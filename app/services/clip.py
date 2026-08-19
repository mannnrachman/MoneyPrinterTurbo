"""Turn a long video into short vertical clips.

Pipeline: extract audio -> whisper transcript (sentence segments with
timestamps) -> LLM scores candidate windows -> ffmpeg renders the top
clips as 9:16 with a blurred background.

Deliberately simple: window boundaries snap to sentence boundaries, so
LLM-selected cuts never land mid-sentence. Face-tracking reframe and
word-level karaoke captions are out of scope for v1.
"""
import json
import math
import os
import re
import subprocess

from loguru import logger

from app.models import const
from app.services import llm, state as sm, subtitle
from app.services import task as tm
from app.utils import utils

_DEFAULT_CLIP_COUNT = 5
_MAX_CLIP_COUNT = 20
_DEFAULT_CLIP_DURATION = 45
_MIN_CLIP_DURATION = 10
_MAX_CLIP_DURATION = 60
_ASPECT_WIDTH = 1080
_ASPECT_HEIGHT = 1920
# Long videos would blow the LLM context. At most N windows sampled across
# the source are sent for scoring; unscored windows remain fallback candidates.
_MAX_PROMPT_WINDOWS = 100
_MAX_WINDOW_TEXT_CHARS = 200

_CLIP_SCORING_PROMPT = """You are an expert short-form video editor. Below is a transcript of a long video split into numbered windows; each window is a candidate short clip for TikTok / YouTube Shorts / Reels.

{transcript}

Score every window from 1 to 10 for how well it works as a standalone short clip. Prefer: a strong hook, a surprising claim, an emotional moment, a clear takeaway, a self-contained story. Penalize: content that depends on earlier context, slow filler, mid-thought rambling.

{extra_instructions}Return ONLY a JSON array, no markdown, no commentary:
[{{"index": 1, "score": 9, "reason": "..."}}]
"""


def _normalize_clip_count(value, default: int = _DEFAULT_CLIP_COUNT) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(count, _MAX_CLIP_COUNT))


def _normalize_clip_duration(value, default: float = _DEFAULT_CLIP_DURATION) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(duration) or duration <= 0:
        return default
    return max(_MIN_CLIP_DURATION, min(duration, _MAX_CLIP_DURATION))


def _fmt_ts(seconds: float) -> str:
    seconds = max(0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Extract a 16kHz mono wav track for whisper transcription."""
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip() or "ffmpeg audio extraction failed"
        )


def _build_windows(segments: list[dict], target_duration: float) -> list[dict]:
    """Group sentence segments into consecutive windows of ~target duration."""
    windows = []
    current = []
    current_start = None
    for seg in segments:
        text = str(seg.get("msg") or "").strip()
        if not text:
            continue
        if current_start is None:
            current_start = float(seg.get("start_time", 0))
        current.append(seg)
        seg_end = float(seg.get("end_time", 0))
        if seg_end - current_start >= target_duration:
            windows.append(_make_window(current, current_start, seg_end))
            current = []
            current_start = None
    if current and current_start is not None:
        last_end = float(current[-1].get("end_time", 0))
        windows.append(_make_window(current, current_start, last_end))
    return windows


def _make_window(segments: list[dict], start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": " ".join(
            str(seg.get("msg") or "").strip() for seg in segments if str(seg.get("msg") or "").strip()
        ),
    }


def _parse_scores(response: str) -> dict | None:
    """Parse the LLM JSON array into {1-based index: {score, reason}}."""
    if not response or "Error: " in response:
        return None
    try:
        data = json.loads(llm._strip_code_fence(response))
    except Exception:
        match = re.search(r"\[.*\]", response or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except Exception:
            return None

    if not isinstance(data, list):
        return None

    scores = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            continue
        scores[index] = {"score": score, "reason": str(item.get("reason") or "")}
    return scores


def _score_windows(windows: list[dict], clip_prompt: str) -> dict | None:
    """Ask the LLM to rank candidate windows. Returns None when LLM is unusable."""
    if len(windows) <= _MAX_PROMPT_WINDOWS:
        window_indices = list(range(len(windows)))
    else:
        # Sample across the entire source instead of silently ignoring the
        # second half of a long recording.
        window_indices = sorted(
            {
                round(i * (len(windows) - 1) / (_MAX_PROMPT_WINDOWS - 1))
                for i in range(_MAX_PROMPT_WINDOWS)
            }
        )

    lines = []
    for window_index in window_indices:
        window = windows[window_index]
        text = window["text"][:_MAX_WINDOW_TEXT_CHARS]
        lines.append(
            f"[{window_index + 1}] {_fmt_ts(window['start'])}-"
            f"{_fmt_ts(window['end'])} {text}"
        )
    transcript_text = "\n".join(lines)

    extra_instructions = ""
    if clip_prompt:
        extra_instructions = f"Prioritize windows that match this theme: {clip_prompt}\n\n"

    prompt = _CLIP_SCORING_PROMPT.format(
        transcript=transcript_text, extra_instructions=extra_instructions
    )
    return _parse_scores(llm._generate_response(prompt))


def _pick_windows(windows: list[dict], scores: dict | None, clip_count: int) -> list[dict]:
    """Pick top-scored windows and fill missing results with even spacing."""
    if scores:
        ranked = []
        scored_indices = set()
        for i, window in enumerate(windows):
            entry = scores.get(i + 1)
            if entry:
                ranked.append((entry["score"], i, window))
                scored_indices.add(i)
        ranked.sort(key=lambda item: item[0], reverse=True)
        picked = [window for _, _, window in ranked[:clip_count]]

        if len(picked) < clip_count:
            remaining = [
                (i, window)
                for i, window in enumerate(windows)
                if i not in scored_indices
            ]
            needed = clip_count - len(picked)
            step = max(1, len(remaining) // max(1, needed))
            picked.extend(window for _, window in remaining[::step][:needed])
        if picked:
            return picked[:clip_count]

    # ponytail: unranked windows (no LLM or all failed) still produce clips,
    # evenly spaced, so the endpoint works end-to-end without a model key.
    if not windows:
        return []
    step = max(1, len(windows) // max(1, clip_count))
    return windows[::step][:clip_count]


def _render_clip(video_path: str, start: float, end: float, output_path: str) -> None:
    """Cut a subclip and render it as 9:16 with a blurred background."""
    width, height = _ASPECT_WIDTH, _ASPECT_HEIGHT
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i",
        video_path,
        # Output-side seeking (-ss after -i): decodes from the start so cuts
        # land exactly on the transcript boundary instead of the nearest
        # keyframe. Slower on long sources, but clip edges must be precise.
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-vf",
        (
            "split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:2[bgblur];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgfit];"
            "[bgblur][fgfit]overlay=(W-w)/2:(H-h)/2"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip() or "ffmpeg clip render failed"
        )


def generate_clips(
    task_id: str,
    video_path: str,
    clip_count=_DEFAULT_CLIP_COUNT,
    clip_duration=_DEFAULT_CLIP_DURATION,
    clip_prompt: str = "",
    subject: str = "",
):
    """Orchestrate long-video -> short-clips for one task, updating task state."""
    subtitle.reset_whisper_model()
    count = _normalize_clip_count(clip_count)
    duration = _normalize_clip_duration(clip_duration)
    task_dir = utils.task_dir(task_id)
    task_metadata = {
        "subject": str(subject or os.path.basename(video_path)),
        "clip_source": str(subject or os.path.basename(video_path)),
    }

    try:
        if not os.path.isfile(video_path):
            return tm._mark_task_failed(task_id, "input", "source video not found")

        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_PROCESSING,
            progress=5,
            **task_metadata,
        )

        audio_path = os.path.join(task_dir, "audio.wav")
        logger.info(f"extracting audio: {video_path}")
        _extract_audio(video_path, audio_path)
        sm.state.update_task(task_id, progress=20, **task_metadata)

        segments = subtitle.transcribe_segments(audio_path)
        if segments is None:
            return tm._mark_task_failed(
                task_id,
                "transcript",
                "whisper transcription unavailable",
                details=task_metadata,
            )
        if not segments:
            return tm._mark_task_failed(
                task_id,
                "transcript",
                "no speech detected in video",
                details=task_metadata,
            )
        sm.state.update_task(task_id, progress=40, **task_metadata)

        windows = _build_windows(segments, duration)
        if not windows:
            return tm._mark_task_failed(
                task_id,
                "windows",
                "transcript too short to clip",
                details=task_metadata,
            )

        scores = _score_windows(windows, clip_prompt)
        picked = _pick_windows(windows, scores, count)
        sm.state.update_task(task_id, progress=60, **task_metadata)

        clip_paths = []
        for i, window in enumerate(picked):
            output_path = os.path.join(task_dir, f"clip-{i + 1}.mp4")
            logger.info(
                f"rendering clip {i + 1}/{len(picked)}: "
                f"{window['start']:.1f}s-{window['end']:.1f}s"
            )
            _render_clip(video_path, window["start"], window["end"], output_path)
            clip_paths.append(output_path)
            sm.state.update_task(
                task_id,
                progress=60 + int(40 * (i + 1) / len(picked)),
                **task_metadata,
            )

        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            clips=clip_paths,
            **task_metadata,
        )
        logger.success(
            f"task {task_id} finished, generated {len(clip_paths)} clips."
        )
        return {"clips": clip_paths}
    except Exception as exc:
        logger.exception(f"clip task failed, task_id: {task_id}, error: {exc}")
        return tm._mark_task_failed(
            task_id,
            "clips",
            f"{type(exc).__name__}: {exc}",
            details=task_metadata,
        )

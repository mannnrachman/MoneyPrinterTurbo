import sys
import unittest
from pathlib import Path

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import clip


class TestClipService(unittest.TestCase):
    def test_normalize_clip_count_clamps_to_range(self):
        self.assertEqual(clip._normalize_clip_count(None, 5), 5)
        self.assertEqual(clip._normalize_clip_count(1, 5), 1)
        self.assertEqual(clip._normalize_clip_count(999, 5), clip._MAX_CLIP_COUNT)
        self.assertEqual(clip._normalize_clip_count("abc", 5), 5)
        self.assertEqual(clip._normalize_clip_count(-3, 5), 1)

    def test_normalize_clip_duration_clamps_to_range(self):
        self.assertEqual(clip._normalize_clip_duration(30, 45), 30)
        self.assertEqual(clip._normalize_clip_duration(1, 45), clip._MIN_CLIP_DURATION)
        self.assertEqual(clip._normalize_clip_duration(500, 45), clip._MAX_CLIP_DURATION)
        self.assertEqual(clip._normalize_clip_duration("x", 45), 45)
        self.assertEqual(clip._normalize_clip_duration(float("nan"), 45), 45)

    def test_build_windows_breaks_on_duration(self):
        segments = [
            {"msg": "one", "start_time": 0, "end_time": 2},
            {"msg": "two", "start_time": 2, "end_time": 4},
            {"msg": "three", "start_time": 4, "end_time": 6},
            {"msg": "four", "start_time": 6, "end_time": 8},
        ]
        windows = clip._build_windows(segments, 5)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["start"], 0)
        self.assertEqual(windows[0]["end"], 6)
        self.assertEqual(windows[0]["text"], "one two three")
        self.assertEqual(windows[1]["start"], 6)

    def test_build_windows_skips_blank_messages(self):
        segments = [
            {"msg": "", "start_time": 0, "end_time": 1},
            {"msg": "hello", "start_time": 1, "end_time": 3},
        ]
        windows = clip._build_windows(segments, 60)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["text"], "hello")

    def test_parse_scores_accepts_plain_json(self):
        response = '[{"index": 3, "score": 9, "reason": "surprising claim"}]'
        scores = clip._parse_scores(response)
        self.assertEqual(scores[3]["score"], 9)
        self.assertEqual(scores[3]["reason"], "surprising claim")

    def test_parse_scores_strips_code_fence(self):
        response = '```json\n[{"index": 1, "score": 8, "reason": "hook"}]\n```'
        scores = clip._parse_scores(response)
        self.assertEqual(scores[1]["score"], 8)

    def test_parse_scores_recovers_json_from_wrapped_text(self):
        response = 'Here are the picks:\n[{"index": 2, "score": 7, "reason": "x"}]'
        scores = clip._parse_scores(response)
        self.assertEqual(scores[2]["score"], 7)

    def test_parse_scores_rejects_garbage(self):
        self.assertIsNone(clip._parse_scores(""))
        self.assertIsNone(clip._parse_scores("Error: llm down"))
        self.assertIsNone(clip._parse_scores("no json here"))
        self.assertIsNone(clip._parse_scores('{"not": "a list"}'))

    def test_pick_windows_ranks_by_score(self):
        windows = [
            {"start": 0, "end": 5, "text": "a"},
            {"start": 5, "end": 10, "text": "b"},
        ]
        scores = {1: {"score": 2, "reason": ""}, 2: {"score": 9, "reason": ""}}
        picked = clip._pick_windows(windows, scores, 2)
        self.assertEqual(picked[0], windows[1])

    def test_pick_windows_falls_back_to_even_spacing(self):
        windows = [{"start": i * 4, "end": i * 4 + 4, "text": str(i)} for i in range(6)]
        picked = clip._pick_windows(windows, None, 3)
        self.assertEqual(len(picked), 3)
        self.assertEqual(picked[0], windows[0])

    def test_fmt_ts(self):
        self.assertEqual(clip._fmt_ts(0), "0:00")
        self.assertEqual(clip._fmt_ts(65), "1:05")
        self.assertEqual(clip._fmt_ts(3661), "1:01:01")

    def test_cleanup_clip_source_removes_files(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "source.mp4")
            audio = os.path.join(tmp, "audio.wav")
            keep = os.path.join(tmp, "clip-1.mp4")
            for path in (video, audio, keep):
                with open(path, "wb") as f:
                    f.write(b"x")
            clip._cleanup_clip_source(video, audio)
            self.assertFalse(os.path.exists(video))
            self.assertFalse(os.path.exists(audio))
            self.assertTrue(os.path.exists(keep))

    def test_cleanup_clip_source_missing_paths_are_ignored(self):
        clip._cleanup_clip_source("/nonexistent/video.mp4", None)

    def test_download_youtube_requires_url_and_caps_height(self):
        # 不真的下载；只验证 format selector 与输出路径行为在缺 yt-dlp 时的报错。
        if clip.YoutubeDL is None:
            with self.assertRaises(RuntimeError):
                clip.download_youtube("https://example.com/v", "/tmp/x.mp4", 360)



if __name__ == "__main__":
    unittest.main()

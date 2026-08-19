"""API endpoints for turning long videos into short vertical clips."""

import os
import pathlib
import shutil

from fastapi import File, Form, Request, UploadFile
from loguru import logger

from app.config import config
from app.controllers import base
from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.v1 import video as video_controller
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import TaskQueryResponse, TaskResponse
from app.services import clip as clip_service
from app.services import state as sm
from app.utils import utils

router = new_router()

_ALLOWED_SUFFIXES = ("mp4", "mov", "mkv", "webm", "avi", "flv")


@router.post(
    "/clips",
    response_model=TaskResponse,
    summary="Generate short clips from a long video",
)
def create_clips(
    request: Request,
    file: UploadFile = File(...),
    clip_count: int = Form(5),
    clip_duration: int = Form(45),
    clip_prompt: str = Form(""),
):
    request_id = base.get_task_id(request)
    task_id = utils.get_uuid()
    task_dir = None
    scheduled = False
    try:
        safe_filename = video_controller._sanitize_upload_filename(
            file.filename, request_id
        )
        suffix = pathlib.Path(safe_filename).suffix.lower().lstrip(".")
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(
                f"Only files with extensions {', '.join(_ALLOWED_SUFFIXES)} "
                "can be used as clip source"
            )

        # The source video lives inside the task dir, so every artifact stays
        # under storage/tasks/<task_id> and the existing /stream + /download
        # endpoints can serve clips without extra path handling.
        task_dir = utils.task_dir(task_id)
        video_path = os.path.join(task_dir, f"source.{suffix}")
        with open(video_path, "wb+") as buffer:
            file.file.seek(0)
            while chunk := file.file.read(1024 * 1024):
                buffer.write(chunk)

        task = {
            "task_id": task_id,
            "request_id": request_id,
            "params": {
                "clip_count": clip_count,
                "clip_duration": clip_duration,
                "clip_prompt": clip_prompt,
            },
        }
        sm.state.update_task(task_id)
        try:
            video_controller.task_manager.add_task(
                clip_service.generate_clips,
                task_id=task_id,
                video_path=video_path,
                clip_count=clip_count,
                clip_duration=clip_duration,
                clip_prompt=clip_prompt,
            )
        except Exception:
            # 状态记录在调度前创建，默认标记为 processing。如果调度器没能
            # 接管任务，必须回滚该记录，否则 API 会展示一个从未运行的任务。
            sm.state.delete_task(task_id)
            raise
        scheduled = True
        logger.success(f"Clip task created: {utils.to_json(task)}")
        return utils.get_response(200, task)
    except TaskQueueFullError as e:
        logger.warning(
            f"reject clip task because queue is full, request_id: {request_id}, "
            f"task_id: {task_id}"
        )
        raise HttpException(
            task_id=task_id, status_code=429, message=f"{request_id}: {str(e)}"
        )
    except ValueError as e:
        raise HttpException(
            task_id=task_id, status_code=400, message=f"{request_id}: {str(e)}"
        )
    finally:
        if not scheduled and task_dir:
            shutil.rmtree(task_dir, ignore_errors=True)


@router.get(
    "/clips/{task_id}",
    response_model=TaskQueryResponse,
    summary="Query clip task status",
)
def get_clip_task(request: Request, task_id: str):
    request_id = base.get_task_id(request)
    endpoint = config.app.get("endpoint", "").rstrip("/")
    task = sm.state.get_task(task_id)
    if task:
        response_task = video_controller._public_task_data(task)
        if "clips" in task:
            response_task["clips"] = [
                video_controller._task_file_to_uri(
                    clip_file, endpoint, utils.task_dir(), request_id
                )
                for clip_file in task["clips"]
            ]
        return utils.get_response(200, response_task)

    raise HttpException(
        task_id=task_id,
        status_code=404,
        message=f"{request_id}: clip task not found",
    )

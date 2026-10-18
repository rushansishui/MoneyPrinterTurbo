"""独立文生视频端点。

与 ``/api/v1/videos`` 的区别：那个端点创建的是完整短视频任务（配音 → 字幕 →
素材 → 剪辑），而这里只做「给提示词，出一段视频」，不复用任务状态机。

计费安全：API 没有会话可以承载 WebUI 的确认框，因此 ``confirm_charge`` 必须
显式传 true；缺失时直接 400，不会创建付费任务。
"""

from fastapi import Depends, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    AIVideoGenerateData,
    AIVideoGenerateRequest,
    AIVideoGenerateResponse,
    AIVideoProviderData,
    AIVideoProvidersData,
    AIVideoProvidersResponse,
    GeneratedClipData,
)
from app.services import ai_video_studio
from app.utils import utils

# 与视频/LLM 端点共用同一鉴权规则；api_key 为空时保持本地免认证默认体验。
router = new_router(dependencies=[Depends(base.verify_token)])


@router.get(
    "/ai_videos/providers",
    response_model=AIVideoProvidersResponse,
    summary="List AI video generation providers",
)
def list_ai_video_providers(request: Request):
    request_id = base.get_task_id(request)
    try:
        providers = ai_video_studio.list_providers()
    except Exception as exc:
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"{request_id}: {exc}",
        ) from exc

    data = AIVideoProvidersData(
        providers=[AIVideoProviderData(**item) for item in providers]
    )
    return utils.get_response(200, data.model_dump())


@router.post(
    "/ai_videos",
    response_model=AIVideoGenerateResponse,
    summary="Generate a single video clip with one AI provider",
)
def generate_ai_video(request: Request, body: AIVideoGenerateRequest):
    request_id = base.get_task_id(request)
    try:
        clips = ai_video_studio.generate_and_save(
            provider_id=body.provider,
            prompt=body.prompt,
            duration=body.duration,
            video_aspect=body.video_aspect,
            first_frame=(
                body.first_frame.model_dump() if body.first_frame else None
            ),
            # 服务层再校验一次：即使调用方绕过 Pydantic 也必须显式确认。
            confirm_charge=body.confirm_charge,
        )
    except ai_video_studio.AIVideoChargeConfirmationRequired as exc:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: {exc}",
        ) from exc
    except ai_video_studio.AIVideoStudioError as exc:
        # 请求本身不合法（未知 provider、空提示词、首帧不被支持等）。
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: {exc}",
        ) from exc
    except ai_video_studio.AIVideoGenerationError as exc:
        # 远端已计费但下载失败等情况，把远端任务 ID 带回，便于人工找回。
        detail = f"{request_id}: {exc}"
        if exc.task_id:
            detail += f" (remote_id={exc.task_id})"
        raise HttpException(
            task_id=request_id,
            status_code=502,
            message=detail,
        ) from exc

    data = AIVideoGenerateData(
        clips=[
            GeneratedClipData(
                provider=clip.provider,
                file=clip.file_path,
                duration=clip.duration,
                width=clip.width,
                height=clip.height,
                remote_task_id=clip.remote_task_id,
            )
            for clip in clips
        ]
    )
    return utils.get_response(200, data.model_dump())

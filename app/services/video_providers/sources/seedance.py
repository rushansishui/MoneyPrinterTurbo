"""火山引擎方舟 Seedance 文生视频客户端。

该模块只负责方舟的异步视频生成协议：提交付费任务、轮询同一个任务、解析
生成结果。素材按需生成、文件下载和成片拼接仍由 ``material`` 服务负责，
避免供应商协议与本地视频工作流相互耦合。

提交与轮询的通用骨架已收敛到 ``video_providers`` 包的公共层，这里只保留
Seedance 特有的部分：三级凭据回退、分辨率白名单、480p 的对齐尺寸、以及
"终止失败必须抛错"的语义。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services.video_providers import client, http_util, polling
from app.services.video_providers.errors import (
    VideoProviderError,
)
from app.services.video_providers.spec import VideoProviderSpec


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
DEFAULT_RESOLUTION = "1080p"
DEFAULT_MIN_DURATION_SECONDS = 2
DEFAULT_MAX_DURATION_SECONDS = 12
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = http_util.RETRYABLE_STATUS_CODES
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "expired"}
)
ACTIVE_STATUSES = frozenset({"queued", "running"})
SUCCESS_STATUSES = frozenset({"succeeded"})
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})


class VolcEngineSeedanceError(VideoProviderError):
    """确定性的配置、请求或响应错误。"""


class VolcEngineSeedanceUnconfirmedTaskError(VolcEngineSeedanceError):
    """远端可能已创建付费任务，但本机无法确认其最终状态。"""


class VolcEngineSeedanceDownloadError(VolcEngineSeedanceError):
    """远端付费任务已成功，但生成的视频未能下载到本机。"""


SPEC = VideoProviderSpec(
    provider_id="volcengine_seedance",
    default_label="Volcano Engine Seedance",
    module="app.services.video_providers.sources.seedance",
    legacy_module="app.services.volcengine_seedance",
    group="ai_video",
    requires_charge_confirmation=True,
    supports_image_to_video=False,
    supports_native_audio=False,
    measure_downloaded_duration=False,
    remote_task_id_attr="task_id",
    error_details_key="volcengine_seedance_task_id",
)


_POLL_CONFIG = polling.PollConfig(
    provider_label="Seedance",
    run_timeout_key="volcengine_seedance_run_timeout",
    poll_interval_key="volcengine_seedance_poll_interval",
    error_cls=VolcEngineSeedanceError,
    unconfirmed_cls=VolcEngineSeedanceUnconfirmedTaskError,
    default_run_timeout=DEFAULT_RUN_TIMEOUT_SECONDS,
    default_poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
    min_poll_interval=0.5,
    max_poll_retries=MAX_POLL_RETRIES,
    retry_base_seconds=RETRY_BASE_SECONDS,
    retryable_status_codes=RETRYABLE_STATUS_CODES,
    active_statuses=ACTIVE_STATUSES,
    success_statuses=SUCCESS_STATUSES,
    terminal_failure_statuses=TERMINAL_FAILURE_STATUSES,
    stop_on_terminal_failure=True,
)

_SUBMIT_CONFIG = client.SubmitConfig(
    provider_label="Seedance",
    error_cls=VolcEngineSeedanceError,
    unconfirmed_cls=VolcEngineSeedanceUnconfirmedTaskError,
    default_task_id_field="id",
)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    按明确且唯一的优先级读取方舟凭据。

    Seedance 专用配置优先级最高；唯一支持的运行时环境变量是语义明确的
    ``VOLCENGINE_ARK_API_KEY``。历史 ``volcengine_api_key`` 只作为共享配置
    兜底，避免已经接入方舟大模型的用户升级后必须重复填写同一把 Key。
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("volcengine_seedance_api_key", "") or "").strip()
    environment_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
    shared_ark_key = str(settings.get("volcengine_api_key", "") or "").strip()
    return configured or environment_key or shared_ark_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("volcengine_seedance_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("volcengine_seedance_model", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    configured = config.app.get(
        "volcengine_seedance_resolution", DEFAULT_RESOLUTION
    )
    value = str(configured).strip().lower()
    if value not in SUPPORTED_RESOLUTIONS:
        # 分辨率会直接影响付费任务的规格。无效值不能静默回退到最高默认
        # 分辨率。配置项缺失时由 get 使用默认值；一旦用户显式写入空值、
        # None、0 等非法值也必须报错，否则仍可能产生超出预期的费用。
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        raise VolcEngineSeedanceError(
            f"Unsupported Seedance resolution {value!r}; expected one of: {supported}"
        )
    return value


def _config_bool(key: str, default: bool) -> bool:
    return http_util.config_bool(key, default)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    return http_util.bounded_float(key, default, minimum, maximum)


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 1 else default

    minimum = read(
        "volcengine_seedance_min_duration", DEFAULT_MIN_DURATION_SECONDS
    )
    maximum = read(
        "volcengine_seedance_max_duration", DEFAULT_MAX_DURATION_SECONDS
    )
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return http_util.tls_verify()


def _status_code(response: Any) -> int:
    return http_util.status_code(response)


def _redact_secret(value: Any, secret: str) -> str:
    return http_util.redact_secret(value, secret)


def _response_error(response: Any, api_key: str) -> str:
    return http_util.response_error(response, api_key)


def _rendition_size(aspect: VideoAspect, resolution: str) -> tuple[int, int]:
    # 方舟的 480p 视频长边按编码对齐实际输出为 864，而不是数学换算得到的
    # 854；720p 和 1080p 分别按官方比例输出 1280、1920。本机真实调用已
    # 验证 480p 竖屏产物为 480x864。来源记录必须描述真实产物，否则后续
    # 审计或素材诊断会看到与文件不一致的尺寸。
    short_edge = {"480p": 480, "720p": 720, "1080p": 1080}[resolution]
    long_edge = {"480p": 864, "720p": 1280, "1080p": 1920}[resolution]
    if aspect == VideoAspect.portrait:
        return short_edge, long_edge
    if aspect == VideoAspect.square:
        return short_edge, short_edge
    return long_edge, short_edge


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: Any = None,
) -> list[MaterialInfo]:
    """提交一个方舟 Seedance 文生视频任务，并等待可下载的结果地址。"""
    if first_frame is not None:
        # 当前接入的是 Seedance 1.0 Pro，不支持图生视频。能力声明在 SPEC 里，
        # 注册表会先行拦截；这里再挡一次，避免绕过注册表直接调用时静默丢弃首帧。
        raise VolcEngineSeedanceError(
            "Seedance does not support image-to-video generation"
        )

    api_key = get_api_key()
    if not api_key:
        raise VolcEngineSeedanceError(
            "Volcano Engine Seedance requires an Ark API key"
        )

    term = str(search_term or "").strip()
    if not term:
        # 空提示词可能来自上游脚本拆分异常。付费生成源不能把它提交到远端，
        # 否则即使接口接受请求，也只会得到无法使用且已经计费的视频。
        raise VolcEngineSeedanceError("Seedance search term must not be empty")

    aspect = VideoAspect(video_aspect)
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        ) from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        # 生成比请求更长不会影响成片：剪辑流程仍按片段时长裁剪；生成比请求
        # 更短只发生在请求超过模型上限时，此时也只能收敛到上限。其它付费视频源
        # 都会说明这次收敛，否则用户无法解释成片片段为何比配置值短。
        logger.info(
            "Seedance clip duration clamped to the configured model range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configured {minimum}-{maximum}s)"
        )
    resolution = _resolution()
    payload = {
        "model": _model_id(),
        "content": [{"type": "text", "text": term}],
        "ratio": aspect.value,
        "duration": duration,
        "resolution": resolution,
        "watermark": _config_bool("volcengine_seedance_watermark", False),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tasks_url = f"{_base_url()}/contents/generations/tasks"
    logger.info(
        "generating video with Volcano Engine Seedance: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    task_id = client.submit_task(
        url=tasks_url,
        payload=payload,
        headers=headers,
        api_key=api_key,
        settings=_SUBMIT_CONFIG,
    )

    task = _wait_for_task(
        task_id=task_id,
        tasks_url=tasks_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []
    content = task.get("content")
    video_url = content.get("video_url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(
        ("http://", "https://")
    ):
        raise VolcEngineSeedanceError(
            f"Seedance task succeeded without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    width, height = _rendition_size(aspect, resolution)
    return [
        MaterialInfo(
            provider="volcengine_seedance",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "volcengine_seedance",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": width,
                    "height": height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    tasks_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    """轮询同一个付费任务。

    参数名保持与迁移前一致：既有测试直接按 ``tasks_url`` 调用本函数。
    """
    return polling.poll_task(
        task_id=task_id,
        status_url=f"{tasks_url}/{task_id}",
        headers=headers,
        api_key=api_key,
        settings=_POLL_CONFIG,
    )

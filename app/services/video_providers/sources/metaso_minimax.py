"""秘塔 MiniMax H3 文生视频客户端。

该模块只负责秘塔代理的 MiniMax V2 协议：提交付费任务、轮询同一个任务、
解析生成结果。素材按需生成、文件下载和成片拼接仍由 ``material`` 服务负责，
避免供应商协议与本地视频工作流相互耦合。

本模块的差异点（相对其它源）：``base_url`` 用 ``urlsplit`` 强校验绝对
HTTP(S) URL；轮询响应外层是 ``{"task": {...}}``，需要解包；错误结构是
MiniMax V2 的 ``error.type/http_code``；时长有 4 秒硬下限；分辨率只支持
``768P``/``2K``；轮询间隔下界是 1.0 秒。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services.video_providers import client, http_util, polling
from app.services.video_providers.errors import (
    VideoProviderDownloadError,
    VideoProviderError,
    VideoProviderUnconfirmedTaskError,
)
from app.services.video_providers.spec import VideoProviderSpec


DEFAULT_BASE_URL = "https://metaso.cn/api/minimax"
DEFAULT_MODEL_ID = "MiniMax-H3"
DEFAULT_RESOLUTION = "2K"
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_PROMPT_LENGTH = 7000
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = http_util.RETRYABLE_STATUS_CODES
ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "canceled"})
SUCCESS_STATUSES = frozenset({"succeeded"})
SUPPORTED_RESOLUTIONS = frozenset({"768P", "2K"})


class MetasoMiniMaxError(VideoProviderError):
    """秘塔 MiniMax 的确定性配置、请求或响应错误。"""
    # 供任务层从异常反查 provider spec。
    provider_id = "metaso_minimax"


class MetasoMiniMaxUnconfirmedTaskError(
    MetasoMiniMaxError, VideoProviderUnconfirmedTaskError
):
    """远端可能已创建付费任务，但本机无法确认其最终状态。"""


class MetasoMiniMaxDownloadError(
    MetasoMiniMaxError, VideoProviderDownloadError
):
    """远端付费任务已成功，但成片未能下载到本机。"""


SPEC = VideoProviderSpec(
    provider_id="metaso_minimax",
    default_label="Metaso MiniMax H3",
    module="app.services.video_providers.sources.metaso_minimax",
    legacy_module="app.services.metaso_minimax",
    group="ai_video",
    requires_charge_confirmation=True,
    supports_image_to_video=False,
    supports_native_audio=False,
    measure_downloaded_duration=False,
    remote_task_id_attr="task_id",
    missing_credentials_message="Metaso MiniMax requires an API key",
)


def _mini_max_error_extractor(payload: dict) -> Any:
    """兼容 MiniMax V2 的嵌套错误结构，并限制日志中的响应长度。"""
    error = payload.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        message = error.get("message")
        http_code = error.get("http_code")
    else:
        error_type = None
        message = payload.get("message") or error
        http_code = payload.get("code")
    return ": ".join(
        str(item)
        for item in (error_type, http_code, message)
        if item not in (None, "")
    )


_SUBMIT_CONFIG = client.SubmitConfig(
    provider_label="Metaso MiniMax",
    error_cls=MetasoMiniMaxError,
    unconfirmed_cls=MetasoMiniMaxUnconfirmedTaskError,
    default_task_id_field="task_id",
    response_error_extractor=_mini_max_error_extractor,
)

_POLL_CONFIG = polling.PollConfig(
    provider_label="Metaso MiniMax",
    run_timeout_key="metaso_minimax_run_timeout",
    poll_interval_key="metaso_minimax_poll_interval",
    error_cls=MetasoMiniMaxError,
    unconfirmed_cls=MetasoMiniMaxUnconfirmedTaskError,
    default_run_timeout=DEFAULT_RUN_TIMEOUT_SECONDS,
    default_poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
    # 秘塔对高频轮询更敏感，下界设为 1.0 秒。
    min_poll_interval=1.0,
    max_poll_retries=MAX_POLL_RETRIES,
    retry_base_seconds=RETRY_BASE_SECONDS,
    retryable_status_codes=RETRYABLE_STATUS_CODES,
    active_statuses=ACTIVE_STATUSES,
    success_statuses=SUCCESS_STATUSES,
    terminal_failure_statuses=TERMINAL_FAILURE_STATUSES,
    stop_on_terminal_failure=True,
    unwrap=lambda body: body.get("task"),
    response_error_extractor=_mini_max_error_extractor,
)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    按固定优先级读取秘塔凭据。

    秘塔 ``mk-`` Key 与 MiniMax 官方 Key 属于不同账户体系，因此不能复用
    项目已有的 ``minimax_api_key``。独立配置和独立环境变量也能防止用户在
    切换 LLM Provider 时意外改变视频生成凭据。
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("metaso_minimax_api_key", "") or "").strip()
    environment_key = os.getenv("METASO_MINIMAX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """返回当前配置是否具备调用秘塔视频接口的凭据。"""
    return bool(get_api_key(settings))


def _base_url() -> str:
    value = (
        str(
            config.app.get("metaso_minimax_base_url", DEFAULT_BASE_URL)
            or DEFAULT_BASE_URL
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MetasoMiniMaxError(
            "metaso_minimax_base_url must be an absolute HTTP(S) URL"
        )
    return value


def _resolution() -> str:
    configured = config.app.get("metaso_minimax_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().upper()
    if value not in SUPPORTED_RESOLUTIONS:
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        # 分辨率直接影响生成费用，用户显式写错时不能静默退回 2K。只有配置项
        # 完全缺失时才使用默认值，避免无意间创建比预期更贵的任务。
        raise MetasoMiniMaxError(
            f"Unsupported Metaso MiniMax resolution {value!r}; "
            f"expected one of: {supported}"
        )
    return value


def _config_bool(key: str, default: bool) -> bool:
    return http_util.config_bool(key, default)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    return http_util.bounded_float(key, default, minimum, maximum)


def _tls_verify() -> bool:
    return http_util.tls_verify()


def _status_code(response: Any) -> int:
    return http_util.status_code(response)


def _redact_secret(value: Any, secret: str) -> str:
    return http_util.redact_secret(value, secret)


def _response_error(response: Any, api_key: str) -> str:
    return http_util.response_error(
        response, api_key, extractor=_mini_max_error_extractor
    )


def _normalize_duration(minimum_duration: int) -> tuple[int, int]:
    """返回“用户请求时长、实际提交时长”，用于日志解释最短 4 秒约束。"""
    try:
        requested = int(minimum_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if requested <= 0:
        raise MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )
    duration = min(
        max(requested, DEFAULT_MIN_DURATION_SECONDS),
        DEFAULT_MAX_DURATION_SECONDS,
    )
    return requested, duration


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: Any = None,
) -> list[MaterialInfo]:
    """提交一个秘塔 MiniMax H3 文生视频任务并等待可下载的结果。"""
    if first_frame is not None:
        raise MetasoMiniMaxError(
            "Metaso MiniMax does not support image-to-video generation"
        )

    api_key = get_api_key()
    if not api_key:
        raise MetasoMiniMaxError("Metaso MiniMax requires an API key")

    term = str(search_term or "").strip()
    if not term:
        # 空提示词通常表示上游脚本拆分失败。付费接口不能用无效输入试探，
        # 否则即使远端接受也只会产生无法使用的计费素材。
        raise MetasoMiniMaxError("Metaso MiniMax search term must not be empty")
    if len(term) > MAX_PROMPT_LENGTH:
        raise MetasoMiniMaxError(
            f"Metaso MiniMax search term exceeds {MAX_PROMPT_LENGTH} characters"
        )

    aspect = VideoAspect(video_aspect)
    requested_duration, duration = _normalize_duration(minimum_duration)
    if duration != requested_duration:
        logger.info(
            "Metaso MiniMax clip duration adjusted to H3 limits: "
            f"requested={requested_duration}s, using={duration}s, "
            f"supported={DEFAULT_MIN_DURATION_SECONDS}-{DEFAULT_MAX_DURATION_SECONDS}s"
        )
    resolution = _resolution()
    payload = {
        "model": DEFAULT_MODEL_ID,
        "content": [{"type": "text", "text": term}],
        "resolution": resolution,
        "duration": duration,
        "ratio": aspect.value,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = _base_url()
    create_url = f"{base_url}/v2/video_generation"
    logger.info(
        "generating video with Metaso MiniMax H3: "
        f"resolution={resolution}, ratio={aspect.value}, duration={duration}s, "
        f"prompt_length={len(term)}"
    )

    task_id = client.submit_task(
        url=create_url,
        payload=payload,
        headers=headers,
        api_key=api_key,
        settings=_SUBMIT_CONFIG,
    )

    task = _wait_for_task(
        task_id=task_id,
        base_url=base_url,
        headers=headers,
        api_key=api_key,
    )
    content = task.get("content")
    video_url = content.get("url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(
        ("http://", "https://")
    ):
        raise MetasoMiniMaxError(
            f"Metaso MiniMax task succeeded without a downloadable video: "
            f"id={task_id}",
            task_id=task_id,
        )

    actual_duration = task.get("duration", duration)
    try:
        actual_duration = int(actual_duration)
    except (TypeError, ValueError, OverflowError):
        actual_duration = duration
    if actual_duration <= 0:
        actual_duration = duration

    return [
        MaterialInfo(
            provider="metaso_minimax",
            url=video_url,
            duration=actual_duration,
            source_info={
                "provider": "metaso_minimax",
                "search_term": term,
                "asset_id": task_id,
                # MiniMax 的 2K/768P 是规格名称，接口没有承诺固定像素尺寸。
                # 不猜测 width/height，后续若需要精确尺寸应以下载文件探测值为准。
                "rendition": {"id": task_id},
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    base_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    """轮询同一个付费任务。参数名保持与迁移前一致。"""
    result = polling.poll_task(
        task_id=task_id,
        status_url=(
            f"{base_url}/v2/query/video_generation/{quote(task_id, safe='')}"
        ),
        headers=headers,
        api_key=api_key,
        settings=_POLL_CONFIG,
    )
    # 该 provider 配置为 stop_on_terminal_failure=True，因此失败一定抛错，
    # 能走到这里必然是成功任务。
    return result if result is not None else {}

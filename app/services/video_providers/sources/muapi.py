"""MuAPI 异步文生视频客户端。

Provider 特有的提交/轮询协议放在这里，让 material 服务只需要处理
``MaterialInfo`` 与本地下载。MuAPI 的任务一经接受即计费，因此响应不明确时
提交**绝不重试**。

本模块的差异点（相对其它源）：base_url 与 endpoint 都做绝对 URL / 相对路径
校验；轮询路径是 ``/predictions/{id}/result``；提交与轮询都要专门处理 3xx
重定向并带上 Location；错误 detail 可能是 list。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

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


DEFAULT_BASE_URL = "https://api.muapi.ai/api/v1"
# The budget Seedance Lite route supports the three MoneyPrinterTurbo aspect
# ratios, 3-12 second clips, and 480p/720p/1080p output.
DEFAULT_ENDPOINT = "seedance-lite-t2v"
DEFAULT_RESOLUTION = "480p"
DEFAULT_MIN_DURATION_SECONDS = 3
DEFAULT_MAX_DURATION_SECONDS = 12
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = http_util.RETRYABLE_STATUS_CODES
ACTIVE_STATUSES = frozenset({"queued", "pending", "processing"})
TERMINAL_SUCCESS_STATUSES = frozenset({"completed"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "expired"}
)


class MuAPIError(VideoProviderError):
    """Deterministic configuration, request, or response error."""
    # 供任务层从异常反查 provider spec。
    provider_id = "muapi"


class MuAPIUnconfirmedTaskError(MuAPIError, VideoProviderUnconfirmedTaskError):
    """The remote job may exist, but its final state cannot be confirmed."""


class MuAPIDownloadError(MuAPIError, VideoProviderDownloadError):
    """A billable job completed, but its output could not be downloaded."""


SPEC = VideoProviderSpec(
    provider_id="muapi",
    default_label="MuAPI AI Video",
    module="app.services.video_providers.sources.muapi",
    legacy_module="app.services.muapi",
    group="ai_video",
    requires_charge_confirmation=True,
    supports_image_to_video=False,
    supports_native_audio=False,
    # MuAPI 是唯一以本地实测文件时长核算素材时长的源。
    measure_downloaded_duration=True,
    remote_task_id_attr="task_id",
    missing_credentials_message=(
        "MuAPI video generation requires a MuAPI API key"
    ),
)


def _muapi_error_extractor(payload: dict) -> Any:
    """MuAPI 的错误 detail 可能是字符串或 list。"""
    error = payload.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or error.get("detail") or error.get("code")
    else:
        detail = payload.get("detail") or payload.get("message") or error
    if isinstance(detail, list):
        detail = "; ".join(str(item) for item in detail)
    return detail


def _submit_redirect_factory(response: Any, api_key: str) -> Exception:
    location = getattr(response, "headers", {}).get("Location", "")
    return MuAPIUnconfirmedTaskError(
        "MuAPI submission returned an unexpected redirect; the paid task "
        "state is unknown: "
        f"HTTP {http_util.status_code(response)}, "
        f"location={http_util.redact_secret(location, api_key)}"
    )


def _poll_redirect_factory(response: Any, task_id: str, api_key: str) -> Exception:
    location = getattr(response, "headers", {}).get("Location", "")
    return MuAPIUnconfirmedTaskError(
        "MuAPI polling returned an unexpected redirect; the paid "
        "task state is unknown: "
        f"HTTP {http_util.status_code(response)}, "
        f"location={http_util.redact_secret(location, api_key)}",
        task_id=task_id,
    )


_SUBMIT_CONFIG = client.SubmitConfig(
    provider_label="MuAPI",
    error_cls=MuAPIError,
    unconfirmed_cls=MuAPIUnconfirmedTaskError,
    default_task_id_field="id",
    response_error_extractor=_muapi_error_extractor,
    redirect_error_factory=_submit_redirect_factory,
)

_POLL_CONFIG = polling.PollConfig(
    provider_label="MuAPI",
    run_timeout_key="muapi_run_timeout",
    poll_interval_key="muapi_poll_interval",
    error_cls=MuAPIError,
    unconfirmed_cls=MuAPIUnconfirmedTaskError,
    default_run_timeout=DEFAULT_RUN_TIMEOUT_SECONDS,
    default_poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
    min_poll_interval=0.5,
    max_poll_retries=MAX_POLL_RETRIES,
    retry_base_seconds=RETRY_BASE_SECONDS,
    retryable_status_codes=RETRYABLE_STATUS_CODES,
    active_statuses=ACTIVE_STATUSES,
    success_statuses=TERMINAL_SUCCESS_STATUSES,
    terminal_failure_statuses=TERMINAL_FAILURE_STATUSES,
    # 远端明确失败意味着任务已结束，可以安全地继续后续片段。
    stop_on_terminal_failure=False,
    redirect_error_factory=_poll_redirect_factory,
    response_error_extractor=_muapi_error_extractor,
)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """Read the dedicated MuAPI key, then the provider-specific environment key."""
    settings = config.app if settings is None else settings
    configured = str(settings.get("muapi_api_key", "") or "").strip()
    environment_key = os.getenv("MUAPI_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    value = str(
        config.app.get("muapi_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MuAPIError("muapi_base_url must be an absolute HTTP(S) URL")
    return value


def _endpoint() -> str:
    value = str(
        config.app.get("muapi_video_endpoint", DEFAULT_ENDPOINT) or DEFAULT_ENDPOINT
    ).strip().strip("/")
    # The setting is a path below /api/v1, never an arbitrary URL.  This keeps
    # credentials in the configured MuAPI host and makes the generated request
    # straightforward to audit.
    if not value or "://" in value or "?" in value or "#" in value:
        raise MuAPIError(
            "muapi_video_endpoint must be a non-empty relative endpoint path"
        )
    return value


def _resolution() -> str:
    value = str(
        config.app.get("muapi_resolution", DEFAULT_RESOLUTION) or ""
    ).strip()
    return value or DEFAULT_RESOLUTION


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

    minimum = read("muapi_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("muapi_max_duration", DEFAULT_MAX_DURATION_SECONDS)
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return http_util.tls_verify()


def _status_code(response: Any) -> int:
    return http_util.status_code(response)


def _redact_secret(value: Any, secret: str) -> str:
    return http_util.redact_secret(value, secret)


def _response_error(response: Any, api_key: str) -> str:
    return http_util.response_error(
        response, api_key, extractor=_muapi_error_extractor
    )


def _extract_task_id(body: dict) -> str:
    return str(body.get("request_id") or body.get("id") or "")


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: Any = None,
) -> list[MaterialInfo]:
    """Submit one MuAPI text-to-video job and wait for its downloadable output."""
    if first_frame is not None:
        raise MuAPIError("MuAPI does not support image-to-video generation")

    api_key = get_api_key()
    if not api_key:
        raise MuAPIError("MuAPI video generation requires a MuAPI API key")

    term = str(search_term or "").strip()
    if not term:
        raise MuAPIError("MuAPI search term must not be empty")

    try:
        aspect = VideoAspect(video_aspect)
    except (TypeError, ValueError) as exc:
        raise MuAPIError("MuAPI video aspect is invalid") from exc
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MuAPIError("MuAPI clip duration must be a positive integer") from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        logger.info(
            "muapi clip duration clamped to the configured endpoint range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configured {minimum}-{maximum}s)"
        )

    base_url = _base_url()
    endpoint = _endpoint()
    videos_url = f"{base_url}/{endpoint}"
    payload = {
        "prompt": term,
        "aspect_ratio": aspect.value,
        "resolution": _resolution(),
        "duration": duration,
    }
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    logger.info(
        "generating video with MuAPI: "
        f"endpoint={endpoint}, term={term!r}, duration={duration}s"
    )

    task_id = client.submit_task(
        url=videos_url,
        payload=payload,
        headers=headers,
        api_key=api_key,
        settings=_SUBMIT_CONFIG,
        task_id_extractor=_extract_task_id,
    )

    task = _wait_for_task(
        task_id=task_id,
        base_url=base_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []

    outputs = task.get("outputs")
    candidates = [outputs] if isinstance(outputs, str) else outputs
    video_url = ""
    for candidate in candidates if isinstance(candidates, list) else []:
        if isinstance(candidate, str) and candidate.startswith(
            ("http://", "https://")
        ):
            video_url = candidate
            break
    if not video_url:
        raise MuAPIError(
            f"MuAPI task completed without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    video_width, video_height = aspect.to_resolution()
    return [
        MaterialInfo(
            provider="muapi",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "muapi",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": video_width,
                    "height": video_height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    base_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    """轮询同一个付费任务。参数名保持与迁移前一致。"""
    return polling.poll_task(
        task_id=task_id,
        status_url=f"{base_url}/predictions/{task_id}/result",
        headers=headers,
        api_key=api_key,
        settings=_POLL_CONFIG,
    )

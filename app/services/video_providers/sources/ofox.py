"""OFox 多模型文生视频客户端。

OFox 用一个 API Key 暴露多个文生视频模型（Seedance、Wan 等），走异步
``/videos`` 接口。本模块的差异点：``ofox_provider`` 上游厂商钉定、
分辨率不做本地白名单（由服务端按模型校验）、成功状态集含 ``succeeded``、
远端明确失败时返回 ``None`` 让调用方换关键词继续、结果 URL 优先取
``mirror_urls`` 再回退 ``unsigned_urls``。
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
    VideoProviderDownloadError,
    VideoProviderError,
    VideoProviderUnconfirmedTaskError,
)
from app.services.video_providers.spec import VideoProviderSpec


DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL_ID = "bytedance/seedance-2.0-fast"
DEFAULT_RESOLUTION = "720p"
# 默认钉定国际厂商通道：面向全球受众时内容政策更一致；显式配置为空则交回
# 网关按权重在可用厂商间分发。
DEFAULT_PROVIDER_TYPE = "byteplus"
# 默认模型 bytedance/seedance-2.0-fast 只接受 4-15 秒（服务端实测校验值）。
# 其它可选模型区间不同（如 alibaba/wan-2.7 为 2-15 秒），切换模型时应同步
# 调整配置里的区间；超出区间的请求会被 API 以明确的 400 拒绝，不会计费。
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = http_util.RETRYABLE_STATUS_CODES
TERMINAL_SUCCESS_STATUSES = frozenset({"completed", "succeeded"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "error", "cancelled", "canceled", "expired"}
)
# 官方成功路径: pending(收单待上游提交) → queued → in_progress → completed
ACTIVE_STATUSES = frozenset({"pending", "queued", "in_progress"})


class OFoxError(VideoProviderError):
    """确定性的配置、请求或响应错误。"""
    # 供任务层从异常反查 provider spec。
    provider_id = "ofox"


class OFoxUnconfirmedTaskError(OFoxError, VideoProviderUnconfirmedTaskError):
    """远端可能已创建付费任务，但本机无法确认其最终状态。"""


class OFoxDownloadError(OFoxError, VideoProviderDownloadError):
    """远端付费任务已成功，但生成的视频未能下载到本机。"""


SPEC = VideoProviderSpec(
    provider_id="ofox",
    default_label="OFox",
    module="app.services.video_providers.sources.ofox",
    legacy_module="app.services.ofox",
    group="ai_video",
    requires_charge_confirmation=True,
    supports_image_to_video=False,
    supports_native_audio=False,
    measure_downloaded_duration=False,
    remote_task_id_attr="task_id",
    missing_credentials_message=(
        "OFox video generation requires an OFox API key"
    ),
)


_SUBMIT_CONFIG = client.SubmitConfig(
    provider_label="OFox",
    error_cls=OFoxError,
    unconfirmed_cls=OFoxUnconfirmedTaskError,
    default_task_id_field="id",
)

_POLL_CONFIG = polling.PollConfig(
    provider_label="OFox",
    run_timeout_key="ofox_run_timeout",
    poll_interval_key="ofox_poll_interval",
    error_cls=OFoxError,
    unconfirmed_cls=OFoxUnconfirmedTaskError,
    default_run_timeout=DEFAULT_RUN_TIMEOUT_SECONDS,
    default_poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
    min_poll_interval=0.5,
    max_poll_retries=MAX_POLL_RETRIES,
    retry_base_seconds=RETRY_BASE_SECONDS,
    retryable_status_codes=RETRYABLE_STATUS_CODES,
    active_statuses=ACTIVE_STATUSES,
    success_statuses=TERMINAL_SUCCESS_STATUSES,
    terminal_failure_statuses=TERMINAL_FAILURE_STATUSES,
    # 远端明确失败意味着任务已经结束，可以安全地继续后续片段；由调用方
    # 决定是否换一个关键词重试。
    stop_on_terminal_failure=False,
)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    按明确且唯一的优先级读取 OFox 凭据。

    配置文件里的专用键优先级最高；唯一支持的运行时环境变量是语义明确的
    ``OFOX_API_KEY``。
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("ofox_api_key", "") or "").strip()
    environment_key = os.getenv("OFOX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("ofox_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("ofox_text_to_video_model", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    """
    读取生成分辨率。

    OFox 按模型在服务端校验分辨率（例如默认 Seedance fast 模型只接受
    480p/720p），无效值会得到一个明确的 400 拒绝且不会创建付费任务，
    因此这里只做去空白，不维护本地白名单——白名单会随远端模型目录
    变化而过期。空值退回默认，避免把空字符串提交到远端。
    """
    value = str(
        config.app.get("ofox_resolution", DEFAULT_RESOLUTION) or ""
    ).strip()
    return value or DEFAULT_RESOLUTION


def _provider_type() -> str:
    """
    读取上游厂商钉定（provider routing）。

    OFox 的部分模型由多个上游厂商供货（如 Seedance 系列的 volcengine 与
    byteplus），各厂商有各自的内容政策与区域可用性。默认钉定 byteplus
    （国际厂商，对全球受众内容政策更一致、路由可预期）；配置为其它厂商名
    则钉定那一家；显式配置为空字符串则不钉定，由网关按权重分发。非法厂商
    名会被 API 以专属的 400 ``invalid_provider_type`` 拒绝且不创建付费
    任务，因此本地不维护厂商白名单。
    """
    value = config.app.get("ofox_provider", DEFAULT_PROVIDER_TYPE)
    if value is None:
        return DEFAULT_PROVIDER_TYPE
    return str(value).strip()


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

    minimum = read("ofox_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("ofox_max_duration", DEFAULT_MAX_DURATION_SECONDS)
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return http_util.tls_verify()


def _status_code(response: Any) -> int:
    return http_util.status_code(response)


def _redact_secret(value: Any, secret: str) -> str:
    return http_util.redact_secret(value, secret)


def _response_error(response: Any, api_key: str) -> str:
    return http_util.response_error(response, api_key)


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: Any = None,
) -> list[MaterialInfo]:
    """提交一个 OFox 文生视频任务，并等待可下载的结果地址。"""
    if first_frame is not None:
        raise OFoxError("OFox does not support image-to-video generation")

    api_key = get_api_key()
    if not api_key:
        raise OFoxError("OFox video generation requires an OFox API key")

    term = str(search_term or "").strip()
    if not term:
        # 空提示词可能来自上游脚本拆分异常。付费生成源不能把它提交到远端，
        # 否则即使接口接受请求，也只会得到无法使用且已经计费的视频。
        raise OFoxError("OFox search term must not be empty")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OFoxError("OFox clip duration must be a positive integer") from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        # 生成比请求更长不会影响成片：剪辑流程仍按片段时长裁剪；生成比请求
        # 更短只发生在请求超过模型上限时，此时也只能收敛到上限。
        logger.info(
            f"ofox clip duration clamped to the configured model range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configured {minimum}-{maximum}s)"
        )
    payload = {
        "model": _model_id(),
        "prompt": term,
        "duration": duration,
        "resolution": _resolution(),
        "aspect_ratio": aspect.value,
    }
    provider_type = _provider_type()
    if provider_type:
        payload["provider"] = {"type": provider_type}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    videos_url = f"{_base_url()}/videos"
    logger.info(
        "generating video with OFox: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    task_id = client.submit_task(
        url=videos_url,
        payload=payload,
        headers=headers,
        api_key=api_key,
        settings=_SUBMIT_CONFIG,
    )

    task = _wait_for_task(
        task_id=task_id,
        videos_url=videos_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []

    # 官方推荐优先使用 mirror_urls（OFox CDN 持久签名地址，仅在上游开启镜像
    # 时返回），缺失时回退到 unsigned_urls（上游临时直链，可能 24 小时内过
    # 期）。两者都必须整体保留并立即用于下载，不写入长期的 source_info。
    video_url = ""
    for field in ("mirror_urls", "unsigned_urls"):
        urls = task.get(field)
        for candidate in urls if isinstance(urls, list) else []:
            if isinstance(candidate, str) and candidate.startswith(
                ("http://", "https://")
            ):
                video_url = candidate
                break
        if video_url:
            break
    if not video_url:
        raise OFoxError(
            f"OFox task completed without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    return [
        MaterialInfo(
            provider="ofox",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "ofox",
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
    videos_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    """轮询同一个付费任务。参数名保持与迁移前一致。"""
    return polling.poll_task(
        task_id=task_id,
        status_url=f"{videos_url}/{task_id}",
        headers=headers,
        api_key=api_key,
        settings=_POLL_CONFIG,
    )

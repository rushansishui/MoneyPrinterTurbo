"""火山引擎方舟 Seedance 2.5 文生/图生视频客户端。

与 ``seedance.py``（Seedance 1.0 Pro）的关系：**同一套方舟协议**——相同的
``POST /contents/generations/tasks`` 提交、相同的 ``GET .../tasks/{id}`` 轮询、
相同的 ``content.video_url`` 结果字段与 ``id`` 任务 ID 字段，因此共用
``video_providers`` 公共层的提交与轮询实现。

差异集中在请求参数与能力上：

* **时长上限** 30 秒（1.0 Pro 为 12 秒），取值 ``[4, 30]`` 或 ``-1``（自动）。
* **分辨率** 多出 ``4k``。
* **图生视频（i2v）** ``content`` 内追加
  ``{"type": "image_url", "image_url": {"url": ...}, "role": "first_frame"}``，
  且此时 ``ratio`` **必须**为 ``adaptive``（由方舟强制，传入具体比例会报错）。
* **原生音频** ``generate_audio: true`` 让模型直接产出带声音的视频。音轨内嵌
  在返回的成片里，因此成片是否使用它由剪辑阶段决定，provider 只负责请求。

本地图片以 base64 data URL 提交（方舟同时接受公网 URL）。
"""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Mapping
from pathlib import Path
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
from app.services.video_providers.spec import FirstFrame, VideoProviderSpec


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# 文档给定的 Seedance 2.5 正式模型 ID。
DEFAULT_MODEL_ID = "doubao-seedance-2-5-260628"
DEFAULT_RESOLUTION = "1080p"
# 官方取值范围 [4, 30] 秒；本项目额外支持 -1（由模型自动决定时长），
# 但按需生成流程需要一个确定的片段时长来累计覆盖进度，因此默认不用 -1。
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 30
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
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p", "4k"})
# 图生视频时方舟要求 ratio 必须为 adaptive（自动跟随首帧宽高比）。
ADAPTIVE_RATIO = "adaptive"
# 单张首帧图片的最大字节数。文档未给出硬上限，这里取一个保守值，
# 避免把超大图片编码进请求体导致提交被拒或超时。
MAX_FIRST_FRAME_BYTES = 10 * 1024 * 1024


class Seedance25Error(VideoProviderError):
    """确定性的配置、请求或响应错误。"""

    provider_id = "seedance25"


class Seedance25UnconfirmedTaskError(
    Seedance25Error, VideoProviderUnconfirmedTaskError
):
    """远端可能已创建付费任务，但本机无法确认其最终状态。"""


class Seedance25DownloadError(Seedance25Error, VideoProviderDownloadError):
    """远端付费任务已成功，但生成的视频未能下载到本机。"""


SPEC = VideoProviderSpec(
    provider_id="seedance25",
    default_label="Volcano Engine Seedance 2.5",
    module="app.services.video_providers.sources.seedance25",
    legacy_module="",
    group="ai_video",
    requires_charge_confirmation=True,
    # Seedance 2.5 原生支持首帧图生视频与带声音视频。
    supports_image_to_video=True,
    supports_native_audio=True,
    measure_downloaded_duration=False,
    remote_task_id_attr="task_id",
    error_details_key="seedance25_task_id",
    missing_credentials_message=(
        "Volcano Engine Seedance 2.5 requires an Ark API key"
    ),
)


_SUBMIT_CONFIG = client.SubmitConfig(
    provider_label="Seedance 2.5",
    error_cls=Seedance25Error,
    unconfirmed_cls=Seedance25UnconfirmedTaskError,
    default_task_id_field="id",
)

_POLL_CONFIG = polling.PollConfig(
    provider_label="Seedance 2.5",
    run_timeout_key="seedance25_run_timeout",
    poll_interval_key="seedance25_poll_interval",
    error_cls=Seedance25Error,
    unconfirmed_cls=Seedance25UnconfirmedTaskError,
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


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    按明确且唯一的优先级读取方舟凭据。

    与 Seedance 1.0 使用同一套方舟账号体系，因此回退顺序一致：专用配置 →
    ``VOLCENGINE_ARK_API_KEY`` → 共享的 ``volcengine_api_key``。这样已经配置
    过方舟的用户不需要为新模型重复填写同一把 Key。
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("seedance25_api_key", "") or "").strip()
    environment_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
    shared_ark_key = str(settings.get("volcengine_api_key", "") or "").strip()
    return configured or environment_key or shared_ark_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("seedance25_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("seedance25_model", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    configured = config.app.get("seedance25_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().lower()
    if value not in SUPPORTED_RESOLUTIONS:
        # 分辨率直接决定付费任务的规格与单价，写错时不能静默回退到默认值。
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        raise Seedance25Error(
            f"Unsupported Seedance 2.5 resolution {value!r}; "
            f"expected one of: {supported}"
        )
    return value


def _config_bool(key: str, default: bool) -> bool:
    return http_util.config_bool(key, default)


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 1 else default

    minimum = read("seedance25_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("seedance25_max_duration", DEFAULT_MAX_DURATION_SECONDS)
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
    """按输出规格与画幅推算记录用的像素尺寸。

    分辨率名称只承诺短边规格（480/720/1080）与 16:9 长边；本项目只生成
    9:16 / 16:9 / 1:1 三种画幅，因此用与 Seedance 1.0 相同的对齐口径，
    避免审计时看到与文件不一致的尺寸。4k 按 3840 长边记录。
    """
    short_edge = {"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}[resolution]
    long_edge = {"480p": 864, "720p": 1280, "1080p": 1920, "4k": 3840}[resolution]
    if aspect == VideoAspect.portrait:
        return short_edge, long_edge
    if aspect == VideoAspect.square:
        return short_edge, short_edge
    return long_edge, short_edge


def _encode_first_frame(first_frame: FirstFrame) -> str:
    """把统一的 ``FirstFrame`` 转成方舟 ``image_url.url`` 接受的取值。

    * ``url`` —— 公网可访问的图片地址，直接提交。
    * ``local_path`` —— 本地文件读成 base64 data URL（方舟同时接受这种形态）。
    * ``asset_id`` —— 方舟图生视频不接受素材 ID，必须显式拒绝而不是静默丢弃。
    """
    if first_frame.kind == "url":
        return first_frame.value

    if first_frame.kind != "local_path":
        raise Seedance25Error(
            "Seedance 2.5 image-to-video accepts a public image URL or a local "
            f"file, not {first_frame.kind!r}"
        )

    path = Path(first_frame.value)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise Seedance25Error(
            f"first frame image is not readable: {path}"
        ) from exc
    if size <= 0:
        raise Seedance25Error(f"first frame image is empty: {path}")
    if size > MAX_FIRST_FRAME_BYTES:
        raise Seedance25Error(
            f"first frame image exceeds {MAX_FIRST_FRAME_BYTES} bytes: {path}"
        )

    mime = first_frame.mime or mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Seedance25Error(
            f"first frame image could not be read: {path}"
        ) from exc
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _build_content(term: str, first_frame: FirstFrame | None) -> list[dict[str, Any]]:
    """构造方舟的 ``content`` 数组：文本提示词，可选首帧图。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": term}]
    if first_frame is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_first_frame(first_frame)},
                "role": "first_frame",
            }
        )
    return content


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: FirstFrame | None = None,
) -> list[MaterialInfo]:
    """提交一个 Seedance 2.5 任务，并等待可下载的结果地址。

    ``first_frame`` 非空时走图生视频：方舟要求此时的 ``ratio`` 必须为
    ``adaptive``，由服务端自动跟随首帧宽高比，因此覆盖面宽高比。
    """
    api_key = get_api_key()
    if not api_key:
        raise Seedance25Error("Volcano Engine Seedance 2.5 requires an Ark API key")

    term = str(search_term or "").strip()
    if not term:
        # 空提示词通常来自上游脚本拆分异常。付费生成源不能把无效输入提交到
        # 远端，否则即使接口接受也只会得到无法使用且已计费的视频。
        raise Seedance25Error("Seedance 2.5 search term must not be empty")

    aspect = VideoAspect(video_aspect)
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Seedance25Error(
            "Seedance 2.5 clip duration must be a positive integer"
        ) from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        logger.info(
            "Seedance 2.5 clip duration clamped to the configured model range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configured {minimum}-{maximum}s)"
        )

    resolution = _resolution()
    # 图生视频必须用 adaptive，否则方舟会拒绝请求。
    ratio = ADAPTIVE_RATIO if first_frame is not None else aspect.value
    payload: dict[str, Any] = {
        "model": _model_id(),
        "content": _build_content(term, first_frame),
        "ratio": ratio,
        "duration": duration,
        "resolution": resolution,
        "watermark": _config_bool("seedance25_watermark", False),
    }
    # 原生音频：让模型直接产出带声音的视频。默认开启，因为这是选用 2.5 的
    # 主要理由；成片是否真正混入该音轨由剪辑阶段决定。
    if _config_bool("seedance25_generate_audio", True):
        payload["generate_audio"] = True

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tasks_url = f"{_base_url()}/contents/generations/tasks"
    logger.info(
        "generating video with Volcano Engine Seedance 2.5: "
        f"model={payload['model']}, duration={duration}s, ratio={ratio}, "
        f"resolution={resolution}, image_to_video={first_frame is not None}, "
        f"prompt_length={len(term)}"
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
        raise Seedance25Error(
            f"Seedance 2.5 task succeeded without a downloadable video: "
            f"id={task_id}",
            task_id=task_id,
        )

    width, height = _rendition_size(aspect, resolution)
    source_info: dict[str, Any] = {
        "provider": "seedance25",
        "search_term": term,
        "asset_id": task_id,
        "rendition": {"id": task_id, "width": width, "height": height},
    }
    if first_frame is not None:
        # 记录首帧来源，便于审计"这一段的画面是从哪张图接下去的"。
        source_info["first_frame_kind"] = first_frame.kind
        source_info["first_frame_origin"] = first_frame.origin

    return [
        MaterialInfo(
            provider="seedance25",
            url=video_url,
            duration=duration,
            source_info=source_info,
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    tasks_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    """轮询同一个付费任务。参数名与 Seedance 1.0 保持一致便于复用调用习惯。"""
    return polling.poll_task(
        task_id=task_id,
        status_url=f"{tasks_url}/{task_id}",
        headers=headers,
        api_key=api_key,
        settings=_POLL_CONFIG,
    )

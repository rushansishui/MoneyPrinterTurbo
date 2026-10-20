"""火山引擎方舟 Seedance 2.5 文生/图生视频客户端。

与 ``seedance.py``（Seedance 1.0 Pro）的关系：**同一套方舟协议**——相同的
``POST /contents/generations/tasks`` 提交、相同的 ``GET .../tasks/{id}`` 轮询、
相同的 ``content.video_url`` 结果字段与 ``id`` 任务 ID 字段，因此共用
``video_providers`` 公共层的提交与轮询实现。

差异集中在请求参数与能力上：

* **时长上限** 30 秒（1.0 Pro 为 12 秒），取值 ``[4, 30]`` 或 ``-1``（自动）。
* **分辨率** 支持 ``480p``、``720p``、``1080p``。
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
from collections.abc import Mapping, Sequence
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
from app.services.video_providers.spec import (
    FirstFrame,
    ReferenceAsset,
    VideoProviderSpec,
)


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
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})
# 图生视频时方舟要求 ratio 必须为 adaptive（自动跟随首帧宽高比）。
ADAPTIVE_RATIO = "adaptive"
# 官方限制单张图片小于 30 MB；超大文件应使用 URL 或素材 ID，避免请求体超过
# 64 MB。
MAX_FIRST_FRAME_BYTES = 30 * 1024 * 1024
# 官方限制：图片小于 30 MB、音频小于 15 MB。视频不能用 Base64 提交。
MAX_REFERENCE_IMAGE_BYTES = MAX_FIRST_FRAME_BYTES
MAX_REFERENCE_AUDIO_BYTES = 15 * 1024 * 1024
VALID_OMNI_REFERENCE_TASK_TYPES = frozenset({"auto", "edit", "extend"})


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
    # Seedance 2.5 原生支持首帧图生视频、带声音视频，以及全模态参考素材
    # （reference_image / reference_video / reference_audio 可混合传入）。
    supports_image_to_video=True,
    supports_native_audio=True,
    supports_reference_assets=True,
    # 方舟允许 duration=-1，表示依据输入素材自动适配时长（edit 任务强制如此）。
    supports_auto_duration=True,
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

    使用官方文档列出的输出像素规格；adaptive 由模型根据参考素材决定，
    提交前无法可靠推导尺寸，故记录为 0×0。
    """
    if aspect == VideoAspect.adaptive:
        return 0, 0
    sizes = {
        "480p": {
            VideoAspect.landscape: (854, 480),
            VideoAspect.landscape_4_3: (752, 560),
            VideoAspect.square: (640, 640),
            VideoAspect.portrait_3_4: (560, 752),
            VideoAspect.portrait: (480, 854),
            VideoAspect.cinematic: (992, 432),
        },
        "720p": {
            VideoAspect.landscape: (1280, 720),
            VideoAspect.landscape_4_3: (1112, 834),
            VideoAspect.square: (960, 960),
            VideoAspect.portrait_3_4: (834, 1112),
            VideoAspect.portrait: (720, 1280),
            VideoAspect.cinematic: (1470, 630),
        },
        "1080p": {
            VideoAspect.landscape: (1920, 1080),
            VideoAspect.landscape_4_3: (1664, 1248),
            VideoAspect.square: (1440, 1440),
            VideoAspect.portrait_3_4: (1248, 1664),
            VideoAspect.portrait: (1080, 1920),
            VideoAspect.cinematic: (2206, 946),
        },
    }
    return sizes[resolution][aspect]


def _encode_first_frame(first_frame: FirstFrame) -> str:
    """把统一的 ``FirstFrame`` 转成方舟 ``image_url.url`` 接受的取值。

    * ``url`` —— 公网可访问的图片地址，直接提交。
    * ``local_path`` —— 本地文件读成 base64 data URL（方舟同时接受这种形态）。
    * ``asset_id`` —— 方舟素材库中的图片素材 ID，直接提交。
    """
    if first_frame.kind in {"url", "asset_id"}:
        return first_frame.value
    if first_frame.kind != "local_path":
        raise Seedance25Error(
            "Seedance 2.5 image-to-video accepts a URL, asset ID, or local "
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


# 每条参考素材在 content 里对应的 type 与 role。方舟用 role 区分用途：
# reference_image 提供外观/角色/商品/场景/风格，reference_video 提供动作、
# 运镜与表演，reference_audio 提供音色与节奏。
_REFERENCE_ROLES: dict[str, tuple[str, str]] = {
    "image": ("image_url", "reference_image"),
    "video": ("video_url", "reference_video"),
    "audio": ("audio_url", "reference_audio"),
}

# 单次请求的参考素材上限（官方：≤30 图、≤10 视频、≤10 音频）。
MAX_REFERENCE_COUNTS: dict[str, int] = {"image": 30, "video": 10, "audio": 10}


def _encode_reference_asset(asset: ReferenceAsset) -> str:
    """把参考素材转成方舟接受的地址形态。

    音频与视频通常体积较大，走 base64 data URL 会让请求体膨胀到不可用的
    量级；因此只有图片支持本地路径（内联为 data URL），视频与音频必须是
    公网可访问的 URL。
    """
    if asset.source in {"url", "asset_id"}:
        return asset.value

    if asset.kind == "video":
        raise Seedance25Error(
            "Seedance 2.5 requires a public URL or asset ID for video reference "
            "assets; local video files are not supported"
        )

    path = Path(asset.value)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise Seedance25Error(f"reference {asset.kind} is not readable: {path}") from exc
    if size <= 0:
        raise Seedance25Error(f"reference {asset.kind} is empty: {path}")
    limit = (
        MAX_REFERENCE_IMAGE_BYTES
        if asset.kind == "image"
        else MAX_REFERENCE_AUDIO_BYTES
    )
    if size > limit:
        raise Seedance25Error(
            f"reference {asset.kind} exceeds {limit} bytes: {path}"
        )
    default_mime = "image/png" if asset.kind == "image" else "audio/mpeg"
    mime = asset.mime or mimetypes.guess_type(path.name)[0] or default_mime
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Seedance25Error(f"reference {asset.kind} could not be read: {path}") from exc
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _build_content(
    term: str,
    first_frame: FirstFrame | None,
    reference_assets: Sequence[ReferenceAsset] | None = None,
) -> list[dict[str, Any]]:
    """构造方舟的 ``content`` 数组。

    顺序是「文本 → 首帧 → 参考素材」。首帧（first_frame）与参考图
    （reference_image）语义不同：前者决定视频从哪里开始，后者是"照着这个
    外观/风格/动作生成"。两者可以同时存在。
    """
    content: list[dict[str, Any]] = []
    if term:
        content.append({"type": "text", "text": term})

    if first_frame is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_first_frame(first_frame)},
                "role": "first_frame",
            }
        )

    assets = list(reference_assets or ())
    if assets:
        counts: dict[str, int] = {}
        for asset in assets:
            counts[asset.kind] = counts.get(asset.kind, 0) + 1
        for kind, count in counts.items():
            limit = MAX_REFERENCE_COUNTS.get(kind, 0)
            if count > limit:
                raise Seedance25Error(
                    f"too many {kind} reference assets: {count} > {limit}"
                )
        for asset in assets:
            if asset.kind not in _REFERENCE_ROLES:
                raise Seedance25Error(
                    f"unsupported Seedance 2.5 reference asset kind: {asset.kind!r}"
                )
            type_name, role = _REFERENCE_ROLES[asset.kind]
            content.append(
                {
                    "type": type_name,
                    type_name: {"url": _encode_reference_asset(asset)},
                    "role": role,
                }
            )

    return content


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: FirstFrame | None = None,
    reference_assets: Sequence[ReferenceAsset] | None = None,
    omni_reference_task_type: str = "",
) -> list[MaterialInfo]:
    """提交一个 Seedance 2.5 任务，并等待可下载的结果地址。

    ``first_frame`` 非空时走图生视频：方舟要求此时的 ``ratio`` 必须为
    ``adaptive``，由服务端自动跟随首帧宽高比，因此覆盖面宽高比。
    """
    api_key = get_api_key()
    if not api_key:
        raise Seedance25Error("Volcano Engine Seedance 2.5 requires an Ark API key")

    term = str(search_term or "").strip()
    assets = list(reference_assets or ())
    reference_task_type = str(omni_reference_task_type or "").strip().lower()
    # 全模态参考任务由任一种 reference_image / reference_video /
    # reference_audio 触发，因此参考音频可以作为唯一素材。
    has_reference_input = first_frame is not None or bool(assets)
    if not term and not has_reference_input:
        # 空提示词通常来自上游脚本拆分异常。付费生成源不能把无效输入提交到
        # 远端，否则即使接口接受也只会得到无法使用且已计费的视频。
        raise Seedance25Error(
            "Seedance 2.5 search term must not be empty unless an image or "
            "video/audio reference asset is supplied"
        )
    if assets and not reference_task_type:
        # 官方推荐显式传 auto，令服务端按提示词识别参考生成、编辑或延长。
        reference_task_type = "auto"
    if reference_task_type not in {"", *VALID_OMNI_REFERENCE_TASK_TYPES}:
        choices = ", ".join(sorted(VALID_OMNI_REFERENCE_TASK_TYPES))
        raise Seedance25Error(
            f"unsupported omni_reference_task_type {reference_task_type!r}; "
            f"expected one of: {choices}"
        )
    if reference_task_type and not assets:
        raise Seedance25Error(
            "omni_reference_task_type requires at least one reference asset"
        )

    aspect = VideoAspect(video_aspect)
    try:
        requested_duration = int(minimum_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Seedance25Error(
            "Seedance 2.5 clip duration must be -1 or a positive integer"
        ) from exc
    if requested_duration == -1:
        duration = -1
    elif requested_duration <= 0:
        raise Seedance25Error(
            "Seedance 2.5 clip duration must be -1 or a positive integer"
        )
    else:
        minimum, maximum = _duration_bounds()
        duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        logger.info(
            "Seedance 2.5 clip duration clamped to the configured model range: "
            f"requested={requested_duration}s, using={duration}s"
        )

    resolution = _resolution()
    has_reference_video = any(asset.kind == "video" for asset in assets)
    if reference_task_type in {"edit", "extend"} and not has_reference_video:
        raise Seedance25Error(
            f"omni_reference_task_type={reference_task_type!r} requires at "
            "least one reference_video asset"
        )
    if reference_task_type in {"edit", "extend"} and aspect != VideoAspect.adaptive:
        raise Seedance25Error(
            f"omni_reference_task_type={reference_task_type!r} requires "
            "ratio='adaptive'"
        )
    if reference_task_type == "edit" and duration != -1:
        raise Seedance25Error(
            "omni_reference_task_type='edit' requires duration=-1"
        )
    # 图生视频必须用 adaptive，否则方舟会拒绝请求。
    ratio = ADAPTIVE_RATIO if first_frame is not None else aspect.value
    payload: dict[str, Any] = {
        "model": _model_id(),
        "content": _build_content(term, first_frame, assets),
        "ratio": ratio,
        "duration": duration,
        "resolution": resolution,
        "watermark": _config_bool("seedance25_watermark", False),
    }
    # 原生音频：让模型直接产出带声音的视频。默认开启，因为这是选用 2.5 的
    # 主要理由；成片是否真正混入该音轨由剪辑阶段决定。
    if _config_bool("seedance25_generate_audio", True):
        payload["generate_audio"] = True
    if reference_task_type:
        # 方舟根据该值提前判定参考生视频/视频编辑/视频延长任务类型。
        payload["omni_reference_task_type"] = reference_task_type

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tasks_url = f"{_base_url()}/contents/generations/tasks"
    logger.info(
        "generating video with Volcano Engine Seedance 2.5: "
        f"model={payload['model']}, duration={duration}s, ratio={ratio}, "
        f"resolution={resolution}, image_to_video={first_frame is not None}, "
        f"references={len(assets)}, "
        f"reference_task_type={reference_task_type or 'auto'}, "
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
    if assets:
        source_info["reference_asset_count"] = len(assets)
        source_info["reference_asset_kinds"] = [asset.kind for asset in assets]
        source_info["omni_reference_task_type"] = reference_task_type

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

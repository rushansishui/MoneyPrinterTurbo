"""独立文生视频（text/image-to-video）能力。

该模块只负责「给一个提示词，产出一段视频」这一件事，与主视频流水线
（配音 → 字幕 → 剪辑 → 混音）完全解耦：

* API 端点与 WebUI 页面共用这里的实现，避免两套逻辑各自演进。
* 生成仍走 ``video_providers`` 注册表，因此新增一个模型不需要改这里。
* provider 的契约依然只返回 URL 元数据，落盘由本模块完成——付费产物必须
  复用带重试的下载路径，不能因为一次下载抖动就重新下单。

计费安全：调用方必须显式传 ``confirm_charge=True``。API 没有会话可以承载
WebUI 那样的会话态确认框，因此这个门禁是服务层的硬性前置条件。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material, video_providers
from app.services.video_providers.spec import FirstFrame, ReferenceAsset


class AIVideoStudioError(ValueError):
    """请求本身不合法（未知 provider、缺少确认、缺少凭据等）。"""


class AIVideoChargeConfirmationRequired(AIVideoStudioError):
    """未确认付费创建任务。"""


class AIVideoGenerationError(RuntimeError):
    """生成或落盘失败，携带远端任务 ID 便于找回。"""

    def __init__(self, message: str, *, provider_id: str = "", task_id: str = ""):
        super().__init__(message)
        self.provider_id = provider_id
        self.task_id = task_id


@dataclass(frozen=True, slots=True)
class GeneratedClip:
    """一段已经落盘的可播放产物。"""

    provider: str
    file_path: str
    duration: int
    width: int
    height: int
    remote_task_id: str = ""
    source_info: Mapping[str, Any] = field(default_factory=dict)


def list_providers() -> list[dict[str, Any]]:
    """列出可用于独立生成端点的 provider。

    同时返回能力标记，让调用方知道能否传首帧、能否生成原生音频，以及是否
    需要先确认费用——前端据此决定展示哪些控件，而不是靠硬编码。
    """
    providers = []
    for spec in video_providers.all_specs(group="ai_video"):
        providers.append(
            {
                "provider_id": spec.provider_id,
                "label": spec.default_label,
                "supports_image_to_video": spec.supports_image_to_video,
                "supports_native_audio": spec.supports_native_audio,
                "supports_reference_assets": spec.supports_reference_assets,
                "supports_auto_duration": spec.supports_auto_duration,
                "requires_charge_confirmation": spec.requires_charge_confirmation,
                "enabled": video_providers.is_enabled(spec.provider_id),
            }
        )
    return providers


def _resolve_spec(provider_id: str):
    normalized = str(provider_id or "").strip()
    if not normalized:
        raise AIVideoStudioError("provider is required")
    spec = video_providers.get_spec(normalized)
    if spec is None:
        raise AIVideoStudioError(f"unknown video provider: {normalized!r}")
    if spec.group != "ai_video":
        # 库存源与本地素材不属于「AI 生成」，不应出现在这个入口。
        raise AIVideoStudioError(
            f"{normalized!r} is not an AI video generation provider"
        )
    return spec


def _normalize_first_frame(value: Any) -> FirstFrame | None:
    """把请求里的首帧描述转成统一的 ``FirstFrame``。

    同时接受 ``FirstFrame`` 实例与字段字典，方便 API（Pydantic 模型）与
    WebUI（直接构造）共用同一条校验路径。
    """
    if value is None:
        return None
    if isinstance(value, FirstFrame):
        return value
    if isinstance(value, Mapping):
        return FirstFrame(
            kind=str(value.get("kind") or "").strip(),
            value=str(value.get("value") or "").strip(),
            mime=str(value.get("mime") or "").strip(),
            width=int(value.get("width") or 0),
            height=int(value.get("height") or 0),
            origin=str(value.get("origin") or "upload").strip() or "upload",
        )
    raise AIVideoStudioError("first_frame must be an object")


def _normalize_reference_assets(value: Any) -> list[ReferenceAsset]:
    """Convert API/WebUI reference-asset dictionaries to provider values."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise AIVideoStudioError("reference_assets must be a list")

    assets: list[ReferenceAsset] = []
    for index, item in enumerate(value):
        if isinstance(item, ReferenceAsset):
            asset = item
        elif isinstance(item, Mapping):
            kind = str(item.get("kind") or "").strip()
            source = str(item.get("source") or "").strip()
            if kind not in {"image", "video", "audio"}:
                raise AIVideoStudioError(
                    f"reference_assets[{index}].kind must be image, video, or audio"
                )
            if source not in {"local_path", "url", "asset_id"}:
                raise AIVideoStudioError(
                    f"reference_assets[{index}].source must be local_path, url, or asset_id"
                )
            try:
                asset = ReferenceAsset(
                    kind=kind,
                    source=source,
                    value=str(item.get("value") or "").strip(),
                    mime=str(item.get("mime") or "").strip(),
                )
            except ValueError as exc:
                raise AIVideoStudioError(
                    f"invalid reference_assets[{index}]: {exc}"
                ) from exc
        else:
            raise AIVideoStudioError(
                f"reference_assets[{index}] must be an object"
            )
        assets.append(asset)
    return assets


def _clip_dimensions(item: MaterialInfo, aspect: VideoAspect) -> tuple[int, int]:
    """优先使用 provider 回报的真实尺寸，缺失时回退画幅换算值。

    ``adaptive`` 没有固定分辨率，provider 也没回报尺寸时只能返回 (0, 0)，
    由调用方决定是否回读文件真实尺寸。
    """
    source_info = item.source_info if isinstance(item.source_info, dict) else {}
    rendition = source_info.get("rendition")
    if isinstance(rendition, dict):
        try:
            width = int(rendition.get("width") or 0)
            height = int(rendition.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0
        if width > 0 and height > 0:
            return width, height
    try:
        return aspect.to_resolution()
    except ValueError:
        return 0, 0


def _probe_downloaded_clip(file_path: str) -> tuple[int, int, int]:
    """读取已下载文件的行之有效的时长与尺寸。

    只在 provider 没有给出可用元数据时调用：``duration=-1``（自动时长）和
    ``adaptive`` 画幅都会让上游回报 ``-1`` / ``0x0``，直接透传会让 API 调用方
    拿到负数时长与零尺寸。探测失败不抛异常——视频已经付费生成并下载完成，
    不能因为读不出元数据就丢掉产物。
    """
    clip = None
    try:
        clip = material.VideoFileClip(file_path)
        duration = float(clip.duration or 0)
        width, height = clip.size
        return (
            int(round(duration)) if duration > 0 else 0,
            int(width or 0),
            int(height or 0),
        )
    except Exception as exc:
        logger.warning(
            "could not probe generated video metadata, leaving it unset: "
            f"path={file_path}, error={type(exc).__name__}, detail={exc}"
        )
        return 0, 0, 0
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    "failed to close generated video after metadata probe: "
                    f"path={file_path}, error={type(close_error).__name__}, "
                    f"detail={close_error}"
                )


def _resolve_clip_metadata(
    item: MaterialInfo,
    aspect: VideoAspect,
    requested_duration: int,
    saved_path: str,
) -> tuple[int, int, int]:
    """把 provider 回报的元数据归一化成可用的 ``(duration, width, height)``。

    上游可能给出三种不可用取值：``duration=-1``（自动时长）、``None``（未回报）
    以及 ``0x0`` 尺寸。任一出现时才回读文件，避免为每段视频都多开一次解码器。
    """
    reported_duration = int(item.duration or 0)
    width, height = _clip_dimensions(item, aspect)
    needs_probe = reported_duration <= 0 or width <= 0 or height <= 0
    probed_duration, probed_width, probed_height = (
        _probe_downloaded_clip(saved_path) if needs_probe else (0, 0, 0)
    )
    if reported_duration > 0:
        duration = reported_duration
    elif probed_duration > 0:
        duration = probed_duration
    elif requested_duration > 0:
        duration = requested_duration
    else:
        duration = 0
    return (
        duration,
        width or probed_width,
        height or probed_height,
    )


def generate_and_save(
    *,
    provider_id: str,
    prompt: str,
    duration: int,
    video_aspect: VideoAspect | str = VideoAspect.portrait,
    first_frame: Any = None,
    reference_assets: Any = None,
    omni_reference_task_type: str = "",
    confirm_charge: bool = False,
    save_dir: str = "",
) -> list[GeneratedClip]:
    """生成一段（或 provider 返回的多段）视频并落盘。

    参数校验 → 计费确认 → 凭据检查 → 生成 → 带重试下载，任何一步失败都
    在落盘前停下，避免产生半成品。
    """
    spec = _resolve_spec(provider_id)

    term = str(prompt or "").strip()
    try:
        clip_duration = int(duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AIVideoStudioError("duration must be a positive integer") from exc
    if clip_duration == 0 or clip_duration < -1:
        raise AIVideoStudioError("duration must be -1 or a positive integer")
    aspect = VideoAspect(video_aspect)
    frame = _normalize_first_frame(first_frame)
    assets = _normalize_reference_assets(reference_assets)
    reference_task_type = str(omni_reference_task_type or "").strip().lower()
    if frame is not None and not spec.supports_image_to_video:
        raise AIVideoStudioError(
            f"{spec.provider_id} does not support image-to-video generation"
        )
    if assets and not spec.supports_reference_assets:
        raise AIVideoStudioError(
            f"{spec.provider_id} does not support reference assets"
        )
    if reference_task_type and not assets:
        raise AIVideoStudioError(
            "omni_reference_task_type requires at least one reference asset"
        )
    if assets and not reference_task_type:
        reference_task_type = "auto"
    if reference_task_type not in {"", "auto", "edit", "extend"}:
        raise AIVideoStudioError(
            "omni_reference_task_type must be auto, edit, or extend"
        )
    # -1 是方舟的「自动时长」语义。其余 provider 会把它夹到 1 秒后照常下单，
    # 必须在提交前拒绝，而不是让调用方拿到一段无法使用的付费视频。
    if clip_duration == -1 and not spec.supports_auto_duration:
        raise AIVideoStudioError(
            f"{spec.provider_id} does not support automatic duration (-1)"
        )

    # Seedance 2.5 的「参考生视频」由任一种参考素材触发，参考音频单独使用
    # 也是合法组合；没有参考素材时才要求文本提示词。
    has_reference_input = frame is not None or bool(assets)
    if not term and not has_reference_input:
        raise AIVideoStudioError(
            "provide a prompt or at least one reference asset"
        )

    # 计费确认必须发生在任何凭据读取与远端请求之前。
    if spec.requires_charge_confirmation and not confirm_charge:
        raise AIVideoChargeConfirmationRequired(
            f"{spec.provider_id} creates a paid task; set confirm_charge=true "
            "to proceed"
        )

    if not video_providers.is_enabled(spec.provider_id):
        raise AIVideoStudioError(
            spec.missing_credentials_message
            or f"{spec.provider_id} requires an API key"
        )

    logger.info(
        f"standalone ai video generation: provider={spec.provider_id}, "
        f"duration={clip_duration}s, aspect={aspect.value}, "
        f"image_to_video={frame is not None}, references={len(assets)}, "
        f"reference_task_type={reference_task_type or 'auto'}"
    )

    try:
        items = video_providers.generate(
            spec.provider_id,
            search_term=term,
            minimum_duration=clip_duration,
            video_aspect=aspect,
            first_frame=frame,
            reference_assets=assets,
            omni_reference_task_type=reference_task_type,
        )
    except video_providers.VideoProviderError as exc:
        raise AIVideoGenerationError(
            str(exc),
            provider_id=spec.provider_id,
            task_id=str(getattr(exc, spec.remote_task_id_attr, "") or ""),
        ) from exc

    if not items:
        raise AIVideoGenerationError(
            f"{spec.provider_id} produced no video for this prompt",
            provider_id=spec.provider_id,
        )

    clips: list[GeneratedClip] = []
    for item in items:
        remote_task_id = str(
            (item.source_info or {}).get("asset_id") or ""
        ).strip()
        saved_path = material._save_generated_video_with_retry(
            item.url, save_dir, spec.provider_id
        )
        if not saved_path:
            # 远端已经生成并计费，下载失败时必须把 ID 交回调用方。
            raise AIVideoGenerationError(
                f"{spec.provider_id} generated a paid video but the result "
                f"could not be downloaded: id={remote_task_id or 'unknown'}",
                provider_id=spec.provider_id,
                task_id=remote_task_id,
            )
        duration, width, height = _resolve_clip_metadata(
            item, aspect, clip_duration, saved_path
        )
        clips.append(
            GeneratedClip(
                provider=spec.provider_id,
                file_path=saved_path,
                duration=duration,
                width=width,
                height=height,
                remote_task_id=remote_task_id,
                source_info=dict(item.source_info or {}),
            )
        )

    logger.success(
        f"standalone ai video generation finished: provider={spec.provider_id}, "
        f"clips={len(clips)}"
    )
    return clips

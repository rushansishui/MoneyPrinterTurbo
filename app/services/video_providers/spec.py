"""视频 Provider 的声明层：能力与元数据。

沿用 ``app/models/llm_provider.py`` 的 ``LLMProviderSpec`` 范式——集中声明
「这个源是什么、支持什么能力」，不实现「怎么调用」。调用实现留在
``sources/`` 下各自的模块里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


def _normalize_identifier(value: Any) -> str:
    """把配置里的标识符归一化成 ``snake_case`` 小写形式。"""
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("-", "_").replace(" ", "_").lower()


@dataclass(frozen=True, slots=True)
class FirstFrame:
    """图生视频（i2v）的首帧描述。

    统一的抽象让 provider 只需关心「怎么把这张图交给自己的 API」，而不必
    关心它来自本地上传、文生图产出还是库存图库。具体的编码方式（base64
    data URI / multipart 上传 / 预签名 URL）由各 provider 按官方文档实现。
    """

    kind: Literal["local_path", "url", "asset_id"]
    value: str
    mime: str = ""
    width: int = 0
    height: int = 0
    origin: Literal["upload", "generated", "stock"] = "upload"

    def __post_init__(self) -> None:
        normalized_value = str(self.value or "").strip()
        if not normalized_value:
            raise ValueError("first frame value must not be empty")
        # frozen+slots 的 dataclass 仍需用 object.__setattr__ 归一化字段。
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(self, "mime", str(self.mime or "").strip())


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    """一条参考素材（图片 / 视频 / 音频）。

    Seedance 2.5 的「全模态参考」允许在同一个请求里混合多种参考素材：图片
    提供外观与风格、视频提供动作与运镜、音频提供音色与节奏。它们与首帧是
    不同概念——首帧是"从哪里开始"，参考素材是"照着什么做"。

    与 :class:`FirstFrame` 一样，这里只描述素材是什么，具体如何编码（base64
    data URL / 公网 URL / 预签名地址）由各 provider 按自己的协议实现。
    """

    kind: Literal["image", "video", "audio"]
    # 素材位置形态：本地文件路径或公网 URL。
    source: Literal["local_path", "url", "asset_id"]
    value: str
    mime: str = ""

    def __post_init__(self) -> None:
        normalized_value = str(self.value or "").strip()
        if not normalized_value:
            raise ValueError("reference asset value must not be empty")
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(self, "mime", str(self.mime or "").strip())


@dataclass(frozen=True, slots=True)
class VideoProviderSpec:
    """一个视频源的静态元数据。

    字段刻意保持精简：只放「会驱动分派、UI 与校验」的信息，其余细节
    （端点、状态枚举、时长边界）留在各自模块里，避免注册表变成一个
    需要频繁维护的大表。
    """

    provider_id: str
    default_label: str
    module: str
    # 旧模块路径。迁移期保留 re-export 垫片，让既有测试与外部调用继续工作。
    legacy_module: str = ""
    group: str = "ai_video"
    # 是否需要在创建付费任务前要求用户显式确认。
    requires_charge_confirmation: bool = True
    supports_image_to_video: bool = False
    supports_native_audio: bool = False
    # 是否支持「参考素材」（reference_image / reference_video / reference_audio
    # 混用）。与首帧是不同概念：首帧决定起点，参考素材决定照着什么生成。
    supports_reference_assets: bool = False
    # 是否支持 ``duration=-1``（由模型依据输入素材自动决定时长）。多数源把
    # 时长当正整数处理，直接透传 -1 会被它们静默夹到 1 秒并照常下单。
    supports_auto_duration: bool = False
    # 下载时长核算策略：False 用 provider 回报的时长，True 用本地实测文件时长。
    measure_downloaded_duration: bool = False
    # 远端任务 ID 在异常对象上的属性名与写入失败详情时的键名。
    remote_task_id_attr: str = "task_id"
    error_details_key: str = ""
    # 缺少凭据时的 preflight 提示文案。为空表示该源不参与 preflight 的凭据
    # 检查（例如 WaveSpeed 历史上没有这道预检，保持原行为不变）。文案逐字保留
    # 原有措辞，因为既有测试与用户排障信息都依赖它。
    missing_credentials_message: str = ""
    # 模块级「异常类名后缀 → 语义」映射，供注册表做统一异常归类。
    error_class_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_id = _normalize_identifier(self.provider_id)
        if not normalized_id:
            raise ValueError("provider_id must not be empty")
        if not str(self.module or "").strip():
            raise ValueError(
                f"provider {normalized_id!r} must declare an importable module"
            )
        object.__setattr__(self, "provider_id", normalized_id)
        object.__setattr__(self, "default_label", str(self.default_label or normalized_id))

    def config_key(self, suffix: str) -> str:
        """派生配置键，如 ``config_key("api_key") -> "ofox_api_key"``。"""
        return f"{self.provider_id}_{suffix}"

    @property
    def label_key(self) -> str:
        """WebUI i18n 用的 label 键。"""
        return f"video_provider_label.{self.provider_id}"

    @property
    def confirm_charge_key(self) -> str:
        """WebUI 会话态里的付费确认复选框键。"""
        return f"{self.provider_id}_confirm_charge"

    @property
    def resolved_error_details_key(self) -> str:
        """写失败详情时使用的键名，默认 ``<provider_id>_task_id``。"""
        return self.error_details_key or f"{self.provider_id}_task_id"

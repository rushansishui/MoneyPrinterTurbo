"""视频 Provider 注册表：把 provider_id 解析成可调用的模块。

设计要点：

* **显式列表注册**，不做 ``pkgutil`` 扫描。启动成本可控、import 顺序确定，
  也避免自动发现时 sources 反向 import registry 造成的循环导入。
* **惰性导入**：``resolve()`` 首次访问某 provider 时才 import 其模块，把
  ``requests`` 等重依赖的加载推迟到真正需要时。
* ``sources/`` 下的模块只允许 import ``spec`` / ``polling`` / ``http_util`` /
  ``errors``，不得 import 本模块。
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from app.services.video_providers.spec import FirstFrame, VideoProviderSpec


# 注册顺序同时决定 WebUI 中的展示顺序。
_REGISTRY: list[VideoProviderSpec] = []
_BY_ID: dict[str, VideoProviderSpec] = {}
_MODULES: dict[str, ModuleType] = {}
_BUILTINS_LOADED = False

# 内置 provider 模块。显式列出而不做 pkgutil 扫描：启动成本可控、import
# 顺序确定，也避免自动发现时 sources 反向 import 本模块造成循环导入。
_BUILTIN_MODULES: tuple[str, ...] = (
    "app.services.video_providers.sources.seedance",
    "app.services.video_providers.sources.ofox",
    "app.services.video_providers.sources.metaso_minimax",
    "app.services.video_providers.sources.muapi",
    "app.services.video_providers.sources.wavespeed",
)


def _ensure_builtins() -> None:
    """首次访问注册表时导入内置 provider 模块。

    各 ``sources/*.py`` 在模块底部调用 :func:`register`，因此这里只需 import。
    惰性执行让「只 import 注册表但不查 provider」的场景（例如仅用 http_util
    的工具函数）不必加载 requests 等重依赖。单个源导入失败不应让注册表整体
    不可用，问题写入日志便于定位。
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    from loguru import logger

    for module_name in _BUILTIN_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # 尚未落地的源在迁移期属于正常状态（例如 P0 阶段 sources/ 为空）。
            # 用 debug 记录避免每次查询注册表都刷屏，同时仍可开启日志排查。
            logger.debug(f"video provider module not present yet: {module_name}")
            continue
        except Exception:  # pragma: no cover - 仅在源模块损坏时触发
            logger.opt(exception=True).error(
                f"failed to register video provider module {module_name}"
            )
            continue

        # 由注册表读取模块级 SPEC 并登记。source 模块因此不需要 import 本模块，
        # 依赖方向保持单向（sources -> spec），不会出现循环导入。
        module_spec = getattr(module, "SPEC", None)
        if isinstance(module_spec, VideoProviderSpec):
            register(module_spec)
        else:  # pragma: no cover - 仅在源模块漏写 SPEC 时触发
            logger.error(f"video provider module {module_name} declares no SPEC")


def register(spec: VideoProviderSpec) -> VideoProviderSpec:
    """登记一个 provider。重复 ``provider_id`` 会直接报错，避免静默覆盖。"""
    if spec.provider_id in _BY_ID:
        raise ValueError(f"duplicate video provider id: {spec.provider_id!r}")
    _REGISTRY.append(spec)
    _BY_ID[spec.provider_id] = spec
    return spec


def get_spec(provider_id: str) -> VideoProviderSpec | None:
    """按 ID 取 spec；未知 ID 返回 ``None``（调用方决定报错方式）。"""
    _ensure_builtins()
    return _BY_ID.get(str(provider_id or "").strip())


def all_specs(group: str | None = None) -> tuple[VideoProviderSpec, ...]:
    """返回全部 spec，或某一分组的 spec，保持注册顺序。"""
    _ensure_builtins()
    if group is None:
        return tuple(_REGISTRY)
    return tuple(spec for spec in _REGISTRY if spec.group == group)


def all_provider_ids(group: str | None = None) -> frozenset[str]:
    """全部（或某分组的）provider_id，供白名单与 choices 生成。"""
    return frozenset(spec.provider_id for spec in all_specs(group))


def resolve(provider_id: str) -> ModuleType:
    """解析 provider 对应的模块对象，带缓存。未知 ID 抛 ``KeyError``。

    声明了 ``legacy_module`` 时**优先**返回该模块（即 ``app/services/<x>.py``
    的兼容垫片）。原因：这些路径是既有的公开地址，调用方与测试都按它打桩
    （``patch.object(metaso_minimax, "is_enabled")``）。若这里改解析到
    ``sources/<x>.py``，patch 会打在另一个模块对象上而静默失效，测试会在
    「看似通过」的同时真的去读全局配置、甚至发出真实请求。
    垫片本身 re-export 了 ``sources`` 的实现，因此两条路径行为一致。
    """
    spec = get_spec(provider_id)
    if spec is None:
        raise KeyError(f"unknown video provider: {provider_id!r}")
    cached = _MODULES.get(spec.provider_id)
    if cached is not None:
        return cached
    module = importlib.import_module(spec.legacy_module or spec.module)
    _MODULES[spec.provider_id] = module
    return module


def is_enabled(
    provider_id: str, settings: Mapping[str, Any] | None = None
) -> bool:
    """该 provider 是否具备可用的凭据。

    未注册的 ID 返回 ``False`` 而不是抛错：preflight 与 WebUI 门禁都会用它
    做「能不能生成」的判断，未知源在此一律视为不可用。
    """
    spec = get_spec(provider_id)
    if spec is None:
        return False
    module = resolve(provider_id)
    checker = getattr(module, "is_enabled", None)
    if not callable(checker):
        return False
    return bool(checker(settings))


def generate(
    provider_id: str,
    *,
    search_term: str,
    minimum_duration: int,
    video_aspect: Any,
    first_frame: FirstFrame | None = None,
):
    """调用 provider 的 ``generate_videos``，并统一执行能力校验。

    不支持图生视频的 provider 在拿到 ``first_frame`` 时直接拒绝，避免把
    首帧静默丢弃后仍然发起一次付费任务。
    """
    spec = get_spec(provider_id)
    if spec is None:
        raise KeyError(f"unknown video provider: {provider_id!r}")
    if first_frame is not None and not spec.supports_image_to_video:
        raise ValueError(
            f"{spec.provider_id} does not support image-to-video generation"
        )
    module = resolve(provider_id)
    return module.generate_videos(
        search_term,
        minimum_duration,
        video_aspect,
        first_frame=first_frame,
    )


def _reset_for_tests() -> None:
    """清空注册状态。仅供测试在隔离注册表时使用。"""
    global _BUILTINS_LOADED
    _REGISTRY.clear()
    _BY_ID.clear()
    _MODULES.clear()
    _BUILTINS_LOADED = False

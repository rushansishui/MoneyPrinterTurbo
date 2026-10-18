"""Provider 注册表与各处分派清单的一致性护栏。

这些断言把「重构的价值」固化成回归护栏：新增一个视频源时，如果有哪个清单
被漏掉，会在这里立刻失败，而不是等到用户遇到「下拉里看不到 / CLI 报错 /
preflight 拒绝」才发现。

同时断言 ``material.download_videos`` 不再按 provider 逐个 ``if source ==``
分派——否则重构会悄悄退回成"每加一个源就要改这一处"的旧形态。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from app.services import material, task, video_providers
from app.services.video_providers import registry


ROOT = Path(__file__).resolve().parents[2]
WEBUI_MAIN = ROOT / "webui" / "Main.py"
CLI_FILE = ROOT / "cli.py"


def _literal_assignment(source_path: Path, name: str):
    """从模块源码里取出某个顶层字面量赋值的值。

    WebUI 与 CLI 的来源清单是用户可见的顺序声明，既有测试刻意按字面量固定
    它们（见 test_webui_ofox_placement.py）。这里用同样的方式读取，才能断言
    它们没有和注册表脱节。
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {source_path}")


def test_task_whitelist_covers_every_registered_provider():
    """task 的白名单必须包含注册表里的全部 provider。"""
    missing = video_providers.all_provider_ids() - task._SUPPORTED_VIDEO_SOURCES
    assert not missing, f"task whitelist is missing registered providers: {missing}"


def test_cli_video_sources_cover_every_registered_provider():
    """CLI 的 --video-source choices 必须包含注册表里的全部 provider。"""
    cli_sources = set(_literal_assignment(CLI_FILE, "_CLI_VIDEO_SOURCES"))
    missing = video_providers.all_provider_ids() - cli_sources
    assert not missing, f"cli --video-source is missing providers: {missing}"


def test_webui_groups_cover_every_registered_provider():
    """WebUI 的来源分组必须包含注册表里的全部 provider。"""
    groups = _literal_assignment(WEBUI_MAIN, "VIDEO_SOURCE_GROUPS")
    listed = {value for values in groups.values() for value in values}
    missing = video_providers.all_provider_ids() - listed
    assert not missing, f"webui VIDEO_SOURCE_GROUPS is missing providers: {missing}"


def test_material_dispatch_has_no_per_provider_if_branches():
    """material 的生成源分派必须是注册表查找，而不是逐源 if。

    只检查 ``download_videos`` 的函数体：其它地方仍可能按来源判断（例如
    本地素材与文生图各有专门流程），那些不属于"每加一个源就要改"的分派。
    这里只统计**生成源**的名字，避免把 pexels/pixabay/coverr/local 这类
    库存源分支误判成违规。
    """
    source = inspect.getsource(material.download_videos)
    tree = ast.parse(inspect.cleandoc(source))
    generated_names = set(material._GENERATED_VIDEO_PROVIDERS)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "source":
            continue
        for comparator in node.comparators:
            if (
                isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
                and comparator.value in generated_names
            ):
                offenders.append(comparator.value)
    assert not offenders, (
        "download_videos still dispatches per generated provider by name; "
        f"route them through the registry instead: {sorted(set(offenders))}"
    )


def test_every_registered_provider_has_orchestration_entry():
    """每个注册的 provider 都要在 material 的编排表里有对应条目。

    否则该源能通过 preflight，却在实际下载素材时无人接手。
    """
    table = material._GENERATED_VIDEO_PROVIDERS
    registered = video_providers.all_provider_ids()
    assert registered == set(table), (
        "material orchestration table and provider registry disagree: "
        f"only-in-registry={registered - set(table)}, "
        f"only-in-table={set(table) - registered}"
    )


def test_every_registered_provider_declares_spec_and_public_functions():
    """每个注册的 provider 都能解析到模块，且声明了 SPEC 与三个公开函数。"""
    for provider_id in sorted(video_providers.all_provider_ids()):
        module = video_providers.resolve(provider_id)
        spec = getattr(module, "SPEC", None)
        assert spec is not None, f"{provider_id} module declares no SPEC"
        assert spec.provider_id == provider_id
        for required in ("get_api_key", "is_enabled", "generate_videos"):
            assert callable(getattr(module, required, None)), (
                f"{provider_id} is missing required callable {required}"
            )


def test_every_registered_provider_declares_provider_id_on_error_base():
    """错误基类要携带 provider_id，任务层才能从异常反查 spec 写恢复信息。"""
    for provider_id in sorted(video_providers.all_provider_ids()):
        module = video_providers.resolve(provider_id)
        error_base = None
        for name in dir(module):
            if not name.endswith("Error"):
                continue
            candidate = getattr(module, name)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, video_providers.VideoProviderError)
                and getattr(candidate, "provider_id", "") == provider_id
            ):
                error_base = candidate
                break
        assert error_base is not None, (
            f"{provider_id} exposes no error class carrying provider_id"
        )


def test_registry_lookup_is_stable():
    """重复查询注册表结果稳定，且不产生重复登记。"""
    first = video_providers.all_specs()
    second = video_providers.all_specs()
    assert first == second
    assert len({spec.provider_id for spec in second}) == len(second)


def test_registry_reports_unknown_provider_as_not_enabled():
    """未注册的来源一律视为不可用，而不是抛错。"""
    assert registry.is_enabled("definitely-not-a-provider") is False
    assert registry.get_spec("definitely-not-a-provider") is None

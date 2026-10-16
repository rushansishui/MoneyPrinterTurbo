"""Seedance 2.5 provider 测试。

覆盖三块相对其它源新增的行为：

1. **图生视频（i2v）** —— ``content`` 追加 ``image_url`` + ``role=first_frame``，
   且 ``ratio`` 必须切换成 ``adaptive``（方舟强制要求，传具体比例会被拒）。
   三种首帧形态（公网 URL / 本地路径 base64 / 不支持的 asset_id）分别验证。
2. **原生音频** —— ``generate_audio`` 跟随配置，关闭时字段不能出现在请求体里，
   避免"以为关了其实还在按带声音的视频计费"。
3. **2.5 参数区间** —— 分辨率白名单新增 4k，时长上限放宽到 30 秒。

全部用 mock 替换 requests/time，CI 不依赖真实网络与计费。
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import material
from app.services.video_providers import errors
from app.services.video_providers.sources import seedance25


PROVIDER = "seedance25"


def _response(payload, status_code: int = 200, headers=None):
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        json=lambda: payload,
    )


@pytest.fixture(autouse=True)
def restore_config():
    """隔离全局配置，避免测试之间互相污染。"""
    original_app = dict(config.app)
    original_proxy = dict(config.proxy)
    config.app.update(
        {
            "seedance25_api_key": "ark-25-test-key",
            "seedance25_base_url": seedance25.DEFAULT_BASE_URL,
            "seedance25_resolution": seedance25.DEFAULT_RESOLUTION,
        }
    )
    config.proxy.clear()
    yield
    config.app.clear()
    config.app.update(original_app)
    config.proxy.clear()
    config.proxy.update(original_proxy)


def _completed_task(video_url: str = "https://cdn.test/out.mp4"):
    return _response(
        {"id": "task-1", "status": "succeeded", "content": {"video_url": video_url}}
    )


# ---------------------------------------------------------------------------
# 文生视频
# ---------------------------------------------------------------------------


def test_text_to_video_submits_expected_payload_and_parses_result():
    submit = _response({"id": "task-1"})
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=submit
        ) as post,
        patch.object(
            seedance25.polling.requests, "get", return_value=_completed_task()
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        items = seedance25.generate_videos(
            "sunrise over mountains",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MaterialInfo)
    assert item.provider == PROVIDER
    assert item.url == "https://cdn.test/out.mp4"
    assert item.duration == 5
    assert item.source_info["asset_id"] == "task-1"
    assert item.source_info["rendition"]["width"] == 1080
    assert item.source_info["rendition"]["height"] == 1920
    # 文生视频不应记录首帧字段。
    assert "first_frame_kind" not in item.source_info

    payload = post.call_args.kwargs["json"]
    assert payload["model"] == seedance25.DEFAULT_MODEL_ID
    assert payload["content"] == [
        {"type": "text", "text": "sunrise over mountains"}
    ]
    assert payload["ratio"] == "9:16"
    assert payload["duration"] == 5
    # 2.5 默认开启原生音频。
    assert payload["generate_audio"] is True
    assert post.call_args.kwargs["timeout"] == (30, 60)
    assert post.call_args.kwargs["allow_redirects"] is False


def test_text_to_video_can_disable_native_audio():
    config.app["seedance25_generate_audio"] = False
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=_response({"id": "t"})
        ) as post,
        patch.object(
            seedance25.polling.requests, "get", return_value=_completed_task()
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        seedance25.generate_videos("x", minimum_duration=5)

    assert "generate_audio" not in post.call_args.kwargs["json"]


def test_missing_api_key_fails_before_submission():
    config.app["seedance25_api_key"] = ""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(seedance25.client.requests, "post") as post,
    ):
        with pytest.raises(seedance25.Seedance25Error, match="API key"):
            seedance25.generate_videos("x", minimum_duration=5)
    post.assert_not_called()


def test_empty_prompt_fails_before_submission():
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="must not be empty"):
            seedance25.generate_videos("   ", minimum_duration=5)
    post.assert_not_called()


def test_duration_is_clamped_to_the_2_5_range():
    # 请求 2 秒低于 2.5 的 4 秒下限；请求 99 秒高于 30 秒上限。
    for requested, expected in ((2, 4), (99, 30)):
        with (
            patch.object(
                seedance25.client.requests, "post", return_value=_response({"id": "t"})
            ) as post,
            patch.object(
                seedance25.polling.requests, "get", return_value=_completed_task()
            ),
            patch.object(seedance25.polling.time, "sleep"),
        ):
            seedance25.generate_videos("x", minimum_duration=requested)
        assert post.call_args.kwargs["json"]["duration"] == expected


def test_resolution_whitelist_includes_4k_and_rejects_unknown_values():
    config.app["seedance25_resolution"] = "4k"
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=_response({"id": "t"})
        ) as post,
        patch.object(
            seedance25.polling.requests, "get", return_value=_completed_task()
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        items = seedance25.generate_videos("x", minimum_duration=5)
    assert post.call_args.kwargs["json"]["resolution"] == "4k"
    # 4k 长边按 3840 记录。
    assert items[0].source_info["rendition"]["width"] == 2160
    assert items[0].source_info["rendition"]["height"] == 3840

    config.app["seedance25_resolution"] = "8k"
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="Unsupported"):
            seedance25.generate_videos("x", minimum_duration=5)
    post.assert_not_called()


def test_task_success_without_video_url_is_an_error():
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=_response({"id": "t"})
        ),
        patch.object(
            seedance25.polling.requests,
            "get",
            return_value=_response({"id": "t", "status": "succeeded", "content": {}}),
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        with pytest.raises(seedance25.Seedance25Error, match="downloadable video"):
            seedance25.generate_videos("x", minimum_duration=5)


# ---------------------------------------------------------------------------
# 图生视频（i2v）
# ---------------------------------------------------------------------------


def test_image_to_video_from_public_url_uses_adaptive_ratio():
    frame = seedance25.FirstFrame(
        kind="url", value="https://img.test/a.png", origin="stock"
    )
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=_response({"id": "t"})
        ) as post,
        patch.object(
            seedance25.polling.requests, "get", return_value=_completed_task()
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        items = seedance25.generate_videos(
            "zoom in slowly", minimum_duration=5, first_frame=frame
        )

    payload = post.call_args.kwargs["json"]
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://img.test/a.png"},
        "role": "first_frame",
    }
    # 方舟要求首帧生视频时 ratio 必须为 adaptive。
    assert payload["ratio"] == "adaptive"
    assert items[0].source_info["first_frame_kind"] == "url"
    assert items[0].source_info["first_frame_origin"] == "stock"


def test_image_to_video_from_local_path_encodes_base64_data_url(tmp_path):
    image = tmp_path / "frame.png"
    raw = b"\x89PNG\r\n\x1a\n fake png bytes"
    image.write_bytes(raw)

    frame = seedance25.FirstFrame(
        kind="local_path", value=str(image), origin="upload"
    )
    with (
        patch.object(
            seedance25.client.requests, "post", return_value=_response({"id": "t"})
        ) as post,
        patch.object(
            seedance25.polling.requests, "get", return_value=_completed_task()
        ),
        patch.object(seedance25.polling.time, "sleep"),
    ):
        seedance25.generate_videos("x", minimum_duration=5, first_frame=frame)

    url = post.call_args.kwargs["json"]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_image_to_video_rejects_unsupported_asset_id_kind():
    frame = seedance25.FirstFrame(kind="asset_id", value="asset-1")
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="public image URL"):
            seedance25.generate_videos("x", minimum_duration=5, first_frame=frame)
    post.assert_not_called()


def test_image_to_video_rejects_missing_and_empty_local_files(tmp_path):
    missing = seedance25.FirstFrame(
        kind="local_path", value=str(tmp_path / "nope.png")
    )
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="not readable"):
            seedance25.generate_videos("x", minimum_duration=5, first_frame=missing)
    post.assert_not_called()

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    frame = seedance25.FirstFrame(kind="local_path", value=str(empty))
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="empty"):
            seedance25.generate_videos("x", minimum_duration=5, first_frame=frame)
    post.assert_not_called()


def test_image_to_video_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(seedance25, "MAX_FIRST_FRAME_BYTES", 4)
    image = tmp_path / "big.png"
    image.write_bytes(b"12345")
    frame = seedance25.FirstFrame(kind="local_path", value=str(image))
    with patch.object(seedance25.client.requests, "post") as post:
        with pytest.raises(seedance25.Seedance25Error, match="exceeds"):
            seedance25.generate_videos("x", minimum_duration=5, first_frame=frame)
    post.assert_not_called()


# ---------------------------------------------------------------------------
# 契约与编排接入
# ---------------------------------------------------------------------------


def test_spec_declares_new_capabilities():
    assert seedance25.SPEC.supports_image_to_video is True
    assert seedance25.SPEC.supports_native_audio is True
    assert seedance25.SPEC.provider_id == PROVIDER
    # 新源没有历史模块路径，因此不设置 legacy_module。
    assert seedance25.SPEC.legacy_module == ""


def test_error_classes_are_marked_for_the_orchestration_layer():
    assert issubclass(
        seedance25.Seedance25UnconfirmedTaskError,
        errors.VideoProviderUnconfirmedTaskError,
    )
    assert issubclass(
        seedance25.Seedance25DownloadError, errors.VideoProviderDownloadError
    )
    # 任务层靠这个属性反查 spec 写恢复信息。
    assert seedance25.Seedance25Error.provider_id == PROVIDER


def test_submission_is_never_retried_on_ambiguous_response():
    with patch.object(
        seedance25.client.requests,
        "post",
        side_effect=requests.exceptions.Timeout("slow"),
    ) as post:
        with pytest.raises(seedance25.Seedance25UnconfirmedTaskError):
            seedance25.generate_videos("x", minimum_duration=5)
    assert post.call_count == 1


def test_material_orchestration_generates_and_counts_coverage(tmp_path):
    """主流水线按 seedance25 生成素材并在覆盖配音时长后停止。"""
    item = MaterialInfo(
        provider=PROVIDER,
        url="https://cdn.test/a.mp4",
        duration=5,
        source_info={"asset_id": "task-1"},
    )
    with (
        patch.object(seedance25, "generate_videos", return_value=[item]) as generate,
        patch.object(
            material, "_save_generated_video_with_retry", return_value="/tmp/a.mp4"
        ),
        patch.object(material, "_persist_material_sources"),
    ):
        paths = material.download_videos(
            task_id="t",
            search_terms=["one", "two"],
            source=PROVIDER,
            video_aspect=VideoAspect.portrait,
            audio_duration=5,
            max_clip_duration=5,
        )

    assert paths == ["/tmp/a.mp4"]
    # 5 秒正好覆盖要求的 5 秒，不该再为第二个关键词下单。
    assert generate.call_count == 1


def test_material_orchestration_propagates_unconfirmed_with_task_id():
    pending = {"count": 0}

    def fake_generate(search_term, minimum_duration, video_aspect, first_frame=None):
        pending["count"] += 1
        raise seedance25.Seedance25UnconfirmedTaskError(
            "state unknown", task_id="task-stuck"
        )

    with (
        patch.object(seedance25, "generate_videos", side_effect=fake_generate),
        patch.object(material, "_persist_material_sources"),
    ):
        with pytest.raises(seedance25.Seedance25UnconfirmedTaskError) as raised:
            material.download_videos(
                task_id="t",
                search_terms=["one", "two", "three"],
                source=PROVIDER,
                audio_duration=30,
                max_clip_duration=5,
            )

    assert raised.value.task_id == "task-stuck"
    # 首个关键词状态不明后必须立即停止，不能继续为后续关键词下单。
    assert pending["count"] == 1


def test_material_orchestration_reports_download_failure_with_remote_id():
    item = MaterialInfo(
        provider=PROVIDER,
        url="https://cdn.test/a.mp4",
        duration=5,
        source_info={"asset_id": "task-dl"},
    )
    with (
        patch.object(seedance25, "generate_videos", return_value=[item]),
        patch.object(material, "_save_generated_video_with_retry", return_value=""),
        patch.object(material, "_persist_material_sources"),
        patch.object(material.time, "sleep"),
    ):
        with pytest.raises(seedance25.Seedance25DownloadError) as raised:
            material.download_videos(
                task_id="t",
                search_terms=["one", "two"],
                source=PROVIDER,
                audio_duration=30,
                max_clip_duration=5,
            )

    assert raised.value.task_id == "task-dl"


def test_task_failure_records_seedance25_details_key(monkeypatch):
    """任务层把失败详情写进 spec 声明的键名，便于按平台找回远端任务。"""
    from app.services import state as sm
    from app.services import task as task_service
    from app.models.schema import VideoParams

    memory_state = sm.MemoryState()
    monkeypatch.setattr(task_service.sm, "state", memory_state)

    failure = seedance25.Seedance25UnconfirmedTaskError(
        "remote state unknown", task_id="task-recover-1"
    )
    with (
        patch.object(seedance25, "is_enabled", return_value=True),
        patch.object(task_service, "generate_script", return_value="script"),
        patch.object(task_service, "generate_terms", return_value=["term"]),
        patch.object(task_service, "generate_audio", return_value=("a.mp3", 5.0, None)),
        patch.object(material, "download_videos", side_effect=failure),
    ):
        task_service.start(
            "task-25-failure",
            VideoParams(video_subject="s", video_source=PROVIDER),
            stop_at="materials",
        )

    failed = memory_state.get_task("task-25-failure")
    assert failed["failed_stage"] == "materials"
    assert failed["seedance25_task_id"] == "task-recover-1"

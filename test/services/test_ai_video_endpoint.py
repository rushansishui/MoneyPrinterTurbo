"""独立文生视频端点测试。

重点覆盖三条安全语义：

1. **计费确认必须先于任何远端请求** —— API 没有会话承载 WebUI 的确认框，
   ``confirm_charge=false`` 必须直接 400，且不能触发生成。
2. **只有 ai_video 分组的源可用** —— 库存源（pexels 等）不属于 AI 生成。
3. **provider 能力差异被正确拒绝** —— 不支持 i2v 的源收到首帧时返回 400，
   而不是静默丢弃首帧后仍然下单。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.models.schema import MaterialInfo
from app.services import ai_video_studio, video_providers
from app.services.video_providers.spec import FirstFrame
from app.services.video_providers.spec import ReferenceAsset


@pytest.fixture
def client():
    from app.asgi import app

    return TestClient(app)


def _enable(provider_id: str, enabled: bool = True):
    """把某个 provider 的凭据检查打桩，避免测试依赖真实配置文件。"""
    return patch.object(
        video_providers, "is_enabled", side_effect=lambda pid, *a, **k: (
            enabled if pid == provider_id else False
        )
    )


# ---------------------------------------------------------------------------
# provider 列表
# ---------------------------------------------------------------------------


def test_providers_endpoint_lists_capabilities(client):
    with patch.object(video_providers, "is_enabled", return_value=False):
        response = client.get("/api/v1/ai_videos/providers")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == 200
    providers = {item["provider_id"]: item for item in payload["data"]["providers"]}
    # Seedance 2.5 是本期新增的能力最全的源。
    assert providers["seedance25"]["supports_image_to_video"] is True
    assert providers["seedance25"]["supports_native_audio"] is True
    assert providers["seedance25"]["supports_reference_assets"] is True
    assert providers["seedance25"]["requires_charge_confirmation"] is True
    # 1.0 不支持 i2v / 原生音频，能力标记必须如实反映。
    assert providers["volcengine_seedance"]["supports_image_to_video"] is False
    assert providers["volcengine_seedance"]["supports_native_audio"] is False
    assert providers["volcengine_seedance"]["supports_reference_assets"] is False
    # 未配置凭据时 enabled 为 false，前端据此提示用户先填 Key。
    assert all(item["enabled"] is False for item in providers.values())


def test_providers_endpoint_excludes_non_ai_sources(client):
    with patch.object(video_providers, "is_enabled", return_value=False):
        payload = client.get("/api/v1/ai_videos/providers").json()
    listed = {item["provider_id"] for item in payload["data"]["providers"]}
    # 库存源与本地素材不属于「AI 生成」，不应出现在这个入口。
    assert "pexels" not in listed
    assert "local" not in listed


# ---------------------------------------------------------------------------
# 计费确认
# ---------------------------------------------------------------------------


def test_missing_charge_confirmation_is_rejected_before_any_generation(client):
    with (
        _enable("seedance25"),
        patch.object(ai_video_studio.video_providers, "generate") as generate,
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "a cat walking on the beach",
                "duration": 5,
            },
        )

    assert response.status_code == 400
    assert "confirm_charge" in response.json()["message"]
    generate.assert_not_called()


def test_confirmed_request_generates_and_saves_clip(client, tmp_path):
    item = MaterialInfo(
        provider="seedance25",
        url="https://cdn.test/clip.mp4",
        duration=5,
        source_info={
            "asset_id": "task-1",
            "rendition": {"width": 1080, "height": 1920},
        },
    )
    saved = str(tmp_path / "clip.mp4")

    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", return_value=[item]
        ),
        patch.object(
            ai_video_studio.material,
            "_save_generated_video_with_retry",
            return_value=saved,
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "a cat walking on the beach",
                "duration": 5,
                "confirm_charge": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    clips = payload["data"]["clips"]
    assert len(clips) == 1
    assert clips[0]["provider"] == "seedance25"
    assert clips[0]["file"] == saved
    assert clips[0]["width"] == 1080
    assert clips[0]["height"] == 1920
    assert clips[0]["remote_task_id"] == "task-1"


# ---------------------------------------------------------------------------
# 请求校验
# ---------------------------------------------------------------------------


def test_unknown_provider_is_rejected(client):
    response = client.post(
        "/api/v1/ai_videos",
        json={
            "provider": "no-such-provider",
            "prompt": "x",
            "duration": 5,
            "confirm_charge": True,
        },
    )
    assert response.status_code == 400
    assert "unknown video provider" in response.json()["message"]


def test_stock_source_is_not_available_through_this_endpoint(client):
    """库存源不在这条端点的可选范围内。

    注册表只登记 AI 生成源，因此 pexels 在这里表现为「未知 provider」——
    这正是期望行为：这个入口不应该接受库存素材搜索。
    """
    response = client.post(
        "/api/v1/ai_videos",
        json={
            "provider": "pexels",
            "prompt": "x",
            "duration": 5,
            "confirm_charge": True,
        },
    )
    assert response.status_code == 400
    assert "pexels" in response.json()["message"]


def test_missing_credentials_is_rejected(client):
    with _enable("seedance25", enabled=False):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
            },
        )
    assert response.status_code == 400
    assert "Ark API key" in response.json()["message"]


def test_image_to_video_is_rejected_for_unsupported_provider(client):
    """不支持 i2v 的源收到首帧时必须报错，而不是丢弃首帧后仍下单。"""
    with (
        _enable("volcengine_seedance"),
        patch.object(ai_video_studio.video_providers, "generate") as generate,
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "volcengine_seedance",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
                "first_frame": {
                    "kind": "url",
                    "value": "https://img.test/a.png",
                },
            },
        )

    assert response.status_code == 400
    assert "image-to-video" in response.json()["message"]
    generate.assert_not_called()


def test_image_to_video_first_frame_is_passed_to_provider(client, tmp_path):
    captured = {}

    def fake_generate(provider_id, **kwargs):
        captured.update(kwargs)
        return []

    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", side_effect=fake_generate
        ),
    ):
        client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "pan across the room",
                "duration": 5,
                "confirm_charge": True,
                "first_frame": {
                    "kind": "url",
                    "value": "https://img.test/a.png",
                    "origin": "stock",
                },
            },
        )

    frame = captured["first_frame"]
    assert isinstance(frame, FirstFrame)
    assert frame.kind == "url"
    assert frame.value == "https://img.test/a.png"
    assert frame.origin == "stock"


def test_reference_assets_are_passed_to_provider_and_allow_empty_prompt(client):
    captured = {}

    def fake_generate(provider_id, **kwargs):
        captured.update(kwargs)
        return []

    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", side_effect=fake_generate
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "",
                "duration": 5,
                "confirm_charge": True,
                "omni_reference_task_type": "auto",
                "reference_assets": [
                    {
                        "kind": "image",
                        "source": "url",
                        "value": "https://img.test/character.png",
                    },
                    {
                        "kind": "video",
                        "source": "url",
                        "value": "https://cdn.test/movement.mp4",
                    },
                    {
                        "kind": "audio",
                        "source": "url",
                        "value": "https://cdn.test/voice.mp3",
                    },
                ],
            },
        )

    assert response.status_code == 502  # fake provider intentionally returns no clips
    assets = captured["reference_assets"]
    assert all(isinstance(asset, ReferenceAsset) for asset in assets)
    assert [asset.kind for asset in assets] == ["image", "video", "audio"]
    assert captured["omni_reference_task_type"] == "auto"


def test_audio_reference_allows_empty_prompt(client):
    captured = {}

    def fake_generate(provider_id, **kwargs):
        captured.update(kwargs)
        return []

    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", side_effect=fake_generate
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "",
                "duration": 5,
                "confirm_charge": True,
                "reference_assets": [
                    {
                        "kind": "audio",
                        "source": "url",
                        "value": "https://cdn.test/voice.mp3",
                    }
                ],
            },
        )

    assert response.status_code == 502  # fake provider intentionally returns no clips
    assert captured["reference_assets"][0].kind == "audio"


def test_reference_assets_are_rejected_for_unsupported_provider(client):
    with (
        _enable("volcengine_seedance"),
        patch.object(ai_video_studio.video_providers, "generate") as generate,
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "volcengine_seedance",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
                "reference_assets": [
                    {
                        "kind": "image",
                        "source": "url",
                        "value": "https://img.test/style.png",
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert "reference assets" in response.json()["message"]
    generate.assert_not_called()


def test_empty_prompt_without_reference_is_rejected(client):
    response = client.post(
        "/api/v1/ai_videos",
        json={
            "provider": "seedance25",
            "prompt": "",
            "duration": 5,
            "confirm_charge": True,
        },
    )
    # 提示词可为空以支持「仅参考素材」；但两者都没有时服务层返回 400。
    assert response.status_code == 400


def test_failed_download_returns_502_with_remote_id(client):
    item = MaterialInfo(
        provider="seedance25",
        url="https://cdn.test/clip.mp4",
        duration=5,
        source_info={"asset_id": "task-recover"},
    )
    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", return_value=[item]
        ),
        patch.object(
            ai_video_studio.material,
            "_save_generated_video_with_retry",
            return_value="",
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
            },
        )

    assert response.status_code == 502
    message = response.json()["message"]
    assert "task-recover" in message


def test_provider_error_is_surfaced_with_remote_id(client):
    from app.services.video_providers.sources import seedance25

    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers,
            "generate",
            side_effect=seedance25.Seedance25UnconfirmedTaskError(
                "state unknown", task_id="task-stuck"
            ),
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
            },
        )

    assert response.status_code == 502
    assert "task-stuck" in response.json()["message"]


def test_empty_result_is_reported_as_generation_failure(client):
    with (
        _enable("seedance25"),
        patch.object(
            ai_video_studio.video_providers, "generate", return_value=[]
        ),
    ):
        response = client.post(
            "/api/v1/ai_videos",
            json={
                "provider": "seedance25",
                "prompt": "x",
                "duration": 5,
                "confirm_charge": True,
            },
        )
    assert response.status_code == 502
    assert "produced no video" in response.json()["message"]

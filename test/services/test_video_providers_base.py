"""P0 公共层测试：HTTP 工具、异常层次、首帧抽象、Spec 与注册表。

这些测试替代原先分散在四个 provider 测试文件里的重复骨架测试。provider
模块自身的测试仍然保留（它们验证各自的协议差异），这里只覆盖「公共层
本身的行为」。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from app.config import config
from app.services.video_providers import (
    client,
    errors,
    http_util,
    polling,
    registry,
    spec as spec_module,
)


def _response(payload, status_code: int = 200, headers=None):
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        json=lambda: payload,
    )


@pytest.fixture(autouse=True)
def restore_config():
    """隔离全局配置与代理设置，避免测试之间互相污染。"""
    original_app = dict(config.app)
    original_proxy = dict(config.proxy)
    yield
    config.app.clear()
    config.app.update(original_app)
    config.proxy.clear()
    config.proxy.update(original_proxy)


# --------------------------------------------------------------------------
# 异常层次
# --------------------------------------------------------------------------


def test_error_hierarchy_carries_task_id():
    error = errors.VideoProviderError("boom", task_id="t-1")
    assert isinstance(error, RuntimeError)
    assert error.task_id == "t-1"

    unconfirmed = errors.VideoProviderUnconfirmedTaskError("maybe", task_id="t-2")
    assert isinstance(unconfirmed, errors.VideoProviderError)
    assert unconfirmed.task_id == "t-2"

    download = errors.VideoProviderDownloadError("partial", task_id="t-3")
    assert isinstance(download, errors.VideoProviderError)
    assert download.task_id == "t-3"


def test_error_task_id_defaults_to_empty_string():
    assert errors.VideoProviderError("boom").task_id == ""


# --------------------------------------------------------------------------
# http_util
# --------------------------------------------------------------------------


def test_status_code_handles_missing_and_unparsable_values():
    assert http_util.status_code(_response({}, status_code=404)) == 404
    assert http_util.status_code(object()) == 200
    assert http_util.status_code(SimpleNamespace(status_code="nope")) == 200


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        (" FALSE ", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("", False),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_config_bool_coercion(raw, expected):
    config.app["probe_flag"] = raw
    assert http_util.config_bool("probe_flag", True) is expected


def test_bounded_float_clamps_and_tolerates_bad_values():
    config.app["probe_timeout"] = 5000
    assert http_util.bounded_float("probe_timeout", 10.0, 1.0, 100.0) == 100.0
    config.app["probe_timeout"] = -5
    assert http_util.bounded_float("probe_timeout", 10.0, 1.0, 100.0) == 1.0
    config.app["probe_timeout"] = float("nan")
    assert http_util.bounded_float("probe_timeout", 7.0, 1.0, 100.0) == 7.0
    config.app["probe_timeout"] = "not-a-number"
    assert http_util.bounded_float("probe_timeout", 7.0, 1.0, 100.0) == 7.0
    assert http_util.bounded_float("missing_key", 3.0, 1.0, 100.0) == 3.0


def test_bounded_float_reads_from_explicit_settings_mapping():
    assert (
        http_util.bounded_float(
            "some_key", 9.0, 1.0, 100.0, settings={"some_key": 42}
        )
        == 42.0
    )


def test_redact_secret_removes_plain_and_url_encoded_key_and_proxy():
    config.proxy["http"] = "http://user:secretpass@proxy.test:3128"
    # quote_plus 把空格编码为 "+"，因此这里用带空格的 Key 同时验证两种编码形态。
    text = (
        "key=abc 123 encoded=abc+123 "
        "proxy=http://user:secretpass@proxy.test:3128"
    )
    redacted = http_util.redact_secret(text, "abc 123")
    assert "abc 123" not in redacted
    assert "abc+123" not in redacted
    assert "secretpass" not in redacted
    assert redacted.count("***") == 3


def test_redact_secret_skips_url_encoded_form_when_identical():
    # quote_plus 对纯字母数字 Key 不改变内容，不应产生额外的无意义替换。
    redacted = http_util.redact_secret("token=plainkey", "plainkey")
    assert redacted == "token=***"


def test_redact_secret_truncates_to_max_length():
    assert len(http_util.redact_secret("x" * 1000, "")) == 500
    assert len(http_util.redact_secret("x" * 1000, "", max_length=20)) == 20


def test_is_retryable_error_covers_network_and_retryable_status():
    assert http_util.is_retryable_error(requests.exceptions.Timeout("slow"))
    assert http_util.is_retryable_error(requests.exceptions.ConnectionError("down"))
    assert http_util.is_retryable_error(
        requests.exceptions.ChunkedEncodingError("truncated")
    )
    assert http_util.is_retryable_error(
        requests.exceptions.HTTPError("503", response=_response({}, status_code=503))
    )
    assert not http_util.is_retryable_error(
        requests.exceptions.HTTPError("401", response=_response({}, status_code=401))
    )
    assert not http_util.is_retryable_error(ValueError("unrelated"))


def test_response_error_defaults_to_ark_ofox_shape():
    response = _response({"error": {"code": "Bad", "message": "nope"}})
    assert http_util.response_error(response, "key") == "Bad: nope"


def test_response_error_falls_back_when_json_is_unavailable():
    class Unreadable:
        status_code = 502

        def json(self):
            raise ValueError("not json")

    assert http_util.response_error(Unreadable(), "key") == "HTTP 502"


def test_response_error_falls_back_when_payload_is_not_a_mapping():
    assert http_util.response_error(_response([1, 2, 3]), "key") == "HTTP 200"


def test_response_error_accepts_custom_extractor():
    def mini_max_extractor(payload):
        error = payload.get("error") or {}
        return ": ".join(
            str(item)
            for item in (error.get("type"), error.get("http_code"), error.get("message"))
            if item not in (None, "")
        )

    response = _response(
        {"error": {"type": "invalid_request", "http_code": 400, "message": "bad"}}
    )
    assert http_util.response_error(
        response, "key", extractor=mini_max_extractor
    ) == "invalid_request: 400: bad"


def test_response_error_redacts_secret_in_extracted_detail():
    response = _response({"error": {"code": "Unauthorized", "message": "sekret"}})
    assert "sekret" not in http_util.response_error(response, "sekret")


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


def test_first_frame_normalizes_value_and_mime():
    frame = spec_module.FirstFrame(
        kind="local_path", value="  /tmp/a.png  ", mime="  image/png  "
    )
    assert frame.value == "/tmp/a.png"
    assert frame.mime == "image/png"
    assert frame.origin == "upload"


def test_first_frame_rejects_empty_value():
    with pytest.raises(ValueError, match="must not be empty"):
        spec_module.FirstFrame(kind="url", value="   ")


def test_spec_derives_config_i18n_and_session_keys():
    provider = spec_module.VideoProviderSpec(
        provider_id="volcengine_seedance",
        default_label="Seedance",
        module="app.services.video_providers.sources.seedance",
    )
    assert provider.config_key("api_key") == "volcengine_seedance_api_key"
    assert provider.label_key == "video_provider_label.volcengine_seedance"
    assert provider.confirm_charge_key == "volcengine_seedance_confirm_charge"
    assert provider.resolved_error_details_key == "volcengine_seedance_task_id"


def test_spec_normalizes_identifier_and_validates_inputs():
    provider = spec_module.VideoProviderSpec(
        provider_id="  Wan-3.0  ",
        default_label="",
        module="app.services.video_providers.sources.wan",
    )
    assert provider.provider_id == "wan_3.0"
    assert provider.default_label == "wan_3.0"

    with pytest.raises(ValueError, match="provider_id must not be empty"):
        spec_module.VideoProviderSpec(provider_id="", default_label="x", module="m")
    with pytest.raises(ValueError, match="must declare an importable module"):
        spec_module.VideoProviderSpec(provider_id="x", default_label="x", module="")


def test_spec_honors_explicit_error_details_key():
    provider = spec_module.VideoProviderSpec(
        provider_id="wavespeed",
        default_label="WaveSpeed",
        module="m",
        error_details_key="prediction_id",
    )
    assert provider.resolved_error_details_key == "prediction_id"


# --------------------------------------------------------------------------
# polling
# --------------------------------------------------------------------------


def _poll_config(**overrides) -> polling.PollConfig:
    base = {
        "provider_label": "Probe",
        "run_timeout_key": "probe_run_timeout",
        "poll_interval_key": "probe_poll_interval",
        "error_cls": errors.VideoProviderError,
        "unconfirmed_cls": errors.VideoProviderUnconfirmedTaskError,
        "success_statuses": frozenset({"succeeded"}),
        "terminal_failure_statuses": frozenset({"failed"}),
        "active_statuses": frozenset({"running"}),
    }
    base.update(overrides)
    return polling.PollConfig(**base)


def test_poll_task_returns_task_on_success():
    responses = [_response({"id": "t", "status": "succeeded"})]
    with (
        patch.object(polling.requests, "get", side_effect=responses),
        patch.object(polling.time, "sleep"),
    ):
        task = polling.poll_task(
            task_id="t",
            status_url="https://probe.test/tasks/t",
            headers={},
            api_key="key",
            settings=_poll_config(),
        )
    assert task == {"id": "t", "status": "succeeded"}


def test_poll_task_raises_unconfirmed_on_expired_deadline_before_polling():
    config.app["probe_run_timeout"] = 60
    clock = iter([0.0, 61.0])
    with (
        patch.object(polling.requests, "get") as get,
        patch.object(polling.time, "monotonic", side_effect=lambda: next(clock)),
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="late",
                status_url="https://probe.test/tasks/late",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    assert raised.value.task_id == "late"
    get.assert_not_called()


def test_poll_task_caps_sleep_by_remaining_deadline():
    config.app.update({"probe_run_timeout": 60, "probe_poll_interval": 5})
    # 首次响应后只剩 0.25s；下一轮在请求前发现截止时间已过。
    clock = iter([0.0, 59.0, 59.75, 60.1])
    with (
        patch.object(
            polling.requests, "get", return_value=_response({"status": "running"})
        ) as get,
        patch.object(polling.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    assert get.call_count == 1
    sleep.assert_called_once_with(0.25)


def test_poll_task_uses_half_of_remaining_for_phase_timeout():
    config.app["probe_run_timeout"] = 60
    clock = iter([0.0, 1.0, 61.0])
    with (
        patch.object(
            polling.requests,
            "get",
            side_effect=requests.exceptions.Timeout("slow"),
        ) as get,
        patch.object(polling.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    assert get.call_count == 1
    assert get.call_args.kwargs["timeout"] == (29.5, 29.5)
    sleep.assert_not_called()


def test_poll_task_linear_backoff_and_retry_exhaustion():
    with (
        patch.object(
            polling.requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ) as get,
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="stuck",
                status_url="https://probe.test/tasks/stuck",
                headers={},
                api_key="key",
                settings=_poll_config(max_poll_retries=5, retry_base_seconds=1.0),
            )
    assert raised.value.task_id == "stuck"
    assert get.call_count == 6  # 首次 + 5 次重试
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_poll_task_returns_none_on_terminal_failure_when_configured():
    with (
        patch.object(
            polling.requests,
            "get",
            return_value=_response({"status": "failed", "error": "bad"}),
        ),
        patch.object(polling.time, "sleep"),
    ):
        result = polling.poll_task(
            task_id="t",
            status_url="https://probe.test/tasks/t",
            headers={},
            api_key="key",
            settings=_poll_config(stop_on_terminal_failure=False),
        )
    assert result is None


def test_poll_task_raises_on_terminal_failure_when_configured_to_stop():
    with (
        patch.object(
            polling.requests,
            "get",
            return_value=_response({"status": "failed", "error": "bad"}),
        ),
        patch.object(polling.time, "sleep"),
    ):
        with pytest.raises(errors.VideoProviderError) as raised:
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(stop_on_terminal_failure=True),
            )
    assert raised.value.task_id == "t"
    assert "failed" in str(raised.value)


def test_poll_task_treats_unknown_status_as_unconfirmed():
    with patch.object(
        polling.requests, "get", return_value=_response({"status": "pausing"})
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    assert "pausing" in str(raised.value)


@pytest.mark.parametrize("payload", [[], None, "running"])
def test_poll_task_rejects_malformed_payload(payload):
    with patch.object(
        polling.requests, "get", return_value=_response(payload)
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    assert raised.value.task_id == "t"


def test_poll_task_retryable_status_enters_retry_path():
    responses = [
        _response({"error": "busy"}, status_code=503),
        _response({"status": "succeeded"}),
    ]
    with (
        patch.object(polling.requests, "get", side_effect=responses),
        patch.object(polling.time, "sleep") as sleep,
    ):
        task = polling.poll_task(
            task_id="t",
            status_url="https://probe.test/tasks/t",
            headers={},
            api_key="key",
            settings=_poll_config(),
        )
    assert task["status"] == "succeeded"
    sleep.assert_called_once_with(1.0)


def test_poll_task_non_retryable_status_is_unconfirmed_and_redacts():
    response = _response(
        {"error": {"code": "Unauthorized", "message": "sekret"}}, status_code=401
    )
    with (
        patch.object(polling.requests, "get", return_value=response),
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="sekret",
                settings=_poll_config(),
            )
    assert "sekret" not in str(raised.value)
    sleep.assert_not_called()


def test_poll_task_uses_unwrap_hook_for_nested_payloads():
    nested = _response({"task": {"status": "succeeded", "id": "inner"}})
    with (
        patch.object(polling.requests, "get", return_value=nested),
        patch.object(polling.time, "sleep"),
    ):
        task = polling.poll_task(
            task_id="t",
            status_url="https://probe.test/tasks/t",
            headers={},
            api_key="key",
            settings=_poll_config(unwrap=lambda body: body.get("task")),
        )
    assert task == {"status": "succeeded", "id": "inner"}


def test_poll_task_unwrap_returning_non_mapping_is_malformed():
    payload = _response({"task": "not-a-dict"})
    with patch.object(polling.requests, "get", return_value=payload):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(unwrap=lambda body: body.get("task")),
            )


def test_poll_task_redirect_factory_handles_3xx_before_retry_logic():
    def factory(response, task_id, api_key):
        location = getattr(response, "headers", {}).get("Location", "")
        return errors.VideoProviderUnconfirmedTaskError(
            f"redirected to {location}", task_id=task_id
        )

    redirect = _response(
        {}, status_code=302, headers={"Location": "https://elsewhere.test"}
    )
    with (
        patch.object(polling.requests, "get", return_value=redirect) as get,
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(redirect_error_factory=factory),
            )
    assert "elsewhere.test" in str(raised.value)
    assert get.call_count == 1
    sleep.assert_not_called()


def test_poll_task_3xx_without_factory_falls_into_non_retryable_branch():
    # 无 redirect_error_factory 时，3xx 不是可重试状态码，会走「状态不明」分支。
    with (
        patch.object(polling.requests, "get", return_value=_response({}, status_code=302)),
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(),
            )
    sleep.assert_not_called()


def test_poll_task_uses_provider_specific_poll_interval_lower_bound():
    # metaso 的 poll 下界是 1.0（其余 0.5）：配置 0.2 应被抬到 1.0。
    config.app.update({"probe_run_timeout": 60, "probe_poll_interval": 0.2})
    running = _response({"status": "running"})
    clock = iter([0.0, 1.0, 2.0, 61.0])
    with (
        patch.object(polling.requests, "get", return_value=running),
        patch.object(polling.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(polling.time, "sleep") as sleep,
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            polling.poll_task(
                task_id="t",
                status_url="https://probe.test/tasks/t",
                headers={},
                api_key="key",
                settings=_poll_config(min_poll_interval=1.0),
            )
    assert sleep.call_args_list[0].args[0] in {1.0, 1.0}


def test_polling_module_calls_requests_and_time_at_call_site():
    """R1 护栏：必须现场访问模块属性，否则测试 patch 会静默失效。

    provider 测试通过 ``patch.object(seedance.requests, "get")`` 打桩，而
    ``seedance.requests`` 就是全局 requests 模块对象。若公共层把它预先绑定
    到局部变量，patch 不再命中，测试会在「看似通过」的同时发出真实网络请求。
    """
    source = inspect.getsource(polling.poll_task)
    assert "requests.get(" in source
    assert "time.monotonic()" in source
    assert "time.sleep(" in source
    # 不得把模块属性预先绑定到局部变量。这里检查赋值语句而非任意出现，
    # 因为 `response = requests.get(` 本身是合法的现场调用。
    for bound in (
        "get = requests.get",
        "monotonic = time.monotonic",
        "sleep = time.sleep",
        "post = requests.post",
    ):
        assert bound not in source, f"unexpected local rebinding: {bound}"


# --------------------------------------------------------------------------
# client.submit_task
# --------------------------------------------------------------------------


def _submit_config(**overrides) -> client.SubmitConfig:
    base = {
        "provider_label": "Probe",
        "error_cls": errors.VideoProviderError,
        "unconfirmed_cls": errors.VideoProviderUnconfirmedTaskError,
    }
    base.update(overrides)
    return client.SubmitConfig(**base)


def test_submit_task_returns_remote_id():
    with patch.object(
        client.requests, "post", return_value=_response({"id": "remote-1"})
    ) as post:
        task_id = client.submit_task(
            url="https://probe.test/videos",
            payload={"prompt": "x"},
            headers={"Authorization": "Bearer k"},
            api_key="k",
            settings=_submit_config(),
        )
    assert task_id == "remote-1"
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == (30, 60)
    assert post.call_args.kwargs["allow_redirects"] is False


def test_submit_task_uses_custom_task_id_field():
    with patch.object(
        client.requests, "post", return_value=_response({"task_id": "remote-2"})
    ):
        task_id = client.submit_task(
            url="https://probe.test/videos",
            payload={},
            headers={},
            api_key="k",
            settings=_submit_config(default_task_id_field="task_id"),
        )
    assert task_id == "remote-2"


def test_submit_task_connection_error_is_unconfirmed_and_never_retried():
    with patch.object(
        client.requests,
        "post",
        side_effect=requests.exceptions.Timeout("slow"),
    ) as post:
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )
    assert post.call_count == 1


@pytest.mark.parametrize("status_code", [301, 302, 307])
def test_submit_task_redirect_is_unconfirmed(status_code):
    with patch.object(
        client.requests, "post", return_value=_response({}, status_code=status_code)
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )


@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_submit_task_server_error_is_unconfirmed(status_code):
    with patch.object(
        client.requests, "post", return_value=_response({}, status_code=status_code)
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError):
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )


@pytest.mark.parametrize("status_code", [400, 401, 422])
def test_submit_task_client_error_is_deterministic_failure(status_code):
    with patch.object(
        client.requests,
        "post",
        return_value=_response(
            {"error": {"code": "Bad", "message": "nope"}}, status_code=status_code
        ),
    ):
        with pytest.raises(errors.VideoProviderError) as raised:
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )
    assert not isinstance(raised.value, errors.VideoProviderUnconfirmedTaskError)
    assert "rejected" in str(raised.value)


def test_submit_task_unreadable_body_is_unconfirmed():
    class Unreadable:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    with patch.object(client.requests, "post", return_value=Unreadable()):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )
    assert "unreadable" in str(raised.value)


def test_submit_task_missing_task_id_is_unconfirmed():
    with patch.object(client.requests, "post", return_value=_response({})):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="k",
                settings=_submit_config(),
            )
    assert "task id" in str(raised.value)


def test_submit_task_redacts_secret_in_connection_error():
    with patch.object(
        client.requests,
        "post",
        side_effect=requests.exceptions.ConnectionError("leaked-sekret"),
    ):
        with pytest.raises(errors.VideoProviderUnconfirmedTaskError) as raised:
            client.submit_task(
                url="https://probe.test/videos",
                payload={},
                headers={},
                api_key="leaked-sekret",
                settings=_submit_config(),
            )
    assert "leaked-sekret" not in str(raised.value)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@pytest.fixture
def isolated_registry():
    """在隔离的注册表上测试，不影响内置 provider 的全局状态。

    同时屏蔽内置 provider 的自动登记：``_reset_for_tests`` 会把
    ``_BUILTINS_LOADED`` 复位，若不禁用发现流程，下一次查询就会把真实
    provider 重新填进注册表，断言随即失效。
    """
    registry._reset_for_tests()
    with patch.object(registry, "_ensure_builtins", lambda: None):
        yield registry
    registry._reset_for_tests()


def _fake_spec(provider_id: str = "probe", **overrides):
    base = {
        "provider_id": provider_id,
        "default_label": "Probe",
        "module": "app.services.video_providers.sources.probe",
    }
    base.update(overrides)
    return spec_module.VideoProviderSpec(**base)


def test_registry_register_and_lookup(isolated_registry):
    registered = isolated_registry.register(_fake_spec())
    assert isolated_registry.get_spec("probe") is registered
    assert isolated_registry.get_spec("unknown") is None
    assert isolated_registry.all_specs() == (registered,)
    assert isolated_registry.all_provider_ids() == frozenset({"probe"})


def test_registry_rejects_duplicate_ids(isolated_registry):
    isolated_registry.register(_fake_spec())
    with pytest.raises(ValueError, match="duplicate video provider id"):
        isolated_registry.register(_fake_spec())


def test_registry_filters_by_group(isolated_registry):
    isolated_registry.register(_fake_spec("a", group="ai_video"))
    isolated_registry.register(_fake_spec("b", group="ai_image"))
    assert [s.provider_id for s in isolated_registry.all_specs("ai_video")] == ["a"]
    assert isolated_registry.all_provider_ids("ai_image") == frozenset({"b"})


def test_registry_resolve_unknown_id_raises_key_error(isolated_registry):
    with pytest.raises(KeyError, match="unknown video provider"):
        isolated_registry.resolve("nope")


def test_registry_is_enabled_false_for_unknown_provider(isolated_registry):
    assert isolated_registry.is_enabled("nope") is False


def test_registry_generate_caches_module_handle(isolated_registry):
    module = SimpleNamespace(
        generate_videos=lambda term, duration, aspect, first_frame=None: [
            (term, duration, first_frame)
        ]
    )
    isolated_registry.register(
        _fake_spec("cached", module="types")  # 用真实可导入模块避免走 importlib
    )
    with patch.object(isolated_registry.importlib, "import_module", return_value=module):
        first = isolated_registry.resolve("cached")
        second = isolated_registry.resolve("cached")
    assert first is module
    assert second is module


def test_registry_generate_rejects_i2v_for_unsupported_provider(isolated_registry):
    isolated_registry.register(
        _fake_spec("t2v_only", supports_image_to_video=False)
    )
    with pytest.raises(ValueError, match="does not support image-to-video"):
        isolated_registry.generate(
            "t2v_only",
            search_term="x",
            minimum_duration=5,
            video_aspect="9:16",
            first_frame=spec_module.FirstFrame(kind="url", value="https://img.test/a.png"),
        )


def test_registry_generate_passes_first_frame_when_supported(isolated_registry):
    captured = {}

    def fake_generate_videos(term, duration, aspect, first_frame=None):
        captured["args"] = (term, duration, aspect, first_frame)
        return []

    isolated_registry.register(
        _fake_spec("i2v_ok", supports_image_to_video=True, module="types")
    )
    with patch.object(
        isolated_registry.importlib,
        "import_module",
        return_value=SimpleNamespace(generate_videos=fake_generate_videos),
    ):
        frame = spec_module.FirstFrame(kind="url", value="https://img.test/a.png")
        isolated_registry.generate(
            "i2v_ok",
            search_term="sunrise",
            minimum_duration=5,
            video_aspect="9:16",
            first_frame=frame,
        )
    assert captured["args"] == ("sunrise", 5, "9:16", frame)


def test_registry_generate_unknown_provider_raises_key_error(isolated_registry):
    with pytest.raises(KeyError):
        isolated_registry.generate(
            "nope", search_term="x", minimum_duration=5, video_aspect="9:16"
        )

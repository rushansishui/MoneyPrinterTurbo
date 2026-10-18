"""Provider 公共 HTTP 工具：配置读取、错误脱敏与响应解析。

这里的所有函数都是无状态的纯工具函数，从原先四个 provider 模块里逐字相同的
私有实现提取而来。**重要**：``redact_secret`` / ``response_error`` 的默认
提取器必须与旧实现保持逐字一致的行为，因为既有测试会对错误文本做断言。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote_plus

import requests

from app.config import config


MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def status_code(response: Any) -> int:
    """读取响应状态码；无法读取时按 200 处理，与旧实现一致。"""
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def config_bool(key: str, default: bool) -> bool:
    """读取布尔配置，兼容字符串形式的 ``"0"/"false"/"no"/"off"``。"""
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def tls_verify() -> bool:
    """是否校验外部 API 的 TLS 证书。"""
    return config_bool("tls_verify", True)


def bounded_float(
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    settings: Mapping[str, Any] | None = None,
) -> float:
    """读取有限浮点配置，并限制在不会压垮远端或本机的安全范围内。"""
    source = config.app if settings is None else settings
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def redact_secret(
    value: Any,
    secret: str,
    *,
    max_length: int = MAX_ERROR_TEXT_LENGTH,
) -> str:
    """保留可排障文本，同时移除 API Key、URL 编码 Key 和代理凭据。"""
    text = str(value or "")
    if secret:
        text = text.replace(secret, "***")
        encoded = quote_plus(secret)
        if encoded != secret:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:max_length]


def is_retryable_error(
    error: Exception,
    retryable_status_codes: frozenset[int] = RETRYABLE_STATUS_CODES,
) -> bool:
    """判断异常是否属于「同一任务可以安全重试」的瞬时网络错误。"""
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    return (
        response is not None
        and status_code(response) in retryable_status_codes
    )


def default_error_extractor(payload: dict) -> Any:
    """方舟 / OFox 形状：``{"error": {"code", "message"}}``，否则回退顶层字段。"""
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or payload.get("code")
        message = error.get("message") or payload.get("message")
    else:
        code = payload.get("code")
        message = payload.get("message") or error
    return ": ".join(str(item) for item in (code, message) if item not in (None, ""))


def response_error(
    response: Any,
    api_key: str,
    *,
    extractor: Callable[[dict], Any] | None = None,
    max_length: int = MAX_ERROR_TEXT_LENGTH,
) -> str:
    """把失败的 HTTP 响应整理成一行可排障文本，并做脱敏与截断。

    ``extractor`` 让各 provider 保留自己的错误结构差异（MiniMax V2 的嵌套
    ``error.type/http_code``、MuAPI 可能是 list 的 ``detail``），而不是被
    强行抹平成同一种格式。
    """
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {status_code(response)}"

    detail = (extractor or default_error_extractor)(payload)
    return redact_secret(
        detail or f"HTTP {status_code(response)}", api_key, max_length=max_length
    )

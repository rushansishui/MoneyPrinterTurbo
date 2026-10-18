"""Provider 公共 HTTP 客户端：统一提交语义。

四个 provider 的**提交**（POST）流程逐字同构，差异只在 URL、请求体、任务 ID
的字段名与消息前缀。这里把这段模式集中成一个函数，顺带把最重要的一条安全
语义固化下来：

**提交绝不自动重试。** 超时或 5xx 可能发生在远端已经创建并计费之后，盲目
重发会造成重复扣费。因此只有拿到明确的 4xx 拒绝才判定为确定性失败；超时、
5xx 与不可读响应一律归类为「任务状态不明」（``UnconfirmedTaskError``），
由上层决定停止后续关键词。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services.video_providers import errors, http_util
from app.services.video_providers.http_util import redact_secret, response_error


# 从响应体里取出远端任务 ID。
TaskIdExtractor = Callable[[dict], str]
# 构建提交阶段 3xx 重定向异常的回调：``(response, api_key) -> Exception``。
RedirectErrorFactory = Callable[[Any, str], Exception]


@dataclass(frozen=True, slots=True)
class SubmitConfig:
    """一次付费任务提交的行为配置。"""

    provider_label: str
    error_cls: type[errors.VideoProviderError]
    unconfirmed_cls: type[errors.VideoProviderError]
    default_task_id_field: str = "id"
    response_error_extractor: Callable[[dict], Any] | None = None
    # 提供时，提交阶段的 3xx 走该回调生成异常（可带上 Location 便于排障）；
    # 未提供时使用默认的通用重定向文案。
    redirect_error_factory: RedirectErrorFactory | None = None

    def _message(self, suffix: str) -> str:
        return f"{self.provider_label} {suffix}"


def submit_task(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    api_key: str,
    settings: SubmitConfig,
    task_id_extractor: TaskIdExtractor | None = None,
) -> str:
    """提交一个付费任务并返回远端任务 ID。

    失败分类：

    * 连接异常 / 3xx / 5xx / 响应不可读 / 缺少任务 ID → ``UnconfirmedTaskError``
      （远端可能已计费，调用方必须停止继续提交）
    * 其它 4xx → ``error_cls``（远端明确拒绝，未创建任务）

    POST 固定使用 ``timeout=(30, 60)``、``allow_redirects=False`` 且**不重试**。
    """
    try:
        response = requests.post(
            url,
            json=dict(payload),
            headers=dict(headers),
            proxies=config.proxy,
            verify=http_util.tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise settings.unconfirmed_cls(
            f"{settings.provider_label} submission returned no response; a paid "
            "task may already exist remotely: "
            f"error={type(exc).__name__}, detail={redact_secret(exc, api_key)}"
        ) from exc

    status_code = http_util.status_code(response)
    if 300 <= status_code < 400:
        if settings.redirect_error_factory is not None:
            raise settings.redirect_error_factory(response, api_key)
        raise settings.unconfirmed_cls(
            f"{settings.provider_label} submission returned a redirect; the paid "
            "task state is unknown and the request was not replayed"
        )
    if status_code >= 500:
        raise settings.unconfirmed_cls(
            f"{settings.provider_label} submission failed with HTTP {status_code}; "
            "a paid task may already exist remotely"
        )
    if not 200 <= status_code < 300:
        # 4xx 是明确拒绝（如 duration/resolution 超出模型支持范围），远端
        # 没有创建任务，不存在重复计费风险；错误信息里带着服务端给出的
        # 合法取值范围，直接抛给用户修配置。
        raise settings.error_cls(
            f"{settings.provider_label} video generation request rejected: "
            f"HTTP {status_code}, "
            f"{response_error(response, api_key, extractor=settings.response_error_extractor)}"
        )

    try:
        body = response.json()
    except Exception as exc:
        raise settings.unconfirmed_cls(
            f"{settings.provider_label} submission returned an unreadable "
            "response; a paid task may already exist remotely: "
            f"error={type(exc).__name__}"
        ) from exc

    extractor = task_id_extractor or _default_task_id_extractor(
        settings.default_task_id_field
    )
    task_id = extractor(body) if isinstance(body, dict) else ""
    task_id = str(task_id or "").strip()
    if not task_id:
        raise settings.unconfirmed_cls(
            f"{settings.provider_label} accepted the submission without "
            "returning a task id"
        )

    logger.info(f"{settings.provider_label} paid task created: id={task_id}")
    return task_id


def _default_task_id_extractor(field: str) -> TaskIdExtractor:
    def extract(body: dict) -> str:
        return str(body.get(field) or "")

    return extract

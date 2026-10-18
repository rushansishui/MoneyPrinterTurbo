"""公共付费任务轮询引擎。

从原先四个 provider 模块里逐字复制的 ``_wait_for_task`` 骨架提取而来。骨架
本身（截止时间、相位超时、重试上限、线性退避、状态三分类）保持完全一致，
差异通过 :class:`PollConfig` 参数化：

* **消息前缀** —— 各源的 ``provider_label``。
* **窗口与间隔** —— 配置键、默认值与下界（metaso 的 poll 下界是 1.0，其余 0.5）。
* **状态枚举** —— 成功 / 终止失败 / 进行中三个集合（ofox 的成功集含
  ``succeeded``，muapi 的是 ``{"completed"}``）。
* **终止失败后的行为** —— seedance/metaso 抛错，ofox/muapi 返回 ``None``
  （远端既已明确结束，调用方可以换关键词继续）。
* **3xx 重定向的处理** —— 仅 muapi 有专门分支，通过 ``redirect_error_factory``
  注入。

**关键实现约束**：本模块必须在调用现场使用 ``requests.get(...)`` /
``time.monotonic()`` / ``time.sleep(...)``，不得把它们预先绑定到局部变量或
默认参数。既有测试通过 ``patch.object(seedance.requests, "get")`` 打桩，而
``seedance.requests`` 就是全局 ``requests`` 模块对象，因此本模块对同一对象的
属性访问同样会被命中。若改成局部绑定，patch 会静默失效并发出真实网络请求。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services.video_providers import errors, http_util
from app.services.video_providers.http_util import (
    RETRYABLE_STATUS_CODES,
    redact_secret,
    response_error,
)


# 构建 3xx 重定向异常的回调：``(response, task_id, api_key) -> Exception``。
RedirectErrorFactory = Callable[[Any, str, str], Exception]
# 把解析后的响应体转换成任务对象；返回非 dict 视为响应格式错误。
TaskUnwrapper = Callable[[dict], Any]


@dataclass(frozen=True, slots=True)
class PollConfig:
    """一个 provider 的轮询行为配置。"""

    provider_label: str
    run_timeout_key: str
    poll_interval_key: str
    error_cls: type[errors.VideoProviderError]
    unconfirmed_cls: type[errors.VideoProviderError]
    default_run_timeout: float = 1800.0
    default_poll_interval: float = 5.0
    min_poll_interval: float = 0.5
    max_poll_retries: int = 5
    retry_base_seconds: float = 1.0
    retryable_status_codes: frozenset[int] = RETRYABLE_STATUS_CODES
    active_statuses: frozenset[str] = frozenset({"queued", "running"})
    success_statuses: frozenset[str] = frozenset({"succeeded"})
    terminal_failure_statuses: frozenset[str] = frozenset(
        {"failed", "cancelled", "canceled"}
    )
    # True：终止失败抛 error_cls；False：记录日志并返回 None。
    stop_on_terminal_failure: bool = True
    # 仅 muapi 需要：在重试判定之前专门处理 3xx。
    redirect_error_factory: RedirectErrorFactory | None = None
    unwrap: TaskUnwrapper | None = None
    response_error_extractor: Callable[[dict], Any] | None = None

    def _message(self, suffix: str) -> str:
        return f"{self.provider_label} {suffix}"

    def make_error(self, suffix: str, task_id: str = "") -> errors.VideoProviderError:
        return self.error_cls(self._message(suffix), task_id=task_id)

    def make_unconfirmed(
        self, suffix: str, task_id: str = ""
    ) -> errors.VideoProviderError:
        return self.unconfirmed_cls(self._message(suffix), task_id=task_id)


def poll_task(
    *,
    task_id: str,
    status_url: str,
    headers: Mapping[str, str],
    api_key: str,
    settings: PollConfig,
) -> dict[str, Any] | None:
    """轮询同一个付费任务，直到成功、明确失败或本地无法确认状态。

    返回远端任务对象；``None`` 表示远端已明确失败且该 provider 允许跳过
    （见 :attr:`PollConfig.stop_on_terminal_failure`）。
    """
    deadline = time.monotonic() + http_util.bounded_float(
        settings.run_timeout_key,
        settings.default_run_timeout,
        60.0,
        7200.0,
    )
    poll_interval = http_util.bounded_float(
        settings.poll_interval_key,
        settings.default_poll_interval,
        settings.min_poll_interval,
        60.0,
    )
    consecutive_failures = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise settings.make_unconfirmed(
                "task is still running after the configured local wait timeout: "
                f"id={task_id}",
                task_id=task_id,
            )

        # connect/read timeout 分别计时，均使用剩余时间的一半，保证一次 GET
        # 不会有意越过任务总截止时间。到期后不会再发起下一次轮询。
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = requests.get(
                status_url,
                headers=headers,
                proxies=config.proxy,
                verify=http_util.tls_verify(),
                timeout=(phase_timeout, phase_timeout),
                allow_redirects=False,
            )
            status_code = http_util.status_code(response)
            if (
                settings.redirect_error_factory is not None
                and 300 <= status_code < 400
            ):
                raise settings.redirect_error_factory(response, task_id, api_key)
            if status_code in settings.retryable_status_codes:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise settings.make_unconfirmed(
                    "task status is unknown: "
                    f"http_status={status_code}, "
                    f"detail={response_error(response, api_key, extractor=settings.response_error_extractor)}",
                    task_id=task_id,
                )
            body = response.json()
            task = (settings.unwrap or _identity)(body) if isinstance(body, dict) else None
            if not isinstance(task, dict):
                raise settings.make_unconfirmed(
                    "task status response is malformed", task_id=task_id
                )
        except errors.VideoProviderError:
            # 已归类好的异常直接透传。各 provider 使用自己的错误子类（保持旧
            # 代码里 `except VolcEngineSeedanceError` 这种窄捕获语义），它们都
            # 继承 VideoProviderError，因此这里按基类判断即可。
            raise
        except Exception as exc:
            if not http_util.is_retryable_error(
                exc, settings.retryable_status_codes
            ):
                raise settings.make_unconfirmed(
                    "polling failed and the paid task state is unknown: "
                    f"error={type(exc).__name__}, "
                    f"detail={redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise settings.make_unconfirmed(
                    "task is still running after the configured local wait "
                    f"timeout: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > settings.max_poll_retries:
                raise settings.make_unconfirmed(
                    "polling failed after retries; the paid task may still be "
                    f"running remotely: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(
                settings.retry_base_seconds * consecutive_failures, remaining
            )
            logger.warning(
                f"{settings.provider_label} polling hit a transient error; "
                f"retrying the same task: id={task_id}, "
                f"attempt={consecutive_failures}/{settings.max_poll_retries}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(task.get("status") or "").strip().lower()
        logger.info(f"{settings.provider_label} task status: id={task_id}, status={status}")
        if status in settings.success_statuses:
            return task
        if status in settings.terminal_failure_statuses:
            detail = redact_secret(task.get("error"), api_key)
            if settings.stop_on_terminal_failure:
                raise settings.make_error(
                    "task did not produce a video: "
                    f"id={task_id}, status={status}, detail={detail}",
                    task_id=task_id,
                )
            logger.error(
                f"{settings.provider_label} task did not produce a video: "
                f"id={task_id}, status={status}, detail={detail}"
            )
            # 远端明确失败意味着任务已经结束，可以安全地继续后续片段；由
            # 调用方决定是否换一个关键词重试。
            return None
        if status not in settings.active_statuses:
            raise settings.make_unconfirmed(
                f"returned an unknown task status: id={task_id}, status={status!r}",
                task_id=task_id,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise settings.make_unconfirmed(
                "task is still running after the configured local wait timeout: "
                f"id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))


def _identity(body: dict) -> dict:
    return body

"""兼容垫片：转发到 ``video_providers.sources.ofox``。

保留旧路径的原因与其它垫片相同：既有测试通过模块属性打桩并直接调用
``ofox._wait_for_task(...)``。新代码请直接使用
``app.services.video_providers`` 包。
"""

from app.services.video_providers.sources.ofox import (  # noqa: F401
    SPEC,
    ACTIVE_STATUSES,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MIN_DURATION_SECONDS,
    DEFAULT_MODEL_ID,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_PROVIDER_TYPE,
    DEFAULT_RESOLUTION,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    MAX_ERROR_TEXT_LENGTH,
    MAX_POLL_RETRIES,
    RETRY_BASE_SECONDS,
    RETRYABLE_STATUS_CODES,
    TERMINAL_FAILURE_STATUSES,
    TERMINAL_SUCCESS_STATUSES,
    OFoxDownloadError,
    OFoxError,
    OFoxUnconfirmedTaskError,
    _base_url,
    _bounded_float,
    _config_bool,
    _duration_bounds,
    _model_id,
    _provider_type,
    _redact_secret,
    _resolution,
    _response_error,
    _status_code,
    _tls_verify,
    _wait_for_task,
    generate_videos,
    get_api_key,
    is_enabled,
)

import requests  # noqa: F401  —— 测试按模块属性打桩，必须存在
import time  # noqa: F401  —— 同上

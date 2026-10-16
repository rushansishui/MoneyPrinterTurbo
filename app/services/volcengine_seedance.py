"""兼容垫片：转发到 ``video_providers.sources.seedance``。

该模块已被迁移到统一的 provider 包，但旧路径仍然保留，原因是：

1. 既有测试通过模块属性打桩（``patch.object(seedance.requests, "get")``），
   并直接调用 ``seedance._wait_for_task(...)``；重新导出这些名字才能让
   这些测试零修改通过。
2. 外部调用方与 ``material`` / ``task`` 服务仍按旧名字 import 错误类。

新代码请直接使用 ``app.services.video_providers`` 包。
"""

from app.services.video_providers.sources.seedance import (  # noqa: F401
    ACTIVE_STATUSES,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MIN_DURATION_SECONDS,
    DEFAULT_MODEL_ID,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_RESOLUTION,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    MAX_ERROR_TEXT_LENGTH,
    MAX_POLL_RETRIES,
    RETRY_BASE_SECONDS,
    RETRYABLE_STATUS_CODES,
    SUCCESS_STATUSES,
    SUPPORTED_RESOLUTIONS,
    TERMINAL_FAILURE_STATUSES,
    VolcEngineSeedanceDownloadError,
    VolcEngineSeedanceError,
    VolcEngineSeedanceUnconfirmedTaskError,
    _base_url,
    _bounded_float,
    _config_bool,
    _duration_bounds,
    _model_id,
    _redact_secret,
    _rendition_size,
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

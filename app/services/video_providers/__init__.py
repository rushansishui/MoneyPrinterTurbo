"""AI 文生视频 Provider 的公共基础层。

该包把原先分散在各 provider 模块里的 HTTP 轮询骨架、配置读取与错误脱敏
统一到一处，并提供注册表把「有哪些源、各自支持什么能力」与「怎么调用」
分离。新增一个视频源只需要在 ``sources/`` 下加一个模块并注册，不需要改动
``material``/``task``/``webui``/``cli`` 的任何分派代码。

依赖方向固定为：``material -> registry -> sources -> {spec, polling, http_util, errors}``。
``sources`` 不得 import ``registry``，避免循环导入。
"""

from app.services.video_providers.errors import (
    VideoProviderDownloadError,
    VideoProviderError,
    VideoProviderUnconfirmedTaskError,
)
from app.services.video_providers.registry import (
    all_specs,
    generate,
    get_spec,
    is_enabled,
    resolve,
)
from app.services.video_providers.spec import FirstFrame, VideoProviderSpec


__all__ = [
    "FirstFrame",
    "VideoProviderDownloadError",
    "VideoProviderError",
    "VideoProviderSpec",
    "VideoProviderUnconfirmedTaskError",
    "all_specs",
    "generate",
    "get_spec",
    "is_enabled",
    "resolve",
]

"""视频 Provider 的统一异常层次。

三类错误的语义与现有各 provider 模块完全一致，只是去掉了源前缀：

* :class:`VideoProviderError` —— 确定性的配置、请求或响应错误。远端没有
  创建付费任务，或任务已经明确结束。
* :class:`VideoProviderUnconfirmedTaskError` —— 远端**可能**已经创建并计费，
  但本机无法确认其最终状态。上层必须停止继续提交新的付费任务。
* :class:`VideoProviderDownloadError` —— 远端任务已成功，但成片没能下载到本机。

与旧模块的兼容性由各源的封装模块负责：``volcengine_seedance.VolcEngineSeedanceError``
等旧名字仍然可用，并且仍是本层次的子类，因此 ``except`` 的宽窄行为不变。
"""

from __future__ import annotations


class VideoProviderError(RuntimeError):
    """确定性的配置、请求或响应错误。"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # 只要远端任务已经创建，所有后续异常都携带同一个 ID。任务服务可以
        # 统一保存恢复线索，不需要了解轮询、结果解析或下载分别在哪一步失败。
        # 字段名保持 ``task_id``：WebUI/API 与既有测试都按这个名字读取。
        self.task_id = task_id


class VideoProviderUnconfirmedTaskError(VideoProviderError):
    """远端可能已创建付费任务，但本机无法确认其最终状态。"""


class VideoProviderDownloadError(VideoProviderError):
    """远端付费任务已成功，但生成的视频未能下载到本机。"""

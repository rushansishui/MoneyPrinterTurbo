"""WaveSpeed AI 文生视频客户端。

WaveSpeed 走 ``{base}/{model_id}`` 提交、``{base}/predictions/{id}/result``
轮询，响应统一包在 ``{"code": 200, "data": {...}}`` 里，并按 ``prediction_id``
追踪任务（而非 ``task_id``）。

**该源的骨架刻意不并入公共 ``poll_task``**：它与其余四个源有实质差异——
响应需要先校验业务码 ``code == 200`` 再解包 ``data``；轮询使用固定的
``WAVESPEED_POLL_INTERVAL_SECONDS`` 而非配置项；轮询请求固定
``timeout=(30, 60)`` 且不设 ``allow_redirects=False``；截止时间检查放在状态
判定之后；提交被业务码拒绝时按「空结果继续」处理而不是抛错。强行统一会改变
这些行为并破坏既有测试，因此这里保留自己的轮询实现，只复用无状态的
``http_util`` 工具。
"""

from __future__ import annotations

import time
from typing import Any, List

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services.video_providers import http_util
from app.services.video_providers.errors import (
    VideoProviderDownloadError,
    VideoProviderError,
    VideoProviderUnconfirmedTaskError,
)
from app.services.video_providers.spec import VideoProviderSpec


# WaveSpeed AI (https://wavespeed.ai) 通过文生视频模型按脚本关键词直接生成素材，
# 与三个库存素材源共用 MaterialInfo 结果结构和后续下载、剪辑流程。
WAVESPEED_API_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_DEFAULT_T2V_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
WAVESPEED_POLL_INTERVAL_SECONDS = 2.0
WAVESPEED_RUN_TIMEOUT_SECONDS = 600.0
# 默认模型 bytedance/seedance-2.0-fast/text-to-video 只接受 4-15 秒；超出
# 范围的请求会被 API 直接拒绝。WebUI 默认片段时长是 3 秒，因此必须在提交
# 前收敛到模型支持区间，多出的时长由现有剪辑流程按片段时长裁掉。
WAVESPEED_MIN_DURATION_SECONDS = 4
WAVESPEED_MAX_DURATION_SECONDS = 15
# 三个失败态语义不同（模型报错 / 用户取消 / 平台超时），但对素材流程都意味着
# 本关键词没有产物，统一按空结果处理，交给上层跳过该片段继续生成。
WAVESPEED_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})
WAVESPEED_SUCCESS_STATUSES = frozenset({"completed"})
# 与 WaveSpeed 官方 Python SDK / n8n 节点保持同一口径：429 与 5xx 属于临时
# 故障，值得有限次退避重试；4xx 是明确的客户端错误，快速失败。
WAVESPEED_RETRYABLE_STATUS_CODES = http_util.RETRYABLE_STATUS_CODES
# 单次轮询允许的连续临时失败次数。一次不走运的 GET 不能让已经计费的任务失联。
WAVESPEED_MAX_POLL_RETRIES = 5
# 线性退避基数，第 n 次重试等待 base * n 秒。
WAVESPEED_RETRY_BASE_SECONDS = 1.0
# 产物下载失败时对同一个签名地址的重试次数。素材已经付费生成，优先重试原
# 地址，不能因为一次下载抖动就重新提交一次付费生成任务。
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2


class WaveSpeedUnconfirmedTaskError(VideoProviderUnconfirmedTaskError):
    """
    付费生成任务已提交，但最终状态无法在本地确认。

    这类异常绝不等价于“该任务失败、可以重来”：远端任务可能仍在运行或已经
    完成并计费。素材流程必须就此停止，不再为后续关键词提交新的付费任务，
    并把已提交的 prediction id 传给任务状态供人工找回。
    """

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message, task_id=prediction_id)
        # 该源对外暴露的是 ``prediction_id``，历史调用方与测试都按这个名字读取。
        self.prediction_id = prediction_id


class WaveSpeedDownloadError(VideoProviderDownloadError):
    """A paid prediction completed, but its video could not be saved locally."""

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message, task_id=prediction_id)
        self.prediction_id = prediction_id


SPEC = VideoProviderSpec(
    provider_id="wavespeed",
    default_label="WaveSpeed AI",
    module="app.services.video_providers.sources.wavespeed",
    legacy_module="app.services.material",
    group="ai_video",
    requires_charge_confirmation=True,
    supports_image_to_video=False,
    supports_native_audio=False,
    measure_downloaded_duration=False,
    # 该源的远端 ID 属性名是 prediction_id（不是 task_id）。
    remote_task_id_attr="prediction_id",
    error_details_key="wavespeed_prediction_id",
)


def _wavespeed_status_code(response: Any) -> int:
    """读取响应状态码；测试替身或异常对象缺少该字段时按 200 处理。"""
    return http_util.status_code(response)


def _is_wavespeed_retryable_error(error: Exception) -> bool:
    """
    判断轮询异常是否值得重试。

    连接、超时一类网络异常没有状态码，按临时故障处理；带状态码的响应只在
    429 和 5xx 时重试，与官方 SDK 的重试集合保持一致。
    """
    return http_util.is_retryable_error(error, WAVESPEED_RETRYABLE_STATUS_CODES)


def _wavespeed_duration_bounds() -> tuple[int, int]:
    """
    返回当前模型支持的生成时长区间（秒）。

    默认区间对应默认 Seedance 模型；用户切换到其它文生视频模型时，可以在
    配置中同步调整区间。任何异常配置都退回默认值，并保证 min <= max，
    避免把用户输入变成必然失败的远端请求。
    """

    def read_bound(key: str, fallback: int) -> int:
        try:
            value = int(config.app.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value >= 1 else fallback

    min_duration = read_bound(
        "wavespeed_min_duration", WAVESPEED_MIN_DURATION_SECONDS
    )
    max_duration = read_bound(
        "wavespeed_max_duration", WAVESPEED_MAX_DURATION_SECONDS
    )
    return min_duration, max(max_duration, min_duration)


def _wavespeed_api_key() -> str:
    """读取 WaveSpeed 凭据。

    WaveSpeed 使用 ``wavespeed_api_keys`` 列表并支持多 Key 轮换，因此不走
    ``http_util`` 的单键读取，而是沿用 material 服务的 Key 轮换逻辑。
    """
    from app.services import material

    return str(material.get_api_key("wavespeed_api_keys") or "")


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    first_frame: Any = None,
) -> List[MaterialInfo]:
    """
    用 WaveSpeed 文生视频模型为一个脚本关键词生成一段素材。

    与库存素材源的 search_videos_* 保持同一签名和空列表失败约定，
    使其可以直接接入 ``download_videos`` 的通用下载与时长核算流程。
    ``minimum_duration`` 在生成语境下就是目标片段时长（秒）。
    """
    if first_frame is not None:
        raise VideoProviderError(
            "WaveSpeed does not support image-to-video generation"
        )

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = _wavespeed_api_key()
    model_id = (
        str(
            config.app.get("wavespeed_text_to_video_model", "")
            or WAVESPEED_DEFAULT_T2V_MODEL
        )
        .strip()
        .strip("/")
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    requested_duration = max(int(minimum_duration), 1)
    min_duration, max_duration = _wavespeed_duration_bounds()
    duration = min(max(requested_duration, min_duration), max_duration)
    if duration != requested_duration:
        # 生成比请求更长不会影响成片：剪辑流程仍按片段时长裁剪；生成比请求
        # 更短的情况只发生在请求超过模型上限时，此时也只能收敛到上限。
        logger.info(
            f"wavespeed clip duration clamped to model-supported range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(supported {min_duration}-{max_duration}s)"
        )
    payload = {
        "prompt": search_term,
        "aspect_ratio": aspect.value,
        "duration": duration,
    }
    logger.info(
        f"generating video on wavespeed: model={model_id}, "
        f"term={search_term!r}, duration={duration}s"
    )

    # 提交 POST 绝不自动重试：请求可能已经在远端创建了付费任务，重发会造成
    # 重复生成和重复扣费（与官方 SDK 的 submission 策略一致）。
    try:
        submit_response = requests.post(
            f"{WAVESPEED_API_BASE_URL}/{model_id}",
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=http_util.tls_verify(),
            timeout=(30, 60),
        )
    except Exception as e:
        # 没有收到响应并不代表任务没有创建。此时状态不明，必须终止整个生成
        # 流程，而不是继续为下一个关键词提交新的付费任务。
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission did not return a response, the task may "
            "already exist remotely: "
            f"error={type(e).__name__}, detail={http_util.redact_secret(e, api_key)}"
        ) from e

    submit_status = _wavespeed_status_code(submit_response)
    if submit_status >= 500:
        # 5xx 可能发生在任务创建之后，无法判断是否已经计费。
        raise WaveSpeedUnconfirmedTaskError(
            f"wavespeed submission failed with HTTP {submit_status}, "
            "the task may already exist remotely"
        )
    try:
        submit_body = submit_response.json()
    except Exception as e:
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission returned an unreadable response, the task "
            f"may already exist remotely: error={type(e).__name__}"
        ) from e

    submit_data = submit_body.get("data") if isinstance(submit_body, dict) else None
    if not isinstance(submit_body, dict) or submit_body.get("code") != 200:
        # 4xx 与业务错误码是明确的拒绝，远端没有创建任务，也就不存在重复
        # 计费风险，按现有素材源约定返回空结果并继续。
        logger.error(
            "wavespeed video generation request rejected: "
            f"http_status={submit_status}, "
            f"code={submit_body.get('code') if isinstance(submit_body, dict) else None}, "
            f"detail={http_util.redact_secret(str((submit_body or {}).get('message') or ''), api_key)}"
        )
        return []
    prediction_id = (
        str(submit_data.get("id") or "") if isinstance(submit_data, dict) else ""
    )
    if not prediction_id:
        # 提交被接受但没拿到 ID：任务可能已经存在却无法追踪，不能继续下单。
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed accepted the submission without returning a prediction id"
        )
    # 生成任务提交成功即产生远端计费副作用，先落日志记录任务 ID，
    # 即使后续轮询失败，用户仍能凭 ID 在 WaveSpeed 控制台找回产物。
    logger.info(f"wavespeed prediction created: id={prediction_id}")

    result_data = _wait_for_wavespeed_prediction(
        prediction_id=prediction_id,
        headers=headers,
        api_key=api_key,
    )
    if result_data is None:
        return []

    try:
        video_items = []
        outputs = result_data.get("outputs")
        for output in outputs if isinstance(outputs, list) else []:
            # 产物 URL 是带签名的临时下载地址，必须整体保留（不能剥离查询参
            # 数），因此不写入 source_info，只用于随后的立即下载。
            if not isinstance(output, str) or not output.startswith(
                ("http://", "https://")
            ):
                continue
            item = MaterialInfo()
            item.provider = "wavespeed"
            item.url = output
            item.duration = duration
            item.source_info = {
                "provider": "wavespeed",
                "search_term": search_term,
                "asset_id": prediction_id,
                "rendition": {
                    "id": None,
                    "width": video_width,
                    "height": video_height,
                },
            }
            video_items.append(item)
        if not video_items:
            logger.error(
                "wavespeed prediction completed without downloadable outputs: "
                f"id={prediction_id}"
            )
        return video_items
    except Exception as e:
        # 产物已经生成并计费，这里的异常只可能来自本地解析。记录后按空结果
        # 返回，让上层跳过该片段，但任务状态本身是确定的，可以继续后续片段。
        logger.error(
            "wavespeed output parsing failed: "
            f"id={prediction_id}, error={type(e).__name__}, "
            f"detail={http_util.redact_secret(e, api_key)}"
        )

    return []


def _wait_for_wavespeed_prediction(
    *,
    prediction_id: str,
    headers: dict,
    api_key: str,
) -> dict | None:
    """
    轮询同一个 prediction id 直到出现确定结果。

    返回 ``completed`` 的 data；远端明确失败（failed / cancelled / timeout）
    时返回 None，表示该任务已经结束、可以安全地继续后续片段。临时故障按
    线性退避重试同一个 ID，绝不重新提交任务；状态始终无法确认时抛出
    :class:`WaveSpeedUnconfirmedTaskError`，由调用方终止整个生成流程。
    """
    deadline = time.monotonic() + WAVESPEED_RUN_TIMEOUT_SECONDS
    consecutive_failures = 0
    while True:
        try:
            response = requests.get(
                f"{WAVESPEED_API_BASE_URL}/predictions/{prediction_id}/result",
                headers=headers,
                proxies=config.proxy,
                verify=http_util.tls_verify(),
                timeout=(30, 60),
            )
            status_code = _wavespeed_status_code(response)
            if status_code in WAVESPEED_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            result_body = response.json()
            result_data = (
                result_body.get("data") if isinstance(result_body, dict) else None
            )
            if not isinstance(result_body, dict) or result_body.get("code") != 200:
                # 轮询被明确拒绝（如 4xx）时任务状态仍然未知：任务已经提交，
                # 只是本地查不到结果，同样不能继续提交新的付费任务。
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction status is unknown: "
                    f"http_status={status_code}, "
                    f"code={result_body.get('code') if isinstance(result_body, dict) else None}, "
                    f"detail={http_util.redact_secret(str((result_body or {}).get('message') or ''), api_key)}",
                    prediction_id=prediction_id,
                )
            if not isinstance(result_data, dict):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction result payload is malformed",
                    prediction_id=prediction_id,
                )
        except WaveSpeedUnconfirmedTaskError:
            raise
        except Exception as e:
            if not _is_wavespeed_retryable_error(e):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed and the task state is "
                    f"unknown: error={type(e).__name__}, "
                    f"detail={http_util.redact_secret(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            consecutive_failures += 1
            if consecutive_failures > WAVESPEED_MAX_POLL_RETRIES:
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed after "
                    f"{WAVESPEED_MAX_POLL_RETRIES + 1} attempts, the task may "
                    "still be running remotely: "
                    f"error={type(e).__name__}, "
                    f"detail={http_util.redact_secret(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            delay = WAVESPEED_RETRY_BASE_SECONDS * consecutive_failures
            logger.warning(
                "wavespeed prediction polling hit a transient error, retry the "
                f"same task: id={prediction_id}, "
                f"attempt={consecutive_failures}/{WAVESPEED_MAX_POLL_RETRIES}, "
                f"error={type(e).__name__}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        # 拿到一次有效响应就重置计数，只有连续失败才消耗重试额度。
        consecutive_failures = 0
        status = str(result_data.get("status") or "")
        if status in WAVESPEED_SUCCESS_STATUSES:
            return result_data
        if status in WAVESPEED_FAILURE_STATUSES:
            logger.error(
                "wavespeed prediction did not produce a video: "
                f"id={prediction_id}, status={status}, "
                f"detail={http_util.redact_secret(str(result_data.get('error') or ''), api_key)}"
            )
            return None
        if time.monotonic() > deadline:
            # 远端任务仍在执行，本地无法确认最终状态，必须停止继续下单。
            raise WaveSpeedUnconfirmedTaskError(
                f"wavespeed prediction is still {status or 'pending'} after "
                f"{WAVESPEED_RUN_TIMEOUT_SECONDS:.0f}s of local waiting",
                prediction_id=prediction_id,
            )
        time.sleep(WAVESPEED_POLL_INTERVAL_SECONDS)


# 旧名字：material 服务的分派与既有测试都按 ``generate_videos_wavespeed`` 调用。
generate_videos_wavespeed = generate_videos

"""原生音轨叠加（P5）测试。

原生音轨的关键约束有三条，这里逐条锁定：

1. **默认关闭时行为逐字节不变** —— ``native_audio_enabled=False`` 时不能给
   MoviePy 传 ``audio`` 关键字，也不能给 concat 命令加音频参数，否则既有
   成片的音轨结构会改变。
2. **时间轴在片段成型时就对齐** —— 音频子片段必须用与视频相同的源时间窗
   与倍速构造，这样后续拼接与最终混音都不需要再算时间轴。
3. **concat 的流结构必须一致** —— 原生模式下每个中间片段都要带音轨，
   无音轨素材补静音，否则 ``-f concat`` 会失败或产出错位。
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import config
from app.models.schema import VideoParams
from app.services import video as vd


class _FakeReader:
    """close_clip 通过 reader.close() 释放句柄，这里用它记录关闭。"""

    def __init__(self, owner=None):
        self.owner = owner
        self.closed = False

    def close(self):
        self.closed = True
        if self.owner is not None:
            self.owner.closed = True


class _FakeAudioClip:
    """最小音频片段替身：记录被裁切的时间窗，便于断言对齐逻辑。

    ``ranges`` 与 ``speed_factors`` 是**共享**列表，因为 ``subclipped`` /
    ``with_speed_scaled`` 都返回新对象，记录必须留在调用方拿得到的容器里。
    """

    def __init__(self, duration=10.0, ranges=None, speed_factors=None):
        self.duration = duration
        self.fps = 48000
        self.ranges = ranges if ranges is not None else []
        self.speed_factors = (
            speed_factors if speed_factors is not None else []
        )
        self.reader = _FakeReader(self)
        self.closed = False

    # generate_video 用 ExitStack 持有旁白片段，替身需要支持上下文管理器协议。
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def subclipped(self, start_time, end_time):
        self.ranges.append((start_time, end_time))
        return _FakeAudioClip(
            max(end_time - start_time, 0.0), self.ranges, self.speed_factors
        )

    def with_speed_scaled(self, factor):
        self.speed_factors.append(factor)
        return _FakeAudioClip(self.duration, self.ranges, self.speed_factors)

    def with_effects(self, effects):
        return self


class _FakeVideoClip:
    def __init__(self, duration, size=(1080, 1920), audio=None):
        self.duration = duration
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.audio = audio
        self.reader = _FakeReader(self)

    # generate_video 用 ExitStack 持有片段，替身需要支持上下文管理器协议。
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def subclipped(self, start_time, end_time):
        return _FakeVideoClip(
            max(end_time - start_time, 0.0), self.size, self.audio
        )

    def with_speed_scaled(self, factor):
        return _FakeVideoClip(self.duration, self.size, self.audio)

    def with_audio(self, audio):
        return _FakeVideoClip(self.duration, self.size, audio)

    def with_effects(self, effects):
        return self

    def resized(self, *args, **kwargs):
        # _fit_clip_to_canvas 会先缩放再裁剪；替身只需要保持尺寸与时长不变。
        return _FakeVideoClip(self.duration, (100, 100), self.audio)

    def cropped(self, *args, **kwargs):
        return _FakeVideoClip(self.duration, self.size, self.audio)

    def close(self):
        pass


class TestNativeAudioSegmentAlignment(unittest.TestCase):
    """音频子片段必须与视频使用同一源时间窗、同一倍速、同一目标时长。"""

    def setUp(self):
        self.original_app = dict(config.app)
        self.original_proxy = dict(config.proxy)
        config.proxy.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)
        config.proxy.clear()
        config.proxy.update(self.original_proxy)

    def test_returns_none_when_source_has_no_audio_track(self):
        """无音轨素材返回 None，由调用方补静音，而不是抛错。"""
        probe = _FakeVideoClip(5.0, audio=None)
        with patch.object(vd, "_open_video_clip_quietly", return_value=probe):
            with patch.object(vd, "AudioFileClip") as audio_clip:
                result = vd._build_native_audio_segment(
                    source_path="clip.mp4",
                    start_time=0.0,
                    end_time=5.0,
                    speed=1.0,
                    target_duration=5.0,
                )
        self.assertIsNone(result)
        # 探测到无音轨后不应白白启动一个音频 reader。
        audio_clip.assert_not_called()

    def test_uses_same_source_window_and_speed_as_video(self):
        probe = _FakeVideoClip(10.0, audio=object())
        audio = _FakeAudioClip(10.0)
        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=probe),
            patch.object(vd, "AudioFileClip", return_value=audio),
        ):
            segment = vd._build_native_audio_segment(
                source_path="clip.mp4",
                start_time=2.0,
                end_time=6.0,
                speed=1.5,
                target_duration=2.5,
            )

        # 先按源时间窗裁切，再按同一倍速变速，最后截到目标时长。
        self.assertIn((2.0, 6.0), audio.ranges)
        self.assertEqual(audio.speed_factors, [1.5])
        self.assertIn((0, 2.5), audio.ranges)
        self.assertIsNotNone(segment)

    def test_skips_speed_change_when_factor_is_one(self):
        probe = _FakeVideoClip(10.0, audio=object())
        audio = _FakeAudioClip(10.0)
        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=probe),
            patch.object(vd, "AudioFileClip", return_value=audio),
        ):
            vd._build_native_audio_segment(
                source_path="clip.mp4",
                start_time=0.0,
                end_time=5.0,
                speed=1.0,
                target_duration=5.0,
            )
        # 1.0 倍速时不能多调一次变速，否则会引入不必要的重采样。
        self.assertEqual(audio.speed_factors, [])

    def test_closes_audio_reader_when_slicing_fails(self):
        """裁切失败时必须关闭 reader，否则 Windows 上临时文件删不掉。"""

        class _BrokenAudio(_FakeAudioClip):
            def subclipped(self, start_time, end_time):
                raise RuntimeError("decode failed")

        probe = _FakeVideoClip(10.0, audio=object())
        audio = _BrokenAudio(10.0)
        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=probe),
            patch.object(vd, "AudioFileClip", return_value=audio),
        ):
            with self.assertRaises(RuntimeError):
                vd._build_native_audio_segment(
                    source_path="clip.mp4",
                    start_time=0.0,
                    end_time=5.0,
                    speed=1.0,
                    target_duration=5.0,
                )
        self.assertTrue(audio.closed)


class TestSilentAudioFallback(unittest.TestCase):
    def test_silent_clip_has_requested_duration_and_shape(self):
        clip = vd._silent_audio_clip(3.5)
        self.assertAlmostEqual(clip.duration, 3.5, places=3)
        frame = clip.get_frame(0.0)
        self.assertEqual(frame.shape, (1, vd._NATIVE_AUDIO_CHANNELS))
        self.assertTrue(np.all(frame == 0))

    def test_negative_duration_is_clamped_to_zero(self):
        clip = vd._silent_audio_clip(-2.0)
        self.assertEqual(clip.duration, 0.0)


class TestConcatAudioWiring(unittest.TestCase):
    """concat 命令必须只在原生模式下追加音频参数。"""

    def setUp(self):
        self.original_app = dict(config.app)
        config.app["video_codec"] = "libx264"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    def _run_concat(self, *, include_audio: bool):
        captured = {}

        def fake_run(command, capture_output, text, check, **kwargs):
            captured["command"] = command
            Path(command[-1]).write_bytes(b"encoded")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")
            with patch.object(vd.subprocess, "run", side_effect=fake_run):
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                    include_audio=include_audio,
                )
        return captured["command"]

    def test_default_mode_adds_no_audio_flags(self):
        command = self._run_concat(include_audio=False)
        # 默认模式必须与改造前逐字节一致：没有 -c:a / -ar / -ac。
        for flag in ("-c:a", "-ar", "-ac"):
            self.assertNotIn(flag, command)

    def test_native_mode_adds_aac_stereo_48k(self):
        command = self._run_concat(include_audio=True)
        self.assertIn("-c:a", command)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-ar") + 1], "48000")
        self.assertEqual(command[command.index("-ac") + 1], "2")

    def test_native_mode_trims_both_streams_with_t(self):
        """``-t`` 同时截断视频与音频，原生音轨不会超过旁白时长。"""
        captured = {}

        def fake_run(command, capture_output, text, check, **kwargs):
            captured["command"] = command
            Path(command[-1]).write_bytes(b"encoded")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")
            with patch.object(vd.subprocess, "run", side_effect=fake_run):
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                    max_duration=12.5,
                    include_audio=True,
                )

        command = captured["command"]
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "12.500")
        # 音频参数必须出现在 -t 之前，保证 -t 作用于两路输出流。
        self.assertLess(command.index("-c:a"), command.index("-t"))


class TestCombineVideosBackwardCompatibility(unittest.TestCase):
    """默认关闭时，combine_videos 的调用形态必须与改造前一致。"""

    def setUp(self):
        self.original_app = dict(config.app)
        config.app["video_codec"] = "libx264"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    def _combine(self, *, native_audio_enabled: bool):
        opened_with = []
        write_kwargs = []

        def fake_open(path, *args, **kwargs):
            opened_with.append(kwargs)
            return _FakeVideoClip(1.0, (480, 480))

        def fake_write(clip, _output, **kwargs):
            write_kwargs.append(kwargs)
            Path(_output).write_bytes(b"encoded")

        audio = _FakeAudioClip(2.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "src.mp4")
            Path(clip_file).write_bytes(b"fake")
            with (
                patch.object(vd, "AudioFileClip", return_value=audio),
                patch.object(vd, "_open_video_clip_quietly", side_effect=fake_open),
                patch.object(
                    vd, "_write_videofile_with_codec_fallback", side_effect=fake_write
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg") as concat,
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    video_paths=[clip_file],
                    audio_file="audio.mp3",
                    max_clip_duration=1,
                    native_audio_enabled=native_audio_enabled,
                )
        return opened_with, write_kwargs, concat

    def test_default_mode_does_not_request_audio_from_the_reader(self):
        opened_with, write_kwargs, concat = self._combine(
            native_audio_enabled=False
        )
        # 不能传 audio 关键字（连 audio=False 都不行），保持原有调用形态。
        for kwargs in opened_with:
            self.assertNotIn("audio", kwargs)
        # 中间片段不写音轨。
        for kwargs in write_kwargs:
            self.assertNotIn("audio_codec", kwargs)
        self.assertFalse(concat.call_args.kwargs["include_audio"])

    def test_native_mode_requests_audio_and_writes_aac_48k(self):
        opened_with, write_kwargs, concat = self._combine(native_audio_enabled=True)
        self.assertTrue(any(kwargs.get("audio") is True for kwargs in opened_with))
        written = [kwargs for kwargs in write_kwargs if "audio_codec" in kwargs]
        self.assertTrue(written)
        self.assertEqual(written[0]["audio_codec"], "aac")
        self.assertEqual(written[0]["audio_fps"], 48000)
        self.assertTrue(concat.call_args.kwargs["include_audio"])


class TestNativeAudioValidation(unittest.TestCase):
    def _params(self, **overrides):
        return VideoParams(video_subject="s", **overrides)

    def test_disabled_skips_validation(self):
        # 关闭时即使音量非法也不该报错——该字段不参与任何计算。
        vd.validate_native_audio(
            self._params(native_audio_enabled=False, native_audio_volume=-1)
        )

    def test_accepts_finite_non_negative_volume(self):
        for volume in (0, 0.3, 1.0, 2.5):
            with self.subTest(volume=volume):
                vd.validate_native_audio(
                    self._params(
                        native_audio_enabled=True, native_audio_volume=volume
                    )
                )

    def test_rejects_negative_volume(self):
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            vd.validate_native_audio(
                self._params(native_audio_enabled=True, native_audio_volume=-0.1)
            )

    def test_schema_refuses_non_finite_volume(self):
        # allow_inf_nan=False 让 NaN/Infinity 在构造 params 时就被拒绝。
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    self._params(
                        native_audio_enabled=True, native_audio_volume=value
                    )


class TestGenerateVideoNativeMix(unittest.TestCase):
    """最终混音：原生音轨作为背景层与旁白叠加，而不是替换旁白。"""

    def setUp(self):
        self.original_app = dict(config.app)
        config.app.update({"video_codec": "libx264", "bgm_type": "none"})

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    def _generate(self, *, native_audio_enabled: bool):
        native_track = _FakeAudioClip(5.0)
        source_video = _FakeVideoClip(5.0, audio=native_track)
        opened = []

        def fake_open(path, *args, **kwargs):
            opened.append(kwargs)
            return source_video

        voice = _FakeAudioClip(5.0)
        composites = []

        class _RecordingComposite:
            def __init__(self, clips=None, *args, **kwargs):
                composites.append(clips)

            def close(self):
                pass

        params = VideoParams(
            video_subject="s",
            subtitle_enabled=False,
            bgm_type="none",
            bgm_volume=0,
            voice_volume=1.0,
            native_audio_enabled=native_audio_enabled,
            native_audio_volume=0.3,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "_open_video_clip_quietly", side_effect=fake_open),
                patch.object(vd, "AudioFileClip", return_value=voice),
                patch.object(vd, "CompositeAudioClip", _RecordingComposite),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "_stage_heartbeat"),
            ):
                vd.generate_video(
                    video_path="combined.mp4",
                    audio_path="audio.mp3",
                    subtitle_path="",
                    output_file=os.path.join(temp_dir, "final.mp4"),
                    params=params,
                )
        return opened, composites, voice, native_track

    def test_default_mode_keeps_single_voice_track(self):
        opened, composites, voice, native_track = self._generate(
            native_audio_enabled=False
        )
        for kwargs in opened:
            self.assertNotIn("audio", kwargs)
        # 没有合成多条音轨，说明旁白仍是唯一音源。
        self.assertEqual(composites, [])

    def test_native_mode_mixes_voice_with_native_background_track(self):
        opened, composites, voice, native_track = self._generate(
            native_audio_enabled=True
        )
        self.assertTrue(any(kwargs.get("audio") is True for kwargs in opened))
        # 旁白 + 原生音轨两条都进入混音，原生音轨没有被用来替换旁白。
        self.assertTrue(composites)
        self.assertIn(voice, composites[0])
        self.assertIn(native_track, composites[0])

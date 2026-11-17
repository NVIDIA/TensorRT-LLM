# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

from tensorrt_llm.inputs import media_io  # noqa: E402
from tensorrt_llm.inputs.media_io import VideoMediaIO, _load_video_by_cv2  # noqa: E402
from tensorrt_llm.inputs.multimodal import default_hasher  # noqa: E402
from tensorrt_llm.inputs.multimodal_data import VideoData  # noqa: E402
from tensorrt_llm.inputs.utils import async_load_video, load_base64_video, load_video  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Encode a tiny mp4 with distinguishable per-frame pixel values."""
    width, height, num_frames = 64, 64, 20
    path = tmp_path_factory.mktemp("video_decode") / "sample video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), (i * 10) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def seek_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Encode enough tiny frames to exercise sparse sampling and seeking."""
    path = tmp_path_factory.mktemp("video_seek") / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    assert writer.isOpened(), "OpenCV must support the mp4v test codec"
    rng = np.random.default_rng(0)
    for _ in range(1025):
        writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def _sequential_sample(path: str, num_frames: int = 3) -> tuple[list[int], np.ndarray]:
    capture = cv2.VideoCapture(path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()
        frames = []
        valid_indices = []
        for index in range(indices[-1] + 1):
            if not capture.grab():
                break
            if index in indices:
                ok, frame = capture.retrieve()
                assert ok
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                valid_indices.append(index)
        return valid_indices, np.stack(frames)
    finally:
        capture.release()


# A real H.264 clip with B-frames (keyframes every 4 frames) kept for other tests.
_H264_B_FRAMES_VIDEO = (
    Path(__file__).parents[1] / "_torch/visual_gen/test_data/cosmos3_v2v_ref_9f_bframes.mp4"
)


def _mp4_box(kind: bytes, payload: bytes = b"") -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + kind + payload


def _mp4_timing_box(kind: bytes, entries: list[tuple[int, int]], version: int = 0) -> bytes:
    payload = bytearray()
    for count, value in entries:
        payload.extend(count.to_bytes(4, "big"))
        payload.extend(value.to_bytes(4, "big", signed=version == 1))
    return _mp4_box(kind, bytes([version]) + bytes(3) + len(entries).to_bytes(4, "big") + payload)


def _mp4_sync_box(samples: list[int]) -> bytes:
    numbers = np.asarray(samples, dtype=">u4").tobytes()
    return _mp4_box(b"stss", bytes(4) + len(samples).to_bytes(4, "big") + numbers)


def _mp4_index(table: bytes, track_extra: bytes = b"") -> bytes:
    """Return the `ftyp` and `moov` boxes of a one-video-track MP4 with sample table `table`."""
    media = _mp4_box(b"hdlr", bytes(8) + b"vide") + _mp4_box(b"minf", _mp4_box(b"stbl", table))
    track = _mp4_box(b"mdia", media) + track_extra
    return _mp4_box(b"ftyp", b"isom" + bytes(4)) + _mp4_box(b"moov", _mp4_box(b"trak", track))


@pytest.mark.parametrize("format", ["np", "pt", "pil"])
def test_sparse_seek_matches_sequential_frames(seek_video_path: str, format: str) -> None:
    indices, expected = _sequential_sample(seek_video_path)
    capture_type = cv2.VideoCapture
    captures = []

    def open_capture(*args: object) -> Mock:
        capture = Mock(wraps=capture_type(*args))
        captures.append(capture)
        return capture

    with patch.object(cv2, "VideoCapture", side_effect=open_capture):
        video = _load_video_by_cv2(seek_video_path, num_frames=3, fps=-1, format=format)

    assert video.metadata["frames_indices"] == indices
    if hasattr(cv2, "CAP_PROP_PTS"):
        assert any(capture.set.called for capture in captures)
    if format == "pt":
        expected_pt = expected.astype(np.float32)
        expected_pt *= 1.0 / 255.0
        assert torch.equal(
            torch.stack(video.frames), torch.from_numpy(expected_pt).permute(0, 3, 1, 2)
        )
    else:
        actual = video.frames if format == "np" else np.stack([np.asarray(f) for f in video.frames])
        assert actual.dtype == np.uint8
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "failure", ["set", "position", "pts", "nan_msec", "grab_error", "get_error"]
)
def test_sparse_seek_failure_falls_back_to_sequential(
    seek_video_path: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(cv2, "CAP_PROP_PTS"):
        pytest.skip("OpenCV cannot verify seek timestamps")
    monkeypatch.setattr(media_io, "_video_seek_fallbacks", 0)
    indices, expected = _sequential_sample(seek_video_path)
    capture_type = cv2.VideoCapture
    captures = []

    def open_capture(*args: object) -> Mock:
        source = capture_type(*args)
        capture = Mock(wraps=source)
        captures.append(capture)
        if len(captures) == 1:
            if failure == "set":
                capture.set.side_effect = lambda key, value: source.set(key, value) and False
            elif failure == "grab_error":

                def grab() -> bool:
                    if capture.set.called:
                        raise cv2.error("seek decode failed")
                    return source.grab()

                capture.grab.side_effect = grab
            else:

                def get(key: int) -> float:
                    if capture.set.called:
                        if failure == "get_error":
                            raise cv2.error("seek position unavailable")
                        if failure == "nan_msec" and key == cv2.CAP_PROP_POS_MSEC:
                            return float("nan")
                        property_id = (
                            cv2.CAP_PROP_POS_FRAMES if failure == "position" else cv2.CAP_PROP_PTS
                        )
                        if key == property_id and failure in ("position", "pts"):
                            return -1
                    return source.get(key)

                capture.get.side_effect = get
        return capture

    with patch.object(cv2, "VideoCapture", side_effect=open_capture):
        video = _load_video_by_cv2(seek_video_path, num_frames=3, fps=-1, format="np")

    assert captures[0].set.called
    assert len(captures) >= 2
    assert media_io._video_seek_fallbacks == 1
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


@pytest.mark.parametrize("failure", ["backend_error", "first_pts_error", "second_msec_nan"])
def test_sparse_seek_unverifiable_preflight_stays_sequential(
    seek_video_path: str, failure: str
) -> None:
    indices, expected = _sequential_sample(seek_video_path)
    source = cv2.VideoCapture(seek_video_path)
    capture = Mock(wraps=source)

    def get(key: int) -> float:
        position = source.get(cv2.CAP_PROP_POS_FRAMES)
        if failure == "backend_error" and key == cv2.CAP_PROP_BACKEND:
            raise cv2.error("backend unavailable")
        if (
            failure == "first_pts_error"
            and key == getattr(cv2, "CAP_PROP_PTS", None)
            and position == 1
        ):
            raise cv2.error("presentation timestamp unavailable")
        if failure == "second_msec_nan" and key == cv2.CAP_PROP_POS_MSEC and position == 2:
            return float("nan")
        return source.get(key)

    capture.get.side_effect = get
    with patch.object(cv2, "VideoCapture", return_value=capture):
        video = _load_video_by_cv2(seek_video_path, num_frames=3, fps=-1, format="np")
    capture.set.assert_not_called()
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


@pytest.mark.parametrize(
    "metadata_case",
    [
        "cfr",
        "integer_ctts",
        "signed_ctts",
        "valid_single_edit",
        "sync_frames",
        "long_gop",
        "bad_sync_frames",
        "complex_edits",
        "unknown",
        "multi_stts",
        "wrong_count",
        "fractional_ctts",
        "ctts_gap",
        "bad_extent",
        "fragmented",
    ],
)
def test_sparse_seek_requires_indexed_constant_rate_metadata(
    seek_video_path: str, metadata_case: str
) -> None:
    indices, expected = _sequential_sample(seek_video_path)

    timing = [(1025, 512)]
    if metadata_case == "multi_stts":
        timing = [(512, 512), (513, 512)]
    elif metadata_case == "wrong_count":
        timing = [(1024, 512)]
    table = _mp4_timing_box(b"stts", timing)
    if metadata_case == "integer_ctts":
        # Reordered presentation timestamps still cover one contiguous grid.
        table += _mp4_timing_box(b"ctts", [(1, 1024), (1, 0)] * 512 + [(1, 512)])
    elif metadata_case == "signed_ctts":
        table += _mp4_timing_box(b"ctts", [(1, 512), (1, -512)] * 512 + [(1, 0)], version=1)
    elif metadata_case == "fractional_ctts":
        table += _mp4_timing_box(b"ctts", [(1025, 1)])
    elif metadata_case == "ctts_gap":
        table += _mp4_timing_box(b"ctts", [(1, 0), (1024, 1024)])
    elif metadata_case == "sync_frames":
        table += _mp4_sync_box(list(range(1, 1026, 30)))
    elif metadata_case == "long_gop":
        # The only keyframe is the first frame: seeking would decode from it again.
        table += _mp4_sync_box([1])
    elif metadata_case == "bad_sync_frames":
        table += _mp4_sync_box([1, 2000])
    track_extra = b""
    if metadata_case in ("valid_single_edit", "complex_edits"):
        edit_count = 1 if metadata_case == "valid_single_edit" else 2
        edit = (1025 * 512).to_bytes(4, "big") + (1024).to_bytes(4, "big") + b"\0\1\0\0"
        track_extra = _mp4_box(
            b"edts", _mp4_box(b"elst", bytes(4) + edit_count.to_bytes(4, "big") + edit * edit_count)
        )
    metadata = _mp4_index(table, track_extra)
    if metadata_case == "unknown":
        metadata = b"not an indexed MP4"
    elif metadata_case == "bad_extent":
        metadata = metadata[:-1]
    elif metadata_case == "fragmented":
        metadata += _mp4_box(b"moof")

    capture = Mock(wraps=cv2.VideoCapture(seek_video_path))
    with (
        patch("builtins.open", side_effect=lambda *_args, **_kwargs: BytesIO(metadata)),
        patch.object(cv2, "VideoCapture", return_value=capture),
    ):
        video = _load_video_by_cv2(seek_video_path, num_frames=3, fps=-1, format="np")
    expected_seek = metadata_case in (
        "cfr",
        "integer_ctts",
        "signed_ctts",
        "valid_single_edit",
        "sync_frames",
    ) and hasattr(cv2, "CAP_PROP_PTS")
    assert capture.set.called == expected_seek
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


@pytest.mark.parametrize("sync_frames", [False, True])
def test_seek_anchors_follow_presentation_order(sync_frames: bool) -> None:
    # Decode order I P B B per group of four; composition offsets of 2/4/1/1
    # frames present them as I B B P, so each I frame is presentation frame 4k.
    count = 1200
    table = _mp4_timing_box(b"stts", [(count, 512)])
    table += _mp4_timing_box(b"ctts", [(1, 1024), (1, 2048), (2, 512)] * (count // 4))
    if sync_frames:
        table += _mp4_sync_box(list(range(1, count + 1, 4)))
    reader = BytesIO(_mp4_index(table))
    targets = [0, 600, 1199]
    assert media_io._iso_video_seek_anchors(reader, count, targets, preroll_frames=16) == (
        [-1, 584, 1180] if sync_frames else [-1, 584, 1183]
    )
    assert media_io._iso_video_seek_anchors(reader, count, targets, preroll_frames=0) == (
        [0, 600, 1196] if sync_frames else targets
    )


def test_seek_plan_first_frame_and_decode_ratio_boundary() -> None:
    indices = [0, 129]
    # With two initial frames, decoding frames 67 through 129 costs exactly 65 of 130.
    assert media_io._plan_video_seeks(indices, [0, 67], first_frame=2) == {2: 129}
    assert media_io._plan_video_seeks(indices, [0, 66], first_frame=2) is None
    assert media_io._plan_video_seeks(indices, [0, 66], first_frame=0) == {1: 129}


def test_dense_sampling_keeps_sequential_decode(seek_video_path: str) -> None:
    # No gap reaches the seek threshold, so decoding matches the sequential
    # path call for call: no index read and no timestamp or backend queries.
    indices, expected = _sequential_sample(seek_video_path, num_frames=32)
    assert max(b - a for a, b in zip(indices, indices[1:])) < media_io._VIDEO_SEEK_MIN_SKIP_FRAMES
    capture = Mock(wraps=cv2.VideoCapture(seek_video_path))
    with (
        patch("builtins.open", side_effect=AssertionError("index must not be read")),
        patch.object(cv2, "VideoCapture", return_value=capture),
    ):
        video = _load_video_by_cv2(seek_video_path, num_frames=32, fps=-1, format="np")
    queried = {call.args[0] for call in capture.get.call_args_list}
    assert queried == {cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS}
    capture.set.assert_not_called()
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


def test_sparse_seek_h264_b_frames_fixture_matches_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The clip is short, so lower the thresholds to seek between its keyframes.
    if not hasattr(cv2, "CAP_PROP_PTS"):
        pytest.skip("OpenCV cannot verify seek timestamps")
    monkeypatch.setattr(media_io, "_VIDEO_SEEK_MIN_SKIP_FRAMES", 1)
    monkeypatch.setattr(media_io, "_CV2_SEEK_PREROLL_FRAMES", 0)
    path = str(_H264_B_FRAMES_VIDEO)
    indices, expected = _sequential_sample(path)
    capture_type = cv2.VideoCapture
    captures = []

    def open_capture(*args: object) -> Mock:
        capture = Mock(wraps=capture_type(*args))
        captures.append(capture)
        return capture

    with patch.object(cv2, "VideoCapture", side_effect=open_capture):
        video = _load_video_by_cv2(path, num_frames=3, fps=-1, format="np")
    # Seeks were verified on the reordered stream without a sequential replay.
    assert len(captures) == 1
    assert captures[0].set.call_count == 2
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


def test_sparse_seek_bytes_matches_sequential_frames(seek_video_path: str) -> None:
    backend = media_io._select_cv2_stream_buffered_backend()
    if backend is None:
        pytest.skip("OpenCV has no stream-buffered video backend")
    indices, expected = _sequential_sample(seek_video_path)
    video = _load_video_by_cv2(
        Path(seek_video_path).read_bytes(),
        num_frames=3,
        fps=-1,
        format="np",
        cv2_backend=backend,
    )
    assert video.metadata["frames_indices"] == indices
    assert np.array_equal(video.frames, expected)


def test_sparse_seek_overstated_frame_count_stops_at_eof(seek_video_path: str) -> None:
    indices, expected = _sequential_sample(seek_video_path)
    capture_type = cv2.VideoCapture

    def open_capture(*args: object) -> Mock:
        source = capture_type(*args)
        capture = Mock(wraps=source)
        capture.get.side_effect = lambda key: (
            2049 if key == cv2.CAP_PROP_FRAME_COUNT else source.get(key)
        )
        return capture

    with patch.object(cv2, "VideoCapture", side_effect=open_capture):
        video = _load_video_by_cv2(seek_video_path, num_frames=3, fps=-1, format="np")

    assert video.metadata["frames_indices"] == [indices[0], indices[-1]]
    assert np.array_equal(video.frames, expected[[0, -1]])
    assert video.frames.base is None


class _KeyframeCapture:
    """Constant-rate capture that counts decoded frames like OpenCV's FFmpeg backend.

    `grab()` decodes one frame. A seek to `target` decodes from the last
    keyframe at or before `target - 16` up to the frame before `target`, and
    the next `grab()` decodes `target`.
    """

    def __init__(self, frame_count: int, keyframes: list[int], fail_seeks: bool) -> None:
        self.frame_count = frame_count
        self.keyframes = keyframes
        self.fail_seeks = fail_seeks
        self.position = 0
        self.decoded = 0
        self.seeks: list[int] = []

    def isOpened(self) -> bool:
        return True

    def get(self, key: int) -> float:
        last = self.position - 1
        return {
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_BACKEND: cv2.CAP_FFMPEG,
            cv2.CAP_PROP_POS_FRAMES: self.position,
            cv2.CAP_PROP_PTS: last,
            cv2.CAP_PROP_POS_MSEC: last * 1000 / 30.0,
        }[key]

    def grab(self) -> bool:
        if self.position >= self.frame_count:
            return False
        self.position += 1
        self.decoded += 1
        return True

    def set(self, key: int, target: float) -> bool:
        assert key == cv2.CAP_PROP_POS_FRAMES
        target = int(target)
        self.seeks.append(target)
        self.decoded += target - max(k for k in self.keyframes if k <= max(target - 16, 0))
        self.position = target
        return not self.fail_seeks

    def retrieve(self, scratch: np.ndarray | None = None) -> tuple[bool, np.ndarray]:
        return True, np.full((2, 2, 3), (self.position - 1) % 256, dtype=np.uint8)

    def release(self) -> None:
        pass


def _keyframe_index(frame_count: int, keyframes: list[int]) -> bytes:
    table = _mp4_timing_box(b"stts", [(frame_count, 512)])
    return _mp4_index(table + _mp4_sync_box([k + 1 for k in keyframes]))


def _decode_with_keyframes(
    frame_count: int, keyframes: list[int], num_frames: int, fail_first_seek: bool = False
) -> tuple[VideoData, list[_KeyframeCapture]]:
    metadata = _keyframe_index(frame_count, keyframes)
    captures: list[_KeyframeCapture] = []

    def open_capture(*_args: object) -> _KeyframeCapture:
        captures.append(_KeyframeCapture(frame_count, keyframes, fail_first_seek and not captures))
        return captures[-1]

    with (
        patch("builtins.open", side_effect=lambda *_args, **_kwargs: BytesIO(metadata)),
        patch.object(cv2, "VideoCapture", side_effect=open_capture),
    ):
        video = _load_video_by_cv2("video.mp4", num_frames=num_frames, fps=-1, format="np")
    return video, captures


@pytest.mark.parametrize(
    ("frame_count", "num_frames", "max_ratio", "planned_frames"),
    [
        # About 17 minutes at 30 fps with a keyframe every 250 frames.
        (30000, 10, None, 1502),
        (30000, 64, None, 9002),
        # A 785-frame clip: its one seek saves too little, so the gate keeps it sequential.
        (785, 10, None, 721),
        (785, 10, 1.0, 721),
        # Many samples between keyframes: some seeks pay, but not most of the decode.
        (30000, 128, None, 18803),
        (30000, 128, 1.0, 18803),
    ],
)
def test_seek_plan_gates_and_matches_decoded_frames(
    frame_count: int,
    num_frames: int,
    max_ratio: float | None,
    planned_frames: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(cv2, "CAP_PROP_PTS"):
        pytest.skip("OpenCV cannot verify seek timestamps")
    if max_ratio is not None:
        monkeypatch.setattr(media_io, "_VIDEO_SEEK_MAX_DECODE_RATIO", max_ratio)
    keyframes = list(range(0, frame_count, 250))
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()
    anchors = media_io._iso_video_seek_anchors(
        BytesIO(_keyframe_index(frame_count, keyframes)),
        frame_count,
        indices,
        media_io._CV2_SEEK_PREROLL_FRAMES,
    )
    assert anchors is not None
    seeks = media_io._plan_video_seeks(indices, anchors, first_frame=2)

    video, captures = _decode_with_keyframes(frame_count, keyframes, num_frames)

    [capture] = captures
    assert video.metadata["frames_indices"] == indices
    assert video.frames[:, 0, 0, 0].tolist() == [i % 256 for i in indices]
    if planned_frames <= media_io._VIDEO_SEEK_MAX_DECODE_RATIO * frame_count:
        # The loop seeks exactly where the plan does and decodes what it counted.
        assert seeks and capture.seeks == list(seeks.values())
        assert capture.decoded == planned_frames
    else:
        # The index is read, but the weak plan keeps the sequential path.
        assert seeks is None
        assert capture.seeks == []
        assert capture.decoded == frame_count


def test_seek_fallback_warns_once_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(cv2, "CAP_PROP_PTS"):
        pytest.skip("OpenCV cannot verify seek timestamps")
    monkeypatch.setattr(media_io, "_video_seek_fallbacks", 0)
    keyframes = list(range(0, 30000, 250))
    indices = np.linspace(0, 29999, 10, dtype=int).tolist()
    with patch.object(media_io.logger, "warning") as warning:
        for _ in range(2):
            video, captures = _decode_with_keyframes(30000, keyframes, 10, fail_first_seek=True)
            # The failed seek reopens the video; the new capture decodes it in order.
            assert [len(capture.seeks) for capture in captures] == [1, 0]
            assert captures[1].decoded == 30000
            assert video.metadata["frames_indices"] == indices
    assert media_io._video_seek_fallbacks == 2
    warning.assert_called_once()
    assert "decoding the video sequentially" in warning.call_args.args[0]


def test_np_format_returns_stacked_uint8_ndarray(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    assert isinstance(video.frames, np.ndarray)
    assert video.frames.shape == (10, 64, 64, 3)
    assert video.frames.dtype == np.uint8
    assert video.frames.flags["C_CONTIGUOUS"]


def test_pt_format_returns_list_of_chw_tensors(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="pt")
    numpy_video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    expected = numpy_video.frames.astype(np.float32)
    expected *= 1.0 / 255.0
    assert isinstance(video.frames, list)
    assert len(video.frames) == 10
    for frame in video.frames:
        assert isinstance(frame, torch.Tensor)
        assert frame.shape == (3, 64, 64)
        assert frame.dtype == torch.float32
        assert frame.is_contiguous()
        assert 0.0 <= frame.min().item() and frame.max().item() <= 1.0
    torch.testing.assert_close(
        torch.stack(video.frames), torch.from_numpy(expected).permute(0, 3, 1, 2), rtol=0, atol=0
    )
    assert video.metadata == numpy_video.metadata
    # The output stays on CPU even when an application changes the default device.
    with torch.device("meta"):
        default_device_video = _load_video_by_cv2(
            sample_video_path, num_frames=10, fps=-1, format="pt"
        )
    torch.testing.assert_close(torch.stack(default_device_video.frames), torch.stack(video.frames))


def test_pil_format_returns_list_of_pil_images(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="pil")
    assert isinstance(video.frames, list)
    assert len(video.frames) == 10
    for frame in video.frames:
        assert isinstance(frame, Image.Image)
        assert frame.size == (64, 64)


def test_np_format_hits_hf_video_processor_fast_path(sample_video_path: str) -> None:
    """HF `make_batched_videos` returns a 4D ndarray input without copying."""
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    batched = make_batched_videos([video.frames])

    assert len(batched) == 1
    assert np.shares_memory(video.frames, batched[0])


@pytest.mark.parametrize(("format", "peak_bytes_per_value"), [("np", 2), ("pil", 2), ("pt", 5)])
def test_decode_limit_is_checked_before_frame_stack_allocation(
    sample_video_path: str, format: str, peak_bytes_per_value: int
) -> None:
    peak_bytes = 10 * 64 * 64 * 3 * peak_bytes_per_value

    with patch("tensorrt_llm.inputs.media_io.np.empty") as allocate:
        with pytest.raises(ValueError, match=f"require {peak_bytes} bytes"):
            _load_video_by_cv2(
                sample_video_path,
                num_frames=10,
                fps=-1,
                format=format,
                max_decoded_frame_bytes=peak_bytes - 1,
            )

    allocate.assert_not_called()


def test_pt_decode_reserves_peak_then_commits_retained_bytes(sample_video_path: str) -> None:
    class _Reservation:
        committed = None
        released = False

        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            self.committed = retained_bytes

        def release(self) -> None:
            self.released = True

    reservation = _Reservation()
    reserved = []

    def reserve(peak_bytes: int, retained_bytes: int) -> _Reservation:
        reserved.append((peak_bytes, retained_bytes))
        return reservation

    _load_video_by_cv2(
        sample_video_path,
        num_frames=10,
        fps=-1,
        format="pt",
        decoded_memory_reserver=reserve,
    )

    assert reserved == [(10 * 64 * 64 * 3 * 5, 10 * 64 * 64 * 3 * 4)]
    assert reservation.committed == 10 * 64 * 64 * 3 * 4
    assert reservation.released is False


def test_decode_releases_reservation_after_allocation_failure(sample_video_path: str) -> None:
    class _Reservation:
        released = False

        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            raise AssertionError("failed decode must not commit")

        def release(self) -> None:
            self.released = True

    reservation = _Reservation()

    with patch("tensorrt_llm.inputs.media_io.np.empty", side_effect=MemoryError("allocation")):
        with pytest.raises(MemoryError, match="allocation"):
            _load_video_by_cv2(
                sample_video_path,
                num_frames=10,
                fps=-1,
                format="np",
                decoded_memory_reserver=lambda _peak_bytes, _retained_bytes: reservation,
            )

    assert reservation.released is True


@pytest.mark.parametrize("end_frame, failed_frame", [(20, None), (15, 8)])
def test_video_conversion_preserves_sequential_sampling(
    monkeypatch: pytest.MonkeyPatch, end_frame: int, failed_frame: int | None
) -> None:
    """Keep decode-order sampling, RGB conversion and readable-frame truncation."""
    bgr = np.arange(4 * 64 * 3, dtype=np.uint8).reshape(4, 64, 3)
    captures = []

    class SequentialCapture:
        def __init__(self, *args: Any) -> None:
            self.next_frame = 0
            self.grabbed = []
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> int:
            return {cv2.CAP_PROP_FRAME_COUNT: 20, cv2.CAP_PROP_FPS: 30}[prop]

        def grab(self) -> bool:
            if self.next_frame >= end_frame:
                return False
            self.grabbed.append(self.next_frame)
            self.next_frame += 1
            return True

        def retrieve(self, scratch: np.ndarray | None) -> tuple[bool, np.ndarray]:
            frame_idx = self.next_frame - 1
            return frame_idx != failed_frame, np.roll(bgr, frame_idx, axis=1)

        def set(self, *args: Any) -> None:
            pytest.fail("decode-order video sampling must not seek")

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", SequentialCapture)
    decoded = _load_video_by_cv2("unused.mp4", num_frames=10, fps=-1, format="pt")
    indices = [i for i in np.linspace(0, 19, 10, dtype=int) if i < end_frame and i != failed_frame]
    expected = np.stack([np.roll(bgr, int(i), axis=1)[..., ::-1] for i in indices]).astype(
        np.float32
    )
    expected *= 1.0 / 255.0
    torch.testing.assert_close(
        torch.stack(decoded.frames), torch.from_numpy(expected).permute(0, 3, 1, 2), rtol=0, atol=0
    )
    assert decoded.metadata["frames_indices"] == indices
    assert captures[0].grabbed == list(range(end_frame))
    assert captures[0].released


def test_truncated_numpy_video_owns_only_the_counted_frames(sample_video_path: str) -> None:
    capture = cv2.VideoCapture(sample_video_path)
    get = capture.get
    committed = []

    class _Reservation:
        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            committed.append(retained_bytes)

        def release(self) -> None:
            pass

    # Container metadata overstates the number of frames that actually decode.
    truncated = Mock(wraps=capture)
    truncated.get.side_effect = lambda key: 50 if key == cv2.CAP_PROP_FRAME_COUNT else get(key)
    with patch.object(cv2, "VideoCapture", return_value=truncated):
        video = _load_video_by_cv2(
            sample_video_path,
            num_frames=10,
            fps=-1,
            format="np",
            decoded_memory_reserver=lambda _peak, _retained: _Reservation(),
        )

    assert 0 < len(video.frames) < 10
    assert video.frames.base is None
    assert committed == [video.frames.nbytes]


@pytest.mark.parametrize("input_kind", ["path", "file_uri", "data_uri"])
@pytest.mark.parametrize("force_tempfile", [False, True], ids=["auto_backend", "tempfile"])
def test_offline_loaders_match_serve_video_hash(
    sample_video_path: str,
    input_kind: str,
    force_tempfile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline loaders must produce the same cache identity as serve's `VideoMediaIO`."""

    def digest(video: VideoData) -> str:
        """Hash the video data and sampling metadata."""
        hasher = default_hasher()
        video.update_hash(hasher)
        return hasher.hexdigest()

    path = Path(sample_video_path)
    encoded = path.read_bytes()
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", len(encoded))
    if input_kind == "file_uri":
        source = path.as_uri()
    elif input_kind == "data_uri":
        source = "data:video/mp4;base64," + base64.b64encode(encoded).decode("ascii")
    else:
        source = str(path)
    if force_tempfile:
        monkeypatch.setattr(
            "tensorrt_llm.inputs.media_io._select_cv2_stream_buffered_backend", lambda: None
        )

    serve = asyncio.run(VideoMediaIO(num_frames=10).async_load(source))
    offline = load_video(source, num_frames=10)
    offline_async = asyncio.run(async_load_video(source, num_frames=10))

    assert serve.raw_bytes_hash == default_hasher(encoded).hexdigest()
    assert offline.raw_bytes_hash == serve.raw_bytes_hash
    assert offline_async.raw_bytes_hash == serve.raw_bytes_hash
    assert digest(offline) == digest(serve) == digest(offline_async)


@pytest.mark.parametrize("file_uri", [False, True], ids=["path", "file_uri"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("read_error", [FileNotFoundError, PermissionError])
def test_offline_local_video_read_errors_are_value_errors(
    tmp_path: Path,
    file_uri: bool,
    use_async: bool,
    read_error: type[OSError],
) -> None:
    """Local read failures retain the offline loaders' ValueError contract."""
    path = tmp_path / "input video.mp4"
    path.write_bytes(b"not decoded")
    source = path.as_uri() if file_uri else str(path)
    failure = read_error("cannot read video")

    with patch.object(Path, "open", side_effect=failure):
        with pytest.raises(ValueError) as error:
            if use_async:
                asyncio.run(async_load_video(source))
            else:
                load_video(source)

    assert error.value.__cause__ is failure


@pytest.mark.parametrize("file_uri", [False, True], ids=["path", "file_uri"])
@pytest.mark.parametrize("loader", ["serve", "sync", "async"])
def test_oversized_local_video_is_rejected_before_read(
    sample_video_path: str,
    file_uri: bool,
    loader: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(sample_video_path)
    source = path.as_uri() if file_uri else str(path)
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", path.stat().st_size - 1)

    with patch.object(Path, "open", side_effect=AssertionError("Oversized file was opened")):
        with pytest.raises(ValueError, match="maximum allowed size"):
            if loader == "serve":
                asyncio.run(VideoMediaIO().async_load(source))
            elif loader == "async":
                asyncio.run(async_load_video(source))
            else:
                load_video(source)


def test_local_media_read_is_bounded_if_file_grows(tmp_path: Path) -> None:
    path = tmp_path / "growing.mp4"
    path.write_bytes(b"x")
    source = BytesIO(b"x" * 100)
    with (
        patch.object(Path, "open", return_value=source),
        patch.object(source, "read", wraps=source.read) as read,
    ):
        with pytest.raises(ValueError, match="maximum allowed size"):
            media_io._safe_read_local_file(str(path), max_bytes=4)
        requested_sizes = [call.args[0] for call in read.call_args_list]
        assert requested_sizes and all(size >= 0 for size in requested_sizes)
        assert sum(requested_sizes) <= 5
    assert source.closed


def test_oversized_video_bytes_are_rejected_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    with patch.object(media_io, "blake3", side_effect=AssertionError("Oversized video was hashed")):
        with pytest.raises(ValueError, match="maximum allowed size"):
            VideoMediaIO().load_bytes(b"video")


@pytest.mark.parametrize("loader", ["serve", "sync", "async", "base64_helper"])
@pytest.mark.parametrize("extra_bytes", [1, 4])
def test_oversized_base64_video_is_rejected(
    loader: str,
    extra_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    encoded = base64.b64encode(b"x" * (4 + extra_bytes)).decode("ascii")
    source = "data:video/mp4;base64," + encoded
    decoder = media_io.base64.b64decode
    with (
        patch.object(media_io.base64, "b64decode", wraps=decoder) as decode,
        patch.object(media_io, "blake3", side_effect=AssertionError("Oversized video was hashed")),
    ):
        with pytest.raises(ValueError, match="maximum allowed size"):
            if loader == "serve":
                asyncio.run(VideoMediaIO().async_load(source))
            elif loader == "async":
                asyncio.run(async_load_video(source))
            elif loader == "base64_helper":
                load_base64_video(source)
            else:
                load_video(source)
        if extra_bytes == 4:
            decode.assert_not_called()


def test_base64_video_helper_accepts_input_at_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    assert load_base64_video("data:video/mp4;base64,eHh4eA==") == b"xxxx"

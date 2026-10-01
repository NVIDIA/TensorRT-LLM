# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

import tensorrt_llm.inputs.media_io as media_io_module  # noqa: E402
from tensorrt_llm.inputs.media_io import _load_video_by_cv2  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Encode a tiny mp4 with distinguishable per-frame pixel values."""
    width, height, num_frames = 64, 64, 20
    path = tmp_path_factory.mktemp("video_decode") / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), (i * 10) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def test_np_format_returns_stacked_uint8_ndarray(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    assert isinstance(video.frames, np.ndarray)
    assert video.frames.shape == (10, 64, 64, 3)
    assert video.frames.dtype == np.uint8
    assert video.frames.flags["C_CONTIGUOUS"]


def test_pt_format_returns_list_of_chw_tensors(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="pt")
    assert isinstance(video.frames, list)
    assert len(video.frames) == 10
    for frame in video.frames:
        assert isinstance(frame, torch.Tensor)
        assert frame.shape == (3, 64, 64)
        assert frame.dtype == torch.float32
        assert 0.0 <= frame.min().item() and frame.max().item() <= 1.0


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


@pytest.mark.parametrize(
    "case, expected_opens, expected_seeks",
    [
        ("constant_frame_rate", 1, 7),
        ("seek_failure", 2, 2),
        ("variable_frame_rate", 1, 0),
        ("seek_budget", 1, 1),
        ("mpeg_program_stream", 1, 0),
    ],
)
def test_sparse_seek_matches_sequential_decode(
    sample_video_path: str,
    tmp_path: Path,
    monkeypatch,
    case: str,
    expected_opens: int,
    expected_seeks: int,
) -> None:
    video_path = sample_video_path
    if case == "mpeg_program_stream":
        # Only ISO-BMFF input seeks: after a seek in an MPEG program stream,
        # OpenCV can decode a different frame than the position it reports.
        video_path = str(tmp_path / "sample.mpg")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"PIM1"), 30, (64, 64))
        for i in range(60):
            writer.write(np.full((64, 64, 3), (i * 10) % 256, dtype=np.uint8))
        writer.release()
    sequential = _load_video_by_cv2(video_path, num_frames=10, fps=-1, format="np")
    original_video_capture = cv2.VideoCapture
    captures = []

    class FaultInjectingCapture:
        def __init__(self, *args, **kwargs):
            self.capture = original_video_capture(*args, **kwargs)
            self.seek_calls = 0
            captures.append(self)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.seek_calls += 1
                if case == "seek_failure" and self.seek_calls == 2:
                    return False
            return self.capture.set(prop, value)

        def get(self, prop):
            value = self.capture.get(prop)
            if case == "variable_frame_rate" and prop == cv2.CAP_PROP_POS_MSEC:
                # Timestamps that disagree with the average frame rate.
                return 1.5 * value
            return value

        def __getattr__(self, name):
            return getattr(self.capture, name)

    monkeypatch.setattr(media_io_module, "_VIDEO_SPARSE_SEEK_MIN_FRAME_RATIO", 1)
    monkeypatch.setattr(media_io_module, "_VIDEO_SPARSE_SEEK_CALIBRATION_FRAMES", 1)
    # Frames 0-2 are decoded in order. Seeks to frames 4-16 add up to 70 frames,
    # so a frame budget of 4 * 20 frames leaves frame 19 to be decoded in order.
    monkeypatch.setattr(media_io_module, "_VIDEO_SPARSE_SEEK_MAX_FRAME_FACTOR", 4)
    # Seeking this short clip costs more than decoding it in order, so only the
    # budget case keeps a time budget: zero, which stops after the first seek.
    monkeypatch.setattr(
        media_io_module,
        "_VIDEO_SPARSE_SEEK_TIME_BUDGET",
        0 if case == "seek_budget" else float("inf"),
    )
    monkeypatch.setattr(cv2, "VideoCapture", FaultInjectingCapture)
    sparse = _load_video_by_cv2(video_path, num_frames=10, fps=-1, format="np")

    assert len(captures) == expected_opens
    assert sum(capture.seek_calls for capture in captures) == expected_seeks
    np.testing.assert_array_equal(sparse.frames, sequential.frames)
    assert sparse.metadata == sequential.metadata

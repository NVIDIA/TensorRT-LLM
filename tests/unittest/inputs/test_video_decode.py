# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

from tensorrt_llm.inputs import media_io  # noqa: E402
from tensorrt_llm.inputs.media_io import _load_video_by_cv2  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("num_frames,expected_sampled", [(0, False), (32, True), (1000, False)])
def test_real_codec_with_injected_interior_retrieve_failure(
    tmp_path, monkeypatch, num_frames, expected_sampled
) -> None:
    """Run the real loader/codec and fail one retrieve at the decoder boundary."""
    path = str(tmp_path / "interior.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
    assert writer.isOpened()
    for i in range(700):
        writer.write(np.full((32, 32, 3), i % 256, dtype=np.uint8))
    writer.release()
    capture = cv2.VideoCapture(path)

    class FaultyCapture:
        index = -1

        def isOpened(self):
            return capture.isOpened()

        def get(self, prop):
            return capture.get(prop)

        def grab(self):
            self.index += 1
            return capture.grab()

        def retrieve(self, buffer):
            if self.index == 350:
                return False, buffer
            return capture.retrieve(buffer)

        def release(self):
            capture.release()

    backend = SimpleNamespace(
        VideoCapture=lambda _: FaultyCapture(),
        CAP_PROP_FRAME_COUNT=cv2.CAP_PROP_FRAME_COUNT,
        CAP_PROP_FPS=cv2.CAP_PROP_FPS,
        COLOR_BGR2RGB=cv2.COLOR_BGR2RGB,
        cvtColor=cv2.cvtColor,
    )
    monkeypatch.setattr(media_io, "_get_cv2", lambda: backend)
    video = _load_video_by_cv2(path, num_frames=num_frames, fps=0, format="np")
    assert video.is_sampled is expected_sampled
    expected = np.linspace(0, 699, 32 if expected_sampled else 700, dtype=int).tolist()
    expected = [i for i in expected if i != 350]
    assert video.metadata["frames_indices"] == expected
    assert len(video.frames) == len(expected)
    assert video.metadata["total_num_frames"] == 700
    assert video.metadata["fps"] == 30
    assert video.metadata["duration"] == 700 / 30


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

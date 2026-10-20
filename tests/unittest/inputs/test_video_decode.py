# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

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


@pytest.mark.parametrize(("format", "peak_bytes_per_value"), [("np", 1), ("pil", 1), ("pt", 9)])
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

        def commit(self, retained_bytes: int) -> None:
            self.committed = retained_bytes

        def release(self) -> None:
            self.released = True

    reservation = _Reservation()
    reserved = []

    def reserve(peak_bytes: int) -> _Reservation:
        reserved.append(peak_bytes)
        return reservation

    _load_video_by_cv2(
        sample_video_path,
        num_frames=10,
        fps=-1,
        format="pt",
        decoded_memory_reserver=reserve,
    )

    assert reserved == [10 * 64 * 64 * 3 * 9]
    assert reservation.committed == 10 * 64 * 64 * 3 * 4
    assert reservation.released is False


def test_decode_releases_reservation_after_allocation_failure(sample_video_path: str) -> None:
    class _Reservation:
        released = False

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
                decoded_memory_reserver=lambda _peak_bytes: reservation,
            )

    assert reservation.released is True

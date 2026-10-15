# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

from tensorrt_llm.inputs.media_io import VideoMediaIO, _load_video_by_cv2  # noqa: E402
from tensorrt_llm.inputs.multimodal import default_hasher  # noqa: E402
from tensorrt_llm.inputs.multimodal_data import VideoData  # noqa: E402
from tensorrt_llm.inputs.utils import async_load_video, load_video  # noqa: E402

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

    with patch.object(Path, "read_bytes", side_effect=failure):
        with pytest.raises(ValueError) as error:
            if use_async:
                asyncio.run(async_load_video(source))
            else:
                load_video(source)

    assert error.value.__cause__ is failure

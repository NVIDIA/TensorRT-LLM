# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from blake3 import blake3
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

import tensorrt_llm.inputs.media_io as media_io_module  # noqa: E402
from tensorrt_llm.inputs.media_io import VideoMediaIO, _load_video_by_cv2  # noqa: E402

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


@pytest.mark.parametrize("end_frame, failed_frame", [(20, None), (15, 8)])
def test_video_conversion_preserves_sequential_sampling(
    monkeypatch: pytest.MonkeyPatch, end_frame: int, failed_frame: int | None
) -> None:
    """Keep decode-order sampling, RGB conversion and readable-frame truncation."""
    bgr = np.arange(4 * 64 * 3, dtype=np.uint8).reshape(4, 64, 3)
    captures = []

    class SequentialCapture:
        def __init__(self, *args):
            self.next_frame = 0
            self.grabbed = []
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_COUNT: 20, cv2.CAP_PROP_FPS: 30}[prop]

        def grab(self):
            if self.next_frame >= end_frame:
                return False
            self.grabbed.append(self.next_frame)
            self.next_frame += 1
            return True

        def retrieve(self, scratch):
            frame_idx = self.next_frame - 1
            return frame_idx != failed_frame, np.roll(bgr, frame_idx, axis=1)

        def set(self, *args):
            pytest.fail("decode-order video sampling must not seek")

        def release(self):
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


def test_load_file_avoids_full_buffer_and_preserves_byte_identity(
    sample_video_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hash a stable local source in bounded reads using the same decoder backend."""
    source = Path(sample_video_path).read_bytes()
    media_io = VideoMediaIO(num_frames=10, fps=-1, format="np")
    baseline = media_io.load_bytes(source)
    original_decode = media_io_module._load_video_by_cv2
    calls = []

    def decode(*args, **kwargs):
        calls.append((args, kwargs))
        return original_decode(*args, **kwargs)

    def fail_read_bytes(path):
        pytest.fail(f"unexpected full-file buffer: {path}")

    monkeypatch.setattr(
        media_io_module, "_select_cv2_stream_buffered_backend", lambda: cv2.CAP_FFMPEG
    )
    monkeypatch.setattr(media_io_module, "_load_video_by_cv2", decode)
    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    video = media_io.load_file(Path(sample_video_path).as_uri())

    np.testing.assert_array_equal(video.frames, baseline.frames)
    assert video.metadata == baseline.metadata
    assert video.raw_bytes_hash == baseline.raw_bytes_hash == blake3(source).hexdigest()
    assert calls[0][0][0] == sample_video_path
    assert calls[0][1]["cv2_backend"] == cv2.CAP_FFMPEG

    class BytesOnlyHasher:
        def __init__(self):
            self.hasher = blake3()

        def update(self, data):
            assert isinstance(data, bytes)
            self.hasher.update(data)

    old_hasher, new_hasher = BytesOnlyHasher(), BytesOnlyHasher()
    baseline.update_hash(old_hasher)
    video.update_hash(new_hasher)
    assert old_hasher.hasher.hexdigest() == new_hasher.hasher.hexdigest()


def test_load_file_rejects_changed_source(sample_video_path: str, monkeypatch: pytest.MonkeyPatch):
    """Do not associate decoded frames with a hash after the file changes."""
    original_decode = media_io_module._load_video_by_cv2

    def decode_then_touch(*args, **kwargs):
        video = original_decode(*args, **kwargs)
        path = Path(sample_video_path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return video

    monkeypatch.setattr(
        media_io_module, "_select_cv2_stream_buffered_backend", lambda: cv2.CAP_FFMPEG
    )
    monkeypatch.setattr(media_io_module, "_load_video_by_cv2", decode_then_touch)
    with pytest.raises(ValueError, match="changed while it was being loaded"):
        VideoMediaIO(format="np").load_file(sample_video_path)


@pytest.mark.parametrize(
    "fallback", ["no_backend", "other_backend", "audio", "subclass", "instance"]
)
def test_load_file_preserves_byte_loader_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    """Keep byte-loader dispatch when direct path decoding is not qualified."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video data")
    calls = []
    expected = object()

    def load_bytes(self, data):
        calls.append(data)
        return expected

    class CustomVideoIO(VideoMediaIO):
        def load_bytes(self, data):
            return load_bytes(self, data)

    if fallback == "subclass":
        media_io = CustomVideoIO()
    else:
        media_io = VideoMediaIO(extract_audio=fallback == "audio")
        if fallback == "instance":
            media_io.load_bytes = lambda data: load_bytes(media_io, data)
        else:
            monkeypatch.setattr(VideoMediaIO, "load_bytes", load_bytes)
    monkeypatch.setattr(
        media_io_module,
        "_select_cv2_stream_buffered_backend",
        lambda: (
            None
            if fallback == "no_backend"
            else cv2.CAP_GSTREAMER
            if fallback == "other_backend"
            else cv2.CAP_FFMPEG
        ),
    )
    assert media_io.load_file(str(source)) is expected
    assert calls == [b"video data"]


@pytest.mark.parametrize("source_kind", ["missing", "directory", "invalid"])
def test_load_file_preserves_error_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    """Keep the errors raised by the original local-file byte path."""
    source = tmp_path / "source.mp4"
    if source_kind == "directory":
        source.mkdir()
    elif source_kind == "invalid":
        source.write_bytes(b"not a video")
    expected = {
        "missing": FileNotFoundError,
        "directory": IsADirectoryError,
        "invalid": ValueError,
    }[source_kind]
    monkeypatch.setattr(
        media_io_module, "_select_cv2_stream_buffered_backend", lambda: cv2.CAP_FFMPEG
    )
    with pytest.raises(expected):
        VideoMediaIO().load_file(str(source))


@pytest.mark.parametrize("source_kind", ["missing", "directory"])
def test_load_file_checks_source_before_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    """Source errors must win even if OpenCV is missing or broken."""
    source = tmp_path / "source.mp4"
    if source_kind == "directory":
        source.mkdir()

    def fail_probe():
        pytest.fail("decoder capability probe ran before checking the source")

    monkeypatch.setattr(media_io_module, "_select_cv2_stream_buffered_backend", fail_probe)
    expected = IsADirectoryError if source_kind == "directory" else FileNotFoundError
    with pytest.raises(expected):
        VideoMediaIO().load_file(str(source))


@pytest.mark.parametrize("decode_error", [False, True])
def test_load_file_hashes_bounded_chunks_and_closes_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decode_error: bool
) -> None:
    """Hash every full/tail block and close the source even when decoding fails."""
    payload = bytes(range(256)) * 8193
    source = tmp_path / "source.mp4"
    source.write_bytes(payload)
    source_file = source.open("rb")
    read_sizes = []
    decoded_hashes = []
    expected = object()

    class ReadSpy:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            source_file.close()

        def fileno(self):
            return source_file.fileno()

        def read(self, size):
            assert size == 1 << 20
            chunk = source_file.read(size)
            read_sizes.append(len(chunk))
            return chunk

    def decode(*args, **kwargs):
        decoded_hashes.append(kwargs["raw_bytes_hash"])
        assert not source_file.closed
        if decode_error:
            raise ValueError("decode failed")
        return expected

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: ReadSpy())
    monkeypatch.setattr(
        media_io_module, "_select_cv2_stream_buffered_backend", lambda: cv2.CAP_FFMPEG
    )
    monkeypatch.setattr(media_io_module, "_load_video_by_cv2", decode)
    if decode_error:
        with pytest.raises(ValueError, match="decode failed"):
            VideoMediaIO().load_file(str(source))
    else:
        assert VideoMediaIO().load_file(str(source)) is expected
    assert source_file.closed
    assert read_sizes == [1 << 20, 1 << 20, 256, 0]
    assert decoded_hashes == [blake3(payload).hexdigest()]

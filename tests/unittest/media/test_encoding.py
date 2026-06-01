# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct unit coverage for :mod:`tensorrt_llm.media.encoding`.

Exercises real PNG round-trips, ``video_to_bytes``, and the path-based
``resolve_video_format`` contract.
"""

from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

from tensorrt_llm.media.encoding import (
    image_to_bytes,
    resolve_video_format,
    save_image,
    save_images,
    save_video,
    save_videos,
    video_to_bytes,
)


def _dummy_image(height: int = 16, width: int = 16) -> torch.Tensor:
    return torch.full((height, width, 3), 200, dtype=torch.uint8)


def _dummy_video(num_frames: int = 4, height: int = 16, width: int = 16) -> torch.Tensor:
    return torch.full((num_frames, height, width, 3), 128, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# save_image / image_to_bytes
# ---------------------------------------------------------------------------


def test_save_image_writes_valid_png(tmp_path):
    """save_image writes a valid PNG that PIL can re-read."""
    target = tmp_path / "out.png"
    saved = save_image(_dummy_image(), target)
    assert Path(saved).exists()
    img = Image.open(saved)
    img.verify()
    assert img.format == "PNG"


def test_save_image_jpeg_via_extension(tmp_path):
    """save_image infers JPEG from the extension."""
    target = tmp_path / "out.jpg"
    saved = save_image(_dummy_image(), target)
    img = Image.open(saved)
    img.verify()
    assert img.format == "JPEG"


def test_save_image_unknown_extension_falls_back_to_png(tmp_path):
    """Unknown extensions fall back to PNG (existing behavior)."""
    target = tmp_path / "out.unknownext"
    saved = save_image(_dummy_image(), target)
    # The fallback rewrites the path to .png.
    assert Path(saved).suffix == ".png"
    img = Image.open(saved)
    img.verify()
    assert img.format == "PNG"


def test_image_to_bytes_returns_nonempty_png():
    """image_to_bytes returns non-empty bytes that decode back to PNG."""
    data = image_to_bytes(_dummy_image(), format="PNG")
    assert isinstance(data, bytes)
    assert len(data) > 50
    img = Image.open(BytesIO(data))
    img.verify()
    assert img.format == "PNG"


def test_save_image_strips_batch_dim(tmp_path):
    """save_image accepts (B, H, W, C) and writes the first slice."""
    batched = torch.stack([_dummy_image(), _dummy_image(), _dummy_image()])
    target = tmp_path / "first.png"
    saved = save_image(batched, target)
    assert Path(saved).exists()
    img = Image.open(saved)
    img.verify()


# ---------------------------------------------------------------------------
# save_video / video_to_bytes
# ---------------------------------------------------------------------------


def test_video_to_bytes_returns_nonempty_avi():
    """video_to_bytes returns non-empty bytes for the AVI fallback.

    AVI uses the pure-Python encoder, so the test runs without requiring
    ffmpeg in the environment.
    """
    data = video_to_bytes(_dummy_video(), frame_rate=8.0, format="avi")
    assert isinstance(data, bytes)
    assert len(data) > 100
    # Minimal RIFF/AVI header sanity check.
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"AVI "


def test_save_video_avi_writes_riff_container(tmp_path):
    """save_video produces a real AVI when ffmpeg is unavailable."""
    target = tmp_path / "out.avi"
    saved = save_video(_dummy_video(), target, frame_rate=8.0, format="avi")
    assert Path(saved).exists()
    with open(saved, "rb") as f:
        head = f.read(12)
    assert head[:4] == b"RIFF"
    assert head[8:12] == b"AVI "


# ---------------------------------------------------------------------------
# resolve_video_format
# ---------------------------------------------------------------------------


def test_resolve_video_format_avi_string():
    fmt, ext = resolve_video_format("avi")
    assert (fmt, ext) == ("avi", ".avi")


def test_resolve_video_format_path_avi():
    """``resolve_video_format(Path("x.avi"))`` returns the same canonical id."""
    fmt, ext = resolve_video_format(Path("clip.avi"))
    assert (fmt, ext) == ("avi", ".avi")


def test_resolve_video_format_path_mp4_with_ffmpeg():
    """Path('x.mp4') resolves to mp4 when ffmpeg is available."""
    with mock.patch("tensorrt_llm.media.encoding._check_ffmpeg_available", return_value=True):
        fmt, ext = resolve_video_format(Path("clip.mp4"))
    assert (fmt, ext) == ("mp4", ".mp4")


def test_resolve_video_format_auto_falls_back_to_avi_without_ffmpeg():
    with mock.patch("tensorrt_llm.media.encoding._check_ffmpeg_available", return_value=False):
        fmt, ext = resolve_video_format("auto")
    assert (fmt, ext) == ("avi", ".avi")


def test_resolve_video_format_mp4_string_without_ffmpeg_raises():
    with mock.patch("tensorrt_llm.media.encoding._check_ffmpeg_available", return_value=False):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            resolve_video_format("mp4")


def test_resolve_video_format_unknown_string_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_video_format("mkv")


def test_resolve_video_format_path_without_suffix_raises():
    with pytest.raises(ValueError, match="without suffix"):
        resolve_video_format(Path("noext"))


# ---------------------------------------------------------------------------
# save_images / save_videos (batch)
# ---------------------------------------------------------------------------


def test_save_images_single_via_batch(tmp_path):
    """A single (H, W, C) tensor saved via save_images produces one file."""
    prefix = str(tmp_path / "single")
    paths = save_images(_dummy_image(), prefix)
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    assert paths[0].endswith("single_0.png")


def test_save_images_batch_prefix(tmp_path):
    """A (B, H, W, C) tensor produces B numbered files under the prefix."""
    batched = torch.stack([_dummy_image() for _ in range(3)], dim=0)
    prefix = str(tmp_path / "batch")
    paths = save_images(batched, prefix)
    assert len(paths) == 3
    for i, p in enumerate(paths):
        assert Path(p).exists()
        assert p.endswith(f"batch_{i}.png")


def test_save_images_custom_format_extension(tmp_path):
    """Explicit format is reflected in the extension."""
    batched = torch.stack([_dummy_image() for _ in range(2)], dim=0)
    prefix = str(tmp_path / "fmt")
    paths = save_images(batched, prefix, format="JPEG")
    assert len(paths) == 2
    for p in paths:
        assert p.endswith(".jpg")
        assert Path(p).exists()


def test_save_images_explicit_path_list(tmp_path):
    """Explicit per-image paths are used as-is."""
    batched = torch.stack([_dummy_image() for _ in range(2)], dim=0)
    paths_in = [str(tmp_path / "alpha.png"), str(tmp_path / "beta.png")]
    paths_out = save_images(batched, paths_in)
    assert paths_out == paths_in
    for p in paths_out:
        assert Path(p).exists()


def test_save_images_path_list_no_extension_appends(tmp_path):
    """Paths without an extension get the format-derived extension."""
    batched = torch.stack([_dummy_image() for _ in range(2)], dim=0)
    paths_in = [str(tmp_path / "img_a"), str(tmp_path / "img_b")]
    paths_out = save_images(batched, paths_in, format="JPEG")
    for p in paths_out:
        assert p.endswith(".jpg")
        assert Path(p).exists()


def test_save_images_path_list_length_mismatch_raises(tmp_path):
    """Mismatched list length raises ValueError."""
    batched = torch.stack([_dummy_image() for _ in range(3)], dim=0)
    with pytest.raises(ValueError, match="does not match batch size"):
        save_images(batched, [str(tmp_path / "only_one.png")])


def test_save_videos_single_via_batch(tmp_path):
    """A single (T, H, W, C) tensor saved via save_videos produces one file."""
    prefix = str(tmp_path / "single_vid")
    paths = save_videos(_dummy_video(), prefix, format="avi")
    assert len(paths) == 1
    assert Path(paths[0]).exists()


def test_save_videos_batch_prefix(tmp_path):
    """A (B, T, H, W, C) tensor produces B files under the prefix."""
    batched = torch.stack([_dummy_video() for _ in range(2)], dim=0)
    prefix = str(tmp_path / "batch_vid")
    paths = save_videos(batched, prefix, format="avi")
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()


def test_save_videos_explicit_path_list(tmp_path):
    """Explicit per-video paths are used as-is."""
    batched = torch.stack([_dummy_video() for _ in range(2)], dim=0)
    paths_in = [str(tmp_path / "clip_a.avi"), str(tmp_path / "clip_b.avi")]
    paths_out = save_videos(batched, paths_in, format="avi")
    assert len(paths_out) == 2
    for p in paths_out:
        assert Path(p).exists()


def test_save_videos_path_list_no_extension_appends(tmp_path):
    """Paths without an extension get the format-derived extension."""
    batched = torch.stack([_dummy_video() for _ in range(1)], dim=0)
    paths_in = [str(tmp_path / "vid_no_ext")]
    paths_out = save_videos(batched, paths_in, format="avi")
    assert paths_out[0].endswith(".avi")
    assert Path(paths_out[0]).exists()

#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for media_storage module."""

import os
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm.serve.media_storage import MediaStorage, PurePythonEncoder, get_video_encoder


def _make_dummy_video_tensor(
    num_frames: int = 4, height: int = 64, width: int = 64
) -> torch.Tensor:
    """Create a small dummy uint8 video tensor (T, H, W, C)."""
    return torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)


class TestMediaStoragePNGFallback:
    """Test PNG fallback path handling when video encoding fails."""

    def test_png_fallback_with_avi_extension(self, tmp_path):
        """Test that PNG fallback uses correct path after .avi extension change."""
        video = _make_dummy_video_tensor()
        output_path = str(tmp_path / "test.avi")

        # Mock encoder to return None (no encoder available)
        with patch("tensorrt_llm.serve.media_storage.get_video_encoder", return_value=None):
            result = MediaStorage._save_mp4(video, None, output_path, 24.0)

        # Should create PNG file with correct name
        assert result.endswith(".png")
        assert os.path.exists(result)
        expected_path = str(tmp_path / "test.png")
        assert result == expected_path

    def test_png_fallback_with_mp4_extension(self, tmp_path):
        """Test that PNG fallback uses correct path after .mp4 extension."""
        video = _make_dummy_video_tensor()
        output_path = str(tmp_path / "test.mp4")

        # Mock encoder to return None (no encoder available)
        with patch("tensorrt_llm.serve.media_storage.get_video_encoder", return_value=None):
            result = MediaStorage._save_mp4(video, None, output_path, 24.0)

        # Should create PNG file with correct name
        assert result.endswith(".png")
        assert os.path.exists(result)
        expected_path = str(tmp_path / "test.png")
        assert result == expected_path


class TestMediaStorageMP4Encoding:
    """Test MP4 encoding with and without ffmpeg."""

    def test_mp4_requires_ffmpeg_error(self, tmp_path):
        """Test that requesting MP4 without ffmpeg raises an error."""
        video = _make_dummy_video_tensor()
        output_path = str(tmp_path / "test.mp4")

        # Mock to simulate pure Python encoder (no ffmpeg)
        with patch(
            "tensorrt_llm.serve.media_storage.get_video_encoder",
            return_value=PurePythonEncoder(),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                MediaStorage._save_mp4(video, None, output_path, 24.0)

            assert "MP4 format requires ffmpeg" in str(exc_info.value)
            assert "apt-get install ffmpeg" in str(exc_info.value)


class TestConvertVideoToBytes:
    """Test convert_video_to_bytes handles actual output path correctly."""

    def test_convert_video_to_bytes_uses_actual_path(self):
        """Test that convert_video_to_bytes reads from the actual returned path."""
        video = _make_dummy_video_tensor()

        # Mock save_video to return a different path than requested
        def mock_save_video(video, output_path, audio, frame_rate, format):
            # Simulate changing extension (e.g., .mp4 -> .avi)
            if output_path.endswith(".mp4"):
                actual_path = output_path[:-4] + ".avi"
            else:
                actual_path = output_path

            # Write dummy content
            with open(actual_path, "wb") as f:
                f.write(b"test video content")
            return actual_path

        with patch.object(MediaStorage, "save_video", side_effect=mock_save_video):
            result = MediaStorage.convert_video_to_bytes(video, None, 24.0, "mp4")

        assert result == b"test video content"


class TestVideoEncoderCaching:
    """Test that video encoder is cached as singleton."""

    def test_encoder_is_cached(self):
        """Test that get_video_encoder returns the same instance on multiple calls."""
        # Clear cache
        import tensorrt_llm.serve.media_storage as ms

        ms._VIDEO_ENCODER = None

        encoder1 = get_video_encoder()
        encoder2 = get_video_encoder()

        # Should be the exact same instance
        assert encoder1 is encoder2

        # Clean up
        ms._VIDEO_ENCODER = None

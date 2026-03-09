#!/usr/bin/env python
# Copyright 2026 NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for media_storage module."""

import os
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm.serve.media_storage import (
    MediaStorage,
    PurePythonEncoder,
    get_video_encoder,
    resolve_video_format,
)


def _make_dummy_video_tensor(
    num_frames: int = 4, height: int = 64, width: int = 64
) -> torch.Tensor:
    """Create a small dummy uint8 video tensor (T, H, W, C)."""
    return torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)


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
                MediaStorage._save_encoded_video(video, None, output_path, 24.0)

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


class TestResolveVideoFormat:
    """Test resolve_video_format() for explicit format selection."""

    def test_resolve_mp4_with_ffmpeg(self):
        """When mp4 is requested and ffmpeg is available, resolve to mp4."""
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=True,
        ):
            fmt, ext = resolve_video_format("mp4")
        assert fmt == "mp4"
        assert ext == ".mp4"

    def test_resolve_mp4_without_ffmpeg_raises(self):
        """When mp4 is requested but ffmpeg is unavailable, raise RuntimeError."""
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                resolve_video_format("mp4")
            assert "ffmpeg" in str(exc_info.value).lower()
            assert "output_format='avi'" in str(exc_info.value)

    def test_resolve_avi_always_succeeds(self):
        """AVI format should always resolve regardless of ffmpeg availability."""
        # Even without ffmpeg
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=False,
        ):
            fmt, ext = resolve_video_format("avi")
        assert fmt == "avi"
        assert ext == ".avi"

    def test_resolve_avi_with_ffmpeg(self):
        """AVI format should resolve to avi even when ffmpeg is available."""
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=True,
        ):
            fmt, ext = resolve_video_format("avi")
        assert fmt == "avi"
        assert ext == ".avi"

    def test_resolve_auto_with_ffmpeg(self):
        """Auto mode with ffmpeg should resolve to mp4."""
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=True,
        ):
            fmt, ext = resolve_video_format("auto")
        assert fmt == "mp4"
        assert ext == ".mp4"

    def test_resolve_auto_without_ffmpeg(self):
        """Auto mode without ffmpeg should resolve to avi."""
        with patch(
            "tensorrt_llm.serve.media_storage._check_ffmpeg_available",
            return_value=False,
        ):
            fmt, ext = resolve_video_format("auto")
        assert fmt == "avi"
        assert ext == ".avi"


class TestExplicitFormatSaveVideo:
    """Test MediaStorage.save_video with explicit output_format."""

    def test_save_video_avi_format(self, tmp_path):
        """Test saving video with explicit avi format uses PurePythonEncoder."""
        video = _make_dummy_video_tensor()
        output_path = str(tmp_path / "test.avi")

        result = MediaStorage.save_video(
            video=video,
            output_path=output_path,
            frame_rate=24.0,
            format="avi",
        )

        assert result.endswith(".avi")
        assert os.path.exists(result)

    def test_save_video_mp4_format_without_ffmpeg_raises(self, tmp_path):
        """Test that explicit mp4 format without ffmpeg raises RuntimeError."""
        video = _make_dummy_video_tensor()
        output_path = str(tmp_path / "test.mp4")

        with patch(
            "tensorrt_llm.serve.media_storage.get_video_encoder",
            return_value=PurePythonEncoder(),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                MediaStorage.save_video(
                    video=video,
                    output_path=output_path,
                    frame_rate=24.0,
                    format="mp4",
                )
            assert "MP4 format requires ffmpeg" in str(exc_info.value)


class TestVideoGenerationRequestOutputFormat:
    """Test VideoGenerationRequest output_format field."""

    def test_default_output_format_is_auto(self):
        """Default output_format should be 'auto'."""
        from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest

        req = VideoGenerationRequest(prompt="test")
        assert req.output_format == "auto"

    def test_valid_output_formats(self):
        """All valid output_format values should be accepted."""
        from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest

        for fmt in ["mp4", "avi", "auto"]:
            req = VideoGenerationRequest(prompt="test", output_format=fmt)
            assert req.output_format == fmt

    def test_invalid_output_format_raises(self):
        """Invalid output_format should raise validation error."""
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest

        with pytest.raises(ValidationError):
            VideoGenerationRequest(prompt="test", output_format="webm")

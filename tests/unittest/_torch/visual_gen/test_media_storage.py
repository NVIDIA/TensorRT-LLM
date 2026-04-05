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


class TestSaveImagesBatch:
    """Test MediaStorage.save_images() batch saving."""

    def test_save_single_image_via_batch(self, tmp_path):
        """A single (H, W, C) tensor should be saved as one file."""
        image = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        prefix = str(tmp_path / "single")
        paths = MediaStorage.save_images(image, prefix)
        assert len(paths) == 1
        assert os.path.exists(paths[0])
        assert "single_0.png" in paths[0]

    def test_save_batch_of_images(self, tmp_path):
        """A (B, H, W, C) tensor should produce B files."""
        images = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
        prefix = str(tmp_path / "batch")
        paths = MediaStorage.save_images(images, prefix)
        assert len(paths) == 3
        for i, p in enumerate(paths):
            assert os.path.exists(p)
            assert f"batch_{i}.png" in p

    def test_save_images_custom_format(self, tmp_path):
        """Explicit format should be reflected in the extension."""
        images = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)
        prefix = str(tmp_path / "fmt")
        paths = MediaStorage.save_images(images, prefix, format="JPEG")
        assert len(paths) == 2
        for p in paths:
            assert p.endswith(".jpg")
            assert os.path.exists(p)


class TestSaveImagesPathList:
    """Test MediaStorage.save_images() with List[str] output paths."""

    def test_save_images_with_path_list(self, tmp_path):
        """Explicit per-image paths should be used as-is."""
        images = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "alpha.png"), str(tmp_path / "beta.png")]
        paths_out = MediaStorage.save_images(images, paths_in)
        assert paths_out == paths_in
        for p in paths_out:
            assert os.path.exists(p)

    def test_save_images_with_relative_path_list(self, tmp_path, monkeypatch):
        """Relative paths should be accepted without error."""
        monkeypatch.chdir(tmp_path)
        images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
        paths_in = ["rel_image.png"]
        paths_out = MediaStorage.save_images(images, paths_in)
        assert paths_out == paths_in
        assert os.path.exists(paths_out[0])

    def test_save_images_path_list_no_extension(self, tmp_path):
        """Paths without an extension get the format-derived extension."""
        images = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "img_a"), str(tmp_path / "img_b")]
        paths_out = MediaStorage.save_images(images, paths_in, format="JPEG")
        for p in paths_out:
            assert p.endswith(".jpg")
            assert os.path.exists(p)

    def test_save_images_path_list_preserves_extension(self, tmp_path):
        """Paths that already have an extension keep it unchanged."""
        images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "photo.webp")]
        paths_out = MediaStorage.save_images(images, paths_in, format="PNG")
        # The explicit extension in the path is preserved
        assert paths_out[0].endswith(".webp")
        assert os.path.exists(paths_out[0])

    def test_save_images_path_list_length_mismatch(self, tmp_path):
        """Mismatched list length should raise ValueError."""
        images = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "only_one.png")]
        with pytest.raises(ValueError, match="does not match batch size"):
            MediaStorage.save_images(images, paths_in)


class TestSaveVideosBatch:
    """Test MediaStorage.save_videos() batch saving."""

    def test_save_single_video_via_batch(self, tmp_path):
        """A single (T, H, W, C) tensor should produce one file."""
        video = _make_dummy_video_tensor()
        prefix = str(tmp_path / "single_vid")
        paths = MediaStorage.save_videos(video, prefix, format="avi")
        assert len(paths) == 1
        assert os.path.exists(paths[0])

    def test_save_batch_of_videos(self, tmp_path):
        """A (B, T, H, W, C) tensor should produce B files."""
        videos = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
        prefix = str(tmp_path / "batch_vid")
        paths = MediaStorage.save_videos(videos, prefix, format="avi")
        assert len(paths) == 2
        for p in paths:
            assert os.path.exists(p)

    def test_save_videos_mp4_without_ffmpeg_raises(self, tmp_path):
        """Requesting mp4 without ffmpeg should raise for each video."""
        videos = torch.randint(0, 256, (1, 4, 64, 64, 3), dtype=torch.uint8)
        prefix = str(tmp_path / "mp4_vid")
        with patch(
            "tensorrt_llm.serve.media_storage.get_video_encoder",
            return_value=PurePythonEncoder(),
        ):
            with pytest.raises(RuntimeError):
                MediaStorage.save_videos(videos, prefix, format="mp4")


class TestSaveVideosPathList:
    """Test MediaStorage.save_videos() with List[str] output paths."""

    def test_save_videos_with_path_list(self, tmp_path):
        """Explicit per-video paths should be used as-is."""
        videos = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "clip_a.avi"), str(tmp_path / "clip_b.avi")]
        paths_out = MediaStorage.save_videos(videos, paths_in, format="avi")
        assert len(paths_out) == 2
        for p in paths_out:
            assert os.path.exists(p)

    def test_save_videos_path_list_no_extension(self, tmp_path):
        """Paths without an extension get the format-derived extension."""
        videos = torch.randint(0, 256, (1, 4, 32, 32, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "vid_no_ext")]
        paths_out = MediaStorage.save_videos(videos, paths_in, format="avi")
        assert paths_out[0].endswith(".avi")
        assert os.path.exists(paths_out[0])

    def test_save_videos_path_list_length_mismatch(self, tmp_path):
        """Mismatched list length should raise ValueError."""
        videos = torch.randint(0, 256, (3, 4, 32, 32, 3), dtype=torch.uint8)
        paths_in = [str(tmp_path / "one.avi")]
        with pytest.raises(ValueError, match="does not match batch size"):
            MediaStorage.save_videos(videos, paths_in, format="avi")


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

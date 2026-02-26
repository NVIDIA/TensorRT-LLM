#!/usr/bin/env python
"""Media Storage for generated images and videos.

This module provides storage handlers for persisting generated media assets
(videos, images) and their associated metadata.
"""

import os
import shutil
import struct
import subprocess  # nosec B404
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import torch
from PIL import Image

from tensorrt_llm.logger import logger

# Video encoder availability flags (cached after first check)
_FFMPEG_PATH: Optional[str] = None
_VIDEO_ENCODER: Optional["VideoEncoder"] = None


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg CLI is available and cache its path."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None:
            try:
                result = subprocess.run(
                    [ffmpeg_path, "-version"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    _FFMPEG_PATH = ffmpeg_path
                else:
                    _FFMPEG_PATH = ""
            except (FileNotFoundError, OSError):
                _FFMPEG_PATH = ""
        else:
            _FFMPEG_PATH = ""
    return bool(_FFMPEG_PATH)


def _get_ffmpeg_path() -> str:
    """Get the cached ffmpeg path. Must call _check_ffmpeg_available() first."""
    if _FFMPEG_PATH is None:
        _check_ffmpeg_available()
    return _FFMPEG_PATH or ""


class VideoEncoder(ABC):
    """Abstract base class for video encoders."""

    @abstractmethod
    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
    ) -> str:
        """Encode video frames to file.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            output_path: Path to save the video
            frame_rate: Frames per second
            audio: Optional audio as torch.Tensor

        Returns:
            Path where the video was saved
        """
        pass

    @staticmethod
    def _validate_video_tensor(video: torch.Tensor) -> None:
        """Validate video tensor format."""
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

    @staticmethod
    def _validate_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
        """Validate and normalize audio tensor format.

        Args:
            audio: Audio tensor in various formats

        Returns:
            Normalized audio tensor in (samples, channels) format as int16
        """
        if not isinstance(audio, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for audio, got {type(audio)}")

        audio_tensor = audio

        # Handle different audio tensor dimensions
        if audio_tensor.ndim == 1:
            # Mono audio: (samples,) -> (samples, 1)
            audio_tensor = audio_tensor[:, None]
        elif audio_tensor.ndim == 2:
            # If shape[1] != 2 and shape[0] == 2, transpose to (samples, channels)
            if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
                audio_tensor = audio_tensor.T
            if audio_tensor.shape[1] > 2:
                audio_tensor = audio_tensor[:, :2]
        elif audio_tensor.ndim == 3:
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            else:
                audio_tensor = audio_tensor[0]
            if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
                audio_tensor = audio_tensor.T
            if audio_tensor.shape[1] > 2:
                audio_tensor = audio_tensor[:, :2]
        else:
            raise ValueError(
                f"Unsupported audio tensor shape: {audio_tensor.shape}. "
                f"Expected 1D, 2D, or 3D tensor."
            )

        if audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2]

        # Convert to int16 if needed
        if audio_tensor.dtype != torch.int16:
            audio_tensor = torch.clip(audio_tensor, -1.0, 1.0)
            audio_tensor = (audio_tensor * 32767.0).to(torch.int16)

        return audio_tensor


class FfmpegCliEncoder(VideoEncoder):
    """Video encoder using ffmpeg CLI for both encoding and muxing.

    This encoder pipes raw RGB frames to ffmpeg for H.264 encoding
    and MP4 container muxing. Does not require any Python video libraries,
    only the ffmpeg CLI tool.
    """

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
    ) -> str:
        """Encode video using ffmpeg CLI.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8 RGB
            output_path: Path to save the video
            frame_rate: Frames per second
            audio: Optional audio as torch.Tensor

        Returns:
            Path where the video was saved
        """
        self._validate_video_tensor(video)

        # Convert video tensor to numpy: (T, H, W, C) uint8
        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape

        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Handle audio if provided
            audio_input_args = []
            audio_output_args = []
            if audio is not None:
                audio_tensor = self._validate_audio_tensor(audio)
                audio_np = audio_tensor.cpu().numpy()
                tmp_audio_path = os.path.join(tmp_dir, "audio.wav")
                self._write_wav(tmp_audio_path, audio_np, sample_rate=24000)
                audio_input_args = ["-i", tmp_audio_path]
                audio_output_args = ["-c:a", "aac", "-shortest"]

            # Build ffmpeg command
            # Input: raw RGB frames piped via stdin
            # Output: H.264 encoded MP4
            ffmpeg_path = _get_ffmpeg_path()
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output
                "-f",
                "rawvideo",  # Input format
                "-pix_fmt",
                "rgb24",  # Pixel format
                "-s",
                f"{width}x{height}",  # Frame size
                "-r",
                str(frame_rate),  # Frame rate
                "-i",
                "-",  # Read from stdin
                *audio_input_args,  # Audio input (if any)
                "-c:v",
                "libx264",  # Video codec
                "-pix_fmt",
                "yuv420p",  # Output pixel format
                "-preset",
                "medium",  # Encoding preset
                "-crf",
                "23",  # Quality (lower = better)
                *audio_output_args,  # Audio output args (if any)
                output_path,
            ]

            try:
                # Run ffmpeg with raw frames piped to stdin
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Write all frames to ffmpeg stdin
                raw_frames = video_np.tobytes()
                stdout, stderr = process.communicate(input=raw_frames)

                if process.returncode != 0:
                    raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")

            except FileNotFoundError:
                raise RuntimeError("ffmpeg not found. Install ffmpeg for video encoding.")

        logger.info(f"Saved video{' with audio' if audio is not None else ''} to {output_path}")
        return output_path

    def _write_wav(self, path: str, audio: Any, sample_rate: int = 24000) -> None:
        """Write audio to WAV file.

        Args:
            path: Output path
            audio: Audio data as numpy array (samples, channels) int16
            sample_rate: Audio sample rate
        """
        import wave

        import numpy as np

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio, audio])

        # Interleave channels for WAV format
        audio_interleaved = audio.flatten().astype(np.int16)

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_interleaved.tobytes())


class PurePythonEncoder(VideoEncoder):
    """Pure Python video encoder using MJPEG in AVI container.

    This encoder creates Motion JPEG (MJPEG) videos in AVI format using only
    Python standard library and PIL (for JPEG encoding). No ffmpeg or other
    external video libraries required.

    Note: Audio is not supported with this encoder.
    """

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
    ) -> str:
        """Encode video as MJPEG in AVI container.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8 RGB
            output_path: Path to save the video
            frame_rate: Frames per second
            audio: Optional audio (NOT SUPPORTED - will be ignored with warning)

        Returns:
            Path where the video was saved
        """
        if audio is not None:
            logger.warning(
                "PurePythonEncoder does not support audio. "
                "Audio will be ignored. Use FfmpegCliEncoder for audio support."
            )

        self._validate_video_tensor(video)

        # Convert video tensor to numpy: (T, H, W, C) uint8
        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape

        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        # Convert frames to JPEG
        jpeg_frames: List[bytes] = []
        for frame in video_np:
            pil_image = Image.fromarray(frame)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            jpeg_frames.append(buffer.getvalue())

        # Write AVI file
        self._write_mjpeg_avi(output_path, jpeg_frames, width, height, frame_rate)

        logger.info(f"Saved video to {output_path}")
        return output_path

    def _write_mjpeg_avi(
        self,
        output_path: str,
        jpeg_frames: List[bytes],
        width: int,
        height: int,
        frame_rate: float,
    ) -> None:
        """Write MJPEG frames to AVI container.

        Args:
            output_path: Output file path
            jpeg_frames: List of JPEG-encoded frames
            width: Frame width
            height: Frame height
            frame_rate: Frames per second
        """
        num_frames = len(jpeg_frames)
        usec_per_frame = int(1000000 / frame_rate)

        # Build movi chunk with all frames
        movi_data = b""
        frame_index: List[tuple] = []  # (offset, size) for each frame

        for jpeg_data in jpeg_frames:
            original_size = len(jpeg_data)
            # Record offset within movi (after 'movi' fourcc)
            frame_index.append((len(movi_data) + 4, original_size))

            # 00dc = compressed video chunk
            chunk = b"00dc" + struct.pack("<I", original_size) + jpeg_data
            # Pad to word boundary
            if original_size % 2:
                chunk += b"\x00"
            movi_data += chunk

        # Build index (idx1)
        idx1_data = b""
        for offset, size in frame_index:
            idx1_data += b"00dc"  # chunk id
            idx1_data += struct.pack("<I", 0x10)  # flags: AVIIF_KEYFRAME
            idx1_data += struct.pack("<I", offset)  # offset from movi
            idx1_data += struct.pack("<I", size)  # chunk size

        # Calculate sizes
        movi_list_size = 4 + len(movi_data)  # 'movi' + data
        idx1_size = len(idx1_data)

        # Build AVI headers
        # Main AVI header (avih)
        avih = struct.pack(
            "<IIIIIIIIIIIIII",
            usec_per_frame,  # dwMicroSecPerFrame
            0,  # dwMaxBytesPerSec
            0,  # dwPaddingGranularity
            0x10,  # dwFlags (AVIF_HASINDEX)
            num_frames,  # dwTotalFrames
            0,  # dwInitialFrames
            1,  # dwStreams
            max(len(f) for f in jpeg_frames),  # dwSuggestedBufferSize
            width,  # dwWidth
            height,  # dwHeight
            0,
            0,
            0,
            0,  # dwReserved[4]
        )

        # Stream header (strh) for video
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIHHHH",
            b"vids",  # fccType
            b"MJPG",  # fccHandler
            0,  # dwFlags
            0,  # wPriority
            0,  # wLanguage
            0,  # dwInitialFrames
            1,  # dwScale
            int(frame_rate),  # dwRate
            0,  # dwStart
            num_frames,  # dwLength
            max(len(f) for f in jpeg_frames),  # dwSuggestedBufferSize
            0,  # dwQuality
            0,  # dwSampleSize
            0,
            0,  # rcFrame left, top
            width,
            height,  # rcFrame right, bottom
        )

        # Stream format (strf) for video - BITMAPINFOHEADER
        strf = struct.pack(
            "<IiiHHIIiiII",
            40,  # biSize
            width,  # biWidth
            height,  # biHeight (positive = bottom-up, but MJPEG handles it)
            1,  # biPlanes
            24,  # biBitCount
            0x47504A4D,  # biCompression = 'MJPG' (little-endian)
            width * height * 3,  # biSizeImage
            0,  # biXPelsPerMeter
            0,  # biYPelsPerMeter
            0,  # biClrUsed
            0,  # biClrImportant
        )

        # Build strl LIST (stream list)
        strh_chunk = b"strh" + struct.pack("<I", len(strh)) + strh
        strf_chunk = b"strf" + struct.pack("<I", len(strf)) + strf
        strl_data = strh_chunk + strf_chunk
        strl_list = b"LIST" + struct.pack("<I", 4 + len(strl_data)) + b"strl" + strl_data

        # Build hdrl LIST (header list)
        avih_chunk = b"avih" + struct.pack("<I", len(avih)) + avih
        hdrl_data = avih_chunk + strl_list
        hdrl_list = b"LIST" + struct.pack("<I", 4 + len(hdrl_data)) + b"hdrl" + hdrl_data

        # Build movi LIST
        movi_list = b"LIST" + struct.pack("<I", movi_list_size) + b"movi" + movi_data

        # Build idx1 chunk
        idx1_chunk = b"idx1" + struct.pack("<I", idx1_size) + idx1_data

        # Calculate total RIFF size
        riff_data = hdrl_list + movi_list + idx1_chunk
        riff_size = 4 + len(riff_data)  # 'AVI ' + data

        # Write the file
        with open(output_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", riff_size))
            f.write(b"AVI ")
            f.write(riff_data)


def get_video_encoder() -> Optional["VideoEncoder"]:
    """Get the best available video encoder (cached singleton).

    Checks availability in order:
    1. ffmpeg CLI - Full-featured solution (supports audio)
    2. Pure Python MJPEG/AVI - No external dependencies (video only, no audio)

    Returns:
        VideoEncoder instance, or None if no encoder is available
    """
    global _VIDEO_ENCODER
    if _VIDEO_ENCODER is None:
        if _check_ffmpeg_available():
            logger.info("Using ffmpeg CLI for video encoding")
            _VIDEO_ENCODER = FfmpegCliEncoder()
        else:
            logger.info("Using pure Python MJPEG/AVI encoder (no audio support)")
            _VIDEO_ENCODER = PurePythonEncoder()
    return _VIDEO_ENCODER


class MediaStorage:
    """Handler for storing images and videos in various formats."""

    @staticmethod
    def save_image(
        image: Any, output_path: str, format: Optional[str] = None, quality: int = 95
    ) -> str:
        """Save image to file.

        Args:
            image: torch.Tensor (H, W, C) uint8
            output_path: Path to save the image
            format: Image format (png, jpg, webp). If None, infer from extension
            quality: Quality for lossy formats (1-100, higher is better)

        Returns:
            Path where the image was saved
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Convert to PIL Image if needed
        pil_image = MediaStorage._to_pil_image(image)

        # Determine format
        if format is None:
            ext = os.path.splitext(output_path)[1].lower()
            if ext in [".png"]:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            elif ext in [".webp"]:
                format = "WEBP"
            else:
                logger.warning(f"Unknown image extension {ext}, defaulting to PNG")
                format = "PNG"
                output_path = output_path.rsplit(".", 1)[0] + ".png"

        # Save image with format-specific handling
        MediaStorage._save_pil_image(pil_image, output_path, format, quality)

        logger.info(f"Saved image to {output_path} (format={format})")
        return output_path

    @staticmethod
    def convert_image_to_bytes(image: Any, format: str = "PNG", quality: int = 95) -> bytes:
        """Convert image to bytes buffer.

        Args:
            image: torch.Tensor (H, W, C) uint8
            format: Image format (PNG, JPEG, WEBP)
            quality: Quality for lossy formats (1-100)

        Returns:
            Image bytes
        """
        pil_image = MediaStorage._to_pil_image(image)

        # Save to bytes buffer
        buffer = BytesIO()
        MediaStorage._save_pil_image(pil_image, buffer, format, quality)

        return buffer.getvalue()

    @staticmethod
    def _to_pil_image(image: torch.Tensor) -> Image.Image:
        """Convert torch.Tensor to PIL Image.

        Args:
            image: torch.Tensor (H, W, C) uint8

        Returns:
            PIL Image
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        # Convert to numpy for PIL
        image_np = image.cpu().numpy()
        return Image.fromarray(image_np)

    @staticmethod
    def _save_pil_image(
        pil_image: Image.Image,
        output: Any,  # Can be path string or BytesIO
        format: str,
        quality: int,
    ):
        """Save PIL Image to file or buffer.

        Args:
            pil_image: PIL Image to save
            output: Output path (str) or BytesIO buffer
            format: Image format (PNG, JPEG, WEBP)
            quality: Quality for lossy formats (1-100)
        """
        format_upper = format.upper()

        if format_upper in ["JPEG", "JPG"]:
            # Convert RGBA to RGB for JPEG
            if pil_image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", pil_image.size, (255, 255, 255))
                if pil_image.mode == "P":
                    pil_image = pil_image.convert("RGBA")
                background.paste(
                    pil_image, mask=pil_image.split()[-1] if pil_image.mode == "RGBA" else None
                )
                pil_image = background
            pil_image.save(output, format="JPEG", quality=quality, optimize=True)
        elif format_upper == "WEBP":
            pil_image.save(output, format="WEBP", quality=quality)
        else:  # PNG or default
            pil_image.save(output, format="PNG", optimize=True)

    @staticmethod
    def save_video(
        video: Any,
        output_path: str,
        audio: Optional[Any] = None,
        frame_rate: float = 24.0,
        format: Optional[str] = None,
    ) -> str:
        """Save video to file with optional audio.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            output_path: Path to save the video
            audio: Optional audio as torch.Tensor
            frame_rate: Frames per second (default: 24.0)
            format: Video format (mp4, gif, png). If None, infer from extension

        Returns:
            Path where the video was saved
        """
        # Ensure output directory exists
        if isinstance(output_path, Path):
            output_path = str(output_path)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Determine format
        if format is None:
            ext = os.path.splitext(output_path)[1].lower()
            format = ext[1:] if ext else "mp4"

        format = format.lower()

        # Save based on format
        if format == "mp4":
            # _save_mp4 may return a different path (e.g., .avi when using PurePythonEncoder)
            output_path = MediaStorage._save_mp4(video, audio, output_path, frame_rate)
        elif format == "gif":
            output_path = MediaStorage._save_gif(video, output_path, frame_rate)
        elif format == "png":
            output_path = MediaStorage._save_middle_frame(video, output_path)
        else:
            logger.warning(f"Unsupported video format: {format}, defaulting to mp4")
            output_path = output_path.rsplit(".", 1)[0] + ".mp4"
            output_path = MediaStorage._save_mp4(video, audio, output_path, frame_rate)

        return output_path

    @staticmethod
    def convert_video_to_bytes(
        video: Any, audio: Optional[Any] = None, frame_rate: float = 24.0, format: str = "mp4"
    ) -> bytes:
        """Convert video to bytes buffer.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            audio: Optional audio as torch.Tensor
            frame_rate: Frames per second
            format: Video format (mp4, gif)

        Returns:
            Video bytes
        """
        import tempfile

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save to temporary file (may return different path than requested)
            actual_path = MediaStorage.save_video(video, tmp_path, audio, frame_rate, format)

            # Read bytes from the actual saved path
            with open(actual_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes
        finally:
            # Clean up temporary file(s) - check both original and actual path
            for path in [tmp_path, actual_path if "actual_path" in locals() else None]:
                if path and os.path.exists(path):
                    os.unlink(path)

    @staticmethod
    def _save_mp4(
        video: torch.Tensor, audio: Optional[torch.Tensor], output_path: str, frame_rate: float
    ) -> str:
        """Save video with optional audio as MP4.

        Uses ffmpeg CLI if available (supports audio), otherwise raises an error
        to prompt ffmpeg installation.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            audio: Optional audio as torch.Tensor
            output_path: Output path for MP4
            frame_rate: Frames per second

        Returns:
            Path where the video was saved

        Raises:
            RuntimeError: If ffmpeg is not available for MP4 encoding
        """
        # Check if ffmpeg is required but not available
        encoder = get_video_encoder()
        if isinstance(encoder, PurePythonEncoder) and output_path.lower().endswith(".mp4"):
            raise RuntimeError(
                "MP4 format requires ffmpeg to be installed. Please install ffmpeg "
                "(e.g., 'apt-get install ffmpeg' on Ubuntu/Debian) or use AVI format instead. "
                "See https://ffmpeg.org/download.html for installation instructions."
            )

        try:
            if encoder is not None:
                return encoder.encode_video(video, output_path, frame_rate, audio)
            else:
                logger.warning(
                    "No video encoder available. Falling back to saving middle frame as PNG."
                )
                png_path = os.path.splitext(output_path)[0] + ".png"
                return MediaStorage._save_middle_frame(video, png_path)
        except Exception as e:
            logger.error(f"Error encoding video: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.warning("Falling back to saving middle frame as PNG.")
            png_path = os.path.splitext(output_path)[0] + ".png"
            return MediaStorage._save_middle_frame(video, png_path)

    @staticmethod
    def _save_gif(video: torch.Tensor, output_path: str, frame_rate: float) -> str:
        """Save video as animated GIF.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            output_path: Output path for GIF
            frame_rate: Frames per second

        Returns:
            Path where the GIF was saved
        """
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

        # Convert to numpy and then to list of PIL Images
        video_np = video.cpu().numpy()
        frames = [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]

        # Save as GIF
        duration_ms = int(1000 / frame_rate)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration_ms,
            loop=0,
        )
        logger.info(f"Saved video as GIF to {output_path} ({len(frames)} frames)")
        return output_path

    @staticmethod
    def _save_middle_frame(video: torch.Tensor, output_path: str) -> str:
        """Save middle frame of video as PNG.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            output_path: Output path for PNG

        Returns:
            Path where the frame was saved
        """
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

        # Extract middle frame
        video_np = video.cpu().numpy()
        frame_idx = video_np.shape[0] // 2
        image = Image.fromarray(video_np[frame_idx])

        image.save(output_path)
        logger.info(f"Saved frame {frame_idx} to {output_path}")
        return output_path

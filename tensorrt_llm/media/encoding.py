# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor → image/video encoding free functions."""

import os
import shutil
import struct
import subprocess  # nosec B404
import tempfile
import wave
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from PIL import Image

from tensorrt_llm.logger import logger

# Video encoder availability flags (cached after first check).
_FFMPEG_PATH: Optional[str] = None
_VIDEO_ENCODER: Optional["_VideoEncoder"] = None


def _check_ffmpeg_available() -> bool:
    """Return True if ffmpeg CLI is installed; cache its path."""
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
                _FFMPEG_PATH = ffmpeg_path if result.returncode == 0 else ""
            except (FileNotFoundError, OSError):
                _FFMPEG_PATH = ""
        else:
            _FFMPEG_PATH = ""
    return bool(_FFMPEG_PATH)


def _get_ffmpeg_path() -> str:
    """Return cached ffmpeg path (after :func:`_check_ffmpeg_available`)."""
    if _FFMPEG_PATH is None:
        _check_ffmpeg_available()
    return _FFMPEG_PATH or ""


# ---------------------------------------------------------------------------
# Encoder backends (private)
# ---------------------------------------------------------------------------


class _VideoEncoder(ABC):
    """Internal abstract base for video encoders."""

    @abstractmethod
    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str: ...

    @staticmethod
    def _validate_video_tensor(video: torch.Tensor) -> None:
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

    @staticmethod
    def _validate_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to ``(samples, channels)`` int16."""
        if not isinstance(audio, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for audio, got {type(audio)}")

        audio_tensor = audio
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor[:, None]
        elif audio_tensor.ndim == 2:
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
                "Expected 1D, 2D, or 3D tensor."
            )

        if audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2]

        if audio_tensor.dtype != torch.int16:
            audio_tensor = torch.clip(audio_tensor, -1.0, 1.0)
            audio_tensor = (audio_tensor * 32767.0).to(torch.int16)
        return audio_tensor


class _FfmpegCliEncoder(_VideoEncoder):
    """Pipe raw RGB frames to ffmpeg for H.264/MP4 encoding (with optional audio)."""

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str:
        self._validate_video_tensor(video)

        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_input_args: list = []
            audio_output_args: list = []
            if audio is not None:
                audio_tensor = self._validate_audio_tensor(audio)
                audio_np = audio_tensor.cpu().numpy()
                tmp_audio_path = os.path.join(tmp_dir, "audio.wav")
                self._write_wav(tmp_audio_path, audio_np, sample_rate=audio_sample_rate)
                audio_input_args = ["-i", tmp_audio_path]
                audio_output_args = ["-c:a", "aac", "-shortest"]

            ffmpeg_path = _get_ffmpeg_path()
            cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(frame_rate),
                "-i",
                "-",
                *audio_input_args,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                *audio_output_args,
                output_path,
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                raw_frames = video_np.tobytes()
                _, stderr = process.communicate(input=raw_frames)
                if process.returncode != 0:
                    raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")
            except FileNotFoundError:
                raise RuntimeError("ffmpeg not found. Install ffmpeg for video encoding.")

        logger.info(f"Saved video{' with audio' if audio is not None else ''} to {output_path}")
        return output_path

    def _write_wav(self, path: str, audio: Any, sample_rate: int) -> None:
        import numpy as np

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio, audio])

        audio_interleaved = audio.flatten().astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_interleaved.tobytes())


class _PurePythonEncoder(_VideoEncoder):
    """Fallback MJPEG/AVI encoder (video only, no audio); requires only PIL."""

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str:
        if audio is not None:
            logger.warning(
                "Pure-Python video encoder does not support audio; audio will be ignored."
            )
        self._validate_video_tensor(video)

        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        jpeg_frames: List[bytes] = []
        for frame in video_np:
            pil_image = Image.fromarray(frame)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            jpeg_frames.append(buffer.getvalue())

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
        num_frames = len(jpeg_frames)
        usec_per_frame = int(1000000 / frame_rate)

        movi_data = b""
        frame_index: List[Tuple[int, int]] = []
        for jpeg_data in jpeg_frames:
            original_size = len(jpeg_data)
            frame_index.append((len(movi_data) + 4, original_size))
            chunk = b"00dc" + struct.pack("<I", original_size) + jpeg_data
            if original_size % 2:
                chunk += b"\x00"
            movi_data += chunk

        idx1_data = b""
        for offset, size in frame_index:
            idx1_data += b"00dc"
            idx1_data += struct.pack("<I", 0x10)
            idx1_data += struct.pack("<I", offset)
            idx1_data += struct.pack("<I", size)

        movi_list_size = 4 + len(movi_data)
        idx1_size = len(idx1_data)

        avih = struct.pack(
            "<IIIIIIIIIIIIII",
            usec_per_frame,
            0,
            0,
            0x10,
            num_frames,
            0,
            1,
            max(len(f) for f in jpeg_frames),
            width,
            height,
            0,
            0,
            0,
            0,
        )
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIHHHH",
            b"vids",
            b"MJPG",
            0,
            0,
            0,
            0,
            1,
            int(frame_rate),
            0,
            num_frames,
            max(len(f) for f in jpeg_frames),
            0,
            0,
            0,
            0,
            width,
            height,
        )
        strf = struct.pack(
            "<IiiHHIIiiII",
            40,
            width,
            height,
            1,
            24,
            0x47504A4D,
            width * height * 3,
            0,
            0,
            0,
            0,
        )

        strh_chunk = b"strh" + struct.pack("<I", len(strh)) + strh
        strf_chunk = b"strf" + struct.pack("<I", len(strf)) + strf
        strl_data = strh_chunk + strf_chunk
        strl_list = b"LIST" + struct.pack("<I", 4 + len(strl_data)) + b"strl" + strl_data

        avih_chunk = b"avih" + struct.pack("<I", len(avih)) + avih
        hdrl_data = avih_chunk + strl_list
        hdrl_list = b"LIST" + struct.pack("<I", 4 + len(hdrl_data)) + b"hdrl" + hdrl_data

        movi_list = b"LIST" + struct.pack("<I", movi_list_size) + b"movi" + movi_data
        idx1_chunk = b"idx1" + struct.pack("<I", idx1_size) + idx1_data
        riff_data = hdrl_list + movi_list + idx1_chunk
        riff_size = 4 + len(riff_data)

        with open(output_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", riff_size))
            f.write(b"AVI ")
            f.write(riff_data)


def _get_video_encoder() -> _VideoEncoder:
    """Return the best available video encoder (cached singleton)."""
    global _VIDEO_ENCODER
    if _VIDEO_ENCODER is None:
        _VIDEO_ENCODER = _FfmpegCliEncoder() if _check_ffmpeg_available() else _PurePythonEncoder()
        logger.info(f"Using {_VIDEO_ENCODER.__class__.__name__} for video encoding")
    return _VIDEO_ENCODER


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def resolve_video_format(output_format) -> Tuple[str, str]:
    """Resolve a requested format to a concrete ``(format, extension)``.

    Args:
        output_format: Either a logical format string (``'mp4'``/``'avi'``/
            ``'auto'``) or a :class:`pathlib.Path` whose suffix indicates the
            desired container.

    Returns:
        Tuple ``(format, extension)`` such as ``('mp4', '.mp4')``.

    Raises:
        RuntimeError: ``'mp4'`` resolved but ffmpeg is not available.
        ValueError: Unsupported value.
    """
    if isinstance(output_format, Path):
        suffix = output_format.suffix.lower().lstrip(".")
        if not suffix:
            raise ValueError(
                f"Cannot resolve video format from path without suffix: {output_format}"
            )
        return resolve_video_format(suffix)

    if output_format == "mp4":
        if _check_ffmpeg_available():
            return "mp4", ".mp4"
        raise RuntimeError(
            "MP4 (H.264) format requires ffmpeg to be installed. "
            "Install ffmpeg (e.g., 'apt-get install ffmpeg' on Ubuntu/Debian) "
            "or use output_format='avi' for MJPEG encoding (no ffmpeg required). "
            "See https://ffmpeg.org/download.html for installation instructions."
        )
    elif output_format == "avi":
        return "avi", ".avi"
    elif output_format == "auto":
        if _check_ffmpeg_available():
            return "mp4", ".mp4"
        return "avi", ".avi"
    else:
        raise ValueError(
            f"Unsupported video format: {output_format}. Please use 'auto' if you "
            "want to automatically choose the best format based on availability."
        )


# ---------------------------------------------------------------------------
# Pixel/PIL helpers
# ---------------------------------------------------------------------------


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    """Convert a uint8 image tensor (``(H, W, C)`` or ``(B, H, W, C)``) to PIL."""
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")
    if image.dim() == 4:
        image = image[0]
    image_np = image.cpu().numpy()
    return Image.fromarray(image_np)


def _save_pil_image(
    pil_image: Image.Image,
    output: Any,
    format: str,
    quality: int,
    png_compress_level: int = 1,
) -> None:
    """Save a PIL image to a path or BytesIO with format-specific defaults."""
    format_upper = format.upper()
    if format_upper in ("JPEG", "JPG"):
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
    else:
        pil_image.save(output, format="PNG", compress_level=png_compress_level)


def _save_encoded_video(
    video: torch.Tensor,
    audio: Optional[torch.Tensor],
    output_path: str,
    frame_rate: float,
    audio_sample_rate: int = 24000,
) -> str:
    """Encode video (and optional audio) to ``output_path`` via the best encoder."""
    encoder = _get_video_encoder()
    if isinstance(encoder, _PurePythonEncoder) and output_path.lower().endswith(".mp4"):
        raise RuntimeError(
            "MP4 format requires ffmpeg to be installed. Please install ffmpeg "
            "(e.g., 'apt-get install ffmpeg' on Ubuntu/Debian) or use AVI format instead. "
            "See https://ffmpeg.org/download.html for installation instructions."
        )
    try:
        return encoder.encode_video(video, output_path, frame_rate, audio, audio_sample_rate)
    except Exception as e:
        logger.error(f"Error encoding video: {e}")
        raise e


# ---------------------------------------------------------------------------
# Public free functions
# ---------------------------------------------------------------------------


def save_image(
    image: torch.Tensor,
    output_path: Any,
    format: Optional[str] = None,
    quality: int = 95,
) -> str:
    """Encode and save an image tensor to disk.

    Args:
        image: Image as ``torch.Tensor`` ``(H, W, C)`` or ``(B, H, W, C)``,
            dtype ``uint8``. If batched, the first image is saved.
        output_path: Output file path (``str`` or :class:`pathlib.Path`).
        format: Image format (``'png'``/``'jpg'``/``'webp'``). If ``None``,
            inferred from the path extension; defaults to PNG when unknown.
        quality: Quality for lossy formats (1-100, higher is better).

    Returns:
        Path string where the image was actually saved.
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)

    if hasattr(image, "dim") and image.dim() == 4:
        image = image[0]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pil_image = _to_pil_image(image)

    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            format = "PNG"
        elif ext in (".jpg", ".jpeg"):
            format = "JPEG"
        elif ext == ".webp":
            format = "WEBP"
        else:
            logger.warning(f"Unknown image extension {ext}, defaulting to PNG")
            format = "PNG"
            output_path = output_path.rsplit(".", 1)[0] + ".png"

    _save_pil_image(pil_image, output_path, format, quality)
    logger.info(f"Saved image to {output_path} (format={format})")
    return output_path


def image_to_bytes(image: torch.Tensor, format: str = "PNG", quality: int = 95) -> bytes:
    """Encode an image tensor to in-memory bytes."""
    pil_image = _to_pil_image(image)
    buffer = BytesIO()
    _save_pil_image(pil_image, buffer, format, quality)
    return buffer.getvalue()


def save_video(
    video: torch.Tensor,
    output_path: Any,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: Optional[str] = None,
    audio_sample_rate: int = 24000,
) -> str:
    """Encode and save a video tensor (with optional audio) to disk.

    Args:
        video: Video as ``torch.Tensor`` ``(T, H, W, C)`` or ``(B, T, H, W, C)``,
            dtype ``uint8``. If batched, the first video is saved.
        output_path: Output file path (``str`` or :class:`pathlib.Path`).
        audio: Optional audio tensor; ignored by the pure-Python AVI fallback.
        frame_rate: Frames per second.
        format: Container (``'mp4'``/``'avi'``). If ``None``, inferred from
            the path extension; defaults to MP4 when unknown.
        audio_sample_rate: Sample rate (Hz) used when muxing audio into the
            output container.

    Returns:
        Path string where the video was actually saved.
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)
    if hasattr(video, "dim") and video.dim() == 5:
        video = video[0]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "mp4"
    format = format.lower()

    if format in ("mp4", "avi"):
        return _save_encoded_video(video, audio, output_path, frame_rate, audio_sample_rate)
    logger.warning(f"Unsupported video format: {format}, defaulting to mp4")
    output_path = output_path.rsplit(".", 1)[0] + ".mp4"
    return _save_encoded_video(video, audio, output_path, frame_rate, audio_sample_rate)


def video_to_bytes(
    video: torch.Tensor,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: str = "mp4",
    audio_sample_rate: int = 24000,
) -> bytes:
    """Encode a video tensor (with optional audio) to in-memory bytes."""
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    actual_path = None
    try:
        actual_path = save_video(video, tmp_path, audio, frame_rate, format, audio_sample_rate)
        with open(actual_path, "rb") as f:
            return f.read()
    finally:
        for path in (tmp_path, actual_path):
            if path and os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Batch save helpers (used by serve endpoints when n > 1)
# ---------------------------------------------------------------------------

_IMAGE_EXT_MAP = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
}


def _image_ext_for_format(format: Optional[str]) -> str:
    """Return the file extension (including leading dot) for an image format."""
    if format is None:
        return ".png"
    return _IMAGE_EXT_MAP.get(format.upper(), ".png")


def _resolve_batch_paths(
    output_paths: Union[str, List[str]],
    batch_size: int,
    ext: str,
) -> List[str]:
    """Build per-item output paths for a batch save operation.

    When *output_paths* is a string it is used as a prefix and each item is
    written to ``{prefix}_{i}{ext}``. When it is a list, each entry is used
    as-is; missing extensions are filled in from *ext*. The list length
    must equal *batch_size*.
    """
    if isinstance(output_paths, list):
        if len(output_paths) != batch_size:
            raise ValueError(
                f"Length of output_paths ({len(output_paths)}) does not "
                f"match batch size ({batch_size})"
            )
        resolved = []
        for p in output_paths:
            if not os.path.splitext(p)[1]:
                p = p + ext
            resolved.append(p)
        return resolved
    return [f"{output_paths}_{i}{ext}" for i in range(batch_size)]


def save_images(
    images: torch.Tensor,
    output_paths: Union[str, List[str]],
    format: Optional[str] = None,
    quality: int = 95,
) -> List[str]:
    """Save a batch of images to individual files.

    Args:
        images: ``torch.Tensor`` of shape ``(B, H, W, C)`` or ``(H, W, C)``,
            dtype ``uint8``.
        output_paths: Either a path prefix string (each image is saved as
            ``{prefix}_{i}.{ext}``) or an explicit list of per-image paths.
        format: Image format (``'png'``/``'jpg'``/``'webp'``). Defaults to PNG.
        quality: Quality for lossy formats (1-100).

    Returns:
        List of paths where the images were saved.
    """
    ext = _image_ext_for_format(format)

    if hasattr(images, "dim") and images.dim() == 3:
        images = images.unsqueeze(0)

    batch_size = images.shape[0]
    resolved = _resolve_batch_paths(output_paths, batch_size, ext)

    paths: List[str] = []
    for i in range(batch_size):
        save_image(images[i], resolved[i], format, quality)
        paths.append(resolved[i])
    return paths


def save_videos(
    videos: torch.Tensor,
    output_paths: Union[str, List[str]],
    audios: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: Optional[str] = None,
    audio_sample_rate: int = 24000,
) -> List[str]:
    """Save a batch of videos to individual files.

    Args:
        videos: ``torch.Tensor`` of shape ``(B, T, H, W, C)`` or
            ``(T, H, W, C)``, dtype ``uint8``.
        output_paths: Either a path prefix string (each video is saved as
            ``{prefix}_{i}.{ext}``) or an explicit list of per-video paths.
        audios: Optional audio tensor. When batched with a matching leading
            dimension, each slice is paired with the corresponding video;
            an unbatched audio tensor is attached to the first video only.
        frame_rate: Frames per second.
        format: Container (``'mp4'``/``'avi'``). Defaults to ``mp4``.
        audio_sample_rate: Audio sample rate (Hz).

    Returns:
        List of paths where the videos were actually saved.
    """
    ext = f".{format}" if format else ".mp4"

    if hasattr(videos, "dim") and videos.dim() == 4:
        videos = videos.unsqueeze(0)

    batch_size = videos.shape[0]
    resolved = _resolve_batch_paths(output_paths, batch_size, ext)

    paths: List[str] = []
    for i in range(batch_size):
        audio_i = None
        if audios is not None:
            if hasattr(audios, "dim") and audios.dim() >= 2 and audios.shape[0] == batch_size:
                audio_i = audios[i]
            elif i == 0:
                audio_i = audios
        actual_path = save_video(
            videos[i],
            resolved[i],
            audio_i,
            frame_rate,
            format,
            audio_sample_rate,
        )
        paths.append(actual_path)
    return paths

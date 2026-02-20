#!/usr/bin/env python
"""Media Storage for generated images and videos.

This module provides storage handlers for persisting generated media assets
(videos, images) and their associated metadata.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from tensorrt_llm.logger import logger


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
            MediaStorage._save_mp4(video, audio, output_path, frame_rate)
        elif format == "gif":
            MediaStorage._save_gif(video, output_path, frame_rate)
        elif format == "png":
            MediaStorage._save_middle_frame(video, output_path)
        else:
            logger.warning(f"Unsupported video format: {format}, defaulting to mp4")
            output_path = output_path.rsplit(".", 1)[0] + ".mp4"
            MediaStorage._save_mp4(video, audio, output_path, frame_rate)

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
            # Save to temporary file
            MediaStorage.save_video(video, tmp_path, audio, frame_rate, format)

            # Read bytes
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _save_mp4(
        video: torch.Tensor, audio: Optional[torch.Tensor], output_path: str, frame_rate: float
    ) -> str:
        """Save video with optional audio as MP4.

        Args:
            video: Video frames as torch.Tensor (T, H, W, C) uint8
            audio: Optional audio as torch.Tensor
            output_path: Output path for MP4
            frame_rate: Frames per second

        Returns:
            Path where the video was saved
        """
        try:
            from fractions import Fraction

            import av

            if not isinstance(video, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

            # Convert video tensor to numpy: (T, H, W, C) uint8
            video_np = video.cpu().numpy()
            num_frames, height, width, channels = video_np.shape

            # Ensure RGB format (3 channels)
            if channels != 3:
                raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

            # Open output container
            container = av.open(output_path, mode="w")

            # Add video stream (H.264 codec)
            video_stream = container.add_stream("libx264", rate=int(frame_rate))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"preset": "medium", "crf": "23"}

            # Pre-process audio and add audio stream BEFORE any muxing.
            # All streams must be registered before the first mux() call
            # (which triggers container header writing).
            audio_stream = None
            audio_tensor = None
            audio_sample_rate = 24000  # Default sample rate
            if audio is not None:
                if not isinstance(audio, torch.Tensor):
                    raise ValueError(f"Expected torch.Tensor for audio, got {type(audio)}")

                # Prepare audio tensor: convert to (samples, channels) format
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

                # Add audio stream now (before any muxing)
                audio_stream = container.add_stream("aac", rate=audio_sample_rate)
                audio_stream.codec_context.sample_rate = audio_sample_rate
                audio_stream.codec_context.layout = "stereo"
                audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)

            # --- Encode video frames ---
            for frame_array in video_np:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            # Flush video encoder
            for packet in video_stream.encode():
                container.mux(packet)

            # --- Encode audio (after video is done) ---
            if audio_stream is not None and audio_tensor is not None:
                # Build packed int16 frame: (1, samples*channels)
                audio_np = audio_tensor.contiguous().reshape(1, -1).cpu().numpy()

                frame_in = av.AudioFrame.from_ndarray(audio_np, format="s16", layout="stereo")
                frame_in.sample_rate = audio_sample_rate

                # Use AudioResampler to convert s16→fltp (AAC's native format)
                cc = audio_stream.codec_context
                audio_resampler = av.audio.resampler.AudioResampler(
                    format=cc.format or "fltp",
                    layout=cc.layout or "stereo",
                    rate=cc.sample_rate or audio_sample_rate,
                )

                audio_next_pts = 0
                for rframe in audio_resampler.resample(frame_in):
                    if rframe.pts is None:
                        rframe.pts = audio_next_pts
                    audio_next_pts += rframe.samples
                    rframe.sample_rate = audio_sample_rate
                    container.mux(audio_stream.encode(rframe))

                # Flush audio encoder
                for packet in audio_stream.encode():
                    container.mux(packet)

            # Close container
            container.close()

            logger.info(f"Saved video{' with audio' if audio is not None else ''} to {output_path}")
            return output_path

        except ImportError:
            logger.warning(
                "PyAV (av) library not available. "
                "Falling back to saving middle frame as PNG. "
                "Install with: pip install av"
            )
            png_path = output_path.replace(".mp4", ".png")
            return MediaStorage._save_middle_frame(video, png_path)
        except Exception as e:
            logger.error(f"Error encoding video with PyAV: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.warning("Falling back to saving middle frame as PNG.")
            png_path = output_path.replace(".mp4", ".png")
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

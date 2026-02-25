"""Unified output handler for diffusion model outputs."""

import os
from typing import Optional

import torch
from PIL import Image

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import MediaOutput


def postprocess_hf_video_tensor(video: torch.Tensor, remove_batch_dim: bool = True) -> torch.Tensor:
    """Post-process video tensor from HuggingFace pipeline output to final format.

    HuggingFace pipelines with output_type="pt" return videos in (B, T, C, H, W) format,
    which is different from VAE decoder output format.

    Args:
        video: Video tensor in (B, T, C, H, W) format from HuggingFace pipeline
        remove_batch_dim: Whether to remove batch dimension. Default True for typical
                         single-batch video generation.

    Returns:
        Post-processed video tensor:
        - If remove_batch_dim=True: (T, H, W, C) uint8 tensor
        - If remove_batch_dim=False: (B, T, H, W, C) uint8 tensor

    Note:
        Assumes video values are in [-1, 1] range (standard pipeline output).
    """
    # Remove batch dimension first if requested
    if remove_batch_dim:
        video = video[0]  # (B, T, C, H, W) -> (T, C, H, W)
        video = video.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
    else:
        video = video.permute(0, 1, 3, 4, 2)  # (B, T, C, H, W) -> (B, T, H, W, C)

    # Normalize to [0, 1] range
    video = (video / 2 + 0.5).clamp(0, 1)

    # Convert to uint8
    video = (video * 255).round().to(torch.uint8)

    return video


def postprocess_hf_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Post-process image tensor from HuggingFace pipeline output to final format.

    HuggingFace pipelines with output_type="pt" return images in (B, C, H, W) format.

    Args:
        image: Image tensor in (B, C, H, W) or (C, H, W) format from HuggingFace pipeline

    Returns:
        Post-processed image tensor in (H, W, C) uint8 format

    Note:
        Assumes image values are in [-1, 1] range (standard pipeline output).
    """
    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]  # (B, C, H, W) -> (C, H, W)

    # Convert to (H, W, C) format
    image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Normalize to [0, 1] range
    image = (image / 2 + 0.5).clamp(0, 1)

    # Convert to uint8
    image = (image * 255).round().to(torch.uint8)

    return image


class OutputHandler:
    """Handle saving of generated outputs in various formats.

    Supports MediaOutput from all models:
    - Video models (WAN): MediaOutput(video=torch.Tensor)
    - Image models: MediaOutput(image=torch.Tensor)
    - Video+Audio models: MediaOutput(video=torch.Tensor, audio=torch.Tensor)

    Supported output formats:
    - .png: Save single image or middle frame
    - .gif: Save video as animated GIF (no audio)
    - .mp4: Save video with audio (requires diffusers export_utils)
    """

    @staticmethod
    def save(output: MediaOutput, output_path: str, frame_rate: float = 24.0):
        """Save output based on content type and file extension.

        Args:
            output: MediaOutput containing model outputs (image/video/audio)
            output_path: Path to save the output file
            frame_rate: Frames per second for video output (default: 24.0)
        """
        if not isinstance(output, MediaOutput):
            raise ValueError(f"Expected output to be MediaOutput, got {type(output)}")

        file_ext = os.path.splitext(output_path)[1].lower()

        # Determine content type
        if output.image is not None:
            OutputHandler._save_image(output.image, output_path, file_ext)
        elif output.video is not None:
            OutputHandler._save_video(output.video, output.audio, output_path, file_ext, frame_rate)
        else:
            raise ValueError("Unknown output format. MediaOutput has no image or video data.")

    @staticmethod
    def _save_image(image: torch.Tensor, output_path: str, file_ext: str):
        """Save single image output.

        Args:
            image: Image as torch tensor (H, W, C) uint8
            output_path: Path to save the image
            file_ext: File extension (.png, .jpg, etc.)
        """
        if file_ext not in [".png", ".jpg", ".jpeg"]:
            logger.warning(f"Image output requested with {file_ext}, defaulting to .png")
            output_path = output_path.replace(file_ext, ".png")

        # Convert torch.Tensor to PIL Image and save
        image_np = image.cpu().numpy()
        Image.fromarray(image_np).save(output_path)
        logger.info(f"Saved image to {output_path}")

    @staticmethod
    def _save_video(
        video: torch.Tensor,
        audio: Optional[torch.Tensor],
        output_path: str,
        file_ext: str,
        frame_rate: float,
    ):
        """Save video output with optional audio.

        Args:
            video: Video frames as torch tensor (T, H, W, C) with dtype uint8
            audio: Optional audio as torch tensor
            output_path: Path to save the video
            file_ext: File extension (.mp4, .gif, .png)
            frame_rate: Frames per second
        """
        if file_ext == ".mp4":
            OutputHandler._save_mp4(video, audio, output_path, frame_rate)
        elif file_ext == ".gif":
            OutputHandler._save_gif(video, output_path, frame_rate)
        elif file_ext == ".png":
            OutputHandler._save_middle_frame(video, output_path)
        else:
            logger.warning(f"Unsupported video output format: {file_ext}, defaulting to .png")
            output_path = output_path.replace(file_ext, ".png")
            OutputHandler._save_middle_frame(video, output_path)

    @staticmethod
    def _save_mp4(
        video: torch.Tensor, audio: Optional[torch.Tensor], output_path: str, frame_rate: float
    ):
        """Save video with optional audio as MP4.

        Args:
            video: Video frames as torch tensor (T, H, W, C) uint8
            audio: Optional audio as torch tensor (float32)
            output_path: Output path for MP4
            frame_rate: Frames per second
        """
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video

            # Prepare audio if present
            audio_prepared = audio.float() if audio is not None else None

            # encode_video expects (T, H, W, C) uint8 video and float32 audio
            encode_video(
                video,
                fps=frame_rate,
                audio=audio_prepared,
                audio_sample_rate=24000 if audio_prepared is not None else None,
                output_path=output_path,
            )
            logger.info(f"Saved video{' with audio' if audio is not None else ''} to {output_path}")

        except ImportError:
            logger.warning(
                "diffusers export_utils (encode_video) not available. "
                "Falling back to saving middle frame as PNG."
            )
            png_path = output_path.replace(".mp4", ".png")
            OutputHandler._save_middle_frame(video, png_path)

    @staticmethod
    def _save_gif(video: torch.Tensor, output_path: str, frame_rate: float):
        """Save video as animated GIF.

        Args:
            video: Video frames as torch tensor (T, H, W, C) uint8
            output_path: Output path for GIF
            frame_rate: Frames per second
        """
        # Convert torch.Tensor to numpy for PIL
        video_np = video.cpu().numpy()

        # Convert to list of PIL Images
        frames = [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]

        # Save as animated GIF
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

    @staticmethod
    def _save_middle_frame(video: torch.Tensor, output_path: str):
        """Save middle frame of video as PNG.

        Args:
            video: Video frames as torch tensor (T, H, W, C) uint8
            output_path: Output path for PNG
        """
        # Convert torch.Tensor to numpy for PIL
        video_np = video.cpu().numpy()

        # Extract middle frame
        frame_idx = video_np.shape[0] // 2
        Image.fromarray(video_np[frame_idx]).save(output_path)
        logger.info(f"Saved frame {frame_idx} to {output_path}")

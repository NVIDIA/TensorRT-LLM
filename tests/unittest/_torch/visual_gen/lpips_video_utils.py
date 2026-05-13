import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _video_tensor_to_lpips_batch(
    video: torch.Tensor,
    image_size: tuple[int, int],
    device: str,
) -> torch.Tensor:
    video = video.detach().cpu()
    if video.dim() == 5:
        video = video[0]
    if video.dim() != 4:
        raise ValueError(f"Expected 4D or 5D video tensor, got shape {tuple(video.shape)}")

    if video.shape[-1] == 3:
        frames = video
    elif video.shape[1] == 3:
        frames = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected RGB video frames, got shape {tuple(video.shape)}")

    frames = frames.float()
    if frames.numel() == 0:
        raise ValueError("Generated video contains no frames")
    if frames.min() < 0:
        frames = (frames + 1.0) / 2.0
    elif frames.max() > 2.0:
        frames = frames / 255.0

    batch = frames.clamp(0, 1).permute(0, 3, 1, 2)
    batch = F.interpolate(
        batch,
        size=(image_size[1], image_size[0]),
        mode="bicubic",
        align_corners=False,
    )
    return batch.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def _decode_video_to_lpips_batch(
    video_path: Path,
    image_size: tuple[int, int],
    device: str,
) -> torch.Tensor:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode golden videos for LPIPS comparison")
    if not video_path.exists():
        raise FileNotFoundError(f"Golden video not found: {video_path}")

    width, height = image_size
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height}:flags=bicubic",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to decode golden video {video_path}: {stderr}")

    frame_size = width * height * 3
    if not result.stdout or len(result.stdout) % frame_size != 0:
        raise ValueError(
            f"Decoded video byte size {len(result.stdout)} is not a multiple of frame size {frame_size}"
        )

    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    batch = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    return batch.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def _sample_frame_indices(frame_count: int, max_frames: int) -> torch.Tensor:
    if frame_count <= 0:
        raise ValueError("Cannot sample from an empty video")
    sample_count = min(frame_count, max_frames)
    return torch.linspace(0, frame_count - 1, steps=sample_count).round().long()


def average_video_lpips_score(
    generated_video: torch.Tensor,
    golden_video_path: Path,
    lpips_model,
    device: str,
    image_size: tuple[int, int] = (256, 256),
    max_frames: int = 8,
) -> float:
    """Average LPIPS over evenly sampled paired frames from generated and golden videos."""
    generated = _video_tensor_to_lpips_batch(generated_video, image_size, device)
    golden = _decode_video_to_lpips_batch(golden_video_path, image_size, device)

    paired_frame_count = min(generated.shape[0], golden.shape[0])
    indices = _sample_frame_indices(paired_frame_count, max_frames).to(device=device)
    generated = generated[:paired_frame_count].index_select(0, indices)
    golden = golden[:paired_frame_count].index_select(0, indices)

    with torch.no_grad():
        scores = lpips_model(generated, golden).flatten()
    return scores.mean().item()


def average_video_file_lpips_score(
    generated_video_path: Path,
    golden_video_path: Path,
    lpips_model,
    device: str,
    image_size: tuple[int, int] = (256, 256),
    max_frames: int = 8,
) -> float:
    """Average LPIPS over paired frames decoded from generated and golden videos."""
    generated = _decode_video_to_lpips_batch(generated_video_path, image_size, device)
    golden = _decode_video_to_lpips_batch(golden_video_path, image_size, device)

    paired_frame_count = min(generated.shape[0], golden.shape[0])
    indices = _sample_frame_indices(paired_frame_count, max_frames).to(device=device)
    generated = generated[:paired_frame_count].index_select(0, indices)
    golden = golden[:paired_frame_count].index_select(0, indices)

    with torch.no_grad():
        scores = lpips_model(generated, golden).flatten()
    return scores.mean().item()

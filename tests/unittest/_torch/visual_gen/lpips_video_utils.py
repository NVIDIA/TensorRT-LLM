# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import shutil
import subprocess
from pathlib import Path

import lpips
import numpy as np
import torch


def _video_tensor_to_lpips_batch(
    video: torch.Tensor,
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
    return batch.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def _probe_video_size(video_path: Path) -> tuple[int, int]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to inspect videos for LPIPS comparison")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to inspect video {video_path}: {stderr}")

    size_lines = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if not size_lines:
        raise RuntimeError(f"Failed to inspect video size for {video_path}: empty ffprobe output")
    size_text = size_lines[0]
    try:
        width_text, height_text = size_text.split("x")
        width, height = int(width_text), int(height_text)
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Failed to parse video size for {video_path}: {size_text!r}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size for {video_path}: {width}x{height}")
    return width, height


def _decode_video_to_lpips_batch(
    video_path: Path,
    device: str,
) -> torch.Tensor:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode golden videos for LPIPS comparison")
    if not video_path.exists():
        raise FileNotFoundError(f"Golden video not found: {video_path}")

    width, height = _probe_video_size(video_path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
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


def _validate_paired_video_shapes(generated: torch.Tensor, golden: torch.Tensor) -> None:
    if generated.shape[1:] != golden.shape[1:]:
        raise ValueError(
            "Generated and golden video frames must have the same native LPIPS tensor shape: "
            f"{tuple(generated.shape[1:])} vs {tuple(golden.shape[1:])}."
        )


def _sample_frame_indices(frame_count: int, max_frames: int | None = None) -> torch.Tensor:
    if frame_count <= 0:
        raise ValueError("Cannot sample from an empty video")
    if max_frames is None or max_frames >= frame_count:
        return torch.arange(frame_count, dtype=torch.long)
    if max_frames <= 0:
        raise ValueError("Video LPIPS max frames must be positive")
    sample_count = min(frame_count, max_frames)
    return torch.linspace(0, frame_count - 1, steps=sample_count).round().long()


def average_video_lpips_score(
    generated_video: torch.Tensor,
    golden_video_path: Path,
    lpips_model,
    device: str,
    max_frames: int | None = None,
) -> float:
    """Average LPIPS over paired frames from generated and golden videos."""
    generated = _video_tensor_to_lpips_batch(generated_video, device)
    golden = _decode_video_to_lpips_batch(golden_video_path, device)
    _validate_paired_video_shapes(generated, golden)

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
    max_frames: int | None = None,
) -> float:
    """Average LPIPS over paired frames decoded from generated and golden videos."""
    generated = _decode_video_to_lpips_batch(generated_video_path, device)
    golden = _decode_video_to_lpips_batch(golden_video_path, device)
    _validate_paired_video_shapes(generated, golden)

    paired_frame_count = min(generated.shape[0], golden.shape[0])
    indices = _sample_frame_indices(paired_frame_count, max_frames).to(device=device)
    generated = generated[:paired_frame_count].index_select(0, indices)
    golden = golden[:paired_frame_count].index_select(0, indices)

    with torch.no_grad():
        scores = lpips_model(generated, golden).flatten()
    return scores.mean().item()


def _load_lpips_model(device: str):
    try:
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"LPIPS model could not be loaded: {exc}") from exc


def _cleanup_lpips_model(lpips_model, device: str) -> None:
    del lpips_model
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_video_lpips_score_below_threshold(
    generated_video: torch.Tensor,
    golden_video_path: Path,
    threshold: float,
    test_label: str,
    device: str = "cuda",
    max_frames: int | None = None,
) -> float:
    """Compute tensor-vs-file video LPIPS and assert it is below the threshold."""
    lpips_model = _load_lpips_model(device)
    try:
        lpips_score = average_video_lpips_score(
            generated_video,
            golden_video_path,
            lpips_model,
            device,
            max_frames=max_frames,
        )
    finally:
        _cleanup_lpips_model(lpips_model, device)

    print(f"\n[{test_label}] mean score: {lpips_score:.6f}")
    assert lpips_score < threshold, (
        f"Mean LPIPS too high: {lpips_score:.6f} (expected < {threshold:.6f})"
    )
    return lpips_score


def assert_video_file_lpips_score_below_threshold(
    generated_video_path: Path,
    golden_video_path: Path,
    threshold: float,
    test_label: str,
    device: str = "cuda",
    max_frames: int | None = None,
) -> float:
    """Compute file-vs-file video LPIPS and assert it is below the threshold."""
    lpips_model = _load_lpips_model(device)
    try:
        lpips_score = average_video_file_lpips_score(
            generated_video_path,
            golden_video_path,
            lpips_model,
            device,
            max_frames=max_frames,
        )
    finally:
        _cleanup_lpips_model(lpips_model, device)

    print(f"\n[{test_label}] mean score: {lpips_score:.6f}")
    assert lpips_score < threshold, (
        f"Mean LPIPS too high: {lpips_score:.6f} (expected < {threshold:.6f})"
    )
    return lpips_score

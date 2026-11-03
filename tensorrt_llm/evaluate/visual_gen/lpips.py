# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared LPIPS image and video scoring for evaluation entry points."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import torch
from PIL import Image


def _to_lpips_tensor(image: Any, device: str) -> torch.Tensor:
    """Convert an RGB image-like object to LPIPS input: NCHW float in [-1, 1]."""
    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.array(image.convert("RGB")))
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        raise TypeError(f"Unsupported image type for LPIPS: {type(image)}")

    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] == 3 and tensor.shape[-1] != 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {tuple(tensor.shape)}")

    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def score_images(
    lpips_model: Any,
    generated_image: Any,
    reference_image: Image.Image,
    device: str,
) -> float:
    generated_tensor = _to_lpips_tensor(generated_image, device)
    reference_tensor = _to_lpips_tensor(reference_image, device)
    if generated_tensor.shape != reference_tensor.shape:
        raise ValueError(
            "Generated image and reference image must have the same LPIPS tensor shape: "
            f"{tuple(generated_tensor.shape)} vs {tuple(reference_tensor.shape)}."
        )
    return _mean_lpips(lpips_model, generated_tensor, reference_tensor)


def _decode_video_to_lpips_batch(
    video_path: pathlib.Path,
    device: str,
) -> torch.Tensor:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for LPIPS comparison: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for LPIPS comparison: {video_path}")

    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Decoded video contains no frames: {video_path}")

    return _video_frames_to_tensor(frames, device)


def _validate_paired_video_shapes(generated: torch.Tensor, reference: torch.Tensor) -> None:
    if generated.shape[1:] != reference.shape[1:]:
        raise ValueError(
            "Generated and reference video frames must have the same LPIPS tensor shape: "
            f"{tuple(generated.shape[1:])} vs {tuple(reference.shape[1:])}."
        )


def score_video_files(
    lpips_model: Any,
    generated_video_path: pathlib.Path,
    reference_video_path: pathlib.Path,
    device: str,
    *,
    strict_pairing: bool = False,
) -> float:
    """Score paired frames, truncating to the shorter video by default.

    Strict pairing requires matching FPS and frame counts, rejects non-finite
    scores, and processes one frame pair at a time to bound memory usage.
    """
    if strict_pairing:
        return _score_video_strict(lpips_model, generated_video_path, reference_video_path, device)
    generated = _decode_video_to_lpips_batch(generated_video_path, device)
    reference = _decode_video_to_lpips_batch(reference_video_path, device)
    _validate_paired_video_shapes(generated, reference)

    paired_frame_count = min(generated.shape[0], reference.shape[0])
    generated = generated[:paired_frame_count]
    reference = reference[:paired_frame_count]

    return _mean_lpips(lpips_model, generated, reference)


def make_lpips_model(net: str, device: str) -> Any:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"LPIPS device is {device}, but CUDA is not available.")
    import lpips

    return lpips.LPIPS(net=net, verbose=False).to(device).eval()


def _video_frames_to_tensor(frames: list[np.ndarray], device: str) -> torch.Tensor:
    """Normalize decoded RGB uint8 frames to NCHW float32 in [-1, 1]."""
    frames_array = np.stack(frames, axis=0)
    batch = torch.from_numpy(frames_array.copy()).permute(0, 3, 1, 2).float() / 255.0
    return batch.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def _mean_lpips(model: Any, generated: torch.Tensor, reference: torch.Tensor) -> float:
    with torch.no_grad():
        return float(model(generated, reference).reshape(-1).mean().item())


def _score_video_strict(
    model: Any, generated_path: pathlib.Path, reference_path: pathlib.Path, device: str
) -> float:
    import cv2

    generated = cv2.VideoCapture(str(generated_path))
    reference = cv2.VideoCapture(str(reference_path))
    total = 0.0
    count = 0
    try:
        if not generated.isOpened() or not reference.isOpened():
            raise ValueError(f"Cannot decode video pair: {generated_path}, {reference_path}")
        generated_fps = generated.get(cv2.CAP_PROP_FPS)
        reference_fps = reference.get(cv2.CAP_PROP_FPS)
        if not (generated_fps > 0 and reference_fps > 0) or not math.isclose(
            generated_fps, reference_fps, rel_tol=1e-6
        ):
            raise ValueError("Frame rate mismatch or invalid frame rate")
        while True:
            has_generated, frame = generated.read()
            has_reference, baseline = reference.read()
            if has_generated != has_reference:
                raise ValueError("Frame count mismatch")
            if not has_generated:
                break
            pair = [
                _video_frames_to_tensor([cv2.cvtColor(value, cv2.COLOR_BGR2RGB)], device)
                for value in (frame, baseline)
            ]
            _validate_paired_video_shapes(*pair)
            score = _mean_lpips(model, *pair)
            if not math.isfinite(score):
                raise ValueError("Non-finite LPIPS score")
            total += score
            count += 1
        if not count:
            raise ValueError("Empty video pair")
        return total / count
    finally:
        generated.release()
        reference.release()

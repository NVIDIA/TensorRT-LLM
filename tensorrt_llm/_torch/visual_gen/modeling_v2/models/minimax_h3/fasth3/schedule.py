# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import MiniMaxH3Scheduler


def set_dmd_timesteps(
    video_scheduler: MiniMaxH3Scheduler,
    audio_scheduler: MiniMaxH3Scheduler,
    manifest: dict,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    steps = manifest["dmd_denoising_steps"]
    if (
        not steps
        or any(type(step) is not int or not 0 < step <= 1000 for step in steps)
        or any(left <= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("FastH3 DMD steps must be strictly decreasing integers in (0, 1000].")
    if num_inference_steps != len(steps) + 1:
        raise ValueError(
            f"FastH3 requires {len(steps) + 1} sigma-grid points for {len(steps)} forwards."
        )
    base = torch.tensor([step / 1000.0 for step in steps] + [0.0], dtype=torch.float32)
    for scheduler, shift_key in (
        (video_scheduler, "video_scheduler_shift"),
        (audio_scheduler, "audio_scheduler_shift"),
    ):
        shift = float(scheduler.shift)
        if shift_key in manifest and shift != float(manifest[shift_key]):
            raise ValueError(f"FastH3 {shift_key} disagrees with the checkpoint scheduler.")
        # Shift each modality's trained noise clock once; explicit sigmas are not shifted.
        sigmas = shift * base / (1 + (shift - 1) * base)
        scheduler.set_timesteps(sigmas=sigmas, device=device)

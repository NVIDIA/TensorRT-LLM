# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canvas and packed text-to-image coordinates for MiniMax-H3-Image."""

import math

import numpy as np
import torch

DEFAULT_MIN_AREA = 2048**2
MIN_AREA_RANGE = (1024, 4096**2)


def generation_canvas(height: int, width: int, min_generation_area: int) -> tuple[int, int]:
    if min(height, width) <= 0:
        raise ValueError("height and width must be positive")
    if (
        type(min_generation_area) is not int
        or not MIN_AREA_RANGE[0] <= min_generation_area <= MIN_AREA_RANGE[1]
    ):
        raise ValueError(f"min_generation_area must be an integer in {MIN_AREA_RANGE}")
    scale = max(1.0, math.sqrt(min_generation_area / (height * width)))
    # 16x VAE compression followed by 2x2 image patch projection.
    return max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32)


def text_image_positions(
    text_length: int, patch_rows: int, patch_cols: int, device: torch.device
) -> torch.Tensor:
    """Positions for [text | image]. Each image occupies one slot on the rotary time axis."""
    if min(text_length, patch_rows, patch_cols) <= 0:
        raise ValueError("A packed request needs text and a nonempty image grid.")
    side = math.sqrt(patch_rows * patch_cols)

    def axis(n: int) -> torch.Tensor:
        ratio = n / side
        left = (1 - ratio) * 0.5
        return torch.from_numpy(32 * np.linspace(left, left + ratio, n, endpoint=False))

    y, x = torch.meshgrid(axis(patch_rows), axis(patch_cols), indexing="ij")
    pos = torch.zeros(text_length + patch_rows * patch_cols, 3, dtype=torch.float64)
    pos[:text_length, 0] = torch.arange(text_length)
    pos[text_length:, 0] = text_length
    pos[text_length:, 1] = y.flatten()
    pos[text_length:, 2] = x.flatten()
    return pos.to(device)

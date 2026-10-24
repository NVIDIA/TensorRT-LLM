# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestep coordinates for H3 checkpoints with pruned AdaLN projections.

Adapted from SGLang MiniMaxH3DiTModel._time_embedding (Apache-2.0).
See checkpoints/COMFY_IMPORT_NOTICE for the pinned source and modifications.
"""

import torch


def interpolate_adaln_curve(table: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Interpolate float32 curve coordinates for normalized timesteps in [0, 1].

    Values outside the interval clamp to its endpoints. The final endpoint
    uses the last interval with interpolation fraction one.
    """
    if table.ndim != 2 or table.shape[0] < 2 or table.dtype != torch.float32:
        raise ValueError("AdaLN curve must be a float32 matrix with at least two rows")
    positions = timesteps.float().clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = positions.floor().long().clamp(max=table.shape[0] - 2)
    fraction = (positions - lower).unsqueeze(-1)
    shape = (*lower.shape, table.shape[1])
    lower_value = table.index_select(0, lower.reshape(-1)).reshape(shape)
    upper_value = table.index_select(0, (lower + 1).reshape(-1)).reshape(shape)
    return torch.lerp(lower_value, upper_value, fraction)

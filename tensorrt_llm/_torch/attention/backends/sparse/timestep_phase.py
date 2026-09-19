# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestep-driven dense-or-sparse phase shared by sparse attention algorithms."""

from __future__ import annotations

import math
import numbers
from typing import Optional

import torch


def timestep_to_float(timestep: object) -> Optional[float]:
    """Return the denoising timestep as a finite float, or ``None`` when absent.

    Per-token timesteps, such as Wan I2V where the conditioning frame keeps
    timestep zero, reduce to their largest value so the schedule stays dense
    until every live token is below the cutoff.
    """

    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        timestep = timestep.amax().item()
    if isinstance(timestep, bool) or not isinstance(timestep, numbers.Real):
        raise TypeError("timestep must be a real scalar or tensor")
    value = float(timestep)
    if not math.isfinite(value):
        raise ValueError("timestep must be finite")
    return value


def graph_phase_for_timestep(
    timestep: object,
    *,
    disabled_until_timestep: Optional[float],
) -> Optional[int]:
    """Return 0 for the dense prefix and 1 for the sparse suffix.

    ``None`` when the algorithm has no cutoff or the call carries no timestep;
    CUDA Graph runners omit their phase key part in that case.
    """

    if disabled_until_timestep is None:
        return None
    value = timestep_to_float(timestep)
    if value is None:
        return None
    return int(value < disabled_until_timestep)


__all__ = ["graph_phase_for_timestep", "timestep_to_float"]

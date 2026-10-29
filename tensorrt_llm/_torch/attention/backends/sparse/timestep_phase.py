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


# The V-Smooth schedule of VC-Attention on the normalized timestep. The paper
# smooths V over the first quarter of the denoising steps and regroups the value
# tokens every fourth step inside that window. Here the window is the top quarter
# of the normalized timestep range and the regroup cadence is one bin of the
# window per refresh, so the schedule no longer depends on the step count the
# scheduler happens to use. The vendored preprocessor takes the schedule as a
# step index and a step count, so the timestep maps onto a fixed virtual
# schedule of VC_SCHEDULE_STEPS steps.
VC_SMOOTH_TIMESTEP_FRACTION = 0.25
VC_REGROUPS = 4
VC_PERM_REFRESH_EVERY = 4
VC_SCHEDULE_STEPS = int(round(VC_REGROUPS * VC_PERM_REFRESH_EVERY / VC_SMOOTH_TIMESTEP_FRACTION))


def vc_denoise_step_for_timestep(timestep: object) -> Optional[tuple[int, int]]:
    """Return the virtual ``(step_index, num_steps)`` of the V-Smooth schedule.

    ``None`` when the call carries no timestep. Timestep one is step zero and
    timestep zero the last step.
    """

    value = timestep_to_float(timestep)
    if value is None:
        return None
    step = int(math.floor((1.0 - min(max(value, 0.0), 1.0)) * VC_SCHEDULE_STEPS))
    return min(step, VC_SCHEDULE_STEPS - 1), VC_SCHEDULE_STEPS


def vc_smooth_phase_for_timestep(timestep: object) -> Optional[int]:
    """Return the V-Smooth phase of a timestep for the CUDA graph key.

    2 regroups the value tokens and demeans V, 1 keeps the last grouping and
    demeans V, 0 runs the plain per-channel E4M3 V after the window. ``None``
    when the call carries no timestep; CUDA Graph runners omit their phase key
    part in that case.
    """

    denoise_step = vc_denoise_step_for_timestep(timestep)
    if denoise_step is None:
        return None
    step, num_steps = denoise_step
    if step >= math.ceil(VC_SMOOTH_TIMESTEP_FRACTION * num_steps):
        return 0
    return 2 if step % VC_PERM_REFRESH_EVERY == 0 else 1


def vc_denoise_step_for_phase(phase: int) -> tuple[int, int]:
    """Return a virtual step of the schedule in ``phase``.

    Under CUDA graph capture the timestep tensor cannot be read, so the host
    resolved phase selects a representative step of that phase instead.
    """

    if phase == 2:
        return 0, VC_SCHEDULE_STEPS
    if phase == 1:
        return 1, VC_SCHEDULE_STEPS
    return VC_SCHEDULE_STEPS - 1, VC_SCHEDULE_STEPS


__all__ = [
    "graph_phase_for_timestep",
    "timestep_to_float",
    "vc_denoise_step_for_phase",
    "vc_denoise_step_for_timestep",
    "vc_smooth_phase_for_timestep",
]

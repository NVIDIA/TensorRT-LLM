# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)
from torch import Tensor


def timeshift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def build_timesteps(n_steps: int, alpha: float) -> Tensor:
    return timeshift(alpha, torch.linspace(1.0, 0.0, n_steps + 1))


class Flux3FlowMatchScheduler:
    """Adapt BFL's shifted timesteps and Euler update to the pipeline interface."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.sigmas = build_timesteps(num_inference_steps, self.alpha)
        self.timesteps = self.sigmas[:-1].to(device)
        self._deltas = self.sigmas.diff().tolist()
        self._step_index = 0

    def step(self, model_output, timestep, sample, return_dict=True):
        previous = sample + self._deltas[self._step_index] * model_output
        self._step_index += 1
        if not return_dict:
            return (previous,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=previous)

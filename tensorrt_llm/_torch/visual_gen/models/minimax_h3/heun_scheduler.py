# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Heun sampling on MiniMax-H3's rectified-flow schedule."""

from typing import Optional

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput

from tensorrt_llm._torch.visual_gen.samplers import heun_derivative, heun_step, heun_timesteps


class MiniMaxH3HeunScheduler(MiniMaxH3Scheduler):
    """Heun's second-order method on the ``MiniMaxH3Scheduler`` sigma grid.

    ``timesteps`` lists Heun's evaluation order (see ``heun_timesteps``), so a
    denoising loop that evaluates the model and calls ``step`` once per entry
    runs Heun unchanged. The first call of a step takes the parent's Euler step
    and returns it as the proposal; the second call, made at the next timestep
    on that proposal, averages the two derivatives. The step that ends at
    ``sigma = 0`` has a single call and keeps its Euler result.
    """

    order = 2
    # (start sample, its derivative, sigma, sigma_next) between a step's two calls.
    _pending: Optional[tuple] = None

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        sigmas: Optional[list[float] | torch.Tensor] = None,
    ) -> None:
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
        self.timesteps = heun_timesteps(self.timesteps)
        # Timesteps repeat, so the step index is tracked rather than looked up by value.
        self._begin_index = 0
        self._pending = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> MiniMaxH3SchedulerOutput | tuple:
        if self._pending is None:
            prev_sample = self._propose(model_output, timestep, sample)
        else:
            prev_sample = self._correct(model_output, timestep, sample)
        if not return_dict:
            return (prev_sample,)
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)

    def _denoised(
        self, model_output: torch.Tensor, timestep: float | torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """The parent's ``x0 = x_t + (1 - t) * v``, with sigma taken from the timestep."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sample + sigma * model_output

    def _propose(
        self, model_output: torch.Tensor, timestep: float | torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        step_index = self._begin_index if self._step_index is None else self._step_index
        if step_index >= len(self.sigmas) - 1:
            raise ValueError(
                "MiniMaxH3HeunScheduler.step() was called past the end of the schedule."
            )
        (proposal,) = super().step(model_output, timestep, sample, return_dict=False)
        # Only the last grid sigma is zero, so every earlier step gets a correction.
        if step_index < len(self.sigmas) - 2:
            sigma = self.sigmas[step_index]
            denoised = self._denoised(model_output, timestep, sample)
            derivative = heun_derivative(sample, denoised, sigma)
            # A copy, because callers may write the proposal back into ``sample`` in place.
            self._pending = (sample.clone(), derivative, sigma, self.sigmas[step_index + 1])
        return proposal

    def _correct(
        self, model_output: torch.Tensor, timestep: float | torch.Tensor, proposal: torch.Tensor
    ) -> torch.Tensor:
        sample, derivative, sigma, sigma_next = self._pending
        self._pending = None
        denoised = self._denoised(model_output, timestep, proposal)
        next_derivative = heun_derivative(proposal, denoised, sigma_next)
        return heun_step(sample, derivative, next_derivative, sigma, sigma_next)

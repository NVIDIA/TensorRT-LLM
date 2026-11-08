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


import torch

from .diffusion_steps import EulerDiffusionStep
from .schedulers import LTX2Scheduler


class NativeSchedulerAdapter:
    """Adapts native LTX-2 scheduler to the interface expected by
    ``BasePipeline._scheduler_step()``.

    Usage::

        sched = NativeSchedulerAdapter()
        sched.set_timesteps(num_inference_steps, latent=latent_5d)
        for t in sched.timesteps:
            noise_pred = transformer(...)
            latents = sched.step(noise_pred, t, latents)[0]
    """

    def __init__(
        self,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
    ):
        self._scheduler = LTX2Scheduler()
        self._diffusion_step = EulerDiffusionStep()
        self.sigmas: torch.FloatTensor | None = None
        self._step_index: int = 0

        self._max_shift = max_shift
        self._base_shift = base_shift
        self._stretch = stretch
        self._terminal = terminal

    # -- properties ---------------------------------------------------------

    @property
    def timesteps(self) -> torch.Tensor:
        """Sigma values used as timesteps (excluding terminal sigma)."""
        assert self.sigmas is not None, "Call set_timesteps() first"
        return self.sigmas[:-1]

    # -- public API --------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        latent: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        merged = dict(
            max_shift=self._max_shift,
            base_shift=self._base_shift,
            stretch=self._stretch,
            terminal=self._terminal,
        )
        merged.update(kwargs)
        self.sigmas = self._scheduler.execute(steps=num_inference_steps, latent=latent, **merged)
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = False,
        **_kwargs,
    ):
        """One Euler step.  Matches the call signature used by
        ``BasePipeline._scheduler_step``:
        ``scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]``
        """
        result = self._diffusion_step.step(
            sample=sample,
            denoised_sample=model_output,
            sigmas=self.sigmas,
            step_index=self._step_index,
        )
        self._step_index += 1
        if return_dict:
            return {"prev_sample": result}
        return (result,)

    # -- helpers -----------------------------------------------------------

    def __deepcopy__(self, memo):
        """Support ``copy.deepcopy`` for creating per-stream schedulers."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        new._scheduler = LTX2Scheduler()
        new._diffusion_step = EulerDiffusionStep()
        new.sigmas = self.sigmas.clone() if self.sigmas is not None else None
        new._step_index = self._step_index
        new._max_shift = self._max_shift
        new._base_shift = self._base_shift
        new._stretch = self._stretch
        new._terminal = self._terminal
        return new

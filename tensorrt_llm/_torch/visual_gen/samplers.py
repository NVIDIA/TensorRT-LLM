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
"""Sampling-algorithm math shared by VisualGen schedulers.

A pipeline that offers more than one sampler lists them in
``BasePipeline.supported_samplers``; a request selects one through
``extra_params["sampler"]``. The helpers below work on the noise level
``sigma`` and the model's clean-sample estimate ``x0``, so a model-specific
scheduler only supplies its own ``x0`` recovery and sigma grid.
"""

from typing import Union

import torch

Sigma = Union[torch.Tensor, float]


def _compute_dtype(sample: torch.Tensor) -> torch.dtype:
    return torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype


def heun_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
    """Expand a one-evaluation-per-step schedule into Heun's evaluation order.

    Each step evaluates the model at its own timestep, and every step except the
    last evaluates it again at the next timestep to correct the Euler proposal.
    The last step ends at ``sigma = 0``, where the derivative is undefined, so it
    keeps the Euler result: ``S`` steps take ``2 * S - 1`` evaluations.
    """
    return torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])


def heun_derivative(sample: torch.Tensor, denoised: torch.Tensor, sigma: Sigma) -> torch.Tensor:
    """``dx/dsigma = (x - x0) / sigma``, computed in float32 for half-precision samples."""
    dtype = _compute_dtype(sample)
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=dtype)
    return (sample.to(dtype) - denoised.to(dtype)) / sigma


def heun_step(
    sample: torch.Tensor,
    derivative: torch.Tensor,
    next_derivative: torch.Tensor,
    sigma: Sigma,
    sigma_next: Sigma,
) -> torch.Tensor:
    """Trapezoidal update ``x + (sigma_next - sigma) * (d + d_next) / 2``.

    ``derivative`` is taken at ``(sample, sigma)`` and ``next_derivative`` at the
    Euler proposal for ``sigma_next``. The result has ``sample``'s dtype.
    """
    dtype = _compute_dtype(sample)
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=dtype)
    sigma_next = torch.as_tensor(sigma_next, device=sample.device, dtype=dtype)
    update = (sigma_next - sigma) * (derivative.to(dtype) + next_derivative.to(dtype)) / 2
    return (sample.to(dtype) + update).to(sample.dtype)

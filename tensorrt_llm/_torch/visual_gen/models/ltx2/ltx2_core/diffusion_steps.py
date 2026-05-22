# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch

from .protocols import DiffusionStepProtocol
from .utils_ltx2 import to_velocity


class EulerDiffusionStep(DiffusionStepProtocol):
    """
    First-order Euler method for diffusion sampling.
    sample + velocity * dt.
    """

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if step_index < 0 or step_index >= len(sigmas) - 1:
            raise ValueError(
                f"step_index={step_index} out of bounds for sigmas with length {len(sigmas)}"
            )

        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)
        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

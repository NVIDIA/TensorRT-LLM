# Copyright 2025 Lightricks. Ported from https://github.com/Lightricks/LTX-2
# Euler diffusion step for flow-matching.

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
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)
        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

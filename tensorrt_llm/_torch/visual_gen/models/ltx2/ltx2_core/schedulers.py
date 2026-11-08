# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# LTX-2 sigma schedulers (token-count-dependent shift).

import math

import torch

from .protocols import SchedulerProtocol

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler(SchedulerProtocol):
    """
    Default scheduler for LTX-2 diffusion sampling.
    Generates sigma schedule with token-count-dependent shifting.
    """

    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> torch.FloatTensor:
        tokens = math.prod(latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
        device = latent.device if latent is not None else None
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = float((tokens) * mm + b)
        exp_shift = math.exp(sigma_shift)
        power = 1
        inv_sigma = torch.where(sigmas != 0, 1.0 / sigmas.clamp(min=1e-8), torch.zeros_like(sigmas))
        denom = exp_shift + (inv_sigma - 1) ** power
        new_sigmas = torch.where(sigmas != 0, exp_shift / denom, sigmas)
        sigmas = new_sigmas

        # Stretch sigmas so final value matches terminal
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas = sigmas.clone()
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)

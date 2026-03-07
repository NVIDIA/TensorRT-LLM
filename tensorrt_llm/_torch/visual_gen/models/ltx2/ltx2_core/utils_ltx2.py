# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch


def to_velocity(
    sample: torch.Tensor,
    sigma: torch.Tensor,
    denoised: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert denoised prediction to flow velocity (flow-matching parameterization).

    velocity = (sample - denoised) / sigma
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised.to(calc_dtype)) / sigma).to(sample.dtype)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMS normalization without learnable weights.

    Used for adaptive layer norm where scale/shift come from external modulation.
    Must match the reference: torch.nn.functional.rms_norm.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 modality bundle.

This is the LTX-2 Modality plus an explicit sigma field, kept distinct from
timesteps because they drive different conditioning: timesteps may vary per
latent token and drives the per-token AdaLN modulation, while sigma is the
global denoising value that derives the prompt_timestep behind the text-context
K/V modulation.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LTX23Modality:
    """Input data for a single modality (video or audio) in the LTX-2.3 transformer."""

    latent: torch.Tensor  # (B, T, D): packed latent tokens
    timesteps: torch.Tensor  # (B,) or (B, T): per-batch or per-token timesteps
    sigma: torch.Tensor  # (B,) global current denoising value (drives prompt cond)
    positions: torch.Tensor  # (B, n_dims, T)[, 2]: index grid
    context: torch.Tensor  # connector text embeddings (already inner_dim)
    enabled: bool = True
    context_mask: torch.Tensor | None = None

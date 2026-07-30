# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 modality bundle.

Extends the LTX-2 ``Modality`` with an explicit ``sigma`` field. The two are
kept distinct on purpose:

* ``timesteps`` may vary per latent token (conditioned / masked generation) and
  drives the per-token AdaLN modulation (MSA / MLP / text-CA query).
* ``sigma`` is the *global* current denoising value. It derives the
  ``prompt_timestep`` that drives the sigma-dependent text-context (K/V)
  modulation via ``prompt_scale_shift_table``. This is the LTX-2.3-specific
  conditioning that makes text K/V step-varying (so it cannot be cached like
  LTX-2's static text K/V).
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

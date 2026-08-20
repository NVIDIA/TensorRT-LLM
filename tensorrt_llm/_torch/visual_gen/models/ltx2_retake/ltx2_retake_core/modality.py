# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """Input data for a single modality (video or audio) in the transformer.

    Bundles latent tokens, timestep values, and positional information.
    """

    latent: torch.Tensor  # (B, T, D): packed latent tokens
    timesteps: torch.Tensor  # (B,) or (B, T): per-batch or per-token timesteps
    positions: torch.Tensor  # (B, n_dims, T) or (B, n_dims, T, 2): index grid
    enabled: bool = True
    sigma: torch.Tensor | None = None  # (B,) scalar sigma for cross-attention AdaLN
    cross_modality_sigma: torch.Tensor | None = None

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

from dataclasses import dataclass, field

import torch

from ...ltx2.ltx2_core.modality import Modality


@dataclass(frozen=True)
class LTX23Modality(Modality):
    """LTX-2 Modality plus the global denoising sigma."""

    sigma: torch.Tensor = field(kw_only=True)  # (B,) drives prompt K/V
    # Optional global sigma for the audio-video cross-attention gate.
    cross_modality_sigma: torch.Tensor | None = field(default=None, kw_only=True)

# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# Perturbation configs for STG (spatiotemporal guidance).
#
# A perturbation masks out the contribution of an attention sub-layer for
# selected transformer blocks and batch elements. During the "perturbed"
# forward pass the mask zeros the attention output, producing a prediction
# that *lacks* that attention signal so the guider can amplify the
# difference.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class PerturbationType(Enum):
    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"


@dataclass(frozen=True)
class Perturbation:
    type: PerturbationType
    blocks: list[int] | None  # None → all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False
        if self.blocks is None:
            return True
        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Per-sample perturbation list (or *None* = no perturbation)."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False
        return any(p.is_perturbed(perturbation_type, block) for p in self.perturbations)


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """Batch of per-sample perturbation configs."""

    perturbations: list[PerturbationConfig]

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return all(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    def mask(
        self,
        perturbation_type: PerturbationType,
        block: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[B]`` tensor: 1.0 = keep, 0.0 = skip."""
        m = torch.ones(len(self.perturbations), device=device, dtype=dtype)
        for i, p in enumerate(self.perturbations):
            if p.is_perturbed(perturbation_type, block):
                m[i] = 0
        return m

    def mask_like(
        self,
        perturbation_type: PerturbationType,
        block: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return broadcastable mask shaped ``[B, 1, …]``."""
        m = self.mask(perturbation_type, block, values.device, values.dtype)
        return m.view(m.numel(), *([1] * (values.ndim - 1)))


def build_stg_perturbation_config(stg_blocks: list[int]) -> PerturbationConfig:
    """Build a perturbation config that skips video self-attention at *stg_blocks*."""
    return PerturbationConfig(
        perturbations=[
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks),
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks),
        ]
    )

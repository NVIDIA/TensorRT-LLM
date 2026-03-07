# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# Multi-modal guidance for LTX-2 diffusion.
#
# Supports three guidance modes combined additively:
#   - CFG (classifier-free guidance): positive vs. negative text conditioning
#   - STG (spatiotemporal guidance): positive vs. attention-perturbed
#   - Modality guidance: positive vs. cross-modal-attention-isolated
# Plus optional variance-preserving rescale.

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class MultiModalGuiderParams:
    """Parameters controlling multi-modal guidance strength.

    Defaults match the Lightricks reference for video generation:
      cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
      modality_scale=3.0, stg_blocks=[29].
    """

    cfg_scale: float = 3.0
    stg_scale: float = 0.0
    stg_blocks: list[int] = field(default_factory=list)
    rescale_scale: float = 0.7
    modality_scale: float = 1.0
    skip_step: int = 0


class MultiModalGuider:
    """Combine CFG, STG, and modality guidance predictions."""

    def __init__(self, params: MultiModalGuiderParams):
        self.params = params

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def do_unconditional_generation(self) -> bool:
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0

    # ------------------------------------------------------------------
    # Guidance calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
    ) -> torch.Tensor:
        """Combine guidance predictions.

        pred = cond
             + (cfg_scale - 1) * (cond - uncond_text)
             + stg_scale         * (cond - uncond_perturbed)
             + (modality_scale - 1) * (cond - uncond_modality)
        """
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )

        if self.params.rescale_scale != 0:
            factor = cond.std() / pred.std().clamp(min=1e-8)
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor

        return pred

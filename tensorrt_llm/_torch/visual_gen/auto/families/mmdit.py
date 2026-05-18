# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic MM-DiT adapter — fallback when no family-specific adapter matches.

Used when an unrecognized but Diffusers-shaped checkpoint routes to the auto
path under ``pipeline_mode="fallback"``. Assumes the transformer follows the
standard MM-DiT calling convention: joint text+image stream, dual modulation
streams, RoPE positional encodings. Best-effort — families with a
custom signature should ship their own adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


class MMDiTAdapter(VisGenFamilyAdapter):
    family = "MMDiT-generic"
    diffusers_transformer_cls = ""  # matches by fallback, not by class name

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        raise NotImplementedError(
            "MMDiTAdapter.example_inputs is a generic-fallback stub. To onboard a new "
            "Diffusers transformer family, subclass VisGenFamilyAdapter and supply "
            "example_inputs / dynamic_shapes (see SD3Adapter or FluxAdapter for examples)."
        )

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        raise NotImplementedError("MMDiTAdapter.dynamic_shapes lands per-family.")

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        return RewritePolicy(attention_backend=cfg.attention.backend)

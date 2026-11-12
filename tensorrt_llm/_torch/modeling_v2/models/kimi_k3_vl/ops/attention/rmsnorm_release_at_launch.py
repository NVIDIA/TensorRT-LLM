# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of the MLA q_a latent that releases its programmatic dependent (q_b) at launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

from ...kernels.k3r_mla_q import q_norm_pdl as _norm


class RmsnormReleaseAtLaunch(OpWrapper):
    """``rmsnorm(x, weight, eps)`` of BF16 ``x`` ``[M, H]`` (unit column stride), a new contiguous
    BF16 tensor; FlashInfer's CuTe DSL ``RMSNormKernel`` with its PDL trigger at launch."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Spec: `tokens` T; `x` is the q_a third of the fused q_a / kv_a projection output
    # [T, 1536 + 512 + 64] (row stride 2112; one row counts as contiguous), weight [1536], eps 1e-6.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(tokens=tokens, width=1536, row_stride=2112, eps=1e-6))
        for why, tokens in (
            ("helix16 ULL c1: one decode token (the contiguous compile)", 1),
            ("helix16 ULL c2 / c4 with draft tokens", 8),
            ("helix8 decode batch", 64),
            ("helix8 c240-sized decode batch", 960),
        )
    )

    note = """
    Bitwise `flashinfer.norm.rmsnorm(x, weight, eps, enable_pdl=True)` (the same kernel code);
    only `griddepcontrol.launch_dependents` moves to the CTA's start, so the dependent q_b GEMM
    streams its weights while q_a and this norm run. The dependent must read nothing but its
    weights before its own `griddepcontrol.wait`. Called only where `q_norm_pdl.can_use` holds
    (`features.mla_q_norm_pdl` and PDL on). A process cache holds one compiled kernel per
    (dtype, H, SM, x contiguous); compile before graph capture.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _norm.rmsnorm_release_at_launch(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """The catalog RMSNorm's FP32 reference (the kernel reduces in FP32, rounds once)."""
        return flashinfer_rmsnorm.reference(x, weight, eps)


rmsnorm_release_at_launch = RmsnormReleaseAtLaunch()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA verify's ``[f_a | b]`` projection (CuTe DSL, the fab kernel's GEMM 1 and reduction), for
a verify kernel that computes the forget gate from ``f_a`` itself."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import k3_kda_fab
from .k3_kda_fab_proj import fab_cells


class K3KdaFabFa(OpWrapper):
    """Return ``bfa = bf16(x @ w_bfa^T)`` (bf16 ``[M, N1]``, fp32 accumulation): columns
    ``[0, 128)`` are ``f_a``, ``[128, 128 + num_heads)`` beta, bitwise those of
    ``k3_kda_fab_proj`` for the same ``w_fb`` row count. ``w_fb`` is only prefetched into L2."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = fab_cells("f_a and beta for the verify's in-kernel forget gate")
    note = """
    Kimi K3's KDA verify calls it on the projection aux stream next to the QKVG GEMM, where the
    verify kernel runs in one wave (or above 224 rows) and can_use_k3_kda_fab_fa accepts the
    operands (as for k3_kda_fab_proj); the verify kernel then computes bf16(f_a @ w_fb^T). Columns
    past 128 + num_heads are the padding rows' products (zero weights). The output is a new
    contiguous tensor.

    Same per-device hand-off workspace and ticket counter as k3_kda_fab_proj (process-wide;
    launches on a device must not overlap); every CTA becomes a worker, so all of them (at most
    28) must become co-resident. Compiled kernels are cached per process. The launch uses PDL.
    Importing the entry needs CuTe DSL (SM100 / SM103 / SM107).
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        num_heads: int,
        w_fb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return k3_kda_fab.k3_kda_fab_fa(x, w_bfa, num_heads, w_fb)

    def reference(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        num_heads: int,
        w_fb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``w_fb`` and ``num_heads`` do not enter the result."""
        return (x.double() @ w_bfa.double().t()).to(torch.bfloat16)


k3_kda_fab_fa = K3KdaFabFa()

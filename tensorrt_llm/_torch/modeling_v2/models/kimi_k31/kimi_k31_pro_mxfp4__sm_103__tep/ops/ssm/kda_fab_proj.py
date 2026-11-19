# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA ``[f_a | b] -> f_b`` gate projection of a few verify rows in one launch, through the family's fab kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.ssm import kda_fab as _kda_fab

#: Width of f_a (the f_b reduction size).
FA_DIM = 128
#: Kimi K3.1 hidden size (the [f_a | b] reduction size).
HIDDEN = 7168


def can_use_k3_kda_fab(
    x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
) -> bool:
    """Whether :data:`k31_kda_fab_proj` takes these operands (the kernel's own non-raising check): see the entry's
    note. False without the CuTe DSL."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    return _kda_fab.can_use_k3_kda_fab(x, w_bfa, w_fb, num_heads)


def _fab(heads: int, rows: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(heads=heads, rows=rows, hidden=HIDDEN))


class K31KdaFabProj(OpWrapper):
    """``(beta, forget_gate)`` of the KDA gate inputs of ``x``.

    ``w_bfa`` is ``[f_a_proj; b_proj]`` (``128 + num_heads`` rows, zero-padded to a multiple of 8, at most 144),
    ``w_fb`` is ``f_b_proj`` (``[N2, 128]``). With ``bfa = bf16(x @ w_bfa^T)``: ``beta = bfa[:, 128:128 + H]`` and
    ``forget_gate = bf16(bfa[:, :128] @ w_fb^T)`` (FP32 accumulation) -- what the ``[f_a | b]`` GEMM, the slice and
    the ``f_b`` GEMM produce. Returns new ``bf16 [M, H]`` and ``bf16 [M, N2]`` tensors.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Kimi K3.1: hidden 7168, f_b out = H * 128 per rank; verify batches of up to 2 requests (1 + 7 rows each).
    # TEP32 (3 heads: 136 [f_a | b] rows), TEP16 (6 heads: 136) and TEP8
    # (12 heads: 144) fit; TEP4's 24 heads (152 rows) do not, so TEP4 keeps the GEMMs.
    CELLS: tuple[Cell, ...] = (
        _fab(3, 8, "TEP32 (ll c1), one request of DFlash2 width 8"),
        _fab(3, 5, "TEP32 (ll c1), one request of DFlash2 width 5"),
        _fab(6, 10, "TEP16 (mid c16), two requests of width 5"),
        _fab(12, 8, "TEP8, one request of 1 + 7 rows"),
        _fab(12, 1, "one row"),
    )

    note = """
    The TEP target computes beta and the forget gate with this kernel only beside its recurrent verify
    (kda_verify_rec): verify batches of at most `_REC_FAB_MAX_REQUESTS` requests, outside torch.compile, and
    where `kda_fab.can_use_k3_kda_fab` holds (x contiguous BF16,
    1 <= M <= 16, 128 < N1 <= 144, N2 % 16 == 0, K % 64 == 0, SM100 / SM103 / SM107, and -- under CUDA graph
    capture -- its per-device workspace created by an eager call first). The products are split over K and
    summed in a fixed order, so the BF16 results can differ from one cuBLAS GEMM in the last bit; launches on
    one device must not overlap (stream order).
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        w_fb: torch.Tensor,
        num_heads: int,
        use_pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``use_pdl=False`` launches without programmatic dependent launch (same bits)."""
        return _kda_fab.k31_kda_fab_proj(x, w_bfa, w_fb, num_heads, use_pdl)

    def reference(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        w_fb: torch.Tensor,
        num_heads: int,
        use_pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two true-FP32 GEMMs, each rounded once to BF16."""
        with true_fp32_matmul():
            bfa = (x.float() @ w_bfa.float().t()).to(torch.bfloat16)
            forget_gate = (bfa[:, :FA_DIM].float() @ w_fb.float().t()).to(torch.bfloat16)
        return bfa[:, FA_DIM : FA_DIM + num_heads].contiguous(), forget_gate

    def is_valid(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        w_fb: torch.Tensor,
        num_heads: int,
        use_pdl: bool = True,
    ) -> None:
        """Nothing to add: the op raises (ValueError) on every operand it would read wrongly -- a dtype other than
        BF16, a strided or misaligned tensor, a shape outside its limits -- which the test drives."""
        return None


k31_kda_fab_proj = K31KdaFabProj()

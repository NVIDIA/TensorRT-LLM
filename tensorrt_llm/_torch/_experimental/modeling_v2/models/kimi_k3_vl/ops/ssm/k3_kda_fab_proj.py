# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA verify's beta and forget gate in one launch (CuTe DSL): ``[f_a | b]`` GEMM, then the
``f_b`` GEMM of its ``f_a`` columns."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import k3_kda_fab

HIDDEN = 7168  # Kimi K3's hidden size, the GEMM's K
FA_DIM = 128  # f_a width

# The verify row counts (requests x (1 + drafts)) of the targets whose KDA head shard the fab
# kernels take (at most 16 heads): (target, heads per rank, rows, what).
FAB_SHAPES = (
    ("helix8", 12, 96, "c240: 16 requests x 6 rows"),
    ("helix8", 12, 64, "c75 / c174: 8 requests x 8 rows"),
    ("helix8", 12, 16, "c24 / c42: 2 requests x 8 rows"),
    ("helix16_moetp4ep4", 6, 8, "ULL c1: one request x 8 rows"),
    ("helix16_moetp4ep4", 6, 32, "ULL c4: 4 requests x 8 rows"),
)


def bfa_rows(heads: int) -> int:
    """Rows of the fused ``[f_a | b]`` weight: ``128 + heads`` padded to a multiple of 8."""
    return -(-(FA_DIM + heads) // 8) * 8


def fab_cells(why_suffix: str) -> tuple[Cell, ...]:
    """One cell per verify shape of :data:`FAB_SHAPES`."""
    return tuple(
        Cell(
            why=f"{target}, {heads} heads, {rows} rows ({what}): {why_suffix}",
            spec=dict(heads=heads, rows=rows),
        )
        for target, heads, rows, what in FAB_SHAPES
    )


class K3KdaFabProj(OpWrapper):
    """Return ``(beta, forget_gate)``: ``bfa = bf16(x @ w_bfa^T)``, ``beta = bfa[:, 128 : 128 + H]``
    (bf16 ``[M, H]``) and ``forget_gate = bf16(bfa[:, :128] @ w_fb^T)`` (bf16 ``[M, N2]``), both
    GEMMs accumulated in fp32."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = fab_cells("beta and the forget gate for the BF16 verify core")
    note = """
    Kimi K3's KDA verify calls it on the projection aux stream next to the QKVG GEMM, for
    M <= 256 rows where can_use_k3_kda_fab accepts the operands (contiguous bf16 x [M, 7168],
    w_bfa = [f_a_proj; b_proj] padded to a multiple of 8 rows, at most 144; w_fb = f_b_proj
    [H * 128, 128]; at most 16 heads). Outputs are new contiguous tensors.

    The CTAs hand their partial sums over through a per-device workspace and ticket counter
    (process-wide, created on the first call outside graph capture; can_use_k3_kda_fab rejects a
    capturing caller until then). Launches sharing a device must not overlap: stream order, and
    with PDL the ticket and the workspace refill come after griddepcontrol.wait. Compiled kernels
    are cached per process by tiling and shape. The launch uses PDL. Importing the entry needs
    CuTe DSL (SM100 / SM103 / SM107).
    """

    def __call__(
        self, x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return k3_kda_fab.k3_kda_fab_proj(x, w_bfa, w_fb, num_heads)

    def reference(
        self, x: torch.Tensor, w_bfa: torch.Tensor, w_fb: torch.Tensor, num_heads: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bfa = (x.double() @ w_bfa.double().t()).to(torch.bfloat16)
        gate = (bfa[:, :FA_DIM].double() @ w_fb.double().t()).to(torch.bfloat16)
        return bfa[:, FA_DIM : FA_DIM + num_heads].contiguous(), gate


k3_kda_fab_proj = K3KdaFabProj()

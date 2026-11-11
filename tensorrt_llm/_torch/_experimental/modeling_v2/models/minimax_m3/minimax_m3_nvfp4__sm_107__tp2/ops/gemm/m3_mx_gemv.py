# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode MXFP8 x MXFP8 GEMV (tcgen05, block-scaled, CuTe DSL) on an activation its producer already quantized."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..moe.m3_moe_engine import mxfp8_dequant
from ..mx_gemv import op as _gemv

__all__ = ["PROJECTIONS", "gemv_is_valid", "m3_mx_gemv", "served_cells", "served_max_rows"]

# (N, K) of this target's MXFP8 projections at tp2, the GEMV's tuned shapes.
QKV_SPARSE = (4992, 6144)  # the fused QKV + index projection (index values disabled)
QKV = (4608, 6144)
O_PROJ = (6144, 4096)
SHARED_GATE_UP = (3072, 6144)
SHARED_DOWN = (6144, 1536)
DENSE_GATE_UP = (12288, 6144)
DENSE_DOWN = (6144, 6144)
PROJECTIONS = {
    "sparse-layer QKV": QKV_SPARSE,
    "dense-attention QKV": QKV,
    "o_proj": O_PROJ,
    "shared gate_up": SHARED_GATE_UP,
    "shared down": SHARED_DOWN,
    "dense gate_up": DENSE_GATE_UP,
    "dense down": DENSE_DOWN,
}


def served_cells(named: tuple[Cell, ...], x_bf16: bool, max_m: int) -> tuple[Cell, ...]:
    """A cell for every (projection, rows) the GEMV serves up to ``max_m`` rows that ``named`` does not list.

    Certification is per projection: a projection's split-K, ring and MMA width differ, so each (N, K, M) the
    decode dispatch can send here is its own cell.
    """
    listed = {(cell.spec["nk"], cell.spec["m"]) for cell in named}
    return tuple(
        Cell(why=f"{name}, {m} rows", spec=dict(nk=nk, m=m))
        for name, nk in PROJECTIONS.items()
        for m in range(1, max_m + 1)
        if (nk, m) not in listed and _gemv.serves(nk[0], nk[1], m, x_bf16)
    )


def served_max_rows(nk: tuple[int, int], x_bf16: bool, lo: int, hi: int) -> int:
    """The largest ``m`` in ``lo..hi`` such that the GEMV serves an (N, K) projection at every row count ``lo..m``;
    ``lo - 1`` when it declines ``lo``. The decode dispatch raises a projection's rows past the batch-1/2 step only
    as far as this (the bf16 GEMV's in-kernel quantize declines the K 6144, split 2 projections at MMA width 16)."""
    m = lo - 1
    while m < hi and _gemv.serves(nk[0], nk[1], m + 1, x_bf16):
        m += 1
    return m


def gemv_reference(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """bf16(x @ dequant(weight)^T) for fp32 activation values x [M, K], summed in fp64 and rounded once."""
    return (x.double() @ mxfp8_dequant(weight, weight_scale).double().t()).to(torch.bfloat16)


def gemv_is_valid(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    n, k = weight.shape
    # The kernel reads both from data_ptr() as dense buffers at the layout it was compiled for.
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous(), (
        "weight must be contiguous e4m3 [N, K]"
    )
    assert weight_scale.numel() == n * k // 32, (
        "weight_scale must be the 128x4-swizzled UE8M0 buffer [N * K / 32]"
    )


class M3MxGemv(OpWrapper):
    """bf16 [M, N] = xq [M, K] (e4m3, 128x4-swizzled UE8M0 scales ``xsf``) x weight [N, K]^T (e4m3, swizzled
    ``weight_scale``), M <= 48; None when the shape is not served (the caller falls back).

    ``trigger_early``: release the PDL dependents before the epilogue. ``cfg``: a (split, ring[, contig, nacc, a_tmem,
    st_async]) tuning override; default the table's. ``use_pdl``: default the environment's. ``ar`` = (multicast
    workspace address, flags address, rank, world, bucket tokens, mode): also push the rows into the next boundary
    all-reduce's exchange (then ``op.last_pushed`` is True).
    """

    ARCHS = frozenset({Arch.SM_107})

    # The projections' (N, K) at the token counts each build differs in: the MMA width (8 columns up to the TP2 decode
    # step's 8 rows) is one, so one cell per projection pins the shape, and the rest are every served row count.
    CELLS: tuple[Cell, ...] = (
        Cell(why="sparse-layer QKV, one token (split 4)", spec=dict(nk=QKV_SPARSE, m=1)),
        Cell(
            why="sparse-layer QKV, 8 rows: two requests verifying 1 + 3",
            spec=dict(nk=QKV_SPARSE, m=8),
        ),
        Cell(
            why="dense-attention QKV, 4 rows: one request verifying 1 + 3", spec=dict(nk=QKV, m=4)
        ),
        Cell(why="o_proj, one token (split 2)", spec=dict(nk=O_PROJ, m=1)),
        Cell(why="o_proj, 8 rows", spec=dict(nk=O_PROJ, m=8)),
        Cell(why="shared gate_up, 4 rows (split 4)", spec=dict(nk=SHARED_GATE_UP, m=4)),
        Cell(why="shared down, one token (split 2, K 1536)", spec=dict(nk=SHARED_DOWN, m=1)),
        Cell(why="dense gate_up, 8 rows (N 12288)", spec=dict(nk=DENSE_GATE_UP, m=8)),
        Cell(why="dense down, 4 rows (K 6144)", spec=dict(nk=DENSE_DOWN, m=4)),
    )
    # Every other (projection, rows) the decode dispatch can send here (modeling's MX_GEMV_ROWS).
    CELLS = CELLS + served_cells(CELLS, x_bf16=False, max_m=48)

    note = """
    The kernel sums in fp32 in split-K order where `reference` sums in fp64: the bf16 outputs agree within the band the
    test derives (SM107: at most as many misrounded elements as the stock FlashInfer CuTe DSL GEMM on the same
    operands, rand / wide-scale / cancellation / tie / zero inputs). Not covered: the all-reduce push (`ar`), which
    needs a multicast exchange; `cfg` overrides beyond the table (the target passes none). Shapes outside the table
    and M > 48 return None; the target sends at most 8 rows.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> Optional[torch.Tensor]:
        return _gemv.m3_mx_gemv(xq, xsf, weight, weight_scale, trigger_early, cfg, use_pdl, ar)

    def reference(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> torch.Tensor:
        """The dequantized product; the launch options are ignored."""
        return gemv_reference(mxfp8_dequant(xq, xsf), weight, weight_scale)

    def is_valid(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> None:
        m, k = xq.shape
        assert xq.dtype == torch.float8_e4m3fn and xq.is_contiguous(), (
            "xq must be contiguous e4m3 [M, K]"
        )
        # A linear-layout [M, K / 32] scale buffer is read as the swizzled one: wrong scales, no error.
        assert xsf.numel() >= (m + 127) // 128 * 128 * (k // 32), (
            "xsf must be the 128x4-swizzled scale buffer (rows padded to 128), not the linear one"
        )
        gemv_is_valid(weight, weight_scale)


m3_mx_gemv = M3MxGemv()

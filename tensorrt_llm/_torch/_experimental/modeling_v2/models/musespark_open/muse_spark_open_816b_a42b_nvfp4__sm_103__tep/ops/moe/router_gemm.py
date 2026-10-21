# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE router logits of a call of 9..512 rows, as a BF16x3 split-K tensor-core GEMM.

The router is an FP32 ``Linear(8192, 256)`` replicated on every rank; its logits
are scaled by the metap multiplier. Above 8 rows the built-in
``MuseSparkOpenMoE.router_logits`` runs ``_router_tensorcore_kernel`` at 16..128 rows
(16-row tiles on TF32x3 ``mma.sync`` instructions, the weight re-read for every 16
rows, 16 K-splits) and the FP32 ``Linear`` above 128 rows, which this stack's torch
runs in plain TF32 (every weight rounded to 11 bits).

This entry runs the tree's CuTe BF16x3 router GEMM
(``cute_dsl_kernels/blackwell/bf16x3_router_gemm.Sm100BF16x3RouterGemm``, the kernel
behind the built-in ``MUSESPARK_OPEN_BF16X3_ROUTER`` option) and sums its split-K
partials with the multiplier folded in. The GEMM computes ``W X^T`` on tcgen05: the
256 experts are the MMA's M dimension (two tiles of 128), the call's rows its N
dimension (one tile of the next power of two of up to 128 rows, then 128-row
tiles), so the weight is streamed once per 128 rows. A call of ``t`` row tiles runs
``splits(rows) = CTAS // (2 * t)`` K splits -- 76, 38, 25, 19 -- so that its
``2 * t * splits`` CTAs fill the GPU's SMs in one wave; each CTA owns every
``splits``-th K tile of 64. It splits each FP32 weight it loads into three BF16
terms, ``w = w0 + w1 + w2`` -- exact for every FP32 value of normal magnitude --
and a BF16 row times a BF16 term is exact in FP32, so the only rounding is the FP32
accumulation on the tensor cores (``w0`` products into one accumulator, ``w1`` and
``w2`` into a second, added at the end of the CTA). It stores its ``[rows, 128]``
partial; the sum kernel loads every split of a block of logits at once, adds them
as a tree (``tl.sum`` over the split axis) and applies the multiplier in FP32.

Values. Not the built-in kernels' bits: a budgeted entry. A logit's arithmetic
depends on its row, the weight and the call's number of 128-row tiles only, so a
row's logits do not depend on the call's other rows, and a call is deterministic.
Its error against the exact product is the FP32 accumulation's alone, and grows
with the K tiles each split accumulates (2 at 76 splits, 7 at 19); the built-in
TF32x3 kernel also rounds every weight to 22 bits, the TF32 ``Linear`` to 11.

Programmatic dependent launch: the GEMM waits (``griddepcontrol.wait``) before its
first load and releases its dependents after its last MMA; the sum kernel waits
before reading the partials and releases its dependents.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's router: 256 experts over the 8192-wide residual.
EXPERTS = 256
HIDDEN = 8192
#: The row counts this entry takes: every decode step above one request's 8 rows (its
#: last accepted token and 7 drafts, the router shard's) up to 64 requests.
MIN_ROWS = 9
MAX_ROWS = 512
#: The GEMM's CTAs per call: one per SM of a GB300 (152), so one wave.
CTAS = 152
#: The GEMM's expert tiles (M) and row tiles (N) of 128.
EXPERT_TILES = EXPERTS // 128
ROW_TILE = 128
#: Partials per program of the sum kernel (splits, padded to a power of two, x logits).
SUM_TILE = 4096


def row_tile(rows: int) -> int:
    """The GEMM's N tile for a call of ``rows`` rows: the next power of two, 8..128."""
    return min(max(triton.next_power_of_2(rows), 8), ROW_TILE)


def splits(rows: int) -> int:
    """The K splits of a call of ``rows`` rows: 76, 38, 25, 19 for 1, 2, 3, 4 row tiles."""
    return CTAS // (EXPERT_TILES * triton.cdiv(rows, ROW_TILE))


def _gemm(tile: int):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.bf16x3_router_gemm import (
        Sm100BF16x3RouterGemm,
    )

    return Sm100BF16x3RouterGemm.compile(tile, HIDDEN)


@triton.jit
def _router_gemm_sum_kernel(
    partials,
    output,
    scale,
    size,
    SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS_POW2)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    values = tl.load(
        partials + split[:, None] * size + offsets[None, :],
        mask=(split[:, None] < SPLITS) & (offsets[None, :] < size),
        other=0.0,
    )
    tl.store(output + offsets, tl.sum(values, axis=0) * scale, mask=offsets < size)


class RouterGemm(OpWrapper):
    """``(hidden_states.float() @ weight.T) * scale`` for ``MIN_ROWS..MAX_ROWS`` rows.

    Args:
        hidden_states: ``[M, 8192]`` BF16, contiguous, ``MIN_ROWS <= M <= MAX_ROWS``.
        weight: the router weight ``[256, 8192]`` FP32, contiguous, read on every call.
        scale: the metap multiplier, a Python float (applied in FP32 after the sum).

    Returns:
        ``[M, 256]`` FP32 logits, newly allocated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the smallest call taken: 9 rows, a 16-row tile with 7 masked rows",
            spec=dict(rows=9),
        ),
        Cell(why="two requests at K=7: 16 rows", spec=dict(rows=16)),
        Cell(why="four requests: 32 rows", spec=dict(rows=32)),
        Cell(
            why="a row count off the power-of-two tiles: 40 rows in a 64-row tile",
            spec=dict(rows=40),
        ),
        Cell(why="sixteen requests: 128 rows, one full tile, 76 splits", spec=dict(rows=128)),
        Cell(
            why="seventeen requests: 136 rows, a second, nearly empty row tile, 38 splits",
            spec=dict(rows=136),
        ),
        Cell(why="thirty-two requests: 256 rows, two full tiles", spec=dict(rows=256)),
        Cell(why="a third row tile: 300 rows, 25 splits", spec=dict(rows=300)),
        Cell(
            why="sixty-four requests: 512 rows, the largest call taken, 19 splits",
            spec=dict(rows=512),
        ),
    )

    note = """
    Budgeted, not bitwise: the test gates every cell against the float64 product
    within a band that grows with the K tiles each split accumulates, checks that it
    is no less accurate than the built-in router at the row counts the built-in
    serves, that a row's logits do not depend on the call's other rows, that
    repeated calls give the same bits, and that a CUDA graph replays changed inputs
    and changed weights.
    """

    def __call__(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, scale: float
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        split_count = splits(rows)
        partials = torch.empty(
            (split_count, rows, EXPERTS), device=hidden_states.device, dtype=torch.float32
        )
        _gemm(row_tile(rows))(hidden_states, weight, partials, split_count)
        output = torch.empty((rows, EXPERTS), device=hidden_states.device, dtype=torch.float32)
        size = rows * EXPERTS
        splits_pow2 = triton.next_power_of_2(split_count)
        block = SUM_TILE // splits_pow2
        _router_gemm_sum_kernel[(triton.cdiv(size, block),)](
            partials,
            output,
            scale,
            size,
            SPLITS=split_count,
            SPLITS_POW2=splits_pow2,
            BLOCK=block,
            num_warps=4,
            launch_pdl=True,
        )
        return output

    def reference(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Float64 product, rounded to FP32, the FP32 multiplier applied after."""
        logits = (hidden_states.double() @ weight.double().t()).float()
        return logits * torch.tensor(scale, dtype=torch.float32, device=logits.device)

    def is_valid(self, hidden_states: torch.Tensor, weight: torch.Tensor, scale: float) -> None:
        """The GEMM reads both operands through dense row-major TMA descriptors."""
        if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
            raise ValueError(f"hidden_states must be 2-D BF16, got {hidden_states.dtype}")
        if not hidden_states.is_contiguous() or hidden_states.shape[1] != HIDDEN:
            raise ValueError(f"hidden_states must be contiguous [M, {HIDDEN}]")
        if not MIN_ROWS <= hidden_states.shape[0] <= MAX_ROWS:
            raise ValueError(f"{MIN_ROWS}..{MAX_ROWS} rows, got {hidden_states.shape[0]}")
        if weight.dtype != torch.float32 or tuple(weight.shape) != (EXPERTS, HIDDEN):
            raise ValueError(f"weight must be FP32 [{EXPERTS}, {HIDDEN}]")
        if not weight.is_contiguous() or weight.device != hidden_states.device:
            raise ValueError("weight must be contiguous, on the input's device")


router_gemm = RouterGemm()

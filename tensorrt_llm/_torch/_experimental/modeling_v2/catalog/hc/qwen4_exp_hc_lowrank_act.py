# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer activations between its two projections: the SiLU of
the rank-`R` bottleneck and the block-injection weights, from the first
projection's fp32 logits, in one Triton row kernel.

Shared by the qwen4_exp tp1 target
(`qwen3_8_flash_next_nvfp4__sm_103__tp1`) and the unified tp4/tep4/dep4 target
(`hf_fc694b5_orig__sm_103__unified`); promoted from their copies. The Triton kernel below was written for the
tp1 target.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper


@triton.jit
def _hc_lowrank_act_kernel(
    logits_ptr,
    low_ptr,
    inject_ptr,
    stride_logits,
    stride_low,
    stride_inject,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    WITH_INJECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    z = tl.load(logits_ptr + row * stride_logits + k, mask=k < RANK, other=0.0).to(tl.float32)
    z = z * (1.0 / HC)
    tl.store(
        low_ptr + row * stride_low + k,
        (z * tl.sigmoid(z)).to(low_ptr.dtype.element_ty),
        mask=k < RANK,
    )
    if WITH_INJECT:
        i = tl.arange(0, HC)
        zi = tl.load(logits_ptr + row * stride_logits + RANK + i).to(tl.float32)
        tl.store(
            inject_ptr + row * stride_inject + i,
            (2.0 * tl.sigmoid(zi * (1.0 / HC))).to(inject_ptr.dtype.element_ty),
        )


class Qwen4ExpHcLowrankAct(OpWrapper):
    """With `R = rank`, `C = hc_count` and `logits` the fp32 `[T, N]` first
    projection `normed @ [down; block_inject; 0]^T` (columns `[0, R)` down,
    `[R, R + C)` injection when `with_inject`, the rest padding)::

        low = silu(logits[:, :R] / C)  # [T, R], `dtype`
        inject = 2 * sigmoid(logits[:, R : R + C] / C)  # [T, C], `dtype`, only with_inject

    fp32 inside, one rounding per output. Returns fresh `(low, inject)`;
    `inject` is `None` exactly when `with_inject` is False.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True), hc_count (4), rank (320), logits_width (384), dtype (bf16).
    # The unified target's HC mixers run this form at every row count, on the fresh
    # fp32 `cublas_mm` output over the first `hc_first_cols` = ceil((320 + 4) / 32) * 32
    # = 352 columns of the first projection: hence the 352-wide cells.
    CELLS: tuple[Cell, ...] = (
        Cell(why="checkpoint rank 320 + 4, padded to 384, 128 rows", spec=dict(rows=128)),
        Cell(why="checkpoint rank 320 + 4, padded to 384, prefill 8192", spec=dict(rows=8192)),
        Cell(why="stack-final mixer: no inject, 8192 rows", spec=dict(rows=8192, inject=False)),
        Cell(
            why="unified decode HC mixer: one token, 352-wide logits",
            spec=dict(rows=1, logits_width=352),
        ),
        Cell(
            why="unified dep4 rank's largest captured decode batch: 32 rows",
            spec=dict(rows=32, logits_width=352),
        ),
        Cell(
            why="unified tep4 MTP-7 verify: 128 requests x 8 rows = 1024",
            spec=dict(rows=1024, logits_width=352),
        ),
        Cell(
            why="unified prefill chunk at max_num_tokens: 8192 rows, 352-wide",
            spec=dict(rows=8192, logits_width=352),
        ),
        Cell(
            why="unified stack-final mixer (no inject): one decode token",
            spec=dict(rows=1, inject=False, logits_width=352),
        ),
    )

    note = """
    Metadata consumed: none. Call sites: every hyper-connection mixer, between
    the first projection (`cublas_mm`, fp32 out, 352 columns) and the up
    projection. HC is replicated, so the rows are this rank's tokens at every
    member: decode 1..128 (tp4 / tep4) or 1..32 (a dep4 rank), MTP verify up to
    128 x 8 = 1024 rows (tep4, draft 7), prefill one chunk of up to 8192 rows
    (`max_num_tokens`). Preconditions: `logits` fp32, unit column stride, at
    least `rank (+ hc_count)` columns; `hc_count` a power of two.
    Graph-capturable.

    Source: a Triton kernel JIT-compiled at first call for the running device;
    the cells ran that JIT path on GB300 (sm_103). One program per row with an
    int64 row offset: 8192 x 352 is 2.9e6 elements, and 2^31 elements is
    2^31 / 352 = 6.1e6 rows, far past the envelope.

    Certified: bf16 outputs from fp32 logits 352 wide (the call site) at 1, 32,
    1024 and 8192 rows with the injection and at 1 row without it, and 384 wide
    at 128 and 8192 rows (without the injection at 8192); graph replay.
    Measured against the float64 reference (logits ~ 20 * randn): bitwise at 1,
    32 and 128 rows; at 1024 and 8192 rows at most 0.99 ulp, on 5 of 327680 and
    52 of 2621440 `low` elements, cosine 1.0, relative L2 <= 1.6e-5.

    Not certified: zero rows; rows past 8192; hc_count other than 4; rank
    other than 320; float16 outputs. The kernel has no row-count switch, so row
    counts between the cells take the same path, and the logits width enters
    only as the row stride (128 rows is driven at 384 wide only).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rows = logits.shape[0]
        low = torch.empty((rows, rank), dtype=dtype, device=logits.device)
        inject = (
            torch.empty((rows, hc_count), dtype=dtype, device=logits.device)
            if with_inject
            else None
        )
        if rows > 0:
            _hc_lowrank_act_kernel[(rows,)](
                logits,
                low,
                inject if with_inject else low,
                logits.stride(0),
                low.stride(0),
                inject.stride(0) if with_inject else 0,
                RANK=rank,
                HC=hc_count,
                WITH_INJECT=with_inject,
                BLOCK=triton.next_power_of_2(rank),
                num_warps=4,
            )
        return low, inject

    def reference(
        self,
        logits: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """float64, one rounding per output."""
        lg = logits.double()
        low = F.silu(lg[:, :rank] / hc_count).to(dtype)
        inject = None
        if with_inject:
            inject = (2.0 * torch.sigmoid(lg[:, rank : rank + hc_count] / hc_count)).to(dtype)
        return low, inject

    def is_valid(
        self,
        logits: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        dtype: torch.dtype,
    ) -> None:
        assert logits.dtype == torch.float32 and logits.stride(1) == 1, "fp32 logits, unit stride"
        assert logits.shape[1] >= rank + (hc_count if with_inject else 0), "logits too narrow"
        assert hc_count & (hc_count - 1) == 0, "hc_count must be a power of two"


qwen4_exp_hc_lowrank_act = Qwen4ExpHcLowrankAct()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer activations between its two projections: the SiLU of
the rank-`R` bottleneck and the block-injection weights, from the first
projection's fp32 logits, in one Triton row kernel.

Called by the qwen4_exp target (`qwen3_8_flash_next_nvfp4__sm_103__default`) at every
layout.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


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

    # spec: rows, inject (True), hc_count (4), rank (320), logits_width (384), dtype (bf16),
    # capture (False: eager; True: replayed from a captured CUDA graph over two steps' logits).
    # The default target's HC mixers run this form on the fresh fp32 `cublas_mm` output
    # over the first `hc_first_cols` = ceil((320 + 4) / 32) * 32 = 352 columns of the
    # first projection: hence the 352-wide cells.
    CELLS: tuple[Cell, ...] = (
        Cell(why="checkpoint rank 320 + 4, padded to 384, 128 rows", spec=dict(rows=128)),
        Cell(why="checkpoint rank 320 + 4, padded to 384, prefill 8192", spec=dict(rows=8192)),
        Cell(why="stack-final mixer: no inject, 8192 rows", spec=dict(rows=8192, inject=False)),
        Cell(
            why="default decode HC mixer: one token, 352-wide logits",
            spec=dict(rows=1, logits_width=352),
        ),
        Cell(
            why="default dep4 rank's largest captured decode batch: 32 rows",
            spec=dict(rows=32, logits_width=352),
        ),
        Cell(
            why="default tep4 MTP-7 verify: 128 requests x 8 rows = 1024",
            spec=dict(rows=1024, logits_width=352),
        ),
        Cell(
            why="default prefill chunk at max_num_tokens: 8192 rows, 352-wide",
            spec=dict(rows=8192, logits_width=352),
        ),
        Cell(
            why="default stack-final mixer (no inject): one decode token",
            spec=dict(rows=1, inject=False, logits_width=352),
        ),
        # The FP8 unified target's HC mixers in its captured graphs at 641..3071 rows (e.g. the
        # breakable prefill buckets): fewer rows run `qwen4_exp_hc_decode_tc_first`, more the
        # rstd form.
        Cell(
            why="FP8 unified captured prefill HC mixer, 641 rows: 352-wide logits",
            spec=dict(rows=641, logits_width=352, capture=True),
        ),
        Cell(
            why="FP8 unified captured prefill HC mixer, 1024 rows: 352-wide logits",
            spec=dict(rows=1024, logits_width=352, capture=True),
        ),
        Cell(
            why="FP8 unified captured prefill HC mixer, 3071 rows (the largest): 352-wide",
            spec=dict(rows=3071, logits_width=352, capture=True),
        ),
    )

    note = """
    Metadata consumed: none. Call sites: `_hc_first`, every hyper-connection
    mixer of a step past `_HC_TC_FIRST_MAX_ROWS` (256) rows -- smaller steps
    run `qwen4_exp_hc_decode_tc_first` -- between the first projection
    (`cublas_mm`, fp32 out, 352 columns) and the up projection. HC is
    replicated, so the rows are this rank's tokens at every member: MTP verify
    up to 128 x 8 = 1024 rows (tep4, draft 7) and prefill chunks of up to 8192
    rows (`max_num_tokens`); at one rank, prefill steps from
    `_HC_RSTD_MIN_ROWS` rows take `qwen4_exp_hc_first_rescale_tc` instead.
    Preconditions: `logits` fp32, unit column stride, at least
    `rank (+ hc_count)` columns; `hc_count` a power of two. Graph-capturable.

    CUDA-graph capture (the `capture` cells): the replay is bitwise the eager
    call, and a second step's logits replay bitwise their own eager call.

    Source: a Triton kernel JIT-compiled at first call for the running device;
    the cells ran on sm_103. One program per row with an int64 row offset;
    2^31 elements is 6.1e6 rows of 352, far past the envelope.

    Certified: bf16 outputs from fp32 logits 352 wide (the call site) at 1, 32,
    1024 and 8192 rows with the injection and at 1 row without it, and 384 wide
    at 128 and 8192 rows (without the injection at 8192); graph replay, and
    captured at 641, 1024 and 3071 rows 352 wide.
    Against the float64 reference: bitwise at 1, 32 and 128 rows; at 1024 and
    8192 rows at most 0.99 ulp on a few `low` elements, relative L2 <= 1.6e-5.

    Not certified: zero rows; rows past 8192; hc_count other than 4; rank
    other than 320; float16 outputs. The kernel has no row-count switch, so row
    counts between the cells take the same path, and the logits width enters
    only as the row stride. Not driven under capture: the form without the
    injection, and the other captured batches.
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

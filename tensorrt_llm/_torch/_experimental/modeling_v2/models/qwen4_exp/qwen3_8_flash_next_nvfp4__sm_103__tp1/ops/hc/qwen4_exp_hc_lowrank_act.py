# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer activations between its two projections: the SiLU of
the rank-`R` bottleneck and the block-injection weights, from the first
projection's fp32 logits, in one Triton row kernel.

Tier 5 (Triton written for this target). Used where the first projection runs on
cuBLAS (prefill, and decode batches past `qwen4_exp_hc_decode_tc_first`'s rows),
between `cublas_mm` (first projection) and the fused up projection
`qwen4_exp_hc_up_gate_mean` (or its NVFP4 variant).
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

    # spec: rows, inject (True), hc_count (4), rank (320), logits_width (384), dtype (bf16).
    CELLS: tuple[Cell, ...] = (
        Cell(why="checkpoint rank 320 + 4, padded to 384, 128 rows", spec=dict(rows=128)),
        Cell(why="checkpoint rank 320 + 4, padded to 384, prefill 8192", spec=dict(rows=8192)),
        Cell(why="stack-final mixer: no inject, 8192 rows", spec=dict(rows=8192, inject=False)),
    )

    note = """
    Metadata consumed: none. Call sites: the trunk's hyper-connection sites in
    `PrefillTarget` (all 97), between the first projection and the up
    projection. Preconditions: `logits` fp32, unit column stride, at least
    `rank (+ hc_count)` columns; `hc_count` a power of two. Graph-capturable.

    Certified: bf16 outputs from 384-wide fp32 logits, with the injection at 128
    and 8192 rows, without it at 8192 rows; graph replay.

    Not certified: zero rows, hc_count other than 4, rank other than 320,
    float16 outputs.
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

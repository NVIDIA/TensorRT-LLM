# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill router logits for MiniMax-M3 without materializing the replicated router input.

``MiniMaxM3Gate`` computes FP32-grade logits from BF16 activations as one tensor-core GEMM,
``cat([x] * terms, -1) @ split_weight.t()``, where ``split_weight`` holds the BF16 terms of the
FP32 router weight side by side along K. For prefill-sized inputs the ``cat`` writes ``terms``
copies of the activations and costs more than the GEMM itself.

``router_split_logits`` computes the same product reading ``x`` once per term from its original
buffer. It walks K in the order the concatenated operand presents it (term 0 over the hidden
dimension, then term 1, then term 2) and chains every BF16 tensor-core product into one FP32
accumulator. When cuBLAS also runs that GEMM as one accumulation chain over K (no split-K), both
round identically and the logits are bitwise equal; that holds on the row counts in
``ROUTER_TRITON_ROWS``, which were checked on SM103 against ``torch.ops.trtllm.cublas_mm``. Outside
that range cuBLAS may choose a split-K kernel, so the stock path runs there.

Switch: ``M3_ROUTER_TRITON_ROWS`` = ``"lo-hi[,lo-hi...]"`` row ranges (default below), or one of
"0", "off", "false", "no" to disable.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

_OFF = ("0", "off", "false", "no", "")

DEFAULT_ROUTER_TRITON_ROWS = "8625-16384"
"""str: Rows where the Triton router matched the cuBLAS router GEMM bit for bit on SM103 (hidden 6144, 128 experts,
every row count checked). Below 8625 cuBLAS alternates between kernels whose K order differs."""

_SUPPORTED_SM = (103,)


def _parse_rows(text: str) -> Tuple[Tuple[int, int], ...]:
    text = text.strip().lower()
    if text in _OFF:
        return ()
    ranges = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        lo_i = int(lo)
        hi_i = int(hi) if hi else lo_i
        if lo_i < 1 or hi_i < lo_i:
            raise ValueError(f"M3_ROUTER_TRITON_ROWS: bad range {part!r}")
        ranges.append((lo_i, hi_i))
    return tuple(ranges)


ROUTER_TRITON_ROWS = _parse_rows(
    os.environ.get("M3_ROUTER_TRITON_ROWS", DEFAULT_ROUTER_TRITON_ROWS)
)

_BM, _BK, _NUM_WARPS, _NUM_STAGES = 128, 128, 8, 3


@triton.jit
def _router_split_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    stride_xm,
    stride_wn,
    stride_om,
    HIDDEN: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TERMS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, NUM_EXPERTS)
    offs_k = tl.arange(0, BK)
    row_mask = offs_m[:, None] < num_rows
    acc = tl.zeros((BM, NUM_EXPERTS), dtype=tl.float32)
    # K order of cat([x] * TERMS, -1): term 0 over the hidden dimension, then term 1, ...
    for t in tl.static_range(TERMS):
        for k0 in range(0, HIDDEN, BK):
            a = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
                mask=row_mask,
                other=0.0,
            )
            b = tl.load(w_ptr + offs_n[None, :] * stride_wn + (t * HIDDEN + k0 + offs_k)[:, None])
            acc = tl.dot(a, b, acc)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc, mask=row_mask)


def router_split_applies(x: torch.Tensor, split_weight: torch.Tensor, hidden_size: int) -> bool:
    """Whether ``router_split_logits`` reproduces the stock router GEMM for this input."""
    if not ROUTER_TRITON_ROWS or x.dim() != 2:
        return False
    rows = x.shape[0]
    if not any(lo <= rows <= hi for lo, hi in ROUTER_TRITON_ROWS):
        return False
    num_experts = split_weight.shape[0]
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and split_weight.dtype == torch.bfloat16
        and x.shape[1] == hidden_size
        and x.stride(1) == 1
        and split_weight.is_contiguous()
        and split_weight.shape[1] % hidden_size == 0
        and hidden_size % _BK == 0
        and num_experts in (64, 128)
        and get_sm_version() in _SUPPORTED_SM
    )


def router_split_logits(
    x: torch.Tensor, split_weight: torch.Tensor, hidden_size: int
) -> torch.Tensor:
    """FP32 ``cat([x] * terms, -1) @ split_weight.t()`` without the ``cat``; see the module docstring."""
    rows = x.shape[0]
    num_experts = split_weight.shape[0]
    out = torch.empty((rows, num_experts), dtype=torch.float32, device=x.device)
    _router_split_kernel[(triton.cdiv(rows, _BM),)](
        x,
        split_weight,
        out,
        rows,
        x.stride(0),
        split_weight.stride(0),
        out.stride(0),
        HIDDEN=hidden_size,
        NUM_EXPERTS=num_experts,
        TERMS=split_weight.shape[1] // hidden_size,
        BM=_BM,
        BK=_BK,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out

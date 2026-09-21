# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-batch router GEMM with FP32 weights, accumulation and output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_mv(
    X, W, Y, SX: tl.constexpr, E: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0)
    expert = tl.program_id(1) * BN + tl.arange(0, BN)
    channel = tl.arange(0, BK)
    x = tl.load(X + row * SX + channel, channel < K, 0).to(tl.float32)
    w = tl.load(
        W + expert[:, None] * K + channel[None, :],
        (expert[:, None] < E) & (channel[None, :] < K),
        0,
    )
    out = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + row * E + expert, out, expert < E)


def fp32_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 router logits for small BF16 batches, preserving FP32 weights.

    x is [tokens, hidden] with contiguous hidden rows; weight is contiguous
    [experts, hidden]. The caller selects this path for 1-8 tokens, 288 experts
    and hidden size 4096; larger batches use the ordinary GEMM.
    """
    rows = x.shape[0]
    if rows == 1:
        bn, warps = 1, 4
    elif rows == 2:
        bn, warps = 1, 8
    elif rows <= 4:
        bn, warps = 2, 8
    else:
        bn, warps = 2, 4
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.float32
    assert x.stride(-1) == 1 and weight.is_contiguous()
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    _router_mv[(x.shape[0], triton.cdiv(weight.shape[0], bn))](
        x,
        weight,
        output,
        x.stride(0),
        weight.shape[0],
        weight.shape[1],
        bn,
        triton.next_power_of_2(weight.shape[1]),
        num_warps=warps,
    )
    return output

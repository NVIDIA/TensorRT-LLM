# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 MoE router logits with a GEMV for few tokens."""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from tensorrt_llm._utils import get_sm_version

from ..flashinfer_utils import get_env_enable_pdl

# Past this many tokens cuBLAS is faster (128 experts x 6144 hidden on Blackwell).
MAX_GEMV_TOKENS = 16


@triton.jit
def _fp32_router_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_tokens,
    hidden_size,
    x_stride,
    w_stride,
    out_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    expert = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    for k0 in range(0, hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_size
        w = tl.load(w_ptr + expert * w_stride + offs_k, mask=k_mask, other=0.0)
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # FP32 FMAs, not tl.dot: tensor cores would round the weight to TF32.
        acc += tl.sum(x.to(tl.float32) * w[None, :], axis=1)

    tl.store(out_ptr + offs_m * out_stride + expert, acc, mask=m_mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@torch.library.custom_op("trtllm::fp32_router_gemm", mutates_args=())
def fp32_router_gemm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(hidden_states.float(), weight)`` for an FP32 ``[num_experts, hidden]`` weight."""
    num_tokens, hidden_size = hidden_states.shape
    if not (
        0 < num_tokens <= MAX_GEMV_TOKENS
        and hidden_size == weight.shape[1]
        and hidden_states.stride(1) == 1
        and weight.stride(1) == 1
    ):
        return torch.nn.functional.linear(hidden_states.to(torch.float32), weight)

    num_experts = weight.shape[0]
    out = hidden_states.new_empty((num_tokens, num_experts), dtype=torch.float32)
    block_m = triton.next_power_of_2(num_tokens)
    # Keep about 32K activation elements in flight per CTA, the fastest in a B200 sweep.
    block_k = min(triton.next_power_of_2(hidden_size), max(128, 32768 // block_m))
    launch_with_pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _fp32_router_gemv_kernel[(num_experts,)](
        hidden_states,
        weight,
        out,
        num_tokens,
        hidden_size,
        hidden_states.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        LAUNCH_WITH_PDL=launch_with_pdl,
        num_warps=8 if block_k >= 4096 else 4,
        launch_pdl=launch_with_pdl,
    )
    return out


@fp32_router_gemm.register_fake
def _(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]), dtype=torch.float32)

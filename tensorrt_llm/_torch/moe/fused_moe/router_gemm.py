# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-tile context router over the loaded stock gate's three BF16 weight terms.

The BF16 activation is read once per K tile and multiplied by every weight term with FP32 accumulation.
BF16 operands are promoted exactly to FP32 for TF32 MMA; their seven mantissa bits fit within TF32,
so this does not round a full FP32 checkpoint weight or discard any of its three BF16 terms.
No term is discarded and no repeated activation tensor is materialized. Each row has the same tile and
K/term order for every M within a split-K regime. Up to SPLITK_MAX_M rows, eight fixed K slices produce
FP32 partials, reduced in slice order; larger calls use the single-pass kernel. Reduction order can differ
across that cap. TRTLLM_CTX_ROUTER_TF32_SPLITK=0 selects single-pass at every M.
"""

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

CFG = (128, 32, 4, 4)
WEIGHT_TERMS = 3
SPLIT_K = 8
REDUCE_BLOCK_M = 32
SPLITK = os.environ.get("TRTLLM_CTX_ROUTER_TF32_SPLITK", "1") == "1"
SPLITK_MAX_M = 8192


@triton.jit(do_not_specialize=["M"])
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            x_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        for term in tl.static_range(TERMS):
            b = tl.load(w_ptr + rn[None, :] * (TERMS * K) + term * K + (k0 + rk)[:, None])
            acc = tl.dot(
                a.to(tl.float32),
                b.to(tl.float32),
                acc,
                input_precision="tf32",
                out_dtype=tl.float32,
            )
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


@triton.jit(do_not_specialize=["M"])
def _router_gemm_partial_kernel(
    x_ptr,
    p_ptr,
    w_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    KS: tl.constexpr,
    TERMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    s = tl.program_id(1)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(s * KS, (s + 1) * KS, BLOCK_K):
        a = tl.load(
            x_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        for term in tl.static_range(TERMS):
            b = tl.load(w_ptr + rn[None, :] * (TERMS * K) + term * K + (k0 + rk)[:, None])
            acc = tl.dot(
                a.to(tl.float32),
                b.to(tl.float32),
                acc,
                input_precision="tf32",
                out_dtype=tl.float32,
            )
    tl.store(p_ptr + (s * M + rm[:, None]).to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


@triton.jit(do_not_specialize=["M"])
def _router_gemm_reduce_kernel(
    p_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(
            p_ptr + (s * M + rm[:, None]).to(tl.int64) * N + rn[None, :],
            mask=mask[:, None],
            other=0.0,
        )
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


def _split_k(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    block_m, block_k, warps, stages = CFG
    m, k = x.shape
    n = weight.shape[0]
    part = torch.empty((SPLIT_K, m, n), dtype=torch.float32, device=x.device)
    _router_gemm_partial_kernel[(triton.cdiv(m, block_m), SPLIT_K)](
        x,
        part,
        weight,
        m,
        k,
        n,
        k // SPLIT_K,
        TERMS=WEIGHT_TERMS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    _router_gemm_reduce_kernel[(triton.cdiv(m, REDUCE_BLOCK_M),)](
        part,
        out,
        m,
        n,
        SPLIT_K,
        BLOCK_M=REDUCE_BLOCK_M,
        num_warps=4,
    )


@torch.library.custom_op("trtllm::router_gemm", mutates_args=())
def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 logits from contiguous CUDA BF16 x[M,K] and three BF16 weight terms[N,3*K]."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous() or not x.is_cuda:
        raise ValueError("router activation must be contiguous CUDA BF16 [M,K]")
    if (
        weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.shape[1] != WEIGHT_TERMS * x.shape[1]
    ):
        raise ValueError("router weight must be contiguous BF16 [N,3*K] on the activation device")
    block_m, block_k, warps, stages = CFG
    m, k = x.shape
    n = weight.shape[0]
    if k <= 0 or k % block_k != 0 or not 16 <= n <= 256 or n & (n - 1):
        raise ValueError(
            "router requires positive K divisible by 32 and power-of-two N in [16,256]"
        )
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if 0 < m <= SPLITK_MAX_M and SPLITK and k % (SPLIT_K * block_k) == 0:
        _split_k(x, weight, out)
        logger.info_once(
            f"Fixed-tile BF16-split context router: {SPLIT_K} K slices taken "
            "(kill TRTLLM_CTX_ROUTER_TF32_SPLITK=0)",
            key="router_gemm_splitk",
        )
    elif m:
        _router_gemm_kernel[(triton.cdiv(m, block_m),)](
            x,
            weight,
            out,
            m,
            k,
            n,
            TERMS=WEIGHT_TERMS,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
    return out


@router_gemm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)

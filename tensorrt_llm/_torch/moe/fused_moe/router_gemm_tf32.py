# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant TF32 router GEMM for prefill: logits[M, N] = x[M, K] (bfloat16) @ w[N, K]^T in float32.

The stock router (fp32 ``F.linear`` with TF32 allowed by the NGC environment) lets cuBLAS pick a kernel, tile and
split-K factor by M, so a token's logits, and through near-ties its experts, depend on the chunk size. This kernel
runs ONE fixed tile and K order for every M, so each output row is a function of its own input row: bitwise
identical for any M on the same side of the split-K cap below (SPLITK_MAX_M), and run to run. Same precision
class as stock: the router weight is rounded to TF32
once at load with round-to-nearest (``prepare_router_weight_tf32``, bit-identical to ``cvt.rna.tf32.f32``, the
same rounding as stock's TF32 path), the bfloat16 input is exact in TF32, and products accumulate in float32.
Reads the bfloat16 hidden states directly (no float32 copy).

K is split into SPLIT_K fixed slices: one CTA per (row block, slice) accumulates its slice in the single-pass
kernel's 32-wide chunk order into a float32 partial, and a second kernel sums the partials in slice order. The tile
and the split are fixed for every M, so each row still depends only on its own input row; the K-parallel grid keeps
the GEMM off a 192-step serial chain per CTA (63 -> 17.5 us/call at 2048 rows on SM107). Up to SPLITK_MAX_M rows
only: above it the float32 partials' round trip (SPLIT_K x M x N) eats the gain (no in-model gain on 16K-row chunks),
so larger steps keep the single-pass kernel, bitwise as before; rows are M-invariant within each side of the cap
(the breakable prefill graphs' padded buckets, <= 4096 rows, are all below it). Across the cap the K reduction order
changes, so a token's logits (and, at near-ties, its experts) can differ between a chunk of <= SPLITK_MAX_M rows and
a larger one. Kill:
TRTLLM_CTX_ROUTER_TF32_SPLITK=0 runs the single-pass kernel at every M."""

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

# (BLOCK_M, BLOCK_K, num_warps, num_stages): one config for every M (no autotune by M; changing it changes bits).
CFG = (128, 32, 4, 4)
# K slices of the split kernel and the row block of its fixed-order reduction (part of the bits, as CFG is).
SPLIT_K = 8
REDUCE_BLOCK_M = 32
SPLITK = os.environ.get("TRTLLM_CTX_ROUTER_TF32_SPLITK", "1") == "1"
# Largest M the split kernel takes (larger steps run the single-pass kernel).
SPLITK_MAX_M = 8192


def prepare_router_weight_tf32(w: torch.Tensor) -> torch.Tensor:
    """float32 router weight [N, K] -> TF32-rounded (round to nearest, ties away; Inf / NaN kept), float32 storage."""
    bits = w.detach().float().contiguous().view(torch.int32)
    rounded = (bits + 0x1000) & ~0x1FFF
    finite = (bits & 0x7F800000) != 0x7F800000
    return torch.where(finite, rounded, bits).view(torch.float32).contiguous()


@triton.jit(
    do_not_specialize=["M"]
)  # one binary for every M (no M == 1 / M % 16 variants, no JIT at capture)
def _router_gemm_tf32_kernel(
    x_ptr, w_ptr, out_ptr, M, K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):  # one K order for every M
        a = tl.load(
            x_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=mask[:, None], other=0.0
        )
        b = tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None])
        acc = tl.dot(a.to(tl.float32), b, acc, input_precision="tf32", out_dtype=tl.float32)
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


@triton.jit(do_not_specialize=["M"])
def _router_gemm_tf32_partial_kernel(
    x_ptr, p_ptr, w_ptr, M, K: tl.constexpr, N: tl.constexpr, KS: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):  # fmt: skip
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    s = tl.program_id(1)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(s * KS, (s + 1) * KS, BLOCK_K):  # the single-pass chunk order, within slice s
        a = tl.load(
            x_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=mask[:, None], other=0.0
        )
        b = tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None])
        acc = tl.dot(a.to(tl.float32), b, acc, input_precision="tf32", out_dtype=tl.float32)
    tl.store(p_ptr + (s * M + rm[:, None]).to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


@triton.jit(do_not_specialize=["M"])
def _router_gemm_tf32_reduce_kernel(
    p_ptr, out_ptr, M, N: tl.constexpr, S: tl.constexpr, BLOCK_M: tl.constexpr
):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    mask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for s in tl.static_range(S):  # slice order: fixed for every M
        acc += tl.load(
            p_ptr + (s * M + rm[:, None]).to(tl.int64) * N + rn[None, :],
            mask=mask[:, None],
            other=0.0,
        )
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=mask[:, None])


def _split_k(x: torch.Tensor, w_tf32: torch.Tensor, out: torch.Tensor) -> None:
    block_m, block_k, warps, stages = CFG
    M, K = x.shape
    N = w_tf32.shape[0]
    part = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    _router_gemm_tf32_partial_kernel[(triton.cdiv(M, block_m), SPLIT_K)](
        x,
        part,
        w_tf32,
        M,
        K,
        N,
        K // SPLIT_K,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    _router_gemm_tf32_reduce_kernel[(triton.cdiv(M, REDUCE_BLOCK_M),)](
        part, out, M, N, SPLIT_K, BLOCK_M=REDUCE_BLOCK_M, num_warps=4
    )


@torch.library.custom_op("trtllm::router_gemm_tf32", mutates_args=())
def router_gemm_tf32(x: torch.Tensor, w_tf32: torch.Tensor) -> torch.Tensor:
    """float32 logits [M, N] = x [M, K] (bfloat16) @ w_tf32 [N, K]^T; ``w_tf32`` from prepare_router_weight_tf32."""
    block_m, block_k, warps, stages = CFG
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous(), (
        "x: contiguous bfloat16 [M, K]"
    )
    assert (
        w_tf32.dtype == torch.float32 and w_tf32.is_contiguous() and w_tf32.shape[1] == x.shape[1]
    )
    M, K = x.shape
    N = w_tf32.shape[0]
    assert K % block_k == 0 and N & (N - 1) == 0 and N <= 256, (K, N)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if 0 < M <= SPLITK_MAX_M and SPLITK and K % (SPLIT_K * block_k) == 0:
        _split_k(x, w_tf32, out)
        logger.info_once(
            f"TF32 prefill router: split-K ({SPLIT_K} slices) taken (kill TRTLLM_CTX_ROUTER_TF32_SPLITK=0)",
            key="router_gemm_tf32_splitk",
        )
    elif M:
        _router_gemm_tf32_kernel[(triton.cdiv(M, block_m),)](
            x,
            w_tf32,
            out,
            M,
            K,
            N,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
    return out


@router_gemm_tf32.register_fake
def _(x: torch.Tensor, w_tf32: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_tf32.shape[0]), dtype=torch.float32)

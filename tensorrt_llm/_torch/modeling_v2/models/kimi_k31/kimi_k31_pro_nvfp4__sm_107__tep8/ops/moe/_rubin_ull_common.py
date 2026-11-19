# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared host-side pieces of the K3.1 low-latency MoE entries on BF16 latent weights (not an
entry).

The rank geometry (whole 128-blocks per rank), the push GEMM's tactics, the bytes of a Lamport
call, and the plain-torch references the entries share. Nothing here touches CUDA, so all of it
runs on the CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul

HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
#: Weight blocks split over the ranks (128 latent columns, latent_down rows or hidden columns).
BLOCK = 128
#: Rows of one call (the low-M tactics' range).
MAX_M = 32

# The split-K GEMM's specializations (FlashInfer dense_bf16_gemm_sm100_splitk): K tile, kernel-M
# (weight rows per CTA), kernel-N (rows of x per CTA), cluster split-K, pipeline depth.
CTA_K = 128
MMA_M = (64, 128)
MMA_N = (8, 16, 32)
SPLIT_K = (1, 2, 3, 4)
MAX_STAGES = 12
#: Per-CTA shared memory of SM100-class parts (cutlass ``get_smem_capacity_in_bytes("sm_100")``).
SMEM_CAPACITY = 232448
#: Weight tiles at or above which split-K 1 already fills the device.
_SPLIT1_TILES = 96

Tactic = Tuple[int, int, int, int]  # (mma_m, mma_n, split_k, ab_stages)


def block_slice(extent: int, world: int, rank: int) -> Tuple[int, int]:
    """``(first, count)`` of ``rank``'s whole 128-blocks of ``extent``: as even as possible, the
    first ranks taking one more (28 latent blocks over 8 ranks: 4 on ranks 0-3, 3 on 4-7)."""
    blocks = extent // BLOCK
    if extent % BLOCK or not 1 <= world <= blocks or not 0 <= rank < world:
        raise ValueError(f"{extent} columns over {world} ranks (rank {rank}) in blocks of {BLOCK}")
    base, extra = divmod(blocks, world)
    first = rank * base + min(rank, extra)
    return first * BLOCK, (base + (1 if rank < extra else 0)) * BLOCK


def _align(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def gemm_smem_bytes(t: Tactic) -> int:
    """Shared memory of the split-K GEMM (the device allocator's layout, as FlashInfer's
    ``_smem_bytes``)."""
    mma_m, mma_n, split_k, stages = t
    cursor = _align(mma_m * CTA_K * 2 * stages, 1024) + mma_n * CTA_K * 2 * stages
    cursor = _align(cursor, 8) + 2 * stages * 8 + 3 * 8
    cursor = _align(cursor, 4) + 4
    if split_k == 1:
        return cursor
    return _align(_align(cursor, 128) + (split_k - 1) * mma_m * mma_n * 4, 8) + 8


def push_smem_bytes(t: Tactic) -> int:
    """Shared memory of the push kernel: the GEMM's plus the token-major staging tile."""
    return _align(gemm_smem_bytes(t), 16) + t[0] * t[1] * 2


def tactic_ok(t: Tactic, m: int, n: int, k: int) -> bool:
    """Whether the push GEMM serves ``x [m, k] @ w[n, k]^T`` with ``t`` (FlashInfer's rules, whole
    weight tiles, the staging tile in shared memory)."""
    mma_m, mma_n, split_k, stages = t
    return (
        mma_m in MMA_M
        and mma_n in MMA_N
        and split_k in SPLIT_K
        and 1 <= stages <= MAX_STAGES
        and 1 <= m <= MAX_M
        and n > 0
        and n % mma_m == 0
        and k > 0
        and k % CTA_K == 0
        and (k // CTA_K) % split_k == 0
        and push_smem_bytes(t) <= SMEM_CAPACITY
    )


def _mma_n_for(m: int) -> int:
    return next(v for v in MMA_N if m <= v)


def _deepest(mma_m: int, mma_n: int, split_k: int, k: int) -> Optional[Tactic]:
    """The tactic with the deepest pipeline that fits, at most one stage per K tile of a CTA."""
    per_cta = k // CTA_K // split_k
    for stages in range(min(MAX_STAGES, max(1, per_cta)), 0, -1):
        t = (mma_m, mma_n, split_k, stages)
        if push_smem_bytes(t) <= SMEM_CAPACITY:
            return t
    return None


def push_tactic(m: int, n: int, k: int) -> Tactic:
    """The push GEMM's default tactic: 64-row weight tiles; split-K 1 where the tiles already fill
    the device, else the largest split that divides K; the deepest pipeline that fits."""
    if not 1 <= m <= MAX_M or n % 64 or k % CTA_K:
        raise ValueError(f"no push tactic for m={m}, n={n}, k={k}")
    splits = [s for s in reversed(SPLIT_K) if (k // CTA_K) % s == 0]
    split_k = 1 if n // 64 >= _SPLIT1_TILES else splits[0]
    t = _deepest(64, _mma_n_for(m), split_k, k)
    if t is None or not tactic_ok(t, m, n, k):
        raise ValueError(f"no push tactic for m={m}, n={n}, k={k}")
    return t


# ---------------------------------------------------------------------------
# Lamport call size (see kernels/comm/rubin_lamport_ar.py)
# ---------------------------------------------------------------------------


def required_bytes(m: int, row: int, world: int) -> int:
    """Bytes of one Lamport buffer a one-shot call with all-gather rows needs
    (``rubin_lamport_ar.required_bytes``)."""
    return m * row * (world + 1) * 2


# ---------------------------------------------------------------------------
# References (plain torch)
# ---------------------------------------------------------------------------


def sanitize_bf16(x: torch.Tensor) -> torch.Tensor:
    """``x`` with every ``-0`` sent as ``+0``, as a Lamport slot carries it."""
    return torch.where(x == 0, torch.zeros_like(x), x)


def gemm_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w^T`` summed in FP32 (TF32 off) and rounded once to BF16."""
    with true_fp32_matmul():
        return (x.float() @ w.float().t()).to(torch.bfloat16)


def push_payload(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """What one rank stores into the call: ``bf16(x @ w^T)``, ``-0`` as ``+0``."""
    return sanitize_bf16(gemm_bf16(x, w))


def reduce_chunk(world: int) -> int:
    """Ranks per partial sum of the C++ one-shot all-reduce a finish reproduces."""
    return min(world, 8)


__all__ = [
    "BLOCK",
    "HIDDEN",
    "LATENT",
    "MAX_M",
    "NUM_EXPERTS",
    "block_slice",
    "gemm_bf16",
    "gemm_smem_bytes",
    "push_payload",
    "push_smem_bytes",
    "push_tactic",
    "reduce_chunk",
    "required_bytes",
    "sanitize_bf16",
    "tactic_ok",
]

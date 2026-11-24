# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-side references of the K3.1 low-latency MoE entries: the sharded front's slot width, the
push GEMM's tactic sweep, the byte offsets of a Lamport call's slots and all-gather rows, and the
plain-torch finish and column-parallel tail."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_ull_common import (  # noqa: E501
    BLOCK,
    CTA_K,
    HIDDEN,
    LATENT,
    MMA_M,
    SPLIT_K,
    Tactic,
    _deepest,
    _mma_n_for,
    block_slice,
    push_payload,
    reduce_chunk,
    sanitize_bf16,
    tactic_ok,
)


def front_slot_cols(world: int) -> int:
    """Features of one (token, rank) latent slot of the sharded front: the largest share."""
    blocks = LATENT // BLOCK
    return (blocks // world + (1 if blocks % world else 0)) * BLOCK


def tactic_candidates(m: int, n: int, k: int) -> List[Tactic]:
    """Every (mma_m, split_k) the GEMM serves at ``m`` rows, each at its deepest pipeline: the
    tactics a sweep (and the kernel's test) tries."""
    out = []
    for mma_m in MMA_M:
        for split_k in SPLIT_K:
            if n % mma_m or (k // CTA_K) % split_k:
                continue
            t = _deepest(mma_m, _mma_n_for(m), split_k, k)
            if t is not None and tactic_ok(t, m, n, k):
                out.append(t)
    return out


def slot_offset(
    tok: int, rank: int, world: int, row: int, feat: int, one_shot: bool, cur: int, nbytes: int
) -> Tuple[int, int]:
    """``(holder, byte offset)`` of feature ``feat`` of ``rank``'s slot of token ``tok``: one-shot,
    in every rank's buffer (holder -1); two-shot, in the scatter stage of the owner
    ``tok % world``."""
    if one_shot:
        return -1, cur * nbytes + (tok * world + rank) * row * 2 + feat * 2
    slot = (tok // world) * world + rank
    return tok % world, cur * 2 * (nbytes // 2) + slot * row * 2 + feat * 2


def ag_offset(m: int, tok: int, world: int, row: int, feat: int, cur: int, nbytes: int) -> int:
    """Byte offset of feature ``feat`` of all-gather row ``tok`` of a one-shot call of ``m`` rows
    (after its ``m x world`` slots), in every rank's buffer."""
    return cur * nbytes + (m * world + tok) * row * 2 + feat * 2


def allreduce(parts: Sequence[torch.Tensor], cpp_one_shot: bool) -> torch.Tensor:
    """The finish of a call: FP32 sums of the slots in rank order (chunks of
    :func:`reduce_chunk` ranks), rounded once to BF16, ``-0`` as ``+0``."""
    acc = torch.zeros(parts[0].shape, dtype=torch.float32)
    chunk = reduce_chunk(len(parts), cpp_one_shot)
    for c0 in range(0, len(parts), chunk):
        part = torch.zeros_like(acc)
        for p in parts[c0 : c0 + chunk]:
            part = part + sanitize_bf16(p).float()
        acc = acc + part
    return sanitize_bf16(acc.to(torch.bfloat16))


def ag_finish(y: torch.Tensor, parts: Sequence[torch.Tensor], cpp_one_shot: bool) -> torch.Tensor:
    """The finish of a call with all-gather rows: ``bf16(y + bf16(sum of the slots))``."""
    return (y.float() + allreduce(parts, cpp_one_shot).float()).to(torch.bfloat16)


def cp_tail(
    latent: torch.Tensor,
    latent_up: torch.Tensor,
    shared_parts: Sequence[torch.Tensor],
    cpp_one_shot: bool,
) -> torch.Tensor:
    """The column-parallel tail's result on every rank: rank ``r`` writes its 128-blocks of
    ``bf16(latent @ latent_up^T)`` into the all-gather rows, the shared partials go into the
    slots, and the finish returns ``bf16(y + bf16(sum of the shared partials))``."""
    world = len(shared_parts)
    y = torch.empty(latent.shape[0], latent_up.shape[0], dtype=torch.bfloat16)
    for r in range(world):
        c0, n = block_slice(HIDDEN, world, r)
        y[:, c0 : c0 + n] = push_payload(latent, latent_up[c0 : c0 + n])
    return ag_finish(y, shared_parts, cpp_one_shot)

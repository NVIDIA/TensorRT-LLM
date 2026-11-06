# Copyright 2026 The VDN Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached decomposition of VDN windows into dense and variable-length attention."""

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention


@dataclass(frozen=True)
class WindowPlan:
    dense_queries: torch.Tensor
    window_queries: torch.Tensor
    window_keys: torch.Tensor
    cu_queries: torch.Tensor
    cu_keys: torch.Tensor
    max_queries: int
    max_keys: int


@lru_cache(maxsize=8)
def window_plan(
    start: int, frames: int, per_frame: int, chunk: int, radius: int, device: torch.device
) -> WindowPlan:
    """Group equal windows, preserving packed-row order within every KV set."""
    dense = list(range(start + per_frame)) + list(
        range(start + (frames - 1) * per_frame, start + frames * per_frame)
    )
    queries, keys, cu_q, cu_k = [], [], [0], [0]
    frame = 1
    max_q = max_k = 0
    while frame < frames - 1:
        end = min((frame // chunk + 1) * chunk, frames - 1) if chunk else frame + 1
        lo = (frame // chunk - radius) * chunk if chunk else frame - radius
        hi = (frame // chunk + radius + 1) * chunk - 1 if chunk else frame + radius
        queries.extend(range(start + frame * per_frame, start + end * per_frame))
        kept = sorted(set(range(max(lo, 0), min(hi + 1, frames))) | {0, frames - 1})
        keys.extend(range(start))
        for key_frame in kept:
            keys.extend(range(start + key_frame * per_frame, start + (key_frame + 1) * per_frame))
        q_length, k_length = (end - frame) * per_frame, start + len(kept) * per_frame
        max_q, max_k = max(max_q, q_length), max(max_k, k_length)
        cu_q.append(cu_q[-1] + q_length)
        cu_k.append(cu_k[-1] + k_length)
        frame = end
    return WindowPlan(
        torch.tensor(dense, device=device),
        torch.tensor(queries, device=device),
        torch.tensor(keys, device=device),
        torch.tensor(cu_q, device=device, dtype=torch.int32),
        torch.tensor(cu_k, device=device, dtype=torch.int32),
        max_q,
        max_k,
    )


@torch.compiler.disable
def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    start: int,
    frames: int,
    per_frame: int,
    chunk: int,
    radius: int,
    dense_attention: VanillaAttention,
) -> torch.Tensor:
    """One cuDNN global call plus one FA4 call for all disjoint window groups."""
    from flash_attn.cute.interface import flash_attn_varlen_func

    plan = window_plan(start, frames, per_frame, chunk, radius, q.device)
    q, k, v = q[0], k[0].contiguous(), v[0].contiguous()
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        dense = dense_attention.forward(
            q[plan.dense_queries].transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
        )
    out[plan.dense_queries] = dense[0].transpose(0, 1)
    windows = flash_attn_varlen_func(
        q[plan.window_queries],
        k[plan.window_keys],
        v[plan.window_keys],
        cu_seqlens_q=plan.cu_queries,
        cu_seqlens_k=plan.cu_keys,
        max_seqlen_q=plan.max_queries,
        max_seqlen_k=plan.max_keys,
        softmax_scale=dense_attention.scale,
    )
    out[plan.window_queries] = windows[0] if isinstance(windows, tuple) else windows
    return out[None]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned state of Kimi K3's fused KDA projection: one set of its three Lamport buffers and per-CTA indices.

A state type, not an entry: it launches nothing per call. Its constructor is eager; the target builds one per
device in ``post_load_weights`` (before any CUDA-graph capture) and passes it to every ``ssm/k3_kda_attn`` and
``ssm/k3_kda_decode_attn`` call on that device, of every KDA layer. ``ssm/k3_kda_attn``'s sibling
``k3_kda_qkvg`` (the projection stream alone) takes a set of its own, made with ``ctas=CTAS``. The contract is the
``## State`` section of ``k3_kda_attn.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op as _op

CTAS = _op.CTAS  # k3_kda_qkvg's grid: 26 clusters of 4 CTAs
FUSED_CTAS = _op.FUSED_CTAS  # k3_kda_attn's and k3_kda_decode_attn's grid


@dataclass(eq=False)
class K3KdaBuffers:
    """The projection's Lamport set: three buffers of every published word, each word the sentinel (all ones) until a
    launch writes it, and each CTA's buffer index. A launch writes buffer ``e = epoch[cta]``, re-arms buffer
    ``(e + 1) % 3`` (the next launch's) to the sentinel and leaves ``epoch[cta] = (e + 1) % 3``, so launches on one
    set run one at a time, in stream order, and every launch moves every index once (see the contract's
    ``## State``)."""

    p1: torch.Tensor
    """int16 [3 * 8 * 1664]: per buffer, 8 token rows of the q, k and f_a columns (bf16 bits)."""
    part: torch.Tensor
    """int32 [3 * 3 * 2 * 8 * 768]: per buffer, the fp32 bits of the two K-half partials of v, og and b."""
    epoch: torch.Tensor
    """int32 [ctas]: each CTA's buffer index (its launch count mod 3)."""
    ctas: int

    @classmethod
    def create(cls, device, ctas: int = FUSED_CTAS) -> "K3KdaBuffers":
        """Allocate and arm a set for ``ctas`` CTAs (``FUSED_CTAS`` for k3_kda_attn / k3_kda_decode_attn, ``CTAS``
        for k3_kda_qkvg) on ``device``: every buffer word the sentinel, every index 0. Eager: it allocates, so it
        refuses to run under CUDA-graph capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("K3KdaBuffers.create allocates: call it before CUDA-graph capture")
        if ctas not in (CTAS, FUSED_CTAS):
            raise ValueError(
                f"K3KdaBuffers: ctas must be {CTAS} (k3_kda_qkvg) or {FUSED_CTAS}, got {ctas}"
            )
        p1, part, epoch = _op.make_buffers(torch.device(device), ctas)
        return cls(p1=p1, part=part, epoch=epoch, ctas=ctas)

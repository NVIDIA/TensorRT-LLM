# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE positions of a DFlash generation step's context latents (Triton, one program)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import dflash_ctx_positions


class K3DflashCtxPositions(OpWrapper):
    """``out[g * width + j] = ctx_len[slots[g]] + j`` for ``g < len(slots)``, ``j < width``:
    int64 ``[len(slots) * width]``, new. ``ctx_len`` is the drafter's per-slot context length
    before the step advances it."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="helix8 c24/c42: 2 requests of 8 positions on the side stream",
            spec=dict(num_gens=2, width=8, max_batch=2),
        ),
        Cell(
            why="helix8 c75/c174: 8 requests of 8 positions",
            spec=dict(num_gens=8, width=8, max_batch=8),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 requests of 8 positions, padding rows on the "
            "dummy slot",
            spec=dict(num_gens=4, width=8, max_batch=4),
        ),
    )

    note = """
    Runs on the side stream that computes a TP-sharded drafter's context latents, before the step
    preparation advances ``ctx_len`` on the main stream (the caller orders the two with an event).
    """

    def __call__(self, ctx_len: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
        return dflash_ctx_positions(ctx_len, slots, width)

    def reference(self, ctx_len: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
        old = ctx_len[slots.to(torch.int64)].to(torch.int64)
        j = torch.arange(width, device=ctx_len.device, dtype=torch.int64)
        return (old[:, None] + j[None, :]).reshape(-1)


k3_dflash_ctx_positions = K3DflashCtxPositions()

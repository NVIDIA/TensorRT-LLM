# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window-bounded context ring: every window a block query reads holds exactly
what a dense per-position arena holds, through prefill chunks longer than the ring, verify writes
with rejected tails, and block writes. The check is plain torch and runs on the CPU."""

import random

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (
    dflash2_window_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Arena:
    """A one-scalar-per-row paged arena addressed like the worker's private arena."""

    def __init__(self, num_slots, rows_per_slot, page_size, seed, ring_rows):
        pages_per_slot = rows_per_slot // page_size
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_slots * pages_per_slot, generator=g)
        self.table = perm.view(num_slots, pages_per_slot).to(_DEVICE)
        self.pool = torch.zeros(num_slots * pages_per_slot, page_size, device=_DEVICE)
        self.page_size = page_size
        self.ring_rows = ring_rows

    def store(self, slot, positions, values):
        """The worker's direct page write (_store_context_kv_paged) with the ring mapping."""
        positions = dflash2_ring_positions(positions, self.ring_rows)
        rows = torch.full_like(positions, slot)
        pages = self.table[rows, positions // self.page_size]
        self.pool[pages, positions % self.page_size] = values

    def prefill(self, slot, start, length):
        """One prefill chunk ``[start, start + length)``: content ``position + 1``."""
        positions = torch.arange(start, start + length, device=_DEVICE)
        keep = dflash2_ring_chunk_start(length, self.ring_rows)
        self.store(slot, positions[keep:], (positions[keep:] + 1).float())

    def window(self, slots, ctx_len, block_size, window):
        rows, lengths = dflash2_window_rows(
            ctx_len,
            self.table.index_select(0, slots),
            block_size,
            self.page_size,
            window,
            window,
            ring_rows=self.ring_rows,
        )
        flat = self.pool.view(-1)
        return [flat[r[:n].long()].tolist() for r, n in zip(rows.cpu(), lengths.cpu().tolist())]


def _expected_window(ctx_len, block_size, window):
    """Context rows ``k + 1`` below the block start, block rows ``-(k + 1)``."""
    out = []
    for c in ctx_len:
        for j in range(block_size):
            p = c + j
            out.append(
                [
                    float(k + 1) if k < c else float(-(k + 1))
                    for k in range(max(0, p - window + 1), p + 1)
                ]
            )
    return out


@pytest.mark.cpu_only
@pytest.mark.parametrize("window,max_draft_len,page_size", [(1024, 4, 32)])
def test_ring_windows_match_dense_through_wraps(window, max_draft_len, page_size):
    """Random prefill chunks (some longer than the ring, one far past 1M positions), then verify
    steps with random acceptance: every query's window reads exactly the dense arena's rows."""
    rng = random.Random(window * 100 + max_draft_len)
    verify_width = max_draft_len + 1
    block_size = max_draft_len + 1
    ring = dflash2_ring_rows(window, verify_width, block_size, page_size)
    num_slots = 3
    arena = _Arena(num_slots, ring, page_size, seed=window + max_draft_len, ring_rows=ring)
    slots = torch.arange(num_slots, device=_DEVICE)
    # Prompts: short, several ring-length chunks, and one ~1M-position prompt in a single chunk.
    prompts = [
        [rng.randint(1, window)],
        [rng.randint(1, 3 * ring) for _ in range(4)],
        [1_048_000 + rng.randint(0, 500), rng.randint(1, ring)],
    ]
    ctx_len = []
    for slot, chunks in enumerate(prompts):
        start = 0
        for n in chunks:
            arena.prefill(slot, start, n)
            start += n
        ctx_len.append(start)

    steps = 2 * (ring // max_draft_len) // 3 + 8
    for _ in range(steps):
        # Verify: V rows written at [c, c + V), the rejected tail zeroed (the worker's masked write).
        accepted = [rng.randint(1, verify_width) for _ in ctx_len]
        for slot, (c, acc) in enumerate(zip(ctx_len, accepted)):
            positions = torch.arange(c, c + verify_width, device=_DEVICE)
            values = torch.where(
                torch.arange(verify_width, device=_DEVICE) < acc,
                positions + 1,
                torch.zeros_like(positions),
            ).float()
            arena.store(slot, positions, values)
        ctx_len = [c + a for c, a in zip(ctx_len, accepted)]
        # Block: B rows at [c', c' + B) (the drafter's own latents), then the window reads.
        for slot, c in enumerate(ctx_len):
            positions = torch.arange(c, c + block_size, device=_DEVICE)
            arena.store(slot, positions, -(positions + 1).float())
        got = arena.window(slots, torch.tensor(ctx_len, device=_DEVICE), block_size, window)
        assert got == _expected_window(ctx_len, block_size, window)

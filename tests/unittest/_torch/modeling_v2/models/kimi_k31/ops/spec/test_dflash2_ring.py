# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window-bounded context ring: sizing, position-to-row mapping, and that
every window a block query reads holds exactly what a dense per-position arena holds, through
prefill chunks longer than the ring, verify writes with rejected tails, and block writes."""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (
    dflash2_window_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "window,width,page_size,expected",
    [(1024, 8, 32, 1056), (1024, 5, 32, 1056), (1024, 8, 64, 1088), (6, 5, 4, 16)],
)
def test_ring_rows_hold_window_and_two_writes(window, width, page_size, expected):
    rows = dflash2_ring_rows(window, width, width, page_size)
    assert rows == expected
    assert rows % page_size == 0
    assert rows >= window + 2 * width  # the read bound W + width - 1 plus a rolled-back step


def test_ring_positions_and_chunk_start():
    pos = torch.tensor([0, 1, 1055, 1056, 1057, 1_048_575, 1_048_580])
    assert dflash2_ring_positions(pos, 1056).tolist() == [p % 1056 for p in pos.tolist()]
    assert dflash2_ring_positions(pos, None) is pos
    assert dflash2_ring_chunk_start(100, 1056) == 0
    assert dflash2_ring_chunk_start(8192, 1056) == 8192 - 1056
    assert dflash2_ring_chunk_start(8192, None) == 0


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


@pytest.mark.parametrize(
    "window,max_draft_len,page_size", [(8, 4, 4), (8, 7, 4), (1024, 4, 32), (1024, 7, 32)]
)
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

    steps = 3 * (ring // max_draft_len) if window < 64 else 2 * (ring // max_draft_len) // 3 + 8
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


def test_too_small_ring_is_rejected():
    table = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="ring"):
        dflash2_window_rows(torch.tensor([3]), table, 5, 4, 8, 8, ring_rows=8)


# -- the worker's arena -----------------------------------------------------------------------

_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the worker allocates CUDA buffers"
)


class _FakeMLADrafter:
    """What the worker reads off the DFlash2 MLA drafter to size and address its arena."""

    config = SimpleNamespace(max_position_embeddings=1 << 20)
    dflash_attention_backend = "VANILLA"
    _paged_ctx_cache = True
    _ctx_ring_rows = None

    def __init__(self, window, block_size):
        self.block_size = block_size
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._ctx_ring_window = window
        self._num_attn_layers = 2
        self._num_heads = 4
        self._num_kv_heads = 1
        self._head_dim = 16
        self._kv_factor = 1

    def _build_fused_kv_buffers(self):
        pass


def _ring_worker(window, max_draft_len, max_seq_len):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31DFlashWorker,
    )
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    config = DFlashDecodingConfig(max_draft_len=max_draft_len, attention_backend="VANILLA")
    worker = K31DFlashWorker(config, Mapping())
    draft = _FakeMLADrafter(window, max_draft_len + 1)
    worker.set_draft_model(draft)
    worker._lazy_init_ctx_buffers(
        draft, SimpleNamespace(max_num_requests=2), SimpleNamespace(max_seq_len=max_seq_len)
    )
    return worker, draft


@_cuda
def test_worker_sizes_the_private_arena_by_the_window():
    window, max_draft_len = 1024, 4
    worker, draft = _ring_worker(window, max_draft_len, 1 << 20)
    ring = dflash2_ring_rows(window, max_draft_len + 1, max_draft_len + 1, worker._ctx_page_size)
    assert worker._ctx_ring_rows == ring and draft._ctx_ring_rows == ring
    assert worker._ctx_pages_per_slot * worker._ctx_page_size == ring
    # [layers, slots x pages, kv_factor, kv_heads, page, head_dim]: 3 slots of one ring each.
    assert worker._ctx_kv_buf.shape == (2, 3 * ring // worker._ctx_page_size, 1, 1, 32, 16)
    assert worker._max_ctx == 1 << 20  # the servable length is unchanged

    # A context no longer than the ring keeps the dense layout.
    worker, draft = _ring_worker(window, max_draft_len, 512)
    assert worker._ctx_ring_rows is None and draft._ctx_ring_rows is None
    assert worker._ctx_pages_per_slot * worker._ctx_page_size >= 512 + max_draft_len + 1


@_cuda
def test_worker_store_wraps_and_windows_read_it_back():
    window, max_draft_len = 64, 7
    worker, _ = _ring_worker(window, max_draft_len, 1 << 20)
    ring, page_size = worker._ctx_ring_rows, worker._ctx_page_size
    assert ring == dflash2_ring_rows(window, max_draft_len + 1, max_draft_len + 1, page_size)
    slot, total = 1, 5 * ring + 13
    hd = worker._ctx_kv_buf.shape[-1]
    positions = torch.arange(total, device="cuda")
    latents = torch.randn(total, 2, 1, hd, device="cuda").to(torch.bfloat16)
    # Prefill in two chunks, each keeping only its last ring rows (the worker's prefill store).
    for lo, hi in ((0, 3 * ring + 5), (3 * ring + 5, total)):
        keep = lo + dflash2_ring_chunk_start(hi - lo, ring)
        rows = torch.full((hi - keep,), slot, dtype=torch.long, device="cuda")
        worker._store_context_kv_paged(latents[keep:hi], None, rows, positions[keep:hi])
    block_size = max_draft_len + 1
    ctx_len = torch.tensor([total - block_size], device="cuda")
    rows, lengths = dflash2_window_rows(
        ctx_len,
        worker._ctx_page_table[slot : slot + 1],
        block_size,
        page_size,
        window,
        window,
        ring_rows=ring,
    )
    for layer in range(2):
        pool = worker._ctx_kv_buf[layer][:, 0, 0].reshape(-1, hd)
        for j in range(block_size):
            p = int(ctx_len) + j
            n = int(lengths[j])
            assert n == window
            torch.testing.assert_close(
                pool[rows[j, :n].long()], latents[p - window + 1 : p + 1, layer, 0], rtol=0, atol=0
            )

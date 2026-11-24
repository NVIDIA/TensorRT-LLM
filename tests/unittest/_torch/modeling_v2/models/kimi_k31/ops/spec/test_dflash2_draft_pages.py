# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's context in the KV cache manager's draft pages: the chunk-tail invariant, the
planner's addresses and histories, ring seeding bit-equal to a ring filled by the request's own
prefill (through a transfer between two managers, and through a reused prefix), the context-only
capture guard, and the window the family manager gives its draft layers."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec import (
    dflash2_draft_pages as pages_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_draft_pages import (  # noqa: E501
    DFlash2DraftPagesWorkerMixin,
    DraftPagePlanner,
    RingSeed,
    dflash2_check_context_kv_dtype,
    dflash2_chunk_tail_start,
    dflash2_draft_window,
    dflash2_history_after_chunk,
    dflash2_seed_ring_from_pages,
    dflash2_store_draft_pages,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.standalone_draft_cache import StandaloneDraftHistory

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LAYERS = 5
_LATENT = 576


# --- Chunk tails and histories ---------------------------------------------------------------------


def _written_after(chunks, window, reused_window_end=None):
    """Positions written by storing each chunk's tail, plus a reused window ending at a reuse point."""
    written = set()
    if reused_window_end is not None:
        written.update(range(max(0, reused_window_end - window), reused_window_end))
    for start, length in chunks:
        written.update(range(dflash2_chunk_tail_start(start, length, window), start + length))
    return written


@pytest.mark.parametrize("seed", range(20))
def test_every_chunk_end_has_a_full_window(seed):
    rng = random.Random(seed)
    window = rng.choice([4, 16, 1024])
    reuse = rng.choice([0, rng.randint(1, 3000)])
    start, chunks = reuse, []
    for _ in range(rng.randint(1, 8)):
        length = rng.choice([1, 3, window - 1, window, window + 1, rng.randint(1, 3 * window)])
        chunks.append((start, length))
        start += length
    written = _written_after(chunks, window, reuse if reuse else None)
    for chunk_start, length in chunks:
        end = chunk_start + length
        assert set(range(max(0, end - window), end)) <= written, (window, reuse, chunks, end)


@pytest.mark.parametrize("dtype, ok", [(None, True), ("auto", True), ("fp8", False)])
def test_context_kv_dtype_matches_the_drafter_storage(dtype, ok):
    # The draft pages hold the ring's latents byte for byte, in the drafter's dtype.
    spec_config = SimpleNamespace(context_kv_dtype=dtype)
    if ok:
        dflash2_check_context_kv_dtype(spec_config)
    else:
        with pytest.raises(ValueError, match="context_kv_dtype='auto'"):
            dflash2_check_context_kv_dtype(spec_config)


def test_history_after_chunk():
    window = 16
    # A fresh request: the chunk alone.
    assert dflash2_history_after_chunk(None, 0, 5, window) == (5, 5)
    assert dflash2_history_after_chunk(StandaloneDraftHistory(0, 0), 0, 40, window) == (16, 40)
    # A continuation adds to the valid window, capped at the window.
    assert dflash2_history_after_chunk(StandaloneDraftHistory(5, 5), 5, 3, window) == (8, 8)
    assert dflash2_history_after_chunk(StandaloneDraftHistory(16, 100), 100, 3, window) == (16, 103)
    # A history that does not end where the chunk starts counts nothing before it.
    assert dflash2_history_after_chunk(StandaloneDraftHistory(16, 90), 100, 3, window) == (3, 103)


# --- A draft-pages manager stand-in ----------------------------------------------------------------


class _Pages:
    """The KV cache manager's draft-page surface the helpers use, with SWA-like eviction: pages that
    end before a request's valid window read as missing (-1)."""

    def __init__(self, num_pages, tokens_per_block, window, dtype, seed):
        self.tokens_per_block = tokens_per_block
        self.window = window
        self.pools = [
            torch.zeros(num_pages, 1, 1, tokens_per_block, _LATENT, dtype=dtype, device=_DEVICE)
            for _ in range(_LAYERS)
        ]
        g = torch.Generator().manual_seed(seed)
        self._free = torch.randperm(num_pages, generator=g).tolist()
        self.kv_cache_map = {}
        self.draft_history = {}

    def add_request(self, request_id, capacity, shared_pages=(), history_length=0):
        pages = list(shared_pages)
        num_blocks = -(-capacity // self.tokens_per_block)
        pages += [self._free.pop() for _ in range(num_blocks - len(pages))]
        self.kv_cache_map[request_id] = SimpleNamespace(
            is_active=True, pages=pages, history_length=history_length
        )

    def get_draft_buffers(self, layer_idx, kv_layout="HND"):
        assert kv_layout == "HND"
        return self.pools[layer_idx]

    def get_draft_history(self, request_id):
        return self.draft_history.get(request_id)

    def set_draft_history(self, request_id, valid_length, position):
        self.draft_history[request_id] = StandaloneDraftHistory(valid_length, position)

    def get_draft_block_indices(self, request_ids, histories=None):
        rows = []
        for row, request_id in enumerate(request_ids):
            pages = list(self.kv_cache_map[request_id].pages)
            history = (
                histories[row] if histories is not None else self.get_draft_history(request_id)
            )
            if history is not None:
                first = (history.position - history.valid_length) // self.tokens_per_block
                pages[:first] = [-1] * first
            rows.append(pages)
        return rows


_MAX_POSITIONS = 4096
_TABLE = torch.randn(
    _MAX_POSITIONS, _LAYERS, 1, _LATENT, generator=torch.Generator().manual_seed(0)
).to(torch.bfloat16)


def _latents(positions, seed=0):
    """A latent per absolute position that depends on the position alone."""
    del seed
    return _TABLE[torch.as_tensor(positions, dtype=torch.long)].to(_DEVICE)


def _store_chunks(manager, request_id, chunks, window, known=None, seed=0):
    """A context server's per-step stores: each chunk's tail into the pages, histories published."""
    known = {} if known is None else known
    for start, length in chunks:
        planner = DraftPagePlanner(manager, window, known)
        planner.add_chunk(request_id, start, length, row_base=0)
        writes = planner.writes(torch.device(_DEVICE))
        if writes is not None:
            latents = _latents(writes.positions.tolist(), seed=seed)
            pools = [pool[:, 0, 0] for pool in manager.pools]
            dflash2_store_draft_pages(pools, latents, writes.pages, writes.offsets)
        for rid, (valid, position) in zip(planner.request_ids, planner.histories):
            manager.set_draft_history(rid, valid, position)
    return known


class _Ring:
    """The worker's private paged ring arena ``[L, pages, 1, 1, page, latent]``."""

    def __init__(self, num_slots, window, width, page_size, dtype):
        self.rows = dflash2_ring_rows(window, width, width, page_size)
        pages_per_slot = self.rows // page_size
        self.page_size = page_size
        self.buf = torch.zeros(
            _LAYERS,
            num_slots * pages_per_slot,
            1,
            1,
            page_size,
            _LATENT,
            dtype=dtype,
            device=_DEVICE,
        )
        self.table = torch.arange(
            num_slots * pages_per_slot, dtype=torch.int32, device=_DEVICE
        ).view(num_slots, pages_per_slot)

    def prefill(self, slot, positions, latents):
        """The worker's prefill store (``_store_context_kv_paged``): a chunk longer than the ring
        keeps its last ``rows`` positions (``dflash2_ring_chunk_start``), one store without
        duplicate destinations."""
        keep = dflash2_ring_chunk_start(len(positions), self.rows)
        positions, latents = positions[keep:], latents[keep:]
        rows = dflash2_ring_positions(torch.as_tensor(positions, device=_DEVICE), self.rows)
        pages = self.table[slot, rows // self.page_size].long()
        offsets = rows % self.page_size
        for layer in range(_LAYERS):
            pool = self.buf[layer][:, 0, 0]
            pool[pages, offsets] = latents[:, layer, 0].to(pool.dtype)

    def read(self, slot, positions):
        rows = dflash2_ring_positions(torch.as_tensor(positions, device=_DEVICE), self.rows)
        pages = self.table[slot, rows // self.page_size].long()
        return self.buf[:, pages, 0, 0, rows % self.page_size]


def test_planner_addresses_and_histories():
    window, tpb = 8, 4
    manager = _Pages(64, tpb, window, torch.bfloat16, seed=1)
    manager.add_request(3, 30)
    manager.add_request(4, 9)
    planner = DraftPagePlanner(manager, window)
    planner.add_chunk(3, 0, 20, row_base=0)  # stores 12..19
    planner.add_chunk(4, 0, 9, row_base=20)  # stores 1..8
    writes = planner.writes(torch.device(_DEVICE))
    positions = list(range(12, 20)) + list(range(1, 9))
    assert writes.positions.tolist() == positions
    assert writes.host_positions.tolist() == positions
    assert writes.rows.tolist() == list(range(12, 20)) + list(range(21, 29))
    pages3, pages4 = manager.kv_cache_map[3].pages, manager.kv_cache_map[4].pages
    expected = [pages3[p // tpb] for p in range(12, 20)] + [pages4[p // tpb] for p in range(1, 9)]
    assert writes.pages.tolist() == expected
    assert writes.offsets.tolist() == [p % tpb for p in positions]
    assert planner.histories == [(8, 20), (8, 9)]
    # The next chunk continues from the planner's own record, unpublished in the manager.
    nxt = DraftPagePlanner(manager, window, planner.known)
    nxt.add_chunk(3, 20, 3, row_base=0)
    assert nxt.histories == [(8, 23)]
    update = nxt.history_update()
    update.publish()
    assert manager.get_draft_history(3) == StandaloneDraftHistory(8, 23)


def test_planner_rejects_unallocated_positions():
    manager = _Pages(16, 4, 8, torch.bfloat16, seed=2)
    manager.add_request(1, 8)
    planner = DraftPagePlanner(manager, 8)
    with pytest.raises(ValueError, match="do not cover"):
        planner.add_chunk(1, 0, 12, row_base=0)


# --- Ring seeding --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,chunks",
    [(7, [7]), (1500, [1500]), (1500, [700, 800]), (2600, [1000, 1000, 600]), (1030, [1024, 6])],
)
def test_transferred_window_seeds_the_ring_bitwise(prompt, chunks):
    window, tpb, width = 1024, 64, 5
    dtype = torch.bfloat16
    starts = [sum(chunks[:i]) for i in range(len(chunks))]
    # Context server: the chunk tails into its draft pages.
    ctx = _Pages(256, tpb, window, dtype, seed=3)
    ctx.add_request(11, prompt + width)
    _store_chunks(ctx, 11, list(zip(starts, chunks)), window)
    valid = min(prompt, window)
    assert ctx.get_draft_history(11) == StandaloneDraftHistory(valid, prompt)
    # Transfer: the window's pages, block by block, into the generation server's own pages.
    gen = _Pages(256, tpb, window, dtype, seed=4)
    gen.add_request(11, prompt + width)
    history = StandaloneDraftHistory(valid, prompt)
    (src,) = ctx.get_draft_block_indices([11], [history])
    (dst,) = gen.get_draft_block_indices([11], [history])
    for block in range((prompt - valid) // tpb, -(-prompt // tpb)):
        for layer in range(_LAYERS):
            gen.pools[layer][dst[block]].copy_(ctx.pools[layer][src[block]])
    gen.set_draft_history(11, valid, prompt)
    # Generation server: seed slot 2 of its ring; aggregated: its own prefill filled slot 2.
    seeded = _Ring(4, window, width, 32, dtype)
    pools = [pool[:, 0, 0] for pool in gen.pools]
    dflash2_seed_ring_from_pages(
        gen, pools, seeded.buf, seeded.table, seeded.rows, 32, [RingSeed(11, 2, valid, prompt)]
    )
    aggregated = _Ring(4, window, width, 32, dtype)
    aggregated.prefill(2, list(range(prompt)), _latents(list(range(prompt))))
    window_positions = list(range(prompt - valid, prompt))
    assert torch.equal(seeded.read(2, window_positions), aggregated.read(2, window_positions))


@pytest.mark.parametrize("reuse,turn2", [(640, 300), (640, 2000), (1500, 64), (100, 50)])
def test_reused_prefix_window_seeds_the_ring(reuse, turn2):
    """Turn 2 reuses turn 1's prompt [0, reuse) (a snapshot point, so a chunk end of turn 1): the
    window behind it comes from the reused pages, the rest from turn 2's own chunk."""
    window, tpb, width = 1024, 64, 5
    manager = _Pages(512, tpb, window, torch.bfloat16, seed=5)
    manager.add_request(1, reuse + width)
    _store_chunks(manager, 1, [(0, reuse)], window)
    # Turn 2 shares turn 1's blocks before the reuse point (whole pages; a partial last page is
    # the manager's copy), with the history a resumed reused prefix gets.
    shared = manager.kv_cache_map[1].pages[: reuse // tpb]
    manager.add_request(2, reuse + turn2 + width, shared_pages=shared)
    partial = reuse % tpb
    if partial:
        src, dst = (
            manager.kv_cache_map[1].pages[reuse // tpb],
            manager.kv_cache_map[2].pages[reuse // tpb],
        )
        for pool in manager.pools:
            pool[dst].copy_(pool[src])
    manager.set_draft_history(2, min(reuse, window), reuse)
    # Aggregated turn 2: seed the ring at the reuse point, then the chunk.
    ring = _Ring(2, window, width, 32, torch.bfloat16)
    pools = [pool[:, 0, 0] for pool in manager.pools]
    dflash2_seed_ring_from_pages(
        manager,
        pools,
        ring.buf,
        ring.table,
        ring.rows,
        32,
        [RingSeed(2, 0, min(reuse, window), reuse)],
    )
    end = reuse + turn2
    ring.prefill(0, list(range(reuse, end)), _latents(list(range(reuse, end))))
    _store_chunks(manager, 2, [(reuse, turn2)], window)
    assert manager.get_draft_history(2) == StandaloneDraftHistory(min(end, window), end)
    # Both the ring and the pages hold the full window ending at the prompt end.
    expected = _latents(list(range(max(0, end - window), end)))
    positions = list(range(max(0, end - window), end))
    assert torch.equal(ring.read(0, positions), expected[:, :, 0].permute(1, 0, 2).contiguous())
    (pages,) = manager.get_draft_block_indices([2])
    for layer in range(_LAYERS):
        got = torch.stack([manager.pools[layer][pages[p // tpb], 0, 0, p % tpb] for p in positions])
        assert torch.equal(got, expected[:, layer, 0])


def test_seeding_rejects_a_dtype_mismatch():
    manager = _Pages(16, 4, 8, torch.bfloat16, seed=6)
    manager.add_request(1, 8)
    ring = _Ring(1, 8, 2, 4, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="share a dtype"):
        dflash2_seed_ring_from_pages(
            manager,
            [pool[:, 0, 0] for pool in manager.pools],
            ring.buf,
            ring.table,
            ring.rows,
            4,
            [RingSeed(1, 0, 8, 8)],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_draft_page_store_refuses_cuda_graph_capture():
    pool = torch.zeros(4, 4, _LATENT, dtype=torch.bfloat16, device="cuda")
    latents = torch.ones(2, 1, 1, _LATENT, dtype=torch.bfloat16, device="cuda")
    index = torch.zeros(2, dtype=torch.long, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="outside every captured CUDA graph"):
        with torch.cuda.graph(graph):
            dflash2_store_draft_pages([pool], latents, index, index)


# --- The worker mixin ---------------------------------------------------------------------------


class _EngineWorker:
    """The engine DFlashWorker state the mixin reads, with its slot assignment."""

    def __init__(self, ring, num_slots=4):
        self._req_to_slot, self._free_slots = {}, list(range(num_slots))
        self._ctx_len_host = [0] * (num_slots + 1)
        self._req_ctx_pos = {}
        self._ctx_buf_inited, self._ctx_paged, self._ctx_block_tables = True, True, None
        self._ctx_kv_buf, self._ctx_page_table = ring.buf, ring.table
        self._ctx_ring_rows, self._ctx_page_size = ring.rows, ring.page_size

    def _write_ctx_len(self, updates):
        for slot, value in updates.items():
            self._ctx_len_host[slot] = value

    def _assign_slot(self, req_id, reset=False, updates=None):
        if req_id not in self._req_to_slot:
            slot = self._free_slots.pop(0)
            self._req_to_slot[req_id] = slot
            (self._write_ctx_len if updates is None else updates.update)({slot: 0})
        return self._req_to_slot[req_id]

    def evict(self, req_id):
        """The engine's slot release for a request that left a step (``DFlashSpecMetadata.prepare``)."""
        slot = self._req_to_slot.pop(req_id)
        self._write_ctx_len({slot: 0})
        self._req_ctx_pos.pop(req_id, None)
        self._free_slots.append(slot)


class _Worker(DFlash2DraftPagesWorkerMixin, _EngineWorker):
    pass


def test_worker_seeds_new_generation_slots_once():
    window, tpb, width, prompt = 1024, 64, 5, 1200
    manager = _Pages(256, tpb, window, torch.bfloat16, seed=7)
    manager.add_request(21, prompt + width, history_length=prompt)
    manager.add_request(22, 40 + width)
    manager.add_request(23, 64 + width)  # arrives without drafter context in its pages
    # Generated past its prompt end before it got a slot (it left a step): not seeded.
    manager.add_request(24, prompt + 200 + width, history_length=prompt + 160)
    _store_chunks(manager, 21, [(0, prompt)], window)
    _store_chunks(manager, 24, [(0, prompt)], window)
    ring = _Ring(5, window, width, 32, torch.bfloat16)
    worker = _Worker(ring)
    worker._assign_slot(22, updates={})  # a prefill's assignment: not queued
    worker._assign_slot(21)  # a generation request's assignment while inputs are prepared
    worker._assign_slot(23)
    worker._assign_slot(24)
    assert worker._draft_page_seeds == [21, 23, 24]
    draft_model = SimpleNamespace(_window=window, _num_attn_layers=_LAYERS, _head_dim=_LATENT)
    spec_metadata = SimpleNamespace(project_captures=False)
    attn_metadata = SimpleNamespace(kv_cache_manager=manager)
    manager.has_standalone_draft_cache = True
    worker.prepare_managed_draft_cache(draft_model, spec_metadata, attn_metadata, None)
    slot = worker._req_to_slot[21]
    assert worker._ctx_len_host[slot] == prompt and worker._req_ctx_pos[21] == prompt
    assert worker._ctx_len_host[worker._req_to_slot[23]] == 0
    assert worker._ctx_len_host[worker._req_to_slot[24]] == 0
    assert worker._draft_page_seeds == []
    counts = (
        worker._draft_page_queued,
        worker._draft_page_arrived,
        worker._draft_page_seeded,
        worker._draft_page_late,
    )
    assert counts == (3, 1, 1, 1)
    positions = list(range(prompt - window, prompt))
    expected = _latents(positions)[:, :, 0].permute(1, 0, 2).contiguous()
    assert torch.equal(ring.read(slot, positions), expected)
    # Only a context-only drafter asks for projected captures, and it reads the plain capture buffer: the
    # projection bound before the forward is None.
    bound = []
    spec_metadata = SimpleNamespace(
        project_captures=True, num_capture_layers=6, bind_capture_projection=bound.append
    )
    worker.prepare_managed_draft_cache(draft_model, spec_metadata, attn_metadata, None)
    assert bound == [None]


def test_seed_counts_are_logged_at_milestones():
    logged = [n for n in range(1, 3001) if pages_mod._seed_count_logged(n)]
    assert logged[:3] == [1, 16, 32] and 80 in logged and 1008 in logged
    assert 1024 not in logged and logged[-3:] == [1008, 2000, 3000]


def test_context_only_server_detection():
    # A context-only drafter, or a full drafter on a cache that serves the context phase only and
    # holds draft pages.
    detect = _Worker._context_only_server

    def cache(context_only, draft_pages):
        return SimpleNamespace(
            kv_cache_manager=SimpleNamespace(
                kda_context_only=context_only, has_standalone_draft_cache=draft_pages
            )
        )

    assert detect(SimpleNamespace(context_kv_only=True), cache(False, True))
    assert detect(SimpleNamespace(context_kv_only=False), cache(True, True))
    assert not detect(SimpleNamespace(context_kv_only=False), cache(False, True))
    assert not detect(SimpleNamespace(), SimpleNamespace(kv_cache_manager=None))
    # Without draft pages (the drafter context transfer not enabled) a full drafter on a
    # context-phase cache fills its own context, as on any other server.
    assert not detect(SimpleNamespace(context_kv_only=False), cache(True, False))


def test_worker_rejects_real_generation_on_a_context_only_drafter():
    worker = _Worker(_Ring(1, 8, 2, 4, torch.bfloat16))
    worker._graph_dummy_id_floor = 1 << 40
    manager = SimpleNamespace(_draft_dummy_request_ids={9})
    attn_metadata = SimpleNamespace(kv_cache_manager=manager)
    spec_metadata = SimpleNamespace(request_ids=[1, 9, 5])
    worker._check_no_real_generation(spec_metadata, attn_metadata, 2, 2)  # only the dummy
    with pytest.raises(RuntimeError, match="context_kv_only"):
        worker._check_no_real_generation(spec_metadata, attn_metadata, 1, 3)


# --- The drafter window of the family manager -------------------------------------------------------


def _drafter_dir(tmp_path, **overrides):
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "num_hidden_layers": _LAYERS,
        "use_sliding_window": True,
        "sliding_window": 1024,
        "max_window_layers": _LAYERS,
        "layer_types": ["sliding_attention"] * _LAYERS,
        "is_causal": True,
        "dflash_config": {"attention_mode": "mla"},
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _dflash_spec(model):
    return SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: True), speculative_model=model
    )


def test_draft_window_from_the_drafter_checkpoint(tmp_path):
    pages_mod.dflash2_drafter_config.cache_clear()
    assert dflash2_draft_window(_dflash_spec(_drafter_dir(tmp_path))) == 1024
    full = tmp_path / "full"
    full.mkdir()
    with pytest.raises(ValueError, match="sliding_attention"):
        dflash2_draft_window(
            _dflash_spec(_drafter_dir(full, layer_types=["full_attention"] * _LAYERS))
        )
    assert dflash2_draft_window(_dflash_spec(None)) is None
    assert dflash2_draft_window(None) is None


def test_family_manager_windows_its_draft_layers(tmp_path):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
        kda_dsa_cache_manager as dsa,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        _estimate_draft_cache_size_components,
        _get_draft_layer_sizes,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.mapping import Mapping

    pages_mod.dflash2_drafter_config.cache_clear()
    spec = _dflash_spec(_drafter_dir(tmp_path))
    draft = dict(
        draft_num_layers=_LAYERS,
        draft_kv_factor=1,
        draft_num_kv_heads=1,
        draft_head_dim=_LATENT,
        draft_dtype=DataType.BF16,
        draft_scratch_tokens=5,
    )
    kwargs = dict(draft, spec_config=spec)
    dsa._set_draft_window(kwargs)
    assert kwargs["draft_window_size"] == 1024
    no_draft = {"spec_config": spec}
    dsa._set_draft_window(no_draft)
    assert "draft_window_size" not in no_draft
    # The static estimate swaps the full-history draft term for the windowed one.
    sizes = _get_draft_layer_sizes(**draft)
    common = dict(generation_capacity_headroom=1, scratch_tokens=5, helix_cp_size=1)
    _, full_slope, full_fixed = _estimate_draft_cache_size_components(sizes, 64, **common)
    _, win_slope, win_fixed = _estimate_draft_cache_size_components(
        sizes, 64, window_size=1024, **common
    )
    slope, fixed = dsa._window_draft_cost(
        1000 + full_slope,
        7 + 3 * full_fixed,
        tokens_per_block=64,
        max_batch_size=3,
        mapping=Mapping(),
        **kwargs,
    )
    assert (slope, fixed) == (1000 + win_slope, 7 + 3 * win_fixed)
    assert win_slope < full_slope


# --- The K3.1 hybrid KV cache manager with windowed draft layers (GPU) ------------------------------


def _hybrid_manager(
    window,
    draft_latent,
    tokens_per_block=32,
    *,
    reuse=False,
    policy="per_request",
    offsets_from_start=(),
    max_seq_len=512,
    max_tokens=2048,
    max_batch_size=2,
    max_num_tokens=512,
    host_cache_size=None,
):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (  # noqa: E501
        KimiK3DSAHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import (
        BlockReuseConfig,
        DeepSeekSparseAttentionConfig,
        KvCacheConfig,
        MambaStateConfig,
    )
    from tensorrt_llm.mapping import Mapping

    text = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )
    return KimiK3DSAHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=2,
        mamba_layer_mask=[True, False, True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=reuse,
            enable_partial_reuse=False,
            block_reuse_config=BlockReuseConfig(policy=policy, max_num_turns=1),
            host_cache_size=host_cache_size,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0,
                additional_snapshot_offsets_from_end=[0],
                additional_snapshot_offsets_from_start=list(offsets_from_start),
            ),
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        layer_mask=[False, True, False, True],
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=text,
        draft_num_layers=_LAYERS,
        draft_kv_factor=1,
        draft_num_kv_heads=1,
        draft_head_dim=draft_latent,
        draft_dtype=DataType.BF16,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=5,
        draft_window_size=window,
    )


def _draft_pool_group(manager):
    layer = manager.layer_offsets[manager._standalone_draft_layer_ids[0]]
    for group in manager.impl.pool_group_descs:
        for variant in group.slot_desc.variants:
            for coalesced in variant.coalesced_buffers:
                if any(int(buffer.layer_id) == layer for buffer in coalesced.buffer_ids):
                    return int(group.pool_group_index)
    raise AssertionError("the manager has no draft pool group")


def _draft_pages(manager, field="unavailable"):
    """Device draft pages of the manager: ``unavailable`` (locked by requests) or ``total``."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel

    stats = manager.impl.get_storage_statistics(CacheLevel(0))
    return int(getattr(stats[_draft_pool_group(manager)], field))


def _llm_request(request_id, tokens):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _run_prefill(manager, requests, chunk_cap, after_step=None):
    """Chunked prefill of every request, all of them in each step as the V2 scheduler batches them:
    chunks of at most ``chunk_cap`` tokens that end at the snapshot points. Returns the peak of the
    draft pages locked during a step (after its allocation) and after its context update."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import _get_forced_context_chunk_size

    manager.prepare_expect_snapshot_points(requests)
    during = after = 0
    while any(request.context_remaining_length > 0 for request in requests):
        batch = ScheduledRequests()
        live = [request for request in requests if request.context_remaining_length > 0]
        for request in live:
            assert manager.prepare_context(request)
            chunk = min(chunk_cap, _get_forced_context_chunk_size(request))
            assert manager.resize_context(request, num_tokens=chunk), "out of KV cache pages"
            request.context_chunk_size = chunk
            batch.append_context_request(request)
        manager.prepare_resources(batch)
        during = max(during, _draft_pages(manager))
        for request in live:
            request.context_current_position += request.context_chunk_size
        manager.update_context_resources(batch)
        after = max(after, _draft_pages(manager))
        if after_step is not None:
            after_step(live)
    return during, after


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("prompt,chunks", [(40, [40]), (300, [100, 200]), (300, [250, 50])])
def test_hybrid_manager_carries_the_window_between_managers(prompt, chunks):
    window, width, tpb = 64, 5, 32
    ctx = _hybrid_manager(window, _LATENT, tpb)
    gen = _hybrid_manager(window, _LATENT, tpb)
    try:
        # The draft layers are a pool group of their own: the MLA (DSA) layers exclude them.
        assert ctx.has_standalone_draft_cache and ctx.draft_window_size == window
        assert len(ctx._attention_local_layers) == 2
        assert ctx.get_unique_primary_pool().shape[1] == 2
        pools_ctx = pages_mod.dflash2_draft_page_pools(ctx, _LAYERS, _LATENT)
        pools_gen = pages_mod.dflash2_draft_page_pools(gen, _LAYERS, _LATENT)

        ctx.add_dummy_requests([5], token_nums=[prompt])
        starts = [sum(chunks[:i]) for i in range(len(chunks))]
        known = {}
        for start, length in zip(starts, chunks):
            planner = DraftPagePlanner(ctx, window, known)
            planner.add_chunk(5, start, length, row_base=0)
            writes = planner.writes(torch.device("cuda"))
            latents = _latents(writes.positions.tolist())
            dflash2_store_draft_pages(pools_ctx, latents, writes.pages, writes.offsets)
            planner.history_update().publish()
        valid = min(prompt, window)
        exported = ctx.export_draft_history(5)
        assert (exported["valid_length"], exported["position"]) == (valid, prompt)

        # The receiving server: the prompt as history, pages for its window only.
        gen.add_dummy_requests([5], token_nums=[prompt], is_gen=True)
        history = StandaloneDraftHistory(valid, prompt)
        (src,) = ctx.get_draft_block_indices([5], [history])
        (dst,) = gen.get_draft_block_indices([5], [history])
        for block in range((prompt - valid) // tpb, -(-prompt // tpb)):
            for layer in range(_LAYERS):
                pools_gen[layer][dst[block]].copy_(pools_ctx[layer][src[block]])
        gen.restore_draft_history(5, exported)

        ring = _Ring(2, window, width, 32, torch.bfloat16)
        dflash2_seed_ring_from_pages(
            gen, pools_gen, ring.buf, ring.table, ring.rows, 32, [RingSeed(5, 1, valid, prompt)]
        )
        positions = list(range(prompt - valid, prompt))
        expected = _latents(positions)[:, :, 0].permute(1, 0, 2).contiguous()
        assert torch.equal(ring.read(1, positions), expected)
    finally:
        ctx.shutdown()
        gen.shutdown()


# --- Draft pages under chunked prefill and on arrival (GPU) -----------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_context_chunks_hold_the_draft_window_only(policy):
    """A prompt many windows long, in chunks that do not end at the snapshot point: each chunk
    locks at most the window plus itself, the out-of-window pages leave after it, the window stays
    for the KV transfer, and the transfer's end releases it."""
    window, tpb, prompt, cap = 64, 32, 2016, 160
    manager = _hybrid_manager(
        window,
        _LATENT,
        tpb,
        reuse=True,
        policy=policy,
        offsets_from_start=[512],
        max_seq_len=4096,
        max_tokens=8192,
        max_num_tokens=cap,
    )
    try:
        window_blocks = -(-(window + 1) // tpb)
        tokens = list(range(100, 100 + prompt))
        request = _llm_request(1, tokens)
        histories = []

        def after_step(live):
            (req,) = live
            histories.append((manager.kv_cache_map[1].history_length, req.context_current_position))

        during, after = _run_prefill(manager, [request], cap, after_step)
        assert all(history == position for history, position in histories)
        assert during <= window_blocks + -(-cap // tpb) + 2
        assert after <= window_blocks + 2
        cache = manager.kv_cache_map[1]
        assert cache.num_committed_tokens == prompt
        # The window before the prompt end stays (get_draft_block_indices raises on a missing page).
        transferred = StandaloneDraftHistory(window, prompt)
        manager.get_draft_block_indices([1], [transferred])
        manager.free_resources(request)
        assert _draft_pages(manager) == 0
        if policy == "per_request":
            # The next turn reuses the prompt with the window committed at its end.
            follow = _llm_request(2, tokens + list(range(7, 57)))
            manager.prepare_expect_snapshot_points([follow])
            assert manager.prepare_context(follow)
            assert follow.context_current_position == prompt
            manager.get_draft_block_indices([2], [transferred])
            manager.free_resources(follow)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_interleaved_long_prompts_lock_their_windows_only():
    """Every slot of the batch in the middle of a long prompt at once: the draft pages hold their
    windows plus one step of chunks, inside the floor the manager sizes them for."""
    window, tpb, num, prompt, cap = 64, 32, 4, 1500, 64
    manager = _hybrid_manager(
        window,
        _LATENT,
        tpb,
        reuse=True,
        offsets_from_start=[512],
        max_seq_len=2048,
        max_tokens=16384,
        max_batch_size=num,
        max_num_tokens=num * cap,
    )
    try:
        window_blocks = -(-(window + 1) // tpb)
        requests = [
            _llm_request(10 + i, list(range(10000 * (i + 1), 10000 * (i + 1) + prompt)))
            for i in range(num)
        ]
        during, after = _run_prefill(manager, requests, cap)
        assert during <= num * (window_blocks + 2) + num * cap // tpb
        assert after <= num * (window_blocks + 2)
        assert _draft_pages(manager, "total") * tpb >= manager._draft_page_floor_tokens()
        for request in requests:
            manager.free_resources(request)
        assert _draft_pages(manager) == 0
    finally:
        manager.shutdown()


def _arrive(manager, request_ids, prompt):
    """Generation requests received from a context server: the prompt as history, its window in the
    draft pages (each position's latent from ``_latents``) and the draft history published."""
    manager.add_dummy_requests(request_ids, token_nums=[prompt + 1] * len(request_ids), is_gen=True)
    planner = DraftPagePlanner(manager, manager.draft_window_size)
    for request_id in request_ids:
        planner.add_chunk(request_id, 0, prompt, row_base=0)
    writes = planner.writes(torch.device("cuda"))
    pools = pages_mod.dflash2_draft_page_pools(manager, _LAYERS, _LATENT)
    dflash2_store_draft_pages(
        pools, _latents(writes.positions.tolist()), writes.pages, writes.offsets
    )
    planner.history_update().publish()


def _generate(manager, request_id, num_tokens):
    """``num_tokens`` accepted tokens of a generation request: the history follows them."""
    cache = manager.kv_cache_map[request_id]
    assert cache.resize(cache.capacity + num_tokens, cache.history_length + num_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_arrivals_are_seeded_once_and_late_slots_are_not():
    """A full step of 16 arrivals seeds 16 rings bit-equal to the transferred windows; an arrival
    scheduled only after others generated for a while is still seeded; a request that generated past
    its prompt end and got a new slot (it left a step) is not seeded and does not fail."""
    window, tpb, width, prompt, num = 64, 32, 5, 300, 16
    manager = _hybrid_manager(
        window, _LATENT, tpb, max_tokens=16384, max_batch_size=num + 2, max_num_tokens=512
    )
    try:
        ring = _Ring(num + 2, window, width, 32, torch.bfloat16)
        worker = _Worker(ring, num_slots=num + 2)
        draft_model = SimpleNamespace(_window=window, _num_attn_layers=_LAYERS, _head_dim=_LATENT)
        spec_metadata = SimpleNamespace(project_captures=False)
        attn_metadata = SimpleNamespace(kv_cache_manager=manager)
        first = list(range(100, 100 + num))
        _arrive(manager, first + [200], prompt)

        def step(request_ids):
            for request_id in request_ids:
                worker._assign_slot(request_id)
            worker.prepare_managed_draft_cache(draft_model, spec_metadata, attn_metadata, None)

        positions = list(range(prompt - window, prompt))
        expected = _latents(positions)[:, :, 0].permute(1, 0, 2).contiguous()
        step(first)
        assert worker._draft_page_seeded == num and worker._draft_page_late == 0
        for request_id in first:
            slot = worker._req_to_slot[request_id]
            assert worker._ctx_len_host[slot] == prompt
            assert torch.equal(ring.read(slot, positions), expected)
        # Three steps of generation by the others, then the delayed arrival's first step.
        for request_id in first:
            _generate(manager, request_id, 3 * width)
        step(first + [200])
        slot = worker._req_to_slot[200]
        assert worker._ctx_len_host[slot] == prompt
        assert torch.equal(ring.read(slot, positions), expected)
        # A request that generated a window's worth and left a step gets a new, unseeded slot.
        worker.evict(100)
        _generate(manager, 100, window + tpb)
        step(first + [200])
        assert worker._ctx_len_host[worker._req_to_slot[100]] == 0
        counts = (
            worker._draft_page_queued,
            worker._draft_page_arrived,
            worker._draft_page_seeded,
            worker._draft_page_late,
        )
        assert counts == (num + 2, num + 1, num + 1, 1)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_arrival_suspended_under_pressure_is_seeded_after_resume():
    """An arrival suspended before its first step while others take the draft pages, then resumed:
    its window comes back with it and seeds its ring bit-equal."""
    window, tpb, width, prompt = 64, 32, 5, 300
    manager = _hybrid_manager(
        window,
        _LATENT,
        tpb,
        max_tokens=4096,
        max_batch_size=8,
        max_num_tokens=512,
        host_cache_size=64 << 20,
    )
    try:
        ring = _Ring(4, window, width, 32, torch.bfloat16)
        worker = _Worker(ring, num_slots=4)
        _arrive(manager, [300], prompt)
        cache = manager.kv_cache_map[300]
        cache.suspend()
        # Others take pages until none is left while the arrival is suspended.
        others = []
        for request_id in range(400, 464):
            taken = manager.add_dummy_requests([request_id], token_nums=[400])
            if not taken:
                break
            others.extend(taken)
        assert others and len(others) < 64, "the pool never ran out"
        for request in others:
            manager.free_resources(request)
        assert cache.resume(torch.cuda.current_stream().cuda_stream)
        worker._assign_slot(300)
        worker.prepare_managed_draft_cache(
            SimpleNamespace(_window=window, _num_attn_layers=_LAYERS, _head_dim=_LATENT),
            SimpleNamespace(project_captures=False),
            SimpleNamespace(kv_cache_manager=manager),
            None,
        )
        positions = list(range(prompt - window, prompt))
        expected = _latents(positions)[:, :, 0].permute(1, 0, 2).contiguous()
        slot = worker._req_to_slot[300]
        assert worker._draft_page_seeded == 1 and worker._ctx_len_host[slot] == prompt
        assert torch.equal(ring.read(slot, positions), expected)
    finally:
        manager.shutdown()


# --- The context-only drafter's RoPE rows (GPU) ------------------------------------------------------

#: A YaRN table of the K3.1 drafter's shape: 64-wide rope slice, 1M positions.
_ROPE = dict(
    dim=64,
    base=50000.0,
    scaling_factor=32.0,
    original_max_position_embeddings=4096,
    max_position_embeddings=1 << 20,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
    mscale_all_dim=1.0,
)


class _RopeDrafter:
    def _mla_rope_kwargs(self):
        return dict(_ROPE)


def _rope_positions(num, seed=0):
    rng = np.random.default_rng(seed)
    edges = [0, 1, 4095, 4096, 32767, 32768, (1 << 20) - 1]
    return np.concatenate([edges, rng.integers(0, 1 << 20, num - len(edges))]).astype(np.int64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rope_rows_are_the_device_table_rows():
    """The rows a context-only drafter rotates with are bitwise the device table's rows."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
        build_mla_yarn_freqs_cis,
    )

    table = build_mla_yarn_freqs_cis(device="cuda", **_ROPE)
    positions = _rope_positions(8192)
    drafter = _RopeDrafter()
    rows, indices = pages_mod.dflash2_rope_rows(drafter, positions, torch.device("cuda"))
    assert indices.tolist() == list(range(len(positions)))
    expected = table[torch.from_numpy(positions).cuda()]
    assert torch.equal(
        torch.view_as_real(rows).view(torch.int32), torch.view_as_real(expected).view(torch.int32)
    )
    # The host table is built once per drafter.
    host = drafter.__dict__["_k31_host_mla_freqs"][1]
    pages_mod.dflash2_rope_rows(drafter, positions[:3], torch.device("cuda"))
    assert drafter.__dict__["_k31_host_mla_freqs"][1] is host


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num", [1, 37, 512, 8192])
def test_context_latents_from_rope_rows_are_bitwise(num):
    """``precompute_context_kv``'s norm / RoPE with the stored positions' rows and indices 0..N-1
    equals the same kernel over the device table at the positions."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
        build_mla_yarn_freqs_cis,
        dflash2_mla_ctx_norm_rope,
    )

    torch.manual_seed(num)
    table = build_mla_yarn_freqs_cis(device="cuda", **_ROPE)
    latent = torch.randn(num, _LAYERS, _LATENT, dtype=torch.bfloat16, device="cuda")
    norm = torch.randn(_LAYERS, 512, dtype=torch.bfloat16, device="cuda")
    positions = _rope_positions(max(num, 8), seed=num)[:num]
    expected = dflash2_mla_ctx_norm_rope(
        latent, norm, 1e-6, table, torch.from_numpy(positions).cuda()
    )
    rows, indices = pages_mod.dflash2_rope_rows(_RopeDrafter(), positions, torch.device("cuda"))
    out = dflash2_mla_ctx_norm_rope(latent, norm, 1e-6, rows, indices)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))

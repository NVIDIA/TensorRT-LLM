# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent CSA2 test fixtures and mathematical references."""

from copy import copy
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2ForwardState,
    CSA2Layout,
    CSA2Mode,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    INDEX_PAGE_BYTES,
    INDEX_PAGE_ROWS,
    gather_rows,
    pack_index_queries_split,
    read_index_rows,
    row_bytes,
    store_index_rows,
    store_rows,
)


def _index_pages(rows: int, device="cpu") -> torch.Tensor:
    """Zeroed native page-footer index pages holding at least ``rows`` slots."""
    pages = (max(rows, 1) + INDEX_PAGE_ROWS - 1) // INDEX_PAGE_ROWS
    return torch.zeros(pages * INDEX_PAGE_BYTES, dtype=torch.uint8, device=device).view(
        pages, INDEX_PAGE_ROWS, 1, INDEX_PAGE_BYTES // INDEX_PAGE_ROWS
    )


def _split_index_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        rows[..., :64].contiguous().view(torch.int8),
        rows[..., 64:].contiguous().view(torch.int32),
    )


def _test_pools(swa, main, index):
    """Synthetic byte pools for pure prediction tests; lifecycle has real-manager tests."""

    def write_global(owner, slots, values, keys, main_transform=None, index_transform=None):
        store_rows(main[owner], slots, values, "main", main_transform)
        store_index_rows(index[owner], slots, keys, index_transform)

    def write_layer_rows(layer, slots, values, transform=None, *, global_rows=None, index_q=None):
        store_rows(swa[layer], slots, values, "swa", transform)
        if global_rows is not None:
            write_global(*global_rows)
        return None if index_q is None else pack_index_queries_split(index_q)

    return SimpleNamespace(
        write_layer_rows=write_layer_rows,
        get_swa_buffer=lambda layer: swa[layer],
        get_main_buffer=lambda owner: main[owner],
        get_index_pages=lambda owner: index[owner],
        gather_indexer_keys=lambda owner, slots: _split_index_rows(
            read_index_rows(index[owner], slots.clamp_min(0))
        ),
        write_swa=lambda layer, slots, values, transform=None: store_rows(
            swa[layer], slots, values, "swa", transform
        ),
        write_global=write_global,
        indexers={},
    )


def _attention_reference(q, kv, valid, sink):
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * q.shape[-1] ** -0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    scores = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
    return torch.einsum("qhk,qkd->qhd", scores.softmax(-1)[..., :-1], kv.float()).to(q.dtype)


def _page_metadata(table, requests, page_size, max_positions):
    # Exercise device-independent metadata transforms without allocating native
    # compute buffers. Constructor/lifecycle coverage uses real V2 managers.
    meta = object.__new__(CSA2TrtllmMetadata)
    meta.csa2_global_page_tables = {0: table}
    meta.csa2_token_requests = requests
    meta.csa2_global_page_sizes = {0: page_size}
    meta.csa2_global_max_positions = {0: max_positions}
    meta.csa2_kv_sources = {0: 0}
    meta.reset_routing()
    return meta


def _model_metadata(
    layout,
    manager,
    swa_reads,
    swa_writes,
    table,
    requests,
    page_size,
    max_positions,
    visible,
    main_writes,
):
    meta = _page_metadata(table, requests, page_size, max_positions)
    meta.kv_cache_manager = manager
    meta.csa2_swa_indices = {i: swa_reads for i in range(len(layout.compress_ratios))}
    meta.csa2_swa_write_slots = {i: swa_writes for i in range(len(layout.compress_ratios))}
    meta.csa2_visible_lengths = {i: visible for i in range(len(layout.compress_ratios))}
    meta.csa2_kv_sources = {
        i: layout.layer(i).kv_source for i in range(len(layout.compress_ratios))
    }
    meta.csa2_main_write_slots = {0: main_writes}
    # These prediction fixtures allow arbitrary visibility per query. Treat
    # each row as a context chunk; real packed request phases are tested with
    # the manager-backed runtime metadata.
    meta.csa2_request_query_ranges = tuple((i, i + 1) for i in range(len(requests)))
    meta.csa2_request_start_positions = tuple(
        max_positions * layout.compress_ratios[0] - 1 for _ in requests
    )
    meta.csa2_num_context_requests = len(requests)
    meta.mapping = None
    meta.is_cuda_graph = False
    return meta


def _run_indexer(layout, layer_idx, state):
    meta = state.metadata
    manager = meta.kv_cache_manager
    layer = layout.layer(layer_idx)
    meta.enter_layer(layer)
    manager.write_swa(
        layer_idx, meta.csa2_swa_write_slots[layer_idx], state.swa_kv, state.swa_transform
    )
    if layer.mode == CSA2Mode.FULL:
        manager.write_global(
            layer.kv_source,
            meta.csa2_main_write_slots[layer.kv_source],
            state.main_kv,
            state.index_k,
            state.main_transform,
            state.index_transform,
        )
    if layer.mode == CSA2Mode.REUSE:
        if layer.index_source not in meta.csa2_indices:
            raise ValueError("CSA2 index source did not run")
        return meta.csa2_indices[layer.index_source]
    if layer_idx not in manager.indexers:
        manager.indexers[layer_idx] = CSA2Indexer(
            layout, layer_idx, state.index_q.shape[1], state.index_q.shape[-1]
        )
    if not state.index_q.is_cuda:
        from tensorrt_llm._torch.modules.top_k import TopKImplementation

        manager.indexers[layer_idx].top_k.prefill_implementation = TopKImplementation.TORCH
        manager.indexers[layer_idx].top_k.decode_implementation = TopKImplementation.TORCH
    return manager.indexers[layer_idx](state, 0, state.swa_kv.shape[0])


def _prediction_reference(layout, layer, q, swa, sink, meta, **kwargs):
    state = CSA2ForwardState(metadata=meta, swa_kv=swa, **kwargs)
    logical = _run_indexer(layout, layer, state)
    manager = meta.kv_cache_manager
    swa_slots = meta.csa2_swa_indices[layer]
    kv = gather_rows(manager.get_swa_buffer(layer), swa_slots, q.shape[-1], "swa")
    valid = swa_slots >= 0
    slots = meta.global_slot_tile(layer, 0, q.shape[0], logical)
    kv = torch.cat(
        (
            kv,
            gather_rows(
                manager.get_main_buffer(layout.layer(layer).kv_source), slots, q.shape[-1], "main"
            ),
        ),
        dim=1,
    )
    valid = torch.cat((valid, slots >= 0), dim=1)
    return _attention_reference(q, kv, valid, sink)


def _selection_state(queries, keys, visible, topk=3):
    width = keys.shape[0]
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        0,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        index_topk=topk,
        window_size=1,
    )
    manager = _test_pools(
        {
            i: torch.zeros(max(queries, 1), row_bytes(128, "swa"), dtype=torch.uint8)
            for i in range(2)
        },
        {0: torch.zeros(max(width, 1), row_bytes(128, "main"), dtype=torch.uint8)},
        {0: _index_pages(width)},
    )
    meta = _model_metadata(
        layout,
        manager,
        torch.arange(queries)[:, None],
        torch.arange(queries),
        torch.arange(width)[None, :],
        torch.zeros(queries, dtype=torch.int64),
        1,
        width,
        visible,
        torch.arange(width),
    )
    state = CSA2ForwardState(
        metadata=meta,
        swa_kv=torch.zeros(queries, 128, dtype=torch.bfloat16),
        index_q=torch.ones(queries, 1, 128, dtype=torch.bfloat16),
        index_weights=torch.ones(queries, 1),
        main_kv=torch.zeros(width, 128, dtype=torch.bfloat16),
        index_k=keys,
    )
    return layout, state


def _run_modes(device, graph_replay=False):
    layout = CSA2Layout(
        (1, 1, 1),
        (0,),
        (0, 2),
        0,
        candidate_topk_blocks=2,
        candidate_block_size=1,
        index_topk=1,
        window_size=1,
    )
    dim = 128
    cache = _test_pools(
        {
            i: torch.zeros(1, row_bytes(dim, "swa"), dtype=torch.uint8, device=device)
            for i in range(3)
        },
        {0: torch.zeros(2, row_bytes(dim, "main"), dtype=torch.uint8, device=device)},
        {0: _index_pages(2, device)},
    )
    base = _model_metadata(
        layout,
        cache,
        torch.tensor([[0]], device=device),
        torch.tensor([0], device=device),
        torch.tensor([[0, 1]], device=device),
        torch.tensor([0], device=device),
        1,
        2,
        torch.tensor([2], device=device),
        torch.tensor([0, 1], device=device),
    )
    q = torch.zeros(1, 1, dim, dtype=torch.bfloat16, device=device)
    swa = torch.full((1, dim), 3.0, dtype=torch.bfloat16, device=device)
    main = torch.stack(
        (torch.full((dim,), 3.0, device=device), torch.full((dim,), 6.0, device=device))
    ).bfloat16()
    key = torch.stack((torch.ones(dim, device=device), -torch.ones(dim, device=device))).bfloat16()
    iq = torch.ones(1, 1, dim, dtype=torch.bfloat16, device=device)
    weights = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    sink = torch.zeros(1, device=device)

    def run():
        meta = copy(base)
        meta.reset_routing()
        outputs = []
        for i in range(3):
            kwargs = dict(index_q=iq if i == 0 else -iq, index_weights=weights) if i != 1 else {}
            if i == 0:
                kwargs.update(main_kv=main, index_k=key)
            outputs.append(_prediction_reference(layout, i, q, swa * (i + 1), sink, meta, **kwargs))
        return outputs, meta

    if not graph_replay:
        outputs, routing = run()
        return outputs, routing, cache, base
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs, routing = run()
    for sign, visible, reverse in (
        (1.0, 2, False),
        (-1.0, 2, True),
        (1.0, 1, False),
    ):
        iq.fill_(sign)
        base.csa2_visible_lengths[0].fill_(visible)
        page_slots = base.csa2_global_page_tables[0]
        page_slots.copy_(torch.tensor([[1, 0] if reverse else [0, 1]], device=device))
        graph.replay()
        actual = [o.clone() for o in outputs]
        actual_indices = routing.csa2_indices[0].clone()
        expected, expected_routing = run()
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)
        torch.testing.assert_close(actual_indices, expected_routing.csa2_indices[0], atol=0, rtol=0)
    if graph_replay:
        for request_id in (-1, 1, 0):
            base.csa2_token_requests.fill_(request_id)
            base.csa2_global_page_tables[0][0, 1] = -1
            graph.replay()
            actual = [o.clone() for o in outputs]
            expected, _ = run()
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
    return outputs, routing, cache, base


WINDOW = 128


class _FakeManager:
    def __init__(self, enable_block_reuse: bool):
        self.enable_block_reuse = enable_block_reuse
        self.layout = SimpleNamespace(window_size=WINDOW)


class _FakeMapping:
    def __init__(self, enable_attention_dp: bool):
        self.enable_attention_dp = enable_attention_dp


class _FakeKvParams:
    def __init__(self, num_cached):
        self.num_cached_tokens_per_seq = list(num_cached)


class _FakeMetadata:
    """The attributes ``plan_decoder_replay`` reads, plus the SWA floors and slot resolution.

    A real ``CSA2TrtllmMetadata`` needs a KV cache manager, pools
    and a device; the planner is pure host arithmetic over these fields, so
    standing them up here keeps this a CPU test and keeps the refusals -- which
    are the part worth pinning -- independent of cache-manager plumbing.

    ``prepare()`` is modelled too, because the boundary's correctness depends on
    what it clears; see the method.
    """

    reset_routing = CSA2TrtllmMetadata.reset_routing

    def __init__(
        self,
        seq_lens,
        num_contexts,
        num_cached=None,
        *,
        captured=False,
        attention_dp=False,
        swa_reuse=False,
    ):
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.seq_lens_cuda = self.seq_lens
        self.num_contexts = num_contexts
        self.num_seqs = len(seq_lens)
        self.num_tokens = int(sum(seq_lens))
        self.is_cuda_graph = captured
        self.mapping = _FakeMapping(attention_dp)
        self.kv_cache_manager = _FakeManager(swa_reuse)
        self.kv_cache_params = _FakeKvParams(num_cached or [0] * len(seq_lens))
        self.csa2_candidates = {}
        self.csa2_candidate_blocks = {}
        self.csa2_candidate_counts = {}
        self.csa2_indices = {}
        self._csa2_last_layer = None
        self.prepare_calls = 0
        self.prompt_lens = list(seq_lens)
        self.kv_lens_cuda = self.seq_lens + torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq
        )
        self.csa2_token_requests = torch.repeat_interleave(
            torch.arange(self.num_seqs), self.seq_lens
        )
        self.csa2_replay_start_positions = torch.zeros(self.num_seqs, dtype=torch.int64)
        self._csa2_swa_resolved = False

    def prepare(self):
        """Only the part of ``prepare()`` the boundary has to survive.

        The real one rebuilds every device index buffer; what matters to
        ``enter_decoder_replay`` is the tail of it (``metadata.py:753-755``), which
        drops all three cross-layer handoffs. Modelling just that keeps the
        boundary testable on CPU and keeps the test honest about *why* the
        reinstatement exists.
        """
        self.prepare_calls += 1
        self.csa2_indices.clear()
        self.csa2_candidates.clear()
        self.csa2_candidate_blocks.clear()
        self.csa2_candidate_counts.clear()
        self.csa2_token_requests = torch.repeat_interleave(
            torch.arange(self.num_seqs), self.seq_lens
        )
        self.csa2_positions = torch.cat(
            [
                torch.arange(start, start + length)
                for start, length in zip(
                    self.kv_cache_params.num_cached_tokens_per_seq, self.seq_lens.tolist()
                )
            ]
        )
        self.csa2_replay_start_positions = torch.zeros(self.num_seqs, dtype=torch.int64)
        self._csa2_swa_read_floors = None
        self._csa2_swa_resolved = False

    def _ensure_swa_slots(self):
        """Identity-page model of the forward-time SWA resolution, honouring replay floors."""
        if self._csa2_swa_resolved:
            return
        self._csa2_swa_resolved = True
        floor = self.csa2_replay_start_positions[self.csa2_token_requests]
        read_floors = getattr(self, "_csa2_swa_read_floors", None)
        read_floor = floor if read_floors is None else read_floors[self.csa2_token_requests]
        logical = self.csa2_positions[:, None] - WINDOW + 1 + torch.arange(WINDOW)
        slots = torch.where((logical >= floor[:, None]) & (logical >= 0), logical, -1)
        self.csa2_swa_indices = {0: slots.masked_fill(logical < read_floor[:, None], -1)}
        self.csa2_swa_write_slots = {0: slots[:, -1]}
        self.csa2_visible_lengths = {0: self.csa2_positions + 1}

    def on_update_kv_lens(self):
        self._csa2_swa_resolved = False
        lengths = self.seq_lens.tolist()
        starts = self.kv_lens_cuda[: self.num_seqs] - self.seq_lens
        self.csa2_positions = torch.cat(
            [torch.arange(s, s + n) for s, n in zip(starts.tolist(), lengths)]
        )

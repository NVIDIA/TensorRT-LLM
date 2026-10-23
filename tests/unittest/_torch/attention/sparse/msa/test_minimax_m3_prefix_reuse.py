# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax-M3 prefix reuse on the MSA path: one mixed hit/miss case with graph refresh.

Component-level: the real ``MiniMaxM3KVCacheManagerV2`` with block reuse enabled and
the MSA attention metadata, at the per-rank M3 geometry (60 sparse layers, 1 KV head,
head and index dim 128, 128-token pages). No checkpoint is loaded. K, V and index-K are
written with the backend's fused cache writer from sentinel tensors that encode
(writer, layer, role, page ordinal, offset, step), so every cache read can be checked
for exact equality.

Flow, mirroring ``KVCacheV2Scheduler`` and ``PyExecutor`` call order:

1. A 2311-token publisher prefills, commits and stays live in generation.
2. Its 2433-token extension and an unrelated 2177-token miss are scheduled together.
   Expected exact reuse: 2311 for the extension, 0 for the miss.
3. Sentinels at all 60 layers verify the page tables (shared full pages, disjoint miss
   pages), suffix placement, that the publisher's pages are untouched by the extension
   and the miss, and that the publisher's first decode write does not reach the
   extension's copy of the partial page.
4. Two real decode rows are prepared in a CUDA-graph metadata, the write kernel is
   captured and replayed, then the rows are reversed with new tags and replayed again.
5. The extension is released; the graph replays with the survivor plus the graph's
   padding row. The released pages stay unchanged and the pool's storage counts return
   to their initial values once every request is freed.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import pytest
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, GPU_LEVEL

LlmRequestState = tensorrt_llm.bindings.LlmRequestState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")

# Per-rank M3 geometry (TP4/EP4 deployment): every target layer is sparse and carries
# K, V and index-K, which is what makes the coalesced pool page 60 * 3 * 128 * 128 * 2
# = 5,898,240 bytes.
NUM_LAYERS = 60
NUM_KV_HEADS = 1
HEAD_DIM = 128
INDEX_DIM = 128
TOKENS_PER_BLOCK = 128
PAGE_BYTES = NUM_LAYERS * 3 * TOKENS_PER_BLOCK * HEAD_DIM * 2
POOL_PAGES = 72
MAX_SEQ_LEN = 2560
MAX_BATCH_SIZE = 4
GRAPH_BATCH = 2

PUBLISHER_LEN = 2311
EXTENSION_LEN = 2433
MISS_LEN = 2177

PUBLISHER_ID, EXTENSION_ID, MISS_ID, PADDING_ID = 11, 12, 13, 99
# Writer codes stored in the sentinel's first element.
PUBLISHER_CODE, EXTENSION_CODE, MISS_CODE, PADDING_CODE = 1, 2, 3, 4
ROLE_K, ROLE_V, ROLE_INDEX_K = 0, 1, 2

# Step tags stored in the sentinel's sixth element.
TAG_PUBLISHER_PREFILL = 1
TAG_MIXED_PREFILL = 2
TAG_PUBLISHER_DECODE = 3
TAG_GRAPH_FORWARD = 4
TAG_GRAPH_REVERSED = 5
TAG_GRAPH_SURVIVOR = 6


def _sentinel(code: int, layer: int, role: int, positions: list[int], tag: int) -> torch.Tensor:
    """One ``[len(positions), HEAD_DIM]`` BF16 row per position; all entries are small
    integers, which BF16 represents exactly."""
    rows = torch.zeros((len(positions), HEAD_DIM), dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.float32)
    rows[:, 0] = code
    rows[:, 1] = layer
    rows[:, 2] = role
    rows[:, 3] = pos % TOKENS_PER_BLOCK
    rows[:, 4] = pos // TOKENS_PER_BLOCK
    rows[:, 5] = tag
    return rows.to(torch.bfloat16).cuda()


def _make_manager() -> MiniMaxM3KVCacheManagerV2:
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=POOL_PAGES * PAGE_BYTES,
            host_cache_size=0,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            copy_on_partial_reuse=True,
            use_kv_cache_manager_v2=True,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        max_num_tokens=MAX_SEQ_LEN,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        vocab_size=65536,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="bf16", fuse_qkv_index_projection=False
        ),
        sparse_layer_ids=list(range(NUM_LAYERS)),
        sparse_index_dim=INDEX_DIM,
    )


def _make_request(request_id: int, tokens: list[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=tokens,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _page_table(manager: MiniMaxM3KVCacheManagerV2, request_id: int) -> list[int]:
    """Slot ids of the request's pages, in block order, without the padding entries."""
    return [
        slot for slot in manager.get_batch_cache_indices([request_id])[0] if slot != BAD_PAGE_INDEX
    ]


def _layer_views(
    manager: MiniMaxM3KVCacheManagerV2, layer: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[num_slots, 1, TOKENS_PER_BLOCK, dim]`` HND views of K, V and index-K, as the
    backend's ``write_layer_caches`` resolves them."""
    kv = manager.get_buffers(layer, kv_layout="HND")
    idx = manager.get_index_k_buffer(layer)
    assert idx is not None, f"layer {layer} has no index-K cache"
    return kv[:, 0], kv[:, 1], idx


def _gather(view: torch.Tensor, page_table: list[int], positions: list[int]) -> torch.Tensor:
    pages = torch.tensor([page_table[p // TOKENS_PER_BLOCK] for p in positions], device="cuda")
    within = torch.tensor([p % TOKENS_PER_BLOCK for p in positions], device="cuda")
    return view[pages, 0, within, :]


@dataclass
class _Ledger:
    """What each request's cache must read back: position -> (writer code, step tag)."""

    entries: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)

    def record(self, request_id: int, code: int, positions: list[int], tag: int) -> None:
        self.entries.setdefault(request_id, {}).update({p: (code, tag) for p in positions})

    def share(self, request_id: int, from_request_id: int, upto: int) -> None:
        """The reused prefix reads back the publisher's writes."""
        source = self.entries[from_request_id]
        self.entries.setdefault(request_id, {}).update(
            {p: source[p] for p in range(upto) if p in source}
        )


def _check_request(
    manager: MiniMaxM3KVCacheManagerV2, request_id: int, ledger: _Ledger, label: str
) -> None:
    """Every recorded position of the request reads back its sentinel at all layers."""
    expected_by_pos = ledger.entries[request_id]
    positions = sorted(expected_by_pos)
    page_table = _page_table(manager, request_id)
    assert len(page_table) * TOKENS_PER_BLOCK > positions[-1], (
        f"{label}: request {request_id} has {len(page_table)} pages for position {positions[-1]}"
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for p in positions:
        groups.setdefault(expected_by_pos[p], []).append(p)
    for layer in range(NUM_LAYERS):
        views = _layer_views(manager, layer)
        for role, view in enumerate(views):
            got = _gather(view, page_table, positions)
            expected = torch.cat(
                [_sentinel(code, layer, role, group, tag) for (code, tag), group in groups.items()]
            )
            order = torch.tensor(
                [p for group in groups.values() for p in group], device="cuda"
            ).argsort()
            expected = expected[order]
            if not torch.equal(got, expected):
                bad = (got != expected).any(dim=1).nonzero().flatten().tolist()
                first = bad[0]
                raise AssertionError(
                    f"{label}: request {request_id} layer {layer} role {role}: "
                    f"{len(bad)} of {len(positions)} positions differ; position "
                    f"{positions[first]} read {got[first, :6].float().tolist()} expected "
                    f"{expected[first, :6].float().tolist()}"
                )


def _storage_counts(manager: MiniMaxM3KVCacheManagerV2) -> list[tuple[int, int, int, int, int]]:
    return [
        (int(s.total), int(s.free), int(s.evictable), int(s.available), int(s.unavailable))
        for s in manager.impl.get_storage_statistics(GPU_LEVEL)
    ]


def _prepare(
    metadata: MiniMaxM3MsaSparseAttentionMetadata,
    rows: list[tuple[LlmRequest, int, int]],
    num_contexts: int,
) -> None:
    """Stage one step: ``rows`` are (request, query tokens, cached tokens) in batch order."""
    metadata.request_ids = [req.py_request_id for req, _, _ in rows]
    metadata.prompt_lens = [req.prompt_len for req, _, _ in rows]
    metadata.num_contexts = num_contexts
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[cached for _, _, cached in rows]
    )
    metadata.seq_lens = torch.tensor([q for _, q, _ in rows], dtype=torch.int)
    metadata.prepare()


def _expected_slots(
    manager: MiniMaxM3KVCacheManagerV2, rows: list[tuple[int, list[int]]]
) -> list[int]:
    """The slot per new token the staged ``msa_out_cache_loc`` must hold, from the page tables."""
    slots = []
    for request_id, positions in rows:
        table = _page_table(manager, request_id)
        slots.extend(
            table[p // TOKENS_PER_BLOCK] * TOKENS_PER_BLOCK + p % TOKENS_PER_BLOCK
            for p in positions
        )
    return slots


def _write_eager(
    manager: MiniMaxM3KVCacheManagerV2,
    metadata: MiniMaxM3MsaSparseAttentionMetadata,
    rows: list[tuple[int, list[int]]],
    tag: int,
) -> None:
    """Write this step's sentinels through the staged slots, as ``write_layer_caches`` does."""
    num_tokens = sum(len(positions) for _, positions in rows)
    out_cache_loc = metadata.msa_out_cache_loc[:num_tokens]
    for layer in range(NUM_LAYERS):
        k_view, v_view, idx_view = _layer_views(manager, layer)
        sources = [
            torch.cat([_sentinel(code, layer, role, positions, tag) for code, positions in rows])
            for role in (ROLE_K, ROLE_V, ROLE_INDEX_K)
        ]
        assert fused_write_layer_caches(k_view, v_view, idx_view, out_cache_loc, *sources), (
            f"fused writer declined layer {layer}"
        )


def _schedule_context(manager: MiniMaxM3KVCacheManagerV2, request: LlmRequest) -> int:
    """One scheduling pass for a context request; returns the reused token count."""
    assert manager.prepare_context(request), f"prepare_context failed for {request.py_request_id}"
    reused = manager.kv_cache_map[request.py_request_id].num_committed_tokens
    assert manager.resize_context(request, request.context_remaining_length), (
        f"resize_context failed for {request.py_request_id}"
    )
    return reused


def _finish_context(manager: MiniMaxM3KVCacheManagerV2, requests: list[LlmRequest]) -> None:
    """Post-forward bookkeeping: advance the chunk, commit the blocks, enter generation."""
    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
        request.move_to_next_context_chunk()
        assert request.context_remaining_length == 0
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
    manager.update_context_resources(batch)
    for request in requests:
        request.add_new_token(7, 0)


def _finish_generation(manager: MiniMaxM3KVCacheManagerV2, requests: list[LlmRequest]) -> None:
    batch = ScheduledRequests()
    for request in requests:
        request.add_new_token(7, 0)
        batch.generation_requests.append(request)
    manager.update_resources(batch)


class _GraphWriter:
    """The per-layer fused cache write for ``GRAPH_BATCH`` decode rows, captured once.

    Sources are static buffers refilled before each replay; the slots are read from the
    metadata's graph-stable ``msa_out_cache_loc``, which ``prepare()`` refreshes in place.
    """

    def __init__(
        self, manager: MiniMaxM3KVCacheManagerV2, metadata: MiniMaxM3MsaSparseAttentionMetadata
    ) -> None:
        self.manager = manager
        self.metadata = metadata
        self.sources = [
            [
                torch.zeros((GRAPH_BATCH, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
                for _ in range(3)
            ]
            for _ in range(NUM_LAYERS)
        ]
        self.graph = torch.cuda.CUDAGraph()
        self.slot_ptr = metadata.msa_out_cache_loc.data_ptr()

    def _launch(self) -> None:
        out_cache_loc = self.metadata.msa_out_cache_loc[:GRAPH_BATCH]
        for layer in range(NUM_LAYERS):
            k_view, v_view, idx_view = _layer_views(self.manager, layer)
            assert fused_write_layer_caches(
                k_view, v_view, idx_view, out_cache_loc, *self.sources[layer]
            )

    def capture(self) -> None:
        # Warm up with every slot negative so the compile pass writes nothing.
        staged = self.metadata.msa_out_cache_loc[:GRAPH_BATCH].clone()
        self.metadata.msa_out_cache_loc[:GRAPH_BATCH].fill_(-1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._launch()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self._launch()
        self.metadata.msa_out_cache_loc[:GRAPH_BATCH].copy_(staged)

    def replay(self, rows: list[tuple[int, list[int]]], tag: int) -> None:
        """``rows`` are (writer code, positions) per graph row, padding rows excluded."""
        assert self.metadata.msa_out_cache_loc.data_ptr() == self.slot_ptr, "slot buffer moved"
        for layer in range(NUM_LAYERS):
            for role in (ROLE_K, ROLE_V, ROLE_INDEX_K):
                values = torch.cat(
                    [_sentinel(code, layer, role, positions, tag) for code, positions in rows]
                )
                self.sources[layer][role][: values.shape[0]].copy_(values)
        self.graph.replay()
        torch.cuda.synchronize()


def test_mixed_hit_miss_prefix_reuse_with_graph_refresh() -> None:
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    manager = _make_manager()
    ledger = _Ledger()
    try:
        assert manager.enable_block_reuse
        assert manager.tokens_per_block == TOKENS_PER_BLOCK
        assert sorted(manager.sparse_layer_ids) == list(range(NUM_LAYERS))
        initial_free_blocks = manager.get_num_free_blocks()
        initial_counts = _storage_counts(manager)

        # The graph's padding row, allocated once up front as the CUDA-graph runner does,
        # so its page is fixed before the extension exists.
        padding = manager.add_dummy_requests([PADDING_ID], is_gen=True)[0]
        padding_table = _page_table(manager, PADDING_ID)
        assert len(padding_table) == 1

        eager = MiniMaxM3MsaSparseAttentionMetadata(
            max_num_requests=MAX_BATCH_SIZE,
            max_num_tokens=MAX_SEQ_LEN,
            kv_cache_manager=manager,
            mapping=manager.mapping,
        )

        # 1. Publisher prefill, commit, live in generation.
        publisher_tokens = [1000 + i for i in range(PUBLISHER_LEN)]
        publisher = _make_request(PUBLISHER_ID, publisher_tokens)
        assert _schedule_context(manager, publisher) == 0
        assert publisher.context_current_position == 0
        publisher_positions = list(range(PUBLISHER_LEN))
        _prepare(eager, [(publisher, PUBLISHER_LEN, 0)], num_contexts=1)
        assert eager.msa_out_cache_loc[:PUBLISHER_LEN].tolist() == _expected_slots(
            manager, [(PUBLISHER_ID, publisher_positions)]
        )
        _write_eager(manager, eager, [(PUBLISHER_CODE, publisher_positions)], TAG_PUBLISHER_PREFILL)
        _finish_context(manager, [publisher])
        ledger.record(PUBLISHER_ID, PUBLISHER_CODE, publisher_positions, TAG_PUBLISHER_PREFILL)
        publisher_table = _page_table(manager, PUBLISHER_ID)
        assert len(publisher_table) == -(-PUBLISHER_LEN // TOKENS_PER_BLOCK) == 19
        assert manager.kv_cache_map[PUBLISHER_ID].num_committed_tokens == PUBLISHER_LEN
        _check_request(manager, PUBLISHER_ID, ledger, "after publisher prefill")

        # 2. Extension (hit) and miss scheduled together while the publisher is live.
        extension = _make_request(
            EXTENSION_ID,
            publisher_tokens + [3000 + i for i in range(EXTENSION_LEN - PUBLISHER_LEN)],
        )
        miss = _make_request(MISS_ID, [50000 + i for i in range(MISS_LEN)])
        reused_extension = _schedule_context(manager, extension)
        reused_miss = _schedule_context(manager, miss)
        assert (reused_extension, reused_miss) == (PUBLISHER_LEN, 0), (
            f"exact reuse expected (2311, 0), got ({reused_extension}, {reused_miss})"
        )
        print(f"reuse: extension={reused_extension} miss={reused_miss}")
        assert extension.context_current_position == PUBLISHER_LEN
        assert extension.context_chunk_size == EXTENSION_LEN - PUBLISHER_LEN
        assert miss.context_current_position == 0

        extension_table = _page_table(manager, EXTENSION_ID)
        miss_table = _page_table(manager, MISS_ID)
        full_pages = PUBLISHER_LEN // TOKENS_PER_BLOCK  # 18
        assert extension_table[:full_pages] == publisher_table[:full_pages], (
            "the reused full pages must be the publisher's pages"
        )
        assert len(extension_table) == -(-EXTENSION_LEN // TOKENS_PER_BLOCK) == 20
        assert len(miss_table) == -(-MISS_LEN // TOKENS_PER_BLOCK) == 18
        assert not set(miss_table) & (set(publisher_table) | set(extension_table)), (
            "the miss must not share pages with the publisher or the extension"
        )
        assert not set(miss_table) & set(padding_table)
        assert not set(extension_table) & set(padding_table)

        # 3. Mixed prefill: extension suffix plus the whole miss, one step.
        suffix_positions = list(range(PUBLISHER_LEN, EXTENSION_LEN))
        miss_positions = list(range(MISS_LEN))
        _prepare(
            eager,
            [(extension, len(suffix_positions), PUBLISHER_LEN), (miss, MISS_LEN, 0)],
            num_contexts=2,
        )
        assert eager.msa_out_cache_loc[
            : len(suffix_positions) + MISS_LEN
        ].tolist() == _expected_slots(
            manager, [(EXTENSION_ID, suffix_positions), (MISS_ID, miss_positions)]
        )
        _write_eager(
            manager,
            eager,
            [(EXTENSION_CODE, suffix_positions), (MISS_CODE, miss_positions)],
            TAG_MIXED_PREFILL,
        )
        _finish_context(manager, [extension, miss])
        ledger.share(EXTENSION_ID, PUBLISHER_ID, PUBLISHER_LEN)
        ledger.record(EXTENSION_ID, EXTENSION_CODE, suffix_positions, TAG_MIXED_PREFILL)
        ledger.record(MISS_ID, MISS_CODE, miss_positions, TAG_MIXED_PREFILL)
        _check_request(manager, EXTENSION_ID, ledger, "after mixed prefill")
        _check_request(manager, MISS_ID, ledger, "after mixed prefill")
        _check_request(manager, PUBLISHER_ID, ledger, "publisher after mixed prefill")

        # Publisher's first decode writes position 2311 into its own partial page. If that
        # page were shared with the extension, the extension's position 2311 would change.
        assert manager.try_allocate_generation(publisher)
        _prepare(eager, [(publisher, 1, PUBLISHER_LEN)], num_contexts=0)
        assert eager.msa_out_cache_loc[:1].tolist() == _expected_slots(
            manager, [(PUBLISHER_ID, [PUBLISHER_LEN])]
        )
        _write_eager(manager, eager, [(PUBLISHER_CODE, [PUBLISHER_LEN])], TAG_PUBLISHER_DECODE)
        _finish_generation(manager, [publisher])
        ledger.record(PUBLISHER_ID, PUBLISHER_CODE, [PUBLISHER_LEN], TAG_PUBLISHER_DECODE)
        _check_request(manager, PUBLISHER_ID, ledger, "publisher after its decode")
        _check_request(manager, EXTENSION_ID, ledger, "extension after publisher decode")
        _check_request(manager, MISS_ID, ledger, "miss after publisher decode")
        print(
            f"page tables: publisher={publisher_table} extension={extension_table} "
            f"miss={miss_table} padding={padding_table}"
        )

        # Checker sensitivity: one corrupted element must be reported at its position
        # (page 5, offset 9 -> position 649) and nowhere else.
        k_view, _, _ = _layer_views(manager, 17)
        probe = (publisher_table[5], 0, 9, 5)
        saved = k_view[probe].clone()
        k_view[probe] = 0
        with pytest.raises(
            AssertionError, match=r"layer 17 role 0: 1 of 2312 positions differ; position 649 "
        ):
            _check_request(manager, PUBLISHER_ID, ledger, "sensitivity")
        k_view[probe] = saved
        _check_request(manager, PUBLISHER_ID, ledger, "after sensitivity restore")

        # 4. Two real decode rows under a CUDA graph: capture, replay, reverse the rows.
        graph_meta = MiniMaxM3MsaSparseAttentionMetadata(
            max_num_requests=GRAPH_BATCH,
            max_num_tokens=GRAPH_BATCH,
            kv_cache_manager=manager,
            mapping=manager.mapping,
            is_cuda_graph=True,
        )
        writer = _GraphWriter(manager, graph_meta)

        publisher_pos = publisher.max_beam_num_tokens - 1  # 2312
        extension_pos = extension.max_beam_num_tokens - 1  # 2433
        assert (publisher_pos, extension_pos) == (PUBLISHER_LEN + 1, EXTENSION_LEN)
        assert manager.try_allocate_generation(publisher)
        assert manager.try_allocate_generation(extension)
        _prepare(
            graph_meta,
            [(publisher, 1, publisher_pos), (extension, 1, extension_pos)],
            num_contexts=0,
        )
        writer.capture()
        block_table_ptr = graph_meta.msa_block_table.data_ptr()
        assert (
            graph_meta.msa_block_table[0, :19].tolist() == _page_table(manager, PUBLISHER_ID)[:19]
        )
        assert (
            graph_meta.msa_block_table[1, :20].tolist() == _page_table(manager, EXTENSION_ID)[:20]
        )
        assert graph_meta.msa_out_cache_loc[:GRAPH_BATCH].tolist() == _expected_slots(
            manager, [(PUBLISHER_ID, [publisher_pos]), (EXTENSION_ID, [extension_pos])]
        )
        writer.replay(
            [(PUBLISHER_CODE, [publisher_pos]), (EXTENSION_CODE, [extension_pos])],
            TAG_GRAPH_FORWARD,
        )
        _finish_generation(manager, [publisher, extension])
        ledger.record(PUBLISHER_ID, PUBLISHER_CODE, [publisher_pos], TAG_GRAPH_FORWARD)
        ledger.record(EXTENSION_ID, EXTENSION_CODE, [extension_pos], TAG_GRAPH_FORWARD)
        for request_id in (PUBLISHER_ID, EXTENSION_ID, MISS_ID):
            _check_request(manager, request_id, ledger, "after graph replay")

        # Reversed row order, new tags: the refreshed staging must route each row's write.
        publisher_pos += 1
        extension_pos += 1
        assert manager.try_allocate_generation(extension)
        assert manager.try_allocate_generation(publisher)
        _prepare(
            graph_meta,
            [(extension, 1, extension_pos), (publisher, 1, publisher_pos)],
            num_contexts=0,
        )
        assert graph_meta.msa_block_table.data_ptr() == block_table_ptr
        assert (
            graph_meta.msa_block_table[0, :20].tolist() == _page_table(manager, EXTENSION_ID)[:20]
        )
        assert (
            graph_meta.msa_block_table[1, :19].tolist() == _page_table(manager, PUBLISHER_ID)[:19]
        )
        writer.replay(
            [(EXTENSION_CODE, [extension_pos]), (PUBLISHER_CODE, [publisher_pos])],
            TAG_GRAPH_REVERSED,
        )
        _finish_generation(manager, [extension, publisher])
        ledger.record(PUBLISHER_ID, PUBLISHER_CODE, [publisher_pos], TAG_GRAPH_REVERSED)
        ledger.record(EXTENSION_ID, EXTENSION_CODE, [extension_pos], TAG_GRAPH_REVERSED)
        for request_id in (PUBLISHER_ID, EXTENSION_ID, MISS_ID):
            _check_request(manager, request_id, ledger, "after reversed graph replay")

        # 5. Release the extension; replay with the survivor and the padding row.
        extension_table = _page_table(manager, EXTENSION_ID)
        released = sorted(set(extension_table) - set(_page_table(manager, PUBLISHER_ID)))
        assert released, "the extension must own pages beyond the shared prefix"
        counts_before_release = _storage_counts(manager)
        manager.free_resources(extension)
        assert EXTENSION_ID not in manager.kv_cache_map
        counts_after_release = _storage_counts(manager)
        print(
            f"released pages={released} storage (total, free, evictable, available, "
            f"unavailable): initial={initial_counts} before_release={counts_before_release} "
            f"after_release={counts_after_release}"
        )
        assert sum(c[4] for c in counts_before_release) - sum(
            c[4] for c in counts_after_release
        ) == len(released), (
            f"unavailable pages before {counts_before_release} after {counts_after_release}"
        )
        snapshot = [
            [view[released].clone() for view in _layer_views(manager, layer)]
            for layer in range(NUM_LAYERS)
        ]

        publisher_pos += 1
        assert manager.try_allocate_generation(publisher)
        _prepare(graph_meta, [(publisher, 1, publisher_pos), (padding, 1, 0)], num_contexts=0)
        assert graph_meta.msa_out_cache_loc[:GRAPH_BATCH].tolist() == _expected_slots(
            manager, [(PUBLISHER_ID, [publisher_pos]), (PADDING_ID, [0])]
        )
        writer.replay([(PUBLISHER_CODE, [publisher_pos]), (PADDING_CODE, [0])], TAG_GRAPH_SURVIVOR)
        _finish_generation(manager, [publisher])
        ledger.record(PUBLISHER_ID, PUBLISHER_CODE, [publisher_pos], TAG_GRAPH_SURVIVOR)
        ledger.record(PADDING_ID, PADDING_CODE, [0], TAG_GRAPH_SURVIVOR)
        _check_request(manager, PUBLISHER_ID, ledger, "survivor after release replay")
        _check_request(manager, MISS_ID, ledger, "miss after release replay")
        _check_request(manager, PADDING_ID, ledger, "padding row after release replay")
        for layer in range(NUM_LAYERS):
            for role, view in enumerate(_layer_views(manager, layer)):
                # Compare bit patterns: BF16 equality would treat NaN bytes as
                # changed and +0/-0 as unchanged.
                assert torch.equal(
                    view[released].view(torch.int16), snapshot[layer][role].view(torch.int16)
                ), f"released pages {released} changed at layer {layer} role {role}"

        # Free-block accounting returns to its initial value.
        for request in (publisher, miss, padding):
            manager.free_resources(request)
        assert not manager.kv_cache_map.keys() - {k for k in manager.kv_cache_map if k < 0}
        assert manager.get_num_free_blocks() == initial_free_blocks
        final_counts = _storage_counts(manager)
        print(
            f"free blocks: initial={initial_free_blocks} final={manager.get_num_free_blocks()} "
            f"storage final={final_counts}"
        )
        assert [c[4] for c in final_counts] == [c[4] for c in initial_counts], (
            f"unavailable pages: initial {initial_counts}, final {final_counts}"
        )
        assert [c[3] for c in final_counts] == [c[3] for c in initial_counts]
        manager.impl.clear_reusable_blocks()
        assert _storage_counts(manager) == initial_counts
    finally:
        manager.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

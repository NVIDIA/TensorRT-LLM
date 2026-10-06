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

"""Sparse offload metadata, attention integration, and CUDA graph replay tests."""

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
    DeepseekV4TrtllmAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    check_sparse_read_table,
    merge_sparse_read_table,
    select_sparse_history_pages,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.hooks import prepare_sparse_runtime_params
from tensorrt_llm._torch.attention.backends.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

from .test_deepseek_v4_cache_manager import _history_case, _manager

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


# Offload metadata allocation and per-batch preparation.


def _metadata(manager, *, enabled=True, graph=False, buffers=None, initialize=True):
    # Exercise the production offload initialization without constructing the
    # unrelated indexer/compressor/native attention workspaces.
    metadata = object.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata.kv_cache_manager = manager
    metadata.sparse_metadata_params = DeepSeekV4SparseAttentionConfig(
        enable_kv_cache_offload=enabled, index_topk=8
    ).to_sparse_metadata_params()
    metadata.max_num_sequences = 4
    metadata.sparse_mla_topk = 8
    metadata.beam_width = 1
    metadata.max_draft_tokens = 0
    metadata.draft_kv_cache_manager = None
    metadata.is_cuda_graph = graph
    metadata.cuda_graph_buffers = buffers
    if initialize:
        metadata._init_sparse_offload_state()
    return metadata


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_csa", [False, True])
@pytest.mark.parametrize(
    "unsupported",
    ["beam", "configured_beam", "draft", "configured_draft", "draft_tree", "draft_cache"],
)
def test_sparse_offload_rejects_configuration_before_allocation(
    unsupported: str, local_csa: bool
) -> None:
    manager = _manager()
    if not local_csa:
        manager.pp_layers = [3]
    metadata = _metadata(manager, initialize=False)
    if unsupported == "beam":
        metadata.beam_width = 2
    elif unsupported == "configured_beam":
        manager.max_beam_width = 2
    elif unsupported == "draft":
        metadata.max_draft_tokens = 1
    elif unsupported == "configured_draft":
        manager.max_draft_len = 1
    elif unsupported == "draft_tree":
        manager.max_total_draft_tokens = 1
    else:
        metadata.draft_kv_cache_manager = object()
    with patch.object(metadata, "get_empty") as allocate:
        with pytest.raises(NotImplementedError, match="single-beam, non-speculative"):
            metadata._init_sparse_offload_state()
        allocate.assert_not_called()
    assert metadata.sparse_offload_state is None


@pytest.mark.cpu_only
def test_sparse_offload_disabled_allocates_nothing():
    manager = _manager()
    with patch.object(manager, "get_sparse_offload_descriptors") as describe:
        metadata = _metadata(manager, enabled=False)
    assert metadata.sparse_offload_state is None
    describe.assert_not_called()


@pytest.mark.cpu_only
def test_sparse_offload_stage_without_local_csa_allocates_nothing():
    manager = _manager()
    manager.pp_layers = [3]
    manager.impl = SimpleNamespace()  # No sparse API lookup is needed on this stage.
    assert _metadata(manager).sparse_offload_state is None


@_requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_sparse_offload_buffers_are_persistent_and_separate(graph):
    manager = _manager()
    buffers = Buffers() if graph else None
    metadata = _metadata(manager, graph=graph, buffers=buffers)
    state = metadata.sparse_offload_state
    assert state.history_blocks_host.is_pinned()
    assert state.selected_history_pages.shape == (4, 6)
    tables = [*state.base_page_tables.values(), state.fetched_page_table, state.compress_read_table]
    assert all(table.shape == (4, 6) for table in tables)
    device_tensors = tables + [
        state.selected_history_pages,
        state.history_blocks,
        state.active_request_count,
    ]
    assert all(tensor.is_cuda and tensor.dtype == torch.int32 for tensor in device_tensors)
    assert len({tensor.data_ptr() for tensor in device_tensors}) == len(device_tensors)
    assert all((table == -1).all() for table in tables)
    assert (state.history_blocks == 0).all() and state.active_request_count.item() == 0
    if graph:
        # Same-name graph buffers can be reused for serial graph buckets.
        reused = _metadata(manager, graph=True, buffers=buffers).sparse_offload_state
        assert reused.base_page_tables[13].data_ptr() == state.base_page_tables[13].data_ptr()
        assert reused.compress_read_table.data_ptr() == state.compress_read_table.data_ptr()
        assert reused.history_blocks_host.data_ptr() != state.history_blocks_host.data_ptr()


@_requires_cuda
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_prepare_sparse_offload_request_order_and_boundaries(tokens_per_block):
    manager = _manager(tokens_per_block)
    manager._stream = torch.cuda.current_stream()
    metadata = _metadata(manager)
    state = metadata.sparse_offload_state
    write = torch.empty((4, 6), device="cuda", dtype=torch.int32)
    pointers = [t.data_ptr() for t in (state.history_blocks, state.base_page_tables[13], write)]
    prepare_stream = torch.cuda.Stream()
    for ids in (
        [30, 10, 20],
        [20, CUDA_GRAPH_DUMMY_REQUEST_ID, CUDA_GRAPH_DUMMY_REQUEST_ID],
        [],
        [10],
    ):
        state.fetched_page_table.fill_(999)
        state.compress_read_table.fill_(999)
        state.selected_history_pages.fill_(999)
        prepare_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(prepare_stream):
            manager.prepare_sparse_offload(state, ids, write, beam_width=1)
        torch.cuda.current_stream().wait_stream(prepare_stream)
        live_ids = [req for req in ids if req != CUDA_GRAPH_DUMMY_REQUEST_ID]
        expected_raw = torch.full((4, 6), -1, dtype=torch.int32)
        expected_write = torch.full_like(expected_raw, -1)
        expected_history = torch.zeros(4, dtype=torch.int32)
        for row, req in enumerate(live_ids):
            cache = manager.kv_cache_map[req]
            count = cache.history_length // tokens_per_block
            expected_history[row] = count
            expected_raw[row, : cache.num_blocks] = torch.tensor(cache.pages)
            for ordinal, page in enumerate(cache.pages):
                if ordinal >= count and page >= 0:
                    expected_write[row, ordinal] = page * 5
        torch.testing.assert_close(state.base_page_tables[13].cpu(), expected_raw)
        torch.testing.assert_close(write.cpu(), expected_write)
        torch.testing.assert_close(state.history_blocks.cpu(), expected_history)
        assert state.active_request_count.item() == len(live_ids)
        assert all(
            (t == -1).all()
            for t in (
                state.selected_history_pages,
                state.fetched_page_table,
                state.compress_read_table,
            )
        )
        if live_ids:
            assert manager.impl.calls[-1][:2] == (live_ids, 13)
            assert manager.impl.calls[-1][3] == prepare_stream.cuda_stream
        assert pointers == [
            t.data_ptr() for t in (state.history_blocks, state.base_page_tables[13], write)
        ]
        # Change the frontier and physical assignment without replacing tensors.
        manager.kv_cache_map[10].history_length = 2 * tokens_per_block
        manager.kv_cache_map[10].pages = [17, 19, 0]
    assert len(manager.impl.calls) == 3  # The empty batch does not call KVCM.


@_requires_cuda
@pytest.mark.parametrize(
    "invalid",
    ["batch", "duplicates", "dummy_order", "unknown", "width", "frontier", "beam", "capture"],
)
def test_sparse_offload_rejects_invalid_preparation(invalid):
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    state = _metadata(manager).sparse_offload_state
    write = torch.empty((4, 6), device="cuda", dtype=torch.int32)
    ids = [10]
    beam = 1
    if invalid == "batch":
        ids = [10] * 5
    elif invalid == "duplicates":
        ids = [10, 10]
    elif invalid == "dummy_order":
        ids = [CUDA_GRAPH_DUMMY_REQUEST_ID, 10]
    elif invalid == "unknown":
        ids = [12345]
    elif invalid == "width":
        manager.kv_cache_map[10].num_blocks = 7
    elif invalid == "frontier":
        manager.kv_cache_map[10].history_length = 99999
    elif invalid == "beam":
        beam = 2
    with patch("torch.cuda.is_current_stream_capturing", return_value=invalid == "capture"):
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            manager.prepare_sparse_offload(state, ids, write, beam_width=beam)
    assert not manager.impl.calls
    assert state.active_request_count.item() == 0


@_requires_cuda
@pytest.mark.parametrize("enabled", [False, True])
def test_metadata_routes_only_sparse_main_table(enabled):
    manager = _manager()
    manager._enable_kv_cache_offload = enabled
    manager._stream = torch.cuda.current_stream()
    metadata = _metadata(manager, enabled=enabled)
    metadata.request_ids = [20, 10]
    metadata._seq_lens = torch.ones(2, dtype=torch.int32)
    metadata._num_contexts = 1
    metadata.sliding_block_tables = torch.empty((1, 1, 4, 6), dtype=torch.int32, device="cuda")
    metadata.compress_block_tables = {
        ratio: torch.empty((4, 6), dtype=torch.int32, device="cuda") for ratio in (4, 128)
    }
    manager.copy_batch_sliding_block_tables = Mock(side_effect=lambda out, *args: out.fill_(71))
    manager.copy_batch_compress_block_tables = Mock(
        side_effect=lambda out, *args, **kwargs: out.fill_(29)
    )
    metadata.prepare_for_block_tables()
    assert (metadata.sliding_block_tables == 71).all()
    assert (metadata.compress_block_tables[128] == 29).all()
    ratios = [
        call.kwargs["compress_ratio"]
        for call in manager.copy_batch_compress_block_tables.call_args_list
    ]
    assert ratios == ([128] if enabled else [4, 128])
    if enabled:
        assert manager.impl.calls[-1][0] == [20, 10]
        assert metadata.compress_block_tables[4][0, 0].item() == -1
        assert metadata.compress_block_tables[4][0, 1].item() == 0
    else:
        assert metadata.sparse_offload_state is None
        assert not manager.impl.calls


@_requires_cuda
def test_sparse_metadata_refresh_before_graph_replay():
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    state = _metadata(manager, graph=True, buffers=Buffers()).sparse_offload_state
    write = torch.empty((4, 6), dtype=torch.int32, device="cuda")
    topk = torch.tensor([[0, 31, 32, 63, 64, -1, 32, 0]] * 4, dtype=torch.int32, device="cuda")
    requests = torch.arange(4, dtype=torch.int32, device="cuda")
    lengths = torch.full((4,), 96, dtype=torch.int32, device="cuda")

    def consume():
        select_sparse_history_pages(
            topk,
            requests,
            state.active_request_count,
            lengths,
            state.base_page_tables[13],
            state.history_blocks,
            32,
            state.selected_history_pages,
        )
        # No fetch is simulated here. History remains absent, and only the
        # prepared resident table can supply valid addresses to the merger.
        merge_sparse_read_table(
            state.fetched_page_table,
            write,
            state.history_blocks,
            state.active_request_count,
            state.compress_read_table,
            fetched_page_scale=1,
        )

    manager.prepare_sparse_offload(state, [20, 10], write, beam_width=1)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        consume()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        consume()
    pointers = state.selected_history_pages.data_ptr(), state.compress_read_table.data_ptr()
    for ids in ([20, 10], [30, CUDA_GRAPH_DUMMY_REQUEST_ID], [], [10, 30]):
        manager.prepare_sparse_offload(state, ids, write, beam_width=1)
        graph.replay()
        expected = torch.full((4, 6), -1, dtype=torch.int32)
        live = [req for req in ids if req != CUDA_GRAPH_DUMMY_REQUEST_ID]
        for row, req in enumerate(live):
            cache = manager.kv_cache_map[req]
            pages = sorted(
                {
                    token // 32
                    for token in topk[row].cpu().tolist()
                    if 0 <= token < 96
                    and token // 32 < cache.history_length // 128
                    and cache.pages[token // 32] >= 0
                }
            )
            expected[row, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        torch.testing.assert_close(state.selected_history_pages.cpu(), expected)
        torch.testing.assert_close(state.compress_read_table, write)
        assert pointers == (
            state.selected_history_pages.data_ptr(),
            state.compress_read_table.data_ptr(),
        )
        manager.kv_cache_map[30].history_length = 2 * 128
        manager.kv_cache_map[30].pages = [0, 16, 3]


# Sparse attention fetch and consumption.


def _host_batch(*, prefill=False):
    manager = _manager()
    manager.compute_sliding_block_tables = Mock()
    for cache in manager.kv_cache_map.values():
        cache.history_length = 0
    return SimpleNamespace(
        sparse_metadata_params=SimpleNamespace(enable_kv_cache_offload=True),
        sparse_offload_state=SimpleNamespace(prepared=True, is_prefill=False),
        kv_cache_manager=manager,
        request_ids=[20, 10],
        beam_width=1,
        max_draft_tokens=0,
        draft_kv_cache_manager=None,
        num_contexts=2 if prefill else 0,
        num_generations=0 if prefill else 2,
        seq_lens=torch.tensor([8, 12] if prefill else [1, 1], dtype=torch.int32),
        seq_lens_kv=torch.tensor([8, 12] if prefill else [1, 1], dtype=torch.int32),
        prompt_lens=[8, 12],
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0, 0]),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefill", [False, True])
def test_offload_admits_decode_or_fresh_prefill(prefill):
    metadata = _host_batch(prefill=prefill)
    DeepseekV4TrtllmAttentionMetadata.validate_sparse_offload_batch(metadata)
    assert not metadata.sparse_offload_state.prepared
    assert metadata.sparse_offload_state.is_prefill == prefill


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "unsupported",
    [
        "mixed",
        "multiquery",
        "cached",
        "chunk",
        "history",
        "kv_lengths",
    ],
)
def test_offload_rejects_batch_before_table_preparation(unsupported):
    metadata = _host_batch(prefill=unsupported in ("cached", "chunk", "history"))
    if unsupported == "mixed":
        metadata.num_contexts = metadata.num_generations = 1
    elif unsupported == "multiquery":
        metadata.seq_lens[:] = metadata.seq_lens_kv[:] = 2
    elif unsupported == "cached":
        metadata.kv_cache_params.num_cached_tokens_per_seq[0] = 1
    elif unsupported == "chunk":
        metadata.prompt_lens[0] += 8
    elif unsupported == "history":
        metadata.kv_cache_manager.kv_cache_map[20].history_length = 1
    else:
        metadata.seq_lens_kv[0] += 1
    metadata.validate_sparse_offload_batch = lambda: (
        DeepseekV4TrtllmAttentionMetadata.validate_sparse_offload_batch(metadata)
    )
    with pytest.raises(NotImplementedError):
        DeepseekV4TrtllmAttentionMetadata.prepare(metadata)
    metadata.kv_cache_manager.compute_sliding_block_tables.assert_not_called()
    assert not metadata.sparse_offload_state.prepared


@triton.jit
def _fetch_double_kernel(
    raw,
    selected,
    history,
    out,
    source,
    pool,
    M: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    OFFSET: tl.constexpr,
    FETCH_SCALE: tl.constexpr,
):
    row, j = tl.program_id(0), tl.program_id(1)
    ordinal = tl.load(selected + row * S + j)
    valid = (ordinal >= 0) & (ordinal < tl.load(history + row))
    host_slot = tl.load(raw + row * M + ordinal, mask=valid, other=0)
    # Positional scratch lives above all resident slots; pages interleave
    # layers at scale 5. Keep the original row position within every page.
    physical_page = (16 + row * S + j) * 5
    x = tl.arange(0, P * D)
    value = tl.load(source + host_slot * P * D + x, mask=valid, other=0)
    tl.store(pool + (OFFSET + physical_page * P) * D + x, value, mask=valid)
    tl.store(out + row * M + ordinal, physical_page // FETCH_SCALE, mask=valid)


class _FetchDouble:
    def __init__(self, state, pool, sources, offsets, page_tokens, calls):
        self.state, self.pool, self.sources = state, pool, sources
        self.offsets, self.page_tokens, self.calls = offsets, page_tokens, calls

    def __call__(self, *, buffers, page_table, selected, num_blocks, out, stream):
        assert len(buffers) == 1
        descriptor = next(d for d in self.state.layers.values() if d.buffer_id == buffers[0])
        assert page_table is self.state.base_page_tables[descriptor.group_id]
        assert selected is self.state.selected_history_pages
        assert num_blocks is self.state.history_blocks
        assert out is self.state.fetched_page_table
        assert stream == torch.cuda.current_stream().cuda_stream
        self.calls.append((buffers[0], stream))
        out.fill_(-1)
        _fetch_double_kernel[(selected.shape[0], selected.shape[1])](
            page_table,
            selected,
            num_blocks,
            out,
            self.sources[buffers[0].layer_id],
            self.pool,
            page_table.shape[1],
            selected.shape[1],
            self.page_tokens,
            self.pool.shape[1],
            self.offsets[buffers[0].layer_id],
            descriptor.fetched_page_scale,
        )


def _attention_case(*, tokens_per_block=128, fetched_page_scale=1, fp8=False):
    manager = _manager(tokens_per_block)
    manager._stream = torch.cuda.current_stream()
    manager.dtype = DataType.FP8 if fp8 else DataType.BF16
    manager.layer_offsets = {2: 0, 3: 1, 4: 2}
    metadata = _metadata(manager)
    state = metadata.sparse_offload_state
    state.layers = {
        layer: replace(d, fetched_page_scale=fetched_page_scale)
        for layer, d in state.layers.items()
    }
    p, dim = tokens_per_block // 4, 16
    manager.kv_cache_map[20].history_length = 2 * tokens_per_block - 1
    manager.kv_cache_map[20].pages = [0, 3, -1]
    manager.kv_cache_map[30].history_length = 2 * tokens_per_block + 3
    manager.kv_cache_map[30].pages = [8, 7, 0]
    manager.kv_cache_map[10].pages = [4, -1, -1]
    metadata.request_ids = [20, 30, 10, CUDA_GRAPH_DUMMY_REQUEST_ID]
    metadata._num_contexts = metadata._num_ctx_tokens = 0
    metadata._num_generations = metadata._num_tokens = 4
    metadata.req_idx_per_token = torch.tensor([1, 0, 2, 3], dtype=torch.int32, device="cuda")
    metadata.compress_block_tables = {4: torch.empty((4, 6), dtype=torch.int32, device="cuda")}
    metadata.compressed_kv_lens_cuda = {
        4: torch.tensor([2 * p, 2 * p + 1, p, 0], dtype=torch.int32, device="cuda")
    }
    metadata.max_compressed_indices = {1: 0, 4: 8, 128: 8}
    metadata.sparse_mla_topk_lens = {
        r: torch.full((4,), 136, dtype=torch.int32, device="cuda") for r in (1, 4, 128)
    }
    metadata.swa_local_indices_cuda = torch.full((4, 128), -1, dtype=torch.int32, device="cuda")
    metadata.swa_local_indices_cuda[:3, :2] = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    metadata.sliding_block_tables = torch.zeros((3, 1, 4, 6), dtype=torch.int32, device="cuda")
    # This pool is only read through returned addresses; the double supplies
    # device source pages to stand in for host history, not an H2D transport.
    pool = (
        torch.arange(64 * 5 * p * dim, device="cuda", dtype=torch.float32).reshape(-1, dim) / 10000
    )
    offsets = {41: p, 19: 3 * p}
    sources = {
        bid: torch.arange(16 * p * dim, device="cuda", dtype=torch.float32).reshape(-1, dim) / 10000
        + bid
        for bid in offsets
    }
    calls = []
    manager.impl.fetch_sparse_pages = _FetchDouble(state, pool, sources, offsets, p, calls)
    stride = dim * (1 if fp8 else 2)
    metadata.sparse_mla_base_ptrs = {1: 100000, 4: pool.data_ptr(), 128: 200000}
    metadata.swa_buffer_ptrs = {
        2: 100000 + 7 * stride,
        3: 100000 + 9 * stride,
        4: 100000 + 11 * stride,
    }
    metadata.compressed_buffer_ptrs = {
        layer: pool.data_ptr() + offsets[d.buffer_id.layer_id] * stride
        for layer, d in state.layers.items()
    }
    config = DeepSeekV4SparseAttentionConfig(enable_kv_cache_offload=True, index_topk=8)
    backends = []
    for layer in state.layers:
        backend = object.__new__(DeepseekV4TrtllmAttention)
        backend.layer_idx, backend.compress_ratio, backend.head_dim = layer, 4, dim
        backend.sparse_attention_config = config
        backend.sparse_params = config.to_sparse_params()
        backend.use_fp8_ds_mla = backend._uses_nvfp4_compress = False
        backend.q_scaling, backend.qk_nope_head_dim, backend.qk_rope_head_dim = 1.0, dim, 0
        backend.kv_scale_quant_orig = torch.tensor([0.25], device="cuda")
        backends.append(backend)
    topk = torch.tensor(
        [
            [2 * p, p + 1, 0, p + 1, 2 * p - 1, p - 1, -1, -1],
            [2 * p - 1, p - 1, 0, p, p - 1, -1, -1, -1],
            [p - 1, 0, 0, -1, -1, -1, -1, -1],
            [0] * 8,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk),
    )
    if fp8:
        forward_args.fmha_scheduler_counter = torch.full((1,), 42, dtype=torch.int32, device="cuda")
        forward_args.mla_bmm1_scale = torch.empty(2, device="cuda")
        forward_args.mla_bmm2_scale = torch.empty(1, device="cuda")
    manager.prepare_sparse_offload(
        state, metadata.request_ids, metadata.compress_block_tables[4], beam_width=1
    )
    return SimpleNamespace(
        manager=manager,
        metadata=metadata,
        state=state,
        backends=backends,
        args=forward_args,
        pool=pool,
        sources=sources,
        offsets=offsets,
        calls=calls,
        p=p,
    )


def _consume(case, backend):
    backend._prepare_sparse_forward_args(case.metadata, case.args)
    return prepare_sparse_runtime_params(
        backend, case.pool[:4], None, None, case.metadata, case.args
    )


def _check_consumption(case, backend, runtime):
    metadata, state = case.metadata, case.state
    descriptor = state.layers[backend.layer_idx]
    offset = case.offsets[descriptor.buffer_id.layer_id]
    raw = state.base_page_tables[descriptor.group_id].cpu()
    history = state.history_blocks.cpu()
    topk = case.args.sparse_backend_args.topk_indices.cpu()
    requests = metadata.req_idx_per_token.cpu().tolist()
    active = state.active_request_count.item()
    indices = runtime.sparse_attn_indices.cpu()
    fetched = state.fetched_page_table.cpu()
    assert runtime.sparse_attn_offsets is None
    assert runtime.aux_kv_cache_pool_ptr == metadata.sparse_mla_base_ptrs[4]
    assert runtime.sparse_attn_kv_lens.data_ptr() == metadata.sparse_mla_topk_lens[4].data_ptr()
    assert runtime.sparse_attn_indices_block_size == backend.sparse_params.indices_block_size
    assert indices.shape == (4, 136)
    expected_indices = torch.full_like(topk, -1)
    for query, row in enumerate(requests):
        if query >= active:
            continue
        for j, token in enumerate(topk[query].tolist()):
            if token < 0:
                continue
            ordinal, position = divmod(token, case.p)
            if ordinal < history[row]:
                physical_page = int(fetched[row, ordinal]) * descriptor.fetched_page_scale
                assert physical_page >= 0  # Every referenced history page must be staged.
                expected_value = case.sources[descriptor.buffer_id.layer_id][
                    int(raw[row, ordinal]) * case.p + position
                ]
            else:
                physical_page = int(raw[row, ordinal]) * descriptor.page_scale
                expected_value = case.pool[offset + physical_page * case.p + position]
            expected_index = offset + physical_page * case.p + position
            expected_indices[query, j] = expected_index
            torch.testing.assert_close(
                case.pool[int(indices[query, 128 + j])], expected_value, rtol=0, atol=0
            )
    torch.testing.assert_close(indices[:, 128:], expected_indices, rtol=0, atol=0)
    # Existing SWA addressing is unaffected by selecting/fetching compressed KV.
    swa_offset = {2: 7, 4: 11}[backend.layer_idx]
    torch.testing.assert_close(
        indices[:active, :2],
        torch.tensor([[swa_offset, swa_offset + 3]] * active, dtype=torch.int32).reshape(active, 2),
    )
    assert (indices[active:] == -1).all()
    assert (indices[:, 2:128] == -1).all()


@_requires_cuda
@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("fetched_page_scale", [1, 5])
def test_per_layer_fetch_reaches_sparse_attention_inputs(tokens_per_block, fetched_page_scale):
    case = _attention_case(tokens_per_block=tokens_per_block, fetched_page_scale=fetched_page_scale)
    original_topk = case.args.sparse_backend_args.topk_indices.clone()
    original_write = case.metadata.compress_block_tables[4].clone()
    for backend in case.backends:
        runtime = _consume(case, backend)
        _check_consumption(case, backend, runtime)
    assert [b.layer_id for b, _ in case.calls] == [41, 19]
    torch.testing.assert_close(case.args.sparse_backend_args.topk_indices, original_topk)
    torch.testing.assert_close(case.metadata.compress_block_tables[4], original_write)


@_requires_cuda
def test_fetch_preserves_fp8_scheduler_prologue():
    case = _attention_case(fp8=True)
    runtime = _consume(case, case.backends[0])
    _check_consumption(case, case.backends[0], runtime)
    assert case.args.fmha_scheduler_counter.item() == 0
    scale = 0.25**2 / math.sqrt(16)
    torch.testing.assert_close(
        case.args.mla_bmm1_scale, torch.tensor([scale, scale * math.log2(math.e)], device="cuda")
    )
    torch.testing.assert_close(case.args.mla_bmm2_scale, torch.tensor([0.25], device="cuda"))


@_requires_cuda
@pytest.mark.parametrize("failure", ["units", "scale", "api", "prepared", "prefill", "transfer"])
def test_sparse_fetch_fails_without_resident_fallback(failure):
    case = _attention_case()
    if failure in ("units", "scale"):
        case.state.layers[2] = replace(
            case.state.layers[2], fetched_page_scale=None if failure == "units" else 0
        )
    elif failure == "api":
        case.manager.impl.fetch_sparse_pages = None
    elif failure == "prepared":
        case.state.prepared = False
    elif failure == "prefill":
        case.state.is_prefill = True
    else:
        case.manager.impl.fetch_sparse_pages = Mock(side_effect=RuntimeError("fetch failed"))
    with pytest.raises((NotImplementedError, RuntimeError, ValueError)):
        _consume(case, case.backends[0])
    assert not case.calls


@_requires_cuda
def test_decode_fetches_even_for_zero_history_or_all_invalid_selection():
    case = _attention_case()
    for cache in case.manager.kv_cache_map.values():
        cache.history_length = 0
    case.args.sparse_backend_args.topk_indices[:3].copy_(
        torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]] * 3, dtype=torch.int32, device="cuda")
    )
    case.manager.prepare_sparse_offload(
        case.state, case.metadata.request_ids, case.metadata.compress_block_tables[4], beam_width=1
    )
    _consume(case, case.backends[0])
    assert (case.state.selected_history_pages == -1).all()
    assert (case.state.fetched_page_table == -1).all()
    torch.testing.assert_close(
        case.state.compress_read_table, case.metadata.compress_block_tables[4]
    )
    case.args.sparse_backend_args.topk_indices.fill_(-1)
    _consume(case, case.backends[1])
    assert len(case.calls) == 2


@_requires_cuda
def test_fresh_prefill_consumes_write_table_without_fetch():
    case = _attention_case()
    for cache in case.manager.kv_cache_map.values():
        cache.history_length = 0
    case.manager.prepare_sparse_offload(
        case.state, case.metadata.request_ids, case.metadata.compress_block_tables[4], beam_width=1
    )
    case.state.is_prefill = True
    case.metadata._num_contexts, case.metadata._num_generations = 2, 0
    case.metadata._num_ctx_tokens = 4
    case.args.attention_input_type = AttentionInputType.context_only
    case.args.sparse_backend_args.topk_indices.copy_(
        torch.tensor([[0, 1, 31, -1, -1, -1, -1, -1]] * 4, dtype=torch.int32, device="cuda")
    )
    # Multiple queries/request are legal only for fresh prefill, where no
    # selection union is required. Repeated ordinals retain their positions.
    case.metadata.req_idx_per_token.copy_(
        torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    )
    runtime = _consume(case, case.backends[0])
    assert not case.calls
    assert (
        runtime.sparse_attn_indices[:, 128:][case.args.sparse_backend_args.topk_indices >= 0] >= 0
    ).all()


@_requires_cuda
def test_offload_disabled_and_other_ratios_do_not_fetch():
    case = _attention_case()
    backend = case.backends[0]
    backend.sparse_attention_config.enable_kv_cache_offload = False
    _consume(case, backend)
    backend.sparse_attention_config.enable_kv_cache_offload = True
    backend.compress_ratio = 1
    runtime = _consume(case, backend)
    assert runtime.sparse_attn_indices.shape == (4, 128)
    backend.compress_ratio = 128
    case.metadata.compress_block_tables[128] = case.metadata.compress_block_tables[4]
    case.metadata.compressed_local_indices_cuda = torch.zeros(
        (4, 8), dtype=torch.int32, device="cuda"
    )
    _consume(case, backend)
    assert not case.calls


@_requires_cuda
@pytest.mark.parametrize("debug_assert", [False, True])
def test_fetch_graph_replay_refreshes_requests_topk_and_padding(monkeypatch, debug_assert):
    monkeypatch.setenv("TLLM_DSV4_OFFLOAD_DEBUG_ASSERT", "1" if debug_assert else "0")
    case = _attention_case(fetched_page_scale=5)
    backend = case.backends[0]
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        _consume(case, backend)  # Compile before capture on the consuming stream.
    torch.cuda.current_stream().wait_stream(consumer)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=consumer):
        runtime = _consume(case, backend)
    pointers = [
        tensor.data_ptr()
        for tensor in (
            case.state.selected_history_pages,
            case.state.fetched_page_table,
            case.state.compress_read_table,
        )
    ]
    for ids, requests, topk in [
        (
            [30, 20, 10],
            [0, 1, 2, 3],
            [
                [0, 32, 64, -1, -1, -1, -1, -1],
                [0, 63, -1, -1, -1, -1, -1, -1],
                [31, -1, -1, -1, -1, -1, -1, -1],
                [0] * 8,
            ],
        ),
        ([10], [0, 1, 2, 3], [[0, 31, -1, -1, -1, -1, -1, -1]] + [[0] * 8] * 3),
        ([], [0, 1, 2, 3], [[0] * 8] * 4),
    ]:
        case.metadata.request_ids = ids + [CUDA_GRAPH_DUMMY_REQUEST_ID] * (4 - len(ids))
        case.manager.prepare_sparse_offload(
            case.state,
            case.metadata.request_ids,
            case.metadata.compress_block_tables[4],
            beam_width=1,
        )
        lengths = [(case.manager.kv_cache_map[i].history_length + 1) // 4 for i in ids] + [0] * (
            4 - len(ids)
        )
        case.metadata.compressed_kv_lens_cuda[4].copy_(
            torch.tensor(lengths, dtype=torch.int32, device="cuda")
        )
        case.metadata.req_idx_per_token.copy_(
            torch.tensor(requests, dtype=torch.int32, device="cuda")
        )
        case.args.sparse_backend_args.topk_indices.copy_(
            torch.tensor(topk, dtype=torch.int32, device="cuda")
        )
        graph.replay()
        _check_consumption(case, backend, runtime)
    assert pointers == [
        tensor.data_ptr()
        for tensor in (
            case.state.selected_history_pages,
            case.state.fetched_page_table,
            case.state.compress_read_table,
        )
    ]
    assert (case.state.fetched_page_table == -1).all()


# History policy validation and stream ordering.


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "policy", ["enable_block_reuse", "kv_compression_manages_history", "_has_cp_helix", "is_draft"]
)
def test_offload_rejects_other_history_policies_during_initialization(policy: str) -> None:
    manager = _manager()
    setattr(manager, policy, True)
    with pytest.raises(NotImplementedError, match="Sparse offload requires"):
        _metadata(manager)


@_requires_cuda
@pytest.mark.parametrize("debug_assert", [False, True])
def test_missing_fetch_result_assertion_is_debug_only(monkeypatch, debug_assert) -> None:
    monkeypatch.delenv("TLLM_DSV4_OFFLOAD_DEBUG_ASSERT", raising=False)
    if debug_assert:
        monkeypatch.setenv("TLLM_DSV4_OFFLOAD_DEBUG_ASSERT", "1")
    case = _attention_case()
    assert case.state.debug_assert == debug_assert
    case.manager.impl.fetch_sparse_pages = lambda **kwargs: kwargs["out"].fill_(-1)

    # Avoid poisoning the suite's CUDA context with a deliberate device assert.
    def check(valid: torch.Tensor, message: str) -> None:
        if not valid.item():
            raise RuntimeError(message)

    with (
        patch("torch._assert_async", side_effect=check) as device_assert,
        patch(
            "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager.check_sparse_read_table",
            wraps=check_sparse_read_table,
        ) as validate,
        patch(
            "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend.deepseek_v4_local_to_global_indices"
        ) as convert,
    ):
        if debug_assert:
            with pytest.raises(RuntimeError, match="missing/out-of-bounds KV page"):
                _consume(case, case.backends[0])
            device_assert.assert_called_once()
            convert.assert_not_called()
        else:
            _consume(case, case.backends[0])
            device_assert.assert_not_called()
            convert.assert_called_once()
        validate.assert_called_once()
        assert case.state.read_table_valid.item() == 0


@_requires_cuda
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_history_demotion_waits_for_all_layer_writers_and_readers(tokens_per_block: int) -> None:
    cache_stream, model_stream = torch.cuda.Stream(), torch.cuda.Stream()
    manager, cache, request, batch = _history_case(tokens_per_block, stream=cache_stream)
    state = _metadata(manager).sparse_offload_state
    write = torch.empty((4, 6), dtype=torch.int32, device="cuda")
    p = tokens_per_block // 4
    # Two CSA layers write different data to the same logical history page.
    source = torch.zeros((2, p, 16), dtype=torch.bfloat16, device="cuda")
    host = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
    read_before_recycle = torch.empty_like(source)
    fetched = torch.empty_like(source)
    model_stream.wait_stream(torch.cuda.current_stream())
    resize = cache.resize

    def demote(capacity: int | None, history: int | None) -> bool:
        previous = cache.history_length
        result = resize(capacity, history)
        if previous < tokens_per_block <= cache.history_length:
            with torch.cuda.stream(cache_stream):
                host.copy_(source, non_blocking=True)
                source.fill_(-99)  # Simulate reusing the released resident slots.
        return result

    cache.resize = demote
    with torch.cuda.stream(model_stream):
        source[0, :-1].fill_(1)
        source[1, :-1].fill_(2)
        manager.update_context_resources(batch)
        manager.prepare_sparse_offload(state, [20], write, beam_width=1)
    model_stream.synchronize()
    assert cache.history_length == tokens_per_block - 1
    assert state.history_blocks[0].item() == 0
    assert write[0, 0].item() == 0

    events = []
    producers = [torch.cuda.Stream(), torch.cuda.Stream()]
    for layer, producer in enumerate(producers):
        producer.wait_stream(model_stream)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(2_000_000 * (layer + 1))
            source[layer, -1].fill_(layer + 3)
            events.append(producer.record_event())
    batch.context_requests, batch.generation_requests = [], [request]
    request.max_beam_num_tokens = tokens_per_block + 1
    with torch.cuda.stream(model_stream):
        for event in events:
            model_stream.wait_event(event)
        read_before_recycle.copy_(source)
        manager.update_resources(batch)
        manager.prepare_sparse_offload(state, [20], write, beam_width=1)
        fetched.copy_(host, non_blocking=True)
    model_stream.synchronize()
    assert cache.history_length == tokens_per_block
    assert state.history_blocks[0].item() == 1
    assert write[0, 0].item() == -1
    assert (source == -99).all()
    expected = torch.empty_like(source)
    expected[0, :-1], expected[1, :-1] = 1, 2
    expected[0, -1], expected[1, -1] = 3, 4
    torch.testing.assert_close(read_before_recycle, expected, rtol=0, atol=0)
    torch.testing.assert_close(fetched, expected, rtol=0, atol=0)

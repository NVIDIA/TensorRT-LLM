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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    merge_sparse_read_table,
    prepare_sparse_write_table,
    select_sparse_history_pages,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _SparseRuntime:
    """Test double for the declared APIs; no sparse transfer implementation."""

    def __init__(self):
        self.groups = {41: 13, 19: 13, 42: 9, 20: 9, 43: 5, 21: 5, 7: 14}
        self.sparse = {41, 19}
        self.converters = {
            41: SimpleNamespace(scale=5, expansion=1, layer_offset=17),
            19: SimpleNamespace(scale=5, expansion=1, layer_offset=23),
        }
        self.pool_bases = {41: 100000, 19: 100000}
        self.calls = []
        self.uploads = []

    def is_sparse(self, layer_id, role):
        return layer_id in self.sparse

    def get_layer_group_id(self, layer_id):
        return self.groups[layer_id]

    def get_page_index_converter(self, layer_id, role):
        return self.converters[layer_id]

    def get_mem_pool_base_address(self, layer_id, role, index_mode):
        assert index_mode == PageIndexMode.SHARED
        return self.pool_bases[layer_id]

    def copy_base_page_indices_to_device(self, kv_caches, layer_group_id, out, stream):
        assert stream == torch.cuda.current_stream().cuda_stream
        assert out.shape[0] == len(kv_caches)
        self.calls.append(
            ([cache.id for cache in kv_caches], layer_group_id, out.data_ptr(), stream)
        )
        table = torch.full(out.shape, -1, dtype=torch.int32, pin_memory=True)
        for row, cache in enumerate(kv_caches):
            table[row, : cache.num_blocks] = torch.tensor(cache.pages, dtype=torch.int32)
        self.uploads.append(table)  # Keep each immutable pinned source alive through its upload.
        out.copy_(table, non_blocking=True)


def _manager(tokens_per_block=128):
    manager = object.__new__(DeepseekV4CacheManager)
    manager._enable_kv_cache_offload = True
    manager._use_nvfp4_compress = False
    manager.use_fp8_ds_mla = False
    manager.pp_layers = [2, 3, 4]
    manager._compress_ratios = [1, 4, 4, 128, 4]
    manager.tokens_per_block = tokens_per_block
    manager.max_blocks_per_seq = 6
    manager._layer_attn_to_layer_id = {
        (2, DeepseekV4AttentionType.COMPRESS): 41,
        (4, DeepseekV4AttentionType.COMPRESS): 19,
        (2, DeepseekV4AttentionType.INDEXER_COMPRESS): 42,
        (4, DeepseekV4AttentionType.INDEXER_COMPRESS): 20,
        (2, DeepseekV4AttentionType.SWA): 43,
        (4, DeepseekV4AttentionType.SWA): 21,
        (3, DeepseekV4AttentionType.COMPRESS): 7,
    }
    manager.impl = _SparseRuntime()
    manager.kv_cache_map = {
        10: SimpleNamespace(id=10, history_length=tokens_per_block - 1, pages=[0, 9, -1]),
        20: SimpleNamespace(id=20, history_length=tokens_per_block, pages=[0, 0, 12]),
        30: SimpleNamespace(id=30, history_length=tokens_per_block + 1, pages=[8, 7, 0]),
    }
    for cache in manager.kv_cache_map.values():
        cache.num_blocks = len(cache.pages)
        cache.capacity = cache.num_blocks * tokens_per_block
        cache.beam_width = 1
    return manager


def _metadata(manager, *, enabled=True, graph=False, buffers=None):
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
    metadata.is_cuda_graph = graph
    metadata.cuda_graph_buffers = buffers
    metadata._init_sparse_offload_state()
    return metadata


@pytest.mark.cpu_only
def test_sparse_offload_descriptors_use_runtime_ids():
    manager = _manager()
    descriptors = manager.get_sparse_offload_descriptors()
    assert list(descriptors) == [2, 4]
    assert [d.buffer_id.layer_id for d in descriptors.values()] == [41, 19]
    for descriptor in descriptors.values():
        assert descriptor.buffer_id.role == DeepseekV4AttentionType.COMPRESS.role
        assert descriptor.group_id == 13
        assert descriptor.page_scale == 5


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "invalid",
    ["group", "scale", "pool", "expansion", "ordinary", "not_sparse", "nvfp4", "footer", "api"],
)
def test_sparse_offload_rejects_incompatible_layout(invalid):
    manager = _manager()
    if invalid == "group":
        manager.impl.groups[19] = 100
    elif invalid == "scale":
        manager.impl.converters[19].scale = 3
    elif invalid == "pool":
        manager.impl.pool_bases[19] += 128
    elif invalid == "expansion":
        manager.impl.converters[41].expansion = 2
    elif invalid == "ordinary":
        manager.impl.groups[42] = 13
    elif invalid == "not_sparse":
        manager.impl.sparse.remove(41)
    elif invalid == "nvfp4":
        manager._use_nvfp4_compress = True
    elif invalid == "footer":
        manager.use_fp8_ds_mla = True
    else:
        manager.impl = SimpleNamespace()
    with pytest.raises((ValueError, NotImplementedError)):
        manager.get_sparse_offload_descriptors()


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


@_requires_cuda
@pytest.mark.parametrize("scale", [1, 5])
def test_prepare_write_table_preserves_invalids_and_strides(scale):
    raw = torch.tensor(
        [[0, 2, -1, 1 << 30], [3, 0, 2, -9], [7, 8, 9, 10]], dtype=torch.int32, device="cuda"
    )
    storage = torch.full((6, 8), 999, dtype=torch.int32, device="cuda")
    out = storage[::2, ::2]
    history = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
    active = torch.tensor([2], dtype=torch.int32, device="cuda")
    prepare_sparse_write_table(raw, history, active, out, page_scale=scale)
    expected = [
        [-1, 2 * scale, -1, (1 << 30) if scale == 1 else -1],
        [-1, -1, 2 * scale, -1],
        [-1] * 4,
    ]
    torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int32))
    assert (storage[1::2] == 999).all() and (storage[:, 1::2] == 999).all()

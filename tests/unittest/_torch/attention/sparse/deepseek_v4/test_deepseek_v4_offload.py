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

"""Native Batch fetch adaptation and persistent DeepSeek-V4 offload metadata."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import backend as deepseek_v4_backend
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    cache_manager as deepseek_v4_cache,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    deepseek_v4_local_to_global_indices,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.attention.backends.sparse.offload import (
    NVFP4_SCRATCH_TAIL_ROWS,
    SparseOffloadState,
    _SparseBatchMember,
    allocate_sparse_offload_state,
)
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

from .test_deepseek_v4_cache_manager import _manager

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


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


@pytest.mark.cpu_only
def test_sparse_metadata_uses_native_capacity_and_configured_width():
    manager = _manager()
    metadata = _metadata(manager, initialize=False)
    metadata.max_num_sequences = 2
    with patch.object(
        metadata,
        "get_empty",
        side_effect=lambda buffers, shape, **kwargs: torch.empty(shape, dtype=torch.int32),
    ):
        metadata._init_sparse_offload_state()
    state = metadata.sparse_offload_state
    assert state.selected_history_units.shape == (4, 8)
    assert state.fetched_indices[1].shape == (4, 1, 8)
    assert state.scatter_start_positions.shape == (4,)
    assert (state.scatter_start_positions == 0).all()
    assert state.batch_rows.shape == (4,) and (state.batch_rows == -1).all()
    assert state.history_lengths is None and state.batch is None
    assert (state.selected_history_units == -1).all()


@pytest.mark.cpu_only
def test_nvfp4_scratch_keeps_a_zeroed_tail_past_its_rows():
    """The sparse MLA FMHA can read one KV tile past the last row its indices name."""
    allocated = {}

    def allocate(name, shape, dtype):
        if dtype.itemsize == 1:
            tensor = torch.full(shape, 0x7F, dtype=torch.uint8).view(dtype)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        allocated[name] = tensor
        return tensor

    layers = {2: SimpleNamespace(buffer_ids=(None, None))}
    state = allocate_sparse_offload_state(
        allocate, layers, max_batch_size=4, topk=8, nvfp4_head_dim=512
    )
    rows = 4 * 8
    flat = allocated["nvfp4_scratch"]
    assert flat.shape == (rows + NVFP4_SCRATCH_TAIL_ROWS, 512)
    assert state.nvfp4_scratch.shape == (4, 8, 512)
    assert state.nvfp4_scratch.data_ptr() == flat.data_ptr()
    assert (flat[:rows].view(torch.uint8) == 0x7F).all()
    assert (flat[rows:].view(torch.uint8) == 0).all()
    assert SparseOffloadState.device_bytes(
        max_batch_size=4, topk=8, buffer_counts={2}, nvfp4_head_dim=512
    ) == sum(tensor.numel() * tensor.element_size() for tensor in allocated.values())


@pytest.mark.cpu_only
def test_sparse_metadata_rejects_different_selection_width():
    metadata = _metadata(_manager(), initialize=False)
    metadata.sparse_mla_topk = 4
    with patch.object(metadata, "get_empty") as allocate:
        with pytest.raises(ValueError, match="configured positive top-k"):
            metadata._init_sparse_offload_state()
        allocate.assert_not_called()


@_requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_sparse_offload_buffers_are_persistent_and_separate(graph):
    manager = _manager()
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    main = next(iter(manager.impl.layers[19].buffers)).role
    manager.impl.configure_buffers(41, [(main, 1), (scale, 0)])
    buffers = Buffers() if graph else None
    metadata = _metadata(manager, graph=graph, buffers=buffers)
    state = metadata.sparse_offload_state
    tensors = [
        state.selected_history_units,
        state.active_request_count,
        state.batch_rows,
        state.scatter_start_positions,
        *state.fetched_indices.values(),
        *state.write_indices.values(),
    ]
    assert state.selected_history_units.shape == (4, 8)
    assert state.fetched_indices[1].shape == (4, 1, 8)
    assert state.fetched_indices[2].shape == (4, 2, 8)
    assert all(t.is_cuda and t.dtype == torch.int32 and t.is_contiguous() for t in tensors)
    assert len({t.data_ptr() for t in tensors}) == len(tensors)
    if graph:
        smaller = _metadata(manager, graph=True, buffers=buffers, initialize=False)
        smaller.max_num_sequences = 2
        smaller._init_sparse_offload_state()
        reused = smaller.sparse_offload_state
        assert reused.selected_history_units.data_ptr() == state.selected_history_units.data_ptr()
        assert reused.fetched_indices[2].data_ptr() == state.fetched_indices[2].data_ptr()
        assert reused.write_indices[2].data_ptr() == state.write_indices[2].data_ptr()
        assert reused.scatter_start_positions.data_ptr() == state.scatter_start_positions.data_ptr()
        assert reused.batch_rows.data_ptr() == state.batch_rows.data_ptr()


class _PublishedBatch:
    """Device watermarks with the same stable DLPack-view ownership as native Batch."""

    def __init__(self, capacity):
        self.max_batch_size = capacity
        self.history = torch.zeros(capacity, dtype=torch.int32, device="cuda")

    def history_lengths(self):
        return self.history

    def sparse_out_slots(self, buffers):
        """One slice per buffer: this double's pools are distinct, so none coincide."""
        return list(range(len(buffers)))


def _publish(manager, request_ids, rows=None):
    """Seat the live requests in ``rows``, by default in input order from row 0."""
    if manager._sparse_batch is None:
        manager._sparse_batch = _PublishedBatch(manager.max_batch_size)
    live = [request for request in request_ids if request != CUDA_GRAPH_DUMMY_REQUEST_ID]
    rows = range(len(live)) if rows is None else rows
    manager._sparse_batch_members = {
        request: _SparseBatchMember(
            cache=manager.kv_cache_map[request],
            row=row,
            blocks=manager.kv_cache_map[request].num_blocks,
        )
        for request, row in zip(live, rows, strict=True)
    }
    manager._sparse_batch_prepared = True
    values = [0] * manager.max_batch_size
    for member in manager._sparse_batch_members.values():
        values[member.row] = member.cache.history_length
    with torch.cuda.stream(manager._stream):
        manager._sparse_batch.history.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))


@_requires_cuda
def test_prepare_sparse_offload_binds_batch_without_copying_watermarks():
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    state = _metadata(manager).sparse_offload_state
    batch = None
    for request_ids in ([20, 10], [10, CUDA_GRAPH_DUMMY_REQUEST_ID], [], [20]):
        _publish(manager, request_ids)
        manager.prepare_sparse_offload(state, request_ids, beam_width=1)
        assert state.prepared
        assert state.batch is manager._sparse_batch
        if batch is not None:
            assert state.batch is batch
        batch = state.batch
        assert state.history_lengths.data_ptr() == batch.history.data_ptr()
        live = len(manager._sparse_batch_members)
        assert state.active_request_count.item() == live
        assert state.batch_rows.tolist() == list(range(live)) + [-1] * (4 - live)
        assert (state.selected_history_units == -1).all()
        assert (state.fetched_indices[1] == -1).all()
        batch.history[0] = 77
        assert state.history_lengths[0].item() == 77


@_requires_cuda
def test_prepare_sparse_offload_maps_input_positions_to_batch_rows():
    """Rows are slots: requests keep theirs whatever their input order."""
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    state = _metadata(manager).sparse_offload_state
    _publish(manager, [20, 10, 30], rows=[3, 0, 2])
    for request_ids, expected in (
        ([20, 10, 30], [3, 0, 2, -1]),
        ([30, 20, 10], [2, 3, 0, -1]),
        ([10, 30, 20, CUDA_GRAPH_DUMMY_REQUEST_ID], [0, 2, 3, -1]),
    ):
        manager.prepare_sparse_offload(state, request_ids, beam_width=1)
        assert state.batch_rows.tolist() == expected
        assert state.active_request_count.item() == 3


@_requires_cuda
@pytest.mark.parametrize(
    "invalid",
    [
        "unpublished",
        "missing",
        "stranger",
        "duplicate",
        "beam",
        "capacity",
        "replacement",
        "capture",
    ],
)
def test_sparse_offload_rejects_invalid_preparation(invalid):
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    state = _metadata(manager).sparse_offload_state
    _publish(manager, [20, 10])
    ids, beam = [20, 10], 1
    if invalid == "unpublished":
        manager._sparse_batch_prepared = False
    elif invalid == "missing":
        ids = [20, CUDA_GRAPH_DUMMY_REQUEST_ID]
    elif invalid == "stranger":
        ids = [20, 30]
    elif invalid == "duplicate":
        ids = [20, 20]
    elif invalid == "beam":
        beam = 2
    elif invalid == "capacity":
        manager._sparse_batch.max_batch_size = 5
    elif invalid == "replacement":
        manager.prepare_sparse_offload(state, ids, beam_width=1)
        manager._sparse_batch = _PublishedBatch(manager.max_batch_size)
    with patch("torch.cuda.is_current_stream_capturing", return_value=invalid == "capture"):
        with pytest.raises((ValueError, RuntimeError)):
            manager.prepare_sparse_offload(state, ids, beam_width=beam)
    assert not state.prepared


@_requires_cuda
@pytest.mark.parametrize("enabled", [False, True])
def test_metadata_routes_only_sparse_main_table(enabled):
    manager = _manager()
    manager._enable_kv_cache_offload = enabled
    manager._stream = torch.cuda.current_stream()
    metadata = _metadata(manager, enabled=enabled)
    metadata.request_ids = [20, 10]
    metadata._seq_lens = torch.ones(2, dtype=torch.int32)
    metadata._num_contexts = 0
    metadata.sliding_block_tables = torch.empty((1, 1, 4, 6), dtype=torch.int32, device="cuda")
    metadata.compress_block_tables = {
        ratio: torch.empty((4, 6), dtype=torch.int32, device="cuda") for ratio in (4, 128)
    }
    manager.copy_batch_sliding_block_tables = Mock(side_effect=lambda out, *args: out.fill_(71))
    manager.copy_batch_compress_block_tables = Mock(
        side_effect=lambda out, *args, **kwargs: out.fill_(29)
    )
    if enabled:
        _publish(manager, metadata.request_ids)
    metadata.prepare_for_block_tables()
    assert (metadata.sliding_block_tables == 71).all()
    assert (metadata.compress_block_tables[128] == 29).all()
    ratios = [
        call.kwargs["compress_ratio"]
        for call in manager.copy_batch_compress_block_tables.call_args_list
    ]
    assert ratios == ([128] if enabled else [4, 128])
    if enabled:
        assert metadata.sparse_offload_state.batch is manager._sparse_batch
        assert (metadata.compress_block_tables[4] == -1).all()
    else:
        assert metadata.sparse_offload_state is None


class _FetchDouble:
    """Compact, per-role physical indices; deliberately accepts only the native ABI."""

    def __init__(self, batch):
        # Attached as an attribute of the Batch, which does not bind a receiver, so the
        # batch the call is standing in for is supplied here instead.
        self.batch = batch
        self.calls = []

    def __call__(self, *, buffers, selected, out, stream):
        assert selected.shape == (self.batch.max_batch_size, 8)
        assert out.shape == (self.batch.max_batch_size, len(buffers), 8)
        assert selected.is_contiguous() and out.is_contiguous()
        assert stream == torch.cuda.current_stream().cuda_stream
        self.calls.append((self.batch, buffers, selected, out, stream))
        for i, buffer in enumerate(buffers):
            # Paired data/scale roles share indices; padding has separate cells.
            offset = (0 if buffer.layer_id == 41 else 1000) + (
                101 if "padding" in buffer.role else 0
            )
            out[:, i, :].copy_(torch.where(selected >= 0, selected + offset, -1))


def _fetch_case(multiple_roles=False, rows=None):
    manager = _manager()
    manager._stream = torch.cuda.current_stream()
    if multiple_roles:
        manager._use_nvfp4_compress = True
        main = manager.impl.layers[41].buffers[0].role
        manager.impl.configure_buffers(
            41,
            [
                (main, 1),
                (deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE, 0),
                ("padding", 1),
                ("scale_padding", 0),
            ],
        )
        manager.impl.configure_buffers(
            19, [(main, 1), (deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE, 0)]
        )
    manager.kv_cache_map[20].history_length = 7
    manager.kv_cache_map[10].history_length = 12
    _publish(manager, [20, 10], rows)
    state = _metadata(manager).sparse_offload_state
    manager.prepare_sparse_offload(state, [20, 10], beam_width=1)
    manager._sparse_batch.fetch_sparse_pages = _FetchDouble(manager._sparse_batch)
    topk = torch.tensor(
        [[2, -1, 0, 1, -2, 3, 99, -1], [1, 0, -1, 2, -1, -1, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    # Query 0 is request 10 at input position 1; query 1 is request 20 at position 0.
    requests = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    return SimpleNamespace(
        manager=manager, state=state, topk=topk, requests=requests, lengths=lengths
    )


def _fetch(case, layer=2):
    return case.manager.fetch_sparse_indices(
        case.state, layer, case.topk, case.requests, case.lengths
    )


def _expected_selection(manager, by_request):
    """Each member's expected selection, in its Batch row; free rows select nothing."""
    expected = torch.full((manager.max_batch_size, 8), -1, dtype=torch.int32, device="cuda")
    for request, member in manager._sparse_batch_members.items():
        expected[member.row] = torch.tensor(by_request[request], dtype=torch.int32)
    return expected


@_requires_cuda
@pytest.mark.parametrize("rows", [None, [3, 1]])
def test_prepare_writes_and_attention_merge_on_ordered_streams(rows):
    case = _fetch_case(rows=rows)
    row_20 = case.manager._sparse_batch_members[20].row
    producer = torch.cuda.Stream()
    seen = []

    def prepare(*, buffers, out, stream):
        assert out.shape == (4, 1) and out.is_contiguous()
        assert stream == producer.cuda_stream
        assert buffers == case.state.layers[2].buffer_ids
        seen.append(out.data_ptr())
        out.fill_(-1)
        # Request 20 has history 7 and capacity 8, so its unit 1 is newly complete.
        out[row_20, 0] = 77

    case.manager._sparse_batch.prepare_sparse_write_indices = prepare
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        writes = case.manager.prepare_sparse_compressor_write_indices(case.state, 2)
    torch.cuda.current_stream().wait_stream(producer)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")),
    ):
        fetched = _fetch(case)
        out = deepseek_v4_local_to_global_indices(
            req_id=case.requests,
            block_table_swa=torch.zeros((2, 1), dtype=torch.int32, device="cuda"),
            swa_local_indices=torch.full((2, 4), -1, dtype=torch.int32, device="cuda"),
            swa_pool_base_ptr=4096,
            swa_buffer_ptr=4096,
            tokens_per_block=128,
            token_stride=1024,
            compress_ratio=4,
            num_compressed_indices=8,
            compressed_local_indices=case.topk,
            compressed_fetched_indices=fetched[:, 0],
            compressed_write_indices=writes[:, 0],
            compressed_history_lengths=case.state.history_lengths,
            compressed_lengths=case.lengths,
            compressed_batch_rows=case.state.batch_rows,
            active_request_count=case.state.active_request_count,
            mask_invalid_pages=True,
        )
    assert seen == [writes.data_ptr()]
    expected = torch.tensor(
        [[[2, -1, 0, 1, -1, -1, -1, -1]], [[77, 0, -1, -1, -1, -1, -1, -1]]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(out[:, 4:], expected[:, 0])
    case.manager.prepare_sparse_offload(case.state, [20, 10], beam_width=1)
    assert case.state.write_layer is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("nvfp4", [False, True])
def test_attention_uses_arena_base_and_physical_indices_without_host_pages(nvfp4):
    manager = _manager()
    manager.dtype = DataType.BF16
    manager.layer_offsets = {2: 0}
    physical = torch.tensor([[[71, -1, 9, 0, -1, -1, -1, -1]]], dtype=torch.int32)
    if nvfp4:
        manager._use_nvfp4_compress = True
        manager._nvfp4_residual_dim = 0
        for layer_id in (41, 19):
            manager.impl.configure_buffers(
                layer_id,
                [
                    (DeepseekV4AttentionType.COMPRESS.role, 1),
                    (deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE, 0),
                ],
            )
        physical = physical.expand(-1, 2, -1).clone()
    physical = physical.expand(4, -1, -1).clone()
    manager.fetch_sparse_indices = Mock(return_value=physical)
    writes = torch.full((4, physical.shape[1], 1), 77, dtype=torch.int32)
    state = SimpleNamespace(
        layers=manager.get_sparse_offload_descriptors(),
        active_request_count=torch.tensor([1], dtype=torch.int32),
        batch_rows=torch.tensor([0, -1, -1, -1], dtype=torch.int32),
        write_indices={physical.shape[1]: writes},
        history_lengths=torch.tensor([4, 0, 0, 0], dtype=torch.int32),
        write_layer=2,
        # The mocked fetch cannot fill this, so stand in for it: one slice per buffer.
        out_slots={2: tuple(range(physical.shape[1]))},
        nvfp4_scratch=torch.empty(1, 8, 512, dtype=torch.float8_e4m3fn) if nvfp4 else None,
        coverage=None,
    )
    descriptor = state.layers[2]
    if nvfp4:
        pool_rows = descriptor.arena_row_counts[descriptor.data_buffer_index]
        manager._sparse_arena_buffers[2] = (
            torch.empty((pool_rows, 1, 256), dtype=torch.uint8),
            torch.empty((pool_rows, 1, 32), dtype=torch.float8_e4m3fn),
        )
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        sparse_offload_state=state,
        num_ctx_tokens=0,
        num_tokens=1,
        num_seqs=1,
        sparse_mla_base_ptrs={1: 4096, 4: 999999},
        swa_buffer_ptrs={2: 4096},
        compressed_buffer_ptrs={2: 888888},
        req_idx_per_token=torch.tensor([0], dtype=torch.int32),
        swa_local_indices_cuda=torch.zeros(1, 4, dtype=torch.int32),
        sliding_block_tables=torch.zeros(1, 4, 1, 1, dtype=torch.int32),
        compress_block_tables={4: None},
        compressed_kv_lens_cuda={4: torch.tensor([2], dtype=torch.int32)},
        max_compressed_indices={4: 8},
    )
    backend = SimpleNamespace(
        layer_idx=2,
        head_dim=512,
        compress_ratio=4,
        use_fp8_ds_mla=False,
        _uses_nvfp4_compress=nvfp4,
        _nvfp4_compress_scale_quant_orig=torch.ones(1),
        sparse_attention_config=SimpleNamespace(index_head_dim=128, enable_kv_cache_offload=True),
    )
    args = SimpleNamespace(
        attention_input_type=AttentionInputType.generation_only,
        sparse_backend_args=SimpleNamespace(topk_indices=torch.zeros(1, 8, dtype=torch.int32)),
        sparse_runtime_params=SimpleNamespace(aux_kv_cache_pool_ptr=0),
    )
    expected = (
        torch.tensor([[0, -1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
        if nvfp4
        else physical[:1, 0]
    )
    prepared = torch.cat((torch.tensor([[10, 11, 12, 13]]), physical[:1, 0]), dim=1).int()

    def gather_rows(
        data_pool, scale_pool, indices, output, global_scale, residual_dim, num_pool_tokens
    ):
        assert data_pool is manager._sparse_arena_buffers[2][0]
        assert scale_pool is manager._sparse_arena_buffers[2][1]
        assert num_pool_tokens == pool_rows
        torch.testing.assert_close(indices, physical[:1, descriptor.data_buffer_index])
        assert output.data_ptr() == state.nvfp4_scratch.data_ptr()
        indices.copy_(expected)

    with (
        patch.object(
            deepseek_v4_backend, "deepseek_v4_local_to_global_indices", return_value=prepared
        ) as convert,
        patch.object(
            torch.ops.trtllm,
            "nvfp4_mla_kv_cache_gather_direct",
            side_effect=gather_rows,
            create=True,
        ) as gather,
    ):
        result = deepseek_v4_backend.DeepseekV4TrtllmAttention.sparse_attn_predict(
            backend, None, None, metadata, args
        )
    manager.fetch_sparse_indices.assert_called_once()
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr == (
        state.nvfp4_scratch.data_ptr()
        if nvfp4
        else descriptor.arena_base_addresses[descriptor.data_buffer_index]
    )
    converted = convert.call_args.kwargs
    torch.testing.assert_close(
        converted["compressed_fetched_indices"], physical[:, descriptor.data_buffer_index]
    )
    torch.testing.assert_close(
        converted["compressed_write_indices"], writes[:, descriptor.data_buffer_index]
    )
    assert converted["compressed_history_lengths"] is state.history_lengths
    assert converted["compressed_batch_rows"] is state.batch_rows
    torch.testing.assert_close(
        result[0][:, :4], torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    )
    torch.testing.assert_close(result[0][:, 4:], expected)
    assert gather.call_count == int(nvfp4)
    state.write_layer = None
    with pytest.raises(RuntimeError, match="Prepare this layer"):
        deepseek_v4_backend.DeepseekV4TrtllmAttention.sparse_attn_predict(
            backend, None, None, metadata, args
        )


@_requires_cuda
@pytest.mark.parametrize("multiple_roles", [False, True])
@pytest.mark.parametrize("rows", [None, [2, 0], [3, 1]])
def test_fetch_uses_batch_rows_and_preserves_topk_slots(multiple_roles, rows):
    case = _fetch_case(multiple_roles, rows)
    original = case.topk.clone()
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")),
    ):
        out = _fetch(case)
    expected = _expected_selection(
        case.manager,
        {20: [-1, 0, -1, -1, -1, -1, -1, -1], 10: [2, -1, 0, 1, -1, -1, -1, -1]},
    )
    torch.testing.assert_close(case.state.selected_history_units, expected)
    torch.testing.assert_close(case.topk, original)
    for i in range(out.shape[1]):
        torch.testing.assert_close(
            out[:, i], torch.where(expected >= 0, expected + (i % 2) * 101, -1)
        )
    call = case.manager._sparse_batch.fetch_sparse_pages.calls[-1]
    assert call[0] is case.manager._sparse_batch
    assert call[1] == case.state.layers[2].buffer_ids
    assert call[2] is case.state.selected_history_units and call[3] is out
    second = _fetch(case, layer=4)
    torch.testing.assert_close(second[:, 0], torch.where(expected >= 0, expected + 1000, -1))
    assert second is case.state.fetched_indices[2 if multiple_roles else 1]
    if not multiple_roles:
        assert second is out


@pytest.mark.cpu_only
@pytest.mark.parametrize("invalid", ["disabled", "unprepared", "batch", "watermarks"])
def test_fetch_rejects_unprepared_metadata(invalid):
    manager = _manager()
    state = SimpleNamespace(prepared=True, batch=object(), history_lengths=object())
    if invalid == "disabled":
        manager._enable_kv_cache_offload = False
    elif invalid == "unprepared":
        state.prepared = False
    elif invalid == "batch":
        state.batch = None
    else:
        state.history_lengths = None
    with pytest.raises(RuntimeError, match="prepared decode metadata"):
        manager.fetch_sparse_indices(state, 2, None, None, None)


@_requires_cuda
def test_fetch_propagates_native_failure():
    case = _fetch_case()
    case.manager._sparse_batch.fetch_sparse_pages = Mock(side_effect=RuntimeError("fetch failed"))
    with pytest.raises(RuntimeError, match="fetch failed"):
        _fetch(case)


@_requires_cuda
@pytest.mark.parametrize("empty", ["topk", "history", "batch"])
def test_fetch_refreshes_all_outputs_for_empty_selection(empty):
    case = _fetch_case(multiple_roles=True)
    _fetch(case)
    if empty == "topk":
        case.topk.fill_(-1)
    elif empty == "history":
        case.manager._sparse_batch.history.zero_()
    else:
        _publish(case.manager, [])
        case.manager.prepare_sparse_offload(
            case.state, [CUDA_GRAPH_DUMMY_REQUEST_ID] * 2, beam_width=1
        )
    out = _fetch(case)
    assert (out == -1).all()
    assert len(case.manager._sparse_batch.fetch_sparse_pages.calls) == 2


@_requires_cuda
def test_fetch_graph_replay_refreshes_batch_watermarks_and_padding():
    case = _fetch_case(multiple_roles=True)
    # Warm up before capture; the graph has two query rows but Batch has four.
    _fetch(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fetch(case)
    ptr = out.data_ptr()
    for ids, rows, history in (
        ([10, 20], None, 8),
        # Both requests moved rows, and row 0 is free: the replay follows the map.
        ([10, 20], [3, 1], 8),
        ([20, CUDA_GRAPH_DUMMY_REQUEST_ID], [2], 12),
        ([], None, 0),
    ):
        case.manager.kv_cache_map[20].history_length = history
        case.manager.kv_cache_map[10].history_length = history
        _publish(case.manager, ids, rows)
        case.manager.prepare_sparse_offload(case.state, ids, beam_width=1)
        case.requests.copy_(
            torch.tensor([0, 0], dtype=torch.int32, device="cuda")
            if len(case.manager._sparse_batch_members) < 2
            else torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        )
        case.topk.copy_(
            torch.tensor([[0, -1, 1, 2, -1, -1, -1, -1]] * 2, dtype=torch.int32, device="cuda")
        )
        case.lengths.fill_(4)
        graph.replay()
        expected = torch.full_like(case.state.selected_history_units, -1)
        for member in case.manager._sparse_batch_members.values():
            expected[member.row, 0] = 0
            if history // 4 > 1:
                expected[member.row, 2] = 1
            if history // 4 > 2:
                expected[member.row, 3] = 2
        for role in range(out.shape[1]):
            torch.testing.assert_close(
                out[:, role], torch.where(expected >= 0, expected + (role % 2) * 101, -1)
            )
        assert out.data_ptr() == ptr


def _host_batch(*, prefill=False):
    manager = _manager()
    manager.compute_sliding_block_tables = Mock()
    for cache in manager.kv_cache_map.values():
        cache.history_length = 0
    metadata = SimpleNamespace(
        sparse_metadata_params=SimpleNamespace(enable_kv_cache_offload=True),
        sparse_offload_state=SimpleNamespace(prepared=True),
        kv_cache_manager=manager,
        request_ids=[20, 10],
        beam_width=1,
        max_draft_tokens=0,
        enable_ragged_verification=False,
        ragged_verify_lens=None,
        draft_kv_cache_manager=None,
        num_contexts=2 if prefill else 0,
        num_generations=0 if prefill else 2,
        seq_lens=torch.tensor([8, 12] if prefill else [1, 1], dtype=torch.int32),
        seq_lens_kv=torch.tensor([8, 12] if prefill else [1, 1], dtype=torch.int32),
        prompt_lens=[8, 12],
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0, 0]),
    )
    metadata._validate_ragged_verification = MethodType(
        DeepseekV4TrtllmAttentionMetadata._validate_ragged_verification, metadata
    )
    metadata._validate_dsa_sparse_offload_batch = MethodType(
        DeepseekV4TrtllmAttentionMetadata._validate_dsa_sparse_offload_batch, metadata
    )
    return metadata


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefill", [False, True])
def test_offload_admits_only_decode(prefill):
    # An offloading worker is a disaggregated decode worker: context requests reach it
    # through the KV transfer, never as a batch of its own.
    metadata = _host_batch(prefill=prefill)
    if prefill:
        with pytest.raises(NotImplementedError, match="decode batches only"):
            DeepseekV4TrtllmAttentionMetadata.validate_sparse_offload_batch(metadata)
    else:
        DeepseekV4TrtllmAttentionMetadata.validate_sparse_offload_batch(metadata)
    assert not metadata.sparse_offload_state.prepared


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

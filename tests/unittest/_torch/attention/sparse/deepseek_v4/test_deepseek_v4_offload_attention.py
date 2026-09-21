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

"""Attention integration with a device-driven fetch double, without native H2D/FMHA."""

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
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.hooks import prepare_sparse_runtime_params
from tensorrt_llm._torch.attention.backends.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    KVCacheManagerV2,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

from .test_deepseek_v4_offload_metadata import _manager, _metadata

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("first", "last"), [(True, False), (False, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_chunked_prefill_checked_from_scheduled_request(first, last, enabled):
    manager = _manager()
    manager._enable_kv_cache_offload = enabled
    batch = SimpleNamespace(
        context_requests=[SimpleNamespace(is_first_context_chunk=first, is_last_context_chunk=last)]
    )
    with patch.object(KVCacheManagerV2, "prepare_resources") as prepare:
        if enabled and not (first and last):
            with pytest.raises(NotImplementedError, match="chunked prefill"):
                manager.prepare_resources(batch)
            prepare.assert_not_called()
        else:
            manager.prepare_resources(batch)
            prepare.assert_called_once_with(batch)


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
        "beam",
        "draft",
        "draft_cache",
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
    elif unsupported == "beam":
        metadata.beam_width = 2
    elif unsupported == "draft":
        metadata.max_draft_tokens = 1
    elif unsupported == "draft_cache":
        metadata.draft_kv_cache_manager = object()
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
        indices[:3, :2], torch.tensor([[swa_offset, swa_offset + 3]] * 3, dtype=torch.int32)
    )
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
@pytest.mark.parametrize("failure", ["units", "scale", "api", "prepared", "phase", "transfer"])
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
    elif failure == "phase":
        case.args.attention_input_type = AttentionInputType.mixed
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
def test_fetch_graph_replay_refreshes_requests_topk_and_padding():
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

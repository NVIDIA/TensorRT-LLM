# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 metadata, bounded replay, decoder boundaries and live device lengths."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2CompressionBatch
from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
    enter_decoder_replay,
    enter_remote_tail_decoder,
    exit_decoder_replay,
    plan_decoder_replay,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

from ._utils import _FakeMetadata, _page_metadata


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decoder_swa_slots_skip_reclaimed_encoder_pages(device):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    if device == "cuda" and not (torch.cuda.is_available() and kernel.dsl_available()):
        pytest.skip("CuTe DSL SWA refresh requires CUDA")
    metadata = object.__new__(CSA2TrtllmMetadata)
    metadata.is_cuda_graph = False
    metadata.kv_lens_cuda = torch.zeros(1, device=device)
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        layout=SimpleNamespace(window_size=4, compress_ratios=(0, 2, 1, 1)),
    )
    # Encoder pages have already been released, so a decoder-only query pass
    # binds only the decoder layers' tables.
    metadata.csa2_query_layer_start = 2
    metadata._csa2_buffers = {}
    metadata._csa2_swa_page_tables = {
        2: torch.tensor([[2, 3]], dtype=torch.int32, device=device),
        3: torch.tensor([[4, 5]], dtype=torch.int32, device=device),
    }
    metadata.csa2_positions = torch.tensor([128, 129], dtype=torch.int32, device=device)
    metadata.csa2_token_requests = torch.zeros(2, dtype=torch.int64, device=device)
    metadata.csa2_replay_start_positions = torch.zeros(1, dtype=torch.int64, device=device)
    if device == "cuda":
        metadata._csa2_swa_descriptors = metadata._swa_table_descriptors(4)
        metadata._csa2_swa_ratios = metadata._swa_ratio_tensor((0, 2, 1, 1))
    metadata._csa2_swa_resolved = False
    metadata._ensure_swa_slots()
    for layer, pages in ((2, (2, 3)), (3, (4, 5))):
        expected = torch.tensor(
            [
                [pages[p // 128] * 128 + p % 128 for p in range(end - 3, end + 1)]
                for end in (128, 129)
            ]
        )
        torch.testing.assert_close(metadata.csa2_swa_indices[layer].cpu(), expected)
        torch.testing.assert_close(metadata.csa2_swa_write_slots[layer].cpu(), expected[:, -1])
        torch.testing.assert_close(
            metadata.csa2_visible_lengths[layer].cpu(), torch.tensor([129, 130])
        )
    # Layers before the boundary bind no pages and read or write nothing.
    for layer in (0, 1):
        assert (metadata.csa2_swa_indices[layer] == -1).all()
        assert (metadata.csa2_swa_write_slots[layer] == -1).all()


@pytest.fixture
def manager_requests(request):
    """Real CSA2 manager with two prefilled requests; ``request.param`` may set
    ``scratch`` and ``speculative``."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    params = getattr(request, "param", {})
    spec_config = (
        DraftTargetDecodingConfig(max_draft_len=4, speculative_model="attention-test-draft")
        if params.get("speculative")
        else None
    )
    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=4096,
            enable_swa_scratch_reuse=params.get("scratch", True),
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3)),
        num_layers=4,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=2,
        max_input_len=512,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=1024,
        spec_config=spec_config,
    )
    requests = []
    for request_id in (41, 97):
        llm_request = LlmRequest(
            request_id=request_id,
            max_new_tokens=256,
            input_tokens=list(range(257)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(llm_request)
        assert manager.resize_context(llm_request, llm_request.context_chunk_size)
        requests.append(llm_request)
    manager._stream.synchronize()
    yield manager, requests
    for llm_request in requests:
        manager.free_resources(llm_request)
    manager.shutdown()


def _values(compressor, rows):
    """The value half of the ratio-two compressor's fused projection."""
    return compressor.wkv_gate(rows)[:, : compressor.head_dim]


def _prepare(metadata, requests, starts, lengths):
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.num_contexts = len(requests)
    metadata.prompt_lens = [257] * len(requests)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    metadata.prepare()


def _swa(metadata, layer):
    """Resolve the SWA slots from the current inputs, as a forward does, and return one layer's."""
    metadata._csa2_swa_resolved = False
    metadata._ensure_swa_slots()
    return (
        metadata.csa2_swa_indices[layer],
        metadata.csa2_swa_write_slots[layer],
        metadata.csa2_visible_lengths[layer],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_manager_metadata_scratch_and_source_capacity(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    _prepare(metadata, requests, [0, 0], [257, 257])
    _assert_swa_cpu_reference(metadata, manager, requests, [0, 0], [257, 257], [0, 0])
    reads, writes, _ = _swa(metadata, 1)
    assert writes[:257].unique().numel() == 257
    # Every query retains its own past, including the beginning of a chunk
    # longer than the 128-token window.
    torch.testing.assert_close(reads[128], writes[1:129])
    assert _swa(metadata, 0)[2].count_nonzero() == 0
    assert metadata.get_compression_batch(3) is None
    assert manager.get_main_buffer(1).stride(0) == 288
    assert manager.get_index_pages(1).shape[1:] == (64, 1, 68)
    metadata.set_source_batch([5, 3], [0, 1])
    _prepare(metadata, requests, [4, 3], [1, 1])
    compression = metadata.get_compression_batch(1)
    assert compression.output_rows == 8
    assert compression.cu_seq_lengths.tolist() == [0, 5, 8]
    assert compression.cu_compressed_lengths.tolist() == [0, 2, 4]
    assert metadata.get_compressed_positions(1).tolist() == [0, 2, 0, 2, 0, 0, 0, 0]
    assert metadata.csa2_main_write_slots[1][4:].tolist() == [-1] * 4
    _prepare(metadata, requests, [0, 1], [1, 1])
    assert metadata.get_compression_batch(1).cu_compressed_lengths.tolist() == [0, 0, 1]
    completed = metadata.csa2_main_write_slots[1].tolist()
    expected_page = manager.get_cache_indices(97, 1, CSA2CacheRole.GLOBAL)[0]
    assert completed == [expected_page * 64, -1]
    # Zero completed ratio-2 groups still prepare partial state and padded slots.
    _prepare(metadata, requests, [0, 0], [1, 1])
    compression = metadata.get_compression_batch(1)
    assert compression.output_rows == 2
    assert compression.cu_seq_lengths.tolist() == [0, 1, 2]
    assert compression.cu_compressed_lengths.tolist() == [0, 0, 0]
    assert metadata.csa2_main_write_slots[1].tolist() == [-1, -1]
    assert metadata.get_compressed_positions(1).tolist() == [0, 0]
    # Consumer and source resolve the same owner pages, without owning extra KV.
    assert manager.get_cache_indices(41, 1, CSA2CacheRole.GLOBAL) == manager.get_cache_indices(
        41, 2, CSA2CacheRole.GLOBAL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_swa_slots_resolve_once_per_forward(manager_requests, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    if not kernel.dsl_available():
        pytest.skip("Counting native resolutions requires the CuTe DSL")
    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    launches = []
    refresh = kernel.refresh_swa_slots
    monkeypatch.setattr(kernel, "refresh_swa_slots", lambda *a: launches.append(refresh(*a)))
    layers = [manager.layout.layer(i) for i in range(len(manager.layout.compress_ratios))]

    def forward():
        metadata.begin_model_forward()
        for layer in layers:
            metadata.enter_layer(layer)

    _prepare(metadata, requests, [0, 0], [257, 257])
    forward()
    forward()
    metadata.on_update_kv_lens()
    metadata.prepare_indexer(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    # Once per forward, after a KV-length update, and once inside a capture.
    assert len(launches) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_manager_partial_compression_graph_and_strided_publication(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        store_rows,
    )

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    metadata = metadata.create_cuda_graph_metadata(2)
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    compressor.wkv_gate.weight.normal_(std=0.1)
    torch.manual_seed(472)
    inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    _prepare(metadata, requests, [0, 0], [1, 1])
    manager.get_main_buffer(1).fill_(73)
    manager.get_index_pages(1).fill_(59)
    compression = metadata.get_compression_batch(1)
    write_slots = metadata.csa2_main_write_slots[1]
    global_page_table = metadata.csa2_global_page_tables[1]
    write_slot_mapping = metadata.csa2_main_write_slots

    def forward():
        output = compressor(inputs, compression)
        store_rows(manager.get_main_buffer(1), write_slots, output, "main")
        return output

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    ptrs = (
        write_slots.data_ptr(),
        compression.start_positions.data_ptr(),
        global_page_table.data_ptr(),
    )
    previous = {}
    for step in range(4):
        inputs.copy_(torch.randn_like(inputs))
        ordered = requests if step < 2 else requests[::-1]
        _prepare(metadata, ordered, [step, step], [1, 1])
        refreshed = metadata.get_compression_batch(1)
        assert metadata.csa2_main_write_slots is not write_slot_mapping
        write_slot_mapping = metadata.csa2_main_write_slots
        assert ptrs == (
            metadata.csa2_main_write_slots[1].data_ptr(),
            refreshed.start_positions.data_ptr(),
            metadata.csa2_global_page_tables[1].data_ptr(),
        )
        manager.get_main_buffer(1).fill_(73)
        values, gates = compressor.wkv_gate(inputs).float().split(512, dim=-1)
        expected_rows = []
        for row, request in enumerate(ordered):
            request_id = request.py_request_id
            if step % 2:
                old_values, old_gates = previous[request_id]
                weights = torch.stack((old_gates, gates[row])).softmax(0)
                expected_rows.append((torch.stack((old_values, values[row])) * weights).sum(0))
            previous[request_id] = (values[row].clone(), gates[row].clone())
        graph.replay()
        torch.cuda.synchronize()
        if step % 2 == 0:
            assert output.count_nonzero() == 0
            assert torch.all(manager.get_main_buffer(1) == 73)
        else:
            # Native pooling rounds the reduced latent before model RMSNorm.
            expected = torch.stack(expected_rows).to(torch.bfloat16).float()
            expected *= torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
            expected = (expected * compressor.norm.weight.float()).to(torch.bfloat16)
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
            slots = metadata.csa2_main_write_slots[1].long()
            torch.testing.assert_close(manager.get_main_buffer(1)[slots], pack_rows(output, "main"))
        # Main stores must never overwrite the adjacent index bytes.
        assert torch.all(manager.get_index_pages(1) == 59)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_odd_prefix_reuse_compression_matches_fresh():
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            enable_swa_scratch_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        # This regression intentionally exercises the retained exact-cache
        # compatibility mode, including persisted FP32 odd-tail state.
        layout=CSA2Layout((2,), (0,), (0,)),
        num_layers=1,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=3,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=8192,
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=3, max_num_tokens=1024, kv_cache_manager=manager)
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    compressor.wkv_gate.weight.normal_(std=0.1)
    torch.manual_seed(671)
    hidden = torch.randn(130, 32, dtype=torch.bfloat16, device="cuda")
    tokens = list(range(129))

    def allocate(request_id, prefix, capacity):
        cache = manager._create_kv_cache(request_id, None, prefix)
        assert cache is not None
        assert manager._resume_and_restore(request_id, cache)
        assert cache.resize(capacity)
        manager._stream.synchronize()
        return cache

    def compress(request_id, start, values):
        _prepare(metadata, [SimpleNamespace(py_request_id=request_id)], [start], [len(values)])
        output = compressor(values, metadata.get_compression_batch(0))
        manager.write_swa(0, _swa(metadata, 0)[1], _values(compressor, values).bfloat16())
        manager.write_global(0, metadata.csa2_main_write_slots[0], output, output[:, :128])
        return output

    try:
        first = allocate(501, [], 129)
        compress(501, 0, hidden[:129])
        first.commit(tokens)
        torch.cuda.synchronize()
        manager.free_resources(SimpleNamespace(py_request_id=501))
        left = allocate(502, tokens + [7000], 130)
        right = allocate(503, tokens + [7001], 130)
        reused = left.num_committed_tokens
        assert reused == right.num_committed_tokens
        assert reused in (128, 129)
        print(
            f"CSA2 odd prefix: committed=129, actually_reused={reused}, recomputed={130 - reused}"
        )
        left_pages = manager.get_cache_indices(502, 0, CSA2CacheRole.GLOBAL)
        right_pages = manager.get_cache_indices(503, 0, CSA2CacheRole.GLOBAL)
        assert left_pages[1] != right_pages[1]
        pool = manager.get_buffers(0, CSA2CacheRole.GLOBAL)
        shared_before = pool[left_pages[0]].clone()
        right_before = pool[right_pages[1]].clone()
        continuation = compress(502, reused, hidden[reused:]).clone()
        torch.cuda.synchronize()
        # Publishing the completed odd-boundary pair must not modify either
        # the shared prefix page or another request's private writable suffix.
        torch.testing.assert_close(pool[left_pages[0]], shared_before, atol=0, rtol=0)
        torch.testing.assert_close(pool[right_pages[0]], shared_before, atol=0, rtol=0)
        torch.testing.assert_close(pool[right_pages[1]], right_before, atol=0, rtol=0)
        allocate(504, [], 130)
        fresh = compress(504, 0, hidden)
        torch.testing.assert_close(continuation[0], fresh[64], atol=0.02, rtol=0.02)
        values, gates = compressor.wkv_gate(hidden[128:]).float().split(512, dim=-1)
        expected = (values * gates.softmax(0)).sum(0).bfloat16().float()
        expected *= torch.rsqrt(expected.square().mean() + 1e-6)
        expected = (expected * compressor.norm.weight.float()).bfloat16()
        torch.testing.assert_close(continuation[0], expected, atol=0.02, rtol=0.02)
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("start", [127, 128, 129])
@pytest.mark.parametrize("accepted_drafts", [0, 1, 4])
@torch.inference_mode()
def test_chain_rewind_preserves_compressor_and_swa(start, accepted_drafts):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    spec = DraftTargetDecodingConfig(max_draft_len=4, speculative_model="attention-test-draft")
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=128 << 20, enable_swa_scratch_reuse=True, dtype="fp8"),
        CacheType.SELFKONLY,
        layout=CSA2Layout((2,), (0,), (0,)),
        num_layers=1,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=512,
        dtype=DataType.BF16,
        vocab_size=8192,
        spec_config=spec,
    )
    request = LlmRequest(
        request_id=71,
        max_new_tokens=32,
        input_tokens=list(range(start)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=512, kv_cache_manager=manager)
    metadata.is_spec_decoding_enabled = True
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    compressor.wkv_gate.weight.normal_(std=0.1)
    torch.manual_seed(719 + start + accepted_drafts)
    hidden = torch.randn(start + 5, 32, device="cuda", dtype=torch.bfloat16)
    projected_swa = []

    def compress(position, values, context=False):
        metadata.request_ids = [71]
        metadata.seq_lens = torch.tensor([len(values)], dtype=torch.int32)
        metadata.num_contexts = int(context)
        metadata.prompt_lens = [start]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[position]
        )
        metadata.prepare()
        output = compressor(values, metadata.get_compression_batch(0))
        swa = _values(compressor, values).bfloat16()
        projected_swa.append(swa.clone())
        manager.write_swa(0, _swa(metadata, 0)[1], swa)
        manager.write_global(0, metadata.csa2_main_write_slots[0], output, output[:, :128])
        return output

    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, start)
        manager._stream.synchronize()
        compress(0, hidden[:start], context=True)
        cache = manager.kv_cache_map[71]
        assert cache.resize(start, start)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.add_new_token(1000, 0)
        request.py_draft_tokens = [1001, 1002, 1003, 1004]
        assert manager.try_allocate_generation(request)
        manager._stream.synchronize()
        compress(start, hidden[start:])
        # The golden token is always retained; accepted_drafts counts only
        # subsequent draft tokens. The final sampled token is not cached yet.
        for offset in range(accepted_drafts + 1):
            request.add_new_token(1100 + offset, 0)
        request.py_num_accepted_draft_tokens = accepted_drafts
        request.py_rewind_len = 4 - accepted_drafts
        scheduled = ScheduledRequests()
        scheduled.generation_requests = [request]
        manager.update_resources(scheduled, metadata, 2)
        accepted_end = start + 1 + accepted_drafts
        assert cache.capacity == accepted_end
        continuation = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        assert cache.resize(accepted_end + 2)
        manager._stream.synchronize()
        actual = compress(accepted_end, continuation)
        pair = (
            torch.cat((hidden[accepted_end - 1 : accepted_end], continuation[:1]))
            if accepted_end % 2
            else continuation
        )
        values, gates = compressor.wkv_gate(pair).float().split(512, dim=-1)
        expected = (values * gates.softmax(0)).sum(0).bfloat16().float()
        expected *= torch.rsqrt(expected.square().mean() + 1e-6)
        expected = (expected * compressor.norm.weight.float()).bfloat16()
        torch.testing.assert_close(actual[0], expected, atol=0.02, rtol=0.02)
        assert torch.count_nonzero(actual[1]) == 0
        # Compare cache preservation against the exact projection inputs at
        # each publication boundary. Re-running GEMM at a different row count
        # can choose a different reduction schedule near BF16 rounding ties.
        retained = torch.cat(
            (projected_swa[0], projected_swa[1][: 1 + accepted_drafts], projected_swa[2][:1])
        )[-128:]
        swa_reference = unpack_rows(pack_rows(retained, "swa"), 512, "swa")
        slots = _swa(metadata, 0)[0][0]
        gathered = gather_rows(manager.get_swa_buffer(0), slots, 512, "swa")
        torch.testing.assert_close(gathered[-len(retained) :], swa_reference, atol=0, rtol=0)
    finally:
        manager.free_resources(request)
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_temporal_prior_identity_rewind_and_replay(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    resets = []
    metadata.register_indexer_reset(1, lambda: resets.append(True))

    def prepare(ordered, positions):
        metadata.request_ids = [request.py_request_id for request in ordered]
        metadata.seq_lens = torch.ones(2, dtype=torch.int32)
        metadata.num_contexts = 0
        metadata.prompt_lens = [257, 257]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=positions
        )
        metadata.prepare()
        return metadata.prepare_indexer_prior(1, 4)

    prior = prepare(requests, [0, 0])
    assert torch.all(prior == -1)
    assert metadata.csa2_indexer_prior_capacity[1].shape == (8, 4)
    selected = torch.arange(8, device="cuda", dtype=torch.int32).view(2, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.publish_indexer_prior(1, selected)
    graph.replay()
    prior = prepare(requests, [1, 1])
    torch.testing.assert_close(prior, selected)
    reset_count = len(resets)
    graph.replay()
    prior = prepare(requests[::-1], [2, 2])
    torch.testing.assert_close(prior, selected.flip(0))
    assert len(resets) > reset_count
    graph.replay()
    # A rewind must not carry a rejected/future query's temporal hint.
    assert torch.all(prepare(requests[::-1], [1, 1]) == -1)
    graph.replay()
    old_epoch = manager.request_epoch(requests[0].py_request_id)
    manager.free_resources(requests[0])
    assert manager.prepare_context(requests[0])
    assert manager.resize_context(requests[0], requests[0].context_chunk_size)
    assert manager.request_epoch(requests[0].py_request_id) != old_epoch
    assert torch.all(prepare(requests, [2, 2])[0] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_retained_metadata_buffers_and_workspace_accounting(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    for length in (1, 2, 3, 4):
        _prepare(metadata, requests, [0, 0], [length, length])
    eager_keys = [key for key in metadata._csa2_buffers if not key[-1]]
    assert len({key[0] for key in eager_keys}) == len(eager_keys)
    frame = metadata.get_query_tile_metadata(
        torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16), 17
    )
    before = metadata.get_workspace_bytes()
    metadata._csa2_buffers["alias", (), torch.bfloat16, False] = frame.swa_pool.view(-1)
    assert metadata.get_workspace_bytes() == before
    extra = torch.empty(129, device="cuda", dtype=torch.uint8)
    metadata._csa2_buffers["extra", (129,), torch.uint8, False] = extra
    assert metadata.get_workspace_bytes() == before + extra.untyped_storage().nbytes()
    # Exact sum of the graph descriptor allocations: block table, contexts,
    # visible lengths, position validity, radix scratch and one schedule. Index
    # rows are read from the owner's cache pages in place, so no page copies are retained.
    expected = 4 * (1 * (4 + 64) + 8 + 80 * 32) + (148 + 1) * 8
    assert metadata.workspace_reservation_bytes(2, 4, 64, 32, 148) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_pageable_metadata_uploads_feed_captured_consumer(manager_requests):
    dtype = torch.int64
    manager, _ = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    width, steps = 257, 64
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buffer = metadata._copy_csa2_tensor(
            "pageable_lifetime", torch.zeros(width, dtype=dtype, device="cpu")
        )
        address = buffer.data_ptr()
        consumed = torch.empty_like(buffer)
        snapshots = torch.empty((steps, width), dtype=dtype, device=buffer.device)
        for _ in range(2):
            torch.add(buffer, 7, out=consumed)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            torch.add(buffer, 7, out=consumed)
        for step in range(steps):
            backing = torch.arange(width, dtype=dtype, device="cpu") + step * width
            if step % 2:
                # Match strided CPU metadata views such as reads[:, -1].
                columns = torch.empty((width, 2), dtype=dtype, device="cpu")
                columns[:, 0].copy_(backing)
                backing = columns
                del columns
                value = backing[:, 0]
                assert value.stride() == (2,)
            else:
                value = backing
            assert not value.is_pinned()
            updated = metadata._copy_csa2_tensor("pageable_lifetime", value)
            assert updated.data_ptr() == address
            del value, backing
            # Encourage immediate pageable allocation reuse before the consumer.
            # This storage is neither retained nor used as a pinned transfer source.
            torch.full((width,), -1, dtype=dtype, device="cpu")
            torch.full((width, 2), -1, dtype=dtype, device="cpu")
            graph.replay()
            snapshots[step].copy_(consumed)
    # No reads or explicit synchronization occurred between queued uploads.
    stream.synchronize()
    expected = torch.arange(steps * width, dtype=dtype, device="cpu").reshape(steps, width) + 7
    torch.testing.assert_close(snapshots.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_representable_draft_switch_restores_target_fields(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.speculative.interface import (
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    target, requests = manager_requests
    draft = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=128 << 20, dtype="fp8"),
        CacheType.SELFKONLY,
        layout=target.layout,
        num_layers=len(target.layout.compress_ratios),
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=1024,
        is_draft=True,
    )
    try:
        for request in requests:
            assert draft.prepare_context(request)
            assert draft.resize_context(request, request.context_chunk_size)
        draft._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=2,
            max_num_tokens=1024,
            kv_cache_manager=target,
            draft_kv_cache_manager=draft,
        )
        _prepare(metadata, requests, [0, 0], [1, 1])
        _assert_swa_cpu_reference(metadata, target, requests, [0, 0], [1, 1], [0, 0])
        target_reads = {
            layer: _swa(metadata, layer)[0].clone()
            for layer in range(len(target.layout.compress_ratios))
        }
        target_pages = dict(metadata._csa2_swa_page_tables)
        target_page_values = {layer: pages.clone() for layer, pages in target_pages.items()}
        target_positions = metadata.csa2_positions
        target_state = metadata.get_compression_batch(1).kv_state
        resets = []
        metadata.register_indexer_reset(1, lambda: resets.append("target"))
        saved = prepare_attn_metadata_for_draft_replay(metadata, draft)
        assert resets == ["target"]
        metadata.register_indexer_reset(1, lambda: resets.append("draft"))
        try:
            assert metadata.kv_cache_manager is draft
            for layer, pages in metadata._csa2_swa_page_tables.items():
                torch.testing.assert_close(
                    target_pages[layer], target_page_values[layer], atol=0, rtol=0
                )
            _assert_swa_cpu_reference(metadata, draft, requests, [0, 0], [1, 1], [0, 0])
        finally:
            restore_attn_metadata_after_draft_replay(metadata, saved)
        assert metadata.kv_cache_manager is target
        _assert_swa_cpu_reference(metadata, target, requests, [0, 0], [1, 1], [0, 0])
        for layer, expected in target_reads.items():
            torch.testing.assert_close(_swa(metadata, layer)[0], expected, atol=0, rtol=0)
        assert metadata.csa2_positions is target_positions
        assert metadata.get_compression_batch(1).kv_state is target_state
        assert resets == ["target", "target"]
        # A later draft switch restores its own cached callback/prior state;
        # contiguous identity must not suppress reset of shared emission state.
        saved = prepare_attn_metadata_for_draft_replay(metadata, draft)
        assert resets[-2:] == ["target", "draft"]
        restore_attn_metadata_after_draft_replay(metadata, saved)
        assert resets[-1] == "target"
        # Legacy direct manager rebinding remains legal: metadata-owned backing
        # is refreshed from the new manager rather than rejected or redirected.
        metadata.draft_kv_cache_manager = None
        for current in (draft, target):
            metadata.kv_cache_manager = current
            _prepare(metadata, requests, [0, 0], [1, 1])
            _assert_swa_cpu_reference(metadata, current, requests, [0, 0], [1, 1], [0, 0])
    finally:
        for request in requests:
            draft.free_resources(request)
        draft.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_speculative_contract_rejects_unrepresentable_paths():
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager

    def construct(spec):
        return CSA2CacheManager(
            None, None, num_layers=1, tokens_per_block=128, mapping=None, spec_config=spec
        )

    with pytest.raises(NotImplementedError, match="token trees"):
        construct(SimpleNamespace(is_linear_tree=False))
    with pytest.raises(NotImplementedError, match="virtual draft layers"):
        construct(
            SimpleNamespace(
                is_linear_tree=True,
                spec_dec_mode=SimpleNamespace(is_mtp_eagle_one_model=lambda: True),
            )
        )
    manager = object.__new__(CSA2CacheManager)
    with pytest.raises(NotImplementedError, match="relocation indices"):
        manager.update_resources(
            SimpleNamespace(
                generation_requests=[SimpleNamespace(py_num_accepted_draft_tokens_indices=[1, 3])]
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "manager_requests",
    [{"scratch": True}, {"scratch": False}],
    indirect=True,
    ids=["scratch", "no_scratch"],
)
def test_batch_page_tables_match_per_request_converters(manager_requests):
    """One device conversion of every per-layer table equals the per-request converters,
    with and without SWA scratch; host allocation queries follow the converted pages."""
    manager, requests = manager_requests
    ids = [request.py_request_id for request in requests]
    specs = list(manager._batch_page_index)
    width = manager.max_blocks_per_seq
    expected = [
        [list(manager.get_cache_indices(request_id, layer, role))[:width] for request_id in ids]
        for layer, role in specs
    ]
    manager.compute_batch_page_tables(ids, 0)
    # The conversion reads a snapshot: rewriting the host rows before it runs
    # does not reach it.
    base = manager.host_kv_cache_block_offsets
    saved = base.clone()
    base.fill_(-1)
    converted = manager._batch_page_tables.cpu().numpy()
    base.copy_(saved)
    for index, spec in enumerate(specs):
        for row, pages in enumerate(expected[index]):
            assert converted[index, row].tolist() == pages + [-1] * (width - len(pages)), spec
    rows, columns = (grid.ravel() for grid in np.indices((len(ids), width + 1)))
    padded = np.pad(converted, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
    np.testing.assert_array_equal(
        manager.batch_pages_allocated(specs, rows, columns, width), padded[:, rows, columns] >= 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_temporal_prefill_seeds_first_decode(manager_requests, monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    _prepare(metadata, requests, [0, 0], [2, 2])
    assert metadata.prepare_indexer_prior(1, 4).shape == (0, 4)
    selected = torch.arange(16, dtype=torch.int32, device="cuda").view(4, 4)
    metadata.publish_indexer_prior(1, selected)
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[2, 2])
    metadata.prepare()
    torch.testing.assert_close(metadata.prepare_indexer_prior(1, 4), selected[[1, 3]])
    if torch.cuda.get_device_capability()[0] != 10:
        return  # Native FP4 paged descriptor coverage is validated on SM100.
    manager.get_index_pages(1).zero_()
    assert metadata.prepare_indexer(1) is metadata
    # Index pages are read in place through the owner's block table.
    assert metadata.csa2_indexer_k_cache.data_ptr() == manager.get_index_pages(1).data_ptr()
    valid = metadata.csa2_indexer_valid_positions
    offsets = torch.arange(valid.shape[1], device=valid.device)
    torch.testing.assert_close(
        valid, offsets[None, :] < metadata.csa2_indexer_visible_lengths[:, None]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_graph_workspace_rejects_underfunded_resolved_cap(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.prompt_lens = [257, 257]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0])
    metadata.prepare()
    manager.fp8_ctx_mla_kv_len_cap = 1
    with pytest.raises(ValueError, match="full admitted request KV bound"):
        metadata.prepare_indexer(1)
    assert not hasattr(metadata, "_csa2_indexer_workspaces")


@pytest.fixture
def replay_metadata():
    manager = _manager(4)
    for request_id in (71, 97):
        cache = manager._create_kv_cache(request_id, None, [])
        assert manager._resume_and_restore(request_id, cache)
        assert cache.resize(32)
    manager._stream.synchronize()
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=64, kv_cache_manager=manager)
    yield manager, metadata
    for request_id in list(manager.kv_cache_map):
        manager.free_resources(SimpleNamespace(py_request_id=request_id))
    manager.shutdown()


def _manager(window):
    return CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, enable_swa_scratch_reuse=True, dtype="fp8"),
        CacheType.SELFKONLY,
        num_layers=5,
        tokens_per_block=128,
        layout=CSA2Layout((0, 2, 2, 2, 1), (1, 4), (1, 2, 4), window_size=window, index_topk=4),
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=512,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
    )


def _prepare_replay(metadata, prefixes, suffixes=None, *, decoder=False, ids=None):
    suffixes = [0] * len(prefixes) if suffixes is None else suffixes
    ranges = metadata.kv_cache_manager.get_swa_replay_ranges(prefixes, suffixes, decoder=decoder)
    metadata.request_ids = ids or [71, 97][: len(prefixes)]
    metadata.num_contexts = len(prefixes)
    metadata.seq_lens = torch.tensor([end - start for start, end in ranges], dtype=torch.int32)
    metadata.prompt_lens = [end for _, end in ranges]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[start for start, _ in ranges]
    )
    metadata.set_swa_bounded_replay(prefixes, decoder=decoder)
    metadata.prepare()
    return ranges


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_owner_source_selection_and_one_shot_reset(replay_metadata):
    manager, metadata = replay_metadata
    assert _prepare_replay(metadata, [5, 6], [3, 2]) == ((1, 8), (2, 8))
    # Ratio2 replays one raw odd tail plus suffix; ratio1 only projects suffix.
    assert metadata.csa2_global_source_indices[1].tolist() == [3, 4, 5, 6, 11, 12]
    assert metadata.csa2_global_source_indices[4].tolist() == [4, 5, 6, 11, 12]
    compression = metadata.get_compression_batch(1)
    assert compression.start_positions.tolist() == [4, 6]
    assert compression.cu_seq_lengths.tolist() == [0, 4, 6]
    assert compression.cu_compressed_lengths.tolist() == [0, 2, 3]
    assert set(metadata.csa2_global_source_indices) == {1, 4}
    assert [metadata.csa2_kv_sources[layer] for layer in (0, 1, 2, 3, 4)] == [None, 1, 1, 1, 4]
    for layer in range(5):
        reads = _swa(metadata, layer)[0]
        assert reads[0, :-1].tolist() == [-1, -1, -1]
        assert reads[7, :-1].tolist() == [-1, -1, -1]
    hidden = torch.randn(13, 32, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        metadata.select_global_source(1, hidden), hidden[[3, 4, 5, 6, 11, 12]]
    )
    with pytest.raises(ValueError, match="external source batch"):
        metadata.select_global_source(1, hidden, hidden)
    # A normal subsequent prepare consumes no stale replay floor or selector.
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[8, 8])
    metadata.prepare()
    assert metadata.csa2_replay_mode is None
    assert metadata.csa2_global_source_indices == {}
    assert metadata.csa2_replay_start_positions.tolist() == [0, 0]
    next_hidden = hidden[:2]
    assert metadata.select_global_source(1, next_hidden) is next_hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_graph_updates_values_but_rejects_shape_or_mode_change(replay_metadata):
    _, metadata = replay_metadata
    metadata.is_cuda_graph = True
    _prepare_replay(metadata, [5], [3])
    hidden = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(7, 8, 512, device="cuda", dtype=torch.bfloat16)
    metadata.get_query_tile_metadata(q, 4, query_start=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.get_query_tile_metadata(q, 4, query_start=0)
        selected = metadata.select_global_source(1, hidden)
        positions = metadata.csa2_positions.clone()
        write_slots = metadata.csa2_main_write_slots[1].clone()
    # Same source/query shape at a different absolute hit position is valid.
    _prepare_replay(metadata, [7], [3], ids=[97])
    hidden.neg_()
    graph.replay()
    torch.testing.assert_close(selected, hidden[3:])
    torch.testing.assert_close(positions, torch.arange(3, 10, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(write_slots, metadata.csa2_main_write_slots[1])
    with pytest.raises(ValueError, match="fresh graph metadata"):
        metadata.prepare()
    with pytest.raises(ValueError, match="fresh graph metadata"):
        _prepare_replay(metadata, [6], [3])
    with pytest.raises(ValueError, match="fresh graph metadata"):
        _prepare_replay(metadata, [7], decoder=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_requires_first_window_row_to_be_physically_writable():
    # Explicit replay must reject allocations missing the first recovery row.
    manager = _manager(128)
    try:
        cache = manager._create_kv_cache(71, None, [])
        assert manager._resume_and_restore(71, cache)
        cache.enable_swa_scratch_reuse = False
        # Ordinary next-query retention at C255 keeps128..254, while replay
        # requires127..254. Page0 is genuinely absent in the real allocator.
        assert cache.resize(255, 255)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=128, kv_cache_manager=manager
        )
        assert manager.get_swa_replay_ranges([255]) == ((127, 255),)
        with pytest.raises(ValueError, match="reserve the complete replay range"):
            _prepare_replay(metadata, [255])
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.parametrize("query_width", [1, 3, 17])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("accepted_end", [128, 129, 130])
def test_live_endpoint_refreshes_query_slots_and_compressed_groups(
    monkeypatch, ratio, accepted_end, query_width
):
    # Base scheduler invalidation is orthogonal; exercise the actual CSA2 tensor
    # refresh without native allocations, including an odd ratio-two boundary.
    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
    metadata._csa2_ready_for_kv_update = True
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=128, layout=SimpleNamespace(window_size=128, compress_ratios=(ratio,))
    )
    metadata._csa2_query_lengths = torch.tensor([query_width], dtype=torch.int32)
    metadata._csa2_query_base = torch.tensor([129], dtype=torch.int32)
    metadata._csa2_query_offsets = torch.arange(query_width, dtype=torch.int32)
    metadata.kv_lens_cuda = torch.tensor([accepted_end], dtype=torch.int32)
    metadata.csa2_token_requests = torch.zeros(query_width, dtype=torch.int64)
    metadata.csa2_positions = torch.empty(query_width, dtype=torch.int32)
    metadata.csa2_replay_start_positions = torch.zeros(1, dtype=torch.int64)
    metadata._csa2_swa_page_tables = {0: torch.tensor([[4, 7]], dtype=torch.int32)}
    metadata.is_cuda_graph = False
    metadata._csa2_source_geometry = {0: (torch.tensor([129]), torch.tensor([query_width]))}
    metadata.csa2_main_write_slots = {0: torch.empty(query_width, dtype=torch.int64)}
    metadata._csa2_compressed_positions = {0: torch.empty(query_width, dtype=torch.int32)}
    metadata.csa2_global_page_sizes = {0: 128 // ratio}
    metadata.csa2_global_page_tables = {0: torch.tensor([[9, 12]], dtype=torch.int32)}
    if ratio == 2:
        metadata._csa2_compression = {
            0: CSA2CompressionBatch(
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(1, dtype=torch.int32),
                torch.empty(1, dtype=torch.int32),
                torch.tensor([0, query_width], dtype=torch.int32),
                torch.empty(2, dtype=torch.int32),
                query_width,
                128,
                query_width,
            )
        }
    identities = (metadata.csa2_positions.data_ptr(), metadata.csa2_main_write_slots[0].data_ptr())
    metadata.on_update_kv_lens()
    positions = list(range(accepted_end - query_width, accepted_end))
    assert metadata.csa2_positions.tolist() == positions
    # The layer resolves its SWA slots from the refreshed positions when entered.
    _, writes, visible = _swa(metadata, 0)
    assert writes.tolist() == [[4, 7][p // 128] * 128 + p % 128 for p in positions]
    assert visible.tolist() == [(p + 1) // ratio for p in positions]
    groups = list(range((accepted_end - query_width) // ratio, accepted_end // ratio))
    page_size = 128 // ratio
    assert metadata.csa2_main_write_slots[0].tolist() == [
        [9, 12][g // page_size] * page_size + g % page_size for g in groups
    ] + [-1] * (query_width - len(groups))
    assert metadata._csa2_compressed_positions[0].tolist() == [g * ratio for g in groups] + [0] * (
        query_width - len(groups)
    )
    if ratio == 2:
        batch = metadata._csa2_compression[0]
        assert batch.start_positions.tolist() == [accepted_end - query_width]
        assert batch.kv_lengths.tolist() == [accepted_end]
        assert batch.cu_compressed_lengths.tolist() == [0, len(groups)]
    assert identities == (
        metadata.csa2_positions.data_ptr(),
        metadata.csa2_main_write_slots[0].data_ptr(),
    )


def test_paged_owner_mapping_and_request_reordering():
    pages = _page_metadata(
        torch.tensor([[2, 0], [1, -1]], dtype=torch.int32), torch.tensor([1, 0]), 2, 4
    )
    assert pages.global_slot_tile(0, 0, 2).tolist() == [[2, 3, -1, -1], [4, 5, 0, 1]]
    logical = torch.tensor([[1, 3, -1], [2, 0, 4]], dtype=torch.int32)
    assert pages.global_slot_tile(0, 0, 2, logical).tolist() == [[3, -1, -1], [0, 4, -1]]


def test_invalid_paged_request_does_not_alias_real_request():
    pages = _page_metadata(torch.tensor([[3], [7]]), torch.tensor([-1, 2, 1]), 2, 2)
    assert pages.global_slot_tile(0, 0, 3).tolist() == [[-1, -1], [-1, -1], [14, 15]]


def test_routing_reset_detaches_shallow_clone():
    metadata = _page_metadata(torch.tensor([[0]]), torch.tensor([0]), 1, 1)
    layout = CSA2Layout((1, 1), (0,), (0,))
    metadata.enter_layer(layout.layer(0))
    indices = torch.tensor([[0]])
    metadata.csa2_indices[0] = indices
    clone = copy(metadata)
    clone.reset_routing()
    assert clone.csa2_indices == {} and clone.csa2_candidates == {}
    assert metadata.csa2_indices[0] is indices
    clone.enter_layer(layout.layer(0))
    with pytest.raises(ValueError, match="across forwards"):
        metadata.enter_layer(layout.layer(0))
    metadata.reset_routing()
    assert metadata.csa2_indices == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staging_excludes_out_of_capacity_rows() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Fused CSA2 gather requires SM100-family validation")
    q = torch.empty((2, 32, 512), dtype=torch.bfloat16, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    swa = pack_rows(torch.randn(3, 512, dtype=torch.bfloat16, device="cuda"), "swa")
    main = pack_rows(torch.randn(3, 512, dtype=torch.bfloat16, device="cuda"), "main")
    swa_slots = torch.tensor([[0, 3, 2**40, -1], [3, -1, -1, -1]], device="cuda")
    main_slots = torch.tensor([[2, 3], [-1, 3]], device="cuda")
    inputs = CSA2BackendForwardArgs(
        swa_pool=swa, swa_indices=swa_slots, main_pool=main, topk_indices=main_slots
    )
    metadata.stage_selected(inputs)
    assert metadata.prepared_lens.tolist() == [2, 1]
    expected = torch.cat(
        (
            _staging_gather_reference(swa, swa_slots[:, :1], 512, "swa"),
            _staging_gather_reference(main, main_slots[:, :1], 512, "main"),
        ),
        dim=1,
    )
    _assert_staging_bf16_decode(metadata.swa_pool[0, :2], expected[0])
    assert torch.count_nonzero(metadata.swa_pool[1, 0]) == 0


def _staging_gather_reference(pool, slots, dim, cache_format):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    valid = (slots >= 0) & (slots < pool.shape[0])
    if pool.shape[0] == 0:
        return torch.zeros((*slots.shape, dim), dtype=torch.bfloat16, device=pool.device)
    selected = pool[torch.where(valid, slots, 0).long()]
    decoded = unpack_rows(selected, dim, cache_format)
    return torch.where(valid[..., None], decoded, 0)


def _assert_staging_bf16_decode(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], atol=0, rtol=0
    )


WINDOW = 128


class _LifecycleMetadata(_FakeMetadata):
    _get_csa2_buffer = CSA2TrtllmMetadata._get_csa2_buffer
    _copy_host = CSA2TrtllmMetadata._copy_host
    _copy_csa2_tensor = CSA2TrtllmMetadata._copy_csa2_tensor

    @property
    def num_generations(self) -> int:
        return self.num_seqs - self.num_contexts

    def prepare(self):
        super().prepare()
        self._csa2_last_layer = -1
        self.csa2_precomputed_kv_layers = set()
        self.kv_lens_cuda[: self.num_seqs].copy_(
            self.seq_lens + torch.tensor(self.kv_cache_params.num_cached_tokens_per_seq)
        )


@pytest.mark.parametrize(
    "private_decoder,pass_requests,extend_context",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_replay_preserves_live_speculative_kv_lifecycle(
    private_decoder, pass_requests, extend_context
) -> None:
    metadata = _LifecycleMetadata([400, 7, 6], 2, [128, 700, 900])
    # The second context row can be a speculative extend request, rather
    # than an ordinary prefill chunk. Only that layout sets this counter.
    metadata.num_chunked_ctx_requests = int(extend_context)
    metadata.request_ids = [10, 11, 12]
    original_params = metadata.kv_cache_params
    metadata.prompt_lens = [400, 7, 700]
    metadata.kv_lens_cuda = torch.cat((metadata.kv_lens_cuda, torch.tensor([-99, -99])))
    metadata.kv_lens_cuda[2] -= 3
    metadata.on_update_kv_lens()
    expected_kv = metadata.kv_lens_cuda.clone()
    positions = metadata.csa2_positions.clone()
    untrimmed_swa = _swa(metadata, 0)[0][400:].clone()
    requests = [SimpleNamespace(py_ced_replay=None) for _ in range(2)] if pass_requests else None
    plan = plan_decoder_replay(metadata, WINDOW, requests, private_decoder=private_decoder)
    if extend_context and not private_decoder:
        assert plan is None
        return
    assert plan is not None
    assert plan.num_encoder_tokens == 413
    assert plan.num_replay_tokens == WINDOW + 7 + 6
    assert plan.replay_num_cached == [400, 700, 900]
    assert plan.swa_floors == [400, 0, 0]
    torch.testing.assert_close(plan.rows, torch.arange(272, 413), atol=0, rtol=0)

    # Encoder handoffs are published after planning and cleared by prepare().
    topk = torch.arange(413 * 3, dtype=torch.int32).reshape(413, 3)
    candidates = torch.arange(413 * 2, dtype=torch.int32).reshape(413, 2)
    metadata.csa2_indices[20] = topk
    metadata.csa2_candidates[20] = candidates
    metadata.csa2_precomputed_kv_layers = {20}
    metadata._csa2_last_layer = 20
    saved_prompt = metadata.prompt_lens
    original_buffer = metadata.kv_lens_cuda
    enter_decoder_replay(metadata, plan)
    assert metadata.prepare_calls == 1
    assert metadata.kv_lens_cuda is original_buffer
    torch.testing.assert_close(metadata.kv_lens_cuda, expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(metadata.csa2_positions, positions[plan.rows], atol=0, rtol=0)
    torch.testing.assert_close(metadata.csa2_indices[20], topk[plan.rows], atol=0, rtol=0)
    torch.testing.assert_close(metadata.csa2_candidates[20], candidates[plan.rows], atol=0, rtol=0)
    assert metadata.csa2_precomputed_kv_layers == {20}
    assert metadata._csa2_last_layer == 20
    assert metadata.csa2_replay_query_rows is plan.rows
    indices = _swa(metadata, 0)[0]
    torch.testing.assert_close((indices[:WINDOW] >= 0).sum(-1), torch.arange(1, WINDOW + 1))
    torch.testing.assert_close(indices[WINDOW - 1], torch.arange(400, 528))
    torch.testing.assert_close(indices[WINDOW:], untrimmed_swa, atol=0, rtol=0)
    assert metadata.prompt_lens == [WINDOW, 7, 700]
    exit_decoder_replay(metadata, plan)
    assert metadata.request_ids == [10, 11, 12] and metadata.num_contexts == 2
    assert metadata.kv_cache_params is original_params
    assert metadata.prompt_lens is saved_prompt
    assert metadata.seq_lens is plan.saved_seq_lens
    assert metadata.seq_lens.tolist() == [400, 7, 6]
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [128, 700, 900]
    assert metadata.csa2_replay_query_rows is None
    torch.testing.assert_close(metadata.kv_lens_cuda, expected_kv, atol=0, rtol=0)


def test_remote_tail_decoder_masks_only_pre_handoff_context_swa() -> None:
    metadata = _LifecycleMetadata([4, 3], 1, [100, 200])
    metadata.prepare()
    metadata._ensure_swa_slots()
    generation_indices = metadata.csa2_swa_indices[0][4:].clone()
    write_slots = metadata.csa2_swa_write_slots[0].clone()

    enter_remote_tail_decoder(metadata, [102])
    # The next decoder layer resolves the slots against the raised floors.
    metadata._ensure_swa_slots()

    context_indices = metadata.csa2_swa_indices[0][:4]
    assert torch.all(context_indices[context_indices >= 0] >= 102)
    torch.testing.assert_close(metadata.csa2_swa_indices[0][4:], generation_indices, atol=0, rtol=0)
    # The floor hides reads only: rows below it still store their SWA.
    torch.testing.assert_close(metadata.csa2_swa_write_slots[0], write_slots, atol=0, rtol=0)


def _assert_swa_cpu_reference(metadata, manager, requests, starts, lengths, floors, observed=None):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole

    positions = torch.tensor(
        [start + i for start, length in zip(starts, lengths) for i in range(length)],
        dtype=torch.int64,
        device="cpu",
    )
    rows = torch.tensor(
        [r for r, length in enumerate(lengths) for _ in range(length)],
        dtype=torch.int64,
        device="cpu",
    )
    window, block = manager.layout.window_size, manager.tokens_per_block
    logical = positions[:, None] - window + 1 + torch.arange(window, device="cpu")
    columns = logical.clamp_min(0) // block
    floor = torch.tensor(floors, dtype=torch.int64, device="cpu")[rows, None]
    for layer, ratio in enumerate(manager.layout.compress_ratios):
        pages = torch.full(
            (len(requests), (manager.max_seq_len + block - 1) // block),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        for r, request in enumerate(requests):
            entries = manager.get_cache_indices(request.py_request_id, layer, CSA2CacheRole.SWA)
            n = min(len(entries), pages.shape[1])
            pages[r, :n] = torch.tensor(entries[:n], dtype=torch.int32, device="cpu")
        physical = pages[rows[:, None], columns].long()
        reads = torch.where(
            (logical >= floor) & (physical >= 0), physical * block + logical % block, -1
        )
        visible = (positions + 1) // ratio if ratio else torch.zeros_like(positions)
        actual = _swa(metadata, layer) if observed is None else observed[layer]
        for value, expected in zip(actual, (reads, reads[:, -1], visible)):
            torch.testing.assert_close(value.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_coalesced_uploads_mixed_dtypes_empty_values_and_slab_growth():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import _CoalescedUploads

    metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
    vars(metadata).update(
        is_cuda_graph=False, kv_lens_cuda=torch.empty(0, dtype=torch.int32, device="cuda")
    )
    for size in (3, 1000):  # The second round regrows the upload slab.
        values = {
            "odd": np.arange(size, dtype=np.int32),
            "wide": np.arange(2 * size, dtype=np.int64).reshape(size, 2) * 7,
            "empty": np.empty(0, dtype=np.int64),
            "scalar": torch.tensor(size, dtype=torch.int32),
        }
        uploads = _CoalescedUploads(metadata)
        outputs = {key: uploads.copy(key, value) for key, value in values.items()}
        uploads.flush()
        for key, value in values.items():
            expected = torch.as_tensor(np.asarray(value))
            torch.testing.assert_close(outputs[key].cpu(), expected, atol=0, rtol=0)


@pytest.mark.cpu_only
def test_swa_write_interval_validation_preserves_errors():
    def check(pages, starts, lengths, floors, allocated, block):
        rows, columns = CSA2TrtllmMetadata._swa_write_pages(
            starts, lengths, floors, allocated, block, pages.shape
        )
        CSA2TrtllmMetadata._check_written_pages(pages.numpy()[rows, columns] >= 0)

    full = [1 << 30]
    check(torch.tensor([[-1, 7]]), [128], [2], [0], full, 128)  # Historical-only hole.
    check(torch.empty(1, 0, dtype=torch.int32), [0], [0], [0], full, 128)
    # Queries beyond the allocated capacity (overlap reservation) need no page.
    check(torch.tensor([[7, -1]]), [127], [2], [0], [128], 128)
    check(torch.tensor([[7]]), [128], [1], [0], [128], 128)
    check(torch.tensor([[7]]), [256], [3], [0], [128], 128)
    for pages, start, length, floor in (
        ([[7, -1]], 128, 2, 0),
        ([[7]], 128, 1, 0),
        ([[]], 0, 1, 0),
        ([[7, 8]], 127, 2, 128),
        ([[7, 8]], 0, 257, 0),
    ):
        with pytest.raises(ValueError, match="writable SWA pages"):
            check(torch.tensor(pages, dtype=torch.int32), [start], [length], [floor], full, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("queries", [1, 2731, 8192])
@torch.inference_mode()
def test_forward_swa_resolution_matches_cpu_reference(queries):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig

    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False, max_gpu_total_bytes=128 << 20, enable_swa_scratch_reuse=True
        ),
        CacheType.SELFKONLY,
        layout=CSA2Layout((0, 2, 1), (1, 2), (1, 2)),
        num_layers=3,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=8320,
        max_batch_size=1,
        max_input_len=8192,
        max_num_tokens=8192,
        dtype=DataType.BF16,
        vocab_size=1024,
    )
    request = LlmRequest(
        request_id=314,
        max_new_tokens=1,
        input_tokens=[i % 1024 for i in range(queries)],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=8192, kv_cache_manager=manager
        )
        metadata.request_ids = [request.py_request_id]
        metadata.seq_lens = torch.tensor([queries], dtype=torch.int32, device="cpu")
        metadata.num_contexts = 1
        metadata.prompt_lens = [queries]
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
        metadata.prepare()
        _assert_swa_cpu_reference(metadata, manager, [request], [0], [queries], [0])
        assert metadata._csa2_ready_for_kv_update
    finally:
        manager.free_resources(request)
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_swa_resolution_integer_outputs_and_graph(monkeypatch, native):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel

    if native and not kernel.dsl_available():
        pytest.skip("The native SWA resolution requires the CuTe DSL")
    if not native:
        monkeypatch.setattr(kernel, "dsl_available", lambda: False)
    ratios, window, block = (0, 1, 2), 128, 128
    for count, table_rows, table_width in ((0, 2, 3), (1, 2, 0), (2, 0, 3), (8, 2, 3)):
        positions_host = [0, 131, 129, -1, 257, 260, 256, 9][:count]
        requests_host = [0, 0, 1, 1, -1, 2, 1, 0][:count]
        floors_host = [0, 129][:table_rows]
        # Decoder floors hide positions from reads while writes keep ``floors``.
        read_floors_host = [0, 258][:table_rows]
        pages_host = [[(1 << 25), -1, 7], [3, 4, 5]][:table_rows]
        pages_host = [row[:table_width] for row in pages_host]
        positions = torch.tensor(positions_host, dtype=torch.int32, device="cuda")
        requests = torch.tensor(requests_host, dtype=torch.int64, device="cuda")
        metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
        vars(metadata).update(
            is_cuda_graph=False,
            _csa2_buffers={},
            kv_lens_cuda=positions,
            kv_cache_manager=SimpleNamespace(
                tokens_per_block=block,
                layout=SimpleNamespace(window_size=window, compress_ratios=ratios),
            ),
            csa2_positions=positions,
            csa2_token_requests=requests,
            csa2_replay_start_positions=torch.tensor(floors_host, dtype=torch.int64, device="cuda"),
            _csa2_swa_read_floors=torch.tensor(read_floors_host, dtype=torch.int64, device="cuda"),
            _csa2_swa_page_tables={
                layer: torch.tensor(pages_host, dtype=torch.int32, device="cuda").reshape(
                    table_rows, table_width
                )
                for layer in (2, 0, 1)
            },
        )

        def bind():
            if native:
                metadata._csa2_swa_descriptors = metadata._swa_table_descriptors(3)
                metadata._csa2_swa_ratios = metadata._swa_ratio_tensor(ratios)

        outputs = [
            tuple(
                torch.full(shape, 91, dtype=torch.int64, device="cuda")
                for shape in ((count, window), (count,), (count,))
            )
            for _ in ratios
        ]

        def resolve_all(force=True):
            if force:
                metadata._csa2_swa_resolved = False
            metadata._ensure_swa_slots()
            for layer, kept in enumerate(outputs):
                for name, out in zip(
                    ("csa2_swa_indices", "csa2_swa_write_slots", "csa2_visible_lengths"), kept
                ):
                    out.copy_(getattr(metadata, name)[layer])

        def check():
            # Independent scalar CPU oracle, including invalid request domains.
            reads, writes = [], []
            for position, request in zip(positions_host, requests_host):
                row = []
                for logical in range(position - window + 1, position + 1):
                    valid = 0 <= request < table_rows and logical >= 0
                    valid = valid and logical >= floors_host[request]
                    valid = valid and logical // block < table_width
                    page = pages_host[request][logical // block] if valid else -1
                    row.append(page * block + logical % block if page >= 0 else -1)
                writes.append(row[-1])
                if 0 <= request < table_rows:
                    floor = read_floors_host[request] - position + window - 1
                    row = [-1] * min(window, max(0, floor)) + row[max(0, floor) :]
                reads.append(row)
            reads = torch.tensor(reads, dtype=torch.int64, device="cpu").reshape(count, window)
            writes = torch.tensor(writes, dtype=torch.int64, device="cpu")
            for kept, ratio in zip(outputs, ratios):
                visible = torch.tensor(
                    [(position + 1) // ratio if ratio else 0 for position in positions_host],
                    dtype=torch.int64,
                    device="cpu",
                )
                for value, expected in zip(kept, (reads, writes, visible)):
                    torch.testing.assert_close(value.cpu(), expected, atol=0, rtol=0)

        bind()
        resolve_all()
        check()
        metadata.is_cuda_graph = True
        bind()
        resolve_all()  # Warm the graph-mode buffers and kernels before capture.
        check()
        if count != 8:
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Eager warmup resolved these inputs; the captured forward clears
            # the flag at its start and records the launch.
            metadata.reset_routing()
            resolve_all(force=False)
        for page in (-1, 13):
            pages_host[0][0] = page
            floors_host[0] = 1 if page < 0 else 0
            requests_host.reverse()
            positions_host[1] += 1
            positions.copy_(torch.tensor(positions_host, dtype=torch.int32, device="cuda"))
            requests.copy_(torch.tensor(requests_host, dtype=torch.int64, device="cuda"))
            metadata.csa2_replay_start_positions.copy_(
                torch.tensor(floors_host, dtype=torch.int64, device="cuda")
            )
            for layer in range(3):
                metadata._csa2_swa_page_tables[layer][0, 0] = page
            for kept in outputs:
                for out in kept:
                    out.fill_(94)
            graph.replay()
            check()


@torch.inference_mode()
def test_indexer_descriptors_expand_pages_and_mask_missing_pages():
    # Descriptor-only high physical IDs must never reach a native cache gather.
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        for page_size, columns in ((64, 2), (128, 2), (128, 0)):
            factor = page_size // 64
            source = [[1 << 25, -1], [3, 5]]
            source = [row[:columns] for row in source]
            visibility = [128, 193, 65, 0, 160, 9]
            tokens = [1, 1, 2, 2, -1, 3]
            table = torch.tensor(source, dtype=torch.int32, device=device).reshape(2, columns)
            decode_visible = torch.tensor(visibility, dtype=torch.int64, device=device)
            requests = torch.tensor(tokens, dtype=torch.int64, device=device)
            blocks = torch.full((6, 4), 92, dtype=torch.int32, device=device)
            contexts = torch.full((6, 1), 93, dtype=torch.int32, device=device)
            visible = torch.full((6,), 94, dtype=torch.int32, device=device)
            valid = torch.ones((6, 256), dtype=torch.bool, device=device)
            outputs = (blocks, contexts, visible, valid)
            pointers = [value.data_ptr() for value in outputs]
            originals = [value.clone() for value in (table, decode_visible, requests)]
            page_ids, page_valid = [], []
            for request in range(2):
                ids, allocated = [], []
                for page in range(4):
                    column = page // factor
                    physical = source[request][column] if column < columns else -1
                    allocated.append(physical >= 0)
                    ids.append(physical * factor + page % factor if physical >= 0 else 0)
                page_ids.append(ids)
                page_valid.append(allocated)
            expected_blocks, expected_visible, expected_valid = [], [], []
            for token, end in zip(tokens, visibility):
                request = token - 1
                admitted = 0 <= request < 2
                expected_blocks.append(page_ids[request] if admitted else [0] * 4)
                expected_visible.append(end if admitted else 0)
                expected_valid.append(
                    [
                        admitted and position < end and page_valid[request][position // 64]
                        for position in range(256)
                    ]
                )
            expected = (
                expected_blocks,
                [[max(length, 1)] for length in expected_visible],
                expected_visible,
                expected_valid,
            )
            with torch.inference_mode():
                CSA2TrtllmMetadata._fill_indexer_descriptors(
                    table, requests, decode_visible, 1, factor, 64, *outputs
                )
                if device == "cuda" and page_size == 128 and columns:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        CSA2TrtllmMetadata._fill_indexer_descriptors(
                            table, requests, decode_visible, 1, factor, 64, *outputs
                        )
                    for value in outputs:
                        value.fill_(96)
                    graph.replay()
            for value, reference in zip(outputs, expected):
                torch.testing.assert_close(
                    value.cpu(), torch.tensor(reference, dtype=value.dtype), atol=0, rtol=0
                )
            assert pointers == [value.data_ptr() for value in outputs]
            for value, original in zip((table, decode_visible, requests), originals):
                torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("published_first", [False, True])
def test_swa_table_backing_stays_fixed_across_publication_and_fallback(published_first):
    metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
    metadata._csa2_buffers = {}
    metadata.kv_lens_cuda = torch.empty(0, dtype=torch.int32)
    metadata.is_cuda_graph = True
    outer = torch.arange(16, dtype=torch.int32).reshape(2, 2, 4)
    published = outer[:, 0, :3]
    host = torch.full((2, 3), 17, dtype=torch.int32)
    first = metadata._bind_swa_pages(0, host, published if published_first else None)
    pointer, stride = first.data_ptr(), first.stride()
    assert (pointer == published.data_ptr()) == published_first
    # Direct fallback refreshes captured backing, even when that backing is an
    # outer-table view rather than the old private contiguous allocation.
    fallback = metadata._bind_swa_pages(0, host, None)
    assert (fallback.data_ptr(), fallback.stride()) == (pointer, stride)
    torch.testing.assert_close(fallback, host, atol=0, rtol=0)
    other = torch.full((2, 2, 4), 29, dtype=torch.int32)[:, 0, :3]
    normal = metadata._bind_swa_pages(0, host, other)
    assert (normal.data_ptr(), normal.stride()) == (pointer, stride)
    torch.testing.assert_close(normal, other, atol=0, rtol=0)

    # The native SWA refresh reads a [layers, 4] page-table descriptor at replay
    # time, so graph metadata keeps one descriptor buffer per table binding.
    def rows(tables):
        return [[t.data_ptr(), t.stride(0), t.shape[0], t.shape[1]] for t in tables]

    small = [torch.zeros((1, 4), dtype=torch.int32) for _ in range(2)]
    large = [torch.zeros((8, 4), dtype=torch.int32) for _ in range(2)]
    metadata._csa2_swa_page_tables = dict(enumerate(small))
    first = metadata._swa_table_descriptors(2)
    metadata._csa2_swa_page_tables = dict(enumerate(large))
    second = metadata._swa_table_descriptors(2)
    assert first.data_ptr() != second.data_ptr() and second.tolist() == rows(large)
    metadata._csa2_swa_page_tables = dict(enumerate(small))
    assert metadata._swa_table_descriptors(2).data_ptr() == first.data_ptr()
    # Eager metadata refreshes a single buffer in place before every launch.
    metadata.is_cuda_graph = False
    eager = metadata._swa_table_descriptors(2)
    metadata._csa2_swa_page_tables = dict(enumerate(large))
    assert metadata._swa_table_descriptors(2).data_ptr() == eager.data_ptr()
    assert eager.tolist() == rows(large) and first.tolist() == rows(small)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_swa_fresh_publication_reuses_tables_and_keeps_captured_backing(
    manager_requests, monkeypatch
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    # Physical publication order need not equal logical model-layer order.
    order = [2, 0, 3, 1]
    monkeypatch.setattr(manager, "pp_layers", order)
    monkeypatch.setattr(manager, "layer_offsets", {layer: i for i, layer in enumerate(order)})
    manager._init_batch_page_plan()
    conversions = []
    convert = manager.compute_batch_page_tables

    def counted(*args):
        conversions.append(True)
        return convert(*args)

    monkeypatch.setattr(manager, "compute_batch_page_tables", counted)
    # Address-only oracle: model COW remapping under stable request IDs by
    # rewriting layer 0's base SWA pages, without dereferencing synthetic pages.
    base = manager.host_kv_cache_block_offsets
    rows = manager.index_mapper.get_copy_index([r.py_request_id for r in requests], 0, 1).long()
    pool = manager.layer_to_pool_mapping_dict[manager._layer_roles[0, CSA2CacheRole.SWA]]
    saved = base[pool, rows, 0, :2].clone()

    def remap(shift):
        base[pool, rows, 0, 0] = -1
        base[pool, rows, 0, 1] = (1 << 20) + shift

    remap(0)
    _prepare(metadata, requests, [130, 131], [2, 1])
    # The publication converts once and aliases the generic upload.
    assert len(conversions) == 1
    pointers = {
        layer: (value.data_ptr(), value.stride())
        for layer, value in metadata._csa2_swa_page_tables.items()
    }
    for layer, value in metadata._csa2_swa_page_tables.items():
        assert (
            value.data_ptr()
            == metadata.kv_cache_block_offsets[manager.layer_offsets[layer], :2, 0, :4].data_ptr()
        )
    output_pointers = tuple(value.data_ptr() for value in _swa(metadata, 0))
    kept = {layer: tuple(value.clone() for value in _swa(metadata, layer)) for layer in range(4)}

    def refresh_and_resolve():
        metadata.on_update_kv_lens()
        for layer, outputs in kept.items():
            for value, out in zip(_swa(metadata, layer), outputs):
                out.copy_(value)

    for _ in range(3):
        refresh_and_resolve()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refresh_and_resolve()
    for direct, order, shift in (
        (True, requests, 1),
        (False, requests[::-1], 2),
        (True, requests[::-1], 3),
    ):
        remap(shift)
        conversions.clear()
        if direct:
            metadata.prepare_csa2()
        else:
            _prepare(metadata, order, [130, 131], [2, 1])
        assert len(conversions) == 1
        assert manager._swa_publication is manager._swa_publication_token is None
        assert pointers == {
            layer: (value.data_ptr(), value.stride())
            for layer, value in metadata._csa2_swa_page_tables.items()
        }
        assert output_pointers == tuple(value.data_ptr() for value in _swa(metadata, 0))
        graph.replay()
        _assert_swa_cpu_reference(metadata, manager, order, [130, 131], [2, 1], [0, 0], kept)
    # A valid publication narrower than the old logical page bound must fall
    # back, without shrinking/rebinding the already captured page storage.
    full_offsets = metadata.kv_cache_block_offsets
    metadata.kv_cache_block_offsets = full_offsets[..., :2]
    conversions.clear()
    try:
        _prepare(metadata, requests, [130, 131], [2, 1])
        assert len(conversions) == 2
        assert pointers == {
            layer: (value.data_ptr(), value.stride())
            for layer, value in metadata._csa2_swa_page_tables.items()
        }
        _assert_swa_cpu_reference(metadata, manager, requests, [130, 131], [2, 1], [0, 0])
    finally:
        metadata.kv_cache_block_offsets = full_offsets
        base[pool, rows, 0, :2] = saved


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows,window", [(0, 128), (1, 1), (3, 128), (4096, 128)])
def test_swa_layer_reference_matches_scalar_oracle(rows, window):
    ratios = (0, 2, 1)
    block = 128
    positions = list(range(rows))
    requests_host = torch.tensor([0] * (rows // 2) + [1] * (rows - rows // 2), dtype=torch.int64)
    # Request 1 replays from position 127 only when its queries lie beyond it.
    floors = [0, 127 if rows > 256 else 0]
    page_count = rows // block + 1
    generator = torch.Generator().manual_seed(rows)
    tables = [
        torch.randint(0, 8, (2, page_count), dtype=torch.int32, generator=generator) for _ in ratios
    ]
    if rows > 256:
        # Request 1 starts at floor 127: page 0 is history below the floor, never a write target.
        tables[1][1, 0] = -1
    metadata = CSA2TrtllmMetadata.__new__(CSA2TrtllmMetadata)
    vars(metadata).update(
        is_cuda_graph=False,
        _csa2_buffers={},
        kv_lens_cuda=torch.empty(0, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=block,
            layout=SimpleNamespace(window_size=window, compress_ratios=ratios),
        ),
        csa2_positions=torch.tensor(positions, dtype=torch.int32),
        csa2_token_requests=requests_host,
        csa2_replay_start_positions=torch.tensor(floors, dtype=torch.int64),
        _csa2_swa_page_tables=dict(enumerate(tables)),
    )
    for layer, ratio in enumerate(ratios):
        reads, writes, visible = _swa(metadata, layer)
        expected = torch.full((rows, window), -1, dtype=torch.int64)
        for row in range(rows):
            request = int(requests_host[row])
            for column in range(window):
                logical = positions[row] - window + 1 + column
                page = int(tables[layer][request, max(logical, 0) // block])
                if logical >= floors[request] and page >= 0:
                    expected[row, column] = page * block + logical % block
        torch.testing.assert_close(reads, expected, atol=0, rtol=0)
        torch.testing.assert_close(writes, expected[:, -1], atol=0, rtol=0)
        expected_visible = torch.tensor(
            [(pos + 1) // ratio if ratio else 0 for pos in positions], dtype=torch.int64
        )
        torch.testing.assert_close(visible, expected_visible, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu_only),
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_swa_refresh_preserves_per_layer_page_domains(device: str) -> None:
    positions_host = [0, 128, 255, -1, 300, 259]
    requests_host = [0, 1, 2, -1, 0, 1]
    floors_host = [0, 129]
    window, block = 128, 128
    pages_host = [
        [[1 << 25, -1, 7], [3, 4, 5]],
        [[9]],
        [[], [], []],
        [],
    ]
    shapes = [(2, 3), (1, 1), (3, 0), (0, 3)]
    ratios = (0, 1, 2, 1)
    positions = torch.tensor(positions_host, dtype=torch.int32, device=device)
    requests = torch.tensor(requests_host, dtype=torch.int64, device=device)
    floors = torch.tensor(floors_host, dtype=torch.int64, device=device)
    layer_inputs = tuple(
        (
            torch.tensor(pages, dtype=torch.int32, device=device).reshape(shape),
            torch.full((len(positions_host), window), 91, dtype=torch.int64, device=device),
            torch.full((len(positions_host),), 92, dtype=torch.int64, device=device),
            torch.full((len(positions_host),), 93, dtype=torch.int64, device=device),
            ratio,
        )
        for pages, shape, ratio in zip(pages_host, shapes, ratios)
    )
    CSA2TrtllmMetadata._refresh_swa_tensor_outputs(
        positions, requests, floors, block, window, layer_inputs
    )
    for (_, reads, writes, visible, ratio), pages, (table_rows, width) in zip(
        layer_inputs, pages_host, shapes
    ):
        expected_rows = []
        for position, request in zip(positions_host, requests_host):
            expected_row = []
            for logical in range(position - window + 1, position + 1):
                valid = 0 <= request < min(table_rows, len(floors_host)) and logical >= 0
                valid = valid and logical >= floors_host[request] and logical // block < width
                page = pages[request][logical // block] if valid else -1
                expected_row.append(page * block + logical % block if page >= 0 else -1)
            expected_rows.append(expected_row)
        expected = torch.tensor(expected_rows, dtype=torch.int64)
        torch.testing.assert_close(reads.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(writes.cpu(), expected[:, -1], atol=0, rtol=0)
        expected_visible = torch.tensor(
            [(position + 1) // ratio if ratio else 0 for position in positions_host],
            dtype=torch.int64,
        )
        torch.testing.assert_close(visible.cpu(), expected_visible, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_context_tile_cache_retains_latest_and_preserves_graph_and_dtype():
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=64)
    q = torch.zeros(64, 8, 512, dtype=torch.bfloat16, device="cuda")
    generation = source.get_query_tile_metadata(q[:2], 0)
    generation.swa_pool.zero_()
    generation.workspace.resize_(32).fill_(47)
    generation_workspace = generation.workspace
    generation_workspace_pointer = generation.workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert source.get_query_tile_metadata(q[:2], 0) is generation
        captured = generation.swa_pool.view(torch.uint8).clone()
        captured_workspace = generation.workspace.clone()
    generation_pointer = generation.swa_pool.data_ptr()
    source._num_ctx_tokens = source._num_tokens = 64
    source._num_contexts = 2
    source.csa2_request_query_ranges = ((0, 32), (32, 64))
    source.csa2_request_start_positions = (0, 512)
    source.csa2_request_lengths = (32, 32)
    protected = {}
    for heads, extra, dtype in (
        (8, 0, torch.bfloat16),
        (8, 0, torch.float8_e4m3fn),
        (8, 17, torch.bfloat16),
        (16, 0, torch.bfloat16),
    ):
        queries = torch.zeros(64, heads, 512, dtype=torch.bfloat16, device="cuda")
        workspace = None
        high_water = 0
        for count in (16, 7, 12, 16, 5, 48, 9, 48):
            frame = source.get_query_tile_metadata(
                queries[:count], extra, query_start=0, staging_dtype=dtype
            )
            assert frame.swa_pool.dtype == dtype
            assert frame.num_tokens == count and frame.num_ctx_tokens == count
            assert frame.num_contexts == (2 if count > 32 else 1)
            if workspace is None:
                workspace = frame.workspace
            assert frame.workspace is frame.cuda_graph_workspace is workspace
            assert frame.workspace.numel() == high_water
            high_water = max(high_water, 32 + count)
            frame.workspace.resize_(high_water).fill_(count)
            for previous in protected.values():
                assert any(tile is previous for tile in source._csa2_query_tiles.values())
                assert previous.workspace is not frame.workspace
            # Only the latest shape of an eager context family stays resident.
            assert (
                sum(
                    1
                    for key, tile in source._csa2_query_tiles.items()
                    if tile.num_contexts
                    and tile.swa_pool.dtype == dtype
                    and key[1] == heads
                    and key[3] == (extra + 127) // 128 * 128
                )
                == 1
            )
            assert generation.swa_pool.data_ptr() == generation_pointer
            assert generation.workspace is generation_workspace
            assert generation.workspace.data_ptr() == generation_workspace_pointer
            assert frame.workspace is not generation_workspace
            assert source.get_query_tile_metadata(q[:2], 0) is generation
            generation.swa_pool.fill_(count)
            expected = generation.swa_pool.view(torch.uint8).clone()
            graph.replay()
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                captured_workspace, torch.full_like(captured_workspace, 47), atol=0, rtol=0
            )
        protected[(heads, extra, dtype)] = frame
    assert len(protected) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_context_domains_follow_owner_source_batch_and_replay(manager_requests):
    manager, requests = manager_requests
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    source.set_source_batch([200, 200], [0, 0])
    _prepare(source, requests, [130, 131], [2, 1])
    q = torch.zeros(3, 8, 512, device="cuda", dtype=torch.bfloat16)
    child = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=1)
    workspace = child.workspace
    workspace.resize_(64).fill_(11)
    other_owner = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=3)
    other_workspace = other_owner.workspace
    other_workspace.resize_(96).fill_(17)
    assert other_workspace is not workspace
    first = child.shared_plan
    assert first is not None
    assert first.main_rows == sum(source._csa2_main_domain_counts[1])
    assert first.main_rows >= 200  # source visibility, plus admitted page capacity
    assert first.swa_rows == 257
    assert first.swa_starts.tolist() == [3, 4]
    assert child.max_num_requests == 2 and child.num_tokens == 3
    assert child.swa_pool.data_ptr() == child.extra_pool.data_ptr() == child.shared_pool.data_ptr()
    assert child.shared_pool.shape == (1 + first.swa_rows + first.main_rows, 512)
    reuse = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=2)
    assert reuse is child
    assert reuse.shared_plan.main_pages.data_ptr() == first.main_pages.data_ptr()
    assert reuse.shared_plan.swa_pages.data_ptr() != first.swa_pages.data_ptr()
    # Pure decoder replay has no newly projected GLOBAL rows, but reads cached C.
    source.set_swa_bounded_replay([130, 130], decoder=True)
    _prepare(source, requests, [2, 2], [128, 128])
    q = torch.zeros(256, 8, 512, device="cuda", dtype=torch.bfloat16)
    child = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=1)
    assert child.shared_plan is not None and child.shared_plan.main_rows >= 130
    assert child.workspace is child.cuda_graph_workspace is workspace
    assert child.workspace.numel() == 64
    other_owner = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=3)
    assert other_owner.workspace is other_owner.cuda_graph_workspace is other_workspace
    assert other_workspace.numel() == 96
    torch.testing.assert_close(workspace, torch.full_like(workspace, 11), atol=0, rtol=0)
    torch.testing.assert_close(
        other_workspace, torch.full_like(other_workspace, 17), atol=0, rtol=0
    )
    assert child.max_num_requests == 2 and child.num_tokens == 256
    assert child.shared_plan.swa_starts.tolist() == [2, 2]
    assert child.shared_pool.shape[0] < q.shape[0] * 640
    # Captured context keeps its existing independent-generation frame.
    source.is_cuda_graph = True
    selected = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=1)
    assert selected.shared_plan is None and selected.swa_pool.shape == (256, 128, 512)
    assert selected.num_contexts == 0
    assert selected.workspace is not workspace and selected.workspace is not other_workspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_context_covers_upward_live_endpoint_inside_published_pages(manager_requests):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

    from .test_backend import _backend

    manager, requests = manager_requests
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    source.set_source_batch([200, 200], [0, 0])
    _prepare(source, requests, [130, 131], [2, 1])
    source.kv_lens_cuda.fill_(260)
    source.on_update_kv_lens()
    logical = torch.tensor([[110], [111], [112]], dtype=torch.int64, device="cuda")
    slots = source.global_slot_tile(1, 0, 3, logical)
    assert torch.all(slots >= 0)
    manager.get_swa_buffer(1).zero_()
    manager.get_main_buffer(1).zero_()
    manager.write_global(
        1,
        slots.flatten(),
        torch.full((3, 512), 32, device="cuda", dtype=torch.bfloat16),
        torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16),
    )
    q = torch.zeros(3, 8, 512, device="cuda", dtype=torch.bfloat16)
    shared = source.get_query_tile_metadata(q, 512, query_start=0, layer_idx=1)
    assert shared.shared_plan is not None
    assert shared.shared_plan.main_rows > 2 * (200 // 2)
    selected = CSA2TrtllmMetadata.for_query_tile(q, 512, context_lengths=[2, 1])
    backend = _backend(8)

    def arguments():
        return AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            attention_sinks=torch.zeros(8, device="cuda"),
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_pool=manager.get_swa_buffer(1),
                swa_indices=_swa(source, 1)[0],
                main_pool=manager.get_main_buffer(1),
                topk_indices=slots,
                main_logical_indices=logical,
            ),
        )

    expected = backend.forward(q.flatten(1), None, None, selected, forward_args=arguments())
    actual = backend.forward(q.flatten(1), None, None, shared, forward_args=arguments())
    assert actual.abs().max() > 0.1
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


# Without SWA scratch the compressor-state pages come from the base page table.
@pytest.mark.parametrize("manager_requests", [{"scratch": False}], indirect=True)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compressor_readiness_uses_current_rows(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    base = manager.host_kv_cache_block_offsets
    row = int(manager.index_mapper.get_copy_index([requests[0].py_request_id], 1, 1)[0])

    def prepare_source(length):
        metadata.set_source_batch([length, 0], [127, 0])
        _prepare(metadata, requests, [4, 4], [1, 1])

    prepare_source(2)  # Source spans logical pages 0 and 1.
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        pool = manager.layer_to_pool_mapping_dict[manager._layer_roles[1, role]]
        saved = int(base[pool, row, 0, 1])
        base[pool, row, 0, 1] = -1
        try:
            with pytest.raises(
                ValueError, match="source rows have no allocated compressor state pages"
            ):
                prepare_source(2)
            prepare_source(0)  # An empty source must not validate unused missing pages.
        finally:
            base[pool, row, 0, 1] = saved
    prepare_source(2)  # A corrected mapping in the next prepare is observed.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_first_decode_after_context_takes_full_prepare(manager_requests):
    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    # A one-token context chunk (a full prefix hit) and then its first decode
    # step: the context geometry must not be reused for the generation rows.
    _prepare(metadata, requests, [126, 129], [1, 1])
    _prepare_decode_metadata(metadata, requests, [127, 130])
    assert metadata.csa2_num_context_requests == 0
    assert metadata.csa2_decode_row_requests is not None


def _decode_derived_outputs(metadata):
    # SWA outputs are resolved when a forward enters its first layer.
    metadata._ensure_swa_slots()
    outputs = {"positions": metadata.csa2_positions}
    for name in (
        "csa2_swa_indices",
        "csa2_swa_write_slots",
        "csa2_visible_lengths",
        "csa2_main_write_slots",
        "_csa2_compressed_positions",
    ):
        outputs.update({(name, layer): tensor for layer, tensor in getattr(metadata, name).items()})
    for owner, batch in metadata._csa2_compression.items():
        for name in ("start_positions", "kv_lengths", "cu_compressed_lengths"):
            outputs[owner, name] = getattr(batch, name)
    return outputs


def _prepare_decode_metadata(metadata, requests, starts, *, deferred=False):
    from contextlib import nullcontext

    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.ones(len(requests), dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.prompt_lens = [request.prompt_len for request in requests]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    with metadata.defer_cuda_graph_decode_outputs() if deferred else nullcontext():
        metadata.prepare()
    assert not metadata._csa2_defer_decode_outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_steady_decode_refreshes_page_tables_across_pages(manager_requests, monkeypatch):
    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    refreshed = []
    refresh = metadata._refresh_steady_page_tables
    monkeypatch.setattr(
        metadata,
        "_refresh_steady_page_tables",
        lambda *args: refreshed.append(args[2][0]) or refresh(*args),
    )
    # Consecutive steps stay on the steady path; crossing the page at 128
    # re-reads the tables, which must equal a full preparation's.
    for start in range(125, 131):
        starts = [start, start + 3]
        _prepare_decode_metadata(metadata, requests, starts)
        reference = CSA2TrtllmMetadata(
            max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager
        )
        _prepare_decode_metadata(reference, requests, starts)
        observed = _decode_derived_outputs(metadata)
        for name, tensor in _decode_derived_outputs(reference).items():
            torch.testing.assert_close(observed[name], tensor)
        for owner, table in reference.csa2_global_page_tables.items():
            torch.testing.assert_close(metadata.csa2_global_page_tables[owner], table)
    assert refreshed == [128]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.inference_mode()
def test_decode_graph_defers_derived_uploads_and_refreshes_all_outputs(
    manager_requests, monkeypatch, batch_size
):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm.bindings import SamplingConfig

    manager, requests = manager_requests
    reference = CSA2TrtllmMetadata(
        max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager
    )
    base = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    metadata = base.create_cuda_graph_metadata(batch_size)
    copied = []
    copy_host = metadata._copy_host

    def record_copy(key, destination, value):
        copied.append(key)
        return copy_host(key, destination, value)

    monkeypatch.setattr(metadata, "_copy_host", record_copy)
    selected = requests[:batch_size]
    starts = [127] * batch_size
    _prepare_decode_metadata(metadata, selected, starts, deferred=True)
    assert metadata._csa2_deferred_decode_outputs
    engine = object.__new__(PyTorchModelEngine)
    engine.enable_spec_decode = False
    engine.guided_decoder = None
    pointers = {
        name: tensor.data_ptr() for name, tensor in _decode_derived_outputs(metadata).items()
    }

    def forward():
        # The real engine's preprocessing is captured ahead of every consumer.
        engine._preprocess_inputs({"attn_metadata": metadata})
        return {name: tensor.clone() for name, tensor in _decode_derived_outputs(metadata).items()}

    for _ in range(2):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed = forward()
    # Capture only records the clone operations; first replay initializes the
    # captured outputs, exactly as the engine does after its first capture.
    graph.replay()
    _prepare_decode_metadata(reference, selected, starts)
    for name, tensor in _decode_derived_outputs(reference).items():
        torch.testing.assert_close(observed[name], tensor, atol=0, rtol=0)
    # Consecutive decode steps inside the same pages take the upstream steady
    # preparation path. Selected graphs still own every live endpoint refresh.
    for steady_start in (130, 131, 132):
        starts = [steady_start] * batch_size
        _prepare_decode_metadata(reference, selected, starts)
        expected = {
            name: tensor.clone() for name, tensor in _decode_derived_outputs(reference).items()
        }
        copied.clear()
        _prepare_decode_metadata(metadata, selected, starts, deferred=True)
        assert metadata._csa2_deferred_decode_outputs
        if steady_start > 130:
            assert not copied
        for tensor in _decode_derived_outputs(metadata).values():
            tensor.fill_(-997)
        graph.replay()
        for name in expected:
            torch.testing.assert_close(observed[name], expected[name], atol=0, rtol=0)
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in _decode_derived_outputs(metadata).items()
        }

    # Odd/even compression endpoints, page crossing, reorder, recycled IDs,
    # newly admitted padding rows and later replay retain captured addresses.
    for step, start in enumerate((127, 128, 129, 255, 256, 0, 128)):
        if step == 4:
            request_id = requests[1].py_request_id
            epoch = manager.request_epoch(request_id)
            manager.free_resources(requests[1])
            replacement = LlmRequest(
                request_id=request_id,
                max_new_tokens=256,
                input_tokens=list(range(257)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            requests[1] = replacement
            assert manager.prepare_context(replacement)
            assert manager.resize_context(replacement, replacement.context_chunk_size)
            assert manager.request_epoch(request_id) != epoch
            manager._stream.synchronize()
        elif step == 5:
            manager.free_resources(requests[1])
            dummy = manager.add_dummy_requests(
                [CUDA_GRAPH_DUMMY_REQUEST_ID], token_nums=[1], is_gen=True
            )
            assert dummy is not None
            requests[1] = dummy[0]
            requests[1].is_cuda_graph_dummy = True
            manager._stream.synchronize()
        selected = (requests if step % 2 == 0 else requests[::-1])[:batch_size]
        starts = [
            0 if getattr(request, "is_cuda_graph_dummy", False) else max(0, start - row)
            for row, request in enumerate(selected)
        ]
        _prepare_decode_metadata(reference, selected, starts)
        expected = {
            name: tensor.clone() for name, tensor in _decode_derived_outputs(reference).items()
        }
        _prepare_decode_metadata(metadata, selected, starts, deferred=True)
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in _decode_derived_outputs(metadata).items()
        }
        for tensor in _decode_derived_outputs(metadata).values():
            tensor.fill_(-997)
        graph.replay()
        for name in expected:
            torch.testing.assert_close(observed[name], expected[name], atol=0, rtol=0)
        _assert_swa_cpu_reference(
            metadata, manager, selected, starts, [1] * batch_size, [0] * batch_size
        )

    # Graph metadata used by a direct caller must still be fully initialized.
    _prepare_decode_metadata(metadata, selected, starts)
    assert not metadata._csa2_deferred_decode_outputs
    for name, tensor in _decode_derived_outputs(metadata).items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "fallback,manager_requests",
    [
        ("eager", {}),
        ("context", {}),
        ("source", {}),
        ("remote_tail", {}),
        ("speculative", {"speculative": True}),
    ],
    indirect=["manager_requests"],
)
@torch.inference_mode()
def test_decode_graph_deferral_keeps_unsupported_prepare_paths_complete(manager_requests, fallback):
    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    metadata.is_cuda_graph = fallback != "eager"
    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 2 if fallback == "context" else 0
    metadata.prompt_lens = [257, 257]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[127, 128])
    if fallback == "source":
        metadata.set_source_batch([1, 1], [127, 128])
    if fallback == "remote_tail":
        metadata.csa2_remote_tail_mode = "decode"
    if fallback == "speculative":
        metadata.is_spec_decoding_enabled = True
    with metadata.defer_cuda_graph_decode_outputs():
        metadata.prepare()
    assert not metadata._csa2_deferred_decode_outputs
    assert metadata.csa2_positions.tolist() == [127, 128]
    _assert_swa_cpu_reference(metadata, manager, requests, [127, 128], [1, 1], [0, 0])


@pytest.mark.cpu_only
def test_decode_graph_prepare_scope_restores_proof_after_exception():
    metadata = object.__new__(CSA2TrtllmMetadata)
    with pytest.raises(RuntimeError, match="prepare failed"):
        with metadata.defer_cuda_graph_decode_outputs():
            assert metadata._csa2_defer_decode_outputs
            raise RuntimeError("prepare failed")
    assert not metadata._csa2_defer_decode_outputs

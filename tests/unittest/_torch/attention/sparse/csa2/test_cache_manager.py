# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 manager ownership, prefix reuse and scheduler lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    read_index_rows,
    write_packed_index_rows,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("disk_bytes", [0, 1 << 30])
@pytest.mark.parametrize("custom_codec", [False, True])
def test_encoder_replay_passes_cold_storage_to_base(
    monkeypatch, tmp_path, disk_bytes, custom_codec
):
    """CSA2 must leave cold-tier compatibility checks to the storage backend."""
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    config = KvCacheConfig(
        enable_block_reuse=True, disk_cache_size=disk_bytes, disk_cache_path=str(tmp_path)
    )
    provider = object() if custom_codec else None

    class BackendReached(Exception):
        pass

    def initialize_backend(self, received_config, *args, **kwargs):
        assert received_config is config
        assert kwargs["cold_page_codec_provider"] is provider
        assert self._encoder_reuse_policy.name == "OPTIONAL"
        raise BackendReached

    monkeypatch.setattr(CSA2CacheManager.__bases__[0], "__init__", initialize_backend)
    with pytest.raises(BackendReached):
        CSA2CacheManager(
            config,
            CacheType.SELFKONLY,
            num_layers=3,
            tokens_per_block=128,
            mapping=Mapping(),
            layout=CSA2Layout((0, 1, 1), (1,), (1,)),
            cold_page_codec_provider=provider,
        )


def test_disabled_layer_mask_rejected_before_allocation():
    with pytest.raises(ValueError, match="disabled layers in layer_mask"):
        CSA2CacheManager(
            KvCacheConfig(dtype="fp8"),
            CacheType.SELFKONLY,
            num_layers=2,
            tokens_per_block=128,
            mapping=Mapping(),
            layout=CSA2Layout((0, 0), (), ()),
            layer_mask=[True, False],
        )


@pytest.fixture
def manager(request, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CSA2 runtime cache storage requires CUDA")
    guard_page = getattr(request, "param", None)
    if guard_page is not None:
        monkeypatch.setenv("TRTLLM_KV_GUARD_PAGE", guard_page)
    layout = CSA2Layout((0, 2, 2, 1, 1), (1, 3), (1, 3))
    result = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20,
            host_cache_size=0,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            enable_swa_scratch_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=5,
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=3,
        max_num_tokens=1024,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=layout,
    )
    yield result
    for request_id in list(result.kv_cache_map):
        result.free_resources(SimpleNamespace(py_request_id=request_id))
    result.shutdown()


def allocate(manager, request_id, tokens, capacity):
    cache = manager._create_kv_cache(request_id, None, tokens)
    assert cache is not None
    assert manager._resume_and_restore(request_id, cache)
    assert cache.resize(capacity)
    return cache


@pytest.mark.parametrize("manager", ["", "nan"], indirect=True, ids=["unguarded", "guarded"])
def test_warmup_cache_cleanup_preserves_swa_guards(manager) -> None:
    allocate(manager, 90, [], 129)
    buffers = {
        (layer, role): manager.get_buffers(layer, role)
        for layer, role in manager._physical_roles.values()
    }
    assert {role for _, role in buffers} == set(CSA2CacheRole)
    assert any(
        manager._layer_roles[layer, CSA2CacheRole.SWA] != layer for layer in manager.pp_layers
    )
    if manager._guard_page_value is not None:
        assert set(manager._guard_page_by_layer) == set(manager.pp_layers)
    for buffer in buffers.values():
        buffer.fill_(7)
    for layer, page in manager._guard_page_by_layer.items():
        buffers[layer, CSA2CacheRole.SWA][page].fill_(0x7F)
    assert not manager.check_invalid_values_in_kv_cache()

    floating_cells = []
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        page = manager.get_cache_indices(90, 1, role)[0]
        cell = buffers[1, role][page, 0, 0]
        floating_cells.append(cell)
        for value in (float("nan"), float("inf")):
            cell.fill_(value)
            assert manager.check_invalid_values_in_kv_cache()
            assert not torch.isfinite(cell)
            cell.fill_(7)
    for buffer in buffers.values():
        assert torch.all(buffer != 0)
    floating_cells[0].fill_(float("nan"))
    floating_cells[1].fill_(float("inf"))
    assert manager.check_invalid_values_in_kv_cache(fill_with_zero=True)

    for (_, role), buffer in buffers.items():
        expected = torch.zeros_like(buffer)
        if role == CSA2CacheRole.SWA:
            for layer, page in manager._guard_page_by_layer.items():
                if buffers[layer, role].data_ptr() == buffer.data_ptr():
                    expected[page].fill_(0x7F)
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
    assert not manager.check_invalid_values_in_kv_cache()


def test_owner_views_and_lifecycle(manager):
    cache = allocate(manager, 10, [], 513)
    assert (
        len({manager.get_cache_indices(10, layer, CSA2CacheRole.SWA)[0] for layer in range(5)}) == 5
    )
    for owner in (1, 3):
        main, index = manager.get_main_buffer(owner), manager.get_index_pages(owner)
        assert main.stride() == (288, 1)
        # Native page-footer index pages, read in place by the paged kernels.
        assert index.shape[1:] == (64, 1, 68) and index.is_contiguous()
        assert index.data_ptr() != main.data_ptr()
        assert manager.get_cache_indices(
            10, owner, CSA2CacheRole.GLOBAL
        ) == manager.get_cache_indices(10, owner + 1, CSA2CacheRole.GLOBAL)
        # Main and index buffers share one page group and therefore one page table.
        assert manager.get_cache_indices(
            10, owner, CSA2CacheRole.INDEX
        ) == manager.get_cache_indices(10, owner, CSA2CacheRole.GLOBAL)
        page = manager.get_cache_indices(10, owner, CSA2CacheRole.GLOBAL)[0]
        row = page * (128 // manager.layout.compress_ratios[owner])
        main[row].fill_(37)
        slot = torch.tensor([row], device=main.device)
        write_packed_index_rows(
            index, slot, torch.full((1, 68), 91, dtype=torch.uint8, device=main.device)
        )
        torch.testing.assert_close(main[row], torch.full_like(main[row], 37))
        torch.testing.assert_close(
            read_index_rows(manager.get_index_pages(owner), slot)[0],
            torch.full((68,), 91, dtype=torch.uint8, device=main.device),
        )
    sizes, windows = manager._get_runtime_cache_size_layer_components()
    assert sum(size for size, window in zip(sizes, windows) if window is None) == 534
    assert sizes.count(4096) == 1
    quota = manager._get_quota_from_max_tokens(512)
    assert quota > 512 * 534
    assert manager._get_max_tokens_from_quota(quota) == pytest.approx(512)
    # Long prefill resolves scratch pages, not just the final sliding ring.
    for role in (CSA2CacheRole.SWA, CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        pages = manager.get_cache_indices(10, 1, role)
        assert len(pages) >= 5
        assert len(set(pages[:5])) == 5
    assert cache.resize(514, history_length=513)
    manager.free_resources(SimpleNamespace(py_request_id=10))
    assert 10 not in manager.kv_cache_map
    allocate(manager, 11, [], 129)
    assert manager.get_cache_indices(11, 1, CSA2CacheRole.GLOBAL)


def test_prefix_reuse_preserves_combined_records(manager):
    tokens = list(range(512))
    first = allocate(manager, 20, [], len(tokens))
    for model_layer, role in manager._physical_roles.values():
        pages = manager.get_cache_indices(20, model_layer, role)
        pool = manager.get_buffers(model_layer, role)
        for page in pages:
            if page >= 0:
                pool[page].fill_(7)
    first.commit(tokens)
    torch.cuda.synchronize()
    manager.free_resources(SimpleNamespace(py_request_id=20))
    second = allocate(manager, 21, tokens + [7000], 513)
    assert second.num_committed_tokens > 0
    page = manager.get_cache_indices(21, 1, CSA2CacheRole.GLOBAL)[0]
    packed = manager.get_buffers(1, CSA2CacheRole.GLOBAL)[page]
    torch.testing.assert_close(packed, torch.full_like(packed, 7))
    # Fork an overlapping prefix and allocate independent writable suffixes.
    third = allocate(manager, 22, tokens + [7001], 513)
    assert third.num_committed_tokens > 0
    pages2 = manager.get_cache_indices(21, 1, CSA2CacheRole.GLOBAL)
    pages3 = manager.get_cache_indices(22, 1, CSA2CacheRole.GLOBAL)
    assert pages2[4] != pages3[4]
    pool = manager.get_buffers(1, CSA2CacheRole.GLOBAL)
    pool[pages2[4]].fill_(11)
    pool[pages3[4]].fill_(19)
    torch.testing.assert_close(pool[pages2[4]], torch.full_like(pool[pages2[4]], 11))
    torch.testing.assert_close(pool[pages3[4]], torch.full_like(pool[pages3[4]], 19))


def test_partial_page_copy_on_write_and_state(manager):
    tokens = list(range(129))
    source = allocate(manager, 30, [], 129)
    for model_layer, role in manager._physical_roles.values():
        pool = manager.get_buffers(model_layer, role)
        for page in manager.get_cache_indices(30, model_layer, role):
            if page >= 0:
                pool[page].fill_(23)
    source.commit(tokens)
    torch.cuda.synchronize()
    manager.free_resources(SimpleNamespace(py_request_id=30))
    left = allocate(manager, 31, tokens + [7000], 130)
    right = allocate(manager, 32, tokens + [7001], 130)
    assert left.num_committed_tokens == right.num_committed_tokens
    assert left.num_committed_tokens >= 128
    # The writable trailing page is private even when its prefix was reused.
    for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        pool = manager.get_buffers(1, role)
        left_pages = manager.get_cache_indices(31, 1, role)
        right_pages = manager.get_cache_indices(32, 1, role)
        assert left_pages[1] != right_pages[1]
        pool[left_pages[1]].fill_(31)
        pool[right_pages[1]].fill_(47)
        torch.testing.assert_close(pool[left_pages[1]], torch.full_like(pool[left_pages[1]], 31))
    # Suspend/resume reconnects page-index buffers before the next forward.
    left.suspend()
    assert manager._resume_and_restore(31, left)
    assert manager.get_cache_indices(31, 1, CSA2CacheRole.GLOBAL)[1] >= 0


@pytest.mark.parametrize("enable_scratch", [False, True])
def test_cache_manager_preserves_disabled_or_enabled_scratch_config(enable_scratch):
    if not torch.cuda.is_available():
        pytest.skip("CSA2 runtime cache storage requires CUDA")
    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=enable_scratch,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=2,
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=256,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=CSA2Layout((0, 2), (1,), (1,)),
    )
    try:
        # Exercise the real runtime config conversion and allocator, including
        # the native optional-property setter used by full model construction.
        assert manager.impl.enable_swa_scratch_reuse == enable_scratch
        cache = allocate(manager, 91, [], 129)
        assert cache.enable_swa_scratch_reuse == enable_scratch
        for role in (CSA2CacheRole.SWA, CSA2CacheRole.GLOBAL, CSA2CacheRole.COMPRESSOR_KV):
            assert manager.get_cache_indices(91, 1, role)
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.parametrize("fallback", ["scratch", "attention_dp", "disaggregated", "late_engram"])
@pytest.mark.parametrize("partial_reuse", [False, True])
def test_decoder_replay_selects_cache_lifecycles(monkeypatch, fallback, partial_reuse):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            enable_block_reuse=True,
            enable_partial_reuse=partial_reuse,
            enable_swa_scratch_reuse=fallback == "scratch",
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=2,
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=256,
        mapping=Mapping(enable_attention_dp=fallback == "attention_dp"),
        is_disagg=fallback == "disaggregated",
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=CSA2Layout((0, 1), (1,), (1,)),
        pretrained_config=SimpleNamespace(
            engram_layer_ids=[1] if fallback == "late_engram" else []
        ),
    )
    try:
        assert manager.enable_partial_reuse == partial_reuse
        assert manager.impl.init_config.enable_partial_reuse == partial_reuse
        replay = fallback in ("attention_dp", "disaggregated")
        assert manager.has_private_swa_suffix(1) == replay
        assert not manager.has_private_swa_suffix(0)
        assert not manager.has_private_swa_suffix(2)
        if replay:
            assert manager._decoder_replay_window == manager.layout.window_size
            for layer in manager.impl.init_config.layers:
                decoder = layer.layer_id == manager._layer_roles[1, CSA2CacheRole.SWA]
                assert layer.reuse_policy == (
                    AttentionReusePolicy.PRIVATE if decoder else AttentionReusePolicy.REQUIRED
                )
        else:
            assert manager._decoder_replay_window == 0
            assert all(
                layer.reuse_policy == AttentionReusePolicy.REQUIRED
                for layer in manager.impl.init_config.layers
            )
    finally:
        manager.shutdown()


def _request(request_id, tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=16,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _metadata(manager, request, metadata=None):
    metadata = metadata or CSA2TrtllmMetadata(
        max_num_requests=1, max_num_tokens=512, kv_cache_manager=manager
    )
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.seq_lens = torch.tensor([request.context_chunk_size], dtype=torch.int32)
    metadata.prompt_lens = [request.context_chunk_size]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[request.context_current_position]
    )
    metadata.prepare()
    return metadata


@pytest.mark.cpu_only
def test_swa_publication_is_scoped_transaction_with_exact_request_and_destination():
    manager = CSA2CacheManager.__new__(CSA2CacheManager)
    manager._swa_publication_token = manager._swa_publication = None
    manager.pp_layers = [0, 1]
    manager.max_blocks_per_seq = 4
    state = {"fail": False, "offset": 0}

    def pages(request, layer, role):
        assert role == CSA2CacheRole.SWA
        if state["fail"] and layer == 1:
            raise ValueError("conversion failed")
        return [request + 10 * layer + state["offset"], -1, 19]

    def compute(request_ids, num_contexts):
        # Stands in for the device conversion: one SWA table per local layer.
        manager._batch_page_tables = torch.tensor(
            [
                [(pages(r, layer, CSA2CacheRole.SWA) + [-1])[:4] for r in request_ids]
                for layer in (0, 1)
            ],
            dtype=torch.int32,
        ).reshape(2, len(request_ids), 4)

    manager.compute_batch_page_tables = compute
    destination = torch.full((2, 4, 2, 4), 99, dtype=torch.int32)
    ids = [3, 3, 1]  # Repeated padding IDs retain packed row order.
    with manager._swa_publication_scope() as token:
        manager.copy_batch_block_offsets(destination, ids, 1, 2, 3)
        device = manager._get_swa_publication(token, destination, ids, 2, 3)
        torch.testing.assert_close(manager._batch_page_tables[:, :, :3], device, atol=0, rtol=0)
        assert device.stride(-2) == 8 and torch.all(destination[:, 3] == -1)
        assert manager._get_swa_publication(token, destination, ids, 2, 5) is None
        assert manager._get_swa_publication(token, destination, [1, 3, 3], 2, 3) is None
        assert manager._get_swa_publication(token, destination, ids, 1, 3) is None
        assert manager._get_swa_publication(token, destination.clone(), ids, 2, 3) is None
        state["fail"] = True
        with pytest.raises(ValueError, match="conversion failed"):
            manager.copy_batch_block_offsets(destination, ids, 1, 2, 3)
        assert manager._get_swa_publication(token, destination, ids, 2, 3) is None
    assert manager._swa_publication_token is manager._swa_publication is None
    state.update(fail=False, offset=100)
    with manager._swa_publication_scope() as fresh:
        manager.copy_batch_block_offsets(destination, ids, 1, 2, 3)
        assert manager._get_swa_publication(token, destination, ids, 2, 3) is None
        assert manager._get_swa_publication(fresh, destination, ids, 2, 3)[0, 0, 0] == 103
        destination.set_(torch.empty_like(destination))
        assert manager._get_swa_publication(fresh, destination, ids, 2, 3) is None
    with pytest.raises(RuntimeError, match="later preparation failed"):
        with manager._swa_publication_scope():
            manager.copy_batch_block_offsets(destination, ids, 1, 2, 3)
            raise RuntimeError("later preparation failed")
    assert manager._swa_publication_token is manager._swa_publication is None


# Embedded DSpark linear verification replayed with the executor protocol.
DRAFT_LEN = 5


def _spec_manager(block):
    return CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=512 << 20, host_cache_size=0, enable_block_reuse=False, dtype="fp8"
        ),
        CacheType.SELFKONLY,
        num_layers=4,
        tokens_per_block=block,
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=4096,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=128, index_topk=4),
        spec_config=DSparkDecodingConfig(max_draft_len=DRAFT_LEN),
    )


def _spec_metadata(manager):
    return CSA2TrtllmMetadata(
        max_num_requests=manager.max_batch_size,
        max_num_tokens=manager.max_num_tokens,
        kv_cache_manager=manager,
    )


def _spec_prepare(manager, requests, cached):
    metadata = _spec_metadata(manager)
    metadata.request_ids = [r.py_request_id for r in requests]
    metadata.num_contexts = 0
    metadata.seq_lens = torch.full((len(requests),), DRAFT_LEN + 1, dtype=torch.int32)
    metadata.prompt_lens = [r.py_prompt_len for r in requests]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached)
    metadata.prepare()
    return metadata


def _spec_finish_context(manager, request):
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.prompt_len)
    _metadata(manager, request, _spec_metadata(manager))
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    # Under overlap, the context step's sampled token is not appended yet.
    request.py_draft_tokens = [0] * DRAFT_LEN


def _spec_accept(manager, requests, accepted):
    batch = ScheduledRequests()
    batch.generation_requests = list(requests)
    for request, count in zip(requests, accepted):
        request.py_num_accepted_draft_tokens = count
        request.py_rewind_len = DRAFT_LEN - count
        for _ in range(count + 1):
            request.add_new_token(1, 0)
    manager.update_resources(batch)


def _spec_cleanup(manager):
    for request_id in list(manager.kv_cache_map):
        manager.free_resources(SimpleNamespace(py_request_id=request_id))
    manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block", [128, 256])
def test_first_verification_step_tolerates_the_overlap_tail(block):
    """Reserved positions beyond the allocated capacity stay unmapped, not fatal."""
    manager = _spec_manager(block)
    try:
        # prompt + 2 * DRAFT_LEN lands exactly on a page boundary for every request.
        prompts = [block * m - 2 * DRAFT_LEN for m in (1, 2, 3)]
        requests = [_request(i, list(range(p))) for i, p in enumerate(prompts)]
        for request in requests:
            _spec_finish_context(manager, request)
            assert manager.try_allocate_generation(request)
        cached = [r.prompt_len - 1 + DRAFT_LEN + 1 for r in requests]
        metadata = _spec_prepare(manager, requests, cached)
        owner = manager.layout.kv_source_layer_ids[-1]
        assert manager.layout.compress_ratios[owner] == 1
        metadata._ensure_swa_slots()
        for row, request in enumerate(requests):
            cache = manager.kv_cache_map[request.py_request_id]
            assert cache.capacity == request.prompt_len + 2 * DRAFT_LEN
            tail = metadata.csa2_request_query_ranges[row][1] - 1
            for layer in range(len(manager.layout.compress_ratios)):
                writes = metadata.csa2_swa_write_slots[layer]
                assert int(writes[tail]) == -1
                assert all(int(v) >= 0 for v in writes[tail - DRAFT_LEN : tail])
            assert int(metadata.csa2_main_write_slots[owner][tail]) == -1
    finally:
        _spec_cleanup(manager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("overlap", [False, True])
def test_linear_verification_steps_stay_resolvable(overlap):
    """Repeated verification steps keep resolving as pages fill and rewind."""
    manager = _spec_manager(128)
    try:
        requests = [_request(i, list(range(p))) for i, p in enumerate((118, 127, 128, 255, 640))]
        for request in requests:
            _spec_finish_context(manager, request)
        pending = None
        for step in range(60):
            for request in requests:
                assert manager.try_allocate_generation(request)
            cached = [
                r.max_beam_num_tokens - 1 + (DRAFT_LEN + 1 if overlap else 0) for r in requests
            ]
            _spec_prepare(manager, requests, cached)
            for request, start in zip(requests, cached):
                assert manager.kv_cache_map[request.py_request_id].capacity >= start + 1
            if overlap and pending is not None:
                # Iteration N-1's acceptance lands after iteration N was prepared.
                _spec_accept(manager, requests, pending)
            accepted = [(step + row) % (DRAFT_LEN + 1) for row in range(len(requests))]
            if overlap:
                pending = accepted
            else:
                _spec_accept(manager, requests, accepted)
    finally:
        _spec_cleanup(manager)

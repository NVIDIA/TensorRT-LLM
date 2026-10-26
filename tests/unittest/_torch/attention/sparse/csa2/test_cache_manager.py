# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 manager ownership, prefix reuse and scheduler lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import cache_manager as cache_module
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
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)


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


@skip_pre_blackwell
@pytest.mark.parametrize("manager", ["", "nan"], indirect=True, ids=["unguarded", "guarded"])
def test_warmup_cache_cleanup_preserves_swa_guards(manager) -> None:
    allocate(manager, 90, [], 129)
    buffers = {
        (layer, role): manager.get_buffers(layer, role)
        for layer, _, role in manager._physical_roles.values()
    }
    assert {role for _, role in buffers} == set(CSA2CacheRole)
    assert any(
        manager._layer_roles[layer, 1, CSA2CacheRole.SWA] != layer for layer in manager.pp_layers
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


@skip_pre_blackwell
def test_owner_views_and_lifecycle(manager):
    cache = allocate(manager, 10, [], 513)
    assert (
        len({manager.get_cache_indices(10, layer, CSA2CacheRole.SWA)[0] for layer in range(5)}) == 5
    )
    for owner in (1, 3):
        main, index = manager.get_main_buffer(owner), manager.get_index_pages(owner)
        ratio = manager.layout.compress_ratios[owner]
        assert manager.get_main_buffer(owner + 1) is main
        assert manager.get_index_pages(owner + 1) is index
        assert manager._global_buffers[owner, ratio, CSA2CacheRole.GLOBAL] is main
        assert manager._global_buffers[owner, ratio, CSA2CacheRole.INDEX] is index
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


@skip_pre_blackwell
def test_prefix_reuse_preserves_combined_records(manager):
    tokens = list(range(512))
    first = allocate(manager, 20, [], len(tokens))
    for model_layer, _, role in manager._physical_roles.values():
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


@skip_pre_blackwell
def test_partial_page_copy_on_write_and_state(manager):
    tokens = list(range(129))
    source = allocate(manager, 30, [], 129)
    for model_layer, _, role in manager._physical_roles.values():
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


@skip_pre_blackwell
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


@skip_pre_blackwell
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
        replay = fallback in ("scratch", "attention_dp", "disaggregated")
        assert manager.has_private_swa_suffix(1) == replay
        assert not manager.has_private_swa_suffix(0)
        assert not manager.has_private_swa_suffix(2)
        if replay:
            assert manager._decoder_replay_window == 140
            for layer in manager.impl.init_config.layers:
                decoder = layer.layer_id == manager._layer_roles[1, 1, CSA2CacheRole.SWA]
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

    def compute(request_ids, num_contexts, swa_out=None):
        # Stands in for the device conversion: one SWA table per local layer.
        manager._batch_page_tables = torch.tensor(
            [
                [(pages(r, layer, CSA2CacheRole.SWA) + [-1])[:4] for r in request_ids]
                for layer in (0, 1)
            ],
            dtype=torch.int32,
        ).reshape(2, len(request_ids), 4)
        if swa_out is not None:
            swa_out.copy_(manager._batch_page_tables[: swa_out.shape[0], :, : swa_out.shape[2]])

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


@skip_pre_blackwell
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


@skip_pre_blackwell
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


def _context_only_layout() -> CSA2Layout:
    return CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer_id=20,
    )


def _declarative_manager(limit: int | None) -> CSA2CacheManager:
    manager = object.__new__(CSA2CacheManager)
    manager.context_swa_layer_limit = limit
    manager.layout = _context_only_layout()
    manager.pp_layers = list(range(40))
    manager.tokens_per_block = 128
    manager.max_draft_len = manager.reuse_match_backoff = 0
    manager._decoder_replay_window = 0
    manager._encoder_replay_enabled = False
    manager._build_cache_config(SimpleNamespace(swa_scratch_reuse=None))
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [0, -1, 40, 41, 20.5, True, "20"])
def test_invalid_context_swa_limit_fails_before_allocation(limit) -> None:
    with pytest.raises(ValueError, match="context SWA layer limit"):
        CSA2CacheManager(
            KvCacheConfig(dtype="fp8"),
            CacheType.SELFKONLY,
            num_layers=40,
            tokens_per_block=128,
            mapping=Mapping(),
            layout=_context_only_layout(),
            context_swa_layer_limit=limit,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [None, 20])
def test_prune_only_physical_swa_groups(limit: int | None) -> None:
    manager = _declarative_manager(limit)
    built = manager._build_cache_config(SimpleNamespace(swa_scratch_reuse=None))
    retained = 40 if limit is None else limit
    assert manager.pp_layers == list(range(40))
    assert len(built.layers) == retained + 7
    identities = set(manager._physical_roles.values())
    assert {layer for layer, _, role in identities if role == CSA2CacheRole.SWA} == set(
        range(retained)
    )
    assert {ratio for _, ratio, role in identities if role == CSA2CacheRole.SWA} == {1}
    for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
        assert {(layer, ratio) for layer, ratio, item in identities if item == role} == {
            (2, 2),
            (8, 2),
            (14, 2),
            (20, 1),
        }
        assert manager._layer_roles[3, 2, role] == manager._layer_roles[2, 2, role]
        assert manager._layer_roles[39, 1, role] == manager._layer_roles[20, 1, role]
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        assert {(layer, ratio) for layer, ratio, item in identities if item == role} == {
            (2, 2),
            (8, 2),
            (14, 2),
        }
    sizes, windows = manager._get_runtime_cache_size_layer_components()
    assert sizes.count(528) == retained
    assert sum(size for size, window in zip(sizes, windows) if window is None) == 890
    assert sizes.count(4096) == 3
    assert (
        sum(buffer.size for layer in built.layers for buffer in layer.buffers) == sum(sizes) * 128
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("remote_tail", [False, True])
def test_static_sizing_removes_only_swa_reservation(
    tokens_per_block: int, remote_tail: bool
) -> None:
    layout = _context_only_layout()
    text_config = SimpleNamespace(
        compress_ratios=layout.compress_ratios,
        kv_source_layer_ids=layout.kv_source_layer_ids,
        index_source_layer_ids=layout.index_source_layer_ids,
        candidate_source_layer_id=layout.candidate_source_layer_id,
        candidate_topk_blocks=layout.candidate_topk_blocks,
        candidate_block_size=layout.candidate_block_size,
        index_topk=layout.index_topk,
        sliding_window=layout.window_size,
    )
    model = SimpleNamespace(pretrained_config=text_config, get_num_attention_layers=lambda: 40)
    ordinary = CSA2CacheManager.get_cache_size_per_token(
        model, Mapping(), tokens_per_block=tokens_per_block, max_batch_size=32
    )
    model.extra_attrs = {"bounded_replay_on_generation": remote_tail}
    full = CSA2CacheManager.get_cache_size_per_token(
        model, Mapping(), tokens_per_block=tokens_per_block, max_batch_size=32
    )
    # Remote-tail retention adds a 128-row lookback to the 141-row replay window.
    extra_pages = int(remote_tail)
    assert full[0] == ordinary[0]
    assert full[1] - ordinary[1] == 20 * extra_pages * tokens_per_block * 528 * 32
    model.extra_attrs["csa2_context_swa_layer_limit"] = 20
    pruned = CSA2CacheManager.get_cache_size_per_token(
        model, Mapping(), tokens_per_block=tokens_per_block, max_batch_size=32
    )
    # The GLOBAL token slope stays fixed. Replay pages include headroom for
    # the worst page alignment, including three pages for 128-token blocks.
    assert full[0] == pruned[0] == 890
    decoder_pages = 3 if tokens_per_block == 128 else 2
    assert full[1] - pruned[1] == 20 * decoder_pages * tokens_per_block * 528 * 32


@pytest.mark.cpu_only
@pytest.mark.parametrize(("quota_seq_len", "expected"), [(None, 190771), (140000, 140000)])
def test_typical_seq_len_fits_a_full_batch(quota_seq_len, expected) -> None:
    manager = _declarative_manager(None)
    manager.max_seq_len, manager.max_batch_size = 190771, 1024
    manager.enable_swa_scratch_reuse = False
    manager._generation_kv_capacity_headroom = 0
    sizes, windows, keys = manager._runtime_cache_size_layer_components()
    _, per_token, per_request = cache_module._estimate_cache_size_components(
        sizes, windows, 128, scratch=False, generation_capacity_headroom=0, scratch_group_keys=keys
    )
    # A quota holding exactly 1024 requests of quota_seq_len tokens.
    quota = 0 if quota_seq_len is None else 1024 * (per_request + per_token * quota_seq_len)
    config = KvCacheConfig(max_gpu_total_bytes=quota, max_util_for_resume=1.0)
    assert manager._get_typical_seq_len(config) == expected


@pytest.mark.cpu_only
def test_logical_page_tables_and_null_pointers(monkeypatch) -> None:
    manager = _declarative_manager(20)
    manager.num_local_layers = 40
    manager.num_pools = 3
    manager.max_batch_size = 2
    manager.max_beam_width = 1
    manager.max_blocks_per_seq = 8
    manager.impl = SimpleNamespace(get_mem_pool_base_address=lambda *args: 4096 + int(args[0]))
    monkeypatch.setattr(cache_module, "prefer_pinned", lambda: False)
    # The device page-table plan needs real converters; pointers do not.
    monkeypatch.setattr(manager, "_init_batch_page_plan", lambda: None)
    manager._prepare_page_table_tensor(2)
    assert manager.kv_cache_pool_pointers.shape == (40, 2)
    assert torch.all(manager.kv_cache_pool_pointers[:20, 0] != 0)
    assert torch.count_nonzero(manager.kv_cache_pool_pointers[20:]) == 0
    assert manager.kv_cache_pool_mapping[:, 0].tolist() == list(range(40))
    with pytest.raises(ValueError, match="has no SWA cache"):
        manager.get_buffers(20)
    with pytest.raises(ValueError, match="has no SWA cache"):
        manager.get_cache_indices(1, 39, CSA2CacheRole.SWA)

    def compute(request_ids, num_contexts, swa_out):
        # Stands in for the device conversion: SWA tables of the retained layers.
        tables = [
            [[r + layer, 88] + [BAD_PAGE_INDEX] * 6 for r in request_ids] for layer in range(20)
        ]
        swa_out.copy_(torch.tensor(tables, dtype=torch.int32))

    manager.compute_batch_page_tables = compute
    destination = torch.full((40, 2, 2, 8), 99, dtype=torch.int32)
    manager._swa_publication_token = manager._swa_publication = None
    manager.copy_batch_block_offsets(destination, [1], 1, 1, 1)
    assert destination[:20, 0, 0, 0].tolist() == list(range(1, 21))
    assert torch.all(destination[20:] == BAD_PAGE_INDEX)


@pytest.mark.cpu_only
def test_fresh_fill_uses_physical_roles_and_preserves_relocated_committed_pages(
    monkeypatch,
) -> None:
    manager = _declarative_manager(20)
    manager._fresh_page_fill = 0
    manager._fresh_fill_announced = True
    manager._fresh_pages_filled = {}
    manager.kv_cache_map = {1: SimpleNamespace(num_committed_tokens=1)}
    # Model layer 20's GLOBAL role remains, despite no SWA at that layer.
    buffers = {
        identity: torch.full((16, 1, 4), 7, dtype=torch.uint8)
        for identity in manager._physical_roles.values()
    }
    page_tables = {identity: [1, 2, 3, 4, BAD_PAGE_INDEX, BAD_PAGE_INDEX] for identity in buffers}
    manager.get_buffers = lambda layer, role: buffers[manager._cache_role_key(layer, role)]
    manager.get_cache_indices = lambda request, layer, role: page_tables[
        manager._cache_role_key(layer, role)
    ]
    manager._get_page_index_converter = lambda *args: SimpleNamespace(expansion=2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    manager._fill_fresh_kv_pages(1)
    for buffer in buffers.values():
        assert torch.all(buffer[[1, 2]] == 7)  # ceil(committed / 128) * expansion
        assert torch.all(buffer[[3, 4]] == 0)
        buffer.fill_(9)
    for identity in page_tables:
        page_tables[identity] = [1, 2, 8, 9, 10, 11]
    manager._fill_fresh_kv_pages(1)
    for buffer in buffers.values():
        assert torch.all(buffer[[1, 2, 8, 9]] == 9)  # protected or relocated
        assert torch.all(buffer[[10, 11]] == 0)  # newly materialized ordinals


@skip_pre_blackwell
@pytest.mark.parametrize("guard", [False, True])
def test_real_context_allocator_omits_decoder_swa(monkeypatch, guard: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CSA2 runtime cache storage requires CUDA")
    monkeypatch.setenv("TRTLLM_KV_GUARD_PAGE", "nan" if guard else "")
    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", "zero")
    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20, host_cache_size=0, enable_block_reuse=False, dtype="fp8"
        ),
        CacheType.SELFKONLY,
        num_layers=40,
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=2,
        max_num_tokens=1024,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=_context_only_layout(),
        context_swa_layer_limit=20,
    )
    try:
        assert manager.num_local_layers == 40
        assert len(manager.impl.init_config.layers) == 27
        assert len(list(manager.impl.all_buffer_ids)) == 34
        assert torch.all(manager.kv_cache_pool_pointers[20:] == 0)
        assert set(manager._guard_page_by_layer) == (set(range(20)) if guard else set())
        assert manager.get_num_free_blocks() > 0
        assert manager.blocks_in_primary_pool > 0
        quota = manager._get_quota_from_max_tokens(1024)
        assert manager._get_max_tokens_from_quota(quota) == pytest.approx(1024)
        cache = manager._create_kv_cache(90, None, [])
        assert cache is not None and manager._resume_and_restore(90, cache)
        assert cache.resize(513)
        manager._fill_fresh_kv_pages(90)
        for owner in (2, 8, 14, 20):
            assert manager.get_cache_indices(90, owner, CSA2CacheRole.GLOBAL)
            assert manager.get_buffers(owner, CSA2CacheRole.GLOBAL).numel() > 0
        for layer in range(20, 40):
            with pytest.raises(ValueError, match="has no SWA cache"):
                manager.get_buffers(layer)
        destination = torch.empty(
            (40, 2, 2, manager.max_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        manager.copy_batch_block_offsets(destination, [90], 1, 1, 1)
        assert torch.all(destination[20:] == BAD_PAGE_INDEX)
        assert sum(role == CSA2CacheRole.SWA for _, role in manager._batch_page_index) == 20
        manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
        assert not manager.check_invalid_values_in_kv_cache()
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize("encoder_replay", [False, True])
@pytest.mark.parametrize("block_reuse", [False, True])
def test_remote_tail_retains_encoder_handoff_lookback(encoder_replay, block_reuse) -> None:
    manager = _declarative_manager(20)
    manager._decoder_replay_window = 140
    manager._encoder_replay_enabled = encoder_replay
    manager.bounded_replay_on_generation = True
    manager.enable_block_reuse = block_reuse
    # Next-query retention at P must reach S-W, not just the last window.
    prompt_end = 1025
    handoff = prompt_end - 140
    assert prompt_end + 1 - manager._swa_retention(0) == handoff - 128
    _, windows = manager._get_runtime_cache_size_layer_components()
    assert windows[0] == manager._swa_retention(0)
    manager.context_swa_layer_limit = None
    assert manager._swa_retention(20) == 141
    sizes, windows = manager._get_runtime_cache_size_layer_components()
    swa_windows = [window for size, window in zip(sizes, windows) if size == 528]
    assert swa_windows == [269] * 20 + [manager._swa_retention(20)] * 20


@pytest.mark.cpu_only
@pytest.mark.parametrize("encoder_replay", [False, True])
@pytest.mark.parametrize("generation_init", [False, True])
@pytest.mark.parametrize("handoff,limit", [(0, None), (385, None), (385, 256)])
def test_remote_tail_reuse_cannot_pass_handoff(
    monkeypatch, encoder_replay, generation_init, handoff, limit
) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    manager = _declarative_manager(20)
    manager._decoder_replay_window = 128
    manager._encoder_replay_enabled = encoder_replay
    request = SimpleNamespace(
        py_csa2_remote_tail_start=handoff,
        py_csa2_remote_tail_mode="destination" if generation_init else "source",
        is_disagg_generation_init_state=generation_init,
        prompt_len=handoff + 128,
        py_return_context_logits=False,
        py_additional_outputs=None,
        py_multimodal_data=None,
    )
    monkeypatch.setattr(
        KVCacheManagerV2, "_context_reuse_tokens", lambda self, req, reuse_limit: reuse_limit
    )
    cap = handoff if generation_init else max(0, handoff - 1)
    assert manager._context_reuse_tokens(request, limit) == (
        cap if limit is None else min(cap, limit)
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("encoder_replay", [False, True])
def test_remote_tail_source_reuse_keeps_a_schedulable_final_token(monkeypatch, encoder_replay):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        KVCacheManagerV2,
        _settle_context_cursor,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig

    manager = _declarative_manager(20)
    manager._decoder_replay_window = 128
    manager._encoder_replay_enabled = encoder_replay
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[0] * 1244,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.py_csa2_remote_tail_mode = "source"
    request.py_csa2_remote_tail_start = 1116
    monkeypatch.setattr(
        KVCacheManagerV2, "_context_reuse_tokens", lambda self, req, reuse_limit: reuse_limit
    )

    reused = manager._context_reuse_tokens(request)
    _settle_context_cursor(request, reused, 128)

    assert reused == 1115
    assert request.context_remaining_length == request.context_chunk_size == 1
    assert request.is_last_context_chunk


@pytest.mark.cpu_only
def test_remote_tail_destination_does_not_recover_received_encoder_state(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    manager = _declarative_manager(20)
    manager._decoder_replay_window = 128
    manager._encoder_replay_enabled = True
    manager._encoder_optional_groups = (0,)
    request = SimpleNamespace(
        py_request_id=1,
        prompt_len=640,
        is_dummy=False,
        is_first_context_chunk=True,
        py_ced_replay=None,
        is_disagg_generation_init_state=False,
        py_csa2_remote_tail_mode="destination",
    )
    cache = SimpleNamespace(
        num_committed_tokens=512,
        is_active=True,
        reuse_status={0: SimpleNamespace(complete=False)},
    )
    monkeypatch.setattr(KVCacheManagerV2, "_resume_context_cache", lambda *args: True)
    assert manager._resume_context_cache(request, cache)
    assert request.py_ced_replay is None

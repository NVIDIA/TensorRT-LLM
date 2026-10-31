# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA decode-step preparation a model opts into (RuntimeHooks.msa).

Covers the short-sequence identity table the metadata stages, and the KV cache
manager's block-table and pool-view shortcuts. With the default policy every
path runs as before; the existing test_msa_backend.py tests cover that.
"""

from unittest.mock import create_autospec

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
    MiniMaxM3SparseMetadataParams,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
    _fill_unused_slots,
)
from tensorrt_llm._torch.pyexecutor import h2d_cache
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NUM_KV_HEADS = 2
PARAMS = MiniMaxM3SparseMetadataParams(
    global_num_q_heads=8, global_num_kv_heads=NUM_KV_HEADS, num_index_heads=4, topk=16
)


@pytest.fixture
def manager(monkeypatch) -> MiniMaxM3KVCacheManagerV2:
    """An M3 cache manager whose V2 base allocates nothing (as in test_msa_backend.py)."""

    def fake_base_init(self, *args, **kwargs) -> None:
        del args, kwargs
        self.is_disagg = False
        self.dtype = DataType.BF16
        self.layer_offsets = {3: 0}
        # _prepare_page_table_tensor's result for the two-pool host table the copy tests build.
        self.num_attention_op_pools = 2

    def no_index_k_buffer(self, layer_idx, **kwargs):
        del self, layer_idx, kwargs
        return None

    monkeypatch.setattr(KVCacheManagerV2, "__init__", fake_base_init)
    monkeypatch.setattr(KVCacheManagerV2, "get_index_k_buffer", no_index_k_buffer)
    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "_compute_num_total_slots", lambda self: 0)
    # The constructor resolves every layer's K/V view, which needs a real pool. Stub it there
    # only: the pool-view tests below call get_buffers against their own geometry.
    with monkeypatch.context() as construction:
        construction.setattr(MiniMaxM3KVCacheManagerV2, "get_buffers", lambda self, *args: None)
        return MiniMaxM3KVCacheManagerV2(
            num_layers=4, sparse_attention_config=MiniMaxM3SparseAttentionConfig()
        )


# ---------------------------------------------------------------------------
# Short-sequence identity table
# ---------------------------------------------------------------------------


def _short_table_metadata(
    policy: MsaRuntimePolicy, manager: MiniMaxM3KVCacheManagerV2, num_contexts: int = 0
) -> MiniMaxM3MsaSparseAttentionMetadata:
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata.msa_policy = policy
    metadata._num_contexts = num_contexts
    metadata.kv_cache_manager = manager
    metadata.max_num_sequences = 4
    metadata.max_num_tokens = 16
    metadata.cuda_graph_buffers = None
    metadata.is_cuda_graph = False
    metadata.mapping = Mapping()
    return metadata


def _identity(n_valid: list[int]) -> torch.Tensor:
    ar = torch.arange(MSA_REQUIRED_TOPK, dtype=torch.int32)
    valid = torch.tensor(n_valid, dtype=torch.int32)
    table = torch.where(ar[None, :] < valid[:, None], ar[None, :], -1)
    return table.unsqueeze(0).expand(NUM_KV_HEADS, len(n_valid), MSA_REQUIRED_TOPK)


def _staged(metadata: MiniMaxM3MsaSparseAttentionMetadata, total_q: int) -> torch.Tensor:
    torch.cuda.synchronize()
    size = NUM_KV_HEADS * total_q * MSA_REQUIRED_TOPK
    return metadata.msa_short_table[:size].view(NUM_KV_HEADS, total_q, MSA_REQUIRED_TOPK).cpu()


@requires_cuda
@pytest.mark.parametrize("n_valid", [[1], [1, 16, 5], [16, 16, 16, 16]])
def test_short_steps_stage_the_identity_table(manager, n_valid):
    metadata = _short_table_metadata(MsaRuntimePolicy(short_sequence_identity=True), manager)

    metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)

    assert metadata._msa_short_ok
    assert metadata.msa_short_table.is_cuda
    assert torch.equal(_staged(metadata, len(n_valid)), _identity(n_valid))


@requires_cuda
def test_the_default_policy_stages_nothing(manager):
    metadata = _short_table_metadata(MsaRuntimePolicy(), manager)

    metadata._stage_msa_short_table(torch.tensor([1, 2], dtype=torch.int32), PARAMS)

    assert not metadata._msa_short_ok
    assert metadata.msa_short_table is None


@requires_cuda
@pytest.mark.parametrize(
    ("n_valid", "num_contexts"),
    [([1, MSA_REQUIRED_TOPK + 1], 0), ([], 0), ([1, 2], 1)],
    ids=["more_than_topk_blocks", "empty_step", "context_step"],
)
def test_ineligible_steps_do_not_use_the_identity_table(manager, n_valid, num_contexts):
    metadata = _short_table_metadata(MsaRuntimePolicy(short_sequence_identity=True), manager)
    # A previous eligible step staged the table.
    metadata._stage_msa_short_table(torch.tensor([3], dtype=torch.int32), PARAMS)
    assert metadata._msa_short_ok

    metadata._num_contexts = num_contexts
    metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)

    assert not metadata._msa_short_ok


@requires_cuda
@pytest.mark.parametrize("skip_cache", [False, True], ids=["plain", "skip_cache"])
def test_the_identity_table_follows_each_step(manager, skip_cache):
    """Also with the engine's skip cache active, which skips the repeated [2, 4] upload."""
    metadata = _short_table_metadata(MsaRuntimePolicy(short_sequence_identity=True), manager)

    with h2d_cache.active(H2DSkipCache() if skip_cache else None):
        for n_valid in ([2, 3], [2, 4], [2, 4], [16], [2, 3]):
            metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)
            assert torch.equal(_staged(metadata, len(n_valid)), _identity(n_valid))


@requires_cuda
def test_the_identity_table_grows_past_its_first_bound(manager):
    metadata = _short_table_metadata(MsaRuntimePolicy(short_sequence_identity=True), manager)
    metadata._stage_msa_short_table(torch.tensor([1], dtype=torch.int32), PARAMS)
    first_numel = metadata.msa_short_table.numel()
    n_valid = [i % MSA_REQUIRED_TOPK + 1 for i in range(first_numel // MSA_REQUIRED_TOPK)]

    metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)

    assert metadata._msa_short_ok
    assert metadata.msa_short_table.numel() >= NUM_KV_HEADS * len(n_valid) * MSA_REQUIRED_TOPK
    assert torch.equal(_staged(metadata, len(n_valid)), _identity(n_valid))


# ---------------------------------------------------------------------------
# Block-table copy shortcut
# ---------------------------------------------------------------------------


@pytest.fixture
def base_copy(monkeypatch):
    """The V2 base copy, recorded instead of launched."""
    copy = create_autospec(KVCacheManagerV2.copy_batch_block_offsets, spec_set=True)
    monkeypatch.setattr(KVCacheManagerV2, "copy_batch_block_offsets", copy)
    return copy.mock


def _block_table_host(manager: MiniMaxM3KVCacheManagerV2) -> torch.Tensor:
    manager.host_kv_cache_block_offsets = torch.arange(2 * 4 * 2 * 8, dtype=torch.int32).view(
        2, 4, 2, 8
    )
    return manager.host_kv_cache_block_offsets


def _decode_copy(manager, dst, request_ids=(0, 1), num_contexts=0, max_blocks=8):
    manager.copy_batch_block_offsets(
        dst, list(request_ids), 1, num_contexts, len(request_ids), max_blocks=max_blocks
    )


def test_block_offsets_shortcut_is_off_by_default(manager, base_copy):
    _block_table_host(manager)
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst)

    assert not manager.skip_unchanged_block_offsets
    assert base_copy.call_count == 2


def test_an_unchanged_decode_copy_is_skipped(manager, base_copy):
    _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst)

    assert base_copy.call_count == 1


def test_a_changed_host_table_is_copied(manager, base_copy):
    host = _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    host[0, 1, 0, 3] += 100  # a request grew into a new page
    _decode_copy(manager, dst)
    _decode_copy(manager, dst)

    assert base_copy.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        dict(request_ids=(1, 0)),
        dict(request_ids=(0, 2)),
        dict(max_blocks=4),
    ],
    ids=["batch_order", "request_set", "max_blocks"],
)
def test_a_changed_batch_is_copied(manager, base_copy, change):
    _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst, **change)

    assert base_copy.call_count == 2


def test_each_destination_keeps_its_own_record(manager, base_copy):
    _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    first = torch.zeros(2, 4, 2, 8, dtype=torch.int32)
    second = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, first)
    _decode_copy(manager, second)
    _decode_copy(manager, first)
    _decode_copy(manager, second)

    assert base_copy.call_count == 2


def test_a_step_with_contexts_always_copies_and_resets_the_record(manager, base_copy):
    _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst, num_contexts=1)
    _decode_copy(manager, dst, num_contexts=1)
    # The context steps wrote dst through the base copy: the next decode copy must run.
    _decode_copy(manager, dst)

    assert base_copy.call_count == 4


def test_a_failed_block_copy_does_not_leave_a_valid_record(manager, base_copy):
    host = _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    dst = torch.zeros_like(host)
    _decode_copy(manager, dst)
    base_copy.side_effect = RuntimeError("copy failed")
    with pytest.raises(RuntimeError, match="copy failed"):
        _decode_copy(manager, dst, request_ids=(1, 0))
    base_copy.side_effect = None
    _decode_copy(manager, dst)
    assert base_copy.call_count == 3


def test_a_different_destination_view_requires_a_new_copy(manager, base_copy):
    host = _block_table_host(manager)
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=True, reuse_pool_views=False)
    storage = torch.zeros(2 * host.numel(), dtype=host.dtype)
    first = storage[: host.numel()].view_as(host)
    second = storage[host.numel() :].view_as(host)
    for dst in (first, second, first):
        _decode_copy(manager, dst)
    assert base_copy.call_count == 3


# ---------------------------------------------------------------------------
# Pool-view shortcut
# ---------------------------------------------------------------------------

NUM_SLOTS, SCALE, PAGE_SHAPE = 3, 4, [2, 8, 16]


@pytest.fixture
def pools(monkeypatch):
    """Two pools the manager's slot geometry resolves to: one for K/V, one for block scales."""
    numel = NUM_SLOTS * SCALE * 2 * 8 * 16
    kv_pool = torch.zeros(numel, dtype=torch.bfloat16, device="cuda")
    scale_pool = torch.zeros(numel, dtype=torch.uint8, device="cuda")

    def kv_slot_geometry(self, layer_idx, kv_layout):
        del self, layer_idx, kv_layout
        return kv_pool.data_ptr(), torch.bfloat16, NUM_SLOTS, SCALE, list(PAGE_SHAPE)

    def kv_scale_slot_geometry(self, layer_idx, kv_layout):
        del self, layer_idx, kv_layout
        return scale_pool.data_ptr(), torch.uint8, NUM_SLOTS, SCALE, list(PAGE_SHAPE)

    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "_kv_slot_geometry", kv_slot_geometry)
    monkeypatch.setattr(
        MiniMaxM3KVCacheManagerV2, "_kv_scale_slot_geometry", kv_scale_slot_geometry
    )
    return kv_pool, scale_pool


def _assert_kv_view(
    view: torch.Tensor, pool: torch.Tensor, manager: MiniMaxM3KVCacheManagerV2
) -> None:
    # The pool is laid out in the manager's physical layout; a view in the other layout swaps the
    # page's head and token axes over the same memory.
    page = list(PAGE_SHAPE)
    if manager._main_kv_view_layout != manager._main_kv_layout_name():
        page[0], page[1] = page[1], page[0]
    assert view.data_ptr() == pool.data_ptr()
    assert list(view.shape) == [NUM_SLOTS, 2, *page]
    assert view.stride(0) == SCALE * 2 * 8 * 16


@requires_cuda
def test_pool_views_are_rebuilt_on_every_call_by_default(manager, pools):
    kv_pool, _ = pools

    first = manager.get_buffers(3)
    second = manager.get_buffers(3)

    assert not manager.reuse_pool_views
    assert first is not second
    _assert_kv_view(first, kv_pool, manager)
    _assert_kv_view(second, kv_pool, manager)


@requires_cuda
def test_reused_pool_views_are_built_once_per_geometry(manager, pools):
    kv_pool, scale_pool = pools
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=False, reuse_pool_views=True)

    view = manager.get_buffers(3)
    scale_view = manager.get_block_scale_buffers(3)

    assert manager.get_buffers(3) is view
    assert manager.get_block_scale_buffers(3) is scale_view
    assert scale_view is not view
    _assert_kv_view(view, kv_pool, manager)
    assert scale_view.data_ptr() == scale_pool.data_ptr()
    assert scale_view.dtype == torch.uint8


@requires_cuda
def test_reused_pool_views_are_kept_per_inference_mode(manager, pools):
    kv_pool, _ = pools
    manager.configure_decode_shortcuts(skip_unchanged_block_offsets=False, reuse_pool_views=True)

    view = manager.get_buffers(3)
    with torch.inference_mode():
        inference_view = manager.get_buffers(3)
        assert manager.get_buffers(3) is inference_view

    assert inference_view is not view
    assert inference_view.is_inference()
    assert not view.is_inference()
    _assert_kv_view(inference_view, kv_pool, manager)


@requires_cuda
def test_unused_slots_are_filled_after_a_full_buffer_step():
    buf = torch.full((8,), 7, dtype=torch.int32, device="cuda")
    with h2d_cache.active(H2DSkipCache()):
        _fill_unused_slots(buf, 3)
        # A step whose live rows cover the whole buffer overwrites the filled tail.
        buf.copy_(torch.arange(8, dtype=torch.int32, device="cuda"))
        _fill_unused_slots(buf, buf.shape[0])
        buf[:3] = 0
        _fill_unused_slots(buf, 3)
    assert buf.tolist() == [0, 0, 0, -1, -1, -1, -1, -1]

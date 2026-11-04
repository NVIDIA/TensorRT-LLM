# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA decode-step preparation, MSA's own behaviour in every configuration.

Covers the short-sequence identity table the metadata stages (also with unchanged uploads skipped), the records the
CUDA graph metadata copies share, and the KV cache manager's block-table and pool-view shortcuts.
"""

from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    MiniMaxM3KVCacheManagerV2,
    msa_backend,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
    MiniMaxM3SparseMetadataParams,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
    write_subpage_block_table,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
    _fill_unused_slots,
)
from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

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
        # Set by the base's page-table setup, which this fake base skips.
        self.num_pools = self.num_attention_op_pools = 1

    def no_index_k_buffer(self, layer_idx, **kwargs):
        del self, layer_idx, kwargs
        return None

    monkeypatch.setattr(KVCacheManagerV2, "__init__", fake_base_init)
    monkeypatch.setattr(KVCacheManagerV2, "get_index_k_buffer", no_index_k_buffer)
    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "_compute_num_total_slots", lambda self: 0)
    # The constructor resolves each layer's K/V views; the bare base has no pools to resolve them
    # from. Stub that only while constructing, so the pool-view tests below build real views.
    with monkeypatch.context() as construct:
        construct.setattr(MiniMaxM3KVCacheManagerV2, "get_buffers", lambda self, *args: None)
        return MiniMaxM3KVCacheManagerV2(
            num_layers=4, sparse_attention_config=MiniMaxM3SparseAttentionConfig()
        )


# ---------------------------------------------------------------------------
# Short-sequence identity table
# ---------------------------------------------------------------------------


def _short_table_metadata(
    manager: MiniMaxM3KVCacheManagerV2, num_contexts: int = 0
) -> MiniMaxM3MsaSparseAttentionMetadata:
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata._num_contexts = num_contexts
    metadata.kv_cache_manager = manager
    metadata.max_num_sequences = 4
    metadata.max_num_tokens = 16
    metadata.cuda_graph_buffers = None
    metadata.is_cuda_graph = False
    metadata.mapping = Mapping()
    metadata._upload_buffers = {}
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
    metadata = _short_table_metadata(manager)

    metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)

    assert metadata._msa_short_ok
    assert metadata.msa_short_table.is_cuda
    assert torch.equal(_staged(metadata, len(n_valid)), _identity(n_valid))


@requires_cuda
@pytest.mark.parametrize(
    ("n_valid", "num_contexts"),
    [([1, MSA_REQUIRED_TOPK + 1], 0), ([], 0), ([1, 2], 1)],
    ids=["more_than_topk_blocks", "empty_step", "context_step"],
)
def test_ineligible_steps_do_not_use_the_identity_table(manager, n_valid, num_contexts):
    metadata = _short_table_metadata(manager)
    # A previous eligible step staged the table.
    metadata._stage_msa_short_table(torch.tensor([3], dtype=torch.int32), PARAMS)
    assert metadata._msa_short_ok

    metadata._num_contexts = num_contexts
    metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)

    assert not metadata._msa_short_ok


@requires_cuda
@pytest.mark.parametrize("skip_cache", [False, True], ids=["plain", "skip_cache"])
def test_the_identity_table_follows_each_step(manager, skip_cache, monkeypatch):
    """Also with the metadata skipping unchanged uploads: the short table is pooled CUDA graph metadata, so even the
    repeated [2, 4] upload is issued."""
    metadata = _short_table_metadata(manager)
    monkeypatch.setattr(type(metadata), "skips_unchanged_uploads", skip_cache)

    issued = []
    real_copy = device_copy.copy

    def counting_copy(site, dst, src, **kwargs):
        issued.append(real_copy(site, dst, src, **kwargs))
        return issued[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(device_copy, "copy", counting_copy)
        for n_valid in ([2, 3], [2, 4], [2, 4], [16], [2, 3]):
            metadata._stage_msa_short_table(torch.tensor(n_valid, dtype=torch.int32), PARAMS)
            assert torch.equal(_staged(metadata, len(n_valid)), _identity(n_valid))
    assert issued == [True] * 5


@requires_cuda
def test_the_identity_table_grows_past_its_first_bound(manager):
    metadata = _short_table_metadata(manager)
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
    manager.skip_unchanged_block_offsets = True
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst)

    assert base_copy.call_count == 1


def test_a_changed_host_table_is_copied(manager, base_copy):
    host = _block_table_host(manager)
    manager.skip_unchanged_block_offsets = True
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
    manager.skip_unchanged_block_offsets = True
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst, **change)

    assert base_copy.call_count == 2


def test_each_destination_keeps_its_own_record(manager, base_copy):
    _block_table_host(manager)
    manager.skip_unchanged_block_offsets = True
    first = torch.zeros(2, 4, 2, 8, dtype=torch.int32)
    second = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, first)
    _decode_copy(manager, second)
    _decode_copy(manager, first)
    _decode_copy(manager, second)

    assert base_copy.call_count == 2


def test_a_step_with_contexts_always_copies_and_resets_the_record(manager, base_copy):
    _block_table_host(manager)
    manager.skip_unchanged_block_offsets = True
    dst = torch.zeros(2, 4, 2, 8, dtype=torch.int32)

    _decode_copy(manager, dst)
    _decode_copy(manager, dst, num_contexts=1)
    _decode_copy(manager, dst, num_contexts=1)
    # The context steps wrote dst through the base copy: the next decode copy must run.
    _decode_copy(manager, dst)

    assert base_copy.call_count == 4


def test_a_failed_block_copy_does_not_leave_a_valid_record(manager, base_copy):
    host = _block_table_host(manager)
    manager.skip_unchanged_block_offsets = True
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
    manager.skip_unchanged_block_offsets = True
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
def test_pool_views_are_rebuilt_on_every_call_without_reuse(manager, pools):
    kv_pool, _ = pools
    manager.reuse_pool_views = False

    first = manager.get_buffers(3)
    second = manager.get_buffers(3)

    assert first is not second
    _assert_kv_view(first, kv_pool, manager)
    _assert_kv_view(second, kv_pool, manager)


@requires_cuda
def test_reused_pool_views_are_built_once_per_geometry(manager, pools):
    kv_pool, scale_pool = pools
    assert manager.reuse_pool_views

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
    """The tail fill is never skipped (pooled CUDA graph metadata), so it follows every live-row write."""
    buf = torch.full((8,), 7, dtype=torch.int32, device="cuda")
    _fill_unused_slots(buf, 3)
    # A step whose live rows cover the whole buffer overwrites the filled tail.
    buf.copy_(torch.arange(8, dtype=torch.int32, device="cuda"))
    _fill_unused_slots(buf, buf.shape[0])
    buf[:3] = 0
    _fill_unused_slots(buf, 3)
    assert buf.tolist() == [0, 0, 0, -1, -1, -1, -1, -1]


class _PageIndexView:
    """A request's KV cache as get_block_ids_per_seq reads it: an int32 page-index view per pool."""

    def __init__(self, view: np.ndarray):
        self._view = view

    def get_base_page_indices(self, pool_id: int) -> np.ndarray:
        assert pool_id == 0
        return self._view


def _page_index_views(seed: int):
    rng = np.random.default_rng(seed)
    batch = int(rng.integers(0, 5))
    same_width = rng.random() < 0.5
    width = int(rng.integers(1, 9000))
    views = {}
    for rid in range(batch):
        w = width if same_width else int(rng.integers(0, 9000))
        view = rng.integers(0, 2**31 - 1, w, dtype=np.int64).astype(np.int32)
        view[int(rng.integers(0, w + 1)) :] = BAD_PAGE_INDEX
        view[rng.random(w) < float(rng.choice([0.0, 0.01, 0.5]))] = BAD_PAGE_INDEX
        view.setflags(write=False)
        views[100 + rid] = _PageIndexView(view)
    return views


def _bare_manager() -> MiniMaxM3KVCacheManagerV2:
    """A manager with only the state get_block_ids_per_seq reads (no pools, no V2 base)."""
    manager = object.__new__(MiniMaxM3KVCacheManagerV2)
    manager.layer_specs = None
    manager.kv_factor = 2
    return manager


@pytest.mark.parametrize("seed", range(24))
def test_block_ids_per_seq_matches_the_cache_index_rows(seed):
    """The view-copied table equals the get_batch_cache_indices rows, zero-padded with BAD_PAGE_INDEX as 0."""
    manager = _bare_manager()
    manager.kv_cache_map = _page_index_views(seed)
    request_ids = list(manager.kv_cache_map)

    table = manager.get_block_ids_per_seq(request_ids)

    rows = manager.get_batch_cache_indices(request_ids)
    width = max((len(row) for row in rows), default=0)
    expected = torch.zeros((len(rows), width), dtype=torch.int32)
    for i, row in enumerate(rows):
        expected[i, : len(row)] = torch.tensor(
            [0 if v == BAD_PAGE_INDEX else v for v in row], dtype=torch.int32
        )
    assert table.dtype == torch.int32
    if table.numel():  # an empty tensor never reports pinned
        assert table.is_pinned() == prefer_pinned()
    assert torch.equal(table, expected)


# ---------------------------------------------------------------------------
# Sub-page block table expanded inside the forward
# ---------------------------------------------------------------------------


def _subpage_metadata(
    rows: int, max_blocks: int, factor: int
) -> MiniMaxM3MsaSparseAttentionMetadata:
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata._msa_subpage_rows = rows  # as prepare() stages them
    metadata.msa_block_table = torch.randint(
        0, 1 << 20, (rows + 2, max_blocks), dtype=torch.int32, device="cuda"
    )
    metadata.msa_subpage_block_table = torch.full(
        (rows + 2, 2, max_blocks), -7, dtype=torch.int32, device="cuda"
    )
    metadata._msa_subpages_per_slot = factor
    metadata._msa_subpage_expanded = False
    return metadata


def _expected_subpages(metadata, rows: int) -> torch.Tensor:
    out = torch.empty_like(metadata.msa_subpage_block_table[:rows])
    write_subpage_block_table(metadata.msa_block_table[:rows], metadata._msa_subpages_per_slot, out)
    return out


@requires_cuda
def test_the_first_reader_of_a_forward_expands_the_subpage_table(monkeypatch):
    monkeypatch.setattr(msa_backend, "SUBPAGE_IN_GRAPH", True)
    metadata = _subpage_metadata(rows=3, max_blocks=40, factor=2)
    calls = []
    real = msa_backend.write_subpage_block_table
    monkeypatch.setattr(
        msa_backend, "write_subpage_block_table", lambda *a: (calls.append(1), real(*a))
    )

    table, factor = metadata.msa_subpage_rows(0, 3)
    again, _ = metadata.msa_subpage_rows(1, 3)

    assert factor == 2 and len(calls) == 1
    assert torch.equal(table, _expected_subpages(metadata, 3))
    assert torch.equal(again, table[1:3])
    # Rows past the step's requests are left alone.
    assert metadata.msa_subpage_block_table[3:].eq(-7).all()


@requires_cuda
def test_each_forward_expands_the_current_block_table(monkeypatch):
    monkeypatch.setattr(msa_backend, "SUBPAGE_IN_GRAPH", True)
    metadata = _subpage_metadata(rows=2, max_blocks=16, factor=4)
    metadata.msa_subpage_rows(0, 2)
    metadata.msa_block_table[:2, :5] += 11
    # The next forward (on_update_kv_lens opens it).
    metadata._msa_subpage_expanded = False

    table, _ = metadata.msa_subpage_rows(0, 2)

    assert torch.equal(table, _expected_subpages(metadata, 2))


@requires_cuda
def test_a_captured_forward_expands_the_table_at_every_replay(monkeypatch):
    monkeypatch.setattr(msa_backend, "SUBPAGE_IN_GRAPH", True)
    metadata = _subpage_metadata(rows=4, max_blocks=32, factor=2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        metadata._msa_subpage_expanded = False
        metadata.msa_subpage_rows(0, 4)
        metadata.msa_subpage_rows(2, 4)
    torch.cuda.current_stream().wait_stream(stream)

    for step in range(3):
        # The pack lands this step's block table; then the replay.
        metadata.msa_block_table[:4].copy_(
            torch.randint(0, 1 << 20, (4, 32), dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(metadata.msa_subpage_block_table[:4], _expected_subpages(metadata, 4)), (
            step
        )


def test_nothing_is_expanded_without_a_prepared_step():
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata.msa_subpage_block_table = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3)
    metadata._msa_subpages_per_slot = 2

    table, factor = metadata.msa_subpage_rows(0, 2)

    assert factor == 2 and torch.equal(table, torch.arange(12, dtype=torch.int32).reshape(2, 2, 3))


# ---------------------------------------------------------------------------
# msa_out_cache_loc tail filled inside the forward
# ---------------------------------------------------------------------------


def _tail_metadata(cap: int, start) -> MiniMaxM3MsaSparseAttentionMetadata:
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata.msa_out_cache_loc = torch.arange(100, 100 + cap, dtype=torch.int32, device="cuda")
    metadata._msa_tail_start = start
    return metadata


@requires_cuda
def test_the_forward_fills_only_past_the_live_slots():
    metadata = _tail_metadata(10, 4)
    metadata._fill_tail_in_forward()
    assert metadata.msa_out_cache_loc.tolist() == [100, 101, 102, 103] + [-1] * 6

    untouched = _tail_metadata(6, None)
    untouched._fill_tail_in_forward()
    assert untouched.msa_out_cache_loc.tolist() == list(range(100, 106))

    full = _tail_metadata(6, 6)
    full._fill_tail_in_forward()
    assert full.msa_out_cache_loc.tolist() == list(range(100, 106))


@requires_cuda
def test_a_captured_forward_refills_its_tail_at_every_replay():
    """Another graph key sharing the pool block writes live slots past this key's tail between replays; each replay of
    this key's forward must invalidate them again before its own readers run."""
    metadata = _tail_metadata(12, 5)
    buf = metadata.msa_out_cache_loc
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        metadata._fill_tail_in_forward()
    torch.cuda.current_stream().wait_stream(stream)

    for step in range(3):
        buf[:5] = (
            torch.arange(5, dtype=torch.int32, device="cuda") + 10 * step
        )  # this step's live slots
        buf[5:9] = 777  # another key's in-graph writes into the shared block
        graph.replay()
        torch.cuda.synchronize()
        assert buf.tolist() == [10 * step + i for i in range(5)] + [-1] * 7, step


# ---------------------------------------------------------------------------
# Owner-side skipping is MSA's default
# ---------------------------------------------------------------------------


def test_msa_metadata_skips_unchanged_uploads_and_arms_its_manager(manager):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    assert MiniMaxM3MsaSparseAttentionMetadata.skips_unchanged_uploads
    assert not TrtllmAttentionMetadata.skips_unchanged_uploads
    assert (
        MiniMaxM3MsaSparseAttentionMetadata.short_long_seq_len_threshold(
            MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128)
        )
        == 16 * 128
    )
    # One generation step's tokens of headroom for the in-flight step's accepted tokens.
    assert (
        MiniMaxM3MsaSparseAttentionMetadata.short_long_seq_len_threshold(
            MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=16, sparse_block_size=128),
            tokens_per_gen_step=4,
        )
        == 16 * 128 - 4
    )
    assert TrtllmAttentionMetadata.short_long_seq_len_threshold(None) is None
    assert not manager.skip_unchanged_block_offsets
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata.kv_cache_manager = manager
    metadata._arm_kv_cache_manager()
    assert manager.skip_unchanged_block_offsets


@requires_cuda
def test_cuda_graph_metadata_copies_share_one_record_set():
    """Graph metadata of different batch sizes are shallow copies writing the same buffers: a write through one
    copy must make the others send their values again."""
    import copy

    from tensorrt_llm._torch.pyexecutor.device_copy import CopySite

    engine_md = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    engine_md._ensure_upload_buffers()
    graphs = [copy.copy(engine_md) for _ in range(3)]
    for g in graphs:
        # CUDA graph metadata re-runs __post_init__, which keeps the shared DeviceCopyBuffers.
        g._ensure_upload_buffers()
    assert all(g._upload_buffers is engine_md._upload_buffers for g in graphs)

    buf = torch.zeros(4, dtype=torch.int32, device="cuda")
    one = torch.tensor([1, 2, 3, 4], dtype=torch.int32).pin_memory()
    two = torch.tensor([5, 6, 7, 8], dtype=torch.int32).pin_memory()
    site = CopySite.ATTN_PROMPT_LENS
    assert graphs[0]._upload_metadata(site, buf, one) is True
    assert graphs[1]._upload_metadata(site, buf, two) is True
    # graphs[1] overwrote the buffer: graphs[0]'s identical values must be sent again, then skipped.
    assert graphs[0]._upload_metadata(site, buf, one) is True
    assert graphs[2]._upload_metadata(site, buf, one) is False
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 2, 3, 4]
    # One DeviceCopyBuffer per destination, with the upload's site, shared by every copy.
    shared = graphs[0]._upload_buffer(site, buf)
    assert shared is graphs[2]._upload_buffer(site, buf) is engine_md._upload_buffer(site, buf)
    assert shared.site is site and shared.tensor is buf

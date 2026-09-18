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
"""Pure-logic tests for MiniMax-M3's shared Eagle3 draft layers: each gets its own
virtual attention-op pool rooted at its K page inside the mega-slot.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    cache_manager as m3_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3DraftSubpageView,
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
    extend_attention_op_pools_for_shared_draft_layers,
    shared_draft_layer_count,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType

DRAFT_LOCAL_LAYER = 60
SCALE = 179  # sub-pages per M3 mega-slot: 3 dense x 2 + 57 sparse x 3 + draft x 2
DRAFT_K_ADDR = 0x7000_0000


def test_virtual_pool_is_rooted_at_the_draft_k_page():
    pool_pointers = torch.tensor([[0x6000_0000, 0]], dtype=torch.int64)
    pool_mapping = torch.tensor([[0, i] for i in range(61)], dtype=torch.int32)

    pointers, mapping, index_scales, kv_offsets, op_pools = (
        extend_attention_op_pools_for_shared_draft_layers(
            pool_pointers, pool_mapping, 1, [(DRAFT_LOCAL_LAYER, DRAFT_K_ADDR, SCALE)]
        )
    )

    assert pointers.tolist() == [[0x6000_0000, 0], [DRAFT_K_ADDR, 0]]
    # The draft layer moves to the new pool at offset 0; target rows are untouched.
    assert mapping[DRAFT_LOCAL_LAYER].tolist() == [1, 0]
    assert mapping[:DRAFT_LOCAL_LAYER].tolist() == pool_mapping[:DRAFT_LOCAL_LAYER].tolist()
    # Slot s -> page s * SCALE for K and s * SCALE + 1 for V.
    assert index_scales.tolist() == [SCALE]
    assert kv_offsets.tolist() == [1]
    assert op_pools == [(1, 0)]


def test_block_offset_copy_fills_the_virtual_pool_from_the_source_pool(monkeypatch):
    """The base copy fills the storage pools; the override then fills the virtual pool
    from the source pool's slot ids, and does nothing extra without draft layers.
    """
    calls = []

    def fake_base_copy(
        self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None
    ):
        calls.append(("base", request_ids, num_seqs, max_blocks))

    def fake_device_copy(host, dst, copy_idx, index_scales, kv_offsets, stream):
        calls.append(
            ("virtual", host, dst, copy_idx, index_scales.tolist(), kv_offsets.tolist(), stream)
        )

    monkeypatch.setattr(KVCacheManagerV2, "copy_batch_block_offsets", fake_base_copy)
    monkeypatch.setattr(m3_cache_manager, "copy_batch_block_offsets_to_device", fake_device_copy)

    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.FP8
    manager._draft_op_pools = ((1, 0),)
    manager._draft_index_scales = torch.tensor([SCALE], dtype=torch.int32)
    manager._draft_kv_offsets = torch.tensor([1], dtype=torch.int32)
    manager.host_kv_cache_block_offsets = torch.zeros((1, 4, 2, 8), dtype=torch.int32)
    copy_idx = torch.tensor([2, 0], dtype=torch.int32)
    manager.index_mapper = SimpleNamespace(get_copy_index=lambda ids, nc, bw: copy_idx)
    manager._stream = SimpleNamespace(cuda_stream=1234)
    dst = torch.zeros((2, 4, 2, 8), dtype=torch.int32)

    manager.copy_batch_block_offsets(dst, [7, 9], 1, 0, 2, max_blocks=5)

    assert calls[0] == ("base", [7, 9], 2, 5)
    kind, host, dst_slice, idx, scales, offsets, stream = calls[1]
    assert kind == "virtual"
    assert (
        host.shape == (1, 4, 2, 8)
        and host.data_ptr() == manager.host_kv_cache_block_offsets.data_ptr()
    )
    assert dst_slice.shape == (1, 4, 2, 8) and dst_slice.data_ptr() == dst[1].data_ptr()
    assert idx is copy_idx
    assert scales == [SCALE] and offsets == [1] and stream == 1234

    # Non-speculative MiniMax-M3: the base copy is all that runs.
    calls.clear()
    plain = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    plain.copy_batch_block_offsets(dst, [7], 1, 0, 1)
    assert calls == [("base", [7], 1, None)]


@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("swa_scratch_reuse", [False, True])
def test_per_layer_page_tables_get_no_virtual_pools(monkeypatch, dtype, swa_scratch_reuse):
    """With per-layer page tables every layer already has its own pool."""
    pointers = torch.tensor([[0x6000_0000 + i, 0] for i in range(61)])
    mapping = torch.tensor([[i, 0] for i in range(61)], dtype=torch.int32)

    def fake_base_prepare(self, index_mapper_capacity):
        self._use_per_layer_page_tables = True
        self.kv_cache_pool_pointers = pointers.clone()
        self.kv_cache_pool_mapping = mapping.clone()
        self.num_attention_op_pools = 61

    monkeypatch.setattr(KVCacheManagerV2, "_prepare_page_table_tensor", fake_base_prepare)
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = dtype
    manager._shared_draft_layer_ids = [DRAFT_LOCAL_LAYER]
    manager.layer_offsets = {i: i for i in range(61)}
    manager.is_draft = False
    manager.enable_swa_scratch_reuse = swa_scratch_reuse
    manager.tokens_per_block = 128

    if swa_scratch_reuse:
        with pytest.raises(NotImplementedError, match="SWA scratch reuse"):
            manager._prepare_page_table_tensor(8)
        return

    manager._prepare_page_table_tensor(8)

    assert torch.equal(manager.kv_cache_pool_pointers, pointers)
    assert torch.equal(manager.kv_cache_pool_mapping, mapping)
    assert manager.num_attention_op_pools == 61
    assert manager._draft_op_pools == ()
    expected_extra_pages = {128} if dtype == DataType.FP8 else set()
    assert manager.trtllm_gen_extra_tokens_per_block == frozenset(expected_extra_pages)


def test_update_resources_refuses_tree_relocation(monkeypatch):
    """Linear acceptance rewinds through the base; tree acceptance is refused."""
    calls = []
    monkeypatch.setattr(KVCacheManagerV2, "update_resources", lambda self, *a: calls.append(a))
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.FP8
    linear = SimpleNamespace(
        py_num_accepted_draft_tokens=2, py_num_accepted_draft_tokens_indices=[]
    )
    tree = SimpleNamespace(
        py_num_accepted_draft_tokens=2, py_num_accepted_draft_tokens_indices=[0, 2]
    )

    manager.update_resources(SimpleNamespace(generation_requests=[linear]), None, 2)
    assert len(calls) == 1

    with pytest.raises(NotImplementedError, match="relocate"):
        manager.update_resources(SimpleNamespace(generation_requests=[linear, tree]), None, 2)


def test_draft_layout_locates_the_appended_tail():
    # num_layers is the target count; a per-layer head list already includes the
    # draft tail.
    assert derive_shared_draft_layout(60, 4, 1) == ([60], 60)
    assert derive_shared_draft_layout(60, [4] * 60 + [64], 1) == ([60], 60)
    assert derive_shared_draft_layout(60, 4, 0) == ([], 60)
    assert derive_shared_draft_layout(None, 4, 1) == ([], None)


def test_shared_draft_layer_count_follows_the_base_manager():
    # Same rule as get_pp_layers: a spec config and no layer_mask.
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=lambda: False,
        is_mtp_vanilla=lambda: False,
        is_eagle3_one_model=lambda: True,
    )
    eagle3 = SimpleNamespace(spec_dec_mode=mode, _num_draft_hidden_layers=None)
    assert shared_draft_layer_count(eagle3, None) == 1
    assert shared_draft_layer_count(eagle3, [True] * 60) == 0
    assert shared_draft_layer_count(None, None) == 0


NVFP4_DRAFT_LAYER = 60
NVFP4_SCALE = 178
NVFP4_ADDR = 0x7000_0000


class _Nvfp4BaseManager:
    tokens_per_block = 128
    max_blocks_per_seq = 16
    num_pools = 1
    # V2's flattened bound uses the target pool's 128-token page units and
    # base pointer. The draft view must not delegate this value.
    blocks_in_primary_pool = 1024 * NVFP4_SCALE

    def __init__(self):
        self.layer_offsets = {NVFP4_DRAFT_LAYER: NVFP4_DRAFT_LAYER}
        self.kv_cache_pool_mapping = torch.zeros((NVFP4_DRAFT_LAYER + 1, 2), dtype=torch.int32)
        self.kv_cache_pool_mapping[NVFP4_DRAFT_LAYER] = torch.tensor([0, 7], dtype=torch.int32)
        self.slot_rows = [[5, 7]]
        self.impl = SimpleNamespace(
            get_layer_group_id=lambda local: int(self.kv_cache_pool_mapping[local, 0])
        )

    def _kv_slot_geometry(self, layer_idx, kv_layout):
        assert layer_idx == NVFP4_DRAFT_LAYER
        page_shape = [self.tokens_per_block, 16, 128]
        return NVFP4_ADDR, torch.int8, 1024, NVFP4_SCALE, page_shape

    def _get_batch_cache_indices_by_pool_id(self, request_ids, *, pool_id):
        assert pool_id == 0
        return self.slot_rows[: len(request_ids)]


class _Nvfp4HybridManager(_Nvfp4BaseManager):
    dtype = DataType.NVFP4
    nvfp4_dense_tokens_per_block = 32
    num_pools = 2

    def __init__(self):
        super().__init__()
        self.kv_cache_pool_mapping[NVFP4_DRAFT_LAYER] = torch.tensor([1, 7], dtype=torch.int32)

    def is_fp8_subpaged_layer(self, layer_idx):
        assert layer_idx == NVFP4_DRAFT_LAYER
        return True

    def _fp8_dense_data_buffers(self, layer_idx):
        assert layer_idx == NVFP4_DRAFT_LAYER

        class _Pointer:
            shape = (1024, NVFP4_SCALE * 4, 16, 32, 128)

            @staticmethod
            def data_ptr():
                return NVFP4_ADDR

        return _Pointer(), None, NVFP4_SCALE * 4, 4

    def _get_batch_cache_indices_by_pool_id(self, request_ids, *, pool_id):
        assert pool_id == 1
        return self.slot_rows[: len(request_ids)]


def test_hybrid_view_publishes_an_fp8_pool_pointer_from_the_draft_pool():
    view = MiniMaxM3DraftSubpageView(_Nvfp4HybridManager(), [NVFP4_DRAFT_LAYER], 32)
    expected = [[NVFP4_ADDR, 0]]
    assert view.kv_cache_pool_pointers.tolist() == expected
    assert view.host_kv_cache_pool_pointers.tolist() == expected
    assert view.dtype == DataType.FP8
    assert view._source_pool_id == 1
    assert view.blocks_in_primary_pool == (1024 - 1) * NVFP4_SCALE * 4 + 8

    dst = torch.full((1, 1, 2, view.max_blocks_per_seq), -7, dtype=torch.int32)
    view.copy_batch_block_offsets(dst, request_ids=[123], beam_width=1, num_contexts=1, num_seqs=1)
    unit = NVFP4_SCALE * 4
    expected_k = [5 * unit + j for j in range(4)] + [7 * unit + j for j in range(4)]
    assert dst[0, 0, 0, :8].tolist() == expected_k
    assert dst[0, 0, 1, :8].tolist() == [page + 4 for page in expected_k]


def test_hybrid_view_rejects_non_p32_draft_pages():
    try:
        MiniMaxM3DraftSubpageView(_Nvfp4HybridManager(), [NVFP4_DRAFT_LAYER], 128)
    except AssertionError as error:
        assert "physical dense-cache page size P32" in str(error)
    else:
        raise AssertionError("expected NVFP4 Eagle draft view to require P32 pages")


@pytest.mark.parametrize("local_draft_layers", [[], [60], [61], [60, 61]])
@pytest.mark.parametrize("swa_scratch_reuse", [False, True])
def test_draft_subpage_accessor_respects_local_layers(
    monkeypatch: pytest.MonkeyPatch, local_draft_layers: list[int], swa_scratch_reuse: bool
) -> None:
    """Only local draft layers may create a view; scratch reuse is unsupported on every rank."""
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.NVFP4
    manager.is_draft = False
    manager._shared_draft_layer_ids = [60, 61]
    manager.layer_offsets = {layer: local for local, layer in enumerate([30, *local_draft_layers])}
    manager.enable_swa_scratch_reuse = swa_scratch_reuse
    manager._draft_subpage_view_obj = None
    view = SimpleNamespace(tokens_per_block=32, blocks_in_primary_pool=1024)
    create_view = Mock(return_value=view)
    monkeypatch.setattr(m3_cache_manager, "MiniMaxM3DraftSubpageView", create_view)

    if swa_scratch_reuse:
        with pytest.raises(NotImplementedError, match="SWA scratch reuse"):
            manager.get_draft_subpage_view()
        create_view.assert_not_called()
        assert manager._draft_subpage_view_obj is None
    elif not local_draft_layers:
        assert manager.get_draft_subpage_view() is None
        create_view.assert_not_called()
        assert manager._draft_subpage_view_obj is None
    else:
        assert manager.get_draft_subpage_view() is view
        assert manager.get_draft_subpage_view() is view
        create_view.assert_called_once_with(manager, local_draft_layers, 32)
    assert manager._shared_draft_layer_ids == [60, 61]


def test_nvfp4_manager_rejects_dynamic_tree_eagle_before_allocation():
    class _DynamicTreeConfig:
        use_dynamic_tree = True

    try:
        MiniMaxM3KVCacheManagerV2(
            dtype=DataType.NVFP4,
            spec_config=_DynamicTreeConfig(),
        )
    except NotImplementedError as error:
        assert "supports linear Eagle3" in str(error)
        assert "block scales" in str(error)
    else:
        raise AssertionError("expected NVFP4 dynamic-tree Eagle to be rejected")

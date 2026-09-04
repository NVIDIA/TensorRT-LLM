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
"""Pure-logic tests for MiniMaxM3DraftKVCacheView."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3DraftKVCacheView,
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType

DRAFT_LAYER = 60
SCALE = 178  # P128 pages per M3 mega-slot
ADDR = 0x7000_0000


class _FakeFlatPool:
    shape = ((1024 - 1) * SCALE + 2,)

    def data_ptr(self):
        return ADDR


class _FakeManager:
    tokens_per_block = 128
    max_blocks_per_seq = 16
    num_pools = 1
    num_attention_op_pools = 1
    enable_swa_scratch_reuse = False
    dtype = DataType.FP8
    _stream = object()

    def __init__(self):
        self.layer_offsets = {DRAFT_LAYER: DRAFT_LAYER}
        self.kv_cache_pool_mapping = torch.zeros((DRAFT_LAYER + 1, 2), dtype=torch.int32)
        self.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([0, 7], dtype=torch.int32)
        self.kv_cache_pool_pointers = torch.tensor([[ADDR - 1024, 0]], dtype=torch.int64)
        self.index_scales = torch.tensor([SCALE], dtype=torch.int32)
        self.kv_offset = torch.tensor([1], dtype=torch.int32)
        self.host_kv_cache_block_offsets = torch.zeros((1, 1, 2, 16), dtype=torch.int32)

    def get_kv_subpage_pool(self, layer_idx, kv_layout):
        assert layer_idx == DRAFT_LAYER
        assert kv_layout == "HND"
        return _FakeFlatPool(), SCALE

    def is_fp8_dense_layer(self, layer_idx):
        assert layer_idx == DRAFT_LAYER
        return False


class _FakeHybridManager(_FakeManager):
    dtype = DataType.NVFP4
    num_pools = 2

    def __init__(self):
        super().__init__()
        self.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([1, 7], dtype=torch.int32)
        self.index_scales = torch.tensor([3, SCALE], dtype=torch.int32)
        self.kv_offset = torch.tensor([1, 1], dtype=torch.int32)
        self.host_kv_cache_block_offsets = torch.zeros((2, 1, 2, 16), dtype=torch.int32)

    def is_fp8_dense_layer(self, layer_idx):
        assert layer_idx == DRAFT_LAYER
        return True


def _make_view():
    return MiniMaxM3DraftKVCacheView(_FakeManager(), [DRAFT_LAYER])


def test_view_geometry():
    view = _make_view()
    assert view.tokens_per_block == 128
    assert view.max_blocks_per_seq == 16
    assert view.num_pools == view.num_attention_op_pools == 1
    assert view.kv_cache_pool_pointers.tolist() == [[ADDR, 0]]
    assert view.host_kv_cache_pool_pointers.tolist() == [[ADDR, 0]]
    assert view.kv_cache_pool_mapping[DRAFT_LAYER].tolist() == [0, 0]
    assert view.blocks_in_primary_pool == (1024 - 1) * SCALE + 2
    assert view.trtllm_gen_extra_tokens_per_block == frozenset({128})


def test_block_table_uses_native_p128_copy(monkeypatch):
    view = _make_view()
    calls = []

    def fake_copy(
        self,
        dst_tensor,
        request_ids,
        beam_width,
        num_contexts,
        num_seqs,
        max_blocks=None,
    ):
        calls.append((request_ids, beam_width, num_contexts, num_seqs, max_blocks))
        assert self.index_scales.tolist() == [SCALE]
        assert self.kv_offset.tolist() == [1]
        dst_tensor.zero_()
        for block_idx, slot in enumerate((5, 7)):
            dst_tensor[0, 0, 0, block_idx] = slot * SCALE
            dst_tensor[0, 0, 1, block_idx] = slot * SCALE + 1

    monkeypatch.setattr(KVCacheManagerV2, "copy_batch_block_offsets", fake_copy)
    dst = torch.full((1, 1, 2, view.max_blocks_per_seq), -7, dtype=torch.int32)

    view.copy_batch_block_offsets(
        dst,
        request_ids=[123],
        beam_width=1,
        num_contexts=1,
        num_seqs=1,
        max_blocks=9,
    )

    assert dst[0, 0, 0, :2].tolist() == [5 * SCALE, 7 * SCALE]
    assert dst[0, 0, 1, :2].tolist() == [5 * SCALE + 1, 7 * SCALE + 1]
    assert dst[0, 0, 0, 2:].tolist() == [0] * 14
    assert dst[0, 0, 1, 2:].tolist() == [0] * 14
    assert calls == [([123], 1, 1, 1, 9)]


def test_hybrid_view_uses_the_draft_layers_actual_fp8_pool():
    manager = _FakeHybridManager()
    # A heterogeneous source pool reports its first layer's page scale, not
    # the rerooted draft layer's flat-pool stride.
    manager.index_scales[1] = SCALE + 11
    view = MiniMaxM3DraftKVCacheView(manager, [DRAFT_LAYER])

    assert view.dtype == DataType.FP8
    assert view._source_pool_id == 1
    assert view.index_scales.tolist() == [SCALE]
    assert view.kv_offset.tolist() == [1]
    assert view.host_kv_cache_block_offsets.data_ptr() == (
        manager.host_kv_cache_block_offsets[1:2].data_ptr()
    )
    assert view.kv_cache_pool_pointers.tolist() == [[ADDR, 0]]
    assert view.blocks_in_primary_pool == (1024 - 1) * SCALE + 2


def test_nvfp4_manager_rejects_dynamic_tree_eagle_before_allocation():
    class _DynamicTreeConfig:
        use_dynamic_tree = True

    with pytest.raises(NotImplementedError, match="block scales"):
        MiniMaxM3KVCacheManagerV2(
            dtype=DataType.NVFP4,
            spec_config=_DynamicTreeConfig(),
        )


def test_free_resources_is_noop():
    _make_view().free_resources(object())


def test_manager_accessor_builds_and_caches_view():
    class _FakeSharedManager(_FakeManager):
        is_draft = False
        sparse_layer_ids = list(range(3, 60))

        def __init__(self):
            super().__init__()
            self._shared_draft_layer_ids = [DRAFT_LAYER]
            self._draft_kv_cache_view_obj = None

    manager = _FakeSharedManager()
    get_view = MiniMaxM3KVCacheManagerV2.get_draft_kv_cache_view
    view = get_view(manager)
    assert isinstance(view, MiniMaxM3DraftKVCacheView)
    assert get_view(manager) is view

    manager_draft = _FakeSharedManager()
    manager_draft.is_draft = True
    assert get_view(manager_draft) is None


def test_view_rejects_non_p128_manager():
    manager = _FakeManager()
    manager.tokens_per_block = 32
    with pytest.raises(ValueError, match="tokens_per_block=128"):
        MiniMaxM3DraftKVCacheView(manager, [DRAFT_LAYER])


def test_view_rejects_multiple_draft_layers():
    with pytest.raises(ValueError, match="exactly one draft layer"):
        MiniMaxM3DraftKVCacheView(_FakeManager(), [DRAFT_LAYER, DRAFT_LAYER + 1])


def test_view_rejects_incompatible_source_pool_kv_offset():
    manager = _FakeHybridManager()
    manager.kv_offset[1] += 1
    with pytest.raises(ValueError, match="block-table mapping is unavailable"):
        MiniMaxM3DraftKVCacheView(manager, [DRAFT_LAYER])


def test_view_rejects_swa_scratch_reuse():
    manager = _FakeManager()
    manager.enable_swa_scratch_reuse = True
    with pytest.raises(ValueError, match="SWA scratch reuse"):
        MiniMaxM3DraftKVCacheView(manager, [DRAFT_LAYER])


def test_draft_layout_target_only_num_layers():
    heads = [4] * 60 + [64]
    draft_ids, num_target = derive_shared_draft_layout(60, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_equal_head_drafter():
    draft_ids, num_target = derive_shared_draft_layout(60, [4] * 61, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_pre_extended_num_layers():
    heads = [4] * 60 + [64]
    draft_ids, num_target = derive_shared_draft_layout(61, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_no_draft():
    draft_ids, num_target = derive_shared_draft_layout(60, [4] * 60, 0)
    assert draft_ids == []
    assert num_target == 60
    assert derive_shared_draft_layout(60, 4, 0) == ([], 60)


def test_draft_layout_unpinned_range():
    assert derive_shared_draft_layout(None, 4, 1) == ([], None)

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

from contextlib import nullcontext

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3DraftKVCacheView,
    MiniMaxM3KVCacheManagerV2,
    _derive_shared_draft_layout,
)

DRAFT_LAYER = 60
SCALE = 178  # P128 pages per M3 mega-slot
ADDR = 0x7000_0000


class _FakeManager:
    tokens_per_block = 128
    max_blocks_per_seq = 16
    num_pools = 1
    num_attention_op_pools = 1
    index_scales = [SCALE]
    kv_offset = [1]
    _stream = object()

    def __init__(self):
        self.layer_offsets = {DRAFT_LAYER: DRAFT_LAYER}
        self.kv_cache_pool_mapping = torch.zeros((DRAFT_LAYER + 1, 2), dtype=torch.int32)
        self.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([0, 7], dtype=torch.int32)
        self.kv_cache_pool_pointers = torch.tensor([[ADDR - 1024, 0]], dtype=torch.int64)
        self.slot_rows = [[5, 7]]
        self.copy_args = None

    def _kv_slot_geometry(self, layer_idx, kv_layout):
        assert layer_idx == DRAFT_LAYER
        page_shape = [self.tokens_per_block, 16, 128]
        return ADDR, torch.int8, 1024, SCALE, page_shape

    def copy_batch_block_offsets(
        self,
        dst_tensor,
        request_ids,
        beam_width,
        num_contexts,
        num_seqs,
        max_blocks=None,
    ):
        self.copy_args = (
            request_ids,
            beam_width,
            num_contexts,
            num_seqs,
            max_blocks,
        )
        dst_tensor.zero_()
        for row_idx, slots in enumerate(self.slot_rows[:num_seqs]):
            for block_idx, slot in enumerate(slots):
                dst_tensor[0, row_idx, 0, block_idx] = slot * SCALE
                dst_tensor[0, row_idx, 1, block_idx] = slot * SCALE + 1


def _make_view():
    return MiniMaxM3DraftKVCacheView(_FakeManager(), [DRAFT_LAYER])


def test_view_geometry():
    view = _make_view()
    assert view.tokens_per_block == 128
    assert view._slot_stride == SCALE
    assert view.max_blocks_per_seq == 16
    assert view.num_pools == view.num_attention_op_pools == 1
    assert view.kv_cache_pool_pointers.tolist() == [[ADDR, 0]]
    assert view.kv_cache_pool_mapping[DRAFT_LAYER].tolist() == [0, 0]
    assert view.blocks_in_primary_pool == (1024 - 1) * SCALE + 2
    assert view.trtllm_gen_extra_tokens_per_block == frozenset({128})


def test_block_table_uses_native_p128_copy(monkeypatch):
    view = _make_view()
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
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
    assert dst[0, 0, 1, 2:].tolist() == [1] * 14
    assert view._manager.copy_args == ([123], 1, 1, 1, 9)


def test_free_resources_is_noop():
    _make_view().free_resources(object())


def test_manager_accessor_builds_and_caches_view():
    class _FakeSharedManager(_FakeManager):
        is_draft = False

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


def test_view_rejects_draft_layer_outside_pool_zero():
    manager = _FakeManager()
    manager.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([1, 0], dtype=torch.int32)
    with pytest.raises(ValueError, match="not in pool 0"):
        MiniMaxM3DraftKVCacheView(manager, [DRAFT_LAYER])


def test_draft_layout_target_only_num_layers():
    heads = [4] * 60 + [64]
    draft_ids, num_target = _derive_shared_draft_layout(60, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_equal_head_drafter():
    draft_ids, num_target = _derive_shared_draft_layout(60, [4] * 61, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_pre_extended_num_layers():
    heads = [4] * 60 + [64]
    draft_ids, num_target = _derive_shared_draft_layout(61, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_no_draft():
    draft_ids, num_target = _derive_shared_draft_layout(60, [4] * 60, 0)
    assert draft_ids == []
    assert num_target == 60
    assert _derive_shared_draft_layout(60, 4, 0) == ([], 60)


def test_draft_layout_unpinned_range():
    assert _derive_shared_draft_layout(None, 4, 1) == ([], None)

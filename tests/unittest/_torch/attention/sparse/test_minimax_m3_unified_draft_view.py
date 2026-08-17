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
"""Pure-logic tests for MiniMaxM3DraftSubpageView.

The view presents the shared manager's draft-layer pool to the attention ops
at a smaller kernel page size (32-token pages inside 128-token logical
blocks). These tests validate the addressing math against a fake manager, so
they run without GPUs: slot ``s`` of the drafter's layer must resolve to K
sub-pages ``s*scale*subdiv + j`` and V sub-pages offset by ``subdiv`` (V is
laid out immediately after K within the slot).
"""

import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3DraftSubpageView,
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
)

DRAFT_LAYER = 60
SCALE = 178  # sub-pages per mega-slot, in units of the drafter's 128-tok page
ADDR = 0x7000_0000


class _FakeManager:
    tokens_per_block = 128
    max_blocks_per_seq = 16
    num_pools = 1
    # V2's flattened bound uses the target pool's 128-token page units and
    # base pointer. The draft view must not delegate this value.
    blocks_in_primary_pool = 1024 * SCALE

    def __init__(self):
        self.layer_offsets = {DRAFT_LAYER: DRAFT_LAYER}
        self.kv_cache_pool_mapping = torch.zeros((DRAFT_LAYER + 1, 2), dtype=torch.int32)
        self.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([0, 7], dtype=torch.int32)
        self.slot_rows = [[5, 7]]

    def _kv_slot_geometry(self, layer_idx, kv_layout):
        assert layer_idx == DRAFT_LAYER
        page_shape = [self.tokens_per_block, 16, 128]
        return ADDR, torch.int8, 1024, SCALE, page_shape

    def _get_batch_cache_indices_by_pool_id(self, request_ids, *, pool_id):
        assert pool_id == 0
        return self.slot_rows[: len(request_ids)]


def _make_view():
    return MiniMaxM3DraftSubpageView(_FakeManager(), [DRAFT_LAYER], 32)


def test_view_geometry():
    view = _make_view()
    assert view.tokens_per_block == 32
    assert view._subdiv == 4
    assert view._slot_units == SCALE * 4
    assert view.max_blocks_per_seq == 16 * 4
    assert view.num_pools == view.num_attention_op_pools == 1
    assert view.kv_cache_pool_pointers.tolist() == [[ADDR, 0]]
    # The draft layer's mapping row is rewritten to the view's single pool.
    assert view.kv_cache_pool_mapping[DRAFT_LAYER].tolist() == [0, 0]
    # FlashInfer wraps the pool as a flat tensor rooted at the draft K
    # pointer. Its upper bound must use 32-token sub-page units and stop after
    # the last slot's V pages, not delegate the target manager's 128-token
    # page bound.
    assert view.blocks_in_primary_pool == (1024 - 1) * SCALE * 4 + 8


def test_block_table_expansion():
    view = _make_view()
    max_units = view.max_blocks_per_seq
    dst = torch.full((1, 1, 2, max_units), -7, dtype=torch.int32)
    view.copy_batch_block_offsets(dst, request_ids=[123], beam_width=1, num_contexts=1, num_seqs=1)
    unit = SCALE * 4
    expect_k = [5 * unit + j for j in range(4)] + [7 * unit + j for j in range(4)]
    expect_v = [v + 4 for v in expect_k]
    assert dst[0, 0, 0, :8].tolist() == expect_k
    assert dst[0, 0, 1, :8].tolist() == expect_v
    # Unallocated tail slots clamp to slot 0, so entries tile its sub-pages —
    # inert pads: kernels never read past the row's real block count (same
    # property as test_bad_page_index_padding_is_safe).
    assert dst[0, 0, 0, 8:].tolist() == [0, 1, 2, 3] * ((max_units - 8) // 4)
    assert dst[0, 0, 1, 8:].tolist() == [4, 5, 6, 7] * ((max_units - 8) // 4)


def test_block_table_source_is_private_per_call():
    # The H2D copy reads its source at execution time, so every call must get
    # its own staging buffer: a persistent one refilled in place would let the
    # next iteration clobber a still-pending copy and the drafter would index
    # another batch's blocks (nvbug 6293536).
    view = _make_view()
    first = view._host_block_table([[5, 7]], 1, 2, torch.int32)
    second = view._host_block_table([[9, 11]], 1, 2, torch.int32)
    assert first.data_ptr() != second.data_ptr()
    unit = SCALE * 4
    # The first table still holds its own batch after the second call.
    assert first[0, 0, :4].tolist() == [5 * unit + j for j in range(4)]
    assert second[0, 0, :4].tolist() == [9 * unit + j for j in range(4)]


def test_bad_page_index_padding_is_safe():
    view = _make_view()
    view._manager.slot_rows = [[5, -1]]
    dst = torch.zeros((1, 1, 2, view.max_blocks_per_seq), dtype=torch.int32)
    view.copy_batch_block_offsets(dst, request_ids=[1], beam_width=1, num_contexts=1, num_seqs=1)
    # BAD_PAGE_INDEX (-1) clamps to slot 0: pad entries index pages 0..subdiv,
    # never negative offsets.
    assert dst[0, 0, 0, 4:8].tolist() == [0, 1, 2, 3]
    assert (dst >= 0).all()


def test_free_resources_is_noop():
    view = _make_view()
    view.free_resources(object())  # must not raise nor touch the manager


def test_subdiv_one_degenerates_to_identity():
    # Retirement path (TRTLLM_M3_DRAFT_KV_TOKENS_PER_BLOCK=128): one
    # sub-page per logical block, so the table is K=slot*scale, V=K+1.
    view = MiniMaxM3DraftSubpageView(_FakeManager(), [DRAFT_LAYER], 128)
    assert view._subdiv == 1
    assert view.tokens_per_block == 128
    assert view.max_blocks_per_seq == 16
    assert view.blocks_in_primary_pool == (1024 - 1) * SCALE + 2
    dst = torch.zeros((1, 1, 2, view.max_blocks_per_seq), dtype=torch.int32)
    view.copy_batch_block_offsets(dst, request_ids=[1], beam_width=1, num_contexts=1, num_seqs=1)
    assert dst[0, 0, 0, :2].tolist() == [5 * SCALE, 7 * SCALE]
    assert dst[0, 0, 1, :2].tolist() == [5 * SCALE + 1, 7 * SCALE + 1]


def test_manager_accessor_builds_and_caches_view():
    # Exercise the accessor itself (construction + the log statement), not
    # just direct view construction: a stale field reference in the log
    # f-string once raised AttributeError here and silently disabled the
    # view.
    class _FakeSharedManager(_FakeManager):
        is_draft = False
        draft_manager_tokens_per_block = 32

        def __init__(self):
            super().__init__()
            self._shared_draft_layer_ids = [DRAFT_LAYER]
            self._draft_subpage_view_obj = None

    manager = _FakeSharedManager()
    get_view = MiniMaxM3KVCacheManagerV2.get_draft_subpage_view
    view = get_view(manager)
    assert isinstance(view, MiniMaxM3DraftSubpageView)
    assert view.tokens_per_block == 32
    assert get_view(manager) is view  # cached on second call

    manager_draft = _FakeSharedManager()
    manager_draft.is_draft = True
    assert get_view(manager_draft) is None


def test_view_rejects_draft_layer_outside_pool_zero():
    manager = _FakeManager()
    manager.kv_cache_pool_mapping[DRAFT_LAYER] = torch.tensor([1, 0], dtype=torch.int32)
    try:
        MiniMaxM3DraftSubpageView(manager, [DRAFT_LAYER], 32)
    except AssertionError as e:
        assert "pool 0" in str(e)
    else:
        raise AssertionError("expected the pool-0 sanity check to fire")


def test_draft_layout_target_only_num_layers():
    # The M3 creation-site flow: num_layers carries the pretrained target
    # count while the per-layer heads list is already draft-extended.
    # Anchoring the tail on num_layers instead of the list marked target
    # layer 59 as draft and dropped its index-K cache (crashed at startup).
    heads = [4] * 60 + [64]
    draft_ids, num_target = derive_shared_draft_layout(60, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_equal_head_drafter():
    # The GQA Eagle head has the target's KV head count, so the heads list
    # is uniform; the draft tail must still resolve from the list length
    # (an equal-head drafter is invisible in the values).
    draft_ids, num_target = derive_shared_draft_layout(60, [4] * 61, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_pre_extended_num_layers():
    # Flows that pass the extended count directly must resolve identically.
    heads = [4] * 60 + [64]
    draft_ids, num_target = derive_shared_draft_layout(61, heads, 1)
    assert draft_ids == [60]
    assert num_target == 60


def test_draft_layout_no_draft():
    draft_ids, num_target = derive_shared_draft_layout(60, [4] * 60, 0)
    assert draft_ids == []
    assert num_target == 60
    # Scalar heads (plain M3, no spec) fall back to num_layers.
    assert derive_shared_draft_layout(60, 4, 0) == ([], 60)


def test_draft_layout_unpinned_range():
    assert derive_shared_draft_layout(None, 4, 1) == ([], None)

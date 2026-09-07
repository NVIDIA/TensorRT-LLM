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

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    cache_manager as m3_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
    extend_attention_op_pools_for_shared_draft_layers,
    shared_draft_layer_count,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

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

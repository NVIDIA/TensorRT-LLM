# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType


class _FakeImpl:
    def __init__(self, pool_group_descs: list[SimpleNamespace]) -> None:
        self.pool_group_descs = pool_group_descs

    @staticmethod
    def get_mem_pool_base_address(layer_id: int, role: object) -> int:
        role_offset = 0 if role == Role.KEY else 100
        return layer_id * 1000 + role_offset

    @staticmethod
    def get_page_index_scale(layer_id: int, role: object) -> int:
        assert role == Role.KEY
        return layer_id + 10

    @staticmethod
    def get_page_stride(layer_id: int, role: object) -> int:
        del layer_id
        assert role == Role.KEY
        return 100


def _make_pool_group_desc(group_id: int, layer_order: list[int]) -> SimpleNamespace:
    buffer_ids = []
    for layer_id in layer_order:
        buffer_ids.extend(
            [
                SimpleNamespace(layer_id=layer_id, role=Role.KEY),
                SimpleNamespace(layer_id=layer_id, role=Role.VALUE),
                SimpleNamespace(layer_id=layer_id, role=Role.INDEX_KEY),
            ]
        )
    variant = SimpleNamespace(
        layer_group_id=group_id,
        coalesced_buffers=[SimpleNamespace(buffer_ids=buffer_ids)],
    )
    return SimpleNamespace(slot_desc=SimpleNamespace(variants=[variant]))


def test_prepare_page_table_uses_pool_descriptors_instead_of_group_order() -> None:
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.enable_swa_scratch_reuse = False
    manager.num_pools = 2
    manager.num_local_layers = 4
    manager.dtype = DataType.HALF
    manager.kv_cache_type = CacheType.SELF
    manager.max_beam_width = 1
    manager.max_blocks_per_seq = 4

    # Physical descriptors and buffers are deliberately not in group/layer-id order.
    manager.impl = _FakeImpl(
        [
            _make_pool_group_desc(1, [3, 2]),
            _make_pool_group_desc(0, [1, 0]),
        ]
    )

    with patch(
        "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager.prefer_pinned",
        return_value=False,
    ):
        manager._prepare_page_table_tensor(index_mapper_capacity=3)

    torch.testing.assert_close(
        manager.kv_cache_pool_pointers,
        torch.tensor([[1000, 0], [3000, 0]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        manager.kv_cache_pool_mapping,
        torch.tensor([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(manager.index_scales, torch.tensor([11, 13], dtype=torch.int32))
    torch.testing.assert_close(manager.kv_offset, torch.ones(2, dtype=torch.int32))
    assert manager.host_kv_cache_block_offsets.shape == (2, 3, 2, 4)

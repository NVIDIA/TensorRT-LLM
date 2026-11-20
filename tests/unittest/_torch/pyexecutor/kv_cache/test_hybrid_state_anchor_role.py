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
"""`MambaHybridCacheManagerV2` addresses state slots through `_state_anchor_role`.

The role defaults to the SSM state. A subclass whose recurrent layers keep other
buffers names one of them, and the manager then reads the slots' layer group, page
index scale and page count from that buffer.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MambaRole,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, LayerId

_WINDOW_ROLE = DataRole("test_window")
_WINDOW_SHAPE = [3, 64]
_SCRATCH_BYTES_PER_SLOT = 128
#: Layer 0 is recurrent, layer 1 is attention.
_RECURRENT_LAYER, _ATTENTION_LAYER = 0, 1


class _WindowStateManager(MambaHybridCacheManagerV2):
    """Recurrent layers that keep one window buffer and neither SSM nor convolution state."""

    _state_anchor_role = _WINDOW_ROLE

    def _get_recurrent_buffer_configs(self, global_layer_id: int) -> list[BufferConfig]:
        return [BufferConfig(role=_WINDOW_ROLE, size=math.prod(_WINDOW_SHAPE) * 2)]

    def _extra_scratch_bytes_per_slot(self) -> int:
        return _SCRATCH_BYTES_PER_SLOT

    def _setup_states(self) -> None:
        self.all_ssm_states = []
        self.all_conv_states = []
        self.windows = [
            self._get_state_buffer(
                self.layer_offsets[layer], _WINDOW_ROLE, torch.bfloat16, _WINDOW_SHAPE
            )
            for layer in self.mamba_pp_layers
        ]


def _build(manager_cls: type) -> MambaHybridCacheManagerV2:
    return manager_cls(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=KvCacheConfig(max_tokens=512),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        layer_mask=[False, True],
        dtype=DataType.HALF,
    )


@pytest.fixture
def manager():
    manager = _build(_WindowStateManager)
    yield manager
    manager.shutdown()


def test_the_default_anchor_is_the_ssm_state():
    assert MambaHybridCacheManagerV2._state_anchor_role == MambaRole.SSM_STATE


def test_the_state_slots_are_addressed_through_the_anchor_role(manager):
    layer = LayerId(manager.layer_offsets[_RECURRENT_LAYER])
    group = int(manager.impl.get_buffer_group_id(layer, _WINDOW_ROLE))
    scale = manager.impl.get_page_index_scale(layer, _WINDOW_ROLE)
    pages = manager.impl.get_page_index_upper_bound(layer, _WINDOW_ROLE)

    assert manager.ssm_layer_group_id == group
    assert manager._ssm_page_index_scale == scale
    assert manager._get_pool_roles(group) == (_WINDOW_ROLE, None)
    attention_layer = manager.layer_offsets[_ATTENTION_LAYER]
    assert manager._get_layer_execution_group(int(layer)) == group
    assert manager._get_layer_execution_group(attention_layer) != group
    assert manager.windows[0].shape[0] == (pages + scale - 1) // scale
    assert manager._get_external_cache_bytes(manager.impl) == (
        (pages + scale - 1) // scale * _SCRATCH_BYTES_PER_SLOT
    )
    assert manager.get_num_free_blocks() > 0


def test_a_request_gets_a_slot_of_the_anchor_group(manager):
    manager.add_dummy_requests([7], token_nums=[8], is_gen=False)
    kv_cache = manager.kv_cache_map[7]
    slot = kv_cache.get_ssm_block_base_index(manager.ssm_layer_group_id)
    assert manager.get_state_indices([7], [False]) == [slot]
    assert 0 <= slot < manager.windows[0].shape[0]

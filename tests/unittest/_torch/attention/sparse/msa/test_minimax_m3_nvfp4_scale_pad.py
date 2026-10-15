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
"""MiniMax-M3 NVFP4 scale-pool padding keeps data and scale page geometry equal.

V2 coalesces buffers by (life_cycle, size). With an FP8 128-dim index key, the
INDEX_KEY page matches a packed NVFP4 K page at two KV heads per rank, so it
joins the data pool; the scale pool is padded to keep one block table valid.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    NVFP4_SCALE_PAD_ROLE,
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping


def _make_manager(num_kv_heads: int) -> MiniMaxM3KVCacheManagerV2:
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=4,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        tokens_per_block=128,
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_nvfp4_scale_pool_mirrors_data_pool(num_kv_heads: int) -> None:
    torch.cuda.init()
    manager = _make_manager(num_kv_heads)
    try:
        sparse = [
            layer
            for layer in manager.sparse_layer_ids
            if layer in manager.layer_offsets and manager.is_nvfp4_layer(layer)
        ]
        assert sparse, "test config must contain NVFP4 sparse layers"

        # The pad is registered only where INDEX_KEY coalesces with packed K/V.
        extra = manager._extra_buffers_per_layer(tokens_per_block=manager.tokens_per_block)
        padded = {
            local
            for local, buffers in extra.items()
            if any(b.role == NVFP4_SCALE_PAD_ROLE for b in buffers)
        }
        expect_pad = num_kv_heads == 2
        assert bool(padded) == expect_pad

        for layer in sparse:
            data = manager._kv_slot_geometry(layer, "HND")
            scale = manager._kv_scale_slot_geometry(layer, "HND")
            # (slots, sub-pages per slot) must match for one shared block table.
            assert data[2:4] == scale[2:4], (layer, data[2:4], scale[2:4])
            assert manager.nvfp4_scale_pool_aligned(layer)
            _, scale_subpages = manager.get_kv_scale_subpage_pool(layer, "HND")
            _, data_subpages = manager.get_kv_subpage_pool(layer, "HND")
            assert scale_subpages == data_subpages
            if expect_pad:
                # K, V and the coalesced INDEX_KEY share the data slot.
                converter = manager.impl.get_page_index_converter(
                    manager.layer_offsets[layer], Role.INDEX_KEY
                )
                assert int(converter.scale) == data[3]
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
def test_nvfp4_scale_pad_is_not_transferred() -> None:
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    assert NVFP4_SCALE_PAD_ROLE in manager.get_disagg_ignored_roles()

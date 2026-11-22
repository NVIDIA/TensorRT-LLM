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
"""MiniMax-M3 KV pools start zeroed, and so do the slots a pool resize adds.

Readers that consume a whole block or page (the CSR sparse prefill's V dequant,
the full-context decode on a last page's tail) also read bytes no request has
written, so every pool view must hold zeros, not uninitialized memory, before
the first request.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PoolDesc

# The smallest unit in which a GPU KV cache pool maps device memory.
_MIN_ALLOCATION_UNIT = 2 << 20


def _nonzero_bytes(view: torch.Tensor) -> int:
    """Count nonzero bytes; a strided view that cannot be reinterpreted in place is copied first."""
    try:
        raw = view.view(torch.uint8)
    except RuntimeError:
        raw = view.contiguous().view(torch.uint8)
    return int(torch.count_nonzero(raw).item())


def _pool_bytes(pool: PoolDesc, slots: range) -> torch.Tensor:
    """Return the bytes of ``slots`` in ``pool``."""
    return convert_to_torch_tensor(
        TensorWrapper(
            pool.base_address + slots.start * pool.slot_bytes,
            torch.uint8,
            [len(slots) * pool.slot_bytes],
        )
    )


def _make_manager(with_draft: bool) -> MiniMaxM3KVCacheManagerV2:
    torch.cuda.init()
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=4,
        num_kv_heads=[2, 2, 2, 2, 1] if with_draft else [2, 2, 2, 2],
        head_dim=128,
        tokens_per_block=128,
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        spec_config=(
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused")
            if with_draft
            else None
        ),
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize("with_draft", [False, True], ids=["target_only", "shared_eagle3"])
def test_every_pool_view_is_zero_after_construction(with_draft: bool) -> None:
    manager = _make_manager(with_draft)
    try:
        views = {
            **{("kv",) + key: view for key, view in manager._kv_buffers.items()},
            **{("scale",) + key: view for key, view in manager._block_scale_buffers.items()},
            **{("index_k", key): view for key, view in manager._index_k_buffers.items()},
        }
        assert any(name[0] == "kv" for name in views)
        assert any(name[0] == "scale" for name in views)
        assert any(name[0] == "index_k" and view is not None for name, view in views.items())
        torch.cuda.synchronize()
        counts = {name: _nonzero_bytes(view) for name, view in views.items() if view is not None}
        nonzero = {name: count for name, count in counts.items() if count}
        assert not nonzero, f"pool views with nonzero bytes after construction: {nonzero}"
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_slots_added_by_a_resize_are_zero() -> None:
    manager = _make_manager(with_draft=False)
    try:
        (pool_group,) = manager.impl.pool_group_descs
        num_slots = pool_group.num_slots
        # A pool that fits in one allocation unit keeps it mapped through a shrink, so the slots the
        # shrink drops from it come back from the grow still holding the bytes written below.
        assert any(num_slots * pool.slot_bytes <= _MIN_ALLOCATION_UNIT for pool in pool_group.pools)
        for pool in pool_group.pools:
            _pool_bytes(pool, range(num_slots)).fill_(0xFF)
        quota = manager.gpu_quota
        manager.resize_gpu_quota(quota // 2)
        (pool_group,) = manager.impl.pool_group_descs
        kept = pool_group.num_slots
        manager.resize_gpu_quota(2 * quota)
        (pool_group,) = manager.impl.pool_group_descs
        assert kept < num_slots < pool_group.num_slots
        torch.cuda.synchronize()

        for pool in pool_group.pools:
            assert _pool_bytes(pool, range(kept)).eq(0xFF).all()
            assert _nonzero_bytes(_pool_bytes(pool, range(kept, pool_group.num_slots))) == 0
        for view in (
            *manager._kv_buffers.values(),
            *manager._block_scale_buffers.values(),
            *manager._index_k_buffers.values(),
        ):
            assert view.shape[0] == pool_group.num_slots
            assert any(
                pool.base_address
                <= view.data_ptr()
                < pool.base_address + pool_group.num_slots * pool.slot_bytes
                for pool in pool_group.pools
            )
            assert _nonzero_bytes(view[kept:]) == 0
    finally:
        manager.shutdown()

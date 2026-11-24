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
"""The DSA metadata's pool view of a layer's own physical K pool.

A DSA cache that keeps some layers' latent KV as FP8 pages beside NVFP4 ones (the Kimi K3.1 DSA
cache with FP8 KV layers) has one physical K pool per format. ``_ensure_pool_view_cached`` keys
its cache by (manager, pool address): a layer of the other pool recomputes it, the NVFP4 (int8)
pool has no token view, the FP8 pool a ``[tokens, 1, head_dim]`` one, and each pool counts its
own tokens. Plain tensors stand in for the pools, so this runs without a GPU.
"""

from types import MethodType, SimpleNamespace

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata

_TOKENS_PER_BLOCK = 64
_HEAD_DIM = 576
_SLOTS = 5
# Production page counts per slot: 22 NVFP4 layers and 2 FP8 layers.
_NVFP4_PAGES, _FP8_PAGES = 22, 2


def _metadata():
    """Metadata over a manager with an NVFP4 pool (layers 3, 47) and an FP8 pool (91, 92), and
    the layers whose pool the metadata asked for."""
    nvfp4_pool = torch.zeros(
        _SLOTS, _NVFP4_PAGES, 1, _TOKENS_PER_BLOCK * _HEAD_DIM // 2, dtype=torch.int8
    )
    fp8_pool = torch.zeros(
        _SLOTS, _FP8_PAGES, 1, _TOKENS_PER_BLOCK * _HEAD_DIM, dtype=torch.float8_e4m3fn
    )
    pools = {3: nvfp4_pool, 47: nvfp4_pool, 91: fp8_pool, 92: fp8_pool}
    asked = []

    def get_primary_pool(layer):
        asked.append(layer)
        return pools[layer]

    def get_unique_primary_pool():
        raise ValueError("Mixed Kimi DSA storage requires a layer-specific primary pool.")

    manager = SimpleNamespace(
        get_primary_pool=get_primary_pool,
        get_unique_primary_pool=get_unique_primary_pool,
        tokens_per_block=_TOKENS_PER_BLOCK,
        head_dim=_HEAD_DIM,
    )
    md = SimpleNamespace(
        kv_cache_manager=manager,
        _pool_cache_valid=False,
        _cached_pool_key=None,
        _cached_kv_mgr_id=0,
        _cached_pool_view=None,
        _cached_num_pool_tokens=0,
        _cached_tokens_per_block=0,
        # One context request (five tokens) and two generation requests (one token each).
        block_table=torch.arange(12, dtype=torch.int32).view(3, 4),
        num_contexts=1,
        num_seqs=3,
        num_ctx_tokens=5,
        num_tokens=7,
        req_idx_per_token=torch.tensor([0, 0, 0, 0, 0, 1, 2], dtype=torch.int32),
    )
    md._ensure_pool_view_cached = MethodType(
        DSAtrtllmAttentionMetadata._ensure_pool_view_cached, md
    )
    return md, nvfp4_pool, fp8_pool, asked


def test_each_physical_pool_has_its_own_view():
    md, nvfp4_pool, fp8_pool, _ = _metadata()
    md._ensure_pool_view_cached(3)
    assert md._cached_pool_view is None
    assert md._cached_num_pool_tokens == _SLOTS * _NVFP4_PAGES * _TOKENS_PER_BLOCK
    assert md._cached_tokens_per_block == _TOKENS_PER_BLOCK

    md._ensure_pool_view_cached(91)
    view = md._cached_pool_view
    assert view is not None and view.dtype == torch.float8_e4m3fn
    assert tuple(view.shape) == (_SLOTS * _FP8_PAGES * _TOKENS_PER_BLOCK, 1, _HEAD_DIM)
    assert view.data_ptr() == fp8_pool.data_ptr()
    assert md._cached_num_pool_tokens == _SLOTS * _FP8_PAGES * _TOKENS_PER_BLOCK

    md._ensure_pool_view_cached(47)
    assert md._cached_pool_view is None
    assert md._cached_num_pool_tokens == _SLOTS * _NVFP4_PAGES * _TOKENS_PER_BLOCK
    assert md._cached_pool_key == (id(md.kv_cache_manager), nvfp4_pool.data_ptr())

    # The batch slices do not depend on the pool.
    assert torch.equal(md._cached_block_table_ctx, md.block_table[:1])
    assert torch.equal(md._cached_block_table_gen, md.block_table[1:3])
    assert md._cached_req_idx_ctx.tolist() == [0] * 5
    assert md._cached_req_idx_gen.tolist() == [0, 1]


def test_a_layer_of_the_same_pool_keeps_the_cached_view():
    md, _, fp8_pool, asked = _metadata()
    md._ensure_pool_view_cached(91)
    sentinel = object()
    md._cached_pool_view = sentinel
    md._ensure_pool_view_cached(92)
    assert md._cached_pool_view is sentinel
    assert asked == [91, 92]

    # A step boundary recomputes it.
    md._pool_cache_valid = False
    md._ensure_pool_view_cached(92)
    assert md._cached_pool_view is not sentinel
    assert md._cached_pool_view.data_ptr() == fp8_pool.data_ptr()


def test_without_a_layer_the_unique_pool_is_asked():
    """Callers that pass no layer (single-pool DSA caches) keep asking for the unique pool, keyed by
    the manager alone and of the manager's KV dtype, as before per-layer pools."""
    md, nvfp4_pool, _, asked = _metadata()
    md.kv_cache_manager.get_unique_primary_pool = lambda: nvfp4_pool
    md.kv_cache_manager.dtype = tensorrt_llm.bindings.DataType.NVFP4
    md._ensure_pool_view_cached()
    assert asked == []
    assert md._cached_pool_view is None
    assert md._cached_pool_key == (id(md.kv_cache_manager), None)

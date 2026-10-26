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
"""MiniMax-M3 KV pools start zeroed.

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
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping


def _nonzero_bytes(view: torch.Tensor) -> int:
    """Count nonzero bytes; a strided view that cannot be reinterpreted in place is copied first."""
    try:
        raw = view.view(torch.uint8)
    except RuntimeError:
        raw = view.contiguous().view(torch.uint8)
    return int(torch.count_nonzero(raw).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize("with_draft", [False, True], ids=["target_only", "shared_eagle3"])
def test_every_pool_view_is_zero_after_construction(with_draft: bool) -> None:
    torch.cuda.init()
    manager = MiniMaxM3KVCacheManagerV2(
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

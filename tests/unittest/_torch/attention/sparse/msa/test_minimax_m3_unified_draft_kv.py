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
"""A standalone DSpark drafter's layers as explicit draft layer specs of the MiniMax-M3 manager."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping

NUM_TARGET = 4  # M3 convention without a checkpoint: layers 0..2 dense, 3 sparse
NUM_DRAFT = 2
DRAFT_WINDOW = 256
HEADS = 2
HEAD_DIM = 128


def _specs() -> tuple:
    """Target layers in the NVFP4 cache dtype, then a windowed BF16 drafter."""
    scale_roles = (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
    target = [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.NVFP4,
            num_kv_heads=HEADS,
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE, *scale_roles),
        )
        for layer in range(NUM_TARGET)
    ]
    draft = [
        KVCacheLayerSpec(
            global_layer_id=NUM_TARGET + layer,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=DataType.BF16,
            num_kv_heads=HEADS,
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=DRAFT_WINDOW,
        )
        for layer in range(NUM_DRAFT)
    ]
    return tuple(target + draft)


def _manager(layer_specs) -> MiniMaxM3KVCacheManagerV2:
    """A hybrid NVFP4 MSA manager, with or without explicit draft layer specs."""
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
        layer_specs=layer_specs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_draft_layer_specs_join_the_hybrid_nvfp4_manager() -> None:
    """Draft layers join the hybrid manager without changing the target's pages."""
    torch.cuda.init()
    plain = _manager(None)
    unified = _manager(_specs())
    try:
        assert unified.target_layer_ids == tuple(range(NUM_TARGET))
        assert unified.draft_layer_ids == tuple(range(NUM_TARGET, NUM_TARGET + NUM_DRAFT))
        # Dense target layers keep the FP8 half of the hybrid layout; draft layers their own dtype.
        dtypes = [unified.get_layer_dtype(layer) for layer in unified.pp_layers]
        assert dtypes == [DataType.FP8] * 3 + [DataType.NVFP4] + [DataType.BF16] * NUM_DRAFT
        for layer in range(NUM_TARGET):
            for role in (Role.KEY, Role.VALUE):
                assert unified.impl.get_page_stride(
                    unified.layer_offsets[layer], role
                ) == plain.impl.get_page_stride(plain.layer_offsets[layer], role)
        assert unified.get_index_k_buffer(3).shape[1:] == plain.get_index_k_buffer(3).shape[1:]

        # Draft layers: their own lifecycle, the base view in their dtype.
        groups = unified.layer_to_pool_mapping_dict
        draft_groups = {groups[unified.layer_offsets[layer]] for layer in unified.draft_layer_ids}
        target_groups = {groups[unified.layer_offsets[layer]] for layer in unified.target_layer_ids}
        assert len(draft_groups) == 1 and not draft_groups & target_groups
        view = unified.get_buffers(NUM_TARGET, kv_layout="HND")
        assert view.dtype == torch.bfloat16
        assert tuple(view.shape[1:]) == (2, HEADS, 128, HEAD_DIM)
        assert unified.get_draft_kv_cache_view() is unified
        assert plain.get_draft_kv_cache_view() is None

        # The unified drafter reads its layers' block offsets, so they keep attention-op pools.
        assert plain.num_attention_op_pools == 0
        assert unified.num_attention_op_pools > 0
        (request,) = unified.add_dummy_requests([7], token_nums=[200])
        offsets = torch.zeros(
            unified.num_attention_op_pools,
            1,
            2,
            unified.max_blocks_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
        unified.copy_batch_block_offsets(offsets, [7], 1, 1, 1)
        torch.cuda.synchronize()
        pool = int(unified.kv_cache_pool_mapping[unified.layer_offsets[NUM_TARGET], 0])
        slots = [s for s in unified.get_batch_cache_indices([7], layer_idx=NUM_TARGET)[0] if s >= 0]
        assert len(slots) == 2
        # The row the drafter reads holds the draft layer's K page of each slot, V right after it.
        scale = unified.get_layer_page_index_scale(NUM_TARGET)
        pages = [[slot * scale + plane for slot in slots] for plane in (0, 1)]
        assert offsets[pool, 0, :, : len(slots)].tolist() == pages
        unified.free_resources(request)

        # The draft lifecycle is sized by its window, not by its share of the bytes per token.
        target_ratio, draft_ratio = unified.kv_cache_manager_py_config.initial_pool_ratio
        assert target_ratio + draft_ratio == pytest.approx(1.0)
        assert (
            0
            < draft_ratio
            < 2 * HEADS * HEAD_DIM * 2 * NUM_DRAFT / (unified.get_cache_bytes_per_token())
        )
    finally:
        unified.shutdown()
        plain.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize("is_disagg", [False, True])
def test_window_sized_draft_ratio(is_disagg: bool) -> None:
    """A windowed draft lifecycle gets its windows plus one chunk, not a bytes-per-token share."""
    manager = object.__new__(KVCacheManagerV2)
    manager.layer_specs = _specs()[NUM_TARGET - 1 :]
    manager.max_batch_size = 2
    manager.max_num_tokens = 256
    manager.mapping = Mapping()
    manager.is_disagg = is_disagg
    manager._generation_kv_capacity_headroom = 1
    configs = [
        SimpleNamespace(
            sliding_window_size=spec.sliding_window_size,
            num_sink_tokens=None,
            reuse_policy=spec.reuse_policy,
            buffers=[SimpleNamespace(size=spec.bytes_per_token() * 128)],
        )
        for spec in manager.layer_specs
    ]
    draft_bytes_per_token = 2 * HEADS * HEAD_DIM * 2 * NUM_DRAFT
    # Four admissible sequences hold three pages of window each, plus one 256-token chunk.
    draft_bytes = (4 * 3 * 128 + 256) * draft_bytes_per_token
    assert manager._windowed_draft_pool_ratio(configs, 1 << 30, 128) == pytest.approx(
        [1 - draft_bytes / (1 << 30), draft_bytes / (1 << 30)]
    )
    # A quota too small for the windows keeps the bytes-per-token split.
    token_share = draft_bytes_per_token / sum(
        spec.bytes_per_token() for spec in manager.layer_specs
    )
    assert manager._windowed_draft_pool_ratio(configs, 1 << 20, 128)[1] == pytest.approx(
        token_share
    )
    # Without a windowed draft lifecycle the backend keeps its own split.
    assert manager._windowed_draft_pool_ratio(configs[:1], 1 << 30, 128) is None

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
"""The family KDA + DSA hybrid cache manager of the Kimi K3.1 Rubin targets: indexer K-cache
views and draft layers."""

import torch

__extra_import_path__ = [".", "../.."]
from _k31_utils import on_sm
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    dsa_sparse_config,
    dsa_text_config,
)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MambaStateConfig
from tensorrt_llm.mapping import Mapping

# sm_107 only: l0_gb300 collects ops/ as a directory.
pytestmark = on_sm(107)

_INDEX_HEAD_DIM = 128
# FP8 keys plus one fp32 scale per 128-wide block.
_INDEXER_BYTES_PER_TOKEN = _INDEX_HEAD_DIM + 4
_TOKENS_PER_BLOCK = 32
# Layers 1 and 3 are DSA MLA (1-indexed 2 and 4); layers 0 and 2 are KDA.
_MLA_MASK = [False, True, False, True]


_MLA_HEAD_DIM = 64


def _make_manager(
    indexer_k_dtype: str = "fp8",
    dtype: DataType = DataType.BF16,
    head_dim: int = _MLA_HEAD_DIM,
    enable_block_reuse: bool = False,
    num_mla_layers: int = 2,
    **draft_kwargs,
) -> RubinKdaDsaHybridCacheManager:
    """``num_mla_layers`` KDA + DSA layer pairs (KDA first), with partial reuse and a snapshot at
    the end of each prompt."""
    return _make_dsa_manager(
        indexer_k_dtype,
        dtype,
        mla_layers=tuple(range(2, 2 * num_mla_layers + 1, 2)),
        num_hidden_layers=2 * num_mla_layers,
        head_dim=head_dim,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
        ),
        **draft_kwargs,
    )


def test_kimi_dsa_hybrid_manager_with_standalone_draft_layers() -> None:
    # A unified standalone MLA drafter (DSpark) stores one BF16 latent per token per layer and is
    # registered as model layers; the DSA views must still cover the two target MLA layers only.
    num_draft_layers = 3
    manager = _make_manager(
        draft_num_layers=num_draft_layers,
        draft_num_kv_heads=1,
        draft_head_dim=_MLA_HEAD_DIM,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_scratch_tokens=8,
        draft_window_size=128,
    )
    try:
        assert len(manager.draft_layer_ids) == num_draft_layers
        draft_local = {manager.layer_offsets[layer] for layer in manager.draft_layer_ids}
        assert draft_local.isdisjoint(manager._attention_local_layers)
        assert manager._attention_local_layers == [1, 3]
        pool = manager.get_unique_primary_pool()
        assert pool.shape[1] == sum(_MLA_MASK)
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM)
        assert manager.indexer_k_cache_local_layer_mask == _MLA_MASK + [False] * num_draft_layers

        # Per token: a BF16 latent and the index K for each MLA layer, a BF16 latent per draft layer.
        mla_bytes = sum(_MLA_MASK) * (2 * _MLA_HEAD_DIM + _INDEXER_BYTES_PER_TOKEN)
        draft_bytes = num_draft_layers * 2 * _MLA_HEAD_DIM
        assert manager._attention_cache_bytes_per_token() == mla_bytes + draft_bytes
        for layer_idx in (1, 3):
            assert manager.get_indexer_k_cache_buffers(layer_idx).shape[1:] == (
                _TOKENS_PER_BLOCK,
                1,
                _INDEXER_BYTES_PER_TOKEN,
            )
    finally:
        manager.shutdown()


def _make_dsa_manager(
    indexer_k_dtype: str = "fp8",
    dtype: DataType = DataType.FP8,
    *,
    mla_layers: tuple,
    num_hidden_layers: int,
    head_dim: int = _MLA_HEAD_DIM,
    kv_lora_rank: int = 32,
    qk_rope_head_dim: int = 32,
    tokens_per_block: int = _TOKENS_PER_BLOCK,
    max_tokens: int = 512,
    max_seq_len: int = 256,
    max_batch_size: int = 2,
    host_cache_size: int | None = None,
    kv_cache_config: KvCacheConfig | None = None,
    mamba_ssm_cache_dtype: torch.dtype = torch.float32,
    **draft_kwargs,
) -> RubinKdaDsaHybridCacheManager:
    # The MLA layers (1-indexed) are the DSA layers; every other layer is KDA.
    mla_mask = [layer + 1 in mla_layers for layer in range(num_hidden_layers)]
    kda_mask = [not is_mla for is_mla in mla_mask]
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(
            max_tokens=max_tokens, enable_block_reuse=False, host_cache_size=host_cache_size
        )
    return RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=sum(kda_mask),
        mamba_layer_mask=kda_mask,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=sum(mla_mask),
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        layer_mask=mla_mask,
        sparse_attention_config=dsa_sparse_config(indexer_k_dtype),
        pretrained_config=dsa_text_config(
            mla_layers, num_hidden_layers, kv_lora_rank, qk_rope_head_dim
        ),
        **draft_kwargs,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA sparse KV cache offload configuration and its lowering."""

import pytest
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
    SparseKvCacheOffloadConfig,
)

_OFFLOAD = {"gpu_cache_factor": 3.0, "host_history_size": 2 * 1024**3}


def _glm_config() -> PretrainedConfig:
    return PretrainedConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=7,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_rope_interleave=True,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_dsa_offload_lowers_into_sparse_params(enabled):
    config = DeepSeekSparseAttentionConfig(
        index_topk=2048,
        enable_kv_cache_offload=enabled,
        kv_cache_offload_config=_OFFLOAD if enabled else {},
    )
    expected = SparseKvCacheOffloadConfig(**_OFFLOAD) if enabled else None
    assert config.get_kv_cache_offload_config() == expected
    assert config.to_sparse_params().enable_kv_cache_offload is enabled
    assert config.to_sparse_metadata_params().enable_kv_cache_offload is enabled


def test_dsa_offload_sizing_requires_offload():
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        DeepSeekSparseAttentionConfig(kv_cache_offload_config=_OFFLOAD)


def test_dsa_offload_defers_the_top_k_check_to_the_checkpoint():
    # index_topk comes from the checkpoint when the user leaves it unset.
    assert DeepSeekSparseAttentionConfig(enable_kv_cache_offload=True).index_topk is None
    with pytest.raises(ValueError, match="positive index_topk"):
        DeepSeekSparseAttentionConfig(enable_kv_cache_offload=True, index_topk=0)


def test_deepseek_v4_offload_keeps_its_ratio_check():
    with pytest.raises(ValueError, match="ratio-4"):
        DeepSeekV4SparseAttentionConfig(compress_ratios=[1, 128], enable_kv_cache_offload=True)
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        DeepSeekV4SparseAttentionConfig(kv_cache_offload_config=_OFFLOAD)


@pytest.mark.parametrize("enabled", [False, True])
def test_glm_model_config_preserves_sparse_offload(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config",
        lambda *args, **kwargs: _glm_config(),
    )
    sparse_config = DeepSeekSparseAttentionConfig(
        enable_kv_cache_offload=enabled,
        kv_cache_offload_config=_OFFLOAD if enabled else {},
    )

    resolved = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    ).sparse_attention_config

    assert resolved.enable_kv_cache_offload is enabled
    assert resolved.kv_cache_offload_config == sparse_config.kv_cache_offload_config
    # The checkpoint fills the selection width the offload cache is sized by.
    assert resolved.index_topk == 2048


@pytest.mark.parametrize(
    ("indexer_mask", "pp_layers", "expected"),
    [
        # GLM-5.2's pattern over seven layers: full indexers at 0, 1, 2 and 6, so
        # layers 3-5 reuse layer 2's top-k and nothing else is shared.
        (
            [True, True, True, False, False, False, True],
            None,
            {0: None, 1: None, 2: 2, 3: 2, 4: 2, 5: 2, 6: None},
        ),
        # Every layer owning an indexer shares nothing, which is what keeps the
        # fused fetch path for models without indexer sharing.
        ([True] * 4, None, {0: None, 1: None, 2: None, 3: None}),
        # A pipeline rank holding global layers 8-13, full at 8 and 11. Leaders are
        # model layers, not local offsets.
        (
            [False] * 8 + [True, False, False, True, False, False],
            list(range(8, 14)),
            {8: 8, 9: 8, 10: 8, 11: 11, 12: 11, 13: 11},
        ),
        # A rank whose first layer owns no indexer: its top-k comes from a layer on
        # another rank, so the run has to start here anyway. The remap path's own
        # scan starts a group at local zero for the same reason.
        (
            [False] * 10 + [True, False],
            list(range(8, 12)),
            {8: 8, 9: 8, 10: 10, 11: 10},
        ),
    ],
    ids=["glm-5.2", "no-sharing", "pipelined", "rank-starts-shared"],
)
def test_selection_leaders_follow_the_indexer_runs(indexer_mask, pp_layers, expected):
    """Which layers are handed one selection, and so share one resident set.

    A run starts at a layer owning an indexer; the layers after it reuse its top-k.
    A run of one names no selection -- a layer that selects for itself never shares.
    """
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2

    manager = SimpleNamespace(
        pp_layers=pp_layers if pp_layers is not None else list(range(len(indexer_mask))),
        _indexer_k_cache_layer_mask=indexer_mask,
        _selection_leader_cache=None,
    )

    assert DSACacheManagerV2._selection_leaders(manager) == expected

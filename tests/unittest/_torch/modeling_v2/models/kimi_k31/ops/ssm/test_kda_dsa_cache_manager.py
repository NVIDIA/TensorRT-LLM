# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA + DSA cache manager: the KV pool log line and the manager selection."""

import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KV_POOL_LOG_EVERY_ENV,
    KimiK3DSAHybridCacheManager,
    select_k31_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_TOKENS_PER_BLOCK = 32
#: Latent width of one MLA token: kv_lora_rank + qk_rope_head_dim.
_LATENT_DIM = 576
_KDA_MASK = [True, False, True, False]
_MLA_MASK = [False, True, False, True]


def _sparse_config(indexer_k_dtype: str) -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_n_heads=4,
        index_head_dim=128,
        index_topk=64,
        indexer_k_dtype=indexer_k_dtype,
    )


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _make(cls, dtype: DataType, indexer_k_dtype: str = "fp8", head_dim: int = _LATENT_DIM, **extra):
    return cls(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        layer_mask=_MLA_MASK,
        sparse_attention_config=_sparse_config(indexer_k_dtype),
        pretrained_config=_text_config(),
        **extra,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "executor_stats", [False, True], ids=["log-reads-counters", "executor-stats"]
)
def test_kv_pool_log_line(monkeypatch, executor_stats):
    """TLLM_KV_POOL_LOG_EVERY turns on the [kv_pool] line: per-level pool counts, and the native reuse /
    onboard / offload / snapshot counters and the reused prompt tokens by cache level when the executor
    does not read them (iteration stats off)."""
    monkeypatch.setenv(KV_POOL_LOG_EVERY_ENV, "3")
    manager = _make(KimiK3DSAHybridCacheManager, DataType.FP8, enable_stats=executor_stats)
    try:
        assert manager._kv_pool_log_every == 3
        assert manager.enable_stats
        assert manager._kv_pool_log_reads_counters == (not executor_stats)
        line = manager.format_kv_pool_stats(3)
        assert line.startswith("[kv_pool] iter=3 live=0 L0[pg0:"), line
        assert "d(new/reused/missed/onboard/offload/hostdrop)=" in line
        if executor_stats:
            assert line.endswith("d(new/reused/missed/onboard/offload/hostdrop)="), line
        else:
            # No request was committed, so no level has reused tokens yet.
            assert re.search(r" cached_tok_by_level=0(/0)*$", line), line
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kv_pool_log_is_off_by_default(monkeypatch):
    monkeypatch.delenv(KV_POOL_LOG_EVERY_ENV, raising=False)
    manager = _make(KimiK3DSAHybridCacheManager, DataType.FP8)
    try:
        assert manager._kv_pool_log_every == 0
        assert not manager._kv_pool_log_reads_counters
        assert not manager.enable_stats
    finally:
        manager.shutdown()


def _model_config(kv_algo):
    return SimpleNamespace(
        sparse_attention_config=_sparse_config("fp8"),
        quant_config=QuantConfig(kv_cache_quant_algo=kv_algo),
    )


def test_selection_refuses_an_nvfp4_kv_cache():
    kv = KvCacheConfig(use_kv_cache_manager_v2=True)
    with pytest.raises(ValueError, match="NVFP4 KV cache"):
        select_k31_kv_cache_manager_cls(_model_config(QuantAlgo.NVFP4), kv)
    assert (
        select_k31_kv_cache_manager_cls(_model_config(QuantAlgo.FP8), kv)
        is KimiK3DSAHybridCacheManager
    )

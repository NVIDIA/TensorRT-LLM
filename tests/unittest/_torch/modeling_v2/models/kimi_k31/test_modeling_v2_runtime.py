# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine hooks the Kimi K3.1 targets declare (``runtime.py``) and the count-first attention-DP admission."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.runtime import (
    K31_RUNTIME_HOOKS,
    K31CountFirstAdmission,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _kimi_dsa_model_config():
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        linear_attn_config={"kda_layers": [1, 3], "full_attn_layers": [2, 4]},
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        pretrained_config=config,
        mapping=Mapping(),
        quant_config=None,
        sparse_attention_config=SimpleNamespace(algorithm="dsa"),
        spec_config=None,
        get_num_mamba_layers=lambda: 2,
    )


def test_dsa_on_the_hybrid_model_needs_the_declared_manager(monkeypatch):
    """The engine resolves its Kimi hybrid manager for DSA only for a model that declares its own manager, which
    the targets' (the family's) is a subclass of."""
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    resolved = get_kv_cache_manager_cls(
        _kimi_dsa_model_config(), kv_cache_config, model_declares_manager=True
    )
    assert resolved is KimiK3HybridCacheManager
    assert issubclass(K31_RUNTIME_HOOKS.kv_cache_manager_cls, resolved)
    with pytest.raises(ValueError, match="'dsa' is not supported with hybrid"):
        get_kv_cache_manager_cls(_kimi_dsa_model_config(), kv_cache_config)
    with pytest.raises(
        ValueError, match="DSA with hybrid .* requires use_kv_cache_manager_v2=True"
    ):
        get_kv_cache_manager_cls(
            _kimi_dsa_model_config(),
            KvCacheConfig(use_kv_cache_manager_v2=False),
            model_declares_manager=True,
        )


def _plan(**overrides):
    kwargs = dict(
        candidate_tokens=[10_000],
        candidate_forced=[False],
        rank_tokens=[100_000, 110_000],
        rank_num_requests=[8, 2],
        max_requests_per_rank=16,
        max_new_requests=1,
        candidate_groups=[0],
    )
    kwargs.update(overrides)
    return K31CountFirstAdmission().plan(**kwargs)


def test_fewest_requests_within_the_kv_slack():
    # Rank 0: light KV, many requests; rank 1: more KV, few requests.
    assert _plan() == [(0, 1)]
    # Past the slack the request goes to the least-KV rank.
    assert _plan(rank_tokens=[100_000, 300_000]) == [(0, 0)]


def test_partial_window_admits_forced_then_queue_order():
    plan = _plan(
        candidate_tokens=[1_000, 2_000, 3_000],
        candidate_forced=[False, False, True],
        rank_tokens=[0, 0],
        rank_num_requests=[0, 0],
        max_requests_per_rank=1,
        max_new_requests=2,
        candidate_groups=[0, 0, 0],
    )
    assert [idx for idx, _ in plan] == [2, 0]
    assert sorted(rank for _, rank in plan) == [0, 1]


def test_whole_window_is_placed_largest_first():
    plan = _plan(
        candidate_tokens=[1_000, 3_000, 2_000],
        candidate_forced=[False, False, False],
        rank_tokens=[0, 0],
        rank_num_requests=[0, 0],
        max_new_requests=8,
        candidate_groups=[0, 0, 0],
    )
    assert [idx for idx, _ in plan] == [1, 2, 0]

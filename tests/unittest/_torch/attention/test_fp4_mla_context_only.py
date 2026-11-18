# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-copy FP4 MLA cache of a disaggregated context worker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
    Fp4MlaKVCacheManagerV2,
    is_fp4_mla_context_only_cache,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
)
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import dummy_requests
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    ResourceManagerType,
    is_context_only_kv_cache,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


@pytest.mark.parametrize("role", ["context", "generation", None])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_layout_and_estimate_follow_worker_role(monkeypatch, role, is_disagg, fused):
    if role is None:
        monkeypatch.delenv(DISAGG_ROLE_ENV, raising=False)
    else:
        monkeypatch.setenv(DISAGG_ROLE_ENV, role)
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused)))
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4", enable_block_reuse=True),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=576,
        pretrained_config=config,
        is_disagg=is_disagg,
    )
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK

    # Only a disaggregated context worker omits the generation-only sidecars.
    context_only = is_disagg and role == "context"
    assert manager.fp4_mla_context_only_cache == context_only
    assert is_fp4_mla_context_only_cache(is_disagg) == context_only
    assert is_context_only_kv_cache(manager) == context_only
    assert manager.mla_v_head_dim == (None if context_only or fused else 512)
    assert (manager._v_scale_bytes_per_page() == 0) == context_only
    if context_only:
        assert manager.get_mla_v_scale_pool() is None
        assert manager.get_mla_v_scale_pool_base() is None

    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(pretrained_config=config),
        Mapping(world_size=1, tp_size=1, rank=0),
        num_layers=1,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=2,
        is_disagg=is_disagg,
    )
    # K (320 B) and K scales (40 B) per token; V scales add 32 B and the
    # unfused CuTe DSL packed V another 256 B.
    assert slope == (360 if context_only else 392 if fused else 648)
    # The HP ring stays: the context worker writes and transfers it.
    assert intercept == 2 * 16 * 576 * 2


def test_context_only_layer_bytes_exclude_v_scales(monkeypatch):
    monkeypatch.setenv(DISAGG_ROLE_ENV, "context")
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4"),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=576,
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        is_disagg=True,
    )
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    manager.num_local_layers = 1
    manager.pp_layers = [0]
    manager.head_dim_per_layer = [576]
    manager.num_kv_heads_per_layer = [1]

    assert manager.get_layer_bytes_per_token(0, Role.ALL) == 320 + 40
    # No V-scale (or packed-V) buffer is registered next to K and K scales.
    assert manager._extra_buffers_per_layer(tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK) == {0: []}


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({}, False),
        ({"kda_context_only": True}, True),
        ({"fp4_mla_context_only_cache": True}, True),
        ({"kda_context_only": False, "fp4_mla_context_only_cache": False}, False),
    ],
)
def test_context_only_kv_cache_predicate(attrs, expected):
    assert is_context_only_kv_cache(SimpleNamespace(**attrs)) is expected
    # A Mock manager answers any attribute with a truthy Mock; only True counts.
    assert is_context_only_kv_cache(Mock()) is False


def _dummy_requests(monkeypatch, context_only: bool):
    cache = SimpleNamespace(fp4_mla_context_only_cache=context_only, context_swa_layer_limit=None)
    resources = SimpleNamespace(
        get_resource_manager=lambda key: cache
        if key == ResourceManagerType.KV_CACHE_MANAGER
        else None
    )
    factory = object.__new__(DummyRequestFactory)
    factory.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    factory._config = SimpleNamespace(spec_config=None)
    # Stop as soon as the factory starts building a batch.
    monkeypatch.setattr(
        dummy_requests,
        "get_draft_kv_cache_manager",
        Mock(side_effect=RuntimeError("warmup batch reached")),
    )
    return factory, resources


@pytest.mark.parametrize("context_only", [False, True])
def test_only_context_workers_skip_generation_warmup(monkeypatch, context_only):
    factory, resources = _dummy_requests(monkeypatch, context_only)

    if context_only:
        assert factory.create_warmup_request(resources, 8, 1) is None
        assert factory.create_cuda_graph_warmup_request(resources, 1, 0) is None
    else:
        with pytest.raises(RuntimeError, match="warmup batch reached"):
            factory.create_warmup_request(resources, 8, 1)
        with pytest.raises(RuntimeError, match="warmup batch reached"):
            factory.create_cuda_graph_warmup_request(resources, 1, 0)


def test_context_warmup_is_preserved_on_context_only_cache(monkeypatch):
    factory, resources = _dummy_requests(monkeypatch, context_only=True)

    with pytest.raises(RuntimeError, match="warmup batch reached"):
        factory.create_warmup_request(resources, 8192, 0)

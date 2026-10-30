# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-copy FP4 MLA cache of a disaggregated context worker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
    Fp4MlaKVCacheManagerV2,
    Fp4MlaV2CacheLayoutPolicy,
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
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
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


def test_hybrid_composition_keeps_single_copy_on_every_layer(monkeypatch):
    """A hybrid linear-attention manager (Kimi K3) composing the layout policy
    keeps the single-copy layout on every MLA layer, with or without a
    one-model drafter: draft-layer decode views apply to the pure MLA manager
    only."""
    monkeypatch.setenv(DISAGG_ROLE_ENV, "context")
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    manager = SimpleNamespace()
    Fp4MlaV2CacheLayoutPolicy.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4"),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=576,
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        spec_config=MTPDecodingConfig(max_draft_len=3),
        is_disagg=True,
    )
    assert manager.fp4_mla_context_only_cache
    assert manager._fp4_mla_num_spec_layers == 0
    assert manager.mla_v_head_dim is None
    assert manager._v_scale_bytes_per_page() == 0
    assert manager.fp4_mla_draft_lifecycle_layer_ids() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_context_only_draft_layers_keep_decode_views(monkeypatch):
    """One-model draft layers of a single-copy cache keep V scales and packed V.

    The drafter runs its later draft steps as generation steps inside every
    context step. Its layer keeps all four roles in a lifecycle of its own,
    with a page table and an FP4 state of its own, while the target layers
    keep K and K scales only.
    """
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_update import (
        _fp4_mla_layer_has_decode_views,
    )
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

    monkeypatch.setenv(DISAGG_ROLE_ENV, "context")
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    num_target_layers = 3
    manager = Fp4MlaKVCacheManagerV2(
        KvCacheConfig(
            max_tokens=16 * FP4_MLA_TOKENS_PER_BLOCK,
            dtype="nvfp4",
            enable_block_reuse=True,
            host_cache_size=0,
        ),
        CacheTypeCpp.SELFKONLY,
        num_layers=num_target_layers,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=4 * FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.NVFP4,
        spec_config=MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1),
        max_num_tokens=16 * FP4_MLA_TOKENS_PER_BLOCK,
        pretrained_config=SimpleNamespace(kv_lora_rank=512),
        is_disagg=True,
    )
    draft = num_target_layers
    try:
        assert manager.fp4_mla_context_only_cache
        assert manager.pp_layers == list(range(num_target_layers + 1))
        # K (320 B) and K scales (40 B) on the target layers; the draft layer
        # also keeps V scales (32 B) and packed V (256 B).
        assert [
            manager.get_layer_bytes_per_token(layer, Role.ALL) for layer in range(draft + 1)
        ] == [360] * draft + [648]
        assert manager.fp4_mla_draft_lifecycle_layer_ids() == [draft]
        target_group = int(manager.impl.get_layer_group_id(LayerId(0)))
        draft_group = int(manager.impl.get_layer_group_id(LayerId(draft)))
        hp_group = int(manager.impl.get_layer_group_id(manager._hp_manager_layer_ids[0]))
        assert len({target_group, draft_group, hp_group}) == 3
        assert manager.get_fp4_mla_page_table_spec(0).cache_pool_id == target_group
        assert manager.get_fp4_mla_page_table_spec(draft).cache_pool_id == draft_group
        # The V-scale pool covers the draft layer only, from its compact index on.
        assert manager.fp4_mla_v_layer_base == draft
        assert manager.get_mla_v_scale_pool().shape[0] == 1
        assert manager.get_mla_v_packed_pool(draft) is not None
        assert _fp4_mla_layer_has_decode_views(manager, draft)
        assert not _fp4_mla_layer_has_decode_views(manager, 0)
        with pytest.raises(RuntimeError, match="cannot execute generation"):
            manager.get_mla_v_scale_page_offset(0)
        with pytest.raises(RuntimeError, match="cannot execute generation"):
            manager.get_mla_v_packed_pool(0)

        seq_lens = [100, 50]
        manager.add_dummy_requests([0, 1], seq_lens)
        metadata = TrtllmAttentionMetadata(
            max_num_requests=len(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            runtime_features=AttentionRuntimeFeatures(),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            num_contexts=len(seq_lens),
            request_ids=[0, 1],
            prompt_lens=seq_lens,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0]),
        )
        metadata.prepare()
        draft_state = metadata.fp4_mla_draft_state
        assert draft_state is not None and draft_state.layer_ids == {draft}
        assert metadata.fp4_mla_state_for_layer(draft) is draft_state
        assert metadata.fp4_mla_state_for_layer(0) is None
        assert metadata.fp4_mla_state.cache_pool_id == target_group
        assert draft_state.cache_pool_id == draft_group
        assert metadata.fp4_mla_state.v_scale_pool is None
        assert draft_state.v_scale_pool is not None and draft_state.v_layer_base == draft
        target_state = metadata.fp4_mla_state
        with metadata.use_fp4_mla_state(draft_state):
            assert metadata.fp4_mla_state is draft_state
        assert metadata.fp4_mla_state is target_state
    finally:
        manager.shutdown()

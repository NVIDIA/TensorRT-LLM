# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-role policy tests; native cache lifecycle is tested separately."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention_backend.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
@pytest.mark.parametrize("role", ["context", "generation", ""])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("legacy_flag", [None, "0", "1"])
def test_layout_and_estimate_follow_worker_role(monkeypatch, backend, role, is_disagg, legacy_flag):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    monkeypatch.setenv("TRTLLM_FP4_MLA_ATTENTION_BACKEND", backend)
    monkeypatch.delenv("TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE", raising=False)
    monkeypatch.delenv("TLLM_KIMI_K3_BF16_KV_LAYERS", raising=False)
    if legacy_flag is None:
        monkeypatch.delenv("TRTLLM_FP4_MLA_CONTEXT_ONLY_CACHE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_FP4_MLA_CONTEXT_ONLY_CACHE", legacy_flag)
    config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4", enable_block_reuse=True),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=128,
        head_dim=576,
        pretrained_config=config,
        is_disagg=is_disagg,
    )
    context_only = is_disagg and role == "context"
    assert manager.fp4_mla_context_only_cache == context_only
    # CuTeDSL generation caches keep the persistent V-packed sidecar.
    v_pack = backend == "cutedsl" and not context_only
    assert manager.mla_v_head_dim == (512 if v_pack else None)
    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(pretrained_config=config),
        Mapping(world_size=1, tp_size=1, rank=0),
        num_layers=1,
        tokens_per_block=128,
        max_batch_size=2,
        is_disagg=is_disagg,
    )
    assert slope == (360 if context_only else 392) + (256 if v_pack else 0)
    assert intercept == 2 * 16 * 576 * 2


@pytest.mark.parametrize("role", ["context", "generation"])
def test_mxfp4_2d_drops_v_sidecars(monkeypatch, role):
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4_mla

    monkeypatch.setattr(fp4_mla, "_FP4_MLA_MXFP4_2D", True)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    monkeypatch.setenv("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "cutedsl")
    monkeypatch.delenv("TLLM_KIMI_K3_BF16_KV_LAYERS", raising=False)
    config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4", enable_block_reuse=False),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=128,
        head_dim=576,
        pretrained_config=config,
        is_disagg=True,
    )
    assert manager.fp4_mla_mxfp4_2d
    assert manager.fp4_mla_v_sidecars_dropped
    assert manager.mla_v_head_dim is None
    slope, _ = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(pretrained_config=config),
        Mapping(world_size=1, tp_size=1, rank=0),
        num_layers=1,
        tokens_per_block=128,
        max_batch_size=2,
        is_disagg=True,
    )
    # K codes (640 dims / 2) + K scales (640 / 16): no V-pack, no V-scale pool.
    assert slope == 360
    with pytest.raises(ValueError, match="enable_block_reuse"):
        Fp4MlaKVCacheManagerV2.configure_manager(
            object.__new__(Fp4MlaKVCacheManagerV2),
            KvCacheConfig(dtype="nvfp4", enable_block_reuse=True),
            CacheTypeCpp.SELFKONLY,
            dtype=DataType.NVFP4,
            tokens_per_block=128,
            head_dim=576,
            pretrained_config=config,
            is_disagg=True,
        )


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_only_context_workers_skip_generation_warmup(monkeypatch, context_only, graph):
    # The old benchmark flag must not suppress GEN warmup any more.
    monkeypatch.setenv("TRTLLM_CTX_ONLY_BENCHMARK", "1")
    cache = SimpleNamespace(fp4_mla_context_only_cache=context_only)
    resources = SimpleNamespace(get_resource_manager=lambda _: cache)
    engine = SimpleNamespace(
        kv_cache_manager_key="cache",
        _get_draft_kv_cache_manager=Mock(side_effect=RuntimeError("generation warmup reached")),
    )
    if graph:
        call = PyTorchModelEngine._create_cuda_graph_warmup_request
        last_arg = 0
    else:
        call = PyTorchModelEngine._create_warmup_request
        last_arg = 1
    if context_only:
        assert call(engine, resources, 1, last_arg) is None
        engine._get_draft_kv_cache_manager.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="generation warmup reached"):
            call(engine, resources, 1, last_arg)


def test_context_warmup_is_preserved():
    resources = SimpleNamespace(
        get_resource_manager=lambda _: SimpleNamespace(fp4_mla_context_only_cache=True)
    )
    engine = SimpleNamespace(
        kv_cache_manager_key="cache",
        _get_draft_kv_cache_manager=Mock(side_effect=RuntimeError("context warmup reached")),
    )
    with pytest.raises(RuntimeError, match="context warmup reached"):
        PyTorchModelEngine._create_warmup_request(engine, resources, 8192, 0)

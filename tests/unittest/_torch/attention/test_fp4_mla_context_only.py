# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA single-copy cache layout on disaggregated context workers."""

from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager as fp4_mla_cache_manager
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    is_fp4_mla_context_only_cache,
    is_fp4_mla_context_only_worker,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention.backends.fp4_mla.config import DISAGG_ROLE_ENV
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_KV_LORA_RANK = 512
_HEAD_DIM = 576


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    monkeypatch.delenv(DISAGG_ROLE_ENV, raising=False)


@pytest.mark.parametrize("role", ["context", "generation", ""])
@pytest.mark.parametrize("is_disagg", [False, True])
def test_context_only_worker_requires_disagg_context_role(
    monkeypatch: pytest.MonkeyPatch, role: str, is_disagg: bool
) -> None:
    monkeypatch.setenv(DISAGG_ROLE_ENV, role)
    assert is_fp4_mla_context_only_worker(is_disagg) == (is_disagg and role == "context")


def test_context_only_cache_flag_must_be_exactly_true() -> None:
    assert is_fp4_mla_context_only_cache(SimpleNamespace(fp4_mla_context_only_cache=True))
    assert not is_fp4_mla_context_only_cache(SimpleNamespace(fp4_mla_context_only_cache=False))
    assert not is_fp4_mla_context_only_cache(SimpleNamespace())
    assert not is_fp4_mla_context_only_cache(None)


@pytest.mark.parametrize("role", ["context", "generation", ""])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("fused_v_transpose", [False, True])
def test_layout_and_estimate_follow_worker_role(
    monkeypatch: pytest.MonkeyPatch, role: str, is_disagg: bool, fused_v_transpose: bool
) -> None:
    monkeypatch.setenv(DISAGG_ROLE_ENV, role)
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v_transpose)))
    # Record the requested sidecar buffers without the native config type.
    monkeypatch.setattr(fp4_mla_cache_manager, "BufferConfig", SimpleNamespace)
    config = SimpleNamespace(kv_lora_rank=_KV_LORA_RANK, qk_rope_head_dim=64)

    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        KvCacheConfig(dtype="nvfp4", enable_block_reuse=True),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=_HEAD_DIM,
        pretrained_config=config,
        is_disagg=is_disagg,
    )
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    manager.num_local_layers = 1
    manager.pp_layers = [0]

    context_only = is_disagg and role == "context"
    assert manager.fp4_mla_context_only_cache == context_only
    assert manager.mla_v_head_dim == (None if context_only or fused_v_transpose else 512)
    cache_roles = manager._fp4_mla_cache_roles()
    assert (Role.MLA_V_SCALE in cache_roles) == (not context_only)
    assert (Role.MLA_V_PACKED in cache_roles) == (manager.mla_v_head_dim is not None)
    sidecars = manager._extra_buffers_per_layer(tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK)[0]
    assert [buffer.role for buffer in sidecars] == cache_roles[2:]
    assert all(buffer.size > 0 for buffer in sidecars)
    if context_only:
        assert manager._v_scale_bytes_per_page() == 0
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
    # K (640 / 2) + K scales (640 / 16) [+ V scales 32] [+ packed V 512 / 2].
    assert slope == (360 if context_only else 392 if fused_v_transpose else 648)
    assert intercept == 2 * HP_BLOCK_SIZE * _HEAD_DIM * 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side composition of the dense FP4 MLA layout into the Kimi K3 hybrid manager."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common as mamba_common
import tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 as mamba_v2
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
    Fp4MlaKVCacheManagerV2,
    Fp4MlaV2CacheLayoutPolicy,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import supports_fp4_mla_attention
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

pytestmark = pytest.mark.cpu_only


def _fp4_quant_config():
    return SimpleNamespace(
        quant_mode=SimpleNamespace(
            has_fp8_kv_cache=lambda: False,
            has_fp4_kv_cache=lambda: True,
        ),
        mamba_ssm_cache_dtype=None,
    )


def _kimi_model_config(*, model_type: str = "kimi_linear") -> SimpleNamespace:
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type=model_type,
        vocab_size=163840,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        pretrained_config=config,
        mapping=Mapping(),
        quant_config=_fp4_quant_config(),
        sparse_attention_config=None,
        spec_config=None,
        attn_backend="TRTLLM",
        get_num_mamba_layers=lambda: 2,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)


def test_kimi_k3_is_the_supported_fp4_mla_hybrid() -> None:
    assert supports_fp4_mla_attention(_kimi_model_config())
    nemotron = _kimi_model_config()
    nemotron.pretrained_config = SimpleNamespace(
        model_type="nemotron_h",
        hybrid_override_pattern="M*",
        kv_lora_rank=32,
        qk_rope_head_dim=8,
    )
    assert not supports_fp4_mla_attention(nemotron)


def test_nvfp4_kimi_k3_routes_to_hybrid_manager() -> None:
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    assert get_kv_cache_manager_cls(_kimi_model_config(), kv_cache_config) is (
        KimiK3HybridCacheManager
    )


def test_nvfp4_hybrid_without_kda_layers_is_rejected() -> None:
    model_config = _kimi_model_config()
    model_config.get_num_mamba_layers = lambda: 0
    with pytest.raises(NotImplementedError, match="only for Kimi K3"):
        get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))


def test_nvfp4_kimi_k3_rejects_reuse_with_pool_rebalance() -> None:
    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=True,
        enable_kv_pool_rebalance=True,
    )
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance"):
        get_kv_cache_manager_cls(_kimi_model_config(), kv_cache_config)


@pytest.mark.parametrize("nvfp4", [False, True])
def test_kimi_k3_manager_receives_mla_config_only_under_nvfp4(
    monkeypatch: pytest.MonkeyPatch, nvfp4: bool
) -> None:
    captured = {}

    class RecordingManager(KimiK3HybridCacheManager):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

    model_config = _kimi_model_config()
    if not nvfp4:
        model_config.quant_config = None
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    assert captured["dtype"] == (DataType.NVFP4 if nvfp4 else DataType.BF16)
    if nvfp4:
        assert captured["pretrained_config"] is model_config.pretrained_config
    else:
        assert "pretrained_config" not in captured


def _hybrid_manager(*, fp4_mla: bool) -> MambaHybridCacheManagerV2:
    """A hybrid manager with local layers [KDA, MLA, KDA, MLA] and two HP layers."""
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.pp_layers = [0, 1, 2, 3]
    manager._mamba_layer_mask = [True, False, True, False]
    manager.tokens_per_block = 128
    manager.impl = SimpleNamespace(
        layer_grouping=[
            [LayerId(0), LayerId(2)],
            [LayerId(1), LayerId(3)],
            [LayerId(4), LayerId(5)],
        ]
    )
    manager._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy if fp4_mla else None
    if fp4_mla:
        manager.fp4_mla_context_only_cache = False
        manager.mla_v_head_dim = None
        manager._fp4_mla_compact_to_local = [1, 3]
        manager._cache_manager_layer_ids = [LayerId(1), LayerId(3)]
        manager._fp8_mla_manager_layer_ids = []
        manager._hp_manager_layer_ids = [LayerId(4), LayerId(5)]
        manager._fp4_mla_cache_roles = lambda: Fp4MlaV2CacheLayoutPolicy._fp4_mla_cache_roles(
            manager
        )
        manager.get_disagg_global_layer_ids = manager._fp4_mla_disagg_global_layer_ids
    return manager


def test_hp_tail_virtual_layers_are_attention_layers() -> None:
    manager = _hybrid_manager(fp4_mla=True)
    assert manager._is_local_mamba_layer(0)
    assert not manager._is_local_mamba_layer(1)
    assert not manager._is_local_mamba_layer(4)
    assert manager._get_pool_roles(0) == (MambaRole.SSM_STATE, None)
    assert manager._get_pool_roles(2) == (Role.MLA_HP_TAIL, None)

    manager._hp_manager_layer_ids = [LayerId(5)]
    with pytest.raises(ValueError, match="virtual layer 4"):
        manager._get_pool_roles(2)


def test_disagg_mapper_kinds_keep_indexed_recurrent_state() -> None:
    plain = _hybrid_manager(fp4_mla=False).get_disagg_role_mapper_kinds()
    assert plain[Role.ALL] == MapperKind.INDEXED
    assert plain[MambaRole.CONV_STATE] == MapperKind.SECTIONED

    kinds = _hybrid_manager(fp4_mla=True).get_disagg_role_mapper_kinds()
    # SSM state falls back to Role.ALL, which a recurrent view needs INDEXED.
    assert kinds[Role.ALL] == MapperKind.INDEXED
    assert kinds[MambaRole.CONV_STATE] == MapperKind.SECTIONED
    assert kinds[Role.KEY] == MapperKind.NHD
    assert kinds[Role.KEY_BLOCK_SCALE] == MapperKind.NHD
    assert kinds[Role.MLA_V_SCALE] == MapperKind.NHD
    assert kinds[Role.MLA_HP_TAIL] == MapperKind.REPLICATED


def test_disagg_ignored_roles_compose_with_fp4_mla() -> None:
    assert _hybrid_manager(fp4_mla=True).get_disagg_ignored_roles() == frozenset(
        (Role.MLA_V_SCALE, Role.MLA_V_PACKED)
    )
    assert _hybrid_manager(fp4_mla=False).get_disagg_ignored_roles() == frozenset()


def test_disagg_global_layer_ids_are_declared_only_for_fp4_mla() -> None:
    # Without the FP4 MLA layout the transfer keeps its default encoding.
    assert getattr(_hybrid_manager(fp4_mla=False), "get_disagg_global_layer_ids", None) is None

    manager = _hybrid_manager(fp4_mla=True)
    assert manager.get_disagg_global_layer_ids(0) == [0, 4]
    assert manager.get_disagg_global_layer_ids(1) == [2, 6]
    assert manager.get_disagg_global_layer_ids(2) == [3, 7]
    manager.impl.layer_grouping.append([LayerId(0), LayerId(1)])
    with pytest.raises(ValueError, match="mixes recurrent"):
        manager.get_disagg_global_layer_ids(3)


def test_snapshot_points_align_to_resumable_fp4_blocks() -> None:
    request = SimpleNamespace(expect_snapshot_points=[100, 128, 300, 1000])
    _hybrid_manager(fp4_mla=True)._align_fp4_mla_snapshot_points(request)
    assert request.expect_snapshot_points == [128, 256, 896]

    request = SimpleNamespace(expect_snapshot_points=[100, 300])
    _hybrid_manager(fp4_mla=False)._align_fp4_mla_snapshot_points(request)
    assert request.expect_snapshot_points == [100, 300]


def test_estimator_uses_fp4_attention_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    def generic_attention_cost(*args, **kwargs):
        raise AssertionError("the FP4 MLA override must replace the generic attention cost")

    monkeypatch.setattr(
        mamba_common,
        "_get_local_mamba_cache_layout",
        lambda *args, **kwargs: (SimpleNamespace(get_states_bytes_per_layer=lambda _: 1000), 2, 3),
    )
    monkeypatch.setattr(
        mamba_common.KVCacheManager, "get_cache_size_per_token", generic_attention_cost
    )
    kv_cache_config = SimpleNamespace(
        enable_block_reuse=True,
        mamba_state_config=SimpleNamespace(
            periodic_snapshot_interval=256,
            additional_snapshot_offsets_from_start=[],
            additional_snapshot_offsets_from_end=[],
        ),
    )
    slope, intercept = mamba_common._estimate_mamba_hybrid_cache_cost(
        SimpleNamespace(),
        Mapping(),
        max_batch_size=2,
        kv_cache_config=kv_cache_config,
        tokens_per_block=128,
        max_seq_len=None,
        num_reserved_dummy_slots=1,
        include_explicit_snapshots=True,
        cap_partial_attention_snapshots=True,
        attention_slope_override=10,
        attention_page_bytes_override=500,
    )
    # Three state slots (two live, one dummy) of 2 x 1000 bytes, plus one
    # retained partial attention page per live sequence with its HP page.
    assert intercept == 3 * 2000 + 2 * (10 * 128 + 500)
    assert slope == 10 + (2000 + 255) // 256


def test_hybrid_cost_adds_fp4_mla_fixed_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def estimate(model_config, mapping, **kwargs):
        captured.update(kwargs)
        return 100, 1000

    monkeypatch.setattr(mamba_v2, "_estimate_mamba_hybrid_cache_cost", estimate)
    monkeypatch.setattr(
        mamba_v2,
        "_get_local_mamba_cache_layout",
        lambda *args, **kwargs: (None, 2, 3),
    )
    fp4_calls = []

    def fp4_cost(model_config, mapping, num_layers=None, **kwargs):
        fp4_calls.append((num_layers, kwargs))
        return 7, 4000

    monkeypatch.setattr(Fp4MlaKVCacheManagerV2, "get_cache_size_per_token", fp4_cost)
    slope, intercept = MambaHybridCacheManagerV2.get_cache_size_per_token(
        _kimi_model_config(),
        Mapping(),
        max_batch_size=4,
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=128,
    )
    assert (slope, intercept) == (100, 5000)
    assert fp4_calls == [
        (
            3,
            dict(
                tokens_per_block=128,
                max_batch_size=4,
                spec_config=None,
                is_disagg=False,
            ),
        )
    ]
    assert captured["attention_slope_override"] == 7
    assert captured["attention_page_bytes_override"] == 1000

    model_config = _kimi_model_config()
    model_config.quant_config = None
    fp4_calls.clear()
    MambaHybridCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(),
        max_batch_size=4,
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=128,
    )
    assert not fp4_calls
    assert captured["attention_slope_override"] is None
    assert captured["attention_page_bytes_override"] == 0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, budgeting, and context-worker admission for 2FP4 KDA."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KimiK3HybridCacheManager,
    kda_2fp4_states_bytes_per_layer,
)
from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import HISTORY_SIZE, STATE_BYTES_PER_HEAD
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _cache_config(**kwargs) -> KvCacheConfig:
    return KvCacheConfig(mamba_state_config={"kda_2fp4_replay": True}, **kwargs)


def _model_config() -> SimpleNamespace:
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        vocab_size=1024,
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        dtype=torch.bfloat16,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
    )
    return SimpleNamespace(
        pretrained_config=config,
        mapping=Mapping(),
        quant_config=None,
        sparse_attention_config=None,
        spec_config=None,
        get_num_mamba_layers=lambda: 2,
    )


def test_2fp4_defaults_to_disabled() -> None:
    # Off by default, so the V2-only check does not reject the V1 manager.
    config = KvCacheConfig(mamba_state_config={}, use_kv_cache_manager_v2=False)
    assert config.mamba_state_config.kda_2fp4_replay is False


@pytest.mark.parametrize("value", [None, {"decode": True}])
def test_2fp4_rejects_non_boolean_enablement(value: object) -> None:
    with pytest.raises(ValueError, match="kda_2fp4_replay"):
        KvCacheConfig(mamba_state_config={"kda_2fp4_replay": value})


def test_2fp4_configuration_and_physical_budget() -> None:
    config = _cache_config()
    assert config.mamba_state_config.kda_2fp4_replay is True
    params = extract_mamba_kv_cache_params(_model_config().pretrained_config)
    mapping = Mapping(world_size=2, tp_size=2)
    actual = kda_2fp4_states_bytes_per_layer(params, mapping)
    conv_bytes = (3 * 4 * 128 // 2) * 3 * torch.bfloat16.itemsize
    assert actual == conv_bytes + 2 * STATE_BYTES_PER_HEAD
    assert params.get_states_bytes_per_layer(mapping) > actual


@pytest.mark.parametrize(
    "kwargs", [{"use_kv_cache_manager_v2": False}, {"enable_kv_pool_rebalance": True}]
)
def test_2fp4_rejects_unsupported_cache_layouts(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="KDA 2FP4"):
        _cache_config(**kwargs)


def test_2fp4_rejects_non_kda_model() -> None:
    model = SimpleNamespace(pretrained_config=SimpleNamespace(model_type="qwen3_next"))
    with pytest.raises(ValueError, match="Kimi KDA"):
        get_kv_cache_manager_cls(model, _cache_config())


def test_2fp4_selects_the_glm5_next_manager() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextCacheManager

    model = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="glm5_next_text"),
        sparse_attention_config=None,
        quant_config=None,
        get_num_mamba_layers=lambda: 2,
    )
    assert get_kv_cache_manager_cls(model, _cache_config()) is Glm5NextCacheManager


@pytest.mark.parametrize("dspark", [False, True])
def test_2fp4_live_history_budget(monkeypatch: pytest.MonkeyPatch, dspark: bool) -> None:
    """The live history table and the per-batch buffers are reserved beside the pools.

    The table holds one row per resident request and reserved dummy slot, each
    with every local layer's K/U/G history and length; no replay region is
    budgeted. A disaggregated context worker keeps no history.
    """
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextCacheManager
    from tensorrt_llm._torch.custom_ops.cute_dsl_kda_2fp4_mtp_commit import DESC_FIELDS
    from tensorrt_llm._torch.custom_ops.cute_dsl_kda_2fp4_replay import REPLAY_WORKSPACE_INT32
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
        _get_num_cuda_graph_padding_dummy_slots,
    )

    model = _model_config()
    params = extract_mamba_kv_cache_params(model.pretrained_config)
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda.cache_manager._get_local_mamba_cache_layout",
        lambda *args, **kwargs: (params, 2, 2),
    )
    spec_config = DSparkDecodingConfig(max_draft_len=3) if dspark else None
    capacity = HISTORY_SIZE + (4 if dspark else 0)
    # Two local layers of four heads, plus each layer's length row.
    row_bytes = 2 * (4 * capacity * 128 * (2 + 2 + 4) + 16)
    rows = 4 + _get_num_cuda_graph_padding_dummy_slots(spec_config, 4)
    if dspark:
        batch_bytes = 2 * 4 * 4 * (3 * 4 * 128) * 2 + 2 * 4 * 4 + 2 * DESC_FIELDS * 8
    else:
        batch_bytes = (4 + REPLAY_WORKSPACE_INT32) * 4
    kwargs = dict(max_batch_size=4, kv_cache_config=_cache_config(), spec_config=spec_config)
    for manager_cls in (KimiK3HybridCacheManager, Glm5NextCacheManager):
        assert manager_cls._estimate_fixed_region_bytes(model, Mapping(), **kwargs) == 0
        assert (
            manager_cls._estimate_reserved_device_bytes(model, Mapping(), **kwargs)
            == rows * row_bytes + batch_bytes
        )
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    assert (
        KimiK3HybridCacheManager._estimate_reserved_device_bytes(
            model, Mapping(), is_disagg=True, **kwargs
        )
        == 0
    )


@pytest.mark.parametrize("mtp", [False, True, "dspark"])
def test_2fp4_factory_keeps_decode_separate_from_mtp(
    monkeypatch: pytest.MonkeyPatch, mtp: bool | str
) -> None:
    captured = {}

    class RecordingManager(MambaHybridCacheManagerV2):
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor._util.get_sm_version", lambda: 100)
    spec = (
        DSparkDecodingConfig(max_draft_len=3)
        if mtp == "dspark"
        else MTPDecodingConfig(max_draft_len=3)
        if mtp
        else None
    )

    def create() -> None:
        _create_kv_cache_manager(
            model_engine=None,
            kv_cache_manager_cls=RecordingManager,
            mapping=Mapping(),
            kv_cache_config=_cache_config(),
            tokens_per_block=64,
            max_seq_len=2048,
            max_batch_size=4,
            spec_config=spec,
            sparse_attention_config=None,
            max_num_tokens=256,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=_model_config(),
            dtype=torch.bfloat16,
        )

    if mtp is True:
        with pytest.raises(ValueError, match="DSpark only"):
            create()
        assert not captured
    else:
        create()
        assert captured["kda_2fp4_replay"] is True
        for name in ("kda_replay_num_spec", "kda_mtp_replay", "kda_draft_records"):
            assert name not in captured
        assert captured["conv_state_layout"] == "q_k_v"


def test_separate_dspark_drafter_does_not_inherit_2fp4_state() -> None:
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    creator = object.__new__(KvCacheCreator)
    creator._get_draft_max_attention_window = lambda *_: None
    target = _cache_config()
    draft = creator._get_one_model_draft_kv_cache_config(target, 2048)
    assert target.mamba_state_config.kda_2fp4_replay
    assert not draft.mamba_state_config.kda_2fp4_replay


def test_dspark_draft_width_is_bounded() -> None:
    from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import validate_speculative_config

    for draft in (1, 3, 6, 7):
        validate_speculative_config(DSparkDecodingConfig(max_draft_len=draft))
    with pytest.raises(ValueError, match="1..7"):
        validate_speculative_config(DSparkDecodingConfig(max_draft_len=8))


@pytest.mark.parametrize("dummy", [False, True])
def test_context_worker_admission_still_allows_attention_dp_dummies(dummy: bool) -> None:
    """A disaggregated context worker keeps no 2FP4 replay history: it runs
    attention-DP padding dummies but rejects real generation requests."""
    manager = object.__new__(KimiK3HybridCacheManager)
    manager._kda_2fp4 = True
    manager.kda_context_only = True
    manager.local_num_mamba_layers = 0
    manager._kda_replay = None
    batch = SimpleNamespace(generation_requests=[SimpleNamespace(is_dummy=dummy)])
    with patch.object(KVCacheManagerV2, "prepare_resources") as prepare:
        if dummy:
            manager.prepare_resources(batch)
            prepare.assert_called_once_with(batch)
        else:
            with pytest.raises(ValueError, match="context worker"):
                manager.prepare_resources(batch)
            prepare.assert_not_called()

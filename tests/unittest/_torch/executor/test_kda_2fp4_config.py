# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, budgeting, and prefill-only admission for 2FP4 KDA."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import STATE_BYTES_PER_HEAD
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _cache_config(prefill_only: bool = False, **kwargs) -> KvCacheConfig:
    return KvCacheConfig(
        mamba_state_config={
            "kda_2fp4_replay": True,
            "kda_2fp4_prefill_only": prefill_only,
        },
        **kwargs,
    )


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


@pytest.mark.parametrize("state_config", [{}, {"kda_2fp4_replay": False}])
def test_2fp4_defaults_to_disabled(state_config: dict) -> None:
    config = KvCacheConfig(
        mamba_state_config=state_config,
        use_kv_cache_manager_v2=False,
    )
    assert config.mamba_state_config.kda_2fp4_replay is False
    assert config.mamba_state_config.kda_2fp4_prefill_only is False


@pytest.mark.parametrize("value", [None, {}, {"prefill_only": True}])
def test_2fp4_rejects_non_boolean_enablement(value: object) -> None:
    with pytest.raises(ValueError, match="kda_2fp4_replay"):
        KvCacheConfig(mamba_state_config={"kda_2fp4_replay": value})


@pytest.mark.parametrize("state_config", [{}, {"kda_2fp4_replay": False}])
def test_2fp4_prefill_only_requires_enablement(state_config: dict) -> None:
    with pytest.raises(ValueError, match="requires kda_2fp4_replay=True"):
        KvCacheConfig(mamba_state_config={**state_config, "kda_2fp4_prefill_only": True})


@pytest.mark.parametrize("prefill_only", [False, True])
def test_2fp4_configuration_and_physical_budget(prefill_only: bool) -> None:
    config = _cache_config(prefill_only)
    assert config.mamba_state_config.kda_2fp4_replay is True
    assert config.mamba_state_config.kda_2fp4_prefill_only is prefill_only
    params = extract_mamba_kv_cache_params(_model_config().pretrained_config)
    mapping = Mapping(world_size=2, tp_size=2)
    actual = params.get_states_bytes_per_layer(mapping, kda_2fp4=True)
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


@pytest.mark.parametrize("snapshots", [False, True])
@pytest.mark.parametrize("dspark", [False, True])
def test_live_history_budget_does_not_scale_with_snapshot_slots(
    monkeypatch: pytest.MonkeyPatch, snapshots: bool, dspark: bool
) -> None:
    model = _model_config()
    params = extract_mamba_kv_cache_params(model.pretrained_config)
    monkeypatch.setattr(KVCacheManager, "get_cache_size_per_token", lambda *args, **kwargs: 17)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager._get_local_mamba_cache_layout",
        lambda *args, **kwargs: (params, 2, 2),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.kda_mtp_replay_num_spec",
        lambda *_: pytest.fail("2FP4 must not budget legacy MTP replay buffers"),
    )
    costs = []
    for prefill_only in [True, False]:
        config = _cache_config(prefill_only)
        config.enable_block_reuse = snapshots
        if snapshots:
            config.mamba_state_config.periodic_snapshot_interval = 64
            config.mamba_state_config.additional_snapshot_offsets_from_start = [32]
            config.mamba_state_config.additional_snapshot_offsets_from_end = [0]
        costs.append(
            MambaHybridCacheManagerV2.get_cache_size_per_token(
                model,
                Mapping(),
                max_batch_size=4,
                kv_cache_config=config,
                max_seq_len=2048,
                tokens_per_block=64,
                spec_config=DSparkDecodingConfig(max_draft_len=3) if dspark else None,
            )
        )
    assert costs[0][0] == costs[1][0]
    # Two local layers, four live requests plus a reserved dummy row, and
    # four heads. Snapshot slots add packed state but never K/U/G histories.
    history_bytes = 2 * 5 * (4 * (20 if dspark else 16) * 128 * 8 + 4)
    candidate_bytes = 2 * 4 * (4 * 3 * 4 * 128 * 2 + 4) if dspark else 0
    assert costs[1][1] - costs[0][1] == history_bytes + candidate_bytes


@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize("mtp", [False, True, "dspark"])
def test_2fp4_factory_keeps_decode_separate_from_mtp(
    monkeypatch: pytest.MonkeyPatch, prefill_only: bool, mtp: bool | str
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
            kv_cache_config=_cache_config(prefill_only),
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
        assert captured["supports_generation"] is not prefill_only
        assert "kda_replay_num_spec" not in captured
        assert captured["conv_state_layout"] == "q_k_v"


def test_separate_dspark_drafter_does_not_inherit_2fp4_state() -> None:
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    creator = object.__new__(KvCacheCreator)
    creator._get_draft_max_attention_window = lambda *_: None
    target = _cache_config(prefill_only=True)
    draft = creator._get_one_model_draft_kv_cache_config(target, 2048)
    assert target.mamba_state_config.kda_2fp4_replay
    assert target.mamba_state_config.kda_2fp4_prefill_only
    assert not draft.mamba_state_config.kda_2fp4_replay
    assert not draft.mamba_state_config.kda_2fp4_prefill_only


def test_dspark_draft_width_is_bounded() -> None:
    from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import validate_speculative_config

    for draft in (1, 3, 6, 7):
        validate_speculative_config(DSparkDecodingConfig(max_draft_len=draft))
    with pytest.raises(ValueError, match="1..7"):
        validate_speculative_config(DSparkDecodingConfig(max_draft_len=8))


@pytest.mark.parametrize("dummy", [False, True])
def test_prefill_only_admission_still_allows_attention_dp_dummies(dummy: bool) -> None:
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.supports_generation = False
    manager.local_num_mamba_layers = 0
    batch = SimpleNamespace(generation_requests=[SimpleNamespace(is_dummy=dummy)])
    with patch.object(KVCacheManagerV2, "prepare_resources") as prepare:
        if dummy:
            manager.prepare_resources(batch)
            prepare.assert_called_once_with(batch)
        else:
            with pytest.raises(ValueError, match="prefill-only"):
                manager.prepare_resources(batch)
            prepare.assert_not_called()

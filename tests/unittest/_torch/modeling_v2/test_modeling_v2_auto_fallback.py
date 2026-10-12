# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When ``auto`` may fall back to the built-in implementation, read on the CPU.

A family whose targets bring their own input processor (DeepSeek-V4.1) has
that processor chosen on the checkpoint alone, before the GPU and the parallel
mapping are known. So under ``auto`` a configuration none of its targets
matches must not fall back silently: the built-in model would run behind the
target's processor and placeholder placement. It raises at model construction
instead, and names the way out. ``off`` stays exactly what it was.

The configs are plain namespaces and the SM is patched, so these run without a
GPU, without weights, and without importing any target's modeling code.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2._router_index as router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_ENV,
    modeling_v2_input_processor,
    modeling_v2_resolve,
)

_V41_TARGET = "ModelingV2DeepseekV41FlashNvfp4Sm103Unified"
_TEP4 = dict(
    world_size=4, tp_size=4, pp_size=1, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=False
)


def _v41_pretrained(**overrides):
    fields = dict(
        architectures=["DeepseekV41ForConditionalGeneration"],
        model_type="deepseek_v41",
        num_hidden_layers=40,
        hidden_size=5120,
        n_routed_experts=384,
        q_lora_rank=1280,
        head_dim=512,
        o_groups=8,
        index_n_heads=32,
        num_nextn_predict_layers=3,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _model_config(pretrained, quant="FP8_BLOCK_SCALES", **mapping_overrides):
    return SimpleNamespace(
        pretrained_config=pretrained,
        mapping=SimpleNamespace(**dict(_TEP4, **mapping_overrides)),
        quant_config=SimpleNamespace(quant_algo=quant),
        spec_config=None,
    )


@pytest.fixture
def sm(monkeypatch):
    """Route as if on the given SM; default a GB300 (10, 3)."""
    state = {"sm": (10, 3)}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a, **k: state["sm"], raising=False
    )
    return state


@pytest.fixture
def imports(monkeypatch):
    """Record the resolver's imports; a target's modeling module is not imported
    for real (it needs the GPU stack), only recorded."""
    seen = []

    def _import(name):
        seen.append(name)
        if name.endswith(".modeling"):
            return SimpleNamespace(__name__=name)
        return import_module(name)

    monkeypatch.setattr(router_index, "import_module", _import)
    return seen


def _mode(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv(MODELING_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_ENV, mode)


@pytest.mark.parametrize("mode", [None, "off"])
@pytest.mark.parametrize(
    "config",
    [
        _model_config(_v41_pretrained()),
        _model_config(_v41_pretrained(), tp_size=2, world_size=2, moe_ep_size=2),
        _model_config(_v41_pretrained(), quant=None),
    ],
    ids=["match", "tp2", "unquantized"],
)
def test_off_changes_nothing(monkeypatch, sm, imports, mode, config):
    """Off: no routing, no processor, no import, matching or not."""
    _mode(monkeypatch, mode)
    assert modeling_v2_resolve(config) is None
    assert modeling_v2_input_processor(config.pretrained_config) is None
    assert imports == []


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_a_matching_configuration_routes(monkeypatch, sm, imports, mode):
    _mode(monkeypatch, mode)
    assert modeling_v2_resolve(_model_config(_v41_pretrained())) == _V41_TARGET
    assert imports[-1].endswith("deepseek_v4_1_flash_nvfp4__sm_103__unified.modeling")


@pytest.mark.parametrize(
    "quant, mapping_overrides, device_sm, missed",
    [
        ("FP8_BLOCK_SCALES", dict(tp_size=2, world_size=2, moe_ep_size=2), (10, 3), "parallel"),
        ("FP8_BLOCK_SCALES", dict(moe_ep_size=2, moe_tp_size=2), (10, 3), "parallel"),
        (None, dict(), (10, 3), "quant"),
        ("FP8_BLOCK_SCALES", dict(), (10, 0), "sm"),
    ],
    ids=["tp2", "ep2xmoe_tp2", "unquantized", "sm100"],
)
def test_auto_refuses_the_builtin_fallback_behind_the_targets_processor(
    monkeypatch, sm, imports, quant, mapping_overrides, device_sm, missed
):
    """The target's checkpoint on a configuration no target covers: auto raises,
    names the criterion that missed and the ways out, and imports no target."""
    _mode(monkeypatch, "auto")
    sm["sm"] = device_sm
    config = _model_config(_v41_pretrained(), quant=quant, **mapping_overrides)
    with pytest.raises(ValueError) as raised:
        modeling_v2_resolve(config)
    message = str(raised.value)
    assert f"{MODELING_V2_ENV}=auto" in message
    assert f"{MODELING_V2_ENV}=off" in message
    assert "input processor" in message
    assert any(line.lstrip().startswith(missed) for line in message.splitlines())
    assert not any(name.endswith(".modeling") for name in imports)


def test_require_keeps_its_own_message(monkeypatch, sm, imports):
    _mode(monkeypatch, "require")
    config = _model_config(_v41_pretrained(), tp_size=2, world_size=2, moe_ep_size=2)
    with pytest.raises(ValueError, match=f"{MODELING_V2_ENV}=require, but no target matched"):
        modeling_v2_resolve(config)


def test_auto_falls_back_for_a_checkpoint_the_targets_do_not_claim(monkeypatch, sm, imports):
    """Another V4.1 shape gets no target processor, so the plain fallback stands."""
    _mode(monkeypatch, "auto")
    config = _model_config(_v41_pretrained(num_nextn_predict_layers=1))
    assert modeling_v2_resolve(config) is None
    assert modeling_v2_input_processor(config.pretrained_config) is None


def test_auto_falls_back_for_a_family_without_its_own_processor(monkeypatch, sm, imports):
    _mode(monkeypatch, "auto")
    gpt_oss = SimpleNamespace(
        architectures=["GptOssForCausalLM"],
        num_hidden_layers=36,
        hidden_size=2880,
        num_local_experts=128,
    )
    assert modeling_v2_resolve(_model_config(gpt_oss, world_size=2, tp_size=2)) is None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``staircase_resolve`` actually does, driven by synthetic configs.

No checkpoint and no weights: routing reads config *shape*, the mapping and
the SM version, all of which can be stated directly. The SM version is
monkeypatched so these run on any device -- the point here is the decision
logic, not the kernels.

The three things worth proving:

* ``off`` changes nothing. This is the whole safety argument for putting the
  hook in ``_resolve_class`` at all.
* a matching configuration reaches a target class, and that class is
  *external* -- so it wins the registry slot rather than losing it to the
  built-in provider, which the lazy zoo may import afterwards.
* ``require`` raises on a near-miss and says which criterion missed.
"""

from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    _is_builtin_model_class,
    get_registered_model_class,
)
from ._router_index import (
    STAIRCASE_ENV,
    StaircaseMode,
    staircase_resolve,
)
from tensorrt_llm.mapping import Mapping

_SM103 = (10, 3)


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _gpt_oss_config(**overrides):
    """The shape gpt-oss-120b's own config.json declares."""
    fields = dict(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=36,
        hidden_size=2880,
        num_local_experts=128,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _r1_config(**overrides):
    """The shape DeepSeek-R1-0528-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=61,
        hidden_size=7168,
        n_routed_experts=256,
        q_lora_rank=1536,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


@pytest.fixture(autouse=True)
def _mode(monkeypatch, request):
    """Default every case to 'auto'; a case that wants another mode sets it.

    The switch is an environment variable, so the tests set one too -- that is
    the surface under test.
    """
    monkeypatch.setenv(STAIRCASE_ENV, "auto")


def _set_mode(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv(STAIRCASE_ENV, raising=False)
    else:
        monkeypatch.setenv(STAIRCASE_ENV, mode)


def _model_config(pretrained_config, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(pretrained_config=pretrained_config, mapping=mapping)


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("unset", [True, False], ids=["env-unset", "env-off"])
def test_off_resolves_nothing(monkeypatch, unset):
    """The default must be indistinguishable from staircase not existing.

    Unset and an explicit "off" have to behave identically: the common case is
    that nobody has heard of this package.
    """
    _set_mode(monkeypatch, None if unset else "off")
    config = _model_config(_gpt_oss_config())
    assert staircase_resolve(config) is None


def test_off_still_reaches_the_builtin_implementation(monkeypatch):
    _set_mode(monkeypatch, "off")
    config = _model_config(_gpt_oss_config())
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == "tensorrt_llm._torch.models.modeling_gpt_oss"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    config = _model_config(_gpt_oss_config())
    assert staircase_resolve(config) == "StaircaseGptOss120bSm103Tp1"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    config = _model_config(_r1_config(), **_DEP4)
    assert staircase_resolve(config) == "StaircaseDeepseekR10528Nvfp4Sm103Dep4"


def test_resolving_registers_the_target_class():
    """The synthetic name is a key; the import behind it is what fills it."""
    config = _model_config(_gpt_oss_config())
    name = staircase_resolve(config)
    cls = get_registered_model_class(name)
    assert cls is not None, f"{name} resolved to no class"
    assert cls.__name__ == name
    assert cls.__module__.endswith(
        "staircase.models.gpt_oss.targets.gpt_oss_120b.sm_103.tp1.modeling"
    )


def test_the_target_registration_counts_as_external():
    """External registrations always win their slot; built-ins only fill
    empty ones. Living beside the zoo rather than inside it is what buys
    this, and a move into _torch/models/ would silently reverse it."""
    config = _model_config(_gpt_oss_config())
    cls = get_registered_model_class(staircase_resolve(config))
    assert not _is_builtin_model_class(cls)


def test_resolve_class_rewrites_the_architecture_end_to_end():
    config = _model_config(_gpt_oss_config())
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved.__name__ == "StaircaseGptOss120bSm103Tp1"


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, "shape"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "parallel"),
    ],
)
def test_gpt_oss_near_misses_do_not_match(monkeypatch, config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert staircase_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=missed):
        staircase_resolve(config)


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert staircase_resolve(config) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    _set_mode(monkeypatch, "require")
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        staircase_resolve(config)
    message = str(excinfo.value)
    assert "sm" in message and "(10, 0)" in message
    assert "no match" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert staircase_resolve(config) is None


def test_an_unrouted_architecture_raises_under_require(monkeypatch):
    _set_mode(monkeypatch, "require")
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        staircase_resolve(config)


@pytest.mark.parametrize(
    "raw,expected", [("off", "off"), ("AUTO", "auto"), (" require ", "require"), ("", "off")]
)
def test_the_env_var_is_read_leniently(monkeypatch, raw, expected):
    monkeypatch.setenv(STAIRCASE_ENV, raw)
    assert StaircaseMode.from_env().value == expected


def test_an_unknown_mode_raises_rather_than_falling_back(monkeypatch):
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    monkeypatch.setenv(STAIRCASE_ENV, "yes")
    with pytest.raises(ValueError, match="not a staircase mode"):
        StaircaseMode.from_env()

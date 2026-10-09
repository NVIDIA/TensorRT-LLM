# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``modeling_v2_resolve`` actually does, driven by synthetic configs.

No checkpoint and no weights: routing reads config *shape*, the mapping, the
SM version and the LLM API arguments the deployment was configured with, all
of which can be stated directly. The SM version is monkeypatched so these run
on any device -- the point here is the decision logic, not the kernels.

The things worth proving:

* ``off`` changes nothing. This is the whole safety argument for making the
  decision in the model loader at all.
* a matching configuration reaches a target class, and that class is
  *external* -- so it wins the registry slot rather than losing it to the
  built-in provider, which the lazy zoo may import afterwards.
* ``require`` raises on a near-miss and says which criterion missed.
* a target's ``within_bounds`` has the last word: outside it, ``auto`` falls
  back to the built-in implementation and ``require`` raises.
* ``AutoModelForCausalLM._resolve_class`` reads the decision off the
  ``ModelConfig`` rather than deciding again.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NULL_TRACE,
    modeling_v2_resolve,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss import routing as gpt_oss_routing
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    _is_builtin_model_class,
    get_registered_model_class,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_TARGET = "ModelingV2GptOss120bSm103Tp1"


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


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """The deployment, as the LLM API would have been given it.

    The switch is an LLM API argument, so the tests set one too -- that is the
    surface under test.
    """
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _model_config(pretrained_config, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(pretrained_config=pretrained_config, mapping=mapping)


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("mode", [None, "off"], ids=["default", "off"])
def test_off_resolves_nothing(mode):
    """The default must be indistinguishable from modeling_v2 not existing.

    Leaving the argument alone and an explicit "off" have to behave
    identically: the common case is that nobody has heard of this package.
    """
    args = TorchLlmArgs(model=_DUMMY_MODEL) if mode is None else _llm_args(mode)
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, args) is None


def test_off_still_reaches_the_builtin_implementation():
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == _BUILTIN_GPT_OSS


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(mode):
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args(mode)) == _GPT_OSS_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(mode):
    config = _model_config(_r1_config(), **_DEP4)
    assert modeling_v2_resolve(config, _llm_args(mode)) == "ModelingV2DeepseekR10528Nvfp4Sm103Dep4"


def test_resolving_registers_the_target_class():
    """The synthetic name is a key; the import behind it is what fills it."""
    config = _model_config(_gpt_oss_config())
    name = modeling_v2_resolve(config, _llm_args())
    cls = get_registered_model_class(name)
    assert cls is not None, f"{name} resolved to no class"
    assert cls.__name__ == name
    assert cls.__module__.endswith("modeling_v2.models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling")


def test_the_target_registration_counts_as_external():
    """External registrations always win their slot; built-ins only fill
    empty ones. Living beside the zoo rather than inside it is what buys
    this, and a move into _torch/models/ would silently reverse it."""
    config = _model_config(_gpt_oss_config())
    cls = get_registered_model_class(modeling_v2_resolve(config, _llm_args()))
    assert not _is_builtin_model_class(cls)


def test_resolve_class_reads_the_decision_off_the_config():
    """The loader decides and writes the result on the ModelConfig; the class
    lookup only has to honour it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args())
    decided = replace(config, modeling_v2_target=target)
    assert AutoModelForCausalLM._resolve_class(decided).__name__ == target


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, "shape"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "parallel"),
    ],
)
def test_gpt_oss_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args()) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "sm" in message and "(10, 0)" in message
    assert "no match" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_an_unrouted_architecture_raises_under_require():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_an_unknown_mode_is_rejected_when_the_arguments_are_built():
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent. Pydantic refuses it before anything is built.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    with pytest.raises(ValidationError, match="modeling_v2"):
        _llm_args("yes")


# --- within_bounds: the target's certified deployment envelope -----------------


def _outside(label, value):
    """A ``within_bounds`` that rejects on one named criterion."""

    def within_bounds(target, args, ctx, trace=NULL_TRACE):
        return trace.check(label, value, False)

    return within_bounds


def test_outside_its_bounds_a_target_falls_back_under_auto(monkeypatch):
    monkeypatch.setattr(gpt_oss_routing, "within_bounds", _outside("max_batch_size", 2048))
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_outside_its_bounds_a_target_raises_under_require(monkeypatch):
    """Falling back silently would break the promise ``require`` makes, the
    same way a missed identity criterion would -- so the error names the
    target that claimed the configuration and the bound it failed."""
    monkeypatch.setattr(gpt_oss_routing, "within_bounds", _outside("max_batch_size", 2048))
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert _GPT_OSS_TARGET in message
    assert "bounds" in message
    assert "max_batch_size" in message and "2048" in message

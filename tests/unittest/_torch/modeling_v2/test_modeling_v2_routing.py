# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``modeling_v2_resolve`` actually does, driven by synthetic configs.

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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_ENV,
    ModelingV2Mode,
    modeling_v2_resolve,
)
from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    _is_builtin_model_class,
    get_registered_model_class,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

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
    monkeypatch.setenv(MODELING_V2_ENV, "auto")


def _set_mode(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv(MODELING_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_ENV, mode)


def _model_config(pretrained_config, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(pretrained_config=pretrained_config, mapping=mapping)


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("unset", [True, False], ids=["env-unset", "env-off"])
def test_off_resolves_nothing(monkeypatch, unset):
    """The default must be indistinguishable from modeling_v2 not existing.

    Unset and an explicit "off" have to behave identically: the common case is
    that nobody has heard of this package.
    """
    _set_mode(monkeypatch, None if unset else "off")
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config) is None


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
    assert modeling_v2_resolve(config) == "ModelingV2GptOss120bSm103Tp1"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    config = _model_config(_r1_config(), **_DEP4)
    assert modeling_v2_resolve(config) == "ModelingV2DeepseekR10528Nvfp4Sm103Dep4"


def test_resolving_registers_the_target_class():
    """The synthetic name is a key; the import behind it is what fills it."""
    config = _model_config(_gpt_oss_config())
    name = modeling_v2_resolve(config)
    cls = get_registered_model_class(name)
    assert cls is not None, f"{name} resolved to no class"
    assert cls.__name__ == name
    assert cls.__module__.endswith("modeling_v2.models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling")


def test_the_target_registration_counts_as_external():
    """External registrations always win their slot; built-ins only fill
    empty ones. Living beside the zoo rather than inside it is what buys
    this, and a move into _torch/models/ would silently reverse it."""
    config = _model_config(_gpt_oss_config())
    cls = get_registered_model_class(modeling_v2_resolve(config))
    assert not _is_builtin_model_class(cls)


def test_resolve_class_rewrites_the_architecture_end_to_end():
    config = _model_config(_gpt_oss_config())
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved.__name__ == "ModelingV2GptOss120bSm103Tp1"


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
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config)


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    _set_mode(monkeypatch, "require")
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config)
    message = str(excinfo.value)
    assert "sm" in message and "(10, 0)" in message
    assert "no match" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert modeling_v2_resolve(config) is None


def test_an_unrouted_architecture_raises_under_require(monkeypatch):
    _set_mode(monkeypatch, "require")
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        modeling_v2_resolve(config)


@pytest.mark.parametrize(
    "raw,expected",
    [(None, "off"), ("off", "off"), ("AUTO", "auto"), (" require ", "require"), ("", "off")],
)
def test_the_env_var_is_read_leniently(monkeypatch, raw, expected):
    """``None`` is the unset case, and it is the one that must never drift.

    Everything in the accuracy suite rests on modeling_v2 being opt-in: unset has
    to read as off on the code path the engine actually takes.
    """
    if raw is None:
        monkeypatch.delenv(MODELING_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_ENV, raw)
    assert ModelingV2Mode.from_env().value == expected


def test_an_unknown_mode_raises_rather_than_falling_back(monkeypatch):
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    monkeypatch.setenv(MODELING_V2_ENV, "yes")
    with pytest.raises(ValueError, match="not a modeling_v2 mode"):
        ModelingV2Mode.from_env()


_SM100 = (10, 0)
_TP16_MOETP4EP4 = dict(world_size=16, tp_size=16, moe_tp_size=4, moe_ep_size=4)
_K3_TARGET = "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4"


def _k3_config(text_as_dict=False, **text_overrides):
    """The shape Kimi K3's own config.json declares: a vision-language wrapper
    whose language model is ``text_config``."""
    text = dict(
        model_type="kimi_linear",
        num_hidden_layers=93,
        hidden_size=7168,
        num_experts=896,
        routed_expert_hidden_size=3584,
    )
    text.update(text_overrides)
    return PretrainedConfig(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
        text_config=text if text_as_dict else PretrainedConfig(**text),
    )


# The quantization the MXFP4 checkpoint declares in its config.json, inside ``text_config``.
_K3_CHECKPOINT_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "group_size": 32,
                "num_bits": 4,
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "kv_cache_scheme": None,
}


def _k3_checkpoint_quant_config():
    """What the engine reads for the MXFP4 checkpoint: ``KimiK3Config`` surfaces the text config's
    declaration, and ``ModelConfig`` parses it."""
    config = KimiK3Config(
        text_config=dict(quantization_config=_K3_CHECKPOINT_QUANTIZATION),
        architectures=["KimiK3ForConditionalGeneration"],
    )
    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        config.quantization_config, "TRTLLM"
    )
    assert layer_quant_config is None
    return quant_config


def _k3_model_config(pretrained_config, quant_config=None, **mapping_kwargs):
    """A model config of the MXFP4 checkpoint's quantization (or ``quant_config``)."""
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs),
        quant_config=quant_config if quant_config is not None else _k3_checkpoint_quant_config(),
    )


@pytest.fixture
def _on_sm100(monkeypatch):
    """Route as if this were a GB200."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def test_kimi_k3_checkpoint_reads_as_w4a16_mxfp4():
    quant_config = _k3_checkpoint_quant_config()
    assert quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
    assert quant_config.group_size == 32


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("text_as_dict", [False, True], ids=["text-config", "text-dict"])
def test_kimi_k3_tp16_moetp4ep4_matches(monkeypatch, mode, text_as_dict):
    _set_mode(monkeypatch, mode)
    config = _k3_model_config(_k3_config(text_as_dict=text_as_dict), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config) == _K3_TARGET


@pytest.mark.usefixtures("_on_sm100")
def test_kimi_k3_target_registers_and_counts_as_external():
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    cls = get_registered_model_class(modeling_v2_resolve(config))
    assert cls is not None and cls.__name__ == _K3_TARGET
    assert not _is_builtin_model_class(cls)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "quant_algo",
    [QuantAlgo.MIXED_PRECISION, None],
    ids=["nvfp4-requant", "unquantized"],
)
def test_kimi_k3_other_quantizations_do_not_match(monkeypatch, quant_algo):
    """The NVFP4 requant (MIXED_PRECISION) and an unquantized checkpoint have the MXFP4 checkpoint's shape; the
    quantization is what keeps them out of a target whose loader reads packed MXFP4 experts."""
    config = _k3_model_config(_k3_config(), QuantConfig(quant_algo=quant_algo), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match="quant"):
        modeling_v2_resolve(config)


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "text_overrides, mapping_kwargs, missed",
    [
        # a Kimi-family checkpoint of another depth
        (dict(num_hidden_layers=61), _TP16_MOETP4EP4, "shape"),
        # route B's expert split: experts tensor-parallel 16 ways, no target yet
        (dict(), dict(world_size=16, tp_size=16, moe_tp_size=16, moe_ep_size=1), "parallel"),
        # attention data parallelism splits the requests, not the heads
        (dict(), dict(_TP16_MOETP4EP4, enable_attention_dp=True), "parallel"),
        # one tray instead of four
        (dict(), dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4), "parallel"),
    ],
)
def test_kimi_k3_near_misses_do_not_match(monkeypatch, text_overrides, mapping_kwargs, missed):
    config = _k3_model_config(_k3_config(**text_overrides), **mapping_kwargs)
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config)


def test_kimi_k3_on_another_sm_does_not_match(monkeypatch):
    """The autouse fixture routes as a GB300 (sm 10.3); the K3 target is sm 10.0 only."""
    config = _k3_model_config(_k3_config(), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config)
    assert "sm" in str(excinfo.value) and "(10, 3)" in str(excinfo.value)

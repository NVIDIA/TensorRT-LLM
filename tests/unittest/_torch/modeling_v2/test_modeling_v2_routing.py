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
    modeling_v2_input_processor,
    modeling_v2_resolve,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    _is_builtin_model_class,
    get_registered_model_class,
)
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

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


def _v41_config(**overrides):
    """The shape DeepSeek-V4.1-Flash-NVFP4's own config.json declares (its
    ``text_config``, which the composite config answers from), under the name
    the engine gives a V4.1 checkpoint that carries its vision tower."""
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
    return PretrainedConfig(**fields)


def _v41_model_config(quant="FP8_BLOCK_SCALES", config_kwargs=None, **mapping_kwargs):
    quant_config = QuantConfig(quant_algo=QuantAlgo(quant)) if quant else QuantConfig()
    return ModelConfig(
        pretrained_config=_v41_config(**(config_kwargs or {})),
        mapping=Mapping(**mapping_kwargs),
        quant_config=quant_config,
    )


_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)
# Every member the unified target folds, as the mapping states it.
_V41_MEMBERS = {
    "tep4": _TEP4,
    "dep4": dict(_TEP4, enable_attention_dp=True),
    "tp4": dict(world_size=4, tp_size=4, moe_ep_size=1, moe_tp_size=4),
    "tep8": dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1),
    "dep8": dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True),
    "pp2xtep2": dict(world_size=4, tp_size=2, pp_size=2, moe_ep_size=2, moe_tp_size=1),
}


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("member", sorted(_V41_MEMBERS))
@pytest.mark.parametrize("quant", ["FP8_BLOCK_SCALES", "MIXED_PRECISION"])
def test_v41_unified_matches(monkeypatch, mode, member, quant):
    """Every member, under either reading of the checkpoint's quantization: the
    engine's DeepSeek-V4.1 loader reads the dense FP8 blocks as FP8_BLOCK_SCALES,
    a modelopt reading gives MIXED_PRECISION."""
    _set_mode(monkeypatch, mode)
    config = _v41_model_config(quant, **_V41_MEMBERS[member])
    assert modeling_v2_resolve(config) == "ModelingV2DeepseekV41FlashNvfp4Sm103Unified"


def test_v41_off_still_reaches_the_builtin_implementation(monkeypatch):
    """With the switch off the checkpoint is the built-in DeepSeek-V4.1's."""
    _set_mode(monkeypatch, "off")
    resolved = AutoModelForCausalLM._resolve_class(_v41_model_config(**_TEP4))
    assert resolved is not None
    assert resolved.__module__ == "tensorrt_llm._torch.models.modeling_deepseekv41_multimodal"


def test_v41_resolve_class_rewrites_the_architecture_end_to_end(monkeypatch):
    _set_mode(monkeypatch, "require")
    resolved = AutoModelForCausalLM._resolve_class(_v41_model_config(**_TEP4))
    assert resolved.__name__ == "ModelingV2DeepseekV41FlashNvfp4Sm103Unified"
    assert not _is_builtin_model_class(resolved)


@pytest.mark.parametrize(
    "config_kwargs, quant, mapping_kwargs, missed",
    [
        # a DeepSeek-V4.1 checkpoint of another shape (one DSpark stage)
        (dict(num_nextn_predict_layers=1), "FP8_BLOCK_SCALES", _TEP4, "shape"),
        # the right shape, unquantized
        (dict(), None, _TEP4, "quant"),
        # the right checkpoint, a topology no member covers
        (dict(), "FP8_BLOCK_SCALES", dict(world_size=2, tp_size=2, moe_ep_size=2), "parallel"),
        (dict(), "FP8_BLOCK_SCALES", dict(_TEP4, moe_ep_size=2, moe_tp_size=2), "parallel"),
    ],
    ids=["shape", "quant", "tp2", "ep2xmoe_tp2"],
)
def test_v41_near_misses_do_not_match(monkeypatch, config_kwargs, quant, mapping_kwargs, missed):
    """Under auto, another checkpoint shape falls back; the target's own
    checkpoint raises rather than run the built-in model behind the target's
    input processor. Under require every near miss names its criterion."""
    config = _v41_model_config(quant, config_kwargs, **mapping_kwargs)
    if missed == "shape":
        assert modeling_v2_resolve(config) is None
    else:
        with pytest.raises(ValueError, match=f"{MODELING_V2_ENV}=off"):
            modeling_v2_resolve(config)

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config)


def test_v41_text_only_architecture_is_not_routed(monkeypatch):
    """A text-only V4.1 config has no vision tower for the target to run."""
    config = _v41_model_config(
        config_kwargs=dict(architectures=["DeepseekV41ForCausalLM"]), **_TEP4
    )
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match="no target exists for architecture"):
        modeling_v2_resolve(config)


@pytest.fixture
def _placeholder_registry(monkeypatch):
    """A copy of the placeholder registry's table, so registrations made here
    do not outlive the test."""
    table = dict(MULTIMODAL_PLACEHOLDER_REGISTRY._multimodal_placeholder_by_model_type)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY, "_multimodal_placeholder_by_model_type", table
    )
    return MULTIMODAL_PLACEHOLDER_REGISTRY


def test_v41_input_processor_is_the_builtin_choice_when_off(monkeypatch, _placeholder_registry):
    _set_mode(monkeypatch, "off")
    assert modeling_v2_input_processor(_v41_config()) is None


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_v41_input_processor_is_the_targets(monkeypatch, mode, _placeholder_registry):
    """With modeling_v2 on, the processor and the placeholder placement are the
    target's: one placeholder per image, before the text, joined by a blank line."""
    _set_mode(monkeypatch, mode)
    processor = modeling_v2_input_processor(_v41_config())
    assert processor is not None
    assert processor.__name__ == "DeepseekV41InputProcessor"
    assert processor.__module__.endswith(
        "models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.input_processor"
    )
    assert processor._registered_model_type == "deepseek_v41"
    registry = _placeholder_registry
    assert registry.get_placeholder("deepseek_v41", "image") == "<\uff5cdeepseek_image\uff5c>"
    assert (
        registry.get_placeholder_placement("deepseek_v41")
        is MultimodalPlaceholderPlacement.BEFORE_TEXT
    )
    assert registry.get_placeholders_separator("deepseek_v41") == "\n\n"
    assert registry.get_content_format("deepseek_v41") is ContentFormat.STRING
    assert not registry.get_interleave_placeholders("deepseek_v41")


def test_v41_input_processor_registration_holds_the_slot(monkeypatch, _placeholder_registry):
    """The built-in zoo module registering the same model type afterwards does
    not replace the target's placeholder metadata."""
    _set_mode(monkeypatch, "auto")
    modeling_v2_input_processor(_v41_config())
    registry = _placeholder_registry
    registry.set_placeholder_metadata(
        "deepseek_v41",
        MultimodalPlaceholderMetadata(placeholder_map={"image": "<image>"}),
        registrant_module="tensorrt_llm._torch.models.modeling_deepseekv41_multimodal",
    )
    assert registry.get_placeholder("deepseek_v41", "image") == "<\uff5cdeepseek_image\uff5c>"


@pytest.mark.parametrize(
    "config_kwargs",
    [dict(num_nextn_predict_layers=1), dict(architectures=["DeepseekV41ForCausalLM"])],
    ids=["other_shape", "text_only"],
)
def test_v41_input_processor_needs_the_targets_checkpoint(
    monkeypatch, config_kwargs, _placeholder_registry
):
    _set_mode(monkeypatch, "require")
    assert modeling_v2_input_processor(_v41_config(**config_kwargs)) is None


def test_a_family_without_its_own_input_processor_keeps_the_builtin_one(
    monkeypatch, _placeholder_registry
):
    _set_mode(monkeypatch, "require")
    assert modeling_v2_input_processor(_gpt_oss_config()) is None

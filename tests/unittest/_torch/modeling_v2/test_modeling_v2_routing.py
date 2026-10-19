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
* a matching configuration resolves to the target class itself, imported
  from its own directory; no registry lookup stands between the decision and
  the class.
* ``require`` raises on a near-miss and says, in words, which criterion
  missed; ``auto`` falls back to the built-in implementation instead.
* speculative decoding is an identity criterion: a target that does not
  implement the requested mode does not claim the deployment.
* ``AutoModelForCausalLM._resolve_class`` reads the decision off the
  ``ModelConfig`` rather than deciding again.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_TARGET = "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling.ModelingV2GptOss120bSm103Tp1"
_R1_TARGET = (
    "models.deepseek_v3.r1_0528_nvfp4__sm_103__dep4.modeling.ModelingV2DeepseekR10528Nvfp4Sm103Dep4"
)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


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


def _model_config(pretrained_config, spec_config=None, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config, mapping=mapping, spec_config=spec_config
    )


# A speculative mode no target implements.
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qwen4_exp_config(**vision_overrides):
    """The shape Qwen3.8-Flash-Next-NVFP4's normalized composite config declares."""
    text = PretrainedConfig(
        num_hidden_layers=48,
        hidden_size=2560,
        num_experts=512,
        num_attention_heads=24,
        linear_num_value_heads=48,
        vocab_size=248320,
    )
    vision = dict(
        depth=27,
        hidden_size=1152,
        num_heads=16,
        spatial_merge_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh",
    )
    vision.update(vision_overrides)
    return PretrainedConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        vision_config=PretrainedConfig(**vision),
    )


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
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GPT_OSS_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(mode):
    config = _model_config(_r1_config(), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _R1_TARGET


def test_resolve_class_reads_the_decision_off_the_config():
    """The loader decides and writes the class on the ModelConfig; the class
    lookup only has to honour it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args())
    decided = replace(config, modeling_v2_target=target)
    assert AutoModelForCausalLM._resolve_class(decided) is target


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, r"shape \(24, 2880, 32\)"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "2 ranks"),
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
    assert "no target fits" in message and "GptOssForCausalLM" in message
    assert "compute capability (10, 0)" in message


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


# --- speculative decoding is an identity criterion ------------------------------


def test_gpt_oss_with_speculative_decoding_falls_back_under_auto():
    """The tp1 target implements no speculative path, so claiming the
    deployment would trade a fallback for a crash inside the target."""
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_gpt_oss_with_speculative_decoding_raises_under_require():
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "NGramDecodingConfig" in message


def test_r1_dep4_with_mtp_matches():
    config = _model_config(_r1_config(), spec_config=MTPDecodingConfig(max_draft_len=3), **_DEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _R1_TARGET


def test_r1_dep4_with_another_speculative_mode_does_not_match():
    """The dep4 target carries the checkpoint's MTP module and nothing else."""
    config = _model_config(_r1_config(), spec_config=_NGRAM, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_r1_dep4_with_vanilla_mtp_does_not_match():
    """``use_mtp_vanilla`` resolves to the multi-model MTP loop; the target
    asserts the one-model MTP-Eagle loop at construction, so route must say
    no before that assert can."""
    vanilla = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
    config = _model_config(_r1_config(), spec_config=vanilla, **_DEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- qwen4_exp: one-model MTP-Eagle over a linear chain, one rank ---------------

_QWEN4_EXP_TARGET = "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."


@pytest.mark.parametrize(
    "spec_config",
    [None, MTPDecodingConfig(max_draft_len=1), MTPDecodingConfig(max_draft_len=3)],
    ids=["no-spec", "mtp1", "mtp3"],
)
def test_qwen4_exp_tp1_matches(spec_config):
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(target).startswith(_QWEN4_EXP_TARGET)


def _mtp(max_draft_len=3, **attributes):
    """An MTP config with fields the engine fills in after load set directly: the
    constructor remaps or recomputes them, so passing them would test nothing."""
    config = MTPDecodingConfig(max_draft_len=max_draft_len)
    for name, value in attributes.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize(
    "spec_config",
    [
        _NGRAM,
        MTPDecodingConfig(max_draft_len=4),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        _mtp(max_total_draft_tokens=5),
        _mtp(num_nextn_predict_layers=2),
    ],
    ids=[
        "ngram",
        "draft4",
        "vanilla-mtp",
        "max-total-draft",
        "two-predictors",
    ],
)
def test_qwen4_exp_unsupported_speculation_falls_back(spec_config):
    """The target asserts each of these at construction; route must decline first."""
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError):
        modeling_v2_resolve(config, _llm_args("require"))


def test_qwen4_exp_deepstack_vision_tower_does_not_match():
    config = _model_config(_qwen4_exp_config(deepstack_visual_indexes=[5, 11, 17]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_qwen4_exp_on_two_ranks_does_not_match():
    config = _model_config(_qwen4_exp_config(), world_size=2, tp_size=2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


# --- nemotron_h -------------------------------------------------------------

_SM107 = (10, 7)
_ULTRA_TARGET = (
    "models.nemotron_h.nemotron3_ultra_nvfp4__sm_107__tep4.modeling."
    "ModelingV2Nemotron3UltraNvfp4Sm107Tep4"
)
_ULTRA_DEP4_TARGET = (
    "models.nemotron_h.nemotron3_ultra_nvfp4__sm_107__dep4.modeling."
    "ModelingV2Nemotron3UltraNvfp4Sm107Dep4"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=False)

_SM100 = (10, 0)
_ULTRA_BF16_TARGET = (
    "models.nemotron_h.nemotron3_ultra_bf16__sm_100__tep8.modeling."
    "ModelingV2Nemotron3UltraBf16Sm100Tep8"
)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=False)


def _ultra_config(**overrides):
    """The shape Nemotron-3 Ultra 550B-A55B NVFP4's own config.json declares."""
    fields = dict(
        architectures=["NemotronHForCausalLM"],
        model_type="nemotron_h",
        hidden_size=8192,
        mamba_num_heads=256,
        mamba_head_dim=64,
        n_groups=8,
        ssm_state_size=128,
        num_attention_heads=64,
        num_key_value_heads=2,
        head_dim=128,
        n_routed_experts=512,
        num_experts_per_tok=22,
        moe_intermediate_size=5120,
        moe_latent_size=2048,
        moe_shared_expert_intermediate_size=10240,
        layers_block_type=["mamba", "moe", "mamba", "attention", "moe"] * 2,
        quantization_config=dict(
            quant_algo="MIXED_PRECISION",
            kv_cache_scheme=dict(dynamic=False, num_bits=8, type="float"),
        ),
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


@pytest.fixture
def _on_sm107(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM107)


@pytest.fixture
def _on_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_tep4_matches(mode, _on_sm107):
    config = _model_config(_ultra_config(), **_TEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_TARGET


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_dep4_matches(mode, _on_sm107):
    """Attention DP is an identity criterion: it selects the replicated-attention target."""
    config = _model_config(_ultra_config(), **dict(_TEP4, enable_attention_dp=True))
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_DEP4_TARGET


def test_ultra_depth_is_not_part_of_the_identity(_on_sm107):
    """Every weight the target declares is per layer and the stack is built from the
    config's own block list, so a prefix of the layers runs the same code."""
    config = _model_config(_ultra_config(layers_block_type=["mamba", "moe", "attention"]), **_TEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _ULTRA_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # Nemotron-3 Super's geometry
        (dict(hidden_size=4096, n_routed_experts=512), {}, r"config shape \(4096,"),
        # a dense-MLP block the target does not implement
        (dict(layers_block_type=["mamba", "mlp", "attention"]), {}, "block kinds"),
        # a BF16 export of the same model
        (dict(quantization_config=None), {}, r"quantization \(None, None\)"),
        # the right checkpoint, the experts tensor-parallel instead of expert-parallel
        (dict(), dict(_TEP4, moe_ep_size=1, moe_tp_size=4), "moe_ep=1"),
        # the right checkpoint on 8 ranks
        (dict(), dict(world_size=8, tp_size=8, moe_ep_size=8), "world_size=8"),
    ],
)
def test_ultra_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed, _on_sm107):
    config = _model_config(_ultra_config(**config_kwargs), **(mapping_kwargs or _TEP4))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_ultra_on_sm103_does_not_match():
    """Certification is per architecture: the autouse fixture routes as a GB300."""
    config = _model_config(_ultra_config(), **_TEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_ultra_with_speculative_decoding_does_not_match(_on_sm107):
    config = _model_config(_ultra_config(), spec_config=_NGRAM, **_TEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


# --- nemotron_h, BF16 on sm_100 -----------------------------------------------


def _ultra_bf16_config(**overrides):
    """The shape Nemotron-3 Ultra 550B-A55B BF16's own config.json declares: the
    NVFP4 release's, with no quantization_config at all."""
    fields = dict(quantization_config=None)
    fields.update(overrides)
    return _ultra_config(**fields)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_ultra_bf16_tep8_matches(mode, _on_sm100):
    config = _model_config(_ultra_bf16_config(), **_TEP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _ULTRA_BF16_TARGET


def test_ultra_bf16_depth_is_not_part_of_the_identity(_on_sm100):
    config = _model_config(
        _ultra_bf16_config(layers_block_type=["mamba", "moe", "attention"]), **_TEP8
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _ULTRA_BF16_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # the NVFP4 release of the same model: sm_100 has no NVFP4 target
        (
            dict(
                quantization_config=dict(
                    quant_algo="MIXED_PRECISION",
                    kv_cache_scheme=dict(dynamic=False, num_bits=8, type="float"),
                )
            ),
            {},
            r"quantization \('MIXED_PRECISION', 8\)",
        ),
        # the right checkpoint on 4 ranks: 1.1 TB of weights do not fit
        (dict(), _TEP4, "world_size=4"),
        # the right checkpoint with attention data parallel: the replicated non-expert
        # weights would not fit beside the experts
        (dict(), dict(_TEP8, enable_attention_dp=True), "attention_dp=True"),
        # the experts tensor-parallel instead of expert-parallel
        (dict(), dict(_TEP8, moe_ep_size=1, moe_tp_size=8), "moe_ep=1"),
        # Nemotron-3 Super's geometry
        (dict(hidden_size=4096), {}, r"config shape \(4096,"),
        # a dense-MLP block the target does not implement
        (dict(layers_block_type=["mamba", "mlp", "attention"]), {}, "block kinds"),
    ],
)
def test_ultra_bf16_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed, _on_sm100):
    config = _model_config(_ultra_bf16_config(**config_kwargs), **(mapping_kwargs or _TEP8))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("capability, text", [((10, 7), r"\(10, 7\)"), ((10, 3), r"\(10, 3\)")])
def test_ultra_bf16_on_another_architecture_does_not_match(capability, text, monkeypatch):
    """Certification is per architecture: the BF16 target is gated on sm_100 only."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    config = _model_config(_ultra_bf16_config(), **_TEP8)
    with pytest.raises(ValueError, match=rf"compute capability {text}"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_ultra_bf16_with_speculative_decoding_does_not_match(_on_sm100):
    config = _model_config(_ultra_bf16_config(), spec_config=_NGRAM, **_TEP8)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_ultra_nvfp4_does_not_route_on_sm100(_on_sm100):
    """The sm_107 targets stay sm_107's: on an sm_100 device the NVFP4 checkpoint falls back."""
    config = _model_config(_ultra_config(), **_TEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

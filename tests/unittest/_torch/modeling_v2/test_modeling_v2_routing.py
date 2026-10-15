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
from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, NGramDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

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


_SM100 = (10, 0)
_TP16_MOETP4EP4 = dict(world_size=16, tp_size=16, moe_tp_size=4, moe_ep_size=4)
_TP16_MOETP16EP1 = dict(world_size=16, tp_size=16, moe_tp_size=16, moe_ep_size=1)
_K3_TARGET = (
    "models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling."
    "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4"
)
_K3_TARGET_B = (
    "models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling."
    "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1"
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

_K3_NVFP4_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "input_activations": {
                "group_size": 16,
                "num_bits": 4,
                "strategy": "tensor_group",
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "group_size": 16,
                "num_bits": 4,
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "ignore": ["re:.*lm_head.*"],
    "kv_cache_scheme": None,
}


def _k3_config(text_as_dict=False, quantization=_K3_CHECKPOINT_QUANTIZATION, **text_overrides):
    """The shape Kimi K3's own config.json declares: a vision-language wrapper whose language model is
    ``text_config``, with the quantization ``KimiK3Config`` surfaces from it."""
    text = dict(
        model_type="kimi_linear",
        num_hidden_layers=93,
        hidden_size=7168,
        num_experts=896,
        routed_expert_hidden_size=3584,
    )
    text.update(text_overrides)
    fields = dict(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
        text_config=text if text_as_dict else PretrainedConfig(**text),
    )
    if quantization is not None:
        fields["quantization_config"] = quantization
    return PretrainedConfig(**fields)


@pytest.fixture
def _on_sm100(monkeypatch):
    """Route as if this were a GB200."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def test_kimi_k3_config_surfaces_the_text_config_quantization():
    config = KimiK3Config(
        text_config=dict(quantization_config=_K3_CHECKPOINT_QUANTIZATION),
        architectures=["KimiK3ForConditionalGeneration"],
    )
    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        config.quantization_config, "TRTLLM"
    )
    assert layer_quant_config is None
    assert quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
    assert quant_config.group_size == 32


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("text_as_dict", [False, True], ids=["text-config", "text-dict"])
@pytest.mark.parametrize(
    "mapping_kwargs, target",
    [(_TP16_MOETP4EP4, _K3_TARGET), (_TP16_MOETP16EP1, _K3_TARGET_B)],
    ids=["tp16_moetp4ep4", "tp16_moetp16ep1"],
)
def test_kimi_k3_targets_match(mode, text_as_dict, mapping_kwargs, target):
    """Route B is the experts split 16 ways by tensor, set explicitly."""
    config = _model_config(_k3_config(text_as_dict=text_as_dict), **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == target


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping_kwargs,parallel",
    [(_TP16_MOETP4EP4, "Moetp4ep4"), (_TP16_MOETP16EP1, "Moetp16ep1")],
)
def test_kimi_k3_mxfp4_routes_on_sm103(mode, mapping_kwargs, parallel):
    config = _model_config(_k3_config(), **mapping_kwargs)
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert target.__name__ == f"ModelingV2KimiK3Mxfp4Sm103Tp16{parallel}"
    assert target.target_spec.sm == _SM103


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "quantization",
    [_K3_NVFP4_QUANTIZATION, None],
    ids=["nvfp4", "unquantized"],
)
def test_kimi_k3_other_quantizations_do_not_match(quantization):
    """An NVFP4 declaration and an unquantized checkpoint have the MXFP4 checkpoint's shape; the quantization is
    what keeps them out of a target whose loader reads packed MXFP4 experts."""
    config = _model_config(_k3_config(quantization=quantization), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="quantization"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_on_sm100")
@pytest.mark.parametrize(
    "text_overrides, mapping_kwargs, missed",
    [
        # a Kimi-family checkpoint of another depth
        (dict(num_hidden_layers=61), _TP16_MOETP4EP4, "shape"),
        # the expert split left unset: the mapping resolves to moe_tp 16 x moe_ep 1, but the built-in model runs
        # that default as expert parallelism over the 16 ranks, a layout no target serves
        (dict(), dict(world_size=16, tp_size=16), "split_set=False"),
        # experts split 16 ways by expert, set explicitly
        (dict(), dict(world_size=16, tp_size=16, moe_tp_size=1, moe_ep_size=16), "moe_ep=16"),
        # attention data parallelism splits the requests, not the heads
        (dict(), dict(_TP16_MOETP4EP4, enable_attention_dp=True), "attention_dp=True"),
        (dict(), dict(_TP16_MOETP16EP1, enable_attention_dp=True), "attention_dp=True"),
        # pipeline parallelism over the 16 ranks
        (
            dict(),
            dict(world_size=16, tp_size=8, pp_size=2, moe_tp_size=8, moe_ep_size=1),
            "pp=2",
        ),
        # one tray instead of four
        (dict(), dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4), "ws=4"),
    ],
)
def test_kimi_k3_near_misses_do_not_match(text_overrides, mapping_kwargs, missed):
    config = _model_config(_k3_config(**text_overrides), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_kimi_k3_on_another_sm_does_not_match(monkeypatch):
    """The MXFP4 fused decode path is scoped to SM100/SM103."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (10, 7))
    config = _model_config(_k3_config(), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 7\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def _k3_nvfp4_mixed_config():
    """ModelOpt's expert declarations, alongside FP8 attention projections."""
    return {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "quantized_layers": {
                **{
                    f"language_model.model.layers.{layer}.mlp.experts": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    }
                    for layer in range(1, 93)
                },
                "language_model.model.layers.1.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
            },
        },
    }


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("quant_method", ["modelopt", "modelopt_mixed"])
@pytest.mark.parametrize("sm", [103, 107])
def test_kimi_k3_nvfp4_routes(monkeypatch, mode, quant_method, sm):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (sm // 10, sm % 10))
    quantization = _k3_nvfp4_mixed_config()
    if quant_method == "modelopt_mixed":
        quantization = dict(quantization["quantization"], quant_method=quant_method)
    config = _model_config(_k3_config(quantization=quantization), **_TP16_MOETP4EP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == (
        f"models.kimi_k3_vl.kimi_k3_nvfp4__sm_{sm}__tp16_moetp4ep4.modeling."
        f"ModelingV2KimiK3Nvfp4Sm{sm}Tp16Moetp4ep4"
    )


@pytest.mark.parametrize("bad_layer", ["missing", "mxfp4", "wrong_group_size"])
@pytest.mark.parametrize("sm", [103, 107])
def test_kimi_k3_nvfp4_rejects_incomplete_expert_declarations(monkeypatch, bad_layer, sm):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (sm // 10, sm % 10))
    quantization = _k3_nvfp4_mixed_config()
    layers = quantization["quantization"]["quantized_layers"]
    key = "language_model.model.layers.92.mlp.experts"
    if bad_layer == "missing":
        del layers[key]
    elif bad_layer == "mxfp4":
        layers[key] = {"quant_algo": "W4A8_MXFP4_MXFP8", "group_size": 32}
    else:
        layers[key]["group_size"] = 32
    config = _model_config(_k3_config(quantization=quantization), **_TP16_MOETP4EP4)
    with pytest.raises(ValueError):
        modeling_v2_resolve(config, _llm_args("require"))


def test_kimi_k3_nvfp4_rejects_uniformly_quantized_attention(monkeypatch):
    """NVFP4 attention weights are distinct from the supported FP8 attention checkpoint."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (10, 7))
    config = _model_config(_k3_config(quantization=_K3_NVFP4_QUANTIZATION), **_TP16_MOETP4EP4)
    with pytest.raises(ValueError, match="no target fits"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_kimi_k3_nvfp4_declines_excluded_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (10, 7))
    quantization = _k3_nvfp4_mixed_config()
    quantization["quantization"]["exclude_modules"] = ["*.layers.92.mlp.experts"]
    config = _model_config(_k3_config(quantization=quantization), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="no target fits"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_kimi_k3_nvfp4_constructor_uses_per_layer_quantization():
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.modeling import (
        KimiK3MoERuntime,
    )

    moe_runtime = KimiK3MoERuntime

    quant, layers = ModelConfig.load_hf_quant_config(_k3_nvfp4_mixed_config(), "TRTLLM")
    config = SimpleNamespace(quant_config=quant, quant_config_dict=layers)
    for layer_idx in (1, 92):
        resolved = moe_runtime._resolve_routed_quant_config(config, layer_idx)
        assert resolved.quant_algo == QuantAlgo.NVFP4
        assert resolved.group_size == 16
    del layers["language_model.model.layers.92.mlp.experts"]
    with pytest.raises(ValueError, match="Missing NVFP4"):
        moe_runtime._resolve_routed_quant_config(config, 92)


@pytest.mark.parametrize("prefix", ["language_model.model.", "model.", ""])
def test_kimi_k3_nvfp4_experts_honor_excluded_aliases(prefix: str) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.modeling import (
        KimiK3MoERuntime,
    )
    from tensorrt_llm.models.modeling_utils import QuantConfig

    name = f"{prefix}layers.92.mlp.experts"
    config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, exclude_modules=[name]),
        quant_config_dict={name: QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)},
    )
    with pytest.raises(ValueError, match="excluded from quantization"):
        KimiK3MoERuntime._resolve_routed_quant_config(config, 92)


def test_kimi_k3_nvfp4_keeps_mla_dspark_worker():
    """The reference MLA drafter must retain its compressed-context-cache worker."""
    import importlib
    from types import SimpleNamespace

    target = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_nvfp4__sm_107__tp16_moetp4ep4.modeling"
    )
    drafter, worker = object(), object()
    model = SimpleNamespace(
        draft_config=SimpleNamespace(
            pretrained_config=PretrainedConfig(kv_lora_rank=512, qk_rope_head_dim=64)
        ),
        draft_model=drafter,
        spec_worker=worker,
        epilogue=[drafter, worker],
    )
    config = SimpleNamespace(
        spec_config=SimpleNamespace(
            spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
            draft_is_embedded_in_target=False,
        )
    )
    target.ModelingV2KimiK3Nvfp4Sm107Tp16Moetp4ep4._own_spec_branch(model, config)
    assert model.draft_model is drafter
    assert model.spec_worker is worker
    assert model.epilogue == [drafter, worker]


@pytest.mark.parametrize("prefix", ["language_model.model.", "model.", ""])
def test_kimi_k3_nvfp4_attention_quantization_preserves_config(prefix: str) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import modeling as target
    from tensorrt_llm.models.modeling_utils import QuantConfig

    projection = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    global_quant = QuantConfig(
        quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=QuantAlgo.NVFP4
    )
    config = ModelConfig(
        quant_config=global_quant,
        quant_config_dict={f"{prefix}layers.3.self_attn.q_a_proj": projection},
    )
    resolved = target.resolve_attention_quant_config(config, 3, "q_a_proj")
    assert resolved.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert resolved.group_size == 128
    assert resolved.kv_cache_quant_algo == QuantAlgo.NVFP4
    assert projection.kv_cache_quant_algo is None
    assert target.resolve_attention_quant_config(config, 3, "g_proj").quant_algo is None


def test_kimi_k3_nvfp4_attention_honors_exclusions() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import modeling as target
    from tensorrt_llm.models.modeling_utils import QuantConfig

    config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION,
            kv_cache_quant_algo=QuantAlgo.NVFP4,
            exclude_modules=["*.g_proj"],
        ),
        quant_config_dict={
            "model.layers.3.self_attn.g_proj": QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
        },
    )
    resolved = target.resolve_attention_quant_config(config, 3, "g_proj")
    assert resolved.quant_algo is None
    assert resolved.kv_cache_quant_algo == QuantAlgo.NVFP4


@pytest.mark.parametrize(
    "algo,group_size", [(QuantAlgo.NVFP4, 16), (QuantAlgo.FP8_BLOCK_SCALES, 64)]
)
def test_kimi_k3_nvfp4_attention_rejects_unsupported_quantization(
    algo: QuantAlgo, group_size: int
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import modeling as target
    from tensorrt_llm.models.modeling_utils import QuantConfig

    config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={
            "model.layers.3.self_attn.q_a_proj": QuantConfig(quant_algo=algo, group_size=group_size)
        },
    )
    with pytest.raises(ValueError, match="quantization|128x128"):
        target.resolve_attention_quant_config(config, 3, "q_a_proj")


def test_kimi_k3_nvfp4_attention_rejects_conflicting_block_sizes() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import modeling as target
    from tensorrt_llm.models.modeling_utils import QuantConfig

    config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={
            f"{prefix}layers.3.self_attn.q_a_proj": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size
            )
            for prefix, group_size in (("language_model.model.", 128), ("model.", 64))
        },
    )
    with pytest.raises(ValueError, match="Conflicting"):
        target.resolve_attention_quant_config(config, 3, "q_a_proj")

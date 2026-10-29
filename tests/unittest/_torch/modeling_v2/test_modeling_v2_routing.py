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
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import (
    DSparkDecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_TARGET = "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling.ModelingV2GptOss120bSm103Tp1"
_R1_TARGET = (
    "models.deepseek_v3.r1_0528_nvfp4__sm_103__dep4.modeling.ModelingV2DeepseekR10528Nvfp4Sm103Dep4"
)
_MISTRAL_LARGE4_NVFP4_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Tep4"
)
_MISTRAL_LARGE4_NVFP4_DEP4_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Dep4"
)
_MISTRAL_LARGE4_NVFP4_TP4_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Tp4"
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


_ML4_LAYERS = 72
#: Mistral Large 4's per-layer structure: every fourth layer NoPE, the NoPE layers past 35
#: reading layer 35's latent, an indexer on the first 17 layers and then on each pair
#: following two that reuse its selection.
_ML4_NOPE = [i % 4 == 3 for i in range(_ML4_LAYERS)]
_ML4_KV_SOURCE = [35 if i > 35 and i % 4 == 3 else i for i in range(_ML4_LAYERS)]
_ML4_INDEX_SHARING = ["full" if i < 17 or i % 4 in (0, 3) else "shared" for i in range(_ML4_LAYERS)]


def _ml4_mixed_precision_layers() -> dict:
    """The NVFP4 export's `quantized_layers`, written out from its module structure: NVFP4 group
    16 for the routed experts of layers 1..71; FP8_PB_WO for every layer's wq and wo, the full
    indexers' wq_b, the shared experts' w1/w2/w3 and layer 0's dense w1/w2/w3."""
    fp8 = {"quant_algo": "FP8_PB_WO"}
    layers = {}
    for i in range(_ML4_LAYERS):
        if i:
            layers[f"layers.{i}.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
        layers[f"layers.{i}.attention.wq"] = dict(fp8)
        layers[f"layers.{i}.attention.wo"] = dict(fp8)
        if _ML4_INDEX_SHARING[i] == "full":
            layers[f"layers.{i}.attention.indexer.wq_b"] = dict(fp8)
        mlp = "shared_experts" if i else "feed_forward"
        for w in ("w1", "w2", "w3"):
            layers[f"layers.{i}.{mlp}.{w}"] = dict(fp8)
    assert len(layers) == 475
    return layers


def _mistral_large4_nvfp4_config(**overrides):
    """The fields Mistral-Large-4-1T-A52B-NVFP4's configuration, adapted from its params.json,
    declares that routing reads: the architecture, YaRN factor 2048 over 1048576 positions, and
    its ModelOpt MIXED_PRECISION quantization."""
    fields = dict(
        architectures=["MistralLarge4ForCausalLM"],
        model_type="mistral_large_4",
        num_hidden_layers=_ML4_LAYERS,
        hidden_size=6144,
        num_attention_heads=64,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        moe_intermediate_size=6144,
        intermediate_size=24576,
        first_k_dense_replace=1,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        vocab_size=204800,
        # transformers validates the YaRN factor against it.
        max_position_embeddings=1048576,
        nope_layer_flags=list(_ML4_NOPE),
        indexer_nope_layer_flags=list(_ML4_NOPE),
        kv_source_layers=list(_ML4_KV_SOURCE),
        index_sharing=list(_ML4_INDEX_SHARING),
        ssmax_scaling=True,
        indexer_ssmax_scaling=True,
        lerope=True,
        softmax_tanh_gating=30.0,
        scoring_func="softmax",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        topk_method=None,
        rope_parameters=dict(
            rope_type="yarn",
            attention_factor=1.0,
            beta_fast=32.0,
            beta_slow=1.0,
            factor=2048.0,
            mscale_all_dim=1,
            original_max_position_embeddings=512,
            rope_theta=10000.0,
        ),
        quantization_config=dict(
            group_size=16,
            ignore=[],
            quant_algo="MIXED_PRECISION",
            quant_method="modelopt",
            quantized_layers=_ml4_mixed_precision_layers(),
        ),
        fp8_kv_cache=True,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _ml4_nvfp4_quant(**overrides) -> dict:
    """The NVFP4 export's `quantization_config` with `overrides` applied."""
    return dict(_mistral_large4_nvfp4_config().quantization_config, **overrides)


def _ml4_layers_with(**changes) -> dict:
    """The NVFP4 export's map with module entries changed (a value of None drops it)."""
    layers = _ml4_mixed_precision_layers()
    for key, value in changes.items():
        key = key.replace("__", ".")
        if value is None:
            del layers[key]
        else:
            layers[key] = value
    return layers


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
_TEP4 = dict(_DEP4, enable_attention_dp=False)
_TP4 = dict(_TEP4, moe_ep_size=1, moe_tp_size=4)
# Eight ranks over two nodes of four, as the engine resolves them (gpus_per_node is the visible
# device count of rank 0's node): tep4's layout over eight ranks, the tep8 deployment.
_TEP8 = dict(_TEP4, world_size=8, tp_size=8, moe_ep_size=8, gpus_per_node=4)
# The same eight ranks with the routed experts tensor parallel (tp8): every expert on every rank,
# cut by width. Stated, as options/core.yaml states it, and bare, as `--tp_size=8` leaves it (the
# mapping resolves the unset MoE split to moe_tp 8, moe_ep 1).
_TP8 = dict(_TEP8, moe_ep_size=1, moe_tp_size=8)
_TP8_BARE = dict(world_size=8, tp_size=8, gpus_per_node=4)
# Eight ranks over two nodes of four with attention data parallel (dep8): dep4 over eight ranks.
_DEP8 = dict(_DEP4, world_size=8, tp_size=8, moe_ep_size=8, gpus_per_node=4)


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


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103 ----------------------------------------


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tep4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TARGET


def _flip(values, i):
    flipped = list(values)
    flipped[i] = not flipped[i]
    return flipped


def test_mistral_large4_rope_factor_and_positions_are_not_criteria():
    """The YaRN factor and max_position_embeddings are configuration the rope table is built
    from: the NVFP4 export at YaRN factor 512 over 262144 positions is still the NVFP4 target."""
    config = _mistral_large4_nvfp4_config(
        max_position_embeddings=262144,
        rope_parameters=dict(_mistral_large4_nvfp4_config().rope_parameters, factor=512.0),
    )
    resolved = modeling_v2_resolve(_model_config(config, **_TEP4), _llm_args("require"))
    assert _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # another trunk depth, expert count or a q-LoRA query path: other declared weights
        (dict(num_hidden_layers=61), _TEP4, r"config shape \(61, 6144"),
        (
            dict(n_routed_experts=256),
            _TEP4,
            r"config shape \(72, 6144, 64, None, 512, 64, 64, 128, 256,",
        ),
        (dict(q_lora_rank=1536), _TEP4, r"config shape \(72, 6144, 64, 1536,"),
        # a RoPE layer turned NoPE, a sharer reading its own latent, an indexer layer turned
        # shared: each is another forward
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _TEP4, "pattern differs"),
        (dict(indexer_nope_layer_flags=_flip(_ML4_NOPE, 3)), _TEP4, "pattern differs"),
        (
            dict(kv_source_layers=[i for i in range(_ML4_LAYERS)]),
            _TEP4,
            "pattern differs",
        ),
        (dict(index_sharing=["shared"] + _ML4_INDEX_SHARING[1:]), _TEP4, "pattern differs"),
        # the math the forward carries
        (dict(ssmax_scaling=False), _TEP4, "switches .* differ"),
        (dict(softmax_tanh_gating=None), _TEP4, "switches .* differ"),
        (dict(fp8_kv_cache=False), _TEP4, "switches .* differ"),
        (dict(scoring_func="sigmoid"), _TEP4, "switches .* differ"),
        (
            dict(
                rope_parameters=dict(
                    _mistral_large4_nvfp4_config().rope_parameters, mscale=1.0, mscale_all_dim=0
                )
            ),
            _TEP4,
            "switches .* differ",
        ),
        # the quantization: method, algorithm, block size and the per-module map, as data
        (
            dict(quantization_config=_ml4_nvfp4_quant(quant_method="fp8")),
            _TEP4,
            "quantization differs",
        ),
        (
            dict(quantization_config=_ml4_nvfp4_quant(quant_algo="NVFP4")),
            _TEP4,
            "quantization differs",
        ),
        (dict(quantization_config=_ml4_nvfp4_quant(group_size=32)), _TEP4, "quantization differs"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(quantized_layers=None)),
            _TEP4,
            "quantization differs",
        ),
        # one MoE layer's routed experts in FP8 blocks
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _TEP4,
            "quantization differs",
        ),
        # NVFP4 at another block size on one layer
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__71__experts={"quant_algo": "NVFP4", "group_size": 32}
                    )
                )
            ),
            _TEP4,
            "quantization differs",
        ),
        # a shared expert left in bf16, and a bf16 module quantized
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(layers__9__shared_experts__w2=None)
                )
            ),
            _TEP4,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__3__attention__wkv_b={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _TEP4,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_on_sm100_does_not_match(monkeypatch):
    """The sparse-MLA kernels are sm103-only, so a B200 is not this target."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TEP4
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, attention data parallel ----------------


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_dep4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode)))
        == _MISTRAL_LARGE4_NVFP4_DEP4_TARGET
    )


def test_mistral_large4_nvfp4_lands_on_its_topologys_target():
    """One checkpoint, two targets, chosen by attention DP alone: off is tep4, on is dep4."""
    config = _mistral_large4_nvfp4_config()
    tep4 = modeling_v2_resolve(_model_config(config, **_TEP4), _llm_args("require"))
    dep4 = modeling_v2_resolve(_model_config(config, **_DEP4), _llm_args("require"))
    assert _qualified(tep4) == _MISTRAL_LARGE4_NVFP4_TARGET
    assert _qualified(dep4) == _MISTRAL_LARGE4_NVFP4_DEP4_TARGET


def test_mistral_large4_tep4_branch_refuses_attention_dp():
    """The tep4 target is never returned for an attention-DP mapping."""
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    resolved = modeling_v2_resolve(config, _llm_args("auto"))
    assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # lm_head tensor parallel under attention DP: the shell's lm_head is vocab-parallel
        (dict(), dict(_DEP4, enable_lm_head_tp_in_adp=True), "lm_head_tp_in_adp=True"),
        # another topology with attention DP on
        (
            dict(),
            dict(_DEP4, world_size=8, tp_size=8, moe_ep_size=4, moe_tp_size=2),
            "world_size=8 tp=8 pp=1 moe_ep=4 moe_tp=2",
        ),
        (dict(), dict(_DEP4, moe_ep_size=2, moe_tp_size=2), "moe_ep=2 moe_tp=2"),
        (
            dict(),
            dict(_DEP4, world_size=8, pp_size=2),
            "world_size=8 tp=4 pp=2",
        ),
        # the architecture and the quantization, as at tep4
        (dict(num_hidden_layers=61), _DEP4, r"config shape \(61, 6144"),
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _DEP4, "pattern differs"),
        (dict(ssmax_scaling=False), _DEP4, "switches .* differ"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(group_size=32)),
            _DEP4,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _DEP4,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_dep4_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_dep4_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_dep4_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_DEP4
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, tensor-parallel experts -----------------


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tp4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP4)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TP4_TARGET
    )


def test_mistral_large4_nvfp4_tp4_from_tensor_parallel_size_alone():
    """A deployment that states only tensor_parallel_size 4 leaves both MoE sizes unset; the
    mapping then gives every rank all the experts, tensor parallel (moe_tp 4, moe_ep 1), and
    that lands on the tp4 target."""
    mapping = Mapping(world_size=4, tp_size=4)
    assert (mapping.moe_tp_size, mapping.moe_ep_size) == (4, 1)
    config = ModelConfig(pretrained_config=_mistral_large4_nvfp4_config(), mapping=mapping)
    resolved = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TP4_TARGET


def test_mistral_large4_nvfp4_lands_on_each_topologys_target():
    """One checkpoint, three targets, chosen by the parallel topology alone: experts split by
    expert is tep4 (attention DP off) or dep4 (on), experts split by width is tp4."""
    config = _mistral_large4_nvfp4_config()
    landed = {
        name: _qualified(modeling_v2_resolve(_model_config(config, **m), _llm_args("require")))
        for name, m in (("tep4", _TEP4), ("dep4", _DEP4), ("tp4", _TP4))
    }
    assert landed == {
        "tep4": _MISTRAL_LARGE4_NVFP4_TARGET,
        "dep4": _MISTRAL_LARGE4_NVFP4_DEP4_TARGET,
        "tp4": _MISTRAL_LARGE4_NVFP4_TP4_TARGET,
    }


def test_mistral_large4_expert_parallel_branches_refuse_tp4():
    """No expert-parallel target is ever returned for the tp4 mapping, and the tp4 target for
    none of the expert-parallel mappings."""
    config = _mistral_large4_nvfp4_config()
    resolved = modeling_v2_resolve(_model_config(config, **_TP4), _llm_args("auto"))
    assert resolved is None or _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TP4_TARGET
    for mapping in (_TEP4, _DEP4):
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("auto"))
        assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TP4_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a mixed split of the experts: half the experts on each rank, each at half its width
        (dict(), dict(_TP4, moe_tp_size=2, moe_ep_size=2), "moe_ep=2 moe_tp=2"),
        # tensor-parallel experts under attention DP: another attention
        (dict(), dict(_TP4, enable_attention_dp=True), "attention_dp=True"),
        # another width split or another world
        (
            dict(),
            dict(_TP4, world_size=8, tp_size=8, moe_tp_size=4, moe_ep_size=2),
            "world_size=8 tp=8 pp=1 moe_ep=2 moe_tp=4",
        ),
        (dict(), dict(_TP4, world_size=8, pp_size=2), "world_size=8 tp=4 pp=2"),
        # the architecture and the quantization, as at tep4
        (dict(num_hidden_layers=61), _TP4, r"config shape \(61, 6144"),
        (
            dict(moe_intermediate_size=3072),
            _TP4,
            r"config shape \(72, 6144, 64, None, 512, 64, 64, 128, 128, 4, 1, 3072,",
        ),
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _TP4, "pattern differs"),
        (dict(ssmax_scaling=False), _TP4, "switches .* differ"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(group_size=32)),
            _TP4,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _TP4,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_tp4_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_tp4_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_tp4_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TP4
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, eight ranks over two nodes ---------------


def _resolves_to(config, mapping) -> str | None:
    """The target `require` returns for the deployment, or None where it raises."""
    try:
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("require"))
    except ValueError:
        return None
    return _qualified(resolved)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tep8_matches_the_tep4_target(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TARGET


def test_mistral_large4_nvfp4_lands_on_its_topologys_target_at_every_world():
    """One checkpoint, three targets, chosen by the topology alone and not by the rank count:
    the experts split by expert is the tep4 target (attention DP off) or the dep4 target (on),
    split by width (stated, or tensor_parallel_size 8 alone) the tp4 target, on four ranks and
    on eight over two nodes of four alike."""
    config = _mistral_large4_nvfp4_config()
    for mapping, target in (
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
        (_TP4, _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
        (_TEP8, _MISTRAL_LARGE4_NVFP4_TARGET),
        (_TP8, _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
        (_TP8_BARE, _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
    ):
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("require"))
        assert _qualified(resolved) == target, mapping


def test_mistral_large4_eight_ranks_land_where_four_ranks_do():
    """Each eight-rank mapping lands on the target of its four-rank counterpart."""
    config = _mistral_large4_nvfp4_config()
    for four, eight in ((_TEP4, _TEP8), (_TP4, _TP8), (_DEP4, _DEP8)):
        assert _resolves_to(config, eight) == _resolves_to(config, four), eight


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        (dict(_TEP4, world_size=2, tp_size=2, moe_ep_size=2), "world_size=2 tp=2 pp=1 moe_ep=2"),
        (
            dict(_TEP4, world_size=16, tp_size=16, moe_ep_size=16),
            "world_size=16 tp=16 pp=1 moe_ep=16",
        ),
        (dict(_TP4, world_size=2, tp_size=2, moe_tp_size=2), "world_size=2 tp=2 pp=1 moe_ep=1"),
        (
            dict(_TP4, world_size=16, tp_size=16, moe_tp_size=16),
            "world_size=16 tp=16 pp=1 moe_ep=1 moe_tp=16",
        ),
        (dict(_DEP4, world_size=2, tp_size=2, moe_ep_size=2), "world_size=2 tp=2 pp=1 moe_ep=2"),
        (
            dict(_DEP4, world_size=16, tp_size=16, moe_ep_size=16),
            "world_size=16 tp=16 pp=1 moe_ep=16",
        ),
    ],
)
def test_mistral_large4_nvfp4_other_rank_counts_do_not_match(mapping_kwargs, missed):
    """The NVFP4 targets serve four ranks or eight: the same layouts on two ranks or sixteen are
    refused, and the refusal states the rank counts each target serves."""
    config = _model_config(_mistral_large4_nvfp4_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    assert "serves 4 ranks on one node or 8 ranks over two nodes of four" in str(excinfo.value)


def test_mistral_large4_refusal_states_the_ranks_each_target_serves():
    """The refusal names every target with the four- and eight-rank mappings it serves: each
    target on 4 ranks on one node or 8 over two nodes of four, with the eight-rank deployment's
    name beside its mapping."""
    config = _model_config(
        _mistral_large4_nvfp4_config(), **dict(_TEP8, tp_size=4, pp_size=2, moe_ep_size=4)
    )
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert (
        "the mistral_large4 targets are Mistral-Large-4-1T-A52B-NVFP4 (ModelOpt MIXED_PRECISION: "
        "NVFP4 routed experts, FP8_PB_WO linears, group 16, its 475-module map) on sm103 at tep4, "
        "at dep4 and at tp4" in message
    )
    assert "each target serves 4 ranks on one node or 8 ranks over two nodes of four" in message
    for served in (
        "world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8",
        "world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: dep8",
        "world_size=4 tp=4 pp=1 moe_ep=1 moe_tp=4, or world_size=8 tp=8 pp=1 moe_ep=1 moe_tp=8: tp8",
    ):
        assert served in message, served


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a mixed split of the experts over the eight ranks: another weight layout
        (dict(), dict(_TEP8, moe_ep_size=4, moe_tp_size=2), "moe_ep=4 moe_tp=2"),
        # eight ranks as two pipeline stages of tep4
        (dict(), dict(_TEP8, tp_size=4, pp_size=2, moe_ep_size=4), "world_size=8 tp=4 pp=2"),
        # the architecture and the quantization, as at tep4
        (dict(num_hidden_layers=61), _TEP8, r"config shape \(61, 6144"),
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _TEP8, "pattern differs"),
        (dict(ssmax_scaling=False), _TEP8, "switches .* differ"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(group_size=32)),
            _TEP8,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _TEP8,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_tep8_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    # The refusal names the eight-rank deployment the tep4 target serves.
    message = str(excinfo.value)
    assert "at tep4" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8" in message


def test_mistral_large4_nvfp4_tep8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_tep8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TEP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, eight ranks, tensor-parallel experts -----


@pytest.mark.parametrize("mapping", [_TP8, _TP8_BARE], ids=["stated", "tp_size_8_alone"])
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tp8_matches_the_tp4_target(mode, mapping):
    model_config = _model_config(_mistral_large4_nvfp4_config(), **mapping)
    assert (model_config.mapping.moe_tp_size, model_config.mapping.moe_ep_size) == (8, 1)
    assert (
        _qualified(modeling_v2_resolve(model_config, _llm_args(mode)))
        == _MISTRAL_LARGE4_NVFP4_TP4_TARGET
    )


def test_mistral_large4_tp4_target_refuses_every_other_topology_at_eight_ranks():
    """The tp4 target is returned for no expert-parallel mapping, at four ranks or eight, and
    for no mixed split of the experts over eight ranks."""
    others = (
        _TEP4,
        _DEP4,
        _TEP8,
        dict(_TEP8, enable_attention_dp=True),
        dict(_TP8, moe_ep_size=2, moe_tp_size=4),
        dict(_TP8, moe_ep_size=4, moe_tp_size=2),
    )
    config = _mistral_large4_nvfp4_config()
    for mapping in others:
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("auto"))
        assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TP4_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a mixed split of the experts over the eight ranks: another weight layout
        (dict(), dict(_TP8, moe_ep_size=4, moe_tp_size=2), "moe_ep=4 moe_tp=2"),
        (dict(), dict(_TP8, moe_ep_size=2, moe_tp_size=4), "moe_ep=2 moe_tp=4"),
        # attention data parallel over eight ranks: the whole attention on every rank
        (dict(), dict(_TP8, enable_attention_dp=True), "attention_dp=True"),
        # eight ranks as two pipeline stages of tp4
        (dict(), dict(_TP8, tp_size=4, pp_size=2, moe_tp_size=4), "world_size=8 tp=4 pp=2"),
        # the architecture, the switches and the quantization, as at tep8
        (dict(num_hidden_layers=61), _TP8, r"config shape \(61, 6144"),
        (
            dict(moe_intermediate_size=4096),
            _TP8,
            r"config shape \(72, 6144, 64, None, 512, 64, 64, 128, 128, 4, 1, 4096,",
        ),
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _TP8, "pattern differs"),
        (dict(index_sharing=["shared"] + _ML4_INDEX_SHARING[1:]), _TP8, "pattern differs"),
        (dict(ssmax_scaling=False), _TP8, "switches .* differ"),
        (dict(fp8_kv_cache=False), _TP8, "switches .* differ"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(group_size=32)),
            _TP8,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _TP8,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_tp8_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    # The refusal names the width-split eight-rank deployment the tp4 target serves, beside the
    # expert-split one the tep4 target serves.
    message = str(excinfo.value)
    assert "at tp4" in message and "world_size=8 tp=8 pp=1 moe_ep=1 moe_tp=8: tp8" in message
    assert "at tep4" in message and "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8" in message


def test_mistral_large4_nvfp4_tp8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_tp8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, attention data parallel over eight ranks ----


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_dep8_matches_the_dep4_target(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP8)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode)))
        == _MISTRAL_LARGE4_NVFP4_DEP4_TARGET
    )


def test_mistral_large4_nvfp4_lands_on_its_topologys_target_at_four_and_eight_ranks():
    """One checkpoint, one target per topology: tep4 and dep4 on four ranks, and attention DP
    over eight ranks is the dep4 target too."""
    config = _mistral_large4_nvfp4_config()
    for mapping, target in (
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
    ):
        assert _resolves_to(config, mapping) == target, mapping


def test_mistral_large4_eight_ranks_without_attention_dp_never_reach_dep4():
    """Eight ranks with attention DP off (tep8), or tensor_parallel_size 8 alone (every expert
    sharded eight ways, moe_tp 8), are other weight layouts: whatever routing returns for them,
    it is not the dep4 target -- the NVFP4 export lands on tep4 and tp4, and tensor-parallel
    experts under attention DP on nothing. The dep4 target is returned for no four-rank mapping
    without attention DP."""
    config = _mistral_large4_nvfp4_config()
    tep8 = dict(_DEP8, enable_attention_dp=False)
    for mapping, landed in (
        (tep8, _MISTRAL_LARGE4_NVFP4_TARGET),
        (dict(tep8, moe_ep_size=1, moe_tp_size=8), _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
        (dict(_DEP8, moe_ep_size=1, moe_tp_size=8), None),
    ):
        resolved = _resolves_to(config, mapping)
        assert resolved != _MISTRAL_LARGE4_NVFP4_DEP4_TARGET, mapping
        assert resolved == landed, mapping
    for mapping in (_TEP4, _TP4):
        assert _resolves_to(config, mapping) != _MISTRAL_LARGE4_NVFP4_DEP4_TARGET, mapping
    assert _resolves_to(config, _DEP8) == _MISTRAL_LARGE4_NVFP4_DEP4_TARGET


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # lm_head tensor parallel under attention DP: the shell's lm_head is vocab-parallel
        (dict(), dict(_DEP8, enable_lm_head_tp_in_adp=True), "lm_head_tp_in_adp=True"),
        # the routed experts split another way over the eight ranks
        (dict(), dict(_DEP8, moe_ep_size=4, moe_tp_size=2), "moe_ep=4 moe_tp=2"),
        # eight ranks as two pipeline stages of dep4
        (dict(), dict(_DEP8, tp_size=4, pp_size=2, moe_ep_size=4), "world_size=8 tp=4 pp=2"),
        # the architecture and the quantization, as at dep4
        (dict(num_hidden_layers=61), _DEP8, r"config shape \(61, 6144"),
        (dict(nope_layer_flags=_flip(_ML4_NOPE, 0)), _DEP8, "pattern differs"),
        (dict(ssmax_scaling=False), _DEP8, "switches .* differ"),
        (
            dict(quantization_config=_ml4_nvfp4_quant(group_size=32)),
            _DEP8,
            "quantization differs",
        ),
        (
            dict(
                quantization_config=_ml4_nvfp4_quant(
                    quantized_layers=_ml4_layers_with(
                        layers__5__experts={"quant_algo": "FP8_PB_WO"}
                    )
                )
            ),
            _DEP8,
            "quantization differs",
        ),
    ],
)
def test_mistral_large4_nvfp4_dep8_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_mistral_large4_nvfp4_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    # The refusal names the eight-rank deployment the dep4 target serves.
    message = str(excinfo.value)
    assert "at dep4" in message and "8 ranks over two nodes of four" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: dep8" in message


def test_mistral_large4_nvfp4_dep8_with_lm_head_tp_names_the_dep4_target():
    """lm_head tensor parallel under attention DP at eight ranks is refused, and the message
    names the dep4 target's eight-rank deployment and the switch that missed."""
    config = _model_config(
        _mistral_large4_nvfp4_config(), **dict(_DEP8, enable_lm_head_tp_in_adp=True)
    )
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1" in message
    assert "lm_head_tp_in_adp=True" in message
    assert "at dep4" in message and "attention data parallel with lm_head TP off" in message
    assert "moe_ep=8 moe_tp=1: dep8" in message


def test_mistral_large4_nvfp4_dep8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_mistral_large4_nvfp4_dep8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_DEP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize(
    "parallel,target",
    [
        ("tep", _MISTRAL_LARGE4_NVFP4_TARGET),
        ("dep", _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
        ("tp", _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
    ],
)
def test_mistral_large4_native_dspark_routes_each_topology(tmp_path, world, parallel, target):
    import json

    (tmp_path / "params.json").write_text(
        json.dumps(
            {
                "vocab_size": 204800,
                "mtp": {
                    "n_layers": 3,
                    "depth": 3,
                    "target_activation_layers": "1,18,36,54,71",
                    "ragged_attention": "2048,2048,2048",
                    "norm_strategy": "eagle",
                    "dspark": {
                        "markov_rank": 512,
                        "head_type": "rnn",
                        "draft_input": "position_specific",
                    },
                },
            }
        )
    )
    spec = DSparkDecodingConfig(max_draft_len=3)
    from tensorrt_llm._torch.models.checkpoints.dspark_config import resolve_dspark_checkpoint

    resolve_dspark_checkpoint(spec, str(tmp_path))
    config = _model_config(
        _mistral_large4_nvfp4_config(),
        spec_config=spec,
        world_size=world,
        tp_size=world,
        gpus_per_node=4,
        moe_ep_size=1 if parallel == "tp" else world,
        moe_tp_size=world if parallel == "tp" else 1,
        enable_attention_dp=parallel == "dep",
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == target
    # Reject a different training contract rather than silently drafting with
    # the target's hard-coded block/window interpretation.
    spec._draft_model_config["norm_strategy"] = "none"
    with pytest.raises(ValueError, match="native DSpark"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral Large 4 vision encoders ----------------------------------------------------------------

# The vision encoder and projector geometry the Mistral Large 4 checkpoint declares.
_ML4_VISION = dict(
    num_hidden_layers=30,
    hidden_size=2048,
    num_attention_heads=16,
    intermediate_size=6144,
    num_channels=3,
    image_size=1540,
    patch_size=14,
    rope_theta=10000.0,
    spatial_merge_size=2,
    mm_projector_id="patch_merge",
    adapter_bias=False,
    add_pre_mm_projector_layer_norm=True,
    image_token_id=10,
    image_break_token_id=12,
    image_end_token_id=13,
)


@pytest.mark.parametrize(
    "mapping, target",
    [
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP4_TARGET),
        (_TP4, _MISTRAL_LARGE4_NVFP4_TP4_TARGET),
    ],
    ids=["tep4", "dep4", "tp4"],
)
def test_mistral_large4_nvfp4_with_a_vision_encoder_serves_images(mapping, target):
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.routing import (
        serves_images,
    )

    pretrained_config = _mistral_large4_nvfp4_config(vision_config=SimpleNamespace(**_ML4_VISION))
    config = _model_config(pretrained_config, **mapping)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == target
    assert serves_images(pretrained_config)
    assert not serves_images(_mistral_large4_nvfp4_config())


def test_mistral_large4_nvfp4_with_another_vision_encoder_does_not_match():
    vision = SimpleNamespace(**dict(_ML4_VISION, patch_size=16))
    config = _model_config(_mistral_large4_nvfp4_config(vision_config=vision), **_TEP4)
    with pytest.raises(ValueError, match="vision encoder or projector geometry"):
        modeling_v2_resolve(config, _llm_args("require"))

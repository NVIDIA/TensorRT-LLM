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
* a ``@disable_v2_by_default`` target is built only when the deployment sets
  ``modeling_v2`` itself; left at its default, the built-in implementation runs.
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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    _RESOLVED_MODES,
    _ROUTES,
    ModelingV2Mode,
    NoTarget,
    disable_v2_by_default,
    modeling_v2_resolve,
    resolved_mode,
)
from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_BUILTIN_GPT_OSS = "tensorrt_llm._torch.models.modeling_gpt_oss"
_GPT_OSS_DEFAULT_TARGET = (
    "models.gpt_oss.gpt_oss_120b__sm_10x__default.modeling.ModelingV2GptOss120bSm10xDefault"
)
_R1_DEFAULT_TARGET = (
    "models.deepseek_v3.r1_0528_nvfp4__sm_10x__default.modeling."
    "ModelingV2DeepseekR10528Nvfp4Sm10xDefault"
)
_GLM52_TARGET = (
    "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.modeling.ModelingV2Glm52Nvfp4Sm103Tep4"
)
_MISTRAL_LARGE4_NVFP4_TEP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Tep"
)
_MISTRAL_LARGE4_NVFP4_DEP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Dep"
)
_MISTRAL_LARGE4_NVFP4_TP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm103Tp"
)
_MISTRAL_LARGE4_NVFP4_SM107_TEP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_107__tep.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm107Tep"
)
_MISTRAL_LARGE4_NVFP4_SM107_DEP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm107Dep"
)
_MISTRAL_LARGE4_NVFP4_SM107_TP_TARGET = (
    "models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp.modeling."
    "ModelingV2MistralLarge4Nvfp4Sm107Tp"
)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True, params=[(10, 0), (10, 3)], ids=["sm100", "sm103"])
def _on_sm10x(request, monkeypatch):
    """Route as if this were each GPU the targets are certified on, wherever the
    test actually runs: every case holds on sm_100 and on sm_103. Nodes are
    GB200/GB300 trays of one NVLink domain unless a test says otherwise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: request.param)
    _set_mnnvl(monkeypatch, True)
    return request.param


def _set_mnnvl(monkeypatch, available: bool) -> None:
    """Whether the nodes share multi-node NVLink, as `MNNVLAllReduce.is_mnnvl` reports."""
    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *a, **k: available))


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


def _glm52_config(**overrides):
    """The shape GLM-5.2-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        hidden_size=6144,
        n_routed_experts=256,
        q_lora_rank=2048,
        index_n_heads=32,
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


# Stands for "whatever this checkpoint's own hf_quant_config declares".
_CHECKPOINT_KV = object()


def _model_config(
    pretrained_config, spec_config=None, kv_cache_quant_algo=_CHECKPOINT_KV, **mapping_kwargs
):
    """The loaded config. Its KV-cache algorithm is the checkpoint's own: FP8 for
    DeepSeek-R1-0528-NVFP4, GLM-5.2-NVFP4 and Mistral-Large-4-1T-A52B-NVFP4, none (a bf16
    pool) for the others."""
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    if kv_cache_quant_algo is _CHECKPOINT_KV:
        fp8_kv = pretrained_config.architectures[0] in (
            "DeepseekV3ForCausalLM",
            "GlmMoeDsaForCausalLM",
            "MistralLarge4ForCausalLM",
        )
        kv_cache_quant_algo = QuantAlgo.FP8 if fp8_kv else None
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec_config,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo),
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
        # The settings the target pins (`modeling._TEXT_SETTINGS`), as the normalized config
        # carries them: rope theta and partial factor flattened to the top level.
        norm_topk_prob=True,
        num_experts_per_tok=10,
        hidden_act="silu",
        num_key_value_heads=2,
        head_dim=256,
        attention_bias=False,
        output_gate_type="sigmoid",
        tie_word_embeddings=False,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        rope_scaling=dict(type="mrope", mrope_section=[11, 11, 10], mrope_interleaved=True),
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


_MTP3 = MTPDecodingConfig(max_draft_len=3)
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


def test_off_resolves_nothing():
    """``off`` must be indistinguishable from modeling_v2 not existing, even for a deployment a target claims."""
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None


class _TargetUsedByDefault:
    """A target the default mode builds."""


@disable_v2_by_default
class _TargetDisabledByDefault:
    """A target the default mode declines."""


def _claimed_by(monkeypatch, target):
    """A config whose architecture's ``route`` claims every deployment for ``target``."""
    arch = "ModelingV2TestForCausalLM"
    monkeypatch.setitem(_ROUTES, arch, lambda *_: target)
    return _model_config(PretrainedConfig(architectures=[arch]))


def test_the_default_mode_builds_a_claiming_target(monkeypatch):
    config = _claimed_by(monkeypatch, _TargetUsedByDefault)
    assert modeling_v2_resolve(config, TorchLlmArgs(model=_DUMMY_MODEL)) is _TargetUsedByDefault


def test_the_default_mode_declines_a_target_disabled_by_default(monkeypatch):
    config = _claimed_by(monkeypatch, _TargetDisabledByDefault)
    assert modeling_v2_resolve(config, TorchLlmArgs(model=_DUMMY_MODEL)) is None


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_a_set_mode_builds_a_target_disabled_by_default(monkeypatch, mode):
    """Setting ``modeling_v2`` is the opt-in, including to the value it defaults to."""
    config = _claimed_by(monkeypatch, _TargetDisabledByDefault)
    assert modeling_v2_resolve(config, _llm_args(mode)) is _TargetDisabledByDefault


@pytest.mark.parametrize(
    "pretrained_config, expected",
    [(_gpt_oss_config, _GPT_OSS_DEFAULT_TARGET), (_r1_config, _R1_DEFAULT_TARGET)],
    ids=["gpt_oss", "deepseek_r1"],
)
def test_gpt_oss_and_r1_run_builtin_unless_modeling_v2_is_set(pretrained_config, expected):
    """Both default targets are ``@disable_v2_by_default``: a deployment they claim keeps the built-in model
    until it sets ``modeling_v2``."""
    config = _model_config(pretrained_config(), **_layout(4, False))
    assert modeling_v2_resolve(config, TorchLlmArgs(model=_DUMMY_MODEL)) is None
    assert _qualified(modeling_v2_resolve(config, _llm_args("auto"))) == expected


def test_off_still_reaches_the_builtin_implementation():
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config, _llm_args("off")) is None
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == _BUILTIN_GPT_OSS


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(mode):
    """One GPU has no specialization of its own: it is a default-target layout."""
    config = _model_config(_gpt_oss_config())
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GPT_OSS_DEFAULT_TARGET


def _layout(ws, attention_dp, moe_tp=None, moe_ep=None):
    """Mapping kwargs for a (world_size, attention_dp, moe_tp, moe_ep) layout.

    Without moe sizes, Mapping derives moe_tp = tp, the LLM API default. More
    than four ranks span 4-GPU trays."""
    kwargs = dict(world_size=ws, tp_size=ws)
    if moe_tp is not None:
        kwargs.update(moe_tp_size=moe_tp, moe_ep_size=moe_ep)
    if attention_dp:
        kwargs["enable_attention_dp"] = True
    if ws > 4:
        kwargs["gpus_per_node"] = 4
    return kwargs


# Every layout the gpt-oss default target was gated on.
_GPT_OSS_DEFAULT_LAYOUTS = {
    "tp1": _layout(1, False),
    "tp2": _layout(2, False),
    "tp2-ep2": _layout(2, False, 1, 2),
    "tp4": _layout(4, False),
    "tp4-ep4": _layout(4, False, 1, 4),
    "tp4-moetp2-ep2": _layout(4, False, 2, 2),
    "dep2": _layout(2, True, 1, 2),
    "dep4": _layout(4, True, 1, 4),
    "dep4-moetp2-ep2": _layout(4, True, 2, 2),
    "tp8": _layout(8, False),
    "tp8-ep8": _layout(8, False, 1, 8),
    "dep8": _layout(8, True, 1, 8),
}


@pytest.mark.parametrize(
    "mapping_kwargs",
    list(_GPT_OSS_DEFAULT_LAYOUTS.values()),
    ids=list(_GPT_OSS_DEFAULT_LAYOUTS),
)
def test_gpt_oss_default_matches(mapping_kwargs):
    """Every gated layout lands on the default target. (A match is returned
    before the mode is read, so `require` alone covers `auto`.)"""
    config = _model_config(_gpt_oss_config(), **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _GPT_OSS_DEFAULT_TARGET


# Every layout the deepseek default target was gated on. It is the only
# deepseek target, so dep4 is one of them.
_R1_DEFAULT_LAYOUTS = {
    "tp2": _layout(2, False),
    "tp2-ep2": _layout(2, False, 1, 2),
    "dep2": _layout(2, True, 1, 2),
    "tp4": _layout(4, False),
    "tp4-ep4": _layout(4, False, 1, 4),
    "tp4-moetp2-ep2": _layout(4, False, 2, 2),
    "dep4": _layout(4, True, 1, 4),
    "dep4-moetp2-ep2": _layout(4, True, 2, 2),
    "dep4-moetp4": _layout(4, True, 4, 1),
    "tp8": _layout(8, False),
    "tp8-ep8": _layout(8, False, 1, 8),
    "tp8-moetp4-ep2": _layout(8, False, 4, 2),
    "tp8-moetp2-ep4": _layout(8, False, 2, 4),
    "dep8": _layout(8, True, 1, 8),
    "dep8-moetp8": _layout(8, True, 8, 1),
    "dep8-moetp2-ep4": _layout(8, True, 2, 4),
    "dep16": _layout(16, True, 1, 16),
}

# The layouts the MTP layer's acceptance was measured at: each residual-norm
# path and each expert split.
_R1_MTP_LAYOUTS = ("dep4", "tp4", "tp4-ep4", "dep4-moetp4", "tp8-moetp2-ep4", "dep8")


@pytest.mark.parametrize(
    "mapping_kwargs",
    list(_R1_DEFAULT_LAYOUTS.values()),
    ids=list(_R1_DEFAULT_LAYOUTS),
)
def test_r1_default_matches(mapping_kwargs):
    """Every gated layout lands on the default target. (A match is returned
    before the mode is read, so `require` alone covers `auto`.)"""
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _R1_DEFAULT_TARGET


@pytest.mark.parametrize("layout", _R1_MTP_LAYOUTS)
def test_r1_mtp_matches_at_its_gated_layouts(layout):
    config = _model_config(_r1_config(), spec_config=_MTP3, **_R1_DEFAULT_LAYOUTS[layout])
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _R1_DEFAULT_TARGET


@pytest.mark.parametrize(
    "layout", [name for name in _R1_DEFAULT_LAYOUTS if name not in _R1_MTP_LAYOUTS]
)
def test_r1_mtp_is_refused_elsewhere(layout):
    """The default forward computes the MTP layer at every layout, but a
    speculative config is admitted only where its acceptance was measured."""
    config = _model_config(_r1_config(), spec_config=_MTP3, **_R1_DEFAULT_LAYOUTS[layout])
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="MTP is gated on"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        # a layout outside the gated table
        (_layout(8, True, 4, 2), "attention_dp=True moe_tp=4 moe_ep=2"),
        # pipeline parallelism, refused before the table is read
        (dict(world_size=4, tp_size=2, pp_size=2), "pipeline"),
        # dep4 with the lm_head split over the attention-DP ranks, which the
        # target asserts against at construction
        (dict(_layout(4, True, 1, 4), enable_lm_head_tp_in_adp=True), "enable_lm_head_tp_in_adp"),
        # tp4 with attention split two ways and helix CP over the other two:
        # attn_tp_size 2 != tp_size 4, which the target asserts against
        (
            dict(_layout(4, False, 4, 1), attn_tp_size=2, cp_config={"cp_type": "HELIX"}),
            "attn_tp_size=2 != tp_size=4",
        ),
    ],
    ids=["dep8-moetp4-ep2", "tp2-pp2", "dep4-lm-head-tp", "tp4-attn-tp2"],
)
def test_r1_near_misses_do_not_match(mapping_kwargs, missed):
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_the_target_reads_the_mode_it_was_resolved_under(mode):
    """A target raises instead of falling back under require, so it has to know which mode built it."""
    config = _model_config(_gpt_oss_config())
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert resolved_mode(target) is ModelingV2Mode(mode)


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
        # the right checkpoint, a layout the default target has not been gated on
        (dict(), _layout(8, False, 2, 4), "moe_tp=2 moe_ep=4"),
        # a gated layout (dep4), with lm_head split over the attention-DP ranks:
        # the target asserts this off, so routing must refuse it first
        (
            dict(),
            {**_layout(4, True, 1, 4), "enable_lm_head_tp_in_adp": True},
            "lm_head_tp_in_adp=True",
        ),
        # a gated layout (tp4), with attention over 2 ranks inside the TP group
        # of 4 (helix splits the other 2 as attention CP): the target asserts
        # attention TP == TP, so routing must refuse it first
        (
            dict(),
            {**_layout(4, False), "attn_tp_size": 2, "cp_config": {"cp_type": "HELIX"}},
            "attn_tp=2 tp=4",
        ),
    ],
    ids=["other-size", "tp8-moetp2-ep4", "dep4-lm-head-tp", "tp4-attn-tp2"],
)
def test_gpt_oss_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None

    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


# Without multi-node NVLink (x86 nodes over IB, say), the targets cannot reduce
# attention TP across nodes; the same layouts inside one node, and attention DP
# across nodes, do not need it.
_NO_MNNVL_CASES = [
    pytest.param(_gpt_oss_config, _layout(8, False), None, id="gpt-oss-tp8-across-nodes"),
    pytest.param(_r1_config, _layout(8, False, 2, 4), None, id="r1-tp8-moetp2-ep4-across-nodes"),
    pytest.param(
        _gpt_oss_config,
        {**_layout(8, False), "gpus_per_node": 8},
        _GPT_OSS_DEFAULT_TARGET,
        id="gpt-oss-tp8-one-node",
    ),
    pytest.param(
        _r1_config,
        {**_layout(8, False, 2, 4), "gpus_per_node": 8},
        _R1_DEFAULT_TARGET,
        id="r1-tp8-moetp2-ep4-one-node",
    ),
    pytest.param(
        _gpt_oss_config, _layout(8, True, 1, 8), _GPT_OSS_DEFAULT_TARGET, id="gpt-oss-dep8"
    ),
    pytest.param(_r1_config, _layout(8, True, 1, 8), _R1_DEFAULT_TARGET, id="r1-dep8"),
]


@pytest.mark.parametrize("make_config, mapping_kwargs, expected", _NO_MNNVL_CASES)
def test_attention_tp_across_nodes_needs_mnnvl(monkeypatch, make_config, mapping_kwargs, expected):
    _set_mnnvl(monkeypatch, False)
    config = _model_config(make_config(), **mapping_kwargs)
    if expected is not None:
        assert _qualified(modeling_v2_resolve(config, _llm_args("auto"))) == expected
        return
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="(multi-node NVLink|tp_reaches_across_nodes=False)"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "no target fits" in message and "GptOssForCausalLM" in message
    assert "compute capability (9, 0)" in message


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


def test_the_default_mode_is_auto():
    assert TorchLlmArgs(model=_DUMMY_MODEL).modeling_v2 == "auto"


def test_no_target_is_a_value_error():
    """A refusal is the ValueError ``require`` has always raised, so callers that catch ValueError still do."""
    assert issubclass(NoTarget, ValueError)


class _RefusingTarget:
    """A target that refuses its runtime at construction."""

    def __init__(self, config):
        raise NoTarget("this runtime is not the one the target is written for")


class _BuiltIn:
    def __init__(self, config):
        self.config = config


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_a_refusal_at_construction_falls_back_unless_required(monkeypatch, mode):
    """A target's construction-time refusal builds the built-in implementation under ``auto`` (the mode
    ``default`` records) and is the error under ``require``."""
    import tensorrt_llm._torch.models.modeling_auto as modeling_auto

    monkeypatch.setattr(modeling_auto, "get_registered_model_class", lambda arch: _BuiltIn)
    monkeypatch.setitem(_RESOLVED_MODES, _RefusingTarget, ModelingV2Mode(mode))
    config = replace(_model_config(_gpt_oss_config()), modeling_v2_target=_RefusingTarget)
    config.extra_attrs["nvfp4_gemm_allowed_backends"] = ["cutlass"]
    if mode == "require":
        with pytest.raises(NoTarget, match="not the one the target is written for"):
            AutoModelForCausalLM.from_config(config)
        return
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, _BuiltIn)
    assert model.config.modeling_v2_target is None
    # The loader's settings reach the built-in model.
    assert model.config.extra_attrs["nvfp4_gemm_allowed_backends"] == ["cutlass"]


# --- speculative decoding is an identity criterion ------------------------------


def test_gpt_oss_with_speculative_decoding_falls_back_under_auto():
    """The default target implements no speculative path, so claiming the
    deployment would trade a fallback for a crash inside the target."""
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


def test_gpt_oss_with_speculative_decoding_raises_under_require():
    config = _model_config(_gpt_oss_config(), spec_config=_NGRAM)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "NGramDecodingConfig" in message


def test_r1_with_another_speculative_mode_does_not_match():
    """The default target carries the checkpoint's MTP module and nothing else."""
    config = _model_config(_r1_config(), spec_config=_NGRAM, **_R1_DEFAULT_LAYOUTS["dep4"])
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="one-model MTP-Eagle"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_r1_with_vanilla_mtp_does_not_match():
    """``use_mtp_vanilla`` resolves to the multi-model MTP loop, which drives a
    draft model this target does not build, so route must say no."""
    vanilla = MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)
    config = _model_config(_r1_config(), spec_config=vanilla, **_R1_DEFAULT_LAYOUTS["dep4"])
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# Each target asserts its KV pool at construction, so routing reads the KV-cache
# algorithm the engine will run with -- the checkpoint's under dtype "auto", the
# dtype's otherwise -- and refuses first, so `auto` falls back instead.
@pytest.mark.parametrize(
    "make_config, mapping_kwargs, kv_dtype, checkpoint_kv, expected",
    [
        pytest.param(
            _gpt_oss_config, _layout(4, False), "fp8", _CHECKPOINT_KV, None, id="gpt-oss-fp8"
        ),
        pytest.param(
            _gpt_oss_config,
            _layout(4, False),
            "auto",
            _CHECKPOINT_KV,
            _GPT_OSS_DEFAULT_TARGET,
            id="gpt-oss-auto",
        ),
        pytest.param(
            _r1_config,
            _layout(4, True, 1, 4),
            "auto",
            None,
            None,
            id="r1-checkpoint-without-fp8-kv",
        ),
        pytest.param(
            _r1_config, _layout(4, True, 1, 4), "nvfp4", _CHECKPOINT_KV, None, id="r1-nvfp4"
        ),
        pytest.param(
            _r1_config,
            _layout(4, True, 1, 4),
            "fp8_ds_mla",
            _CHECKPOINT_KV,
            None,
            id="r1-fp8-ds-mla",
        ),
        pytest.param(
            _r1_config,
            _layout(4, True, 1, 4),
            "fp8",
            None,
            _R1_DEFAULT_TARGET,
            id="r1-fp8-over-a-bf16-checkpoint",
        ),
    ],
)
def test_routes_on_the_kv_cache_algorithm_the_engine_runs(
    make_config, mapping_kwargs, kv_dtype, checkpoint_kv, expected
):
    config = _model_config(make_config(), kv_cache_quant_algo=checkpoint_kv, **mapping_kwargs)
    args = dict(kv_cache_config=KvCacheConfig(dtype=kv_dtype))
    if expected is not None:
        assert _qualified(modeling_v2_resolve(config, _llm_args("auto", **args))) == expected
        return
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


@pytest.mark.parametrize(
    "mapping_kwargs, parallel",
    [
        (dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1), "tp4"),
        (dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1), "tp8"),
        # experts split by expert rather than by tensor: another weight layout, no target
        (dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8), None),
        (dict(world_size=8, tp_size=8, moe_tp_size=8, enable_attention_dp=True), None),
        (dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1), "tp2"),
        (dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2), None),
    ],
)
def test_minimax_m3_parallel_names_the_target_topology(mapping_kwargs, parallel):
    """MiniMax-M3 has one target per tensor-parallel width: tp2, tp4 and tp8 load different per-rank shapes."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    assert routing._parallel(Mapping(**mapping_kwargs)) == parallel


# The MiniMax-M3 text model's shape fields, in ``routing._SHAPE`` order.
_M3_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_local_experts",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
)
# The MiniMax-M3 RoPE base and sparse-attention layers (dense 0-2, sparse 3-59), as its config.json reads.
_M3_LAYOUT = dict(
    rope_theta=5_000_000, sparse_attention_config=dict(sparse_attention_freq=[0] * 3 + [1] * 57)
)
# Another checkpoint revision with the same architecture class and shape: another RoPE base and sparse layout.
_OTHER_LAYOUT = dict(
    rope_theta=1_000_000, sparse_attention_config=dict(sparse_attention_freq=[0] * 4 + [1] * 56)
)


def _m3_text_config(**layout):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    return dict(zip(_M3_SHAPE_FIELDS, routing._SHAPE), **(layout or _M3_LAYOUT))


_M3_TARGETS = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
_M3_TP2 = dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1)
_M3_TP4 = dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1)
_M3_TP8 = dict(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)


@pytest.mark.parametrize(
    "capability, mapping_kwargs, target",
    [
        ((10, 3), _M3_TP4, "minimax_m3_nvfp4__sm_103__tp4.ModelingV2MiniMaxM3Nvfp4Sm103Tp4"),
        ((10, 3), _M3_TP8, "minimax_m3_nvfp4__sm_103__tp8.ModelingV2MiniMaxM3Nvfp4Sm103Tp8"),
        ((10, 7), _M3_TP2, "minimax_m3_nvfp4__sm_107__tp2.ModelingV2MiniMaxM3Nvfp4Sm107Tp2"),
        ((10, 7), _M3_TP4, "minimax_m3_nvfp4__sm_107__tp4.ModelingV2MiniMaxM3Nvfp4Sm107Tp4"),
        ((10, 7), _M3_TP8, "minimax_m3_nvfp4__sm_107__tp8.ModelingV2MiniMaxM3Nvfp4Sm107Tp8"),
        # no GB300 tp2 target
        ((10, 3), _M3_TP2, None),
    ],
)
def test_minimax_m3_route_picks_the_target_by_architecture_and_width(
    monkeypatch, capability, mapping_kwargs, target
):
    """Each (SM, tensor-parallel width) has its own target directory. The target modules are stubbed: this checks the
    decision, not that the modeling code imports on this host."""
    import sys
    import types

    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm.quantization.mode import QuantAlgo

    for directory, name in (
        ("minimax_m3_nvfp4__sm_103__tp4", "ModelingV2MiniMaxM3Nvfp4Sm103Tp4"),
        ("minimax_m3_nvfp4__sm_103__tp8", "ModelingV2MiniMaxM3Nvfp4Sm103Tp8"),
        ("minimax_m3_nvfp4__sm_107__tp2", "ModelingV2MiniMaxM3Nvfp4Sm107Tp2"),
        ("minimax_m3_nvfp4__sm_107__tp4", "ModelingV2MiniMaxM3Nvfp4Sm107Tp4"),
        ("minimax_m3_nvfp4__sm_107__tp8", "ModelingV2MiniMaxM3Nvfp4Sm107Tp8"),
    ):
        stub = types.ModuleType(f"{_M3_TARGETS}.{directory}.modeling")
        setattr(stub, name, type(name, (), {"__module__": stub.__name__}))
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(routing, "_quant_algo", lambda _config: QuantAlgo.MIXED_PRECISION)

    model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(text_config=_m3_text_config()),
        mapping=Mapping(**mapping_kwargs),
        spec_config=None,
    )
    llm_args = _m3_llm_args("auto", "nvfp4")
    if target is None:
        with pytest.raises(NoTarget, match="sm107 \\(Rubin\\) with 2, 4 or 8 ranks"):
            routing.route(model_config, llm_args, capability)
        return
    resolved = routing.route(model_config, llm_args, capability)
    directory, name = target.split(".")
    assert resolved.__module__ == f"{_M3_TARGETS}.{directory}.modeling"
    assert resolved.__name__ == name


@pytest.mark.parametrize("spec", [None, "eagle3"], ids=["no_speculation", "one_model_eagle3"])
@pytest.mark.parametrize("tp", [4, 8])
def test_minimax_m3_route_takes_the_released_checkpoint(monkeypatch, spec, tp):
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    monkeypatch.setattr(routing, "_eagle3_one_model_or_none", lambda spec: True)
    spec_config = (
        None
        if spec is None
        else SimpleNamespace(spec_dec_mode=SimpleNamespace(name="EAGLE3_ONE_MODEL"))
    )
    target = routing.route(
        SimpleNamespace(
            pretrained_config=SimpleNamespace(text_config=_m3_text_config()),
            mapping=Mapping(world_size=tp, tp_size=tp, moe_tp_size=tp, moe_ep_size=1),
            spec_config=spec_config,
        ),
        _m3_llm_args("auto", "nvfp4"),
        routing._SM,
    )
    assert target.__name__ == f"ModelingV2MiniMaxM3Nvfp4Sm103Tp{tp}"


@pytest.mark.parametrize(
    "layout, reason",
    [
        (_OTHER_LAYOUT, "rope_theta 1000000, not 5000000; sparse attention layers other than 3-59"),
        (dict(_M3_LAYOUT, rope_theta=1_000_000), "rope_theta 1000000, not 5000000"),
        (
            dict(_M3_LAYOUT, sparse_attention_config=_OTHER_LAYOUT["sparse_attention_config"]),
            "sparse attention layers other than 3-59",
        ),
        (
            dict(sparse_attention_config=_M3_LAYOUT["sparse_attention_config"]),
            "rope_theta None, not 5000000",
        ),
        (
            dict(
                _M3_LAYOUT,
                sparse_attention_config=dict(
                    _M3_LAYOUT["sparse_attention_config"], use_sparse_attention=False
                ),
            ),
            "sparse attention layers other than 3-59",
        ),
    ],
    ids=["other_revision", "rope_base", "sparse_layers", "no_rope_base", "sparse_disabled"],
)
def test_minimax_m3_route_declines_another_rope_base_or_sparse_layout(monkeypatch, layout, reason):
    """The architecture name and the shape do not pin the RoPE base or which layers are sparse: a checkpoint with
    the target shape but another of either (another checkpoint revision) is not routed to the MiniMax-M3 targets,
    even with their quantization."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    text_config = _m3_text_config(**layout)
    assert routing._layout_mismatch(text_config) == reason
    with pytest.raises(NoTarget, match=reason.split(";")[0]):
        routing.route(
            SimpleNamespace(
                pretrained_config=SimpleNamespace(text_config=text_config),
                mapping=Mapping(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
                spec_config=None,
            ),
            _m3_llm_args("auto", "nvfp4"),
            routing._SM,
        )


@pytest.mark.parametrize("mapping_kwargs", [_M3_TP2, _M3_TP4, _M3_TP8], ids=["tp2", "tp4", "tp8"])
@pytest.mark.parametrize(
    "layout",
    [
        _OTHER_LAYOUT,
        dict(
            _M3_LAYOUT,
            sparse_attention_config=dict(
                _M3_LAYOUT["sparse_attention_config"], use_sparse_attention=False
            ),
        ),
    ],
    ids=["other_revision", "sparse_disabled"],
)
def test_minimax_m3_rubin_route_declines_another_layout(monkeypatch, mapping_kwargs, layout):
    """The sm107 targets build the same RoPE table and sparse layout as the sm103 ones, so they decline the same
    checkpoints."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    with pytest.raises(NoTarget, match="sparse attention layers other than 3-59|rope_theta"):
        routing.route(
            SimpleNamespace(
                pretrained_config=SimpleNamespace(text_config=_m3_text_config(**layout)),
                mapping=Mapping(**mapping_kwargs),
                spec_config=None,
            ),
            _m3_llm_args("auto", "nvfp4"),
            routing._SM_RUBIN,
        )


def test_minimax_m3_layout_reads_rope_theta_from_rope_parameters():
    """A config that keeps the RoPE base under ``rope_parameters`` (newer transformers) reads the same."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    sparse = _M3_LAYOUT["sparse_attention_config"]
    assert routing._layout_mismatch(_m3_text_config()) is None
    for rope in (dict(rope_theta=5_000_000), dict(full_attention=dict(rope_theta=5_000_000))):
        assert (
            routing._layout_mismatch(dict(sparse_attention_config=sparse, rope_parameters=rope))
            is None
        )


def _m3_sparse(**overrides):
    """The sparse-attention runtime of the PR deployment (MSA, FP8 indexer, fused projection), with overrides."""
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    fields = dict(
        implementation="msa",
        indexer_kv_dtype="fp8",
        fuse_qkv_index_projection=True,
        sparse_disable_index_value=True,
    )
    fields.update(overrides)
    return MiniMaxM3SparseAttentionConfig(**fields)


def _m3_resolve_config(sparse, kv_checkpoint, tp=4, layout=None):
    """A ModelConfig as the loader holds it when it resolves: the MiniMax-M3 VL wrapper with the target text shape,
    the checkpoint's KV quantization (before kv_cache_config.dtype is applied) and the sparse-attention runtime."""
    pretrained = PretrainedConfig(
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config=_m3_text_config(**(layout or {})),
    )
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(world_size=tp, tp_size=tp, moe_tp_size=tp, moe_ep_size=1),
        quant_config=QuantConfig(kv_cache_quant_algo=kv_checkpoint),
        sparse_attention_config=sparse,
    )


def _m3_llm_args(mode, kv_dtype):
    return _llm_args(mode, kv_cache_config=dict(dtype=kv_dtype))


@pytest.fixture
def _minimax_m3_sm(_on_sm10x):
    """The MiniMax-M3 targets route on sm_103 only; on sm_100 they route nowhere."""
    if _on_sm10x != (10, 3):
        pytest.skip("the MiniMax-M3 targets are certified on sm_103 only")


@pytest.mark.usefixtures("_minimax_m3_sm")
@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_minimax_m3_pr_runtime_resolves_to_the_target(monkeypatch, mode, tp):
    """The PR deployment (MSA, FP8 indexer, fused projection, kv_cache_config.dtype='nvfp4') takes the target."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(), None, tp)
    target = modeling_v2_resolve(config, _m3_llm_args(mode, "nvfp4"))
    assert target.__name__ == f"ModelingV2MiniMaxM3Nvfp4Sm103Tp{tp}"


@pytest.mark.usefixtures("_minimax_m3_sm")
@pytest.mark.parametrize("tp", [4, 8])
def test_minimax_m3_bf16_kv_cache_falls_back_under_auto(monkeypatch, tp):
    """The targets run an FP8 or NVFP4 KV cache: ``auto`` with the BF16 cache resolves nothing, so the loader builds
    the registered built-in MiniMax-M3 model."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    # kv_cache_config.dtype 'auto' keeps the checkpoint's KV quantization (None: the BF16 cache).
    config = _m3_resolve_config(_m3_sparse(fuse_qkv_index_projection=False), None, tp)
    assert modeling_v2_resolve(config, _m3_llm_args("auto", "auto")) is None
    resolved = replace(config, modeling_v2_target=None)
    assert AutoModelForCausalLM._resolve_class(resolved) is MiniMaxM3VLForConditionalGeneration


@pytest.mark.usefixtures("_minimax_m3_sm")
def test_minimax_m3_bf16_kv_cache_raises_under_require(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(fuse_qkv_index_projection=False), None)
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _m3_llm_args("require", "auto"))
    message = str(excinfo.value)
    assert "MiniMaxM3SparseForConditionalGeneration" in message
    assert "KV cache bf16" in message


@pytest.mark.usefixtures("_minimax_m3_sm")
@pytest.mark.parametrize("tp", [4, 8])
def test_minimax_m3_other_checkpoint_revision_is_declined(monkeypatch, tp):
    """Another checkpoint revision (same architecture class and shape, another RoPE base and sparse layout) with the
    PR runtime: ``auto`` builds the built-in model and ``require`` names the layout."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(), None, tp, layout=_OTHER_LAYOUT)
    assert modeling_v2_resolve(config, _m3_llm_args("auto", "nvfp4")) is None
    with pytest.raises(ValueError, match="rope_theta 1000000, not 5000000"):
        modeling_v2_resolve(config, _m3_llm_args("require", "nvfp4"))


@pytest.mark.usefixtures("_minimax_m3_sm")
def test_minimax_m3_kv_cache_dtype_overrides_the_checkpoint(monkeypatch):
    """An explicit kv_cache_config.dtype is the KV cache the engine runs with, whatever the checkpoint declares."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    config = _m3_resolve_config(_m3_sparse(fuse_qkv_index_projection=False), None)
    assert modeling_v2_resolve(config, _m3_llm_args("auto", "auto")) is None
    target = modeling_v2_resolve(config, _m3_llm_args("auto", "fp8"))
    assert target.__name__ == "ModelingV2MiniMaxM3Nvfp4Sm103Tp4"


@pytest.mark.usefixtures("_minimax_m3_sm")
@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("role", ["aggregated", "context", "generation"])
def test_minimax_m3_is_claimed_in_every_serving_role(monkeypatch, tp: int, role: str) -> None:
    """Every serving role builds the target, under the default mode as under a set one."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    config = _m3_resolve_config(_m3_sparse(), None, tp)
    kwargs = dict(kv_cache_config=dict(dtype="nvfp4"))
    if role != "aggregated":
        kwargs["cache_transceiver_config"] = CacheTransceiverConfig()
    for args in (
        TorchLlmArgs(model=_DUMMY_MODEL, **kwargs),
        _llm_args("auto", **kwargs),
        _llm_args("require", **kwargs),
    ):
        target = modeling_v2_resolve(config, args)
        assert target.__name__ == f"ModelingV2MiniMaxM3Nvfp4Sm103Tp{tp}"
        assert resolved_mode(target) is ModelingV2Mode(args.modeling_v2)


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
    """Routed as a GB300 (sm 10.3), the K3 targets, sm 10.0 only, match nothing."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_k3_config(), **_TP16_MOETP4EP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- qwen4_exp: one-model MTP-Eagle over a linear chain, one rank ---------------

_QWEN4_EXP_TARGET = "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.modeling."


@pytest.fixture
def _qwen4_exp_sm(_on_sm10x):
    """The qwen4_exp target is certified on sm_103 only; on sm_100 it routes nowhere."""
    if _on_sm10x != (10, 3):
        pytest.skip("the qwen4_exp target is certified on sm_103 only")


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize(
    "spec_config",
    [
        None,
        MTPDecodingConfig(max_draft_len=1),
        MTPDecodingConfig(max_draft_len=3),
        MTPDecodingConfig(max_draft_len=7),
    ],
    ids=["no-spec", "mtp1", "mtp3", "mtp7"],
)
def test_qwen4_exp_tp1_matches(monkeypatch, spec_config):
    # mtp7 needs the GDN cached replay, which TRTLLM_USE_GDN_REPLAY=0 turns off.
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    # The recurrent-state cache at either dtype the target serves; `auto` is bf16 here.
    for dtype in ("float32", "bfloat16", "auto"):
        kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype=dtype)
        target = modeling_v2_resolve(config, _llm_args("require", kv_cache_config=kv_cache_config))
        assert _qualified(target).startswith(_QWEN4_EXP_TARGET), dtype


def _mtp(max_draft_len=3, **attributes):
    """An MTP config with fields the engine fills in after load set directly: the
    constructor remaps or recomputes them, so passing them would test nothing."""
    config = MTPDecodingConfig(max_draft_len=max_draft_len)
    for name, value in attributes.items():
        setattr(config, name, value)
    return config


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize(
    "spec_config",
    [
        _NGRAM,
        MTPDecodingConfig(max_draft_len=8),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        _mtp(max_total_draft_tokens=5),
        _mtp(num_nextn_predict_layers=2),
    ],
    ids=[
        "ngram",
        "draft8",
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


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_deepstack_vision_tower_does_not_match():
    config = _model_config(_qwen4_exp_config(deepstack_visual_indexes=[5, 11, 17]))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_on_two_ranks_does_not_match():
    config = _model_config(_qwen4_exp_config(), world_size=2, tp_size=2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_with_an_fp8_ds_mla_kv_cache_does_not_match():
    """fp8_ds_mla is FP8 to the loader but selects the packed sparse-MLA cache layout."""
    config = _model_config(_qwen4_exp_config())
    args = dict(kv_cache_config=KvCacheConfig(dtype="fp8_ds_mla"))
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


# --- qwen4_exp: engine choices the constructor would refuse -----------------------


def _declined(config, llm_args_kwargs=None, match=None):
    """``auto`` falls back to the built-in model and ``require`` says why."""
    kwargs = llm_args_kwargs or {}
    assert modeling_v2_resolve(config, _llm_args("auto", **kwargs)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", **kwargs))


def _claimed(config, llm_args_kwargs=None):
    target = modeling_v2_resolve(config, _llm_args("require", **(llm_args_kwargs or {})))
    assert _qualified(target).startswith(_QWEN4_EXP_TARGET)


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("algo", ["fp8", "nvfp4"])
def test_qwen4_exp_quantized_kv_cache_dtypes_match(algo):
    _claimed(_model_config(_qwen4_exp_config()), dict(kv_cache_config=KvCacheConfig(dtype=algo)))


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("algo", [QuantAlgo.INT8, QuantAlgo.FP8_BLOCK_SCALES])
def test_qwen4_exp_other_kv_cache_algorithms_are_declined(algo):
    """`kv_cache_config.dtype: auto` keeps the checkpoint's own KV-cache algorithm."""
    _declined(_model_config(_qwen4_exp_config(), kv_cache_quant_algo=algo), match="KV cache")


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_piecewise_prefill_cuda_graph_falls_back():
    config = _model_config(_qwen4_exp_config())
    _declined(config, dict(prefill_cuda_graph_backend="piecewise"), match="piecewise")


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("backend", ["disabled", "breakable"])
def test_qwen4_exp_matches_without_or_with_the_breakable_prefill_cuda_graph(backend):
    _claimed(
        _model_config(_qwen4_exp_config()),
        dict(prefill_cuda_graph_backend=backend),
    )


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("backend", ["CUTLASS", "TRITON", "DEEPGEMM", "VANILLA"])
def test_qwen4_exp_unimplemented_moe_backend_falls_back(backend):
    config = replace(_model_config(_qwen4_exp_config()), moe_backend_requested=backend)
    _declined(config, match="moe_config.backend")


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM", "CUTEDSL"])
def test_qwen4_exp_implemented_moe_backend_matches(backend):
    _claimed(replace(_model_config(_qwen4_exp_config()), moe_backend_requested=backend))


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        dict(max_concurrency=8),
        dict(draft_len_schedule={4: 3, 8: 1}),
        dict(acceptance_rate_threshold=0.5),
    ],
    ids=["max-concurrency", "draft-len-schedule", "acceptance-rate"],
)
@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_runtime_draft_length_falls_back(spec_kwargs):
    """The verify step runs at one width; a draft length that changes at runtime
    would trip the width assert in the first reduced step."""
    spec = MTPDecodingConfig(max_draft_len=3, **spec_kwargs)
    _declined(_model_config(_qwen4_exp_config(), spec_config=spec), match="draft length")


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("max_draft_len", [1, 3, 7])
def test_qwen4_exp_mtp_needs_the_cached_replay(monkeypatch, max_draft_len):
    """The target verifies on the GDN cached replay only, never on the legacy per-step
    recurrent states."""
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(
        _qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=max_draft_len)
    )
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float32")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="TRTLLM_USE_GDN_REPLAY=0")
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    _claimed(config, dict(kv_cache_config=kv_cache_config))


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_qwen4_exp_cutedsl_declines_drafts_beyond_three(monkeypatch, max_draft_len):
    """The CuteDSL expert entries capture up to 128 rows (32 requests of 4 verify rows)."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float32")
    spec = MTPDecodingConfig(max_draft_len=max_draft_len)
    cutedsl = replace(
        _model_config(_qwen4_exp_config(), spec_config=spec), moe_backend_requested="CUTEDSL"
    )
    _declined(cutedsl, dict(kv_cache_config=kv_cache_config), match="CUTEDSL")
    _claimed(
        _model_config(_qwen4_exp_config(), spec_config=spec), dict(kv_cache_config=kv_cache_config)
    )


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_cutedsl_matches_up_to_three_drafted_tokens(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    spec = MTPDecodingConfig(max_draft_len=3)
    cutedsl = replace(
        _model_config(_qwen4_exp_config(), spec_config=spec), moe_backend_requested="CUTEDSL"
    )
    _claimed(cutedsl, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32")))


@pytest.mark.usefixtures("_qwen4_exp_sm")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "auto"])
def test_qwen4_exp_mtp_matches_a_float32_or_bfloat16_state_cache(monkeypatch, dtype):
    """GDN cached replay runs on either checkpoint dtype; `auto` resolves to bf16 here."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype)))


@pytest.mark.parametrize(
    "spec_config", [None, MTPDecodingConfig(max_draft_len=3)], ids=["no-spec", "mtp3"]
)
@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_declines_a_float16_state_cache(monkeypatch, spec_config):
    """The engine's replay gate accepts fp16; the target's pool contract does not."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    config = _model_config(_qwen4_exp_config(), spec_config=spec_config)
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float16")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="recurrent-state cache")


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_mtp_on_a_float16_state_cache_declines_with_replay_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float16")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="recurrent-state cache")


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_without_speculation_matches_with_replay_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    _claimed(_model_config(_qwen4_exp_config()))


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_no_speculation_matches_the_default_state_cache(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    _claimed(_model_config(_qwen4_exp_config()))


# --- GLM-5.2 on sm103 -----------------------------------------------------------


@pytest.fixture
def _glm52_sm(_on_sm10x):
    """The GLM-5.2 target is certified on sm_103 only; on sm_100 it routes nowhere."""
    if _on_sm10x != (10, 3):
        pytest.skip("the GLM-5.2 target is certified on sm_103 only")


@pytest.mark.usefixtures("_glm52_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_glm52_tep4_matches(mode):
    config = _model_config(_glm52_config(), **_TEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _GLM52_TARGET


@pytest.mark.usefixtures("_glm52_sm")
@pytest.mark.parametrize("max_draft_len", [1, 3, 5], ids=["mtp1", "mtp3", "mtp5"])
def test_glm52_tep4_with_mtp_matches(max_draft_len):
    config = _model_config(
        _glm52_config(), spec_config=MTPDecodingConfig(max_draft_len=max_draft_len), **_TEP4
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _GLM52_TARGET


@pytest.mark.usefixtures("_glm52_sm")
@pytest.mark.parametrize(
    "spec_config, missed",
    [
        (MTPDecodingConfig(max_draft_len=2), r"max_draft_len=2, max_total_draft_tokens=2"),
        (MTPDecodingConfig(max_draft_len=4), r"max_draft_len=4, max_total_draft_tokens=4"),
        (
            MTPDecodingConfig(max_draft_len=1, use_dynamic_tree=True, dynamic_tree_max_topK=2),
            r"max_draft_len=1, max_total_draft_tokens=2, dynamic tree",
        ),
        # A dynamic tree capped at its chain length has the row count of a certified
        # linear chain, but still carries tree masks and position offsets.
        (
            MTPDecodingConfig(
                max_draft_len=3,
                max_total_draft_tokens=3,
                use_dynamic_tree=True,
                dynamic_tree_max_topK=2,
            ),
            r"max_draft_len=3, max_total_draft_tokens=3, dynamic tree",
        ),
        (
            MTPDecodingConfig(
                max_draft_len=5,
                max_total_draft_tokens=5,
                use_dynamic_tree=True,
                dynamic_tree_max_topK=2,
            ),
            r"max_draft_len=5, max_total_draft_tokens=5, dynamic tree",
        ),
    ],
    ids=[
        "draft2",
        "draft4",
        "dynamic_tree",
        "capped_dynamic_tree_mtp3",
        "capped_dynamic_tree_mtp5",
    ],
)
def test_glm52_tep4_with_uncertified_draft_chain_does_not_match(spec_config, missed):
    """The target is certified on linear chains of 1, 3 and 5 draft tokens (the decode
    attention's grid is compiled for D + 1 rows per sequence); other chain lengths route
    nowhere, and so does a dynamic tree, even one capped at a certified chain length:
    the MLA / RoPE and top-k paths do not consume its tree masks or position offsets."""
    config = _model_config(_glm52_config(), spec_config=spec_config, **_TEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_glm52_sm")
@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a DeepSeek-V3.2-shaped indexer: same trunk shape, 64 indexer heads
        (dict(index_n_heads=64), _TEP4, "index_n_heads=64"),
        # attention DP replicates attention: a different weight layout
        (dict(), _DEP4, "attention_dp=True"),
    ],
)
def test_glm52_near_misses_do_not_match(config_kwargs, mapping_kwargs, missed):
    config = _model_config(_glm52_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_glm52_sm")
def test_glm52_on_sm100_does_not_match(monkeypatch):
    """The context phase runs an sm103-only kernel, so a B200 is not this target."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_glm52_config(), **_TEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_glm52_sm")
def test_glm52_with_another_speculative_mode_does_not_match():
    config = _model_config(_glm52_config(), spec_config=_NGRAM, **_TEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None


@pytest.mark.usefixtures("_glm52_sm")
def test_glm52_without_an_fp8_kv_cache_does_not_match():
    config = _model_config(_glm52_config(), kv_cache_quant_algo=None, **_TEP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="KV cache bf16"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103 and sm107 ------------------------------


@pytest.fixture
def _mistral_large4_sm(_on_sm10x):
    """The Mistral-Large-4 targets are certified on sm_103 and sm_107; on sm_100 they route
    nowhere. `_on_sm10x` covers sm_100 and sm_103, so the sm_107 cases set the capability."""
    if _on_sm10x != (10, 3):
        pytest.skip("the Mistral-Large-4 targets are certified on sm_103 and sm_107 only")


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tep4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP4)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TEP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping,target",
    [
        (_TEP4, _MISTRAL_LARGE4_NVFP4_SM107_TEP_TARGET),
        (_TEP8, _MISTRAL_LARGE4_NVFP4_SM107_TEP_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_SM107_DEP_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_SM107_DEP_TARGET),
        (_TP4, _MISTRAL_LARGE4_NVFP4_SM107_TP_TARGET),
        (_TP8, _MISTRAL_LARGE4_NVFP4_SM107_TP_TARGET),
    ],
    ids=["tep4", "tep8", "dep4", "dep8", "tp4", "tp8"],
)
def test_mistral_large4_nvfp4_sm107_topologies_match(monkeypatch, mode, mapping, target):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    config = _model_config(_mistral_large4_nvfp4_config(), **mapping)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == target


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize(
    "mapping",
    [
        dict(_TEP8, world_size=16, tp_size=16, moe_ep_size=16),
        dict(_TEP8, tp_size=4, pp_size=2, moe_ep_size=4),
        dict(_TEP4, moe_ep_size=2, moe_tp_size=2),
        dict(_DEP4, enable_lm_head_tp_in_adp=True),
    ],
)
def test_mistral_large4_nvfp4_sm107_refuses_other_topologies(monkeypatch, mapping):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    config = _model_config(_mistral_large4_nvfp4_config(), **mapping)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 7\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mapping", [_TEP4, _TEP8, _DEP4, _DEP8, _TP4, _TP8])
def test_mistral_large4_nvfp4_sm107_refuses_non_native_speculation(monkeypatch, mapping):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **mapping
    )
    with pytest.raises(ValueError, match=r"compute capability \(10, 7\)"):
        modeling_v2_resolve(config, _llm_args("require"))


def _flip(values, i):
    flipped = list(values)
    flipped[i] = not flipped[i]
    return flipped


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_rope_factor_and_positions_are_not_criteria():
    """The YaRN factor and max_position_embeddings are configuration the rope table is built
    from: the NVFP4 export at YaRN factor 512 over 262144 positions is still the NVFP4 target."""
    config = _mistral_large4_nvfp4_config(
        max_position_embeddings=262144,
        rope_parameters=dict(_mistral_large4_nvfp4_config().rope_parameters, factor=512.0),
    )
    resolved = modeling_v2_resolve(_model_config(config, **_TEP4), _llm_args("require"))
    assert _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TEP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
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


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_on_sm100_does_not_match(monkeypatch):
    """The sparse-MLA kernels are sm103-only, so a B200 is not this target."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TEP4
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, attention data parallel ----------------


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_dep4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_DEP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_lands_on_its_topologys_target():
    """One checkpoint, two targets, chosen by attention DP alone: off is tep4, on is dep4."""
    config = _mistral_large4_nvfp4_config()
    tep4 = modeling_v2_resolve(_model_config(config, **_TEP4), _llm_args("require"))
    dep4 = modeling_v2_resolve(_model_config(config, **_DEP4), _llm_args("require"))
    assert _qualified(tep4) == _MISTRAL_LARGE4_NVFP4_TEP_TARGET
    assert _qualified(dep4) == _MISTRAL_LARGE4_NVFP4_DEP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_tep_branch_refuses_attention_dp():
    """The tep target is never returned for an attention-DP mapping."""
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    resolved = modeling_v2_resolve(config, _llm_args("auto"))
    assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TEP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
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


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_dep4_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_dep4_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_DEP4
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, tensor-parallel experts -----------------


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tp4_matches(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP4)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tp4_from_tensor_parallel_size_alone():
    """A deployment that states only tensor_parallel_size 4 leaves both MoE sizes unset; the
    mapping then gives every rank all the experts, tensor parallel (moe_tp 4, moe_ep 1), and
    that lands on the tp target."""
    mapping = Mapping(world_size=4, tp_size=4)
    assert (mapping.moe_tp_size, mapping.moe_ep_size) == (4, 1)
    config = ModelConfig(pretrained_config=_mistral_large4_nvfp4_config(), mapping=mapping)
    resolved = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_lands_on_each_topologys_target():
    """One checkpoint, three targets, chosen by the parallel topology alone: experts split by
    expert is tep4 (attention DP off) or dep4 (on), experts split by width is tp4."""
    config = _mistral_large4_nvfp4_config()
    landed = {
        name: _qualified(modeling_v2_resolve(_model_config(config, **m), _llm_args("require")))
        for name, m in (("tep4", _TEP4), ("dep4", _DEP4), ("tp4", _TP4))
    }
    assert landed == {
        "tep4": _MISTRAL_LARGE4_NVFP4_TEP_TARGET,
        "dep4": _MISTRAL_LARGE4_NVFP4_DEP_TARGET,
        "tp4": _MISTRAL_LARGE4_NVFP4_TP_TARGET,
    }


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_expert_parallel_branches_refuse_tp4():
    """No expert-parallel target is ever returned for the tp4 mapping, and the tp target for
    none of the expert-parallel mappings."""
    config = _mistral_large4_nvfp4_config()
    resolved = modeling_v2_resolve(_model_config(config, **_TP4), _llm_args("auto"))
    assert resolved is None or _qualified(resolved) == _MISTRAL_LARGE4_NVFP4_TP_TARGET
    for mapping in (_TEP4, _DEP4):
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("auto"))
        assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
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


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tp4_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP4)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
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


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tep8_matches_the_tep_target(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP8)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_TEP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_lands_on_its_topologys_target_at_every_world():
    """One checkpoint, three targets, chosen by the topology alone and not by the rank count:
    the experts split by expert is the tep target (attention DP off) or the dep target (on),
    split by width (stated, or tensor_parallel_size 8 alone) the tp target, on four ranks and
    on eight over two nodes of four alike."""
    config = _mistral_large4_nvfp4_config()
    for mapping, target in (
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
        (_TP4, _MISTRAL_LARGE4_NVFP4_TP_TARGET),
        (_TEP8, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (_TP8, _MISTRAL_LARGE4_NVFP4_TP_TARGET),
        (_TP8_BARE, _MISTRAL_LARGE4_NVFP4_TP_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
    ):
        resolved = modeling_v2_resolve(_model_config(config, **mapping), _llm_args("require"))
        assert _qualified(resolved) == target, mapping


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_eight_ranks_land_where_four_ranks_do():
    """Each eight-rank mapping lands on the target of its four-rank counterpart."""
    config = _mistral_large4_nvfp4_config()
    for four, eight in ((_TEP4, _TEP8), (_TP4, _TP8), (_DEP4, _DEP8)):
        assert _resolves_to(config, eight) == _resolves_to(config, four), eight


@pytest.mark.usefixtures("_mistral_large4_sm")
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


@pytest.mark.usefixtures("_mistral_large4_sm")
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
        "NVFP4 routed experts, FP8_PB_WO linears, group 16, its 475-module map) on sm103 or sm107 "
        "in the tep, dep and tp layouts" in message
    )
    assert "each target serves 4 ranks on one node or 8 ranks over two nodes of four" in message
    for served in (
        "world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8",
        "world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: dep8",
        "world_size=4 tp=4 pp=1 moe_ep=1 moe_tp=4, or world_size=8 tp=8 pp=1 moe_ep=1 moe_tp=8: tp8",
    ):
        assert served in message, served


@pytest.mark.usefixtures("_mistral_large4_sm")
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
    # The refusal names the eight-rank deployment the tep target serves.
    message = str(excinfo.value)
    assert "tep is tensor-parallel attention, expert-parallel MoE" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8" in message


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tep8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TEP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tep8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TEP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, eight ranks, tensor-parallel experts -----


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mapping", [_TP8, _TP8_BARE], ids=["stated", "tp_size_8_alone"])
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_tp8_matches_the_tp_target(mode, mapping):
    model_config = _model_config(_mistral_large4_nvfp4_config(), **mapping)
    assert (model_config.mapping.moe_tp_size, model_config.mapping.moe_ep_size) == (8, 1)
    assert (
        _qualified(modeling_v2_resolve(model_config, _llm_args(mode)))
        == _MISTRAL_LARGE4_NVFP4_TP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_tp_target_refuses_every_other_topology_at_eight_ranks():
    """The tp target is returned for no expert-parallel mapping, at four ranks or eight, and
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
        assert resolved is None or _qualified(resolved) != _MISTRAL_LARGE4_NVFP4_TP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
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
    # The refusal names the width-split eight-rank deployment the tp target serves, beside the
    # expert-split one the tep target serves.
    message = str(excinfo.value)
    assert "tp is tensor-parallel attention, tensor-parallel MoE" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=1 moe_tp=8: tp8" in message
    assert "tep is tensor-parallel attention, expert-parallel MoE" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: tep8" in message


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tp8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_TP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_tp8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- Mistral-Large-4-1T-A52B-NVFP4 on sm103, attention data parallel over eight ranks ----


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_mistral_large4_nvfp4_dep8_matches_the_dep_target(mode):
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP8)
    assert (
        _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _MISTRAL_LARGE4_NVFP4_DEP_TARGET
    )


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_lands_on_its_topologys_target_at_four_and_eight_ranks():
    """One checkpoint, one target per topology: tep4 and dep4 on four ranks, and attention DP
    over eight ranks is the dep target too."""
    config = _mistral_large4_nvfp4_config()
    for mapping, target in (
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
    ):
        assert _resolves_to(config, mapping) == target, mapping


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_eight_ranks_without_attention_dp_never_reach_the_dep_target():
    """Eight ranks with attention DP off (tep8), or tensor_parallel_size 8 alone (every expert
    sharded eight ways, moe_tp 8), are other weight layouts: whatever routing returns for them,
    it is not the dep target -- the NVFP4 export lands on tep4 and tp4, and tensor-parallel
    experts under attention DP on nothing. The dep target is returned for no four-rank mapping
    without attention DP."""
    config = _mistral_large4_nvfp4_config()
    tep8 = dict(_DEP8, enable_attention_dp=False)
    for mapping, landed in (
        (tep8, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (dict(tep8, moe_ep_size=1, moe_tp_size=8), _MISTRAL_LARGE4_NVFP4_TP_TARGET),
        (dict(_DEP8, moe_ep_size=1, moe_tp_size=8), None),
    ):
        resolved = _resolves_to(config, mapping)
        assert resolved != _MISTRAL_LARGE4_NVFP4_DEP_TARGET, mapping
        assert resolved == landed, mapping
    for mapping in (_TEP4, _TP4):
        assert _resolves_to(config, mapping) != _MISTRAL_LARGE4_NVFP4_DEP_TARGET, mapping
    assert _resolves_to(config, _DEP8) == _MISTRAL_LARGE4_NVFP4_DEP_TARGET


@pytest.mark.usefixtures("_mistral_large4_sm")
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
    # The refusal names the eight-rank deployment the dep target serves.
    message = str(excinfo.value)
    assert "dep is attention data parallel" in message
    assert "8 ranks over two nodes of four" in message
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1: dep8" in message


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_dep8_with_lm_head_tp_names_the_dep_target():
    """lm_head tensor parallel under attention DP at eight ranks is refused, and the message
    names the dep target's eight-rank deployment and the switch that missed."""
    config = _model_config(
        _mistral_large4_nvfp4_config(), **dict(_DEP8, enable_lm_head_tp_in_adp=True)
    )
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "world_size=8 tp=8 pp=1 moe_ep=8 moe_tp=1" in message
    assert "lm_head_tp_in_adp=True" in message
    assert "dep is attention data parallel with lm_head TP off" in message
    assert "moe_ep=8 moe_tp=1: dep8" in message


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_dep8_on_sm100_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_mistral_large4_nvfp4_config(), **_DEP8)
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_dep8_with_speculative_decoding_does_not_match():
    config = _model_config(
        _mistral_large4_nvfp4_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_DEP8
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding MTP"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("sm", [103, 107])
@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize(
    "parallel,target",
    [
        ("tep", _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        ("dep", _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
        ("tp", _MISTRAL_LARGE4_NVFP4_TP_TARGET),
    ],
)
def test_mistral_large4_native_dspark_routes_each_topology(
    tmp_path, monkeypatch, world, parallel, target, sm, with_images
):
    import json

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (sm // 10, sm % 10))
    target = target.replace("sm_103", f"sm_{sm}").replace("Sm103", f"Sm{sm}")
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
        _mistral_large4_nvfp4_config(
            **({"vision_config": SimpleNamespace(**_ML4_VISION)} if with_images else {})
        ),
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


@pytest.mark.usefixtures("_mistral_large4_sm")
@pytest.mark.parametrize("sm", [103, 107])
@pytest.mark.parametrize(
    "mapping, target",
    [
        (_TEP4, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (_DEP4, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
        (_TP4, _MISTRAL_LARGE4_NVFP4_TP_TARGET),
        (_TEP8, _MISTRAL_LARGE4_NVFP4_TEP_TARGET),
        (_DEP8, _MISTRAL_LARGE4_NVFP4_DEP_TARGET),
        (_TP8, _MISTRAL_LARGE4_NVFP4_TP_TARGET),
    ],
    ids=["tep4", "dep4", "tp4", "tep8", "dep8", "tp8"],
)
def test_mistral_large4_nvfp4_with_a_vision_encoder_serves_images(monkeypatch, mapping, target, sm):
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.routing import (
        serves_images,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (sm // 10, sm % 10))
    target = target.replace("sm_103", f"sm_{sm}").replace("Sm103", f"Sm{sm}")
    pretrained_config = _mistral_large4_nvfp4_config(vision_config=SimpleNamespace(**_ML4_VISION))
    config = _model_config(pretrained_config, **mapping)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == target
    assert serves_images(pretrained_config)
    assert not serves_images(_mistral_large4_nvfp4_config())


@pytest.mark.usefixtures("_mistral_large4_sm")
def test_mistral_large4_nvfp4_with_another_vision_encoder_does_not_match():
    vision = SimpleNamespace(**dict(_ML4_VISION, patch_size=16))
    config = _model_config(_mistral_large4_nvfp4_config(vision_config=vision), **_TEP4)
    with pytest.raises(ValueError, match="vision encoder or projector geometry"):
        modeling_v2_resolve(config, _llm_args("require"))

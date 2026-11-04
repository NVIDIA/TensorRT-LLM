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

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    _RESOLVED_MODES,
    ModelingV2Mode,
    NoTarget,
    modeling_v2_resolve,
    resolved_mode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import (
    routing as qwen4_exp_routing,
)
from tensorrt_llm._torch.configs.kimi_k3 import KimiK3Config
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
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
    DeepSeek-R1-0528-NVFP4 and GLM-5.2-NVFP4, none (a bf16 pool) for the others."""
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    if kv_cache_quant_algo is _CHECKPOINT_KV:
        fp8_kv = pretrained_config.architectures[0] in (
            "DeepseekV3ForCausalLM",
            "GlmMoeDsaForCausalLM",
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


def test_the_default_mode_is_off():
    assert TorchLlmArgs(model=_DUMMY_MODEL).modeling_v2 == "off"


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
        (dict(world_size=2, tp_size=2, moe_tp_size=2), None),
    ],
)
def test_minimax_m3_parallel_names_the_target_topology(mapping_kwargs, parallel):
    """MiniMax-M3 has one target per tensor-parallel width: tp4 and tp8 load different per-rank shapes."""
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
def test_minimax_m3_requires_explicit_opt_in(monkeypatch, tp: int, role: str) -> None:
    """Every serving role stays on the built-in model unless modeling_v2 is requested."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing

    monkeypatch.setattr(routing, "_quant_algo", lambda cfg: QuantAlgo.MIXED_PRECISION)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    config = _m3_resolve_config(_m3_sparse(), None, tp)
    kwargs = dict(kv_cache_config=dict(dtype="nvfp4"))
    if role != "aggregated":
        kwargs["cache_transceiver_config"] = CacheTransceiverConfig()
    assert modeling_v2_resolve(config, TorchLlmArgs(model=_DUMMY_MODEL, **kwargs)) is None
    for mode in ("auto", "require"):
        target = modeling_v2_resolve(config, _llm_args(mode, **kwargs))
        assert target.__name__ == f"ModelingV2MiniMaxM3Nvfp4Sm103Tp{tp}"
        assert resolved_mode(target) is ModelingV2Mode(mode)


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

_QWEN4_EXP_TARGET = "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."


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
    choices = qwen4_exp_routing.engine_choices(
        _model_config(_qwen4_exp_config()), _llm_args("auto")
    )
    with pytest.raises(NoTarget, match="KV cache"):
        qwen4_exp_routing._check_engine(None, algo, choices)


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
@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_qwen4_exp_draft_beyond_three_needs_the_cached_replay(monkeypatch, max_draft_len):
    """The legacy per-step recurrent states are certified at up to three drafted tokens and
    hold one fp32 state per verify row."""
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(
        _qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=max_draft_len)
    )
    kv_cache_config = KvCacheConfig(mamba_ssm_cache_dtype="float32")
    _declined(config, dict(kv_cache_config=kv_cache_config), match="TRTLLM_USE_GDN_REPLAY=0")
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    _claimed(config, dict(kv_cache_config=kv_cache_config))


@pytest.mark.usefixtures("_qwen4_exp_sm")
def test_qwen4_exp_legacy_verify_matches_up_to_three_drafted_tokens(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32")))


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
def test_qwen4_exp_mtp_on_a_bf16_state_cache_matches_with_replay_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(_qwen4_exp_config(), spec_config=MTPDecodingConfig(max_draft_len=3))
    _claimed(config)


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
def test_glm52_tep4_with_mtp_matches():
    config = _model_config(_glm52_config(), spec_config=MTPDecodingConfig(max_draft_len=1), **_TEP4)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _GLM52_TARGET


@pytest.mark.usefixtures("_glm52_sm")
@pytest.mark.parametrize(
    "spec_config, missed",
    [
        (MTPDecodingConfig(max_draft_len=2), r"max_draft_len=2, max_total_draft_tokens=2"),
        (MTPDecodingConfig(max_draft_len=3), r"max_draft_len=3, max_total_draft_tokens=3"),
        (
            MTPDecodingConfig(max_draft_len=1, use_dynamic_tree=True, dynamic_tree_max_topK=2),
            r"max_draft_len=1, max_total_draft_tokens=2",
        ),
    ],
    ids=["draft2", "draft3", "dynamic_tree"],
)
def test_glm52_tep4_with_more_than_one_draft_token_does_not_match(spec_config, missed):
    """The decode attention, the base-row index kind and the MLA rope kernel are
    built for two query rows per sequence; a longer draft chain or a tree puts
    more rows on a sequence and is a different forward."""
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

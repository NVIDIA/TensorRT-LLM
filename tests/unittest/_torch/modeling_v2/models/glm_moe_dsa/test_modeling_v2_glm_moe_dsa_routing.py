# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GlmMoeDsaForCausalLM deployment lands, driven through ``modeling_v2_resolve``.

Synthetic configs, as in test_modeling_v2_routing.py: routing reads the config
shape and layer pattern, the declared quantization, the mapping, the compute
capability (patched here, so these run on any machine) and the
speculative-decoding config.

The three targets claim one checkpoint (GLM-5.2 NVFP4 with its bf16 MTP layer), on
sm103, with one-model MTP-Eagle as a linear chain, at disjoint parallel layouts:
tp8 (attention and experts tensor-parallel over eight ranks) and tep (attention
tensor-parallel over four or eight ranks, experts expert-parallel over the same
ranks), both without attention DP and at five draft tokens, and dep (attention DP
with the experts expert-parallel over the same ranks: sixteen ranks at three draft
tokens, eight ranks at three, or eight ranks with the LM head tensor-parallel in
attention DP at five). Every other value of each criterion declines -- ``auto``
falls back to the built-in implementation and ``require`` raises naming the fact
that missed -- including quantization declarations the engine's own parser would
reject.
"""

from __future__ import annotations

import re

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    ModelingV2Mode,
    modeling_v2_resolve,
    resolved_mode,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    MTPDecodingConfig,
    NGramDecodingConfig,
    SAEnhancerConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET = "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.modeling.ModelingV2Glm52Nvfp4Sm103Tp8"
_TEP_TARGET = "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep.modeling.ModelingV2Glm52Nvfp4Sm103Tep"
_DEP_TARGET = "models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__dep.modeling.ModelingV2Glm52Nvfp4Sm103Dep"

# The excluded-module patterns of the GLM-5.2 NVFP4 export, abridged: the head, the
# embedding, the dense layers 0-2, the MTP layer 78, and per MoE layer the attention
# and the shared expert.
_NVFP4_IGNORE = [
    "lm_head",
    "model.embed_tokens",
    "model.layers.0*",
    "model.layers.1.*",
    "model.layers.2.*",
    "model.layers.78*",
    "model.layers.3.self_attn*",
    "model.layers.3.mlp.shared_experts*",
]


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were an sm_103 device, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _nvfp4_quantization(ignore=_NVFP4_IGNORE) -> dict:
    """The ``quantization_config`` the NVFP4 export writes into its config.json."""
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt"},
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "ignore": list(ignore),
    }


# The shape and layer pattern GLM-5.2-NVFP4's own config.json declares.
_GLM52_SHAPE = dict(
    num_hidden_layers=78,
    hidden_size=6144,
    n_routed_experts=256,
    q_lora_rank=2048,
    num_attention_heads=64,
    index_topk=2048,
    first_k_dense_replace=3,
    index_topk_freq=4,
    index_skip_topk_offset=3,
)
_SHAPE_FIELDS = (*_GLM52_SHAPE, "index_topk_pattern")


def _glm52_fields(**overrides) -> dict:
    fields = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_nextn_predict_layers=1,
        quantization_config=_nvfp4_quantization(),
        **_GLM52_SHAPE,
    )
    fields.update(overrides)
    return fields


def _glm52_config(**overrides) -> PretrainedConfig:
    """The shape, layer pattern and quantization GLM-5.2-NVFP4's own config.json declares."""
    return PretrainedConfig(**_glm52_fields(**overrides))


def _shape_words(**overrides) -> str:
    """The config shape a NoTarget message names for the GLM-5.2 shape with ``overrides``."""
    values = {**_GLM52_SHAPE, "index_topk_pattern": None, **overrides}
    return re.escape(f"config shape {tuple(values[name] for name in _SHAPE_FIELDS)}")


def _llm_args(mode: str = "auto") -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode)


def _model_config(pretrained_config, spec_config, **mapping_kwargs) -> ModelConfig:
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
    )


_TP8 = dict(world_size=8, tp_size=8)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1)
_DEP16 = dict(world_size=16, tp_size=16, moe_ep_size=16, moe_tp_size=1, enable_attention_dp=True)
_DEP8 = dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True)
_DEP8_LM_HEAD_TP = dict(_DEP8, enable_lm_head_tp_in_adp=True)
# The deployments the dep target claims: (mapping, draft tokens per request).
_DEP_CELLS = {
    "dep16-mtp3": (_DEP16, 3),
    "dep8-mtp3": (_DEP8, 3),
    "dep8-lm-head-tp-mtp5": (_DEP8_LM_HEAD_TP, 5),
}


def _mtp(max_draft_len: int = 5, **kwargs) -> MTPDecodingConfig:
    return MTPDecodingConfig(max_draft_len=max_draft_len, **kwargs)


@pytest.mark.parametrize("mode", [None, "off"], ids=["default", "off"])
def test_off_resolves_nothing(mode):
    args = TorchLlmArgs(model=_DUMMY_MODEL) if mode is None else _llm_args(mode)
    config = _model_config(_glm52_config(), _mtp(), **_TP8)
    assert modeling_v2_resolve(config, args) is None


def test_tp8_is_one_expert_group_without_attention_dp():
    """The default tp8 mapping is the layout the target was written for."""
    mapping = Mapping(**_TP8)
    assert (mapping.moe_tp_size, mapping.moe_ep_size, mapping.enable_attention_dp) == (
        8,
        1,
        False,
    )


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_glm52_nvfp4_tp8_mtp5_matches(mode):
    config = _model_config(_glm52_config(), _mtp(), **_TP8)
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert _qualified(target) == _TARGET
    assert resolved_mode(target) is ModelingV2Mode(mode)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("mapping_kwargs", [_TEP4, _TEP8], ids=["tep4", "tep8"])
def test_glm52_nvfp4_tep_mtp5_matches(mode, mapping_kwargs):
    """Four or eight ranks of tensor-parallel attention with the experts expert-parallel over
    the same ranks land on the tep target."""
    config = _model_config(_glm52_config(), _mtp(), **mapping_kwargs)
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert _qualified(target) == _TEP_TARGET
    assert resolved_mode(target) is ModelingV2Mode(mode)


def test_the_tep_mappings_are_one_expert_per_rank_group_without_attention_dp():
    for kwargs, ranks in ((_TEP4, 4), (_TEP8, 8)):
        mapping = Mapping(**kwargs)
        assert (mapping.world_size, mapping.tp_size, mapping.moe_ep_size, mapping.moe_tp_size) == (
            ranks,
            ranks,
            ranks,
            1,
        )
        assert not mapping.enable_attention_dp


def test_the_three_claims_are_disjoint():
    """No mapping satisfies two parallel criteria: tp8 takes moe_ep 1, tep moe_ep = tp > 1, both
    without attention DP, and dep takes attention DP."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa import routing

    for kwargs in (_TP8, _TEP4, _TEP8, _DEP16, _DEP8, _DEP8_LM_HEAD_TP):
        mapping = Mapping(**kwargs)
        assert routing._tp8(mapping) + routing._tep(mapping) + routing._dep(mapping) == 1, kwargs


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping_kwargs, draft_len", list(_DEP_CELLS.values()), ids=list(_DEP_CELLS)
)
def test_glm52_nvfp4_dep_matches(mode, mapping_kwargs, draft_len):
    """Attention DP with the experts expert-parallel over the same ranks lands on the dep target
    at the draft length of its rank count and LM-head split."""
    config = _model_config(_glm52_config(), _mtp(draft_len), **mapping_kwargs)
    target = modeling_v2_resolve(config, _llm_args(mode))
    assert _qualified(target) == _DEP_TARGET
    assert resolved_mode(target) is ModelingV2Mode(mode)


def test_the_dep_mappings_are_one_expert_per_rank_group_with_attention_dp():
    for kwargs, ranks, lm_head_tp in ((_DEP16, 16, False), (_DEP8, 8, False)) + (
        (_DEP8_LM_HEAD_TP, 8, True),
    ):
        mapping = Mapping(**kwargs)
        assert (mapping.world_size, mapping.tp_size, mapping.moe_ep_size, mapping.moe_tp_size) == (
            ranks,
            ranks,
            ranks,
            1,
        )
        assert mapping.enable_attention_dp
        assert mapping.enable_lm_head_tp_in_adp is lm_head_tp


def test_the_dep_cells_are_the_routed_ones():
    """The target's cell table and ``route``'s claim name the same deployments."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa import routing
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__dep import (  # noqa: E501
        modeling,
    )

    assert set(modeling.CELLS) == {
        (ranks, lm_head_tp, draft) for (ranks, lm_head_tp), draft in routing._DEP.items()
    }
    assert {
        (Mapping(**kwargs).world_size, Mapping(**kwargs).enable_lm_head_tp_in_adp, draft)
        for kwargs, draft in _DEP_CELLS.values()
    } == set(modeling.CELLS)


def test_the_dep_target_is_the_builtin_causal_lm():
    from tensorrt_llm._torch.models.modeling_deepseekv3 import GlmMoeDsaForCausalLM

    config = _model_config(_glm52_config(), _mtp(3), **_DEP8)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert issubclass(target, GlmMoeDsaForCausalLM)
    assert (
        target.get_preferred_kv_cache_manager_version()
        == GlmMoeDsaForCausalLM.get_preferred_kv_cache_manager_version()
    )
    assert (
        target.get_preferred_transceiver_runtime()
        == GlmMoeDsaForCausalLM.get_preferred_transceiver_runtime()
    )


def test_the_tep_target_is_the_builtin_causal_lm():
    from tensorrt_llm._torch.models.modeling_deepseekv3 import GlmMoeDsaForCausalLM

    config = _model_config(_glm52_config(), _mtp(), **_TEP8)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert issubclass(target, GlmMoeDsaForCausalLM)
    assert (
        target.get_preferred_kv_cache_manager_version()
        == GlmMoeDsaForCausalLM.get_preferred_kv_cache_manager_version()
    )


def test_the_engine_config_class_keeps_the_routed_fields():
    """The loader builds glm_moe_dsa configs through the registered class; route reads the same fields."""
    from tensorrt_llm._torch.pyexecutor.config_utils import _CONFIG_REGISTRY

    pretrained_config = _CONFIG_REGISTRY["glm_moe_dsa"].from_dict(_glm52_fields())
    config = _model_config(pretrained_config, _mtp(), **_TP8)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


def test_the_target_is_the_builtin_causal_lm():
    """The target keeps the built-in class's engine preferences, read before routing."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import GlmMoeDsaForCausalLM

    config = _model_config(_glm52_config(), _mtp(), **_TP8)
    target = modeling_v2_resolve(config, _llm_args("require"))
    assert issubclass(target, GlmMoeDsaForCausalLM)
    assert (
        target.get_preferred_transceiver_runtime()
        == GlmMoeDsaForCausalLM.get_preferred_transceiver_runtime()
    )
    assert (
        target.get_preferred_kv_cache_manager_version()
        == GlmMoeDsaForCausalLM.get_preferred_kv_cache_manager_version()
    )


# Each case changes one fact of the claimed deployment: (config overrides, mapping,
# speculative config, compute capability, what the NoTarget message must name).
_FOREIGN = {
    "tp4": ({}, dict(world_size=4, tp_size=4), _mtp(), _SM103, r"ws=4 tp=4"),
    "tp16": ({}, dict(world_size=16, tp_size=16), _mtp(), _SM103, r"ws=16 tp=16"),
    "attention-dp": (
        {},
        dict(_TP8, enable_attention_dp=True),
        _mtp(),
        _SM103,
        r"attention_dp=True",
    ),
    "tep2": (
        {},
        dict(world_size=2, tp_size=2, moe_ep_size=2, moe_tp_size=1),
        _mtp(),
        _SM103,
        r"ws=2 tp=2 .* moe_tp=1 moe_ep=2",
    ),
    "tep16": (
        {},
        dict(world_size=16, tp_size=16, moe_ep_size=16, moe_tp_size=1),
        _mtp(),
        _SM103,
        r"ws=16 tp=16 .* moe_tp=1 moe_ep=16",
    ),
    "tep8-attention-dp": (
        {},
        dict(_TEP8, enable_attention_dp=True),
        _mtp(),
        _SM103,
        r"moe_ep=8 attention_dp=True",
    ),
    "tep4-attention-dp": (
        {},
        dict(_TEP4, enable_attention_dp=True),
        _mtp(),
        _SM103,
        r"moe_ep=4 attention_dp=True",
    ),
    "tp8-ep4-tp2": (
        {},
        dict(_TP8, moe_ep_size=4, moe_tp_size=2),
        _mtp(),
        _SM103,
        r"moe_tp=2 moe_ep=4",
    ),
    "tep4-pp2": (
        {},
        dict(world_size=8, tp_size=4, pp_size=2, moe_ep_size=4, moe_tp_size=1),
        _mtp(),
        _SM103,
        r"ws=8 tp=4 pp=2",
    ),
    "tep8-sm100": ({}, _TEP8, _mtp(), (10, 0), r"compute capability \(10, 0\)"),
    "tep4-mtp3": ({}, _TEP4, _mtp(3), _SM103, r"max_draft_len=3 "),
    "tep8-bf16-checkpoint": (
        dict(quantization_config=None),
        _TEP8,
        _mtp(),
        _SM103,
        r"quantization none",
    ),
    "sm100": ({}, _TP8, _mtp(), (10, 0), r"compute capability \(10, 0\)"),
    "sm120": ({}, _TP8, _mtp(), (12, 0), r"compute capability \(12, 0\)"),
    "sm90": ({}, _TP8, _mtp(), (9, 0), r"compute capability \(9, 0\)"),
    "bf16-checkpoint": (
        dict(quantization_config=None),
        _TP8,
        _mtp(),
        _SM103,
        r"quantization none",
    ),
    "fp8-checkpoint": (
        dict(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}),
        _TP8,
        _mtp(),
        _SM103,
        r"quantization quant_method 'fp8' \(not a modelopt export\)",
    ),
    "fp8-modelopt-checkpoint": (
        dict(quantization_config={**_nvfp4_quantization(), "quant_algo": "FP8"}),
        _TP8,
        _mtp(),
        _SM103,
        r"quantization FP8 with the MTP layer unquantized",
    ),
    "quantized-mtp-layer": (
        dict(quantization_config=_nvfp4_quantization(["lm_head", "model.embed_tokens"])),
        _TP8,
        _mtp(),
        _SM103,
        r"quantization NVFP4 with the MTP layer quantized",
    ),
    **{
        f"shape-{name}": (
            {name: value},
            _TP8,
            _mtp(),
            _SM103,
            _shape_words(**{name: value}),
        )
        for name, value in (
            ("num_hidden_layers", 92),
            ("hidden_size", 7168),
            ("n_routed_experts", 160),
            ("q_lora_rank", 1536),
            ("num_attention_heads", 128),
            ("index_topk", 1024),
            ("first_k_dense_replace", 1),
            ("index_topk_freq", 1),
            ("index_skip_topk_offset", 2),
            ("index_topk_pattern", "FFFSSSF"),
        )
    },
    "no-speculation": ({}, _TP8, None, _SM103, r"speculative decoding off"),
    "mtp1": ({}, _TP8, _mtp(1), _SM103, r"max_draft_len=1 "),
    "mtp2": ({}, _TP8, _mtp(2), _SM103, r"max_draft_len=2 "),
    "mtp3": ({}, _TP8, _mtp(3), _SM103, r"max_draft_len=3 "),
    "mtp4": ({}, _TP8, _mtp(4), _SM103, r"max_draft_len=4 "),
    "mtp6": ({}, _TP8, _mtp(6), _SM103, r"max_draft_len=6 "),
    "vanilla-mtp": ({}, _TP8, _mtp(use_mtp_vanilla=True), _SM103, r"speculative decoding MTP "),
    "dynamic-tree": (
        {},
        _TP8,
        _mtp(use_dynamic_tree=True, dynamic_tree_max_topK=1),
        _SM103,
        r"dynamic tree",
    ),
    "suffix-automaton-enhancer": (
        {},
        _TP8,
        _mtp(sa_config=SAEnhancerConfig()),
        _SM103,
        r"with the suffix-automaton enhancer",
    ),
    "relaxed-acceptance": (
        {},
        _TP8,
        _mtp(use_relaxed_acceptance_for_thinking=True),
        _SM103,
        r"with relaxed acceptance",
    ),
    "ngram": (
        {},
        _TP8,
        NGramDecodingConfig(max_draft_len=5, max_matching_ngram_size=2),
        _SM103,
        r"speculative decoding NGRAM",
    ),
    # The dep claim: one fact of a claimed deployment changed.
    "dep4-mtp3": (
        {},
        dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True),
        _mtp(3),
        _SM103,
        r"ws=4 tp=4 .* attention_dp=True",
    ),
    "dep32-mtp3": (
        {},
        dict(world_size=32, tp_size=32, moe_ep_size=32, moe_tp_size=1, enable_attention_dp=True),
        _mtp(3),
        _SM103,
        r"ws=32 tp=32",
    ),
    "dep16-mtp5": ({}, _DEP16, _mtp(), _SM103, r"ws=16 .* max_draft_len=5 "),
    "dep16-lm-head-tp-mtp3": (
        {},
        dict(_DEP16, enable_lm_head_tp_in_adp=True),
        _mtp(3),
        _SM103,
        r"ws=16 .* lm_head_tp_in_adp=True, .* max_draft_len=3 ",
    ),
    "dep16-lm-head-tp-mtp5": (
        {},
        dict(_DEP16, enable_lm_head_tp_in_adp=True),
        _mtp(),
        _SM103,
        r"ws=16 .* lm_head_tp_in_adp=True, .* max_draft_len=5 ",
    ),
    "dep8-lm-head-tp-mtp3": (
        {},
        _DEP8_LM_HEAD_TP,
        _mtp(3),
        _SM103,
        r"lm_head_tp_in_adp=True, .* max_draft_len=3 ",
    ),
    **{
        f"dep8-mtp{draft}": ({}, _DEP8, _mtp(draft), _SM103, rf"max_draft_len={draft} ")
        for draft in (1, 2, 4, 5)
    },
    **{
        f"dep8-lm-head-tp-mtp{draft}": (
            {},
            _DEP8_LM_HEAD_TP,
            _mtp(draft),
            _SM103,
            rf"max_draft_len={draft} ",
        )
        for draft in (1, 2, 4, 6)
    },
    "dep8-moe-tp2": (
        {},
        dict(_DEP8, moe_ep_size=4, moe_tp_size=2),
        _mtp(3),
        _SM103,
        r"moe_tp=2 moe_ep=4 attention_dp=True",
    ),
    "dep8-pp2": (
        {},
        dict(
            world_size=16,
            tp_size=8,
            pp_size=2,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
        ),
        _mtp(3),
        _SM103,
        r"ws=16 tp=8 pp=2",
    ),
    "dep8-no-speculation": ({}, _DEP8, None, _SM103, r"speculative decoding off"),
    "dep8-vanilla-mtp": (
        {},
        _DEP8,
        _mtp(3, use_mtp_vanilla=True),
        _SM103,
        r"speculative decoding MTP ",
    ),
    "dep8-dynamic-tree": (
        {},
        _DEP8,
        _mtp(3, use_dynamic_tree=True, dynamic_tree_max_topK=1),
        _SM103,
        r"dynamic tree",
    ),
    "dep8-suffix-automaton-enhancer": (
        {},
        _DEP8,
        _mtp(3, sa_config=SAEnhancerConfig()),
        _SM103,
        r"with the suffix-automaton enhancer",
    ),
    "dep8-relaxed-acceptance": (
        {},
        _DEP8,
        _mtp(3, use_relaxed_acceptance_for_thinking=True),
        _SM103,
        r"with relaxed acceptance",
    ),
    "dep8-sm100": ({}, _DEP8, _mtp(3), (10, 0), r"compute capability \(10, 0\)"),
    "dep16-bf16-checkpoint": (
        dict(quantization_config=None),
        _DEP16,
        _mtp(3),
        _SM103,
        r"quantization none",
    ),
    "dep8-lm-head-tp-quantized-mtp-layer": (
        dict(quantization_config=_nvfp4_quantization(["lm_head", "model.embed_tokens"])),
        _DEP8_LM_HEAD_TP,
        _mtp(),
        _SM103,
        r"quantization NVFP4 with the MTP layer quantized",
    ),
    "dep8-shape-num_attention_heads": (
        {"num_attention_heads": 128},
        _DEP8,
        _mtp(3),
        _SM103,
        _shape_words(num_attention_heads=128),
    ),
}


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, spec_config, capability, missed",
    list(_FOREIGN.values()),
    ids=list(_FOREIGN),
)
def test_a_foreign_deployment_does_not_match(
    monkeypatch, config_kwargs, mapping_kwargs, spec_config, capability, missed
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    config = _model_config(_glm52_config(**config_kwargs), spec_config, **mapping_kwargs)

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="GlmMoeDsaForCausalLM") as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(excinfo.value)
    assert "the glm_moe_dsa targets are" in message
    assert "tp8 (8 ranks" in message and "tep (4 or 8 ranks" in message
    assert "dep (attention DP" in message
    assert re.search(missed, message), message


def test_a_config_without_the_sparse_attention_fields_does_not_match():
    """A GlmMoeDsa config that lacks a shape field declines instead of failing in route."""
    pretrained_config = _glm52_config()
    del pretrained_config.index_topk
    config = _model_config(pretrained_config, _mtp(), **_TP8)

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=_shape_words(index_topk=None)):
        modeling_v2_resolve(config, _llm_args("require"))


class _NoToDict:
    """A quantization declaration that is neither a dict nor convertible to one."""

    quant_method = "nvfp4"


# Quantization declarations route must decline without parsing through the engine's
# builders, which raise other errors than NoTarget on them: (declaration, what the
# message names when the rest of the deployment matches).
_UNREADABLE_QUANTIZATION = {
    "modelopt-mixed-precision-flat": (
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {"group_0": {}},
            "ignore": ["lm_head"],
        },
        r"quantization MIXED_PRECISION with the MTP layer quantized",
    ),
    "modelopt-mixed-precision-legacy": (
        {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION"}},
        r"quantization MIXED_PRECISION with the MTP layer quantized",
    ),
    "modelopt-unknown-algorithm": (
        {"quant_method": "modelopt", "quant_algo": "NOT_AN_ALGORITHM"},
        r"quantization a modelopt config without a readable global algorithm",
    ),
    "modelopt-no-algorithm": (
        {"quant_method": "modelopt", "ignore": ["model.layers.78*"]},
        r"quantization a modelopt config without a readable global algorithm",
    ),
    "modelopt-excluded-modules-not-a-list": (
        {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": 5},
        r"quantization modelopt NVFP4 with unreadable excluded modules",
    ),
    "fp8-64x64-blocks": (
        {"quant_method": "fp8", "weight_block_size": [64, 64]},
        r"quantization quant_method 'fp8'",
    ),
    "nvfp4-group-32": (
        {"quant_method": "nvfp4", "group_size": 32},
        r"quantization quant_method 'nvfp4'",
    ),
    "mxfp8-1x64-blocks": (
        {"quant_method": "mxfp8", "weight_block_size": [1, 64]},
        r"quantization quant_method 'mxfp8'",
    ),
    "compressed-tensors-without-strategy": (
        {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}},
        },
        r"quantization quant_method 'compressed-tensors'",
    ),
    "not-a-dict": (_NoToDict(), r"quantization declared as a _NoToDict, not a dict"),
}


@pytest.mark.parametrize("capability", [_SM103, (9, 0)], ids=["sm103", "sm90"])
@pytest.mark.parametrize(
    "quantization, named",
    list(_UNREADABLE_QUANTIZATION.values()),
    ids=list(_UNREADABLE_QUANTIZATION),
)
def test_a_quantization_route_cannot_read_declines(monkeypatch, quantization, named, capability):
    """Declined, never an exception other than NoTarget, so ``auto`` still falls back.

    The quantization is read only for the deployment the target serves; on any
    other deployment the message says it was not read.
    """
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    config = _model_config(_glm52_config(quantization_config=quantization), _mtp(), **_TP8)

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="GlmMoeDsaForCausalLM") as excinfo:
        modeling_v2_resolve(config, _llm_args("require"))
    expected = named if capability == _SM103 else r"quantization not read"
    assert re.search(expected, str(excinfo.value)), str(excinfo.value)

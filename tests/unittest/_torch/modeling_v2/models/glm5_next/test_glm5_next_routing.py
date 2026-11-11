# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash target routing, driven by synthetic configs on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM100 = (10, 0)
_DUMMY_MODEL = "/tmp/dummy_model"
_SPLIT = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling."
    "ModelingV2Glm53FlashNvfp4Sm100Default"
)
_FUSED = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default_mtp_fused.modeling."
    "ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm100(monkeypatch):
    """Route as if this were a B200, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM100)


def _llm_args(mode: str, **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _flash_config(**text_overrides) -> PretrainedConfig:
    """The composite config GLM-5.3-Flash's config.json declares, its text_config cut to the routed fields."""
    text_config = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        hidden_size=4096,
        n_routed_experts=288,
        q_lora_rank=1536,
        linear_num_heads=64,
    )
    text_config.update(text_overrides)
    return PretrainedConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=text_config,
    )


def _model_config(
    pretrained_config,
    quant_algo=QuantAlgo.NVFP4,
    spec_config=None,
    kv_cache_quant_algo=None,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TEP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
    )


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, enable_attention_dp=True)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config, mapping_kwargs",
    [
        (MTPDecodingConfig(max_draft_len=3), _TEP4),
        (MTPDecodingConfig(max_draft_len=1), dict(world_size=8, tp_size=8, moe_tp_size=8)),
        (MTPDecodingConfig(max_draft_len=3), _DEP4),
    ],
    ids=["tep4-mtp3", "tp8-mtp1", "dep4-mtp3"],
)
def test_linear_mtp_eagle_routes_to_the_fused_target(mode, spec_config, mapping_kwargs):
    """The fused target takes every linear one-model MTP-Eagle deployment it supports."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _FUSED


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "spec_config, mapping_kwargs, llm_kwargs",
    [
        (None, _TEP4, {}),
        (None, _DEP4, {}),
        (MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3, use_rejection_sampling=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3, use_relaxed_acceptance_for_thinking=True), _TEP4, {}),
        (MTPDecodingConfig(max_draft_len=3), _TEP4, dict(guided_decoding_backend="xgrammar")),
        (SADecodingConfig(max_draft_len=3), _TEP4, {}),
    ],
    ids=[
        "no-spec",
        "dep4-no-spec",
        "vanilla-mtp",
        "rejection-sampling",
        "relaxed-acceptance",
        "guided-decoding",
        "suffix-automaton",
    ],
)
def test_other_deployments_route_to_the_split_target(mode, spec_config, mapping_kwargs, llm_kwargs):
    """What the fused target does not compute falls to the split target."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **llm_kwargs))) == _SPLIT


@pytest.mark.parametrize(
    "text_overrides, quant_algo, spec_config, mapping_kwargs, missed",
    [
        (dict(num_hidden_layers=44), QuantAlgo.NVFP4, None, _TEP4, r"shape \(44, 4096"),
        (dict(), QuantAlgo.FP8_BLOCK_SCALES, None, _TEP4, "quantization FP8_BLOCK_SCALES"),
        (dict(), QuantAlgo.NVFP4, None, dict(world_size=8, tp_size=4, pp_size=2), "pp=2"),
        (
            dict(),
            QuantAlgo.NVFP4,
            NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            _TEP4,
            "NGRAM",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused"),
            _TEP4,
            "Eagle3DecodingConfig",
        ),
        (
            dict(),
            QuantAlgo.NVFP4,
            MTPDecodingConfig(max_draft_len=3),
            dict(_DEP4, enable_lm_head_tp_in_adp=True),
            "LM head split",
        ),
    ],
    ids=["other-shape", "fp8-checkpoint", "pipeline-parallel", "ngram", "eagle3", "adp-split-head"],
)
def test_flash_near_misses_do_not_match(
    text_overrides, quant_algo, spec_config, mapping_kwargs, missed
):
    config = _model_config(
        _flash_config(**text_overrides),
        quant_algo=quant_algo,
        spec_config=spec_config,
        **mapping_kwargs,
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_flash_on_sm103_does_not_match(monkeypatch):
    """The targets have been run on B200 only; a GB300 falls back."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_flash_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


# The block-FP8 target: the FP8_BLOCK_SCALES checkpoint on sm_100 / sm_103, tp4, FP8 KV cache,
# text only, with no speculation or one-model MTP-Eagle at three linear drafts.

_FP8 = "models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.modeling.ModelingV2Glm53FlashFp8Sm10xTp4"
_TP4 = dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, gpus_per_node=8)


def _fp8_config(
    spec_config=None, kv_cache_quant_algo=QuantAlgo.FP8, text_overrides=None, **mapping
):
    """The block-FP8 checkpoint at the deployment its target admits, unless overridden."""
    return _model_config(
        _flash_config(**(text_overrides or {})),
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        spec_config=spec_config,
        kv_cache_quant_algo=kv_cache_quant_algo,
        **(mapping or _TP4),
    )


def _fp8_llm_args(mode: str, **kwargs) -> TorchLlmArgs:
    kwargs.setdefault("disable_mm_encoder", True)
    return _llm_args(mode, **kwargs)


def _mtp3(**fields) -> MTPDecodingConfig:
    """MTP-Eagle at three drafts with ``fields`` set as given, bypassing the validators that would
    reject some of them on their own (a draft checkpoint path that does not exist, a tree without
    its sizes): routing reads the fields, not their consistency."""
    return MTPDecodingConfig(max_draft_len=3).model_copy(update=fields)


def _fp8_declined(config, llm_args_of, missed: str) -> str:
    """``auto`` falls back to the built-in model; ``require`` raises with ``missed`` in the reason,
    which it returns."""
    assert modeling_v2_resolve(config, llm_args_of("auto")) is None
    with pytest.raises(ValueError, match=missed) as raised:
        modeling_v2_resolve(config, llm_args_of("require"))
    return str(raised.value)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("spec_config", [None, _mtp3()], ids=["no-spec", "mtp3"])
@pytest.mark.parametrize("moe_backend", ["AUTO", "TRTLLM"])
@pytest.mark.parametrize("sm", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
def test_fp8_tp4_routes_to_the_fp8_target(monkeypatch, sm, moe_backend, spec_config, mode):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    llm_args = _fp8_llm_args(mode, moe_config={"backend": moe_backend})
    assert _qualified(modeling_v2_resolve(_fp8_config(spec_config), llm_args)) == _FP8


@pytest.mark.parametrize(
    "spec_config, missed",
    [
        (_mtp3(use_dynamic_tree=True), "a dynamic draft tree"),
        (_mtp3(speculative_model="/draft"), "a separate speculative checkpoint"),
        (_mtp3(draft_lm_head_dtype="mxfp8"), r"draft_lm_head_dtype=mxfp8 \(only auto\)"),
        (_mtp3(moe_backend="CUTLASS"), "draft MoE backend CUTLASS"),
    ],
    ids=["dynamic-tree", "draft-checkpoint", "mxfp8-draft-head", "draft-moe-backend"],
)
def test_fp8_drafter_options_it_does_not_build_are_declined(spec_config, missed):
    _fp8_declined(_fp8_config(spec_config), _fp8_llm_args, missed)


@pytest.mark.parametrize(
    "moe_update, args_update, missed",
    [
        (dict(backend="DEEPGEMM"), {}, r"MoE backend DEEPGEMM \(only AUTO or TRTLLM\)"),
        (dict(backend="CUTLASS"), {}, r"MoE backend CUTLASS \(only AUTO or TRTLLM\)"),
        (dict(load_balancer="eplb.yaml"), {}, r"a MoE load balancer \(eplb\.yaml\)"),
        ({}, dict(dwdp_config="dwdp"), r"DWDP \(dwdp\)"),
    ],
    ids=["deepgemm", "cutlass", "load-balancer", "dwdp"],
)
def test_fp8_moe_options_it_does_not_honour_are_declined(moe_update, args_update, missed):
    def llm_args_of(mode):
        args = _fp8_llm_args(mode)
        moe_config = args.moe_config.model_copy(update=moe_update)
        return args.model_copy(update=dict(args_update, moe_config=moe_config))

    _fp8_declined(_fp8_config(), llm_args_of, missed)


def test_fp8_shared_expert_fusion_is_declined(monkeypatch):
    monkeypatch.setenv("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION", "1")
    missed = r"shared-expert fusion \(TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1\)"
    _fp8_declined(_fp8_config(), _fp8_llm_args, missed)


@pytest.mark.parametrize(
    "config_kwargs, llm_kwargs, missed",
    [
        (
            dict(text_overrides=dict(num_hidden_layers=78)),
            {},
            r"text_config shape \(78, 4096, 288, 1536, 64\) \(only",
        ),
        (dict(kv_cache_quant_algo=None), {}, r"KV cache bf16 \(only FP8\)"),
        (dict(kv_cache_quant_algo=QuantAlgo.NVFP4), {}, r"KV cache NVFP4 \(only FP8\)"),
        (dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1), {}, "moe_tp=1 moe_ep=4"),
        (
            dict(world_size=4, tp_size=4, moe_tp_size=4, enable_attention_dp=True),
            {},
            "attention DP",
        ),
        (dict(world_size=8, tp_size=8, moe_tp_size=8), {}, "world_size=8 tp=8"),
        (dict(spec_config=MTPDecodingConfig(max_draft_len=2)), {}, r"max_draft_len=2 \(only 3\)"),
        (
            dict(spec_config=MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True)),
            {},
            "only one-model MTP-Eagle",
        ),
        ({}, dict(disable_mm_encoder=False), "the vision encoder"),
    ],
    ids=[
        "other-shape",
        "bf16-kv",
        "nvfp4-kv",
        "expert-parallel",
        "attention-dp",
        "tp8",
        "mtp2",
        "vanilla-mtp",
        "vision-tower",
    ],
)
def test_fp8_near_misses_do_not_match(config_kwargs, llm_kwargs, missed):
    def llm_args_of(mode):
        return _fp8_llm_args(mode, **llm_kwargs)

    _fp8_declined(_fp8_config(**config_kwargs), llm_args_of, missed)


def test_fp8_refusal_names_only_what_failed():
    """A deployment that misses one requirement is told that one, not every fact it has."""
    reason = _fp8_declined(
        _fp8_config(kv_cache_quant_algo=None), _fp8_llm_args, r"KV cache bf16 \(only FP8\)"
    )
    for admitted in ("world_size", "moe_tp", "attention DP", "MoE backend", "vision", "draft"):
        assert admitted not in reason, f"{admitted!r} is admitted but reported: {reason}"


@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, None], ids=["nvfp4", "unquantized"])
def test_other_checkpoints_do_not_reach_the_fp8_target(quant_algo):
    """Another quantization of the same model never lands on the block-FP8 target."""
    config = _model_config(
        _flash_config(), quant_algo=quant_algo, kv_cache_quant_algo=QuantAlgo.FP8, **_TP4
    )
    for mode in ("auto", "require"):
        try:
            target = modeling_v2_resolve(config, _fp8_llm_args(mode))
        except ValueError:
            continue
        assert target is None or _qualified(target) != _FP8


@pytest.mark.parametrize("sm", [(9, 0), (12, 0)], ids=["sm90", "sm120"])
def test_fp8_on_other_gpus_does_not_match(monkeypatch, sm):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    missed = rf"compute capability \({sm[0]}, {sm[1]}\) \(only sm_100 / sm_103\)"
    _fp8_declined(_fp8_config(), _fp8_llm_args, missed)

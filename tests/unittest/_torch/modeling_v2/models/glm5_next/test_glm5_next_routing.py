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
    KvCacheConfig,
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
_TP4_TARGET = (
    "models.glm5_next.glm_5_3_flash_nvfp4__sm_10x__tp4.modeling.ModelingV2Glm53FlashNvfp4Sm10xTp4"
)
_TEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1)
_TP4 = dict(world_size=4, tp_size=4, moe_ep_size=1, moe_tp_size=4)


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


def _tp4_engine(**kwargs) -> dict:
    """The engine the tp4 target is served with: 128-token KV cache pages, up to 256 sequences per step."""
    return {
        "max_batch_size": 32,
        "kv_cache_config": KvCacheConfig(tokens_per_block=128),
        **kwargs,
    }


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
    pretrained_config, quant_algo=QuantAlgo.NVFP4, spec_config=None, **mapping_kwargs
):
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**(mapping_kwargs or _TEP4)),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo),
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
    """The default targets have been run on B200 only; a GB300 the tp4 target declines falls back."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    config = _model_config(_flash_config())
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 3\)"):
        modeling_v2_resolve(config, _llm_args("require"))


# --- The tp4 target: sm_100 and sm_103, with the engine it is written for ---------


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("sm", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
@pytest.mark.parametrize(
    "spec_config",
    [
        None,
        MTPDecodingConfig(max_draft_len=3),
        # the served deployment: temperature-only sampling, which the step's own head samples
        MTPDecodingConfig(max_draft_len=3, advanced_sampling_mode="no_topk_no_topp"),
    ],
    ids=["no-spec", "mtp3", "mtp3-temperature-only"],
)
@pytest.mark.parametrize("layout", [_TP4, _TEP4], ids=["tp4", "tp4-ep4"])
def test_tp4_engine_routes_to_the_tp4_target(mode, sm, spec_config, layout, monkeypatch):
    """The tp4 target takes every deployment it supports, ahead of the default targets."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(_flash_config(), spec_config=spec_config, **layout)
    target = modeling_v2_resolve(config, _llm_args(mode, **_tp4_engine()))
    assert _qualified(target) == _TP4_TARGET


@pytest.mark.parametrize(
    "spec_config, mapping_kwargs, engine, on_sm100, missed",
    [
        # attention DP splits the attention weights differently: another layout, not a knob
        (None, _DEP4, {}, _SPLIT, "attention_dp=True"),
        # a measured layout splits the experts by expert or by intermediate, not both
        (
            None,
            dict(world_size=4, tp_size=4, moe_ep_size=2, moe_tp_size=2),
            {},
            _SPLIT,
            "moe_tp=2 moe_ep=2",
        ),
        # the KDA replay records hold exactly three drafts
        (MTPDecodingConfig(max_draft_len=2), _TP4, {}, _FUSED, "max_draft_len=2"),
        # vanilla MTP is the multi-model loop; the target carries the one-model drafter
        (
            MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
            _TP4,
            {},
            _SPLIT,
            "speculative decoding",
        ),
        # the kernels read and write a BF16 latent cache
        (
            None,
            _TP4,
            dict(kv_cache_config=KvCacheConfig(dtype="fp8", tokens_per_block=128)),
            _SPLIT,
            "KV cache dtype fp8",
        ),
        # the kernels address 128-token pages (the LLM API's default is 32)
        (
            MTPDecodingConfig(max_draft_len=3),
            _TEP4,
            dict(kv_cache_config=KvCacheConfig()),
            _FUSED,
            "tokens_per_block 32",
        ),
        # the step program's head takes up to 256 sequences per step (the LLM API's default
        # max_batch_size is 2048)
        (None, _TP4, dict(max_batch_size=512), _SPLIT, "max_batch_size 512"),
        # a step's rows are split over the 4 ranks
        (None, _TP4, dict(max_num_tokens=8190), _SPLIT, "max_num_tokens 8190"),
    ],
    ids=[
        "attention-dp",
        "moe-tp2-ep2",
        "mtp2",
        "vanilla-mtp",
        "fp8-kv",
        "pages-32",
        "batch-512",
        "max-num-tokens-8190",
    ],
)
def test_what_the_tp4_target_declines_goes_to_the_default_targets(
    spec_config, mapping_kwargs, engine, on_sm100, missed, monkeypatch
):
    """A deployment the tp4 target declines lands on a default target on sm_100, and on none on sm_103."""
    config = _model_config(_flash_config(), spec_config=spec_config, **mapping_kwargs)
    engine = _tp4_engine(**engine)
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **engine))) == on_sm100

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    assert modeling_v2_resolve(config, _llm_args("auto", **engine)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **engine))


@pytest.mark.parametrize(
    "spec_kwargs, engine, on_sm100, missed",
    [
        (dict(use_rejection_sampling=True), {}, _SPLIT, "rejection sampling"),
        (dict(use_relaxed_acceptance_for_thinking=True), {}, _SPLIT, "relaxed acceptance"),
        (dict(enable_penalty=True), {}, _SPLIT, "occurrence penalties"),
        (
            dict(use_dynamic_tree=True, dynamic_tree_max_topK=2),
            {},
            _SPLIT,
            "a dynamic draft tree",
        ),
        (dict(draft_len_schedule={4: 3, 16: 1}), {}, _SPLIT, "a draft-length schedule"),
        # speculation off above a batch size is a draft-length schedule too
        (dict(max_concurrency=8), {}, _SPLIT, "a draft-length schedule"),
        (
            dict(acceptance_rate_window_size=32, acceptance_rate_threshold=0.2),
            {},
            _FUSED,
            "an acceptance-rate threshold",
        ),
        # the step's head samples without a grammar's bitmask
        ({}, dict(guided_decoding_backend="xgrammar"), _SPLIT, "guided decoding"),
    ],
    ids=[
        "rejection-sampling",
        "relaxed-acceptance",
        "penalties",
        "dynamic-tree",
        "draft-len-schedule",
        "max-concurrency",
        "acceptance-gate",
        "guided-decoding",
    ],
)
def test_mtp_features_the_tp4_forward_does_not_compute_go_to_the_default_targets(
    spec_kwargs, engine, on_sm100, missed, monkeypatch
):
    """The tp4 forward runs the MTP layer as a linear chain of three drafts with strict acceptance, every step.

    A feature that changes what a step computes, or how many drafts it runs, routes elsewhere rather than being
    served as something else.
    """
    spec_config = MTPDecodingConfig(max_draft_len=3, **spec_kwargs)
    config = _model_config(_flash_config(), spec_config=spec_config, **_TP4)
    engine = _tp4_engine(**engine)
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **engine))) == on_sm100

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    assert modeling_v2_resolve(config, _llm_args("auto", **engine)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **engine))


@pytest.mark.parametrize("sm", [(9, 0), (12, 0)], ids=["sm90", "sm120"])
def test_flash_on_other_sms_does_not_match(sm, monkeypatch):
    """Every target's kernels are sm_10x code."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    config = _model_config(_flash_config(), **_TP4)
    assert modeling_v2_resolve(config, _llm_args("auto", **_tp4_engine())) is None

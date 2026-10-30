# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-NVFP4 routing at four ranks, driven by synthetic configs on CPU.

Two families route `Qwen4ExpForConditionalGeneration`: `qwen4_exp` claims one rank and
`qwen4_exp_for_conditional_generation` claims tp4, tep4 and dep4. Each layout lands on
its own family's target, and the layouts neither one serves fall back under `auto`.
"""

from dataclasses import replace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    modeling_v2_resolve,
    registered_routes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import routing as tp1_routing
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation import (  # noqa: E501
    routing,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
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
_ARCH = "Qwen4ExpForConditionalGeneration"
_TARGET = (
    "models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.modeling."
    "ModelingV2HfFc694b5OrigSm103Unified"
)
_TP1_TARGET = (
    "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."
    "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1"
)

# The three layouts, as the LLM API's parallel arguments build them. tp4 leaves the
# expert split to Mapping's default, which is the TP size.
_LAYOUTS = {
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}

_SPECS = [
    None,
    MTPDecodingConfig(max_draft_len=1),
    MTPDecodingConfig(max_draft_len=3),
    MTPDecodingConfig(max_draft_len=7),
]
_SPEC_IDS = ["no-spec", "mtp1", "mtp3", "mtp7"]


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs, with the GDN
    cached replay at its default (on)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _qwen4_exp_config(text_overrides=None, **vision_overrides):
    """The shape Qwen3.8-Flash-Next-NVFP4's normalized composite config declares."""
    text = dict(
        num_hidden_layers=48,
        hidden_size=2560,
        num_experts=512,
        num_attention_heads=24,
        linear_num_value_heads=48,
        vocab_size=248320,
    )
    text.update(text_overrides or {})
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
        architectures=[_ARCH],
        text_config=PretrainedConfig(**text),
        vision_config=PretrainedConfig(**vision),
    )


def _model_config(spec_config=None, pretrained_config=None, **mapping_kwargs) -> ModelConfig:
    """The loaded config; the checkpoint's own KV is bf16 (no KV-cache algorithm)."""
    return ModelConfig(
        pretrained_config=pretrained_config or _qwen4_exp_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(kv_cache_quant_algo=None),
    )


def _claimed(config, **llm_args_kwargs) -> None:
    for mode in ("auto", "require"):
        target = modeling_v2_resolve(config, _llm_args(mode, **llm_args_kwargs))
        assert target is not None and _qualified(target) == _TARGET, mode


def _declined(config, match=None, **llm_args_kwargs) -> None:
    """``auto`` falls back to the built-in model and ``require`` says why."""
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_args_kwargs)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", **llm_args_kwargs))


# --- the registry -----------------------------------------------------------------


def test_both_families_route_the_architecture_one_rank_first():
    """The single-rank family registers first (`models/__init__.py` imports in order)."""
    assert registered_routes()[_ARCH] == (tp1_routing.route, routing.route)


def test_routing_admits_exactly_what_the_target_implements():
    """`route` declines a MoE backend, a draft length or a recurrent-state dtype the
    target does not run; this keeps routing's constants and the target's together."""
    assert set(routing._MOE_BACKENDS) == set(M._TRUNK_MOE_BACKENDS)
    assert routing._MAX_DRAFT_LEN == M._MAX_VERIFY_WIDTH - 1
    assert routing._MAX_LEGACY_DRAFT_LEN == M._MAX_LEGACY_VERIFY_WIDTH - 1
    assert routing._SSM_STATE_DTYPES == M._SSM_STATE_DTYPES
    assert routing._SM == M._SM


# --- each layout lands on its own family's target ---------------------------------


@pytest.mark.parametrize("spec_config", _SPECS, ids=_SPEC_IDS)
@pytest.mark.parametrize("member", sorted(_LAYOUTS))
def test_each_member_routes_to_the_unified_target(member, spec_config):
    config = _model_config(spec_config, **_LAYOUTS[member])
    assert routing.layout(config.mapping) == member
    for dtype in ("bfloat16", "auto"):
        _claimed(config, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))


@pytest.mark.parametrize("spec_config", [None, MTPDecodingConfig(max_draft_len=3)])
def test_the_one_rank_deployment_still_routes_to_tp1(spec_config):
    config = _model_config(spec_config)
    for mode in ("auto", "require"):
        assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TP1_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(world_size=2, tp_size=2),
        dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
        dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
        dict(world_size=8, tp_size=8),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_attention_dp=True),
    ],
    ids=["tp2", "tep2", "dep2", "tp8", "tep8", "dep8"],
)
def test_two_or_eight_ranks_decline_in_both_families(mapping_kwargs):
    """Neither family claims it, and ``require`` names each family's reason."""
    config = _model_config(**mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError) as raised:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(raised.value)
    assert "qwen4_exp: " in message, message
    assert "qwen4_exp_for_conditional_generation: " in message, message
    assert f"world_size={mapping_kwargs['world_size']}" in message, message


# --- near misses at four ranks ------------------------------------------------------


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # attention DP with the experts split by intermediate
        dict(_LAYOUTS["dep4"], moe_tp_size=4, moe_ep_size=1),
        # experts split two ways by intermediate and two by expert
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
        # pipeline parallelism
        dict(world_size=4, tp_size=2, pp_size=2),
        # lm_head TP under attention DP
        dict(_LAYOUTS["dep4"], enable_lm_head_tp_in_adp=True),
        # context parallelism
        dict(world_size=4, tp_size=2, cp_size=2),
        # four ranks over two nodes
        dict(_LAYOUTS["tep4"], gpus_per_node=2),
    ],
    ids=["dep4-moetp4", "tp4-moetp2-ep2", "tp2-pp2", "dep4-lm-head-tp", "tp2-cp2", "two-nodes"],
)
def test_other_layouts_at_four_ranks_decline(mapping_kwargs):
    config = _model_config(**mapping_kwargs)
    assert routing.layout(config.mapping) is None
    _declined(config)


@pytest.mark.parametrize(
    "spec_config",
    [
        NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
        MTPDecodingConfig(max_draft_len=8),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
    ],
    ids=["ngram", "draft8", "vanilla-mtp"],
)
def test_unsupported_speculation_declines(spec_config):
    _declined(_model_config(spec_config, **_LAYOUTS["tep4"]))


def test_another_checkpoint_shape_declines():
    config = _model_config(
        pretrained_config=_qwen4_exp_config(dict(num_hidden_layers=60)), **_LAYOUTS["tep4"]
    )
    _declined(config, match="text shape")


def test_a_deepstack_vision_tower_declines():
    config = _model_config(
        pretrained_config=_qwen4_exp_config(deepstack_visual_indexes=[5, 11, 17]),
        **_LAYOUTS["tep4"],
    )
    _declined(config, match="vision shape")


def test_sm100_declines(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    _declined(_model_config(**_LAYOUTS["tep4"]), match=r"compute capability \(10, 0\)")


# --- engine choices the target does not implement ---------------------------------


@pytest.mark.parametrize("dtype", ["fp8", "nvfp4"])
@pytest.mark.parametrize("member", sorted(_LAYOUTS))
def test_quantized_kv_cache_dtypes_match(member, dtype):
    _claimed(_model_config(**_LAYOUTS[member]), kv_cache_config=KvCacheConfig(dtype=dtype))


def test_other_kv_cache_algorithms_decline():
    choices = routing.engine_choices(_model_config(**_LAYOUTS["tep4"]), _llm_args())
    for algo in (QuantAlgo.INT8, QuantAlgo.FP8_BLOCK_SCALES):
        with pytest.raises(routing.NoTarget, match="KV cache"):
            routing._check_engine(None, algo, choices)
    _declined(
        _model_config(**_LAYOUTS["tep4"]),
        match="KV cache",
        kv_cache_config=KvCacheConfig(dtype="fp8_ds_mla"),
    )


@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM"])
def test_the_trtllm_moe_backend_matches(backend):
    _claimed(replace(_model_config(**_LAYOUTS["tep4"]), moe_backend_requested=backend))


@pytest.mark.parametrize("backend", ["CUTEDSL", "CUTLASS", "TRITON", "DEEPGEMM"])
def test_other_moe_backends_decline(backend):
    config = replace(_model_config(**_LAYOUTS["tep4"]), moe_backend_requested=backend)
    _declined(config, match="moe_config.backend")


def test_a_float32_state_cache_declines():
    _declined(
        _model_config(**_LAYOUTS["dep4"]),
        match="recurrent-state cache",
        kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32"),
    )


@pytest.mark.parametrize("backend", ["disabled", "breakable"])
def test_no_or_the_breakable_prefill_cuda_graph_matches(backend):
    _claimed(_model_config(**_LAYOUTS["tep4"]), prefill_cuda_graph_backend=backend)


def test_the_piecewise_prefill_cuda_graph_declines():
    _declined(
        _model_config(**_LAYOUTS["tep4"]), match="piecewise", prefill_cuda_graph_backend="piecewise"
    )


@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_a_draft_beyond_three_needs_the_cached_replay(monkeypatch, max_draft_len):
    config = _model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_LAYOUTS["tep4"])
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    _declined(config, match="TRTLLM_USE_GDN_REPLAY=0")
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    _claimed(config)


def test_three_drafts_match_without_the_cached_replay(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    _claimed(_model_config(MTPDecodingConfig(max_draft_len=3), **_LAYOUTS["tep4"]))

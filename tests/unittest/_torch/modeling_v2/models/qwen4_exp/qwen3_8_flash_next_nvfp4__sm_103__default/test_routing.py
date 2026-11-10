# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-NVFP4 routing at its layouts, driven by synthetic configs on CPU.

`qwen4_exp` routes to its one target one rank (tp1), tp4 and tep4 with or without MTP, and
the attention-DP layouts dep4 (with or without MTP) and dep2 (the disaggregated-serving
engines' layout, without speculation), neither with a breakable prefill graph. The layouts
the target does not serve fall back under `auto`. The tp1 engine choices are exercised in
`tests/unittest/_torch/modeling_v2/test_modeling_v2_routing.py`.
"""

from dataclasses import replace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    MoeConfig,
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
    "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.modeling."
    "ModelingV2Qwen38FlashNextNvfp4Sm103Default"
)

# The four-rank layouts, as the LLM API's parallel arguments build them. tp4 leaves the
# expert split to Mapping's default, which is the TP size.
_LAYOUTS = {
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}

# The two-rank layout both engines of the disaggregated-serving test run: attention DP,
# the routed experts in two 256-expert windows.
_DEP2 = dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True)

# The layouts with attention TP, which serve MTP and the breakable prefill graph, and the
# attention-DP ones, which serve neither.
_ATTENTION_TP = ["tp4", "tep4"]
_ATTENTION_DP = {"dep4": _LAYOUTS["dep4"], "dep2": _DEP2}

# Every draft length the target runs.
_DRAFT_LENS = list(range(1, 8))


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


# The checkpoint's `mtp` block.
_MTP = dict(
    hybrid=True,
    layer_types=["full_attention"],
    mtp_use_hidden_state_from_layer=None,
    num_hidden_layers=1,
    rope_theta=10000000,
)


def _qwen4_exp_config(text_overrides=None, **vision_overrides):
    """The shape Qwen3.8-Flash-Next-NVFP4's normalized composite config declares."""
    text = dict(
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
        mtp=dict(_MTP),
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
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


def test_routing_and_the_target_agree_on_the_gpu():
    """`route` gates on the GPU before it imports the target, whose constructor asserts the
    same one; the target's `check_deployment` holds every other admission fact."""
    assert routing._SM == M._SM


# --- each layout lands on the default target ------------------------------------


@pytest.mark.parametrize("member", sorted(_LAYOUTS))
def test_each_member_routes_to_the_default_target(member):
    config = _model_config(**_LAYOUTS[member])
    assert routing.layout(config.mapping) == member
    for dtype in ("bfloat16", "auto"):
        _claimed(config, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))


@pytest.mark.parametrize("max_draft_len", _DRAFT_LENS)
@pytest.mark.parametrize("member", _ATTENTION_TP)
def test_mtp_at_every_draft_length_routes_under_attention_tp(member, max_draft_len):
    """The tune runs MTP at tp4 and tep4 over draft lengths 1 to 7."""
    config = _model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_LAYOUTS[member])
    assert routing.layout(config.mapping) == member
    for dtype in ("bfloat16", "auto"):
        _claimed(config, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))


@pytest.mark.parametrize("dtype", ["fp8", "nvfp4", "auto"])
def test_dep2_routes_to_the_default_target(dtype):
    config = _model_config(**_DEP2)
    assert routing.layout(config.mapping) == "dep2"
    _claimed(config, kv_cache_config=KvCacheConfig(dtype=dtype, mamba_ssm_cache_dtype="bfloat16"))


@pytest.mark.parametrize("max_draft_len", _DRAFT_LENS)
def test_mtp_at_every_draft_length_routes_at_dep4(max_draft_len):
    """Each dep4 rank drafts for its own requests and the draft layer's FP8 expert window runs
    the trunk's attention-DP round trip: the tune's ht_dep4 config may serve MTP."""
    config = _model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_LAYOUTS["dep4"])
    assert routing.layout(config.mapping) == "dep4"
    for dtype in ("bfloat16", "auto"):
        _claimed(config, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))
    _claimed(config, kv_cache_config=KvCacheConfig(dtype="fp8", mamba_ssm_cache_dtype="bfloat16"))


# --- what the target does not run under attention DP ------------------------------


@pytest.mark.parametrize("max_draft_len", [1, 3, 7])
def test_dep2_declines_speculation(max_draft_len):
    """MTP under attention DP is admitted only where its draft acceptance was measured (dep4);
    neither family claims dep2 with speculation."""
    config = _model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_DEP2)
    assert routing.layout(config.mapping) == "dep2"
    _declined(config, match=r"speculative decoding with attention DP \(dep2\)")


@pytest.mark.parametrize("member", sorted(_ATTENTION_DP))
def test_attention_dp_declines_the_breakable_prefill_graph(member):
    """The engine pads every attention-DP rank to one prefill bucket, which a rank holding
    only generation requests cannot take; neither family claims it."""
    config = _model_config(**_ATTENTION_DP[member])
    _declined(
        config,
        match=rf"prefill_cuda_graph_backend='breakable' with attention DP \({member}\)",
        prefill_cuda_graph_backend="breakable",
    )
    _claimed(config, prefill_cuda_graph_backend="disabled")


def test_attention_dp_declines_speculation_before_the_engine_choices(monkeypatch):
    """MTP without the cached replay is declined at dep2 for attention DP: that is the
    structural reason, and no engine choice would lift it. At dep4 the same draft is declined
    for the missing cached replay alone."""
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _model_config(MTPDecodingConfig(max_draft_len=7), **_DEP2)
    _declined(config, match=r"speculative decoding with attention DP \(dep2\)")
    config = _model_config(MTPDecodingConfig(max_draft_len=7), **_LAYOUTS["dep4"])
    _declined(config, match="TRTLLM_USE_GDN_REPLAY=0")


def test_dep4_declines_mtp_with_the_breakable_prefill_graph():
    """Admitting MTP at dep4 keeps the breakable prefill graph's refusal under attention DP."""
    config = _model_config(MTPDecodingConfig(max_draft_len=3), **_LAYOUTS["dep4"])
    _declined(
        config,
        match=r"prefill_cuda_graph_backend='breakable' with attention DP \(dep4\)",
        prefill_cuda_graph_backend="breakable",
    )


@pytest.mark.parametrize("spec_config", [None, MTPDecodingConfig(max_draft_len=3)])
def test_the_one_rank_deployment_routes_to_the_default_target_at_tp1(spec_config):
    config = _model_config(spec_config)
    assert routing.layout(config.mapping) == "tp1"
    _claimed(config)
    # The one-rank engine choices: the CuteDSL experts and the fp32 recurrent state.
    _claimed(
        replace(config, moe_backend_requested="CUTEDSL"),
        kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32"),
    )


def test_attention_dp_at_one_rank_declines():
    config = _model_config(world_size=1, tp_size=1, enable_attention_dp=True)
    assert routing.layout(config.mapping) is None
    _declined(config)


@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        dict(world_size=2, tp_size=2),
        dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
        dict(_DEP2, moe_tp_size=2, moe_ep_size=1),
        dict(_DEP2, enable_lm_head_tp_in_adp=True),
        dict(world_size=8, tp_size=8),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_attention_dp=True),
    ],
    ids=["tp2", "tep2", "dep2-moetp2", "dep2-lm-head-tp", "tp8", "tep8", "dep8"],
)
def test_other_two_or_eight_rank_layouts_decline(mapping_kwargs):
    """The target does not claim it, and ``require`` names the layout and what the target
    serves."""
    config = _model_config(**mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError) as raised:
        modeling_v2_resolve(config, _llm_args("require"))
    message = str(raised.value)
    assert "the qwen4_exp target is" in message, message
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


_ROPE = dict(type="mrope", mrope_section=[11, 11, 10], mrope_interleaved=True)


@pytest.mark.parametrize(
    "override, field",
    [
        (dict(norm_topk_prob=False), "norm_topk_prob"),
        (dict(num_experts_per_tok=8), "num_experts_per_tok"),
        (dict(hidden_act="gelu"), "hidden_act"),
        (dict(num_key_value_heads=4), "num_key_value_heads"),
        (dict(attention_bias=True), "attention_bias"),
        (dict(output_gate_type="silu"), "output_gate_type"),
        (dict(tie_word_embeddings=True), "tie_word_embeddings"),
        (dict(rope_theta=1e6), "rope_theta"),
        (dict(partial_rotary_factor=0.5), "partial_rotary_factor"),
        (dict(rope_scaling=dict(_ROPE, mrope_interleaved=False)), "mrope_interleaved"),
        (dict(rope_scaling=dict(_ROPE, mrope_section=[16, 8, 8])), "mrope_section"),
        (dict(rope_scaling=dict(_ROPE, factor=4.0)), "rope_factor"),
        (dict(attn_output_gate=False), "attn_output_gate"),
        (dict(mtp_num_hidden_layers=2), "mtp_num_hidden_layers"),
        (dict(mtp_use_dedicated_embeddings=True), "mtp_use_dedicated_embeddings"),
        (dict(mtp=dict(_MTP, num_hidden_layers=2)), "mtp.num_hidden_layers"),
        (dict(mtp=dict(_MTP, layer_types=["linear_attention"])), "mtp.layer_types"),
        (
            dict(mtp=dict(_MTP, mtp_use_hidden_state_from_layer=40)),
            "mtp.mtp_use_hidden_state_from_layer",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_a_same_shape_checkpoint_with_other_settings_declines(override, field):
    """A fine-tune with the checkpoint's shape that changes what it computes, which the target
    would run with wrong logits (or wrong drafts) and no error: `auto` keeps the built-in
    model."""
    config = _model_config(pretrained_config=_qwen4_exp_config(override), **_LAYOUTS["tep4"])
    _declined(config, match=f"'{field}'")


def test_the_mtp_block_is_read_as_a_mapping_or_a_config_object():
    """An `mtp` block that is a config object is read field by field as a mapping is, and
    one that omits the block's fields (or the whole block) takes the checkpoint's values."""
    as_object = PretrainedConfig(**_MTP)
    tep4 = _LAYOUTS["tep4"]
    _claimed(_model_config(pretrained_config=_qwen4_exp_config(dict(mtp=as_object)), **tep4))
    as_object.mtp_use_hidden_state_from_layer = 40
    _declined(
        _model_config(pretrained_config=_qwen4_exp_config(dict(mtp=as_object)), **tep4),
        match="'mtp.mtp_use_hidden_state_from_layer'",
    )
    bare = dict(mtp=None, mtp_num_hidden_layers=None, mtp_use_dedicated_embeddings=None)
    _claimed(_model_config(pretrained_config=_qwen4_exp_config(bare), **tep4))


def test_the_settings_read_off_the_rope_parameters_dict():
    """A config that keeps its rope fields in `rope_parameters` (theta and partial factor
    inside) is read the same way, as the target reads it."""
    rope = dict(
        rope_type="default",
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    text = dict(
        rope_parameters=rope, rope_scaling=None, rope_theta=None, partial_rotary_factor=None
    )
    _claimed(_model_config(pretrained_config=_qwen4_exp_config(text), **_LAYOUTS["tep4"]))
    text["rope_parameters"] = dict(rope, rope_theta=5e6)
    _declined(
        _model_config(pretrained_config=_qwen4_exp_config(text), **_LAYOUTS["tep4"]),
        match="'rope_theta'",
    )


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


@pytest.mark.parametrize(
    "chunked, reuse",
    [(False, False), (True, False), (False, True)],
    ids=["neither", "chunked_prefill", "block_reuse"],
)
@pytest.mark.parametrize("dtype", ["fp8", "nvfp4"])
def test_a_quantized_kv_cache_matches_with_or_without_the_paged_context_path(dtype, chunked, reuse):
    """The engine enables its paged-context attention only for chunked prefill or block reuse;
    the target takes it for an NVFP4 pool either way, so neither setting is declined."""
    _claimed(
        _model_config(**_LAYOUTS["tep4"]),
        enable_chunked_prefill=chunked,
        kv_cache_config=KvCacheConfig(dtype=dtype, enable_block_reuse=reuse),
    )


@pytest.mark.parametrize("value", ["1", "force"])
@pytest.mark.parametrize("max_draft_len", [3, 7])
def test_mtp_matches_with_the_flashinfer_ring_replay(monkeypatch, max_draft_len, value):
    """`TRTLLM_USE_GDN_FLASHINFER_REPLAY` lays the replay history out as FlashInfer's ring at 3
    and 7 drafts over a bf16 state, which the target's verify step runs."""
    monkeypatch.setenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", value)
    _claimed(_model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_LAYOUTS["tep4"]))


@pytest.mark.parametrize("algo", [QuantAlgo.INT8, QuantAlgo.FP8_BLOCK_SCALES])
def test_another_checkpoint_kv_cache_algorithm_declines(algo):
    """`kv_cache_config.dtype: auto` keeps the checkpoint's own KV-cache algorithm."""
    config = _model_config(**_LAYOUTS["tep4"])
    config.quant_config.kv_cache_quant_algo = algo
    _declined(config, match="KV cache")


def test_the_fp8_ds_mla_kv_cache_declines():
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
    """CUTEDSL included: its expert pipeline runs every expert on one rank (tp1)."""
    config = replace(_model_config(**_LAYOUTS["tep4"]), moe_backend_requested=backend)
    _declined(config, match="moe_config.backend")


def test_a_float32_state_cache_declines():
    _declined(
        _model_config(**_LAYOUTS["dep4"]),
        match="recurrent-state cache",
        kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32"),
    )


@pytest.mark.parametrize("backend", ["disabled", "breakable"])
@pytest.mark.parametrize("member", _ATTENTION_TP)
def test_no_or_the_breakable_prefill_cuda_graph_matches(member, backend):
    _claimed(_model_config(**_LAYOUTS[member]), prefill_cuda_graph_backend=backend)


def test_the_piecewise_prefill_cuda_graph_declines():
    _declined(
        _model_config(**_LAYOUTS["tep4"]), match="piecewise", prefill_cuda_graph_backend="piecewise"
    )


@pytest.mark.parametrize("member", ["tep4", "dep4", "dep2"])
def test_eplb_declines_at_the_expert_parallel_layouts(member):
    """The engine builds an EPLB balancer only with expert parallelism; the target's windows
    are fixed, so `auto` keeps the built-in model, which balances."""
    mapping = _DEP2 if member == "dep2" else _LAYOUTS[member]
    eplb = MoeConfig(load_balancer={"num_slots": 512})
    _declined(_model_config(**mapping), match=r"load_balancer \(EPLB\)", moe_config=eplb)
    _claimed(_model_config(**mapping))


@pytest.mark.parametrize("mapping_kwargs", [{}, _LAYOUTS["tp4"]], ids=["tp1", "tp4"])
def test_eplb_without_expert_parallelism_still_matches(mapping_kwargs):
    """Without expert parallelism the engine builds no balancer, as for the built-in model."""
    _claimed(
        _model_config(**mapping_kwargs), moe_config=MoeConfig(load_balancer={"num_slots": 512})
    )


@pytest.mark.parametrize("max_draft_len", [1, 3, 7])
def test_mtp_needs_the_cached_replay(monkeypatch, max_draft_len):
    """The target verifies on the GDN cached replay only; without speculation it is unused."""
    config = _model_config(MTPDecodingConfig(max_draft_len=max_draft_len), **_LAYOUTS["tep4"])
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    _declined(config, match="TRTLLM_USE_GDN_REPLAY=0")
    _claimed(_model_config(**_LAYOUTS["tep4"]))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    _claimed(config)

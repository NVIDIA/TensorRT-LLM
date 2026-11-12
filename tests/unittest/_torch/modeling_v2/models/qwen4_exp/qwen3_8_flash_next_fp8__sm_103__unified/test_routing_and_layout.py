# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-FP8 unified target: routing and deployment checks, each rank's share of the
core (on the meta device) and the per-rank KV / recurrent pool layout."""

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    pool as P,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import QSASparseAttentionConfig
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    MultimodalConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# ------------------------------------------------------------------ the members ---

NUM_LAYERS = 48

#: The seven members; tp2 and tp4 leave the experts to Mapping's default split (the TP size).
MEMBERS = {
    "tp1": dict(world_size=1, tp_size=1),
    "tp2": dict(world_size=2, tp_size=2),
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}

_RANKS = [(m, r) for m, layout in MEMBERS.items() for r in range(layout["world_size"])]
#: The last rank of one member per attention layout: one rank, TP2, TP4, attention DP.
LAYOUT_RANKS = [("tp1", 0), ("tp2", 1), ("tp4", 3), ("dep4", 3)]


def _mapping(member: str, rank: int) -> Mapping:
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _world(member: str) -> int:
    return MEMBERS[member]["world_size"]


_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# --------------------------------------------------------------------- routing ---

_DUMMY_MODEL = "/tmp/dummy_model"
_ARCH = "Qwen4ExpForConditionalGeneration"
_TARGET = (
    "models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.modeling."
    "ModelingV2Qwen38FlashNextFp8Sm103Unified"
)
_NVFP4_DEFAULT_TARGET = (
    "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.modeling."
    "ModelingV2Qwen38FlashNextNvfp4Sm103Default"
)

# The quantization the engine resolves for the FP8 (128x128 blocks) and the NVFP4 exports.
_FP8 = QuantAlgo.FP8_BLOCK_SCALES
_NVFP4 = QuantAlgo.MIXED_PRECISION


def _qualified(target) -> str:
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture
def on_sm103(monkeypatch):
    """Route as if on a GB300, with the GDN cached replay at its default (on)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    for env in (
        "TRTLLM_USE_GDN_FLASHINFER_REPLAY",
        M.MOE_ALLTOALL_ENV,
        M.ROWTILE_SCRATCH_GROUPS_ENV,
        W.MOE_TPB_ENV,
    ):
        monkeypatch.delenv(env, raising=False)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


# The checkpoint's `mtp` block.
_MTP_BLOCK = dict(
    hybrid=True,
    layer_types=["full_attention"],
    mtp_use_hidden_state_from_layer=None,
    num_hidden_layers=1,
    rope_theta=10000000,
)


def _qwen4_exp_config(text_overrides=None, composite_overrides=None, **vision_overrides):
    """Qwen3.8-Flash-Next's normalized composite config (the FP8 and NVFP4 exports alike)."""
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
        mtp=dict(_MTP_BLOCK),
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
        **(composite_overrides or {}),
    )


def _routed_config(
    member=None, spec_config=None, pretrained_config=None, quant_algo=_FP8, **mapping_kwargs
) -> ModelConfig:
    if member is not None:
        mapping_kwargs = dict(MEMBERS[member], **mapping_kwargs)
    return ModelConfig(
        pretrained_config=pretrained_config or _qwen4_exp_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=None),
    )


def _admitted(config, **llm_args_kwargs) -> None:
    for mode in ("auto", "require"):
        target = modeling_v2_resolve(config, _llm_args(mode, **llm_args_kwargs))
        assert target is not None and _qualified(target) == _TARGET, mode


def _mtp_with(**attributes):
    """Sets post-load MTP fields directly: the constructor would remap or recompute them."""
    config = MTPDecodingConfig(max_draft_len=3)
    for name, value in attributes.items():
        setattr(config, name, value)
    return config


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, parallel",
    [
        ("tp1", dict(tensor_parallel_size=1, moe_expert_parallel_size=1)),
        ("tp2", dict(tensor_parallel_size=2, moe_expert_parallel_size=1)),
        ("tep2", dict(tensor_parallel_size=2, moe_expert_parallel_size=2)),
        (
            "dep2",
            dict(tensor_parallel_size=2, moe_expert_parallel_size=2, enable_attention_dp=True),
        ),
        ("tp4", dict(tensor_parallel_size=4)),
        ("tep4", dict(tensor_parallel_size=4, moe_expert_parallel_size=4)),
        (
            "dep4",
            dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True),
        ),
    ],
)
def test_each_member_is_admitted_as_the_llm_api_spells_it(on_sm103, member, parallel, monkeypatch):
    """Each member's LLM API spelling builds its mapping and is admitted. The node size is pinned:
    inferred from one visible GPU, a multi-rank world would span nodes, which `layout` declines."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    mapping = _llm_args(gpus_per_node=4, **parallel).parallel_config.to_mapping()
    assert routing.layout(mapping, routing._FP8_LAYOUTS) == member
    _admitted(_routed_config(member))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, spec_config, llm_args_kwargs",
    [
        ("tp1", MTPDecodingConfig(max_draft_len=1), {}),
        ("tep2", MTPDecodingConfig(max_draft_len=7), dict(prefill_cuda_graph_backend="breakable")),
        (
            "tep4",
            MTPDecodingConfig(max_draft_len=3),
            dict(kv_cache_config=KvCacheConfig(dtype="fp8")),
        ),
        (
            "dep4",
            MTPDecodingConfig(max_draft_len=3),
            dict(kv_cache_config=KvCacheConfig(dtype="nvfp4")),
        ),
        ("tp4", None, dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="bfloat16"))),
    ],
    ids=[
        "tp1-mtp1",
        "tep2-mtp7-breakable",
        "tep4-mtp3-fp8-kv",
        "dep4-mtp3-nvfp4-kv",
        "tp4-bf16-state",
    ],
)
def test_the_engine_choices_the_target_runs_are_admitted(
    on_sm103, member, spec_config, llm_args_kwargs
):
    _admitted(_routed_config(member, spec_config), **llm_args_kwargs)


_MTP3 = MTPDecodingConfig(max_draft_len=3)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config_kwargs, llm_args_kwargs, match",
    [
        (dict(member="dep2", spec_config=_MTP3), {}, r"attention DP \(dep2\)"),
        (dict(member="dep4"), dict(prefill_cuda_graph_backend="breakable"), "breakable"),
        (dict(member="tep4"), dict(prefill_cuda_graph_backend="piecewise"), "piecewise"),
        (
            dict(member="tep4", spec_config=_MTP3),
            dict(kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32")),
            "recurrent-state cache",
        ),
        (dict(member="tep4"), dict(kv_cache_config=KvCacheConfig(dtype="fp8_ds_mla")), "KV cache"),
        (
            dict(
                member="tep4",
                spec_config=NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            ),
            {},
            "one-model MTP-Eagle",
        ),
        (
            dict(member="tep4", spec_config=MTPDecodingConfig(max_draft_len=8)),
            {},
            "max_draft_len=8",
        ),
        (
            dict(member="tep4", spec_config=_mtp_with(max_total_draft_tokens=5)),
            {},
            "linear chain",
        ),
        (
            dict(member="tep4", spec_config=MTPDecodingConfig(max_draft_len=3, max_concurrency=8)),
            {},
            "draft length",
        ),
        (dict(world_size=2, tp_size=1, pp_size=2), {}, "tp2 or tep2"),
        (dict(MEMBERS["tep4"], gpus_per_node=2), {}, "tp2 or tep2"),
        (dict(MEMBERS["dep4"], enable_lm_head_tp_in_adp=True), {}, "tp2 or tep2"),
        (dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8), {}, "tp2 or tep2"),
        (
            dict(member="tep4", pretrained_config=_qwen4_exp_config(dict(num_hidden_layers=60))),
            {},
            "text shape",
        ),
        (
            dict(
                member="tep4",
                pretrained_config=_qwen4_exp_config(composite_overrides=dict(dtype="float16")),
            ),
            {},
            "torch.float16",
        ),
        (
            dict(member="tep4", pretrained_config=_qwen4_exp_config(dict(norm_topk_prob=False))),
            {},
            "'norm_topk_prob'",
        ),
        (
            dict(member="tep4", pretrained_config=_qwen4_exp_config(dict(rope_theta=1e6))),
            {},
            "'rope_theta'",
        ),
        (
            dict(
                member="tep4",
                pretrained_config=_qwen4_exp_config(
                    dict(mtp=dict(_MTP_BLOCK, layer_types=["linear_attention"]))
                ),
            ),
            {},
            "'mtp.layer_types'",
        ),
        (
            dict(member="tep4"),
            dict(max_seq_len=8192, kv_cache_config=KvCacheConfig(max_attention_window=[4096])),
            "sliding window",
        ),
        (
            dict(member="tep4"),
            dict(moe_config=MoeConfig(load_balancer={"num_slots": 512})),
            r"load_balancer \(EPLB\)",
        ),
        (
            dict(member="tep4"),
            dict(multimodal_config=MultimodalConfig(vision_gpu_normalize=True)),
            "vision_gpu_normalize",
        ),
    ],
    ids=[
        "dep2-mtp",
        "dep4-breakable",
        "piecewise",
        "fp32-state",
        "kv-algorithm",
        "ngram",
        "draft8",
        "max-total-draft",
        "runtime-draft-length",
        "pp2",
        "two-nodes",
        "dep4-lm-head-tp",
        "tep8",
        "text-shape",
        "fp16-model",
        "other-routing-setting",
        "other-rope-theta",
        "other-mtp-layer",
        "sliding-window",
        "eplb",
        "vision-gpu-normalize",
    ],
)
def test_what_the_target_does_not_run_declines(on_sm103, config_kwargs, llm_args_kwargs, match):
    """``auto`` falls back to the built-in model and ``require`` says why, once per reason."""
    config = _routed_config(**config_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_args_kwargs)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", **llm_args_kwargs))


@pytest.mark.cpu_only
def test_another_moe_backend_or_gpu_or_no_gdn_replay_declines(on_sm103, monkeypatch):
    """The engine choices read off the `ModelConfig` or the process rather than `TorchLlmArgs`."""
    config = replace(_routed_config("tep4"), moe_backend_requested="CUTLASS")
    with pytest.raises(ValueError, match="moe_config.backend"):
        modeling_v2_resolve(config, _llm_args("require"))
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    with pytest.raises(ValueError, match="TRTLLM_USE_GDN_REPLAY=0"):
        modeling_v2_resolve(_routed_config("tep4", _MTP3), _llm_args("require"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(_routed_config("tep4"), _llm_args("require"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY")
    monkeypatch.setenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", "1")
    with pytest.raises(ValueError, match="TRTLLM_USE_GDN_FLASHINFER_REPLAY"):
        modeling_v2_resolve(_routed_config("tep4", _MTP3), _llm_args("require"))
    monkeypatch.delenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY")
    # An all-to-all MoE the deployment cannot run: no attention DP, or no MNNVL.
    monkeypatch.setenv(M.MOE_ALLTOALL_ENV, "1")
    with pytest.raises(ValueError, match=M.MOE_ALLTOALL_ENV):
        modeling_v2_resolve(_routed_config("tep4"), _llm_args("require"))
    with mock.patch(
        "tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided.NVLinkOneSided"
        ".is_platform_supported",
        return_value=False,
    ):
        with pytest.raises(ValueError, match=M.MOE_ALLTOALL_ENV):
            modeling_v2_resolve(_routed_config("dep4"), _llm_args("require"))
    # A misspelled switch raises rather than routing as if it were off.
    monkeypatch.setenv(M.MOE_ALLTOALL_ENV, "enabled")
    with pytest.raises(ValueError, match=f"{M.MOE_ALLTOALL_ENV}='enabled'"):
        modeling_v2_resolve(_routed_config("tep4"), _llm_args("auto"))
    monkeypatch.delenv(M.MOE_ALLTOALL_ENV)
    monkeypatch.setenv(W.MOE_TPB_ENV, "1")
    with pytest.raises(ValueError, match=f"{W.MOE_TPB_ENV}='1': expected rr"):
        modeling_v2_resolve(_routed_config("tep4"), _llm_args("auto"))
    monkeypatch.delenv(W.MOE_TPB_ENV)


@pytest.mark.cpu_only
def test_the_quantization_picks_the_checkpoint(on_sm103):
    """MIXED_PRECISION routes to the NVFP4 target (its own checks mocked), FP8_BLOCK_SCALES here."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
        modeling as nvfp4_modeling,
    )

    with mock.patch.object(
        nvfp4_modeling.ModelingV2Qwen38FlashNextNvfp4Sm103Default, "check_deployment"
    ):
        target = modeling_v2_resolve(
            _routed_config("tep4", quant_algo=_NVFP4), _llm_args("require")
        )
    assert _qualified(target) == _NVFP4_DEFAULT_TARGET
    _admitted(_routed_config("tep4"))


# ------------------------------------------------------------------------ the core ---

# Qwen3.8-Flash-Next's text config (the FP8 export's `text_config`).
# fmt: off
_TEXT = dict(
    architectures=["Qwen4ExpForCausalLM"], attention_bias=False, attention_dropout=0.0,
    bos_token_id=248044, dtype="bfloat16", eos_token_id=248044, full_attention_interval=4,
    hc_count=4, hc_lowrank=320, head_dim=256, heads_per_ngram=8, hidden_act="silu",
    hidden_size=2560, indexer_budget=2048, indexer_compress_ratio=4, indexer_head_dim=128,
    indexer_kv_heads=1, indexer_n_heads=4,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(NUM_LAYERS)],
    linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=16,
    linear_num_value_heads=48, linear_value_head_dim=128, make_ngram_vocab_size_divisible_by=128,
    mamba_ssm_dtype="float32", max_position_embeddings=262144, moe_intermediate_size=640,
    mtp=dict(hybrid=True, layer_types=["full_attention"], mtp_use_hidden_state_from_layer=None,
             num_hidden_layers=1, rope_theta=10000000),
    mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False, ngram_size=3,
    ngram_vocab_size_base=20000000, num_attention_heads=24, num_experts=512,
    num_experts_per_tok=10, num_hidden_layers=NUM_LAYERS, num_key_value_heads=2,
    output_gate_type="sigmoid", partial_rotary_factor=0.25, ple_conv_kernel_size=4,
    ple_embed_dim=2560, ple_layer_ids=[2], rms_norm_eps=1e-06,
    rope_parameters=dict(mrope_interleaved=True, mrope_section=[11, 11, 10],
                         partial_rotary_factor=0.25, rope_theta=10000000, rope_type="default"),
    shared_expert_intermediate_size=640, split_ngram_parts=128, tie_word_embeddings=False,
    vocab_size=248320,
)
_VISION = dict(
    deepstack_visual_indexes=[], depth=27, dtype="bfloat16", hidden_act="gelu_pytorch_tanh",
    hidden_size=1152, in_channels=3, intermediate_size=4304, num_heads=16,
    num_position_embeddings=2304, out_hidden_size=2560, patch_size=16, spatial_merge_size=2,
    temporal_patch_size=2,
)
# fmt: on

HIDDEN = 2560
NUM_EXPERTS = 512
TOPK = 10

# Per member: (q heads, KV heads, attn row, rope "v" heads, GDN k / v heads, in_proj split, o K).
_GEOMETRY = {
    "tp1": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
    "tp2": (12, 1, 7424, 16, 8, 24, (5120, 3072, 24, 24), 3072),
    "tep2": (12, 1, 7424, 16, 8, 24, (5120, 3072, 24, 24), 3072),
    "dep2": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
    "tp4": (6, 1, 4352, 10, 4, 12, (2560, 1536, 12, 12), 1536),
    "tep4": (6, 1, 4352, 10, 4, 12, (2560, 1536, 12, 12), 1536),
    "dep4": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
}


def _mtp(max_draft_len: int = 3) -> MTPDecodingConfig:
    return MTPDecodingConfig(max_draft_len=max_draft_len)


def _core(member: str, rank: int, spec_config=None):
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=_mapping(member, rank),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        moe_backend="TRTLLM",
        moe_backend_requested="TRTLLM",
        sparse_attention_config=QSASparseAttentionConfig(),
        spec_config=spec_config,
    )
    composite = SimpleNamespace(
        vision_config=Qwen4ExpVisionConfig(**_VISION), image_token_id=248056, video_token_id=248057
    )
    # The core asserts the device it runs on; the weights it declares stay on meta.
    with (
        mock.patch.object(torch.cuda, "get_device_capability", return_value=M._SM),
        torch.device("meta"),
    ):
        return M.Qwen4ExpUnifiedModelingV2(model_config, composite)


#: Every rank; MTP with 7 drafts at a member's last rank, 1 elsewhere, and none at dep2.
_CORE_CASES = [(m, r, None if m == "dep2" else 7 if r == _world(m) - 1 else 1) for m, r in _RANKS]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank, max_draft_len",
    _CORE_CASES,
    ids=[f"{m}-rank{r}-mtp{d}" for m, r, d in _CORE_CASES],
)
def test_each_rank_builds_its_share_of_the_core(member, rank, max_draft_len) -> None:
    """This rank's heads (rank `r` of two holds KV head `r`, of four `r // 2`) and one
    whole-expert window at the full intermediate, whatever the MoE TP/EP split; with MTP the draft
    layer 48 is cut like a trunk layer and a verify step carries `max_draft_len + 1` rows."""
    hq, hkv, row, tail, hk, hv, split, out_k = _GEOMETRY[member]
    world, adp = _world(member), MEMBERS[member].get("enable_attention_dp", False)
    attn_tp = 1 if adp else world
    core = _core(member, rank)
    assert (core.heads_q, core.heads_kv) == (hq, hkv)
    assert core.kv_head_start == (rank * 2 // attn_tp if attn_tp > 1 else 0)
    assert (core.attn_in_width, core.attn_in_tail_heads) == (row, tail)
    assert core.index_col == core.qkv_width + core.q_width == 2 * (hq + hkv) * 256
    assert row == 2 * hq * 256 + 2 * hkv * 256 + 640 + 128
    assert tuple(core.w["l3_attn_in_w"].shape) == (row, HIDDEN)
    assert tuple(core.w["l3_o_w"].shape) == (HIDDEN, out_k)
    assert (core.num_k_heads, core.num_v_heads) == (hk, hv)
    assert tuple(core.in_proj_split) == split
    # Padded with zero rows to a multiple of 16, the decode GEMM's whole-tile width.
    assert core.in_proj_rows == -(-sum(split) // 16) * 16
    assert tuple(core.w["l0_in_proj_w"].shape) == (core.in_proj_rows, HIDDEN)
    assert tuple(core.w["l0_out_w"].shape) == (HIDDEN, out_k)
    assert (core.attn_tp_size, core.dp_size) == (attn_tp, world if adp else 1)

    count = NUM_EXPERTS // world
    assert (core.local_experts, core.expert_offset, core.moe_inter) == (count, count * rank, 640)
    window = W.expert_window(core.w, core.dims, 0)
    assert (window.offset, window.count) == (count * rank, count)
    assert tuple(window.fc1_w.shape) == (count, 2 * 640, HIDDEN)
    assert window.fc1_w.dtype == window.fc2_w.dtype == torch.float8_e4m3fn
    assert tuple(window.fc1_s.shape) == (count, 2 * 640 // 128, HIDDEN // 128)
    assert window.fc1_s.dtype == torch.float32
    assert tuple(window.fc2_w.shape) == (count, HIDDEN, 640)
    assert tuple(window.fc2_s.shape) == (count, HIDDEN // 128, 640 // 128)
    assert tuple(core.w["l0_router"].shape) == (NUM_EXPERTS, HIDDEN)
    assert not core.mtp_enabled and core.spec_rollback is None
    assert not any("_mtp_" in k for k in core.w)
    if max_draft_len is None:
        return

    mtp = _core(member, rank, _mtp(max_draft_len))
    assert mtp.mtp_enabled and mtp.mtp_layer_idx == NUM_LAYERS and mtp.spec_rollback is not None
    assert max_draft_len + 1 <= M._MAX_VERIFY_WIDTH
    assert mtp.mtp_theta == mtp.theta == 1e7
    draft = W.expert_window(mtp.w, mtp.dims, NUM_LAYERS)
    assert (draft.offset, draft.count) == (window.offset, window.count)
    assert draft.fc1_w.shape == window.fc1_w.shape and draft.fc2_s.shape == window.fc2_s.shape
    assert draft.fc1_w.dtype == draft.fc2_w.dtype == torch.float8_e4m3fn
    p = f"l{NUM_LAYERS}_mtp_"
    assert tuple(mtp.w[f"{p}attn_in_w"].shape) == (row, HIDDEN)
    assert tuple(mtp.w[f"{p}o_w"].shape) == (HIDDEN, out_k)
    assert tuple(mtp.w[f"{p}router"].shape) == (NUM_EXPERTS, HIDDEN)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, dp_size, step, grouped_min, calls",
    [
        (1, 1, 8192, 0, [("runner", 1)]),
        (20000, 1, 8192, 0, [("runner", 8192), ("runner", 8192), ("runner", 3616)]),
        (4095, 1, 8192, 4096, [("runner", 4095)]),
        (16384, 1, 16384, 4096, [("grouped", 16384)]),
        # An attention-DP window: up to four 8192-row chunks a runner call, never grouped.
        (40000, 4, 8192, 4096, [("runner", 32768), ("runner", 7232)]),
    ],
    ids=["one-row", "tp-chunks", "tp-below-grouped", "tp-context-grouped", "adp-window"],
)
def test_a_moe_layer_enters_the_fp8_runner_over_its_window(
    rows, dp_size, step, grouped_min, calls, monkeypatch
) -> None:
    """`_routed_experts` enters the FP8 runner pre-routed over this rank's window in row-ordered
    chunks of `core.moe_max_rows` rows, or `moe_prefill_grouped` for a large TP context chunk."""
    core = SimpleNamespace(
        topk=TOPK, num_experts=NUM_EXPERTS, moe_inter=640, dp_size=dp_size, moe_max_rows=step
    )
    experts = W.ExpertWindow("fc1_w", "fc1_s", "fc2_w", "fc2_s", 384, 128)
    x = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, 4).contiguous()
    seen = []

    def router(chunk, mat_b):
        assert mat_b == "router"
        return chunk[:, :1]

    def routing(logits, topk, output_dtype):
        assert topk == TOPK
        return logits.int().expand(-1, topk).contiguous(), logits.expand(-1, topk).contiguous()

    def quantize(chunk, use_ue8m0):
        assert use_ue8m0 is False
        return chunk + 0.5, chunk[:, :2].t().contiguous()

    def runner(**kw):
        seen.append(("runner", kw["hidden_states"], kw["hidden_states_scale"], kw))
        assert kw["routing_logits"] is None and kw["routing_bias"] is None
        assert (kw["gemm1_weights"], kw["gemm1_weights_scale"]) == ("fc1_w", "fc1_s")
        assert (kw["gemm2_weights"], kw["gemm2_weights_scale"]) == ("fc2_w", "fc2_s")
        assert (kw["num_experts"], kw["top_k"], kw["intermediate_size"]) == (512, 10, 640)
        assert (kw["local_expert_offset"], kw["local_num_experts"]) == (384, 128)
        assert kw["routing_method_type"] == 1 and kw["act_type"] == 0
        return kw["hidden_states"] * 2

    def grouped(xq, xsf, ids, w, *weights):
        assert weights == ("fc1_w", "fc1_s", "fc2_w", "fc2_s", 384)
        seen.append(("grouped", xq, xsf, dict(topk_ids=ids, topk_weights=w)))
        return xq * 2

    monkeypatch.setattr(M, "cublas_mm", router)
    monkeypatch.setattr(M, "renorm_moe_routing_op", routing)
    monkeypatch.setattr(M, "fp8_quantize_1x128", quantize)
    monkeypatch.setattr(M, "fp8_block_scale_moe_runner", runner)
    monkeypatch.setattr(M, "moe_prefill_grouped", grouped)
    monkeypatch.setattr(M, "prefill_grouped_ready", lambda local: local == 128)
    monkeypatch.setattr(M, "_MOE_PREFILL_GROUPED_MIN_ROWS", grouped_min)
    out = M._routed_experts(core, x, "router", experts)
    assert [(kind, xq.shape[0]) for kind, xq, *_ in seen] == calls
    torch.testing.assert_close(out, (x + 0.5) * 2, rtol=0, atol=0)
    lo = 0
    for _, xq, xsf, kw in seen:
        n = xq.shape[0]
        torch.testing.assert_close(xsf, x[lo : lo + n, :2].t(), rtol=0, atol=0)
        torch.testing.assert_close(kw["topk_ids"][:, 0], x[lo : lo + n, 0].int(), rtol=0, atol=0)
        torch.testing.assert_close(kw["topk_weights"][:, 0], x[lo : lo + n, 0], rtol=0, atol=0)
        lo += n


def _pad(x: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat([x, x.new_zeros(rows - x.shape[0], x.shape[1])], 0)


def _expert_out(x, ids, wts, start, count):
    """A stand-in MoE over experts `[start, start + count)`: whole-expert windows sum to all 512."""
    keep = ((ids >= start) & (ids < start + count)).to(x.dtype)
    return (wts * keep * (ids + 1).to(x.dtype)).sum(-1, keepdim=True) * x


#: The core's side stream on the host, where there is nothing to order.
_HOST_STREAM = SimpleNamespace(wait_stream=lambda other: None)


class _HostMoe:
    """A MoE layer's ops in float64 over a fixed router: top-10 softmax routing, `_expert_out`
    over the window, chunks of `chunk_rows`, two chunks a runner call on an attention-DP window."""

    def __init__(self, monkeypatch, chunk_rows: int):
        gen = torch.Generator().manual_seed(0)
        self.router = torch.randn(HIDDEN, NUM_EXPERTS, generator=gen, dtype=torch.float64)
        self.chunk_rows = chunk_rows
        self.calls, self.windows = [], []
        monkeypatch.setattr(M, "_MOE_MAX_ROWS", chunk_rows)
        monkeypatch.setattr(M, "_MOE_MAX_ROWS_TP", chunk_rows)
        monkeypatch.setattr(M, "_MOE_PREFILL_GROUPED_MIN_ROWS", 0)  # the runner path
        monkeypatch.setattr(M, "_ADP_WINDOW_MAX_ROWS", 2 * chunk_rows)
        monkeypatch.setattr(M, "cublas_mm", lambda a, b, *args, **kwargs: a @ b)
        monkeypatch.setattr(M, "renorm_moe_routing_op", self.routing)
        # Scales in the `[hidden / 128, rows]` layout, which grouped chunks concatenate.
        monkeypatch.setattr(
            M, "fp8_quantize_1x128", lambda t, use_ue8m0: (t, t.new_zeros(HIDDEN // 128, len(t)))
        )
        monkeypatch.setattr(M, "fp8_block_scale_moe_runner", self.runner)
        monkeypatch.setattr(M, "silu_and_mul_rows", lambda gug, inter: gug[:, :inter])
        # The shared expert and its gate are replicated and checked elsewhere.
        monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda routed, gate, shared: routed)
        # `_mtp_moe_block` forks the shared expert onto a side stream; the host runs it in order.
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: _HOST_STREAM)
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())

    def route(self, x):
        top_logits, top_ids = torch.topk(x @ self.router, TOPK, dim=-1)
        return top_ids, torch.softmax(top_logits, dim=-1)

    def routing(self, logits, topk, output_dtype):
        assert topk == TOPK and logits.shape[1] == NUM_EXPERTS
        top_logits, top_ids = torch.topk(logits, topk, dim=-1)
        return top_ids, torch.softmax(top_logits, dim=-1)

    def runner(self, **kw):
        assert kw["num_experts"] == NUM_EXPERTS and kw["intermediate_size"] == 640
        x = kw["hidden_states"]
        assert 0 < x.shape[0] <= 2 * self.chunk_rows
        self.calls.append(x.shape[0])
        start, count = kw["local_expert_offset"], kw["local_num_experts"]
        self.windows.append((start, count))
        return _expert_out(x, kw["topk_ids"], kw["topk_weights"], start, count)

    def views(self, core):
        core._side_stream = _HOST_STREAM
        return SimpleNamespace(
            router=self.router,
            experts=W.expert_window(core.w, core.dims, NUM_LAYERS),
            sh_gug=torch.zeros(HIDDEN, 2 * 640 + 8, dtype=torch.float64),
            sh_dn=torch.zeros(640, HIDDEN, dtype=torch.float64),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "counts, draft",
    [
        ([3, 7], False),
        ([0, 0, 5, 3], False),
        ([4, 4, 4, 4], False),
        ([3, 7, 1, 4], True),
        ([1, 2, 1, 3], True),
        ([4, 4, 4, 4], True),
    ],
    ids=[
        "dep2-trunk",
        "dep4-trunk-two-ranks-without-rows",
        "dep4-trunk-uniform",
        "dep4-draft-extend",
        "dep4-draft-decode-with-a-dummy-rank",
        "dep4-draft-captured-batch",
    ],
)
def test_the_attention_dp_round_trip_returns_each_ranks_rows(counts, draft, monkeypatch) -> None:
    """Each rank gathers the group's rows, runs its window over the real rows only (a padding
    row's zero logits would route to the lowest experts) and reduce-scatters, with the same
    collectives on every rank (ragged when counts differ), getting back its own rows' MoE."""
    world = len(counts)
    moe = _HostMoe(monkeypatch, chunk_rows=5)
    gen = torch.Generator().manual_seed(1)
    xs = [torch.randn(n, HIDDEN, generator=gen, dtype=torch.float64) for n in counts]
    rows, n = max(counts), sum(counts)
    split = min(counts) != rows
    if split:
        gathered = torch.cat(xs, 0)
        starts = [sum(counts[:r]) for r in range(world)]
        real = torch.arange(n)
    else:
        gathered = torch.cat([_pad(x, rows) for x in xs], 0)
        starts = [r * rows for r in range(world)]
        real = torch.cat([torch.arange(r * rows, r * rows + c) for r, c in enumerate(counts)])
    real_rows = gathered[real]
    full = _expert_out(gathered, *moe.route(gathered), 0, NUM_EXPERTS)
    state = {}

    def fake_allgather(t, sizes, group, process_group):
        assert sizes == (list(counts) if split else None)
        assert group == list(range(world)) and process_group is None
        state["collectives"].append(("allgather", tuple(t.shape)))
        x = xs[state["rank"]]
        torch.testing.assert_close(t, x if split else _pad(x, rows), rtol=0, atol=0)
        return gathered

    def fake_reducescatter(t, sizes, group, process_group):
        assert sizes == (list(counts) if split else None)
        assert group == list(range(world)) and process_group is None
        state["collectives"].append(("reducescatter", tuple(t.shape)))
        state["partial"] = t
        r = state["rank"]
        return full[starts[r] : starts[r] + (counts[r] if split else rows)]

    def no_tp_sum(t):
        raise AssertionError("the MoE under attention DP went through the TP sum")

    monkeypatch.setattr(M, "allgather", fake_allgather)
    monkeypatch.setattr(M, "reducescatter", fake_reducescatter)
    monkeypatch.setattr(M, "mpi_disabled", lambda: False)
    per_rank = []
    for rank in range(world):
        core = _core(f"dep{world}", rank, _mtp() if draft else None)
        core.expert_sum, core.tp_sum = core._expert_sum_op(), no_tp_sum
        state.update(rank=rank, collectives=[])
        moe.calls, moe.windows = [], []
        if draft:
            pad_rows = M._draft_pad_rows(core, counts, counts[rank])
            out = M._mtp_moe_block(core, moe.views(core), xs[rank], pad_rows)
        else:
            md = SimpleNamespace(all_rank_num_tokens=counts)
            pad_rows = core._dp_pad_rows(md, counts[rank])
            trunk = W.expert_window(core.w, core.dims, 0)
            window = lambda t: M._routed_experts(core, t, moe.router, trunk)  # noqa: E731
            out = core.expert_sum(window, xs[rank], pad_rows)
        count = NUM_EXPERTS // world
        assert moe.calls == [min(10, n - lo) for lo in range(0, n, 10)]
        assert set(moe.windows) == {(count * rank, count)}
        torch.testing.assert_close(
            state["partial"][real],
            _expert_out(real_rows, *moe.route(real_rows), count * rank, count),
        )
        assert tuple(out.shape) == (counts[rank], HIDDEN)
        torch.testing.assert_close(out, _expert_out(xs[rank], *moe.route(xs[rank]), 0, NUM_EXPERTS))
        # Each rank hands the gather its own rows: the ragged form's shapes differ by rank.
        assert state["collectives"][0] == ("allgather", (counts[rank] if split else rows, HIDDEN))
        per_rank.append(state["collectives"][1:])
    assert all(c == per_rank[0] for c in per_rank), per_rank
    assert per_rank[0] == [("reducescatter", (n if split else world * rows, HIDDEN))]


class _HostA2a:
    """The NVLink MoE all-to-all at one rank: `dispatch` hands it each source's rows routed into
    its window (padded with -1 routes); `combine` checks its partials and returns its rows' MoE."""

    def __init__(self, xs, routes, rank):
        self.xs, self.routes, self.rank = xs, routes, rank
        self.world, self.cap = len(xs), max(len(x) for x in xs)
        self.max_num_tokens_per_rank = 1 << 20
        self.calls = []

    def dispatch(self, x, sf, ids, w, counts):
        self.calls.append("dispatch")
        assert sf is None and counts == [len(t) for t in self.xs]
        own_ids, own_w = self.routes[self.rank]
        torch.testing.assert_close(x, self.xs[self.rank], rtol=0, atol=0)
        # The router runs over chunks of the rows, so the weights match to rounding.
        assert torch.equal(ids, own_ids), "not routed on its own rows"
        torch.testing.assert_close(w, own_w.float())  # fp32 on the wire, as the stock MoE's
        lo, count = self.rank * (NUM_EXPERTS // self.world), NUM_EXPERTS // self.world
        rows, rids, rw = [], [], []
        for src, (sx, (sids, sw)) in enumerate(zip(self.xs, self.routes)):
            hit = ((sids >= lo) & (sids < lo + count)).any(1)
            pad = self.cap - int(hit.sum())
            rows.append(torch.cat([sx[hit], sx.new_zeros(pad, HIDDEN)]))
            rids.append(torch.cat([sids[hit], sids.new_full((pad, TOPK), -1)]))
            rw.append(torch.cat([sw[hit], sw.new_zeros(pad, TOPK)]))
        self.got = torch.cat(rows), torch.cat(rids), torch.cat(rw), lo, count
        return self.got[0], None, self.got[1], self.got[2]

    def combine(self, partials):
        self.calls.append("combine")
        rows, ids, w, lo, count = self.got
        torch.testing.assert_close(partials, _expert_out(rows, ids, w, lo, count))
        own = self.xs[self.rank]
        return _expert_out(own, *self.routes[self.rank], 0, NUM_EXPERTS)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "counts, draft",
    [([3, 7], False), ([0, 5, 2, 4], False), ([4, 4, 4, 4], True)],
    ids=["dep2-trunk-ragged", "dep4-trunk-a-rank-without-rows", "dep4-draft-captured-batch"],
)
def test_the_attention_dp_alltoall_round_trip(counts, draft, monkeypatch) -> None:
    """`TRTLLM_MODELING_V2_MOE_ALLTOALL=1`: each rank routes its own rows through the all-to-all
    and returns what the combine sums; no all-gather, reduce-scatter or TP sum."""
    world = len(counts)
    moe = _HostMoe(monkeypatch, chunk_rows=5)
    gen = torch.Generator().manual_seed(2)
    xs = [torch.randn(n, HIDDEN, generator=gen, dtype=torch.float64) for n in counts]
    routes = [moe.route(x) for x in xs]

    def forbidden(*args, **kwargs):
        raise AssertionError("the all-to-all round trip took the default collectives")

    monkeypatch.setenv(M.MOE_ALLTOALL_ENV, "1")
    monkeypatch.setattr(M, "mpi_disabled", lambda: False)
    for name in ("allgather", "reducescatter"):
        monkeypatch.setattr(M, name, forbidden)
    for rank in range(world):
        core = _core(f"dep{world}", rank, _mtp() if draft else None)
        a2a = _HostA2a(xs, routes, rank)
        monkeypatch.setattr(M, "_build_moe_a2a", lambda mapping, max_tokens: a2a)
        core.expert_sum, core.tp_sum = core._expert_sum_op(), forbidden
        assert core.moe_a2a is a2a
        if draft:
            pad_rows = M._draft_pad_rows(core, counts, counts[rank])
            out = M._mtp_moe_block(core, moe.views(core), xs[rank], pad_rows)
        else:
            md = SimpleNamespace(all_rank_num_tokens=counts)
            run = M._WindowRun(core, moe.router, W.expert_window(core.w, core.dims, 0))
            out = core.expert_sum(run, xs[rank], core._dp_pad_rows(md, counts[rank]))
        assert a2a.calls == ["dispatch", "combine"]
        torch.testing.assert_close(out, _expert_out(xs[rank], *routes[rank], 0, NUM_EXPERTS))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "accepted, rows_per_request, index_share, dp_size, want",
    [
        ([1, 3, 2, 1], 3, False, 1, [0, 5, 7, 9]),
        ([1, 3, 2, 1] * 4, 3, False, 1, [r * 3 + a - 1 for r, a in enumerate([1, 3, 2, 1] * 4)]),
        ([1, 3, 2, 1], 1, False, 1, None),
        ([1, 3, 2, 1], 3, True, 1, None),  # IndexShare's capture keeps every row
        ([1, 3, 2, 1], 3, False, 2, None),  # attention DP pads by the worker's counts
        (None, 3, False, 1, None),
    ],
    ids=["draft-extend", "ht-batch", "not-extend", "index-share", "attention-dp", "no-counts"],
)
def test_a_draft_extend_step_continues_from_each_requests_last_accepted_row(
    accepted, rows_per_request, index_share, dp_size, want
) -> None:
    """Request r's draft-extend outputs are read from row `num_accepted[r] - 1` of its
    `rows_per_request` (the golden token counts as accepted); None means every row."""
    counts = None if accepted is None else torch.tensor(accepted, dtype=torch.int32)
    md = SimpleNamespace(qsa_mtp_num_accepted=counts)
    requests = 4 if accepted is None else len(accepted)
    got = M._draft_continue_rows(
        SimpleNamespace(dp_size=dp_size), md, requests, rows_per_request, index_share
    )
    assert (None if got is None else got.tolist()) == want


# ------------------------------------------- sequence-parallel prefill at attention TP ---


def _sp_target(rank: int, world: int, comm: bool = False, ce=None):
    target = object.__new__(M.PrefillTarget)
    target._sequence_parallel = (rank, world, list(range(world)), None)
    target._ce_exchange = ce
    target._comm_stream = target._comm_fork = target._comm_join = target._sp_rows = None
    if comm:
        target._comm_stream = torch.cuda.Stream(priority=-1)
        target._comm_fork, target._comm_join = torch.cuda.Event(), torch.cuda.Event()
    return target


def _sp_share(monkeypatch, num_tokens, world, rank, want, configured=True):
    """From `_SP_MIN_ROWS` rows that divide evenly a rank keeps its consecutive `T / world` rows,
    gathered and scattered uniformly over the attention-TP group; otherwise there is no split."""
    target = _sp_target(rank, world)
    if not configured:
        target._sequence_parallel = None
    calls = []
    monkeypatch.setattr(
        M, "allgather", lambda t, sizes, group, pg: calls.append(("ag", sizes, group)) or t
    )
    monkeypatch.setattr(
        M, "reducescatter", lambda t, sizes, group, pg: calls.append(("rs", sizes, group)) or t
    )
    sp = target._sp_split(num_tokens)
    if want is None:
        assert sp is None
        return
    lo, hi, gather, scatter = sp
    assert (lo, hi) == want
    gather(torch.zeros(1))
    scatter(torch.zeros(1))
    assert calls == [("ag", None, list(range(world))), ("rs", None, list(range(world)))]


def _sp_moe_own_rows(monkeypatch, rows, step, padded):
    """`_routed_experts_sp` routes (padding rows to `_NO_EXPERT`) and quantizes only this rank's
    rows; the gathered codes, scales, ids and weights reach the runner exactly, in row order."""
    hidden, topk = 384, 10
    core = SimpleNamespace(topk=topk, num_experts=512, moe_inter=640, moe_max_rows=step)
    experts = W.ExpertWindow("fc1_w", "fc1_s", "fc2_w", "fc2_s", 0, 256)
    gen = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, hidden, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, 512, (rows, topk), generator=gen, dtype=torch.int32)
    weights = torch.rand(rows, topk, generator=gen).to(torch.bfloat16)
    codes = (torch.randn(rows, hidden, generator=gen) * 8).to(torch.float8_e4m3fn)
    scales = torch.rand(hidden // 128, rows, generator=gen)
    monkeypatch.setattr(M, "cublas_mm", lambda a, b, *args: ("logits", a.shape[0]))
    monkeypatch.setattr(
        M, "renorm_moe_routing_op", lambda logits, topk, output_dtype: (ids.clone(), weights)
    )
    monkeypatch.setattr(M, "fp8_quantize_1x128", lambda t, use_ue8m0: (codes, scales))
    calls = []
    monkeypatch.setattr(
        M,
        "fp8_block_scale_moe_runner",
        lambda **kw: calls.append(kw) or torch.zeros(kw["hidden_states"].shape[0], 4),
    )
    mask = (torch.arange(rows) >= rows - padded)[:, None] if padded else None
    want_ids = ids.clone()
    want_ids[rows - padded :] = M._NO_EXPERT
    gathers = []
    out = M._routed_experts_sp(
        core, x, "router", experts, lambda t: gathers.append(t.dtype) or t.clone(), mask
    )
    assert gathers == [torch.bfloat16, torch.bfloat16]
    assert tuple(out.shape) == (rows, 4)
    assert [c["hidden_states"].shape[0] for c in calls] == [
        min(step, rows - lo) for lo in range(0, rows, step)
    ]
    got_codes = torch.cat([c["hidden_states"] for c in calls])
    assert torch.equal(got_codes.view(torch.uint8), codes.view(torch.uint8))
    assert torch.equal(torch.cat([c["hidden_states_scale"] for c in calls], 1), scales)
    assert all(c["hidden_states_scale"].is_contiguous() for c in calls)
    assert torch.equal(torch.cat([c["topk_ids"] for c in calls]), want_ids)
    assert torch.equal(torch.cat([c["topk_weights"] for c in calls]), weights)
    assert all(c["topk_ids"].dtype == torch.int32 for c in calls)


def _sp_ce_scatter(monkeypatch, rank):
    """A copy-engine two-rank scatter sends the peer's rows and adds its partial (the same bits in
    either order) without NCCL; `CopyEngineExchange.create` builds only a one-node pair."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm.ce_exchange__sp_pair import (  # noqa: E501
        CopyEngineExchange,
    )

    num_tokens, hidden = 4096, 64
    rows = num_tokens // 2
    gen = torch.Generator().manual_seed(3)
    partial = [torch.randn(num_tokens, hidden, generator=gen).to(torch.bfloat16) for _ in range(2)]
    sent = []

    class Exchange:
        def exchange(self, x):
            sent.append(x)
            return partial[1 - rank][rank * rows : (rank + 1) * rows].clone()

    monkeypatch.setattr(M, "reducescatter", lambda *a: pytest.fail("NCCL reduce-scatter called"))
    lo, hi, _, scatter = _sp_target(rank, 2, ce=Exchange())._sp_split(num_tokens)
    got = scatter(partial[rank])
    assert (lo, hi) == (rank * rows, (rank + 1) * rows)
    assert torch.equal(sent[0], partial[rank][(1 - rank) * rows : (2 - rank) * rows])
    assert torch.equal(got, partial[0][lo:hi] + partial[1][lo:hi])
    assert torch.equal(got, partial[1][lo:hi] + partial[0][lo:hi])
    four = SimpleNamespace(tp_size=4, gpus_per_node=4, tp_rank=0)
    across = SimpleNamespace(tp_size=2, gpus_per_node=1, tp_rank=0)
    assert CopyEngineExchange.create(four, 1 << 20) is None
    assert CopyEngineExchange.create(across, 1 << 20) is None


def _sp_project(monkeypatch, rank, overlap):
    """`_project_overlapped` projects this rank's rows while the comm stream gathers the others',
    each into its own rows of one output; without the comm stream the block gathers first."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11 + rank)
    rows, hidden, width = 512, 256, 384
    shares = [
        torch.randn(rows, hidden, generator=gen, device=dev).to(torch.bfloat16) for _ in range(2)
    ]
    w = (torch.randn(width, hidden, generator=gen, device=dev) * 0.1).to(torch.bfloat16).t()
    target = _sp_target(rank, 2, comm=overlap)
    gathered = []

    def gather(t):
        assert t.data_ptr() == shares[rank].data_ptr()
        gathered.append(torch.cuda.current_stream())
        return torch.cat(shares)

    target._sp_rows = (rank * rows, (rank + 1) * rows, gather)
    assert target._overlap_projection() is overlap
    if overlap:
        out = target._project_overlapped(shares[rank], w)
        torch.cuda.synchronize()
        assert gathered == [target._comm_stream], "the gather runs on the comm stream"
        assert tuple(out.shape) == (2 * rows, width)
        want = torch.cat(shares).float() @ w.float()
        torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)
    else:
        assert torch.equal(target._block_rows(shares[rank]), torch.cat(shares))
    target._sp_rows = None
    assert not target._overlap_projection()
    assert target._block_rows(shares[rank]) is shares[rank]


def _sp_ce_project_reduce(monkeypatch, rank):
    """With the copy-engine pair exchange, `_project_overlapped` and `_reduce_overlapped` exchange
    on the comm stream and match NCCL bitwise; the reduction sums both ranks' products."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(23)
    rows, hidden, width = 384, 256, 320
    shares = [
        torch.randn(rows, hidden, generator=gen, device=dev).to(torch.bfloat16) for _ in range(2)
    ]
    a = [
        torch.randn(2 * rows, hidden, generator=gen, device=dev).to(torch.bfloat16)
        for _ in range(2)
    ]
    w = (torch.randn(width, hidden, generator=gen, device=dev) * 0.1).to(torch.bfloat16).t()
    sent = [torch.mm(a[r][(1 - r) * rows : (2 - r) * rows], w) for r in range(2)]
    calls = []

    class Exchange:
        def exchange(self, x):
            assert torch.cuda.current_stream() == with_ce._comm_stream
            if torch.equal(x, shares[rank]):
                calls.append("rows")
                return shares[1 - rank].clone()
            assert torch.equal(x, sent[rank])
            calls.append("partials")
            return sent[1 - rank].clone()

    def gather(t):
        return torch.cat(shares) if torch.equal(t, shares[rank]) else torch.cat(sent)

    with_ce, nccl = _sp_target(rank, 2, comm=True, ce=Exchange()), _sp_target(rank, 2, comm=True)
    for t in (with_ce, nccl):
        t._sp_rows = (rank * rows, (rank + 1) * rows, gather)
    assert with_ce._overlap_reduce()
    got = with_ce._project_overlapped(shares[rank], w)
    want = nccl._project_overlapped(shares[rank], w)
    red = with_ce._reduce_overlapped(a[rank], w)
    red_want = nccl._reduce_overlapped(a[rank], w)
    torch.cuda.synchronize()
    assert calls == ["rows", "partials"]
    assert torch.equal(got, want) and torch.equal(red, red_want)
    full = sum(torch.mm(a[r], w).float() for r in range(2))
    torch.testing.assert_close(
        red.float(), full[rank * rows : (rank + 1) * rows], rtol=2e-2, atol=2e-2
    )
    for world, comm in ((4, True), (2, False)):  # four ranks, or no comm stream: one GEMM
        target = _sp_target(rank, world, comm=comm)
        target._sp_rows = with_ce._sp_rows
        assert not target._overlap_reduce()


_cpu = pytest.mark.cpu_only
_SP_CASES = [
    *(
        pytest.param(_sp_share, dict(num_tokens=t, world=w, rank=r, want=want, configured=c),
                     marks=_cpu, id=f"share-{t}-tp{w}-rank{r}{'' if c else '-off'}")
        for t, w, r, want, c in [
            (16384, 2, 1, (8192, 16384), True),
            (4096, 2, 0, (0, 2048), True),
            (4095, 2, 0, None, True),  # an odd eager step keeps every row
            (2048, 2, 0, None, True),  # below `_SP_MIN_ROWS`
            (8192, 4, 3, (6144, 8192), True),
            (16384, 2, 0, None, False),  # one rank, attention DP, or the switch off
        ]
    ),
    *(
        pytest.param(_sp_moe_own_rows, dict(rows=rows, step=step, padded=padded), marks=_cpu,
                     id=f"moe-own-rows-{rows}-step{step}-pad{padded}")
        for rows, step, padded in [(96, 8192, 0), (96, 40, 5), (7, 8192, 3)]
    ),
    *(
        pytest.param(_sp_ce_scatter, dict(rank=r), marks=_cpu, id=f"ce-scatter-rank{r}")
        for r in (0, 1)
    ),
    *(
        pytest.param(_sp_project, dict(rank=r, overlap=o), marks=_needs_gpu,
                     id=f"project-rank{r}-{'overlap' if o else 'gather'}")
        for r, o in [(0, True), (1, True), (1, False)]
    ),
    *(
        pytest.param(_sp_ce_project_reduce, dict(rank=r), marks=_needs_gpu,
                     id=f"ce-project-reduce-rank{r}")
        for r in (0, 1)
    ),
]  # fmt: skip


@pytest.mark.parametrize("check, kwargs", _SP_CASES)
def test_a_sequence_parallel_prefill_step(check, kwargs, monkeypatch) -> None:
    """Sequence-parallel prefill at TP2 / TP4: row shares, own-row MoE, overlapped projections."""
    check(monkeypatch, **kwargs)


# ------------------------------------------------- the split context selection ---


def _selection_balance(monkeypatch, chunks, starts, world):
    """`_selection_cuts` ascend from 0 to the step's rows, interior cuts on the index scores'
    64-row tile, each rank's summed row cost (position + 1) within one tile of the even share."""
    cuts = M._selection_cuts(chunks, starts, world)
    rows = sum(chunks)
    assert len(cuts) == world + 1 and cuts[0] == 0 and cuts[-1] == rows
    assert all(a <= b for a, b in zip(cuts, cuts[1:]))
    assert all(c % M._SELECTION_CUT_ALIGN == 0 for c in cuts[1:-1])
    cost = [s0 + j + 1 for c, s0 in zip(chunks, starts) for j in range(c)]
    parts = [sum(cost[a:b]) for a, b in zip(cuts, cuts[1:])]
    even = sum(cost) / world
    assert max(abs(x - even) for x in parts) <= M._SELECTION_CUT_ALIGN * max(cost) + 1, parts


def _selection_lands_rows(monkeypatch, chunks, starts, world, balanced, pieces):
    """`_split_context_selection`: each rank selects only its share of the context rows, gathered
    in `pieces` on the comm stream, and every rank ends bitwise as one rank selecting them all."""
    dev = torch.device("cuda" if pieces else "cpu")
    lo, width = 5, 2051
    rows = sum(chunks)
    cuts = M._selection_cuts(chunks, starts, world) if balanced else None
    bounds = cuts or [min(r * -(-rows // world), rows) for r in range(world + 1)]
    pipeline = None
    if pieces:
        pipeline = (pieces, torch.cuda.Stream(priority=-1), torch.cuda.Event(), torch.cuda.Event())

    def select_rows(a, b, context_rows, out):
        assert context_rows
        out.copy_(
            torch.arange(a, b, dtype=torch.int32, device=dev)[:, None] * 4099
            + torch.arange(width, dtype=torch.int32, device=dev)
        )

    want = torch.full((lo + rows + 3, width), -1, dtype=torch.int32, device=dev)
    select_rows(lo, lo + rows, True, want[lo : lo + rows])
    buffers = {}  # (rank, piece) -> the rank's buffer

    def run(rank, gather_of):
        scored, count = [], [0]

        def gather(mine, sizes, group, pg):
            if pipeline is not None:
                assert torch.cuda.current_stream() == pipeline[1], "a piece gathers on comm"
            count[0] += 1
            return gather_of(rank, count[0] - 1, mine)

        def record(a, b, context_rows, out):
            scored.append((a, b))
            select_rows(a, b, context_rows, out)

        monkeypatch.setattr(M, "allgather", gather)
        sel = torch.full_like(want, -1)
        M._split_context_selection(
            (rank, world, list(range(world)), None), record, sel, lo, lo + rows, pipeline, cuts
        )
        if pieces:
            torch.cuda.synchronize()
        return sel, scored

    for rank in range(world):  # each rank's own buffers (the stand-in returns a gather's shape)
        _, scored = run(
            rank, lambda r, j, mine: torch.cat([buffers.setdefault((r, j), mine.clone())] * world)
        )
        assert scored[0][0] == lo + bounds[rank] and scored[-1][1] == lo + bounds[rank + 1]
        assert all(b == a2 for (_, b), (a2, _) in zip(scored, scored[1:])), scored
    assert all(b.dtype == torch.bfloat16 and b.shape[1] == 2 * width for b in buffers.values())
    if not pieces:
        per = max(b - a for a, b in zip(bounds, bounds[1:]))
        assert all(b.shape[0] == per for b in buffers.values())
    for rank in range(world):
        sel, _ = run(rank, lambda r, j, mine: torch.cat([buffers[(q, j)] for q in range(world)]))
        assert torch.equal(sel, want), rank


_SELECTION_CASES = [
    *(
        pytest.param(_selection_balance, dict(chunks=c, starts=s, world=w), marks=_cpu,
                     id=f"balance-{len(c)}x{c[0]}-tp{w}")
        for c, s, w in [
            ([32766], [88064], 2),
            ([2728] * 12, [30000 + 9000 * i for i in range(12)], 2),
            ([2046] * 16, [88064] * 16, 2),
            ([32766], [88064], 4),
            ([5000, 3, 7000], [100000, 5, 0], 2),
            ([1024], [0], 2),
        ]
    ),
    *(
        pytest.param(_selection_lands_rows, dict(chunks=c, starts=s, world=w, balanced=b,
                     pieces=0), marks=_cpu, id=f"lands-{sum(c)}-tp{w}-{'balanced' if b else 'even'}")
        for c, s, w, b in [
            ([16384], [0], 2, False),
            ([4093], [0], 2, False),
            ([1025], [0], 4, False),
            ([9000, 4000], [60000, 120], 2, True),
            ([9000, 4000], [60000, 120], 4, True),
        ]
    ),
    *(
        pytest.param(_selection_lands_rows, dict(chunks=c, starts=s, world=2, balanced=b,
                     pieces=p), marks=_needs_gpu, id=f"pipelined-{sum(c)}-{p}-pieces")
        for c, s, b, p in [([32766], [88064], True, 4), ([16382], [0], False, 4)]
    ),
]  # fmt: skip


@pytest.mark.parametrize("check, kwargs", _SELECTION_CASES)
def test_a_tp_ranks_context_selection_is_its_balanced_share(check, kwargs, monkeypatch) -> None:
    """An attention-TP rank selects its cost-balanced share of the context rows, then gathers."""
    check(monkeypatch, **kwargs)


# ----------------------------------------------------------------------- the pools ---

ATTN_LAYERS = [i for i in range(NUM_LAYERS) if i % 4 == 3]
GDN_LAYERS = [i for i in range(NUM_LAYERS) if i % 4 != 3]
PLE_LAYER = 1  # `ple_layer_ids: [2]` is one-based
MTP_LAYER = NUM_LAYERS
TOKENS_PER_BLOCK = 32  # `kv_cache_config.tokens_per_block`'s default, which the runtime keeps
MAX_BATCH_SIZE = 4


_ALL = dict(heads_kv=2, kv_head_start=0, num_k_heads=16, k_head_start=0, num_v_heads=48,
            v_head_start=0, conv_dim=10240)  # fmt: skip


def _tp2(r):
    return dict(heads_kv=1, kv_head_start=r, num_k_heads=8, k_head_start=8 * r, num_v_heads=24,
                v_head_start=24 * r, conv_dim=5120)  # fmt: skip


def _tp4(r):
    return dict(heads_kv=1, kv_head_start=r // 2, num_k_heads=4, k_head_start=4 * r,
                num_v_heads=12, v_head_start=12 * r, conv_dim=2560)  # fmt: skip


#: Per member and rank: KV heads and the first's global index, GDN key and value heads and their
#: first, and the conv channels. TP4 repeats each KV head on two neighbouring ranks.
LAYOUT = {
    **{(m, r): _ALL for m in ("tp1", "dep2", "dep4") for r in range(_world(m))},
    **{(m, r): _tp2(r) for m in ("tp2", "tep2") for r in range(2)},
    **{(m, r): _tp4(r) for m in ("tp4", "tep4") for r in range(4)},
}

COMMON = dict(
    tokens_per_block=TOKENS_PER_BLOCK,
    head_dim=256,
    index_kv_heads=1,
    index_head_dim=128,
    conv_kernel=4,
    head_k_dim=128,
    head_v_dim=128,
    stream=4 * 2560,
    ple_state_len=(4 - 1) * 3,
    ngram_context_len=3 - 1,
    dtype=torch.bfloat16,
)

#: K/V storage per KV-cache algorithm: (pool dtype, innermost extent, whether the QSA index and
#: position pages share one side page). A 32-token bf16 index page is 8192 bytes: an FP8 K page
#: of one head, an NVFP4 one of two, never a bf16 K/V page (16384 bytes a head).
KV_STORAGE = {
    None: (torch.bfloat16, 256, False),
    QuantAlgo.FP8: (torch.float8_e4m3fn, 256, True),
    QuantAlgo.NVFP4: (torch.uint8, 128, True),
}
KV_IDS = {None: "kv_bf16", QuantAlgo.FP8: "kv_fp8", QuantAlgo.NVFP4: "kv_nvfp4"}
_ALGOS = list(KV_STORAGE)


def _geometry(member: str, rank: int, kv_cache_quant_algo=None) -> P.PoolGeometry:
    return P.pool_geometry(
        Qwen4ExpTextConfig(**_TEXT),
        _mapping(member, rank),
        tokens_per_block=TOKENS_PER_BLOCK,
        kv_cache_quant_algo=kv_cache_quant_algo,
        dtype=torch.bfloat16,
    )


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


class _MetaManager:
    """The stock manager's views on meta, sized by a geometry, with the QSA planes as own roles or
    slices of one packed side page. Empty `gdn_layers` stands for a separate draft manager."""

    PAGES = 64

    def __init__(self, g: P.PoolGeometry, mapping: Mapping, layers, gdn_layers=GDN_LAYERS):
        self.g, self.mapping, self.max_batch_size = g, mapping, MAX_BATCH_SIZE
        self.attn_layers, self.gdn_layers = list(layers), list(gdn_layers)
        mapped = sorted([*self.attn_layers, *self.gdn_layers])
        self.layer_offsets = {layer: local for local, layer in enumerate(mapped)}
        self.conv_section_dims = list(g.conv_sections)
        self.slots = MAX_BATCH_SIZE + P.reserved_dummy_slots(mapping)
        self.ssm_dtype, self.scale_pages_per_slot = torch.bfloat16, 1
        tpb = g.tokens_per_block
        if g.qsa_side_planes:
            index, position = g.index_page_bytes, g.position_page_bytes
            side = _meta([self.PAGES, len(self.attn_layers) * index + position + 16], torch.uint8)
            self._index = {
                layer: side[:, k * index : (k + 1) * index]
                .view(torch.bfloat16)
                .view(self.PAGES, tpb, g.index_kv_heads, g.index_head_dim)
                for k, layer in enumerate(self.attn_layers)
            }
            start = len(self.attn_layers) * index
            self._position = (
                side[:, start : start + position].view(torch.int32).view(self.PAGES, tpb, 3)
            )
        else:
            self._index = {
                layer: _meta([self.PAGES, tpb, g.index_kv_heads, g.index_head_dim], torch.bfloat16)
                for layer in self.attn_layers
            }
            self._position = _meta([self.PAGES, tpb, 3], torch.int32)

    def get_buffers(self, layer, kv_layout="NHD"):
        assert kv_layout == "HND"
        g = self.g
        return _meta([self.PAGES, 2, g.heads_kv, g.tokens_per_block, g.kv_width], g.kv_dtype)

    def get_block_scale_buffers(self, layer, kv_layout="NHD"):
        assert kv_layout == "HND"
        g = self.g
        page = [g.heads_kv, g.tokens_per_block, g.head_dim // P.NVFP4_BLOCK]
        return _meta([self.PAGES, 2, *page], torch.uint8)

    def get_index_k_buffer(self, layer):
        return self._index[layer]

    def get_qsa_position_buffer(self):
        return self._position

    def get_conv_states(self, layer):
        return _meta([self.slots, self.g.conv_dim, self.g.conv_kernel - 1], torch.bfloat16)

    def get_ssm_states(self, layer):
        g = self.g
        return _meta([self.slots, g.num_v_heads, g.head_v_dim, g.head_k_dim], self.ssm_dtype)

    def ple_layer_cache(self, layer):
        if not self.gdn_layers:
            return None
        g = self.g
        return (
            _meta([self.slots, g.stream, g.ple_state_len], torch.bfloat16),
            _meta([self.slots, g.ngram_context_len], torch.int64),
        )


#: Every rank under bf16 K/V; each quantized K/V storage at one member per attention layout (the
#: storage sets the pool's dtype, width and side-page packing alone, the rank its heads alone).
_POOL_CASES = [(m, r, None) for m, r in _RANKS] + [
    (m, r, a) for m, r in LAYOUT_RANKS for a in (QuantAlgo.FP8, QuantAlgo.NVFP4)
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank, algo", _POOL_CASES, ids=[f"{m}-rank{r}-{KV_IDS[a]}" for m, r, a in _POOL_CASES]
)
def test_each_rank_binds_the_pools_of_its_layout(member, rank, algo):
    """`pool_geometry` is `LAYOUT` and a rank's KV head matches the stock loader's split.
    `take_pools` binds a stock-like manager; `take_draft_pools` takes the draft-only manager under
    attention TP (unless a cache transceiver forbids it), else the trunk's, refusing the other."""
    g, mapping = _geometry(member, rank, algo), _mapping(member, rank)
    kv_dtype, kv_width, side_planes = KV_STORAGE[algo]
    assert g._asdict() == dict(
        COMMON,
        **LAYOUT[member, rank],
        kv_dtype=kv_dtype,
        kv_width=kv_width,
        qsa_side_planes=side_planes,
    )
    assert g.conv_sections == (g.num_k_heads * 128, g.num_k_heads * 128, g.num_v_heads * 128)
    assert g.kv_block_scales == (algo == QuantAlgo.NVFP4)
    heads, rows_per_head = 2, 4
    weight = torch.arange(heads * rows_per_head, dtype=torch.float32).reshape(-1, 1)
    tp = P.attention_tp_size(mapping)
    local = duplicate_kv_weight(weight, heads, tp).chunk(tp)[rank if tp > 1 else 0]
    start, stop = g.kv_head_start * rows_per_head, (g.kv_head_start + g.heads_kv) * rows_per_head
    assert torch.equal(local, weight[start:stop])

    trunk = _MetaManager(g, mapping, ATTN_LAYERS)
    pools = P.take_pools(trunk, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)
    assert sorted(pools.kv) == ATTN_LAYERS and sorted(pools.conv) == GDN_LAYERS
    first = pools.kv[ATTN_LAYERS[0]]
    assert tuple(first.k.shape[1:]) == (g.heads_kv, TOKENS_PER_BLOCK, g.kv_width)
    assert (first.k_scale is not None) == (algo == QuantAlgo.NVFP4)

    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    separate = P.draft_kv_cache_is_separate(mapping)
    assert separate == (
        not mapping.enable_attention_dp and should_use_separate_draft_kv_cache(spec)
    )
    assert P.draft_kv_cache_is_separate(mapping, spec) == separate
    spec._allow_separate_draft_kv_cache = False  # what a cache transceiver makes the executor do
    assert not should_use_separate_draft_kv_cache(spec)
    assert not P.draft_kv_cache_is_separate(mapping, spec)
    draft_alone = _MetaManager(g, mapping, [MTP_LAYER], [])
    trunk_and_draft = _MetaManager(g, mapping, [*ATTN_LAYERS, MTP_LAYER])
    for sep in sorted({separate, False}):  # the engine's choice, and a disaggregated engine's
        cm, wrong = (draft_alone, trunk_and_draft) if sep else (trunk_and_draft, draft_alone)
        draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=sep)
        assert draft.pool_idx == cm.layer_offsets[MTP_LAYER]
        assert tuple(draft.kv.v.shape[1:]) == (g.heads_kv, TOKENS_PER_BLOCK, g.kv_width)
        with pytest.raises(AssertionError, match="draft layer"):
            P.take_draft_pools(wrong, g, MTP_LAYER, separate=sep)
        with pytest.raises(AssertionError, match="does not map"):
            P.take_draft_pools(trunk, g, MTP_LAYER, separate=sep)


# ------------------------------------------------ the stock manager, on GPU ---


@contextlib.contextmanager
def _manager(member: str, rank: int, kv_cache_quant_algo=None, spec_config=None, is_draft=False):
    """The trunk manager (or with `is_draft` the separate draft KV cache) the engine builds on
    this rank, with the core's cache config: bf16 recurrent state, no block reuse, manager V2."""
    # Imported here: the stock factory pulls in the executor, which the cpu_only cases skip.
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

    mapping = _mapping(member, rank)
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=mapping,
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES, kv_cache_quant_algo=kv_cache_quant_algo
        ),
        sparse_attention_config=QSASparseAttentionConfig(),
        spec_config=spec_config,
    )
    # What the loader does with `kv_cache_config.mamba_ssm_cache_dtype: bfloat16`.
    validate_and_set_mamba_ssm_cache_dtype(model_config, "bfloat16")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="bfloat16",
        max_tokens=4096,
        use_kv_cache_manager_v2=True,
    )
    cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    cm = _create_kv_cache_manager(
        None,
        cls,
        mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=2048,
        max_batch_size=MAX_BATCH_SIZE,
        spec_config=spec_config,
        sparse_attention_config=model_config.sparse_attention_config,
        max_num_tokens=2048,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=is_draft,
        max_cuda_graph_batch_size=MAX_BATCH_SIZE,
    )
    try:
        yield cm
    finally:
        cm.shutdown()


def _stock_trunk(member, rank, algo):
    """Each trunk view passes the contract; the manager's per-rank numbers are the geometry's."""
    from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager
    from tensorrt_llm.bindings import DataType

    binding = {None: DataType.BF16, QuantAlgo.FP8: DataType.FP8, QuantAlgo.NVFP4: DataType.NVFP4}
    g, mapping = _geometry(member, rank, algo), _mapping(member, rank)
    with _manager(member, rank, algo) as cm:
        assert type(cm) is Qwen4ExpHybridCacheManager
        assert cm.dtype == binding[algo]
        assert cm.ssm_state_dtype == torch.bfloat16
        pools = P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)
        assert {cm.num_kv_heads_per_layer[cm.layer_offsets[i]] for i in ATTN_LAYERS} == {g.heads_kv}
        assert (cm.qsa_index_kv_heads, cm.qsa_index_dim) == (g.index_kv_heads, g.index_head_dim)
        assert list(cm.conv_state_shape) == [g.conv_dim, g.conv_kernel - 1]
        assert list(cm.ssm_state_shape) == [g.num_v_heads, g.head_v_dim, g.head_k_dim]
        assert (cm.qsa_side_planes_size is not None) == g.qsa_side_planes
        assert cm._num_reserved_dummy_slots == P.reserved_dummy_slots(mapping)
        assert len(set(pools.pool_idx.values())) == len(ATTN_LAYERS)
        # One slot space for every QSA view of the layers, one for the recurrent and PLE state.
        slots = {view.shape[0] for view in (pools.position_cache, *pools.index_cache.values())}
        slots |= {planes.k.shape[0] for planes in pools.kv.values()}
        assert len(slots) == 1
        assert {t.shape[0] for t in (*pools.conv.values(), *pools.ssm.values())} == {
            pools.ple_state[0].shape[0]
        }
        assert MTP_LAYER not in cm.layer_offsets


def _stock_mtp(member, rank, max_draft_len):
    """With MTP the verify buffers (conv windows, GDN replay history) hold this rank's heads at
    width `max_draft_len + 1`; the trunk manager maps the draft layer only under attention DP."""
    g, mapping = _geometry(member, rank), _mapping(member, rank)
    spec = MTPDecodingConfig(max_draft_len=max_draft_len, num_nextn_predict_layers=1)
    with _manager(member, rank, spec_config=spec) as cm:
        pools = P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)
        assert cm.is_speculative()
        assert cm._num_reserved_dummy_slots == P.reserved_dummy_slots(mapping)
        ssm0 = pools.ssm[GDN_LAYERS[0]]
        caches = {i: cm.mamba_layer_cache(i) for i in GDN_LAYERS}
        for i, cache in caches.items():
            P.check_step_buffers(
                g, cache.intermediate_conv_window, pools.conv[i], cm.intermediate_state_indices
            )
            assert cache.intermediate_ssm is None, "the cached replay keeps no per-step state"
        assert not cm.use_gdn_flashinfer_ring_replay
        meta = cm.get_replay_state_update_metadata()
        P.check_replay_state(
            meta.replay_history_size,
            meta.replay_step_width,
            meta.prev_num_accepted_tokens,
            meta.cache_buf_idx,
            ssm0,
        )
        assert meta.replay_step_width == max_draft_len + 1
        P.check_verify_width(meta.replay_step_width)
        for i, cache in caches.items():
            P.check_replay_history(
                g,
                i,
                meta.replay_history_size,
                ssm0.shape[0],
                cache.old_x,
                cache.old_B,
                cache.old_dt,
            )
        separate = P.draft_kv_cache_is_separate(mapping)
        assert (MTP_LAYER in cm.layer_offsets) == (not separate)
        if not separate:
            draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=False)
            assert draft.position_cache.data_ptr() == pools.position_cache.data_ptr()


def _stock_draft(member, rank, algo):
    """Under attention TP the separate draft manager holds the draft layer alone at trunk shapes."""
    g = _geometry(member, rank, algo)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    with _manager(member, rank, algo, spec_config=spec, is_draft=True) as cm:
        draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=True)
        assert cm.ple_layer_cache(PLE_LAYER) is None
        assert cm.local_num_mamba_layers == 0
        assert cm.num_kv_heads_per_layer[draft.pool_idx] == g.heads_kv
        assert (cm.qsa_side_planes_size is not None) == g.qsa_side_planes
        assert draft.kv.k.shape[0] == draft.index_cache.shape[0] == draft.position_cache.shape[0]


_STOCK_CASES = [
    *(
        pytest.param(_stock_trunk, m, r, dict(algo=a), id=f"trunk-{m}-rank{r}-{KV_IDS[a]}")
        for m, r in LAYOUT_RANKS
        for a in _ALGOS
    ),
    # Each attention layout at three drafts, and the widest verify under attention TP and DP.
    *(
        pytest.param(_stock_mtp, m, r, dict(max_draft_len=d), id=f"mtp{d}-{m}-rank{r}")
        for m, r, d in [*((m, r, 3) for m, r in LAYOUT_RANKS), ("tp4", 3, 7), ("dep4", 3, 7)]
    ),
    *(
        pytest.param(_stock_draft, m, r, dict(algo=a), id=f"draft-kv-{m}-rank{r}-{KV_IDS[a]}")
        for m, r in LAYOUT_RANKS
        if m != "dep4"
        for a in _ALGOS
    ),
]


@_needs_gpu
@pytest.mark.parametrize("check, member, rank, kwargs", _STOCK_CASES)
def test_the_stock_manager_serves_the_layout(check, member, rank, kwargs, monkeypatch):
    """The stock `Qwen4ExpHybridCacheManager`, built as the engine does, passes the contract."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    check(member, rank, **kwargs)


# ------------------------------------------------------ the decode step's runtime paths ---


def _runtime_hooks(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    hooks = M._runtime_hooks()
    assert isinstance(hooks.decode_prologue, M.SpecStepMetadataPrologue)
    assert replace(hooks, decode_prologue=None) == RuntimeHooks(loop_on_execution_stream=True)


def _decode_stream(monkeypatch):
    """A generation-only step of up to `_DECODE_HIGH_PRIORITY_MAX_ROWS` rows runs on the core's
    high-priority decode stream, ordered after the caller's work and before its next, eager and
    captured; a larger step, a prefill step or a core without the stream stays on the caller's."""
    hi = torch.cuda.Stream(priority=-100)
    assert hi.priority < torch.cuda.Stream().priority
    core = SimpleNamespace(_decode_stream=hi)
    caller = torch.cuda.current_stream()
    seen = []
    src = torch.full((1 << 20,), 1.0, device="cuda")

    def step(scale):
        seen.append(torch.cuda.current_stream())
        return src * scale

    src.fill_(3.0)  # the caller's pending write is ordered before the step
    out = M._on_decode_stream(core, 3, step, 2.0)
    assert seen[-1] == hi and torch.cuda.current_stream() == caller
    assert torch.equal(out, torch.full_like(src, 6.0))
    # Captured: the waits become graph edges and a replay reads the refilled source.
    side = torch.cuda.Stream()
    side.wait_stream(caller)
    with torch.cuda.stream(side):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            captured = M._on_decode_stream(core, 3, step, 5.0)
    caller.wait_stream(side)
    src.fill_(1.5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, torch.full_like(src, 7.5))
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.forward = lambda *a, **k: seen.append(torch.cuda.current_stream())
    target.mtp_forward = lambda *a, **k: seen.append(torch.cuda.current_stream())
    prefill = SimpleNamespace(forward=target.forward, mtp_forward=target.mtp_forward)
    rows = M._DECODE_HIGH_PRIORITY_MAX_ROWS
    for chosen, n, want in ((target, rows, hi), (target, rows + 1, caller), (prefill, 3, caller)):
        ids = torch.zeros(n, dtype=torch.int32, device="cuda")
        core.__dict__.update(_step_begun=True, _select_target=lambda md, t=chosen: t)
        M.Qwen4ExpUnifiedModelingV2.forward(core, None, input_ids=ids)
        assert seen[-1] == want, (n, chosen)
        M.MTPLayer.forward(SimpleNamespace(core=core), None, None, ids, None, None, None)
        assert seen[-1] == want, (n, chosen)
    M._on_decode_stream(SimpleNamespace(), 3, step, 1.0)
    assert seen[-1] == caller, "a core without the stream runs on the caller's"


class _ShardHead:
    """A COLUMN-parallel `lm_head` stand-in recording each call's `gather_output`."""

    def __init__(self, padding_size: int = 0) -> None:
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        self.tp_mode = TensorParallelMode.COLUMN
        self.gather_output = True
        self.padding_size = padding_size
        self.calls = []

    def __call__(self, x, gather_output=True):
        self.calls.append(gather_output)
        return "shard"


def _draft_head_shard(monkeypatch, case, sharded):
    """With the worker's TP key exchange a greedy draft pass keeps this rank's vocab shard of the
    logits; rejection sampling, guidance, excess rows, a padded vocab or no exchange gather them."""
    monkeypatch.setattr(M, "_hc_site", lambda *args: (None, torch.zeros(3, 8), None))
    core = SimpleNamespace(_mtp=SimpleNamespace(fhc_norm=None, fhc_first=None, fhc_up=None))
    head = _ShardHead(padding_size=16 if case == "padded_vocab" else 0)
    worker = SimpleNamespace(key_exchange=SimpleNamespace(MAX_ROWS=8), guided_decoder=None)
    if case == "no_exchange":
        worker.key_exchange = None
    if case == "guided":
        worker.guided_decoder = object()
    if case == "rows_past_exchange":
        worker.key_exchange.MAX_ROWS = 2
    layer = M.MTPLayer(core, logits_processor=SimpleNamespace(forward=lambda *args: "gathered"))
    layer.worker = worker
    layer.spec_metadata = SimpleNamespace(
        wants_advanced_draft_sampling=case == "rejection_sampling"
    )
    out = layer.shared_head(torch.zeros(3, 32), head, None, True)
    assert out == ("shard" if sharded else "gathered")
    assert head.calls == ([False] if sharded else [])


_RUNTIME_CASES = [
    pytest.param(_runtime_hooks, {}, marks=_cpu, id="runtime-hooks"),
    pytest.param(_decode_stream, {}, marks=_needs_gpu, id="decode-stream"),
    *(
        pytest.param(_draft_head_shard, dict(case=c, sharded=c == "greedy"), marks=_cpu,
                     id=f"draft-head-{c}")
        for c in ("greedy", "rejection_sampling", "no_exchange", "guided", "rows_past_exchange",
                  "padded_vocab")
    ),
]  # fmt: skip


@pytest.mark.parametrize("check, kwargs", _RUNTIME_CASES)
def test_the_decode_steps_runtime_paths(check, kwargs, monkeypatch) -> None:
    """The runtime hooks, the high-priority decode stream and the draft head's vocab-shard path."""
    check(monkeypatch, **kwargs)

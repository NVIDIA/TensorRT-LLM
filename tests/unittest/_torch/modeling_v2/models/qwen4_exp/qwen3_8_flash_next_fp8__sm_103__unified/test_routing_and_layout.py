# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-FP8 routing, and the per-rank layout of the unified target's core and pools.

Routing (cpu_only, synthetic configs): `route` tells the FP8 export from the NVFP4 one by the
quantization the engine resolves for the checkpoint (FP8_BLOCK_SCALES here, MIXED_PRECISION for
NVFP4), and `routing.fp8_member` admits the FP8 checkpoint at tp1, tp2, tep2, dep2, tp4, tep4 and
dep4 -- one-model MTP-Eagle (max_draft_len 1..7) under attention TP and at dep4, the breakable
prefill graph under attention TP, a bf16 recurrent state, a bf16, FP8 or NVFP4 KV cache and the
trtllm-gen experts -- and declines everything else, saying why. Every admitted deployment resolves
to this target under both `auto` and `require` (`_admitted`). A choice routing checks the same way
at every member is driven at the representative members tp1, tep4 and dep4; each decline is driven
once per reason it is refused for.

The core (cpu_only): built on the meta device from the checkpoint's text and vision config at the
members' ranks, so nothing touches a device. What is pinned is what the forward addresses: the
fused attention row and the split the norm + rope reads it through, the Gated-DeltaNet
projection's column split, the whole-expert window a rank holds (by expert at every member,
whatever the `Mapping`'s MoE TP/EP split), how a MoE layer enters the FP8 runner over that window,
and the attention-DP round trip's padding and slicing. With MTP: the draft layer's attention share
and expert window, the verify width, and the draft MoE's padding and round trip under attention DP
by the row counts the spec worker hands each draft step.

The pools: `pool.pool_geometry` derives one rank's pool shapes from the text config and the rank's
`Mapping`. The cpu_only cases hold it to `LAYOUT` (what the target binds at every rank of the seven
members) and drive `take_pools` and `take_draft_pools` over a meta-device stand-in laid out like the
stock manager's views (the HND K/V role pair, the NVFP4 block-scale pair, the packed QSA side page),
with each wrong pool a consuming kernel would accept. The engine facts the contract restates -- the
reserved dummy slots, when the draft layer gets a manager of its own (never in a disaggregated
engine: a cache transceiver disallows a separate draft KV cache) -- are compared with the engine's
own functions. The GPU cases build the stock `Qwen4ExpHybridCacheManager` the way the engine does
-- `get_kv_cache_manager_cls` and `_create_kv_cache_manager` over a `ModelConfig`, with the core
runtime's cache configuration (bf16 recurrent state, block reuse off, KV cache manager V2)
-- and take every view through the contract: the trunk's pools; with MTP the verify buffers and the
GDN cached replay; and the draft layer's pools, in the trunk's manager under attention DP and in the
separate draft KV cache the engine builds under attention TP.

Which members a layout case runs at: tp2 / tep2 and tp4 / tep4 differ only in how the experts
are split, which moves no head and no pool (`pool.py` reads the attention layout alone), and under
attention DP a rank holds every head as tp1's does. The geometry test holds every rank of every
member to `LAYOUT`, which states exactly that; the cases that bind or cut heads then run every rank
of tp1, tp2 and tp4 and one rank of dep2 and dep4 (`HEAD_RANKS`), and those that bind whole pools
the last rank of one member per attention layout (`LAYOUT_RANKS`). The expert window, which the
MoE split could move, is held at every member.
"""

import ast
import contextlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
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
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    _get_num_cuda_graph_padding_dummy_slots,
)
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import QSASparseAttentionConfig
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# ------------------------------------------------------------------ the members ---

NUM_LAYERS = 48
CHECKPOINT = "Qwen3.8-Flash-Next-FP8"

#: The seven members, as the LLM API's parallel arguments build their `Mapping`. tp2 and tp4
#: leave the experts to Mapping's default split, which is the TP size; tep and dep split them by
#: expert; dep replicates attention.
MEMBERS = {
    "tp1": dict(world_size=1, tp_size=1),
    "tp2": dict(world_size=2, tp_size=2),
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}
_ATTENTION_TP = ["tp1", "tp2", "tep2", "tp4", "tep4"]
#: One member of each kind routing tells apart: one rank, attention TP, attention DP.
_REPRESENTATIVE = ["tp1", "tep4", "dep4"]

#: Every rank of every member.
_RANKS = [(m, r) for m, layout in MEMBERS.items() for r in range(layout["world_size"])]
_RANK_IDS = [f"{member}-rank{rank}" for member, rank in _RANKS]
#: The ranks that hold different heads: every rank of tp1, tp2 and tp4 (tep2 and tep4 hold the
#: same), and one rank of each attention-DP member (every rank holds every head).
HEAD_RANKS = [(m, r) for m, r in _RANKS if m in ("tp1", "tp2", "tp4")] + [("dep2", 1), ("dep4", 3)]
_HEAD_RANK_IDS = [f"{member}-rank{rank}" for member, rank in HEAD_RANKS]
#: The last rank of one member per attention layout: one rank, TP2, TP4, attention DP.
LAYOUT_RANKS = [("tp1", 0), ("tp2", 1), ("tp4", 3), ("dep4", 3)]
_LAYOUT_RANK_IDS = [f"{member}-rank{rank}" for member, rank in LAYOUT_RANKS]


def _mapping(member: str, rank: int) -> Mapping:
    """Rank `rank` of `member` in one node of four GB300s."""
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _world(member: str) -> int:
    return MEMBERS[member]["world_size"]


_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the manager allocates its pools on the GPU"
)


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
_NVFP4_TP1_TARGET = (
    "models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling."
    "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1"
)

# The quantization the engine resolves for Qwen3.8-Flash-Next-FP8 (`quant_method: fp8` with
# 128x128 weight blocks), and for the NVFP4 export.
_FP8 = QuantAlgo.FP8_BLOCK_SCALES
_NVFP4 = QuantAlgo.MIXED_PRECISION


# The node every member spans: one GB300 node of four devices.
_GPUS_PER_NODE = 4


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture
def on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs, with the GDN
    cached replay at its default (on)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _qwen4_exp_config(text_overrides=None, composite_overrides=None, **vision_overrides):
    """The shape Qwen3.8-Flash-Next's normalized composite config declares (FP8 and NVFP4
    alike)."""
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
        **(composite_overrides or {}),
    )


def _routed_config(
    member=None, spec_config=None, pretrained_config=None, quant_algo=_FP8, **mapping_kwargs
) -> ModelConfig:
    """The loaded config of the FP8 checkpoint (unless `quant_algo` says otherwise) at
    `member`, or at `mapping_kwargs`; its own KV is bf16 (no KV-cache algorithm)."""
    if member is not None:
        mapping_kwargs = dict(MEMBERS[member], **mapping_kwargs)
    return ModelConfig(
        pretrained_config=pretrained_config or _qwen4_exp_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=None),
    )


def _admitted(config, member, **llm_args_kwargs) -> None:
    """`fp8_member` admits the deployment as `member`, and `route` returns the FP8 target for
    it under both modes."""
    assert routing.fp8_member(config, _llm_args(**llm_args_kwargs)) == member
    for mode in ("auto", "require"):
        target = modeling_v2_resolve(config, _llm_args(mode, **llm_args_kwargs))
        assert target is not None and _qualified(target) == _TARGET, mode


def _declined(config, match=None, **llm_args_kwargs) -> None:
    """``auto`` falls back to the built-in model and ``require`` says why."""
    assert modeling_v2_resolve(config, _llm_args("auto", **llm_args_kwargs)) is None
    with pytest.raises(ValueError, match=match):
        modeling_v2_resolve(config, _llm_args("require", **llm_args_kwargs))


def _mtp(max_draft_len=3, **attributes):
    """An MTP config with fields the engine fills in after load set directly: the
    constructor remaps or recomputes them, so passing them would test nothing."""
    config = MTPDecodingConfig(max_draft_len=max_draft_len)
    for name, value in attributes.items():
        setattr(config, name, value)
    return config


@pytest.mark.cpu_only
def test_routing_admits_the_seven_members_and_their_engine_choices(on_sm103):
    assert sorted(routing._FP8_LAYOUTS.values()) == sorted(MEMBERS)
    assert set(routing._FP8_MOE_BACKENDS) == {"AUTO", "TRTLLM"}
    assert routing._FP8_SSM_STATE_DTYPES == (torch.bfloat16,)
    assert routing._FP8_ATTENTION_DP_MTP_LAYOUTS == ("dep4",)
    assert routing._FP8_QUANT_ALGO == _FP8 and routing._NVFP4_QUANT_ALGO == _NVFP4


@pytest.mark.cpu_only
def test_routing_admits_exactly_what_the_target_implements(on_sm103):
    """`route` declines a MoE backend, a draft length, a recurrent-state dtype or a
    quantization the target does not run; this keeps routing's constants and the target's
    together."""
    assert set(routing._FP8_MOE_BACKENDS) == set(M._TRUNK_MOE_BACKENDS)
    assert routing._MAX_DRAFT_LEN == M._MAX_VERIFY_WIDTH - 1
    assert routing._FP8_SSM_STATE_DTYPES == M._SSM_STATE_DTYPES
    assert routing._FP8_QUANT_ALGO == M._QUANT_ALGO
    assert routing._SM == M._SM


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", sorted(MEMBERS))
def test_each_member_is_admitted(on_sm103, member):
    config = _routed_config(member)
    assert routing.layout(config.mapping, routing._FP8_LAYOUTS) == member
    for dtype in ("bfloat16", "auto"):
        _admitted(config, member, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))


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
def test_the_llm_api_spells_each_member_as_routed(on_sm103, member, parallel, monkeypatch):
    """The LLM API's parallel arguments build the mapping each member is keyed by: a tp member
    with or without `moe_expert_parallel_size=1` leaves the experts to the TP split.

    The node is stated (`gpus_per_node`, four GB300s): left unset, `TorchLlmArgs` takes it from
    the devices this process sees, and a job holding one of them reads a two- or four-rank world
    as several nodes, which `layout` declines. The test sees one device either way."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    mapping = _llm_args(gpus_per_node=_GPUS_PER_NODE, **parallel).parallel_config.to_mapping()
    assert not mapping.is_multi_node()
    assert routing.layout(mapping, routing._FP8_LAYOUTS) == member


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, max_draft_len",
    [(member, 3) for member in _ATTENTION_TP] + [("tp1", 1), ("tp1", 7)],
)
def test_mtp_is_admitted_under_attention_tp(on_sm103, member, max_draft_len):
    """One-model MTP-Eagle at every attention-TP member, and at the shortest and the longest
    draft the target verifies (1 and 7; 8 is declined below)."""
    config = _routed_config(member, MTPDecodingConfig(max_draft_len=max_draft_len))
    for dtype in ("bfloat16", "auto"):
        _admitted(config, member, kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype=dtype))


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_draft_len", [1, 7])
def test_mtp_is_admitted_at_dep4(on_sm103, max_draft_len):
    """Each dep4 rank drafts for its own requests: the high-throughput tune serves MTP here."""
    config = _routed_config("dep4", MTPDecodingConfig(max_draft_len=max_draft_len))
    _admitted(config, "dep4")
    _admitted(config, "dep4", kv_cache_config=KvCacheConfig(dtype="fp8"))


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", ["fp8", "nvfp4", "auto"])
@pytest.mark.parametrize("member", _REPRESENTATIVE)
def test_a_bf16_fp8_or_nvfp4_kv_cache_is_admitted(on_sm103, member, dtype):
    _admitted(_routed_config(member), member, kv_cache_config=KvCacheConfig(dtype=dtype))


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["disabled", "breakable"])
@pytest.mark.parametrize("member", ["tp1", "tep4"])
def test_no_or_the_breakable_prefill_cuda_graph_is_admitted_under_attention_tp(
    on_sm103, member, backend
):
    _admitted(_routed_config(member), member, prefill_cuda_graph_backend=backend)


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["AUTO", "TRTLLM"])
@pytest.mark.parametrize("member", _REPRESENTATIVE)
def test_the_trtllm_moe_backend_is_admitted(on_sm103, member, backend):
    _admitted(replace(_routed_config(member), moe_backend_requested=backend), member)


# --- what attention DP does not run ----------------------------------------------


@pytest.mark.cpu_only
def test_dep2_declines_speculation(on_sm103):
    """MTP under attention DP is admitted where the tune serves it (dep4); dep2 is the
    disaggregated-serving engines' layout, served without speculation."""
    config = _routed_config("dep2", MTPDecodingConfig(max_draft_len=3))
    _declined(config, match=r"speculative decoding with attention DP \(dep2\)")


@pytest.mark.cpu_only
def test_attention_dp_declines_the_breakable_prefill_graph(on_sm103):
    _declined(
        _routed_config("dep2"),
        match=r"prefill_cuda_graph_backend='breakable' with attention DP \(dep2\)",
        prefill_cuda_graph_backend="breakable",
    )


@pytest.mark.cpu_only
def test_dep4_declines_mtp_with_the_breakable_prefill_graph(on_sm103):
    config = _routed_config("dep4", MTPDecodingConfig(max_draft_len=3))
    _declined(
        config,
        match=r"prefill_cuda_graph_backend='breakable' with attention DP \(dep4\)",
        prefill_cuda_graph_backend="breakable",
    )


# --- engine choices the target does not implement ---------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", _REPRESENTATIVE)
def test_the_piecewise_prefill_cuda_graph_declines(on_sm103, member):
    _declined(_routed_config(member), match="piecewise", prefill_cuda_graph_backend="piecewise")


@pytest.mark.cpu_only
def test_another_moe_backend_declines(on_sm103):
    config = replace(_routed_config("tep4"), moe_backend_requested="CUTLASS")
    _declined(config, match="moe_config.backend")


@pytest.mark.cpu_only
def test_a_recurrent_state_cache_other_than_bf16_declines(on_sm103):
    """The checkpoint's own `mamba_ssm_dtype` is float32; the target serves bf16 state."""
    _declined(
        _routed_config("tep4", MTPDecodingConfig(max_draft_len=3)),
        match="recurrent-state cache",
        kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32"),
    )


@pytest.mark.cpu_only
def test_another_model_dtype_declines(on_sm103):
    config = _routed_config(
        "tep4", pretrained_config=_qwen4_exp_config(composite_overrides=dict(dtype="float16"))
    )
    _declined(config, match="runs the model in torch.float16")


@pytest.mark.cpu_only
def test_other_kv_cache_algorithms_decline(on_sm103):
    choices = routing.engine_choices(_routed_config("tep4"), _llm_args())
    for algo in (QuantAlgo.INT8, QuantAlgo.FP8_BLOCK_SCALES):
        with pytest.raises(routing.NoTarget, match="KV cache"):
            routing._check_fp8_engine(None, algo, choices)
    _declined(
        _routed_config("tep4"), match="KV cache", kv_cache_config=KvCacheConfig(dtype="fp8_ds_mla")
    )


@pytest.mark.cpu_only
def test_mtp_without_the_gdn_cached_replay_declines(on_sm103, monkeypatch):
    """The target's verify step runs the GDN cached replay at every draft length; the legacy
    per-step recurrent states `TRTLLM_USE_GDN_REPLAY=0` selects are not assembled."""
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
    config = _routed_config("tep4", MTPDecodingConfig(max_draft_len=3))
    _declined(config, match="TRTLLM_USE_GDN_REPLAY=0")
    # Without speculation the variable selects nothing the target runs.
    _admitted(_routed_config("tep4"), "tep4")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "spec_config",
    [
        NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
        MTPDecodingConfig(max_draft_len=8),
        MTPDecodingConfig(max_draft_len=3, use_mtp_vanilla=True),
        _mtp(max_total_draft_tokens=5),
        _mtp(num_nextn_predict_layers=2),
    ],
    ids=["ngram", "draft8", "vanilla-mtp", "max-total-draft", "two-predictors"],
)
def test_unsupported_speculation_declines(on_sm103, spec_config):
    _declined(_routed_config("tep4", spec_config), match="one-model MTP-Eagle")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "spec_kwargs",
    [
        dict(max_concurrency=8),
        dict(draft_len_schedule={4: 3, 8: 1}),
        dict(acceptance_rate_threshold=0.5),
    ],
    ids=["max-concurrency", "draft-len-schedule", "acceptance-rate"],
)
def test_a_runtime_draft_length_declines(on_sm103, spec_kwargs):
    """The verify step runs at the one width max_draft_len + 1."""
    spec = MTPDecodingConfig(max_draft_len=3, **spec_kwargs)
    _declined(_routed_config("tep4", spec), match="draft length")


# --- other layouts ------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # pipeline or context parallelism: TP narrower than the world
        dict(world_size=2, tp_size=1, pp_size=2),
        # four ranks over two nodes
        dict(MEMBERS["tep4"], gpus_per_node=2),
        # lm_head TP under attention DP
        dict(MEMBERS["dep4"], enable_lm_head_tp_in_adp=True),
        # attention DP with the experts split by intermediate
        dict(MEMBERS["dep4"], moe_tp_size=4, moe_ep_size=1),
        # experts split two ways by intermediate and two by expert
        dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
        # eight ranks
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
    ],
    ids=["pp2", "two-nodes", "dep4-lm-head-tp", "dep4-moetp4", "tp4-moetp2-ep2", "tep8"],
)
def test_other_layouts_decline(on_sm103, mapping_kwargs):
    """One layout per way `routing.layout` refuses one: a TP group narrower than the world,
    a group spanning nodes, lm_head TP under attention DP, and a (world, attention DP, MoE TP,
    MoE EP) key no member has."""
    config = _routed_config(**mapping_kwargs)
    assert routing.layout(config.mapping, routing._FP8_LAYOUTS) is None
    _declined(config, match="the FP8 target serves Qwen3.8-Flash-Next-FP8 in one node")


# --- the checkpoint -----------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tp1", "tep4"])
def test_the_quantization_picks_the_checkpoint(on_sm103, member):
    """One geometry, two exports: at MIXED_PRECISION a member the NVFP4 targets serve lands on
    them, at FP8_BLOCK_SCALES the same deployment is the FP8 target's and never theirs."""
    nvfp4 = _routed_config(member, quant_algo=_NVFP4)
    expected = _NVFP4_TP1_TARGET if member == "tp1" else _NVFP4_DEFAULT_TARGET
    assert _qualified(modeling_v2_resolve(nvfp4, _llm_args("require"))) == expected
    _admitted(_routed_config(member), member)


@pytest.mark.cpu_only
@pytest.mark.parametrize("quant_algo", [None, QuantAlgo.FP8])
def test_another_quantization_declines(on_sm103, quant_algo):
    """An unquantized checkpoint, and per-tensor FP8 rather than 128x128 blocks."""
    _declined(
        _routed_config("tep4", quant_algo=quant_algo), match=f"quantization {quant_algo or 'none'}"
    )


@pytest.mark.cpu_only
def test_another_checkpoint_shape_declines(on_sm103):
    config = _routed_config("tep4", pretrained_config=_qwen4_exp_config(dict(num_hidden_layers=60)))
    _declined(config, match="text shape")


@pytest.mark.cpu_only
def test_a_deepstack_vision_tower_declines(on_sm103):
    config = _routed_config(
        "tep4", pretrained_config=_qwen4_exp_config(deepstack_visual_indexes=[5, 11, 17])
    )
    _declined(config, match="vision shape")


@pytest.mark.cpu_only
def test_sm100_declines(on_sm103, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    _declined(_routed_config("tep4"), match=r"compute capability \(10, 0\)")


# ------------------------------------------------------------------------ the core ---

# Qwen3.8-Flash-Next's text config (the FP8 export's `text_config`).
_TEXT = dict(
    architectures=["Qwen4ExpForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=248044,
    dtype="bfloat16",
    eos_token_id=248044,
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=320,
    head_dim=256,
    heads_per_ngram=8,
    hidden_act="silu",
    hidden_size=2560,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    indexer_kv_heads=1,
    indexer_n_heads=4,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(NUM_LAYERS)],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    make_ngram_vocab_size_divisible_by=128,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    moe_intermediate_size=640,
    mtp=dict(
        hybrid=True,
        layer_types=["full_attention"],
        mtp_use_hidden_state_from_layer=None,
        num_hidden_layers=1,
        rope_theta=10000000,
    ),
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    ngram_size=3,
    ngram_vocab_size_base=20000000,
    num_attention_heads=24,
    num_experts=512,
    num_experts_per_tok=10,
    num_hidden_layers=NUM_LAYERS,
    num_key_value_heads=2,
    output_gate_type="sigmoid",
    partial_rotary_factor=0.25,
    ple_conv_kernel_size=4,
    ple_embed_dim=2560,
    ple_layer_ids=[2],
    rms_norm_eps=1e-06,
    rope_parameters=dict(
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        rope_type="default",
    ),
    shared_expert_intermediate_size=640,
    split_ngram_parts=128,
    tie_word_embeddings=False,
    vocab_size=248320,
)
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=27,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)


# Per member: (q heads, KV heads, fused attention row, rope "v" heads, GDN key / value heads,
# GDN in_proj split, o / out_proj K). tp2 == tep2 and tp4 == tep4: the MoE split moves none.
_GEOMETRY = {
    "tp1": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
    "tp2": (12, 1, 7424, 16, 8, 24, (5120, 3072, 24, 24), 3072),
    "tep2": (12, 1, 7424, 16, 8, 24, (5120, 3072, 24, 24), 3072),
    "dep2": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
    "tp4": (6, 1, 4352, 10, 4, 12, (2560, 1536, 12, 12), 1536),
    "tep4": (6, 1, 4352, 10, 4, 12, (2560, 1536, 12, 12), 1536),
    "dep4": (24, 2, 14080, 29, 16, 48, (10240, 6144, 48, 48), 6144),
}


def _core(member: str, rank: int, spec_config=None, quant_algo=QuantAlgo.FP8_BLOCK_SCALES):
    """The core of rank `rank` of `member`, every parameter on meta."""
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=_mapping(member, rank),
        quant_config=QuantConfig(quant_algo=quant_algo),
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", HEAD_RANKS, ids=_HEAD_RANK_IDS)
def test_each_rank_runs_its_share_of_the_heads(member, rank) -> None:
    """The fused row is `[q | k | v | gate | index q, k | v slot]`; the main norm + rope reads
    it as q, k and "v" heads of 256 (the row stride alone). The KV head a rank holds is the
    one its K/V pool holds: rank `r` of two holds head `r`, of four head `r // 2`."""
    core = _core(member, rank)
    hq, hkv, row, tail, hk, hv, split, out_k = _GEOMETRY[member]
    world, adp = _world(member), MEMBERS[member].get("enable_attention_dp", False)
    attn_tp = 1 if adp else world
    assert (core.heads_q, core.heads_kv) == (hq, hkv)
    assert core.kv_head_start == (rank * 2 // attn_tp if attn_tp > 1 else 0)
    assert (core.attn_in_width, core.attn_in_tail_heads) == (row, tail)
    assert core.index_col == core.qkv_width + core.q_width == 2 * (hq + hkv) * 256
    assert tuple(core.w["l3_attn_in_w"].shape) == (row, 2560)
    assert tuple(core.w["l3_o_w"].shape) == (2560, out_k)
    assert (core.num_k_heads, core.num_v_heads) == (hk, hv)
    assert tuple(core.in_proj_split) == split
    assert tuple(core.w["l0_in_proj_w"].shape) == (sum(split), 2560)
    assert tuple(core.w["l0_out_w"].shape) == (2560, out_k)
    assert core.attn_tp_size == attn_tp
    assert core.dp_size == (world if adp else 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", list(MEMBERS))
def test_each_rank_holds_one_whole_expert_window(member) -> None:
    """Rank `r` of `t` holds experts `[512r / t, 512(r + 1) / t)` at the whole 640-row
    intermediate at every member: tp2 and tp4, which the `Mapping` puts in MoE TP, hold whole
    experts too (640 / 2 and 640 / 4 rows do not take the 128-row weight block). Held at each
    member's last rank, whose window starts furthest in."""
    world = _world(member)
    rank = world - 1
    core = _core(member, rank)
    count = 512 // world
    assert (core.local_experts, core.expert_offset, core.moe_inter) == (count, count * rank, 640)
    window = W.expert_window(core.w, core.dims, 0)
    assert (window.offset, window.count) == (count * rank, count)
    assert tuple(window.fc1_w.shape) == (count, 2 * 640, 2560)
    assert window.fc1_w.dtype == torch.float8_e4m3fn
    assert tuple(window.fc1_s.shape) == (count, 2 * 640 // 128, 2560 // 128)
    assert window.fc1_s.dtype == torch.float32
    assert tuple(window.fc2_w.shape) == (count, 2560, 640)
    assert tuple(window.fc2_s.shape) == (count, 2560 // 128, 640 // 128)
    assert tuple(core.w["l0_router"].shape) == (512, 2560)


@pytest.mark.cpu_only
def test_the_core_refuses_another_quantization() -> None:
    with pytest.raises(AssertionError, match="block-FP8 checkpoint"):
        _core("tep2", 0, quant_algo=QuantAlgo.MIXED_PRECISION)


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 8192, 8193, 20000])
def test_a_moe_layer_enters_the_fp8_runner_over_its_window(rows, monkeypatch) -> None:
    """`_routed_experts` routes and quantizes each chunk of at most 8192 rows and hands the
    runner the routing space (512), the window, the whole intermediate, renormalized routing,
    SwiGLU and the activation scales as the quantizer returned them."""
    core = _core("tep2", 1)
    experts = W.ExpertWindow("fc1_w", "fc1_s", "fc2_w", "fc2_s", 256, 256)
    calls = []

    def router(chunk, mat_b):
        assert mat_b == "router"
        return ("logits", chunk.shape[0])

    def routing(logits, topk, output_dtype):
        assert topk == 10 and logits[0] == "logits"
        return ("ids", logits[1]), ("w", logits[1])

    def quantize(chunk, use_ue8m0):
        assert use_ue8m0 is False
        return ("xq", chunk.shape[0]), ("xsf", chunk.shape[0])

    def runner(**kwargs):
        calls.append(kwargs)
        return torch.zeros(kwargs["hidden_states"][1], 4)

    monkeypatch.setattr(M, "cublas_mm", router)
    monkeypatch.setattr(M, "renorm_moe_routing_op", routing)
    monkeypatch.setattr(M, "fp8_quantize_1x128", quantize)
    monkeypatch.setattr(M, "fp8_block_scale_moe_runner", runner)
    out = M._routed_experts(core, torch.zeros(rows, 4), "router", experts)
    assert tuple(out.shape) == (rows, 4)
    chunks = [min(8192, rows - lo) for lo in range(0, rows, 8192)]
    assert [c["hidden_states"][1] for c in calls] == chunks
    for c, n in zip(calls, chunks):
        assert c["hidden_states_scale"] == ("xsf", n)
        assert (c["topk_ids"], c["topk_weights"]) == (("ids", n), ("w", n))
        assert c["routing_logits"] is None and c["routing_bias"] is None
        assert (c["gemm1_weights"], c["gemm1_weights_scale"]) == ("fc1_w", "fc1_s")
        assert (c["gemm2_weights"], c["gemm2_weights_scale"]) == ("fc2_w", "fc2_s")
        assert (c["num_experts"], c["top_k"], c["intermediate_size"]) == (512, 10, 640)
        assert (c["local_expert_offset"], c["local_num_experts"]) == (256, 256)
        assert c["routing_method_type"] == 1 and c["act_type"] == 0


@pytest.mark.cpu_only
def test_attention_dp_pads_to_the_groups_largest_row_count() -> None:
    """Each rank pads to the largest of `all_rank_num_tokens`; a count that disagrees with
    the rows this rank holds would gather the wrong split, and is refused."""
    core = SimpleNamespace(dp_size=2, dp_rank=1)
    md = SimpleNamespace(all_rank_num_tokens=[17, 9])
    assert M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(core, md, 9) == 8
    with pytest.raises(AssertionError, match="wrong split"):
        M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(core, md, 10)
    assert M.Qwen4ExpUnifiedModelingV2._dp_pad_rows(SimpleNamespace(dp_size=1), None, 3) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("world", [2, 4])
def test_the_attention_dp_round_trip_returns_this_ranks_rows_through_every_window(
    world, monkeypatch
) -> None:
    """The round trip pads this rank's rows, gathers the group's in rank order, runs this
    rank's window over all of them, reduce-scatters and drops the padding: this rank gets back
    the sum over windows of its own rows. The collectives are stood in for by the group's
    arithmetic."""
    counts = [3, 7, 1, 4][:world]
    hidden = 8
    gen = torch.Generator().manual_seed(world)
    xs = [torch.randn(n, hidden, generator=gen) for n in counts]
    rows = max(counts)

    def pad(x):
        return torch.cat([x, x.new_zeros(rows - x.shape[0], hidden)], 0)

    gathered = torch.cat([pad(x) for x in xs], 0)

    def window(w):
        return lambda t: t * (w + 1) + w

    full = sum(window(w)(gathered) for w in range(world))
    for rank in range(world):

        def fake_allgather(t, sizes, group, process_group):
            assert sizes is None and group == list(range(world)) and process_group is None
            torch.testing.assert_close(t, pad(xs[rank]), rtol=0, atol=0)
            return gathered

        def fake_reducescatter(t, sizes, group, process_group):
            torch.testing.assert_close(t, window(rank)(gathered), rtol=0, atol=0)
            return full[rank * rows : (rank + 1) * rows]

        monkeypatch.setattr(M, "allgather", fake_allgather)
        monkeypatch.setattr(M, "reducescatter", fake_reducescatter)
        monkeypatch.setattr(M, "mpi_disabled", lambda: False)
        core = SimpleNamespace(dp_size=world, dp_group=list(range(world)), mapping=None)
        expert_sum = M.Qwen4ExpUnifiedModelingV2._expert_sum_op(core)
        out = expert_sum(window(rank), xs[rank], rows - counts[rank])
        torch.testing.assert_close(out, sum(window(w)(xs[rank]) for w in range(world)))


# --- MTP: the draft layer ------------------------------------------------------------

HIDDEN = 2560
NUM_EXPERTS = 512
TOPK = 10


def _mtp(max_draft_len: int = 3) -> MTPDecodingConfig:
    return MTPDecodingConfig(max_draft_len=max_draft_len)


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_mtp_declares_the_draft_layer_after_the_trunk_at_the_trunk_window(member) -> None:
    """With speculation the MTP module is declared at global layer 48, its experts at the
    trunk's window; without it, nothing of the module is declared."""
    core = _core(member, 1, MTPDecodingConfig(max_draft_len=3))
    assert core.mtp_enabled and core.max_verify_rows == 4 and core.mtp_layer_idx == NUM_LAYERS
    trunk, draft = (W.expert_window(core.w, core.dims, i) for i in (0, NUM_LAYERS))
    assert (draft.offset, draft.count) == (trunk.offset, trunk.count)
    assert draft.fc1_w.shape == trunk.fc1_w.shape
    plain = _core(member, 1)
    assert not plain.mtp_enabled and plain.spec_rollback is None
    assert not any("_mtp_" in k for k in plain.w)


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_draft_len", [1, 7])
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_mtp_widens_the_verify_step_to_the_draft_length(member, max_draft_len) -> None:
    """A verify step carries `max_draft_len + 1` rows a generation request: two at the
    shortest draft, at the longest the eight the GDN cached replay and the pools are sized
    for. The draft layer is global layer 48 at the trunk's rope base, and the rollback handler
    exists exactly when MTP does."""
    core = _core(member, 0, _mtp(max_draft_len))
    assert core.mtp_enabled and core.max_verify_rows == max_draft_len + 1
    assert core.max_verify_rows <= M._MAX_VERIFY_WIDTH
    assert core.mtp_layer_idx == NUM_LAYERS
    assert core.mtp_theta == core.theta == 1e7
    assert core.spec_rollback is not None
    assert any(k.startswith(f"l{NUM_LAYERS}_mtp_") for k in core.w)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank", [(m, r) for m, r in HEAD_RANKS if m not in ("tp1", "dep2")]
)
def test_each_rank_holds_its_share_of_the_draft_attention(member, rank) -> None:
    """The draft layer's attention is cut like a trunk attention layer: under attention TP this
    rank's gated q heads, its KV head and its o_proj columns; under attention DP every head and
    the whole o_proj. Its indexer and router are replicated."""
    core = _core(member, rank, _mtp())
    hq, hkv, *_, out_k = _GEOMETRY[member]
    p = f"l{NUM_LAYERS}_mtp_"
    w = core.w
    assert tuple(w[f"{p}q_w"].shape) == (2 * hq * 256, HIDDEN)
    assert tuple(w[f"{p}k_w"].shape) == tuple(w[f"{p}v_w"].shape) == (hkv * 256, HIDDEN)
    assert tuple(w[f"{p}o_w"].shape) == (HIDDEN, out_k)
    assert tuple(w[f"{p}idx_w"].shape) == (640, HIDDEN)
    assert tuple(w[f"{p}router"].shape) == (NUM_EXPERTS, HIDDEN)


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tp4", "tep4", "dep4", "tep2"])
def test_the_draft_expert_windows_cover_the_routing_space_once(member) -> None:
    """Each rank's draft experts are the trunk's window at the whole 640-row intermediate, in
    the FP8 runner's form; the group's windows cover the 512 experts exactly once."""
    world = _world(member)
    covered = []
    for rank in range(world):
        core = _core(member, rank, _mtp())
        draft = W.expert_window(core.w, core.dims, NUM_LAYERS)
        assert (draft.offset, draft.count) == (core.expert_offset, core.local_experts)
        assert draft.fc1_w.dtype == draft.fc2_w.dtype == torch.float8_e4m3fn
        assert tuple(draft.fc1_w.shape) == (512 // world, 2 * 640, HIDDEN)
        assert tuple(draft.fc2_s.shape) == (512 // world, HIDDEN // 128, 640 // 128)
        covered += range(draft.offset, draft.offset + draft.count)
    assert sorted(covered) == list(range(NUM_EXPERTS))


@pytest.mark.cpu_only
def test_a_draft_step_pads_by_the_counts_the_worker_hands_it() -> None:
    """The spec worker hands each draft step the group's row counts -- every rank's trunk
    tokens on the loop's first step, its request count on the later ones -- and does not write
    them into the metadata. A count that disagrees with this rank's rows would gather the wrong
    split, and a missing one cannot pad at all; without attention DP nothing is padded."""
    core = _core("dep4", 2, _mtp())
    assert M._draft_pad_rows(core, [5, 17, 9, 1], 9) == 8
    assert M._draft_pad_rows(core, [2, 4, 3, 1], 3) == 1
    assert M._draft_pad_rows(core, [4, 4, 4, 4], 4) == 0
    with pytest.raises(AssertionError, match="wrong split"):
        M._draft_pad_rows(core, [5, 17, 9, 1], 10)
    with pytest.raises(AssertionError, match="without the group's row counts"):
        M._draft_pad_rows(core, None, 9)
    assert M._draft_pad_rows(_core("tep4", 0, _mtp()), None, 7) == 0


def _pad(x: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat([x, x.new_zeros(rows - x.shape[0], x.shape[1])], 0)


def _expert_out(x, ids, wts, start, count):
    """A stand-in MoE whose output names the experts a row routes to inside `[start, start +
    count)`: windows of whole experts sum to the all-expert output."""
    keep = ((ids >= start) & (ids < start + count)).to(x.dtype)
    return (wts * keep * (ids + 1).to(x.dtype)).sum(-1, keepdim=True) * x


class _DraftMoe:
    """Stand-ins for the draft MoE's ops, in float64, over a fixed router: the routing is the
    top-10 softmax of the logits, the runner `_expert_out` over the window it is handed."""

    def __init__(self, monkeypatch, chunk_rows: int):
        gen = torch.Generator().manual_seed(0)
        self.router = torch.randn(HIDDEN, NUM_EXPERTS, generator=gen, dtype=torch.float64)
        self.chunk_rows = chunk_rows
        self.runner_rows = 0
        self.windows = []
        monkeypatch.setattr(M, "_MOE_MAX_ROWS", chunk_rows)
        monkeypatch.setattr(M, "cublas_mm", lambda a, b, *args, **kwargs: a @ b)
        monkeypatch.setattr(M, "renorm_moe_routing_op", self.routing)
        monkeypatch.setattr(M, "fp8_quantize_1x128", lambda t, use_ue8m0: (t, None))
        monkeypatch.setattr(M, "fp8_block_scale_moe_runner", self.runner)
        monkeypatch.setattr(M, "silu_and_mul_rows", lambda gug, inter: gug[:, :inter])
        # The shared expert and its gate are replicated and checked elsewhere.
        monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda routed, gate, shared: routed)

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
        assert 0 < x.shape[0] <= self.chunk_rows
        self.runner_rows += x.shape[0]
        start, count = kw["local_expert_offset"], kw["local_num_experts"]
        self.windows.append((start, count))
        return _expert_out(x, kw["topk_ids"], kw["topk_weights"], start, count)

    def views(self, core):
        return SimpleNamespace(
            router=self.router,
            experts=W.expert_window(core.w, core.dims, NUM_LAYERS),
            sh_gug=torch.zeros(HIDDEN, 2 * 640 + 8, dtype=torch.float64),
            sh_dn=torch.zeros(640, HIDDEN, dtype=torch.float64),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "counts",
    [[3, 7, 1, 4], [1, 2, 1, 3], [4, 4, 4, 4]],
    ids=["draft-extend", "draft-decode-with-a-dummy-rank", "captured-batch"],
)
def test_every_dep4_rank_completes_its_draft_rows_through_the_round_trip(
    counts, monkeypatch
) -> None:
    """At every draft step every rank pads its rows to the group's largest count, gathers the
    group's rows (one allgather), routes, quantizes and runs its window over all of them in
    chunks, and reduce-scatters (one reduce-scatter): the same two collectives at the same shape
    on every rank, a rank holding one dummy request included. Each rank gets back the
    all-expert MoE of its own rows; nothing goes through the TP sum."""
    world = 4
    moe = _DraftMoe(monkeypatch, chunk_rows=5)
    gen = torch.Generator().manual_seed(1)
    xs = [torch.randn(n, HIDDEN, generator=gen, dtype=torch.float64) for n in counts]
    rows = max(counts)
    gathered = torch.cat([_pad(x, rows) for x in xs], 0)
    full = _expert_out(gathered, *moe.route(gathered), 0, NUM_EXPERTS)
    state = {}

    def fake_allgather(t, sizes, group, process_group):
        assert sizes is None and group == list(range(world)) and process_group is None
        state["collectives"].append(("allgather", tuple(t.shape)))
        torch.testing.assert_close(t, _pad(xs[state["rank"]], rows), rtol=0, atol=0)
        return gathered

    def fake_reducescatter(t, sizes, group, process_group):
        assert sizes is None and group == list(range(world)) and process_group is None
        state["collectives"].append(("reducescatter", tuple(t.shape)))
        state["partial"] = t
        r = state["rank"]
        return full[r * rows : (r + 1) * rows]

    monkeypatch.setattr(M, "allgather", fake_allgather)
    monkeypatch.setattr(M, "reducescatter", fake_reducescatter)
    monkeypatch.setattr(M, "mpi_disabled", lambda: False)

    per_rank = []
    for rank in range(world):
        core = _core("dep4", rank, _mtp())
        core.expert_sum = core._expert_sum_op()

        def no_tp_sum(t):
            raise AssertionError("the draft MoE under attention DP went through the TP sum")

        core.tp_sum = no_tp_sum
        state.update(rank=rank, collectives=[])
        moe.runner_rows, moe.windows = 0, []
        pad_rows = M._draft_pad_rows(core, counts, counts[rank])
        out = M._mtp_moe_block(core, moe.views(core), xs[rank], pad_rows)
        assert moe.runner_rows == world * rows
        assert set(moe.windows) == {(128 * rank, 128)}
        torch.testing.assert_close(
            state["partial"], _expert_out(gathered, *moe.route(gathered), 128 * rank, 128)
        )
        assert tuple(out.shape) == (counts[rank], HIDDEN)
        torch.testing.assert_close(out, _expert_out(xs[rank], *moe.route(xs[rank]), 0, 512))
        per_rank.append(state["collectives"])
    assert all(c == per_rank[0] for c in per_rank), per_rank
    assert per_rank[0] == [
        ("allgather", (rows, HIDDEN)),
        ("reducescatter", (world * rows, HIDDEN)),
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tp4", "tep4"])
def test_the_draft_windows_sum_to_the_whole_draft_moe_under_attention_tp(
    member, monkeypatch
) -> None:
    """Under attention TP every rank runs its window over the same rows and the TP sum
    completes the draft MoE: the windows' partials add to the all-expert output, at a TP rank's
    verify rows (128 requests of MTP7's 8 rows) in several chunks."""
    world = _world(member)
    moe = _DraftMoe(monkeypatch, chunk_rows=300)
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1024, HIDDEN, generator=gen, dtype=torch.float64)
    partials = []
    for rank in range(world):
        core = _core(member, rank, _mtp(7))
        core.tp_sum = lambda t: partials.append(t) or t
        core.expert_sum = core._expert_sum_op()
        moe.windows = []
        M._mtp_moe_block(core, moe.views(core), x, M._draft_pad_rows(core, None, x.shape[0]))
        assert set(moe.windows) == {(rank * 512 // world, 512 // world)}
    assert len(partials) == world
    torch.testing.assert_close(sum(partials), _expert_out(x, *moe.route(x), 0, NUM_EXPERTS))


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", [M.PrefillTarget, M.DecodeTarget])
def test_both_draft_targets_pad_the_draft_moe_by_the_workers_counts(target) -> None:
    """A draft step runs on the prefill target while the metadata still carries context rows
    and on the decode target otherwise; ranks of one attention-DP group can be on either at the
    same draft step. Both pad the draft MoE by the `all_rank_num_tokens` the worker passes them
    for that step, never by the metadata's, which still holds the trunk step's counts."""
    fn = ast.parse(inspect.getsource(target.mtp_forward).lstrip())
    (pad,) = _calls(fn, "_draft_pad_rows")
    assert [ast.unparse(a) for a in pad.args] == ["core", "all_rank_num_tokens", "rows"]
    (tail,) = _calls(fn, "_mtp_tail")
    assert ast.unparse(tail.args[-1]) == "pad_rows"
    assert "_dp_pad_rows" not in ast.unparse(fn)


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


def _tp4(r, kv_head):
    return dict(heads_kv=1, kv_head_start=kv_head, num_k_heads=4, k_head_start=4 * r,
                num_v_heads=12, v_head_start=12 * r, conv_dim=2560)  # fmt: skip


#: What the target binds on each member and rank: KV heads and the first one's global index, GDN
#: key heads and the first, GDN value heads and the first, and the conv channels ([q | k | v]).
#: One rank (tp1) and attention DP hold every head; TP2 one KV head and 8 / 24 GDN heads a rank;
#: TP4 one KV head, repeated on two neighbouring ranks, and 4 / 12 GDN heads a rank.
LAYOUT = {
    ("tp1", 0): _ALL,
    ("tp2", 0): _tp2(0),
    ("tp2", 1): _tp2(1),
    ("tep2", 0): _tp2(0),
    ("tep2", 1): _tp2(1),
    ("dep2", 0): _ALL,
    ("dep2", 1): _ALL,
    ("tp4", 0): _tp4(0, 0),
    ("tp4", 1): _tp4(1, 0),
    ("tp4", 2): _tp4(2, 1),
    ("tp4", 3): _tp4(3, 1),
    ("tep4", 0): _tp4(0, 0),
    ("tep4", 1): _tp4(1, 0),
    ("tep4", 2): _tp4(2, 1),
    ("tep4", 3): _tp4(3, 1),
    ("dep4", 0): _ALL,
    ("dep4", 1): _ALL,
    ("dep4", 2): _ALL,
    ("dep4", 3): _ALL,
}

#: The same on every member and rank: the page size, the head dims, the replicated QSA index and
#: PLE pools, and the activation dtype the conv, index and PLE pools hold.
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

#: The K/V storage per resolved KV-cache algorithm: (pool dtype, innermost extent, whether the
#: QSA index and position pages are packed into one side page). A 32-token bf16 index page is
#: 8192 bytes: an FP8 K page of one 256-wide head, an NVFP4 one of two 128-byte heads; never a
#: bf16 K/V page (16384 bytes a head).
KV_STORAGE = {
    None: (torch.bfloat16, 256, False),
    QuantAlgo.FP8: (torch.float8_e4m3fn, 256, True),
    QuantAlgo.NVFP4: (torch.uint8, 128, True),
}
KV_IDS = {None: "kv_bf16", QuantAlgo.FP8: "kv_fp8", QuantAlgo.NVFP4: "kv_nvfp4"}
_ALGOS = list(KV_STORAGE)
_ALGO_IDS = list(KV_IDS.values())


def _geometry(member: str, rank: int, kv_cache_quant_algo=None) -> P.PoolGeometry:
    return P.pool_geometry(
        Qwen4ExpTextConfig(**_TEXT),
        _mapping(member, rank),
        tokens_per_block=TOKENS_PER_BLOCK,
        kv_cache_quant_algo=kv_cache_quant_algo,
        dtype=torch.bfloat16,
    )


@pytest.mark.cpu_only
def test_the_text_config_is_the_checkpoints():
    """Every value of `_TEXT` is the checkpoint's own, read off its `config.json`."""
    root = os.environ.get("LLM_MODELS_ROOT")
    path = None if root is None else Path(root) / CHECKPOINT / "config.json"
    if path is None or not path.is_file():
        pytest.skip(f"{CHECKPOINT} is not under LLM_MODELS_ROOT")
    text = json.loads(path.read_text())["text_config"]
    differ = {
        k: (v, text.get(k)) for k, v in _TEXT.items() if k != "architectures" and text.get(k) != v
    }
    assert not differ, f"(this test, the checkpoint) differ: {differ}"


#: Every rank under bf16 K/V, and each quantized K/V storage at one rank: the storage sets the
#: pool's dtype, width and side-page packing alone, the rank its heads alone.
_GEOMETRY_CASES = [(m, r, None) for m, r in _RANKS] + [
    ("dep4", 3, QuantAlgo.FP8),
    ("tp4", 2, QuantAlgo.NVFP4),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank, algo",
    _GEOMETRY_CASES,
    ids=[f"{m}-rank{r}-{KV_IDS[a]}" for m, r, a in _GEOMETRY_CASES],
)
def test_the_geometry_is_the_layout_the_target_binds(member, rank, algo):
    g = _geometry(member, rank, algo)
    kv_dtype, kv_width, side_planes = KV_STORAGE[algo]
    expected = dict(
        COMMON,
        **LAYOUT[member, rank],
        kv_dtype=kv_dtype,
        kv_width=kv_width,
        qsa_side_planes=side_planes,
    )
    assert g._asdict() == expected
    assert g.conv_sections == (g.num_k_heads * 128, g.num_k_heads * 128, g.num_v_heads * 128)
    assert g.kv_block_scales == (algo == QuantAlgo.NVFP4)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [16, 128])
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
def test_side_plane_packing_does_not_depend_on_the_page_size(algo, tokens_per_block):
    """Every page scales with the page size, so the packing choice is the K/V storage's alone."""
    g = P.pool_geometry(
        Qwen4ExpTextConfig(**_TEXT),
        _mapping("tep2", 0),
        tokens_per_block=tokens_per_block,
        kv_cache_quant_algo=algo,
        dtype=torch.bfloat16,
    )
    assert g.qsa_side_planes == KV_STORAGE[algo][2]


@pytest.mark.cpu_only
def test_another_kv_algorithm_is_refused():
    with pytest.raises(ValueError, match="kv_cache_quant_algo"):
        _geometry("dep4", 0, QuantAlgo.INT8)


@pytest.mark.cpu_only
def test_a_pipeline_or_context_parallel_layout_is_refused():
    with pytest.raises(ValueError, match="pp_size"):
        P.pool_geometry(
            Qwen4ExpTextConfig(**_TEXT),
            Mapping(rank=0, gpus_per_node=4, world_size=4, tp_size=2, pp_size=2),
            tokens_per_block=TOKENS_PER_BLOCK,
            kv_cache_quant_algo=None,
            dtype=torch.bfloat16,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", HEAD_RANKS, ids=_HEAD_RANK_IDS)
def test_a_rank_holds_the_kv_head_the_weight_split_gives_it(member, rank):
    """With more attention-TP ranks than KV heads the stock loader repeats each head
    (`duplicate_kv_weight`) before the even row split, so the K/V a rank writes into its pages
    is that of global head `kv_head_start`; at one rank or under attention DP a rank holds all of
    them."""
    g = _geometry(member, rank)
    heads, rows_per_head = 2, 4
    weight = torch.arange(heads * rows_per_head, dtype=torch.float32).reshape(-1, 1)
    tp = P.attention_tp_size(_mapping(member, rank))
    local = duplicate_kv_weight(weight, heads, tp).chunk(tp)[rank if tp > 1 else 0]
    start, stop = g.kv_head_start * rows_per_head, (g.kv_head_start + g.heads_kv) * rows_per_head
    assert torch.equal(local, weight[start:stop])


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tp1", "tp2", "tp4", "dep4"])
def test_a_members_ranks_hold_every_gdn_head_once(member):
    """Under attention TP the ranks' GDN head ranges tile the heads; one rank, or every rank
    under attention DP, holds them all."""
    world = MEMBERS[member]["world_size"]
    gs = [_geometry(member, rank) for rank in range(world)]
    for count, start, total in (
        ("num_k_heads", "k_head_start", 16),
        ("num_v_heads", "v_head_start", 48),
    ):
        ranges = sorted({(getattr(g, start), getattr(g, count)) for g in gs})
        covered = [h for first, n in ranges for h in range(first, first + n)]
        assert covered == list(range(total)), (member, count, ranges)


# ------------------------------------------------------------ engine facts ---


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_draft_len", [None, 1, 3, 7], ids=["no-spec", "mtp1", "mtp3", "mtp7"])
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_the_reserved_dummy_slots_are_the_engines(member, max_draft_len):
    """`MambaHybridCacheManagerV2` reserves the CUDA-graph padding dummies of every draft length
    it can run, plus the attention-DP dummy request's slot under attention DP."""
    spec = None if max_draft_len is None else MTPDecodingConfig(max_draft_len=max_draft_len)
    mapping = _mapping(member, 0)
    engine = _get_num_cuda_graph_padding_dummy_slots(spec, 128) + int(mapping.enable_attention_dp)
    assert P.reserved_dummy_slots(mapping) == engine


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_the_draft_kv_cache_is_separate_exactly_under_attention_tp(member):
    """One-engine MTP gets a separate draft KV cache (`should_use_separate_draft_kv_cache`) unless
    attention DP folds the draft layer into the trunk's manager."""
    mapping = _mapping(member, 0)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    engine = not mapping.enable_attention_dp and should_use_separate_draft_kv_cache(spec)
    assert P.draft_kv_cache_is_separate(mapping) == engine


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_a_disaggregated_engine_keeps_the_draft_layer_in_the_trunks_manager(member):
    """A cache transceiver makes the executor disallow a separate draft KV cache
    (`_allow_separate_draft_kv_cache`) at every layout, so the pools it moves carry the draft
    layer: the target's answer is the engine's, attention TP included."""
    mapping = _mapping(member, 0)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    assert P.draft_kv_cache_is_separate(mapping, spec) == P.draft_kv_cache_is_separate(mapping)
    spec._allow_separate_draft_kv_cache = False
    assert not should_use_separate_draft_kv_cache(spec)
    assert not P.draft_kv_cache_is_separate(mapping, spec)


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 2, 8])
def test_every_verify_width_up_to_seven_drafts_is_admitted(rows):
    """A plain decode row, the shortest verify (one draft) and the widest (seven)."""
    P.check_verify_width(rows)
    if rows >= 2:
        slots = 8
        bookkeeping = torch.zeros(slots, dtype=torch.int32, device="meta")
        ssm = torch.empty(slots, 48, 128, 128, dtype=torch.bfloat16, device="meta")
        P.check_replay_state(P.GDN_REPLAY_HISTORY, rows, bookkeeping, bookkeeping, ssm)


@pytest.mark.cpu_only
def test_a_verify_wider_than_eight_or_an_fp32_replay_is_refused():
    with pytest.raises(AssertionError, match="predicted_tokens_per_seq 9"):
        P.check_verify_width(9)
    slots = 8
    bookkeeping = torch.zeros(slots, dtype=torch.int32, device="meta")
    bf16 = torch.empty(slots, 48, 128, 128, dtype=torch.bfloat16, device="meta")
    with pytest.raises(AssertionError, match="replay_step_width 9"):
        P.check_replay_state(P.GDN_REPLAY_HISTORY, 9, bookkeeping, bookkeeping, bf16)
    with pytest.raises(AssertionError, match="replay_history_size 8"):
        P.check_replay_state(8, 4, bookkeeping, bookkeeping, bf16)
    fp32 = bf16.to(torch.float32)
    with pytest.raises(AssertionError, match="bf16 checkpoint"):
        P.check_replay_state(P.GDN_REPLAY_HISTORY, 4, bookkeeping, bookkeeping, fp32)
    with pytest.raises(AssertionError, match="step width 4"):
        P.check_replay_width(type("Replay", (), {"width": 4})(), 1)


# --------------------------------------------- the stock manager, on meta ---


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


class _MetaManager:
    """The stock manager's views on the meta device, sized by a geometry: `get_buffers`'s
    `[pages, 2, H, block, D]` HND role pair, the NVFP4 block-scale pair holding as many pages a
    slot, the QSA planes as their own roles or as slices of one packed side page, and the
    recurrent and PLE pools over one slot space. `layers` are the attention layers it maps;
    `gdn_layers` none for a separate draft manager. Keyword arguments plant one wrong pool."""

    PAGES = 64

    def __init__(
        self,
        g: P.PoolGeometry,
        mapping: Mapping,
        layers,
        gdn_layers=GDN_LAYERS,
        *,
        slots=None,
        ssm_dtype=torch.bfloat16,
        section_dims=None,
        scale_pages_per_slot=1,
        packed=None,
    ):
        self.g, self.mapping, self.max_batch_size = g, mapping, MAX_BATCH_SIZE
        self.attn_layers, self.gdn_layers = list(layers), list(gdn_layers)
        mapped = sorted([*self.attn_layers, *self.gdn_layers])
        self.layer_offsets = {layer: local for local, layer in enumerate(mapped)}
        self.conv_section_dims = list(section_dims or g.conv_sections)
        self.slots = MAX_BATCH_SIZE + P.reserved_dummy_slots(mapping) if slots is None else slots
        self.ssm_dtype, self.scale_pages_per_slot = ssm_dtype, scale_pages_per_slot
        packed = g.qsa_side_planes if packed is None else packed
        tpb = g.tokens_per_block
        if packed:
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
        both = _meta([self.PAGES, 2 * self.scale_pages_per_slot, *page], torch.uint8)
        return both[:, :2]

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


@pytest.mark.cpu_only
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
@pytest.mark.parametrize("member, rank", LAYOUT_RANKS, ids=_LAYOUT_RANK_IDS)
def test_take_pools_binds_the_managers_layout(member, rank, algo):
    g, mapping = _geometry(member, rank, algo), _mapping(member, rank)
    cm = _MetaManager(g, mapping, ATTN_LAYERS)
    pools = P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)
    assert sorted(pools.kv) == ATTN_LAYERS and sorted(pools.conv) == GDN_LAYERS
    first = pools.kv[ATTN_LAYERS[0]]
    assert tuple(first.k.shape[1:]) == (g.heads_kv, TOKENS_PER_BLOCK, g.kv_width)
    assert (first.k_scale is not None) == (algo == QuantAlgo.NVFP4)


def _refused(cm, g, match):
    with pytest.raises(AssertionError, match=match):
        P.take_pools(cm, g, ATTN_LAYERS, GDN_LAYERS, PLE_LAYER)


@pytest.mark.cpu_only
def test_an_fp32_recurrent_state_is_refused():
    g, mapping = _geometry("tep2", 0), _mapping("tep2", 0)
    _refused(_MetaManager(g, mapping, ATTN_LAYERS, ssm_dtype=torch.float32), g, "serves")


@pytest.mark.cpu_only
def test_another_members_heads_are_refused():
    """tp1's manager holds both KV heads and every GDN head; a tep2 rank binds one and a half."""
    cm = _MetaManager(_geometry("tp1", 0), _mapping("tp1", 0), ATTN_LAYERS)
    _refused(cm, _geometry("tep2", 0), "paged K/V")


@pytest.mark.cpu_only
def test_an_x_b_c_conv_layout_is_refused():
    g, mapping = _geometry("tp2", 1), _mapping("tp2", 1)
    q, k, v = g.conv_sections
    _refused(_MetaManager(g, mapping, ATTN_LAYERS, section_dims=(v, q, k)), g, "q_k_v")


@pytest.mark.cpu_only
def test_nvfp4_scales_at_another_page_count_are_refused():
    g, mapping = _geometry("dep4", 0, QuantAlgo.NVFP4), _mapping("dep4", 0)
    _refused(_MetaManager(g, mapping, ATTN_LAYERS, scale_pages_per_slot=2), g, "pages per slot")


@pytest.mark.cpu_only
def test_unpacked_side_planes_under_an_fp8_pool_are_refused():
    g, mapping = _geometry("tep4", 3, QuantAlgo.FP8), _mapping("tep4", 3)
    _refused(_MetaManager(g, mapping, ATTN_LAYERS, packed=False), g, "one per-slot page")


@pytest.mark.cpu_only
def test_a_pool_short_of_the_reserved_dummies_is_refused():
    """Under attention DP the attention-DP dummy request needs one slot more than attention TP:
    a dep2 pool sized for attention TP's dummies is refused."""
    g, mapping = _geometry("dep2", 0), _mapping("dep2", 0)
    short = MAX_BATCH_SIZE + P.reserved_dummy_slots(mapping) - 1
    assert short == MAX_BATCH_SIZE + P.reserved_dummy_slots(_mapping("tep2", 0))
    _refused(_MetaManager(g, mapping, ATTN_LAYERS, slots=short), g, "reserved dummies")


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", LAYOUT_RANKS, ids=_LAYOUT_RANK_IDS)
def test_the_replay_history_and_step_buffers_are_this_ranks_heads(member, rank):
    g, slots, history = _geometry(member, rank), 8, P.GDN_REPLAY_HISTORY
    conv = _meta([slots, g.conv_dim, g.conv_kernel - 1], torch.bfloat16)
    int_conv = _meta([slots, 8, g.conv_dim, g.conv_kernel - 1], torch.bfloat16)
    P.check_step_buffers(g, int_conv, conv, torch.zeros(slots, dtype=torch.int32, device="meta"))
    old_x = _meta([slots, 2, history, g.num_v_heads, g.head_v_dim], torch.bfloat16)
    old_b = _meta([slots, 2, history, g.num_k_heads, g.head_k_dim], torch.bfloat16)
    old_dt = _meta([slots, 2, g.num_v_heads, history], torch.float32)
    P.check_replay_history(g, GDN_LAYERS[0], history, slots, old_x, old_b, old_dt)
    other = _geometry("tp1" if g.num_v_heads != 48 else "tep4", 0)
    with pytest.raises(AssertionError, match="old_x"):
        P.check_replay_history(other, GDN_LAYERS[0], history, slots, old_x, old_b, old_dt)


@pytest.mark.cpu_only
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
@pytest.mark.parametrize("member, rank", LAYOUT_RANKS, ids=_LAYOUT_RANK_IDS)
def test_take_draft_pools_reads_the_manager_the_draft_step_is_handed(member, rank, algo):
    """Under attention TP a separate manager mapping the draft layer alone; under attention DP the
    trunk's, mapping it after the trunk's layers. Either way at the trunk's per-rank shapes."""
    g, mapping = _geometry(member, rank, algo), _mapping(member, rank)
    separate = P.draft_kv_cache_is_separate(mapping)
    draft_alone = _MetaManager(g, mapping, [MTP_LAYER], [])
    trunk_and_draft = _MetaManager(g, mapping, [*ATTN_LAYERS, MTP_LAYER])
    cm, wrong = (draft_alone, trunk_and_draft) if separate else (trunk_and_draft, draft_alone)
    draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=separate)
    assert draft.pool_idx == cm.layer_offsets[MTP_LAYER]
    assert tuple(draft.kv.v.shape[1:]) == (g.heads_kv, TOKENS_PER_BLOCK, g.kv_width)
    with pytest.raises(AssertionError, match="draft layer"):
        P.take_draft_pools(wrong, g, MTP_LAYER, separate=separate)
    with pytest.raises(AssertionError, match="does not map"):
        P.take_draft_pools(_MetaManager(g, mapping, ATTN_LAYERS), g, MTP_LAYER, separate=separate)


@pytest.mark.cpu_only
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
@pytest.mark.parametrize("member, rank", LAYOUT_RANKS, ids=_LAYOUT_RANK_IDS)
def test_a_disaggregated_draft_step_reads_the_trunks_manager(member, rank, algo):
    """In a disaggregated engine the draft step is handed the trunk's manager at every layout
    (attention TP too): the draft layer's views at the trunk's per-rank shapes, and a manager
    holding the draft layer alone is refused."""
    g, mapping = _geometry(member, rank, algo), _mapping(member, rank)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    spec._allow_separate_draft_kv_cache = False
    separate = P.draft_kv_cache_is_separate(mapping, spec)
    assert not separate
    trunk_and_draft = _MetaManager(g, mapping, [*ATTN_LAYERS, MTP_LAYER])
    draft = P.take_draft_pools(trunk_and_draft, g, MTP_LAYER, separate=separate)
    assert draft.pool_idx == trunk_and_draft.layer_offsets[MTP_LAYER]
    assert tuple(draft.kv.v.shape[1:]) == (g.heads_kv, TOKENS_PER_BLOCK, g.kv_width)
    with pytest.raises(AssertionError, match="trunk's layers too"):
        P.take_draft_pools(
            _MetaManager(g, mapping, [MTP_LAYER], []), g, MTP_LAYER, separate=separate
        )


# ------------------------------------------------ the stock manager, on GPU ---


def _manager_model_config(mapping, kv_cache_quant_algo, spec_config):
    # Imported here: the stock factory pulls in the executor, which the cpu_only cases skip.
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_mamba_ssm_cache_dtype

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
    return model_config


@contextlib.contextmanager
def _manager(member: str, rank: int, kv_cache_quant_algo=None, spec_config=None, is_draft=False):
    """The trunk manager the engine builds on `rank` of `member` -- or with `is_draft` the
    separate one-model draft KV cache -- through the engine's own class choice and factory."""
    from tensorrt_llm._torch.pyexecutor._util import (
        _create_kv_cache_manager,
        get_kv_cache_manager_cls,
    )

    mapping = _mapping(member, rank)
    model_config = _manager_model_config(mapping, kv_cache_quant_algo, spec_config)
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


@_needs_gpu
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
@pytest.mark.parametrize("member, rank", LAYOUT_RANKS, ids=_LAYOUT_RANK_IDS)
def test_the_stock_manager_serves_the_layout(member, rank, algo):
    """Every trunk pool view of the stock manager passes the contract at this rank's geometry,
    and the manager's own per-rank numbers are the geometry's."""
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
        # The packing the geometry predicts is the manager's choice.
        assert (cm.qsa_side_planes_size is not None) == g.qsa_side_planes
        # The dummies the contract reserves are the manager's.
        assert cm._num_reserved_dummy_slots == P.reserved_dummy_slots(mapping)
        # Each attention layer has a pool offset of its own.
        assert len(set(pools.pool_idx.values())) == len(ATTN_LAYERS)
        # One slot space for every QSA view of the layers, and one for the recurrent and PLE
        # state.
        slots = {view.shape[0] for view in (pools.position_cache, *pools.index_cache.values())}
        slots |= {planes.k.shape[0] for planes in pools.kv.values()}
        assert len(slots) == 1
        assert {t.shape[0] for t in (*pools.conv.values(), *pools.ssm.values())} == {
            pools.ple_state[0].shape[0]
        }
        # No draft layer without a speculative configuration.
        assert MTP_LAYER not in cm.layer_offsets


#: Each attention layout's member at three drafts (MTP is not admitted at dep2), and the
#: shortest and the widest verify (one draft, seven: eight rows) under attention TP and DP.
_MTP_CASES = [(member, rank, 3) for member, rank in LAYOUT_RANKS] + [
    (member, rank, draft) for member, rank in (("tp4", 3), ("dep4", 3)) for draft in (1, 7)
]
_MTP_CASE_IDS = [f"{member}-rank{rank}-mtp{draft}" for member, rank, draft in _MTP_CASES]


@_needs_gpu
@pytest.mark.parametrize("member, rank, max_draft_len", _MTP_CASES, ids=_MTP_CASE_IDS)
def test_the_mtp_verify_buffers_hold_this_ranks_heads(monkeypatch, member, rank, max_draft_len):
    """With MTP (the GDN cached replay) the verify buffers -- the per-step conv windows and every
    GDN layer's replay history -- are this rank's heads, at the step width `max_draft_len + 1`.
    Under attention TP the draft layer is not the trunk manager's; under attention DP it is, with
    the trunk's per-rank pool shapes."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
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
        assert meta is not None
        P.check_replay_state(
            meta.replay_history_size,
            meta.replay_step_width,
            meta.prev_num_accepted_tokens,
            meta.cache_buf_idx,
            ssm0,
        )
        assert meta.replay_step_width == max_draft_len + 1
        P.check_verify_width(meta.replay_step_width)
        history, slots = meta.replay_history_size, ssm0.shape[0]
        for i, cache in caches.items():
            P.check_replay_history(g, i, history, slots, cache.old_x, cache.old_B, cache.old_dt)

        separate = P.draft_kv_cache_is_separate(mapping)
        assert (MTP_LAYER in cm.layer_offsets) == (not separate)
        if not separate:
            draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=False)
            assert draft.position_cache.data_ptr() == pools.position_cache.data_ptr()


#: The attention-TP layouts' ranks, where the draft layer has a KV cache of its own.
_ATTN_TP_RANKS = [(m, r) for m, r in LAYOUT_RANKS if m != "dep4"]


@_needs_gpu
@pytest.mark.parametrize("algo", _ALGOS, ids=_ALGO_IDS)
@pytest.mark.parametrize(
    "member, rank", _ATTN_TP_RANKS, ids=[f"{m}-rank{r}" for m, r in _ATTN_TP_RANKS]
)
def test_the_separate_draft_kv_cache_holds_the_draft_layer_alone(monkeypatch, member, rank, algo):
    """Under attention TP the engine builds the draft layer a manager of its own (`is_draft`):
    the draft layer alone, attention only, at the trunk's per-rank K/V, index and position shapes
    under the trunk's K/V storage."""
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    g = _geometry(member, rank, algo)
    spec = MTPDecodingConfig(max_draft_len=3, num_nextn_predict_layers=1)
    with _manager(member, rank, algo, spec_config=spec, is_draft=True) as cm:
        draft = P.take_draft_pools(cm, g, MTP_LAYER, separate=True)
        assert cm.ple_layer_cache(PLE_LAYER) is None
        assert cm.local_num_mamba_layers == 0
        assert cm.num_kv_heads_per_layer[draft.pool_idx] == g.heads_kv
        assert (cm.qsa_side_planes_size is not None) == g.qsa_side_planes
        assert draft.kv.k.shape[0] == draft.index_cache.shape[0] == draft.position_cache.shape[0]

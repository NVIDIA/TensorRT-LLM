# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

`route` compares the facts that fix which code runs against the deployments the
targets were written for, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). The checkpoint's published
architecture name is kept by `load_pretrained_config`, so `route` registers under it.
All criteria are read off `text_config` / `vision_config`; the composite parent's
mirrored fields are unreliable.

Qwen3.8-Flash-Next is published at two quantizations of one geometry, so `route` keys on
the quantization the engine resolves for the checkpoint (`quant_config.quant_algo`) as well
as its shape:

* Qwen3.8-Flash-Next-NVFP4 (`MIXED_PRECISION`) on sm103 lands on one of two targets by its
  parallel layout: one rank on `qwen3_8_flash_next_nvfp4__sm_103__tp1`, and tp4, tep4, dep4
  or dep2 in one node on `qwen3_8_flash_next_nvfp4__sm_103__default`.
* Qwen3.8-Flash-Next-FP8 (`FP8_BLOCK_SCALES`) on sm103 is one target,
  `qwen3_8_flash_next_fp8__sm_103__unified`, at tp1, tp2, tep2, dep2, tp4, tep4 and dep4 in
  one node (`fp8_member` names the member, or says why the deployment is none).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The GPU every target is certified on.
_SM = (10, 3)

# The quantization the engine resolves for each checkpoint (`ModelConfig.load_hf_quant_config`
# and the modelopt reader): the NVFP4 export's `hf_quant_config.json` declares MIXED_PRECISION
# (NVFP4 routed experts, an FP8 n-gram table, bf16 elsewhere); the FP8 export's
# `quantization_config` is `quant_method: fp8` with 128x128 weight blocks (FP8 routed experts,
# bf16 elsewhere), which the engine reads as FP8_BLOCK_SCALES. The two share every shape.
_NVFP4_QUANT_ALGO = QuantAlgo.MIXED_PRECISION
_FP8_QUANT_ALGO = QuantAlgo.FP8_BLOCK_SCALES

# (layers, hidden, experts, attention heads, linear value heads, vocab). Shape-sniffing
# also matches fine-tunes; the accuracy gate pins identity.
_TEXT_SHAPE = (48, 2560, 512, 24, 48, 248320)

# (depth, hidden, heads, merge, out_hidden, num_pos, deepstack_indexes, act).
# `deepstack_visual_indexes == ()` is the key member: a deepstack sibling has a merger
# list these targets lack.
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# The longest MTP draft either target runs (`_MAX_VERIFY_WIDTH - 1` in modeling.py); the
# legacy per-step recurrent states (`_MAX_LEGACY_VERIFY_WIDTH - 1`) are certified at a
# shorter one.
_MAX_DRAFT_LEN = 7
_MAX_LEGACY_DRAFT_LEN = 3

# The tp1 target: `moe_config.backend` requests it implements (`_TRUNK_MOE_BACKENDS`), the
# longest draft of its CUTEDSL experts (`_CUTEDSL_MAX_DRAFT_LEN`) and its recurrent-state
# cache dtypes (`_SSM_STATE_DTYPES`), all in its modeling.py.
_MOE_BACKENDS = ("AUTO", "TRTLLM", "CUTEDSL")
_MAX_CUTEDSL_DRAFT_LEN = 3
_SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)

# The default target's counterparts, in its modeling.py.
_DEFAULT_MOE_BACKENDS = ("AUTO", "TRTLLM")
_DEFAULT_SSM_STATE_DTYPES = (torch.bfloat16,)

# (world size, attention DP, MoE TP, MoE EP) -> the default target's layout. Every layout
# spans its whole world in one node, attention TP over all of it unless attention DP
# replicates attention. At four ranks the routed experts are split by intermediate (tp4) or
# by expert (tep4, dep4). At two ranks only attention DP with two 256-expert windows (dep2),
# where each rank's attention, Gated-DeltaNet and PLE geometry is dep4's.
_LAYOUTS = {
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (2, True, 1, 2): "dep2",
}

# The attention-DP layouts MTP is admitted at: those its draft acceptance was measured at.
_ATTENTION_DP_MTP_LAYOUTS = ("dep4",)

# The FP8 target's layouts, keyed like `_LAYOUTS`. At one rank everything is local (tp1). At two
# and four ranks attention TP over the whole group with the routed experts split by Mapping's
# default MoE TP (tp2, tp4) or by expert (tep2, tep4), or attention DP with the experts split by
# expert (dep2, dep4). The FP8 experts' 128x128 weight blocks do not take a 320- or 160-row
# intermediate share, so the target runs every layout's experts as whole-expert windows over
# the TP group; the layout only says which ranks attention and the recurrent heads span.
_FP8_LAYOUTS = {
    (1, False, 1, 1): "tp1",
    (2, False, 2, 1): "tp2",
    (2, False, 1, 2): "tep2",
    (2, True, 1, 2): "dep2",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
}

# The FP8 target's engine choices: the trtllm-gen FP8 block-scale experts, a bf16
# recurrent-state cache, and MTP under attention DP at dep4 alone (the high-throughput tune's
# layout; dep2 is the disaggregated-serving engines' layout, served without speculation).
_FP8_MOE_BACKENDS = ("AUTO", "TRTLLM")
_FP8_SSM_STATE_DTYPES = (torch.bfloat16,)
_FP8_ATTENTION_DP_MTP_LAYOUTS = ("dep4",)


class EngineChoices(NamedTuple):
    """Engine choices that fix which code a target runs but are not in the checkpoint's
    config, as the engine will resolve them."""

    torch_dtype: torch.dtype  # the dtype the engine runs the model in
    moe_backend: str  # `moe_config.backend` as requested, before AUTO resolves
    mamba_ssm_cache_dtype: torch.dtype  # the recurrent-state cache's dtype
    gdn_replay: bool  # TRTLLM_USE_GDN_REPLAY, which the engine reads for eligible MTP workloads
    prefill_cuda_graph_backend: str  # `prefill_cuda_graph_backend`: disabled, piecewise, breakable


def engine_choices(model_config: ModelConfig, llm_args: TorchLlmArgs) -> EngineChoices:
    # Imported on a claimed branch only, as the targets are.
    from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype
    from tensorrt_llm._torch.utils import is_gdn_replay_enabled

    kv_cache_config = llm_args.kv_cache_config
    return EngineChoices(
        torch_dtype=model_config.torch_dtype,
        moe_backend=model_config.moe_backend_requested,
        mamba_ssm_cache_dtype=effective_mamba_ssm_cache_dtype(
            model_config, kv_cache_config.mamba_ssm_cache_dtype
        ),
        gdn_replay=is_gdn_replay_enabled(),
        prefill_cuda_graph_backend=str(llm_args.prefill_cuda_graph_backend.value),
    )


def layout(mapping, layouts: dict = _LAYOUTS) -> Optional[str]:
    """The layout ``mapping`` describes in ``layouts``, or None: by default the NVFP4 default
    target's (``tp4``, ``tep4``, ``dep4``, ``dep2``), with `_FP8_LAYOUTS` the FP8 target's.

    Pipeline and context parallelism, an attention TP narrower than the TP group, MoE
    clusters, lm_head TP under attention DP and a group spanning nodes are other
    forwards.
    """
    if mapping.tp_size != mapping.world_size or (mapping.pp_size, mapping.cp_size) != (1, 1):
        return None
    if mapping.is_multi_node() or mapping.moe_cluster_size != 1:
        return None
    if mapping.enable_attention_dp:
        if mapping.enable_lm_head_tp_in_adp:
            return None
    elif mapping.attn_tp_size != mapping.tp_size:
        return None
    return layouts.get(
        (
            mapping.world_size,
            mapping.enable_attention_dp,
            mapping.moe_tp_size,
            mapping.moe_ep_size,
        )
    )


def _describe(mapping) -> str:
    return (
        f"world_size={mapping.world_size} (tp={mapping.tp_size}, pp={mapping.pp_size}, "
        f"cp={mapping.cp_size}, attention_dp={mapping.enable_attention_dp}, "
        f"moe_tp={mapping.moe_tp_size}, moe_ep={mapping.moe_ep_size})"
    )


def _check_engine(
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
    engine: EngineChoices,
    moe_backends: Tuple[str, ...] = _MOE_BACKENDS,
    ssm_state_dtypes: Tuple[torch.dtype, ...] = _SSM_STATE_DTYPES,
) -> None:
    """Decline what a target's constructor or first forward would refuse; the defaults are
    the tp1 target's, `_check_default_engine` passes the default target's.

    Without this ``auto`` claims the deployment and then fails mid-build instead of
    falling back to the built-in implementation.
    """
    if spec_config is not None and (
        getattr(spec_config, "draft_len_schedule", None) is not None
        or getattr(spec_config, "acceptance_rate_threshold", None) is not None
    ):
        raise NoTarget(
            "the draft length changes at runtime (draft_len_schedule, max_concurrency or "
            "acceptance_rate_threshold), but this target's verify step runs at the one "
            "width max_draft_len + 1"
        )
    if kv_cache_quant_algo not in (None, QuantAlgo.FP8, QuantAlgo.NVFP4):
        raise NoTarget(
            f"a {kv_cache_quant_algo} KV cache; the target stores K/V as the checkpoint's bf16, "
            "as FP8 or as NVFP4"
        )
    if engine.torch_dtype != torch.bfloat16:
        raise NoTarget(f"the engine runs the model in {engine.torch_dtype}; the target is bf16")
    if str(engine.moe_backend).upper() not in moe_backends:
        raise NoTarget(
            f"moe_config.backend={engine.moe_backend!r}; the target runs the trunk's routed "
            f"experts on {', '.join(moe_backends)}"
        )
    if engine.mamba_ssm_cache_dtype not in ssm_state_dtypes:
        served = " or ".join(str(dtype).removeprefix("torch.") for dtype in ssm_state_dtypes)
        raise NoTarget(
            f"a {engine.mamba_ssm_cache_dtype} recurrent-state cache; the target serves "
            f"{served} (kv_cache_config.mamba_ssm_cache_dtype)"
        )
    if engine.prefill_cuda_graph_backend == "piecewise":
        raise NoTarget(
            "prefill_cuda_graph_backend='piecewise' makes the engine torch.compile the model, "
            "which the target does not implement; 'breakable' runs the prefill step as a "
            "breakable CUDA graph"
        )
    if spec_config is None:
        return
    draft = spec_config.max_draft_len
    if not engine.gdn_replay and draft > _MAX_LEGACY_DRAFT_LEN:
        raise NoTarget(
            f"max_draft_len={draft} with TRTLLM_USE_GDN_REPLAY=0: the legacy per-step recurrent "
            f"states are certified at max_draft_len up to {_MAX_LEGACY_DRAFT_LEN}; leave "
            "TRTLLM_USE_GDN_REPLAY unset for a longer draft"
        )
    if str(engine.moe_backend).upper() == "CUTEDSL" and draft > _MAX_CUTEDSL_DRAFT_LEN:
        raise NoTarget(
            f"max_draft_len={draft} with moe_config.backend=CUTEDSL: the CuteDSL expert "
            f"pipeline is certified at max_draft_len up to {_MAX_CUTEDSL_DRAFT_LEN}"
        )


def _check_default_engine(
    spec_config: Optional[Any], kv_cache_quant_algo: Optional[Any], engine: EngineChoices
) -> None:
    _check_engine(
        spec_config, kv_cache_quant_algo, engine, _DEFAULT_MOE_BACKENDS, _DEFAULT_SSM_STATE_DTYPES
    )


def _check_attention_dp(
    member: str,
    spec_config: Optional[Any],
    engine: EngineChoices,
    mtp_layouts: Tuple[str, ...] = _ATTENTION_DP_MTP_LAYOUTS,
) -> None:
    """Decline what a target does not run under attention DP (dep4, dep2), where each rank
    serves its own requests and only the routed experts cross ranks; `mtp_layouts` are the
    attention-DP layouts the target admits MTP at (the NVFP4 default target's by default).

    MTP is admitted at dep4, where each rank drafts for its own requests and the draft layer's
    FP8 expert window runs the trunk's attention-DP round trip. Routing admits both MTP and
    the breakable prefill graph at the attention-TP layouts.
    """
    if spec_config is not None and member not in mtp_layouts:
        raise NoTarget(
            f"speculative decoding with attention DP ({member}): no draft acceptance was "
            "measured at this layout; MTP under attention DP is admitted at "
            f"{', '.join(mtp_layouts)}"
        )
    if engine.prefill_cuda_graph_backend == "breakable":
        raise NoTarget(
            f"prefill_cuda_graph_backend='breakable' with attention DP ({member}): the engine "
            "pads every rank to one prefill bucket, so a rank holding only generation requests "
            "is handed more rows than its batch, which the target's decode step refuses"
        )


def _mtp_eagle_or_none(spec_config: Optional[Any]) -> bool:
    """Whether the speculation is none, or the one draft path every target implements.

    The targets implement the one-model MTP-Eagle loop over the checkpoint's single MTP
    module, and no other draft path. A draft longer than seven needs more verify rows per
    request than the GDN cached replay and thop_attention are certified at. The draft must
    also be a linear chain over that one module; the targets assert each of these at
    construction, so declining here is what lets ``auto`` fall back instead of crashing.
    """
    return spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and getattr(spec_config, "num_nextn_predict_layers", None) in (None, 1)
        and 1 <= spec_config.max_draft_len <= _MAX_DRAFT_LEN
        and not getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "eagle_choices", None) is None
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )


def _describe_spec(spec_config: Optional[Any]) -> str:
    if spec_config is None:
        return "speculative decoding off"
    return (
        f"speculative decoding {spec_config.spec_dec_mode.name} "
        f"max_draft_len={spec_config.max_draft_len}"
    )


def _check_fp8_engine(
    spec_config: Optional[Any], kv_cache_quant_algo: Optional[Any], engine: EngineChoices
) -> None:
    """Decline what the FP8 target's constructor or first forward would refuse.

    The engine choices of `_check_engine` at the FP8 target's MoE backends and recurrent-state
    dtypes, and before them a verify step without the GDN cached replay: with
    `TRTLLM_USE_GDN_REPLAY=0` the manager allocates the legacy per-step recurrent states, a
    verify path the FP8 target does not assemble at any draft length.
    """
    if spec_config is not None and not engine.gdn_replay:
        raise NoTarget(
            f"max_draft_len={spec_config.max_draft_len} with TRTLLM_USE_GDN_REPLAY=0: the FP8 "
            "target's verify step runs the GDN cached replay and does not assemble the legacy "
            "per-step recurrent states; leave TRTLLM_USE_GDN_REPLAY unset"
        )
    _check_engine(
        spec_config, kv_cache_quant_algo, engine, _FP8_MOE_BACKENDS, _FP8_SSM_STATE_DTYPES
    )


def fp8_member(model_config: ModelConfig, llm_args: TorchLlmArgs) -> str:
    """The member of the FP8 target (`qwen3_8_flash_next_fp8__sm_103__unified`) this deployment
    of Qwen3.8-Flash-Next-FP8 is (`_FP8_LAYOUTS`: tp1, tp2, tep2, dep2, tp4, tep4, dep4), or
    raise ``NoTarget`` saying why it is none.

    `route` calls it once the checkpoint's geometry, quantization (FP8_BLOCK_SCALES) and GPU
    have matched. It admits no speculation or one-model MTP-Eagle with max_draft_len 1..7 under
    attention TP and at dep4, never at dep2; the breakable prefill graph under attention TP
    only; and the engine choices `_check_fp8_engine` admits.
    """
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    if not _mtp_eagle_or_none(spec_config):
        raise NoTarget(
            f"{_describe_spec(spec_config)}; the FP8 target runs one-model MTP-Eagle over the "
            f"checkpoint's one MTP module as a linear chain (max_draft_len 1..{_MAX_DRAFT_LEN}), "
            "or no speculative decoding"
        )
    member = layout(mapping, _FP8_LAYOUTS)
    if member is None:
        raise NoTarget(
            f"{_describe(mapping)}; the FP8 target serves Qwen3.8-Flash-Next-FP8 in one node at "
            "tp1, at 2 or 4 ranks with attention TP over the group and the experts split by "
            "Mapping's default MoE TP (tp2, tp4) or by expert (tep2, tep4), or with attention DP "
            "and the experts split by expert (dep2, dep4)"
        )
    engine = engine_choices(model_config, llm_args)
    if mapping.enable_attention_dp:
        _check_attention_dp(member, spec_config, engine, _FP8_ATTENTION_DP_MTP_LAYOUTS)
    _check_fp8_engine(spec_config, engine_kv_cache_quant_algo(model_config, llm_args), engine)
    return member


# Qwen3.8-Flash-Next publishes ``Qwen4ExpForConditionalGeneration``, and the
# config loader's composite branch writes the same name back.
@routes("Qwen4ExpForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    A target is imported where it is returned, so declining costs no import of
    modeling code. Routing also declines the engine choices (`EngineChoices`) the
    target's constructor would refuse.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    quant_algo = getattr(model_config.quant_config, "quant_algo", None)
    text = getattr(pretrained_config, "text_config", None)
    vision = getattr(pretrained_config, "vision_config", None)
    # Absent means this is not the normalized Qwen4-Exp composite at all, and
    # nothing below would be readable.
    if text is None or vision is None:
        raise NoTarget(
            f"{type(pretrained_config).__name__} carries no text_config/vision_config; "
            "the qwen4_exp targets read both halves of the normalized composite"
        )

    shape = (
        text.num_hidden_layers,
        text.hidden_size,
        text.num_experts,
        text.num_attention_heads,
        text.linear_num_value_heads,
        text.vocab_size,
    )
    vshape = (
        vision.depth,
        vision.hidden_size,
        vision.num_heads,
        vision.spatial_merge_size,
        vision.out_hidden_size,
        vision.num_position_embeddings,
        tuple(getattr(vision, "deepstack_visual_indexes", None) or ()),
        vision.hidden_act,
    )
    mtp_eagle_or_none = _mtp_eagle_or_none(spec_config)

    geometry = compute_capability == _SM and shape == _TEXT_SHAPE and vshape == _VISION_SHAPE
    nvfp4 = geometry and quant_algo == _NVFP4_QUANT_ALGO
    # Qwen3.8-Flash-Next-NVFP4 on one GB300 (sm103), MTP-Eagle or no speculation.
    if nvfp4 and mapping.world_size == 1 and mtp_eagle_or_none:
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling import ModelingV2Qwen38FlashNextNvfp4Sm103Tp1
        # isort: on
        # fmt: on

        _check_engine(spec_config, kv_cache_quant_algo, engine_choices(model_config, llm_args))
        return ModelingV2Qwen38FlashNextNvfp4Sm103Tp1

    # Qwen3.8-Flash-Next-NVFP4 on four GB300s (sm103) at tp4, tep4 or dep4, MTP-Eagle or no
    # speculation; on two at dep2, without speculation.
    member = layout(mapping)
    if nvfp4 and member is not None and mtp_eagle_or_none:
        engine = engine_choices(model_config, llm_args)
        if mapping.enable_attention_dp:
            _check_attention_dp(member, spec_config, engine)
        _check_default_engine(spec_config, kv_cache_quant_algo, engine)
        # fmt: off
        # isort: off
        from .qwen3_8_flash_next_nvfp4__sm_103__default.modeling import ModelingV2Qwen38FlashNextNvfp4Sm103Default
        # isort: on
        # fmt: on

        return ModelingV2Qwen38FlashNextNvfp4Sm103Default

    # Qwen3.8-Flash-Next-FP8 on one, two or four GB300s (sm103): `fp8_member` names the member
    # or declines the deployment.
    if geometry and quant_algo == _FP8_QUANT_ALGO:
        fp8_member(model_config, llm_args)
        # fmt: off
        # isort: off
        from .qwen3_8_flash_next_fp8__sm_103__unified.modeling import ModelingV2Qwen38FlashNextFp8Sm103Unified
        # isort: on
        # fmt: on

        return ModelingV2Qwen38FlashNextFp8Sm103Unified

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, vision shape "
        f"{vshape}, quantization {quant_algo or 'none'}, {_describe(mapping)}, "
        f"{_describe_spec(spec_config)}, KV cache {kv_cache_quant_algo or 'bf16'}; the qwen4_exp "
        f"targets are Qwen3.8-Flash-Next-NVFP4 ({_NVFP4_QUANT_ALGO}) on sm103 in one node: at 1 "
        "rank (tp1), and at 4 ranks tp4 (attention TP4, experts split by intermediate), tep4 "
        "(attention TP4, experts EP4) or dep4 (attention DP, experts EP4), with one-model "
        f"MTP-Eagle (max_draft_len 1..{_MAX_DRAFT_LEN}) or no speculative decoding; at 2 ranks "
        "dep2 (attention DP, experts EP2) without speculative decoding; and "
        f"Qwen3.8-Flash-Next-FP8 ({_FP8_QUANT_ALGO}) on sm103 in one node at tp1, tp2, tep2, "
        "dep2, tp4, tep4 or dep4"
    )

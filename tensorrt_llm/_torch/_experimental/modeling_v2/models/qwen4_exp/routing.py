# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

`route` compares the facts that fix which code runs against the deployments the
targets were written for, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). The checkpoint's published
architecture name is kept by `load_pretrained_config`, so `route` registers under it.
All criteria are read off `text_config` / `vision_config`; the composite parent's
mirrored fields are unreliable.

Qwen3.8-Flash-Next-NVFP4 on sm103 lands on `qwen3_8_flash_next_nvfp4__sm_103__default`
at one of its parallel layouts (`layout`): tp1 on one rank, and tp4, tep4, dep4 or dep2 in
one node.
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

# The GPU the target is certified on.
_SM = (10, 3)

# (layers, hidden, experts, attention heads, linear value heads, vocab). Shape-sniffing
# also matches fine-tunes; the accuracy gate pins identity.
_TEXT_SHAPE = (48, 2560, 512, 24, 48, 248320)

# (depth, hidden, heads, merge, out_hidden, num_pos, deepstack_indexes, act).
# `deepstack_visual_indexes == ()` is the key member: a deepstack sibling has a merger
# list these targets lack.
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# The longest MTP draft the target runs (`_MAX_VERIFY_WIDTH - 1` in modeling.py).
_MAX_DRAFT_LEN = 7

# The engine choices the target implements at one rank (tp1), all in its modeling.py: the
# `moe_config.backend` requests (`_TRUNK_MOE_BACKENDS`), the longest draft of the CUTEDSL
# experts (`_CUTEDSL_MAX_DRAFT_LEN`) and the recurrent-state cache dtypes
# (`_ONE_RANK_SSM_STATE_DTYPES`).
_ONE_RANK_MOE_BACKENDS = ("AUTO", "TRTLLM", "CUTEDSL")
_MAX_CUTEDSL_DRAFT_LEN = 3
_ONE_RANK_SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)

# Their counterparts at every other layout: no CUTEDSL experts (`_TRUNK_MOE_BACKENDS` less
# CUTEDSL) and a bf16 recurrent state (`_SSM_STATE_DTYPES`).
_MOE_BACKENDS = ("AUTO", "TRTLLM")
_SSM_STATE_DTYPES = (torch.bfloat16,)

# (world size, attention DP, MoE TP, MoE EP) -> the target's layout. Every layout spans its
# whole world in one node, attention TP over all of it unless attention DP replicates
# attention. One rank (tp1) holds everything. At four ranks the routed experts are split by
# intermediate (tp4) or by expert (tep4, dep4). At two ranks only attention DP with two
# 256-expert windows (dep2), where each rank's attention, Gated-DeltaNet and PLE geometry is
# dep4's.
_LAYOUTS = {
    (1, False, 1, 1): "tp1",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (2, True, 1, 2): "dep2",
}

# The attention-DP layouts MTP is admitted at: those its draft acceptance was measured at.
_ATTENTION_DP_MTP_LAYOUTS = ("dep4",)


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


def layout(mapping) -> Optional[str]:
    """The target's layout ``mapping`` describes (``tp1``, ``tp4``, ``tep4``, ``dep4``,
    ``dep2``), or None.

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
    return _LAYOUTS.get(
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
    member: str,
) -> None:
    """Decline what the target's constructor or first forward would refuse at layout
    `member`: at tp1 the CuteDSL experts and the fp32 recurrent state are served too.

    Without this ``auto`` claims the deployment and then fails mid-build instead of
    falling back to the built-in implementation.
    """
    if member == "tp1":
        moe_backends, ssm_state_dtypes = _ONE_RANK_MOE_BACKENDS, _ONE_RANK_SSM_STATE_DTYPES
    else:
        moe_backends, ssm_state_dtypes = _MOE_BACKENDS, _SSM_STATE_DTYPES
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
            f"moe_config.backend={engine.moe_backend!r}; at {member} the target runs the "
            f"trunk's routed experts on {', '.join(moe_backends)}"
        )
    if engine.mamba_ssm_cache_dtype not in ssm_state_dtypes:
        served = " or ".join(str(dtype).removeprefix("torch.") for dtype in ssm_state_dtypes)
        raise NoTarget(
            f"a {engine.mamba_ssm_cache_dtype} recurrent-state cache; at {member} the target "
            f"serves {served} (kv_cache_config.mamba_ssm_cache_dtype)"
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
    if not engine.gdn_replay:
        raise NoTarget(
            "MTP with TRTLLM_USE_GDN_REPLAY=0: the target verifies its Gated-DeltaNet rows on "
            "the cached replay only, not on the legacy per-step recurrent states; leave "
            "TRTLLM_USE_GDN_REPLAY unset"
        )
    if str(engine.moe_backend).upper() == "CUTEDSL" and draft > _MAX_CUTEDSL_DRAFT_LEN:
        raise NoTarget(
            f"max_draft_len={draft} with moe_config.backend=CUTEDSL: the CuteDSL expert "
            f"pipeline is certified at max_draft_len up to {_MAX_CUTEDSL_DRAFT_LEN}"
        )


def _check_attention_dp(member: str, spec_config: Optional[Any], engine: EngineChoices) -> None:
    """Decline what the target does not run under attention DP (dep4, dep2), where
    each rank serves its own requests and only the routed experts cross ranks.

    MTP is admitted at dep4, where each rank drafts for its own requests and the draft layer's
    FP8 expert window runs the trunk's attention-DP round trip. Routing admits both MTP and
    the breakable prefill graph at the attention-TP layouts (tp4, tep4).
    """
    if spec_config is not None and member not in _ATTENTION_DP_MTP_LAYOUTS:
        raise NoTarget(
            f"speculative decoding with attention DP ({member}): no draft acceptance was "
            "measured at this layout; MTP under attention DP is admitted at "
            f"{', '.join(_ATTENTION_DP_MTP_LAYOUTS)}"
        )
    if engine.prefill_cuda_graph_backend == "breakable":
        raise NoTarget(
            f"prefill_cuda_graph_backend='breakable' with attention DP ({member}): the engine "
            "pads every rank to one prefill bucket, so a rank holding only generation requests "
            "is handed more rows than its batch, which the target's decode step refuses"
        )


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
    text = getattr(pretrained_config, "text_config", None)
    vision = getattr(pretrained_config, "vision_config", None)
    # Absent means this is not the normalized Qwen4-Exp composite at all, and
    # nothing below would be readable.
    if text is None or vision is None:
        raise NoTarget(
            f"{type(pretrained_config).__name__} carries no text_config/vision_config; "
            "the qwen4_exp target reads both halves of the normalized composite"
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

    # The target implements the one-model MTP-Eagle loop over the checkpoint's
    # single MTP module, and no other draft path. A draft longer than seven needs
    # more verify rows per request than the GDN cached replay and thop_attention
    # are certified at. The draft must also be a linear chain over that one
    # module; the target asserts each of these at construction, so declining here
    # is what lets ``auto`` fall back instead of crashing.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and getattr(spec_config, "num_nextn_predict_layers", None) in (None, 1)
        and 1 <= spec_config.max_draft_len <= _MAX_DRAFT_LEN
        and not getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "eagle_choices", None) is None
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )

    checkpoint = compute_capability == _SM and shape == _TEXT_SHAPE and vshape == _VISION_SHAPE
    # Qwen3.8-Flash-Next-NVFP4 on GB300s (sm103): on one at tp1, on four at tp4, tep4 or dep4,
    # MTP-Eagle or no speculation; on two at dep2, without speculation.
    member = layout(mapping)
    if checkpoint and member is not None and mtp_eagle_or_none:
        engine = engine_choices(model_config, llm_args)
        if mapping.enable_attention_dp:
            _check_attention_dp(member, spec_config, engine)
        _check_engine(spec_config, kv_cache_quant_algo, engine, member)
        # One line on purpose: the claims suite reads the routed target off the
        # `from .<target>.modeling import <Class>` line.
        # fmt: off
        # isort: off
        from .qwen3_8_flash_next_nvfp4__sm_103__default.modeling import ModelingV2Qwen38FlashNextNvfp4Sm103Default
        # isort: on
        # fmt: on

        return ModelingV2Qwen38FlashNextNvfp4Sm103Default

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, vision shape "
        f"{vshape}, {_describe(mapping)}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f"{f' max_draft_len={spec_config.max_draft_len}' if spec_config is not None else ''}, "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; the qwen4_exp target is "
        "Qwen3.8-Flash-Next-NVFP4 on sm103 in one node: at 1 rank (tp1), and at 4 ranks tp4 "
        "(attention TP4, experts split by intermediate), tep4 (attention TP4, experts EP4) or "
        f"dep4 (attention DP, experts EP4), with one-model MTP-Eagle (max_draft_len "
        f"1..{_MAX_DRAFT_LEN}) or no speculative decoding; at 2 ranks dep2 (attention DP, "
        "experts EP2) without speculative decoding"
    )

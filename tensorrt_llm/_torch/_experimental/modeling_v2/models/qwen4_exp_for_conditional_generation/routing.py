# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration on two or four ranks.

`route` compares the facts that fix which code runs against the deployments the
unified target serves, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). `models/qwen4_exp/routing.py`
routes the same architecture at one rank; neither family claims a deployment the
other does, because they claim different world sizes.

All criteria are read off `text_config` / `vision_config`; the composite parent's
mirrored fields are unreliable.
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
# list this target lacks.
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# (world size, attention DP, MoE TP, MoE EP) -> the layout. Every layout spans its whole
# world in one node, attention TP over all of it unless attention DP replicates attention.
# At four ranks the routed experts are split by intermediate (tp4) or by expert (tep4,
# dep4). At two ranks only attention DP with two 256-expert windows (dep2): each rank's
# attention, Gated-DeltaNet and PLE geometry is dep4's, and it is the layout of both
# engines of the disaggregated-serving test.
_LAYOUTS = {
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (2, True, 1, 2): "dep2",
}

# `moe_config.backend` requests the target implements (`_TRUNK_MOE_BACKENDS` in modeling.py).
_MOE_BACKENDS = ("AUTO", "TRTLLM")

# The longest MTP draft the target runs (`_MAX_VERIFY_WIDTH - 1` in modeling.py); without the
# GDN cached replay the per-step recurrent states (`_MAX_LEGACY_VERIFY_WIDTH - 1`) bound it.
_MAX_DRAFT_LEN = 7
_MAX_LEGACY_DRAFT_LEN = 3

# Recurrent-state cache dtypes the target serves (`_SSM_STATE_DTYPES` in modeling.py).
_SSM_STATE_DTYPES = (torch.bfloat16,)


class EngineChoices(NamedTuple):
    """Engine choices that fix which code the target runs but are not in the checkpoint's
    config, as the engine will resolve them."""

    torch_dtype: torch.dtype  # the dtype the engine runs the model in
    moe_backend: str  # `moe_config.backend` as requested, before AUTO resolves
    mamba_ssm_cache_dtype: torch.dtype  # the recurrent-state cache's dtype
    gdn_replay: bool  # TRTLLM_USE_GDN_REPLAY, which the engine reads for eligible MTP workloads
    prefill_cuda_graph_backend: str  # `prefill_cuda_graph_backend`: disabled, piecewise, breakable


def engine_choices(model_config: ModelConfig, llm_args: TorchLlmArgs) -> EngineChoices:
    # Imported when a layout matched only, as the target is.
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
    """The target's layout ``mapping`` describes (``tp4``, ``tep4``, ``dep4``, ``dep2``),
    or None.

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
    spec_config: Optional[Any], kv_cache_quant_algo: Optional[Any], engine: EngineChoices
) -> None:
    """Decline what the target's constructor or first forward would refuse.

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
    if str(engine.moe_backend).upper() not in _MOE_BACKENDS:
        raise NoTarget(
            f"moe_config.backend={engine.moe_backend!r}; the target runs the routed experts "
            f"on {', '.join(_MOE_BACKENDS)}"
        )
    if engine.mamba_ssm_cache_dtype not in _SSM_STATE_DTYPES:
        raise NoTarget(
            f"a {engine.mamba_ssm_cache_dtype} recurrent-state cache; the target serves "
            "bfloat16 (kv_cache_config.mamba_ssm_cache_dtype)"
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
            f"max_draft_len={draft} with TRTLLM_USE_GDN_REPLAY=0: the per-step recurrent "
            f"states are certified at max_draft_len up to {_MAX_LEGACY_DRAFT_LEN}; leave "
            "TRTLLM_USE_GDN_REPLAY unset for a longer draft"
        )


def _check_attention_dp(member: str, spec_config: Optional[Any], engine: EngineChoices) -> None:
    """Decline what the target does not run under attention DP (dep4, dep2), where each rank
    serves its own requests and only the routed experts cross ranks.

    Routing admits both at the attention-TP layouts (tp4, tep4).
    """
    if spec_config is not None:
        raise NoTarget(
            f"speculative decoding with attention DP ({member}): the target's MTP draft layer "
            "completes its FP8 expert window with the attention-TP sum, not the attention-DP "
            "allgather / reduce-scatter round trip, so it would drop the other ranks' expert "
            "partials, and no draft acceptance was measured under attention DP"
        )
    if engine.prefill_cuda_graph_backend == "breakable":
        raise NoTarget(
            f"prefill_cuda_graph_backend='breakable' with attention DP ({member}): the engine "
            "pads every rank to one prefill bucket, so a rank holding only generation requests "
            "is handed more rows than its batch, which the target's decode step refuses"
        )


@routes("Qwen4ExpForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The engine choices are checked before the target is imported, so declining costs no
    import of modeling code.
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

    # The one-model MTP-Eagle loop over the checkpoint's single MTP module, and no
    # other draft path. A draft longer than seven needs more verify rows per request
    # than the target runs, and the draft must be a linear chain over that one module.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and getattr(spec_config, "num_nextn_predict_layers", None) in (None, 1)
        and 1 <= spec_config.max_draft_len <= _MAX_DRAFT_LEN
        and not getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "eagle_choices", None) is None
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )

    # Qwen3.8-Flash-Next-NVFP4 on four GB300s (sm103) at tp4 or tep4, MTP-Eagle or no
    # speculation, or at dep4; on two at dep2. Attention DP runs without speculation.
    member = layout(mapping)
    if (
        compute_capability == _SM
        and shape == _TEXT_SHAPE
        and vshape == _VISION_SHAPE
        and member is not None
        and mtp_eagle_or_none
    ):
        engine = engine_choices(model_config, llm_args)
        if mapping.enable_attention_dp:
            _check_attention_dp(member, spec_config, engine)
        _check_engine(spec_config, kv_cache_quant_algo, engine)
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .hf_fc694b5_orig__sm_103__unified.modeling import ModelingV2HfFc694b5OrigSm103Unified
        # isort: on
        # fmt: on

        return ModelingV2HfFc694b5OrigSm103Unified

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, vision shape "
        f"{vshape}, {_describe(mapping)}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f"{f' max_draft_len={spec_config.max_draft_len}' if spec_config is not None else ''}, "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; the unified qwen4_exp target is "
        "Qwen3.8-Flash-Next-NVFP4 on sm103 in one node: at 4 ranks tp4 (attention TP4, "
        "experts split by intermediate) or tep4 (attention TP4, experts EP4) with one-model "
        f"MTP-Eagle (max_draft_len 1..{_MAX_DRAFT_LEN}) or no speculative decoding, or dep4 "
        "(attention DP, experts EP4); at 2 ranks dep2 (attention DP, experts EP2); attention "
        "DP without speculative decoding"
    )

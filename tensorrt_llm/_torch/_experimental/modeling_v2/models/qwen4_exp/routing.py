# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

`route` compares the facts that fix which code runs against the one deployment the
target was written for, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). The checkpoint's published
architecture name is kept by `load_pretrained_config`, so `route` registers under it.
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

# (layers, hidden, experts, attention heads, linear value heads, vocab). Shape-sniffing
# also matches fine-tunes; the accuracy gate pins identity.
_TEXT_SHAPE = (48, 2560, 512, 24, 48, 248320)

# (depth, hidden, heads, merge, out_hidden, num_pos, deepstack_indexes, act).
# `deepstack_visual_indexes == ()` is the key member: a deepstack sibling has a merger
# list this target lacks.
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# `moe_config.backend` requests the target implements (`_TRUNK_MOE_BACKENDS` in modeling.py).
_MOE_BACKENDS = ("AUTO", "TRTLLM", "CUTEDSL")

# The longest MTP draft the target runs (`_MAX_VERIFY_WIDTH - 1` in modeling.py); the legacy
# per-step recurrent states (`_MAX_LEGACY_VERIFY_WIDTH - 1`) and the CUTEDSL experts
# (`_CUTEDSL_MAX_DRAFT_LEN`) are certified at shorter ones.
_MAX_DRAFT_LEN = 7
_MAX_LEGACY_DRAFT_LEN = 3
_MAX_CUTEDSL_DRAFT_LEN = 3

# Recurrent-state cache dtypes the target serves (`_SSM_STATE_DTYPES` in modeling.py).
_SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)


class EngineChoices(NamedTuple):
    """Engine choices that fix which code the target runs but are not in the checkpoint's
    config, as the engine will resolve them."""

    torch_dtype: torch.dtype  # the dtype the engine runs the model in
    moe_backend: str  # `moe_config.backend` as requested, before AUTO resolves
    mamba_ssm_cache_dtype: torch.dtype  # the recurrent-state cache's dtype
    gdn_replay: bool  # TRTLLM_USE_GDN_REPLAY, which the engine reads for eligible MTP workloads
    prefill_cuda_graph_backend: str  # `prefill_cuda_graph_backend`: disabled, piecewise, breakable


def engine_choices(model_config: ModelConfig, llm_args: TorchLlmArgs) -> EngineChoices:
    # Imported on the claimed branch only, as the target is.
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
            f"moe_config.backend={engine.moe_backend!r}; the target runs the trunk's routed "
            f"experts on {', '.join(_MOE_BACKENDS)}"
        )
    if engine.mamba_ssm_cache_dtype not in _SSM_STATE_DTYPES:
        raise NoTarget(
            f"a {engine.mamba_ssm_cache_dtype} recurrent-state cache; the target serves "
            "float32 or bfloat16 (kv_cache_config.mamba_ssm_cache_dtype)"
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


# Qwen3.8-Flash-Next publishes ``Qwen4ExpForConditionalGeneration``, and the
# config loader's composite branch writes the same name back.
@routes("Qwen4ExpForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import
    of modeling code. It also declines the engine choices (`EngineChoices`) the
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

    # Qwen3.8-Flash-Next-NVFP4 on one GB300 (sm103), MTP-Eagle or no speculation.
    if (
        compute_capability == (10, 3)
        and shape == _TEXT_SHAPE
        and vshape == _VISION_SHAPE
        and mapping.world_size == 1
        and mtp_eagle_or_none
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .qwen3_8_flash_next_nvfp4__sm_103__tp1.modeling import ModelingV2Qwen38FlashNextNvfp4Sm103Tp1
        # isort: on
        # fmt: on

        _check_engine(spec_config, kv_cache_quant_algo, engine_choices(model_config, llm_args))
        return ModelingV2Qwen38FlashNextNvfp4Sm103Tp1

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, vision shape "
        f"{vshape}, {mapping.world_size} ranks, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f"{f' max_draft_len={spec_config.max_draft_len}' if spec_config is not None else ''}, "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the one qwen4_exp target is Qwen3.8-Flash-Next-NVFP4 on sm103, tp1, with "
        f"one-model MTP-Eagle (max_draft_len 1..{_MAX_DRAFT_LEN}) or no speculative decoding"
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

`route` compares the facts that fix which code runs against the one deployment the
target was written for, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). The checkpoint's published
architecture name is kept by `load_pretrained_config`, so `_router_index` keys on it.
All criteria are read off `text_config` / `vision_config`; the composite parent's
mirrored fields are unreliable.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import EngineFacts, NoTarget

# (layers, hidden, experts, attention heads, linear value heads, vocab). Shape-sniffing
# also matches fine-tunes; the accuracy gate pins identity.
_TEXT_SHAPE = (48, 2560, 512, 24, 48, 248320)

# (depth, hidden, heads, merge, out_hidden, num_pos, deepstack_indexes, act).
# `deepstack_visual_indexes == ()` is the key member: a deepstack sibling has a merger
# list this target lacks.
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# `moe_config.backend` requests the target implements (`_TRUNK_MOE_BACKENDS` in modeling.py).
_MOE_BACKENDS = ("AUTO", "TRTLLM", "CUTEDSL")


def _check_engine(spec_config: Optional[Any], engine: Optional[EngineFacts]) -> None:
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
    if engine is None:
        return
    if engine.torch_dtype != torch.bfloat16:
        raise NoTarget(f"the engine runs the model in {engine.torch_dtype}; the target is bf16")
    if engine.kv_cache_quant_algo not in (None, QuantAlgo.FP8):
        raise NoTarget(
            f"the engine resolved kv_cache_quant_algo {engine.kv_cache_quant_algo}; the target "
            "stores K/V as the checkpoint's bf16 or as FP8"
        )
    if str(engine.moe_backend).upper() not in _MOE_BACKENDS:
        raise NoTarget(
            f"moe_config.backend={engine.moe_backend!r}; the target runs the trunk's routed "
            f"experts on {', '.join(_MOE_BACKENDS)}"
        )
    if (
        spec_config is not None
        and engine.gdn_replay
        and engine.mamba_ssm_cache_dtype != torch.float32
    ):
        raise NoTarget(
            f"MTP with a {engine.mamba_ssm_cache_dtype} recurrent-state cache: the engine "
            "verifies with GDN cached replay, which the target runs on an fp32 cache. Set "
            "kv_cache_config.mamba_ssm_cache_dtype to float32 or export TRTLLM_USE_GDN_REPLAY=0"
        )


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    engine: Optional[EngineFacts] = None,
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import
    of modeling code.
    """
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
    # single MTP module, and no other draft path. A draft longer than three
    # needs predicted_tokens_per_seq past the 1..4 thop_attention certifies,
    # which the target asserts; declining here falls back instead of crashing.
    # The draft must also be a linear chain over that one module; the target
    # asserts each of these at construction, so declining here is what lets
    # ``auto`` fall back instead of crashing.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and getattr(spec_config, "num_nextn_predict_layers", None) in (None, 1)
        and 1 <= spec_config.max_draft_len <= 3
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

        _check_engine(spec_config, engine)
        return ModelingV2Qwen38FlashNextNvfp4Sm103Tp1

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, vision shape "
        f"{vshape}, {mapping.world_size} ranks, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f"{f' max_draft_len={spec_config.max_draft_len}' if spec_config is not None else ''}; "
        "the one qwen4_exp target is Qwen3.8-Flash-Next-NVFP4 on sm103, tp1, with "
        "one-model MTP-Eagle (max_draft_len 1..3) or no speculative decoding"
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV41ForConditionalGeneration config lands. Read this file and you know.

The engine names the DeepSeek-V4.1 checkpoint that carries its vision tower
``DeepseekV41ForConditionalGeneration``; every target here runs that tower, so
a text-only V4.1 config (``DeepseekV41ForCausalLM``) is not routed here.

One forward-reading decision tree per architecture family: the criteria are
evaluated in the order a reader would ask them, and every branch that does not
end in a target returns None (in ``auto`` the engine then uses the built-in
DeepSeek-V4.1 implementation; in ``require`` it raises, quoting the trace below).
"""

from __future__ import annotations

from typing import Optional

from ..._router_index import NULL_TRACE, ModelingV2Context, Trace

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target.
_SM = (10, 3)

# Config-shape fingerprint -> checkpoint identity; see the note in the gpt_oss
# routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, n_routed_experts, q_lora_rank, head_dim,
#  o_groups, index_n_heads, num_nextn_predict_layers). Every field changes an
# operand shape the target declares: the expert stack, the q-LoRA pair, the
# 512-wide shared latent, the grouped output projection, the indexer heads and
# the DSpark stage count the weight table predicts. The attributes are read
# off the top-level config, which answers them from the checkpoint's
# `text_config`.
_CHECKPOINTS = {
    (40, 5120, 384, 1280, 512, 8, 32, 3): "deepseek_v4_1_flash_nvfp4",
}

_TARGETS = {
    ("deepseek_v4_1_flash_nvfp4", "unified"): "ModelingV2DeepseekV41FlashNvfp4Sm103Unified",
}

# Synthetic architecture name -> the module whose import registers it.
TARGET_MODULES = {
    "ModelingV2DeepseekV41FlashNvfp4Sm103Unified": (
        "models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.modeling"
    ),
}

# Checkpoint identity -> the module carrying the input processor its targets
# read their multimodal inputs from (``INPUT_PROCESSOR``, ``MODEL_TYPE``,
# ``PLACEHOLDER_METADATA``); ``modeling_v2_input_processor`` registers it.
INPUT_PROCESSORS = {
    "deepseek_v4_1_flash_nvfp4": (
        "models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.input_processor"
    ),
}

# The global quant_algo the engine reads for this checkpoint's export -- fp8
# 32x32 dense weights plus NVFP4 routed experts. A modelopt reading of the
# export gives MIXED_PRECISION; the engine's DeepSeek-V4.1 loader reads the
# dense FP8 blocks as FP8_BLOCK_SCALES and keeps the experts' NVFP4 per layer.
# The target loads the checkpoint's own tensors under either reading, and a
# checkpoint re-exported another way fails the weight table's dtype asserts.
_QUANT = frozenset({"MIXED_PRECISION", "FP8_BLOCK_SCALES"})


def _quant(q) -> Optional[str]:
    """The global quant_algo as a string, or None for an unquantized config."""
    algo = getattr(q, "quant_algo", None)
    return None if algo is None else str(getattr(algo, "value", algo))


def _parallel(m) -> Optional[str]:
    """Name the parallel segment, or None if no target implements it.

    The unified target folds its members at runtime from the mapping:
    tep4 (attention TP4 + MoE EP4), dep4 (attention DP4 + MoE EP4) and tp4
    (attention TP4 + MoE TP4) on one node, tep8 / dep8 (EP8) over two NVL72
    trays, and any of them per pipeline stage (pp2 x tep2 on one node, pp2 x
    tep4 over two trays). Without pipeline parallelism the TP group is 4 or 8
    ranks; with it, 2 or 4 per stage. Whether a pipeline split is valid is the
    target's to say (a stage must hold every layer its layers read from).
    """
    tp, pp = m.tp_size, m.pp_size
    if m.world_size != tp * pp or tp not in ((4, 8) if pp == 1 else (2, 4)):
        return None
    ep, mtp = m.moe_ep_size, m.moe_tp_size
    if ep == tp and mtp == 1:
        return "unified"  # tep, or dep with attention DP
    if tp in (2, 4) and ep == 1 and mtp == tp and not m.enable_attention_dp:
        return "unified"  # tp (MoE TP)
    return None


def _shape(c) -> tuple:
    """The config-shape fingerprint ``_CHECKPOINTS`` is keyed by."""
    return (
        c.num_hidden_layers,
        c.hidden_size,
        c.n_routed_experts,
        c.q_lora_rank,
        c.head_dim,
        c.o_groups,
        c.index_n_heads,
        c.num_nextn_predict_layers,
    )


def input_processor(c) -> Optional[str]:
    """The input-processor module for this checkpoint's targets, or None.

    Decided by the checkpoint alone: the input processor is built where the GPU
    and the parallel mapping are not known, so this cannot replay ``route``.
    """
    return INPUT_PROCESSORS.get(_CHECKPOINTS.get(_shape(c)))


def route(ctx: ModelingV2Context, trace: Trace = NULL_TRACE) -> Optional[str]:
    c, m = ctx.pretrained_config, ctx.mapping

    if not trace.check("sm", ctx.sm, ctx.sm == _SM):
        return None

    shape = _shape(c)
    ckpt = trace.resolve("shape", shape, _CHECKPOINTS.get(shape))
    if ckpt is None:
        return None

    quant = _quant(ctx.quant_config)
    if not trace.check("quant", quant, quant in _QUANT):
        return None

    parallel = trace.resolve(
        "parallel",
        f"ws={m.world_size} tp={m.tp_size} ep={m.moe_ep_size} "
        f"moe_tp={m.moe_tp_size} attention_dp={m.enable_attention_dp}",
        _parallel(m),
    )
    if parallel is None:
        return None

    return _TARGETS.get((ckpt, parallel))

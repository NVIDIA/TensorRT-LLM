# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK3ForConditionalGeneration config lands. Read this file and you know.

One forward-reading decision tree per architecture family: the criteria are
evaluated in the order a reader would ask them, and every branch that does not
end in a target returns None (in ``auto`` the engine then uses the built-in
Kimi K3 implementation; in ``require`` it raises, quoting the trace below).

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower and refuse image input.
"""

from __future__ import annotations

from typing import Any, Optional

from ..._router_index import NULL_TRACE, ModelingV2Context, Trace

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target. The K3
# decode kernels (the tcgen05 GEMVs, the fused MoE) are built for GB200 only.
_SM = (10, 0)

# Config-shape fingerprint -> checkpoint identity; see the note in the sibling
# gpt_oss routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, num_experts, routed_expert_hidden_size), all
# read from ``text_config``. The NVFP4 requant of the same checkpoint has this
# shape too; the quantization criterion below is what keeps it out.
_CHECKPOINTS = {
    (93, 7168, 896, 3584): "kimi_k3_mxfp4",
}

_TARGETS = {
    ("kimi_k3_mxfp4", "tp16_moetp4ep4"): "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4",
}

# Synthetic architecture name -> the module whose import registers it.
TARGET_MODULES = {
    "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4": (
        "models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling"
    ),
}


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _mxfp4(quant_config: Any) -> bool:
    """Whether the checkpoint is the MXFP4 one these targets load.

    The MXFP4 checkpoint declares its quantization only inside
    ``text_config.quantization_config`` (compressed-tensors), which the model
    config does not surface, so it reads as no quantization at all. The NVFP4
    requant ships ``hf_quant_config.json`` and reads as MIXED_PRECISION; its
    experts would land in a loader that reads packed MXFP4 tensors.
    """
    return quant_config is None or quant_config.quant_algo is None


def _parallel(m) -> Optional[str]:
    """Name the parallel topology, or None if no target implements it.

    Attention and the dense layers are sharded 16 ways; the routed experts are
    split 4 ways by tensor and 4 ways by expert. The expert split decides which
    expert weights each rank loads and into which shapes, so it selects a
    target rather than a runtime branch.
    """
    if (
        m.world_size == 16
        and m.tp_size == 16
        and m.pp_size == 1
        and m.moe_tp_size == 4
        and m.moe_ep_size == 4
        and not m.enable_attention_dp
    ):
        return "tp16_moetp4ep4"
    return None


def route(ctx: ModelingV2Context, trace: Trace = NULL_TRACE) -> Optional[str]:
    c, m = ctx.pretrained_config, ctx.mapping

    if not trace.check("sm", ctx.sm, ctx.sm == _SM):
        return None

    t = getattr(c, "text_config", None)
    if not trace.check("text_config", type(t).__name__, t is not None):
        return None

    shape = tuple(
        _field(t, name)
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "num_experts",
            "routed_expert_hidden_size",
        )
    )
    ckpt = trace.resolve("shape", shape, _CHECKPOINTS.get(shape))
    if ckpt is None:
        return None

    quant_algo = getattr(ctx.quant_config, "quant_algo", None)
    if not trace.check("quant", quant_algo, _mxfp4(ctx.quant_config)):
        return None

    parallel = trace.resolve(
        "parallel",
        f"ws={m.world_size} tp={m.tp_size} pp={m.pp_size} moe_tp={m.moe_tp_size} "
        f"moe_ep={m.moe_ep_size} attention_dp={m.enable_attention_dp}",
        _parallel(m),
    )
    if parallel is None:
        return None

    return _TARGETS.get((ckpt, parallel))

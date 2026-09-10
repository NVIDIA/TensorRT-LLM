# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV3ForCausalLM config lands. Read this file and you know.

One forward-reading decision tree per architecture family: the criteria are
evaluated in the order a reader would ask them, and every branch that does not
end in a target returns None (in ``auto`` the engine then uses the built-in
DeepseekV3 implementation; in ``require`` it raises, quoting the trace below).
"""

from __future__ import annotations

from typing import Optional

from ..._router_index import NULL_TRACE, StaircaseContext, Trace

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target.
_SM = (10, 3)

# Config-shape fingerprint -> checkpoint identity; see the note in the sibling
# gpt_oss routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, n_routed_experts, q_lora_rank). q_lora_rank
# is in the fingerprint because it changes the attention weight layout -- a
# checkpoint that matched the first three and not this one would be a
# different assembly, not a variant.
_CHECKPOINTS = {
    (61, 7168, 256, 1536): "r1_0528_nvfp4",
    # (30, 2560, 72, 0): "v3_lite_nvfp4",   <- second batch
}

_TARGETS = {
    ("r1_0528_nvfp4", "dep4"): "StaircaseDeepseekR10528Nvfp4Sm103Dep4",
}

# Synthetic architecture name -> the module whose import registers it.
TARGET_MODULES = {
    "StaircaseDeepseekR10528Nvfp4Sm103Dep4": "models.deepseek_v3.targets.r1_0528_nvfp4.sm_103.dep4.modeling",
}


def _parallel(m) -> Optional[str]:
    """Name the parallel topology, or None if no target implements it.

    This is a weight-layout question, which is why it selects a target rather
    than a runtime branch: dep4 replicates attention across ranks and shards
    only the experts, while tep4 (not in this batch) shards the attention
    heads. The two load different weights into different shapes.
    """
    if m.world_size == 4 and m.moe_ep_size == 4 and m.moe_tp_size == 1 and m.enable_attention_dp:
        return "dep4"
    return None


def route(ctx: StaircaseContext, trace: Trace = NULL_TRACE) -> Optional[str]:
    c, m = ctx.pretrained_config, ctx.mapping

    if not trace.check("sm", ctx.sm, ctx.sm == _SM):
        return None

    shape = (c.num_hidden_layers, c.hidden_size, c.n_routed_experts, c.q_lora_rank)
    ckpt = trace.resolve("shape", shape, _CHECKPOINTS.get(shape))
    if ckpt is None:
        return None

    parallel = trace.resolve(
        "parallel",
        f"ws={m.world_size} ep={m.moe_ep_size} "
        f"moe_tp={m.moe_tp_size} attention_dp={m.enable_attention_dp}",
        _parallel(m),
    )
    if parallel is None:
        return None

    return _TARGETS.get((ckpt, parallel))

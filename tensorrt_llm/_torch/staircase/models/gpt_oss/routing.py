# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GptOssForCausalLM config lands. Read this file and you know.

One forward-reading decision tree per architecture family: the criteria are
evaluated in the order a reader would ask them, and every branch that does not
end in a target returns None (in ``auto`` the engine then uses the built-in
GptOss implementation; in ``require`` it raises, quoting the trace below).
"""

from __future__ import annotations

from typing import Optional

from ..._router_index import NULL_TRACE, StaircaseContext, Trace

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target.
_SM = (10, 3)

# Config-shape fingerprint -> checkpoint identity. Sniffing the shape is the
# upstream idiom (``is_mla``, ``is_nemotron_hybrid`` do the same). It buys
# automatic routing at a stated cost: a *fine-tune* of this checkpoint has the
# same shape and is routed here silently. See TARGET.md -- the gate record is
# what pins the identity, and it says which checkpoint it was measured on.
#
# (num_hidden_layers, hidden_size, num_local_experts). Layer count alone
# separates 120b from 20b, but the expert count is what makes the MoE operand
# geometry this target declares correct, so it is part of the fingerprint.
_CHECKPOINTS = {
    (36, 2880, 128): "gpt_oss_120b",
}

_TARGETS = {
    ("gpt_oss_120b", "tp1"): "StaircaseGptOss120bSm103Tp1",
}

# Synthetic architecture name -> the module whose import registers it.
TARGET_MODULES = {
    "StaircaseGptOss120bSm103Tp1": "models.gpt_oss.targets.gpt_oss_120b.sm_103.tp1.modeling",
}


def route(ctx: StaircaseContext, trace: Trace = NULL_TRACE) -> Optional[str]:
    c, m = ctx.pretrained_config, ctx.mapping

    if not trace.check("sm", ctx.sm, ctx.sm == _SM):
        return None

    shape = (c.num_hidden_layers, c.hidden_size, c.num_local_experts)
    ckpt = trace.resolve("shape", shape, _CHECKPOINTS.get(shape))
    if ckpt is None:
        return None

    parallel = trace.resolve("parallel", f"ws={m.world_size}", "tp1" if m.world_size == 1 else None)
    if parallel is None:
        return None

    return _TARGETS.get((ckpt, parallel))

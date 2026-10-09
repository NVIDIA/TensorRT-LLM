# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GptOssForCausalLM config lands. Read this file and you know.

One forward-reading decision tree per architecture family, in two stages.
``route`` reads the configuration's identity and names a target;
``within_bounds`` reads the deployment's LLM API arguments and says whether
that target was certified for it. The criteria are evaluated in the order a
reader would ask them, and every branch that does not end in an accepted
target returns None or False (in ``auto`` the engine then uses the built-in
GptOss implementation; in ``require`` it raises, quoting the trace).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..._router_index import NULL_TRACE, ModelingV2Context, Trace

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target.
_SM = (10, 3)

# Config-shape fingerprint -> checkpoint identity. Sniffing the shape is the
# upstream idiom (``is_mla``, ``is_nemotron_hybrid`` do the same). It buys
# automatic routing at a stated cost: a *fine-tune* of this checkpoint has the
# same shape and is routed here silently. The accuracy gate is what pins the
# identity: it names the checkpoint this target was measured on.
#
# (num_hidden_layers, hidden_size, num_local_experts). Layer count alone
# separates 120b from 20b, but the expert count is what makes the MoE operand
# geometry this target declares correct, so it is part of the fingerprint.
_CHECKPOINTS = {
    (36, 2880, 128): "gpt_oss_120b",
}

_TARGETS = {
    ("gpt_oss_120b", "tp1"): "ModelingV2GptOss120bSm103Tp1",
}

# Synthetic architecture name -> the module whose import registers it.
TARGET_MODULES = {
    "ModelingV2GptOss120bSm103Tp1": "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling",
}


def route(ctx: ModelingV2Context, trace: Trace = NULL_TRACE) -> Optional[str]:
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


def within_bounds(
    target: str, args: "TorchLlmArgs", ctx: ModelingV2Context, trace: Trace = NULL_TRACE
) -> bool:
    """Whether this deployment is one ``target`` was certified for.

    ``route`` above says which target a checkpoint, GPU architecture and
    parallel topology map to. This says whether the LLM API arguments the
    deployment was configured with -- ``max_batch_size``, ``max_num_tokens``,
    speculative decoding, CUDA graphs, anything a target's accuracy gate did
    or did not exercise -- fall inside the envelope that gate measured.
    Outside it, ``auto`` builds the built-in implementation and ``require``
    raises quoting the trace. Criteria are written as
    ``trace.check(label, value, ok)`` so ``explain`` replays them.

    Bounds are evaluated on the arguments as the user configured them,
    before model defaults. A value the engine derives later -- an inferred
    ``max_seq_len``, a resolved MoE backend -- is not available here, and an
    ``"auto"`` the target itself resolves through the model-class preference
    hooks needs no bound: the target is the model class that decides it.

    No target in this family bounds anything yet: every deployment the
    identity stage routes here is accepted. The hook exists so that a bound
    can be added where a reader will look for it, next to the identity tree.
    """
    if target not in TARGET_MODULES:
        raise ValueError(
            f"{target!r} is not a target of this routing module; targets: {sorted(TARGET_MODULES)}"
        )
    return True

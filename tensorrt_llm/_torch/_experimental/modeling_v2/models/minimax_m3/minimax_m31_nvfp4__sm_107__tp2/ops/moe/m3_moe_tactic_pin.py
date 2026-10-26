# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2 WIDE decode tactics on the existing TRTLLM-Gen NVFP4 MoE runner.

The caller retains quantization, routing logits and finalization. Only the tactic
choice changes. Invalid configurations decline to the existing tuner path.
"""

from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

ENABLED = not m31_disabled("TP2W_MOE_TACTIC_PIN")


def tactic_for_rows(rows: int):
    """The CTX v2 natural-routing table; no interpolation outside its bands."""
    if 96 <= rows <= 127:
        return (16, 128)
    if rows == 232:
        return (32, 64)
    if rows == 256:
        return (32, 96)
    return None


def run_pinned(core, inputs, do_finalize: bool, routing_method_type: int, act_type: int, topk=None):
    """Run with a validated tactic, or return None so the caller keeps its tuner.

    inputs are the stock runner's first sixteen tensor arguments, in their
    original order. Routing uses the logits (inputs[0]), or with ``topk`` (int32 ids, bf16 weights; S1B_FOLD,
    inputs[0] None) routes from those ids.
    The caller limits this adapter to decode; prefill keeps the original path.
    """
    if not ENABLED:
        return None
    x = inputs[2]
    rows = x.shape[0]
    tactic = tactic_for_rows(rows)
    if tactic is None:
        return None
    if (core.mapping.tp_size, core.hidden, core.moe_inter, core.num_experts, core.top_k) != (
        2,
        6144,
        1536,
        128,
        4,
    ):
        return None
    assert len(inputs) == 16 and (inputs[0] is not None) != (topk is not None)
    assert x.shape[1] * 2 == core.hidden
    config = (
        core.num_experts,
        core.top_k,
        None,
        None,
        core.moe_inter,
        0,
        core.num_experts,
        core.routed_scaling_factor,
        routing_method_type,
        do_finalize,
        act_type,
    )
    key = (x.device, rows, config, core.max_num_tokens)
    cache = core.__dict__.setdefault("_tp2w_moe_tactic_pins", {})
    if key not in cache:
        runner = FP4BlockScaleMoERunner(
            *config, tune_max_num_tokens=core.max_num_tokens, use_dp=False
        )
        valid = runner.get_runner().get_valid_configs(
            core.top_k, core.hidden, core.moe_inter, core.num_experts, rows
        )
        accepted = tactic in {tuple(v) for v in valid}
        cache[key] = runner if accepted else None
        status = "taken" if accepted else "declined"
        logger.info(
            f"[M3.1] TP2W_MOE_TACTIC_PIN {status} at {rows} rows tactic {list(tactic)} "
            f"on rank {core.mapping.tp_rank} finalize={do_finalize}"
        )
    runner = cache[key]
    if runner is None:
        return None
    tw, ti = (None, None) if topk is None else (topk[1], topk[0])
    return runner.forward([*inputs, tw, ti], tactic=list(tactic))

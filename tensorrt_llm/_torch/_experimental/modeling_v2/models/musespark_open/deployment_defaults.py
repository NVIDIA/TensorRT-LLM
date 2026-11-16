# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kernel switches a Muse Spark Open GB300 deployment runs when its environment does not set them.

The GB300 fast paths are opt-in environment switches. The measured AgentX points set them
per role, so the measurement lived in the recipes. This table moves it into the code: before
``routing.route`` picks a target, ``apply`` sets every switch of the deployment's profile
that the environment leaves unset (``os.environ.setdefault``). A switch the environment sets,
to any value, wins. ``MUSESPARK_OPEN_DEPLOYMENT_DEFAULTS=0`` turns the whole table off
(the behaviour before it existed).

Profiles, sm_103 only, keyed on the tensor-parallel size, the role and ``max_batch_size``:

* TP4, any role: the context-path stack of the measured TEP4 points (output-projection /
  reduce-scatter overlap, vectorized pre-attention prep, CuTe DSL prefill experts,
  multi-instance context FMHA, context sequence parallelism) plus the TEP4 decode kernels
  (CL8 sliding-window block for 1..3 requests, 64-token split-KV pages, the fmk context
  FMHA of mixed steps, the four TEP4 extra paths of the TEP8 composition, the persistent
  experts' slot-major order and the router GEMM's 8-aligned row tiles). A context server
  also drops the sequence-parallel ``ce_gather`` and runs the SP tail pipeline.
* TP8 generation, ``max_batch_size`` 2..31: the GB300 TEP8 F11 target with the switches of
  the measured point at that batch size (31, 16, 8, 4, 2); another size takes the smallest
  measured size above it. F11 wins there by 3.6-13.9 % (gen-only A/B).
* TP8 at one request, and aggregated TP8: the ``sm_103__tep`` target (ARIA), which is
  3.2 % faster than F11 at batch 1; above one request the large-batch MoE switches.
* Every profile: the MNNVL o_proj all-reduce, the fused finalize all-reduce over MNNVL,
  sharded pool / unpool projections, the bit-identical context-step fusions and the
  FlashInfer BF16 MoE tactic pin (tiling only, same bits).

The role is ``MUSESPARK_OPEN_LEVER__ROLE`` when set. Otherwise a server without a cache
transceiver is ``agg``, and a disaggregated one is ``gen`` when its ``max_num_tokens`` is at
most ``GEN_MAX_NUM_TOKENS`` (a decode server's step budget) and ``ctx`` above it; the
inferred role is then also set as the lever table's role hint.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

#: ``0`` turns the table off.
ENABLE_ENV = "MUSESPARK_OPEN_DEPLOYMENT_DEFAULTS"
ROLE_ENV = "MUSESPARK_OPEN_LEVER__ROLE"
F11_ENV = "MUSESPARK_OPEN_SM103_TEP8_F11"

#: A disaggregated server whose ``max_num_tokens`` is at most this is a generation server.
GEN_MAX_NUM_TOKENS = 1024

_SM103 = 103

#: Every profile.
COMMON: Mapping[str, str] = {
    "MUSESPARK_OPEN_ALLREDUCE": "MNNVL",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL": "1",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS": "1",
    # Bit-identical context-step fusions (ftp/tekit!11077).
    "MUSESPARK_OPEN_CTX_FUSIONS": "1",
    # FlashInfer BF16 MoE (layers 0 and 61): a 128-wide tile from 2048 rows (tiling only).
    "TRTLLM_FI_BF16_MOE_TACTIC": "2048:128:1",
}

#: Above one request.
LARGE_BATCH: Mapping[str, str] = {
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED": "1",
    "MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES": "1",
    "MUSESPARK_OPEN_FULL_ROUTING_TORCH_ORDER": "1",
    "MUSESPARK_OPEN_LARGE_NORM_GATE": "1",
    "MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP": "1",
    "MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE": "1",
}

#: TP4, every role.
TEP4: Mapping[str, str] = {
    # Context path of the measured TEP4 points.
    "MUSESPARK_OPEN_OVERLAP_SP_PROJECTION": "1",
    "MUSESPARK_OPEN_OVERLAP_SP_PROJECTION_GRAPH": "1",
    "MUSESPARK_OPEN_OVERLAP_SP_PROJECTION_CHUNKS": "2",
    "TRTLLM_MSO_ENABLE_CTX4_PREP_VEC": "1",
    "MUSESPARK_OPEN_LEVER_PREFILL_CUTEDSL_EXPERTS": "on",
    "TRTLLM_MULTI_INSTANCE_CONTEXT_FMHA": "1",
    "MUSESPARK_OPEN_LEVER_CONTEXT_SEQUENCE_PARALLEL": "on",
    # TEP4 decode and mixed-step kernels (ftp/tekit!11080).
    "MUSESPARK_OPEN_SWA_CL8": "all",
    "MUSESPARK_OPEN_SWA_BLOCK_MULTI": "1",
    "MUSESPARK_OPEN_MULTI_REQUEST_SPLIT_KV_PAGE64": "1",
    "MUSESPARK_OPEN_FMK_CONTEXT_FMHA": "1",
    "MUSESPARK_OPEN_PERSISTENT_EXPERTS_OPTIONS": "order=0",
    "TRTLLM_ROUTER_GEMM_ALIGNED_TILE": "1",
}

#: TP4 servers that run decode steps (aggregated and generation) on top of ``TEP4``: the
#: decode-path fast paths of the TEP8 composition. Never on a context server: it runs no decode
#: step, and ``early_expert_ids`` refuses to start without the persistent expert window, which a
#: context server with a static ``MUSESPARK_OPEN_EXPERT_PLACEMENT`` never builds.
TEP4_DECODE: Mapping[str, str] = {
    "MUSESPARK_OPEN_SM103_TEP4_EXTRA_PATHS": (
        "early_expert_ids,pool_into_route,unpool_into_boundary,attention_output_into_hop"
    ),
}

#: TP4 context servers on top of ``TEP4``.
TEP4_CTX: Mapping[str, str] = {
    "MUSESPARK_OPEN_SP_DROP_PARTS": "ce_gather",
    "MUSESPARK_OPEN_SP_TAIL_PIPELINE": "1",
}

#: The TEP8 F11 switches every measured generation point sets.
_F11_SHARED: Mapping[str, str] = {
    F11_ENV: "1",
    "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_SPLITK_ONLY": "1",
    "MUSESPARK_OPEN_SM103_TEP8_UNPOOL_DIRECT_WIDE": "1",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS": "9",
    "MUSESPARK_OPEN_TEP8_SWA_CL8_2CTA": "1",
    "MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS": "2",
}

#: The wide hop / publish / balance switches of the measured points at 31, 16, 4 and 2 requests.
_F11_WIDE: Mapping[str, str] = {
    "MUSESPARK_OPEN_SM103_TEP8_HOP128_PRODUCER": "grid",
    "MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT128": "1",
    "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE": "splitk",
    "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_ONE": "1",
    "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_TACTICS": (
        "pool=64,32,4,8;pool@129=64,64,4,5;unpool=64,64,4,5"
    ),
    "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS": "128",
    "MUSESPARK_OPEN_TEP8_PREROUTED_MAX_ROWS": "128",
    "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS": "128",
}


def _merge(*parts: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in parts:
        out.update(part)
    return out


#: TP8 generation ``max_batch_size`` -> the F11 switches of the measured point at that size
#: (the consolidated GB300 pareto, ftp/tekit!11063: points 2..6).
F11_POINTS: Mapping[int, Mapping[str, str]] = {
    31: _merge(
        _F11_SHARED,
        _F11_WIDE,
        {
            "MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS": "qkv,pool,unpool",
            "MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS": "wide_moe_allreduce",
            "MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT_MAX_ROWS": "256",
            "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_MAX_ROWS": "248",
            "MUSESPARK_OPEN_TEP8_WIDE_BALANCE_ROWS": "256",
            "TRTLLM_MUSESPARK_ROUTER_STAGED_GEMM": "1",
        },
    ),
    16: _merge(
        _F11_SHARED,
        _F11_WIDE,
        {
            "MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS": "qkv,pool,unpool",
            "MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS": (
                "wide_draft_topk,wide_drafter_window_attention"
            ),
            "MUSESPARK_OPEN_DRAFTER_FUSED_PREP": "1",
            "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE": "128",
            "MUSESPARK_OPEN_TEP8_WIDE_WINDOW_GEOMETRY": "1",
        },
    ),
    8: _merge(
        _F11_SHARED,
        {
            "MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS": "unpool",
            "MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS": "wide_draft_topk",
            "MUSESPARK_OPEN_DRAFTER_FUSED_PREP": "1",
        },
    ),
    4: _merge(
        _F11_SHARED,
        _F11_WIDE,
        {
            "MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS": "qkv,o_proj,pool,unpool",
            "MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS": "wide_draft_topk",
            "MUSESPARK_OPEN_DRAFTER_FUSED_PREP": "1",
            "MUSESPARK_OPEN_FA_STAGES": "20",
            "MUSESPARK_OPEN_TEP8_PERSISTENT_EXPERTS_MAX_ROWS": "32",
        },
    ),
    2: _merge(
        _F11_SHARED,
        _F11_WIDE,
        {"MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS": "qkv,pool,unpool"},
    ),
}


def infer_role(env: Mapping[str, str], facts: Any) -> Tuple[Optional[str], str]:
    """``(role, source)``: ``agg``, ``ctx``, ``gen``, or None without engine facts."""
    hint = env.get(ROLE_ENV, "").strip().lower()
    if hint in ("agg", "ctx", "gen"):
        return hint, "hint"
    if facts is None:
        return None, "no engine facts"
    if not getattr(facts, "disaggregated", False):
        return "agg", "no cache transceiver"
    tokens = getattr(facts, "max_num_tokens", None)
    if tokens is not None and tokens <= GEN_MAX_NUM_TOKENS:
        return "gen", f"disaggregated, max_num_tokens {tokens} <= {GEN_MAX_NUM_TOKENS}"
    return "ctx", f"disaggregated, max_num_tokens {tokens} > {GEN_MAX_NUM_TOKENS}"


def f11_point(max_batch_size: Optional[int]) -> Optional[int]:
    """The measured F11 size a TP8 generation server of ``max_batch_size`` takes, or None (ARIA)."""
    if max_batch_size is None or max_batch_size < 2:
        return None
    above = [size for size in sorted(F11_POINTS) if size >= max_batch_size]
    return above[0] if above else None


def profile(
    sm: int, tp: int, role: Optional[str], max_batch_size: Optional[int]
) -> Tuple[Dict[str, str], str]:
    """``(switches, name)`` of a deployment; empty off sm_103 or outside TP4 / TP8."""
    if sm != _SM103 or tp not in (4, 8):
        return {}, "none"
    large = max_batch_size is None or max_batch_size > 1
    base = _merge(COMMON, LARGE_BATCH if large else {})
    if tp == 4:
        if role == "ctx":
            return _merge(base, TEP4, TEP4_CTX), "tep4-ctx"
        return _merge(base, TEP4, TEP4_DECODE), f"tep4-{role or 'agg'}"
    point = f11_point(max_batch_size) if role == "gen" else None
    if point is not None:
        return _merge(base, F11_POINTS[point]), f"tep8-gen-f11-b{point}"
    return base, f"tep8-{role or 'agg'}-aria"


def apply(
    compute_capability: Tuple[int, int],
    mapping: Any,
    facts: Any,
    env: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    """Set the deployment's default switches the environment leaves unset; return what was set."""
    env = os.environ if env is None else env
    if env.get(ENABLE_ENV, "1").strip() == "0":
        return {}
    major, minor = compute_capability
    sm = 10 * int(major) + int(minor)
    role, source = infer_role(env, facts)
    switches, name = profile(sm, int(mapping.tp_size), role, getattr(facts, "max_batch_size", None))
    if not switches:
        return {}
    # An explicit F11 opt-out keeps the ARIA target; its F11-only switches would be unread.
    if env.get(F11_ENV, "1").strip() != "1":
        switches = {
            k: v
            for k, v in switches.items()
            if k == F11_ENV
            or not (k.startswith(("MUSESPARK_OPEN_SM103_TEP8_", "MUSESPARK_OPEN_TEP8_")))
        }
    if source != "hint" and role in ("ctx", "gen"):
        switches[ROLE_ENV] = role
    applied = {k: v for k, v in switches.items() if k not in env}
    for key, value in applied.items():
        env[key] = value
    kept = sorted(k for k in switches if k not in applied)
    from tensorrt_llm.logger import logger

    logger.info(
        f"[musespark_open] deployment defaults {name} (sm{sm} tp{mapping.tp_size}, role {role} "
        f"from {source}, max_batch_size {getattr(facts, 'max_batch_size', None)}): set "
        f"{len(applied)} switch(es) {sorted(applied)}; environment kept {len(kept)} {kept}; "
        f"{ENABLE_ENV}=0 turns this off"
    )
    return applied

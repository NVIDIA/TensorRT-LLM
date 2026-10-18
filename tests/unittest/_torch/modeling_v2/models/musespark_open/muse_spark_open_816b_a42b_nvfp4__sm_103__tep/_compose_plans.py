# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The composition harnesses' arms as dispatch plans.

The harnesses (``test_modeling_v2_musespark_open_compose_*.py``) name each arm by
the short ids of the changes it carries, comma-separated, as their recorded
tables and digests do. ``plan_of_ids`` turns such a name into the plan the
model is adopted with: each id selects the fast path that carries it, closed
under ``depends_on``. ``ids_of`` gives back every id the plan's fast paths
carry, which is what the harnesses' expectations read; a fast path may carry
more than the ids an arm names (the post-attention hop always triggers early
and polls only the stale slots, for one). An arm naming an id no fast path
carries is not a plan; ``served_arms`` leaves such arms out of a table.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Tuple

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)

#: The ids each fast path carries.
PATH_IDS: Dict[str, FrozenSet[str]] = {
    "preattn_prep": frozenset({"preattn_prep"}),
    "swa_paged_decode": frozenset({"paged_decode_q8"}),
    "swa_block": frozenset({"swa_prewait", "swa_fused_prep", "swa_fused_gate"}),
    "full_attention_prep_append": frozenset({"prep_append"}),
    "full_attention_split_kv": frozenset({"split_kv", "spill_free"}),
    "full_attention_prep_fold": frozenset({"prep_fold"}),
    "dense_gemms": frozenset({"dense_gemms"}),
    "dense_gemm_prefetch": frozenset({"gemm_prefetch", "prep_append_prefetch_trim"}),
    "next_qkv_prefetch": frozenset({"qkv_prefetch"}),
    "post_attention_hop": frozenset({"hop", "hop_early_trigger", "stale_poll"}),
    "attention_output_into_hop": frozenset({"direct_publish"}),
    "projection_gathers": frozenset({"gathers"}),
    "boundary_early_release": frozenset({"early_release"}),
    "unpool_into_boundary": frozenset({"unpool_direct"}),
    "capture_fold": frozenset({"capture_fold"}),
    "moe_prologue": frozenset({"moe_prologue", "router_fork"}),
    "pool_into_route": frozenset({"pool_direct"}),
    "replicated_experts": frozenset({"ring_buddy"}),
    "three_window_placement": frozenset({"three_windows"}),
    "persistent_experts": frozenset({"persistent_experts"}),
    "bf16_routed_experts": frozenset({"bf16_routing"}),
    "early_expert_ids": frozenset({"early_ids"}),
    "finalize_transport": frozenset({"finalize_transport"}),
    "replicated_embedding": frozenset({"replicated_embedding"}),
    "split_sampler": frozenset({"split_sampler"}),
    "lm_head_inbox": frozenset({"lm_head_inbox"}),
    "context_kv_fork": frozenset({"context_kv_fork"}),
    "fused_accept": frozenset({"fused_accept"}),
    "verify_fold": frozenset({"verify_fold"}),
    "path_walk_tail": frozenset({"path_walk_tail"}),
    "drafter_block_conv": frozenset({"block_conv"}),
    "drafter_block_decode": frozenset({"block_decode"}),
    "draft_head": frozenset({"draft_head"}),
    "drafter_fc_shard": frozenset({"fc_shard"}),
    "drafter_fixed_gemms": frozenset({"fixed_gemms"}),
    "drafter_hops": frozenset({"drafter_hops_early"}),
    "drafter_pdl_chain": frozenset(
        {"pdl_conv", "pdl_merge", "pdl_silu", "pdl_head", "selector_fork"}
    ),
    "deferred_input_prep": frozenset({"input_staging"}),
    "sampler_store": frozenset({"input_staging"}),
}
assert set(PATH_IDS) == dispatch.FAST_PATH_NAMES

#: Ids that name several fast paths' ids at once.
_UMBRELLAS = {
    "swa_block": ("swa_prewait", "swa_fused_prep", "swa_fused_gate"),
    "drafter_pdl": ("pdl_conv", "pdl_merge", "pdl_silu", "pdl_head", "selector_fork"),
}

_ID_PATHS: Dict[str, Tuple[str, ...]] = {}
for _path, _ids in PATH_IDS.items():
    for _id in _ids:
        _ID_PATHS[_id] = _ID_PATHS.get(_id, ()) + (_path,)

#: The ids the load-time expert fast path carries.
LOADED = PATH_IDS["replicated_experts"]


class NotAPlan(ValueError):
    """An arm names an id no fast path carries."""


def _tokens(value: str) -> FrozenSet[str]:
    ids = set()
    for token in value.split(","):
        token = token.strip().lower()
        if token:
            ids.update(_UMBRELLAS.get(token, (token,)))
    return frozenset(ids)


def plan_of_ids(value: Optional[str]) -> Optional["dispatch.Plan"]:
    """The plan an arm's ids select; None for the built-in model (``value`` None).

    Raises:
        NotAPlan: when an id is carried by no fast path.
    """
    if value is None:
        return None
    ids = _tokens(value)
    unknown = sorted(ids - set(_ID_PATHS))
    if unknown:
        raise NotAPlan(f"no fast path carries {unknown}")
    return dispatch.plan_of(*(path for token in ids for path in _ID_PATHS[token]))


def ids_of(plan: Optional["dispatch.Plan"]) -> FrozenSet[str]:
    """Every id ``plan``'s fast paths carry (empty for None)."""
    if plan is None:
        return frozenset()
    return frozenset().union(*(PATH_IDS[path] for path in plan.paths))


def selection(value: Optional[str]) -> FrozenSet[str]:
    """``ids_of(plan_of_ids(value))``: what an arm runs, as ids."""
    return ids_of(plan_of_ids(value))


def is_plan(value: Optional[str]) -> bool:
    try:
        plan_of_ids(value)
    except NotAPlan:
        return False
    return True


def served_arms(arms: Iterable[tuple]) -> tuple:
    """``arms`` without those whose ids are not a plan; a reference to a left-out arm is None."""
    kept = [arm for arm in arms if is_plan(arm[1])]
    names = {arm[0] for arm in kept}
    return tuple(
        (arm[0], arm[1], arm[2] if arm[2] in names else None) + tuple(arm[3:]) for arm in kept
    )

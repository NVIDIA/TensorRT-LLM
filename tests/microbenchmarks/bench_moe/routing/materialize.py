# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialise a :class:`RoutingPlan` into runtime tensors and observers.

The materialiser:

* turns the per-rank slot dispatch matrix into a flat list of expert ids,
* repacks it column-major so each token row spans different destinations,
* repacks it group-aware instead for grouped routing methods (DeepSeek-V3 style
  ``noaux_tc``), keeping every token row within ``topk_group`` expert groups so
  the routing kernel can realise the plan, and falling back to the column-major
  packing when the requested histogram cannot satisfy that constraint,
* runs a small repair pass to enforce per-token expert uniqueness,
* derives uniform top-k scales,
* observes the realised plan to compute slot / token traffic and per-rank
  expert histograms used for the result schema's accuracy metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

from .builders import RoutingPlan, _largest_remainder_split


def _split_slot_count_to_experts(
    slot_count: int,
    target_histogram_row: List[int],
) -> List[int]:
    """Allocate ``slot_count`` slots across local experts proportionally.

    Largest-remainder over ``target_histogram_row`` ensures the per-local-expert
    distribution within this (src, dst) cell tracks the global histogram for
    the target rank. Returns a list of length ``len(target_histogram_row)``.
    """
    weights = [float(v) for v in target_histogram_row]
    return _largest_remainder_split(int(slot_count), weights)


def _flatten_plan_slots_for_rank(
    plan: RoutingPlan,
    src_rank: int,
    top_k: int,
    experts_per_rank: int,
    moe_ep_size: int,
) -> List[int]:
    """Flatten one plan row into expert ids while preserving slot counts.

    ``local_num_tokens`` is derived from the dispatch-matrix row sum rather
    than from ``per_rank_num_tokens[src_rank]``.  In MoE-TP + attention-DP
    layouts (DTP / CUSTOM-DP) the dispatch matrix is EP-axis indexed while
    ``per_rank_num_tokens`` is world-rank indexed; the row sum is always the
    correct EP-axis aggregate (``source_tokens[src_rank] * top_k``).
    """
    row = list(plan.dispatch_matrix[src_rank])
    row_sum = sum(row)
    if top_k > 0 and row_sum % top_k != 0:
        raise ValueError(
            f"dispatch_matrix row {src_rank} sum ({row_sum}) is not divisible by top_k ({top_k})"
        )
    local_num_tokens = row_sum // top_k if top_k > 0 else 0

    flat: List[int] = []
    for dst in range(moe_ep_size):
        cell = int(row[dst])
        if cell == 0:
            continue
        target_hist = list(plan.expert_histogram[dst])
        per_le = _split_slot_count_to_experts(cell, target_hist)
        for le, cnt in enumerate(per_le):
            if cnt <= 0:
                continue
            expert_id = dst * experts_per_rank + le
            flat.extend([expert_id] * int(cnt))

    expected = local_num_tokens * top_k
    if len(flat) != expected:
        raise ValueError(
            f"materialiser flat length {len(flat)} != local_num_tokens*top_k={expected}"
        )
    return flat


def _pack_slots_column_major(flat: List[int], local_num_tokens: int, top_k: int) -> List[List[int]]:
    """Pack flat slots as k-major columns to spread destinations across tokens."""
    out = [[0] * top_k for _ in range(local_num_tokens)]
    for i, val in enumerate(flat):
        k_idx = i // local_num_tokens
        t_idx = i % local_num_tokens
        out[t_idx][k_idx] = val
    return out


def _pack_slots_group_aware(
    flat: List[int],
    local_num_tokens: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    num_experts: int,
) -> Optional[List[List[int]]]:
    """Pack flat slots so every token row spans at most ``topk_group`` groups.

    Grouped routing methods (DeepSeek-V3 style ``noaux_tc``) score a group by
    the sum of its top-2 expert scores and then keep only ``topk_group``
    groups. The high/low logits built by ``_project_router_logits_for_plan``
    give a group with >=2 selected experts a score of ``2*sigmoid(high)``, a
    group with exactly one ``sigmoid(high) + sigmoid(low)``, and an unselected
    group ``2*sigmoid(low)``. Column-major packing spreads a token's ``top_k``
    slots over every group, so all groups tie on the middle value and the
    kernel's tie-break decides which ``topk_group`` survive -- collapsing the
    load onto the lowest-indexed groups (and therefore the lowest-indexed EP
    ranks) no matter what the plan asked for.

    Packing each token into at most ``topk_group`` groups instead makes the
    selected groups score strictly above the unselected ones, so the routing
    kernel reproduces the plan exactly.

    Each token takes the group with the most unused experts left plus the
    ``topk_group - 1`` emptiest non-empty groups, then draws experts from those
    round-robin. Pairing the fullest group with the emptiest ones drains
    stragglers while a group that can still carry the rest is open; picking
    groups by a rotation fixed up front instead keeps landing on groups that
    are already empty, and gives up on layouts that are in fact packable --
    most visibly for small ``local_num_tokens``, where a balanced plan only
    populates a fraction of the experts. Equally loaded groups are ordered by a
    per-token rotation, so a balanced plan -- where capacity never decides --
    still spreads consecutive tokens over different groups.

    ``flat`` is consumed as a multiset, so the realised per-expert slot counts
    still match ``plan.expert_histogram`` exactly.

    The packing is greedy and does not backtrack, so it returns ``None`` for a
    histogram it cannot place -- always for a genuinely unsatisfiable one (a
    hotspot needing more groups per token than ``topk_group``), and in rare
    cases for one a full search could still place. The caller then falls back
    to column-major packing, and ``_classify_native_projection`` reports
    ``status="projected"`` because it inspects the materialised ids, so the
    fallback is never silent.
    """
    if local_num_tokens <= 0 or top_k <= 0 or n_group <= 1:
        return None
    if num_experts % n_group != 0:
        return None
    experts_per_group = num_experts // n_group
    # A token cannot need more slots than ``topk_group`` groups can supply
    # with distinct experts.
    if top_k > topk_group * experts_per_group:
        return None

    # Remaining slot count per expert, bucketed by group.
    pools: List[Dict[int, int]] = [{} for _ in range(n_group)]
    for eid in flat:
        g = eid // experts_per_group
        if g >= n_group:
            return None
        pools[g][eid] = pools[g].get(eid, 0) + 1

    out: List[List[int]] = []
    for t in range(local_num_tokens):
        live = [g for g in range(n_group) if pools[g]]
        if not live:
            return None

        def rotated(g: int, _t: int = t) -> int:
            return (g - _t) % n_group

        fullest = max(live, key=lambda g: (len(pools[g]), -rotated(g)))
        others = sorted(
            (g for g in live if g != fullest), key=lambda g: (len(pools[g]), rotated(g))
        )
        chosen = [fullest] + others[: topk_group - 1]

        row: List[int] = []
        while len(row) < top_k:
            progressed = False
            for g in chosen:
                if len(row) == top_k:
                    break
                eid = _take_loaded_expert(pools[g], exclude=set(row))
                if eid is not None:
                    row.append(eid)
                    progressed = True
            if not progressed:
                break
        if len(row) != top_k or len(set(row)) != top_k:
            return None
        out.append(row)

    if any(sum(pool.values()) > 0 for pool in pools):
        # Slots left over means the rows do not reproduce the histogram.
        return None
    return out


def _take_loaded_expert(pool: Dict[int, int], exclude: Set[int]) -> Optional[int]:
    """Take the expert with the most remaining slots, skipping ``exclude``.

    Draining the experts with the largest remaining slot count first keeps the
    per-expert histogram from developing a tail of leftovers that no token can
    absorb. ``pool`` is mutated in place; ``None`` means nothing usable is left.
    """
    best = None
    for eid, count in pool.items():
        if eid in exclude:
            continue
        if best is None or count > pool[best] or (count == pool[best] and eid < best):
            best = eid
    if best is None:
        return None
    pool[best] -= 1
    if pool[best] == 0:
        del pool[best]
    return best


def _repair_duplicate_experts(out: List[List[int]], top_k: int) -> None:
    """Best-effort repair so each token row has distinct selected experts."""
    max_passes = 4
    local_num_tokens = len(out)
    for _pass in range(max_passes):
        any_repair = False
        for t in range(local_num_tokens):
            seen: Dict[int, int] = {}
            for k in range(top_k):
                eid = out[t][k]
                if eid in seen:
                    # Prefer swapping with the same k slot in another row; this
                    # preserves per-k distribution better than reshuffling the row.
                    target_k = k
                    swapped = False
                    for t2 in range(local_num_tokens):
                        if t2 == t:
                            continue
                        partner = out[t2][target_k]
                        if partner == eid:
                            continue
                        if partner in seen:
                            continue
                        out[t][target_k], out[t2][target_k] = partner, eid
                        swapped = True
                        any_repair = True
                        break
                    if not swapped:
                        # Last-resort intra-row swap. Some pathological plans
                        # cannot be repaired, and tests intentionally document
                        # those duplicate-producing cases.
                        for k2 in range(top_k):
                            if k2 == k:
                                continue
                            alt = out[t][k2]
                            if alt == eid or alt in seen:
                                continue
                            out[t][k], out[t][k2] = alt, eid
                            any_repair = True
                            break
                    seen[out[t][k]] = k
                else:
                    seen[eid] = k
        if not any_repair:
            break


def _make_uniform_topk_scales(
    local_num_tokens: int,
    top_k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.full((local_num_tokens, top_k), 1.0 / max(top_k, 1), dtype=dtype, device=device)


def _materialize_selected_experts_for_rank(
    plan: RoutingPlan,
    src_rank: int,
    top_k: int,
    experts_per_rank: int,
    moe_ep_size: int,
    device: torch.device,
    scale_dtype: torch.dtype,
    group_constraint: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise ``[local_num_tokens, top_k]`` expert ids + uniform scales.

    The algorithm:
      1. Flatten ``dispatch_matrix[src_rank]`` into a slot-count-per-(dst, le)
         table by splitting the row counts across local experts proportional
         to the target rank's global histogram.
      2. Build a flat list of expert ids of length ``local_num_tokens * top_k``.
      3. Reshape column-major (k=0 first across tokens, then k=1, ...) so
         that within a row consecutive slots come from different "buckets" and
         per-token expert ids stay distinct in practice.
      4. Run a small repair pass that swaps duplicated expert ids between
         rows until each token has ``top_k`` distinct experts.

    ``group_constraint`` is ``(n_group, topk_group)`` for routing methods that
    enforce DeepSeek-V3 style expert grouping. When set, step 3 is replaced by
    :func:`_pack_slots_group_aware`, which keeps each token inside at most
    ``topk_group`` groups so the routing kernel can realise the plan exactly
    (see that function for why column-major packing cannot). The group-aware
    packer falls back to the column-major path when the requested histogram
    cannot satisfy the constraint.
    """
    # Derive the effective token count from the dispatch-matrix row sum so that
    # MoE-TP + attention-DP layouts (DTP / CUSTOM-DP) are handled correctly.
    # In those layouts the row sum equals the aggregated source tokens for the
    # EP rank, while per_rank_num_tokens[src_rank] would only reflect one DP
    # shard's contribution.
    row_sum = sum(plan.dispatch_matrix[src_rank])
    local_num_tokens = row_sum // max(top_k, 1)
    if local_num_tokens == 0:
        ids = torch.zeros((0, top_k), dtype=torch.int32, device=device)
        scales = torch.zeros((0, top_k), dtype=scale_dtype, device=device)
        return ids, scales

    flat = _flatten_plan_slots_for_rank(plan, src_rank, top_k, experts_per_rank, moe_ep_size)
    out = None
    if group_constraint is not None:
        n_group, topk_group = group_constraint
        out = _pack_slots_group_aware(
            list(flat),
            local_num_tokens,
            top_k,
            int(n_group),
            int(topk_group),
            num_experts=int(experts_per_rank) * int(moe_ep_size),
        )
    if out is None:
        out = _pack_slots_column_major(flat, local_num_tokens, top_k)
        _repair_duplicate_experts(out, top_k)

    ids = torch.tensor(out, dtype=torch.int32, device=device)
    scales = _make_uniform_topk_scales(local_num_tokens, top_k, device=device, dtype=scale_dtype)
    return ids, scales


def _observe_routing_metrics(
    plan: RoutingPlan,
    selected_experts_per_rank: List[torch.Tensor],
    experts_per_rank: int,
    moe_ep_size: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Derive observed slot/token traffic and expert histogram from materialised ids."""
    slot_traffic = [[0] * moe_ep_size for _ in range(moe_ep_size)]
    token_traffic = [[0] * moe_ep_size for _ in range(moe_ep_size)]
    expert_hist = [[0] * experts_per_rank for _ in range(moe_ep_size)]
    for src, ids in enumerate(selected_experts_per_rank):
        if ids is None or ids.numel() == 0:
            continue
        ids_cpu = ids.detach().cpu().numpy() if not isinstance(ids, list) else ids
        for row in ids_cpu:
            dst_visited = set()
            for eid in row:
                eid_int = int(eid)
                dst = eid_int // experts_per_rank
                le = eid_int % experts_per_rank
                if 0 <= dst < moe_ep_size and 0 <= le < experts_per_rank:
                    slot_traffic[src][dst] += 1
                    expert_hist[dst][le] += 1
                    if dst not in dst_visited:
                        token_traffic[src][dst] += 1
                        dst_visited.add(dst)
    return slot_traffic, token_traffic, expert_hist


def _observe_summary(
    requested_slot: List[List[int]],
    observed_slot: List[List[int]],
) -> Tuple[int, float]:
    """Return ``(max_abs_slot_error, max_relative_slot_error)``."""
    max_abs = 0
    max_rel = 0.0
    for src in range(len(observed_slot)):
        for dst in range(len(observed_slot[src])):
            req = int(requested_slot[src][dst]) if src < len(requested_slot) else 0
            obs = int(observed_slot[src][dst])
            abs_err = abs(obs - req)
            if abs_err > max_abs:
                max_abs = abs_err
            denom = max(req, 1)
            rel_err = abs_err / denom
            if rel_err > max_rel:
                max_rel = rel_err
    return int(max_abs), float(max_rel)

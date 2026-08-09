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
* runs a small repair pass to enforce per-token expert uniqueness,
* derives uniform top-k scales,
* observes the realised plan to compute slot / token traffic and per-rank
  expert histograms used for the result schema's accuracy metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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

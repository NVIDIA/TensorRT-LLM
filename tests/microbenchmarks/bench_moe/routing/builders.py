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

"""Construction of the canonical :class:`RoutingPlan`.

The builders translate :class:`RoutingControlSpec` plus the runtime topology
into the normalised plan shape (per-rank token counts, slot dispatch matrix,
expert histogram) consumed by the materialiser. JSON sidecar loaders for the
``--routing_pattern_file`` knob also live here so all plan-construction paths
share the same validation surface.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..specs import RoutingControlSpec, WorkloadSpec, _to_jsonable_dict
from ..utils import _distribute_tokens, _validate_per_rank_token_list
from .parsing import _parse_comm_pattern, _parse_expert_pattern


@dataclass(frozen=True)
class RoutingPlan:
    """Canonical normalised routing plan.

    ``per_rank_num_tokens[src]`` is the local input token count on source rank
    ``src``. ``dispatch_matrix[src][dst]`` is the *slot* count (each selected
    (token, expert) slot counts once) sent from ``src`` to ``dst``. Row sums
    are ``per_rank_num_tokens[src] * top_k``. ``expert_histogram[dst][le]`` is
    the global slot count owned by local expert ``le`` on rank ``dst``.
    """

    per_rank_num_tokens: Tuple[int, ...]
    dispatch_matrix: Tuple[Tuple[int, ...], ...]
    expert_histogram: Tuple[Tuple[int, ...], ...]
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable_dict(self)


@dataclass(frozen=True)
class RoutingProjectionResult:
    """Outcome of trying to realise a ``RoutingPlan`` for a given routing method."""

    router_logits: Optional[torch.Tensor]
    status: str  # "exact" | "projected" | "rejected" | "forced_exact" | "not_applicable"
    reason: str
    observed_slot_dispatch_matrix: Tuple[Tuple[int, ...], ...]
    observed_token_dispatch_matrix: Tuple[Tuple[int, ...], ...]
    observed_expert_histogram: Tuple[Tuple[int, ...], ...]
    max_abs_slot_error: int
    max_relative_slot_error: float
    selected_experts: Optional[torch.Tensor]
    selected_scales: Optional[torch.Tensor]
    warnings: Tuple[str, ...] = ()


def _largest_remainder_split(total: int, weights: List[float]) -> List[int]:
    """Split ``total`` integer units among bins using largest-remainder method.

    All weights must be non-negative. Zero-weight bins always receive zero
    units. The result is deterministic for ties (ties broken by lower index).
    """
    n = len(weights)
    if n == 0:
        return []
    total = int(total)
    if total <= 0:
        return [0] * n
    s = sum(weights)
    if s <= 0.0:
        # Distribute evenly when all weights are zero.
        base = total // n
        rem = total - base * n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    raw = [total * (w / s) for w in weights]
    floors = [int(x) for x in raw]
    used = sum(floors)
    remainders = sorted(
        ((raw[i] - floors[i], -i) for i in range(n)), key=lambda pair: pair[0], reverse=True
    )
    for k in range(total - used):
        _, neg_idx = remainders[k % n]
        floors[-neg_idx] += 1
    return floors


def _random_weights(n: int, *, seed: int, label: str, index: int) -> List[float]:
    rng = random.Random(f"bench_moe:{int(seed)}:{label}:{int(index)}")
    return [rng.random() for _ in range(n)]


def _build_per_rank_num_tokens(
    spec: RoutingControlSpec,
    num_tokens: int,
    world_size: int,
    enable_dp: bool,
) -> List[int]:
    """Resolve ``per_rank_num_tokens`` for a workload.

    Explicit ``spec.per_rank_num_tokens`` wins; otherwise the token count per
    rank depends on the attention-DP setting:

    * ``enable_dp=True``  (DEP / DTP): tokens are DP-sharded across ranks, so
      each rank holds ``num_tokens / world_size``.
    * ``enable_dp=False`` (TEP / TTP): attention is tensor-parallel, so every
      rank sees the complete batch and holds ``num_tokens``.

    When an explicit list is provided its sum is validated against the expected
    total (``num_tokens`` for DP modes, ``num_tokens * world_size`` for non-DP).
    """
    if spec.per_rank_num_tokens is None:
        if not enable_dp:
            return [int(num_tokens)] * world_size
        return _distribute_tokens(int(num_tokens), world_size)
    expected_total = int(num_tokens) * (1 if enable_dp else world_size)
    return _validate_per_rank_token_list(
        spec.per_rank_num_tokens, world_size=world_size, expected_total=expected_total
    )


def _per_rank_tokens(workload: WorkloadSpec, world_size: int, enable_dp: bool) -> List[int]:
    """Materialize the ``per_rank_num_tokens`` list for a workload + world size."""
    return _build_per_rank_num_tokens(
        workload.routing_control, int(workload.num_tokens), world_size, enable_dp
    )


def _aggregate_dispatch_source_tokens(
    per_rank_num_tokens: List[int],
    ep_size: int,
    enable_dp: bool,
) -> List[int]:
    """Project world-rank token counts onto EP-source rows.

    TRT-LLM Mapping orders MoE ranks with ``moe_ep_rank = tp_rank % moe_ep_size``.
    In attention-DP modes each world rank owns a distinct token shard, so TP
    shards targeting the same EP row are summed. In non-DP MoE-TP modes those TP
    shards carry the same logical tokens, so only the first TP shard contributes
    to the logical dispatch plan.
    """
    if ep_size <= 0:
        return []
    if len(per_rank_num_tokens) == ep_size:
        return [int(v) for v in per_rank_num_tokens]

    source_tokens = [0] * ep_size
    if not enable_dp:
        for ep_rank in range(ep_size):
            if ep_rank < len(per_rank_num_tokens):
                source_tokens[ep_rank] = int(per_rank_num_tokens[ep_rank])
    else:
        for rank, num_tokens in enumerate(per_rank_num_tokens):
            source_tokens[rank % ep_size] += int(num_tokens)

    return source_tokens


def _build_dispatch_matrix(
    comm_pattern: str,
    per_rank_num_tokens: List[int],
    top_k: int,
    ep_size: int,
    enable_dp: bool,
    seed: int = 0,
) -> List[List[int]]:
    """Build the canonical slot ``dispatch_matrix`` for ``comm_pattern``.

    Row sums equal the EP-source token counts projected from
    ``per_rank_num_tokens`` times ``top_k``. When world ranks outnumber EP
    ranks (DTP / TTP / CUSTOM MoE-TP layouts), multiple world-rank rows are
    aggregated onto the same EP-source row.
    """
    name, kwargs = _parse_comm_pattern(comm_pattern)
    source_tokens = _aggregate_dispatch_source_tokens(per_rank_num_tokens, ep_size, enable_dp)
    matrix: List[List[int]] = [[0] * ep_size for _ in range(ep_size)]
    for src in range(ep_size):
        row_total = int(source_tokens[src]) * int(top_k)
        if row_total == 0:
            continue
        if name == "file":
            # Loaded separately by ``_load_dispatch_matrix_file``.
            raise ValueError(
                "file:<path> dispatch matrices are loaded via _load_dispatch_matrix_file"
            )
        elif name == "random":
            weights = _random_weights(ep_size, seed=seed, label="comm", index=src)
        elif name == "balanced_alltoall":
            weights = [1.0] * ep_size
        elif name == "receiver_hotspot":
            hot_rank = int(kwargs["rank"])
            if not 0 <= hot_rank < ep_size:
                raise ValueError(f"receiver_hotspot rank={hot_rank} out of range [0, {ep_size})")
            hotness = float(kwargs["hotness"])
            weights = [(1.0 - hotness) / max(ep_size, 1)] * ep_size
            weights[hot_rank] += hotness
        elif name == "pair_hotspot":
            pair_src = int(kwargs["src"])
            pair_dst = int(kwargs["dst"])
            if not 0 <= pair_src < ep_size or not 0 <= pair_dst < ep_size:
                raise ValueError(
                    f"pair_hotspot src/dst must be in [0, {ep_size}); got src={pair_src}, dst={pair_dst}"
                )
            hotness = float(kwargs["hotness"])
            if src == pair_src:
                weights = [(1.0 - hotness) / max(ep_size, 1)] * ep_size
                weights[pair_dst] += hotness
            else:
                weights = [1.0] * ep_size
        elif name == "local_only":
            weights = [0.0] * ep_size
            weights[src] = 1.0
        elif name == "ring":
            weights = [0.0] * ep_size
            weights[(src + 1) % ep_size] = 1.0
        else:
            raise ValueError(f"unknown comm_pattern {name!r}")
        matrix[src] = _largest_remainder_split(row_total, weights)
    return matrix


def _build_expert_histogram(
    expert_pattern: str,
    dispatch_matrix: List[List[int]],
    experts_per_rank: int,
    ep_size: int,
    seed: int = 0,
) -> List[List[int]]:
    """Build the canonical global ``expert_histogram[ep_size][experts_per_rank]``."""
    name, kwargs = _parse_expert_pattern(expert_pattern)
    histogram: List[List[int]] = [[0] * experts_per_rank for _ in range(ep_size)]
    # Per-target slot totals come from column sums of the dispatch matrix.
    col_sums = [sum(dispatch_matrix[src][dst] for src in range(ep_size)) for dst in range(ep_size)]
    for dst in range(ep_size):
        target_total = int(col_sums[dst])
        if target_total <= 0:
            continue
        if name == "file":
            raise ValueError(
                "file:<path> expert histograms are loaded via _load_expert_histogram_file"
            )
        elif name == "random":
            weights = _random_weights(experts_per_rank, seed=seed, label="expert", index=dst)
        elif name == "balanced":
            weights = [1.0] * experts_per_rank
        elif name == "hotspot":
            if "active_experts" in kwargs:
                active = int(kwargs["active_experts"])
                active = max(1, min(active, experts_per_rank))
                weights = [1.0 if le < active else 0.0 for le in range(experts_per_rank)]
            else:
                hotness = float(kwargs["hotness"])
                # Concentrate ``hotness`` fraction onto expert 0; spread the
                # remainder uniformly.
                weights = [(1.0 - hotness) / max(experts_per_rank, 1)] * experts_per_rank
                weights[0] += hotness
        else:
            raise ValueError(f"unknown expert_pattern {name!r}")
        histogram[dst] = _largest_remainder_split(target_total, weights)
    return histogram


def _load_2d_matrix_json(
    path: str,
    *,
    matrix_key: str,
    n_rows: int,
    n_cols: int,
    label: str,
    extra_dims: Tuple[Tuple[str, int], ...] = (),
) -> List[List[int]]:
    """Load and validate a 2D integer matrix from a JSON sidecar.

    ``extra_dims`` is a list of ``(payload_key, expected_value)`` checks that
    happen alongside the always-present ``ep_size`` guard, so the dispatch and
    histogram loaders share their full structural-validation pipeline.
    """
    with open(path) as f:
        payload = json.load(f)
    for key, expected in (("ep_size", n_rows),) + tuple(extra_dims):
        if int(payload.get(key, -1)) != int(expected):
            raise ValueError(
                f"{label} {path!r}: {key}={payload.get(key)} mismatches "
                f"runtime {key}={int(expected)}"
            )
    matrix = payload.get(matrix_key)
    if (
        not isinstance(matrix, list)
        or len(matrix) != n_rows
        or any(not isinstance(row, list) or len(row) != n_cols for row in matrix)
    ):
        raise ValueError(f"{label} {path!r}: {matrix_key} must be a {n_rows}x{n_cols} integer list")
    return [[int(v) for v in row] for row in matrix]


def _load_dispatch_matrix_file(path: str, ep_size: int) -> List[List[int]]:
    """Load and validate a ``file:<path>`` slot dispatch matrix."""
    return _load_2d_matrix_json(
        path,
        matrix_key="slot_dispatch_matrix",
        n_rows=ep_size,
        n_cols=ep_size,
        label="dispatch matrix",
    )


def _load_expert_histogram_file(path: str, ep_size: int, experts_per_rank: int) -> List[List[int]]:
    """Load and validate a ``file:<path>`` expert histogram."""
    return _load_2d_matrix_json(
        path,
        matrix_key="expert_histogram",
        n_rows=ep_size,
        n_cols=experts_per_rank,
        label="expert histogram",
        extra_dims=(("experts_per_rank", experts_per_rank),),
    )


def _load_routing_pattern_file(
    path: str, ep_size: int, experts_per_rank: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Load a single file that fixes both dispatch traffic and expert histogram."""
    return (
        _load_dispatch_matrix_file(path, ep_size),
        _load_expert_histogram_file(path, ep_size, experts_per_rank),
    )


def _build_routing_plan(
    spec: RoutingControlSpec,
    num_tokens: int,
    world_size: int,
    top_k: int,
    num_experts: int,
    moe_ep_size: int,
    enable_dp: bool,
) -> RoutingPlan:
    """Translate a ``RoutingControlSpec`` into a canonical normalised plan."""
    if moe_ep_size <= 0 or num_experts % moe_ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
        )
    experts_per_rank = num_experts // moe_ep_size
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    per_rank = _build_per_rank_num_tokens(spec, num_tokens, world_size, enable_dp)
    # The dispatch matrix stays on EP axes. When MoE-TP makes multiple world
    # ranks share one EP rank, the world-rank token counts are aggregated onto
    # the corresponding EP-source row before building the matrix.
    if spec.routing_pattern_file:
        default_patterns = {("balanced_alltoall", "balanced"), ("random", "random")}
        if (spec.comm_pattern, spec.expert_pattern) not in default_patterns:
            raise ValueError(
                "--routing_pattern_file cannot be combined with non-default "
                "--comm_pattern or --expert_pattern"
            )
        dispatch_matrix, expert_histogram = _load_routing_pattern_file(
            spec.routing_pattern_file, moe_ep_size, experts_per_rank
        )
    else:
        dispatch_matrix = _build_dispatch_matrix(
            spec.comm_pattern,
            per_rank,
            top_k,
            moe_ep_size,
            enable_dp=enable_dp,
            seed=spec.seed,
        )
        expert_histogram = _build_expert_histogram(
            spec.expert_pattern, dispatch_matrix, experts_per_rank, moe_ep_size, seed=spec.seed
        )

    # Per-row sums are an invariant; emit a clearer error than the materialiser would.
    source_tokens = _aggregate_dispatch_source_tokens(per_rank, moe_ep_size, enable_dp)
    for src in range(moe_ep_size):
        expected = int(source_tokens[src]) * int(top_k) if src < len(source_tokens) else 0
        actual = sum(dispatch_matrix[src])
        if actual != expected:
            raise ValueError(
                f"dispatch_matrix row {src} sums to {actual}, expected aggregate source tokens * top_k = {expected}"
            )
    # Global expert histogram total must match total slots.
    total_slots = sum(int(t) for t in source_tokens) * int(top_k)
    hist_total = sum(sum(row) for row in expert_histogram)
    if hist_total != total_slots:
        raise ValueError(
            f"expert_histogram sum={hist_total} must equal aggregate source tokens * top_k = {total_slots}"
        )

    return RoutingPlan(
        per_rank_num_tokens=tuple(int(v) for v in per_rank),
        dispatch_matrix=tuple(tuple(int(v) for v in row) for row in dispatch_matrix),
        expert_histogram=tuple(tuple(int(v) for v in row) for row in expert_histogram),
        seed=int(spec.seed),
    )

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

"""Result scoring, bottleneck labelling, and row serialization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm._utils import mpi_allgather

from .mapping import _resolve_mapping_layout
from .routing import _per_rank_tokens
from .specs import ConfigSpec, ModelSpec, RunResult, WorkloadSpec
from .utils import _compute_stats

_ROBUST_SCORE_MIN_SAMPLES = 8
_ROBUST_SCORE_MAD_Z_THRESHOLD = 3.5
_ROBUST_SCORE_ZERO_SPREAD_REL_TOL = 0.05


def _gather_per_iteration_times(times_ms: List[float]) -> List[List[float]]:
    """All-gather raw per-iteration latencies; returns ``[ [rank0_iters], ... ]``."""
    return mpi_allgather(times_ms)


def _slowest_rank_iter_maxes(per_rank_iters: List[List[float]]) -> List[float]:
    """Return ``max_r(latency_ms[rank=r][iteration=i])`` for each common iteration.

    Falls back gracefully when ranks reported different iteration counts; the
    common length is used and trailing entries are ignored.
    """
    if not per_rank_iters:
        return []
    lengths = [len(r) for r in per_rank_iters if r]
    if not lengths:
        return []
    n = min(lengths)
    if n == 0:
        return []
    iter_max: List[float] = []
    for i in range(n):
        per_iter_vals = [r[i] for r in per_rank_iters if i < len(r)]
        iter_max.append(max(per_iter_vals))
    return iter_max


def _slowest_rank_mean_score(per_rank_iters: List[List[float]]) -> float:
    """Compute raw ``mean_i(max_r(latency_ms[rank=r][iteration=i]))``."""
    iter_max = _slowest_rank_iter_maxes(per_rank_iters)
    return sum(iter_max) / len(iter_max) if iter_max else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _robust_slowest_rank_score(iter_max: List[float]) -> Tuple[float, str, Dict[str, Any]]:
    """Score distributed latency while reporting, not hiding, extreme samples."""
    if not iter_max:
        return (
            0.0,
            "slowest_rank_trimmed_mean",
            {
                "method": "modified_z_score_mad",
                "count": 0,
                "samples_used": 0,
                "samples_total": 0,
                "items": [],
            },
        )

    if len(iter_max) < _ROBUST_SCORE_MIN_SAMPLES:
        return (
            _median(iter_max),
            "slowest_rank_median",
            {
                "method": "disabled_insufficient_samples",
                "count": 0,
                "samples_used": len(iter_max),
                "samples_total": len(iter_max),
                "items": [],
            },
        )

    center = _median(iter_max)
    deviations = [abs(v - center) for v in iter_max]
    mad = _median(deviations)
    outliers: List[Dict[str, Any]] = []
    keep: List[float] = []

    for idx, value in enumerate(iter_max):
        diff = value - center
        score: Optional[float]
        if mad > 0.0:
            score = 0.6745 * diff / mad
            is_outlier = abs(score) > _ROBUST_SCORE_MAD_Z_THRESHOLD
        else:
            tolerance = max(abs(center) * _ROBUST_SCORE_ZERO_SPREAD_REL_TOL, 1.0e-12)
            score = None
            is_outlier = abs(diff) > tolerance

        if is_outlier:
            outliers.append(
                {
                    "index": int(idx),
                    "value": float(value),
                    "center": float(center),
                    "modified_z_score": float(score) if score is not None else None,
                    "absolute_deviation": float(abs(diff)),
                    "direction": "high" if diff > 0 else "low",
                }
            )
        else:
            keep.append(value)

    samples = keep if keep else iter_max
    score = sum(samples) / len(samples)
    return (
        float(score),
        "slowest_rank_trimmed_mean",
        {
            "method": "modified_z_score_mad",
            "threshold": float(_ROBUST_SCORE_MAD_Z_THRESHOLD),
            "center": float(center),
            "mad": float(mad),
            "count": len(outliers),
            "samples_used": len(samples),
            "samples_total": len(iter_max),
            "items": outliers,
        },
    )


def _build_latency_block(per_rank_iters: List[List[float]]) -> Dict[str, Any]:
    iter_max = _slowest_rank_iter_maxes(per_rank_iters)
    raw_score = sum(iter_max) / len(iter_max) if iter_max else 0.0
    score, score_type, outliers = _robust_slowest_rank_score(iter_max)
    return {
        "score": float(score),
        "score_type": score_type,
        "raw_score": float(raw_score),
        "raw_score_type": "slowest_rank_mean",
        "iter_max_stats": _compute_stats(iter_max),
        "iter_max_outliers": outliers,
        "per_rank": {f"rank{i}": _compute_stats(times) for i, times in enumerate(per_rank_iters)},
    }


def _gather_kernel_breakdown(
    detailed_stats: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """All-gather per-kernel timings and produce per-rank summary stats."""
    kernel_breakdown, _raw_kernel_times = _gather_kernel_timing_blocks(detailed_stats)
    return kernel_breakdown


def _kernel_times_payload(detailed_stats: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    categories = ("moe_forward_kernels", "other_kernels")
    local_payload: Dict[str, Dict[str, List[float]]] = {}
    for cat in categories:
        local_payload[cat] = {
            kernel["name"]: kernel.get("_times", []) for kernel in detailed_stats.get(cat, [])
        }
    return local_payload


def _kernel_breakdown_from_payload(
    all_payload: List[Dict[str, Dict[str, List[float]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    categories = ("moe_forward_kernels", "other_kernels")
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for cat in categories:
        seen = set()
        kernel_names: List[str] = []
        for rank_payload in all_payload:
            for name in rank_payload.get(cat, {}):
                if name not in seen:
                    seen.add(name)
                    kernel_names.append(name)

        kernels: List[Dict[str, Any]] = []
        for name in kernel_names:
            per_rank_times: List[List[float]] = []
            for rank_payload in all_payload:
                times = rank_payload.get(cat, {}).get(name, [])
                per_rank_times.append(times if isinstance(times, list) else [])

            per_rank = {f"rank{i}": _compute_stats(times) for i, times in enumerate(per_rank_times)}
            kernels.append(
                {
                    "name": name,
                    "count": max((len(times) for times in per_rank_times), default=0),
                    "per_rank": per_rank,
                }
            )
        merged[cat] = kernels
    return merged


def _raw_kernel_times_from_payload(
    all_payload: List[Dict[str, Dict[str, List[float]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    categories = ("moe_forward_kernels", "other_kernels")
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for cat in categories:
        seen = set()
        kernel_names: List[str] = []
        for rank_payload in all_payload:
            for name in rank_payload.get(cat, {}):
                if name not in seen:
                    seen.add(name)
                    kernel_names.append(name)

        kernels: List[Dict[str, Any]] = []
        for name in kernel_names:
            per_rank: Dict[str, List[float]] = {}
            for rank_idx, rank_payload in enumerate(all_payload):
                times = rank_payload.get(cat, {}).get(name, [])
                per_rank[f"rank{rank_idx}"] = list(times) if isinstance(times, list) else []
            kernels.append(
                {
                    "name": name,
                    "count": max((len(times) for times in per_rank.values()), default=0),
                    "per_rank": per_rank,
                }
            )
        merged[cat] = kernels
    return merged


def _gather_kernel_timing_blocks(
    detailed_stats: Dict[str, Any],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """All-gather kernel timings once and return summary plus raw timing blocks."""
    all_payload = mpi_allgather(_kernel_times_payload(detailed_stats))
    return _kernel_breakdown_from_payload(all_payload), _raw_kernel_times_from_payload(all_payload)


def _build_raw_data_block(
    per_rank_iters: List[List[float]],
    raw_kernel_times: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build the raw sample block written to JSON and the analysis workbook."""
    return {
        "forward_times_ms": {
            "per_rank": {
                f"rank{i}": list(times) if isinstance(times, list) else []
                for i, times in enumerate(per_rank_iters)
            }
        },
        "kernel_times_ms": raw_kernel_times
        or {
            "moe_forward_kernels": [],
            "other_kernels": [],
        },
    }


# ---------------------------------------------------------------------------
# Bottleneck classification (best-effort, phase-data dependent)
# ---------------------------------------------------------------------------


# Heuristic thresholds for ``_classify_bottleneck``. Tuned for "label looks
# right at a glance on the dashboard"; not intended as hard cutoffs.
_BOTTLENECK_COMM_FRACTION = 0.5
_BOTTLENECK_COMPUTE_FRACTION = 0.5
_BOTTLENECK_ROUTING_FRACTION = 0.4
_BOTTLENECK_LAUNCH_MIN_KERNELS = 50
_BOTTLENECK_LAUNCH_MAX_KERNEL_MS = 0.005  # 5 us per launch on rank0


def _classify_bottleneck(
    phase_times_ms_agg: Dict[str, Dict[str, float]],
    kernel_breakdown: Dict[str, Any],
    forward_score_ms: float,
) -> Optional[str]:
    """Return a coarse bottleneck label.

    The classification is intentionally minimal: with no scheduler-side phase
    markers (Phase 5 of the design) we can only inspect kernel breakdown. When
    we can identify dispatch/combine/GEMM heuristically by name we use that;
    otherwise we return ``None`` so the dashboard can show ``unknown``.
    """
    if phase_times_ms_agg:
        comm_ms = sum(
            v.get("score", 0.0)
            for k, v in phase_times_ms_agg.items()
            if k in ("dispatch", "combine", "all_reduce_or_reduce_results")
        )
        gemm_ms = phase_times_ms_agg.get("backend_run_moe", {}).get(
            "score", 0.0
        ) or phase_times_ms_agg.get("fused_comm_backend_run_moe", {}).get("score", 0.0)
        routing_ms = phase_times_ms_agg.get("routing", {}).get("score", 0.0)
        total = comm_ms + gemm_ms + routing_ms
        if total <= 0:
            return None
        if comm_ms / total > _BOTTLENECK_COMM_FRACTION:
            return "communication_bound"
        if gemm_ms / total > _BOTTLENECK_COMPUTE_FRACTION:
            return "compute_bound"
        if routing_ms / total > _BOTTLENECK_ROUTING_FRACTION:
            return "routing_bound"
        return "unknown"

    # No phase markers: inspect kernel breakdown for a rough hint.
    moe_kernels = kernel_breakdown.get("moe_forward_kernels", [])
    if not moe_kernels:
        return None
    total_count = sum(k.get("count", 0) for k in moe_kernels)
    rank0_total = 0.0
    rank0_count = 0
    for k in moe_kernels:
        rank0_stats = k.get("per_rank", {}).get("rank0", {})
        mean_ms = float(rank0_stats.get("mean", 0.0))
        count = int(k.get("count", 0))
        rank0_total += mean_ms * count
        rank0_count += count
    avg_ms = (rank0_total / rank0_count) if rank0_count else 0.0
    if (
        total_count > _BOTTLENECK_LAUNCH_MIN_KERNELS
        and 0.0 < avg_ms < _BOTTLENECK_LAUNCH_MAX_KERNEL_MS
        and forward_score_ms > 0
    ):
        return "launch_overhead_bound"
    return None


def _runresult_to_row(result: RunResult) -> Dict[str, Any]:
    """Convert ``RunResult`` to the v2 row schema."""
    return {
        "workload": result.workload.to_dict(per_rank_num_tokens=result.per_rank_num_tokens),
        "requested_config": result.config.to_dict(),
        "actual_config": {
            "backend": result.actual_backend,
            "comm_method": result.actual_comm_method,
            "comm_fallback_reason": result.actual_comm_fallback_reason,
            "scheduler_kind": result.scheduler_kind,
            "moe_ep_size": result.moe_ep_size,
            "moe_tp_size": result.moe_tp_size,
            "enable_attention_dp": result.enable_attention_dp,
            "num_chunks": result.num_chunks,
        },
        "status": result.status,
        "skip_reason": result.skip_reason,
        "status_per_rank": result.status_per_rank,
        "instrumentation": result.instrumentation,
        "latency_ms": result.latency_ms
        or {
            "score": None,
            "score_type": "slowest_rank_trimmed_mean",
            "raw_score": None,
            "raw_score_type": "slowest_rank_mean",
            "iter_max_stats": {},
            "iter_max_outliers": {},
            "per_rank": {},
        },
        "phase_times_ms": result.phase_times_ms or {"agg": {}, "per_rank": {}},
        "overlap": result.overlap or {"overlap_ms": None, "overlap_ratio": None},
        "bottleneck": result.bottleneck,
        "kernel_breakdown": result.kernel_breakdown
        or {
            "moe_forward_kernels": [],
            "other_kernels": [],
        },
        "raw_data": result.raw_data
        or {
            "forward_times_ms": {"per_rank": {}},
            "kernel_times_ms": {
                "moe_forward_kernels": [],
                "other_kernels": [],
            },
        },
        "routing_control": result.routing_control or None,
    }


def _make_skipped_run_result(
    *,
    model: ModelSpec,
    workload: WorkloadSpec,
    config: ConfigSpec,
    world_size: int,
    analysis: Tuple[str, ...],
    reason: str,
) -> RunResult:
    """Build a worker-level skipped row without entering MPI collectives."""
    r = RunResult(model=model, workload=workload, config=config)
    r.status = "skipped"
    r.skip_reason = reason
    _, _, _enable_dp = _resolve_mapping_layout(config, world_size)
    r.per_rank_num_tokens = _per_rank_tokens(workload, world_size, enable_dp=bool(_enable_dp))
    r.status_per_rank = {f"rank{i}": "skipped" for i in range(world_size)}
    r.instrumentation = {
        "level": ",".join(sorted(analysis)) if analysis else "summary",
        "cuda_graph": bool(config.cuda_graph),
        "cupti_available": False,
        "nsys_capture": False,
        "phase_timing_available": False,
        "kernel_breakdown_available": False,
    }
    return r


def _make_upstream_skipped_row(
    *,
    model: ModelSpec,
    workload: WorkloadSpec,
    config: ConfigSpec,
    world_size: int,
    analysis: Tuple[str, ...],
    reason: str,
) -> Dict[str, Any]:
    """Build a row marking a candidate as not-attempted due to an upstream crash.

    Uses a reason ending in ``_upstream`` (or starting with one of the
    upstream prefixes) so :func:`_is_completed_for_resume` treats it as
    not-done; a subsequent ``--resume_from`` run will retry it.
    """
    placeholder = _make_skipped_run_result(
        model=model,
        workload=workload,
        config=config,
        world_size=world_size,
        analysis=analysis,
        reason=reason,
    )
    return _runresult_to_row(placeholder)

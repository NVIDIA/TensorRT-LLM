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
"""Analyze opt-in disaggregated KV-transfer admission diagnostics.

The analyzer intentionally depends only on the Python standard library so it
can run directly against CI log artifacts without importing TensorRT-LLM.
Malformed, incomplete, and unmatched diagnostic events are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

_CATEGORY_PATTERN = re.compile(r"\[DISAGG_DIAG\]\[([^]]+)]")
_FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
_RANK_PATTERN = re.compile(r"\[RANK\s+(\d+)]")


@dataclass(frozen=True)
class DiagnosticEvent:
    """One parsed ``[DISAGG_DIAG]`` event."""

    category: str
    time_s: float
    rank: str
    fields: dict[str, str]
    source: str | None = None


@dataclass(frozen=True)
class Admission:
    """An admission decision usable by the offline model."""

    time_s: float
    sequence: str | None
    admitted: int
    deferred: int
    budget_blocks: float | None
    active_blocks: float | None
    candidate_requests: tuple[tuple[str, float], ...]
    admitted_requests: tuple[str, ...]
    deferred_requests: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """A lightweight admission-controller invocation."""

    time_s: float
    sequence: str | None
    admitted: int
    deferred: int
    budget_blocks: float | None


@dataclass(frozen=True)
class PointEvent:
    """A timestamp associated with a request."""

    time_s: float
    request: str
    service_start_s: float | None = None
    logged_time_s: float | None = None
    call_ms: float | None = None


@dataclass(frozen=True)
class ServiceInterval:
    """A completed request service interval."""

    request: str
    start_s: float
    end_s: float
    blocks: float | None
    start_kind: str
    end_kind: str


@dataclass(frozen=True)
class SlotInterval:
    """A matched receiver-slot acquisition and release."""

    request: str
    manager: str
    buffer: str
    manager_index: str | None
    start_s: float
    end_s: float
    wait_ms: float | None


@dataclass(frozen=True)
class ReleasePoint:
    """A point where transfer capacity may be reusable."""

    time_s: float
    request: str
    source: str


def parse_diagnostic_line(line: str) -> DiagnosticEvent | None:
    """Parse one diagnostic line, returning ``None`` when it is unusable.

    Args:
        line: An arbitrary application log line.

    Returns:
        The parsed event when the category, timestamp, and rank are valid.
    """
    category_match = _CATEGORY_PATTERN.search(line)
    if category_match is None:
        return None

    fields = dict(_FIELD_PATTERN.findall(line))
    event_time = _as_float(fields.get("t"))
    if event_time is None:
        return None

    rank = fields.get("rank")
    if rank is None:
        rank_match = _RANK_PATTERN.search(line)
        rank = rank_match.group(1) if rank_match is not None else "unknown"
    return DiagnosticEvent(category_match.group(1), event_time, rank, fields)


def read_diagnostic_events(paths: Iterable[str | Path]) -> list[DiagnosticEvent]:
    """Read parseable diagnostics from one or more log paths.

    Args:
        paths: Text log paths. Unreadable paths are skipped.

    Returns:
        Parsed events in file/line order.
    """
    events: list[DiagnosticEvent] = []
    for path_like in paths:
        try:
            with Path(path_like).open(errors="replace") as log_file:
                for line in log_file:
                    event = parse_diagnostic_line(line)
                    if event is not None:
                        events.append(
                            DiagnosticEvent(
                                event.category,
                                event.time_s,
                                event.rank,
                                event.fields,
                                str(path_like),
                            )
                        )
        except OSError:
            continue
    return events


def analyze_events(events: Iterable[DiagnosticEvent]) -> dict[str, object]:
    """Calculate admission-window measurements from parsed events.

    The reported throughput is completed blocks divided by the union of
    request-service intervals on each rank. This avoids double-counting time
    when transfers overlap. The shadow multiplier is observational only and
    never changes runtime admission or physical memory allocation.

    Args:
        events: Parsed diagnostic events.

    Returns:
        A JSON-serializable analysis dictionary.
    """
    sorted_events = sorted(
        events,
        key=lambda event: (event.source or "", event.rank, event.time_s),
    )
    sources = {event.source for event in sorted_events if event.source is not None}
    namespace_by_source = len(sources) > 1
    category_counts = Counter(event.category for event in sorted_events)
    events_by_rank: dict[str, list[DiagnosticEvent]] = defaultdict(list)
    for event in sorted_events:
        rank_key = f"{event.source}::rank={event.rank}" if namespace_by_source else event.rank
        events_by_rank[rank_key].append(event)

    global_blocks = _collect_global_request_blocks(sorted_events)
    blocks_by_source = {
        source: _collect_global_request_blocks(
            [event for event in sorted_events if event.source == source]
        )
        for source in sources
    }
    ranks: dict[str, object] = {}
    aggregate_service_intervals: list[ServiceInterval] = []
    aggregate_selected_gaps: list[dict[str, object]] = []
    aggregate_gaps_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    aggregate_slot_refill_gaps: list[float] = []
    aggregate_progress_credits: list[float] = []
    aggregate_fixed_multipliers: list[float] = []
    aggregate_poll_durations_ms: list[float] = []
    aggregate_progress_poll_durations_ms: list[float] = []
    aggregate_no_progress_poll_durations_ms: list[float] = []
    aggregate_reported_ready_to_reap_ms: list[float] = []
    aggregate_physical_release_to_reap_s: list[float] = []
    aggregate_invalid_ready_to_reap_samples = 0
    aggregate_busy_s = 0.0
    aggregate_completed_blocks = 0.0

    for rank in sorted(events_by_rank, key=_rank_sort_key):
        rank_events = events_by_rank[rank]
        request_blocks = global_blocks
        if namespace_by_source and rank_events:
            request_blocks = blocks_by_source.get(rank_events[0].source, {})
        rank_analysis, rank_intervals, selected_gaps = _analyze_rank(rank_events, request_blocks)
        ranks[rank] = rank_analysis
        aggregate_service_intervals.extend(rank_intervals)
        aggregate_selected_gaps.extend(selected_gaps)
        release_analysis = rank_analysis["release_to_admission"]
        if isinstance(release_analysis, dict):
            by_source = release_analysis["by_source"]
            if isinstance(by_source, dict):
                for source, source_analysis in by_source.items():
                    if isinstance(source_analysis, dict):
                        aggregate_gaps_by_source[source].extend(source_analysis["samples"])

        service = rank_analysis["service"]
        if isinstance(service, dict):
            aggregate_busy_s += float(service["busy_s"])
            aggregate_completed_blocks += float(service["completed_blocks"])
        receiver_slots = rank_analysis["receiver_slots"]
        if isinstance(receiver_slots, dict):
            aggregate_slot_refill_gaps.extend(receiver_slots["backlog_refill_gap_samples_s"])
        progress = rank_analysis["linear_progress_credit"]
        if isinstance(progress, dict):
            aggregate_progress_credits.extend(progress["credit_samples_blocks"])
        counterfactual = rank_analysis["fixed_multiplier_counterfactual"]
        if isinstance(counterfactual, dict):
            aggregate_fixed_multipliers.extend(
                counterfactual["next_deferred_required_multiplier_samples"]
            )
        status_poll = rank_analysis["status_poll"]
        if isinstance(status_poll, dict):
            aggregate_poll_durations_ms.extend(status_poll["duration_samples_ms"])
            aggregate_progress_poll_durations_ms.extend(status_poll["progress_duration_samples_ms"])
            aggregate_no_progress_poll_durations_ms.extend(
                status_poll["no_progress_duration_samples_ms"]
            )
        scheduler_visibility = rank_analysis["scheduler_visibility"]
        if isinstance(scheduler_visibility, dict):
            aggregate_reported_ready_to_reap_ms.extend(
                scheduler_visibility["reported_ready_to_reap_samples_ms"]
            )
            aggregate_physical_release_to_reap_s.extend(
                scheduler_visibility["physical_release_to_reap_samples_s"]
            )
            aggregate_invalid_ready_to_reap_samples += int(
                scheduler_visibility["invalid_reported_ready_to_reap_samples"]
            )

    aggregate_throughput = _safe_ratio(aggregate_completed_blocks, aggregate_busy_s)
    selected_decision_gaps = [
        float(sample["decision_gap_s"])
        for sample in aggregate_selected_gaps
        if sample.get("decision_gap_s") is not None
    ]
    selected_successful_admission_gaps = [
        float(sample["successful_admission_gap_s"])
        for sample in aggregate_selected_gaps
        if sample.get("successful_admission_gap_s") is not None
    ]
    selected_refill_gaps = [
        float(sample["refill_gap_s"])
        for sample in aggregate_selected_gaps
        if sample.get("refill_gap_s") is not None
    ]
    shadow_samples = [
        float(sample["shadow_multiplier"])
        for sample in aggregate_selected_gaps
        if sample.get("shadow_multiplier") is not None
    ]
    aggregate_release_bounds = {
        source: {
            "decision_gap_s": _summary([float(sample["decision_gap_s"]) for sample in samples]),
            "successful_admission_gap_s": _summary(
                [
                    float(sample["successful_admission_gap_s"])
                    for sample in samples
                    if sample.get("successful_admission_gap_s") is not None
                ]
            ),
            "refill_gap_s": _summary(
                [
                    float(sample["refill_gap_s"])
                    for sample in samples
                    if sample.get("refill_gap_s") is not None
                ]
            ),
            "shadow_multiplier": _summary(
                [
                    float(sample["shadow_multiplier"])
                    for sample in samples
                    if sample.get("shadow_multiplier") is not None
                ]
            ),
        }
        for source, samples in sorted(aggregate_gaps_by_source.items())
    }

    return {
        "schema_version": 1,
        "rank_namespace": "source-path::rank" if namespace_by_source else "rank",
        "aggregate_scope": "all-input-sources" if namespace_by_source else "single-source",
        "parsed_event_count": len(sorted_events),
        "event_counts": dict(sorted(category_counts.items())),
        "ranks": ranks,
        "aggregate": {
            "completed_service_intervals": len(aggregate_service_intervals),
            "completed_blocks": aggregate_completed_blocks,
            "busy_rank_seconds": aggregate_busy_s,
            "throughput_blocks_per_s": aggregate_throughput,
            "service_latency_s": _summary(
                [interval.end_s - interval.start_s for interval in aggregate_service_intervals]
            ),
            "selected_physical_release_to_next_decision_gap_s": _summary(selected_decision_gaps),
            "selected_physical_release_to_successful_admission_gap_s": _summary(
                selected_successful_admission_gaps
            ),
            "selected_physical_release_to_refill_gap_s": _summary(selected_refill_gaps),
            "release_bounds_by_source": aggregate_release_bounds,
            "receiver_slot_refill_gap_s": _summary(aggregate_slot_refill_gaps),
            "selected_physical_shadow_multiplier": _summary(shadow_samples),
            "next_deferred_required_fixed_multiplier": _summary(aggregate_fixed_multipliers),
            "linear_progress_credit_blocks": _summary(aggregate_progress_credits),
            "status_poll": {
                "duration_ms": _summary(aggregate_poll_durations_ms),
                "progress_duration_ms": _summary(aggregate_progress_poll_durations_ms),
                "no_progress_duration_ms": _summary(aggregate_no_progress_poll_durations_ms),
            },
            "scheduler_visibility": {
                "reported_ready_to_reap_ms": _summary(aggregate_reported_ready_to_reap_ms),
                "physical_release_to_reap_s": _summary(aggregate_physical_release_to_reap_s),
                "invalid_reported_ready_to_reap_samples": (aggregate_invalid_ready_to_reap_samples),
            },
        },
        "model": {
            "shadow_multiplier": "1 + throughput_blocks_per_s * refill_gap_s / budget_blocks",
            "fixed_multiplier_counterfactual": (
                "max(1, (active_blocks + FCFS_prefix_blocks) / budget_blocks)"
            ),
            "linear_progress_credit": (
                "sum(request_blocks * elapsed_service_s / realized_service_s)"
            ),
            "caveat": (
                "Retrospective service and progress use completed intervals; they are validation "
                "estimates, not online remaining-work measurements. Python local-ready is a "
                "rank-local bound and reap is scheduler-visible; runtime control requires "
                "conservative cross-rank aggregation or global-ready semantics."
            ),
        },
    }


def analyze_log_paths(paths: Iterable[str | Path]) -> dict[str, object]:
    """Analyze logs without merging overlapping rank IDs across inputs."""
    path_list = list(paths)
    events = read_diagnostic_events(path_list)
    result = analyze_events(events)
    if len(path_list) > 1:
        source_aggregates: dict[str, object] = {}
        for path_like in path_list:
            source = str(path_like)
            source_result = analyze_events(event for event in events if event.source == source)
            source_aggregates[source] = {
                "parsed_event_count": source_result["parsed_event_count"],
                "event_counts": source_result["event_counts"],
                "aggregate": source_result["aggregate"],
            }
        result["source_aggregates"] = source_aggregates
    return result


def _analyze_rank(
    events: list[DiagnosticEvent], global_blocks: dict[str, float]
) -> tuple[dict[str, object], list[ServiceInterval], list[dict[str, object]]]:
    admissions, request_blocks = _collect_admissions(events)
    decisions = _collect_decisions(events, admissions)
    unsuccessful_requests = _collect_unsuccessful_requests(events)
    for request, blocks in global_blocks.items():
        request_blocks.setdefault(request, blocks)

    submits = _collect_points(events, "submit")
    local_ready = _collect_points(
        events,
        "python-transfer",
        action="local-ready",
        excluded_requests=unsuccessful_requests,
        completed_only=True,
    )
    reaps = _collect_points(
        events,
        "reap",
        excluded_requests=unsuccessful_requests,
        completed_only=True,
    )
    for event in events:
        if event.category == "submit":
            blocks = _as_float(event.fields.get("blocks"))
            request = event.fields.get("request")
            if request is not None and blocks is not None and blocks >= 0.0:
                request_blocks[request] = blocks
        elif event.category == "reap":
            blocks = _as_float(event.fields.get("blocks"))
            request = event.fields.get("request")
            if request is not None and blocks is not None and blocks >= 0.0:
                request_blocks.setdefault(request, blocks)

    raw_slot_intervals, unmatched_acquires, unmatched_releases = _match_slot_intervals(events)
    slot_intervals = [
        interval for interval in raw_slot_intervals if interval.request not in unsuccessful_requests
    ]
    physical_service_intervals = _build_request_slot_intervals(slot_intervals, request_blocks)
    physical_queue_samples = _submit_to_interval_start_gaps(submits, physical_service_intervals)
    service_intervals = _build_service_intervals(
        submits, local_ready, reaps, physical_service_intervals, request_blocks
    )
    busy_s = _union_duration(service_intervals)
    completed_blocks = sum(interval.blocks or 0.0 for interval in service_intervals)
    throughput = _safe_ratio(completed_blocks, busy_s)

    release_points = {
        "local-ready": [
            ReleasePoint(point.time_s, point.request, "local-ready") for point in local_ready
        ],
        "reap": [ReleasePoint(point.time_s, point.request, "reap") for point in reaps],
        "receiver-slot": [
            ReleasePoint(interval.end_s, interval.request, "receiver-slot")
            for interval in physical_service_intervals
        ],
    }
    gaps_by_source = {
        source: _match_release_gaps(points, decisions, admissions, submits, throughput)
        for source, points in release_points.items()
    }
    selected_source = _select_release_source(release_points)
    selected_gaps = gaps_by_source[selected_source] if selected_source is not None else []

    slot_refill_gaps = _slot_refill_gaps(
        raw_slot_intervals,
        decisions,
        admissions,
        unsuccessful_requests,
    )
    progress_samples = _linear_progress_credit(admissions, service_intervals)
    fixed_multiplier_samples = _fixed_multiplier_counterfactual(admissions)
    ready_to_reap_samples = _point_pair_gaps(local_ready, reaps)
    physical_release_to_reap_samples = _point_pair_gaps(
        [PointEvent(interval.end_s, interval.request) for interval in physical_service_intervals],
        reaps,
    )
    reported_ready_to_reap_samples = _reported_ready_to_reap_samples(events, unsuccessful_requests)
    submit_to_service_start_samples = _submit_to_service_start_gaps(submits, local_ready)
    status_poll_samples = _status_poll_samples(events)
    progress_poll_durations = [
        float(sample["duration_ms"]) for sample in status_poll_samples if sample["made_progress"]
    ]
    no_progress_poll_durations = [
        float(sample["duration_ms"])
        for sample in status_poll_samples
        if not sample["made_progress"]
    ]
    slot_latencies = [interval.end_s - interval.start_s for interval in slot_intervals]
    wait_samples = [interval.wait_ms for interval in slot_intervals if interval.wait_ms is not None]

    analysis: dict[str, object] = {
        "admission": {
            "invocations": len(decisions),
            "detailed_snapshots": len(admissions),
            "deferred_invocations": sum(decision.deferred > 0 for decision in decisions),
            "successful_invocations": sum(decision.admitted > 0 for decision in decisions),
            "admitted_requests": sum(decision.admitted for decision in decisions),
            "max_deferred": max((decision.deferred for decision in decisions), default=0),
            "budgets_blocks": sorted(
                {
                    decision.budget_blocks
                    for decision in decisions
                    if decision.budget_blocks is not None
                }
            ),
        },
        "service": {
            "intervals": [_service_interval_json(interval) for interval in service_intervals],
            "excluded_unsuccessful_requests": sorted(unsuccessful_requests),
            "latency_s": _summary(
                [interval.end_s - interval.start_s for interval in service_intervals]
            ),
            "busy_s": busy_s,
            "completed_blocks": completed_blocks,
            "throughput_blocks_per_s": throughput,
        },
        "python_transfer": {
            "submit_to_service_start_samples_s": [
                float(sample["gap_s"]) for sample in submit_to_service_start_samples
            ],
            "submit_to_service_start_s": _summary(
                [float(sample["gap_s"]) for sample in submit_to_service_start_samples]
            ),
            "submit_to_service_start_pairs": submit_to_service_start_samples,
            "ready_to_reap_samples_s": [float(sample["gap_s"]) for sample in ready_to_reap_samples],
            "ready_to_reap_s": _summary(
                [float(sample["gap_s"]) for sample in ready_to_reap_samples]
            ),
            "pairs": ready_to_reap_samples,
        },
        "status_poll": {
            "samples": status_poll_samples,
            "duration_samples_ms": [float(sample["duration_ms"]) for sample in status_poll_samples],
            "progress_duration_samples_ms": progress_poll_durations,
            "no_progress_duration_samples_ms": no_progress_poll_durations,
            "duration_ms": _summary(
                [float(sample["duration_ms"]) for sample in status_poll_samples]
            ),
            "progress_duration_ms": _summary(progress_poll_durations),
            "no_progress_duration_ms": _summary(no_progress_poll_durations),
        },
        "scheduler_visibility": {
            "reported_ready_to_reap_samples_ms": [
                float(sample["duration_ms"]) for sample in reported_ready_to_reap_samples
            ],
            "reported_ready_to_reap_ms": _summary(
                [float(sample["duration_ms"]) for sample in reported_ready_to_reap_samples]
            ),
            "reported_ready_to_reap_samples": reported_ready_to_reap_samples,
            "invalid_reported_ready_to_reap_samples": (
                _invalid_reported_ready_to_reap_sample_count(events, unsuccessful_requests)
            ),
            "physical_release_to_reap_samples_s": [
                float(sample["gap_s"]) for sample in physical_release_to_reap_samples
            ],
            "physical_release_to_reap_s": _summary(
                [float(sample["gap_s"]) for sample in physical_release_to_reap_samples]
            ),
            "physical_release_to_reap_pairs": physical_release_to_reap_samples,
        },
        "receiver_slots": {
            "submit_to_service_start_samples_s": [
                float(sample["gap_s"]) for sample in physical_queue_samples
            ],
            "submit_to_service_start_s": _summary(
                [float(sample["gap_s"]) for sample in physical_queue_samples]
            ),
            "submit_to_service_start_pairs": physical_queue_samples,
            "intervals": [_slot_interval_json(interval) for interval in slot_intervals],
            "service_latency_s": _summary(slot_latencies),
            "wait_ms": _summary(wait_samples),
            "unmatched_acquisitions": unmatched_acquires,
            "unmatched_releases": unmatched_releases,
            "excluded_unsuccessful_intervals": len(raw_slot_intervals) - len(slot_intervals),
            "backlog_refill_gap_samples_s": slot_refill_gaps,
            "backlog_refill_gap_s": _summary(slot_refill_gaps),
        },
        "release_to_admission": {
            "selected_release_source": selected_source,
            "selected_samples": selected_gaps,
            "selected_decision_gap_s": _summary(
                [
                    float(sample["decision_gap_s"])
                    for sample in selected_gaps
                    if sample.get("decision_gap_s") is not None
                ]
            ),
            "selected_refill_gap_s": _summary(
                [
                    float(sample["refill_gap_s"])
                    for sample in selected_gaps
                    if sample.get("refill_gap_s") is not None
                ]
            ),
            "selected_successful_admission_gap_s": _summary(
                [
                    float(sample["successful_admission_gap_s"])
                    for sample in selected_gaps
                    if sample.get("successful_admission_gap_s") is not None
                ]
            ),
            "by_source": {
                source: {
                    "samples": samples,
                    "decision_gap_s": _summary(
                        [
                            float(sample["decision_gap_s"])
                            for sample in samples
                            if sample.get("decision_gap_s") is not None
                        ]
                    ),
                    "refill_gap_s": _summary(
                        [
                            float(sample["refill_gap_s"])
                            for sample in samples
                            if sample.get("refill_gap_s") is not None
                        ]
                    ),
                    "successful_admission_gap_s": _summary(
                        [
                            float(sample["successful_admission_gap_s"])
                            for sample in samples
                            if sample.get("successful_admission_gap_s") is not None
                        ]
                    ),
                }
                for source, samples in gaps_by_source.items()
            },
        },
        "shadow_multiplier": {
            "fitted_source": selected_source,
            "by_source": {
                source: {
                    "samples": [
                        sample["shadow_multiplier"]
                        for sample in samples
                        if sample.get("shadow_multiplier") is not None
                    ],
                    "summary": _summary(
                        [
                            float(sample["shadow_multiplier"])
                            for sample in samples
                            if sample.get("shadow_multiplier") is not None
                        ]
                    ),
                }
                for source, samples in gaps_by_source.items()
            },
            "policy_note": (
                "Python local-ready is a rank-local idle-opportunity bound; reap is a "
                "conservative scheduler-visible bound. An adaptive policy must aggregate "
                "conservatively across ranks or use a global-ready signal."
            ),
        },
        "fixed_multiplier_counterfactual": {
            "samples": fixed_multiplier_samples,
            "next_deferred_required_multiplier_samples": [
                float(sample["next_deferred_required_multiplier"])
                for sample in fixed_multiplier_samples
            ],
            "next_deferred_required_multiplier": _summary(
                [
                    float(sample["next_deferred_required_multiplier"])
                    for sample in fixed_multiplier_samples
                ]
            ),
            "all_prefix_required_multiplier": _summary(
                [
                    float(prefix["required_multiplier"])
                    for sample in fixed_multiplier_samples
                    for prefix in sample["prefixes"]
                ]
            ),
        },
        "linear_progress_credit": {
            "samples": progress_samples,
            "credit_samples_blocks": [
                float(sample["estimated_progress_credit_blocks"]) for sample in progress_samples
            ],
            "credit_blocks": _summary(
                [float(sample["estimated_progress_credit_blocks"]) for sample in progress_samples]
            ),
            "credit_fraction": _summary(
                [float(sample["estimated_progress_fraction"]) for sample in progress_samples]
            ),
        },
    }
    return analysis, service_intervals, selected_gaps


def _collect_admissions(events: list[DiagnosticEvent]) -> tuple[list[Admission], dict[str, float]]:
    admissions: list[Admission] = []
    request_blocks: dict[str, float] = {}
    for event in events:
        if event.category != "admission":
            continue
        candidate_requests = _parse_request_blocks(event.fields.get("candidate_requests"))
        admitted_request_blocks = _parse_request_blocks(event.fields.get("admitted_requests"))
        deferred_request_blocks = _parse_request_blocks(event.fields.get("deferred_requests"))
        for request, blocks in (
            candidate_requests + admitted_request_blocks + deferred_request_blocks
        ):
            request_blocks[request] = blocks

        admitted = _as_int(event.fields.get("admitted"))
        deferred = _as_int(event.fields.get("deferred"))
        if admitted is None:
            admitted = len(admitted_request_blocks)
        if deferred is None:
            deferred = len(deferred_request_blocks)
        if admitted < 0 or deferred < 0:
            continue
        budget = _as_float(event.fields.get("budget"))
        if budget is not None and budget <= 0.0:
            budget = None
        active_blocks = _as_float(event.fields.get("active_blocks"))
        admissions.append(
            Admission(
                time_s=event.time_s,
                sequence=event.fields.get("sequence"),
                admitted=admitted,
                deferred=deferred,
                budget_blocks=budget,
                active_blocks=active_blocks,
                candidate_requests=tuple(candidate_requests),
                admitted_requests=tuple(request for request, _ in admitted_request_blocks),
                deferred_requests=tuple(request for request, _ in deferred_request_blocks),
            )
        )
    admissions.sort(key=lambda admission: admission.time_s)
    return admissions, request_blocks


def _collect_decisions(
    events: list[DiagnosticEvent], admissions: list[Admission]
) -> list[Decision]:
    decisions: list[Decision] = []
    for event in events:
        if event.category != "decision":
            continue
        admitted = _as_int(event.fields.get("admitted"))
        deferred = _as_int(event.fields.get("deferred"))
        if admitted is None or deferred is None or admitted < 0 or deferred < 0:
            continue
        budget = _as_float(event.fields.get("budget"))
        if budget is not None and budget <= 0.0:
            budget = None
        decisions.append(
            Decision(
                time_s=event.time_s,
                sequence=event.fields.get("sequence"),
                admitted=admitted,
                deferred=deferred,
                budget_blocks=budget,
            )
        )
    if not decisions:
        decisions = [
            Decision(
                time_s=admission.time_s,
                sequence=admission.sequence,
                admitted=admission.admitted,
                deferred=admission.deferred,
                budget_blocks=admission.budget_blocks,
            )
            for admission in admissions
        ]
    return sorted(decisions, key=lambda decision: decision.time_s)


def _collect_global_request_blocks(events: list[DiagnosticEvent]) -> dict[str, float]:
    blocks_by_request: dict[str, float] = {}
    conflicts: set[str] = set()
    for event in events:
        pairs: list[tuple[str, float]] = []
        if event.category == "admission":
            for field in ("candidate_requests", "admitted_requests", "deferred_requests"):
                pairs.extend(_parse_request_blocks(event.fields.get(field)))
        elif event.category in {"submit", "reap"}:
            request = event.fields.get("request")
            blocks = _as_float(event.fields.get("blocks"))
            if request is not None and blocks is not None and blocks >= 0.0:
                pairs.append((request, blocks))
        for request, blocks in pairs:
            previous = blocks_by_request.get(request)
            if previous is not None and previous != blocks:
                conflicts.add(request)
            else:
                blocks_by_request[request] = blocks
    for request in conflicts:
        blocks_by_request.pop(request, None)
    return blocks_by_request


def _collect_points(
    events: list[DiagnosticEvent],
    category: str,
    action: str | None = None,
    excluded_requests: set[str] | None = None,
    completed_only: bool = False,
) -> list[PointEvent]:
    excluded_requests = excluded_requests or set()
    points: list[PointEvent] = []
    for event in events:
        if event.category != category:
            continue
        if action is not None and event.fields.get("action") != action:
            continue
        request = event.fields.get("request")
        if (
            request is not None
            and request not in excluded_requests
            and (not completed_only or _event_outcome(event) is not False)
        ):
            service_start = _as_float(event.fields.get("service_start_t"))
            if service_start is not None and (service_start < 0.0 or service_start > event.time_s):
                service_start = None
            point_time = event.time_s
            if category == "submit":
                submit_start = _as_float(event.fields.get("submit_start_t"))
                if (
                    submit_start is not None
                    and submit_start >= 0.0
                    and submit_start <= event.time_s
                ):
                    point_time = submit_start
            points.append(
                PointEvent(
                    point_time,
                    request,
                    service_start,
                    event.time_s,
                    _as_float(event.fields.get("submit_call_ms")),
                )
            )
    return sorted(points, key=lambda point: point.time_s)


def _collect_unsuccessful_requests(events: list[DiagnosticEvent]) -> set[str]:
    return {
        request
        for event in events
        if (request := event.fields.get("request")) is not None and _event_outcome(event) is False
    }


def _event_outcome(event: DiagnosticEvent) -> bool | None:
    outcome = event.fields.get("outcome", "").lower()
    if outcome in {"completed", "complete", "success", "successful", "succeeded", "ok"}:
        return True
    if outcome in {
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
        "aborted",
        "timeout",
        "timed-out",
    }:
        return False

    action = event.fields.get("action", "").lower()
    if event.category == "receiver-transfer" and action in {
        "failed",
        "cancelled",
        "canceled",
        "aborted",
        "timeout",
    }:
        return False

    state = event.fields.get("state", "").upper()
    if any(marker in state for marker in ("ERROR", "FAIL", "CANCEL", "TIMEOUT")):
        return False
    if "COMPLETE" in state:
        return True
    return None


def _status_poll_samples(events: list[DiagnosticEvent]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for event in events:
        if event.category != "status-poll":
            continue
        duration_ms = _as_float(event.fields.get("poll_call_ms"))
        completed = _as_int(event.fields.get("completed"))
        failed = _as_int(event.fields.get("failed"))
        cancelled = _as_int(event.fields.get("cancelled"))
        if (
            duration_ms is None
            or duration_ms < 0.0
            or completed is None
            or failed is None
            or cancelled is None
            or min(completed, failed, cancelled) < 0
        ):
            continue
        samples.append(
            {
                "t": event.time_s,
                "poll_start_t": _as_float(event.fields.get("poll_start_t")),
                "duration_ms": duration_ms,
                "at_least_num": _as_int(event.fields.get("at_least_num")),
                "tracked": _as_int(event.fields.get("tracked")),
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "made_progress": completed + failed + cancelled > 0,
            }
        )
    return samples


def _reported_ready_to_reap_samples(
    events: list[DiagnosticEvent], excluded_requests: set[str]
) -> list[dict[str, object]]:
    """Collect scheduler-visible delay reported by completed reap events."""
    samples: list[dict[str, object]] = []
    for event in events:
        if event.category != "reap" or _event_outcome(event) is False:
            continue
        request = event.fields.get("request")
        duration_ms = _as_float(event.fields.get("ready_to_reap_ms"))
        if (
            request is None
            or request in excluded_requests
            or duration_ms is None
            or duration_ms < 0.0
        ):
            continue
        samples.append(
            {
                "t": event.time_s,
                "request": request,
                "duration_ms": duration_ms,
            }
        )
    return samples


def _invalid_reported_ready_to_reap_sample_count(
    events: list[DiagnosticEvent], excluded_requests: set[str]
) -> int:
    """Count negative reported delays excluded from the summary."""
    return sum(
        1
        for event in events
        if event.category == "reap"
        and _event_outcome(event) is not False
        and event.fields.get("request") not in excluded_requests
        and (duration_ms := _as_float(event.fields.get("ready_to_reap_ms"))) is not None
        and duration_ms < 0.0
    )


def _match_slot_intervals(
    events: list[DiagnosticEvent],
) -> tuple[list[SlotInterval], int, int]:
    acquisitions: dict[tuple[str, str], deque[DiagnosticEvent]] = defaultdict(deque)
    intervals: list[SlotInterval] = []
    unmatched_releases = 0
    for event in sorted(events, key=lambda item: item.time_s):
        if event.category != "receiver-slot":
            continue
        action = event.fields.get("action")
        manager = event.fields.get("manager")
        buffer = event.fields.get("buffer")
        if manager is None or buffer in (None, "-1"):
            continue
        key = (manager, buffer)
        if action in {"acquire", "acquired"}:
            acquisitions[key].append(event)
        elif action in {"release", "released"}:
            if not acquisitions[key]:
                unmatched_releases += 1
                continue
            acquired = acquisitions[key].popleft()
            if acquired.time_s > event.time_s:
                unmatched_releases += 1
                continue
            request = acquired.fields.get("request") or event.fields.get("request")
            if request is None:
                continue
            intervals.append(
                SlotInterval(
                    request=request,
                    manager=manager,
                    buffer=buffer,
                    manager_index=acquired.fields.get("manager_index"),
                    start_s=acquired.time_s,
                    end_s=event.time_s,
                    wait_ms=_as_float(acquired.fields.get("wait_ms")),
                )
            )
    unmatched_acquires = sum(len(queue) for queue in acquisitions.values())
    intervals.sort(key=lambda interval: (interval.start_s, interval.end_s))
    return intervals, unmatched_acquires, unmatched_releases


def _build_service_intervals(
    submits: list[PointEvent],
    local_ready: list[PointEvent],
    reaps: list[PointEvent],
    physical_intervals: list[ServiceInterval],
    request_blocks: dict[str, float],
) -> list[ServiceInterval]:
    ready_by_request: dict[str, list[PointEvent]] = defaultdict(list)
    reap_by_request: dict[str, list[PointEvent]] = defaultdict(list)
    for point in local_ready:
        ready_by_request[point.request].append(point)
    for point in reaps:
        reap_by_request[point.request].append(point)

    # Receiver-slot timestamps directly measure the C++ physical service
    # interval. Prefer them over Python submit/reap observations, which include
    # different parts of the lifecycle and can exist for the same request.
    intervals = list(physical_intervals)
    requests_with_physical_interval = {interval.request for interval in physical_intervals}
    for submit in submits:
        if submit.request in requests_with_physical_interval:
            continue
        endpoint = _first_point_after(ready_by_request.get(submit.request, []), submit.time_s)
        end_kind = "local-ready"
        if endpoint is None:
            endpoint = _first_point_after(reap_by_request.get(submit.request, []), submit.time_s)
            end_kind = "reap"
        if endpoint is None:
            continue
        service_start = (
            endpoint.service_start_s if endpoint.service_start_s is not None else submit.time_s
        )
        intervals.append(
            ServiceInterval(
                request=submit.request,
                start_s=service_start,
                end_s=endpoint.time_s,
                blocks=request_blocks.get(submit.request),
                start_kind=(
                    "python-service-start" if endpoint.service_start_s is not None else "submit"
                ),
                end_kind=end_kind,
            )
        )
    intervals.sort(key=lambda interval: (interval.start_s, interval.end_s, interval.request))
    return intervals


def _build_request_slot_intervals(
    slots: list[SlotInterval], request_blocks: dict[str, float]
) -> list[ServiceInterval]:
    slots_by_request: dict[str, list[SlotInterval]] = defaultdict(list)
    for slot in slots:
        slots_by_request[slot.request].append(slot)
    intervals = [
        ServiceInterval(
            request=request,
            start_s=min(slot.start_s for slot in request_slots),
            end_s=max(slot.end_s for slot in request_slots),
            blocks=request_blocks.get(request),
            start_kind="receiver-slot-acquired",
            end_kind="receiver-slot-released",
        )
        for request, request_slots in slots_by_request.items()
    ]
    return sorted(
        intervals, key=lambda interval: (interval.start_s, interval.end_s, interval.request)
    )


def _first_point_after(points: list[PointEvent], start_s: float) -> PointEvent | None:
    return next((point for point in points if point.time_s >= start_s), None)


def _point_pair_gaps(starts: list[PointEvent], ends: list[PointEvent]) -> list[dict[str, object]]:
    ends_by_request: dict[str, deque[PointEvent]] = defaultdict(deque)
    for point in ends:
        ends_by_request[point.request].append(point)
    samples: list[dict[str, object]] = []
    for start in starts:
        candidates = ends_by_request[start.request]
        while candidates and candidates[0].time_s < start.time_s:
            candidates.popleft()
        if not candidates:
            continue
        end = candidates.popleft()
        samples.append(
            {
                "request": start.request,
                "ready_t": start.time_s,
                "reap_t": end.time_s,
                "gap_s": end.time_s - start.time_s,
            }
        )
    return samples


def _submit_to_service_start_gaps(
    submits: list[PointEvent], local_ready: list[PointEvent]
) -> list[dict[str, object]]:
    submits_by_request: dict[str, list[PointEvent]] = defaultdict(list)
    for submit in submits:
        submits_by_request[submit.request].append(submit)
    samples: list[dict[str, object]] = []
    for ready in local_ready:
        if ready.service_start_s is None:
            continue
        submit = next(
            (
                candidate
                for candidate in reversed(submits_by_request[ready.request])
                if candidate.time_s <= ready.service_start_s
            ),
            None,
        )
        if submit is None:
            submit = next(
                (
                    candidate
                    for candidate in reversed(submits_by_request[ready.request])
                    if candidate.time_s <= ready.time_s
                ),
                None,
            )
        if submit is None:
            continue
        samples.append(
            {
                "request": ready.request,
                "submit_t": submit.time_s,
                "submit_return_t": submit.logged_time_s,
                "submit_call_ms": submit.call_ms,
                "service_start_t": ready.service_start_s,
                "gap_s": ready.service_start_s - submit.time_s,
            }
        )
    return samples


def _submit_to_interval_start_gaps(
    submits: list[PointEvent], intervals: list[ServiceInterval]
) -> list[dict[str, object]]:
    submits_by_request: dict[str, list[PointEvent]] = defaultdict(list)
    for submit in submits:
        submits_by_request[submit.request].append(submit)
    samples: list[dict[str, object]] = []
    for interval in intervals:
        submit = next(
            (
                candidate
                for candidate in reversed(submits_by_request[interval.request])
                if candidate.time_s <= interval.start_s
            ),
            None,
        )
        if submit is None:
            submit = next(
                (
                    candidate
                    for candidate in reversed(submits_by_request[interval.request])
                    if candidate.time_s <= interval.end_s
                ),
                None,
            )
        if submit is None:
            continue
        samples.append(
            {
                "request": interval.request,
                "submit_t": submit.time_s,
                "submit_return_t": submit.logged_time_s,
                "submit_call_ms": submit.call_ms,
                "service_start_t": interval.start_s,
                "gap_s": interval.start_s - submit.time_s,
            }
        )
    return samples


def _match_release_gaps(
    releases: list[ReleasePoint],
    decisions: list[Decision],
    admissions: list[Admission],
    submits: list[PointEvent],
    throughput_blocks_per_s: float | None,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for release in sorted(releases, key=lambda point: point.time_s):
        prior = _latest_backlog_signal(decisions, admissions, release.time_s)
        if prior is None or prior[0] <= 0:
            continue
        next_decision = next(
            (decision for decision in decisions if decision.time_s > release.time_s),
            None,
        )
        if next_decision is None:
            continue
        backlog_requests = _backlog_request_ids_at(admissions, release.time_s)
        backlog_identity_unknown = not backlog_requests
        successful_admission = None
        matched_backlog_requests: set[str] = set()
        for decision in decisions:
            if decision.time_s <= release.time_s or decision.admitted <= 0:
                continue
            if backlog_identity_unknown:
                successful_admission = decision
                break
            detailed_admission = _matching_admission(decision, admissions)
            if detailed_admission is None:
                backlog_identity_unknown = True
                successful_admission = decision
                break
            matched = set(detailed_admission.admitted_requests).intersection(backlog_requests)
            if matched:
                successful_admission = decision
                matched_backlog_requests = matched
                break
        refill = (
            _find_refill_submit(
                submits,
                successful_admission,
                admissions,
                matched_backlog_requests or None,
            )
            if successful_admission is not None
            else None
        )
        budget = prior[1] or (
            successful_admission.budget_blocks
            if successful_admission is not None
            else next_decision.budget_blocks
        )
        refill_gap = refill.time_s - release.time_s if refill is not None else None
        shadow_multiplier = None
        eligible_for_multiplier_fit = (
            not backlog_identity_unknown and bool(matched_backlog_requests) and refill is not None
        )
        if (
            eligible_for_multiplier_fit
            and throughput_blocks_per_s is not None
            and refill_gap is not None
            and budget is not None
            and budget > 0.0
        ):
            shadow_multiplier = 1.0 + throughput_blocks_per_s * refill_gap / budget
        samples.append(
            {
                "release_source": release.source,
                "release_request": release.request,
                "release_t": release.time_s,
                "backlog_request_ids": sorted(backlog_requests),
                "backlog_identity_unknown": backlog_identity_unknown,
                "matched_backlog_request_ids": sorted(matched_backlog_requests),
                "eligible_for_multiplier_fit": eligible_for_multiplier_fit,
                "decision_t": next_decision.time_s,
                "decision_sequence": next_decision.sequence,
                "decision_gap_s": next_decision.time_s - release.time_s,
                "successful_admission_t": (
                    successful_admission.time_s if successful_admission is not None else None
                ),
                "successful_admission_sequence": (
                    successful_admission.sequence if successful_admission is not None else None
                ),
                "successful_admission_gap_s": (
                    successful_admission.time_s - release.time_s
                    if successful_admission is not None
                    else None
                ),
                "refill_t": refill.time_s if refill is not None else None,
                "refill_submit_return_t": (refill.logged_time_s if refill is not None else None),
                "refill_submit_call_ms": (refill.call_ms if refill is not None else None),
                "refill_request": refill.request if refill is not None else None,
                "refill_gap_s": refill_gap,
                "budget_blocks": budget,
                "throughput_blocks_per_s": throughput_blocks_per_s,
                "shadow_multiplier": shadow_multiplier,
            }
        )
    return samples


def _find_refill_submit(
    submits: list[PointEvent],
    decision: Decision,
    admissions: list[Admission],
    required_requests: set[str] | None = None,
) -> PointEvent | None:
    candidates = [submit for submit in submits if submit.time_s >= decision.time_s]
    if required_requests:
        return next(
            (submit for submit in candidates if submit.request in required_requests),
            None,
        )
    admission = _matching_admission(decision, admissions)
    if admission is not None and admission.admitted_requests:
        admitted = set(admission.admitted_requests)
        return next((submit for submit in candidates if submit.request in admitted), None)
    return candidates[0] if candidates else None


def _backlog_request_ids_at(admissions: list[Admission], time_s: float) -> set[str]:
    admission = next(
        (candidate for candidate in reversed(admissions) if candidate.time_s <= time_s),
        None,
    )
    if admission is None or admission.deferred <= 0:
        return set()
    return set(admission.deferred_requests)


def _matching_admission(decision: Decision, admissions: list[Admission]) -> Admission | None:
    if decision.sequence is not None:
        match = next(
            (admission for admission in admissions if admission.sequence == decision.sequence),
            None,
        )
        if match is not None:
            return match
    return next(
        (
            admission
            for admission in admissions
            if math.isclose(
                admission.time_s,
                decision.time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        None,
    )


def _latest_backlog_signal(
    decisions: list[Decision], admissions: list[Admission], time_s: float
) -> tuple[int, float | None] | None:
    signals = [
        (decision.time_s, 1, decision.deferred, decision.budget_blocks)
        for decision in decisions
        if decision.time_s <= time_s
    ]
    signals.extend(
        (admission.time_s, 0, admission.deferred, admission.budget_blocks)
        for admission in admissions
        if admission.time_s <= time_s
    )
    if not signals:
        return None
    _, _, deferred, budget = max(signals, key=lambda signal: (signal[0], signal[1]))
    return deferred, budget


def _select_release_source(release_points: dict[str, list[ReleasePoint]]) -> str | None:
    # Only the C++ path has a directly observed physical release signal.
    # Python local-ready and consensus reap are complementary bounds, so the
    # report intentionally does not collapse them into one selected source.
    return "receiver-slot" if release_points["receiver-slot"] else None


def _slot_refill_gaps(
    intervals: list[SlotInterval],
    decisions: list[Decision],
    admissions: list[Admission],
    excluded_requests: set[str],
) -> list[float]:
    by_slot: dict[tuple[str, str], list[SlotInterval]] = defaultdict(list)
    for interval in intervals:
        by_slot[(interval.manager, interval.buffer)].append(interval)
    gaps: list[float] = []
    for slot_intervals in by_slot.values():
        slot_intervals.sort(key=lambda interval: interval.start_s)
        for current, following in zip(slot_intervals, slot_intervals[1:]):
            if current.request in excluded_requests or following.request in excluded_requests:
                continue
            prior = _latest_backlog_signal(decisions, admissions, current.end_s)
            if prior is not None and prior[0] > 0 and following.start_s >= current.end_s:
                gaps.append(following.start_s - current.end_s)
    return gaps


def _fixed_multiplier_counterfactual(
    admissions: list[Admission],
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for admission in admissions:
        if (
            admission.deferred <= 0
            or admission.active_blocks is None
            or admission.budget_blocks is None
            or not admission.candidate_requests
        ):
            continue

        deferred_requests = set(admission.deferred_requests)
        first_deferred_index = next(
            (
                index
                for index, (request, _) in enumerate(admission.candidate_requests)
                if request in deferred_requests
            ),
            None,
        )
        if first_deferred_index is None and admission.admitted < len(admission.candidate_requests):
            first_deferred_index = admission.admitted
        if first_deferred_index is None:
            continue

        prefix_blocks = 0.0
        prefixes: list[dict[str, object]] = []
        for index, (request, blocks) in enumerate(admission.candidate_requests):
            prefix_blocks += blocks
            required_multiplier = max(
                1.0,
                (admission.active_blocks + prefix_blocks) / admission.budget_blocks,
            )
            prefixes.append(
                {
                    "prefix_length": index + 1,
                    "last_request": request,
                    "prefix_blocks": prefix_blocks,
                    "required_multiplier": required_multiplier,
                    "minimum_integer_multiplier": math.ceil(required_multiplier),
                    "observed_status": (
                        "deferred"
                        if request in deferred_requests or index >= admission.admitted
                        else "admitted"
                    ),
                }
            )

        next_deferred = prefixes[first_deferred_index]
        samples.append(
            {
                "decision_t": admission.time_s,
                "active_blocks": admission.active_blocks,
                "budget_blocks": admission.budget_blocks,
                "next_deferred_request": next_deferred["last_request"],
                "next_deferred_prefix_blocks": next_deferred["prefix_blocks"],
                "next_deferred_required_multiplier": next_deferred["required_multiplier"],
                "next_deferred_minimum_integer_multiplier": next_deferred[
                    "minimum_integer_multiplier"
                ],
                "prefixes": prefixes,
            }
        )
    return samples


def _linear_progress_credit(
    admissions: list[Admission], intervals: list[ServiceInterval]
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for admission in admissions:
        if admission.deferred <= 0:
            continue
        in_progress = [
            interval
            for interval in intervals
            if interval.blocks is not None
            and interval.start_s <= admission.time_s < interval.end_s
            and interval.end_s > interval.start_s
        ]
        if not in_progress:
            continue
        original_blocks = sum(interval.blocks or 0.0 for interval in in_progress)
        credit = sum(
            (interval.blocks or 0.0)
            * (admission.time_s - interval.start_s)
            / (interval.end_s - interval.start_s)
            for interval in in_progress
        )
        fraction = _safe_ratio(credit, original_blocks) or 0.0
        samples.append(
            {
                "decision_t": admission.time_s,
                "in_progress_requests": len(in_progress),
                "logged_active_blocks": admission.active_blocks,
                "original_in_progress_blocks": original_blocks,
                "estimated_progress_credit_blocks": credit,
                "estimated_remaining_blocks": original_blocks - credit,
                "estimated_progress_fraction": fraction,
            }
        )
    return samples


def _union_duration(intervals: list[ServiceInterval]) -> float:
    ranges = sorted((interval.start_s, interval.end_s) for interval in intervals)
    if not ranges:
        return 0.0
    merged: list[list[float]] = []
    for start_s, end_s in ranges:
        if end_s < start_s:
            continue
        if not merged or start_s > merged[-1][1]:
            merged.append([start_s, end_s])
        else:
            merged[-1][1] = max(merged[-1][1], end_s)
    return sum(end_s - start_s for start_s, end_s in merged)


def _summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    samples = sorted(value for value in values if value is not None and math.isfinite(value))
    if not samples:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "min": samples[0],
        "mean": fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "max": samples[-1],
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _parse_request_blocks(value: str | None) -> list[tuple[str, float]]:
    if value in (None, "", "-"):
        return []
    pairs: list[tuple[str, float]] = []
    for item in value.split(","):
        request, separator, blocks_text = item.partition(":")
        blocks = _as_float(blocks_text) if separator else None
        if request and blocks is not None and blocks >= 0.0:
            pairs.append((request, blocks))
    return pairs


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _service_interval_json(interval: ServiceInterval) -> dict[str, object]:
    return {
        "request": interval.request,
        "start_t": interval.start_s,
        "end_t": interval.end_s,
        "latency_s": interval.end_s - interval.start_s,
        "blocks": interval.blocks,
        "start_kind": interval.start_kind,
        "end_kind": interval.end_kind,
    }


def _slot_interval_json(interval: SlotInterval) -> dict[str, object]:
    return {
        "request": interval.request,
        "manager": interval.manager,
        "manager_index": interval.manager_index,
        "buffer": interval.buffer,
        "acquired_t": interval.start_s,
        "released_t": interval.end_s,
        "service_s": interval.end_s - interval.start_s,
        "wait_ms": interval.wait_ms,
    }


def _rank_sort_key(rank: str) -> tuple[int, int | str]:
    try:
        return 0, int(rank)
    except ValueError:
        return 1, rank


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze [DISAGG_DIAG] admission and KV-transfer events."
    )
    parser.add_argument("logs", nargs="+", help="Worker or preserved diagnostic log paths")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line analyzer."""
    args = _build_argument_parser().parse_args(argv)
    print(json.dumps(analyze_log_paths(args.logs), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

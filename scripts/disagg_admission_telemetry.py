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

_ClockDomain = tuple[str, str, str, str, str, str]


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
    candidate_requests_omitted: int
    admitted_requests_omitted: int
    deferred_requests_omitted: int


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


@dataclass(frozen=True)
class _LifecycleMark:
    """One request-scoped lifecycle point in a single clock domain."""

    time_s: float
    request: str
    local_request: str | None
    tag: str
    category: str
    action: str
    log_source: str
    emitter_source: str | None
    host: str
    instance: str
    role: str
    rank: str
    clock: str


_CTX_ACTIONS = {
    "queued",
    "send-queued",
    "timer-start",
    "deadline-observed",
    "cancel-requested",
    "cancel-result",
    "receiver-info-ready",
    "credit-received",
    "request-info-complete",
    "first-write",
    "first-write-submitted",
    "service-start",
    "local-complete",
    "physical-complete",
    "kv-physical-complete",
}
_GEN_ACTIONS = {
    "timer-start",
    "deadline-observed",
    "cancel-requested",
    "cancel-result",
    "capacity-prepared",
    "request-info-sent",
    "request-info-submitted",
    "peer-ready",
    "ready",
    "local-ready",
}
_TERMINAL_ACTIONS = {
    "completed",
    "complete",
    "reaped",
    "failed",
    "failure",
    "cancelled",
    "canceled",
}
_SUCCESS_TERMINAL_ACTIONS = {"completed", "complete", "reaped"}
_LIFECYCLE_INTERVALS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ctx_timer_to_credit": (
        ("ctx-timer-start",),
        ("sender-credit",),
    ),
    "ctx_timer_to_deadline": (
        ("ctx-timer-start",),
        ("ctx-deadline",),
    ),
    "ctx_deadline_to_credit": (
        ("ctx-deadline",),
        ("sender-credit",),
    ),
    "ctx_deadline_to_cancel_result": (
        ("ctx-deadline",),
        ("ctx-cancel-result",),
    ),
    "ctx_deadline_to_terminal": (
        ("ctx-deadline",),
        ("sender-terminal", "ctx-terminal"),
    ),
    "ctx_queued_to_credit": (
        ("ctx-queued", "sender-queued"),
        ("sender-credit",),
    ),
    "ctx_credit_to_first_write": (
        ("sender-credit",),
        ("sender-first-write",),
    ),
    "ctx_first_write_to_terminal": (
        ("sender-first-write",),
        ("sender-terminal", "ctx-terminal"),
    ),
    "ctx_first_write_to_kv_physical_complete": (
        ("sender-first-write",),
        ("sender-kv-physical-complete",),
    ),
    "ctx_kv_physical_complete_to_terminal": (
        ("sender-kv-physical-complete",),
        ("sender-terminal", "ctx-terminal"),
    ),
    "ctx_deadline_to_kv_physical_complete": (
        ("ctx-deadline",),
        ("sender-kv-physical-complete",),
    ),
    "gate1_blocked_to_fitting": (
        ("gate1-blocked",),
        ("gate1-fitting",),
    ),
    "gen_arrival_to_first_gate1": (
        ("gen-arrival",),
        ("gate1-fitting", "gate1-blocked"),
    ),
    "gen_timer_to_deadline": (
        ("gen-timer-start",),
        ("gen-deadline",),
    ),
    "gen_deadline_to_cancel_result": (
        ("gen-deadline",),
        ("gen-cancel-result",),
    ),
    "gen_deadline_to_terminal": (
        ("gen-deadline",),
        ("receiver-terminal",),
    ),
    "gen_arrival_to_first_gate2": (
        ("gen-arrival",),
        ("gate2-seen",),
    ),
    "gen_arrival_to_activation": (
        ("gen-arrival",),
        ("gen-activation",),
    ),
    "gen_activation_to_first_gate2": (
        ("gen-activation",),
        ("gate2-seen",),
    ),
    "gate2_defer_to_admit": (
        ("gate2-deferred",),
        ("gate2-admitted",),
    ),
    "gate2_admit_to_submit": (
        ("gate2-admitted",),
        ("submit", "receiver-submitted"),
    ),
    "submit_to_request_info": (
        ("submit", "receiver-submitted"),
        ("request-info",),
    ),
    "ready_to_reap": (
        ("local-ready", "receiver-completed"),
        ("reap",),
    ),
    "gen_arrival_to_decode_start": (
        ("gen-arrival",),
        ("decode-start",),
    ),
    "ready_to_decode_start": (
        ("local-ready", "receiver-completed"),
        ("decode-start",),
    ),
    "reap_to_decode_start": (
        ("reap",),
        ("decode-start",),
    ),
}
_ACTIVE_AGE_BUCKETS = (
    (0.01, "lt_10ms"),
    (0.1, "10ms_to_100ms"),
    (1.0, "100ms_to_1s"),
    (10.0, "1s_to_10s"),
    (60.0, "10s_to_60s"),
    (math.inf, "gte_60s"),
)


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
    known_instances: dict[tuple[str | None, str, str, str], set[str]] = defaultdict(set)
    for event in sorted_events:
        instance = event.fields.get("instance", "-")
        if instance in {"-", "unknown"}:
            continue
        category = _normalize_diag_token(event.category)
        action = _normalize_diag_token(event.fields.get("action", ""))
        known_instances[
            (
                event.source,
                _event_host(event),
                _event_role(event, category, action),
                event.rank,
            )
        ].add(instance)

    def resolved_instance(event: DiagnosticEvent) -> str:
        instance = event.fields.get("instance", "-")
        if instance not in {"-", "unknown"}:
            return instance
        category = _normalize_diag_token(event.category)
        action = _normalize_diag_token(event.fields.get("action", ""))
        candidates = known_instances.get(
            (
                event.source,
                _event_host(event),
                _event_role(event, category, action),
                event.rank,
            ),
            set(),
        )
        return next(iter(candidates)) if len(candidates) == 1 else instance

    sorted_events = [
        event
        if resolved_instance(event) == event.fields.get("instance", "-")
        else DiagnosticEvent(
            event.category,
            event.time_s,
            event.rank,
            {
                **event.fields,
                "instance": resolved_instance(event),
            },
            event.source,
        )
        for event in sorted_events
    ]

    emitters_by_legacy_rank: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for event in sorted_events:
        legacy_rank = f"{event.source}::rank={event.rank}" if namespace_by_source else event.rank
        category = _normalize_diag_token(event.category)
        action = _normalize_diag_token(event.fields.get("action", ""))
        emitters_by_legacy_rank[legacy_rank].add(
            (
                _event_host(event),
                resolved_instance(event),
                _event_role(event, category, action),
            )
        )
    split_legacy_ranks = {
        legacy_rank
        for legacy_rank, emitters in emitters_by_legacy_rank.items()
        if len(emitters) > 1
        and any(
            host != "<unspecified>" or instance not in {"-", "unknown"}
            for host, instance, _ in emitters
        )
    }
    events_by_rank: dict[str, list[DiagnosticEvent]] = defaultdict(list)
    for event in sorted_events:
        rank_key = f"{event.source}::rank={event.rank}" if namespace_by_source else event.rank
        if rank_key in split_legacy_ranks:
            category = _normalize_diag_token(event.category)
            action = _normalize_diag_token(event.fields.get("action", ""))
            rank_key = (
                f"{rank_key}::host={_event_host(event)}"
                f"::instance={resolved_instance(event)}"
                f"::role={_event_role(event, category, action)}"
            )
        events_by_rank[rank_key].append(event)

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
        block_scope = sorted_events
        if rank_events:
            representative = rank_events[0]
            instance = resolved_instance(representative)
            if instance not in {"-", "unknown"}:
                category = _normalize_diag_token(representative.category)
                action = _normalize_diag_token(representative.fields.get("action", ""))
                role = _event_role(representative, category, action)
                block_scope = [
                    event
                    for event in sorted_events
                    if event.source == representative.source
                    and _event_host(event) == _event_host(representative)
                    and resolved_instance(event) == instance
                    and _event_role(
                        event,
                        _normalize_diag_token(event.category),
                        _normalize_diag_token(event.fields.get("action", "")),
                    )
                    == role
                ]
            elif namespace_by_source:
                block_scope = [
                    event for event in sorted_events if event.source == representative.source
                ]
        request_blocks = _collect_global_request_blocks(block_scope)
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
    lifecycle = _analyze_request_lifecycles(sorted_events)
    cross_host_correlation = _analyze_cross_host_correlation(sorted_events)
    remaining_work_ground_truth = _analyze_remaining_work_ground_truth(sorted_events)
    known_backlog_release = _known_backlog_release_analysis(ranks)

    return {
        "schema_version": 3,
        "rank_namespace": (
            "source-path::rank[::host::instance::role]"
            if split_legacy_ranks
            else ("source-path::rank" if namespace_by_source else "rank")
        ),
        "aggregate_scope": "all-input-sources" if namespace_by_source else "single-source",
        "parsed_event_count": len(sorted_events),
        "event_counts": dict(sorted(category_counts.items())),
        "ranks": ranks,
        "lifecycle": lifecycle,
        "cross_host_correlation": cross_host_correlation,
        "remaining_work_ground_truth": remaining_work_ground_truth,
        "known_backlog_release": known_backlog_release,
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


def _analyze_request_lifecycles(events: list[DiagnosticEvent]) -> dict[str, object]:
    marks = _collect_lifecycle_marks(events)
    timelines: dict[tuple[_ClockDomain, str], list[_LifecycleMark]] = defaultdict(list)
    tag_domains: dict[tuple[str, str], set[_ClockDomain]] = defaultdict(set)
    for mark in marks:
        domain = _lifecycle_domain(mark)
        timelines[(domain, mark.request)].append(mark)
        tag_domains[(mark.request, mark.tag)].add(domain)

    request_records: list[dict[str, object]] = []
    for (domain, request), request_marks in sorted(
        timelines.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        request_marks.sort(key=lambda mark: (mark.time_s, mark.tag))
        marks_by_tag: dict[str, list[_LifecycleMark]] = defaultdict(list)
        for mark in request_marks:
            marks_by_tag[mark.tag].append(mark)

        intervals = {
            name: _evaluate_lifecycle_interval(
                request,
                domain,
                marks_by_tag,
                start_tags,
                end_tags,
                tag_domains,
            )
            for name, (start_tags, end_tags) in _LIFECYCLE_INTERVALS.items()
        }
        first_timestamps = {
            tag: tag_marks[0].time_s for tag, tag_marks in sorted(marks_by_tag.items()) if tag_marks
        }
        local_requests = sorted(
            {mark.local_request for mark in request_marks if mark.local_request is not None}
        )
        emitter_sources = sorted(
            {mark.emitter_source for mark in request_marks if mark.emitter_source is not None}
        )
        request_records.append(
            {
                "request": request,
                "local_requests": local_requests,
                "log_source": domain[0],
                "host": domain[1],
                "role": domain[2],
                "rank": domain[3],
                "clock": domain[4],
                "instance": domain[5],
                "clock_domain": _clock_domain_label(domain),
                "emitter_sources": emitter_sources,
                "first_timestamps": first_timestamps,
                "deadline_phase": _classify_deadline_phase(marks_by_tag),
                "intervals": intervals,
            }
        )

    deadline_phase_counts = Counter(
        str(record["deadline_phase"])
        for record in request_records
        if record["deadline_phase"] is not None
    )
    return {
        "clock_domain_policy": (
            "Durations require the same input log source, host, instance, role, rank, "
            "and clock. Matching request IDs in another source or clock domain are "
            "correlated only for censoring; their raw timestamps are never subtracted."
        ),
        "correlation_request_policy": (
            "Prefer a nonzero context_request/disaggregated request ID; otherwise use request."
        ),
        "request_count": len({mark.request for mark in marks}),
        "clock_domain_request_count": len(request_records),
        "deadline_phase_counts": dict(sorted(deadline_phase_counts.items())),
        "requests": request_records,
        "interval_coverage": _lifecycle_interval_coverage(
            request_records,
            marks,
        ),
    }


def _collect_lifecycle_marks(events: list[DiagnosticEvent]) -> list[_LifecycleMark]:
    marks: list[_LifecycleMark] = []
    seen: set[tuple[object, ...]] = set()
    for event in events:
        category = _normalize_diag_token(event.category)
        action = _normalize_diag_token(event.fields.get("action", ""))
        role = _event_role(event, category, action)
        direct_request = _event_correlation_request(event)
        local_request = _event_local_request(event)

        def add(
            tag: str,
            request: str | None = direct_request,
            time_s: float = event.time_s,
            detail: str | None = None,
        ) -> None:
            if request is None or not math.isfinite(time_s) or time_s < 0.0:
                return
            mark = _LifecycleMark(
                time_s=time_s,
                request=request,
                local_request=local_request if request == direct_request else None,
                tag=tag,
                category=category,
                action=detail or action,
                log_source=event.source or "<in-memory>",
                emitter_source=event.fields.get("source"),
                host=_event_host(event),
                instance=event.fields.get("instance", "-"),
                role=role,
                rank=event.rank,
                clock=_event_clock(event),
            )
            identity = (
                _lifecycle_domain(mark),
                mark.request,
                mark.tag,
                mark.time_s,
                mark.category,
                mark.action,
            )
            if identity not in seen:
                seen.add(identity)
                marks.append(mark)

        if category == "gen-arrival":
            add("gen-arrival")
        if category == "gen-activation":
            add("gen-activation")

        if category in {"gate1", "gate-1"}:
            for field, tag in (
                ("waiting_requests", "gate1-waiting"),
                ("fitting_requests", "gate1-fitting"),
                ("blocked_requests", "gate1-blocked"),
            ):
                for request in _parse_request_ids(event.fields.get(field)):
                    add(tag, request=request, detail=field)
            if action in {"fitting", "fit"}:
                add("gate1-fitting")
            elif action in {"blocked", "deferred"}:
                add("gate1-blocked")

        if category in {"decision", "admission", "gate2", "gate-2"}:
            for field, tag in (
                ("active_requests", "gate2-active"),
                ("candidate_requests", "gate2-seen"),
                ("waiting_requests", "gate2-seen"),
                ("fitting_requests", "gate2-seen"),
                ("admitted_requests", "gate2-admitted"),
                ("deferred_requests", "gate2-deferred"),
            ):
                for request in _parse_request_ids(event.fields.get(field)):
                    add(tag, request=request, detail=field)
                    if tag in {"gate2-admitted", "gate2-deferred"}:
                        add("gate2-seen", request=request, detail=field)
            if action in {"admit", "admitted"}:
                add("gate2-admitted")
                add("gate2-seen")
            elif action in {"defer", "deferred"}:
                add("gate2-deferred")
                add("gate2-seen")

        if category == "submit":
            submit_time = _as_float(event.fields.get("submit_start_t"))
            if submit_time is None or submit_time < 0.0 or submit_time > event.time_s:
                submit_time = event.time_s
            add("submit", time_s=submit_time)

        if category == "reap":
            add("reap")
            ready_time = _as_float(event.fields.get("ready_t"))
            ready_time_source = event.fields.get("ready_time_source")
            if (
                ready_time_source != "cpp-global"
                and ready_time is not None
                and 0.0 <= ready_time <= event.time_s
            ):
                add("local-ready", time_s=ready_time, detail="ready_t")

        if category == "gen-service" and action == "decode-start-proxy":
            add("decode-start")

        if category == "ctx-transfer":
            if action in {"queued", "send-queued"}:
                add("ctx-queued")
            if action == "timer-start":
                add("ctx-timer-start")
            elif action in {"deadline-observed", "timeout", "timed-out"}:
                add("ctx-deadline")
            elif action == "cancel-requested":
                add("ctx-cancel-requested")
            elif action == "cancel-result":
                add("ctx-cancel-result")
            if action in _TERMINAL_ACTIONS:
                add("ctx-terminal")

        if category == "gen-transfer":
            if action == "timer-start":
                add("gen-timer-start")
            elif action in {"deadline-observed", "timeout", "timed-out"}:
                add("gen-deadline")
            elif action == "cancel-requested":
                add("gen-cancel-requested")
            elif action == "cancel-result":
                add("gen-cancel-result")

        if category == "sender-transfer":
            if action in {"queued", "send-queued"}:
                add("sender-queued")
                add("ctx-queued")
            if action in {
                "credit-received",
                "receiver-info-ready",
                "request-info-complete",
            }:
                add("sender-credit")
            if action in {"service-start", "first-write", "first-write-submitted"}:
                add("sender-first-write")
            if action in {
                "local-complete",
                "physical-complete",
                "kv-physical-complete",
            }:
                add("sender-kv-physical-complete")
            if action in _TERMINAL_ACTIONS:
                add("sender-terminal")

        if category == "receiver-transfer":
            if action in {"submitted", "request-info-submitted"}:
                add("receiver-submitted")
            if action in {
                "request-info-sent",
                "request-info-submitted",
            }:
                add("request-info")
            if action in {"peer-ready", "ready"}:
                add("peer-ready")
            if action == "local-ready":
                add("local-ready")
            if action in _TERMINAL_ACTIONS:
                add("receiver-terminal")
                if action in _SUCCESS_TERMINAL_ACTIONS:
                    add("receiver-completed")

        if category == "python-transfer":
            if role == "ctx":
                if action in {"queued", "send-queued"}:
                    add("ctx-queued")
                    add("sender-queued")
                if action in {
                    "credit-received",
                    "receiver-info-ready",
                    "request-info-complete",
                }:
                    add("sender-credit")
                if action in {
                    "service-start",
                    "first-write",
                    "first-write-submitted",
                }:
                    add("sender-first-write")
                if action in {
                    "local-complete",
                    "physical-complete",
                    "kv-physical-complete",
                }:
                    add("sender-kv-physical-complete")
                if action in _TERMINAL_ACTIONS:
                    add("sender-terminal")
                    add("ctx-terminal")
            elif role == "gen":
                if action in {"submitted", "capacity-prepared"}:
                    add("receiver-submitted")
                if action in {"request-info-sent", "request-info-submitted"}:
                    add("request-info")
                if action == "local-ready":
                    add("local-ready")
                if action in _TERMINAL_ACTIONS:
                    add("receiver-terminal")
                    if action in _SUCCESS_TERMINAL_ACTIONS:
                        add("receiver-completed")

        if category == "receiver-slot" and action in {"release", "released"}:
            add("physical-release")

    return sorted(
        marks,
        key=lambda mark: (
            _lifecycle_domain(mark),
            mark.request,
            mark.time_s,
            mark.tag,
        ),
    )


def _analyze_cross_host_correlation(
    events: list[DiagnosticEvent],
) -> dict[str, object]:
    """Join CTX and GEN causal points using diagnostic Unix timestamps."""
    points: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))

    def add(
        event: DiagnosticEvent,
        tag: str,
        request: str | None = None,
    ) -> None:
        wall_time_s = _event_wall_time(event)
        request = request or _event_correlation_request(event)
        if wall_time_s is None or request is None:
            return
        points[request][tag].append(
            {
                "wall_t": wall_time_s,
                "wall_semantics": event.fields.get("wall_semantics", "unknown"),
                "local_t": event.time_s,
                "local_clock": _event_clock(event),
                "host": _event_host(event),
                "instance": event.fields.get("instance", "-"),
                "role": _event_role(
                    event,
                    _normalize_diag_token(event.category),
                    _normalize_diag_token(event.fields.get("action", "")),
                ),
                "rank": event.rank,
                "log_source": event.source or "<in-memory>",
                "category": _normalize_diag_token(event.category),
                "action": _normalize_diag_token(event.fields.get("action", "")),
                "sequence": event.fields.get("sequence"),
                "previous": event.fields.get("previous"),
            }
        )

    for event in events:
        category = _normalize_diag_token(event.category)
        action = _normalize_diag_token(event.fields.get("action", ""))
        role = _event_role(event, category, action)

        if category == "gen-arrival":
            add(event, "gen-arrival")
        elif category == "gen-activation":
            add(event, "gen-activation")
        elif category in {"gate1", "gate-1"}:
            if action in {"fitting", "fit"}:
                add(event, "gate1-fitting")
            elif action in {"blocked", "deferred"}:
                add(event, "gate1-blocked")
            for field, tag in (
                ("fitting_requests", "gate1-fitting"),
                ("blocked_requests", "gate1-blocked"),
            ):
                for request in _parse_request_ids(event.fields.get(field)):
                    add(event, tag, request)
        elif category in {"decision", "admission", "gate2", "gate-2"}:
            if action in {"admit", "admitted"}:
                add(event, "gate2-admitted")
            elif action in {"defer", "deferred"}:
                add(event, "gate2-deferred")
            elif action == "ineligible":
                add(event, "gate2-ineligible")
            for field, tag in (
                ("admitted_requests", "gate2-admitted"),
                ("deferred_requests", "gate2-deferred"),
            ):
                for request in _parse_request_ids(event.fields.get(field)):
                    add(event, tag, request)
        elif category == "submit":
            add(event, "gen-submit")
        elif category == "gen-service" and action == "decode-start-proxy":
            add(event, "gen-service-start")
        elif category == "ctx-transfer":
            if action in {"queued", "send-queued"}:
                add(event, "ctx-queued")
            elif action == "timer-start":
                add(event, "ctx-timer-start")
            elif action in {"deadline-observed", "timeout", "timed-out"}:
                add(event, "ctx-deadline")
            elif action == "cancel-result":
                add(event, "ctx-cancel-result")
            elif action in _TERMINAL_ACTIONS:
                add(event, "ctx-terminal")
        elif category == "gen-transfer":
            if action == "timer-start":
                add(event, "gen-timer-start")
            elif action in {"deadline-observed", "timeout", "timed-out"}:
                add(event, "gen-deadline")
            elif action == "cancel-result":
                add(event, "gen-cancel-result")
        elif category in {"sender-transfer", "python-transfer"} and role == "ctx":
            if action in {
                "credit-received",
                "receiver-info-ready",
                "request-info-complete",
            }:
                add(event, "ctx-receiver-credit")
            elif action in {
                "service-start",
                "first-write",
                "first-write-submitted",
            }:
                add(event, "ctx-first-write")
            elif action in {
                "local-complete",
                "physical-complete",
                "kv-physical-complete",
            }:
                add(event, "ctx-kv-physical-complete")
            elif action in _TERMINAL_ACTIONS:
                add(event, "ctx-terminal")
        elif category in {"receiver-transfer", "python-transfer"} and role == "gen":
            if action in {"request-info-sent", "request-info-submitted"}:
                add(event, "gen-request-info")
            elif action == "local-ready":
                add(event, "gen-local-ready")
            elif action in _TERMINAL_ACTIONS:
                add(event, "gen-terminal")

    records: list[dict[str, object]] = []
    relationship_counts: Counter[str] = Counter()
    joined_request_count = 0
    for request, tags in sorted(points.items()):
        selected_points = {
            tag: _select_cross_host_point(tag, tag_points)
            for tag, tag_points in sorted(tags.items())
            if tag_points
        }
        first_gate1_points = tags.get("gate1-fitting", []) + tags.get("gate1-blocked", [])
        if first_gate1_points:
            selected_points["gate1-first"] = _select_cross_host_point(
                "gate1-first", first_gate1_points
            )
        deadline_point = selected_points.get("ctx-deadline")
        if deadline_point is not None:
            gate2_state = _gate2_state_at_deadline(tags, float(deadline_point["wall_t"]))
            if gate2_state is not None:
                selected_points["gate2-state-at-ctx-deadline"] = gate2_state
        has_ctx = any(tag.startswith("ctx-") for tag in selected_points)
        has_gen = any(tag.startswith("gen-") or tag.startswith("gate") for tag in selected_points)
        if has_ctx and has_gen:
            joined_request_count += 1

        deadline_relationship = _classify_cross_host_deadline(selected_points)
        if deadline_relationship is not None:
            relationship_counts[deadline_relationship] += 1

        records.append(
            {
                "request": request,
                "joined_ctx_gen": has_ctx and has_gen,
                "points": selected_points,
                "wall_intervals_s": {
                    name: _wall_interval(selected_points, start_tag, end_tag)
                    for name, start_tag, end_tag in (
                        ("ctx_timer_to_gen_arrival", "ctx-timer-start", "gen-arrival"),
                        ("ctx_timer_to_first_gate1", "ctx-timer-start", "gate1-first"),
                        ("ctx_timer_to_first_gate2_defer", "ctx-timer-start", "gate2-deferred"),
                        ("ctx_timer_to_gate2_admit", "ctx-timer-start", "gate2-admitted"),
                        ("ctx_timer_to_gen_submit", "ctx-timer-start", "gen-submit"),
                        ("ctx_timer_to_deadline", "ctx-timer-start", "ctx-deadline"),
                        ("gate2_defer_to_ctx_deadline", "gate2-deferred", "ctx-deadline"),
                    )
                },
                "ctx_deadline_relationship": deadline_relationship,
                "ctx_deadline_phase_coverage": _cross_host_phase_coverage(selected_points),
                "wall_clock_anomalies": _cross_host_wall_anomalies(selected_points),
            }
        )

    return {
        "clock_policy": (
            "Unix wall timestamps permit CTX/GEN request correlation across "
            "hosts but depend on cluster clock synchronization and may step. "
            "Boundary-sampled points are preferred; emission points include "
            "logger and polling delay. Cross-rank progress boundaries select "
            "the latest observed emitter while deadline/defer triggers select "
            "the earliest. Emitter coverage is observed, not proof that every "
            "required rank logged. Use this view for causal classification and "
            "long queue delays, not sub-millisecond service or throughput "
            "estimation."
        ),
        "request_count": len(records),
        "joined_ctx_gen_request_count": joined_request_count,
        "ctx_deadline_relationship_counts": dict(sorted(relationship_counts.items())),
        "requests": records,
    }


def _select_cross_host_point(
    tag: str,
    tag_points: list[dict[str, object]],
) -> dict[str, object]:
    """Select an earliest trigger or conservative latest progress boundary."""
    points_by_emitter: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    select_earliest = tag in {
        "ctx-queued",
        "ctx-timer-start",
        "ctx-deadline",
        "gate2-deferred",
        "gate1-first",
    }
    for point in tag_points:
        emitter = (
            point["log_source"],
            point["host"],
            point["instance"],
            point["role"],
            point["rank"],
        )
        points_by_emitter[emitter].append(point)

    selector = min if select_earliest else max
    emitter_points = []
    for emitter_candidates in points_by_emitter.values():
        boundary_candidates = [
            point for point in emitter_candidates if point["wall_semantics"] == "boundary-sampled"
        ]
        preferred = boundary_candidates or emitter_candidates
        emitter_points.append(selector(preferred, key=lambda point: float(point["wall_t"])))
    selected = dict(selector(emitter_points, key=lambda point: float(point["wall_t"])))
    selected["selection"] = "earliest" if select_earliest else "latest"
    selected["observed_emitter_count"] = len(emitter_points)
    selected["observed_hosts"] = sorted({str(point["host"]) for point in emitter_points})
    selected["observed_ranks"] = sorted(
        {str(point["rank"]) for point in emitter_points},
        key=_rank_sort_key,
    )
    selected["wall_semantics_seen"] = sorted({str(point["wall_semantics"]) for point in tag_points})
    return selected


def _event_wall_time(event: DiagnosticEvent) -> float | None:
    if _normalize_diag_token(event.fields.get("wall_clock", "")) != "unix":
        return None
    wall_time_s = _as_float(event.fields.get("wall_t"))
    if wall_time_s is None or not math.isfinite(wall_time_s):
        return None
    return wall_time_s


def _gate2_state_at_deadline(
    tags: dict[str, list[dict[str, object]]],
    deadline_t: float,
) -> dict[str, object] | None:
    transitions_by_emitter: dict[tuple[object, ...], list[tuple[str, dict[str, object]]]] = (
        defaultdict(list)
    )
    for state, tag in (
        ("deferred", "gate2-deferred"),
        ("ineligible", "gate2-ineligible"),
        ("admitted", "gate2-admitted"),
    ):
        for point in tags.get(tag, ()):
            if float(point["wall_t"]) > deadline_t:
                continue
            emitter = (
                point["log_source"],
                point["host"],
                point["instance"],
                point["role"],
                point["rank"],
            )
            transitions_by_emitter[emitter].append((state, point))
    if not transitions_by_emitter:
        return None

    emitter_states = [
        max(
            transitions,
            key=lambda item: (
                float(item[1]["wall_t"]),
                {"deferred": 0, "ineligible": 1, "admitted": 2}[item[0]],
            ),
        )
        for transitions in transitions_by_emitter.values()
    ]
    observed_states = {state for state, _ in emitter_states}
    if len(observed_states) == 1:
        state = next(iter(observed_states))
    elif "admitted" in observed_states:
        state = "partial-admission"
    elif "ineligible" in observed_states:
        state = "ineligible"
    else:
        state = "deferred"
    _, point = max(emitter_states, key=lambda item: float(item[1]["wall_t"]))
    result = dict(point)
    result["state"] = state
    result["observed_emitter_count"] = len(emitter_states)
    result["observed_emitter_state_counts"] = dict(
        sorted(Counter(emitter_state for emitter_state, _ in emitter_states).items())
    )
    result["selection"] = "latest-transition-at-or-before-deadline"
    return result


def _wall_interval(
    points: dict[str, dict[str, object]],
    start_tag: str,
    end_tag: str,
) -> float | None:
    start = points.get(start_tag)
    end = points.get(end_tag)
    if start is None or end is None:
        return None
    interval_s = float(end["wall_t"]) - float(start["wall_t"])
    return interval_s if interval_s >= 0.0 else None


def _classify_cross_host_deadline(
    points: dict[str, dict[str, object]],
) -> str | None:
    deadline = points.get("ctx-deadline")
    if deadline is None:
        return None
    deadline_t = float(deadline["wall_t"])
    timer_start = points.get("ctx-timer-start")
    if timer_start is not None and deadline_t < float(timer_start["wall_t"]):
        return "wall-clock-order-uncertain"
    phase_coverage = _cross_host_phase_coverage(points)
    if phase_coverage is not None and phase_coverage["timestamp_inversions"]:
        return "wall-clock-order-uncertain"

    gate2_admitted = points.get("gate2-admitted")
    if gate2_admitted is None or float(gate2_admitted["wall_t"]) > deadline_t:
        gate2_state = points.get("gate2-state-at-ctx-deadline")
        if gate2_state is not None and gate2_state.get("state") in {
            "admitted",
            "partial-admission",
        }:
            return "partial-gate2-admission-before-global-admission"
        if gate2_state is not None and gate2_state.get("state") == "deferred":
            return "during-gate2-deferral"
        if gate2_state is not None and gate2_state.get("state") == "ineligible":
            return "gate2-ineligible-at-deadline"
        gate1 = points.get("gate1-first")
        if gate1 is None or float(gate1["wall_t"]) > deadline_t:
            return "before-gen-gate1"
        return "before-gen-gate2-admission"

    ordered_milestones = (
        ("gate2-admitted", "after-gen-admission-before-submit"),
        ("gen-submit", "after-gen-submit-before-ctx-credit"),
        ("ctx-receiver-credit", "after-ctx-credit-before-first-write"),
        ("ctx-first-write", "during-kv-physical-transfer"),
        (
            "ctx-kv-physical-complete",
            "after-kv-physical-complete-before-terminal",
        ),
        ("ctx-terminal", "after-terminal"),
    )
    furthest_phase = "after-gen-admission-before-submit"
    for tag, phase in ordered_milestones:
        milestone = points.get(tag)
        if milestone is not None and float(milestone["wall_t"]) <= deadline_t:
            furthest_phase = phase
    return furthest_phase


def _cross_host_phase_coverage(
    points: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    deadline = points.get("ctx-deadline")
    if deadline is None:
        return None
    deadline_t = float(deadline["wall_t"])
    ordered_tags = (
        "gate2-admitted",
        "gen-submit",
        "ctx-receiver-credit",
        "ctx-first-write",
        "ctx-kv-physical-complete",
        "ctx-terminal",
    )
    observed = [
        tag for tag in ordered_tags if tag in points and float(points[tag]["wall_t"]) <= deadline_t
    ]
    if not observed:
        return {
            "observed_at_or_before_deadline": [],
            "missing_before_furthest": [],
            "timestamp_inversions": [],
        }
    furthest_index = max(ordered_tags.index(tag) for tag in observed)
    missing = [tag for tag in ordered_tags[:furthest_index] if tag not in observed]
    inversions = []
    prior_tag = None
    prior_time = None
    for tag in observed:
        tag_time = float(points[tag]["wall_t"])
        if prior_time is not None and tag_time < prior_time:
            inversions.append(f"{prior_tag}->{tag}")
        prior_tag = tag
        prior_time = tag_time
    return {
        "observed_at_or_before_deadline": observed,
        "missing_before_furthest": missing,
        "timestamp_inversions": inversions,
    }


def _cross_host_wall_anomalies(
    points: dict[str, dict[str, object]],
) -> list[str]:
    anomalies: list[str] = []
    for start_tag, end_tag in (
        ("ctx-timer-start", "ctx-deadline"),
        ("gate2-admitted", "gen-submit"),
        ("gen-submit", "ctx-receiver-credit"),
        ("ctx-receiver-credit", "ctx-first-write"),
        ("ctx-first-write", "ctx-kv-physical-complete"),
        ("ctx-kv-physical-complete", "ctx-terminal"),
    ):
        start = points.get(start_tag)
        end = points.get(end_tag)
        if start is not None and end is not None and float(end["wall_t"]) < float(start["wall_t"]):
            used_multi_emitter_selection = (
                int(start.get("observed_emitter_count", 1)) > 1
                or int(end.get("observed_emitter_count", 1)) > 1
            )
            selected_different_emitters = _cross_host_emitter(start) != _cross_host_emitter(end)
            prefix = (
                "cross-emitter-selection"
                if used_multi_emitter_selection and selected_different_emitters
                else "negative"
            )
            anomalies.append(f"{prefix}:{start_tag}->{end_tag}")
    return anomalies


def _cross_host_emitter(point: dict[str, object]) -> tuple[object, ...]:
    return (
        point.get("log_source"),
        point.get("host"),
        point.get("instance"),
        point.get("role"),
        point.get("rank"),
    )


def _evaluate_lifecycle_interval(
    request: str,
    domain: _ClockDomain,
    marks_by_tag: dict[str, list[_LifecycleMark]],
    start_tags: tuple[str, ...],
    end_tags: tuple[str, ...],
    tag_domains: dict[tuple[str, str], set[_ClockDomain]],
) -> dict[str, object]:
    starts = [mark for tag in start_tags for mark in marks_by_tag.get(tag, ())]
    ends = [mark for tag in end_tags for mark in marks_by_tag.get(tag, ())]
    starts.sort(key=lambda mark: mark.time_s)
    ends.sort(key=lambda mark: mark.time_s)
    if not starts and not ends:
        return {
            "status": "not-applicable",
            "duration_s": None,
            "censor_reason": None,
        }

    start = starts[0] if starts else None
    end = (
        next((candidate for candidate in ends if candidate.time_s >= start.time_s), None)
        if start is not None
        else None
    )
    if start is not None and end is not None:
        return {
            "status": "observed",
            "duration_s": end.time_s - start.time_s,
            "start_t": start.time_s,
            "start_kind": f"{start.category}:{start.action or start.tag}",
            "end_t": end.time_s,
            "end_kind": f"{end.category}:{end.action or end.tag}",
            "censor_reason": None,
        }

    if start is None:
        reason = _cross_domain_endpoint_reason(
            "start",
            request,
            domain,
            start_tags,
            tag_domains,
        )
        if reason is None:
            reason = "missing_start"
        return {
            "status": "censored",
            "duration_s": None,
            "start_t": None,
            "end_t": ends[0].time_s if ends else None,
            "censor_reason": reason,
        }

    if ends:
        reason = "end_precedes_start"
    else:
        reason = _cross_domain_endpoint_reason(
            "end",
            request,
            domain,
            end_tags,
            tag_domains,
        )
        if reason is None:
            reason = "missing_end"
    return {
        "status": "censored",
        "duration_s": None,
        "start_t": start.time_s,
        "end_t": None,
        "censor_reason": reason,
    }


def _classify_deadline_phase(
    marks_by_tag: dict[str, list[_LifecycleMark]],
) -> str | None:
    """Locate a CTX deadline in the sender-side transfer lifecycle."""
    deadlines = marks_by_tag.get("ctx-deadline", ())
    if not deadlines:
        return None
    deadline_t = deadlines[0].time_s

    phase_boundaries = (
        ("pre-credit", "sender-credit"),
        ("sender-queue", "sender-first-write"),
        ("kv-transfer-service", "sender-kv-physical-complete"),
        ("completion-visibility", "ctx-terminal"),
    )
    for phase, boundary_tag in phase_boundaries:
        boundaries = marks_by_tag.get(boundary_tag, ())
        if not boundaries or boundaries[0].time_s > deadline_t:
            return phase
    return "after-terminal"


def _cross_domain_endpoint_reason(
    endpoint: str,
    request: str,
    domain: _ClockDomain,
    tags: tuple[str, ...],
    tag_domains: dict[tuple[str, str], set[_ClockDomain]],
) -> str | None:
    other_domains = {
        candidate
        for tag in tags
        for candidate in tag_domains.get((request, tag), ())
        if candidate != domain
    }
    if not other_domains:
        return None
    if any(candidate[0] != domain[0] for candidate in other_domains):
        return f"{endpoint}_in_other_log_source"
    return f"{endpoint}_in_other_clock_domain"


def _lifecycle_interval_coverage(
    request_records: list[dict[str, object]],
    marks: list[_LifecycleMark],
) -> dict[str, object]:
    records_by_request: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in request_records:
        records_by_request[str(record["request"])].append(record)
    mark_domains: dict[tuple[str, str], set[_ClockDomain]] = defaultdict(set)
    for mark in marks:
        mark_domains[(mark.request, mark.tag)].add(_lifecycle_domain(mark))

    coverage: dict[str, object] = {}
    for name, (start_tags, end_tags) in _LIFECYCLE_INTERVALS.items():
        durations: list[float] = []
        eligible_requests = 0
        observed_requests = 0
        reasons: Counter[str] = Counter()
        for request, records in records_by_request.items():
            start_domains = {
                domain for tag in start_tags for domain in mark_domains.get((request, tag), ())
            }
            end_domains = {
                domain for tag in end_tags for domain in mark_domains.get((request, tag), ())
            }
            if not start_domains and not end_domains:
                continue
            eligible_requests += 1
            request_durations = [
                float(interval["duration_s"])
                for record in records
                if isinstance((interval := record["intervals"][name]), dict)
                and interval.get("status") == "observed"
                and interval.get("duration_s") is not None
            ]
            if request_durations:
                observed_requests += 1
                durations.extend(request_durations)
                continue
            if start_domains and end_domains:
                common_domains = start_domains.intersection(end_domains)
                if common_domains:
                    reasons["end_precedes_start"] += 1
                elif any(
                    start_domain[0] != end_domain[0]
                    for start_domain in start_domains
                    for end_domain in end_domains
                ):
                    reasons["cross_log_source_only"] += 1
                else:
                    reasons["cross_clock_domain_only"] += 1
            elif start_domains:
                reasons["missing_end"] += 1
            else:
                reasons["missing_start"] += 1
        coverage[name] = {
            "eligible_requests": eligible_requests,
            "observed_requests": observed_requests,
            "observed_samples": len(durations),
            "censored_requests": eligible_requests - observed_requests,
            "censor_reasons": dict(sorted(reasons.items())),
            "duration_s": _summary(durations),
        }
    return coverage


def _analyze_remaining_work_ground_truth(
    events: list[DiagnosticEvent],
) -> dict[str, object]:
    marks = _collect_lifecycle_marks(events)
    timelines: dict[tuple[_ClockDomain, str], dict[str, list[_LifecycleMark]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tag_domains: dict[tuple[str, str], set[_ClockDomain]] = defaultdict(set)
    for mark in marks:
        domain = _lifecycle_domain(mark)
        timelines[(domain, mark.request)][mark.tag].append(mark)
        tag_domains[(mark.request, mark.tag)].add(domain)
    for tags in timelines.values():
        for tag_marks in tags.values():
            tag_marks.sort(key=lambda mark: mark.time_s)

    samples: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    omission_seen: set[tuple[object, ...]] = set()
    membership_overflow = _collect_membership_overflow(events)
    active_request_ids_omitted = 0
    active_request_ids_recovered_from_overflow = 0
    for event in events:
        category = _normalize_diag_token(event.category)
        if category not in {"decision", "admission"}:
            continue
        if "active_requests" not in event.fields and "active_requests_omitted" not in event.fields:
            continue
        sequence = event.fields.get("sequence")
        domain = _event_domain(event)
        active_overflow = _membership_overflow_for_event(event, "active", membership_overflow)
        active_pairs = _parse_request_blocks(event.fields.get("active_requests")) + active_overflow
        active_blocks = dict(active_pairs)
        active_requests = [request for request, _ in active_pairs]
        omitted = _as_int(event.fields.get("active_requests_omitted")) or 0
        omission_identity = (
            domain,
            sequence or event.time_s,
        )
        if omission_identity not in omission_seen:
            recovered = min(omitted, len(active_overflow))
            active_request_ids_recovered_from_overflow += recovered
            active_request_ids_omitted += max(0, omitted - recovered)
            omission_seen.add(omission_identity)
        if not active_requests:
            continue
        for request in active_requests:
            identity = (domain, sequence or event.time_s, request)
            if identity in seen:
                continue
            seen.add(identity)
            tags = timelines.get((domain, request), {})
            ready = min(
                (
                    mark
                    for tag in ("local-ready", "receiver-completed")
                    for mark in tags.get(tag, ())
                    if mark.time_s >= event.time_s
                ),
                key=lambda mark: mark.time_s,
                default=None,
            )
            reap = next(
                (mark for mark in tags.get("reap", ()) if mark.time_s >= event.time_s),
                None,
            )
            prior_submits = [mark for mark in tags.get("submit", ()) if mark.time_s <= event.time_s]
            active_age_s = event.time_s - prior_submits[-1].time_s if prior_submits else None
            ready_reason = _remaining_endpoint_censor_reason(
                request,
                domain,
                ("local-ready", "receiver-completed"),
                event.time_s,
                tags,
                tag_domains,
            )
            reap_reason = _remaining_endpoint_censor_reason(
                request,
                domain,
                "reap",
                event.time_s,
                tags,
                tag_domains,
            )
            samples.append(
                {
                    "request": request,
                    "blocks": active_blocks.get(request),
                    "sequence": sequence,
                    "decision_t": event.time_s,
                    "log_source": domain[0],
                    "host": domain[1],
                    "role": domain[2],
                    "rank": domain[3],
                    "clock": domain[4],
                    "instance": domain[5],
                    "clock_domain": _clock_domain_label(domain),
                    "active_age_s": active_age_s,
                    "active_age_bucket": _active_age_bucket(active_age_s),
                    "ready_t": ready.time_s if ready is not None else None,
                    "ready_kind": (
                        f"{ready.category}:{ready.action or ready.tag}"
                        if ready is not None
                        else None
                    ),
                    "residual_ready_s": (
                        ready.time_s - event.time_s if ready is not None else None
                    ),
                    "ready_censor_reason": (None if ready is not None else ready_reason),
                    "reap_t": reap.time_s if reap is not None else None,
                    "residual_reap_s": (reap.time_s - event.time_s if reap is not None else None),
                    "reap_censor_reason": (None if reap is not None else reap_reason),
                }
            )

    samples.sort(
        key=lambda sample: (
            str(sample["log_source"]),
            str(sample["clock_domain"]),
            float(sample["decision_t"]),
            str(sample["request"]),
        )
    )
    by_age_bucket: dict[str, object] = {}
    bucket_names = [label for _, label in _ACTIVE_AGE_BUCKETS] + ["unknown"]
    for bucket in bucket_names:
        bucket_samples = [sample for sample in samples if sample["active_age_bucket"] == bucket]
        if not bucket_samples:
            continue
        by_age_bucket[bucket] = {
            "samples": len(bucket_samples),
            "residual_ready_s": _summary(
                [
                    float(sample["residual_ready_s"])
                    for sample in bucket_samples
                    if sample["residual_ready_s"] is not None
                ]
            ),
            "residual_reap_s": _summary(
                [
                    float(sample["residual_reap_s"])
                    for sample in bucket_samples
                    if sample["residual_reap_s"] is not None
                ]
            ),
            "ready_coverage": _remaining_coverage(bucket_samples, "ready"),
            "reap_coverage": _remaining_coverage(bucket_samples, "reap"),
        }

    return {
        "definition": (
            "For each Gate-2 admission snapshot and active request, residual_ready_s and "
            "residual_reap_s use only later GEN events in the identical input source, "
            "instance, and clock domain. CTX timestamps are never used."
        ),
        "active_decision_samples": len(samples),
        "active_request_ids_omitted": active_request_ids_omitted,
        "active_request_ids_recovered_from_overflow": (active_request_ids_recovered_from_overflow),
        "identity_coverage": {
            "observed": len(samples),
            "omitted": active_request_ids_omitted,
            "fraction": _safe_ratio(
                len(samples),
                len(samples) + active_request_ids_omitted,
            ),
        },
        "unique_requests": len({str(sample["request"]) for sample in samples}),
        "samples": samples,
        "residual_ready_s": _summary(
            [
                float(sample["residual_ready_s"])
                for sample in samples
                if sample["residual_ready_s"] is not None
            ]
        ),
        "residual_reap_s": _summary(
            [
                float(sample["residual_reap_s"])
                for sample in samples
                if sample["residual_reap_s"] is not None
            ]
        ),
        "active_age_s": _summary(
            [
                float(sample["active_age_s"])
                for sample in samples
                if sample["active_age_s"] is not None
            ]
        ),
        "ready_coverage": _remaining_coverage(samples, "ready"),
        "reap_coverage": _remaining_coverage(samples, "reap"),
        "by_active_age_bucket": by_age_bucket,
    }


def _remaining_endpoint_censor_reason(
    request: str,
    domain: _ClockDomain,
    tag: str | tuple[str, ...],
    decision_time_s: float,
    tags: dict[str, list[_LifecycleMark]],
    tag_domains: dict[tuple[str, str], set[_ClockDomain]],
) -> str:
    endpoint_tags = (tag,) if isinstance(tag, str) else tag
    endpoint_name = tag if isinstance(tag, str) else "ready"
    same_domain = [mark for endpoint_tag in endpoint_tags for mark in tags.get(endpoint_tag, ())]
    if any(mark.time_s < decision_time_s for mark in same_domain):
        return f"{endpoint_name}_precedes_decision"
    other_domains = {
        candidate
        for endpoint_tag in endpoint_tags
        for candidate in tag_domains.get((request, endpoint_tag), ())
        if candidate != domain
    }
    if any(candidate[0] != domain[0] for candidate in other_domains):
        return f"{endpoint_name}_in_other_log_source"
    if other_domains:
        return f"{endpoint_name}_in_other_clock_domain"
    return f"missing_{endpoint_name}"


def _remaining_coverage(
    samples: list[dict[str, object]],
    endpoint: str,
) -> dict[str, object]:
    residual_field = f"residual_{endpoint}_s"
    reason_field = f"{endpoint}_censor_reason"
    observed = sum(sample.get(residual_field) is not None for sample in samples)
    reasons = Counter(
        str(sample[reason_field])
        for sample in samples
        if sample.get(residual_field) is None and sample.get(reason_field) is not None
    )
    return {
        "eligible": len(samples),
        "observed": observed,
        "censored": len(samples) - observed,
        "censor_reasons": dict(sorted(reasons.items())),
    }


def _known_backlog_release_analysis(ranks: dict[str, object]) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for rank, rank_analysis in ranks.items():
        if not isinstance(rank_analysis, dict):
            continue
        release_analysis = rank_analysis.get("release_to_admission")
        if not isinstance(release_analysis, dict):
            continue
        by_source = release_analysis.get("by_source")
        if not isinstance(by_source, dict):
            continue
        for release_source, source_analysis in by_source.items():
            if not isinstance(source_analysis, dict):
                continue
            source_samples = source_analysis.get("samples")
            if not isinstance(source_samples, list):
                continue
            for sample in source_samples:
                if isinstance(sample, dict) and not sample.get("backlog_identity_unknown", True):
                    samples.append(
                        {
                            "rank": rank,
                            "release_source": release_source,
                            **sample,
                        }
                    )

    return {
        "known_backlog_only": True,
        "samples": samples,
        "release_samples": len(samples),
        "next_decision_coverage": {
            "observed": sum(sample.get("decision_gap_s") is not None for sample in samples),
            "censored": sum(sample.get("decision_gap_s") is None for sample in samples),
        },
        "successful_admission_coverage": {
            "observed": sum(
                sample.get("successful_admission_gap_s") is not None for sample in samples
            ),
            "censored": sum(sample.get("successful_admission_gap_s") is None for sample in samples),
        },
        "release_to_next_decision_s": _summary(
            [
                float(sample["decision_gap_s"])
                for sample in samples
                if sample.get("decision_gap_s") is not None
            ]
        ),
        "release_to_successful_admission_s": _summary(
            [
                float(sample["successful_admission_gap_s"])
                for sample in samples
                if sample.get("successful_admission_gap_s") is not None
            ]
        ),
    }


def _parse_request_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    request_ids: list[str] = []
    for item in value.strip("[](){}").split(","):
        token = item.strip()
        if not token or token in {"-", "none", "None", "null"}:
            continue
        request = token.split(":", 1)[0].strip()
        if request and request not in {"-", "none", "None", "null"}:
            request_ids.append(request)
    return request_ids


def _event_correlation_request(event: DiagnosticEvent) -> str | None:
    for field in (
        "context_request",
        "context_request_id",
        "disagg_request",
        "disagg_request_id",
    ):
        request = event.fields.get(field)
        if request not in {None, "", "-", "-1", "0"}:
            return request
    for field in ("request", "request_id"):
        request = event.fields.get(field)
        if request not in {None, "", "-", "-1"}:
            return request
    return None


def _event_local_request(event: DiagnosticEvent) -> str | None:
    for field in ("local_request", "local_request_id", "request", "request_id"):
        request = event.fields.get(field)
        if request not in {None, "", "-", "-1"}:
            return request
    return None


def _normalize_diag_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _event_role(event: DiagnosticEvent, category: str, action: str) -> str:
    explicit_role = _normalize_diag_token(event.fields.get("role", ""))
    if explicit_role in {"ctx", "context", "sender"}:
        return "ctx"
    if explicit_role in {"gen", "generation", "receiver"}:
        return "gen"
    if category in {"ctx-transfer", "sender-transfer"}:
        return "ctx"
    if category in {
        "gen-arrival",
        "gen-activation",
        "gen-transfer",
        "gate1",
        "gate-1",
        "decision",
        "admission",
        "gate2",
        "gate-2",
        "submit",
        "decision-members",
        "receiver-transfer",
        "receiver-slot",
        "reap",
        "gen-service",
        "status-poll",
    }:
        return "gen"
    if category == "python-transfer":
        if action in _CTX_ACTIONS:
            return "ctx"
        if action in _GEN_ACTIONS or action in _TERMINAL_ACTIONS:
            return "gen"
    return "unknown"


def _event_host(event: DiagnosticEvent) -> str:
    return (
        event.fields.get("host")
        or event.fields.get("hostname")
        or event.fields.get("node")
        or "<unspecified>"
    )


def _event_clock(event: DiagnosticEvent) -> str:
    return event.fields.get("clock_domain") or event.fields.get("clock") or "local_steady"


def _event_domain(event: DiagnosticEvent) -> _ClockDomain:
    category = _normalize_diag_token(event.category)
    action = _normalize_diag_token(event.fields.get("action", ""))
    return (
        event.source or "<in-memory>",
        _event_host(event),
        _event_role(event, category, action),
        event.rank,
        _event_clock(event),
        event.fields.get("instance", "-"),
    )


def _lifecycle_domain(mark: _LifecycleMark) -> _ClockDomain:
    return mark.log_source, mark.host, mark.role, mark.rank, mark.clock, mark.instance


def _clock_domain_label(domain: _ClockDomain) -> str:
    source, host, role, rank, clock, instance = domain
    return f"{source}::host={host}::role={role}::rank={rank}::clock={clock}::instance={instance}"


def _active_age_bucket(active_age_s: float | None) -> str:
    if active_age_s is None or active_age_s < 0.0:
        return "unknown"
    for upper_bound, label in _ACTIVE_AGE_BUCKETS:
        if active_age_s < upper_bound:
            return label
    return "unknown"


def _membership_snapshot_key(
    event: DiagnosticEvent, membership: str, *, definition: bool = False
) -> tuple[_ClockDomain, str, str, str] | None:
    reference_field = "snapshot_version" if definition else f"{membership}_snapshot"
    reference = event.fields.get(reference_field) or event.fields.get("sequence")
    if reference is None:
        return None
    return (
        _event_domain(event),
        event.fields.get("instance", "-"),
        reference,
        membership,
    )


def _membership_overflow_for_event(
    event: DiagnosticEvent,
    membership: str,
    overflow: dict[
        tuple[_ClockDomain, str, str, str],
        list[tuple[str, float]],
    ],
) -> list[tuple[str, float]]:
    key = _membership_snapshot_key(event, membership)
    if key is None:
        return []
    tail = overflow.get(key, [])
    omitted = _as_int(event.fields.get(f"{membership}_requests_omitted"))
    if omitted is not None and omitted != len(tail):
        return []
    return tail


def _collect_admissions(events: list[DiagnosticEvent]) -> tuple[list[Admission], dict[str, float]]:
    admissions: list[Admission] = []
    request_blocks: dict[str, float] = {}
    membership_overflow = _collect_membership_overflow(events)
    decision_keys = {
        (
            _event_domain(event),
            event.fields.get("instance", "-"),
            event.fields.get("sequence"),
        )
        for event in events
        if _normalize_diag_token(event.category) == "decision"
        and event.fields.get("sequence") is not None
    }
    compatibility_admissions = {
        (
            _event_domain(event),
            event.fields.get("instance", "-"),
            event.fields.get("sequence"),
        ): event
        for event in events
        if _normalize_diag_token(event.category) == "admission"
        and event.fields.get("sequence") is not None
    }
    for event in events:
        category = _normalize_diag_token(event.category)
        if category not in {"decision", "admission"}:
            continue
        event_key = (
            _event_domain(event),
            event.fields.get("instance", "-"),
            event.fields.get("sequence"),
        )
        if category == "admission" and event_key in decision_keys:
            continue
        if category == "decision" and event_key in compatibility_admissions:
            compatibility_event = compatibility_admissions[event_key]
            event = DiagnosticEvent(
                category=event.category,
                time_s=event.time_s,
                rank=event.rank,
                fields={**compatibility_event.fields, **event.fields},
                source=event.source,
            )
        candidate_overflow = _membership_overflow_for_event(event, "candidate", membership_overflow)
        admitted_overflow = _membership_overflow_for_event(event, "admitted", membership_overflow)
        deferred_overflow = _membership_overflow_for_event(event, "deferred", membership_overflow)
        candidate_requests = (
            _parse_request_blocks(event.fields.get("candidate_requests")) + candidate_overflow
        )
        admitted_request_blocks = (
            _parse_request_blocks(event.fields.get("admitted_requests")) + admitted_overflow
        )
        deferred_request_blocks = (
            _parse_request_blocks(event.fields.get("deferred_requests")) + deferred_overflow
        )
        admitted = _as_int(event.fields.get("admitted"))
        deferred = _as_int(event.fields.get("deferred"))
        if admitted is None:
            admitted = len(admitted_request_blocks)
        if deferred is None:
            deferred = len(deferred_request_blocks)
        if admitted < 0 or deferred < 0:
            continue
        candidate_omitted = max(
            0,
            (_as_int(event.fields.get("candidate_requests_omitted")) or 0)
            - len(candidate_overflow),
        )
        if (
            event.fields.get("candidate_snapshot") is not None
            and candidate_omitted == 0
            and len(candidate_requests) >= admitted + deferred
        ):
            admitted_request_blocks = candidate_requests[:admitted]
            deferred_request_blocks = candidate_requests[admitted : admitted + deferred]
            admitted_omitted = 0
            deferred_omitted = 0
        else:
            admitted_omitted = max(
                0,
                (_as_int(event.fields.get("admitted_requests_omitted")) or 0)
                - len(admitted_overflow),
            )
            deferred_omitted = max(
                0,
                (_as_int(event.fields.get("deferred_requests_omitted")) or 0)
                - len(deferred_overflow),
            )
        for request, blocks in (
            candidate_requests + admitted_request_blocks + deferred_request_blocks
        ):
            request_blocks[request] = blocks
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
                candidate_requests_omitted=candidate_omitted,
                admitted_requests_omitted=admitted_omitted,
                deferred_requests_omitted=deferred_omitted,
            )
        )
    admissions.sort(key=lambda admission: admission.time_s)
    return admissions, request_blocks


def _collect_membership_overflow(
    events: list[DiagnosticEvent],
) -> dict[
    tuple[_ClockDomain, str, str, str],
    list[tuple[str, float]],
]:
    chunks: dict[
        tuple[_ClockDomain, str, str, str],
        dict[int, list[tuple[str, float]]],
    ] = defaultdict(dict)
    expected_chunk_counts: dict[tuple[_ClockDomain, str, str, str], int] = {}
    conflicts: set[tuple[_ClockDomain, str, str, str]] = set()
    for event in events:
        if _normalize_diag_token(event.category) != "decision-members":
            continue
        membership = _normalize_diag_token(event.fields.get("membership", ""))
        if membership not in {"active", "candidate", "admitted", "deferred"}:
            continue
        chunk_index = _as_int(event.fields.get("chunk_index"))
        chunk_count = _as_int(event.fields.get("chunk_count"))
        if (
            chunk_index is None
            or chunk_index <= 0
            or chunk_count is None
            or chunk_count <= 0
            or chunk_index > chunk_count
        ):
            continue
        key = _membership_snapshot_key(event, membership, definition=True)
        if key is None:
            continue
        previous_count = expected_chunk_counts.setdefault(key, chunk_count)
        if previous_count != chunk_count:
            conflicts.add(key)
            continue
        request_blocks = _parse_request_blocks(event.fields.get("requests"))
        previous_chunk = chunks[key].get(chunk_index)
        if previous_chunk is not None and previous_chunk != request_blocks:
            conflicts.add(key)
            continue
        chunks[key][chunk_index] = request_blocks

    overflow: dict[
        tuple[_ClockDomain, str, str, str],
        list[tuple[str, float]],
    ] = {}
    for key, key_chunks in chunks.items():
        expected = expected_chunk_counts[key]
        if key in conflicts or set(key_chunks) != set(range(1, expected + 1)):
            continue
        request_blocks = [
            request_block
            for chunk_index in range(1, expected + 1)
            for request_block in key_chunks[chunk_index]
        ]
        request_ids = [request for request, _ in request_blocks]
        if len(request_ids) != len(set(request_ids)):
            continue
        overflow[key] = request_blocks
    return overflow


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
        category = _normalize_diag_token(event.category)
        if category in {"decision", "admission"}:
            for field in ("candidate_requests", "admitted_requests", "deferred_requests"):
                pairs.extend(_parse_request_blocks(event.fields.get(field)))
        elif category == "decision-members":
            pairs.extend(_parse_request_blocks(event.fields.get("requests")))
        elif category in {"gate1", "gate-1", "gate2", "gate-2"}:
            request = _event_correlation_request(event)
            blocks = _as_float(event.fields.get("blocks"))
            if request is not None and blocks is not None and blocks >= 0.0:
                pairs.append((request, blocks))
        elif category in {"submit", "reap"}:
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
        if (request := _event_correlation_request(event)) is not None
        and _event_outcome(event) is False
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
    transfer_categories = {
        "ctx-transfer",
        "gen-transfer",
        "sender-transfer",
        "receiver-transfer",
        "python-transfer",
    }
    if event.category in transfer_categories and action in {
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "aborted",
    }:
        return False
    if event.category in transfer_categories and action in {
        "completed",
        "complete",
        "success",
        "succeeded",
    }:
        return True

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
            request = _event_correlation_request(acquired) or _event_correlation_request(event)
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
    if admission is None or admission.deferred <= 0 or admission.deferred_requests_omitted > 0:
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
            or admission.candidate_requests_omitted > 0
            or admission.deferred_requests_omitted > 0
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

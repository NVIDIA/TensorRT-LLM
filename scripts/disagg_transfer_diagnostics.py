#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize opt-in disaggregated KV-transfer diagnostic events.

The runtime writes compact JSON objects after ``[DISAGG_TRANSFER_DIAG]``. This
tool tolerates unrelated and malformed log lines, groups valid events by their
canonical request ID, and derives durations only when both boundaries share a
``(host, pid)`` monotonic-clock domain.
"""

from __future__ import annotations

import argparse
import fileinput
import json
import statistics
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DIAGNOSTICS_LOG_PREFIX = "[DISAGG_TRANSFER_DIAG] "
_GEN_KV_ADMISSION_EVENT = "gen_kv_admission_result"
_TIMELINE_FIELDS = (
    "event",
    "side",
    "wall_ns",
    "monotonic_ns",
    "host",
    "pid",
    "instance",
    "rank",
    "tp_rank",
    "pp_rank",
    "cp_rank",
    "dp_rank",
    "local_request_id",
    "outcome",
    "policy",
    "slice_id",
    "peer_rank",
    "peer_instance",
    "receiver_slice_id",
    "is_last_slice",
    "worker_queue_index",
    "transfer_entries",
    "ownership_enabled",
    "block_reuse_id",
    "state",
    "reason",
    "timeout_ms",
    "timeout_owner",
    "timer_start_monotonic_ns",
    "elapsed_ms",
    "cancellation_requested",
    "session_status",
    "resources_drained",
    "transfer_bytes",
    "expected_receivers",
    "expected_writers",
    "prompt_tokens",
    "tokens_per_block",
    "cache_present",
    "capacity_tokens",
    "history_tokens",
    "capacity_block_equivalent",
    "request_blocks",
    "active_transfer_blocks",
    "admitted_transfer_blocks",
    "transfer_block_budget",
    "legacy_budget_outcome",
    "legacy_active_transfer_blocks",
    "legacy_admitted_transfer_blocks",
    "legacy_limited_by_budget",
    "source_kv_request_owned",
    "source_kv_reuse_pinned",
    "init_requests",
    "transfers_in_progress",
    "transfers_complete",
    "kv_admitted_this_iteration",
    "decode_requests",
    "kv_pool_max_blocks",
    "kv_pool_free_blocks",
    "kv_pool_used_blocks",
    "index_free_slots",
    "dropped_events",
)

Event = dict[str, object]
ClockDomain = tuple[str, int]
Participant = tuple[str | None, str | None, int | None, int | None]


@dataclass(frozen=True)
class _ParsedEvent:
    record: Event
    line_number: int


@dataclass(frozen=True)
class ParseResult:
    """Events and accounting produced while scanning mixed runtime logs."""

    events: list[_ParsedEvent]
    total_lines: int
    ignored_lines: int
    malformed_diagnostic_lines: int


@dataclass(frozen=True)
class _Phase:
    name: str
    start_event: str
    end_event: str
    correlation_fields: tuple[str, ...] = ()
    start_outcomes: frozenset[str] = frozenset()
    end_outcomes: frozenset[str] = frozenset()
    start_fields: tuple[tuple[str, object], ...] = ()
    end_fields: tuple[tuple[str, object], ...] = ()
    single_pair: bool = False
    signed_offset: bool = False
    report_unmatched: bool = True


_PHASES = (
    _Phase(
        "gen_gate1_admission_wait",
        "gen_ingress",
        _GEN_KV_ADMISSION_EVENT,
        end_outcomes=frozenset({"admitted"}),
        single_pair=True,
    ),
    _Phase(
        "gen_transfer_window_admission_wait",
        _GEN_KV_ADMISSION_EVENT,
        "gen_transfer_window_result",
        start_outcomes=frozenset({"admitted"}),
        end_outcomes=frozenset({"admitted"}),
        single_pair=True,
        report_unmatched=False,
    ),
    _Phase("gen_receive_lifetime", "gen_receive_start", "gen_transfer_settled"),
    _Phase(
        "gen_transfer_to_service",
        "gen_transfer_settled",
        "gen_decode_ready",
        start_outcomes=frozenset({"completed"}),
    ),
    _Phase(
        "ctx_receiver_readiness_offset",
        "ctx_send_ready",
        "ctx_all_receivers_ready",
        signed_offset=True,
    ),
    _Phase(
        "ctx_worker_queue_wait",
        "ctx_transfer_queued",
        "ctx_backend_submit_start",
        correlation_fields=("slice_id", "peer_rank"),
    ),
    _Phase(
        "ctx_backend_submission",
        "ctx_backend_submit_start",
        "ctx_backend_submitted",
        correlation_fields=("slice_id", "peer_rank"),
    ),
    _Phase(
        "ctx_backend_service",
        "ctx_backend_submitted",
        "ctx_backend_complete",
        correlation_fields=("slice_id", "peer_rank"),
    ),
    _Phase("ctx_transfer_lifetime", "ctx_send_ready", "ctx_transfer_settled"),
    _Phase(
        "ctx_source_kv_request_ownership",
        "ctx_send_ready",
        "ctx_source_kv_released",
    ),
    _Phase(
        "gen_writer_first_response",
        "gen_request_data_sent",
        "gen_writer_result_received",
        correlation_fields=("slice_id", "peer_rank"),
        single_pair=True,
    ),
    _Phase(
        "gen_destination_drain",
        "gen_writer_result_received",
        "gen_destination_complete",
        correlation_fields=("slice_id", "peer_rank"),
        start_fields=(("outcome", "success"), ("is_last_slice", True)),
        report_unmatched=False,
    ),
    _Phase(
        "ctx_send_to_timeout_start",
        "ctx_send_ready",
        "transfer_timeout_started",
        correlation_fields=("side",),
        end_fields=(("side", "ctx"),),
    ),
    _Phase(
        "gen_receive_to_timeout_start",
        "gen_receive_start",
        "transfer_timeout_started",
        correlation_fields=("side",),
        end_fields=(("side", "gen"),),
    ),
    _Phase(
        "transfer_timeout_window",
        "transfer_timeout_started",
        "transfer_timeout_observed",
        correlation_fields=("side", "timeout_owner"),
    ),
)


def parse_lines(lines: Iterable[str]) -> ParseResult:
    """Extract valid diagnostic JSON objects from mixed log lines."""
    events: list[_ParsedEvent] = []
    total_lines = 0
    ignored_lines = 0
    malformed_lines = 0

    for line_number, line in enumerate(lines, start=1):
        total_lines += 1
        marker = line.find(DIAGNOSTICS_LOG_PREFIX)
        if marker < 0:
            ignored_lines += 1
            continue

        payload = line[marker + len(DIAGNOSTICS_LOG_PREFIX) :].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            malformed_lines += 1
            continue
        events.append(_ParsedEvent(record=record, line_number=line_number))

    return ParseResult(
        events=events,
        total_lines=total_lines,
        ignored_lines=ignored_lines,
        malformed_diagnostic_lines=malformed_lines,
    )


def _clock_domain(record: Event) -> ClockDomain | None:
    host = record.get("host")
    pid = record.get("pid")
    if not isinstance(host, str) or not host or not isinstance(pid, int) or isinstance(pid, bool):
        return None
    return host, pid


def _monotonic_ns(record: Event) -> int | None:
    value = (
        record.get("timer_start_monotonic_ns", record.get("monotonic_ns"))
        if record.get("event") == "transfer_timeout_started"
        else record.get("monotonic_ns")
    )
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _matches_phase_boundary(
    event: _ParsedEvent,
    name: str,
    outcomes: frozenset[str],
    required_fields: tuple[tuple[str, object], ...],
) -> bool:
    record = event.record
    return (
        record.get("event") == name
        and (not outcomes or record.get("outcome") in outcomes)
        and all(record.get(field) == value for field, value in required_fields)
    )


def _correlation(record: Event, fields: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(record.get(field) for field in fields)


def _domain_json(domain: ClockDomain) -> dict[str, object]:
    return {"host": domain[0], "pid": domain[1]}


def _participant(record: Event) -> Participant:
    side = record.get("side")
    host = record.get("host")
    pid = record.get("pid")
    rank = record.get("rank")
    return (
        side if isinstance(side, str) else None,
        host if isinstance(host, str) else None,
        pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
    )


def _participant_json(participant: Participant) -> dict[str, object]:
    side, host, pid, rank = participant
    return {"side": side, "host": host, "pid": pid, "rank": rank}


def _timeline(events: list[_ParsedEvent]) -> list[dict[str, object]]:
    timeline = []
    for event in sorted(events, key=_timeline_sort_key):
        entry = {field: event.record[field] for field in _TIMELINE_FIELDS if field in event.record}
        entry["line_number"] = event.line_number
        timeline.append(entry)
    return timeline


def _timeline_sort_key(event: _ParsedEvent) -> tuple[int, int, int]:
    wall_ns = event.record.get("wall_ns")
    if isinstance(wall_ns, int) and not isinstance(wall_ns, bool) and wall_ns >= 0:
        return 0, wall_ns, event.line_number
    return 1, 0, event.line_number


def _derive_phase(
    events: list[_ParsedEvent], phase: _Phase
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    starts = [
        event
        for event in events
        if _matches_phase_boundary(
            event,
            phase.start_event,
            phase.start_outcomes,
            phase.start_fields,
        )
    ]
    ends = [
        event
        for event in events
        if _matches_phase_boundary(
            event,
            phase.end_event,
            phase.end_outcomes,
            phase.end_fields,
        )
    ]
    if not starts and not ends:
        return [], []
    if not phase.report_unmatched and (not starts or not ends):
        return [], []

    Boundary = tuple[ClockDomain, tuple[object, ...], int]

    def timed(boundaries: list[_ParsedEvent]) -> list[Boundary]:
        result = []
        for event in boundaries:
            domain = _clock_domain(event.record)
            timestamp = _monotonic_ns(event.record)
            if domain is not None and timestamp is not None:
                result.append(
                    (domain, _correlation(event.record, phase.correlation_fields), timestamp)
                )
        return result

    timed_starts = timed(starts)
    timed_ends = timed(ends)
    grouped_starts: dict[tuple[ClockDomain, tuple[object, ...]], list[int]] = defaultdict(list)
    grouped_ends: dict[tuple[ClockDomain, tuple[object, ...]], list[int]] = defaultdict(list)
    for domain, correlation, timestamp in timed_starts:
        grouped_starts[(domain, correlation)].append(timestamp)
    for domain, correlation, timestamp in timed_ends:
        grouped_ends[(domain, correlation)].append(timestamp)

    durations: list[dict[str, object]] = []
    unmeasured: list[dict[str, object]] = []
    all_keys = sorted(grouped_starts.keys() | grouped_ends.keys(), key=lambda key: repr(key))
    for domain, correlation in all_keys:
        domain_starts = sorted(grouped_starts.get((domain, correlation), []))
        domain_ends = sorted(grouped_ends.get((domain, correlation), []))
        used_end_indexes: set[int] = set()
        unmatched_starts = 0

        for start_ns in domain_starts[:1] if phase.single_pair else domain_starts:
            candidates = [
                index
                for index, end_ns in enumerate(domain_ends)
                if index not in used_end_indexes and (phase.signed_offset or end_ns >= start_ns)
            ]
            if not candidates:
                unmatched_starts += 1
                continue
            key = (
                (lambda index: abs(domain_ends[index] - start_ns))
                if phase.signed_offset
                else (lambda index: domain_ends[index])
            )
            end_index = min(candidates, key=key)
            used_end_indexes.add(end_index)
            duration_ns = domain_ends[end_index] - start_ns
            result: dict[str, object] = {
                "phase": phase.name,
                "clock_domain": _domain_json(domain),
            }
            if phase.signed_offset:
                result.update(
                    {
                        "signed_offset_ns": duration_ns,
                        "signed_offset_ms": duration_ns / 1_000_000,
                        "readiness_wait_ms": max(duration_ns, 0) / 1_000_000,
                        "readiness_lead_ms": max(-duration_ns, 0) / 1_000_000,
                    }
                )
            else:
                result.update(
                    {
                        "duration_ns": duration_ns,
                        "duration_ms": duration_ns / 1_000_000,
                    }
                )
            if phase.correlation_fields:
                result["correlation"] = dict(zip(phase.correlation_fields, correlation))
            durations.append(result)

        if phase.single_pair:
            missing_end_count = int(bool(domain_starts) and not used_end_indexes)
            missing_start_count = int(not domain_starts and bool(domain_ends))
        else:
            missing_end_count = unmatched_starts
            missing_start_count = len(domain_ends) - len(used_end_indexes)
        if phase.report_unmatched and (missing_end_count or missing_start_count):
            for reason, count in (
                ("missing_end", missing_end_count),
                ("missing_start", missing_start_count),
            ):
                if count == 0:
                    continue
                gap: dict[str, object] = {
                    "phase": phase.name,
                    "reason": reason,
                    "count": count,
                    "clock_domain": _domain_json(domain),
                }
                if phase.correlation_fields:
                    gap["correlation"] = dict(zip(phase.correlation_fields, correlation))
                unmeasured.append(gap)

    if len(timed_starts) != len(starts) or len(timed_ends) != len(ends):
        unmeasured.append(
            {
                "phase": phase.name,
                "reason": "invalid_clock_metadata",
                "start_count": len(starts) - len(timed_starts),
                "end_count": len(ends) - len(timed_ends),
            }
        )
    if durations:
        return durations, unmeasured
    start_domains = {start[0] for start in timed_starts}
    end_domains = {end[0] for end in timed_ends}
    if not phase.single_pair and unmeasured:
        if starts and ends and start_domains and start_domains.isdisjoint(end_domains):
            unmeasured.append({"phase": phase.name, "reason": "clock_domain_mismatch"})
        return [], unmeasured
    if not starts or not ends:
        reason = "missing_start" if not starts else "missing_end"
    elif len(timed_starts) != len(starts) or len(timed_ends) != len(ends):
        reason = "invalid_clock_metadata"
    elif start_domains.isdisjoint(end_domains):
        reason = "clock_domain_mismatch"
    else:
        reason = "no_ordered_matching_boundaries"
    summary = {"phase": phase.name, "reason": reason}
    if summary not in unmeasured:
        unmeasured.append(summary)
    return [], unmeasured


def _has_event(
    events: list[_ParsedEvent],
    name: str,
    *,
    outcomes: frozenset[str] = frozenset(),
    excluded_policies: frozenset[str] = frozenset(),
    required_fields: tuple[tuple[str, object], ...] = (),
) -> bool:
    return any(
        event.record.get("event") == name
        and (not outcomes or event.record.get("outcome") in outcomes)
        and event.record.get("policy") not in excluded_policies
        and all(event.record.get(field) == value for field, value in required_fields)
        for event in events
    )


def _missing_boundaries(events: list[_ParsedEvent]) -> list[str]:
    expected: set[str] = set()
    observed = {event.record.get("event") for event in events}

    if _has_event(events, "gen_ingress"):
        expected.add(_GEN_KV_ADMISSION_EVENT)
    if _has_event(
        events,
        "gen_transfer_window_result",
        outcomes=frozenset({"admitted"}),
        excluded_policies=frozenset({"not_applicable"}),
    ):
        expected.add("gen_receive_start")
    if _has_event(events, "gen_receive_start"):
        expected.update(("gen_request_data_sent", "gen_transfer_settled"))
    if _has_event(events, "gen_request_data_sent"):
        expected.add("gen_writer_result_received")
    if _has_event(
        events,
        "gen_writer_result_received",
        outcomes=frozenset({"success"}),
        required_fields=(("is_last_slice", True),),
    ):
        expected.add("gen_destination_complete")
    if _has_event(events, "gen_transfer_settled", outcomes=frozenset({"completed"})):
        expected.add("gen_decode_ready")
    if _has_event(events, "transfer_timeout_observed"):
        expected.add("transfer_timeout_started")

    if _has_event(events, "ctx_send_ready"):
        expected.update(
            (
                "ctx_all_receivers_ready",
                "ctx_source_kv_released",
                "ctx_transfer_settled",
            )
        )
    if _has_event(events, "ctx_transfer_queued"):
        expected.add("ctx_backend_submit_start")
    if _has_event(events, "ctx_backend_submit_start"):
        expected.add("ctx_backend_submitted")
    if _has_event(events, "ctx_backend_submitted"):
        expected.add("ctx_backend_complete")

    return sorted(event for event in expected if event not in observed)


def _request_sort_key(request_id: object) -> tuple[str, str]:
    return type(request_id).__name__, str(request_id)


def _participant_summaries(events: list[_ParsedEvent]) -> list[dict[str, object]]:
    grouped: dict[Participant, list[_ParsedEvent]] = defaultdict(list)
    for event in events:
        grouped[_participant(event.record)].append(event)

    summaries = []
    for participant, participant_events in sorted(grouped.items(), key=lambda item: repr(item[0])):
        event_counts = Counter(str(event.record["event"]) for event in participant_events)
        summary = _participant_json(participant)
        summary.update(
            {
                "event_count": len(participant_events),
                "event_counts": dict(sorted(event_counts.items())),
                "missing_boundaries": _missing_boundaries(participant_events),
            }
        )
        summaries.append(summary)
    return summaries


def _summarize_request(request_id: object, events: list[_ParsedEvent]) -> dict[str, object]:
    event_counts = Counter(str(event.record["event"]) for event in events)
    sides = sorted(
        {side for event in events if isinstance((side := event.record.get("side")), str)}
    )
    domains = sorted(
        {domain for event in events if (domain := _clock_domain(event.record)) is not None}
    )
    durations: list[dict[str, object]] = []
    unmeasured: list[dict[str, object]] = []
    for phase in _PHASES:
        phase_durations, phase_unmeasured = _derive_phase(events, phase)
        durations.extend(phase_durations)
        unmeasured.extend(phase_unmeasured)

    return {
        "request_id": request_id,
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "sides": sides,
        "clock_domains": [_domain_json(domain) for domain in domains],
        "missing_boundaries": _missing_boundaries(events),
        "participants": _participant_summaries(events),
        "durations": durations,
        "unmeasured_phases": unmeasured,
        "timeline": _timeline(events),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _phase_summary(requests: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    samples: dict[str, list[float]] = defaultdict(list)
    for request in requests:
        durations = request["durations"]
        assert isinstance(durations, list)
        for duration in durations:
            assert isinstance(duration, dict)
            phase = duration.get("phase")
            value = duration.get("duration_ms")
            if value is None:
                value = duration.get("signed_offset_ms")
            if isinstance(phase, str) and isinstance(value, (int, float)):
                samples[phase].append(float(value))

    summary: dict[str, dict[str, object]] = {}
    for phase, unsorted_values in sorted(samples.items()):
        values = sorted(unsorted_values)
        summary[phase] = {
            "count": len(values),
            "min_ms": values[0],
            "mean_ms": statistics.fmean(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": values[-1],
        }
    return summary


def _snapshot_cadence(events: list[_ParsedEvent]) -> list[dict[str, object]]:
    samples: dict[tuple[ClockDomain, int | None], list[int]] = defaultdict(list)
    for event in events:
        if event.record.get("event") != "gen_kv_pool_snapshot":
            continue
        domain = _clock_domain(event.record)
        timestamp = _monotonic_ns(event.record)
        rank = event.record.get("rank")
        rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        if domain is not None and timestamp is not None:
            samples[(domain, rank)].append(timestamp)

    result = []
    for (domain, rank), timestamps in sorted(samples.items(), key=lambda item: repr(item[0])):
        ordered = sorted(timestamps)
        intervals_ms = [
            (current - previous) / 1_000_000 for previous, current in zip(ordered, ordered[1:])
        ]
        item: dict[str, object] = {
            **_domain_json(domain),
            "rank": rank,
            "snapshot_count": len(ordered),
            "interval_count": len(intervals_ms),
        }
        if intervals_ms:
            item.update(
                {
                    "min_ms": min(intervals_ms),
                    "mean_ms": statistics.fmean(intervals_ms),
                    "max_ms": max(intervals_ms),
                }
            )
        result.append(item)
    return result


def _transfer_settled_to_next_decision(events: list[_ParsedEvent]) -> list[dict[str, object]]:
    decisions: dict[tuple[ClockDomain, int | None], list[int]] = defaultdict(list)
    settled_events: dict[tuple[ClockDomain, int | None], list[int]] = defaultdict(list)
    for event in events:
        domain = _clock_domain(event.record)
        timestamp = _monotonic_ns(event.record)
        rank = event.record.get("rank")
        rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        if domain is None or timestamp is None:
            continue
        key = domain, rank
        if event.record.get("event") == "gen_kv_pool_snapshot":
            decisions[key].append(timestamp)
        elif event.record.get("event") == "gen_transfer_settled":
            settled_events[key].append(timestamp)

    result = []
    for (domain, rank), settled_times in sorted(
        settled_events.items(), key=lambda item: repr(item[0])
    ):
        decision_times = sorted(decisions.get((domain, rank), []))
        delays_ms = []
        unmatched = 0
        for settled_time in settled_times:
            index = bisect_left(decision_times, settled_time)
            if index == len(decision_times):
                unmatched += 1
            else:
                delays_ms.append((decision_times[index] - settled_time) / 1_000_000)
        item: dict[str, object] = {
            **_domain_json(domain),
            "rank": rank,
            "settled_count": len(settled_times),
            "matched_count": len(delays_ms),
            "unmatched_count": unmatched,
        }
        if delays_ms:
            item.update(
                {
                    "min_ms": min(delays_ms),
                    "mean_ms": statistics.fmean(delays_ms),
                    "max_ms": max(delays_ms),
                }
            )
        result.append(item)
    return result


def analyze(parse_result: ParseResult) -> dict[str, object]:
    """Group parsed events and produce request- and phase-level summaries."""
    grouped: dict[tuple[str, str], tuple[object, list[_ParsedEvent]]] = {}
    ungrouped_events: list[_ParsedEvent] = []
    ungrouped_event_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    schema_versions: Counter[str] = Counter()

    for event in parse_result.events:
        record = event.record
        event_name = str(record["event"])
        event_counts[event_name] += 1
        schema_versions[str(record.get("schema_version", "missing"))] += 1
        request_id = record.get("request_id")
        if request_id is None:
            ungrouped_events.append(event)
            ungrouped_event_counts[event_name] += 1
            continue
        key = _request_sort_key(request_id)
        if key not in grouped:
            grouped[key] = (request_id, [])
        grouped[key][1].append(event)

    requests = [
        _summarize_request(request_id, events)
        for _, (request_id, events) in sorted(grouped.items())
    ]
    return {
        "clock_semantics": {
            "durations": "Derived only from monotonic_ns within the same (host, pid).",
            "timeline": (
                "wall_ns is preserved for manual cross-process correlation; cross-host ordering "
                "is clock-sync-sensitive and no wall-clock deltas are derived."
            ),
            "gen_transfer_settled_to_next_scheduler_decision": (
                "A same-domain proxy for release-to-next-admission opportunity; "
                "gen_transfer_settled precedes final session close and state transition."
            ),
        },
        "summary": {
            "total_lines": parse_result.total_lines,
            "ignored_lines": parse_result.ignored_lines,
            "malformed_diagnostic_lines": parse_result.malformed_diagnostic_lines,
            "parsed_events": len(parse_result.events),
            "events_without_request_id": sum(ungrouped_event_counts.values()),
            "request_count": len(requests),
        },
        "event_counts": dict(sorted(event_counts.items())),
        "schema_versions": dict(sorted(schema_versions.items())),
        "ungrouped_event_counts": dict(sorted(ungrouped_event_counts.items())),
        "aggregate_timeline": _timeline(ungrouped_events),
        "scheduler_decision_cadence": _snapshot_cadence(ungrouped_events),
        "gen_transfer_settled_to_next_scheduler_decision": (
            _transfer_settled_to_next_decision(parse_result.events)
        ),
        "phase_durations": _phase_summary(requests),
        "requests": requests,
    }


def analyze_lines(lines: Iterable[str]) -> dict[str, object]:
    """Parse and analyze mixed runtime log lines."""
    return analyze(parse_lines(lines))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        help="Log files to analyze. Read standard input when no file is provided.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write JSON to this file.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line analyzer."""
    args = _parse_args(argv)
    with fileinput.input(files=args.logs or ("-",), encoding="utf-8", errors="replace") as lines:
        result = analyze_lines(lines)

    if args.output is None:
        json.dump(result, sys.stdout, indent=args.indent, sort_keys=True)
        sys.stdout.write("\n")
    else:
        with args.output.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=args.indent, sort_keys=True)
            output.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

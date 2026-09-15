#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize opt-in disaggregated KV-transfer diagnostic events.

The runtime writes compact JSON objects after ``[DISAGG_TRANSFER_DIAG]``. This
tool tolerates unrelated and malformed log lines. Requests are correlated across
processes only with a shared run UUID; otherwise analysis stays process-local.
Durations require matching run/process UUIDs, host, and PID. Legacy records
without a process UUID remain readable but cannot establish safe timing.

Startup ``diagnostic_capabilities`` records describe which event groups each
executor/transceiver can emit. Unsupported boundaries are not missing events;
logs without this metadata retain identity-safe measured pairs but have
unassessed expectations.
"""

from __future__ import annotations

import argparse
import fileinput
import json
import math
import statistics
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from uuid import UUID

DIAGNOSTICS_LOG_PREFIX = "[DISAGG_TRANSFER_DIAG] "
_SUPPORTED_SCHEMA_VERSION = 1
_GEN_KV_ADMISSION_EVENT = "gen_kv_admission_result"
_CAPABILITY_FIELDS = ("executor_events", "scheduler_kv_admission_events", "python_transfer_events")
_TIMELINE_FIELDS = (
    "event",
    "side",
    "wall_ns",
    "monotonic_ns",
    "host",
    "pid",
    "run_uuid",
    "process_uuid",
    "run_uuid_status",
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
    "source_kv_reuse_block_count",
    "state",
    "reason",
    "timeout_ms",
    "timeout_expected",
    "timeout_owner",
    "timer_start_monotonic_ns",
    "elapsed_ms",
    "cancellation_requested",
    "session_status",
    "session_found",
    "resources_drained",
    "transfer_bytes",
    "expected_receivers",
    "expected_writers",
    "writer_cohort_known",
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
    "capability_schema_version",
    "transceiver_runtime",
    *_CAPABILITY_FIELDS,
)

Event = dict[str, object]
ClockDomain = tuple[str, int, str, str | None]
Participant = tuple[str | None, str | None, int | None, int | None, str | None, str | None]
CapabilityScope = tuple[str, int, str, str | None, int | None]

_STRING_FIELDS = frozenset(
    {
        "event",
        "host",
        "instance",
        "legacy_budget_outcome",
        "outcome",
        "peer_instance",
        "policy",
        "reason",
        "session_status",
        "side",
        "state",
        "timeout_owner",
    }
)
_IDENTIFIER_FIELDS = frozenset({"request_id", "local_request_id"})
_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "active_transfer_blocks",
        "admitted_transfer_blocks",
        "capacity_block_equivalent",
        "capacity_tokens",
        "cp_rank",
        "decode_requests",
        "dp_rank",
        "dropped_events",
        "expected_receivers",
        "expected_writers",
        "history_tokens",
        "index_free_slots",
        "init_requests",
        "kv_admitted_this_iteration",
        "kv_pool_free_blocks",
        "kv_pool_max_blocks",
        "kv_pool_used_blocks",
        "legacy_active_transfer_blocks",
        "legacy_admitted_transfer_blocks",
        "monotonic_ns",
        "peer_rank",
        "pid",
        "pp_rank",
        "prompt_tokens",
        "rank",
        "receiver_slice_id",
        "request_blocks",
        "slice_id",
        "source_kv_reuse_block_count",
        "timer_start_monotonic_ns",
        "timeout_ms",
        "tokens_per_block",
        "tp_rank",
        "transfer_block_budget",
        "transfer_bytes",
        "transfer_entries",
        "transfers_complete",
        "transfers_in_progress",
        "wall_ns",
        "worker_queue_index",
    }
)
_NONNEGATIVE_NUMBER_FIELDS = frozenset({"elapsed_ms"})
_BOOLEAN_FIELDS = frozenset(
    {
        "cache_present",
        "cancellation_requested",
        "is_last_slice",
        "legacy_limited_by_budget",
        "ownership_enabled",
        "resources_drained",
        "session_found",
        "source_kv_request_owned",
        "source_kv_reuse_pinned",
        "timeout_expected",
        "writer_cohort_known",
    }
)
_MAX_TIMESTAMP_NS = (1 << 63) - 1
_MAX_IDENTIFIER = (1 << 64) - 1


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


def _is_diagnostic_scalar(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (str, int, bool))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _has_valid_field_types(record: Event) -> bool:
    """Reject records that cannot conform to the runtime's scalar schema."""
    if not all(_is_diagnostic_scalar(value) for value in record.values()):
        return False
    schema_version = record.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _SUPPORTED_SCHEMA_VERSION
    ):
        return False
    for field in _IDENTIFIER_FIELDS:
        value = record.get(field)
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= _MAX_IDENTIFIER
        ):
            return False
    for field in _NONNEGATIVE_INTEGER_FIELDS:
        value = record.get(field)
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= _MAX_TIMESTAMP_NS
        ):
            return False
    for field in _NONNEGATIVE_NUMBER_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            return False
        if isinstance(value, int):
            if value > _MAX_TIMESTAMP_NS:
                return False
        elif not math.isfinite(value):
            return False
    for field in _STRING_FIELDS:
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            return False
    for field in _BOOLEAN_FIELDS:
        value = record.get(field)
        if value is not None and not isinstance(value, bool):
            return False
    return True


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
        "ctx_worker_dequeued",
        correlation_fields=("slice_id", "peer_rank"),
    ),
    _Phase(
        "ctx_worker_preparation",
        "ctx_worker_dequeued",
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
        start_fields=(("timeout_expected", True),),
        end_fields=(("side", "ctx"),),
    ),
    _Phase(
        "gen_receive_to_timeout_start",
        "gen_receive_start",
        "transfer_timeout_started",
        correlation_fields=("side",),
        start_fields=(("timeout_expected", True),),
        end_fields=(("side", "gen"),),
    ),
    _Phase(
        "transfer_timeout_window",
        "transfer_timeout_started",
        "transfer_timeout_observed",
        correlation_fields=("side", "timeout_owner"),
        report_unmatched=False,
    ),
)

_EVENT_CAPABILITIES = {
    event: "python_transfer_events"
    for phase in _PHASES
    for event in (phase.start_event, phase.end_event)
}
_EVENT_CAPABILITIES.update(
    dict.fromkeys(
        (
            "gen_ingress",
            "gen_transfer_window_result",
            "gen_decode_ready",
            "ctx_send_ready",
            "ctx_source_kv_released",
            "ctx_source_unpinned",
            "transfer_timeout_started",
            "transfer_timeout_observed",
        ),
        "executor_events",
    )
)
_EVENT_CAPABILITIES.update(
    dict.fromkeys(
        (_GEN_KV_ADMISSION_EVENT, "gen_kv_pool_snapshot"), "scheduler_kv_admission_events"
    )
)


@dataclass(frozen=True)
class _Capabilities:
    groups: dict[str, bool | None]
    runtime: str | None = None
    issues: tuple[str, ...] = ()

    def supports(self, event: str) -> bool | None:
        return self.groups[_EVENT_CAPABILITIES[event]]

    def to_json(self) -> dict[str, object]:
        known = sum(value is not None for value in self.groups.values())
        return {
            "status": "known"
            if known == len(_CAPABILITY_FIELDS)
            else "partial"
            if known
            else "unknown",
            "transceiver_runtime": self.runtime,
            **self.groups,
            "issues": list(self.issues),
        }


def _capability_scope(record: Event) -> CapabilityScope | None:
    domain = _clock_domain(record)
    return (*domain, _participant(record)[3]) if domain is not None else None


def _known_capability_version(record: Event) -> bool:
    return (
        type(record.get("capability_schema_version")) is int
        and record["capability_schema_version"] == 1
    )


def _capability_index(events: list[_ParsedEvent]) -> dict[CapabilityScope, _Capabilities]:
    declarations: dict[CapabilityScope, list[Event]] = defaultdict(list)
    for event in events:
        if event.record["event"] == "diagnostic_capabilities":
            scope = _capability_scope(event.record)
            if scope is not None:
                declarations[scope].append(event.record)
    result = {}
    for scope, records in declarations.items():
        issues = set()
        groups: dict[str, bool | None] = {}
        if not all(_known_capability_version(record) for record in records):
            issues.add("invalid_capability_schema_version")
        for field in _CAPABILITY_FIELDS:
            field_records = [record for record in records if field in record]
            values = [record[field] for record in field_records]
            if any(type(value) is not bool for value in values) or not all(
                _known_capability_version(record) for record in field_records
            ):
                issues.add(f"invalid_{field}")
                groups[field] = None
            elif len(set(values)) > 1:
                issues.add(f"conflicting_{field}")
                groups[field] = None
            else:
                groups[field] = values[0] is True if values else None
        runtimes = {
            record["transceiver_runtime"] for record in records if "transceiver_runtime" in record
        }
        runtime = next(iter(runtimes)) if len(runtimes) == 1 else None
        if runtimes and (
            len(runtimes) != 1
            or runtime not in ("CPP", "PYTHON")
            or any(
                not _known_capability_version(record)
                for record in records
                if "transceiver_runtime" in record
            )
        ):
            issues.add("invalid_or_conflicting_transceiver_runtime")
            groups["python_transfer_events"] = None
            runtime = None
        elif runtime is not None and groups["python_transfer_events"] not in (
            None,
            runtime == "PYTHON",
        ):
            issues.add("runtime_capability_mismatch")
            groups["python_transfer_events"] = None
        result[scope] = _Capabilities(
            groups, runtime if isinstance(runtime, str) else None, tuple(sorted(issues))
        )
    return result


def _participant_capabilities(
    events: list[_ParsedEvent], index: dict[CapabilityScope, _Capabilities]
) -> _Capabilities:
    scope = _capability_scope(events[0].record)
    profile = index.get(scope) if scope is not None else None
    if profile is None:
        return _Capabilities(
            dict.fromkeys(_CAPABILITY_FIELDS), issues=("missing_capability_metadata",)
        )
    # Contradictory records must not turn an observed Python path into a false
    # C++ exemption. Keep actual measurements, but mark the declaration unknown.
    groups = profile.groups.copy()
    issues = set(profile.issues)
    for event in events:
        field = _EVENT_CAPABILITIES.get(str(event.record["event"]))
        if field is not None and groups[field] is False:
            groups[field] = None
            issues.add(f"observed_unsupported_{field}")
    return _Capabilities(groups, profile.runtime, tuple(sorted(issues)))


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
        except (json.JSONDecodeError, RecursionError, ValueError):
            malformed_lines += 1
            continue
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("event"), str)
            or not _has_valid_field_types(record)
        ):
            malformed_lines += 1
            continue
        events.append(_ParsedEvent(record=record, line_number=line_number))

    return ParseResult(
        events=events,
        total_lines=total_lines,
        ignored_lines=ignored_lines,
        malformed_diagnostic_lines=malformed_lines,
    )


def _uuid_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _identity_issues(record: Event) -> list[str]:
    issues = []
    for field in ("run_uuid", "process_uuid"):
        if _uuid_value(record.get(field)) is None:
            invalid = record.get(field) is not None or (
                field == "run_uuid" and record.get("run_uuid_status") == "invalid"
            )
            issues.append(f"{'invalid' if invalid else 'missing'}_{field}")
    return issues


def _clock_domain(record: Event) -> ClockDomain | None:
    host = record.get("host")
    pid = record.get("pid")
    if not isinstance(host, str) or not host or not isinstance(pid, int) or isinstance(pid, bool):
        return None
    process_uuid = _uuid_value(record.get("process_uuid"))
    if process_uuid is None:
        return None
    return host, pid, process_uuid, _uuid_value(record.get("run_uuid"))


def _monotonic_ns(record: Event) -> int | None:
    value = (
        record.get("timer_start_monotonic_ns", record.get("monotonic_ns"))
        if record.get("event") == "transfer_timeout_started"
        else record.get("monotonic_ns")
    )
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_TIMESTAMP_NS:
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
    return {
        "host": domain[0],
        "pid": domain[1],
        "process_uuid": domain[2],
        "run_uuid": domain[3],
    }


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
        _uuid_value(record.get("process_uuid")),
        _uuid_value(record.get("run_uuid")),
    )


def _participant_json(participant: Participant) -> dict[str, object]:
    side, host, pid, rank, process_uuid, run_uuid = participant
    return {
        "side": side,
        "host": host,
        "pid": pid,
        "rank": rank,
        "process_uuid": process_uuid,
        "run_uuid": run_uuid,
    }


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


def _broadcast_writer_expectations(
    starts: list[_ParsedEvent], ends: list[_ParsedEvent]
) -> tuple[set[tuple[ClockDomain, Participant, object]], list[dict[str, object]]]:
    """Do not require every candidate in a GEN-first ADP broadcast to respond."""
    cohorts: dict[tuple[ClockDomain, Participant, object], list[Event]] = defaultdict(list)
    responders: dict[tuple[ClockDomain, Participant, object], set[object]] = defaultdict(set)
    for boundaries, is_start in ((starts, True), (ends, False)):
        for event in boundaries:
            record = event.record
            domain = _clock_domain(record)
            if (
                domain is None
                or _monotonic_ns(record) is None
                or record.get("slice_id") is None
                or record.get("peer_rank") is None
            ):
                continue
            key = domain, _participant(record), record["slice_id"]
            if is_start:
                cohorts[key].append(record)
            elif record.get("session_found") is not False:
                responders[key].add(record["peer_rank"])

    relaxed = set()
    gaps = []
    for (domain, participant, slice_id), records in cohorts.items():
        flags = {record.get("writer_cohort_known") for record in records}
        if flags == {True}:
            continue
        key = domain, participant, slice_id
        relaxed.add(key)
        observed = len(responders[key])
        counts = {record.get("expected_writers") for record in records}
        expected = next(iter(counts)) if len(counts) == 1 else None
        issue = None
        if flags != {False}:
            # Older or partial logs cannot prove that every published peer was
            # selected. Still retain all observed per-peer timing below.
            issue = "missing_or_conflicting_writer_cohort"
        elif expected is None or not isinstance(expected, int) or expected <= 0:
            issue = "missing_invalid_or_conflicting_expected_writers"
        if issue is None and observed == expected:
            continue
        gap: dict[str, object] = {
            "phase": "gen_writer_first_response",
            "clock_domain": _domain_json(domain),
            "correlation": {"slice_id": slice_id},
            "observed_writers": observed,
        }
        if participant[3] is not None:
            gap["rank"] = participant[3]
        if issue is not None:
            gap.update(reason="unknown_writer_cohort", detail=issue)
        else:
            gap.update(
                reason="missing_writer_responses"
                if observed < expected
                else "unexpected_writer_responses",
                expected_writers=expected,
                count=abs(expected - observed),
            )
        gaps.append(gap)
    return relaxed, gaps


def _derive_phase(
    events: list[_ParsedEvent], phase: _Phase, profiles: dict[Participant, _Capabilities]
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

    Boundary = tuple[ClockDomain, Participant, tuple[object, ...], int]

    def timed(boundaries: list[_ParsedEvent]) -> tuple[list[Boundary], int, int, int]:
        result = []
        invalid_clock_count = 0
        missing_correlation_count = 0
        unverified_identity_count = 0
        for event in boundaries:
            if _uuid_value(event.record.get("process_uuid")) is None:
                unverified_identity_count += 1
                continue
            domain = _clock_domain(event.record)
            timestamp = _monotonic_ns(event.record)
            if domain is None or timestamp is None:
                invalid_clock_count += 1
                continue
            correlation = _correlation(event.record, phase.correlation_fields)
            if any(value is None for value in correlation):
                missing_correlation_count += 1
                continue
            result.append((domain, _participant(event.record), correlation, timestamp))
        return result, invalid_clock_count, missing_correlation_count, unverified_identity_count

    timed_starts, invalid_clock_starts, missing_correlation_starts, unverified_starts = timed(
        starts
    )
    timed_ends, invalid_clock_ends, missing_correlation_ends, unverified_ends = timed(ends)
    grouped_starts: dict[tuple[ClockDomain, Participant, tuple[object, ...]], list[int]] = (
        defaultdict(list)
    )
    grouped_ends: dict[tuple[ClockDomain, Participant, tuple[object, ...]], list[int]] = (
        defaultdict(list)
    )
    for domain, participant, correlation, timestamp in timed_starts:
        grouped_starts[(domain, participant, correlation)].append(timestamp)
    for domain, participant, correlation, timestamp in timed_ends:
        grouped_ends[(domain, participant, correlation)].append(timestamp)

    durations: list[dict[str, object]] = []
    relaxed_writer_cohorts, unmeasured = (
        _broadcast_writer_expectations(starts, ends)
        if phase.name == "gen_writer_first_response"
        else (set(), [])
    )
    all_keys = sorted(grouped_starts.keys() | grouped_ends.keys(), key=lambda key: repr(key))
    for domain, participant, correlation in all_keys:
        domain_starts = sorted(grouped_starts.get((domain, participant, correlation), []))
        domain_ends = sorted(grouped_ends.get((domain, participant, correlation), []))
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
            if participant[3] is not None:
                result["rank"] = participant[3]
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
        if (
            relaxed_writer_cohorts
            and not domain_ends
            and (domain, participant, correlation[0]) in relaxed_writer_cohorts
        ):
            missing_end_count = 0
        if phase.report_unmatched and (missing_end_count or missing_start_count):
            for reason, count in (
                ("missing_end", missing_end_count),
                ("missing_start", missing_start_count),
            ):
                if count == 0:
                    continue
                gap: dict[str, object] = {
                    "phase": phase.name,
                    "reason": _phase_gap_reason(profiles[participant], phase, reason),
                    "count": count,
                    "clock_domain": _domain_json(domain),
                }
                if participant[3] is not None:
                    gap["rank"] = participant[3]
                if gap["reason"] != reason:
                    gap["missing_boundary"] = reason.removeprefix("missing_")
                if phase.correlation_fields:
                    gap["correlation"] = dict(zip(phase.correlation_fields, correlation))
                unmeasured.append(gap)

    for reason, start_count, end_count in (
        ("unverified_process_identity", unverified_starts, unverified_ends),
        ("invalid_clock_metadata", invalid_clock_starts, invalid_clock_ends),
        (
            "missing_correlation_fields",
            missing_correlation_starts,
            missing_correlation_ends,
        ),
    ):
        if start_count or end_count:
            unmeasured.append(
                {
                    "phase": phase.name,
                    "reason": reason,
                    "start_count": start_count,
                    "end_count": end_count,
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
        # Participant-specific gaps above already describe capability support;
        # do not add a second, unconditional missing-boundary summary.
        return [], unmeasured
    elif len(timed_starts) != len(starts) or len(timed_ends) != len(ends):
        return [], unmeasured
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


def _phase_gap_reason(profile: _Capabilities, phase: _Phase, reason: str) -> str:
    support = (profile.supports(phase.start_event), profile.supports(phase.end_event))
    if False in support:
        return "unsupported_capability"
    if None in support:
        return "unknown_capability"
    return reason


def _boundary_expectations(
    events: list[_ParsedEvent], profile: _Capabilities
) -> dict[str, list[str]]:
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
    writer_results = [
        event for event in events if event.record.get("event") == "gen_writer_result_received"
    ]
    if (
        writer_results
        and all(event.record.get("outcome") == "success" for event in writer_results)
        and any(event.record.get("is_last_slice") is True for event in writer_results)
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
        expected.add("ctx_worker_dequeued")
    if _has_event(events, "ctx_worker_dequeued"):
        expected.add("ctx_backend_submit_start")
    if _has_event(events, "ctx_backend_submit_start"):
        expected.add("ctx_backend_submitted")
    if _has_event(events, "ctx_backend_submitted"):
        expected.add("ctx_backend_complete")

    result: dict[str, list[str]] = {
        "missing_boundaries": [],
        "unsupported_boundaries": [],
        "unassessed_boundaries": [],
    }
    for event in sorted(expected - observed):
        support = profile.supports(event)
        key = (
            "missing_boundaries"
            if support is True
            else "unsupported_boundaries"
            if support is False
            else "unassessed_boundaries"
        )
        result[key].append(event)
    return result


def _request_sort_key(request_id: object) -> tuple[str, str]:
    return type(request_id).__name__, str(request_id)


def _request_group_key(event: _ParsedEvent) -> tuple[str, str, str, str]:
    record = event.record
    process_uuid = _uuid_value(record.get("process_uuid"))
    run_uuid = _uuid_value(record.get("run_uuid"))
    if process_uuid is None:
        # Even records from one input file can span restarts. Keep unidentified
        # records separate instead of manufacturing a request from reused IDs.
        scope, identity = "unverified", str(event.line_number)
    elif run_uuid is not None:
        scope, identity = "run", run_uuid
    else:
        scope, identity = "process", repr((process_uuid, record.get("host"), record.get("pid")))
    return scope, identity, *_request_sort_key(record["request_id"])


def _participant_summaries(
    grouped: dict[Participant, list[_ParsedEvent]], profiles: dict[Participant, _Capabilities]
) -> list[dict[str, object]]:
    summaries = []
    for participant, participant_events in sorted(grouped.items(), key=lambda item: repr(item[0])):
        event_counts = Counter(str(event.record["event"]) for event in participant_events)
        summary = _participant_json(participant)
        summary.update(
            {
                "event_count": len(participant_events),
                "event_counts": dict(sorted(event_counts.items())),
                "capabilities": profiles[participant].to_json(),
                **_boundary_expectations(participant_events, profiles[participant]),
            }
        )
        summaries.append(summary)
    return summaries


def _summarize_request(
    request_id: object,
    events: list[_ParsedEvent],
    capability_index: dict[CapabilityScope, _Capabilities],
) -> dict[str, object]:
    event_counts = Counter(str(event.record["event"]) for event in events)
    sides = sorted(
        {side for event in events if isinstance((side := event.record.get("side")), str)}
    )
    domains = sorted(
        {domain for event in events if (domain := _clock_domain(event.record)) is not None}
    )
    durations: list[dict[str, object]] = []
    unmeasured: list[dict[str, object]] = []
    grouped: dict[Participant, list[_ParsedEvent]] = defaultdict(list)
    for event in events:
        grouped[_participant(event.record)].append(event)
    profiles = {
        participant: _participant_capabilities(participant_events, capability_index)
        for participant, participant_events in grouped.items()
    }
    participants = _participant_summaries(grouped, profiles)
    for phase in _PHASES:
        phase_durations, phase_unmeasured = _derive_phase(events, phase, profiles)
        durations.extend(phase_durations)
        unmeasured.extend(phase_unmeasured)

    return {
        "request_id": request_id,
        "run_uuid": _uuid_value(events[0].record.get("run_uuid")),
        "correlation_scope": _request_group_key(events[0])[0],
        "identity_issues": sorted(
            {issue for event in events for issue in _identity_issues(event.record)}
        ),
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "sides": sides,
        "clock_domains": [_domain_json(domain) for domain in domains],
        **{
            field: sorted(
                {boundary for participant in participants for boundary in participant[field]}
            )
            for field in ("missing_boundaries", "unsupported_boundaries", "unassessed_boundaries")
        },
        "participants": participants,
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
        elif (
            event.record.get("event") == "gen_transfer_settled"
            and event.record.get("resources_drained") is True
        ):
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
    grouped: dict[tuple[str, str, str, str], tuple[object, list[_ParsedEvent]]] = {}
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
        key = _request_group_key(event)
        if key not in grouped:
            grouped[key] = (request_id, [])
        grouped[key][1].append(event)

    capabilities = _capability_index(parse_result.events)
    requests = [
        _summarize_request(request_id, events, capabilities)
        for _, (request_id, events) in sorted(grouped.items())
    ]
    return {
        "identity_semantics": {
            "requests": "Join across processes only by a shared run_uuid and request_id.",
            "process_local": (
                "Without a valid run_uuid, correlate only within one process_uuid, host and PID."
            ),
            "unverified": (
                "Without a valid process_uuid, retain individual records "
                "without joining or deriving timing."
            ),
            "configuration": (
                "Set TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID to the same fresh UUID "
                "on CTX and GEN for each launch."
            ),
        },
        "identity_issues": dict(
            sorted(
                Counter(
                    issue
                    for event in parse_result.events
                    for issue in _identity_issues(event.record)
                ).items()
            )
        ),
        "capability_semantics": {
            "scope": (
                "Startup declarations are matched by run/process UUIDs, host, pid and rank, "
                "then applied per participant."
            ),
            "missing_boundaries": "Expected events supported by this participant but not observed.",
            "unsupported_boundaries": "Events not instrumented by this participant's implementation.",
            "unassessed_boundaries": (
                "Expected events whose capability metadata is absent or ambiguous; "
                "not a healthy result."
            ),
            "durations": (
                "Observed matching pairs remain measurable without capability metadata; "
                "unsupported or unknown gaps are labeled separately."
            ),
        },
        "clock_semantics": {
            "durations": (
                "Derived only from monotonic_ns within matching run/process UUIDs, host and PID."
            ),
            "timeline": (
                "wall_ns is preserved for manual cross-process correlation; cross-host ordering "
                "is clock-sync-sensitive and no wall-clock deltas are derived."
            ),
            "gen_transfer_settled_to_next_scheduler_decision": (
                "A same-domain measure from successful transceiver session retirement "
                "to the next admission opportunity."
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

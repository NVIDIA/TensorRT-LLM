#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report observed native KV-transfer timing from opt-in diagnostic logs.

Example: python scripts/disagg_transfer_diagnostics.py ctx.log gen.log -o timing.json

Set TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID to one fresh UUID shared by CTX
and GEN for a launch. Cross-process request correlation requires that UUID and
an explicitly run-scoped request ID. Durations require a matching native_v1
startup profile and are always process-local.

This is a timing aid, not a health/completeness checker. It does not infer
missing participants, ADP writer cohorts, physical buffer use, or admission
policy. Unmatched, repeated, and unsupported evidence remains visible. A
submission-to-completion duration includes backend queueing, not just wire time.
The native wait timeout is not an active-service or resource-safety deadline.
"""

from __future__ import annotations

import argparse
import fileinput
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from uuid import UUID

DIAGNOSTICS_LOG_PREFIX = "[DISAGG_TRANSFER_DIAG] "
_DESCRIPTOR = ("slice_id", "receiver_slice_id", "peer_rank")
# Only unique observed pairs are measured; retries/chunks with repeated keys
# stay ambiguous rather than being matched by arrival order.
_PAIRS = (
    ("gen_admission_wait", "gen_ingress", "gen_kv_admission_result", ()),
    ("gen_receive_lifetime", "gen_receive_start", "gen_transfer_settled", ()),
    ("ctx_queue_wait", "ctx_transfer_queued", "ctx_worker_dequeued", _DESCRIPTOR),
    ("ctx_preparation", "ctx_worker_dequeued", "ctx_backend_submit_start", _DESCRIPTOR),
    ("ctx_submission", "ctx_backend_submit_start", "ctx_backend_submitted", _DESCRIPTOR),
    ("ctx_backend_residence", "ctx_backend_submitted", "ctx_backend_complete", _DESCRIPTOR),
    # This starts at the final-chunk handoff, not the beginning of prefill.
    ("ctx_final_handoff_to_release", "ctx_send_ready", "ctx_source_kv_released", ()),
    (
        "executor_timeout_window",
        "transfer_timeout_started",
        "transfer_timeout_observed",
        ("timeout_owner",),
    ),
    (
        "native_wait_timeout_window",
        "ctx_native_deadline_started",
        "ctx_native_deadline_observed",
        ("timeout_owner",),
    ),
)
_EVENTS = {event for _, start, end, _ in _PAIRS for event in (start, end)} | {
    "diagnostic_profile",
    "diagnostics_events_dropped",
    "gen_kv_rollback",
    "gen_request_data_sent",
    "ctx_request_data_received",
    "gen_writer_result_received",
    "gen_destination_complete",
    "ctx_transfer_settled",
    "gen_decode_ready",
    "transfer_cancel_requested",
    "ctx_source_unpinned",
}
_INTEGER_FIELDS = (
    "pid",
    "rank",
    "slice_id",
    "receiver_slice_id",
    "peer_rank",
    "monotonic_ns",
    "timer_start_monotonic_ns",
    "deadline_monotonic_ns",
    "dropped_events",
)
Scalar = str | int | float | bool | None
Record = dict[str, Scalar]
Domain = tuple[str | None, str, str, int, int]


def _uuid(value: Scalar) -> str | None:
    """Normalize a UUID, or return None for absent/invalid identity.

    Args:
        value: Logged identity field.

    Returns:
        Canonical UUID spelling when valid.
    """
    if not isinstance(value, str):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _valid_record(record: object) -> bool:
    """Validate the flat event envelope and fields used for correlation.

    Args:
        record: Decoded JSON value.

    Returns:
        Whether the record can be retained safely.
    """
    if not isinstance(record, dict):
        return False
    if any(
        value is not None
        and (
            not isinstance(value, (str, int, float, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
        )
        for value in record.values()
    ):
        return False
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        return False
    if not isinstance(record.get("event"), str) or not record["event"]:
        return False
    owner = record.get("timeout_owner")
    if owner is not None and (not isinstance(owner, str) or not owner):
        return False
    for field in (*_INTEGER_FIELDS, "request_id"):
        value = record.get(field)
        limit = (1 << (64 if field == "request_id" else 63)) - 1
        if value is not None and (type(value) is not int or not 0 <= value <= limit):
            return False
    return True


def _domain(record: Record) -> Domain | None:
    """Identify one process/rank clock without trusting reused PIDs.

    Args:
        record: Validated event.

    Returns:
        Clock identity, or None when identity is insufficient.
    """
    process = _uuid(record.get("process_uuid"))
    host, pid, rank = record.get("host"), record.get("pid"), record.get("rank")
    if (
        process is None
        or not isinstance(host, str)
        or not host
        or type(pid) is not int
        or type(rank) is not int
    ):
        return None
    return _uuid(record.get("run_uuid")), process, host, pid, rank


def _request_key(record: Record, line: int) -> tuple[object, ...]:
    """Join canonical requests across workers; keep fallback IDs local.

    Args:
        record: Validated request event.
        line: Unique input position for records without safe identity.

    Returns:
        Correlation key, isolated when identity is unknown.
    """
    domain = _domain(record)
    scope = record.get("request_id_scope")
    if (
        domain is None
        or scope not in ("run", "process")
        or record.get("side") not in ("ctx", "gen")
    ):
        return "unverified", line
    if scope == "run" and domain[0] is not None:
        return "run", domain[0], record["request_id"]
    return "process", domain, scope, record["request_id"]


def _timestamp(record: Record) -> int | None:
    """Use a recorded timer anchor instead of the later log-write time.

    Args:
        record: Validated event.

    Returns:
        Process-local nanoseconds when present.
    """
    field = (
        "timer_start_monotonic_ns"
        if record["event"] in ("transfer_timeout_started", "ctx_native_deadline_started")
        else "monotonic_ns"
    )
    value = record.get(field)
    return value if type(value) is int else None


def _timings(events: list[Record]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure unique, ordered boundaries within an already verified participant.

    Args:
        events: Events from one request, process, rank, and side.

    Returns:
        Measured intervals and explicit reasons for unmeasured observed pairs.
    """
    measured, unmeasured = [], []
    for name, start, end, fields in _PAIRS:
        grouped: dict[tuple[Scalar, ...], dict[str, list[Record]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in events:
            event = record["event"]
            if event not in (start, end):
                continue
            if event == "gen_kv_admission_result" and record.get("outcome") != "admitted":
                continue
            grouped[tuple(record.get(field) for field in fields)][event].append(record)
        for key, boundaries in grouped.items():
            item: dict[str, object] = {"phase": name, "correlation": dict(zip(fields, key))}
            starts, ends = boundaries.get(start, []), boundaries.get(end, [])
            if None in key:
                reason = "missing_correlation"
            elif len(starts) > 1 or len(ends) > 1:
                reason = "ambiguous_repeated_boundary"
            elif not starts or not ends:
                reason = "unmatched_observed_boundary"
            else:
                begin, finish = _timestamp(starts[0]), _timestamp(ends[0])
                if begin is None or finish is None:
                    reason = "missing_timestamp"
                elif finish < begin:
                    reason = "reversed_boundaries"
                else:
                    measured.append(
                        {
                            **item,
                            "duration_ms": (finish - begin) / 1_000_000,
                            "start_line": starts[0]["line"],
                            "end_line": ends[0]["line"],
                        }
                    )
                    continue
            unmeasured.append(
                {**item, "reason": reason, "start_count": len(starts), "end_count": len(ends)}
            )
    return measured, unmeasured


def analyze_lines(lines: Iterable[str]) -> dict[str, object]:
    """Build a native timing report without inferring end-to-end health.

    Args:
        lines: Combined runtime logs, possibly containing unrelated output.

    Returns:
        JSON-compatible report with raw events, local timings, and evidence gaps.
    """
    counts: Counter[str] = Counter()
    events: list[Record] = []
    evidence: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        counts["total_lines"] += 1
        marker = line.find(DIAGNOSTICS_LOG_PREFIX)
        if marker < 0:
            counts["ignored_lines"] += 1
            continue
        try:
            record = json.loads(line[marker + len(DIAGNOSTICS_LOG_PREFIX) :])
        except (ValueError, RecursionError):
            record = None
        if not _valid_record(record):
            counts["malformed_lines"] += 1
            evidence.append({"line": line_number, "reason": "malformed_record"})
            continue
        record["line"] = line_number
        events.append(record)
        if record["event"] not in _EVENTS:
            evidence.append(
                {"line": line_number, "reason": "unsupported_event", "event": record["event"]}
            )
        if record["event"] == "diagnostics_events_dropped":
            counts["reported_dropped_events"] += record.get("dropped_events") or 0
            evidence.append(
                {
                    "line": line_number,
                    "reason": "dropped_events",
                    "count": record.get("dropped_events"),
                }
            )

    profiles: dict[Domain, set[Scalar]] = defaultdict(set)
    for record in events:
        domain = _domain(record)
        if record["event"] == "diagnostic_profile" and domain is not None:
            profiles[domain].add(record.get("profile"))

    grouped: dict[tuple[object, ...], list[Record]] = defaultdict(list)
    ungrouped = []
    for record in events:
        if record.get("request_id") is None:
            ungrouped.append(record)
        else:
            grouped[_request_key(record, record["line"])].append(record)

    requests = []
    for key, records in grouped.items():
        participants: dict[tuple[Domain | None, Scalar], list[Record]] = defaultdict(list)
        for record in records:
            participants[(_domain(record), record.get("side"))].append(record)
        summaries = []
        for (domain, side), local_events in participants.items():
            profile = profiles.get(domain, set())
            qualified = key[0] != "unverified" and profile == {"native_v1"}
            timings, unmeasured = _timings(local_events) if qualified else ([], [])
            if not qualified:
                unmeasured.append(
                    {
                        "reason": "unverified_identity"
                        if key[0] == "unverified"
                        else "unsupported_or_missing_native_profile"
                    }
                )
            first = local_events[0]
            summaries.append(
                {
                    "identity": {
                        field: first.get(field)
                        for field in ("run_uuid", "process_uuid", "host", "pid", "rank")
                    },
                    "side": side,
                    "profile": "native_v1" if qualified else "unassessed",
                    "timings": timings,
                    "unmeasured": unmeasured,
                    "events": local_events,
                }
            )
        requests.append(
            {
                "request_id": records[0]["request_id"],
                "run_uuid": _uuid(records[0].get("run_uuid")),
                "request_id_scope": records[0].get("request_id_scope"),
                "correlation_scope": key[0],
                "participants": summaries,
            }
        )
    return {
        "scope": (
            "Observed native_v1 pairs only; not a health or completeness verdict. "
            "Unsupported events remain raw evidence."
        ),
        "clock_semantics": (
            "Durations use one process/rank monotonic clock. Wall timestamps are display-only; "
            "no remote-clock deltas are inferred."
        ),
        "readiness_semantics": (
            "REQUEST_DATA receipts use receiver slice IDs; "
            "no source-slice join or all-receiver barrier is inferred."
        ),
        "summary": {
            name: counts[name]
            for name in (
                "total_lines",
                "ignored_lines",
                "malformed_lines",
                "reported_dropped_events",
            )
        },
        "evidence": evidence,
        "ungrouped_events": ungrouped,
        "requests": requests,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Read log files/stdin and emit a JSON timing report.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero after successfully writing the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="Log files; defaults to stdin.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file; defaults to stdout.")
    args = parser.parse_args(argv)
    with fileinput.input(files=args.logs or ("-",), encoding="utf-8", errors="replace") as lines:
        result = analyze_lines(lines)
    if args.output is None:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        with args.output.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

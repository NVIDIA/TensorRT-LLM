# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualified native timing, identity, and incomplete-evidence regressions."""

import json
from pathlib import Path

import pytest

__extra_import_path__ = ["~/scripts"]
from disagg_transfer_diagnostics import DIAGNOSTICS_LOG_PREFIX, analyze_lines, main  # noqa: E402

pytestmark = pytest.mark.cpu_only
_RUN = "205d1c66-cc2a-4bb9-a9c4-1559c6f95fd8"
_PROCESS = "6d905e36-6967-4db1-b44b-e8ab47cb3053"
_OTHER = "4f88878a-b7bf-4c30-ad73-b14615738452"


def _event(name: str, timestamp: int = 1_000_000, **fields: object) -> str:
    """Build one flat runtime log record.

    Args:
        name: Event boundary name.
        timestamp: Local monotonic nanoseconds.
        fields: Event/identity overrides, including malformed test inputs.

    Returns:
        A diagnostic JSON log line.
    """
    record = {
        "schema_version": 1,
        "event": name,
        "request_id": 7,
        "request_id_scope": "run",
        "side": "ctx",
        "run_uuid": _RUN,
        "process_uuid": _PROCESS,
        "host": "node-a",
        "pid": 17,
        "rank": 0,
        "monotonic_ns": timestamp,
        "wall_ns": timestamp + 100_000_000_000,
        **fields,
    }
    return f"worker-prefix {DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"


def _profile(**fields: object) -> str:
    """Create a participant's native-path declaration.

    Args:
        fields: Profile or identity overrides.

    Returns:
        Startup log line.
    """
    return _event(
        "diagnostic_profile",
        0,
        **{"request_id": None, "side": "runtime", "profile": "native_v1", **fields},
    )


def _timings(report: dict[str, object]) -> list[dict[str, object]]:
    """Flatten timings without discarding participant identity in the report.

    Args:
        report: Analyzer result.

    Returns:
        Measured timing entries.
    """
    return [
        timing
        for request in report["requests"]
        for participant in request["participants"]
        for timing in participant["timings"]
    ]


def test_native_queue_stages_keep_descriptor_identity() -> None:
    """Distinct source/receiver slices and peers do not satisfy one another."""
    descriptor = {"slice_id": 1, "receiver_slice_id": 2, "peer_rank": 3}
    result = analyze_lines(
        [_profile()]
        + [
            _event(name, timestamp * 1_000_000, **descriptor)
            for name, timestamp in (
                ("ctx_transfer_queued", 1),
                ("ctx_worker_dequeued", 3),
                ("ctx_backend_submit_start", 6),
                ("ctx_backend_submitted", 7),
                ("ctx_backend_complete", 11),
            )
        ]
        + [_event("ctx_transfer_queued", peer_rank=4, slice_id=1, receiver_slice_id=2)]
    )
    measured = _timings(result)
    assert {item["phase"]: item["duration_ms"] for item in measured} == {
        "ctx_queue_wait": 2.0,
        "ctx_preparation": 3.0,
        "ctx_submission": 1.0,
        "ctx_backend_residence": 4.0,
    }
    assert all(item["correlation"] == descriptor for item in measured)
    unmatched = result["requests"][0]["participants"][0]["unmeasured"]
    assert any(item["correlation"]["peer_rank"] == 4 for item in unmatched)


@pytest.mark.parametrize(
    "changed",
    [
        {"run_uuid": _OTHER},
        {"process_uuid": _OTHER},
        {"host": "node-b"},
        {"pid": 18},
        {"rank": 1},
        {"side": "gen"},
    ],
)
def test_local_timing_never_crosses_identity_boundaries(changed: dict[str, object]) -> None:
    """Reused IDs cannot bridge processes, ranks, sides, restarts, or runs.

    Args:
        changed: The endpoint identity component that differs.
    """
    result = analyze_lines(
        [
            _profile(),
            _profile(**changed),
            _event("ctx_send_ready"),
            _event("ctx_source_kv_released", 3_000_000, **changed),
        ]
    )
    assert not _timings(result)


def test_run_identity_correlates_ctx_and_gen_without_cross_process_deltas() -> None:
    """A shared request contains separate local CTX and GEN intervals."""
    gen = {"process_uuid": _OTHER, "pid": 18, "side": "gen"}
    result = analyze_lines(
        [
            _profile(),
            _profile(**gen),
            _event("ctx_send_ready", 900_000_000),
            _event("ctx_source_kv_released", 903_000_000),
            _event("gen_receive_start", 1_000_000, **gen),
            _event("gen_transfer_settled", 5_000_000, **gen),
        ]
    )
    assert len(result["requests"]) == 1
    assert len(result["requests"][0]["participants"]) == 2
    assert {sample["duration_ms"] for sample in _timings(result)} == {3.0, 4.0}


@pytest.mark.parametrize("scope", ["process", None, "unknown", "global", False])
def test_fallback_or_unknown_request_ids_do_not_join_workers(scope: object) -> None:
    """Only explicitly run-scoped IDs can join across processes.

    Args:
        scope: Process-local or unsupported request-ID namespace.
    """
    result = analyze_lines(
        [
            _profile(),
            _profile(process_uuid=_OTHER),
            _event("ctx_send_ready", request_id_scope=scope),
            _event(
                "ctx_source_kv_released", 3_000_000, process_uuid=_OTHER, request_id_scope=scope
            ),
        ]
    )
    assert len(result["requests"]) == 2
    assert not _timings(result)


@pytest.mark.parametrize("run_uuid", [None, "invalid", _RUN])
def test_process_local_ids_remain_measurable_without_shared_run(run_uuid: object) -> None:
    """A verified process supports local intervals without a shared run UUID.

    Args:
        run_uuid: Shared launch identity or missing/invalid value.
    """
    fields = {"run_uuid": run_uuid, "request_id_scope": "process"}
    result = analyze_lines(
        [
            _profile(**fields),
            _event("ctx_send_ready", **fields),
            _event("ctx_source_kv_released", 3_000_000, **fields),
        ]
    )
    assert result["requests"][0]["correlation_scope"] == "process"
    assert _timings(result)[0]["duration_ms"] == 2.0
    assert _timings(result)[0]["phase"] == "ctx_final_handoff_to_release"


@pytest.mark.parametrize(
    "fields",
    [
        {"process_uuid": None},
        {"process_uuid": "invalid"},
        {"host": ""},
        {"pid": None},
        {"rank": None},
        {"side": None},
    ],
)
def test_unverified_identity_retains_records_without_timing(fields: dict[str, object]) -> None:
    """Untrusted identity is explicit rather than a fabricated healthy trace.

    Args:
        fields: Missing or invalid process-clock identity.
    """
    result = analyze_lines(
        [
            _profile(**fields),
            _event("ctx_send_ready", **fields),
            _event("ctx_source_kv_released", 3_000_000, **fields),
        ]
    )
    assert len(result["requests"]) == 2
    assert not _timings(result)
    assert all(
        request["participants"][0]["unmeasured"] == [{"reason": "unverified_identity"}]
        for request in result["requests"]
    )


def test_canonical_and_process_namespaces_are_separate_and_uuids_normalized() -> None:
    """UUID spelling is normalized, but equal IDs in different scopes stay apart."""
    result = analyze_lines(
        [
            _profile(run_uuid=_RUN.upper(), process_uuid=_PROCESS.upper()),
            _event("ctx_send_ready"),
            _event("ctx_source_kv_released", 3_000_000, run_uuid=_RUN.upper()),
            _event("ctx_source_kv_released", 4_000_000, request_id_scope="process"),
        ]
    )
    assert len(result["requests"]) == 2
    assert [item["duration_ms"] for item in _timings(result)] == [2.0]


@pytest.mark.parametrize(
    "profiles",
    [[], [_profile(profile="cpp")], [_profile(), _profile(profile="unknown")], [_profile(rank=1)]],
)
def test_missing_unsupported_conflicting_or_other_rank_profile_is_unassessed(
    profiles: list[str],
) -> None:
    """The small analyzer does not implement a heterogeneous capability matrix.

    Args:
        profiles: Missing, conflicting, unsupported, or unrelated declarations.
    """
    result = analyze_lines(
        profiles + [_event("ctx_send_ready"), _event("ctx_source_kv_released", 3_000_000)]
    )
    participant = result["requests"][0]["participants"][0]
    assert participant["profile"] == "unassessed"
    assert participant["timings"] == []
    assert participant["unmeasured"] == [{"reason": "unsupported_or_missing_native_profile"}]
    assert len(participant["events"]) == 2


@pytest.mark.parametrize(
    "end,extra,reason",
    [
        (None, [], "unmatched_observed_boundary"),
        (0, [], "reversed_boundaries"),
        (None, [_event("ctx_source_kv_released", monotonic_ns=None)], "missing_timestamp"),
        (3_000_000, [_event("ctx_send_ready", 2_000_000)], "ambiguous_repeated_boundary"),
    ],
)
def test_incomplete_and_repeated_boundaries_are_not_guessed(
    end: int | None, extra: list[str], reason: str
) -> None:
    """Do not manufacture timing from gaps, retries, or unordered evidence.

    Args:
        end: Optional release timestamp.
        extra: Additional ambiguous/incomplete evidence.
        reason: Expected unmeasured classification.
    """
    lines = [_profile(), _event("ctx_send_ready"), *extra]
    if end is not None:
        lines.append(_event("ctx_source_kv_released", end))
    result = analyze_lines(lines)
    assert not _timings(result)
    assert result["requests"][0]["participants"][0]["unmeasured"][0]["reason"] == reason


def test_missing_descriptor_keys_and_receiver_slice_ids_are_not_guessed() -> None:
    """REQUEST_DATA receiver slices never masquerade as source work slices."""
    result = analyze_lines(
        [
            _profile(),
            _event("ctx_send_ready"),
            _event("ctx_request_data_received", slice_id=2, peer_rank=3),
            _event("ctx_transfer_queued", slice_id=1, peer_rank=3),
            _event("ctx_worker_dequeued", 3_000_000, slice_id=1, peer_rank=3),
        ]
    )
    participant = result["requests"][0]["participants"][0]
    assert not participant["timings"]
    assert any(item["reason"] == "missing_correlation" for item in participant["unmeasured"])
    assert any(event["event"] == "ctx_request_data_received" for event in participant["events"])


def test_timeout_uses_real_anchor_and_separates_timeout_owners() -> None:
    """Log latency and another timeout owner cannot alter a measured deadline."""
    result = analyze_lines(
        [
            _profile(),
            _event(
                "transfer_timeout_started",
                9_000_000,
                timer_start_monotonic_ns=1_000_000,
                timeout_owner="executor",
            ),
            _event("transfer_timeout_observed", 61_000_000, timeout_owner="executor"),
            _event("transfer_timeout_observed", 75_000_000, timeout_owner="other"),
            _event("transfer_cancel_requested", 62_000_000),
        ]
    )
    assert [item["duration_ms"] for item in _timings(result)] == [60.0]
    assert _timings(result)[0]["correlation"] == {"timeout_owner": "executor"}


def test_native_wait_uses_authoritative_anchor_and_retains_absolute_deadline() -> None:
    """Native wait timing is independent of delayed logging and active service."""
    result = analyze_lines(
        [
            _profile(),
            _event(
                "ctx_native_deadline_started",
                9_000_000,
                timer_start_monotonic_ns=1_000_000,
                deadline_monotonic_ns=61_000_000,
                timeout_owner="native_tx_session",
            ),
            _event(
                "ctx_native_deadline_observed",
                62_000_000,
                deadline_monotonic_ns=61_000_000,
                timeout_owner="native_tx_session",
            ),
        ]
    )
    assert _timings(result)[0]["phase"] == "native_wait_timeout_window"
    assert _timings(result)[0]["duration_ms"] == 61.0
    assert (
        result["requests"][0]["participants"][0]["events"][0]["deadline_monotonic_ns"] == 61_000_000
    )


def test_gen_admission_retries_do_not_count_failed_allocation_as_admission() -> None:
    """The report measures a unique successful GEN allocation only."""
    result = analyze_lines(
        [
            _profile(),
            _event("gen_ingress", side="gen"),
            _event("gen_kv_admission_result", 2_000_000, side="gen", outcome="deferred"),
            _event("gen_kv_admission_result", 4_000_000, side="gen", outcome="admitted"),
        ]
    )
    assert [item["duration_ms"] for item in _timings(result)] == [3.0]


@pytest.mark.parametrize(
    "fields",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"event": None},
        {"request_id": "7"},
        {"request_id": 1 << 64},
        {"rank": True},
        {"peer_rank": []},
        {"pid": -1},
        {"monotonic_ns": 1 << 63},
        {"timeout_owner": False},
        {"timeout_owner": ""},
        {"extra": float("nan")},
        {"extra": float("inf")},
        {"extra": {}},
    ],
)
def test_malformed_envelopes_and_correlation_fields_are_quarantined(
    fields: dict[str, object],
) -> None:
    """Malformed records cannot corrupt keys or arithmetic.

    Args:
        fields: Invalid envelope/correlation values.
    """
    result = analyze_lines([_event("ctx_send_ready", **fields)])
    assert result["summary"]["malformed_lines"] == 1
    assert result["requests"] == []
    assert result["evidence"] == [{"line": 1, "reason": "malformed_record"}]


def test_noise_drops_unsupported_events_and_deep_json_remain_visible() -> None:
    """Incomplete logs never receive an implicit all-clear result."""
    result = analyze_lines(
        [
            "normal runtime message\n",
            DIAGNOSTICS_LOG_PREFIX + "not json",
            DIAGNOSTICS_LOG_PREFIX + "[" * 2_000 + "0" + "]" * 2_000,
            _event("diagnostics_events_dropped", request_id=None, dropped_events=3),
            _event("gen_transfer_window_result", policy="legacy"),
        ]
    )
    assert result["summary"] == {
        "total_lines": 5,
        "ignored_lines": 1,
        "malformed_lines": 2,
        "reported_dropped_events": 3,
    }
    assert {item["reason"] for item in result["evidence"]} == {
        "malformed_record",
        "unsupported_event",
        "dropped_events",
    }
    assert result["requests"][0]["participants"][0]["events"][0]["policy"] == "legacy"
    assert "not a health" in result["scope"]


@pytest.mark.parametrize("file_output", [False, True])
def test_cli_reads_logs_and_writes_json(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], file_output: bool
) -> None:
    """Both CLI output paths execute the real analyzer.

    Args:
        tmp_path: Isolated input/output directory.
        capfd: Captured process output.
        file_output: Whether to write a file instead of stdout.
    """
    log = tmp_path / "ctx.log"
    log.write_text(
        _profile() + _event("ctx_send_ready") + _event("ctx_source_kv_released", 3_000_000)
    )
    output = tmp_path / "timing.json"
    assert main([str(log), *(["-o", str(output)] if file_output else [])]) == 0
    result = json.loads(output.read_text() if file_output else capfd.readouterr().out)
    assert _timings(result)[0]["duration_ms"] == 2.0

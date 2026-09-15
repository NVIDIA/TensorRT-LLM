# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and process identity regressions for the transfer diagnostic analyzer."""

import json

import pytest

__extra_import_path__ = ["~/scripts"]
from disagg_transfer_diagnostics import DIAGNOSTICS_LOG_PREFIX, analyze_lines  # noqa: E402

pytestmark = pytest.mark.cpu_only

_RUN_A = "205d1c66-cc2a-4bb9-a9c4-1559c6f95fd8"
_RUN_B = "e199df92-446e-4a55-847a-18078be332d8"
_PROCESS_A = "6d905e36-6967-4db1-b44b-e8ab47cb3053"
_PROCESS_B = "4f88878a-b7bf-4c30-ad73-b14615738452"


def _event(
    name: str,
    timestamp: int = 1_000_000,
    *,
    request_id: int | str | None = 7,
    side: str = "ctx",
    **details: object,
) -> str:
    record = {
        "schema_version": 1,
        "event": name,
        "request_id": request_id,
        "side": side,
        "host": "node-a",
        "pid": 17,
        "rank": 0,
        "run_uuid": _RUN_A,
        "process_uuid": _PROCESS_A,
        "monotonic_ns": timestamp,
        "wall_ns": 100_000_000_000 + timestamp,
        **details,
    }
    return f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"


def _ownership_pair(**identity: object) -> list[str]:
    return [
        _event("ctx_send_ready", **identity),
        _event("ctx_source_kv_released", 3_000_000, **identity),
    ]


def _ownership_durations(request: dict[str, object]) -> list[float]:
    return [
        duration["duration_ms"]
        for duration in request["durations"]
        if duration["phase"] == "ctx_source_kv_request_ownership"
    ]


def test_different_runs_cannot_pair_reused_request_ids_and_pids() -> None:
    result = analyze_lines(
        [
            _event("ctx_send_ready"),
            _event("ctx_source_kv_released", 61_000_000, run_uuid=_RUN_B),
        ]
    )

    assert len(result["requests"]) == 2
    assert {request["run_uuid"] for request in result["requests"]} == {_RUN_A, _RUN_B}
    assert all(request["correlation_scope"] == "run" for request in result["requests"])
    assert all(request["durations"] == [] for request in result["requests"])
    assert "ctx_source_kv_request_ownership" not in result["phase_durations"]


def test_shared_run_correlates_ctx_and_gen_without_cross_process_timing() -> None:
    result = analyze_lines(
        _ownership_pair()
        + [
            _event(
                "gen_transfer_settled",
                side="gen",
                process_uuid=_PROCESS_B,
                pid=18,
                outcome="completed",
            ),
            _event("gen_decode_ready", 4_000_000, side="gen", process_uuid=_PROCESS_B, pid=18),
        ]
    )

    assert len(result["requests"]) == 1
    request = result["requests"][0]
    assert request["run_uuid"] == _RUN_A
    assert request["correlation_scope"] == "run"
    assert {participant["side"] for participant in request["participants"]} == {"ctx", "gen"}
    assert {participant["process_uuid"] for participant in request["participants"]} == {
        _PROCESS_A,
        _PROCESS_B,
    }
    assert all(participant["run_uuid"] == _RUN_A for participant in request["participants"])
    assert _ownership_durations(request) == [2.0]
    assert any(
        duration["phase"] == "gen_transfer_to_service" and duration["duration_ms"] == 3.0
        for duration in request["durations"]
    )
    assert {duration["clock_domain"]["process_uuid"] for duration in request["durations"]} == {
        _PROCESS_A,
        _PROCESS_B,
    }


def test_restart_with_reused_pid_cannot_pair_local_boundaries() -> None:
    result = analyze_lines(
        [
            _event("ctx_send_ready"),
            _event("ctx_source_kv_released", 61_000_000, process_uuid=_PROCESS_B),
        ]
    )

    assert len(result["requests"]) == 1
    request = result["requests"][0]
    assert request["durations"] == []
    assert len(request["participants"]) == 2
    assert {domain["process_uuid"] for domain in request["clock_domains"]} == {
        _PROCESS_A,
        _PROCESS_B,
    }
    assert all(domain["run_uuid"] == _RUN_A for domain in request["clock_domains"])


@pytest.mark.parametrize("new_identity", ({"run_uuid": _RUN_B}, {"process_uuid": _PROCESS_B}))
def test_capability_metadata_does_not_leak_across_runs_or_restarts(
    new_identity: dict[str, str],
) -> None:
    result = analyze_lines(
        [
            _event(
                "diagnostic_capabilities",
                0,
                request_id=None,
                side="runtime",
                capability_schema_version=1,
                transceiver_runtime="CPP",
                executor_events=True,
                scheduler_kv_admission_events=True,
                python_transfer_events=False,
            ),
            _event("ctx_send_ready", **new_identity),
        ]
    )

    request = result["requests"][0]
    assert request["participants"][0]["capabilities"]["status"] == "unknown"
    assert request["unsupported_boundaries"] == []
    assert "ctx_transfer_settled" in request["unassessed_boundaries"]


@pytest.mark.parametrize("new_identity", ({"run_uuid": _RUN_B}, {"process_uuid": _PROCESS_B}))
def test_scheduler_intervals_do_not_bridge_runs_or_restarts(
    new_identity: dict[str, str],
) -> None:
    result = analyze_lines(
        [
            _event("gen_kv_pool_snapshot", request_id=None, side="gen"),
            _event("gen_transfer_settled", 2_000_000, side="gen", resources_drained=True),
            _event("gen_kv_pool_snapshot", 3_000_000, request_id=None, side="gen", **new_identity),
        ]
    )

    cadence = result["scheduler_decision_cadence"]
    assert len(cadence) == 2
    assert all(item["snapshot_count"] == 1 and item["interval_count"] == 0 for item in cadence)
    settled = result["gen_transfer_settled_to_next_scheduler_decision"]
    assert len(settled) == 1
    assert settled[0]["settled_count"] == 1
    assert settled[0]["matched_count"] == 0
    assert settled[0]["unmatched_count"] == 1


@pytest.mark.parametrize("run_uuid", (None, "", "not-a-run-uuid"))
def test_missing_or_invalid_shared_run_allows_only_process_local_analysis(
    run_uuid: str | None,
) -> None:
    result = analyze_lines(
        _ownership_pair(run_uuid=run_uuid)
        + _ownership_pair(run_uuid=run_uuid, process_uuid=_PROCESS_B)
    )

    assert len(result["requests"]) == 2
    for request in result["requests"]:
        assert request["run_uuid"] is None
        assert request["correlation_scope"] == "process"
        assert len(request["participants"]) == 1
        assert _ownership_durations(request) == [2.0]


@pytest.mark.parametrize("process_uuid", (None, "", "not-a-process-uuid"))
def test_invalid_process_identity_keeps_events_readable_without_joining(
    process_uuid: str | None,
) -> None:
    result = analyze_lines(_ownership_pair(process_uuid=process_uuid))

    assert result["summary"]["parsed_events"] == 2
    assert len(result["requests"]) == 2
    for request in result["requests"]:
        assert request["correlation_scope"] == "unverified"
        assert len(request["timeline"]) == 1
        assert request["durations"] == []


def test_legacy_records_without_identity_do_not_derive_timing() -> None:
    lines = []
    for line in _ownership_pair():
        record = json.loads(line.removeprefix(DIAGNOSTICS_LOG_PREFIX))
        record.pop("run_uuid")
        record.pop("process_uuid")
        lines.append(f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n")

    result = analyze_lines(lines)

    assert len(result["requests"]) == 2
    assert all(request["correlation_scope"] == "unverified" for request in result["requests"])
    assert all(request["durations"] == [] for request in result["requests"])


def test_unverified_process_identity_does_not_derive_scheduler_aggregates() -> None:
    result = analyze_lines(
        [
            _event("gen_kv_pool_snapshot", request_id=None, side="gen", process_uuid=None),
            _event(
                "gen_transfer_settled",
                2_000_000,
                side="gen",
                process_uuid=None,
                resources_drained=True,
            ),
            _event(
                "gen_kv_pool_snapshot",
                3_000_000,
                request_id=None,
                side="gen",
                process_uuid=None,
            ),
        ]
    )

    assert result["scheduler_decision_cadence"] == []
    assert result["gen_transfer_settled_to_next_scheduler_decision"] == []
    assert len(result["aggregate_timeline"]) == 2


def test_uuid_spelling_is_normalized_for_matching() -> None:
    result = analyze_lines(
        [
            _event("ctx_send_ready", run_uuid=_RUN_A.upper(), process_uuid=_PROCESS_A.upper()),
            _event("ctx_source_kv_released", 3_000_000),
        ]
    )

    assert len(result["requests"]) == 1
    request = result["requests"][0]
    assert request["run_uuid"] == _RUN_A
    assert request["participants"][0]["process_uuid"] == _PROCESS_A
    assert _ownership_durations(request) == [2.0]


def test_invalid_string_request_id_is_not_coerced_into_valid_run_group() -> None:
    result = analyze_lines(_ownership_pair(request_id=7) + _ownership_pair(request_id="7"))

    assert result["summary"]["malformed_diagnostic_lines"] == 2
    assert len(result["requests"]) == 1
    assert result["requests"][0]["request_id"] == 7
    assert result["requests"][0]["event_count"] == 2
    assert _ownership_durations(result["requests"][0]) == [2.0]

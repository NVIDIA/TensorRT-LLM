# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-capability regressions for the transfer diagnostic analyzer."""

import json
from uuid import NAMESPACE_DNS, uuid5

import pytest

__extra_import_path__ = ["~/scripts"]
from disagg_transfer_diagnostics import DIAGNOSTICS_LOG_PREFIX, analyze_lines  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _event(
    name: str,
    timestamp: int = 1_000_000,
    *,
    request_id: int | None = 7,
    side: str = "ctx",
    host: str = "node-a",
    pid: int = 17,
    rank: int = 0,
    **details: object,
) -> str:
    record = {
        "schema_version": 1,
        "event": name,
        "request_id": request_id,
        "request_id_scope": "run",
        "side": side,
        "host": host,
        "pid": pid,
        "rank": rank,
        "run_uuid": "11111111-1111-4111-8111-111111111111",
        "process_uuid": str(uuid5(NAMESPACE_DNS, f"{host}:{pid}")),
        "monotonic_ns": timestamp,
        "wall_ns": 100_000_000_000 + timestamp,
        **details,
    }
    return f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"


def _capability_record(**details: object) -> str:
    return _event(
        "diagnostic_capabilities",
        0,
        request_id=None,
        side="runtime",
        **{"capability_schema_version": 1, **details},
    )


def _profile(runtime: str = "CPP", *, scheduler_v2: bool = True, **identity: object) -> list[str]:
    return [
        _capability_record(
            transceiver_runtime=runtime,
            python_transfer_events=runtime == "PYTHON",
            **identity,
        ),
        _capability_record(
            executor_events=True,
            scheduler_kv_admission_events=scheduler_v2,
            **identity,
        ),
    ]


def _request(lines: list[str]) -> dict[str, object]:
    result = analyze_lines(lines)
    assert len(result["requests"]) == 1
    return result["requests"][0]


def _phase_reasons(request: dict[str, object], phase: str) -> set[str]:
    return {gap["reason"] for gap in request["unmeasured_phases"] if gap["phase"] == phase}


def test_healthy_cpp_trace_reports_unsupported_python_boundaries_not_missing() -> None:
    request = _request(
        _profile()
        + [
            _event("ctx_send_ready"),
            _event("ctx_source_kv_released", 3_000_000),
            _event("ctx_source_unpinned", 4_000_000),
            _event("gen_ingress", side="gen"),
            _event("gen_kv_admission_result", 2_000_000, side="gen", outcome="admitted"),
            _event("gen_transfer_window_result", 3_000_000, side="gen", outcome="admitted"),
            _event("gen_decode_ready", 4_000_000, side="gen"),
        ]
    )

    assert request["missing_boundaries"] == []
    assert request["unassessed_boundaries"] == []
    assert request["unsupported_boundaries"] == [
        "ctx_all_receivers_ready",
        "ctx_transfer_settled",
        "gen_receive_start",
    ]
    for participant in request["participants"]:
        assert participant["capabilities"] == {
            "status": "known",
            "transceiver_runtime": "CPP",
            "python_transfer_events": False,
            "executor_events": True,
            "scheduler_kv_admission_events": True,
            "issues": [],
        }
        assert participant["missing_boundaries"] == []
    assert _phase_reasons(request, "ctx_transfer_lifetime") == {"unsupported_capability"}
    assert _phase_reasons(request, "gen_transfer_to_service") == {"unsupported_capability"}
    assert any(
        duration["phase"] == "ctx_source_kv_request_ownership" and duration["duration_ms"] == 2.0
        for duration in request["durations"]
    )


def test_cpp_runtime_still_reports_missing_shared_executor_boundary() -> None:
    request = _request(_profile() + [_event("ctx_send_ready")])

    assert request["missing_boundaries"] == ["ctx_source_kv_released"]
    assert request["participants"][0]["missing_boundaries"] == ["ctx_source_kv_released"]
    assert _phase_reasons(request, "ctx_source_kv_request_ownership") == {"missing_end"}
    assert _phase_reasons(request, "ctx_receiver_readiness_offset") == {"unsupported_capability"}


def test_incomplete_python_trace_keeps_missing_transfer_boundaries() -> None:
    request = _request(
        _profile("PYTHON") + [_event("ctx_send_ready"), _event("ctx_source_kv_released", 3_000_000)]
    )

    assert request["missing_boundaries"] == ["ctx_all_receivers_ready", "ctx_transfer_settled"]
    assert request["unsupported_boundaries"] == []
    assert request["unassessed_boundaries"] == []
    assert _phase_reasons(request, "ctx_transfer_lifetime") == {"missing_end"}


def test_scheduler_v1_support_is_independent_of_python_transfer_support() -> None:
    request = _request(
        _profile("PYTHON", scheduler_v2=False)
        + [
            _event("gen_ingress", side="gen"),
            _event("gen_transfer_window_result", 2_000_000, side="gen", outcome="admitted"),
        ]
    )

    assert request["missing_boundaries"] == ["gen_receive_start"]
    assert request["unsupported_boundaries"] == ["gen_kv_admission_result"]
    assert request["unassessed_boundaries"] == []
    assert _phase_reasons(request, "gen_gate1_admission_wait") == {"unsupported_capability"}
    assert request["participants"][0]["capabilities"]["scheduler_kv_admission_events"] is False


@pytest.mark.parametrize("other_identity", ({"host": "node-b"}, {"pid": 18}, {"rank": 1}))
def test_capabilities_do_not_leak_across_participants(other_identity: dict[str, object]) -> None:
    request = _request(
        _profile("CPP")
        + _profile("PYTHON", **other_identity)
        + [
            _event("gen_transfer_window_result", side="gen", outcome="admitted"),
            _event("gen_transfer_window_result", side="gen", outcome="admitted", **other_identity),
        ]
    )

    participants = {
        participant["capabilities"]["transceiver_runtime"]: participant
        for participant in request["participants"]
    }
    assert len(participants) == 2
    assert participants["CPP"]["missing_boundaries"] == []
    assert participants["CPP"]["unsupported_boundaries"] == ["gen_receive_start"]
    assert participants["PYTHON"]["missing_boundaries"] == ["gen_receive_start"]
    assert participants["PYTHON"]["unsupported_boundaries"] == []
    assert request["missing_boundaries"] == ["gen_receive_start"]


def test_absent_metadata_is_unknown_not_implicitly_cpp() -> None:
    request = _request([_event("ctx_send_ready")])

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] == "unknown"
    assert capabilities["transceiver_runtime"] is None
    assert capabilities["python_transfer_events"] is None
    assert capabilities["executor_events"] is None
    assert request["missing_boundaries"] == []
    assert request["unsupported_boundaries"] == []
    assert request["unassessed_boundaries"] == [
        "ctx_all_receivers_ready",
        "ctx_source_kv_released",
        "ctx_transfer_settled",
    ]
    assert _phase_reasons(request, "ctx_transfer_lifetime") == {"unknown_capability"}


@pytest.mark.parametrize(
    "invalid_fields",
    (
        {"capability_schema_version": 2},
        {"capability_schema_version": True},
        {"python_transfer_events": 0},
        {"python_transfer_events": "false"},
        {"transceiver_runtime": "AUTO"},
    ),
)
def test_invalid_factory_metadata_does_not_silence_missing_python_edges(
    invalid_fields: dict[str, object],
) -> None:
    declaration = {
        "transceiver_runtime": "CPP",
        "python_transfer_events": False,
        **invalid_fields,
    }
    request = _request([_capability_record(**declaration), _event("ctx_send_ready")])

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] != "known"
    assert capabilities["python_transfer_events"] is None
    assert capabilities["issues"]
    assert request["missing_boundaries"] == []
    assert request["unsupported_boundaries"] == []
    assert "ctx_transfer_settled" in request["unassessed_boundaries"]
    assert _phase_reasons(request, "ctx_transfer_lifetime") == {"unknown_capability"}


def test_conflicting_runtime_declarations_are_not_resolved_by_input_order() -> None:
    profiles = _profile("CPP") + _profile("PYTHON")
    for declarations in (profiles, list(reversed(profiles))):
        request = _request(declarations + [_event("ctx_send_ready")])

        capabilities = request["participants"][0]["capabilities"]
        assert capabilities["status"] == "partial"
        assert capabilities["transceiver_runtime"] is None
        assert capabilities["python_transfer_events"] is None
        assert capabilities["executor_events"] is True
        assert capabilities["issues"]
        assert request["missing_boundaries"] == ["ctx_source_kv_released"]
        assert request["unsupported_boundaries"] == []
        assert request["unassessed_boundaries"] == [
            "ctx_all_receivers_ready",
            "ctx_transfer_settled",
        ]


def test_executor_only_metadata_preserves_shared_checks_without_assuming_runtime() -> None:
    request = _request(
        [
            _capability_record(executor_events=True, scheduler_kv_admission_events=True),
            _event("ctx_send_ready"),
        ]
    )

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] == "partial"
    assert capabilities["executor_events"] is True
    assert capabilities["python_transfer_events"] is None
    assert request["missing_boundaries"] == ["ctx_source_kv_released"]
    assert request["unassessed_boundaries"] == [
        "ctx_all_receivers_ready",
        "ctx_transfer_settled",
    ]


def test_factory_only_metadata_does_not_imply_executor_instrumentation() -> None:
    request = _request(
        [
            _capability_record(transceiver_runtime="CPP", python_transfer_events=False),
            _event("ctx_send_ready"),
        ]
    )

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] == "partial"
    assert capabilities["python_transfer_events"] is False
    assert capabilities["executor_events"] is None
    assert request["missing_boundaries"] == []
    assert request["unsupported_boundaries"] == ["ctx_all_receivers_ready", "ctx_transfer_settled"]
    assert request["unassessed_boundaries"] == ["ctx_source_kv_released"]
    assert _phase_reasons(request, "ctx_source_kv_request_ownership") == {"unknown_capability"}


def test_observed_pairs_still_measure_durations_without_capability_metadata() -> None:
    request = _request(
        [
            _event("ctx_send_ready"),
            _event("ctx_all_receivers_ready", 2_000_000),
            _event("ctx_source_kv_released", 3_000_000),
            _event("ctx_transfer_settled", 5_000_000, outcome="completed"),
        ]
    )

    assert request["participants"][0]["capabilities"]["status"] == "unknown"
    assert request["missing_boundaries"] == []
    assert request["unassessed_boundaries"] == []
    assert request["unmeasured_phases"] == []
    durations = {duration["phase"]: duration for duration in request["durations"]}
    assert durations["ctx_transfer_lifetime"]["duration_ms"] == 4.0
    assert durations["ctx_source_kv_request_ownership"]["duration_ms"] == 2.0
    assert durations["ctx_receiver_readiness_offset"]["signed_offset_ms"] == 1.0


def test_observed_python_event_overrides_unsupported_claim_without_hiding_gaps() -> None:
    request = _request(
        _profile("CPP")
        + [
            _event("ctx_send_ready"),
            _event("ctx_all_receivers_ready", 2_000_000),
            _event("ctx_source_kv_released", 3_000_000),
        ]
    )

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] == "partial"
    assert capabilities["python_transfer_events"] is None
    assert "observed_unsupported_python_transfer_events" in capabilities["issues"]
    assert capabilities["executor_events"] is True
    assert request["missing_boundaries"] == []
    assert request["unsupported_boundaries"] == []
    assert request["unassessed_boundaries"] == ["ctx_transfer_settled"]
    assert _phase_reasons(request, "ctx_transfer_lifetime") == {"unknown_capability"}
    durations = {duration["phase"]: duration for duration in request["durations"]}
    assert durations["ctx_receiver_readiness_offset"]["signed_offset_ms"] == 1.0
    assert durations["ctx_source_kv_request_ownership"]["duration_ms"] == 2.0


def test_boundaries_on_different_ranks_in_one_process_are_not_paired() -> None:
    request = _request(
        _profile("CPP", rank=0)
        + _profile("CPP", rank=1)
        + [
            _event("ctx_send_ready", rank=0),
            _event("ctx_source_kv_released", 3_000_000, rank=1),
        ]
    )

    assert not any(
        duration["phase"] == "ctx_source_kv_request_ownership" for duration in request["durations"]
    )
    gaps = {
        (gap["rank"], gap["reason"])
        for gap in request["unmeasured_phases"]
        if gap["phase"] == "ctx_source_kv_request_ownership"
    }
    assert gaps == {(0, "missing_end"), (1, "missing_start")}
    assert request["missing_boundaries"] == ["ctx_source_kv_released"]


@pytest.mark.parametrize("runtime,python_events", (("CPP", True), ("PYTHON", False)))
def test_runtime_flag_mismatch_leaves_python_support_unknown(
    runtime: str, python_events: bool
) -> None:
    request = _request(
        [
            _capability_record(transceiver_runtime=runtime, python_transfer_events=python_events),
            _capability_record(executor_events=True, scheduler_kv_admission_events=True),
            _event("ctx_send_ready"),
        ]
    )

    capabilities = request["participants"][0]["capabilities"]
    assert capabilities["status"] == "partial"
    assert capabilities["python_transfer_events"] is None
    assert "runtime_capability_mismatch" in capabilities["issues"]
    assert capabilities["executor_events"] is True
    assert request["missing_boundaries"] == ["ctx_source_kv_released"]
    assert request["unsupported_boundaries"] == []
    assert request["unassessed_boundaries"] == [
        "ctx_all_receivers_ready",
        "ctx_transfer_settled",
    ]

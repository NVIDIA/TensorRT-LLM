# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the disaggregated-transfer diagnostic log analyzer."""

import json
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

import pytest

__extra_import_path__ = ["~/scripts"]
from disagg_transfer_diagnostics import DIAGNOSTICS_LOG_PREFIX, analyze_lines, main  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _domain(host: str = "node-a", pid: int = 17) -> dict[str, object]:
    return {
        "host": host,
        "pid": pid,
        "run_uuid": "11111111-1111-4111-8111-111111111111",
        "process_uuid": str(uuid5(NAMESPACE_DNS, f"{host}:{pid}")),
    }


def _event(
    name: str,
    request_id: int | None,
    monotonic_ns: int,
    *,
    host: str = "node-a",
    pid: int = 17,
    side: str = "gen",
    **details: object,
) -> str:
    record: dict[str, object] = {
        "schema_version": 1,
        "event": name,
        "request_id": request_id,
        "side": side,
        **_domain(host, pid),
        "monotonic_ns": monotonic_ns,
        "wall_ns": 100_000_000_000 + monotonic_ns,
    }
    if name == "gen_request_data_sent":
        record["writer_cohort_known"] = True
    record.update(details)
    return f"worker-prefix {DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"


def _request(result: dict[str, object], request_id: int) -> dict[str, object]:
    requests = result["requests"]
    assert isinstance(requests, list)
    return next(request for request in requests if request["request_id"] == request_id)


def _capabilities(*, pid: int = 17, rank: int | None = None) -> str:
    return _event(
        "diagnostic_capabilities",
        None,
        0,
        pid=pid,
        rank=rank,
        side="runtime",
        capability_schema_version=1,
        transceiver_runtime="PYTHON",
        python_transfer_events=True,
        executor_events=True,
        scheduler_kv_admission_events=True,
    )


def test_noisy_logs_are_counted_and_grouped_by_canonical_request() -> None:
    lines = [
        "ordinary runtime log\n",
        f"{DIAGNOSTICS_LOG_PREFIX}not-json\n",
        f"{DIAGNOSTICS_LOG_PREFIX}[]\n",
        _event("gen_ingress", 22, 1_000_000),
        _event("gen_kv_admission_result", 22, 3_000_000, outcome="admitted"),
        _event(
            "gen_transfer_window_result",
            22,
            6_000_000,
            outcome="admitted",
            policy="bypassed",
            legacy_budget_outcome="deferred",
            legacy_active_transfer_blocks=64,
            legacy_admitted_transfer_blocks=0,
            legacy_limited_by_budget=True,
        ),
        _event("transfer_timeout_observed", None, 7_000_000),
    ]

    result = analyze_lines(lines)

    assert result["summary"] == {
        "total_lines": 7,
        "ignored_lines": 1,
        "malformed_diagnostic_lines": 2,
        "parsed_events": 4,
        "events_without_request_id": 1,
        "request_count": 1,
    }
    assert result["event_counts"] == {
        "gen_kv_admission_result": 1,
        "gen_ingress": 1,
        "gen_transfer_window_result": 1,
        "transfer_timeout_observed": 1,
    }

    request = _request(result, 22)
    assert request["missing_boundaries"] == []
    assert request["unassessed_boundaries"] == ["gen_receive_start"]
    assert [event["event"] for event in request["timeline"]] == [
        "gen_ingress",
        "gen_kv_admission_result",
        "gen_transfer_window_result",
    ]
    transfer_window = request["timeline"][-1]
    assert transfer_window["legacy_budget_outcome"] == "deferred"
    assert transfer_window["legacy_limited_by_budget"] is True
    assert "clock-sync-sensitive" in result["clock_semantics"]["timeline"]
    durations = {duration["phase"]: duration["duration_ms"] for duration in request["durations"]}
    assert durations == {
        "gen_gate1_admission_wait": 2.0,
        "gen_transfer_window_admission_wait": 3.0,
    }


def test_semantically_malformed_correlation_fields_are_quarantined() -> None:
    lines = [
        _event("ctx_transfer_queued", 1, 1, side="ctx", slice_id={}),
        _event("ctx_transfer_queued", 2, 2, side="ctx", peer_rank=[]),
        _event("ctx_transfer_queued", 3, 3, side="ctx", slice_id=True),
        _event("ctx_transfer_queued", 4, 4, side="ctx", slice_id=0, peer_rank=1),
    ]

    result = analyze_lines(lines)

    assert result["summary"] == {
        "total_lines": 4,
        "ignored_lines": 0,
        "malformed_diagnostic_lines": 3,
        "parsed_events": 1,
        "events_without_request_id": 0,
        "request_count": 1,
    }
    assert result["requests"][0]["request_id"] == 4


@pytest.mark.parametrize(
    "payload",
    (
        '{"event":"gen_ingress","request_id":NaN}',
        '{"event":"gen_ingress","request_id":Infinity}',
        '{"event":"gen_ingress","request_id":18446744073709551616}',
        '{"event":"gen_ingress","request_id":' + "1" * 5_000 + "}",
    ),
)
def test_non_finite_and_oversized_json_numbers_are_quarantined(payload: str) -> None:
    result = analyze_lines([f"{DIAGNOSTICS_LOG_PREFIX}{payload}\n"])

    assert result["summary"] == {
        "total_lines": 1,
        "ignored_lines": 0,
        "malformed_diagnostic_lines": 1,
        "parsed_events": 0,
        "events_without_request_id": 0,
        "request_count": 0,
    }


def test_excessively_nested_json_is_quarantined() -> None:
    nested = "[" * 2_000 + "0" + "]" * 2_000
    payload = f'{{"event":"gen_ingress","detail":{nested}}}'

    result = analyze_lines([f"{DIAGNOSTICS_LOG_PREFIX}{payload}\n"])

    assert result["summary"]["malformed_diagnostic_lines"] == 1
    assert result["summary"]["parsed_events"] == 0


@pytest.mark.parametrize("field", ("is_last_slice", "session_found"))
def test_numeric_boolean_field_is_quarantined(field: str) -> None:
    line = _event(
        "gen_writer_result_received",
        7,
        1,
        outcome="success",
        **{field: 1},
    )

    result = analyze_lines([line])

    assert result["summary"]["malformed_diagnostic_lines"] == 1
    assert result["summary"]["parsed_events"] == 0


@pytest.mark.parametrize("schema_version", (None, "1", True, 0, 2))
def test_unsupported_schema_version_is_quarantined(schema_version: object) -> None:
    line = _event("gen_ingress", 5, 1)
    record = json.loads(line.split(DIAGNOSTICS_LOG_PREFIX, 1)[1])
    if schema_version is None:
        record.pop("schema_version")
    else:
        record["schema_version"] = schema_version

    result = analyze_lines([f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"])

    assert result["summary"]["malformed_diagnostic_lines"] == 1
    assert result["summary"]["parsed_events"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pid", -5),
        ("slice_id", -1),
        ("rank", "0"),
        ("source_kv_reuse_block_count", True),
        ("outcome", 7),
        ("elapsed_ms", -0.5),
        ("elapsed_ms", int("9" * 400)),
        ("monotonic_ns", 1 << 63),
    ),
)
def test_invalid_known_field_is_quarantined(field: str, value: object) -> None:
    line = _event("ctx_transfer_settled", 5, 1, side="ctx", outcome="completed")
    record = json.loads(line.split(DIAGNOSTICS_LOG_PREFIX, 1)[1])
    record[field] = value

    result = analyze_lines([f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record)}\n"])

    assert result["summary"]["malformed_diagnostic_lines"] == 1
    assert result["summary"]["parsed_events"] == 0


def test_missing_fanout_correlation_is_unmeasured() -> None:
    result = analyze_lines(
        [
            _event("ctx_transfer_queued", 6, 1, side="ctx", slice_id=0),
            _event("ctx_worker_dequeued", 6, 2, side="ctx", slice_id=0),
        ]
    )

    request = _request(result, 6)
    assert all(duration["phase"] != "ctx_worker_queue_wait" for duration in request["durations"])
    assert {
        "phase": "ctx_worker_queue_wait",
        "reason": "missing_correlation_fields",
        "start_count": 1,
        "end_count": 1,
    } in request["unmeasured_phases"]


def test_single_pair_invalid_metadata_is_reported_once() -> None:
    result = analyze_lines(
        [
            _event("gen_request_data_sent", 8, 1, host="", slice_id=0, peer_rank=1),
            _event(
                "gen_writer_result_received",
                8,
                2,
                host="",
                outcome="success",
                slice_id=0,
                peer_rank=1,
            ),
        ]
    )

    invalid = [
        phase
        for phase in _request(result, 8)["unmeasured_phases"]
        if phase["phase"] == "gen_writer_first_response"
        and phase["reason"] == "invalid_clock_metadata"
    ]
    assert invalid == [
        {
            "phase": "gen_writer_first_response",
            "reason": "invalid_clock_metadata",
            "start_count": 1,
            "end_count": 1,
        }
    ]


def test_aggregate_kv_pool_snapshots_keep_headroom_fields() -> None:
    result = analyze_lines(
        [
            _event(
                "gen_kv_pool_snapshot",
                None,
                12_000_000,
                init_requests=4,
                transfers_in_progress=2,
                transfers_complete=1,
                kv_admitted_this_iteration=1,
                decode_requests=8,
                kv_pool_max_blocks=100,
                kv_pool_free_blocks=25,
                kv_pool_used_blocks=75,
                index_free_slots=6,
            ),
            _event("gen_kv_pool_snapshot", None, 22_000_000),
            _event(
                "diagnostics_events_dropped",
                None,
                32_000_000,
                side="runtime",
                dropped_events=3,
            ),
        ]
    )

    assert result["summary"]["request_count"] == 0
    snapshot = result["aggregate_timeline"][0]
    assert snapshot["event"] == "gen_kv_pool_snapshot"
    assert snapshot["init_requests"] == 4
    assert snapshot["kv_pool_free_blocks"] == 25
    assert snapshot["index_free_slots"] == 6
    assert result["aggregate_timeline"][-1]["dropped_events"] == 3
    assert result["scheduler_decision_cadence"] == [
        {
            **_domain(),
            "rank": None,
            "snapshot_count": 2,
            "interval_count": 1,
            "min_ms": 10.0,
            "mean_ms": 10.0,
            "max_ms": 10.0,
        }
    ]


def test_gen_transfer_settled_uses_the_next_same_rank_scheduler_decision() -> None:
    result = analyze_lines(
        [
            _event(
                "gen_transfer_settled",
                70,
                10_000_000,
                rank=0,
                outcome="completed",
                resources_drained=True,
            ),
            _event("gen_kv_pool_snapshot", None, 16_000_000, rank=0),
            _event(
                "gen_transfer_settled",
                71,
                20_000_000,
                rank=0,
                outcome="completed",
                resources_drained=True,
            ),
            _event(
                "gen_transfer_settled",
                72,
                21_000_000,
                rank=0,
                outcome="failed",
                resources_drained=False,
            ),
            _event("gen_kv_pool_snapshot", None, 25_000_000, rank=1),
        ]
    )

    assert result["gen_transfer_settled_to_next_scheduler_decision"] == [
        {
            **_domain(),
            "rank": 0,
            "settled_count": 2,
            "matched_count": 1,
            "unmatched_count": 1,
            "min_ms": 6.0,
            "mean_ms": 6.0,
            "max_ms": 6.0,
        }
    ]


def test_boundary_completeness_is_reported_per_participant() -> None:
    result = analyze_lines(
        [
            _capabilities(rank=0),
            _capabilities(pid=18, rank=1),
            _event("gen_ingress", 31, 1_000_000, rank=0),
            _event("gen_kv_admission_result", 31, 2_000_000, rank=0, outcome="admitted"),
            _event("gen_ingress", 31, 3_000_000, pid=18, rank=1),
        ]
    )

    request = _request(result, 31)
    # A peer's event cannot satisfy rank 1's expected admission boundary.
    assert request["missing_boundaries"] == ["gen_kv_admission_result"]
    rank_one = next(
        participant for participant in request["participants"] if participant["rank"] == 1
    )
    assert rank_one["missing_boundaries"] == ["gen_kv_admission_result"]


def test_kv_admission_does_not_infer_transfer_window_ownership() -> None:
    result = analyze_lines(
        [
            _event("gen_ingress", 33, 1_000_000, rank=0, pp_rank=0),
            _event(
                "gen_kv_admission_result",
                33,
                2_000_000,
                rank=0,
                pp_rank=0,
                outcome="admitted",
            ),
            _event("gen_ingress", 33, 1_000_000, pid=18, rank=1, pp_rank=1),
            _event(
                "gen_kv_admission_result",
                33,
                2_000_000,
                pid=18,
                rank=1,
                pp_rank=1,
                outcome="admitted",
            ),
        ]
    )

    participants = _request(result, 33)["participants"]
    rank_zero = next(participant for participant in participants if participant["rank"] == 0)
    rank_one = next(participant for participant in participants if participant["rank"] == 1)
    assert rank_zero["missing_boundaries"] == []
    assert rank_one["missing_boundaries"] == []
    assert all(
        phase["phase"] != "gen_transfer_window_admission_wait"
        for phase in _request(result, 33)["unmeasured_phases"]
    )


def test_gate2_retries_form_one_admission_span_without_false_missing_pairs() -> None:
    result = analyze_lines(
        [
            _event("gen_ingress", 32, 1_000_000),
            _event("gen_kv_admission_result", 32, 2_000_000, outcome="admitted"),
            _event("gen_transfer_window_result", 32, 3_000_000, outcome="deferred"),
            _event("gen_kv_admission_result", 32, 5_000_000, outcome="admitted"),
            _event("gen_transfer_window_result", 32, 8_000_000, outcome="admitted"),
        ]
    )

    request = _request(result, 32)
    durations = {duration["phase"]: duration["duration_ms"] for duration in request["durations"]}
    assert durations["gen_gate1_admission_wait"] == 1.0
    assert durations["gen_transfer_window_admission_wait"] == 6.0
    assert not any(
        phase["phase"] in {"gen_gate1_admission_wait", "gen_transfer_window_admission_wait"}
        for phase in request["unmeasured_phases"]
    )


def test_cross_domain_monotonic_timestamps_are_not_subtracted() -> None:
    result = analyze_lines(
        [
            _event("ctx_send_ready", 9, 900_000_000, host="ctx-node", pid=11, side="ctx"),
            _event(
                "ctx_all_receivers_ready",
                9,
                100,
                host="another-node",
                pid=22,
                side="ctx",
            ),
        ]
    )

    request = _request(result, 9)
    assert all(
        duration["phase"] != "ctx_receiver_readiness_offset" for duration in request["durations"]
    )
    assert {
        "phase": "ctx_receiver_readiness_offset",
        "reason": "clock_domain_mismatch",
    } in request["unmeasured_phases"]
    assert request["missing_boundaries"] == []
    assert request["unassessed_boundaries"] == [
        "ctx_all_receivers_ready",
        "ctx_source_kv_released",
        "ctx_transfer_settled",
    ]
    assert [event["event"] for event in request["timeline"]] == [
        "ctx_all_receivers_ready",
        "ctx_send_ready",
    ]


def test_receiver_ready_before_final_send_is_reported_as_readiness_lead() -> None:
    result = analyze_lines(
        [
            _event("ctx_all_receivers_ready", 10, 3_000_000, side="ctx"),
            _event("ctx_send_ready", 10, 8_000_000, side="ctx"),
        ]
    )

    readiness = next(
        duration
        for duration in _request(result, 10)["durations"]
        if duration["phase"] == "ctx_receiver_readiness_offset"
    )
    assert readiness["signed_offset_ms"] == -5.0
    assert readiness["readiness_lead_ms"] == 5.0
    assert readiness["readiness_wait_ms"] == 0.0


def test_fanout_boundaries_are_correlated_by_slice_and_peer() -> None:
    result = analyze_lines(
        [
            _event("gen_request_data_sent", 41, 1_000_000, slice_id=0, peer_rank=0),
            _event("gen_request_data_sent", 41, 2_000_000, slice_id=0, peer_rank=1),
            _event("gen_writer_result_received", 41, 7_000_000, slice_id=0, peer_rank=1),
            _event("gen_writer_result_received", 41, 4_000_000, slice_id=0, peer_rank=0),
        ]
    )

    durations = [
        duration
        for duration in _request(result, 41)["durations"]
        if duration["phase"] == "gen_writer_first_response"
    ]
    by_peer = {
        duration["correlation"]["peer_rank"]: duration["duration_ms"] for duration in durations
    }
    assert by_peer == {0: 3.0, 1: 5.0}


def test_writer_timeline_preserves_late_session_evidence() -> None:
    result = analyze_lines(
        [
            _event(
                "gen_writer_result_received",
                43,
                4_000_000,
                slice_id=0,
                peer_rank=2,
                session_found=False,
            ),
        ]
    )

    timeline = _request(result, 43)["timeline"]
    assert timeline[0]["session_found"] is False


def test_writer_first_response_reports_an_unmatched_fanout_peer() -> None:
    result = analyze_lines(
        [
            _capabilities(),
            _event("gen_request_data_sent", 42, 1_000_000, slice_id=0, peer_rank=0),
            _event("gen_request_data_sent", 42, 2_000_000, slice_id=0, peer_rank=1),
            _event("gen_writer_result_received", 42, 4_000_000, slice_id=0, peer_rank=0),
        ]
    )

    request = _request(result, 42)
    assert any(
        duration["phase"] == "gen_writer_first_response"
        and duration["correlation"]["peer_rank"] == 0
        for duration in request["durations"]
    )
    assert {
        "phase": "gen_writer_first_response",
        "reason": "missing_end",
        "count": 1,
        "clock_domain": _domain(),
        "correlation": {"slice_id": 0, "peer_rank": 1},
    } in request["unmeasured_phases"]


def _broadcast_send(peer_rank: int, *, expected_writers: int | None = 1, **details: object) -> str:
    return _event(
        "gen_request_data_sent",
        58,
        1_000_000,
        slice_id=0,
        peer_rank=peer_rank,
        writer_cohort_known=False,
        expected_writers=expected_writers,
        **details,
    )


def _broadcast_response(peer_rank: int, **details: object) -> str:
    return _event(
        "gen_writer_result_received",
        58,
        3_000_000,
        peer_rank=peer_rank,
        **{"slice_id": 0, "outcome": "success", "is_last_slice": True, **details},
    )


def _writer_gaps(result: dict[str, object]) -> list[dict[str, object]]:
    return [
        gap
        for gap in _request(result, 58)["unmeasured_phases"]
        if gap["phase"] == "gen_writer_first_response"
    ]


def test_complete_adp_broadcast_does_not_require_unselected_peers_to_respond() -> None:
    result = analyze_lines(
        [
            _capabilities(),
            _broadcast_send(0),
            _broadcast_send(1),
            _broadcast_response(1),
            _event("gen_destination_complete", 58, 4_000_000, slice_id=0, peer_rank=1),
        ]
    )

    assert _writer_gaps(result) == []
    duration = next(
        sample
        for sample in _request(result, 58)["durations"]
        if sample["phase"] == "gen_writer_first_response"
    )
    assert duration["duration_ms"] == 2.0
    assert duration["correlation"] == {"slice_id": 0, "peer_rank": 1}
    assert _request(result, 58)["timeline"][0]["writer_cohort_known"] is False


def test_incomplete_adp_broadcast_reports_missing_writer_count_not_candidate_peers() -> None:
    result = analyze_lines(
        [
            _broadcast_send(0, expected_writers=2),
            _broadcast_send(1, expected_writers=2),
            _broadcast_send(2, expected_writers=2),
            _broadcast_response(1, is_last_slice=False),
            _broadcast_response(1),
            _broadcast_response(2, session_found=False),
        ]
    )

    assert _writer_gaps(result) == [
        {
            "phase": "gen_writer_first_response",
            "reason": "missing_writer_responses",
            "clock_domain": _domain(),
            "correlation": {"slice_id": 0},
            "observed_writers": 1,
            "expected_writers": 2,
            "count": 1,
        }
    ]


@pytest.mark.parametrize("counts", ((None, None), (0, 0), (1, 2), (None, 1)))
def test_ambiguous_broadcast_writer_count_is_unknown(counts: tuple[int | None, int | None]) -> None:
    result = analyze_lines(
        [
            _broadcast_send(0, expected_writers=counts[0]),
            _broadcast_send(1, expected_writers=counts[1]),
            _broadcast_response(1),
        ]
    )

    gaps = _writer_gaps(result)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "unknown_writer_cohort"
    assert gaps[0]["detail"] == "missing_invalid_or_conflicting_expected_writers"


@pytest.mark.parametrize("cohort_flag", (None, True))
def test_missing_or_conflicting_writer_cohort_does_not_assume_exact_peers(
    cohort_flag: bool | None,
) -> None:
    result = analyze_lines(
        [
            _broadcast_send(0),
            _event(
                "gen_request_data_sent",
                58,
                1_000_000,
                slice_id=0,
                peer_rank=1,
                writer_cohort_known=cohort_flag,
                expected_writers=1,
            ),
            _broadcast_response(1),
        ]
    )

    assert [gap["reason"] for gap in _writer_gaps(result)] == ["unknown_writer_cohort"]


@pytest.mark.parametrize(
    "other_identity",
    (
        {"host": "node-b"},
        {"pid": 18},
        {"rank": 1},
        {"slice_id": 1},
        {"process_uuid": "22222222-2222-4222-8222-222222222222"},
    ),
)
def test_other_participant_or_slice_cannot_complete_broadcast_cohort(
    other_identity: dict[str, object],
) -> None:
    result = analyze_lines(
        [
            _broadcast_send(0),
            _broadcast_response(0, **other_identity),
        ]
    )

    missing = [gap for gap in _writer_gaps(result) if gap["reason"] == "missing_writer_responses"]
    assert len(missing) == 1
    assert missing[0]["observed_writers"] == 0
    assert missing[0]["expected_writers"] == 1


def test_duplicate_broadcast_publications_and_chunk_results_count_distinct_writers() -> None:
    result = analyze_lines(
        [
            _broadcast_send(0),
            _broadcast_send(0),
            _broadcast_send(1),
            _broadcast_response(1, is_last_slice=False),
            _broadcast_response(1),
            _broadcast_response(1),
        ]
    )

    assert _writer_gaps(result) == []
    durations = [
        sample
        for sample in _request(result, 58)["durations"]
        if sample["phase"] == "gen_writer_first_response"
    ]
    assert len(durations) == 1


def test_broadcast_does_not_hide_response_that_precedes_publication() -> None:
    result = analyze_lines(
        [
            _capabilities(),
            _broadcast_send(0, expected_writers=2),
            _broadcast_send(1, expected_writers=2),
            _event("gen_writer_result_received", 58, 0, slice_id=0, peer_rank=0),
            _broadcast_response(1),
        ]
    )

    gaps = _writer_gaps(result)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "missing_end"
    assert gaps[0]["correlation"] == {"slice_id": 0, "peer_rank": 0}


def test_pipelined_chunks_use_first_response_and_final_successful_destination() -> None:
    result = analyze_lines(
        [
            _event("gen_request_data_sent", 55, 1_000_000, slice_id=0, peer_rank=3),
            _event("gen_request_data_sent", 55, 1_500_000, slice_id=0, peer_rank=4),
            _event(
                "gen_writer_result_received",
                55,
                2_000_000,
                slice_id=0,
                peer_rank=3,
                outcome="success",
                is_last_slice=False,
            ),
            _event(
                "gen_writer_result_received",
                55,
                5_000_000,
                slice_id=0,
                peer_rank=4,
                outcome="success",
                is_last_slice=True,
            ),
            _event(
                "gen_writer_result_received",
                55,
                6_000_000,
                slice_id=0,
                peer_rank=3,
                outcome="success",
                is_last_slice=True,
            ),
            _event(
                "gen_destination_complete",
                55,
                9_000_000,
                slice_id=0,
                peer_rank=3,
                outcome="completed",
            ),
        ]
    )

    request = _request(result, 55)
    writer_durations = {
        duration["correlation"]["peer_rank"]: duration["duration_ms"]
        for duration in request["durations"]
        if duration["phase"] == "gen_writer_first_response"
    }
    destination_duration = next(
        duration
        for duration in request["durations"]
        if duration["phase"] == "gen_destination_drain"
    )
    assert writer_durations == {3: 1.0, 4: 3.5}
    assert destination_duration["duration_ms"] == 3.0
    assert not any(
        phase["phase"] in {"gen_writer_first_response", "gen_destination_drain"}
        for phase in request["unmeasured_phases"]
    )


def test_failed_writer_result_does_not_expect_destination_completion() -> None:
    result = analyze_lines(
        [
            _event("gen_request_data_sent", 56, 1_000_000, slice_id=0, peer_rank=3),
            _event(
                "gen_writer_result_received",
                56,
                4_000_000,
                slice_id=0,
                peer_rank=3,
                outcome="failed",
                is_last_slice=True,
            ),
        ]
    )

    request = _request(result, 56)
    assert "gen_destination_complete" not in request["missing_boundaries"]
    assert all(duration["phase"] != "gen_destination_drain" for duration in request["durations"])


def test_mixed_writer_results_do_not_expect_destination_completion() -> None:
    result = analyze_lines(
        [
            _event("gen_request_data_sent", 57, 1_000_000, slice_id=0, peer_rank=3),
            _event("gen_request_data_sent", 57, 1_500_000, slice_id=0, peer_rank=4),
            _event(
                "gen_writer_result_received",
                57,
                4_000_000,
                slice_id=0,
                peer_rank=3,
                outcome="success",
                is_last_slice=True,
            ),
            _event(
                "gen_writer_result_received",
                57,
                5_000_000,
                slice_id=0,
                peer_rank=4,
                outcome="failed",
                is_last_slice=True,
            ),
        ]
    )

    request = _request(result, 57)
    assert "gen_destination_complete" not in request["missing_boundaries"]
    assert all(duration["phase"] != "gen_destination_drain" for duration in request["durations"])


def test_timeout_phases_use_the_endpoint_local_monotonic_clock() -> None:
    result = analyze_lines(
        [
            _event(
                "ctx_send_ready",
                77,
                1_000_000,
                side="ctx",
                timeout_expected=True,
            ),
            _event(
                "transfer_timeout_started",
                77,
                2_000_000,
                side="ctx",
                timer_start_monotonic_ns=1_500_000,
                timeout_ms=60,
                timeout_owner="pyexecutor",
                state="DISAGG_CONTEXT_TRANS_IN_PROGRESS",
            ),
            _event(
                "transfer_timeout_observed",
                77,
                62_000_000,
                side="ctx",
                timeout_owner="pyexecutor",
            ),
        ]
    )

    durations = {
        duration["phase"]: duration["duration_ms"] for duration in _request(result, 77)["durations"]
    }
    assert durations["ctx_send_to_timeout_start"] == 0.5
    assert durations["transfer_timeout_window"] == 60.5
    timeout_start = next(
        event
        for event in _request(result, 77)["timeline"]
        if event["event"] == "transfer_timeout_started"
    )
    assert timeout_start["timer_start_monotonic_ns"] == 1_500_000
    assert timeout_start["timeout_ms"] == 60
    assert timeout_start["timeout_owner"] == "pyexecutor"
    assert timeout_start["state"] == "DISAGG_CONTEXT_TRANS_IN_PROGRESS"


def test_healthy_or_inapplicable_timeout_phases_are_not_reported_missing() -> None:
    result = analyze_lines(
        [
            _event(
                "ctx_send_ready",
                78,
                1_000_000,
                side="ctx",
                timeout_expected=True,
            ),
            _event(
                "transfer_timeout_started",
                78,
                2_000_000,
                side="ctx",
                timeout_owner="pyexecutor",
            ),
            _event(
                "gen_receive_start",
                79,
                3_000_000,
                timeout_expected=False,
            ),
            _event(
                "ctx_send_ready",
                80,
                4_000_000,
                side="ctx",
                timeout_expected=False,
            ),
        ]
    )

    healthy = _request(result, 78)
    assert all(
        phase["phase"] != "transfer_timeout_window" for phase in healthy["unmeasured_phases"]
    )
    sync_gen = _request(result, 79)
    assert all(
        phase["phase"] != "gen_receive_to_timeout_start" for phase in sync_gen["unmeasured_phases"]
    )
    timeout_disabled_ctx = _request(result, 80)
    assert all(
        phase["phase"] != "ctx_send_to_timeout_start"
        for phase in timeout_disabled_ctx["unmeasured_phases"]
    )


def test_ctx_worker_queue_wait_is_correlated_per_slice_and_peer() -> None:
    result = analyze_lines(
        [
            _capabilities(),
            _event(
                "ctx_transfer_queued",
                88,
                2_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=0,
                receiver_slice_id=2,
                is_last_slice=True,
            ),
            _event(
                "ctx_transfer_queued",
                88,
                3_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=1,
                receiver_slice_id=2,
                is_last_slice=True,
            ),
            _event(
                "ctx_worker_dequeued",
                88,
                7_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=0,
            ),
            _event(
                "ctx_backend_submit_start",
                88,
                11_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=0,
            ),
            _event(
                "ctx_backend_submitted",
                88,
                12_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=0,
            ),
            _event(
                "ctx_backend_complete",
                88,
                20_000_000,
                side="ctx",
                slice_id=1,
                peer_rank=0,
            ),
        ]
    )

    request = _request(result, 88)
    queue_wait = next(
        duration
        for duration in request["durations"]
        if duration["phase"] == "ctx_worker_queue_wait"
    )
    assert queue_wait["duration_ms"] == 5.0
    assert queue_wait["correlation"] == {"slice_id": 1, "peer_rank": 0}
    preparation = next(
        duration
        for duration in request["durations"]
        if duration["phase"] == "ctx_worker_preparation"
    )
    assert preparation["duration_ms"] == 4.0
    assert preparation["correlation"] == {"slice_id": 1, "peer_rank": 0}
    assert {
        "phase": "ctx_worker_queue_wait",
        "reason": "missing_end",
        "count": 1,
        "clock_domain": _domain(),
        "correlation": {"slice_id": 1, "peer_rank": 1},
    } in request["unmeasured_phases"]
    queued = request["timeline"][0]
    assert queued["receiver_slice_id"] == 2
    assert queued["is_last_slice"] is True


def test_cli_reads_files_and_emits_json(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "worker.log"
    log.write_text(_event("gen_ingress", 5, 100), encoding="utf-8")

    assert main([str(log), "--indent", "0"]) == 0

    result = json.loads(capfd.readouterr().out)
    assert result["summary"]["request_count"] == 1
    assert result["requests"][0]["request_id"] == 5

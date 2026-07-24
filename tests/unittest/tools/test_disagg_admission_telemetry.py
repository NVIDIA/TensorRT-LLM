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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "disagg_admission_telemetry.py"
_SPEC = importlib.util.spec_from_file_location("disagg_admission_telemetry", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load telemetry analyzer from {_SCRIPT_PATH}")
_TELEMETRY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TELEMETRY
_SPEC.loader.exec_module(_TELEMETRY)

analyze_events = _TELEMETRY.analyze_events
main = _TELEMETRY.main
parse_diagnostic_line = _TELEMETRY.parse_diagnostic_line
DiagnosticEvent = _TELEMETRY.DiagnosticEvent


def _parse_lines(lines: list[str]):
    return [event for line in lines if (event := parse_diagnostic_line(line)) is not None]


def _parse_source_line(line: str, source: str):
    event = parse_diagnostic_line(line)
    assert event is not None
    return DiagnosticEvent(
        event.category,
        event.time_s,
        event.rank,
        event.fields,
        source,
    )


def test_parse_diagnostic_line_accepts_rank_prefix_and_ignores_malformed_lines():
    event = parse_diagnostic_line(
        "INFO [RANK 3] [DISAGG_DIAG][admission] t=12.5 active_blocks=8 "
        "candidate_requests=101:4,102:4 admitted=1 deferred=1 budget=16"
    )

    assert event is not None
    assert event.category == "admission"
    assert event.time_s == 12.5
    assert event.rank == "3"
    assert event.fields["candidate_requests"] == "101:4,102:4"
    assert parse_diagnostic_line("ordinary log line") is None
    assert parse_diagnostic_line("[DISAGG_DIAG][submit] t=not-a-number rank=0") is None


def test_python_transfer_analysis_derives_refill_multiplier_and_progress_credit():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision] t=0.0 rank=0 sequence=1 runtime=Python "
            "active_blocks=0 candidates=2 candidate_blocks=20 admitted=1 "
            "admitted_blocks=10 deferred=1 deferred_blocks=10 budget=10",
            "[DISAGG_DIAG][admission] t=0.0 rank=0 active_blocks=0 "
            "candidate_requests=1:10,2:10 admitted=1 admitted_requests=1:10 "
            "deferred=1 deferred_requests=2:10 budget=10 sequence=1",
            "[DISAGG_DIAG][submit] t=0.1 rank=0 request=1 blocks=10 "
            "submit_start_t=0.05 submit_call_ms=50",
            "[DISAGG_DIAG][decision] t=0.5 rank=0 sequence=2 runtime=Python "
            "active_blocks=10 candidates=1 candidate_blocks=10 admitted=0 "
            "admitted_blocks=0 deferred=1 deferred_blocks=10 budget=10",
            "[DISAGG_DIAG][admission] t=0.5 rank=0 active_blocks=10 "
            "candidate_requests=2:10 admitted=0 admitted_requests=- deferred=1 "
            "deferred_requests=2:10 budget=10 sequence=2",
            "[DISAGG_DIAG][python-transfer] t=1.1 rank=0 action=local-ready request=1 "
            "bytes=4096 service_start_t=0.3 outcome=completed",
            "[DISAGG_DIAG][reap] t=1.2 rank=0 request=1 blocks=10 ready_t=1.1 "
            "ready_to_reap_ms=100 outcome=completed",
            "[DISAGG_DIAG][decision] t=1.25 rank=0 sequence=3 runtime=Python "
            "active_blocks=10 candidates=1 candidate_blocks=10 admitted=0 "
            "admitted_blocks=0 deferred=1 deferred_blocks=10 budget=10",
            "[DISAGG_DIAG][decision] t=1.3 rank=0 sequence=4 runtime=Python "
            "active_blocks=0 candidates=1 candidate_blocks=10 admitted=1 "
            "admitted_blocks=10 deferred=0 deferred_blocks=0 budget=10",
            "[DISAGG_DIAG][admission] t=1.3 rank=0 active_blocks=0 "
            "candidate_requests=2:10 admitted=1 admitted_requests=2:10 deferred=0 "
            "deferred_requests=- budget=10 sequence=4",
            "[DISAGG_DIAG][submit] t=1.4 rank=0 request=2 blocks=10 "
            "submit_start_t=1.35 submit_call_ms=50",
            "[DISAGG_DIAG][python-transfer] t=2.4 rank=0 action=local-ready request=2 "
            "bytes=4096 service_start_t=1.6 outcome=completed",
            "[DISAGG_DIAG][reap] t=2.5 rank=0 request=2 blocks=10 ready_t=2.4 "
            "ready_to_reap_ms=-1 outcome=completed",
            "[DISAGG_DIAG][status-poll] t=2.6 rank=0 poll_start_t=2.598 "
            "poll_call_ms=2.0 at_least_num=1 tracked=1 completed=0 failed=0 "
            "cancelled=0",
            "[DISAGG_DIAG][status-poll] t=2.7 rank=0 poll_start_t=2.6995 "
            "poll_call_ms=0.5 at_least_num=1 tracked=1 completed=1 failed=0 "
            "cancelled=0",
            "[DISAGG_DIAG][submit] t=bad rank=0 request=broken blocks=10",
        ]
    )

    result = analyze_events(events)
    rank = result["ranks"]["0"]
    service = rank["service"]
    python_transfer = rank["python_transfer"]
    status_poll = rank["status_poll"]
    visibility = rank["scheduler_visibility"]
    release = rank["release_to_admission"]
    progress = rank["linear_progress_credit"]
    counterfactual = rank["fixed_multiplier_counterfactual"]

    assert result["parsed_event_count"] == 15
    assert service["completed_blocks"] == 20.0
    assert service["busy_s"] == pytest.approx(1.6)
    assert service["throughput_blocks_per_s"] == pytest.approx(12.5)
    assert service["latency_s"]["p50"] == pytest.approx(0.8)
    assert python_transfer["submit_to_service_start_s"]["p50"] == pytest.approx(0.25)
    assert python_transfer["ready_to_reap_s"]["p50"] == pytest.approx(0.1)
    assert status_poll["no_progress_duration_ms"]["p50"] == pytest.approx(2.0)
    assert status_poll["progress_duration_ms"]["p50"] == pytest.approx(0.5)
    assert visibility["reported_ready_to_reap_ms"]["p50"] == pytest.approx(100.0)
    assert visibility["invalid_reported_ready_to_reap_samples"] == 1
    assert result["aggregate"]["status_poll"]["no_progress_duration_ms"]["p50"] == pytest.approx(
        2.0
    )

    assert release["selected_release_source"] is None
    assert release["by_source"]["reap"]["decision_gap_s"]["p50"] == pytest.approx(0.05)
    assert release["by_source"]["reap"]["successful_admission_gap_s"]["p50"] == pytest.approx(0.1)
    assert release["by_source"]["reap"]["refill_gap_s"]["p50"] == pytest.approx(0.15)
    assert rank["shadow_multiplier"]["by_source"]["reap"]["summary"]["p50"] == pytest.approx(1.1875)

    assert len(progress["samples"]) == 1
    assert progress["samples"][0]["estimated_progress_credit_blocks"] == pytest.approx(2.5)
    assert progress["samples"][0]["estimated_remaining_blocks"] == pytest.approx(7.5)
    assert progress["samples"][0]["estimated_progress_fraction"] == pytest.approx(0.25)
    assert counterfactual["next_deferred_required_multiplier"]["count"] == 2
    assert counterfactual["next_deferred_required_multiplier"]["p50"] == pytest.approx(2.0)
    assert counterfactual["samples"][0]["next_deferred_request"] == "2"
    assert [
        prefix["required_multiplier"] for prefix in counterfactual["samples"][0]["prefixes"]
    ] == pytest.approx([1.0, 2.0])


def test_receiver_slot_analysis_matches_reuse_and_backlog_refill_gap():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][admission] t=0.0 rank=2 active_blocks=0 "
            "candidate_requests=11:4,12:4 admitted=1 admitted_requests=11:4 "
            "deferred=1 deferred_requests=12:4 budget=4",
            "[DISAGG_DIAG][submit] t=0.05 rank=2 request=11 blocks=4",
            "[DISAGG_DIAG][receiver-slot] t=0.1 rank=2 action=acquire request=11 "
            "manager_index=0 manager=0xabc buffer=7 wait_ms=2.5",
            "[DISAGG_DIAG][receiver-slot] t=0.12 rank=2 action=acquired request=11 "
            "manager_index=1 manager=0xdef buffer=9 wait_ms=3.0",
            "[DISAGG_DIAG][receiver-slot] t=0.5 rank=2 action=release request=11 "
            "manager=0xabc buffer=7",
            "[DISAGG_DIAG][receiver-slot] t=0.6 rank=2 action=released request=11 "
            "manager=0xdef buffer=9",
            "[DISAGG_DIAG][reap] t=0.65 rank=2 request=11 blocks=4 "
            "ready_to_reap_ms=25 outcome=completed",
            "[DISAGG_DIAG][admission] t=0.7 rank=2 active_blocks=0 "
            "candidate_requests=12:4 admitted=1 admitted_requests=12:4 deferred=0 "
            "deferred_requests=- budget=4",
            "[DISAGG_DIAG][submit] t=0.75 rank=2 request=12 blocks=4",
            "[DISAGG_DIAG][receiver-slot] t=0.8 rank=2 action=acquired request=12 "
            "manager_index=0 manager=0xabc buffer=7 wait_ms=1.0",
            "[DISAGG_DIAG][receiver-slot] t=0.82 rank=2 action=acquired request=12 "
            "manager_index=1 manager=0xdef buffer=9 wait_ms=1.5",
            "[DISAGG_DIAG][python-transfer] t=0.9 rank=2 action=local-ready request=11 bytes=4096",
            "[DISAGG_DIAG][receiver-slot] t=1.2 rank=2 action=released request=12 "
            "manager=0xabc buffer=7",
            "[DISAGG_DIAG][receiver-slot] t=1.3 rank=2 action=released request=12 "
            "manager=0xdef buffer=9",
            "[DISAGG_DIAG][reap] t=1.35 rank=2 request=12 blocks=4 "
            "ready_to_reap_ms=35 outcome=completed",
            "[DISAGG_DIAG][python-transfer] t=1.5 rank=2 action=local-ready request=12 bytes=4096",
            "[DISAGG_DIAG][receiver-slot] t=1.4 rank=2 action=released request=999 "
            "manager=0xmissing buffer=3",
        ]
    )

    result = analyze_events(events)
    rank = result["ranks"]["2"]
    slots = rank["receiver_slots"]
    service = rank["service"]
    release = rank["release_to_admission"]
    visibility = rank["scheduler_visibility"]

    assert release["selected_release_source"] == "receiver-slot"
    assert slots["service_latency_s"]["count"] == 4
    assert service["latency_s"]["count"] == 2
    assert service["latency_s"]["p50"] == pytest.approx(0.5)
    assert service["completed_blocks"] == 8.0
    assert slots["submit_to_service_start_s"]["p50"] == pytest.approx(0.05)
    assert all(
        interval["start_kind"] == "receiver-slot-acquired" for interval in service["intervals"]
    )
    assert slots["wait_ms"]["p50"] == pytest.approx(2.0)
    assert slots["unmatched_releases"] == 1
    assert slots["backlog_refill_gap_s"]["p50"] == pytest.approx(0.26)
    assert release["selected_decision_gap_s"]["p50"] == pytest.approx(0.1)
    assert release["selected_refill_gap_s"]["p50"] == pytest.approx(0.15)
    assert release["selected_samples"][0]["release_t"] == pytest.approx(0.6)
    assert visibility["reported_ready_to_reap_ms"]["p50"] == pytest.approx(30.0)
    assert visibility["physical_release_to_reap_s"]["p50"] == pytest.approx(0.05)
    assert result["aggregate"]["scheduler_visibility"]["reported_ready_to_reap_ms"][
        "p50"
    ] == pytest.approx(30.0)


def test_reap_release_uses_first_decision_then_matching_deferred_refill():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision] t=0.0 rank=0 sequence=1 active_blocks=0 "
            "candidates=2 candidate_blocks=8 admitted=1 admitted_blocks=4 "
            "deferred=1 deferred_blocks=4 budget=4",
            "[DISAGG_DIAG][admission] t=0.0 rank=0 sequence=1 active_blocks=0 "
            "candidate_requests=1:4,2:4 admitted=1 admitted_requests=1:4 "
            "deferred=1 deferred_requests=2:4 budget=4",
            "[DISAGG_DIAG][submit] t=0.1 rank=0 request=1 blocks=4",
            "[DISAGG_DIAG][python-transfer] t=0.9 rank=0 action=local-ready "
            "request=1 service_start_t=0.2 outcome=completed",
            "[DISAGG_DIAG][reap] t=1.0 rank=0 request=1 blocks=4 outcome=completed",
            "[DISAGG_DIAG][decision] t=1.1 rank=0 sequence=2 active_blocks=4 "
            "candidates=1 candidate_blocks=4 admitted=0 admitted_blocks=0 "
            "deferred=1 deferred_blocks=4 budget=4",
            "[DISAGG_DIAG][decision] t=1.5 rank=0 sequence=3 active_blocks=0 "
            "candidates=1 candidate_blocks=4 admitted=1 admitted_blocks=4 "
            "deferred=0 deferred_blocks=0 budget=4",
            "[DISAGG_DIAG][admission] t=1.5 rank=0 sequence=3 active_blocks=0 "
            "candidate_requests=2:4 admitted=1 admitted_requests=2:4 deferred=0 "
            "deferred_requests=- budget=4",
            "[DISAGG_DIAG][submit] t=1.55 rank=0 request=2 blocks=4 "
            "submit_start_t=1.5 submit_call_ms=50",
        ]
    )

    result = analyze_events(events)
    sample = result["ranks"]["0"]["release_to_admission"]["by_source"]["reap"]["samples"][0]

    assert sample["decision_gap_s"] == pytest.approx(0.1)
    assert sample["successful_admission_gap_s"] == pytest.approx(0.5)
    assert sample["refill_gap_s"] == pytest.approx(0.5)
    assert sample["backlog_identity_unknown"] is False
    assert sample["matched_backlog_request_ids"] == ["2"]
    assert sample["eligible_for_multiplier_fit"] is True


def test_failed_transfer_contributes_no_service_or_release_samples():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][submit] t=0.1 rank=0 request=9 blocks=4 outcome=failed",
            "[DISAGG_DIAG][receiver-slot] t=0.2 rank=0 action=acquired request=9 "
            "manager_index=0 manager=0xabc buffer=1",
            "[DISAGG_DIAG][python-transfer] t=0.5 rank=0 action=local-ready "
            "request=9 service_start_t=0.3",
            "[DISAGG_DIAG][receiver-slot] t=0.6 rank=0 action=released request=9 "
            "manager=0xabc buffer=1",
            "[DISAGG_DIAG][reap] t=0.7 rank=0 request=9 blocks=4 outcome=failed "
            "state=DISAGG_TRANS_ERROR",
            "[DISAGG_DIAG][receiver-transfer] t=0.8 rank=0 action=failed "
            "request=7 context_request=9",
        ]
    )

    rank = analyze_events(events)["ranks"]["0"]

    assert rank["service"]["intervals"] == []
    assert rank["service"]["completed_blocks"] == 0
    assert rank["service"]["throughput_blocks_per_s"] is None
    assert rank["receiver_slots"]["excluded_unsuccessful_intervals"] == 1
    assert rank["python_transfer"]["ready_to_reap_s"]["count"] == 0
    assert all(
        not source["samples"] for source in rank["release_to_admission"]["by_source"].values()
    )


def test_unknown_backlog_identity_is_excluded_from_multiplier_fit():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][admission] t=0.0 rank=0 active_blocks=0 "
            "candidate_requests=1:4 admitted=0 admitted_requests=- deferred=1 "
            "deferred_requests=- budget=4",
            "[DISAGG_DIAG][submit] t=0.1 rank=0 request=8 blocks=4",
            "[DISAGG_DIAG][python-transfer] t=0.8 rank=0 action=local-ready "
            "request=8 service_start_t=0.2 outcome=completed",
            "[DISAGG_DIAG][reap] t=1.0 rank=0 request=8 blocks=4 outcome=completed",
            "[DISAGG_DIAG][admission] t=1.2 rank=0 active_blocks=0 "
            "candidate_requests=99:4 admitted=1 admitted_requests=99:4 deferred=0 "
            "deferred_requests=- budget=4",
            "[DISAGG_DIAG][submit] t=1.2 rank=0 request=99 blocks=4",
        ]
    )

    rank = analyze_events(events)["ranks"]["0"]
    sample = rank["release_to_admission"]["by_source"]["reap"]["samples"][0]

    assert sample["backlog_identity_unknown"] is True
    assert sample["eligible_for_multiplier_fit"] is False
    assert sample["shadow_multiplier"] is None
    assert rank["shadow_multiplier"]["by_source"]["reap"]["summary"]["count"] == 0


def test_cli_reads_log_paths_and_prints_json(tmp_path, capsys):
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "noise\n[DISAGG_DIAG][submit] t=1.0 rank=0 request=5 blocks=2\n",
        encoding="utf-8",
    )

    assert main([str(log_path), "--indent", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["parsed_event_count"] == 1
    assert output["event_counts"] == {"submit": 1}


def test_cli_namespaces_overlapping_ranks_from_distinct_logs(tmp_path, capsys):
    ctx_log = tmp_path / "ctx.log"
    gen_log = tmp_path / "gen.log"
    ctx_log.write_text(
        "[DISAGG_DIAG][status-poll] t=1.0 rank=0 poll_call_ms=5 completed=0 failed=0 cancelled=0\n",
        encoding="utf-8",
    )
    gen_log.write_text(
        "[DISAGG_DIAG][submit] t=2.0 rank=0 request=5 blocks=2\n",
        encoding="utf-8",
    )

    assert main([str(ctx_log), str(gen_log), "--indent", "0"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["rank_namespace"] == "source-path::rank"
    assert output["aggregate_scope"] == "all-input-sources"
    assert sorted(output["ranks"]) == [
        f"{ctx_log}::rank=0",
        f"{gen_log}::rank=0",
    ]
    assert output["source_aggregates"][str(ctx_log)]["event_counts"] == {"status-poll": 1}
    assert output["source_aggregates"][str(gen_log)]["event_counts"] == {"submit": 1}


def test_multi_log_block_lookup_is_scoped_by_source(tmp_path, capsys):
    ctx_log = tmp_path / "ctx.log"
    gen_log = tmp_path / "gen.log"
    ctx_log.write_text(
        "[DISAGG_DIAG][admission] t=0 rank=0 active_blocks=0 "
        "candidate_requests=shared:4 admitted=1 admitted_requests=shared:4 "
        "deferred=0 deferred_requests=- budget=4\n"
        "[DISAGG_DIAG][receiver-slot] t=0.1 rank=1 action=acquired "
        "request=shared manager=ctx buffer=0\n"
        "[DISAGG_DIAG][receiver-slot] t=0.2 rank=1 action=released "
        "request=shared manager=ctx buffer=0\n",
        encoding="utf-8",
    )
    gen_log.write_text(
        "[DISAGG_DIAG][admission] t=0 rank=0 active_blocks=0 "
        "candidate_requests=shared:2 admitted=1 admitted_requests=shared:2 "
        "deferred=0 deferred_requests=- budget=2\n"
        "[DISAGG_DIAG][receiver-slot] t=0.1 rank=1 action=acquired "
        "request=shared manager=gen buffer=0\n"
        "[DISAGG_DIAG][receiver-slot] t=0.2 rank=1 action=released "
        "request=shared manager=gen buffer=0\n",
        encoding="utf-8",
    )

    assert main([str(ctx_log), str(gen_log), "--indent", "0"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["ranks"][f"{ctx_log}::rank=1"]["service"]["completed_blocks"] == 4
    assert output["ranks"][f"{gen_log}::rank=1"]["service"]["completed_blocks"] == 2


def test_cpp_lifecycle_and_remaining_work_use_same_domain_completion():
    source = "gen-worker.log"
    events = [
        _parse_source_line(line, source)
        for line in [
            "[DISAGG_DIAG][gen-arrival] t=0.0 rank=0 request=42",
            "[DISAGG_DIAG][gen-activation] t=0.1 rank=0 request=42",
            "[DISAGG_DIAG][decision] t=0.2 rank=0 sequence=1 "
            "active_requests=- candidate_requests=42:8 admitted_requests=42:8 "
            "deferred_requests=- admitted=1 deferred=0 budget=8",
            "[DISAGG_DIAG][submit] t=0.25 rank=0 request=42 blocks=8",
            "[DISAGG_DIAG][receiver-transfer] t=0.3 rank=0 "
            "action=request-info-submitted request=7 context_request=42",
            "[DISAGG_DIAG][decision] t=0.4 rank=0 sequence=2 "
            "active_requests=42:8 candidate_requests=- admitted_requests=- "
            "deferred_requests=- admitted=0 deferred=0 budget=8",
            "[DISAGG_DIAG][receiver-transfer] t=0.9 rank=0 "
            "action=completed request=7 context_request=42",
            "[DISAGG_DIAG][reap] t=1.0 rank=0 request=42 blocks=8 outcome=completed",
            "[DISAGG_DIAG][gen-service] t=1.05 rank=0 action=decode-start-proxy request=42",
        ]
    ]

    result = analyze_events(events)
    remaining = result["remaining_work_ground_truth"]
    sample = remaining["samples"][0]

    assert sample["request"] == "42"
    assert sample["active_age_s"] == pytest.approx(0.15)
    assert sample["residual_ready_s"] == pytest.approx(0.5)
    assert sample["ready_kind"] == "receiver-transfer:completed"
    assert sample["residual_reap_s"] == pytest.approx(0.6)
    assert remaining["ready_coverage"] == {
        "eligible": 1,
        "observed": 1,
        "censored": 0,
        "censor_reasons": {},
    }
    lifecycle = result["lifecycle"]["interval_coverage"]
    assert lifecycle["gen_arrival_to_activation"]["duration_s"]["p50"] == pytest.approx(0.1)
    assert lifecycle["gen_activation_to_first_gate2"]["duration_s"]["p50"] == pytest.approx(0.1)
    assert lifecycle["gate2_admit_to_submit"]["duration_s"]["p50"] == pytest.approx(0.05)
    assert lifecycle["submit_to_request_info"]["duration_s"]["p50"] == pytest.approx(0.05)
    assert lifecycle["ready_to_reap"]["duration_s"]["p50"] == pytest.approx(0.1)
    assert lifecycle["gen_arrival_to_decode_start"]["duration_s"]["p50"] == pytest.approx(1.05)
    assert lifecycle["ready_to_decode_start"]["duration_s"]["p50"] == pytest.approx(0.15)
    assert lifecycle["reap_to_decode_start"]["duration_s"]["p50"] == pytest.approx(0.05)


def test_deadline_is_nonterminal_and_classified_by_sender_phase():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0.00 clock=local_steady host=ctx "
            "rank=0 action=timer-start request=42",
            "[DISAGG_DIAG][ctx-transfer] t=0.01 clock=local_steady host=ctx "
            "rank=0 action=queued request=42",
            "[DISAGG_DIAG][sender-transfer] t=0.20 clock=local_steady host=ctx "
            "rank=0 action=credit-received request=42",
            "[DISAGG_DIAG][python-transfer] t=0.30 clock=local_steady host=ctx "
            "rank=0 role=ctx source=native action=first-write-submitted request=42",
            "[DISAGG_DIAG][ctx-transfer] t=0.60 clock=local_steady host=ctx "
            "rank=0 action=deadline-observed request=42",
            "[DISAGG_DIAG][ctx-transfer] t=0.65 clock=local_steady host=ctx "
            "rank=0 action=cancel-result request=42 result=retry",
            "[DISAGG_DIAG][python-transfer] t=0.80 clock=local_steady host=ctx "
            "rank=0 role=ctx source=native action=kv-physical-complete request=42",
            "[DISAGG_DIAG][ctx-transfer] t=0.90 clock=local_steady host=ctx "
            "rank=0 action=reaped request=42 outcome=completed",
        ]
    )

    result = analyze_events(events)
    lifecycle = result["lifecycle"]
    record = next(
        request
        for request in lifecycle["requests"]
        if request["request"] == "42" and request["role"] == "ctx"
    )

    assert result["schema_version"] == 3
    assert lifecycle["deadline_phase_counts"] == {"kv-transfer-service": 1}
    assert record["deadline_phase"] == "kv-transfer-service"
    assert record["intervals"]["ctx_timer_to_deadline"]["duration_s"] == pytest.approx(0.6)
    assert record["intervals"]["ctx_deadline_to_kv_physical_complete"][
        "duration_s"
    ] == pytest.approx(0.2)
    assert record["intervals"]["ctx_deadline_to_cancel_result"]["duration_s"] == pytest.approx(0.05)
    assert _TELEMETRY._collect_unsuccessful_requests(events) == set()


def test_cross_host_wall_clock_joins_ctx_deadline_to_gen_deferral():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=100.0 clock=local_steady "
            "wall_t=1000.0 wall_clock=unix wall_semantics=boundary-sampled "
            "host=ctx rank=0 action=timer-start request=42",
            "[DISAGG_DIAG][gate1] t=1.0 clock=local_steady "
            "wall_t=1005.0 wall_clock=unix wall_semantics=boundary-sampled "
            "host=gen rank=0 action=fitting request=42",
            "[DISAGG_DIAG][gate2] t=2.0 clock=local_steady "
            "wall_t=1006.0 wall_clock=unix wall_semantics=boundary-sampled "
            "host=gen rank=0 action=deferred request=42",
            "[DISAGG_DIAG][ctx-transfer] t=160.0 clock=local_steady "
            "wall_t=1060.0 wall_clock=unix wall_semantics=boundary-sampled "
            "host=ctx rank=0 action=deadline-observed request=42",
            "[DISAGG_DIAG][gate2] t=62.0 clock=local_steady "
            "wall_t=1062.0 wall_clock=unix wall_semantics=boundary-sampled "
            "host=gen rank=0 action=admitted request=42",
        ]
    )

    correlation = analyze_events(events)["cross_host_correlation"]
    record = correlation["requests"][0]

    assert correlation["joined_ctx_gen_request_count"] == 1
    assert correlation["ctx_deadline_relationship_counts"] == {"during-gate2-deferral": 1}
    assert record["ctx_deadline_relationship"] == "during-gate2-deferral"
    assert record["wall_intervals_s"]["ctx_timer_to_gate2_admit"] == pytest.approx(62.0)
    assert record["wall_intervals_s"]["gate2_defer_to_ctx_deadline"] == pytest.approx(54.0)


def test_cross_host_partial_gate2_admission_is_not_labeled_before_gate1():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0 wall_t=0 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=timer-start request=42",
            "[DISAGG_DIAG][gate2] t=2 wall_t=50 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=admitted request=42",
            "[DISAGG_DIAG][gate2] t=3 wall_t=70 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=1 "
            "action=admitted request=42",
            "[DISAGG_DIAG][submit] t=4 wall_t=55 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 request=42",
            "[DISAGG_DIAG][ctx-transfer] t=60 wall_t=60 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=deadline-observed request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert record["points"]["gate2-state-at-ctx-deadline"]["state"] == "admitted"
    assert record["ctx_deadline_relationship"] == "partial-gate2-admission-before-global-admission"
    assert "negative:gate2-admitted->gen-submit" not in record["wall_clock_anomalies"]
    assert "cross-emitter-selection:gate2-admitted->gen-submit" in record["wall_clock_anomalies"]


def test_cross_host_wall_clock_preserves_same_emitter_negative_order():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=10 wall_t=10 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx instance=ctx rank=0 "
            "action=timer-start request=42",
            "[DISAGG_DIAG][ctx-transfer] t=5 wall_t=5 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx instance=ctx rank=0 "
            "action=deadline-observed request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert "negative:ctx-timer-start->ctx-deadline" in record["wall_clock_anomalies"]


def test_cross_host_wall_clock_preserves_single_cross_emitter_inversion():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][submit] t=10 wall_t=10 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen instance=gen rank=0 "
            "request=42",
            "[DISAGG_DIAG][sender-transfer] t=9 wall_t=9 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx instance=ctx rank=0 "
            "role=ctx action=receiver-info-ready request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert "negative:gen-submit->ctx-receiver-credit" in record["wall_clock_anomalies"]


def test_cross_host_wall_clock_preserves_multi_rank_same_emitter_inversion():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][gate2] t=70 wall_t=70 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen instance=gen rank=0 "
            "action=admitted request=42",
            "[DISAGG_DIAG][gate2] t=60 wall_t=60 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen instance=gen rank=1 "
            "action=admitted request=42",
            "[DISAGG_DIAG][submit] t=55 wall_t=55 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen instance=gen rank=0 "
            "request=42",
            "[DISAGG_DIAG][submit] t=50 wall_t=50 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen instance=gen rank=1 "
            "request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert "negative:gate2-admitted->gen-submit" in record["wall_clock_anomalies"]


def test_cross_host_progress_uses_latest_observed_rank():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0 wall_t=0 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=timer-start request=42",
            "[DISAGG_DIAG][gate2] t=1 wall_t=10 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=deferred request=42",
            "[DISAGG_DIAG][gate2] t=2 wall_t=50 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=admitted request=42",
            "[DISAGG_DIAG][gate2] t=3 wall_t=70 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=1 "
            "action=admitted request=42",
            "[DISAGG_DIAG][ctx-transfer] t=60 wall_t=60 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=deadline-observed request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]
    admission = record["points"]["gate2-admitted"]

    assert admission["wall_t"] == pytest.approx(70.0)
    assert admission["selection"] == "latest"
    assert admission["observed_emitter_count"] == 2
    assert record["ctx_deadline_relationship"] == "partial-gate2-admission-before-global-admission"


def test_cross_host_gate2_ineligible_closes_deferral_episode():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0 wall_t=0 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=timer-start request=42",
            "[DISAGG_DIAG][gate1] t=1 wall_t=5 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=fitting request=42",
            "[DISAGG_DIAG][gate2] t=2 wall_t=10 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=deferred request=42",
            "[DISAGG_DIAG][gate1] t=3 wall_t=20 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=blocked request=42",
            "[DISAGG_DIAG][gate2] t=3 wall_t=20 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=ineligible request=42",
            "[DISAGG_DIAG][ctx-transfer] t=60 wall_t=60 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=deadline-observed request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert record["points"]["gate2-state-at-ctx-deadline"]["state"] == "ineligible"
    assert record["ctx_deadline_relationship"] == "gate2-ineligible-at-deadline"


def test_cross_host_phase_uses_furthest_milestone_with_missing_credit():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0 wall_t=0 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=timer-start request=42",
            "[DISAGG_DIAG][gate2] t=1 wall_t=10 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 "
            "action=admitted request=42",
            "[DISAGG_DIAG][submit] t=2 wall_t=20 wall_clock=unix "
            "wall_semantics=boundary-sampled host=gen rank=0 request=42",
            "[DISAGG_DIAG][python-transfer] t=3 wall_t=30 wall_clock=unix "
            "wall_semantics=emission host=ctx rank=0 role=ctx "
            "action=first-write-submitted request=42",
            "[DISAGG_DIAG][ctx-transfer] t=40 wall_t=40 wall_clock=unix "
            "wall_semantics=boundary-sampled host=ctx rank=0 "
            "action=deadline-observed request=42",
        ]
    )

    record = analyze_events(events)["cross_host_correlation"]["requests"][0]

    assert record["ctx_deadline_relationship"] == "during-kv-physical-transfer"
    assert record["ctx_deadline_phase_coverage"]["missing_before_furthest"] == [
        "ctx-receiver-credit"
    ]


def test_uncapped_gate_transitions_cover_every_request():
    events = _parse_lines(
        [
            event
            for request_id in range(1, 71)
            for event in (
                f"[DISAGG_DIAG][gen-arrival] t=0.0 rank=0 request={request_id}",
                f"[DISAGG_DIAG][gate1] t=0.1 rank=0 action=fitting "
                f"request={request_id} blocks=1 sequence=1",
                f"[DISAGG_DIAG][gate2] t=0.2 rank=0 action=deferred "
                f"request={request_id} blocks=1 sequence=1",
            )
        ]
    )

    lifecycle = analyze_events(events)["lifecycle"]

    assert lifecycle["request_count"] == 70
    assert lifecycle["interval_coverage"]["gen_arrival_to_first_gate1"]["observed_requests"] == 70
    assert lifecycle["interval_coverage"]["gen_arrival_to_first_gate2"]["observed_requests"] == 70
    assert _TELEMETRY._collect_global_request_blocks(events)["70"] == 1


def test_versioned_candidate_snapshot_reconstructs_unchanged_decisions():
    prefix = ",".join(f"{request}:1" for request in range(1, 65))
    tail = ",".join(f"{request}:1" for request in range(65, 71))
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision-members] t=1 rank=0 role=gen "
            "instance=gen sequence=1 snapshot_version=1 membership=candidate "
            f"chunk_index=1 chunk_count=1 requests={tail}",
            "[DISAGG_DIAG][decision] t=1 rank=0 instance=gen sequence=1 "
            "candidate_snapshot=1 active_snapshot=1 active_blocks=0 active_requests=- "
            "active_requests_omitted=0 "
            f"candidate_requests={prefix} candidate_requests_omitted=6 "
            "admitted=0 admitted_requests=- admitted_requests_omitted=0 "
            f"deferred=70 deferred_requests={prefix} "
            "deferred_requests_omitted=6 budget=1",
            "[DISAGG_DIAG][decision] t=2 rank=0 instance=gen sequence=2 "
            "candidate_snapshot=1 active_snapshot=1 active_blocks=0 active_requests=- "
            "active_requests_omitted=0 "
            f"candidate_requests={prefix} candidate_requests_omitted=6 "
            "admitted=0 admitted_requests=- admitted_requests_omitted=0 "
            f"deferred=70 deferred_requests={prefix} "
            "deferred_requests_omitted=6 budget=1",
        ]
    )

    admissions, _ = _TELEMETRY._collect_admissions(events)

    assert len(admissions) == 2
    assert all(len(admission.candidate_requests) == 70 for admission in admissions)
    assert all(len(admission.deferred_requests) == 70 for admission in admissions)
    assert all(admission.candidate_requests_omitted == 0 for admission in admissions)
    assert admissions[1].deferred_requests[-1] == "70"
    fixed = analyze_events(events)["ranks"]["0"]["fixed_multiplier_counterfactual"]
    assert len(fixed["samples"]) == 2
    assert fixed["next_deferred_required_multiplier"]["count"] == 2
    assert [sample["next_deferred_request"] for sample in fixed["samples"]] == ["1", "1"]


def test_remaining_work_ignores_gate_transition_and_reports_incomplete_tail():
    prefix = ",".join(f"{request}:1" for request in range(1, 65))
    partial_tail = ",".join(f"{request}:1" for request in range(65, 68))
    events = _parse_lines(
        [
            "[DISAGG_DIAG][gate2] t=0.9 rank=0 instance=gen sequence=1 "
            "action=deferred request=99 blocks=1",
            "[DISAGG_DIAG][decision-members] t=1 rank=0 role=gen "
            "instance=gen sequence=1 snapshot_version=1 membership=active "
            f"chunk_index=1 chunk_count=2 requests={partial_tail}",
            "[DISAGG_DIAG][decision] t=1 rank=0 instance=gen sequence=1 "
            "active_snapshot=1 candidate_snapshot=1 "
            f"active_requests={prefix} active_requests_omitted=6 "
            "candidate_requests=- candidate_requests_omitted=0 "
            "admitted=0 deferred=0 budget=1",
        ]
    )

    remaining = _TELEMETRY._analyze_remaining_work_ground_truth(events)

    assert remaining["active_decision_samples"] == 64
    assert remaining["active_request_ids_recovered_from_overflow"] == 0
    assert remaining["active_request_ids_omitted"] == 6
    assert remaining["identity_coverage"]["fraction"] == pytest.approx(64 / 70)


def test_remaining_work_recovers_complete_active_snapshot_overflow():
    prefix = ",".join(f"{request}:1" for request in range(1, 65))
    tail = ",".join(f"{request}:1" for request in range(65, 71))
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision-members] t=1 rank=0 role=gen "
            "instance=gen sequence=1 snapshot_version=1 membership=active "
            f"chunk_index=1 chunk_count=1 requests={tail}",
            "[DISAGG_DIAG][decision] t=1 rank=0 instance=gen sequence=1 "
            "active_snapshot=1 candidate_snapshot=1 "
            f"active_requests={prefix} active_requests_omitted=6 "
            "candidate_requests=- candidate_requests_omitted=0 "
            "admitted=0 deferred=0 budget=1",
            *[
                "[DISAGG_DIAG][python-transfer] t=2 rank=0 role=gen "
                f"instance=gen action=local-ready request={request} outcome=completed"
                for request in range(1, 71)
            ],
        ]
    )

    remaining = _TELEMETRY._analyze_remaining_work_ground_truth(events)

    assert remaining["active_decision_samples"] == 70
    assert remaining["active_request_ids_recovered_from_overflow"] == 6
    assert remaining["active_request_ids_omitted"] == 0
    assert remaining["identity_coverage"]["fraction"] == 1.0
    assert remaining["ready_coverage"] == {
        "eligible": 70,
        "observed": 70,
        "censored": 0,
        "censor_reasons": {},
    }
    assert remaining["residual_ready_s"]["count"] == 70
    assert remaining["residual_ready_s"]["p50"] == pytest.approx(1.0)


def test_membership_snapshots_do_not_cross_instances_or_ranks():
    prefix = ",".join(f"{request}:1" for request in range(1, 65))
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision-members] t=1 rank=0 role=gen "
            "instance=gen-a snapshot_version=1 membership=candidate "
            "chunk_index=1 chunk_count=1 requests=65:1",
            "[DISAGG_DIAG][decision] t=1 rank=0 instance=gen-a sequence=1 "
            "candidate_snapshot=1 active_snapshot=1 active_requests=- "
            "active_requests_omitted=0 "
            f"candidate_requests={prefix} candidate_requests_omitted=1 "
            "admitted=0 deferred=65 budget=1",
            "[DISAGG_DIAG][decision-members] t=1 rank=1 role=gen "
            "instance=gen-b snapshot_version=1 membership=candidate "
            "chunk_index=1 chunk_count=1 requests=165:1",
            "[DISAGG_DIAG][decision] t=1 rank=1 instance=gen-b sequence=1 "
            "candidate_snapshot=1 active_snapshot=1 active_requests=- "
            "active_requests_omitted=0 "
            f"candidate_requests={prefix} candidate_requests_omitted=1 "
            "admitted=0 deferred=65 budget=1",
        ]
    )

    admissions, _ = _TELEMETRY._collect_admissions(events)

    assert {admission.candidate_requests[-1][0] for admission in admissions} == {
        "65",
        "165",
    }


def test_single_log_same_rank_is_namespaced_by_instance():
    events = [
        _parse_source_line(
            "[DISAGG_DIAG][decision] t=1 host=node instance=gen-a rank=0 "
            "sequence=1 active_blocks=0 candidate_requests=1:1 admitted=1 "
            "admitted_requests=1:1 deferred=0 budget=1",
            "combined.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][decision] t=1 host=node instance=gen-b rank=0 "
            "sequence=1 active_blocks=0 candidate_requests=101:1 admitted=1 "
            "admitted_requests=101:1 deferred=0 budget=1",
            "combined.log",
        ),
    ]

    result = analyze_events(events)

    assert sorted(result["ranks"]) == [
        "0::host=node::instance=gen-a::role=gen",
        "0::host=node::instance=gen-b::role=gen",
    ]
    assert result["rank_namespace"] == "source-path::rank[::host::instance::role]"


def test_lifecycle_and_remaining_work_are_namespaced_by_instance():
    events = [
        _parse_source_line(line, "combined.log")
        for line in [
            "[DISAGG_DIAG][gen-arrival] t=0 host=node instance=gen-a rank=0 request=42",
            "[DISAGG_DIAG][gen-activation] t=0.5 host=node instance=gen-a rank=0 request=42",
            "[DISAGG_DIAG][decision] t=1 host=node instance=gen-a rank=0 "
            "sequence=1 active_requests=42:1 active_requests_omitted=0 "
            "candidate_requests=- admitted=0 deferred=0 budget=1",
            "[DISAGG_DIAG][python-transfer] t=2 host=node instance=gen-a "
            "rank=0 role=gen action=local-ready request=42 outcome=completed",
            "[DISAGG_DIAG][gen-arrival] t=10 host=node instance=gen-b rank=0 request=42",
            "[DISAGG_DIAG][gen-activation] t=11 host=node instance=gen-b rank=0 request=42",
            "[DISAGG_DIAG][decision] t=11 host=node instance=gen-b rank=0 "
            "sequence=1 active_requests=42:1 active_requests_omitted=0 "
            "candidate_requests=- admitted=0 deferred=0 budget=1",
            "[DISAGG_DIAG][python-transfer] t=20 host=node instance=gen-b "
            "rank=0 role=gen action=local-ready request=42 outcome=completed",
        ]
    ]

    result = analyze_events(events)
    lifecycle = result["lifecycle"]
    remaining = result["remaining_work_ground_truth"]

    assert lifecycle["clock_domain_request_count"] == 2
    assert {record["instance"] for record in lifecycle["requests"]} == {
        "gen-a",
        "gen-b",
    }
    activation_coverage = lifecycle["interval_coverage"]["gen_arrival_to_activation"]
    assert activation_coverage["observed_samples"] == 2
    assert sorted(
        record["intervals"]["gen_arrival_to_activation"]["duration_s"]
        for record in lifecycle["requests"]
        if record["intervals"]["gen_arrival_to_activation"]["status"] == "observed"
    ) == [0.5, 1.0]
    assert remaining["active_decision_samples"] == 2
    assert {sample["instance"] for sample in remaining["samples"]} == {
        "gen-a",
        "gen-b",
    }
    assert sorted(sample["residual_ready_s"] for sample in remaining["samples"]) == [
        1.0,
        9.0,
    ]


def test_missing_native_instance_aliases_to_single_known_instance():
    events = [
        _parse_source_line(
            "[DISAGG_DIAG][gen-arrival] t=0 host=node instance=gen rank=0 request=1",
            "worker.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][gen-activation] t=0.5 host=node rank=0 request=1",
            "worker.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][submit] t=1 host=node instance=gen rank=0 request=1 blocks=4",
            "worker.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][decision] t=1.5 host=node instance=gen rank=0 "
            "sequence=1 active_requests=1:4 active_requests_omitted=0 "
            "candidate_requests=- admitted=0 deferred=0 budget=4",
            "worker.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][python-transfer] t=2 host=node rank=0 role=gen "
            "action=local-ready request=1 outcome=completed",
            "worker.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][reap] t=3 host=node instance=gen rank=0 "
            "request=1 blocks=4 outcome=completed",
            "worker.log",
        ),
    ]

    result = analyze_events(events)

    assert list(result["ranks"]) == ["0"]
    assert result["ranks"]["0"]["service"]["latency_s"]["p50"] == pytest.approx(1.0)
    lifecycle_interval = result["lifecycle"]["requests"][0]["intervals"][
        "gen_arrival_to_activation"
    ]
    assert lifecycle_interval["status"] == "observed"
    assert lifecycle_interval["duration_s"] == pytest.approx(0.5)
    remaining = result["remaining_work_ground_truth"]
    assert remaining["active_decision_samples"] == 1
    assert remaining["residual_ready_s"]["p50"] == pytest.approx(0.5)


@pytest.mark.parametrize("deadline_action", ["timeout", "timed-out"])
def test_legacy_timeout_action_is_an_observation_not_a_terminal_outcome(
    deadline_action,
):
    events = _parse_lines(
        [
            "[DISAGG_DIAG][ctx-transfer] t=0.0 rank=0 action=timer-start request=42",
            f"[DISAGG_DIAG][ctx-transfer] t=1.0 rank=0 action={deadline_action} request=42",
            "[DISAGG_DIAG][ctx-transfer] t=2.0 rank=0 action=reaped request=42 outcome=completed",
        ]
    )

    result = analyze_events(events)

    assert result["lifecycle"]["deadline_phase_counts"] == {"pre-credit": 1}
    assert _TELEMETRY._collect_unsuccessful_requests(events) == set()


def test_remaining_work_censors_cross_source_endpoints():
    events = [
        _parse_source_line(
            "[DISAGG_DIAG][decision] t=1.0 rank=0 sequence=1 "
            "active_requests=42:8 admitted=0 deferred=1 budget=8",
            "scheduler.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][receiver-transfer] t=1.1 rank=0 "
            "action=completed request=7 context_request=42",
            "receiver.log",
        ),
        _parse_source_line(
            "[DISAGG_DIAG][reap] t=1.2 rank=0 request=42 outcome=completed",
            "receiver.log",
        ),
    ]

    sample = analyze_events(events)["remaining_work_ground_truth"]["samples"][0]

    assert sample["residual_ready_s"] is None
    assert sample["ready_censor_reason"] == "ready_in_other_log_source"
    assert sample["residual_reap_s"] is None
    assert sample["reap_censor_reason"] == "reap_in_other_log_source"


def test_cpp_global_ready_timestamp_is_not_subtracted_from_local_reap_clock():
    events = _parse_lines(
        [
            "[DISAGG_DIAG][decision] t=0.5 rank=0 sequence=1 "
            "active_requests=42:8 admitted=0 deferred=1 budget=8",
            "[DISAGG_DIAG][reap] t=1.0 rank=0 request=42 ready_t=100.0 "
            "ready_time_source=cpp-global ready_to_reap_ms=2.0 outcome=completed",
        ]
    )

    sample = analyze_events(events)["remaining_work_ground_truth"]["samples"][0]

    assert sample["residual_ready_s"] is None
    assert sample["ready_censor_reason"] == "missing_ready"
    assert sample["residual_reap_s"] == pytest.approx(0.5)

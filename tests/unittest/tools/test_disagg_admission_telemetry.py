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


def _parse_lines(lines: list[str]):
    return [event for line in lines if (event := parse_diagnostic_line(line)) is not None]


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
            "[DISAGG_DIAG][reap] t=1.2 rank=0 request=1 blocks=10 ready_t=1.1 outcome=completed",
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
            "[DISAGG_DIAG][reap] t=2.5 rank=0 request=2 blocks=10 ready_t=2.4 outcome=completed",
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
            "[DISAGG_DIAG][receiver-transfer] t=0.8 rank=0 action=failed request=9",
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

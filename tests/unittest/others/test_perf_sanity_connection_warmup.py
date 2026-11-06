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
"""The perf-sanity connection warmup's records must stay out of the d_tb_* time breakdown.

The servers append them to the same perf_metrics JSONLs the breakdown reads, and nothing
gates on d_tb_* metrics, so a leak would upload wrong numbers silently.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

# Loaded by path: importing it through the ``defs`` package pulls in torch.
_SPEC = importlib.util.spec_from_file_location(
    "time_breakdown_metrics",
    Path(__file__).resolve().parents[2] / "integration/defs/perf/time_breakdown_metrics.py",
)
tbm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tbm)


def _record(arrival: float, ctx_queue_s: float) -> str:
    """One disagg-server JSONL line with just the fields this test reads."""
    ctx = {"arrival_time": arrival, "first_scheduled_time": arrival + ctx_queue_s}
    gen = {"last_token_time": arrival + 1.0}
    record = {
        "ctx_perf_metrics": {"perf_metrics": {"timing_metrics": ctx}},
        "gen_perf_metrics": {"perf_metrics": {"timing_metrics": gen}},
    }
    return json.dumps(record) + "\n"


def test_connection_warmup_records_are_skipped(tmp_path: Path) -> None:
    jsonl = tmp_path / "perf_metrics-disagg-node0-1-x.jsonl"
    # An overlapping burst of cold connection-warmup requests. The server has not written
    # the last record yet when DisaggTestCmds starts its line-count snapshot.
    connection_warmup_records = [_record(i * 0.001, ctx_queue_s=5.0) for i in range(32)]
    jsonl.write_text("".join(connection_warmup_records[:-1]))
    late_records = connection_warmup_records[-1:]

    def flush_late_records() -> None:
        with jsonl.open("a") as handle:
            handle.writelines(late_records)
        late_records.clear()

    # The server flushes the last record while the snapshot polls.
    _, after_connection_warmup = tbm.wait_for_perf_metrics_files(
        str(tmp_path),
        expected_requests=len(connection_warmup_records),
        wait_for_expected_requests=True,
        stable_seconds=0,
        sleep=lambda _seconds: flush_late_records(),
    )
    # Does nothing if the snapshot waited for the late record. Otherwise the record lands
    # after the snapshot, and the assertion below catches the leak.
    flush_late_records()

    # The measured client's isolated test request, then its measured burst.
    with jsonl.open("a") as handle:
        handle.write(_record(100.0, ctx_queue_s=1.0))
        handle.writelines(_record(200.0 + i * 0.01, ctx_queue_s=0.1) for i in range(60))

    metrics, _ = tbm.compute_time_breakdown_metrics(
        [str(jsonl)],
        "e2e",
        drop_warmup_request=True,
        leading_lines_to_skip=after_connection_warmup["line_counts"],
    )

    # Only the measured requests count. Without the skip, the burst would also keep
    # drop_warmup_request from finding the test request, which would no longer be the
    # earliest record.
    assert metrics["d_tb_ctx_queue_mean"] == pytest.approx(100.0)

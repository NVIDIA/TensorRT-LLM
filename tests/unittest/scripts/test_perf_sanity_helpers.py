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

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch._inductor")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "integration"))

from defs.perf import test_perf_sanity as perf_sanity  # noqa: E402


def test_run_benchmark_with_log_returns_successful_output(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    command = [sys.executable, "-c", "print('benchmark succeeded')"]

    output = perf_sanity._run_benchmark_with_log(command, {}, str(benchmark_log))

    assert output == "benchmark succeeded\n"
    assert benchmark_log.read_text(encoding="utf-8") == output


def test_run_benchmark_with_log_preserves_failed_output(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('benchmark stdout'); "
            "print('benchmark stderr', file=sys.stderr); "
            "sys.exit(7)"
        ),
    ]

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        perf_sanity._run_benchmark_with_log(command, {}, str(benchmark_log))

    expected_output = "benchmark stdout\nbenchmark stderr\n"
    assert exc_info.value.returncode == 7
    assert exc_info.value.output.decode() == expected_output
    assert benchmark_log.read_text(encoding="utf-8") == expected_output


def test_benchmark_log_is_included_in_report_logs(tmp_path: Path) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.touch()
    aggr_commands = perf_sanity.AggrTestCmds(
        server_cmds=[[]],
        client_cmds={0: [[]]},
        timeout=1,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )
    disagg_commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=0,
        num_gen_servers=0,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )

    assert str(benchmark_log) in aggr_commands.get_server_logs(0)
    assert str(benchmark_log) in disagg_commands.get_server_logs(0)


def test_sentinel_timeout_falls_back_to_current_gen_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark_log = tmp_path / "trtllm-benchmark.0.0.log"
    benchmark_log.write_text("benchmark output", encoding="utf-8")
    warmup_benchmark_log = tmp_path / "trtllm-benchmark.0.1.log"
    warmup_benchmark_log.write_text("warmup benchmark output", encoding="utf-8")
    outputs = ["benchmark output", "warmup benchmark output"]
    pending = [
        # Legacy record without skip_leading_requests: parses the full window.
        {
            "output_index": 0,
            "benchmark_file_path": str(benchmark_log),
            "start_offsets": [10, 20],
        },
        # Warmup-lane record: the parser must skip the leading request.
        {
            "output_index": 1,
            "benchmark_file_path": str(warmup_benchmark_log),
            "start_offsets": [30, 40],
            "skip_leading_requests": 1,
        },
    ]
    commands = perf_sanity.DisaggTestCmds(
        server_cmds=[],
        client_cmds={},
        timeout=1,
        hostname="localhost",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=1,
        num_gen_servers=2,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        perf_sanity.DisaggTestCmds,
        "wait_for_gen_log_sentinels",
        lambda self: False,
    )
    parse_calls: list[tuple[str, int, list[int], int]] = []

    def parse_device_step_time(
        output_dir: str,
        num_gen_servers: int,
        start_offsets: list[int],
        skip_leading_requests: int = 0,
    ) -> float:
        parse_calls.append((output_dir, num_gen_servers, start_offsets, skip_leading_requests))
        return 7.25

    monkeypatch.setattr(
        perf_sanity,
        "parse_gen_worker_device_step_time",
        parse_device_step_time,
    )

    commands._append_gen_worker_device_step_time(pending, outputs)

    assert parse_calls == [
        (str(tmp_path), 2, [10, 20], 0),
        (str(tmp_path), 2, [30, 40], 1),
    ]
    assert outputs == [
        "benchmark output\nAverage Per Iter Device Step Time (ms): 7.25\n",
        "warmup benchmark output\nAverage Per Iter Device Step Time (ms): 7.25\n",
    ]
    assert benchmark_log.read_text(encoding="utf-8").endswith(
        "\nAverage Per Iter Device Step Time (ms): 7.25\n"
    )


def _write_gen_worker_log(path: Path, rows: list[tuple[int, float, int]]) -> None:
    """Write gen_server_{i}.log iter lines from (iter, prev_device_ms, total_requests)."""
    lines = []
    for iter_num, device_ms, total_requests in rows:
        lines.append(
            f"[TRT-LLM] [I] [_torch][RANK 0] iter = {iter_num}, "
            f"num_scheduled_requests = 1, "
            f"currank_total_requests = 0/{total_requests}, "
            f"host_step_time = 5.0ms, prev_device_step_time = {device_ms}ms, "
            "states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, "
            "'num_generation_tokens': 4}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_skip_leading_requests_excludes_warmup_and_boundary_stall(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only measured steady rows enter the mean on a warmup lane.

    The warmup request, the inter-request stall (row +1 after the request
    count flips, because prev_device_step_time lags one row), and the
    settle rows must all stay out.
    """
    rows = [(i, 10.0, 1) for i in range(5, 21)]  # warmup request
    rows += [
        (21, 10.0, 2),  # +0: previous request's last step
        (22, 500.0, 2),  # +1: inter-request stall
        (23, 50.0, 2),  # +2: settling
        (24, 30.0, 2),  # +3, +4: margin rows
        (25, 30.0, 2),
    ]
    rows += [(i, 20.0, 2) for i in range(26, 56)]  # measured steady state
    _write_gen_worker_log(tmp_path / "gen_server_0.log", rows)

    blended = perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1)
    measured_only = perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, skip_leading_requests=1
    )

    assert measured_only == pytest.approx(20.0)
    output = capsys.readouterr().out
    assert "Dropped 5 post-boundary device-step rows" in output
    assert "[10.0, 500.0, 50.0, 30.0, 30.0]" in output
    # The full window folds in the warmup rows and the 500ms stall.
    assert blended != pytest.approx(20.0)


def test_skip_leading_requests_without_boundary_falls_back(tmp_path: Path) -> None:
    """A log without a request boundary falls back to the full window.

    This covers e.g. a warmup request that never reached this worker; the
    metric must degrade to the pre-warmup-aware value instead of None.
    """
    rows = [(i, 10.0, 1) for i in range(5, 25)]
    _write_gen_worker_log(tmp_path / "gen_server_0.log", rows)

    assert perf_sanity.parse_gen_worker_device_step_time(
        str(tmp_path), 1, skip_leading_requests=1
    ) == pytest.approx(10.0)


def test_skip_leading_requests_with_empty_measured_window_returns_none(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A detected boundary must not fall back when no measured row survives."""
    rows = [(i, 10.0, 1) for i in range(5, 15)]
    rows += [(i, 500.0, 2) for i in range(15, 20)]
    _write_gen_worker_log(tmp_path / "gen_server_0.log", rows)

    assert (
        perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1, skip_leading_requests=1)
        is None
    )
    output = capsys.readouterr().out
    assert "all_count=15" in output
    assert "post_rows_seen=5" in output


def test_skip_leading_requests_zero_keeps_full_window(tmp_path: Path) -> None:
    """skip_leading_requests=0 (every non-warmup lane) parses the whole window.

    Behavior must match the pre-warmup-aware parser even when the log
    contains a request boundary.
    """
    rows = [(i, 10.0, 1) for i in range(5, 15)]
    rows += [(i, 30.0, 2) for i in range(15, 25)]
    _write_gen_worker_log(tmp_path / "gen_server_0.log", rows)

    assert perf_sanity.parse_gen_worker_device_step_time(str(tmp_path), 1) == pytest.approx(20.0)

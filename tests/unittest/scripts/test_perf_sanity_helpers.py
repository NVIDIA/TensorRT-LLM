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
    outputs = ["benchmark output"]
    pending = [
        {
            "output_index": 0,
            "benchmark_file_path": str(benchmark_log),
            "start_offsets": [10, 20],
        }
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
    parse_calls: list[tuple[str, int, list[int]]] = []

    def parse_device_step_time(
        output_dir: str,
        num_gen_servers: int,
        start_offsets: list[int],
    ) -> float:
        parse_calls.append((output_dir, num_gen_servers, start_offsets))
        return 7.25

    monkeypatch.setattr(
        perf_sanity,
        "parse_gen_worker_device_step_time",
        parse_device_step_time,
    )

    commands._append_gen_worker_device_step_time(pending, outputs)

    assert parse_calls == [(str(tmp_path), 2, [10, 20])]
    assert outputs == ["benchmark output\nAverage Per Iter Device Step Time (ms): 7.25\n"]
    assert benchmark_log.read_text(encoding="utf-8").endswith(
        "\nAverage Per Iter Device Step Time (ms): 7.25\n"
    )

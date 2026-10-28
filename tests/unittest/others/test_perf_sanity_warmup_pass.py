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
"""The perf tests' warmup pass must send the measured traffic shape and stay unmeasured.

A pass whose command drifts from the measured one warms the wrong prefill shapes, and
one that reuses the measured prompts lets measured requests hit its KV cache blocks;
either way the measured numbers move without any runtime change. The servers also
record the pass's requests in the perf_metrics JSONLs the d_tb_* time breakdown reads,
so those records must be left out, however many files they are split across.
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import pytest

pytestmark = pytest.mark.cpu_only


def _load_perf_module(name: str):
    # Loaded by path: importing it through the ``defs`` package pulls in torch.
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / f"integration/defs/perf/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


warmup_pass = _load_perf_module("warmup_pass")
tbm = _load_perf_module("time_breakdown_metrics")

_BENCHMARK_SERVING = ["python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving"]

# Shaped like ClientConfig._to_default_benchmark_cmd's output for the three modes.
# fmt: off
_AGGR_RANDOM_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "40", "--max-concurrency", "4",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos", "--no-test-input",
    "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
    "--random-input-len", "8192", "--random-output-len", "1024",
    "--random-range-ratio", "0.0", "--backend", "openai", "--trust-remote-code",
]
_E2E_DATASET_TIME_BREAKDOWN_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "43010", "--max-concurrency", "4301",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos",
    "--dataset-name", "trtllm_custom", "--dataset-path", "/datasets/d.json",
    "--backend", "openai", "--trust-remote-code",
    "--save-request-time-breakdown", "/out/time_breakdown",
]
_CTX_ONLY_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "6660", "--max-concurrency", "666",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos",
    "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
    "--random-input-len", "7386", "--random-output-len", "1",
    "--random-range-ratio", "0.0", "--backend", "openai", "--non-streaming",
]
# Shaped like test_perf.py's get_trtllm_serve_client_command for a real-text client
# that saves its detailed results.
_SINGLE_NODE_SERVE_REAL_TEXT_CMD = _BENCHMARK_SERVING + [
    "--model", "/models/m", "--tokenizer", "/models/m",
    "--num-prompts", "64", "--tokenize-on-client", "--no-test-input",
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--save-result", "--save-detailed", "--result-dir", "/engine",
    "--result-filename", "result_8192_1024.json",
    "--dataset-name", "trtllm_custom", "--dataset-path", "/engine/dataset_custom.jsonl",
    "--max-concurrency", "16", "--trust-remote-code",
]
# fmt: on

# The flags the warmup pass may change, and the output files the measured client writes.
_REPLACED_FLAGS_WITH_VALUE = (
    "--num-prompts",
    "--seed",
    "--save-request-time-breakdown",
    "--result-dir",
    "--result-filename",
)
_REPLACED_FLAGS = ("--no-test-input", "--save-result", "--save-detailed")
_MEASURED_OUTPUT_FILES = ("/out/time_breakdown", "/engine", "result_8192_1024.json")


def _value_of(cmd: Sequence[str], flag: str) -> str:
    assert cmd.count(flag) == 1, f"{flag} appears {cmd.count(flag)} times in {cmd}"
    return cmd[cmd.index(flag) + 1]


def _traffic_shape(cmd: Sequence[str]) -> List[str]:
    """The command without the flags the warmup pass may change."""
    shape, args = [], iter(cmd)
    for arg in args:
        if arg in _REPLACED_FLAGS_WITH_VALUE:
            next(args, None)
        elif arg not in _REPLACED_FLAGS:
            shape.append(arg)
    return shape


@pytest.mark.parametrize(
    "measured_cmd",
    [
        _AGGR_RANDOM_CMD,
        _E2E_DATASET_TIME_BREAKDOWN_CMD,
        _CTX_ONLY_CMD,
        _SINGLE_NODE_SERVE_REAL_TEXT_CMD,
    ],
)
def test_warmup_pass_cmd_keeps_the_measured_traffic_shape(measured_cmd: List[str]) -> None:
    concurrency = int(_value_of(measured_cmd, "--max-concurrency"))
    warmup_cmd = warmup_pass.build_warmup_pass_cmd(measured_cmd, concurrency)

    assert _traffic_shape(warmup_cmd) == _traffic_shape(measured_cmd)
    assert _value_of(warmup_cmd, "--num-prompts") == str(concurrency)
    assert warmup_cmd.count("--no-test-input") == 1
    # The pass writes none of the measured client's output files.
    assert not set(warmup_cmd) & {"--save-result", "--save-detailed", *_MEASURED_OUTPUT_FILES}


@pytest.mark.parametrize(
    ("concurrency", "num_requests", "max_batch_size", "expected_wave"),
    [
        (4, 40, 512, 4),
        # Never more than the benchmark sends.
        (16, 8, 512, 8),
        # A client without a concurrency limit sends everything at once; the
        # server's batch size caps what is in flight.
        (-1, 512, 128, 128),
        (-1, 32, 512, 32),
        (-1, 0, 64, 64),
    ],
)
def test_one_wave_is_what_the_client_keeps_in_flight(
    concurrency: int, num_requests: int, max_batch_size: int, expected_wave: int
) -> None:
    assert warmup_pass.one_wave_requests(concurrency, num_requests, max_batch_size) == expected_wave


@pytest.mark.parametrize(
    ("seed_args", "expected_warmup_seed"),
    [
        # benchmark_serving's own default seed is 0.
        ([], "1"),
        (["--seed", "7"], "8"),
    ],
)
def test_warmup_pass_draws_other_prompts_than_the_measured_client(
    seed_args: List[str], expected_warmup_seed: str
) -> None:
    measured_cmd = _AGGR_RANDOM_CMD + seed_args
    warmup_cmd = warmup_pass.build_warmup_pass_cmd(measured_cmd, 4)

    assert _value_of(warmup_cmd, "--seed") == expected_warmup_seed


def test_warmup_pass_cmd_leaves_the_measured_cmd_untouched() -> None:
    measured_cmd = list(_E2E_DATASET_TIME_BREAKDOWN_CMD)
    warmup_pass.build_warmup_pass_cmd(measured_cmd, 4301)

    assert measured_cmd == _E2E_DATASET_TIME_BREAKDOWN_CMD


@pytest.mark.parametrize("num_requests", [0, -1])
def test_warmup_pass_needs_a_request(num_requests: int) -> None:
    with pytest.raises(ValueError):
        warmup_pass.build_warmup_pass_cmd(_AGGR_RANDOM_CMD, num_requests)


@pytest.mark.parametrize(
    ("environ", "enabled"),
    [
        ({}, True),
        ({warmup_pass.WARMUP_PASS_ENV: "1"}, True),
        ({warmup_pass.WARMUP_PASS_ENV: "0"}, False),
        ({warmup_pass.WARMUP_PASS_ENV: " False "}, False),
        ({warmup_pass.WARMUP_PASS_ENV: "OFF"}, False),
        ({warmup_pass.WARMUP_PASS_ENV: "no"}, False),
    ],
)
def test_warmup_pass_env_switch(environ: dict, enabled: bool) -> None:
    assert warmup_pass.warmup_pass_enabled(environ) is enabled


@pytest.mark.parametrize(
    ("benchmark_mode", "applies"),
    [
        (None, True),
        ("", True),
        ("ctx_only", True),
        ("e2e", True),
        ("gen_only", False),
        ("gen_only_no_context", False),
    ],
)
def test_warmup_pass_modes(benchmark_mode, applies: bool) -> None:
    assert warmup_pass.warmup_pass_applies(benchmark_mode) is applies


def _send_warmup_pass(
    run_benchmark: Callable[[List[str]], str], num_requests: int = 4
) -> Tuple[int, List[str]]:
    """Return send_warmup_pass's successful requests and the warnings it logged."""
    warnings: List[str] = []
    succeeded = warmup_pass.send_warmup_pass(
        ["benchmark_serving"],
        num_requests,
        run_benchmark,
        log_path="warmup-pass.log",
        timeout=60.0,
        log_info=lambda _message: None,
        log_warning=warnings.append,
    )
    return succeeded, warnings


def _benchmark_summary(successful: int, failed: int) -> str:
    return f"Successful requests:   {successful}\nFailed requests:   {failed}\n"


@pytest.mark.parametrize(
    ("output", "expected_succeeded", "warns"),
    [
        (_benchmark_summary(4, 0), 4, False),
        # The pass still warmed the shapes it reached; the measured client decides.
        (_benchmark_summary(3, 1), 3, True),
    ],
)
def test_warmup_pass_returns_its_successful_requests(
    output: str, expected_succeeded: int, warns: bool
) -> None:
    sent: List[List[str]] = []

    def run_benchmark(cmd: List[str]) -> str:
        sent.append(cmd)
        return output

    succeeded, warnings = _send_warmup_pass(run_benchmark)

    assert sent == [["benchmark_serving"]]
    assert succeeded == expected_succeeded
    assert bool(warnings) is warns


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, ["benchmark_serving"]),
        subprocess.TimeoutExpired(["benchmark_serving"], 60.0),
    ],
)
def test_warmup_pass_client_error_fails_the_test(error: subprocess.SubprocessError) -> None:
    def run_benchmark(_cmd: List[str]) -> str:
        raise error

    with pytest.raises(RuntimeError) as excinfo:
        _send_warmup_pass(run_benchmark)
    assert excinfo.value.__cause__ is error


def test_warmup_pass_with_every_request_failed_fails_the_test() -> None:
    with pytest.raises(RuntimeError):
        _send_warmup_pass(lambda _cmd: _benchmark_summary(0, 4))


class _FakeClock:
    """Drives wait_for_perf_metrics_files without wall time; calls on_sleep at each poll."""

    def __init__(self, on_sleep: Callable[[], None] = lambda: None):
        self.now = 0.0
        self._on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self._on_sleep()


def _aggregated_server_record(arrival: float, ctx_queue_s: float) -> str:
    """One JSONL line of an aggregated server (a ctx_only case), with the fields read here."""
    timing = {"arrival_time": arrival, "first_scheduled_time": arrival + ctx_queue_s}
    return json.dumps({"perf_metrics": {"timing_metrics": timing}}) + "\n"


def test_warmup_records_split_across_frontend_files_are_skipped(tmp_path: Path) -> None:
    # A ctx_only server with two frontend processes (num_serve_frontends: 2). Each
    # process writes its own file, so each file holds only its share of the warmup
    # records, and the second has not written its last one yet.
    frontend_files = [tmp_path / f"perf_metrics-server-node0-{pid}-x.jsonl" for pid in (11, 12)]
    warmup_records = [_aggregated_server_record(i * 0.001, ctx_queue_s=5.0) for i in range(8)]
    frontend_files[0].write_text("".join(warmup_records[0::2]))
    frontend_files[1].write_text("".join(warmup_records[1:-1:2]))
    late_records = warmup_records[-1:]

    def flush_late_records() -> None:
        with frontend_files[1].open("a") as handle:
            handle.writelines(late_records)
        late_records.clear()

    clock = _FakeClock(on_sleep=flush_late_records)
    _, after_warmup = tbm.wait_for_perf_metrics_files(
        str(tmp_path),
        expected_requests=len(warmup_records),
        wait_for_expected_requests=True,
        stable_seconds=0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert after_warmup["stable"]
    # Does nothing if the wait saw the late record. Otherwise the record lands after the
    # snapshot, and the assertion below catches the leak.
    flush_late_records()

    for frontend_idx, path in enumerate(frontend_files):
        with path.open("a") as handle:
            handle.writelines(
                _aggregated_server_record(100.0 + frontend_idx + i * 0.01, ctx_queue_s=0.1)
                for i in range(30)
            )
    metrics, _ = tbm.compute_time_breakdown_metrics(
        [str(path) for path in frontend_files],
        "ctx_only",
        leading_lines_to_skip=after_warmup["line_counts"],
    )

    assert metrics["d_tb_ctx_queue_mean"] == pytest.approx(100.0)


def test_warmup_census_does_not_add_up_server_roles(tmp_path: Path) -> None:
    # A disagg e2e run: the disagg server, the ctx worker and the gen worker each record
    # every request. Each has written two of the four warmup records so far, which is
    # not all four, however many files hold them.
    first_records = {
        "perf_metrics-disagg-node0-1-x.jsonl": {"ctx_perf_metrics": {}, "gen_perf_metrics": {}},
        "perf_metrics-server-node1-2-x.jsonl": {
            "time_breakdown_metrics": {"ctx_chunk_metrics": []}
        },
        "perf_metrics-server-node2-3-x.jsonl": {"time_breakdown_metrics": {"step_metrics": []}},
    }
    for name, record in first_records.items():
        (tmp_path / name).write_text((json.dumps(record) + "\n") * 2)

    clock = _FakeClock()
    _, info = tbm.wait_for_perf_metrics_files(
        str(tmp_path),
        expected_requests=4,
        wait_for_expected_requests=True,
        stable_seconds=0,
        timeout_seconds=5.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert not info["stable"]

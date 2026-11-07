# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the QA benchmark harness on CPU; substitute only GPU/process boundaries."""

import csv
import subprocess
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

__extra_import_path__ = ["~/tests/integration"]
pytest_plugins = ["pytester"]
pytestmark = pytest.mark.cpu_only

CASE = (
    "perf/test_perf.py::test_perf[llama_v3.1_8b_instruct-bench-pytorch-streaming-"
    "float16-input_output_len:128,128-reqs:4]"
)
BENCH_OUTPUT = """Total Latency (ms): 1000
Total Output Throughput (tokens/sec): 512
Total Token Throughput (tokens/sec): 1024
Per GPU Output Throughput (tps/gpu): 512
Per User Output Throughput [w/ ctx] (tps/user): 128
Request Throughput (req/sec): 4
Average time-to-first-token [TTFT] (ms): 10
Average time-per-output-token [TPOT] (ms): 2
Allocated 1.5 GiB for max tokens in paged KV cache
"""
EXPECTED_METRICS = {
    "INFERENCE_TIME": 1000,
    "TOTAL_OUTPUT_THROUGHPUT": 512,
    "TOTAL_TOKEN_THROUGHPUT": 1024,
    "PER_GPU_OUTPUT_THROUGHPUT": 512,
    "PER_USER_OUTPUT_THROUGHPUT": 128,
    "SEQ_THROUGHPUT": 4,
    "KV_CACHE_SIZE": 1.5,
    "FIRST_TOKEN_TIME": 10,
    "OUTPUT_TOKEN_TIME": 2,
}
EXPECTED_SERVE_METRICS = {
    "KV_CACHE_SIZE": 1.5,
    "SEQ_THROUGHPUT": 4,
    "TOTAL_OUTPUT_THROUGHPUT": 512,
    "TOTAL_TOKEN_THROUGHPUT": 1024,
    "USER_THROUGHPUT": 128,
    "FIRST_TOKEN_TIME": 10,
    "MEDIAN_FIRST_TOKEN_TIME": 9,
    "P99_FIRST_TOKEN_TIME": 12,
    "OUTPUT_TOKEN_TIME": 2,
    "MEDIAN_OUTPUT_TOKEN_TIME": 1,
    "P99_OUTPUT_TOKEN_TIME": 3,
    "INTER_TOKEN_TIME": 4,
    "MEDIAN_INTER_TOKEN_TIME": 3,
    "P99_INTER_TOKEN_TIME": 5,
    "INFERENCE_TIME": 1000,
    "MEDIAN_INFERENCE_TIME": 900,
    "P99_INFERENCE_TIME": 1200,
}


@pytest.fixture
def perf_modules() -> tuple[ModuleType, ModuleType]:
    from defs.perf import test_perf, utils

    return test_perf, utils


def test_dynamic_qa_cases_inherit_fixtures(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, perf_modules
) -> None:
    perf_test, _ = perf_modules
    executed = []
    # The nested session only needs core pytest; CI plugins can start persistent workers.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    def run_case(perf_case_name: str, qa_fixture: str) -> None:
        assert qa_fixture == "fixture visible"
        executed.append(perf_case_name)

    monkeypatch.setattr(perf_test, "run_perf_test", run_case)
    pytester.makepyfile("def test_collection_anchor(): pass")
    pytester.makeconftest("""
import pytest
from defs.perf.test_perf import generate_perf_tests

def pytest_addoption(parser):
    parser.addoption("--test-list")
    parser.addoption("--test-prefix", default="")

def pytest_collection_modifyitems(session, config, items):
    generated = generate_perf_tests(session, config, items)
    items[:] = [item for item in generated if item.name.startswith("test_perf[")]

@pytest.fixture
def perf_case_name(request):
    return request.node.nodeid

@pytest.fixture
def qa_fixture():
    return "fixture visible"
""")
    cases = [CASE, "llm/" + CASE.replace("-bench-", "-serve-")]
    test_list = pytester.path / "qa.txt"
    test_list.write_text("\n".join(cases) + "\n")
    result = pytester.runpytest_inprocess(
        "-q", "--confcutdir=.", "--test-prefix=qa", f"--test-list={test_list}"
    )
    result.assert_outcomes(passed=len(cases))
    assert Counter(executed) == Counter(cases)


@pytest.fixture
def qa_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perf_modules
) -> tuple[MagicMock, SimpleNamespace, SimpleNamespace, Path]:
    perf_test, utils = perf_modules
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(perf_test, "get_device_count", lambda: 1)
    monkeypatch.setattr(perf_test, "get_mpi_world_size", lambda: 1)
    monkeypatch.setattr(perf_test, "llm_models_root", lambda: str(tmp_path / "models"))
    monkeypatch.setattr(perf_test, "get_model_dir", lambda name: str(tmp_path / "model"))
    monkeypatch.setattr(utils, "get_trt_llm_lib_dir", lambda venv: str(tmp_path))
    monkeypatch.setattr(utils, "resolve_node_hostname", lambda: "test-node")
    monkeypatch.setenv("TRTLLM_TOTAL_GPU_COUNT", "1")
    monkeypatch.delenv("TRTLLM_PERF_WARMUP_PASS", raising=False)
    clock = MagicMock()
    clock.get_state_data.return_value = []
    clock.get_gpu_properties.return_value = {}
    clock.get_device_subtype.return_value = "test-gpu"
    venv = SimpleNamespace(get_working_directory=lambda: str(tmp_path), _new_env={})
    writer = SimpleNamespace(_output_formats={"csv", "yaml"})
    return clock, venv, writer, output


@pytest.mark.parametrize(
    "failure", [None, "dataset", "benchmark", "missing_metric", "failed_requests", "write"]
)
def test_qa_benchmark_results_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perf_modules,
    qa_environment,
    failure: str | None,
) -> None:
    perf_test, utils = perf_modules
    clock, venv, writer, output = qa_environment
    commands = []

    def prepare_dataset(command: list[str], **kwargs) -> bytes:
        commands.append(command)
        if failure == "dataset":
            raise subprocess.CalledProcessError(7, command, stderr=b"dataset failed")
        return b'{"input_ids": [1, 2], "output_len": 128}\n'

    def benchmark(command: list[str], **kwargs) -> str:
        commands.append(command)
        if failure == "benchmark":
            raise subprocess.CalledProcessError(9, command, stderr=b"benchmark failed")
        if failure == "missing_metric":
            return BENCH_OUTPUT.replace("Total Latency (ms): 1000\n", "")
        if failure == "failed_requests":
            return BENCH_OUTPUT + "Failed requests: 1\n"
        return BENCH_OUTPUT

    monkeypatch.setattr(
        utils,
        "subprocess",
        SimpleNamespace(
            check_output=prepare_dataset,
            PIPE=subprocess.PIPE,
            CalledProcessError=subprocess.CalledProcessError,
        ),
    )
    monkeypatch.setattr(utils, "_run_command_with_captured_output", benchmark)
    if failure == "write":
        # A file cannot be used as an output directory; real result writing must fail.
        output = tmp_path / "not_a_directory"
        output.write_text("occupied")

    def run() -> None:
        perf_test.run_perf_test(CASE, None, clock, writer, str(output), venv, str(tmp_path))

    if failure in {"dataset", "benchmark"}:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run()
        assert caught.value.returncode == (7 if failure == "dataset" else 9)
    elif failure == "missing_metric":
        with pytest.raises(RuntimeError, match="Cannot find perf result"):
            run()
    elif failure == "failed_requests":
        with pytest.raises(Exception, match="Benchmark has 1 failed requests"):
            run()
    elif failure == "write":
        with pytest.raises(OSError):
            run()
    else:
        run()
        with (output / "perf_script_test_results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        yaml_rows = yaml.safe_load((output / "perf_script_test_results.yaml").read_text())
        metrics = {row["metric_type"]: float(row["perf_metric"]) for row in rows}
        assert EXPECTED_METRICS.items() <= metrics.items()
        assert len(rows) == len(metrics) == len(yaml_rows)
        for row in rows:
            assert row["state"] == "valid"
            assert row["original_test_name"] == CASE
            result = yaml_rows[row["test_name"]]["test_result"]
            assert result["perf_metric"] == float(row["perf_metric"])
            assert result["original_test_name"] == CASE

    assert len(commands) == (1 if failure == "dataset" else 2)
    assert "prepare-dataset" in commands[0]
    if failure != "dataset":
        assert commands[1][0] == "trtllm-bench"
        assert {
            "throughput",
            "--backend=pytorch",
            "--streaming",
            "--num_requests=4",
            "--warmup=4",
        }.issubset(commands[1])


@pytest.mark.parametrize(
    "failure", [None, "startup", "client", "missing_metric", "cleanup_timeout"]
)
def test_qa_serve_results_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perf_modules,
    qa_environment,
    failure: str | None,
) -> None:
    perf_test, utils = perf_modules
    clock, venv, writer, output = qa_environment
    case = CASE.replace("-bench-", "-serve-")
    process = MagicMock()
    if failure == "cleanup_timeout":
        process.wait.side_effect = [subprocess.TimeoutExpired("trtllm-serve", 1), None]
    server_commands = []
    client_commands = []
    # Exercise measured traffic separately from the optional warmup pass.
    monkeypatch.setenv("TRTLLM_PERF_WARMUP_PASS", "0")

    def start_server(commands) -> None:
        server_commands.append(commands.server_cmd)
        commands._server_proc = process
        commands._host, commands._port = "localhost", 12345
        log = output / "server.log"
        log.write_text("Allocated 1.5 GiB for max tokens in paged KV cache\n")
        commands._server_log_path = str(log)
        if failure == "startup":
            raise RuntimeError("server failed to become ready")

    def run_client(command: list[str], **kwargs) -> str:
        client_commands.append(command)
        if failure == "client":
            raise subprocess.CalledProcessError(11, command, stderr=b"client failed")
        if failure == "missing_metric":
            return "no metrics\n"
        return "\n".join(
            [
                "Request throughput (req/s): 4",
                "Output token throughput (tok/s): 512",
                "Total Token throughput (tok/s): 1024",
                "User throughput (tok/s): 128",
                "Mean TTFT (ms): 10",
                "Median TTFT (ms): 9",
                "P99 TTFT (ms): 12",
                "Mean TPOT (ms): 2",
                "Median TPOT (ms): 1",
                "P99 TPOT (ms): 3",
                "Mean ITL (ms): 4",
                "Median ITL (ms): 3",
                "P99 ITL (ms): 5",
                "Mean E2EL (ms): 1000",
                "Median E2EL (ms): 900",
                "P99 E2EL (ms): 1200",
            ]
        )

    monkeypatch.setattr(utils.PerfServeScriptTestCmds, "start_server", start_server)
    monkeypatch.setattr(utils, "_run_command_with_captured_output", run_client)

    def run() -> None:
        perf_test.run_perf_test(case, None, clock, writer, str(output), venv, str(tmp_path))

    if failure == "startup":
        with pytest.raises(RuntimeError, match="server failed to become ready"):
            run()
    elif failure == "client":
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run()
        assert caught.value.returncode == 11
    elif failure == "missing_metric":
        with pytest.raises(RuntimeError, match="Cannot find perf result"):
            run()
    else:
        run()
        with (output / "perf_script_test_results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        metrics = {row["metric_type"]: float(row["perf_metric"]) for row in rows}
        assert EXPECTED_SERVE_METRICS.items() <= metrics.items()
        assert len(rows) == len(metrics)
        assert all(row["state"] == "valid" and row["original_test_name"] == case for row in rows)
    assert len(server_commands) == 1
    assert server_commands[0][0] == "trtllm-serve"
    assert "--config" in server_commands[0]
    assert len(client_commands) == (0 if failure == "startup" else 1)
    if client_commands:
        assert "--port" in client_commands[0] and "12345" in client_commands[0]
    process.terminate.assert_called()
    cleanup_calls = [call[0] for call in process.method_calls]
    first_wait_index = cleanup_calls.index("wait")
    assert cleanup_calls.index("terminate") < first_wait_index
    args, kwargs = process.wait.call_args_list[0]
    timeout = kwargs.get("timeout", args[0] if args else None)
    assert timeout is not None and 0 < timeout < float("inf")
    if failure == "cleanup_timeout":
        kill_index = cleanup_calls.index("kill")
        assert first_wait_index < kill_index
        assert "wait" in cleanup_calls[kill_index + 1 :]

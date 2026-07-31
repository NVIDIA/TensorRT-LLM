#!/usr/bin/env python3
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

import copy
import importlib.util
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from test_common.perf_sanity_agreement import (
    BENCHMARK_STATUS_FAILED,
    BENCHMARK_STATUS_SUCCESS,
    CONTEXT_ACTIVATION_DIGEST_ALGORITHM,
    expected_disagg_lifecycle_roles,
    extract_agreement_arm_log,
    extract_log_after_offset,
    extract_measured_context_lifecycle_log,
    find_backend_event_loop_fatal,
    format_agreement_arm_marker,
    is_paired_agreement_configuration,
    parse_context_activation_sequences,
    read_coordination_text,
    run_polled_command,
    terminate_subprocess,
    validate_completed_benchmark_output,
    validate_context_agreement_mode,
    write_atomic_coordination_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMIT_PATHS = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py",
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
DISAGG_CONFIG_DIR = REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "disaggregated"
AGREEMENT_AB_CONFIGS = (
    DISAGG_CONFIG_DIR
    / (
        "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_"
        "eplb0_mtp1_ccb-NIXL-async-terminal.yaml"
    ),
    DISAGG_CONFIG_DIR
    / (
        "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_"
        "eplb0_mtp1_ccb-NIXL-sync-terminal.yaml"
    ),
)
AGREEMENT_AB_TEST_LIST = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "test_lists"
    / "test-db"
    / "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8.yml"
)
SLURM_LAUNCH_DRAFT = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "disaggregated" / "slurm_launch_draft.sh"
)


def test_extract_agreement_arm_log_scopes_cumulative_log_to_requested_arm():
    async_lines = [f"0: {format_agreement_arm_marker('START', 0, 'CTX_0', '0')}"]
    async_lines.extend(
        (f"PYTHON_ASYNC_CONSENSUS transition=mode_active rank={rank} terminal=1 peer_ready=1")
        for rank in range(4)
    )
    async_lines.extend([f"0: {format_agreement_arm_marker('END', 0, 'CTX_0', '0')}"])
    sync_lines = [f"0: {format_agreement_arm_marker('START', 1, 'CTX_0', '0')}"]
    sync_lines.extend(
        (f"PYTHON_ASYNC_CONSENSUS transition=mode_active rank={rank} terminal=0 peer_ready=1")
        for rank in range(4)
    )
    sync_lines.extend([f"0: {format_agreement_arm_marker('END', 1, 'CTX_0', '0')}"])

    scoped_log = extract_agreement_arm_log(
        "\n".join((*async_lines, *sync_lines)),
        server_idx=1,
        expected_ctx_roles=1,
    )

    assert scoped_log is not None
    assert "terminal=0 peer_ready=1" in scoped_log
    assert "terminal=1 peer_ready=1" not in scoped_log


def test_extract_agreement_arm_log_waits_for_outer_ctx_controller_end():
    log_lines = [format_agreement_arm_marker("START", 0, "CTX_0", "0")]

    assert (
        extract_agreement_arm_log(
            "\n".join(log_lines),
            server_idx=0,
            expected_ctx_roles=1,
        )
        is None
    )


def test_extract_agreement_arm_log_includes_late_evidence_before_next_arm():
    first_arm = [
        format_agreement_arm_marker("START", 0, "CTX_0", "0"),
        format_agreement_arm_marker("END", 0, "CTX_0", "0"),
        "PYTHON_CONTEXT_ACTIVATION_SEQUENCE rank=3 count=256 digest=abc123",
    ]
    second_arm = [
        format_agreement_arm_marker("START", 1, "CTX_0", "0"),
    ]

    scoped_log = extract_agreement_arm_log(
        "\n".join((*first_arm, *second_arm)),
        server_idx=0,
        expected_ctx_roles=1,
    )

    assert scoped_log is not None
    assert "count=256 digest=abc123" in scoped_log


def test_expected_disagg_lifecycle_roles_matches_three_node_topology():
    assert expected_disagg_lifecycle_roles(
        num_ctx_servers=1,
        num_gen_servers=1,
    ) == {
        "CTX_0.0",
        "GEN_0.0",
        "DISAGG_SERVER.0",
        "BENCHMARK.0",
    }


def _context_mode_lines(terminal_mode):
    return [
        (
            "PYTHON_ASYNC_CONSENSUS transition=mode_active "
            f"rank={rank} terminal={terminal_mode} peer_ready=1"
        )
        for rank in range(4)
    ]


def _context_shutdown_lines():
    return [
        f"PYTHON_ASYNC_CONSENSUS transition=shutdown_summary rank={rank} counters={{}}"
        for rank in range(4)
    ]


def _context_activation_lines(count):
    return [
        (f"PYTHON_CONTEXT_ACTIVATION_SEQUENCE rank={rank} count={count} digest=abc{rank}")
        for rank in range(4)
    ]


def test_measured_context_lifecycle_excludes_stale_and_profiling_modes():
    stale_prefix = "stale previous run terminal=0\n"
    current_run = "\n".join(
        (
            *_context_mode_lines(0),
            *_context_shutdown_lines(),
            *_context_activation_lines(0),
            *_context_mode_lines(1),
        )
    )

    cumulative_log = (stale_prefix + current_run).encode()
    scoped_log = extract_log_after_offset(cumulative_log, str(len(stale_prefix.encode())))
    measured_log = extract_measured_context_lifecycle_log(
        scoped_log.decode(),
        set(range(4)),
    )

    assert measured_log is not None
    assert "terminal=0" not in measured_log
    assert "count=0" not in measured_log
    assert validate_context_agreement_mode(measured_log, set(range(4)), (1, 1))


def test_measured_context_lifecycle_waits_for_complete_profiling_shutdown():
    partial_log = "\n".join((*_context_shutdown_lines()[:-1], *_context_mode_lines(1)))

    assert extract_measured_context_lifecycle_log(partial_log, set(range(4))) is None


def test_measured_context_lifecycle_rejects_wrong_final_mode():
    log_text = "\n".join((*_context_shutdown_lines(), *_context_mode_lines(0)))
    measured_log = extract_measured_context_lifecycle_log(log_text, set(range(4)))

    assert measured_log is not None
    with pytest.raises(ValueError, match="does not match"):
        validate_context_agreement_mode(measured_log, set(range(4)), (1, 1))


def test_context_activation_evidence_requires_prompt_digest_algorithm():
    digest = "a" * 64
    valid_line = (
        "PYTHON_CONTEXT_ACTIVATION_SEQUENCE "
        f"rank=3 count=256 digest={digest} "
        f"algorithm={CONTEXT_ACTIVATION_DIGEST_ALGORITHM}"
    )
    stale_id_algorithm_line = valid_line.replace(
        CONTEXT_ACTIVATION_DIGEST_ALGORITHM,
        "sha256-length-prefixed-decimal-v1",
    )

    assert parse_context_activation_sequences(valid_line) == {3: (256, digest)}
    assert parse_context_activation_sequences(stale_id_algorithm_line) == {}


@pytest.mark.parametrize("offset_text", ("invalid", "-1", "999"))
def test_context_log_offset_rejects_invalid_or_out_of_range_values(offset_text):
    with pytest.raises(ValueError):
        extract_log_after_offset(b"short log", offset_text)


def test_paired_coordination_files_are_namespaced_per_script_invocation():
    launch_script = SLURM_LAUNCH_DRAFT.read_text()

    assert "export TRTLLM_PERF_RUN_TOKEN=" in launch_script
    assert 'srunArgs+=("--container-env=TRTLLM_PERF_RUN_TOKEN")' in launch_script


@pytest.mark.parametrize(
    ("modes", "expected"),
    (
        ([("e2e", 1, 1), ("e2e", 0, 1)], True),
        ([("e2e", 1, 1)], True),
        ([], False),
        ([("e2e", 1, 1), ("e2e", None, 1)], False),
        ([("e2e", 1, 1), ("gen_only", 0, 1)], False),
    ),
)
def test_is_paired_agreement_configuration(modes, expected):
    assert is_paired_agreement_configuration(modes) is expected


def test_python_consensus_ab_config_changes_only_terminal_mode():
    configs = []
    effective_envs = []
    arms = []
    for config_path in AGREEMENT_AB_CONFIGS:
        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
        configs.append(config)
        benchmark = config["benchmark"]
        assert len(benchmark["agreement_ab_arms"]) == 1
        arm = benchmark["agreement_ab_arms"][0]
        arms.append(arm)
        base_env = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in config["environment"]["ctx_worker_env_var"].split()
            if "=" in item
        }
        effective_env = dict(base_env)
        effective_env.update(
            item.split("=", 1) for item in arm.get("ctx_worker_env_var", "").split() if "=" in item
        )
        effective_envs.append(effective_env)

    assert [arm["name"] for arm in arms] == ["async-terminal", "sync-terminal"]
    assert [
        env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS"] for env in effective_envs
    ] == ["1", "0"]
    assert [
        env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS"] for env in effective_envs
    ] == ["1", "1"]
    normalized_configs = copy.deepcopy(configs)
    for config in normalized_configs:
        arm = config["benchmark"]["agreement_ab_arms"][0]
        arm["name"] = "<arm>"
        arm["ctx_worker_env_var"] = "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS=<mode>"
        arm["expected_async_terminal_commits"] = "<arm-specific>"
        arm["expected_terminal_mode"] = "<mode>"
    assert normalized_configs[0] == normalized_configs[1]
    for config in configs:
        benchmark = config["benchmark"]
        assert benchmark["multi_round"] == 1
        assert benchmark["concurrency_list"] == "256"
        assert benchmark["client_timeout_seconds"] == 10_800
        assert config["slurm"]["job_time"] == "04:00:00"
        assert config["slurm"]["timeout"] == 12_000
        assert (
            config["worker_config"]["ctx"]["cache_transceiver_config"]["kv_transfer_timeout_ms"]
            == 11_400_000
        )
        assert (
            config["worker_config"]["gen"]["cache_transceiver_config"]["kv_transfer_timeout_ms"]
            == 11_400_000
        )

    test_list = yaml.safe_load(AGREEMENT_AB_TEST_LIST.read_text())
    post_merge_tests = next(
        entry["tests"]
        for entry in test_list["l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8"]
        if entry["condition"]["terms"]["stage"] == "post_merge"
    )
    assert post_merge_tests == [
        (
            "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-"
            "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_"
            "eplb0_mtp1_ccb-NIXL-async-terminal] TIMEOUT (210)"
        ),
        (
            "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-"
            "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_"
            "eplb0_mtp1_ccb-NIXL-sync-terminal] TIMEOUT (210)"
        ),
    ]


@pytest.mark.parametrize(
    ("config_path", "expected_terminal_mode"),
    zip(AGREEMENT_AB_CONFIGS, ("1", "0")),
)
def test_split_agreement_arm_mode_reaches_outer_ctx_worker_env(
    submit_module,
    config_path,
    expected_terminal_mode,
):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    env_config = submit_module.get_env_config(
        config,
        "disaggregated",
        "e2e",
        "",
    )
    ctx_env = dict(
        item.split("=", 1) for item in env_config["ctx_worker_env_var"].split() if "=" in item
    )
    gen_env = dict(
        item.split("=", 1) for item in env_config["gen_worker_env_var"].split() if "=" in item
    )

    assert ctx_env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS"] == (
        expected_terminal_mode
    )
    assert ctx_env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS"] == "1"
    assert "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS" not in gen_env


@pytest.mark.parametrize("config_path", AGREEMENT_AB_CONFIGS)
def test_split_agreement_arm_mode_is_not_applied_to_gen_only(
    submit_module,
    config_path,
):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    env_config = submit_module.get_env_config(
        config,
        "disaggregated",
        "gen_only",
        "",
    )
    ctx_env = dict(
        item.split("=", 1) for item in env_config["ctx_worker_env_var"].split() if "=" in item
    )

    assert "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS" not in ctx_env


def test_completed_benchmark_output_requires_exact_requests_and_metrics():
    output = "\n".join(
        (
            "Successful requests: 256",
            "Failed requests: 0",
            "Output token throughput (tok/s): 454.92",
        )
    )

    validate_completed_benchmark_output(
        output,
        expected_requests=256,
        required_metric_patterns={
            "token_throughput": re.compile(r"Output token throughput \(tok/s\):\s+([\d.]+)")
        },
    )


@pytest.mark.parametrize(
    ("output", "error"),
    (
        ("Failed requests: 0\nmetric: 1", "successful-request"),
        ("Successful requests: 255\nFailed requests: 0\nmetric: 1", "unexpected successful"),
        ("Successful requests: 256\nmetric: 1", "failed-request"),
        ("Successful requests: 256\nFailed requests: 1\nmetric: 1", "1 failed requests"),
        (
            "Successful requests: 256\nFailed requests: 0\n!FAILED REQUESTS!\nmetric: 1",
            "failure markers",
        ),
        ("Successful requests: 256\nFailed requests: 0", "missing required metrics"),
    ),
)
def test_completed_benchmark_output_rejects_incomplete_evidence(output, error):
    with pytest.raises(RuntimeError, match=error):
        validate_completed_benchmark_output(
            output,
            expected_requests=256,
            required_metric_patterns={"metric": re.compile(r"metric:\s+\d+")},
        )


def test_coordination_text_is_published_atomically(tmp_path):
    marker_path = tmp_path / "benchmark_status.txt"

    assert read_coordination_text(str(marker_path)) is None
    write_atomic_coordination_text(str(marker_path), BENCHMARK_STATUS_FAILED)
    assert read_coordination_text(str(marker_path)) == BENCHMARK_STATUS_FAILED
    write_atomic_coordination_text(str(marker_path), BENCHMARK_STATUS_SUCCESS)
    assert read_coordination_text(str(marker_path)) == BENCHMARK_STATUS_SUCCESS
    assert list(tmp_path.iterdir()) == [marker_path]


def test_concurrent_coordination_writers_leave_one_complete_marker(tmp_path):
    marker_path = tmp_path / "paired_abort.txt"
    values = [f"complete-value-{index}" for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda value: write_atomic_coordination_text(
                    str(marker_path),
                    value,
                ),
                values,
            )
        )

    assert read_coordination_text(str(marker_path)) in values
    assert list(tmp_path.iterdir()) == [marker_path]


def test_backend_event_loop_fatal_detection_is_specific(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text("RuntimeError: recoverable request error\nnormal progress\n")

    fatal_line, offset = find_backend_event_loop_fatal(str(log_path))
    assert fatal_line is None

    expected_fatal_line = "RequestError: Event loop terminated with error: bad optional access"
    with open(log_path, "a") as log_file:
        log_file.write(expected_fatal_line[:28])
    fatal_line, offset = find_backend_event_loop_fatal(str(log_path), offset)
    assert fatal_line is None

    with open(log_path, "a") as log_file:
        log_file.write(f"{expected_fatal_line[28:]}\n")
    fatal_line, _ = find_backend_event_loop_fatal(str(log_path), offset)
    assert fatal_line == expected_fatal_line


def test_polled_command_retains_partial_log_on_timeout(tmp_path):
    log_path = tmp_path / "benchmark.log"

    with pytest.raises(TimeoutError, match="timed out"):
        run_polled_command(
            [
                sys.executable,
                "-c",
                "import time; print('partial evidence', flush=True); time.sleep(30)",
            ],
            env=None,
            log_path=str(log_path),
            timeout_seconds=2,
            abort_check=lambda: None,
            poll_interval_seconds=0.01,
            terminate_timeout_seconds=0.5,
        )

    assert "partial evidence" in log_path.read_text()


def test_polled_command_is_interrupted_by_sibling_abort(tmp_path):
    log_path = tmp_path / "benchmark.log"
    start_time = time.monotonic()

    def abort_check():
        if time.monotonic() - start_time > 0.1:
            raise RuntimeError("sibling failed")

    with pytest.raises(RuntimeError, match="sibling failed"):
        run_polled_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=None,
            log_path=str(log_path),
            timeout_seconds=10,
            abort_check=abort_check,
            poll_interval_seconds=0.01,
            terminate_timeout_seconds=0.5,
        )

    assert time.monotonic() - start_time < 2


def test_terminate_subprocess_reports_forced_kill():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "print('ready', flush=True);"
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"

    teardown_error = terminate_subprocess(
        process,
        "test process",
        timeout_seconds=0.1,
    )

    assert teardown_error is not None
    assert "was killed" in str(teardown_error)
    assert process.poll() is not None


@pytest.fixture(params=SUBMIT_PATHS, ids=("ci", "local"))
def submit_module(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(request.param.parent))
    spec = importlib.util.spec_from_file_location(
        f"perf_submit_{request.param.parent.name}", request.param
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("config_name", "expected_queue_size"),
    (
        ("gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL", 1),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL",
            128,
        ),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL",
            512,
        ),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL",
            4096,
        ),
    ),
)
def test_gen_only_queue_size_does_not_exceed_executor_capacity(
    submit_module, config_name, expected_queue_size
):
    with open(DISAGG_CONFIG_DIR / f"{config_name}.yaml") as config_file:
        config = yaml.safe_load(config_file)

    concurrency = config["benchmark"]["concurrency_list"]

    assert (
        submit_module.get_benchmark_request_queue_size(config, concurrency) == expected_queue_size
    )


def test_gen_only_queue_size_preserves_reachable_concurrency(submit_module):
    config = {
        "worker_config": {
            "gen": {
                "max_batch_size": 8,
                "tensor_parallel_size": 4,
                "enable_attention_dp": True,
            }
        }
    }

    assert submit_module.get_benchmark_request_queue_size(config, 16) == 16

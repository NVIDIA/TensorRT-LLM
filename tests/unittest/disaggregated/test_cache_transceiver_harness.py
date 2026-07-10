# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Integration test for the KV cache transceiver harness.

Drives ``run_cache_transceiver_test.py`` on a single node by launching two
independent ``mpirun`` subprocesses (ctx and gen), each forming its own MPI
world of size 2 (TP=2). All processes share the same physical GPU.

Requires: 1 GPU, mpirun, mpi4py, tensorrt_llm.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock, call

import pytest

CTT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "examples",
        "disaggregated",
        "slurm",
        "cache_transceiver_test",
    )
)

DRIVER_SCRIPT = os.path.join(CTT_DIR, "run_cache_transceiver_test.py")
REPORT_SCRIPT = os.path.join(CTT_DIR, "report.py")
_LOG_TAIL_CHARS = 16 * 1024
_PROCESS_TIMEOUT_SECONDS = 120
_TERMINATE_GRACE_SECONDS = 5


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _find_mpirun():
    path = shutil.which("mpirun")
    if path is None:
        pytest.skip("mpirun not found on PATH")
    return path


def _log_tail(log: str) -> str:
    if len(log) <= _LOG_TAIL_CHARS:
        return log
    omitted = len(log) - _LOG_TAIL_CHARS
    return f"... {omitted} earlier characters omitted ...\n{log[-_LOG_TAIL_CHARS:]}"


def _wait_for_processes(processes: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.wait(timeout=remaining)


def _terminate_process_groups(processes: list[subprocess.Popen], group_ids: list[int]) -> None:
    for group_id in group_ids:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    for group_id in group_ids:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass


class TestProcessHelpers:
    def test_wait_for_processes_uses_shared_deadline(self, monkeypatch):
        first = MagicMock()
        second = MagicMock()
        monotonic = MagicMock(side_effect=[100.0, 101.0, 111.0])
        monkeypatch.setattr(time, "monotonic", monotonic)

        _wait_for_processes([first, second])

        first.wait.assert_called_once_with(timeout=119.0)
        second.wait.assert_called_once_with(timeout=109.0)

    def test_terminate_process_groups_bounds_post_kill_wait(self, monkeypatch):
        proc = MagicMock(pid=1234)
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("mpirun", 4.0),
            subprocess.TimeoutExpired("mpirun", 4.0),
        ]
        killpg = MagicMock()
        monotonic = MagicMock(side_effect=[100.0, 101.0, 200.0, 201.0])
        monkeypatch.setattr(os, "killpg", killpg)
        monkeypatch.setattr(time, "monotonic", monotonic)

        _terminate_process_groups([proc], [proc.pid])

        assert proc.wait.call_args_list == [call(timeout=4.0), call(timeout=4.0)]
        assert killpg.call_args_list == [
            call(proc.pid, signal.SIGTERM),
            call(proc.pid, signal.SIGKILL),
        ]

    def test_terminate_process_groups_signals_group_after_leader_exit(self, monkeypatch):
        proc = MagicMock(pid=1234)
        proc.poll.return_value = 1
        killpg = MagicMock()
        monkeypatch.setattr(os, "killpg", killpg)

        _terminate_process_groups([proc], [proc.pid])

        assert killpg.call_args_list == [
            call(proc.pid, signal.SIGTERM),
            call(proc.pid, signal.SIGKILL),
        ]
        proc.wait.assert_not_called()


def _build_config(work_dir: str) -> dict:
    return {
        "hardware": {"gpus_per_node": 2},
        "environment": {
            "container_image": "",
            "work_dir": work_dir,
        },
        "test_matrix": {
            "combinations": [
                {"backend": "NIXL", "runtime": "PYTHON"},
            ],
            "cache_manager_versions": ["V2"],
            "request_lengths": [100],
            "num_requests_per_length": 8,
            "warmup_requests": 1,
        },
        "kv_cache": {
            "num_layers": 4,
            "num_kv_heads": 2,
            "head_dim": 128,
            "tokens_per_block": 8,
            "dtype": "HALF",
            "max_tokens_in_buffer": 256,
        },
        "parallel": {"ctx_tp": 2, "ctx_pp": 1, "gen_tp": 2, "gen_pp": 1},
        "ucx_env_sweep": [
            {"name": "default", "env": {"UCX_TLS": "all"}},
        ],
        "run": {
            "timeout_per_cell_s": 30,
            "max_sweep_s": 60,
        },
    }


@pytest.mark.timeout(180)
def test_single_node_transfer(tmp_path):
    """Launch ctx and gen via mpirun on a single node, verify transfer passes."""
    pytest.importorskip("mpi4py")
    mpirun = _find_mpirun()

    work_dir = str(tmp_path / "work")
    cfg = _build_config(work_dir)
    config_path = str(tmp_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    zmq_port = _find_free_port()

    env = os.environ.copy()
    env.update(
        {
            "CTT_CONFIG": config_path,
            "CTX_NODE": "127.0.0.1",
            "ZMQ_PORT": str(zmq_port),
            "CTT_SWEEP": "0",
            "CTT_SWEEP_NAME": "default",
            "CUDA_VISIBLE_DEVICES": "0",
            # Emit protocol-selection tables while transfers are active so the
            # report can deterministically identify the CUDA KV-data transport.
            "UCX_PROTO_INFO": "y",
        }
    )
    sweep_env = cfg["ucx_env_sweep"][0].get("env") or {}
    env.update({k: str(v) for k, v in sweep_env.items()})

    mpi_args = [
        mpirun,
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "2",
        sys.executable,
        DRIVER_SCRIPT,
    ]

    # Write directly to the filenames report.py consumes. Full UCX protocol
    # tables are verbose, so files also prevent either child from blocking on a
    # full stdout pipe while the other child is being drained.
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ctx_log_path = os.path.join(log_dir, "sweep0_ctx_rank0.log")
    gen_log_path = os.path.join(log_dir, "sweep0_gen_rank0.log")
    timeout_error = None
    processes = []
    process_group_ids = []
    processes_succeeded = False
    with open(ctx_log_path, "wb") as ctx_log_file, open(gen_log_path, "wb") as gen_log_file:
        try:
            ctx_proc = subprocess.Popen(
                mpi_args + ["--role", "ctx"],
                env=env,
                stdout=ctx_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(ctx_proc)
            process_group_ids.append(ctx_proc.pid)
            gen_proc = subprocess.Popen(
                mpi_args + ["--role", "gen"],
                env=env,
                stdout=gen_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(gen_proc)
            process_group_ids.append(gen_proc.pid)
            _wait_for_processes(processes)
            processes_succeeded = all(proc.returncode == 0 for proc in processes)
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
        finally:
            if not processes_succeeded:
                _terminate_process_groups(processes, process_group_ids)

    with open(ctx_log_path, errors="replace") as f:
        ctx_log = f.read()
    with open(gen_log_path, errors="replace") as f:
        gen_log = f.read()

    if timeout_error is not None:
        pytest.fail(
            f"mpirun timed out: {timeout_error}\n"
            f"ctx log tail:\n{_log_tail(ctx_log)}\n"
            f"gen log tail:\n{_log_tail(gen_log)}"
        )

    if ctx_proc.returncode != 0:
        pytest.fail(f"ctx mpirun failed (rc={ctx_proc.returncode}):\n{_log_tail(ctx_log)}")
    if gen_proc.returncode != 0:
        pytest.fail(f"gen mpirun failed (rc={gen_proc.returncode}):\n{_log_tail(gen_log)}")

    # Parse gen status JSONL and verify all entries are PASS.
    status_path = os.path.join(work_dir, "status", "sweep0_gen.jsonl")
    assert os.path.exists(status_path), (
        f"gen status file not found at {status_path}\ngen log tail:\n{_log_tail(gen_log)}"
    )

    with open(status_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0, "No status records found"
    for rec in records:
        assert rec["status"] == "PASS", (
            f"Expected PASS, got {rec['status']} "
            f"(combination_idx={rec.get('combination_idx')}, "
            f"reqlen_idx={rec.get('reqlen_idx')}, "
            f"reason={rec.get('reason', '')})\n"
            f"gen log tail:\n{_log_tail(gen_log)}"
        )

    # Verify report aggregation produces valid results.
    results_path = os.path.join(work_dir, "results.json")
    agg_result = subprocess.run(
        [
            sys.executable,
            REPORT_SCRIPT,
            config_path,
            "--aggregate",
            "--require-kv-transport",
            "--out",
            results_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert agg_result.returncode == 0, (
        f"report.py --aggregate failed (rc={agg_result.returncode}):\n"
        f"stdout: {agg_result.stdout}\nstderr: {agg_result.stderr}"
    )

    assert os.path.exists(results_path), "results.json was not created"
    with open(results_path) as f:
        results = json.load(f)

    assert "by_combination" in results
    assert len(results["by_combination"]) > 0
    for combo in results["by_combination"]:
        assert "combination" in combo
        assert "sweeps" in combo
        for sweep in combo["sweeps"]:
            assert sweep["status"] == "PASS", (
                f"Aggregated status not PASS for {combo['combination']} "
                f"sweep={sweep['sweep']}: {sweep['status']} "
                f"{sweep.get('error_detail', '')}"
            )
            assert sweep["selected_transport"], (
                f"selected_transport is empty for {combo['combination']} "
                f"sweep={sweep['sweep']} — UCX did not emit a parseable protocol table\n"
                f"ctx log tail:\n{_log_tail(ctx_log)}\n"
                f"gen log tail:\n{_log_tail(gen_log)}"
            )

    best_path = os.path.splitext(results_path)[0] + ".best.json"
    assert os.path.exists(best_path), "results.best.json was not created"

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
import subprocess
import sys

import pytest

pytest.importorskip("mpi4py")

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
            "UCX_PROTO_INFO": "used",
        }
    )

    mpi_args = [
        mpirun,
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "2",
        sys.executable,
        DRIVER_SCRIPT,
    ]

    ctx_proc = subprocess.Popen(
        mpi_args + ["--role", "ctx"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    gen_proc = subprocess.Popen(
        mpi_args + ["--role", "gen"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    ctx_out, _ = ctx_proc.communicate(timeout=120)
    gen_out, _ = gen_proc.communicate(timeout=120)

    ctx_log = ctx_out.decode(errors="replace")
    gen_log = gen_out.decode(errors="replace")

    # Persist logs to work_dir matching the glob pattern that report.py expects
    # (sweep<N>_<role>_rank<R>.log). mpirun merges all rank outputs into one
    # stream, so we write the combined output as rank0; report still picks it up.
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    for role, content in [("ctx", ctx_log), ("gen", gen_log)]:
        with open(os.path.join(log_dir, f"sweep0_{role}_rank0.log"), "w") as f:
            f.write(content)

    if ctx_proc.returncode != 0:
        pytest.fail(f"ctx mpirun failed (rc={ctx_proc.returncode}):\n{ctx_log}")
    if gen_proc.returncode != 0:
        pytest.fail(f"gen mpirun failed (rc={gen_proc.returncode}):\n{gen_log}")

    # Parse gen status JSONL and verify all entries are PASS.
    status_path = os.path.join(work_dir, "status", "sweep0_gen.jsonl")
    assert os.path.exists(status_path), (
        f"gen status file not found at {status_path}\ngen log:\n{gen_log}"
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
            f"gen log:\n{gen_log}"
        )

    # Verify report aggregation produces valid results.
    results_path = os.path.join(work_dir, "results.json")
    agg_result = subprocess.run(
        [sys.executable, REPORT_SCRIPT, config_path, "--aggregate", "--out", results_path],
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
                f"sweep={sweep['sweep']} — UCX_PROTO_INFO may not be set or "
                f"log filenames may not match report.py's glob pattern"
            )

    best_path = os.path.splitext(results_path)[0] + ".best.json"
    assert os.path.exists(best_path), "results.best.json was not created"

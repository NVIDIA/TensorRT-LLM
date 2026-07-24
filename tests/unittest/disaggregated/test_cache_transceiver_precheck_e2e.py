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
"""End-to-end test for the disagg perf-sanity cache-transceiver PRECHECK driver.

Drives ``run_precheck.py`` (tests/scripts/perf-sanity/cache_transceiver_precheck)
exactly like the SLURM gate does — one ``mpirun`` world per ctx/gen server
instance, a shared ``--work-dir`` for rendezvous/status, the disagg yaml as
the single config source — but on ONE node with every process sharing one
physical GPU. This is the CI net for the precheck's NON-network failure
modes: internal TRT-LLM API drift, KV pool construction, transceiver
setup/transfer/verification, multi-instance pairing, and failure verdicts.

Requires: 1 GPU, mpirun, mpi4py, tensorrt_llm. Modeled on
tests/unittest/disaggregated/test_cache_transceiver_harness.py (same
single-node NIXL/PYTHON transceiver combination that stage already runs).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml

# Same single-GPU environment pinning as kv_transfer_harness.py:
#   - One NIXL worker thread per agent: the default (8) causes heavy
#     contention when many agents share a single GPU.
#   - ``^ib,gdr_copy`` disables InfiniBand and GDR copy, which are
#     unavailable (and flaky) on single-node loopback.
# Set here (not just in the child env) so any in-process tensorrt_llm
# import sees them too; the mpirun children inherit via os.environ.copy().
os.environ["TRTLLM_NIXL_NUM_THREADS"] = "1"
os.environ["UCX_TLS"] = "^ib,gdr_copy"

PRECHECK_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "tests",
        "scripts",
        "perf-sanity",
        "cache_transceiver_precheck",
    )
)
DRIVER_SCRIPT = os.path.join(PRECHECK_DIR, "run_precheck.py")
_LOG_TAIL_CHARS = 16 * 1024
_PROCESS_TIMEOUT_SECONDS = 240
_TERMINATE_GRACE_SECONDS = 5

TINY_MODEL_CONFIG = {
    # Llama-shaped so model_kv_shape() exercises the real config.json path
    # (2 layers x 1 kv head x 64 dim: a few hundred KB of KV per process).
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_size": 256,
    "vocab_size": 32000,
}


def _find_mpirun():
    path = shutil.which("mpirun")
    if path is None:
        pytest.skip("mpirun not found on PATH")
    return path


def _log_tail(path):
    try:
        with open(path, errors="replace") as f:
            log = f.read()
    except OSError:
        return f"(no log at {path})"
    if len(log) <= _LOG_TAIL_CHARS:
        return log
    return (
        f"... {len(log) - _LOG_TAIL_CHARS} earlier characters omitted ...\n{log[-_LOG_TAIL_CHARS:]}"
    )


def _terminate_process_groups(processes):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # ALWAYS signal the group, even when the mpirun leader already
        # exited: its ranks stay in the group and would otherwise leak GPU
        # memory into the following tests (same contract as the harness's
        # test_terminate_process_groups_signals_group_after_leader_exit).
        for proc in processes:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass


def _disagg_yaml(num_ctx, num_gen, ctx_tp, gen_tp, request_lengths=(64,)):
    """Minimal disagg perf-sanity yaml shaped like the checked-in configs."""
    tokens_per_block = 32

    def side(tp):
        return {
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1,
            "kv_cache_config": {
                "dtype": "bf16",
                "tokens_per_block": tokens_per_block,
                # Same combination the merged single-node harness test runs:
                # NIXL + PYTHON transceiver requires the V2 manager.
                "use_kv_cache_manager_v2": True,
            },
            "cache_transceiver_config": {
                "backend": "NIXL",
                "transceiver_runtime": "PYTHON",
                "max_tokens_in_buffer": 512,
            },
        }

    return {
        "metadata": {"model_dir_name": "tiny-llama"},
        "benchmark": {"mode": "e2e", "input_length": 64, "output_length": 8},
        "hardware": {
            "gpus_per_node": 1,
            "num_ctx_servers": num_ctx,
            "num_gen_servers": num_gen,
        },
        "worker_config": {"ctx": side(ctx_tp), "gen": side(gen_tp)},
        "cache_transceiver_precheck": {
            "request_lengths": list(request_lengths),
            "num_requests": 1,
            "warmup_requests": 1,
            "wave_timeout_s": 60,
            "wireup_timeout_s": 30,
            "rendezvous_timeout_s": 90,
        },
    }


def _write_inputs(tmp_path, cfg, name="precheck"):
    models_root = tmp_path / "models"
    model_dir = models_root / "tiny-llama"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(TINY_MODEL_CONFIG))
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return str(config_path), str(models_root)


def _launch_instances(tmp_path, jobs, models_root):
    """One mpirun per (role, idx, world, config) job; returns [(name, proc, log)]."""
    mpirun = _find_mpirun()
    work_dir = str(tmp_path / "work")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()  # carries the module-level UCX/NIXL pinning
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    # run_precheck resolves metadata.model_dir_name under LLM_MODELS_ROOT.
    env["LLM_MODELS_ROOT"] = models_root

    launched = []
    try:
        for role, idx, world, config_path in jobs:
            _launch_one(launched, mpirun, role, idx, world, config_path, work_dir, log_dir, env)
    except BaseException:
        # A failed Popen mid-list must not leak the instances already
        # started: they never reach _wait_all's cleanup.
        _terminate_process_groups([p for _, p, _ in launched])
        raise
    return work_dir, launched


def _launch_one(launched, mpirun, role, idx, world, config_path, work_dir, log_dir, env):
    name = f"{role}_{idx}"
    log_path = str(log_dir / f"{name}.log")
    cmd = [
        mpirun,
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        str(world),
        sys.executable,
        DRIVER_SCRIPT,
        "--role",
        role,
        "--server-idx",
        str(idx),
        "--config",
        config_path,
        "--work-dir",
        work_dir,
    ]
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launched.append((name, proc, log_path))


def _wait_all(launched):
    procs = [p for _, p, _ in launched]
    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    try:
        for name, proc, log_path in launched:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                tails = "\n".join(f"----- {n} -----\n{_log_tail(lp)}" for n, _, lp in launched)
                pytest.fail(f"precheck instance {name} timed out\n{tails}")
    finally:
        _terminate_process_groups(procs)


def _read_status(work_dir, name):
    text_path = os.path.join(work_dir, "status", f"{name}.status")
    json_path = os.path.join(work_dir, "status", f"{name}.json")
    with open(text_path) as f:
        text = f.read()
    with open(json_path) as f:
        doc = json.load(f)
    return text, doc


def _assert_all_passed(work_dir, launched):
    failures = []
    for name, proc, log_path in launched:
        if proc.returncode != 0:
            failures.append(f"{name} exited rc={proc.returncode}:\n{_log_tail(log_path)}")
    if failures:
        pytest.fail("\n".join(failures))
    for name, _, log_path in launched:
        text, doc = _read_status(work_dir, name)
        assert text.startswith(f"PASS {name}"), (
            f"{name} status not PASS: {text}\n{_log_tail(log_path)}"
        )
        assert doc["overall"] == "PASS"
        assert doc["transceiver_runtime"] == "PYTHON"
        assert doc["kv_cache_manager"] == "V2"


def _jobs(cfg, config_path):
    ctx_world = cfg["worker_config"]["ctx"]["tensor_parallel_size"]
    gen_world = cfg["worker_config"]["gen"]["tensor_parallel_size"]
    jobs = []
    for i in range(cfg["hardware"]["num_ctx_servers"]):
        jobs.append(("ctx", i, ctx_world, config_path))
    for i in range(cfg["hardware"]["num_gen_servers"]):
        jobs.append(("gen", i, gen_world, config_path))
    return jobs


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "num_ctx,num_gen,ctx_tp,gen_tp",
    [
        pytest.param(1, 1, 1, 1, id="symmetric_1x1"),
        pytest.param(2, 2, 1, 1, id="multi_instance_2x2"),
        pytest.param(1, 1, 2, 1, id="asymmetric_tp2_to_tp1"),
    ],
)
def test_precheck_passes(tmp_path, num_ctx, num_gen, ctx_tp, gen_tp):
    """All (ctx, gen) pairs transfer + verify on one GPU; every verdict PASS."""
    pytest.importorskip("mpi4py")
    cfg = _disagg_yaml(num_ctx, num_gen, ctx_tp, gen_tp)
    config_path, models_root = _write_inputs(tmp_path, cfg)
    work_dir, launched = _launch_instances(tmp_path, _jobs(cfg, config_path), models_root)
    _wait_all(launched)
    _assert_all_passed(work_dir, launched)
    # Every gen instance must have exercised every ctx peer.
    for gj in range(num_gen):
        _, doc = _read_status(work_dir, f"gen_{gj}")
        peers = {c["peer"] for c in doc["cases"] if c["status"] == "PASS"}
        assert peers == {f"ctx_{ci}" for ci in range(num_ctx)}


@pytest.mark.timeout(300)
def test_precheck_fails_fast_on_fingerprint_mismatch(tmp_path):
    """Mismatched ctx/gen yamls must produce FAIL verdicts, not a hang.

    This is the failure-attribution path the SLURM gate consumes: non-zero
    exit codes plus .status files whose first line names the root cause.
    """
    pytest.importorskip("mpi4py")
    ctx_cfg = _disagg_yaml(1, 1, 1, 1, request_lengths=(64,))
    gen_cfg = _disagg_yaml(1, 1, 1, 1, request_lengths=(32, 64))  # different fingerprint
    ctx_path, models_root = _write_inputs(tmp_path, ctx_cfg, name="ctx")
    gen_path, _ = _write_inputs(tmp_path, gen_cfg, name="gen")
    jobs = [("ctx", 0, 1, ctx_path), ("gen", 0, 1, gen_path)]
    work_dir, launched = _launch_instances(tmp_path, jobs, models_root)
    _wait_all(launched)

    for name, proc, log_path in launched:
        assert proc.returncode != 0, (
            f"{name} unexpectedly passed with mismatched yamls\n{_log_tail(log_path)}"
        )
        text, doc = _read_status(work_dir, name)
        assert text.startswith(f"FAIL {name}"), f"{name} status: {text}"
        assert doc["overall"] == "FAIL"
    # The gen driver saw the ctx abort its handshake: the reason must name it.
    text, _ = _read_status(work_dir, "gen_0")
    assert "fingerprint" in text or "abort" in text.lower()

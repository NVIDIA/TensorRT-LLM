# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node, model-free regressions for the remote MPI worker lifecycle."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

_MPI_RUNTIME_AVAILABLE = (
    sys.platform == "linux"
    and ENABLE_MULTI_DEVICE
    and shutil.which("mpirun") is not None
    and importlib.util.find_spec("mpi4py") is not None
)


def _live_processes(identities: dict[int, float]) -> list[psutil.Process]:
    result = []
    for pid, created in identities.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                result.append(process)
        except psutil.NoSuchProcess:
            pass
    return result


def _observe_processes(root: psutil.Process, directory: Path, identities: dict[int, float]) -> None:
    try:
        for child in root.children(recursive=True):
            try:
                identities[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    for path in directory.glob("identity-*.json"):
        identity = json.loads(path.read_text())
        identities[identity["pid"]] = identity["created"]


def _cleanup_processes(identities: dict[int, float]) -> None:
    processes = _live_processes(identities)
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=2)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=2)


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not _MPI_RUNTIME_AVAILABLE, reason="Linux and a multi-device MPI runtime required"
)
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize(
    "scenario",
    [
        "mixed_failure",
        "mixed_collective",
        "all_hang",
        "all_return",
        "no_submission",
        "reuse",
        "sync_recovery",
        "async_drain",
    ],
)
def test_remote_mpi_worker_lifecycle(scenario: str, ranks: int, tmp_path: Path) -> None:
    """Verify bounded failure and clean reuse without leaving owned processes alive."""
    launcher = shutil.which("trtllm-llmapi-launch")
    assert launcher is not None, "The matching trtllm-llmapi-launch must be installed"
    test_directory = Path(__file__).resolve().parent
    # This is a separate local MPI job, including when pytest itself runs in
    # a Slurm step. Inherited rank identities must not override its own ranks.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "OMPI_COMM_", "PMIX_", "PMI_"))
        and key
        not in (
            "TLLM_SPAWN_PROXY_PROCESS",
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR",
            "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY",
            "tllm_mpi_size",
        )
    }
    env.update(
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        PRTE_ALLOW_RUN_AS_ROOT="1",
        PRTE_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        TLLM_MGMN_SHUTDOWN_GRACE_SECONDS="5",
        TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT="90",
        TLLM_LLMAPI_ZMQ_DEBUG="1",
        TLLM_LOG_LEVEL="info",
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(test_directory)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--host",
        "localhost",
        "--oversubscribe",
        "--bind-to",
        "none",
    ]
    for variable in (
        "OMPI_ALLOW_RUN_AS_ROOT",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
        "PRTE_ALLOW_RUN_AS_ROOT",
        "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM",
        "TLLM_MGMN_SHUTDOWN_GRACE_SECONDS",
        "TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT",
        "TLLM_LLMAPI_ZMQ_DEBUG",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
    ):
        command.extend(["-x", variable])
    command.extend(
        [
            "-np",
            str(ranks),
            launcher,
            sys.executable,
            "-m",
            "_run_mpi_lifecycle_task",
            scenario,
            str(tmp_path),
            "--ranks",
            str(ranks),
        ]
    )

    identities: dict[int, float] = {}
    log_path = tmp_path / "launcher.log"
    started = time.monotonic()
    timed_out = False
    survivors: list[int] = []
    with log_path.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=test_directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                _observe_processes(root, tmp_path, identities)
                if time.monotonic() - started >= 180:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(0.05)
            process.wait(timeout=10)
            exited = time.monotonic()
            _observe_processes(root, tmp_path, identities)
            cleanup_deadline = time.monotonic() + 3
            while _live_processes(identities) and time.monotonic() < cleanup_deadline:
                time.sleep(0.05)
            survivors = [child.pid for child in _live_processes(identities)]
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            _cleanup_processes(identities)

    evidence = {
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "survivors_before_cleanup": survivors,
        "process_identities": identities,
    }
    (tmp_path / "result.json").write_text(json.dumps(evidence, indent=2))
    log = log_path.read_text()
    print(f"MPI lifecycle evidence: {tmp_path}\n{json.dumps(evidence, indent=2)}\n{log}")
    assert not timed_out, f"The outer safety timeout ended the MPI run:\n{log}"
    assert not survivors, f"Owned processes survived MPI launcher exit: {survivors}\n{log}"
    assert not _live_processes(identities), "Failed to clean up the test's own processes"
    assert (tmp_path / "engine-started.json").exists(), log
    assert "ZMQ thread safety violation" not in log, log

    if scenario in ("mixed_failure", "mixed_collective", "all_hang"):
        assert process.returncode != 0, log
        assert (tmp_path / "workers-started.json").exists(), log
        expected_marker = "engine-exiting" if scenario == "all_hang" else "error-observed"
        marker = tmp_path / f"{expected_marker}.json"
        assert marker.exists(), log
        triggered = json.loads(marker.read_text())["monotonic"]
        # The stop helper imports TensorRT-LLM afresh after engine exit. The
        # mixed-failure path needs no new interpreter and has a tighter bound.
        teardown_budget = 120 if scenario == "all_hang" else 20
        assert exited - triggered < teardown_budget, f"Teardown exceeded its bounded budget:\n{log}"
    elif scenario == "no_submission":
        assert process.returncode == 3, log
    elif scenario == "async_drain":
        assert process.returncode == 0, log
        assert (tmp_path / "engine-exiting.json").exists(), log
    else:
        assert process.returncode == 0, log
        assert (tmp_path / "engine-completed.json").exists(), log

    batches = {"all_return": 1, "reuse": 8, "sync_recovery": 2, "async_drain": 3}.get(scenario, 1)
    if scenario != "no_submission":
        for rank in range(ranks):
            events = [
                json.loads(line)
                for line in (tmp_path / f"events-{rank}.jsonl").read_text().splitlines()
            ]
            assert [event["batch"] for event in events] == list(range(batches)), events

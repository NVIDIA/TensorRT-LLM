# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import threading
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Literal

cur_dir = os.path.dirname(os.path.abspath(__file__))

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_IDENTITY_TIMEOUT,
                                             MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             _identity_barrier_timeout,
                                             split_mpi_env)

# isort: off
sys.path.append(os.path.join(cur_dir, '..'))
# isort: on


def task0():
    if MPINodeState.state is None:
        MPINodeState.state = 0
    MPINodeState.state += 1
    return MPINodeState.state


@pytest.fixture(autouse=True)
def _enable_mpi(monkeypatch):
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_session_basic():
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    n_workers = 4
    executor = MpiPoolSession(n_workers)
    results = executor.submit_sync(task0)
    assert results == [1, 1, 1, 1], results

    results = executor.submit_sync(task0)
    assert results == [2, 2, 2, 2], results


def simple_task(x):
    print(f"** simple_task {x} returns {x * 2}\n", "green")
    res = x * 2
    print(f"simple_task {x} returns {res}")


def run_client(server_addr, values_to_process, hmac_key: bytes):
    """Function to run in a separate process that creates a client and submits tasks"""
    try:
        client = RemoteMpiCommSessionClient(server_addr, hmac_key=hmac_key)

        for val in values_to_process:
            print(f"Client Submitting task for value {val}")
            client.submit(simple_task, val)

        client.shutdown()

    except Exception as e:
        return f"Error in client: {str(e)}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("task_type", [
    "submit", "submit_sync", "flashinfer_workspace",
    "flashinfer_temporary_cleanup"
])
def test_remote_mpi_session(
    task_type: Literal["submit", "submit_sync", "flashinfer_workspace",
                       "flashinfer_temporary_cleanup"],
    tmp_path: Path,
) -> None:
    """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, "_test_remote_mpi_session.sh")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = ["bash", test_file, task_type]
    print(' '.join(command))
    env = os.environ.copy()
    if task_type == "flashinfer_workspace":
        env["HOME"] = str(tmp_path)
        env.pop("FLASHINFER_WORKSPACE_BASE", None)
        env.pop("FLASHINFER_CUBIN_DIR", None)
        env.pop("TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS", None)
    elif task_type == "flashinfer_temporary_cleanup":
        invalid_home = tmp_path / "home-file"
        invalid_home.touch()
        env["HOME"] = str(invalid_home)
        env["TMPDIR"] = str(tmp_path)

    with Popen(command,
               env=env,
               stdout=PIPE,
               stderr=PIPE,
               bufsize=1,
               start_new_session=True,
               universal_newlines=True,
               cwd=os.path.dirname(os.path.abspath(__file__))) as process:

        # Function to read from a stream and write to output
        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream,
                                         args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream,
                                         args=(process.stderr, sys.stderr))

        # Start both threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait()

        # Wait for both threads to finish reading
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    if task_type == "flashinfer_temporary_cleanup":
        assert not list(tmp_path.glob("trtllm-flashinfer-rank-*"))


def task1():
    non_mpi_env, mpi_env = split_mpi_env()
    assert non_mpi_env
    assert mpi_env


@pytest.mark.cpu_only
def test_split_mpi_env():
    session = MpiPoolSession(n_workers=4)
    session.submit_sync(task1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "task_script", ["_run_mpi_comm_task.py", "_run_multi_mpi_comm_tasks.py"])
def test_llmapi_launch_multiple_tasks(task_script: str):
    """
    Test that the trtllm-llmapi-launch can run multiple tasks.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, task_script)
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = [
        "mpirun", "-n", "2", "--allow-run-as-root", "trtllm-llmapi-launch",
        "python3", test_file
    ]
    print(' '.join(command))

    with Popen(command,
               env=os.environ,
               stdout=PIPE,
               stderr=PIPE,
               bufsize=1,
               start_new_session=True,
               universal_newlines=True,
               cwd=os.path.dirname(os.path.abspath(__file__))) as process:
        # Function to read from a stream and write to output
        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream,
                                         args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream,
                                         args=(process.stderr, sys.stderr))

        # Start both threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait()

        # Wait for both threads to finish reading
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


@pytest.mark.cpu_only
def test_llmapi_launch_isolates_pmi_rank_without_size(tmp_path: Path) -> None:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    python_stub = stub_bin / "python3"
    python_stub.write_text("#!/bin/sh\n"
                           "if [ \"$1\" = \"-c\" ]; then\n"
                           "    echo ipc:///tmp/trtllm-pmi-workspace-test\n"
                           "fi\n")
    python_stub.chmod(0o755)
    openssl_stub = stub_bin / "openssl"
    openssl_stub.write_text("#!/bin/sh\nprintf '%064d\\n' 0\n")
    openssl_stub.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    for name in (
            "SLURM_NTASKS",
            "SLURM_PROCID",
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "PMI_SIZE",
            "PMI_ID",
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_CUBIN_DIR",
            "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS",
    ):
        env.pop(name, None)
    env["PMI_RANK"] = "0"
    env["HOME"] = str(home)
    env["PATH"] = f"{stub_bin}{os.pathsep}{env['PATH']}"

    launcher = (Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" /
                "trtllm-llmapi-launch")
    result = subprocess.run(  # nosec B603
        ["bash", str(launcher), "/usr/bin/env"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )

    workspace = home / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-0"
    assert f"FLASHINFER_WORKSPACE_BASE={workspace}" in result.stdout
    assert "TRTLLM_FLASHINFER_WORKSPACE_MANAGED=1" in result.stdout


@pytest.mark.cpu_only
def test_llmapi_launch_aborts_when_no_workspace_is_available(
        tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in (
            "SLURM_NTASKS",
            "SLURM_PROCID",
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "PMI_SIZE",
            "PMI_ID",
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_CUBIN_DIR",
            "TRTLLM_FLASHINFER_WORKSPACE_MANAGED",
            "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS",
    ):
        env.pop(name, None)
    env["PMI_RANK"] = "0"
    env["HOME"] = ""
    env["TMPDIR"] = str(tmp_path / "missing")

    launcher = (Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" /
                "trtllm-llmapi-launch")
    result = subprocess.run(  # nosec B603
        ["/bin/bash", str(launcher), "/usr/bin/true"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert (
        "Failed to create a temporary FlashInfer JIT workspace; aborting launch"
        in result.stderr)


# ---- wait_shutdown: shutdown blocks until worker processes actually exit ----


def _wait_workers_exit(identities, timeout: float) -> None:
    """Call the unbound method on an inert stand-in (no MPI spawn).

    ``_wait_workers_exit`` only reads ``self._worker_identities``; a real
    ``MpiPoolSession`` shell would trigger the base class's abort machinery
    at garbage collection.
    """
    import types

    stand_in = types.SimpleNamespace(_worker_identities=identities)
    MpiPoolSession._wait_workers_exit(stand_in, timeout=timeout)


def test_process_start_time_live_and_gone():
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    assert _process_start_time(os.getpid()) is not None
    child = Popen(["true"])  # nosec B603, B607
    child.wait()
    assert _process_start_time(child.pid) is None  # reaped: /proc entry gone


def test_wait_workers_exit_returns_once_workers_are_gone():
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    child = Popen(["true"])  # nosec B603, B607
    identity = (child.pid, _process_start_time(child.pid))
    child.wait()
    # Dead worker -> returns immediately; a None start_time is skipped
    # (identity collection failed for that worker: nothing to wait on).
    _wait_workers_exit((identity, (os.getpid(), None)), timeout=5.0)


def test_wait_workers_exit_bounded_by_timeout_on_live_worker():
    import time as _time

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    t0 = _time.monotonic()
    _wait_workers_exit((me, ), timeout=0.2)  # this process will not exit
    waited = _time.monotonic() - t0
    assert 0.2 <= waited < 2.0  # bounded: a wedged worker cannot hang teardown


def _collect_identities(monkeypatch,
                        results,
                        pending=0,
                        n_workers=2,
                        observed_timeouts=None):
    """Drive _collect_worker_identities on an inert stand-in (no MPI spawn)."""
    import types
    from concurrent.futures import Future

    futs = []
    for r in results:
        f = Future()
        f.set_result(r)
        futs.append(f)
    never = [Future() for _ in range(pending)]  # never resolve

    from tensorrt_llm.llmapi import mpi_session as m

    def _fake_wait(fs, timeout):
        if observed_timeouts is not None:
            observed_timeouts.append(timeout)
        return futs, never

    monkeypatch.setattr(m, "futures_wait", _fake_wait)
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    it = iter(futs + never)
    stand_in = types.SimpleNamespace(
        n_workers=n_workers,
        mpi_pool=types.SimpleNamespace(submit=lambda fn: next(it),
                                       shutdown=lambda wait=True: None),
        _teardown_unidentified_pool=lambda ids: MpiPoolSession.
        _teardown_unidentified_pool(stand_in, ids),
    )
    result = MpiPoolSession._collect_worker_identities(stand_in)
    return result, killed


def test_identity_collection_complete_returns_identities(monkeypatch):
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    other = (1, b"1")  # pid 1: exists but start_time won't match -> unique pid
    ids, killed = _collect_identities(monkeypatch, [me, other])
    assert set(ids) == {me, other} and not killed


def test_identity_collection_fails_closed_on_timeout(monkeypatch):
    # A pending barrier task means the pool cannot honor wait_shutdown:
    # the session must be torn down and rejected, NOT handed out with the
    # contract silently downgraded (review requirement).
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    with _pytest.raises(RuntimeError, match="incomplete"):
        _collect_identities(monkeypatch, [me], pending=1)


def test_identity_collection_fails_closed_on_duplicate_pids(monkeypatch):
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    with _pytest.raises(RuntimeError, match="incomplete"):
        _collect_identities(monkeypatch, [me, me])  # one worker answered twice


def test_identity_collection_uses_configured_timeout(monkeypatch):
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "123.5")
    observed_timeouts = []
    me = (os.getpid(), _process_start_time(os.getpid()))
    _collect_identities(monkeypatch, [me],
                        n_workers=1,
                        observed_timeouts=observed_timeouts)
    assert observed_timeouts == [123.5]


def test_identity_timeout_covers_worker_bootstrap(monkeypatch):
    # The deadline bounds spawn + `import tensorrt_llm`, not barrier latency:
    # it must exceed the slowest bootstrap the repo measures (~117s busy node).
    monkeypatch.delenv("TRTLLM_MPI_IDENTITY_TIMEOUT", raising=False)
    assert _identity_barrier_timeout() > 117.0


# Invalid values (unparsable, non-positive) fall back to the default rather
# than turning the barrier into a busy-wait or an unbounded block.
@pytest.mark.parametrize("raw, expected",
                         [("90", 90.0), ("0.5", 0.5),
                          ("", _DEFAULT_IDENTITY_TIMEOUT),
                          ("0", _DEFAULT_IDENTITY_TIMEOUT),
                          ("-1", _DEFAULT_IDENTITY_TIMEOUT),
                          ("abc", _DEFAULT_IDENTITY_TIMEOUT),
                          ("nan", _DEFAULT_IDENTITY_TIMEOUT),
                          ("inf", _DEFAULT_IDENTITY_TIMEOUT),
                          ("-inf", _DEFAULT_IDENTITY_TIMEOUT),
                          ("1e309", _DEFAULT_IDENTITY_TIMEOUT)])
def test_identity_timeout_env_override(monkeypatch, raw, expected):
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", raw)
    assert _identity_barrier_timeout() == expected


def test_prefetch_fallback_identity_timeout_matches_mpi_default():
    from test_common.session_prefetcher import _FALLBACK_IDENTITY_TIMEOUT

    assert _FALLBACK_IDENTITY_TIMEOUT == _DEFAULT_IDENTITY_TIMEOUT

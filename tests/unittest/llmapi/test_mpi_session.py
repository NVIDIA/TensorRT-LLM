# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import threading
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Literal

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_IDENTITY_TIMEOUT,
                                             MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             _identity_barrier_timeout,
                                             split_mpi_env)


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


_LAUNCHER = (Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" /
             "trtllm-llmapi-launch")

# Variables that would steer the launcher's workspace setup; each launcher test
# starts from an environment without them and adds back only what it exercises.
_LAUNCHER_ENV_SCRUB = (
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
)


def _launcher_env(tmp_path: Path, home: str) -> dict:
    """Environment for a rank-0 launcher-managed run of ``trtllm-llmapi-launch``.

    ``python3`` and ``openssl`` are stubbed so the launcher can hand out an IPC
    address and an HMAC key without importing tensorrt_llm; the stubbed
    ``python3 -S`` used for the workspace lock exits 0, so the persistent slot
    is treated as acquired.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(exist_ok=True)
    python_stub = stub_bin / "python3"
    python_stub.write_text("#!/bin/sh\n"
                           "if [ \"$1\" = \"-c\" ]; then\n"
                           "    echo ipc:///tmp/trtllm-pmi-workspace-test\n"
                           "fi\n")
    python_stub.chmod(0o755)
    openssl_stub = stub_bin / "openssl"
    openssl_stub.write_text("#!/bin/sh\nprintf '%064d\\n' 0\n")
    openssl_stub.chmod(0o755)

    env = os.environ.copy()
    for name in _LAUNCHER_ENV_SCRUB:
        env.pop(name, None)
    env["PMI_RANK"] = "0"
    env["HOME"] = home
    env["PATH"] = f"{stub_bin}{os.pathsep}{env['PATH']}"
    return env


def _run_launcher_env(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(  # nosec B603
        ["bash", str(_LAUNCHER), "/usr/bin/env"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )


@pytest.mark.cpu_only
def test_llmapi_launch_isolates_pmi_rank_without_size(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _launcher_env(tmp_path, str(home))

    result = _run_launcher_env(env)

    workspace = home / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-0"
    assert f"FLASHINFER_WORKSPACE_BASE={workspace}" in result.stdout
    assert "TRTLLM_FLASHINFER_WORKSPACE_MANAGED=1" in result.stdout
    # The artifact cache must follow the per-rank workspace, so the launcher
    # must not pin FLASHINFER_CUBIN_DIR to a shared directory.
    assert "FLASHINFER_CUBIN_DIR=" not in result.stdout


@pytest.mark.cpu_only
def test_llmapi_launch_preserves_explicit_cubin_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    shared = tmp_path / "shared-cubins"
    env = _launcher_env(tmp_path, str(home))
    env["FLASHINFER_CUBIN_DIR"] = str(shared)

    result = _run_launcher_env(env)

    workspace = home / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-0"
    assert f"FLASHINFER_WORKSPACE_BASE={workspace}" in result.stdout
    lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("FLASHINFER_CUBIN_DIR=")
    ]
    assert lines == [f"FLASHINFER_CUBIN_DIR={shared}"]


@pytest.mark.cpu_only
def test_llmapi_launch_temporary_fallback_leaves_cubin_dir_unset(
        tmp_path: Path) -> None:
    # A regular file as HOME makes the persistent workspace root impossible to
    # create, which sends the launcher down the temporary-workspace fallback.
    invalid_home = tmp_path / "home-file"
    invalid_home.touch()
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    env = _launcher_env(tmp_path, str(invalid_home))
    env["TMPDIR"] = str(tmpdir)

    result = _run_launcher_env(env)

    prefix = f"FLASHINFER_WORKSPACE_BASE={tmpdir}/trtllm-flashinfer-rank-0."
    assert any(line.startswith(prefix)
               for line in result.stdout.splitlines()), result.stdout
    assert "TRTLLM_FLASHINFER_WORKSPACE_MANAGED=1" in result.stdout
    assert "FLASHINFER_CUBIN_DIR=" not in result.stdout
    assert "temporary FlashInfer JIT workspace" in result.stderr
    # The fallback workspace, artifacts included, is removed at exit.
    assert list(tmpdir.iterdir()) == []


@pytest.mark.cpu_only
def test_llmapi_launch_aborts_when_no_workspace_is_available(
        tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in _LAUNCHER_ENV_SCRUB:
        env.pop(name, None)
    env["PMI_RANK"] = "0"
    env["HOME"] = ""
    env["TMPDIR"] = str(tmp_path / "missing")

    result = subprocess.run(  # nosec B603
        ["/bin/bash", str(_LAUNCHER), "/usr/bin/true"],
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


class _FakeCommExecutor:
    """Stand-in for the entered MPICommExecutor context manager."""

    def __init__(self, block: bool = False, fail: bool = False):
        self.block = block
        self.fail = fail
        self.release = threading.Event()
        self.exited = threading.Event()

    def __exit__(self, exc_type, exc_value, tb):
        if self.block:
            # A wedged worker rank: the join does not return until the test
            # releases it (so the closer thread does not leak past the test).
            self.release.wait(30)
        self.exited.set()
        if self.fail:
            raise RuntimeError("simulated executor close failure")


def _wait_closer_thread_gone(timeout: float = 5.0) -> None:
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if not any(t.name == "MpiCommExecutorCloser"
                   for t in threading.enumerate()):
            return
        _time.sleep(0.02)
    raise AssertionError("MpiCommExecutorCloser remained alive after "
                         f"{timeout}s")


@pytest.fixture
def _global_executor_state():
    """Snapshot/restore the module-global executor slots around a test."""
    saved = (MPINodeState._global_comm_executor, MPINodeState._global_mpi_pool)
    yield
    (MPINodeState._global_comm_executor, MPINodeState._global_mpi_pool) = saved


def test_server_close_releases_the_global_comm_executor(_global_executor_state):
    """The server's final shutdown must close the shared COMM_WORLD executor.

    ``MpiCommSession.shutdown()`` deliberately leaves the shared pool running
    (multiple LLM instances reuse it), so without this close the non-leader
    ranks stay blocked in the executor's task loop after the client is gone
    -- the teardown hang behind a serve that could only die by hard kill.
    """
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    fake = _FakeCommExecutor()
    MPINodeState._global_comm_executor = fake
    MPINodeState._global_mpi_pool = object()
    aborted = []

    RemoteMpiCommSessionServer._close_global_comm_executor(
        grace=5.0, abort=lambda: aborted.append(True))

    assert fake.exited.is_set()
    assert not aborted
    _wait_closer_thread_gone()
    assert MPINodeState._global_comm_executor is None
    assert MPINodeState._global_mpi_pool is None


def test_server_close_escalates_to_abort_when_the_join_wedges(
        _global_executor_state):
    """A worker stranded in a collective must not block teardown forever."""
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    fake = _FakeCommExecutor(block=True)
    MPINodeState._global_comm_executor = fake
    aborted = []

    RemoteMpiCommSessionServer._close_global_comm_executor(
        grace=0.2, abort=lambda: aborted.append(True))

    assert aborted == [True]
    # Release the wedged join so the closer thread ends inside the test
    # (pytest-threadleak checks for leaked threads per test).
    fake.release.set()
    fake.exited.wait(5)
    _wait_closer_thread_gone()


def test_server_close_escalates_to_abort_when_exit_raises(
        _global_executor_state):
    """Escalate to abort when the executor's ``__exit__`` raises.

    A raised ``__exit__`` means the executor did not cleanly release, so peers
    can stay blocked; the global refs are already cleared, so nothing else will
    close them -- escalate the same way a timeout does.
    """
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    fake = _FakeCommExecutor(fail=True)
    MPINodeState._global_comm_executor = fake
    aborted = []

    RemoteMpiCommSessionServer._close_global_comm_executor(
        grace=5.0, abort=lambda: aborted.append(True))

    assert fake.exited.is_set()
    assert aborted == [True]
    _wait_closer_thread_gone()
    assert MPINodeState._global_comm_executor is None


def test_server_close_is_a_noop_without_a_global_executor(
        _global_executor_state):
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    MPINodeState._global_comm_executor = None
    # Must not touch MPI at all (no abort callable is even constructed).
    RemoteMpiCommSessionServer._close_global_comm_executor(grace=0.1)

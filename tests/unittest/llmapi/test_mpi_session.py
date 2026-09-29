# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import shutil
import subprocess  # nosec B404
import sys
import threading
import time
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


def rendezvous_environment_probe():
    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    return os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_pool_session_forwards_current_rendezvous(monkeypatch):
    # The MPI launcher may outlive a pool and retain its original environment.
    for address, port in [("127.0.0.1", "33271"), ("localhost", "51003")]:
        monkeypatch.setenv("MASTER_ADDR", address)
        monkeypatch.setenv("MASTER_PORT", port)
        session = MpiPoolSession(n_workers=2, wait_shutdown=True)
        try:
            assert session.submit_sync(rendezvous_environment_probe) == [
                (address, port), (address, port)
            ]
        finally:
            session.shutdown()


def flashinfer_environment_probe():
    """Return this worker's FlashInfer isolation paths."""
    # Keep importing this from initializing MPI in the submitting process.
    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    return (
        os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        os.environ.get("FLASHINFER_CUBIN_DIR"),
    )


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_pool_session_flashinfer_workspace_isolation(monkeypatch):
    # A singleton spawn reuses OpenMPI's already-running DVM once one exists in
    # this process, so a spawned worker's HOME can't be relied on to follow a
    # monkeypatched HOME set here; compare against the real home directory
    # instead.
    monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS", raising=False)

    session = MpiPoolSession(n_workers=2, wait_shutdown=True)
    try:
        worker_envs = session.submit_sync(flashinfer_environment_probe)
    finally:
        session.shutdown()

    workspaces = {workspace for workspace, _ in worker_envs}
    cubin_dirs = {cubin_dir for _, cubin_dir in worker_envs}
    assert None not in workspaces
    assert len(workspaces) == 2
    workspace_root = Path.home() / ".cache" / "tensorrt_llm" / "flashinfer"
    assert all(
        Path(workspace).parent == workspace_root for workspace in workspaces)
    # Unset means FlashInfer derives the artifact cache from each worker's
    # isolated workspace, keeping downloaded compiler inputs per-rank.
    assert cubin_dirs == {None}


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


# ---- fail fast when the rank-0 task exits and the worker world is wedged ----

_MPI_RUNTIME_MISSING = (shutil.which("mpirun") is None
                        or importlib.util.find_spec("mpi4py") is None)


@pytest.mark.cpu_only
@pytest.mark.skipif(_MPI_RUNTIME_MISSING,
                    reason="mpirun and mpi4py are required")
def test_remote_mpi_session_fails_fast_when_worker_world_hangs() -> None:
    """The rank-0 task exits while every worker rank is stuck in a task.

    Before ``RemoteMpiCommSessionServer`` polled the control socket while
    worker futures were outstanding, the launcher's stop request was never
    read, rank 0 never exited, and only the ``timeout`` in the driver script
    ended the run (exit code 124). Now the server sees the stop request while
    the futures are outstanding, gives the worker world
    ``TLLM_MGMN_SHUTDOWN_GRACE_SECONDS`` to drain, then calls MPI_Abort so the
    whole run ends non-zero well inside the driver's timeout.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, "_test_remote_mpi_session.sh")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = ["bash", test_file, "hang"]
    print(' '.join(command))
    env = os.environ.copy()
    env["TLLM_MGMN_SHUTDOWN_GRACE_SECONDS"] = "5"
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = "20"
    # The abort decision is logged at warning and critical level; make sure
    # both lines are emitted so the assertions below can see them.
    env["TLLM_LOG_LEVEL"] = "info"

    start = time.monotonic()
    result = subprocess.run(  # nosec B603
        command,
        env=env,
        cwd=cur_dir,
        stdout=PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    elapsed = time.monotonic() - start
    print(result.stdout)
    print(f"hang run ended in {elapsed:.1f}s with exit code "
          f"{result.returncode}")

    # 124 is `timeout` killing a run that never ended on its own.
    assert result.returncode != 124, "the rank-0 launcher never exited"
    assert result.returncode != 0, "a rank-0 task that exits 1 must not end 0"
    assert ("shutdown requested while 2/2 MPI worker task(s) are still "
            "running") in result.stdout
    assert "calling MPI_Abort so no rank outlives the leader" in result.stdout


def _pending_futures_stand_in(messages):
    """A server stand-in whose control socket replays ``messages`` (no MPI)."""
    import types

    class _Queue:

        def poll(self, timeout):
            return bool(messages)

        def get(self):
            return messages.pop(0)

    return types.SimpleNamespace(queue=_Queue(),
                                 _deferred_messages=[],
                                 PENDING_POLL_SECONDS=0.01)


def test_wait_for_pending_futures_sees_shutdown_while_workers_run():
    from concurrent.futures import Future

    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    wait_for_pending = RemoteMpiCommSessionServer._wait_for_pending_futures
    wedged = [Future(), Future()]  # never resolve: the worker world is stuck
    server = _pending_futures_stand_in(["task", None])
    t0 = time.monotonic()
    assert wait_for_pending(server, wedged) is True
    assert time.monotonic() - t0 < 5.0  # did not block on the futures
    # The task that arrived before the stop request is kept, not dropped.
    assert server._deferred_messages == ["task"]


def test_wait_for_pending_futures_completes_without_shutdown():
    from concurrent.futures import Future

    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    wait_for_pending = RemoteMpiCommSessionServer._wait_for_pending_futures
    done = []
    for _ in range(2):
        f = Future()
        f.set_result(None)
        done.append(f)
    server = _pending_futures_stand_in([])
    assert wait_for_pending(server, done) is False
    assert server._deferred_messages == []


def test_mgmn_shutdown_grace_default(monkeypatch):
    from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_MGMN_SHUTDOWN_GRACE,
                                                 _mgmn_shutdown_grace_seconds)

    monkeypatch.delenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", raising=False)
    assert _mgmn_shutdown_grace_seconds() == _DEFAULT_MGMN_SHUTDOWN_GRACE


# Invalid values (unparsable, non-finite, non-positive) fall back to the
# default rather than turning every shutdown into an immediate MPI_Abort.
@pytest.mark.parametrize("raw, use_default", [("5", False), ("0.5", False),
                                              ("", True), ("0", True),
                                              ("-1", True), ("abc", True),
                                              ("nan", True), ("inf", True)])
def test_mgmn_shutdown_grace_env_override(monkeypatch, raw, use_default):
    from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_MGMN_SHUTDOWN_GRACE,
                                                 _mgmn_shutdown_grace_seconds)

    monkeypatch.setenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", raw)
    expected = _DEFAULT_MGMN_SHUTDOWN_GRACE if use_default else float(raw)
    assert _mgmn_shutdown_grace_seconds() == expected


def _shutdown_worker_world(monkeypatch, shutdown, pending_futures, grace):
    """Drive ``_shutdown_worker_world`` on an inert stand-in (no MPI launch).

    ``shutdown`` stands in for ``session.shutdown``. The MPI_Abort call, the
    shared executor teardown and the module logger are replaced by recorders,
    so the test sees exactly which branch ran. Returns the recorded abort
    calls, the warning/critical log lines and the wall time spent.
    """
    import types

    from tensorrt_llm.llmapi import mpi_session as m

    monkeypatch.setenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", str(grace))
    aborts = []
    logged = {"warning": [], "critical": []}
    monkeypatch.setattr(
        m, "logger",
        types.SimpleNamespace(warning=logged["warning"].append,
                              critical=logged["critical"].append,
                              error=lambda *args, **kwargs: None,
                              info=lambda *args, **kwargs: None,
                              debug=lambda *args, **kwargs: None))
    monkeypatch.setattr(m.MPINodeState, "close_global_comm_executor",
                        classmethod(lambda cls: None))
    stand_in = types.SimpleNamespace(session=types.SimpleNamespace(
        shutdown=shutdown, abort=lambda: aborts.append("abort")))
    t0 = time.monotonic()
    m.RemoteMpiCommSessionServer._shutdown_worker_world(stand_in,
                                                        pending_futures)
    return aborts, logged, time.monotonic() - t0


@pytest.mark.parametrize("still_running", [False, True])
def test_shutdown_worker_world_returns_without_abort_when_teardown_finishes(
        monkeypatch, still_running):
    # ``session.shutdown`` returns at once, so the teardown thread finishes
    # inside the grace period: no MPI_Abort, whether or not a worker future
    # was still outstanding when shutdown was requested.
    from concurrent.futures import Future

    done = Future()
    done.set_result(None)
    pending = [done, Future()] if still_running else [done]
    shutdowns = []
    aborts, logged, elapsed = _shutdown_worker_world(
        monkeypatch,
        shutdown=lambda: shutdowns.append("shutdown"),
        pending_futures=pending,
        grace=0.2)

    assert shutdowns == ["shutdown"]
    assert aborts == []
    assert logged["critical"] == []
    assert elapsed < 2.0
    if still_running:
        assert len(logged["warning"]) == 1
        assert ("shutdown requested while 1/2 MPI worker task(s) are still "
                "running; waiting up to 0.2s before calling MPI_Abort"
                ) in logged["warning"][0]
    else:
        assert logged["warning"] == []


def test_shutdown_worker_world_aborts_when_teardown_exceeds_grace(monkeypatch):
    # ``session.shutdown`` blocks like a wedged worker world would; the
    # bounded wait must give up after the grace period and call MPI_Abort
    # exactly once.
    from concurrent.futures import Future

    release = threading.Event()
    try:
        aborts, logged, elapsed = _shutdown_worker_world(
            monkeypatch,
            shutdown=release.wait,
            pending_futures=[Future(), Future()],
            grace=0.2)
    finally:
        release.set()  # let the daemon teardown thread finish

    assert aborts == ["abort"]
    assert 0.2 <= elapsed < 5.0
    assert len(logged["warning"]) == 1
    assert ("shutdown requested while 2/2 MPI worker task(s) are still "
            "running; waiting up to 0.2s before calling MPI_Abort"
            ) in logged["warning"][0]
    assert len(logged["critical"]) == 1
    assert ("worker world did not shut down within 0.2s; calling MPI_Abort "
            "so no rank outlives the leader") in logged["critical"][0]


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


@pytest.mark.cpu_only
def test_llmapi_launch_fails_when_mpi_comm_server_must_be_killed(
        tmp_path: Path) -> None:
    """The task and the stop request both return 0, but the server never exits.

    The stubbed MPI Comm server blocks until it is signalled, so the
    launcher's bounded wait has to kill it. That forced kill must surface as
    a non-zero launcher exit code even though every command the subshell ran
    returned 0.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = _launcher_env(tmp_path, str(home))
    # Re-stub python3: ``-m tensorrt_llm.llmapi.mgmn_leader_node`` with no
    # ``--action`` is the server and blocks until signalled; the stop request
    # and the workspace lock keep exiting 0, ``-c`` keeps printing the address.
    python_stub = tmp_path / "bin" / "python3"
    python_stub.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "    echo ipc:///tmp/trtllm-pmi-workspace-test\n"
        "elif [ $# -eq 2 ] && "
        "[ \"$2\" = \"tensorrt_llm.llmapi.mgmn_leader_node\" ]; then\n"
        "    trap 'kill $sleeper 2>/dev/null; exit 143' TERM\n"
        "    sleep 60 &\n"
        "    sleeper=$!\n"
        "    wait $sleeper\n"
        "fi\n")
    env["TLLM_LLMAPI_LAUNCH_STOP_TIMEOUT"] = "1"

    # A one-second task keeps the stop request behind the server start-up.
    result = subprocess.run(  # nosec B603
        ["bash", str(_LAUNCHER), "sleep", "1"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )

    assert "Task exit code: 0" in result.stderr
    assert "MPI Comm server exit code: 0" in result.stderr
    assert ("still running 1s after the stop request (task exit code 0); "
            "killing it so this rank exits") in result.stderr
    assert "Subshell exit code: 1" in result.stderr
    assert result.returncode == 1, result.stderr


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

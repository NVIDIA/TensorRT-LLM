# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import importlib.util
import json
import os
import shlex
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path
from subprocess import PIPE, Popen
from types import ModuleType
from typing import Literal

import pytest
import zmq
from integration.defs.trt_test_alternative import check_output

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_IDENTITY_TIMEOUT,
                                             MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             RemoteMpiCommSessionServer,
                                             _identity_barrier_timeout,
                                             split_mpi_env)

_STOP_HELPER = (Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" /
                "_llmapi_mpi_stop.py")
_STOP_HELPER_RUNNER = """
import importlib.util
import sys

forbidden = {"tensorrt_llm", "torch", "mpi4py"}
assert not forbidden.intersection(sys.modules)

def forbid_heavy_imports(event: str, args: tuple[object, ...]) -> None:
    if event == "import" and str(args[0]).split(".")[0] in forbidden:
        raise AssertionError(f"stop helper imported {args[0]}")

sys.addaudithook(forbid_heavy_imports)
spec = importlib.util.spec_from_file_location("stop_helper", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.exit(helper.main(ack_timeout=float(sys.argv[2]),
                     max_attempts=int(sys.argv[3])))
"""


def _stop_helper_command(ack_timeout: float, max_attempts: int) -> list[str]:
    """Return a helper subprocess command with import guards and bounded retries."""
    return [
        sys.executable, "-c", _STOP_HELPER_RUNNER,
        str(_STOP_HELPER),
        str(ack_timeout),
        str(max_attempts)
    ]


def _stop_helper_environment(address: str, hmac_key: bytes) -> dict[str, str]:
    """Return the inherited runtime environment with this server's credentials."""
    env = os.environ.copy()
    env["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"] = address
    env["TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"] = hmac_key.hex()
    return env


def _request_stop(stop_client: Literal["module", "standalone"], address: str,
                  hmac_key: bytes, ack_timeout: float,
                  max_attempts: int) -> bool:
    """Run either stop client and return whether the server acknowledged it."""
    if stop_client == "module":
        from tensorrt_llm.llmapi.mgmn_leader_node import stop_server_main
        return stop_server_main(ack_timeout=ack_timeout,
                                max_attempts=max_attempts)
    result = subprocess.run(
        _stop_helper_command(ack_timeout, max_attempts),
        env=_stop_helper_environment(address, hmac_key),
        capture_output=True,
        text=True,
        timeout=ack_timeout * max_attempts + 10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def task0() -> int:
    if MPINodeState.state is None:
        MPINodeState.state = 0
    MPINodeState.state += 1
    return MPINodeState.state


def worker_rank():
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank()


def occupy_worker(seconds):
    import time

    time.sleep(seconds)


@pytest.fixture(autouse=True)
def _enable_mpi(monkeypatch):
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)


# Bounds every wait on pool workers. A fresh pool's first task also covers
# worker spawn and import, so reuse the library's bootstrap deadline.
_POOL_TIMEOUT = _identity_barrier_timeout()


@contextlib.contextmanager
def _pool_session(*args, **kwargs):
    """``MpiPoolSession`` that is abandoned instead of joined on failure.

    A failed test may leave workers stuck in a task, and joining them would
    hang the teardown and the interpreter exit.
    """
    session = MpiPoolSession(*args, **kwargs)
    try:
        yield session
    except BaseException:
        session.abandon()
        raise
    session.shutdown()


def _submit_sync(session, task, *args, **kwargs):
    """``session.submit_sync`` with a deadline."""
    futures = session.submit(task, *args, **kwargs)
    _, not_done = futures_wait(futures, timeout=_POOL_TIMEOUT)
    if not_done:
        raise TimeoutError(f"{len(not_done)}/{len(futures)} copies of {task} "
                           f"did not finish within {_POOL_TIMEOUT}s")
    return [future.result() for future in futures]


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_session_basic() -> None:
    with _pool_session(4, wait_shutdown=True) as executor:
        results = _submit_sync(executor, task0)
        assert results == [1, 1, 1, 1], results

        results = _submit_sync(executor, task0)
        assert results == [2, 2, 2, 2], results


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_pool_session_submit_runs_once_per_worker():
    n_workers = 4
    with _pool_session(n_workers) as executor:
        # Warm the pool so every worker is up before one gets occupied.
        assert sorted(_submit_sync(executor,
                                   worker_rank)) == list(range(n_workers))

        # With one worker busy, the idle workers must not drain its copy.
        busy = executor.mpi_pool.submit(occupy_worker, 3)
        results = _submit_sync(executor, worker_rank)
        busy.result(timeout=_POOL_TIMEOUT)
        assert sorted(results) == list(range(n_workers)), results


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_comm_session_submit_runs_once_per_rank():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        "mpirun", "-n", "4", "--allow-run-as-root", "--oversubscribe",
        sys.executable, "_run_mpi_comm_session_pinning.py"
    ]
    stdout = check_output(command, timeout=300, cwd=cur_dir)
    assert "MpiCommSession pinning checks passed" in stdout


@pytest.mark.cpu_only
def test_mpi_pool_session_concurrent_submits_keep_copies_contiguous():
    """Each submission's copies must sit together in the pool's queue.

    The pinning barrier does not tell submissions apart, so interleaved
    copies would let workers of different submissions pass it together.
    """
    import time
    import types

    queue = []

    def fake_submit(fn, task, *args, **kwargs):
        queue.append(task)
        time.sleep(0.001)  # yield so concurrent submits would interleave

    n_workers = 4
    session = types.SimpleNamespace(
        n_workers=n_workers,
        mpi_pool=types.SimpleNamespace(submit=fake_submit),
        _submit_lock=threading.Lock(),
    )
    tasks = [f"task{i}" for i in range(8)]
    threads = [
        threading.Thread(target=MpiPoolSession.submit, args=(session, task))
        for task in tasks
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    runs = [queue[i:i + n_workers] for i in range(0, len(queue), n_workers)]
    assert all(len(set(run)) == 1 for run in runs), queue
    assert sorted(run[0] for run in runs) == tasks


def rendezvous_environment_probe():
    return os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_pool_session_forwards_current_rendezvous(monkeypatch):
    # The MPI launcher may outlive a pool and retain its original environment.
    for address, port in [("127.0.0.1", "33271"), ("localhost", "51003")]:
        monkeypatch.setenv("MASTER_ADDR", address)
        monkeypatch.setenv("MASTER_PORT", port)
        with _pool_session(n_workers=2, wait_shutdown=True) as session:
            assert _submit_sync(session, rendezvous_environment_probe) == [
                (address, port), (address, port)
            ]


_STALE_SPAWN_ENV = "TLLM_TEST_STALE_SPAWN_ENV"
_STALE_SPAWN_SCRIPT = f"""
import json
import os
import sys
import tempfile
from pathlib import Path

from tensorrt_llm.llmapi.mpi_session import MpiPoolSession


def spawn_and_read():
    session = MpiPoolSession(n_workers=1)
    try:
        return session.submit_sync(os.getenv, {_STALE_SPAWN_ENV!r})[0]
    finally:
        session.shutdown()


os.environ[{_STALE_SPAWN_ENV!r}] = "1"
while_set = spawn_and_read()
del os.environ[{_STALE_SPAWN_ENV!r}]
Path(sys.argv[1]).write_text(json.dumps([while_set, spawn_and_read()]))
"""


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_pool_session_does_not_resurrect_unset_forwarded_env(
        tmp_path: Path) -> None:
    # A singleton's MPI runtime is forked at the process's first spawn and
    # launches every later worker with that environment, so a fresh process
    # is needed to make the first spawn happen while the variable is set.
    env, _ = split_mpi_env()
    env.pop(_STALE_SPAWN_ENV, None)
    # Keep PMIX_HOSTNAME, which split_mpi_env() drops with the other PMIX_*
    # variables: it only overrides the hostname PMIx uses, and CI sets it on
    # pods whose names are too long for a singleton's MPI_Comm_spawn.
    if "PMIX_HOSTNAME" in os.environ:
        env["PMIX_HOSTNAME"] = os.environ["PMIX_HOSTNAME"]
    worker_env = tmp_path / "worker_env.json"
    check_output(
        [sys.executable, "-c", _STALE_SPAWN_SCRIPT,
         str(worker_env)],
        timeout=300,
        env=env,
    )
    assert json.loads(worker_env.read_text()) == ["1", None]


def flashinfer_environment_probe():
    """Return this worker's FlashInfer isolation paths."""
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

    with _pool_session(n_workers=2, wait_shutdown=True) as session:
        worker_envs = _submit_sync(session, flashinfer_environment_probe)

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
@pytest.mark.parametrize(
    "response",
    [[1, 1], RuntimeError("remote task failed"),
     SystemExit(7)],
    ids=["results", "error", "system-exit"])
def test_remote_mpi_submit_sync_response_after_empty_poll(
        monkeypatch: pytest.MonkeyPatch,
        response: list[int] | BaseException) -> None:
    """A response following an empty poll must wake the waiting client."""
    first_empty_poll = threading.Event()
    release_server = threading.Event()
    address: Future[tuple[str, bytes]] = Future()
    empty_poll_at: list[float] = []
    monkeypatch.setattr(RemoteMpiCommSessionClient, "_global_instance", None)

    def send_response() -> None:
        # Each socket is created, used, and closed on its owning thread.
        server = ZeroMqQueue(is_server=True)
        server.socket.setsockopt(zmq.SNDTIMEO, 2000)
        server.socket.setsockopt(zmq.LINGER, 0)
        try:
            address.set_result(server.address)
            request = server.get(timeout=10)
            assert request.sync
            assert request.args == (-1, )
            assert first_empty_poll.wait(10)
            server.put(response)
            assert release_server.wait(10)
        finally:
            server.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        peer = executor.submit(send_response)
        client = None
        try:
            endpoint, key = address.result(timeout=10)
            client = RemoteMpiCommSessionClient(endpoint, hmac_key=key)
            client.queue.socket.setsockopt(zmq.SNDTIMEO, 2000)
            client.queue.socket.setsockopt(zmq.LINGER, 0)
            poll = client.poll

            def observe_poll() -> object:
                if empty_poll_at and time.monotonic() - empty_poll_at[0] > 10:
                    raise TimeoutError("Client did not receive the response")
                result = poll()
                if not result and not first_empty_poll.is_set():
                    empty_poll_at.append(time.monotonic())
                    first_empty_poll.set()
                return result

            # Observe the real socket poll; do not alter its timeout or result.
            monkeypatch.setattr(client, "poll", observe_poll)
            actual = client.submit_sync(abs, -1)
            elapsed = time.monotonic() - empty_poll_at[0]
        finally:
            release_server.set()
            if client is not None:
                client.queue.close()
            peer.result(timeout=10)

    assert first_empty_poll.is_set()
    assert elapsed < 2, f"Client waited {elapsed:.3f}s after an empty poll"
    if isinstance(response, BaseException):
        assert type(actual) is type(response)
        assert actual.args == response.args
    else:
        assert actual == response


@pytest.mark.cpu_only
def test_lightweight_stop_helper_queue_compatibility() -> None:
    """The standalone sender must receive an ACK through the production queue."""
    server = ZeroMqQueue(is_server=True, socket_type=zmq.PAIR)
    server.socket.setsockopt(zmq.LINGER, 0)
    server.socket.setsockopt(zmq.SNDTIMEO, 5000)
    try:
        endpoint, key = server.address
        with subprocess.Popen(
                _stop_helper_command(5.0, 1),
                env=_stop_helper_environment(endpoint, key),
                stdout=PIPE,
                stderr=PIPE,
                text=True,
        ) as process:
            try:
                assert server.poll(5), "The helper did not send a stop request"
                assert server.get(timeout=5) is None
                server.put(RemoteMpiCommSessionServer.SHUTDOWN_ACK)
                stdout, stderr = process.communicate(timeout=10)
                assert process.returncode == 0, stdout + stderr
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
    finally:
        server.close()


@pytest.fixture
def _stop_credential_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the standalone helper with isolated canonical key and cache state."""
    from tensorrt_llm.executor import utils as executor_utils

    monkeypatch.setattr(executor_utils, "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY",
                        None)
    monkeypatch.delenv(executor_utils.LlmLauncherEnvs.
                       TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value,
                       raising=False)
    spec = importlib.util.spec_from_file_location("stop_credential_test",
                                                  _STOP_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
@pytest.mark.parametrize(("key", "expected"), [
    pytest.param("ab" * 32, b"\xab" * 32, id="lowercase"),
    pytest.param("AB" * 32, b"\xab" * 32, id="uppercase"),
    pytest.param("", None, id="empty"),
    pytest.param("ab" * 31, None, id="short"),
    pytest.param("ab" * 33, None, id="long"),
    pytest.param("ab" * 31 + " \t", None, id="whitespace"),
    pytest.param("ab" * 32 + "\n", None, id="newline"),
    pytest.param("gg" * 32, None, id="nonhex"),
])
def test_lightweight_stop_key_matches_canonical_validation(
        _stop_credential_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
        key: str, expected: bytes | None) -> None:
    """Match real key validation and consume the canonical enum's environment."""
    from tensorrt_llm.executor import utils as executor_utils

    env_name = executor_utils.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    if expected is None:
        with pytest.raises(ValueError):
            executor_utils._normalize_spawn_proxy_process_ipc_hmac_key(key)
    else:
        assert executor_utils._normalize_spawn_proxy_process_ipc_hmac_key(
            key) == expected
    for consume in (executor_utils.get_spawn_proxy_process_ipc_hmac_key_env,
                    _stop_credential_module._consume_hmac_key):
        monkeypatch.setenv(env_name, key)
        if expected is None:
            with pytest.raises(ValueError):
                consume()
        else:
            assert consume() == expected
        assert env_name not in os.environ
    assert executor_utils._SPAWN_PROXY_PROCESS_IPC_HMAC_KEY == expected


@pytest.mark.cpu_only
def test_lightweight_stop_key_is_one_shot_unlike_canonical_cache(
        _stop_credential_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the helper's one-shot key and missing-key exception intentional."""
    from tensorrt_llm.executor import utils as executor_utils

    env_name = executor_utils.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    with pytest.raises(RuntimeError, match="HMAC encryption is required"):
        executor_utils.get_spawn_proxy_process_ipc_hmac_key_env()
    with pytest.raises(KeyError):
        _stop_credential_module._consume_hmac_key()
    monkeypatch.setenv(env_name, "ab" * 32)
    assert executor_utils.get_spawn_proxy_process_ipc_hmac_key_env(
    ) == b"\xab" * 32
    assert env_name not in os.environ
    assert executor_utils.get_spawn_proxy_process_ipc_hmac_key_env(
    ) == b"\xab" * 32
    monkeypatch.setenv(env_name, "ab" * 32)
    assert _stop_credential_module._consume_hmac_key() == b"\xab" * 32
    assert env_name not in os.environ
    with pytest.raises(KeyError):
        _stop_credential_module._consume_hmac_key()


@pytest.mark.cpu_only
@pytest.mark.skipif(sys.platform != "linux",
                    reason="requires Linux libc getenv")
@pytest.mark.parametrize("key", ["ab" * 32, "bad-\u00e9-value"],
                         ids=["valid", "invalid-nonascii"])
def test_lightweight_stop_key_matches_canonical_libc_scrub(
        _stop_credential_module: ModuleType, monkeypatch: pytest.MonkeyPatch,
        key: str) -> None:
    """Both consumers erase real libc bytes, including rejected multibyte keys."""
    from tensorrt_llm.executor import utils as executor_utils

    env_name = executor_utils.LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    key_bytes = os.fsencode(key)
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_void_p
    for consume in (executor_utils.get_spawn_proxy_process_ipc_hmac_key_env,
                    _stop_credential_module._consume_hmac_key):
        monkeypatch.setenv(env_name, key)
        value_ptr = libc.getenv(os.fsencode(env_name))
        assert value_ptr
        assert ctypes.string_at(value_ptr, len(key_bytes)) == key_bytes
        if key == "ab" * 32:
            assert consume() == b"\xab" * 32
        else:
            with pytest.raises(ValueError):
                consume()
        assert env_name not in os.environ
        assert ctypes.string_at(value_ptr,
                                len(key_bytes)) == b"\0" * len(key_bytes)


@pytest.mark.cpu_only
@pytest.mark.parametrize("task_type", [
    "submit", "submit_sync", "flashinfer_workspace",
    "flashinfer_temporary_cleanup", "task_kwargs", "client_sys_path"
])
def test_remote_mpi_session(
    task_type: Literal["submit", "submit_sync", "flashinfer_workspace",
                       "flashinfer_temporary_cleanup", "task_kwargs",
                       "client_sys_path"],
    tmp_path: Path,
) -> None:
    """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, "_test_remote_mpi_session.sh")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = ["bash", test_file, task_type]
    print(' '.join(command))
    env = os.environ.copy()
    if task_type == "client_sys_path":
        env.pop("PYTHONPATH", None)
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
        # trtllm-llmapi-launch otherwise puts its IPC socket under TMPDIR, and
        # tmp_path can exceed the 107-byte limit on a Unix socket path.
        env["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"] = (
            f"ipc://{tempfile.gettempdir()}/rpc_test_{uuid.uuid4()}")

    print(check_output(command, timeout=120, stderr=PIPE, env=env, cwd=cur_dir))

    if task_type == "flashinfer_temporary_cleanup":
        assert not list(tmp_path.glob("trtllm-flashinfer-rank-*"))


@pytest.mark.cpu_only
def test_remote_task_wrapper_preserves_paths_and_kwargs(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tensorrt_llm.llmapi import mpi_session

    existing = str(tmp_path / "existing")
    additional = str(tmp_path / "additional")
    monkeypatch.setattr(sys, "path", [existing])
    monkeypatch.setattr(mpi_session, "mpi_rank", lambda: 0)
    monkeypatch.setattr(mpi_session, "mpi_world_size", lambda: 1)

    result = RemoteMpiCommSessionServer.task_wrapper(
        (existing, additional, additional, "relative", ""),
        dict,
        client_sys_path="task-owned value",
        task="another task-owned value")

    assert sys.path == [existing, additional]
    assert result == {
        "client_sys_path": "task-owned value",
        "task": "another task-owned value"
    }


def task1():
    non_mpi_env, mpi_env = split_mpi_env()
    assert non_mpi_env
    assert mpi_env


@pytest.mark.cpu_only
def test_split_mpi_env():
    with _pool_session(n_workers=4) as session:
        _submit_sync(session, task1)


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

    print(check_output(command, timeout=300, stderr=PIPE, cwd=cur_dir))


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
    address and an HMAC key without importing tensorrt_llm. Stdlib-only
    ``python3 -S`` calls, including the process guard, use the test interpreter
    so launcher registration and cleanup retain their normal behavior.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(exist_ok=True)
    python_stub = stub_bin / "python3"
    stop_fifo = tmp_path / "launcher-stop"
    os.mkfifo(stop_fifo)
    python_stub.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-S\" ]; then\n"
        f"    exec {shlex.quote(sys.executable)} \"$@\"\n"
        "elif [ \"$1\" = \"-c\" ]; then\n"
        "    echo ipc:///tmp/trtllm-pmi-workspace-test\n"
        "elif [ \"${1##*/}\" = \"_llmapi_mpi_stop.py\" ]; then\n"
        "    echo stop > \"$LAUNCHER_TEST_STOP_FIFO\"\n"
        "elif [ \"$1\" = \"-m\" ]; then\n"
        "    if [ \"$4\" = \"stop\" ]; then\n"
        "        echo stop > \"$LAUNCHER_TEST_STOP_FIFO\"\n"
        "    else\n"
        "        read message < \"$LAUNCHER_TEST_STOP_FIFO\"\n"
        "    fi\n"
        "fi\n")
    python_stub.chmod(0o755)
    openssl_stub = stub_bin / "openssl"
    openssl_stub.write_text("#!/bin/sh\nprintf '%064d\\n' 0\n")
    openssl_stub.chmod(0o755)

    env = os.environ.copy()
    for name in _LAUNCHER_ENV_SCRUB:
        env.pop(name, None)
    env["PMI_RANK"] = "0"
    env["LAUNCHER_TEST_STOP_FIFO"] = str(stop_fifo)
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
def test_llmapi_launch_respects_unified_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "home" / "unified"
    env = _launcher_env(tmp_path, str(tmp_path / "home"))
    env["TRTLLM_CACHE_DIR"] = "~/unified"
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache_root / "flashinfer")

    result = _run_launcher_env(env)

    workspace = cache_root / "flashinfer" / "rank-0"
    assert f"FLASHINFER_WORKSPACE_BASE={workspace}" in result.stdout
    assert "TRTLLM_FLASHINFER_WORKSPACE_MANAGED=1" in result.stdout
    assert "better cache reuse" in result.stderr


@pytest.mark.cpu_only
def test_llmapi_launch_matches_bootstrap_expansion_with_trailing_home(
        tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cache_root = home / "unified"
    env = _launcher_env(tmp_path, f"{home}{os.sep}")
    env["TRTLLM_CACHE_DIR"] = "~/unified"
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache_root / "flashinfer")

    result = _run_launcher_env(env)

    workspace = cache_root / "flashinfer" / "rank-0"
    assert f"FLASHINFER_WORKSPACE_BASE={workspace}" in result.stdout
    assert "TRTLLM_FLASHINFER_WORKSPACE_MANAGED=1" in result.stdout


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
        mpi_pool=types.SimpleNamespace(submit=lambda *args, **kwargs: next(it),
                                       shutdown=lambda wait=True: None),
        submit=lambda task: MpiPoolSession.submit(stand_in, task),
        _submit_lock=threading.Lock(),
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
    saved = (MPINodeState._global_comm_executor, MPINodeState._global_mpi_pool,
             MPINodeState._global_rank0_dispatcher)
    yield
    (MPINodeState._global_comm_executor, MPINodeState._global_mpi_pool,
     MPINodeState._global_rank0_dispatcher) = saved


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_server_close_is_a_noop_without_a_global_executor(
        _global_executor_state):
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer

    MPINodeState._global_comm_executor = None
    # Must not touch MPI at all (no abort callable is even constructed).
    RemoteMpiCommSessionServer._close_global_comm_executor(grace=0.1)


def _pair_client(addr: str, hmac_key: bytes):
    from tensorrt_llm.executor.ipc import ZeroMqQueue
    return ZeroMqQueue((addr, hmac_key),
                       is_server=False,
                       socket_type=zmq.PAIR,
                       use_hmac_encryption=True)


def _bare_server_on(addr: str, hmac_key: bytes):
    """A RemoteMpiCommSessionServer with a real control socket and no MPI."""
    from tensorrt_llm.executor.ipc import ZeroMqQueue
    server = object.__new__(RemoteMpiCommSessionServer)
    server.addr = addr
    server.queue = ZeroMqQueue((addr, hmac_key),
                               is_server=True,
                               socket_type=zmq.PAIR,
                               use_hmac_encryption=True)
    return server


@pytest.fixture
def _spawn_proxy_ipc(monkeypatch, tmp_path: Path):
    """The IPC address and HMAC key trtllm-llmapi-launch exports."""
    from tensorrt_llm.executor import utils as executor_utils
    addr = f"ipc://{tmp_path / 'server.sock'}"
    hmac_key = os.urandom(32)
    monkeypatch.setenv("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR", addr)
    monkeypatch.setattr(executor_utils, "_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY",
                        hmac_key)
    return addr, hmac_key


@pytest.mark.cpu_only
@pytest.mark.parametrize("stop_client", ["module", "standalone"])
def test_stop_request_reaches_server_still_waiting_for_workers(
        _spawn_proxy_ipc, monkeypatch,
        stop_client: Literal["module", "standalone"]) -> None:
    """The stop client must stop a server whose workers are still running.

    This is ``trtllm-llmapi-launch``'s teardown: the engine (the PAIR client
    that submitted the worker task) exits, then the stop client connects and
    asks the server to shut down while the server is still waiting for the
    task's worker futures. A PAIR socket only takes the new client once it has
    processed the engine's disconnect, so a single request sent right away was
    dropped and the server, and every follower rank, hung. The request must
    also be acknowledged during the wait: the client gives up on an attempt
    after ``ack_timeout`` and reconnects, so an acknowledgment sent after the
    wait finds no peer and the client resends until it runs out of attempts.
    """
    from concurrent.futures import Future

    from tensorrt_llm.llmapi import mpi_session
    from tensorrt_llm.llmapi.mpi_session import RemoteTask

    addr, hmac_key = _spawn_proxy_ipc
    server = _bare_server_on(addr, hmac_key)
    worker_future = Future()
    stopped = threading.Event()

    class _Session:
        """The MPI session the server drives, with one worker."""
        n_workers = 1

        def submit(self, *args, **kwargs):
            return [worker_future]

        def shutdown(self) -> None:
            stopped.set()

        def abort(self) -> None:
            pytest.fail("worker drain unexpectedly required MPI abort")

    server.session = _Session()
    monkeypatch.setattr(mpi_session, "mpi_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(RemoteMpiCommSessionServer,
                        "_close_global_comm_executor",
                        staticmethod(lambda: None))
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    try:
        engine = _pair_client(addr, hmac_key)
        engine.put(RemoteTask(print, (), {}))
        engine.close()
        # The worker keeps running until the stop client has its answer.
        assert _request_stop(stop_client,
                             addr,
                             hmac_key,
                             ack_timeout=0.5,
                             max_attempts=20)
        assert not stopped.is_set()
        worker_future.set_result(None)
        assert stopped.wait(10)
    finally:
        if not worker_future.done():
            worker_future.set_result(None)
        server_thread.join(10)
        # Close the socket only once the thread using it has exited.
        if not server_thread.is_alive():
            server.queue.close()


@pytest.mark.cpu_only
@pytest.mark.parametrize("stop_client", ["module", "standalone"])
def test_stop_request_reads_past_other_messages_to_the_ack(
        _spawn_proxy_ipc, stop_client: Literal["module", "standalone"]) -> None:
    """A worker message sent to the stop client must not hide the ack."""
    from tensorrt_llm.llmapi.mpi_session import RemoteWorkerDeath

    addr, hmac_key = _spawn_proxy_ipc
    server = _bare_server_on(addr, hmac_key)
    stopped = threading.Event()

    def answer_stop_request():
        # Bounded, so a missing request ends the thread instead of leaving it
        # blocked on the socket.
        if server.queue.poll(10) and server.queue.get() is None:
            server.queue.put(
                RemoteWorkerDeath.from_exception(RuntimeError("rank 1 failed")))
            server.queue.put(RemoteMpiCommSessionServer.SHUTDOWN_ACK)
            stopped.set()

    server_thread = threading.Thread(target=answer_stop_request, daemon=True)
    server_thread.start()
    try:
        # One attempt only: a client that read just the first message would
        # miss the ack and give up.
        assert _request_stop(stop_client,
                             addr,
                             hmac_key,
                             ack_timeout=5.0,
                             max_attempts=1)
        assert stopped.wait(10)
    finally:
        server_thread.join(15)
        # Close the socket only once the thread using it has exited.
        if not server_thread.is_alive():
            server.queue.close()


@pytest.mark.cpu_only
def test_stop_request_gives_up_without_a_server(_spawn_proxy_ipc) -> None:
    from tensorrt_llm.llmapi.mgmn_leader_node import stop_server_main

    start = time.monotonic()
    assert not stop_server_main(ack_timeout=0.1, max_attempts=3)
    # Every attempt is dropped with linger 0, so nothing blocks the exit.
    assert time.monotonic() - start < 5


@pytest.mark.cpu_only
@pytest.mark.parametrize("acknowledged, exit_code", [(True, 0), (False, 1)])
def test_stop_cli_exit_code_reports_the_ack(monkeypatch, acknowledged,
                                            exit_code) -> None:
    from click.testing import CliRunner

    from tensorrt_llm.llmapi import mgmn_leader_node

    monkeypatch.setattr(mgmn_leader_node, "stop_server_main",
                        lambda: acknowledged)
    result = CliRunner().invoke(mgmn_leader_node.main, ["--action", "stop"])
    assert result.exit_code == exit_code

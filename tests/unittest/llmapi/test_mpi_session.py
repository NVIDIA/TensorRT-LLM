import os
import subprocess  # nosec B404
import sys
import threading
from subprocess import PIPE, Popen
from typing import Literal

cur_dir = os.path.dirname(os.path.abspath(__file__))

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             split_mpi_env)

# isort: off
sys.path.append(os.path.join(cur_dir, '..'))
from utils.util import skip_single_gpu
# isort: on


def task0():
    if MPINodeState.state is None:
        MPINodeState.state = 0
    MPINodeState.state += 1
    return MPINodeState.state


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


@pytest.mark.parametrize("task_type", ["submit", "submit_sync"])
def test_remote_mpi_session(task_type: Literal["submit", "submit_sync"]):
    """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, "_test_remote_mpi_session.sh")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = ["bash", test_file, task_type]
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


def task1():
    non_mpi_env, mpi_env = split_mpi_env()
    assert non_mpi_env
    assert mpi_env


def test_split_mpi_env():
    session = MpiPoolSession(n_workers=4)
    session.submit_sync(task1)


@skip_single_gpu
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

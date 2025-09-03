import os
import subprocess  # nosec B404
import sys
import threading
from subprocess import PIPE, Popen
from typing import Literal

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             split_mpi_env)


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


def run_client(server_addr, values_to_process):
    """Function to run in a separate process that creates a client and submits tasks"""
    try:
        client = RemoteMpiCommSessionClient(server_addr)

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

import os
import subprocess  # nosec B404

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (MPINodeState,
                                             RemoteMpiCommSessionClient)


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


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5179666")
def test_remote_mpi_session():
    """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""
    command = [
        "mpirun", "--allow-run-as-root", "-np", "4", "trtllm-llmapi-launch",
        "python3", "_run_mpi_comm_task.py"
    ]
    subprocess.run(command,
                   check=True,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   env=os.environ)  # nosec B603

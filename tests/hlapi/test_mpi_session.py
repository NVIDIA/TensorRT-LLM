import os
import subprocess  # nosec B404

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.hlapi.mpi_session import MPINodeState


def task0():
    if MPINodeState.state is None:
        MPINodeState.state = 0
    MPINodeState.state += 1
    return MPINodeState.state


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_session_basic():
    from tensorrt_llm.hlapi.mpi_session import MpiPoolSession

    n_workers = 4
    executor = MpiPoolSession(n_workers)
    results = executor.submit_sync(task0)
    assert results == [1, 1, 1, 1], results

    results = executor.submit_sync(task0)
    assert results == [2, 2, 2, 2], results


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_session_multi_node():
    nworkers = 4
    test_case_file = os.path.join(os.path.dirname(__file__), "mpi_test_task.py")
    command = f"mpirun --allow-run-as-root -n {nworkers} python {test_case_file}"
    subprocess.run(command, shell=True, check=True,
                   env=os.environ)  # nosec B603


if __name__ == '__main__':
    test_mpi_session_basic()
    test_mpi_session_multi_node()

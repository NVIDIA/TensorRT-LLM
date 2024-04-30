import pickle  # nosec B403
import socket
import sys
from concurrent.futures import Future
from typing import Any, List, Optional

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

if ENABLE_MULTI_DEVICE:
    from mpi4py.futures import MPIPoolExecutor


class MPINodeState:
    ''' MPINodeState acts as a central global state shares between tasks on MPI node.

    An example:
        def task():
            if MPINodeState.state is None:
                MPINodeState.state = 0
            MPINodeState.state += 1
            return MPINodeState.state

        n_workers = 4
        with MPIPoolExecutor(max_workers=n_workers) as executor:
            for i in range(2):
                futures = [executor.submit(task) for i in range(n_workers)]

        This should produce the following output:
        - [1, 1, 1, 1]
        - [2, 2, 2, 2]
    '''

    state = None

    @staticmethod
    def is_initialized() -> bool:
        return MPINodeState.state is not None


class MpiSession:

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self._start_mpi_pool()

    def submit(self, task: (...), *args, **kwargs) -> List[Future]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: (...), *args, **kwargs) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown()
            self.mpi_pool = None

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers,
                                        path=sys.path)

    def __del__(self):
        self.shutdown()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

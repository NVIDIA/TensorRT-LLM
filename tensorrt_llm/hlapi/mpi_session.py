from concurrent.futures import Future
from typing import Any, List, Optional

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

comm = MPI.COMM_WORLD


class NodeSession:
    ''' NodeSession Act as a central global state shares between tasks on MPI node.

    An example:
        def task():
            if NodeSession.state is None:
                NodeSession.state = 0
            NodeSession.state += 1
            return NodeSession.state

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
        return NodeSession.state


class MpiSession:

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self._start()

    def submit(self, task: (...), *args) -> List[Future]:
        return [
            self.mpi_pool.submit(task, *args) for i in range(self.n_workers)
        ]

    def submit_sync(self, task: (...), *args) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args) for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        assert self.mpi_pool, 'MPI session not started'
        self.mpi_pool.shutdown()
        self.mpi_pool = None

    def _start(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers)


def mpi_rank() -> int:
    return comm.Get_rank()


def mpi_size() -> int:
    return comm.Get_size()

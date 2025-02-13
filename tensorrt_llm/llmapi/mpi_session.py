import abc
import socket
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, TypeVar

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

if ENABLE_MULTI_DEVICE:
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

    from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size

T = TypeVar("T")


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


def external_mpi_comm_available(model_world_size: int) -> bool:
    ''' Check if the current process is launched by mpirun and does not use MPIPoolExecutor to spawn processes.
    e.g. mpirun -np 4 python script.py
    '''
    if ENABLE_MULTI_DEVICE:
        return mpi_world_size() == model_world_size and model_world_size > 1
    else:
        return False


def need_spawn_mpi_workers(model_world_size: int) -> bool:
    ''' Check if the current process needs to spawn MPI workers. '''
    if ENABLE_MULTI_DEVICE:
        return mpi_world_size() == 1 and model_world_size > 1
    else:
        return False


class MpiSession(abc.ABC):

    @abc.abstractmethod
    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        raise NotImplementedError()

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError()


class MpiPoolSession(MpiSession):

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self._start_mpi_pool()

    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown(wait=False)
            self.mpi_pool = None

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers,
                                        path=sys.path)

    def __del__(self):
        self.shutdown()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class MpiCommSession(MpiSession):

    def __init__(self, n_workers: int = 1):
        if n_workers <= 0:
            raise ValueError(
                f'n_workers must be non-negative, but got {n_workers}')
        if n_workers != mpi_world_size():
            raise ValueError(
                f'n_workers must be equal to the number of processes launched by mpirun, got {n_workers} vs {mpi_world_size()}'
            )

        if mpi_rank() != 0:
            raise RuntimeError(
                f'only rank 0 can start multi-node session, got {mpi_rank()}')
        if not external_mpi_comm_available(n_workers):
            raise RuntimeError('The LLM instance should be launched by mpirun.')

        self.n_workers = n_workers
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.mpi_pool: Optional[MPIPoolExecutor] = None

        self._start_mpi_pool()

    def submit(self,
               task: (...),
               rank0_extra_kwargs: Optional[dict] = None,
               *args,
               **kwargs) -> List[Future]:
        ''' Submit a task to MPI workers.

        Args:
            task: The task to be submitted.
            rank0_extra_kwargs: Extra keyword arguments for rank0 task.
            args: Positional arguments for the task.
            kwargs: Keyword arguments for the task.
        '''
        assert self.mpi_pool is not None, 'MPI session not started'
        # Trick: The MPICommExecutor excludes rank0 from workers, thus an extra
        # task dispatching to rank0 is needed
        rank0_extra_kwargs = rank0_extra_kwargs or {}

        rank0_params = kwargs.copy()
        if rank0_extra_kwargs:
            rank0_params.update(rank0_extra_kwargs)

        worker_futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers - 1)
        ]

        # A trick to wait for rank0 to be ready, or the collective tasks will hang
        # TODO[chunweiy]: Remove this trick
        time.sleep(10)

        rank0_future = self.thread_pool.submit(task, *args, **rank0_params)
        return [rank0_future] + worker_futures

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = self.submit(task, *args, **kwargs)
        return [future.result() for future in futures]

    def shutdown(self):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown(wait=False)
            self.mpi_pool = None

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        comm_executor = MPICommExecutor(mpi_comm())
        self.mpi_pool = comm_executor.__enter__()

    def __del__(self):
        self.shutdown()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

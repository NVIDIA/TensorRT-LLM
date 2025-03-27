import abc
import itertools
import os
import socket
import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import zmq

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

from .._utils import global_mpi_rank, mpi_comm
from .utils import print_colored_debug

if ENABLE_MULTI_DEVICE:
    import mpi4py
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

    from tensorrt_llm._utils import global_mpi_size, mpi_rank, mpi_world_size

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
        return (get_mpi_world_size() == model_world_size
                and model_world_size > 1) or (global_mpi_size()
                                              > get_mpi_world_size())
    else:
        return False


def need_spawn_mpi_workers(model_world_size: int) -> bool:
    ''' Check if the current process needs to spawn MPI workers. '''
    if ENABLE_MULTI_DEVICE:
        return get_mpi_world_size() == 1 and model_world_size > 1
    else:
        return False


def set_mpi_session_cpp(comm):
    if ENABLE_MULTI_DEVICE:
        comm_fortran = comm.py2f()
        from tensorrt_llm.bindings import MpiComm
        MpiComm.set_raw_mpi_session_by_fortran_handle(comm_fortran)


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

    @abc.abstractmethod
    def barrier(self):
        raise NotImplementedError()


class MpiPoolSession(MpiSession):

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self._start_mpi_pool()
        if ENABLE_MULTI_DEVICE:
            self.comm = mpi4py.MPI.COMM_WORLD

    def get_comm(self):
        return self.comm

    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def barrier(self):
        # MPIPoolExecutor does not need a barrier, all the MPI processes are spawned by the same mpirun command
        pass

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

    def __init__(self, comm=None, n_workers: int = 1):
        if not external_mpi_comm_available(n_workers):
            raise RuntimeError('The LLM instance should be launched by mpirun.')

        self.comm = comm
        self.n_workers = n_workers
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.mpi_pool: Optional[MPIPoolExecutor] = None

        if n_workers <= 0:
            raise ValueError(
                f'n_workers must be non-negative, but got {n_workers}')

        if ENABLE_MULTI_DEVICE:
            if not self.comm:
                self.comm = mpi_comm()

            if self.comm.Get_rank() != 0:
                raise RuntimeError(
                    f'only rank 0 can start multi-node session, got {self.comm.Get_rank()}'
                )

            if self.comm.Get_size() != n_workers:
                raise ValueError(
                    f'n_workers must be equal to the number of processes in MPI, got {n_workers} vs {get_mpi_world_size()}'
                )

        self._start_mpi_pool()

    def get_comm(self):
        return self.comm

    def submit(self, task: Callable[..., T], *args,
               **kwargs) -> List[Future[T]]:
        ''' Submit a task to MPI workers.

        Args:
            task: The task to be submitted.
            args: Positional arguments for the task.
            kwargs: Keyword arguments for the task.
        '''
        assert self.mpi_pool is not None, 'MPI session not started'
        worker_futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers - 1)
        ]

        rank0_future = self.thread_pool.submit(task, *args, **kwargs)
        return [rank0_future] + worker_futures

    @staticmethod
    def _mpi_barrier():
        print_colored_debug("MPI barrier\n", "yellow")
        mpi_comm().Barrier()
        print_colored_debug("MPI barrier done\n", "yellow")

    def barrier(self):
        # Wait until the MPI processes are ready
        assert mpi_rank() == 0, 'Barrier should be called by rank0'
        self.submit_sync(MpiCommSession._mpi_barrier)

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
        comm_executor = MPICommExecutor(self.comm)
        self.mpi_pool = comm_executor.__enter__()

    def __del__(self):
        self.shutdown()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class RemoteTask(NamedTuple):
    task: Callable[..., T] | str  # str for special tasks like barrier
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class RemoteMpiCommSessionClient():
    '''
    RemoteMpiCommSessionClient is a variant of MpiCommSession that is used to connect to a remote MPI pool.
    '''

    def __init__(self, addr: str):
        print_colored_debug(
            f"RemoteMpiCommSessionClient connecting to {addr}\n", "yellow")
        self.addr = addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(addr)
        self._is_shutdown = False

    def barrier(self):
        self.socket.send_pyobj(RemoteTask(MpiCommSession._mpi_barrier, (), {}))

    def submit(self, task: Callable[..., T], *args, **kwargs) -> list:
        if self._is_shutdown:
            print_colored_debug(
                "RemoteMpiCommSessionClient is already shut down\n", "yellow")
            return []
        print_colored_debug(
            f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] Sending task {task} to {self.addr}\n",
            "yellow")
        self.socket.send_pyobj(RemoteTask(task, args, kwargs))
        return []

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> list:
        return self.submit(task, *args, **kwargs)

    def shutdown(self):
        if self._is_shutdown:
            return

        try:
            print_colored_debug(
                f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] send shutdown signal to server\n",
                "green")
            self.socket.send_pyobj(
                None)  # ask RemoteMpiCommSessionServer to shutdown
            self.socket.close()
            self.context.term()
        except zmq.error.ZMQError as e:
            print_colored_debug(
                f"Error during RemoteMpiCommSessionClient shutdown: {e}\n",
                "red")
        finally:
            self._is_shutdown = True


class RemoteMpiCommSessionServer():
    '''
    RemoteMpiCommSessionServer is a variant of MpiCommSession that is used to create a remote MPI pool.
    '''

    def __init__(self,
                 n_workers: int = 0,
                 addr: str = f'tcp://127.0.0.1:*',
                 comm=None,
                 is_comm: bool = False):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(addr)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.comm = comm

        if self.comm is not None:
            self.session = MpiCommSession(n_workers=self.comm.Get_size(),
                                          comm=self.comm)
        else:
            self.session = MpiCommSession(
                n_workers=n_workers) if is_comm else MpiPoolSession(
                    n_workers=n_workers)

    def serve(self):
        print_colored_debug(
            f"RemoteMpiCommSessionServer listening on {self.addr}\n", "yellow")
        while True:
            message: Optional[RemoteTask] = self.socket.recv_pyobj()
            if message is None:
                print_colored_debug(
                    f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received shutdown signal\n",
                    "green")
                self.session.shutdown()
                break
            else:
                if message.task == MpiCommSession._mpi_barrier:
                    # This will block the server until the barrier is done
                    self.session.submit_sync(message.task, *message.args,
                                             **message.kwargs)
                else:
                    print_colored_debug(
                        f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received task from {self.addr}\n",
                        "green")
                self.session.submit(message.task, *message.args,
                                    **message.kwargs)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_mpi_world_size() -> int:
    # avoid cyclic import
    from ..executor.utils import get_spawn_proxy_process_env

    # If the proxy process is spawned, the MPI-related env will be cleaned in the proxy process, thus we made another env for the mpi_world_size
    if get_spawn_proxy_process_env():
        return int(os.getenv("tllm_mpi_size") or 1)
    else:
        return mpi_world_size()


def split_mpi_env(mpi_env_keys: List[str] | None = None) -> Tuple[dict, dict]:
    '''
    Splits the environment variables into MPI-related and non-MPI-related dictionaries.

    Args:
        mpi_env_keys: Additional environment variables to be considered as MPI-related.

    Returns:
        Tuple[dict, dict]: (non_mpi_env, mpi_env)
            - non_mpi_env: Environment dictionary without MPI-related variables
            - mpi_env: Environment dictionary containing only MPI-related variables
    '''
    current_env = os.environ.copy()

    # Identify MPI-related variables
    mpi_vars = set(
        itertools.chain([
            var for var in current_env if var.startswith((
                'MPI_',
                'OMPI_',
                'PMIX_',
                'PMI_',
                'OMPI_',
                'PMIX_',
                'PMI_',
                'SLURM_',
                'MPI_',
                'UCX_',
                'I_MPI_',
                'HYDRA_',
                'KMP_',
                'MPICH_',
                'MV2_',
                'CRAY_',
            ))
        ], mpi_env_keys or []))

    # Split into two dictionaries
    non_mpi_env = {k: v for k, v in current_env.items() if k not in mpi_vars}
    mpi_env = {k: v for k, v in current_env.items() if k in mpi_vars}

    return non_mpi_env, mpi_env

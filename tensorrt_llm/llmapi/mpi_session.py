import abc
import itertools
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import zmq

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.logger import logger

from .._utils import global_mpi_rank, mpi_barrier, mpi_rank
from .utils import logger_debug, print_colored

if ENABLE_MULTI_DEVICE:
    import mpi4py
    from mpi4py.futures import MPICommExecutor, MPIPoolExecutor

    from tensorrt_llm._utils import global_mpi_size, mpi_world_size

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
    def shutdown(self, wait=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def abort(self):
        raise NotImplementedError()

    def is_comm_session(self) -> bool:
        return isinstance(self, (MpiCommSession, RemoteMpiCommSessionClient))

    def _abort_on_timeout(self, fut: Future, timeout: float, reason=None):
        try:
            fut.result(timeout=timeout)
        except TimeoutError:
            logger.critical("MpiSession shutdown timeout, aborting...")
            if reason is not None:
                logger.info(f"Reason to shutdown: {repr(reason)}")
            self.abort()

    def shutdown_abort(self, grace: float = 60, reason=None):
        if sys.is_finalizing():
            # cannot start thread at interpreter shutdown
            # simply don't wait to avoid hang
            return self.shutdown(wait=False)

        fut = Future()
        killer = threading.Thread(group=None,
                                  target=self._abort_on_timeout,
                                  name="MpiSessionTimeoutKiller",
                                  args=(fut, grace, reason))
        killer.start()
        self.shutdown()
        fut.set_result(None)
        killer.join()


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

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown(wait=wait)
            self.mpi_pool = None

    def abort(self):
        self.get_comm().Abort(1)

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers,
                                        path=sys.path)

    def __del__(self):
        self.shutdown_abort()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class MpiCommSession(MpiSession):

    def __init__(self, comm=None, n_workers: int = 1):
        self.comm = comm
        self.n_workers = n_workers
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.mpi_pool: Optional[MPIPoolExecutor] = None

        if n_workers <= 0:
            raise ValueError(
                f'n_workers must be non-negative, but got {n_workers}')

        if ENABLE_MULTI_DEVICE:
            if not self.comm:
                self.comm = mpi4py.MPI.COMM_WORLD

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

    def submit_sync(self, task: Callable[..., T], *args, **kwargs) -> List[T]:
        futures = self.submit(task, *args, **kwargs)
        return [future.result() for future in futures]

    def shutdown(self, wait=True):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown(wait=wait)
            self.mpi_pool = None
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=wait)
            self.thread_pool = None

    def abort(self):
        self.get_comm().Abort(1)

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        comm_executor = MPICommExecutor(self.comm)
        self.mpi_pool = comm_executor.__enter__()

    def __del__(self):
        self.shutdown_abort()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class RemoteTask(NamedTuple):
    task: Callable[..., T]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    sync: bool = False  # if True, the result will be sent back to the client


class RemoteMpiCommSessionClient(MpiSession):
    '''
    RemoteMpiCommSessionClient is a variant of MpiCommSession that is used to connect to a remote MPI pool.
    '''

    def __init__(self, addr: str, hmac_key: Optional[bytes] = None):
        # FIXME: this is a hack to avoid circular import, resolve later
        from tensorrt_llm.executor.ipc import ZeroMqQueue
        self.addr = addr
        logger_debug(f"RemoteMpiCommSessionClient connecting to {addr}\n",
                     "yellow")
        self.queue = ZeroMqQueue((addr, hmac_key),
                                 is_server=False,
                                 socket_type=zmq.PAIR,
                                 use_hmac_encryption=bool(hmac_key))
        self._is_shutdown = False

    def submit(self,
               task: Callable[..., T],
               *args,
               sync: bool = False,
               **kwargs) -> list:
        ''' Submit a task to the remote MPI pool. '''
        if self._is_shutdown:
            logger_debug("RemoteMpiCommSessionClient is already shut down\n",
                         "yellow")
            return []
        logger_debug(
            f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] sending task {task} to {self.addr}\n",
            "yellow")
        self.queue.put(RemoteTask(task, args, kwargs, sync=sync))
        return []

    SYNC_IDLE_INTERVAL = 8

    def submit_sync(self, task, *args, **kwargs) -> List[T]:
        ''' Submit a task to the remote MPI pool and wait for task completion. '''
        self.submit(task, *args, sync=True, **kwargs)

        while not ((res := self.poll()) or self._is_shutdown):
            logger_debug(f"Waiting for task completion... {res}\n", "grey")
            time.sleep(self.SYNC_IDLE_INTERVAL)

        logger_debug(
            f"rank{global_mpi_rank()} RemoteMpiCommSessionClient.send_sync received results: {res}\n",
            "green")

        if not res:
            raise RuntimeError(
                "RemoteMpiCommSessionClient received unexpected response")
        return res

    def poll(self) -> bool:
        ''' Poll the queue for a response.
        Returns:
            True if a response is received, False otherwise.
        '''
        if self._is_shutdown:
            return False
        response = self.queue.poll(0.1)
        if response:
            return self.queue.get()  # should get a True if success
        return False

    def abort(self):
        self.shutdown()

    def shutdown(self, wait=True):
        pass

    def shutdown_abort(self, grace: float = 60, reason=None):
        pass


class RemoteMpiCommSessionServer():
    '''
    RemoteMpiCommSessionServer is a variant of MpiCommSession that is used to create a remote MPI pool.
    '''

    def __init__(self,
                 n_workers: int = 0,
                 addr: str = f'tcp://127.0.0.1:*',
                 hmac_key: Optional[bytes] = None,
                 comm=None,
                 is_comm: bool = False):
        # FIXME: this is a hack to avoid circular import, resolve later
        from tensorrt_llm.executor.ipc import ZeroMqQueue
        self.addr = addr
        self.queue = ZeroMqQueue((addr, hmac_key),
                                 is_server=True,
                                 socket_type=zmq.PAIR,
                                 use_hmac_encryption=bool(hmac_key))
        self.comm = comm
        self.results = []  # the results may arrive in any order

        if self.comm is not None:
            self.session = MpiCommSession(n_workers=self.comm.Get_size(),
                                          comm=self.comm)
        else:
            self.session = MpiCommSession(
                n_workers=n_workers) if is_comm else MpiPoolSession(
                    n_workers=n_workers)

    @staticmethod
    def task_wrapper(task: Callable[..., T], *args, **kwargs) -> T:
        logger_debug(
            f"MpiCommSession rank{mpi_rank()} with world_size {mpi_world_size()}\n",
            "green")
        logger_debug(
            f"MpiCommSession rank{mpi_rank()} start task [{task}] with args: {args} and kwargs: {kwargs}\n",
            "green")

        # wait for all ranks to start the task
        mpi_barrier()

        try:
            return task(*args, **kwargs)
        except Exception as e:
            print_colored(
                f"MpiCommSession rank{mpi_rank()} task [{task}] failed with exception: {e}\n",
                "red")
            traceback.print_exc()
            raise e
        finally:
            logger_debug(
                f"MpiCommSession rank{mpi_rank()} task [{task}] finished\n",
                "green")
            mpi_barrier()

    def serve(self):
        logger_debug(f"RemoteMpiCommSessionServer listening on {self.addr}\n",
                     "yellow")
        while True:
            message: Optional[RemoteTask] = self.queue.get()
            if message is None:
                logger_debug(
                    f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received shutdown signal\n",
                    "green")
                self.session.shutdown_abort()
                break
            else:
                logger_debug(
                    f"RemoteMpiCommSessionServer [rank{global_mpi_rank()}] received task [{message.task}] from {self.addr}\n",
                    "green")
                futures = self.session.submit(
                    RemoteMpiCommSessionServer.task_wrapper, message.task,
                    *message.args, **message.kwargs)
                self.num_results = self.session.n_workers
                assert len(futures) == self.num_results == mpi_world_size()
                if message.sync:
                    for future in futures:
                        future.add_done_callback(self.mpi_future_callback)

    def mpi_future_callback(self, future):
        logger_debug(f"rank{global_mpi_rank()} got future: {future}\n", "red")
        if future.exception() is not None:
            logger_debug(
                f"mpi_future got exception: {future.exception()}, quitting\n",
                "red")
            self.queue.put(future.exception())
            return

        result = future.result()
        self.results.append(result)
        logger_debug(
            f"RemoteMpiCommSessionServer working status: {len(self.results)}/{self.num_results}\n",
            "grey")
        if len(self.results) == self.num_results:
            logger_debug(
                f"RemoteMpiCommSessionServer received all results, sending to client\n",
                "green")
            try:
                self.queue.put_noblock(self.results, retry=2)
            except zmq.ZMQError as e:
                # The client could be shutdown first.
                if e.errno == zmq.EAGAIN:
                    pass
                else:
                    raise e

            logger_debug(f"RemoteMpiCommSessionServer sent results to client\n",
                         "green")
            self.results.clear()


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

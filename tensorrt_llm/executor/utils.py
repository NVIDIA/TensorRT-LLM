import asyncio
import concurrent.futures
import os
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, List, NamedTuple, Optional

from strenum import StrEnum

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi.utils import enable_llm_debug, logger_debug

from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient)
from ..llmapi.utils import logger_debug
from ..logger import logger


class LlmLauncherEnvs(StrEnum):
    # Spawn a process for the LLM-API Proxy
    TLLM_SPAWN_PROXY_PROCESS = "TLLM_SPAWN_PROXY_PROCESS"
    TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR = "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"
    TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"

    # Whether to use periodical responses handler in await_responses
    TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT = "TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT"


def get_spawn_proxy_process_ipc_addr_env() -> str | None:
    ''' Get the IPC address for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR)


def get_spawn_proxy_process_ipc_hmac_key_env() -> bytes:
    ''' Get the HMAC key for the spawn proxy process dynamically. '''
    key = os.getenv("TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY")
    assert key is not None, (
        f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY} is not set. "
        "HMAC encryption is required for IPC communication.")
    return bytes.fromhex(key)


def get_spawn_proxy_process_env() -> bool:
    ''' Get the environment variable for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS) == "1"


# ---------------------------------------------------------------------------
# In-process rank-0 worker registry (RPC local fast-path).
#
# Under MpiCommSession the rank-0 RpcWorker runs in a thread of the proxy
# process, yet the proxy still reaches it over a loopback ZMQ socket
# (pickle + HMAC + dispatch) -- pure self-inflicted overhead that competes for
# the GIL with the co-located executor loop. The worker registers itself here
# keyed by its rpc_addr (which the proxy also knows), so the proxy can call it
# directly. A lookup miss -- e.g. the spawn-proxy topology where the worker
# lives in a different process, so this process's registry does not contain it
# -- transparently falls back to RPC. This makes the local/remote decision
# automatic per-process, with no explicit topology detection.
# ---------------------------------------------------------------------------
_local_rpc_workers: dict = {}
_local_rpc_workers_lock = threading.Lock()


def register_local_rpc_worker(rpc_addr: str, worker: Any) -> None:
    with _local_rpc_workers_lock:
        _local_rpc_workers[rpc_addr] = worker


def unregister_local_rpc_worker(rpc_addr: str) -> None:
    with _local_rpc_workers_lock:
        _local_rpc_workers.pop(rpc_addr, None)


def get_local_rpc_worker(rpc_addr: str) -> Optional[Any]:
    with _local_rpc_workers_lock:
        return _local_rpc_workers.get(rpc_addr)


def create_mpi_comm_session(
        n_workers: int) -> RemoteMpiCommSessionClient | MpiPoolSession:
    assert mpi_rank(
    ) == 0, f"create_mpi_comm_session must be called by rank 0, but it was called by rank {mpi_rank()}"
    if get_spawn_proxy_process_env():
        assert get_spawn_proxy_process_ipc_addr_env(
        ), f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR} is not set."
        logger_debug(
            f"Using RemoteMpiPoolSessionClient to bind to external MPI processes at {get_spawn_proxy_process_ipc_addr_env()}\n",
            "yellow")
        hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
        return RemoteMpiCommSessionClient(
            addr=get_spawn_proxy_process_ipc_addr_env(), hmac_key=hmac_key)
    else:
        logger_debug(
            f"Using MpiCommSession to bind to external MPI processes\n",
            "yellow")
        return MpiCommSession(n_workers=n_workers)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class RequestError(RuntimeError):
    ''' The error raised when the request is failed. '''


class ProcessPoolExecutorSession(MpiSession):
    # This process pool is introduced for better recoverable exceptions handling.
    # It replaces MpiPoolExecutor for single-gpu case.

    def __init__(self, n_workers: int, **kwargs):
        self.n_workers = n_workers
        self.mpi_pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                            **kwargs)

    def submit(self, task: Callable, *args,
               **kwargs) -> List[concurrent.futures.Future]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable, *args, **kwargs) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        self.mpi_pool.shutdown(wait=True)


class ErrorResponse(NamedTuple):
    client_id: int
    error_msg: str
    request_id: int


class IntraProcessQueue:
    ''' A Queue-like container for IPC within the same process. '''

    def __init__(self):
        self.queue = Queue()

    def put(self, obj: Any):
        self.queue.put(obj)

    def get(self, timeout=None) -> Any:
        return self.queue.get(timeout=timeout)

    def close(self):
        pass

    def drain(self) -> list:
        """Non-blocking drain: return all currently available messages."""
        results = []
        while True:
            try:
                results.append(self.queue.get_nowait())
            except Empty:
                break
        return results

    def poll(self, timeout=None) -> bool:
        with self.queue.not_empty:
            if self.queue._qsize() > 0:
                return True
            if timeout is not None and timeout > 0:
                self.queue.not_empty.wait(timeout=timeout)
                return self.queue._qsize() > 0
            return False


class WorkerCommIpcAddrs(NamedTuple):
    ''' IPC addresses (str) and HMAC keys (bytes) for communication with the worker processes. '''
    request_queue_addr: tuple[str, Optional[bytes]]
    worker_init_status_queue_addr: tuple[str, Optional[bytes]]
    result_queue_addr: tuple[str, Optional[bytes]]
    resource_governor_queue_addr: Optional[tuple[str, Optional[bytes]]] = None


def is_llm_response(instance):
    # Duck typing, expect one of:
    #  tensorrt_llm.bindings.executor.Response
    #  tensorrt_llm._torch.pyexecutor.llm_request.LlmResponse
    # Avoid testing for "result", because an error bindings.executor.Response
    # throws when accessing its result property.
    return hasattr(instance, "has_error")


def print_alive_threads():
    assert enable_llm_debug(
    ), "print_alive_threads must be called with enable_llm_debug() enabled"

    # Print all alive threads for debugging
    alive_threads = [t for t in threading.enumerate() if t.is_alive()]
    logger.info(
        f'All alive threads after shutdown: {[t.name for t in alive_threads]}\n',
        "red")
    for t in alive_threads:
        logger.info(f'Thread {t.name} (daemon={t.daemon}) is still alive')
        # Get the stack trace for this thread
        stack = sys._current_frames().get(t.ident)
        if stack is not None:
            logger.info(f'Stack trace for thread {t.name}:')
            traceback.print_stack(stack, file=sys.stdout)
            logger.info('')

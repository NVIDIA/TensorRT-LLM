# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
import ctypes
import os
import re
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


_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY: bytes | None = None
_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _scrub_process_env_value(key_name: str, value: str) -> None:
    if sys.platform != "linux":
        return

    libc = ctypes.CDLL(None, use_errno=True)
    libc.getenv.restype = ctypes.c_void_p
    value_ptr = libc.getenv(os.fsencode(key_name))
    if value_ptr:
        ctypes.memset(value_ptr, 0, len(os.fsencode(value)))


def _normalize_spawn_proxy_process_ipc_hmac_key(key: str) -> bytes:
    if not _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_PATTERN.fullmatch(key):
        raise ValueError("IPC HMAC key must be a 64-character hex string.")

    try:
        key_bytes = bytes.fromhex(key)
    except ValueError as exc:
        raise ValueError(
            "IPC HMAC key must be a 64-character hex string.") from exc

    if len(key_bytes) != 32:
        raise ValueError("IPC HMAC key must be 32 bytes.")
    return key_bytes


def get_spawn_proxy_process_ipc_addr_env() -> str | None:
    ''' Get the IPC address for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR)


def get_spawn_proxy_process_ipc_hmac_key_env() -> bytes:
    ''' Get the HMAC key for the spawn proxy process dynamically. '''
    global _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY
    if _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY is not None:
        return _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY

    env_name = LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    key = os.environ.get(env_name)
    if key is None:
        raise RuntimeError(
            f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY} is not set. "
            "HMAC encryption is required for IPC communication.")
    _scrub_process_env_value(env_name, key)
    os.environ.pop(env_name, None)

    _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = (
        _normalize_spawn_proxy_process_ipc_hmac_key(key))
    return _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY


def get_spawn_proxy_process_env() -> bool:
    ''' Get the environment variable for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS) == "1"


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


class EngineDeadError(RuntimeError):
    """Raised by pending and new requests once the engine is known dead.

    Sticky and engine-level (unlike the per-request ``RequestError``): when a
    worker process dies, every queued ``GenerationResult`` is unblocked with this
    error and every subsequent ``submit()`` raises it immediately, instead of
    blocking forever on a response queue whose producer is gone.
    """

    def __init__(self, root_cause: Optional[BaseException] = None):
        msg = "Engine has died"
        if root_cause is not None:
            msg += f": {type(root_cause).__name__}: {root_cause}"
        super().__init__(msg)
        self.root_cause = root_cause


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

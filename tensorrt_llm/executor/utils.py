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
    # Multi-frontend serving (classic IPC path): one result lane per frontend
    # process, selected by the frontend id in client_id's top bits. When set,
    # the rank0 worker BINDS the request queue (PULL) so every frontend can
    # PUSH-connect, and routes responses to these lanes instead of
    # result_queue_addr (which then aliases lane 0, the launcher's).
    frontend_result_queue_addrs: Optional[list[tuple[str,
                                                     Optional[bytes]]]] = None


# Multi-frontend client_id namespacing: the top bits of the uint64 client id
# carry the frontend id, the low FRONTEND_ID_SHIFT bits carry the per-frontend
# request counter. Frontend id 0 keeps client ids bit-identical to the legacy
# single-frontend scheme.
FRONTEND_ID_SHIFT = 48
FRONTEND_COUNTER_MASK = (1 << FRONTEND_ID_SHIFT) - 1


def get_frontend_id(client_id: Optional[int]) -> int:
    """Extract the originating frontend id from a namespaced client id."""
    if not isinstance(client_id, int):
        return 0
    return client_id >> FRONTEND_ID_SHIFT


def namespace_client_id(frontend_id: int, client_id: int) -> int:
    """Embed frontend_id in the top bits of a per-frontend client id."""
    return (frontend_id << FRONTEND_ID_SHIFT) | (client_id
                                                 & FRONTEND_COUNTER_MASK)


def frontend_lane_index(client_id: Optional[int], num_lanes: int) -> int:
    """The result-lane index for a response's originating frontend.

    Responses without a usable client_id (e.g. ADP dummy requests carry
    client_id=None) and ids with an out-of-range frontend go to lane 0
    (the launcher), matching legacy single-client visibility where such
    responses are silently discarded by the launcher's dispatcher.
    """
    frontend_id = get_frontend_id(client_id)
    return frontend_id if frontend_id < num_lanes else 0


def bucket_responses_by_frontend(responses: list,
                                 num_frontends: int) -> list[list]:
    """Bucket responses by their originating frontend id (client_id top bits).

    Lane selection (including the route-to-launcher fallback) follows
    frontend_lane_index.
    """
    buckets = [[] for _ in range(num_frontends)]
    for rsp in responses:
        buckets[frontend_lane_index(rsp.client_id,
                                    num_frontends)].append(rsp)
    return buckets


def get_num_serve_frontends() -> int:
    """The number of serving frontend processes (TLLM_SERVE_NUM_FRONTENDS).

    Single source for parsing and range-validating the env knob; used by
    trtllm-serve (the launcher) and GenerationExecutorProxy so they always
    agree on the frontend count. 1 (single frontend) when unset.
    """
    num_frontends = int(os.getenv("TLLM_SERVE_NUM_FRONTENDS", "1") or "1")
    if not 0 < num_frontends <= (1 << 16):
        raise ValueError(
            f"TLLM_SERVE_NUM_FRONTENDS out of range: {num_frontends}")
    return num_frontends


def get_multi_frontend_ipc_info() -> Optional[tuple[str, bytes]]:
    """The shared ipc dir and HMAC key for multi-frontend serving.

    Pre-generated by trtllm-serve (the launcher) on the classic IPC executor
    path before the executor is created. None outside that mode.
    """
    ipc_dir = os.getenv("TLLM_MULTI_FRONTEND_IPC_DIR")
    hmac_hex = os.getenv("TLLM_MULTI_FRONTEND_HMAC")
    if ipc_dir and hmac_hex:
        return ipc_dir, bytes.fromhex(hmac_hex)
    return None


def multi_frontend_request_addr(ipc_dir: str) -> str:
    """The request ingress endpoint bound by the rank0 worker (PULL).

    Every frontend PUSH-connects to it.
    """
    return f"ipc://{os.path.join(ipc_dir, 'request.sock')}"


def multi_frontend_result_addr(ipc_dir: str, frontend_id: int) -> str:
    """The result lane endpoint bound by frontend ``frontend_id`` (PULL).

    The worker/postproc processes PUSH-connect to it. Deterministic so a
    respawned process can rebind the same lane.
    """
    return f"ipc://{os.path.join(ipc_dir, f'result_{frontend_id}.sock')}"


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

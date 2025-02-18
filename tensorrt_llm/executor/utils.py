import asyncio
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, List, NamedTuple, Optional

import torch

from tensorrt_llm.logger import logger

from ..bindings import executor as tllm
from ..llmapi.mpi_session import MpiSession

BATCH_RESP_IN_AWAIT = os.getenv("TLLM_EXECUTOR_BATCH_RESP_IN_AWAIT") == "1"

if BATCH_RESP_IN_AWAIT:
    logger.info("Using batched responses in await_responses")


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
        self.mpi_pool.shutdown(wait=False)


class ExecutorResponseTensors(NamedTuple):
    output_token_ids: List[List[int]]
    # context_logits is a tensor or a string denoting the path to the shared memory.
    context_logits: Optional[torch.Tensor | str]
    # generation_logits is a tensor or a string denoting the path to the shared memory.
    generation_logits: Optional[torch.Tensor | str]
    log_probs: Optional[list]
    cum_log_probs: Optional[list]


class ExecutorResponse(NamedTuple):
    """ The response from the cpp-executor to the Python main thread. """
    client_id: int
    tensors: Optional[ExecutorResponseTensors]
    finish_reasons: Optional[List[tllm.FinishReason]]
    is_final: Optional[bool]
    sequence_index: Optional[int]
    # There are two types of errors:
    # 1. str for the errors from the cpp-executor.await_responses, this will be dispatched to the user's
    #    generate_async as a per-request error, and won't stop the whole service.
    # 2. Exception for the errors from the background threads/processes, this will be processed in the main thread,
    #    and stop the whole service.
    error: Optional[str | Exception]
    # The timestamp of the creation of the response. We use this to track the IPC overhead.
    timestamp: Optional[float] = None


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

    def poll(self, timeout=None) -> bool:
        try:
            # Try to get an item from the queue without blocking
            item = self.queue.get(timeout=timeout)
            # If successful, put the item back to not alter the state
            self.queue.put(item)
            return True
        except Empty:
            # If the queue thread is empty, return False
            return False


class WorkerCommIpcAddrs(NamedTuple):
    ''' IPC addresses for communication with the worker processes. '''
    request_queue_addr: str
    request_error_queue_addr: str
    result_queue_addr: str
    stats_queue_addr: str


class WorkerCommQueues(NamedTuple):
    ''' Queues for communication with the worker in the same process. '''
    request_queue: IntraProcessQueue
    request_error_queue: IntraProcessQueue
    # result_queue could be an IPC address when postproc worker is enabled.
    result_queue: IntraProcessQueue | str
    stats_queue: IntraProcessQueue

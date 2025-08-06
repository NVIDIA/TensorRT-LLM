from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import tqdm
from zmq import PUSH
from zmq.asyncio import Context

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.bench.dataclasses.general import InferenceRequest
from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple, StatsKeeper
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger


class LlmManager:
    """LLM Manager class for providing a high-level API for running benchmarks."""

    def __init__(self,
                 llm: LLM,
                 outbox: asyncio.Queue[PerfItemTuple],
                 streaming: bool,
                 concurrency: int = -1,
                 modality: Optional[str] = None) -> None:
        self.llm = llm
        self._inbox: asyncio.Queue[Tuple[InferenceRequest,
                                         SamplingParams]] = asyncio.Queue()
        self._outbox = outbox

        self._stop = asyncio.Event()
        self._running = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._task_errors: List[BaseException] = []
        self._backend_task: Optional[asyncio.Task] = None
        self._iteration_log_task: Optional[asyncio.Task] = None
        self._concurrency_semaphore = asyncio.Semaphore(
            concurrency) if concurrency > 0 else None
        self.streaming = streaming
        self.request_seen = asyncio.Event()
        self.modality = modality

    async def process_request(self, request: InferenceRequest,
                              sampling_params: SamplingParams,
                              post_proc_params: PostprocParams):
        # Set up sampling params with inference request
        self.request_seen.set()
        sampling_params.max_tokens = request.output_tokens

        async with semaphore_guard(self._concurrency_semaphore):
            request_start_timestamp = time.perf_counter_ns()
            time_on_first_token = None
            # Schedule the request in the LLM API (asynchronously)
            logger.debug(f"request.lora_request: {request.lora_request}")
            output: RequestOutput = self.llm.generate_async(
                request.input_ids if self.modality is None else request.prompt,
                sampling_params=sampling_params,
                _postproc_params=post_proc_params,
                streaming=self.streaming,
                lora_request=request.lora_request)
            if self.streaming:
                async for stream_output in output:
                    if time_on_first_token is None:
                        time_on_first_token = time.perf_counter_ns()
                        response = stream_output
            else:
                # Wait for the response to return to us.
                response: RequestOutput = await output.aresult()

        response_end_timestamp = time.perf_counter_ns()

        # Mark that the response returned. Construct a record to send to statistics.
        tokens = list(chain(*(beam.token_ids for beam in response.outputs)))
        request_perf_item = PerfItemTuple(
            start_timestamp=request_start_timestamp,
            end_timestamp=response_end_timestamp,
            request_id=response.id,
            num_input_tokens=len(output.prompt_token_ids),
            response_is_final=response.finished,
            error=False,
            tokens=tokens,
            decoding_iteration=response.decoding_iter,
            time_on_first_token=time_on_first_token,
        )

        # Register the new request perf items in the outbound queue for statistics keeping
        await self._outbox.put(request_perf_item)

    def _raise_for_failed_tasks(self):
        if not self._task_errors:
            return

        error_counts: Dict[str, int] = {}
        for error in self._task_errors:
            error_str = str(error)
            error_counts[error_str] = error_counts.get(error_str, 0) + 1

        task_errors_str = ", ".join(f"{error} ({count} requests)"
                                    for error, count in error_counts.items())
        raise ValueError(f"Requests failed: {task_errors_str}")

    def _task_done_callback(self, task: asyncio.Task):
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._task_errors.append(error)

    async def worker(self) -> None:
        try:
            while not self._stop.is_set():
                self._raise_for_failed_tasks()
                try:
                    request, sampling_params, post_proc_params = self._inbox.get_nowait(
                    )
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)  # yield to concurrent tasks
                    continue
                task = asyncio.create_task(
                    self.process_request(request,
                                         sampling_params=sampling_params,
                                         post_proc_params=post_proc_params))
                task.add_done_callback(self._task_done_callback)
                self._tasks.add(task)
            logger.debug("Worker task finishing...")
        except asyncio.CancelledError:
            logger.info("Worker task cancelled.")
        finally:
            logger.debug("Worker task cancelling remaining requests...")
            for task in self._tasks:
                task.cancel()
            logger.debug("Waiting for requests...")
            if self._tasks:
                await asyncio.wait(self._tasks)
            self._raise_for_failed_tasks()

    # This asynchronous function acts as a worker that logs iteration statistics.
    # It connects to a given address using a PUSH socket and sends JSON-encoded
    # statistics data until a stop signal is received.
    async def iteration_worker(self, iteration_addr: str) -> None:
        logger.info("Iteration log worker starting up...")
        context = None
        socket = None

        try:
            # Create a ZMQ context and socket for sending data
            context = Context.instance(io_threads=1)
            socket = context.socket(PUSH)
            socket.connect(iteration_addr)

            # Wait until a request is seen before proceeding
            await self.request_seen.wait()
            logger.debug(
                f"Iteration log worker connected to '{iteration_addr}'.")

            # Continuously send statistics data while the stop signal is not set
            while not self._stop.is_set():
                async for stats in self.llm.get_stats_async(2):
                    await socket.send_json(stats)
                # NOTE: This is a WAR to force this loop to relinquish control
                # that was preventing other async tasks from holding the event
                # loop. If we don't
                await asyncio.sleep(0)

            # Wrap up by sending any remaining statistics data
            logger.debug("Iteration log worker wrapping up...")
            async for stats in self.llm.get_stats_async(2):
                await socket.send_json(stats)
        except asyncio.CancelledError:
            # Handle task cancellation
            logger.debug("Iteration log worker cancelled.")
        except Exception as e:
            # Raise any other exceptions encountered
            raise e
        finally:
            # Ensure the socket sends a termination message and is properly closed
            logger.debug("Iteration log worker sending None...")
            socket.send_json({"end": True})
            if socket is not None:
                logger.debug("Closing socket...")
                socket.close()
            if context is not None:
                logger.debug("Terminating context...")
                context.term()

        logger.info("Iteration log worker exiting.")

    async def stop(self) -> None:
        logger.info("Stopping LLM backend.")
        self._stop.set()
        if self._iteration_log_task:
            await self._iteration_log_task
        assert self._backend_task is not None
        await self._backend_task
        logger.info("LLM Backend stopped.")

    @property
    def busy(self) -> bool:
        return not self._inbox.empty() or any(not task.done()
                                              for task in self._tasks)

    def run(self, iteration_addr: str = None) -> None:
        self._backend_task = asyncio.create_task(self.worker())
        if iteration_addr is not None:
            self._iteration_log_task = asyncio.create_task(
                self.iteration_worker(iteration_addr))

    async def enqueue(self, request: InferenceRequest,
                      sampling_params: SamplingParams,
                      post_proc_params: PostprocParams) -> None:
        await self._inbox.put((request, sampling_params, post_proc_params))


@asynccontextmanager
async def semaphore_guard(semaphore: Optional[asyncio.Semaphore] = None):
    if semaphore is not None:
        await semaphore.acquire()
    try:
        yield
    finally:
        if semaphore is not None:
            semaphore.release()


async def enqueue_messages(backend: LlmManager,
                           requests: List[InferenceRequest],
                           sampling_params: SamplingParams,
                           post_proc_params: PostprocParams,
                           submit_finished: asyncio.Event) -> None:
    num_requests = 0
    submit_start = time.perf_counter_ns()
    for request in requests:
        await backend.enqueue(request, sampling_params, post_proc_params)
        num_requests += 1
    submit_time = (time.perf_counter_ns() - submit_start) * 1.0e-9
    logger.info(
        "Request submission complete. "
        f"[count={num_requests}, time={submit_time:.4f}s, rate={num_requests / submit_time:.2f} req/s]"
    )
    submit_finished.set()


async def async_benchmark(
    llm: LLM,
    sampling_params: SamplingParams,
    post_proc_params: PostprocParams,
    requests: List[InferenceRequest],
    streaming: bool,
    concurrency: int = -1,
    iteration_log_addr: str = None,
    modality: Optional[str] = None,
) -> StatsKeeper:
    outbox = asyncio.Queue()
    statistics = StatsKeeper()
    submit_finished = asyncio.Event()

    logger.info("Starting benchmarking async task.")
    backend = LlmManager(llm,
                         outbox,
                         streaming,
                         concurrency=concurrency,
                         modality=modality)
    enqueue_task: Optional[asyncio.Task] = None
    try:
        backend.run(iteration_addr=iteration_log_addr)

        enqueue_task = asyncio.create_task(
            enqueue_messages(backend, requests, sampling_params,
                             post_proc_params, submit_finished))

        logger.info("Starting benchmark...")
        pbar = tqdm.tqdm(total=len(requests), desc="Benchmarking")
        finished_requests = 0

        while not submit_finished.is_set() or backend.busy or not outbox.empty(
        ):
            try:
                item: PerfItemTuple = await asyncio.wait_for(outbox.get(),
                                                             timeout=1.0)
                statistics.register_request_perf_item(item)
                pbar.update(1)
                finished_requests += 1
            except asyncio.TimeoutError:
                logger.debug("No items in queue. Continuing.")

        assert finished_requests == len(requests), "Benchmark failed"
        logger.info("Benchmark complete.")

        return statistics

    finally:
        if enqueue_task is not None and not enqueue_task.done():
            enqueue_task.cancel()
            await enqueue_task
        await backend.stop()

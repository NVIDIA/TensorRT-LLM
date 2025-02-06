from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from itertools import chain
from typing import List, Optional, Set, Tuple

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bench.dataclasses.general import InferenceRequest
from tensorrt_llm.bench.dataclasses.reporting import (NewRequestPerfItemTuple,
                                                      StatsKeeper)
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger


class LlmManager:
    """LLM Manager class for providing a high-level API for running benchmarks."""

    def __init__(self,
                 llm: LLM,
                 outbox: asyncio.Queue[NewRequestPerfItemTuple],
                 streaming: bool,
                 concurrency: int = -1) -> None:
        self.llm = llm
        self._inbox: asyncio.Queue[Tuple[InferenceRequest,
                                         SamplingParams]] = asyncio.Queue()
        self._outbox = outbox

        self._stop = asyncio.Event()
        self._running = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._backend_task = None
        self._concurrency_semaphore = asyncio.Semaphore(
            concurrency) if concurrency > 0 else None
        self.streaming = streaming

    async def process_request(self, request: InferenceRequest,
                              sampling_params: SamplingParams):
        # Set up sampling params with inference request
        sampling_params.max_tokens = request.output_tokens

        async with semaphore_guard(self._concurrency_semaphore):
            request_start_timestamp = time.perf_counter_ns()
            time_on_first_token = None
            # Schedule the request in the LLM API (asynchronously)
            output: RequestOutput = self.llm.generate_async(
                request.input_ids,
                sampling_params=sampling_params,
                streaming=self.streaming)
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
        tokens = list(chain(*[beam.token_ids for beam in response.outputs]))
        request_perf_item = NewRequestPerfItemTuple(
            start_timestamp=request_start_timestamp,
            end_timestamp=response_end_timestamp,
            request_id=response.request_id,
            num_input_tokens=len(request.input_ids),
            response_is_final=response.finished,
            error=False,
            tokens=tokens,
            decoding_iteration=len(tokens),
            time_on_first_token=time_on_first_token,
        )

        # Register the new request perf items in the outbound queue for statistics keeping
        await self._outbox.put(request_perf_item)

    async def worker(self) -> None:
        while not self._stop.is_set():
            request, sampling_params = await self._inbox.get()
            task = asyncio.create_task(
                self.process_request(request, sampling_params=sampling_params))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def stop(self) -> None:
        logger.info("Stopping LLM backend.")
        self._stop.set()
        logger.info(f"Cancelling all {len(self._tasks)} tasks to complete.")
        for task in self._tasks:
            task.cancel()
        logger.info("All tasks cancelled.")
        self._backend_task.cancel()
        logger.info("LLM Backend stopped.")

    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    def run(self) -> None:
        self._backend_task = asyncio.create_task(self.worker())

    async def enqueue(self, request: InferenceRequest,
                      sampling_params: SamplingParams) -> None:
        await self._inbox.put((request, sampling_params))


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
                           submit_finished: asyncio.Event) -> None:
    num_requests = 0
    submit_start = time.perf_counter_ns()
    for request in requests:
        await backend.enqueue(request, sampling_params)
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
    requests: List[InferenceRequest],
    streaming: bool,
    concurrency: int = -1,
) -> StatsKeeper:
    outbox = asyncio.Queue()
    statistics = StatsKeeper()
    submit_finished = asyncio.Event()

    try:
        logger.info("Starting benchmarking async task.")
        backend = LlmManager(llm, outbox, streaming, concurrency=concurrency)
        backend.run()

        enqueue_task = asyncio.create_task(
            enqueue_messages(backend, requests, sampling_params,
                             submit_finished))

        logger.info("Starting benchmark...")
        while not submit_finished.is_set() or backend.busy or not outbox.empty(
        ):
            try:
                item: NewRequestPerfItemTuple = await asyncio.wait_for(
                    outbox.get(), timeout=1.0)
                statistics.register_request_perf_item(item)
            except asyncio.TimeoutError:
                logger.debug("No items in queue. Continuing.")

        logger.info("Benchmark complete.")

        return statistics

    except asyncio.CancelledError:
        enqueue_task.cancel()

    finally:
        backend.stop()

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from itertools import chain
from multiprocessing.synchronize import Event as MpEvent
from typing import List, Set

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import LLM
from tensorrt_llm.bench.benchmark.dataclasses import RuntimeConfig
from tensorrt_llm.bench.benchmark.utils.reporting import (NewRequestTuple,
                                                          ResponseTuple,
                                                          StatsKeeper,
                                                          report_statistics)
from tensorrt_llm.bench.dataclasses import InferenceRequest
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.llmapi.utils import SamplingParams
from tensorrt_llm.logger import logger


class LlmManager:
    """LLM Manager class for providing a high-level API for running benchmarks."""

    def __init__(self, llm: LLM, outbox: asyncio.Queue[RequestOutput],
                 streaming) -> None:
        self.llm = llm
        self._inbox = asyncio.Queue()
        self._outbox = outbox

        self._stop = asyncio.Event()
        self._running = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._backend_task = None
        self.streaming = streaming

    async def process_request(self, request: InferenceRequest,
                              sampling_params: SamplingParams):
        # Set up sampling params with inference request
        sampling_params.max_tokens = request.output_tokens
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
        # Register the new request in the outbound queue for statistics keeping
        await self._outbox.put(
            NewRequestTuple(request_start_timestamp, response.request_id,
                            len(request.input_ids)))

        # Mark that the response returned. Construct a record to send to statistics.
        tokens = list(chain(*[beam.token_ids for beam in response.outputs]))
        end_time = time.perf_counter_ns()
        response_record = ResponseTuple(end_time, response.request_id,
                                        response.finished, False, tokens,
                                        len(tokens), time_on_first_token)
        # Push the record to the outbound queue.
        await self._outbox.put(response_record)

    async def worker(self) -> None:
        while not self._stop.is_set():
            request = await self._inbox.get()
            task = asyncio.create_task(
                self.process_request(request[0], request[1]))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def stop(self) -> None:
        logger.info("Stopping LLM backend.")
        self._stop.set()
        logger.info(f"Cancelling all {len(self._tasks)} tasks to complete.")
        for task in self._tasks:
            task.cancel()
        logger.info("All tasks cancelled.")
        self.backend.cancel()
        logger.info("LLM Backend stopped.")

    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    async def run(self) -> None:
        self.backend = asyncio.create_task(self.worker())
        await self.backend

    async def enqueue(self, request: InferenceRequest,
                      sampling_params: SamplingParams) -> None:
        await self._inbox.put((request, sampling_params))


async def enqueue_messages(backend: LlmManager,
                           requests: List[InferenceRequest],
                           sampling_params: SamplingParams,
                           submit_finished: MpEvent) -> None:
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


async def async_benchmark(runtime_config: RuntimeConfig,
                          requests: List[InferenceRequest],
                          streaming: bool) -> None:
    outbox = asyncio.Queue()
    sampling_params = SamplingParams(end_id=-1, pad_id=-1, beam_width=1)
    statistics = StatsKeeper()
    submit_finished = mp.Event()

    try:
        logger.info("Setting up throughput benchmark.")
        llm_args = LlmArgs(
            model=runtime_config.engine_dir,
            skip_tokenizer_init=True,
            pipeline_parallel_size=runtime_config.world_config.pp_size,
            tensor_parallel_size=runtime_config.world_config.tp_size,
            trust_remote_code=True,
            scheduler_config=runtime_config.settings_config.
            get_scheduler_config(),
            kv_cache_config=runtime_config.settings_config.get_kvcache_config(),
            decoding_config=runtime_config.decoding_config.get_decoding_config(
            ),
            batching_type=trtllm.BatchingType.INFLIGHT,
        )

        llm = LLM(**llm_args.to_dict())

        backend = LlmManager(llm, outbox, streaming)
        logger.info("Creating backend and request enqueue tasks.")
        asyncio.create_task(backend.run())
        enqueue_task = asyncio.create_task(
            enqueue_messages(backend, requests, sampling_params,
                             submit_finished))

        logger.info("Starting benchmark...")
        while not submit_finished.is_set() or backend.busy or not outbox.empty(
        ):
            try:
                item = await asyncio.wait_for(outbox.get(), timeout=1.0)
                if isinstance(item, NewRequestTuple):
                    statistics.register_request(item.request_id, item.timestamp,
                                                item.input_length)
                elif isinstance(item, ResponseTuple):
                    statistics.register_response(item.request_id,
                                                 item.timestamp, item.final,
                                                 item.error,
                                                 item.decoding_iteration,
                                                 item.tokens,
                                                 item.time_on_first_token)
                else:
                    raise TypeError(
                        f"Encountered unexpected type '{type(item)}' when "
                        "processing responses.")
            except asyncio.TimeoutError:
                logger.debug("No items in queue. Continuing.")
        logger.info("Benchmark complete.")
        report_statistics(statistics, runtime_config, logger, streaming)
    except asyncio.CancelledError:
        enqueue_task.cancel()
    finally:
        backend.stop()
        llm._shutdown()

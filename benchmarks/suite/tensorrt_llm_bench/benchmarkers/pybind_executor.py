from datetime import timedelta
from time import sleep, time
from typing import List

from mpi4py.MPI import COMM_WORLD
from transformers import PreTrainedTokenizer
from utils.dataclasses import BenchmarkConfig, BenchmarkResults
from utils.enums import IFBSchedulingPolicy, ResultsSchedulingPolicy

from tensorrt_llm.bindings.executor import (Executor, ExecutorConfig,
                                            KvCacheConfig, ModelType,
                                            OutputConfig, Request,
                                            SchedulerConfig)

from . import InferenceRequest


class PybindExecutorBenchmarker:
    """Utility class for running inflight benchmarks via the Executor API."""

    def __init__(
        self,
        config: BenchmarkConfig,
    ):
        """Initialize a gptSessionBenchmark instance.

        Args:
            config (BenchmarkConfig): Benchmark configuration for build/run.
        """
        self.config: BenchmarkConfig = config

    @staticmethod
    def get_request(request: InferenceRequest,
                    tokenizer: PreTrainedTokenizer) -> Request:
        return Request(
            input_token_ids=request.logits,
            max_new_tokens=request.output_tokens,
            stop_words=[],
            bad_words=[],
            streaming=False,
            output_config=OutputConfig(exclude_input_from_output=True),
            pad_id=tokenizer.pad_token_id,
            end_id=tokenizer.eos_token_id,
        )

    def initialize_executor(self) -> Executor:
        """
        Initialize an Executor instance.

        Returns:
            Executor: An instance of a TensorRT-LLM Executor.
        """
        policy = IFBSchedulingPolicy(self.config.scheduling_policy).value
        executor_config: ExecutorConfig = ExecutorConfig(
            max_beam_width=1,
            enable_chunked_context=self.config.chunking,
            scheduler_config=SchedulerConfig(
                capacity_scheduler_policy=policy, ),
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=self.config.kv_cache_mem_percentage, ),
        )

        executor: Executor = Executor(
            model_path=self.config.engine_path,
            model_type=ModelType.DECODER_ONLY,
            executor_config=executor_config,
        )

        return executor

    def benchmark_dataset(self, rate: int,
                          dataset: List[InferenceRequest]) -> BenchmarkResults:
        """Benchmark the Executor Pybind interface.

        Args:
            dataset (List[InferenceRequest]): List of inference requests to
            benchmark with.

        Returns:
            BenchmarkResults: Final results from running the specified dataset.
        """
        request_ids = []
        num_finished = 0
        num_errored = 0
        num_input_tokens = 0
        num_output_tokens = 0
        delay = 1.0 / float(rate)
        last_request = len(dataset) - 1
        bench_result = None

        executor = self.initialize_executor()
        if executor.can_enqueue_requests():
            print(f"[RANK {COMM_WORLD.rank}] Submitting requests...")
            start = time()
            for i, request in enumerate(dataset):
                sleep_time = delay if i != last_request else 0
                request_ids.append(executor.enqueue_request(request))
                num_input_tokens += len(request.input_token_ids)
                sleep(sleep_time)
            print(f"[RANK {COMM_WORLD.rank}] Completed request submission.")

            while num_finished <= last_request:
                responses = executor.await_responses(timeout=timedelta(
                    milliseconds=1))
                for response in responses:
                    has_error = response.has_error()
                    num_finished += 1
                    num_errored += 1 if has_error else 0

                    if not has_error:
                        result = response.result
                        for out_tokens in result.output_token_ids:
                            num_output_tokens += len(out_tokens)
            end = time()
            print(f"[RANK {COMM_WORLD.rank}] Calculating results.")
            e2e_time = end - start
            e2e_time * 1000.0
            policy = ResultsSchedulingPolicy(
                IFBSchedulingPolicy(self.config.scheduling_policy).value)

            bench_result = BenchmarkResults(
                model=self.config.model,
                dtype=self.config.dtype.value,
                quantization=str(self.config.quantization.value),
                max_batch_size=self.config.max_batch_size,
                total_input_tokens=num_input_tokens,
                total_output_tokens=num_output_tokens,
                tp_size=self.config.tensor_parallel,
                pp_size=self.config.pipeline_parallel,
                kv_mem_fraction=self.config.kv_cache_mem_percentage,
                scheduler=policy.value,
                max_tokens=self.config.max_tokens,
                inflight_batching=True,
                total_latency=e2e_time,
                first_token_latency=0,
                time_per_output_token=0,
                latency_units="ms",
                throughput=num_output_tokens / e2e_time,
                throughput_units="tokens/second",
                peak_gpu_mem=0.0,
                peak_gpu_mem_units="GB",
                build_cmd="",
                benchmark_cmd="",
            )

        return bench_result

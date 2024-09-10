import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from tensorrt_llm import logger

from .._utils import release_gc
from ..profiler import device_memory_info, host_memory_info
from .llm import LLM, SamplingParams
from .utils import is_directory_empty, print_colored


@dataclass
class PerfItem:
    start: float
    end: float
    time_on_first_token: Optional[float] = None
    num_out_tokens: int = 0

    @property
    def lantency(self) -> float:
        return self.end - self.start

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.time_on_first_token is None:
            return None
        else:
            return self.time_on_first_token - self.start

    @property
    def inter_token_latency(self) -> Optional[float]:
        if self.time_on_first_token is None or self.num_out_tokens <= 1:
            return None
        else:
            return (self.end -
                    self.time_on_first_token) / (self.num_out_tokens - 1)


@dataclass
class Report:
    num_samples: int
    total_latency: float
    seq_throughput: float
    token_throughput: float
    avg_sl: float
    max_sl: float
    min_sl: float
    p99_sl: float
    p90_sl: float
    p50_sl: float
    avg_ttft: Optional[float] = None
    max_ttft: Optional[float] = None
    min_ttft: Optional[float] = None
    p99_ttft: Optional[float] = None
    p90_ttft: Optional[float] = None
    p50_ttft: Optional[float] = None
    avg_itl: Optional[float] = None
    max_itl: Optional[float] = None
    min_itl: Optional[float] = None
    p99_itl: Optional[float] = None
    p90_itl: Optional[float] = None
    p50_itl: Optional[float] = None
    memory_usage_samples: "MemoryContinuousMonitorThread.RecordList" = field(
        default_factory=list)

    def display(self):
        print_colored("Performance Report:\n",
                      color="bold_green",
                      writer=sys.stdout)
        print(f"num_samples: {self.num_samples}")
        print(f"total_latency (ms): {self.total_latency*1000:.2f}")
        print(f"seq_throughput (seq/sec): {self.seq_throughput:.2f}")
        print(f"token_throughput (token/sec): {self.token_throughput:.2f}")
        print("", flush=True)

        print(f"avg_sequence_latency (ms): {self.avg_sl*1000:.2f}")
        print(f"max_sequence_latency (ms): {self.max_sl*1000:.2f}")
        print(f"min_sequence_latency (ms): {self.min_sl*1000:.2f}")
        print(f"p99_sequence_latency (ms): {self.p99_sl*1000:.2f}")
        print(f"p90_sequence_latency (ms): {self.p90_sl*1000:.2f}")
        print(f"p50_sequence_latency (ms): {self.p50_sl*1000:.2f}")
        print("", flush=True)

        if self.avg_ttft:
            print(f"avg_time_to_first_token (ms): {self.avg_ttft*1000:.2f}")
            print(f"max_time_to_first_token (ms): {self.max_ttft*1000:.2f}")
            print(f"min_time_to_first_token (ms): {self.min_ttft*1000:.2f}")
            print(f"p99_time_to_first_token (ms): {self.p99_ttft*1000:.2f}")
            print(f"p90_time_to_first_token (ms): {self.p90_ttft*1000:.2f}")
            print(f"p50_time_to_first_token (ms): {self.p50_ttft*1000:.2f}")
            print("", flush=True)

        if self.avg_itl:
            print(f"avg_inter_token_latency (ms): {self.avg_itl*1000:.2f}")
            print(f"max_inter_token_latency (ms): {self.max_itl*1000:.2f}")
            print(f"min_inter_token_latency (ms): {self.min_itl*1000:.2f}")
            print(f"p99_inter_token_latency (ms): {self.p99_itl*1000:.2f}")
            print(f"p90_inter_token_latency (ms): {self.p90_itl*1000:.2f}")
            print(f"p50_inter_token_latency (ms): {self.p50_itl*1000:.2f}")
            print("", flush=True)

        if self.memory_usage_samples:
            print("Memory Usage:\n")
            min_record, max_record, average_record = self.memory_usage_samples.get_min(
            ), self.memory_usage_samples.get_max(
            ), self.memory_usage_samples.get_average()
            print(
                f"  host memory usage: (min: {min_record[0]}, max: {max_record[0]}, average: {average_record[0]})"
            )
            print(
                f"  gpu memory usage: (min: {min_record[1]}, max: {max_record[1]}, average: {average_record[1]})"
            )
        print_colored("__________________________________\n",
                      color="green",
                      writer=sys.stdout)

    def save_json(self, path: Path, **kwargs):
        with open(path, 'w') as f:
            data = self.__dict__.copy()
            data.update(kwargs)
            data["memory_usage_samples"] = [
                asdict(record) for record in self.memory_usage_samples
            ]
            json.dump(data, f)


@dataclass
class Sample:
    input_ids: List[int]
    output_len: int


class MemoryContinuousMonitorThread(threading.Thread):
    ''' Monitor the host memory and GPU memory usage. '''

    @dataclass
    class Record:
        time: float
        host_memory: float  # GiB
        gpu_memory: List[float]

        def to_dict(self):
            return dict(time=self.time,
                        host_memory=self.host_memory,
                        gpu_memory=self.gpu_memory)

    class RecordList(list):

        def __init__(self, *args, **kwargs):
            super(MemoryContinuousMonitorThread.RecordList,
                  self).__init__(*args, **kwargs)

        def get_min(self):
            return self._get_memory_usage('min')

        def get_max(self):
            return self._get_memory_usage('max')

        def get_average(self):
            return self._get_memory_usage('average')

        def _get_memory_usage(self, op: str):
            host_memory = [record.host_memory for record in self]
            gpu_memory = [record.gpu_memory for record in self]

            ops = dict(min=np.min, max=np.max, average=np.mean)
            theop = ops[op]

            return theop(host_memory), [theop(gpu) for gpu in zip(*gpu_memory)]

    def __init__(self, sampling_interval: float = 1):
        super(MemoryContinuousMonitorThread, self).__init__()
        self.sampling_interval = sampling_interval
        self._stop_event = threading.Event()
        self.memory_samples = MemoryContinuousMonitorThread.RecordList()

    def run(self):
        while not self._stop_event.is_set():
            record = self.monitor()
            logger.info(f'record: {record}')
            self.memory_samples.append(record)
            time.sleep(self.sampling_interval)

    def monitor(self) -> "MemoryContinuousMonitorThread.Record":
        return self.Record(time.perf_counter(), get_host_memory_usage(),
                           list(get_gpu_memory_usage()))

    def stop(self):
        self._stop_event.set()


def get_host_memory_usage() -> float:
    return host_memory_info(os.getpid())[0] / 1024**3  # GiB


def get_gpu_memory_usage() -> Iterable[float]:
    for device in range(torch.cuda.device_count()):
        yield device_memory_info(device)[0] / 1024**3


class LLMPerfEvaluator:

    @classmethod
    def create(cls,
               model: str,
               samples_path: Path,
               num_samples: Optional[int] = None,
               streaming: bool = False,
               warmup: int = 2,
               concurrency: Optional[int] = None,
               engine_cache_path: Optional[Path] = None,
               memory_monitor_interval: Optional[int] = None,
               **kwargs) -> 'LLMPerfEvaluator':
        '''
        Args:
            model: The model name or a local path to the model directory.
            samples_path: path to the input data samples
            num_samples: number of the heading samples to run, if set to -1, all samples will be used
            streaming: Whether to enable streaming generation
            warmup: number of samples for warmup runs
            concurrency: Concurrent number of connections with LLM. if left default, the concurrency will be the number of samples
            engine_cache_path: path to the engine file, if provided, the engine will save the built engine to the path and reuse it for the next runs
            memory_monitor_interval: the interval to monitor the host and GPU memory usage, if set to None, the memory monitor will be disabled
            kwargs: the additional arguments are for the LLM constructor
        '''

        from_cache = False
        if engine_cache_path and Path.exists(
                engine_cache_path
        ) and not is_directory_empty(engine_cache_path):
            print(f"Loading engine from {engine_cache_path}\n")
            from_cache = True
            model = engine_cache_path

        memory_monitor_thread = None
        if memory_monitor_interval is not None:
            memory_monitor_thread = MemoryContinuousMonitorThread(
                sampling_interval=memory_monitor_interval)
            memory_monitor_thread.start()

        # TODO[chunweiy]: Fixit, this barely work, the cpp runtime will trigger RuntimeError, which cannot be caught
        try:
            llm = LLM(model, skip_tokenizer_init=True, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM with {model} and {kwargs}")
            raise e

        if engine_cache_path is not None and not from_cache:
            print_colored(f"Saving engine to {engine_cache_path}\n",
                          "green",
                          writer=sys.stdout)
            llm.save(engine_cache_path)

        samples: List[Sample] = list(cls.load_dataset(samples_path))
        if num_samples is not None:
            assert len(samples) >= num_samples, \
                f"num_samples {num_samples} is too large. The dataset only has {len(samples)} samples."
            samples = samples[:num_samples]

        return cls(llm,
                   samples=samples,
                   streaming=streaming,
                   warmup=warmup,
                   concurrency=concurrency,
                   memory_monitor_thread=memory_monitor_thread)

    def __init__(self,
                 llm: LLM,
                 samples: List[Sample],
                 streaming: bool = False,
                 warmup: int = 2,
                 concurrency: Optional[int] = None,
                 memory_monitor_thread: Optional[
                     MemoryContinuousMonitorThread] = None):
        self.llm = llm
        self.samples = samples
        self.streaming = streaming
        self.warmup = warmup
        self.concurrency = len(
            self.samples) if concurrency is None else concurrency
        self.memory_monitor_thread = memory_monitor_thread

        self.perf_items: List[PerfItem] = []
        self.start = None
        self.end = None

    def run(self, end_id: int = -1, beam_width: int = 1) -> Report:
        # reset states
        self.perf_items = []
        sample_offset = 0

        sampling_params = SamplingParams(
            end_id=end_id,
            pad_id=end_id,
            beam_width=beam_width,
        )

        async def lane(sampling_params: SamplingParams,
                       is_warmup: bool = False):
            nonlocal sample_offset
            num_samples = self.warmup if is_warmup else len(self.samples)

            while sample_offset < num_samples:
                sample = self.samples[sample_offset]
                sample_offset += 1
                sampling_params.max_tokens = sample.output_len
                sampling_params.end_id = -2
                sampling_params.pad_id = -2

                start = time.perf_counter()
                time_on_first_token = None
                output = self.llm.generate_async(
                    sample.input_ids,
                    sampling_params=sampling_params,
                    streaming=self.streaming)
                if self.streaming:
                    async for stream_output in output:
                        if time_on_first_token is None:
                            time_on_first_token = time.perf_counter()
                    output = stream_output
                else:
                    output = await output.aresult()
                end = time.perf_counter()

                perf_item = PerfItem(start=start,
                                     end=end,
                                     time_on_first_token=time_on_first_token,
                                     num_out_tokens=sum(
                                         beam_output.length
                                         for beam_output in output.outputs))
                if not is_warmup:
                    self.perf_items.append(perf_item)

        if self.warmup > 0:
            logger.warning("warming up ...")

            async def run_lanes():
                lanes = [
                    lane(sampling_params, is_warmup=True)
                    for _ in range(min(self.concurrency, self.warmup))
                ]
                await asyncio.gather(*lanes)

            asyncio.run(run_lanes())
            sample_offset = 0

        logger.warning("running ...")

        async def run_lanes():
            lanes = [lane(sampling_params) for _ in range(self.concurrency)]
            await asyncio.gather(*lanes)

        self.start = time.perf_counter()
        asyncio.run(run_lanes())
        self.end = time.perf_counter()

        return self._generate_report()

    @staticmethod
    def load_dataset(path: Path) -> Iterable[Sample]:
        with open(path, 'r') as f:
            dataset = json.load(f)

        for sample in dataset["samples"]:
            yield Sample(input_ids=sample["input_ids"],
                         output_len=sample["output_len"])

    def _generate_report(self) -> Report:
        num_samples = len(self.perf_items)
        total_tokens = sum(
            [perf_item.num_out_tokens for perf_item in self.perf_items])
        total_latency = self.end - self.start
        seq_throughput = num_samples / total_latency
        token_throughput = total_tokens / total_latency

        sls = [perf_item.lantency for perf_item in self.perf_items]
        ttfts = [
            perf_item.time_to_first_token for perf_item in self.perf_items
            if perf_item.time_to_first_token is not None
        ]
        itls = [
            perf_item.inter_token_latency for perf_item in self.perf_items
            if perf_item.inter_token_latency is not None
        ]

        return Report(
            num_samples=num_samples,
            total_latency=total_latency,
            seq_throughput=seq_throughput,
            token_throughput=token_throughput,
            avg_sl=np.mean(sls),
            max_sl=np.max(sls),
            min_sl=np.min(sls),
            p99_sl=np.percentile(sls, 99),
            p90_sl=np.percentile(sls, 90),
            p50_sl=np.median(sls),
            avg_ttft=np.mean(ttfts) if ttfts else None,
            max_ttft=np.max(ttfts) if ttfts else None,
            min_ttft=np.min(ttfts) if ttfts else None,
            p99_ttft=np.percentile(ttfts, 99) if ttfts else None,
            p90_ttft=np.percentile(ttfts, 90) if ttfts else None,
            p50_ttft=np.median(ttfts) if ttfts else None,
            avg_itl=np.mean(itls) if itls else None,
            max_itl=np.max(itls) if itls else None,
            min_itl=np.min(itls) if itls else None,
            p99_itl=np.percentile(itls, 99) if itls else None,
            p90_itl=np.percentile(itls, 90) if itls else None,
            p50_itl=np.median(itls) if itls else None,
            memory_usage_samples=self.memory_monitor_thread.memory_samples
            if self.memory_monitor_thread else [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.llm.__exit__(exc_type, exc_value, traceback)
        del self.llm
        release_gc()

        if self.memory_monitor_thread:
            self.memory_monitor_thread.stop()
            self.memory_monitor_thread.join()
            del self.memory_monitor_thread

import asyncio
import json
import math
import os
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
    num_out_tokens: int = 0

    @property
    def lantency(self) -> float:
        return self.end - self.start


@dataclass
class Report:
    num_samples: int
    total_latency: float
    avg_latency: float
    seq_throughput: float
    token_throughput: float
    ave_tokens_per_sample: float
    memory_usage_samples: "MemoryContinuousMonitorThread.RecordList" = field(
        default_factory=list)

    def display(self):
        print_colored("Performance Report:\n", color="bold_green")
        print(f"num_samples: {self.num_samples}")
        print(f"total_latency: {self.total_latency:.2f}")
        print(f"avg_latency: {self.avg_latency:.2f}")
        print(f"seq_throughput: {self.seq_throughput:.2f}")
        print(f"token_throughput: {self.token_throughput:.2f}")
        print(f"average tokens per sample: {self.ave_tokens_per_sample:.2f}")
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
        print_colored("__________________________________\n", color="green")

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
        return self.Record(time.time(), get_host_memory_usage(),
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
               num_samples: int = -1,
               warmup: int = 100,
               batch_size: int = -1,
               engine_cache_path: Optional[Path] = None,
               memory_monitor_interval: Optional[int] = None,
               **kwargs) -> 'LLMPerfEvaluator':
        '''
        Args:
            model: The model name or a local path to the model directory.
            samples_path: path to the input data samples
            num_samples: number of the heading samples to run, if set to -1, all samples will be used
            warmup: number of samples for warmup runs
            batch_size: batch size for the runs, if left default, the batch size will be the same as the number of samples
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
            print_colored(f"Saving engine to {engine_cache_path}\n", "green")
            llm.save(engine_cache_path)

        samples: List[Sample] = list(cls.load_dataset(samples_path))
        assert len(
            samples
        ) >= num_samples, f"num_samples {num_samples} is too large. The dataset only has {len(samples)} samples."
        samples = samples[:num_samples] if num_samples > 0 else samples

        return cls(llm,
                   warmup=warmup,
                   samples=samples,
                   max_num_samples=num_samples,
                   batch_size=batch_size,
                   memory_monitor_thread=memory_monitor_thread)

    def __init__(self,
                 llm: LLM,
                 samples: List[Sample],
                 warmup: int,
                 max_num_samples: int,
                 batch_size: int,
                 memory_monitor_thread: Optional[
                     MemoryContinuousMonitorThread] = None):
        self.llm = llm
        self.samples = samples
        self.warmup = warmup
        self.max_num_samples = max_num_samples
        self.perf_items = []
        self.batch_size = batch_size if batch_size > 0 else len(self.samples)
        self.memory_monitor_thread = memory_monitor_thread

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

        async def lane(num_tasks: int,
                       sampling_params: SamplingParams,
                       warmup=False):
            nonlocal sample_offset

            for i in range(num_tasks):
                sample = self.samples[sample_offset]
                sample_offset += 1
                sampling_params.max_new_tokens = sample.output_len
                sampling_params.end_id = -2
                sampling_params.pad_id = -2

                start = time.time()
                output = self.llm.generate_async(
                    sample.input_ids, sampling_params=sampling_params)
                output = await output.aresult()
                end = time.time()

                perf_item = PerfItem(start=start,
                                     end=end,
                                     num_out_tokens=sum(
                                         beam_output.length
                                         for beam_output in output.outputs))
                if not warmup:
                    self.perf_items.append(perf_item)

        if self.warmup > 0:
            logger.warning("warming up ...")
            for i in range(math.ceil(self.warmup / len(self.samples))):
                asyncio.run(
                    lane(min(self.warmup, len(self.samples)),
                         sampling_params,
                         warmup=True))
            sample_offset = 0

        logger.warning("running ...")
        self.start = time.time()

        async def run_lanes():
            num_tasks = len(self.samples) // self.batch_size
            lanes = [
                lane(num_tasks, sampling_params) for _ in range(self.batch_size)
            ]
            await asyncio.gather(*lanes)

        asyncio.run(run_lanes())
        self.end = time.time()

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
        total_latency = self.end - self.start
        avg_latency = total_latency / num_samples
        seq_throughput = num_samples / total_latency
        token_throughput = sum(
            [perf_item.num_out_tokens
             for perf_item in self.perf_items]) / total_latency
        total_tokens = sum(
            [perf_item.num_out_tokens for perf_item in self.perf_items])

        return Report(
            num_samples=num_samples,
            total_latency=total_latency,
            avg_latency=avg_latency,
            seq_throughput=seq_throughput,
            token_throughput=token_throughput,
            ave_tokens_per_sample=total_tokens / num_samples,
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

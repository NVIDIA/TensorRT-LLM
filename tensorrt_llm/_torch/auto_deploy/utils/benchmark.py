import json
import os
import pathlib
from collections import deque
from contextlib import contextmanager
from typing import Callable, Collection, Deque

import torch

from .logger import ad_logger


class GenerationProfiler:
    def __init__(self, num_runs: int):
        self.prefill_start, self.prefill_end = self._create_events()
        self.decode_start, self.decode_end = self._create_events()
        self.num_runs = num_runs

        self.prefill_times: Deque[float] = deque(maxlen=self.num_runs)
        self.decode_times: Deque[float] = deque(maxlen=self.num_runs)

    def _create_events(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return start, end

    def _record_event(self, event: torch.cuda.Event):
        event.record()

    def _get_elapsed_time(self, start_event, end_event):
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)

    def _store_time(self, time_list: Deque[float], elapsed_time):
        time_list.append(elapsed_time)

    def record_prefill_start(self):
        self._record_event(self.prefill_start)

    def record_prefill_end(self) -> float:
        self._record_event(self.prefill_end)
        elapsed_time_ms = self._get_elapsed_time(self.prefill_start, self.prefill_end)
        self._store_time(self.prefill_times, elapsed_time_ms)
        return elapsed_time_ms

    def record_decode_start(self):
        self._record_event(self.decode_start)

    def record_decode_end(self) -> float:
        self._record_event(self.decode_end)
        elapsed_time_ms = self._get_elapsed_time(self.decode_start, self.decode_end)
        self._store_time(self.decode_times, elapsed_time_ms)
        return elapsed_time_ms

    def get_average_prefill_time(self) -> float:
        return self._get_average_time(self.prefill_times)

    def get_average_decode_time(self) -> float:
        return self._get_average_time(self.decode_times)

    def _get_average_time(self, time_list: Collection[float]) -> float:
        if len(time_list):
            return sum(time_list) / len(time_list)
        return 0.0

    def reset(self):
        self.prefill_start, self.prefill_end = self._create_events()
        self.decode_start, self.decode_end = self._create_events()

    @contextmanager
    def record_prefill(self):
        try:
            self.record_prefill_start()
            yield
        finally:
            self.record_prefill_end()

    @contextmanager
    def record_decode(self):
        try:
            self.record_decode_start()
            yield
        finally:
            self.record_decode_end()


def benchmark(
    func: Callable[[], None], num_runs: int, log_prefix: str = "", results_path: str | None = None
) -> Callable[[Callable], Callable]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []

    # warmup
    for _ in range(1):
        func()

    use_nsys_profiling = bool(os.environ.get("NSYS_PROFILING_SESSION_ID", None))
    if use_nsys_profiling:
        torch.cuda.cudart().cudaProfilerStart()
        func()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        for _ in range(num_runs):
            start.record()
            func()
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        ad_logger.info(
            f"{log_prefix} Average of {len(latencies)} "
            f"runs: {sum(latencies) / len(latencies): 0.2f} (millisecond)"
        )

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "avg_latency_num_runs": num_runs if not use_nsys_profiling else 0,
    }


def store_benchmark_results(results: dict, results_path: str):
    results_path = pathlib.Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as results_file:
        json.dump(results, results_file)

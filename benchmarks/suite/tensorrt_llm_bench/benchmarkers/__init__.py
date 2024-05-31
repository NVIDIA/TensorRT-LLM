from typing import List, Protocol

from utils.dataclasses import BenchmarkResults, InferenceRequest


class Benchmarker(Protocol):
    """Protocol for defining benchmarking classes for building/benchmarking."""

    def build(self) -> None:
        """Build a model to be benchmarked."""
        ...

    def benchmark(self) -> BenchmarkResults:
        """Benchmark the constructed model container by a benchmarker."""
        ...


class DatasetBenchmarker(Protocol):

    def benchmark_dataset(self,
                          dataset: List[InferenceRequest]) -> BenchmarkResults:
        """_summary_

        Args:
            dataset (List[InferenceRequest]): List of inference requests to benchmark.

        Returns:
            BenchmarkResults: The results of the benchmark run.
        """
        ...

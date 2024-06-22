from typing import Protocol

from utils.dataclasses import BenchmarkResults


class Benchmarker(Protocol):
    """Protocol for defining benchmarking classes for building/benchmarking."""

    def build(self) -> None:
        """Build a model to be benchmarked."""
        ...

    def benchmark(self) -> BenchmarkResults:
        """Benchmark the constructed model container by a benchmarker."""
        ...

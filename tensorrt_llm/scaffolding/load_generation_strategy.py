"""
Load generation strategies for TensorRT-LLM Scaffolding Benchmark.
Inspired by guidellm's scheduling strategies.
"""
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional

from pydantic import BaseModel, Field


class _UnlimitedSemaphore:
    """
    A dummy semaphore that supports async with but never blocks.
    Used for strategies that don't limit concurrency.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class LoadGenerationStrategy(BaseModel, ABC):
    """
    Abstract base class for load generation strategies.

    Defines how requests are generated and sent over time to simulate different load patterns.
    The core method request_times() generates timestamps for when each request should be sent.
    """

    strategy_type: str = Field(description="Type of load generation strategy")
    start_time: Optional[float] = Field(
        default=None,
        description="Strategy start time (set automatically on first request)")

    class Config:
        arbitrary_types_allowed = True

    def get_semaphore(self):
        """
        Get semaphore for concurrency control.

        Returns an object that supports async with context manager.
        Default implementation returns unlimited (non-blocking) semaphore.
        Subclasses can override to provide specific concurrency limits.

        Returns:
            Semaphore-like object supporting async with
        """
        return _UnlimitedSemaphore()

    async def request_times(self) -> AsyncGenerator[float, None]:
        """
        Template method that ensures start_time is set, then delegates to subclass.

        Returns:
            Timestamp (float): Absolute time when the request should be sent
        """
        # Ensure start_time is set once (lazily on first call)
        if self.start_time is None:
            self.start_time = time.time()

        # Delegate to subclass implementation
        async for timestamp in self._generate_request_times():
            yield timestamp

    @abstractmethod
    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """
        Subclass-specific implementation for generating request timestamps.

        This method is called by request_times() after start_time is initialized.
        Subclasses should implement this method instead of request_times().

        Returns:
            Timestamp (float): Absolute time when the request should be sent
        """
        ...

    def __str__(self) -> str:
        return f"{self.strategy_type}"


class SynchronousStrategy(LoadGenerationStrategy):
    """
    Synchronous strategy: sends requests serially, one after another.

    Characteristics:
    - Single-threaded execution
    - Rate = 1 / average latency
    - Suitable for baseline latency testing

    Example:
        strategy = SynchronousStrategy()
        # Requests will be sent strictly in serial
    """

    strategy_type: str = "synchronous"

    def get_semaphore(self):
        """Return semaphore with concurrency limit of 1."""
        import asyncio
        return asyncio.Semaphore(1)

    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """Send each request immediately (but controlled by concurrency=1 externally)"""
        while True:
            yield time.time()


class ConcurrentStrategy(LoadGenerationStrategy):
    """
    Concurrent strategy: N independent streams, each stream is serial internally.

    Characteristics:
    - N parallel streams executing independently
    - Indirectly controls rate through concurrency
    - Simulates fixed number of concurrent clients

    Args:
        concurrency: Number of concurrent streams

    Example:
        strategy = ConcurrentStrategy(concurrency=8)
        # Simulates 8 clients making requests concurrently
    """

    strategy_type: str = "concurrent"
    concurrency: int = Field(gt=0, description="Number of concurrent streams")

    def get_semaphore(self):
        """Return semaphore with specified concurrency limit."""
        import asyncio
        return asyncio.Semaphore(self.concurrency)

    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """Send immediately (concurrency controlled by external semaphore)"""
        while True:
            yield time.time()

    def __str__(self) -> str:
        return f"{self.strategy_type}@{self.concurrency}"


class ThroughputStrategy(LoadGenerationStrategy):
    """
    Throughput strategy: sends all requests as fast as possible asynchronously.

    Characteristics:
    - Sends all requests immediately
    - Tests maximum server throughput
    - Suitable for stress testing

    Args:
        max_concurrency: Maximum concurrency limit

    Example:
        strategy = ThroughputStrategy(max_concurrency=512)
        # Sends all requests immediately, max 512 processing concurrently
    """

    strategy_type: str = "throughput"
    max_concurrency: Optional[int] = Field(
        default=None, description="Maximum concurrency limit")

    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """All requests sent at start time immediately"""
        init_time = self.start_time
        while True:
            yield init_time


class ConstantRateStrategy(LoadGenerationStrategy):
    """
    Constant rate strategy: sends requests at fixed time intervals.

    Characteristics:
    - Interval = 1 / rate
    - Precise rate control
    - No fluctuation

    Args:
        rate: Requests per second
        initial_burst: Whether to send floor(rate) requests at start to quickly reach target rate
        max_concurrency: Maximum concurrency limit

    Example:
        strategy = ConstantRateStrategy(rate=10.0, initial_burst=True)
        # Timeline:
        # t=0s: send 10 requests (initial burst)
        # t=0.1s: send request 11
        # t=0.2s: send request 12
        # ... one request every 100ms
    """

    strategy_type: str = "constant"
    rate: float = Field(gt=0, description="Target rate (req/s)")
    initial_burst: bool = Field(
        default=False,
        description="Whether to send initial burst to quickly reach target rate"
    )
    max_concurrency: Optional[int] = Field(
        default=None, description="Maximum concurrency limit")

    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """Generate send times at fixed intervals"""
        constant_increment = 1.0 / self.rate
        current_time = self.start_time

        # Initial burst: quickly send floor(rate) requests
        if self.initial_burst:
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield current_time
            current_time += constant_increment

        # Then send at fixed intervals
        counter = 0
        while True:
            yield current_time + constant_increment * counter
            counter += 1

    def __str__(self) -> str:
        return f"{self.strategy_type}@{self.rate:.2f}"


class PoissonRateStrategy(LoadGenerationStrategy):
    """
    Poisson rate strategy: request intervals follow exponential distribution.

    Characteristics:
    - Average rate = rate
    - Random intervals (exponentially distributed)
    - Simulates real user behavior

    Args:
        rate: Average requests per second
        initial_burst: Whether to send floor(rate) requests at start
        max_concurrency: Maximum concurrency limit
        random_seed: Random seed for reproducibility

    Mathematical principle:
        Poisson process inter-arrival times follow exponential distribution with parameter λ=rate
        inter_arrival_time ~ Exp(λ)

    Example:
        strategy = PoissonRateStrategy(rate=10.0, random_seed=42)
        # Timeline (example):
        # t=0s: send 10 requests (initial burst)
        # t=0.08s: send request (interval 80ms)
        # t=0.21s: send request (interval 130ms)
        # t=0.25s: send request (interval 40ms)
        # ... random intervals but average 100ms
    """

    strategy_type: str = "poisson"
    rate: float = Field(gt=0, description="Average rate (req/s)")
    initial_burst: bool = Field(default=False,
                                description="Whether to send initial burst")
    max_concurrency: Optional[int] = Field(
        default=None, description="Maximum concurrency limit")
    random_seed: int = Field(default=42, description="Random seed")

    async def _generate_request_times(self) -> AsyncGenerator[float, None]:
        """Generate send times following Poisson process"""
        current_time = self.start_time

        # Initial burst
        if self.initial_burst:
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield current_time
        else:
            yield current_time

        # Set random seed for reproducibility
        rand = random.Random(self.random_seed)

        # Poisson process: inter-arrival times follow exponential distribution
        while True:
            inter_arrival_time = rand.expovariate(self.rate)
            current_time += inter_arrival_time
            yield current_time

    def __str__(self) -> str:
        return f"{self.strategy_type}@{self.rate:.2f}"


# Export all strategies
__all__ = [
    "LoadGenerationStrategy",
    "SynchronousStrategy",
    "ConcurrentStrategy",
    "ThroughputStrategy",
    "ConstantRateStrategy",
    "PoissonRateStrategy",
]

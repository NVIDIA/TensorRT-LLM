"""Store-based broadcast signaling between a driver rank and workers.
Workers poll an incrementing counter in the process group's store instead of
blocking directly on a collective. An idle worker therefore neither trips the
NCCL watchdog timeout nor keeps an NCCL receive kernel spinning on its GPU
while the driver is still producing what it will broadcast.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed as dist


class BroadcastCoordinator:
    """Coordinates broadcast signaling between the driver and workers via a distributed store.
    Uses an incrementing counter to signal when a broadcast is ready. Workers
    poll for counter changes to avoid NCCL timeout issues during idle periods.
    The counter wraps at a large value to prevent overflow.
    """

    def __init__(self, store: dist.Store, is_driver: bool) -> None:
        """Initialize the broadcast coordinator.
        Args:
            store: Distributed store (TCPStore, PrefixStore, etc.) for coordination across ranks.
            is_driver: Whether this rank is the driver (signals broadcasts).
        """
        self.store = store
        self.is_driver = is_driver
        self.broadcast_key = "dist_pipeline_broadcast_ready"
        self.signal = 0
        self.signal_max = 2**31 - 1  # ~2.1 billion calls before wraparound

        if is_driver:
            self.store.set(self.broadcast_key, str(self.signal))

        self.last_seen_signal = self.get_current_signal() if not is_driver else 0

    @contextmanager
    def broadcast_context(self) -> Iterator[None]:
        """Signal a broadcast (driver only).
        Increments the signal counter on entry to notify workers that a
        broadcast is ready. The counter stays incremented for change detection.
        Raises:
            RuntimeError: If called on a non-driver rank.
        """
        self.notify()
        yield

    def notify(self) -> None:
        """Increment the signal counter so workers waiting on it proceed (driver only).
        Raises:
            RuntimeError: If called on a non-driver rank.
        """
        if not self.is_driver:
            raise RuntimeError("only the driver rank can signal a broadcast")
        self.signal = (self.signal + 1) % self.signal_max
        self.store.set(self.broadcast_key, str(self.signal))

    def wait_for_signal_change(self, poll_interval: float = 0.01, timeout: float | None = None) -> None:
        """Wait for the driver's next signal (worker only).
        Each call consumes exactly one signal. If the driver has signaled several times since
        the previous call, this returns at once and the next call consumes the next signal, so
        a worker that polls late never skips one.
        Args:
            poll_interval: Seconds to sleep between polls (default: 0.01).
            timeout: Seconds to wait before raising; ``None`` waits indefinitely.
        Raises:
            TimeoutError: If no new signal arrives within ``timeout``.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.get_current_signal() == self.last_seen_signal:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"no broadcast signal from the driver within {timeout} s")
            time.sleep(poll_interval)
        self.last_seen_signal = (self.last_seen_signal + 1) % self.signal_max

    def get_current_signal(self) -> int:
        """Get the current signal value from the store."""
        return int(self.store.get(self.broadcast_key).decode("utf-8"))

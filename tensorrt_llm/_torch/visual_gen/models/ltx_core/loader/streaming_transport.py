"""Per-device transfer queues and completion markers for streamed units.
A transport knows one device's mechanism: where transfers run, how they are ordered after earlier
work, and how completion is observed. It never sees the scheduler or the units it moves.
Compute completions use the calling thread's current stream at call time. The forward hooks call them, so
they cover the stream the triggering module runs on. A module that reads its streamed weights on
another stream must join that stream back before returning.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

import torch


class Completion(Protocol):
    """Completion of work issued on one device queue."""

    def wait_host(self) -> None:
        """Block the calling thread until the work finishes."""
        ...

    def wait_device(self) -> None:
        """Order later work on the calling thread's current stream after this work, without blocking the CPU.
        Only the consumer calls this, from the forward thread. Transports order their own queues explicitly.
        """
        ...


class Transfer(Protocol):
    """One transfer in progress. ``completion`` is valid once its context exits, normally or by an exception."""

    completion: Completion

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy on the transfer queue with this device's blocking rule."""
        ...


class Transport(Protocol):
    """Transfer mechanism for one streaming device."""

    device: torch.device
    transfer_thread: Literal["any", "consumer"]

    def transfer(self, after: Completion | None) -> AbstractContextManager[Transfer]:
        """Run the body on this device's transfer queue, ordered after ``after``.
        ``after`` is a completion this transport created, from ``compute_completion`` or an earlier transfer.
        """
        ...

    def compute_completion(self) -> Completion:
        """Completion of the work issued so far on the calling thread's current stream."""
        ...

    def close(self) -> None:
        """Wait for all work on this device, transfers and compute, so buffer storage can be freed."""
        ...


@dataclass(frozen=True, slots=True)
class CudaCompletion:
    """A recorded CUDA event on one device."""

    event: torch.cuda.Event
    device: torch.device

    def wait_host(self) -> None:
        # cudaEventSynchronize runs on the calling thread's current device and can create a context there.
        with torch.cuda.device(self.device):
            self.event.synchronize()

    def wait_device(self) -> None:
        torch.cuda.current_stream(self.device).wait_event(self.event)


@dataclass(frozen=True, slots=True)
class MpsCompletion:
    """A recorded event on MPS's single stream."""

    event: torch.mps.Event

    def wait_host(self) -> None:
        self.event.synchronize()

    def wait_device(self) -> None:
        self.event.wait()


class _Done:
    """Already complete: CPU work runs synchronously."""

    def wait_host(self) -> None:
        pass

    def wait_device(self) -> None:
        pass


DONE: Completion = _Done()


@dataclass(slots=True)
class _Transfer:
    non_blocking: bool = False
    completion: Completion = DONE

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src, non_blocking=self.non_blocking)


class CudaTransport:
    """Transfers on a side stream, ordered by events; no CPU synchronization per unit."""

    transfer_thread: Literal["any", "consumer"] = "any"

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError("CUDA transport needs an indexed CUDA device")
        self.device = device
        self._stream = torch.cuda.Stream(device=device)

    @contextmanager
    def transfer(self, after: Completion | None) -> Iterator[Transfer]:
        handle = _Transfer(non_blocking=True)
        if after is not None and (not isinstance(after, CudaCompletion) or after.device != self.device):
            raise TypeError("CUDA transfers order only after completions from the same device")
        # Worker threads keep their default device: every CUDA call here runs under this device.
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            if after is not None:
                self._stream.wait_event(after.event)
            try:
                yield handle
            finally:
                event = torch.cuda.Event()
                event.record(self._stream)
                handle.completion = CudaCompletion(event, self.device)

    def compute_completion(self) -> Completion:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return CudaCompletion(event, self.device)

    def close(self) -> None:
        # Not just the side stream: device buffers were allocated on the build thread's stream, and the caching
        # allocator orders their reuse only against that stream, not the one the forward ran on.
        torch.cuda.synchronize(self.device)


class MpsTransport:
    """Transfers issued from the model thread on MPS's single stream."""

    transfer_thread: Literal["any", "consumer"] = "consumer"

    def __init__(self, device: torch.device) -> None:
        if device.type != "mps":
            raise ValueError("MPS transport needs an MPS device")
        self.device = device

    @contextmanager
    def transfer(self, after: Completion | None) -> Iterator[Transfer]:
        # Transfers run on the consumer thread's stream, which already orders them after earlier compute.
        if after is not None:
            after.wait_device()
        handle = _Transfer()
        try:
            yield handle
        finally:
            event = torch.mps.Event()
            event.record()
            handle.completion = MpsCompletion(event)

    def compute_completion(self) -> Completion:
        event = torch.mps.Event()
        event.record()
        return MpsCompletion(event)

    def close(self) -> None:
        torch.mps.synchronize()


class CpuTransport:
    """Synchronous copies between CPU buffers."""

    transfer_thread: Literal["any", "consumer"] = "any"

    def __init__(self, device: torch.device) -> None:
        if device.type != "cpu":
            raise ValueError("CPU transport needs a CPU device")
        self.device = device

    @contextmanager
    def transfer(self, after: Completion | None) -> Iterator[Transfer]:
        if after is not None:
            after.wait_host()
        yield _Transfer()

    def compute_completion(self) -> Completion:
        return DONE

    def close(self) -> None:
        pass


def make_transport(device: torch.device) -> Transport:
    """Choose the transport for a streaming device; the only branch on device type."""
    if device.type == "cuda":
        return CudaTransport(device)
    if device.type == "mps":
        return MpsTransport(device)
    if device.type == "cpu":
        return CpuTransport(device)
    raise ValueError(f"unsupported streaming device {device}")

"""Shared CUDA multi-stream utilities for multi-stream transforms.

This module provides the core infrastructure for executing parts of an FX graph
on auxiliary CUDA streams.  It is consumed by the multi-stream MoE and MLA
attention transforms in ``..transform.library``.

Key components:
  - ``CudaStreamManager``: per-device singleton managing auxiliary streams/events.
  - Custom ops ``record_event`` / ``wait_event``: graph-safe event primitives.
  - Passthrough helpers that switch streams while preserving the data-flow edges
    required by FX graph execution and CUDA graph capture.
  - ``_make_aux_stream_impl``: factory for building an implementation that runs
    a base op on the auxiliary CUDA stream.
"""

from threading import RLock
from typing import Any, Callable, Dict, List

import torch

from .logger import ad_logger

# ---------------------------------------------------------------------------
# Singleton metaclass
# ---------------------------------------------------------------------------


class _Singleton(type):
    _instances: Dict[type, Any] = {}
    _lock = RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # double-checked locking
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ---------------------------------------------------------------------------
# CudaStreamManager
# ---------------------------------------------------------------------------

# Previously, CudaStreamManager and the custom ops that use the cuda streams and events were
# placed in custom_ops folder. However doing so resulted in CudaStreamManager
# being created only in the parent process, but we need each rank to have its own CudaStreamManager that
# manages the cuda streams and events for that rank. Placing the logic to instantiate
# CudaStreamManager and the custom ops that use the cuda streams and events at the transform level ensures that
# each rank has its own CudaStreamManager since each rank applies the transform independently.


class CudaStreamManager(metaclass=_Singleton):
    AUX_STREAM_NAME = "aux"
    MAIN_STREAM_NAME = "main"
    devices: List[torch.device] = []
    events: Dict[torch.device, Dict[str, Any]] = {}
    streams: Dict[torch.device, Dict[str, Any]] = {}
    # Per-device save slot for the caller's stream.  ``begin_aux_stream_passthrough``
    # saves the real current stream here so that ``end_aux_stream_passthrough`` can
    # restore it — this is critical during CUDA graph capture where the capture stream
    # differs from ``torch.cuda.default_stream()``.
    _caller_streams: Dict[int, Any] = {}

    def __init__(self) -> None:
        # In case __init__ ever gets called twice, guard against re-init
        if hasattr(self, "streams"):
            return

        self._lock = RLock()
        self.add_device(torch.cuda.current_device())

    def add_device(self, device: int) -> None:
        if device not in self.devices:
            self.devices.append(device)
            with torch.cuda.device(device):
                self.events[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Event(),
                    self.MAIN_STREAM_NAME: torch.cuda.Event(),
                }
                self.streams[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Stream(),
                    self.MAIN_STREAM_NAME: torch.cuda.default_stream(),
                }
        else:
            ad_logger.warning(f"CudaStreamManager: Device {device} already added")

    def get_stream(self, device: int, stream_name: str) -> torch.cuda.Stream:
        return self.streams[device][stream_name]

    def get_event(self, device: int, event_name: str) -> torch.cuda.Event:
        return self.events[device][event_name]


# Every device will have a singleton instance of CudaStreamManager.
cuda_stream_manager = CudaStreamManager()


# ---------------------------------------------------------------------------
# Custom ops — graph-safe CUDA event primitives
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.wait()


# ---------------------------------------------------------------------------
# Passthrough helpers
# ---------------------------------------------------------------------------


@torch._dynamo.disable
def record_event_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the main stream and return the input unchanged.

    Inserted after the gating/routing computation to mark a synchronization
    point.  The aux stream waits for this event before starting the MoE
    computation, enabling overlap between the shared expert (main stream)
    and routed experts (aux stream).
    """
    if device < 0:
        device = torch.cuda.current_device()
    torch.ops.auto_deploy.record_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    return x


@torch._dynamo.disable
def begin_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the main stream, switch to aux, and wait for it.

    After this function returns the thread-local current stream is the
    auxiliary stream.  All subsequent GPU ops dispatched by the FX graph
    interpreter will be recorded on aux until ``end_aux_stream_passthrough``
    switches back to main.
    """
    if device < 0:
        device = torch.cuda.current_device()
    # Save the *actual* current stream so ``end_aux`` can restore it.
    # During CUDA graph capture the current stream is the capture stream,
    # which is NOT ``torch.cuda.default_stream()``.
    caller_stream = torch.cuda.current_stream(device)
    cuda_stream_manager._caller_streams[device] = caller_stream
    # Record where the caller's stream has reached so aux knows when data is ready.
    main_event = cuda_stream_manager.get_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    main_event.record(caller_stream)
    # Switch the thread-local current stream to aux.
    aux_stream = cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.cuda.set_stream(aux_stream)
    # Make aux wait for the main-stream event before executing any work.
    aux_stream.wait_event(main_event)
    return x


@torch._dynamo.disable
def end_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the aux stream and switch back to the caller's stream.

    This does **not** make the caller's stream wait for aux.  The caller must
    insert ``wait_aux_stream_passthrough`` at the point where both branches
    need to be synchronised (typically right before the ``add`` that merges
    shared-expert and routed-expert outputs).
    """
    if device < 0:
        device = torch.cuda.current_device()
    # Record the aux-stream progress so the caller's stream can wait for it later.
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    aux_event.record()
    # Restore the caller's stream saved by ``begin_aux_stream_passthrough``.
    # This is critical during CUDA graph capture where the capture stream
    # differs from ``torch.cuda.default_stream()``.
    caller_stream = cuda_stream_manager._caller_streams.pop(device, None)
    if caller_stream is not None:
        torch.cuda.set_stream(caller_stream)
    else:
        torch.cuda.set_stream(
            cuda_stream_manager.get_stream(device, cuda_stream_manager.MAIN_STREAM_NAME)
        )
    return x


@torch._dynamo.disable
def wait_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Make the current stream wait for the auxiliary stream's last recorded event.

    This is a GPU-side wait (non-blocking on the CPU).  Insert this right
    before the ``add`` that merges shared-expert output (computed on aux)
    with routed-expert output (computed on main).

    Uses ``torch.cuda.current_stream()`` rather than the stored default stream
    so that the correct stream is waited on during CUDA graph capture.
    """
    if device < 0:
        device = torch.cuda.current_device()
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.cuda.current_stream(device).wait_event(aux_event)
    return x


# ---------------------------------------------------------------------------
# Aux-stream implementation factory
# ---------------------------------------------------------------------------


def _make_aux_stream_impl(base_overload: Callable) -> Callable:
    """Build an implementation that runs *base_overload* on the auxiliary CUDA stream."""

    def _impl(*args, **kwargs):
        device = torch.cuda.current_device()
        with torch.cuda.stream(
            cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
        ):
            torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
            output = base_overload(*args, **kwargs)
            torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
        return output

    return _impl

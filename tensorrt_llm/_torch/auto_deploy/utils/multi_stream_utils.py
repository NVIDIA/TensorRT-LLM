# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
  - ``disable_multi_stream`` context manager: turn all passthroughs and aux-stream
    impls into no-ops for code paths where multi-stream execution is unsafe
    (e.g. piecewise CUDA graph capture/replay for prefill/mixed batches).
"""

from threading import RLock
from typing import Any, Callable, Dict, List, Tuple, TypeVar

import torch

from .logger import ad_logger

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Runtime enable/disable flag
# ---------------------------------------------------------------------------
# When False, all multi-stream passthroughs return ``x`` unchanged and
# ``_make_aux_stream_impl`` impls run the base op on the caller's stream.
# Used by the piecewise CUDA graph path to keep prefill/mixed batches on a
# single stream — host-side ``caller_stream.synchronize()`` between captured
# segments is incompatible with stable-address invariants required for replay.
# Decode-only batches use the monolithic CG path and leave the flag set.
_multi_stream_enabled: bool = True

# DEBUG: counters for diagnosing why MoE layers' multi_stream doesn't fire in
# trace. Output via atexit hook below.
import os as _os  # noqa: E402

_DEBUG_COUNTERS = _os.environ.get("AD_DEBUG_MULTI_STREAM", "") == "1"
_begin_aux_count = 0
_begin_aux_early_returns = 0
_begin_aux_in_capture = 0
_begin_aux_out_capture = 0
_end_aux_count = 0
_end_aux_early_returns = 0
_wait_aux_count = 0
_wait_aux_early_returns = 0

if _DEBUG_COUNTERS:
    import atexit as _atexit

    @_atexit.register
    def _print_multi_stream_counts():
        pid = _os.getpid()
        print(
            f"[MS-COUNT pid={pid}] begin_aux calls={_begin_aux_count} early={_begin_aux_early_returns} in_capture={_begin_aux_in_capture} out_capture={_begin_aux_out_capture}",  # noqa: E501
            flush=True,
        )
        print(
            f"[MS-COUNT pid={pid}] end_aux   calls={_end_aux_count} early={_end_aux_early_returns}",
            flush=True,
        )
        print(
            f"[MS-COUNT pid={pid}] wait_aux  calls={_wait_aux_count} early={_wait_aux_early_returns}",
            flush=True,
        )


def is_multi_stream_enabled() -> bool:
    return _multi_stream_enabled


class disable_multi_stream:
    """Context manager that disables multi-stream execution.

    Nestable: saves the previous flag value on ``__enter__`` and restores it
    on ``__exit__``.
    """

    def __enter__(self) -> "disable_multi_stream":
        global _multi_stream_enabled
        self._prev = _multi_stream_enabled
        _multi_stream_enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _multi_stream_enabled
        _multi_stream_enabled = self._prev


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
    # V13: per-fork-id event pool. Each (device, name, event_id) gets its own
    # CUDA event so that the cuda graph capture engine does not see the same
    # event recorded N times (which causes events to be deduplicated/dropped
    # in the graph DAG, reducing actual aux-stream activity).
    events_per_id: Dict[int, Dict[Tuple[str, int], Any]] = {}
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

    def get_event_for_id(self, device: int, event_name: str, event_id: int) -> torch.cuda.Event:
        """V13: lazily allocate a unique CUDA event per (event_name, event_id) tuple.

        Forks that share the same name+id share the same event (e.g., the begin/
        end of a single MLA fork). Forks with different ids get distinct events,
        preventing CUDA-graph-capture-time event deduplication that occurs when
        the same event is recorded many times in one captured graph.
        """
        if event_id < 0:
            # Backwards-compat: caller did not pass an id, use shared global event
            return self.events[device][event_name]
        if device not in self.events_per_id:
            self.events_per_id[device] = {}
        key = (event_name, event_id)
        pool = self.events_per_id[device]
        if key not in pool:
            with torch.cuda.device(device):
                pool[key] = torch.cuda.Event()
        return pool[key]


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


def _record_stream_for_tensor_outputs(x: object, stream: torch.cuda.Stream) -> None:
    if isinstance(x, torch.Tensor):
        x.record_stream(stream)
        return
    if isinstance(x, (list, tuple)):
        for item in x:
            _record_stream_for_tensor_outputs(item, stream)
        return
    if isinstance(x, dict):
        for item in x.values():
            _record_stream_for_tensor_outputs(item, stream)


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
    if not _multi_stream_enabled:
        return x
    if device < 0:
        device = torch.cuda.current_device()
    torch.ops.auto_deploy.record_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    return x


@torch._dynamo.disable
def begin_aux_stream_passthrough(
    x: T,
    *,
    device: int = -1,
    event_id: int = -1,
) -> T:
    """Record a CUDA event on the main stream, switch to aux, and wait for it.

    After this function returns the thread-local current stream is the
    auxiliary stream.  All subsequent GPU ops dispatched by the FX graph
    interpreter will be recorded on aux until ``end_aux_stream_passthrough``
    switches back to main.

    Args:
        event_id: V13 — if >= 0, uses a per-id event pool so each fork gets a
            unique CUDA event (avoids cuda graph capture deduplication of
            shared events).
    """
    global _begin_aux_count, _begin_aux_early_returns
    global _begin_aux_in_capture, _begin_aux_out_capture
    _begin_aux_count += 1
    if not _multi_stream_enabled:
        _begin_aux_early_returns += 1
        return x
    if torch.cuda.is_current_stream_capturing():
        _begin_aux_in_capture += 1
    else:
        _begin_aux_out_capture += 1
    if device < 0:
        device = torch.cuda.current_device()
    # Save the *actual* current stream so ``end_aux`` can restore it.
    # During CUDA graph capture the current stream is the capture stream,
    # which is NOT ``torch.cuda.default_stream()``.
    caller_stream = torch.cuda.current_stream(device)
    cuda_stream_manager._caller_streams[device] = caller_stream
    # Record where the caller's stream has reached so aux knows when data is ready.
    if event_id >= 0:
        main_event = cuda_stream_manager.get_event_for_id(
            device, cuda_stream_manager.MAIN_STREAM_NAME, event_id
        )
    else:
        main_event = cuda_stream_manager.get_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    main_event.record(caller_stream)
    # Switch the thread-local current stream to aux.
    aux_stream = cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    # Tell the caching allocator that x is also live on aux_stream.  Without
    # this, MLIR/Triton kernels that produced x on the main stream can cause
    # the allocator to recycle x's backing storage before aux-stream work
    # has consumed it, leading to silent data corruption or illegal accesses.
    # record_stream is the idiomatic PyTorch solution for this cross-stream
    # liveness problem: it is a CPU-only allocator hint, never a GPU sync,
    # so it cannot deadlock with in-flight NCCL collectives.
    _record_stream_for_tensor_outputs(x, aux_stream)
    torch.cuda.set_stream(aux_stream)
    # Make aux wait for the main-stream event before executing any work.
    aux_stream.wait_event(main_event)
    return x


@torch._dynamo.disable
def end_aux_stream_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
    event_id: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the aux stream and switch back to the caller's stream.

    This does **not** make the caller's stream wait for aux.  The caller must
    insert ``wait_aux_stream_passthrough`` at the point where both branches
    need to be synchronised (typically right before the ``add`` that merges
    shared-expert and routed-expert outputs).

    Args:
        event_id: V13 — if >= 0, uses a per-id event pool (paired with begin/wait).
    """
    global _end_aux_count, _end_aux_early_returns
    _end_aux_count += 1
    if not _multi_stream_enabled:
        _end_aux_early_returns += 1
        return x
    if device < 0:
        device = torch.cuda.current_device()
    # Record the aux-stream progress so the caller's stream can wait for it later.
    if event_id >= 0:
        aux_event = cuda_stream_manager.get_event_for_id(
            device, cuda_stream_manager.AUX_STREAM_NAME, event_id
        )
    else:
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
    event_id: int = -1,
) -> torch.Tensor:
    """Make the current stream wait for the auxiliary stream's last recorded event.

    This is a GPU-side wait (non-blocking on the CPU).  Insert this right
    before the ``add`` that merges shared-expert output (computed on aux)
    with routed-expert output (computed on main).

    Uses ``torch.cuda.current_stream()`` rather than the stored default stream
    so that the correct stream is waited on during CUDA graph capture.

    Args:
        event_id: V13 — if >= 0, uses a per-id event pool (paired with begin/end).
    """
    global _wait_aux_count, _wait_aux_early_returns
    _wait_aux_count += 1
    if not _multi_stream_enabled:
        _wait_aux_early_returns += 1
        return x
    if device < 0:
        device = torch.cuda.current_device()
    if event_id >= 0:
        aux_event = cuda_stream_manager.get_event_for_id(
            device, cuda_stream_manager.AUX_STREAM_NAME, event_id
        )
    else:
        aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.cuda.current_stream(device).wait_event(aux_event)
    return x


# ---------------------------------------------------------------------------
# Aux-stream implementation factory
# ---------------------------------------------------------------------------


def _make_aux_stream_impl(base_overload: Callable) -> Callable:
    """Build an implementation that runs *base_overload* on the auxiliary CUDA stream."""

    def _impl(*args, **kwargs):
        if not _multi_stream_enabled:
            return base_overload(*args, **kwargs)
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


# ---------------------------------------------------------------------------
# V19: MoE shared-expert wrap helper
# ---------------------------------------------------------------------------


@torch._dynamo.disable
def dsv3_moe_shared_mlp_in_aux_wrapped(
    input_tensor: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_proj_input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    gate_proj_alpha: torch.Tensor,
    down_proj_input_scale: torch.Tensor,
    down_proj_weight_scale: torch.Tensor,
    down_proj_alpha: torch.Tensor,
) -> torch.Tensor:
    """V19: Run fused_nvfp4_swiglu_mlp (shared expert) in aux stream.

    Wraps begin_aux + fused MLP + end_aux into a single @torch._dynamo.disable
    function. The corresponding wait_aux is left as a separate fx node by the
    transform (it does not call set_stream(), so it does not split capture).

    Purpose: collapse 3 fx-graph nodes (begin_aux, fused_mlp, end_aux) into 1
    single call_function node, so cuda graph capture does not allocate
    per-MoE-layer dedicated streams.
    """

    def _mlp():
        return torch.ops.auto_deploy.fused_nvfp4_swiglu_mlp.default(
            input_tensor,
            gate_up_weight,
            down_proj_weight,
            gate_proj_input_scale,
            gate_up_weight_scale,
            gate_proj_alpha,
            down_proj_input_scale,
            down_proj_weight_scale,
            down_proj_alpha,
        )

    if not _multi_stream_enabled:
        return _mlp()

    device = torch.cuda.current_device()
    aux_stream = cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    main_event = cuda_stream_manager.get_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)

    caller_stream = torch.cuda.current_stream(device)
    main_event.record(caller_stream)
    if not torch.cuda.is_current_stream_capturing():
        input_tensor.record_stream(aux_stream)
        gate_up_weight.record_stream(aux_stream)
        down_proj_weight.record_stream(aux_stream)
    with torch.cuda.stream(aux_stream):
        aux_stream.wait_event(main_event)
        result = _mlp()
        aux_event.record(aux_stream)
    # NOTE: caller_stream does NOT wait_event(aux_event) here — the wait is
    # left as a separate wait_aux fx node placed before the merge add, so the
    # routed-expert work on main can run concurrently with this shared MLP.
    # We record aux_event on a shared event (cuda_stream_manager.events[...][AUX_STREAM])
    # so the wait_aux passthrough fx node can wait on the same event.
    return result

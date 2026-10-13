# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
UGPU Localization utilities for PyTorch backend.

Provides per-device global resources for UGPU localization including:
- UgpuLocalizationHandle: Handle for UGPU operations (one per GPU device)
- Streams: One for each UGPU (uGPU0 and uGPU1) per GPU device
- Allocators: PyTorch CUDA allocators for each UGPU per GPU device

All resources are lazily initialized on first use for each device.
"""

import ctypes
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
from torch.cuda.memory import CUDAPluggableAllocator

import tensorrt_llm as trtllm
import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._utils import get_sm_version

__all__ = [
    "get_ugpu_stream",
    "get_ugpu_mempool",
    "get_ugpu_compute_sm_counts",
    "is_ugpu_supported",
    "initialize_ugpu_resources",
    "ugpu_device",
    "get_current_ugpu",
    "start_for_all_ugpu",
    "end_for_all_ugpu",
    "get_reserved_remainder_stream",
    "node_local_max_active_clusters",
]


class UgpuResourceManager:
    """
    Manager class that holds all UGPU-related global resources.

    This class centralizes UGPU resources (streams, memory pools, events,
    allocators) and provides explicit control over their lifecycle,
    including manual cleanup/destruction.

    Usage:
        # Get the global instance
        manager = get_ugpu_resource_manager()

        # Access resources
        streams = manager.streams
        mempools = manager.mempools

        # Manual cleanup when needed
        manager.cleanup()

        # Or reset the global manager entirely
        reset_ugpu_resource_manager()
    """

    def __init__(self):
        # Per-device resources, Key: (device_id, ugpu_id)
        self.streams: dict[tuple[int, int], torch.cuda.Stream] = {}
        self.mempools: dict[tuple[int, int], torch.cuda.MemPool] = {}
        # Per-device compute topology, Key: (device_id, ugpu_id),
        # Value: (localized partition SM count, full-device SM count).
        self.compute_sm_counts: dict[tuple[int, int], tuple[int, int] | None] = {}
        # Per-device non-localized remainder stream for the strict public split.
        self.reserved_remainder_streams: dict[int, torch.cuda.Stream | None] = {}
        # Per-device events, Key: device_id
        self.events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = {}
        # Shared allocator holders and allocators (not per-device)
        self.allocator_holders: list = []
        self.allocators: list = []
        # Set of initialized device IDs
        self.initialized_devices: set[int] = set()
        # Thread-local storage for current uGPU ID
        self.current_ugpu = threading.local()
        # Thread-local storage for tracking if we're inside a mem_pool context
        # This prevents nested use_mem_pool calls which cause "already recording to mempool_id" error
        self.in_mem_pool_context = threading.local()

        # This hacky WAR is used to avoid crash during application exit.
        # Which is caused by cudaGraph may cause Custom Allocator's refcount not zero.
        # The WAR is to increase the uGPU resource's refcount to not release it by Python at exit.
        pythonapi = ctypes.pythonapi
        pythonapi.Py_IncRef.argtypes = [ctypes.py_object]
        pythonapi.Py_DecRef.argtypes = [ctypes.py_object]
        pythonapi.Py_IncRef(self)

    def cleanup(self) -> None:
        """
        This method clears all stored resources including streams, memory pools,
        events, and allocators. After calling this, the manager will be in a
        fresh state and resources will be lazily re-initialized on next use.

        Note: This does NOT destroy the underlying CUDA resources immediately,
        as they may still be referenced elsewhere. It only clears our references.
        """
        self.streams.clear()
        self.compute_sm_counts.clear()
        self.reserved_remainder_streams.clear()
        self.events.clear()
        import gc

        torch.cuda.empty_cache()
        gc.collect()
        self.allocator_holders.clear()
        self.allocators.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        self.mempools.clear()
        self.initialized_devices.clear()
        # Reset thread-local storage
        if hasattr(self.current_ugpu, "id"):
            delattr(self.current_ugpu, "id")
        if hasattr(self.in_mem_pool_context, "active"):
            delattr(self.in_mem_pool_context, "active")

    def __del__(self) -> None:
        self.cleanup()

    def is_initialized(self, device_id: int) -> bool:
        """Check if resources are initialized for a specific device."""
        return device_id in self.initialized_devices

    def mark_initialized(self, device_id: int) -> None:
        """Mark a device as initialized."""
        self.initialized_devices.add(device_id)


# Global manager instance (lazily created)
_ugpu_resource_manager: UgpuResourceManager | None = None
_manager_lock = threading.Lock()


def get_ugpu_resource_manager() -> UgpuResourceManager:
    """
    Get the global UGPU resource manager instance.

    Returns:
        UgpuResourceManager: The global manager instance.
    """
    global _ugpu_resource_manager
    if _ugpu_resource_manager is None:
        with _manager_lock:
            if _ugpu_resource_manager is None:
                _ugpu_resource_manager = UgpuResourceManager()
    return _ugpu_resource_manager


def reset_ugpu_resource_manager() -> None:
    """
    Reset the global UGPU resource manager.

    This performs cleanup on the existing manager (if any) and then
    sets the global manager to None, so a fresh one will be created
    on next access.
    """
    global _ugpu_resource_manager
    with _manager_lock:
        if _ugpu_resource_manager is not None:
            _ugpu_resource_manager.cleanup()
            _ugpu_resource_manager = None


def cleanup_ugpu_resources() -> None:
    """
    Cleanup all UGPU resources.

    This clears all stored resources including streams, memory pools,
    events, and allocators. After calling this, resources will be
    lazily re-initialized on next use.
    """
    global _ugpu_resource_manager
    with _manager_lock:
        if _ugpu_resource_manager is not None:
            _ugpu_resource_manager.cleanup()


def is_ugpu_supported() -> bool:
    """
    Check if UGPU localization is supported on this system.

    Returns:
        bool: True if UGPU localization is supported, False otherwise.
    """
    try:
        handle = _tbr.UgpuLocalizationHandle()
        return handle.supports_ugpu_localization()
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_ugpu_enabled() -> bool:
    """
    Check if UGPU localization is enabled on this system.
    """
    if os.getenv("DISABLE_UGPU", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    if get_sm_version() != 107:
        return False
    return is_ugpu_supported()


def get_current_ugpu() -> int | None:
    """
    Get the current UGPU ID for this thread.

    Returns:
        int | None: The current UGPU ID (0 or 1), or None if not set.

    Example:
        >>> with ugpu_device(0):
        ...     ugpu_id = get_current_ugpu()  # Returns 0
        >>> ugpu_id = get_current_ugpu()  # Returns None
    """
    manager = get_ugpu_resource_manager()
    return getattr(manager.current_ugpu, "id", None)


@contextmanager
def optional_ugpu_mem_pool(use_ugpu: bool = True):
    current_ugpu = get_current_ugpu()
    if not use_ugpu or current_ugpu is None:
        # No change on current allocator
        yield
    else:
        manager = get_ugpu_resource_manager()
        # Check if we're already in a mem_pool context to prevent nested calls
        # PyTorch's use_mem_pool doesn't support re-entry and will raise
        # "RuntimeError: beginAllocateToPool: already recording to mempool_id"
        if getattr(manager.in_mem_pool_context, "active", False):
            # Already in a mem_pool context, just yield without entering again
            yield
        else:
            # enter torch.cuda.use_mem_pool
            assert isinstance(current_ugpu, int)
            pool = get_ugpu_mempool(current_ugpu)
            manager.in_mem_pool_context.active = True
            try:
                with torch.cuda.use_mem_pool(pool):
                    yield
            finally:
                manager.in_mem_pool_context.active = False


@contextmanager
def ugpu_device(ugpu_id: int | None):
    """
    Context manager to set the current UGPU device.

    This context manager allows you to specify which UGPU should be used
    within its scope. It properly saves and restores the previous UGPU ID,
    allowing for nested usage.

    If UGPU is not enabled (is_ugpu_enabled() returns False), this context
    manager becomes a no-op and the current UGPU will remain None.

    Args:
        ugpu_id: The UGPU ID to use (0, 1, or None to disable UGPU).

    Yields:
        None

    Raises:
        ValueError: If ugpu_id is not 0, 1, or None.

    Example:
        >>> # Use uGPU 0
        >>> with ugpu_device(0):
        ...     stream = get_ugpu_stream(0)
        ...     # Operations here use uGPU 0
        >>> # Nested usage
        >>> with ugpu_device(0):
        ...     print(get_current_ugpu())  # 0
        ...     with ugpu_device(1):
        ...         print(get_current_ugpu())  # 1
        ...     print(get_current_ugpu())  # 0
        >>> # Disable UGPU
        >>> with ugpu_device(None):
        ...     print(get_current_ugpu())  # None
    """
    if ugpu_id is not None and ugpu_id not in [0, 1]:
        raise ValueError(f"ugpu_id must be 0, 1, or None, got {ugpu_id}")

    # If UGPU is not enabled, do nothing and keep current UGPU as None
    if not is_ugpu_enabled():
        yield
        return

    # Save old value
    old_ugpu_id = get_current_ugpu()
    manager = get_ugpu_resource_manager()

    try:
        # Set new value
        manager.current_ugpu.id = ugpu_id
        yield
    finally:
        # Restore old value
        manager.current_ugpu.id = old_ugpu_id


def initialize_ugpu_allocators():
    """
    Initialize UGPU allocators.
    - Two uGPU allocator, one for uGPU0, one for uGPU1, all devices share same allocator.
    """
    manager = get_ugpu_resource_manager()
    if len(manager.allocators) > 0:
        assert len(manager.allocators) == 2
        return
    trtllm_dir = Path(trtllm.__file__).parent
    th_common_libname = "th_common"
    th_common_dir = trtllm_dir / "libs"
    th_common_so = th_common_dir / f"lib{th_common_libname}.so"
    ugpu_allocator_holder0 = CUDAPluggableAllocator(
        th_common_so, "trtllm_ugpu0_alloc", "trtllm_ugpu0_free"
    )
    ugpu_allocator_holder1 = CUDAPluggableAllocator(
        th_common_so, "trtllm_ugpu1_alloc", "trtllm_ugpu1_free"
    )
    ugpu_allocator0 = ugpu_allocator_holder0.allocator()
    ugpu_allocator1 = ugpu_allocator_holder1.allocator()
    manager.allocator_holders = [ugpu_allocator_holder0, ugpu_allocator_holder1]
    manager.allocators = [ugpu_allocator0, ugpu_allocator1]


def initialize_ugpu_resources() -> None:
    """
    Initialize UGPU resources for current device including:
    - Two CUDA streams (one for uGPU0, one for uGPU1)
    - Two CUDA uGPU MemPool (one for uGPU0, one for uGPU1)

    This function is idempotent - calling it multiple times for the same device is safe.
    Resources are only initialized on the first call for each device.

    Raises:
        RuntimeError: If UGPU localization is not supported on this system.
    """
    manager = get_ugpu_resource_manager()

    # Get current device
    device_id = torch.cuda.current_device()

    # Already initialized for this device
    if manager.is_initialized(device_id):
        return

    initialize_ugpu_allocators()

    # Create the UgpuLocalizationHandle for this device
    with torch.cuda.device(device_id):
        ugpu_handle = _tbr.UgpuLocalizationHandle()

        if not ugpu_handle.supports_ugpu_localization():
            raise RuntimeError(
                f"UGPU localization is not supported on device {device_id}. "
                "Please ensure you are running on a system with UGPU support."
            )

        # Initialize resources for uGPU0 and uGPU1
        for ugpu_id in [0, 1]:
            get_sm_counts = getattr(ugpu_handle, "get_ugpu_compute_sm_counts", None)
            sm_counts = get_sm_counts(ugpu_id) if get_sm_counts else (0, 0)
            partition_sm_count, total_sm_count = (int(value) for value in sm_counts)
            manager.compute_sm_counts[(device_id, ugpu_id)] = (
                (partition_sm_count, total_sm_count)
                if 0 < partition_sm_count <= total_sm_count
                else None
            )

            # Create UGPU localized stream
            stream_ptr = ugpu_handle.create_ugpu_localized_stream(ugpu_id)
            # The C++ process-lifetime singleton owns the cached raw stream.
            manager.streams[(device_id, ugpu_id)] = torch.cuda.ExternalStream(
                stream_ptr, device=device_id
            )

            ugpu_mempool = torch.cuda.MemPool(manager.allocators[ugpu_id])
            manager.mempools[(device_id, ugpu_id)] = ugpu_mempool

        # The strict public split creates a third, non-localized green context
        # from the remainder resource (12 SMs on Rubin). Its stream is owned by
        # the process-lifetime C++ uGPU resource and borrowed by PyTorch.
        get_remainder_stream = getattr(
            ugpu_handle,
            "get_reserved_remainder_stream",
            None,
        )
        remainder_ptr = get_remainder_stream() if get_remainder_stream else 0
        manager.reserved_remainder_streams[device_id] = (
            torch.cuda.ExternalStream(remainder_ptr, device=device_id) if remainder_ptr else None
        )

        manager.events[device_id] = (
            torch.cuda.Event(blocking=False, interprocess=False),
            torch.cuda.Event(blocking=False, interprocess=False),
            torch.cuda.Event(blocking=False, interprocess=False),
        )

        manager.mark_initialized(device_id)


def get_ugpu_stream(ugpu_id: int) -> torch.cuda.Stream:
    """
    Get the CUDA stream for the specified UGPU on a specific device.

    Args:
        ugpu_id (int): The UGPU ID (0 or 1).

    Returns:
        torch.cuda.Stream: The CUDA stream for the specified UGPU and device.

    Raises:
        ValueError: If ugpu_id is not 0 or 1.
        RuntimeError: If UGPU localization is not supported.
    """
    if ugpu_id not in [0, 1]:
        raise ValueError(f"ugpu_id must be 0 or 1, got {ugpu_id}")

    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()

    if not manager.is_initialized(device_id):
        initialize_ugpu_resources()

    return manager.streams[(device_id, ugpu_id)]


def get_ugpu_compute_sm_counts(ugpu_id: int) -> tuple[int, int] | None:
    """Return the actual compute-partition and full-device SM counts."""
    if ugpu_id not in [0, 1]:
        raise ValueError(f"ugpu_id must be 0 or 1, got {ugpu_id}")

    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_ugpu_resources()
    return manager.compute_sm_counts.get((device_id, ugpu_id))


def node_local_max_active_clusters(max_active_full_device: int) -> int | None:
    """Scale a full-device cluster limit to the active uGPU partition.

    The cached counts come from the validated public CUDA resource split, so
    strict partitions scale to the locality-domain SM count while balanced
    partitions include their backfilled SMs.
    """
    ugpu_id = get_current_ugpu()
    if ugpu_id is None:
        return None

    sm_counts = get_ugpu_compute_sm_counts(ugpu_id)
    if sm_counts is None:
        return None
    partition_sm_count, total_sm_count = sm_counts
    if max_active_full_device <= 0 or not 0 < partition_sm_count <= total_sm_count:
        return None
    return max(1, max_active_full_device * partition_sm_count // total_sm_count)


def get_reserved_remainder_stream() -> torch.cuda.Stream | None:
    """Return the strict split's non-localized remainder stream, if present."""
    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_ugpu_resources()
    return manager.reserved_remainder_streams.get(device_id)


def start_for_all_ugpu():
    """
    Start uGPU work, will wait on current stream.
    """
    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_ugpu_resources()
    base_event = manager.events[device_id][2]
    current_stream = torch.cuda.current_stream()
    current_stream.record_event(base_event)
    for ugpu_id in [0, 1]:
        ugpu_stream = get_ugpu_stream(ugpu_id)
        ugpu_stream.wait_event(base_event)


def end_for_all_ugpu():
    """
    End uGPU work, will record event.
    """
    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()
    current_stream = torch.cuda.current_stream()
    for ugpu_id in [0, 1]:
        ugpu_stream = get_ugpu_stream(ugpu_id)
        ugpu_event = manager.events[device_id][ugpu_id]
        ugpu_stream.record_event(ugpu_event)

    for ugpu_id in [0, 1]:
        ugpu_event = manager.events[device_id][ugpu_id]
        current_stream.wait_event(ugpu_event)


def get_ugpu_mempool(ugpu_id: int) -> torch.cuda.MemPool:
    """
    Get the CUDA memory pool for the specified UGPU on a specific device.

    Args:
        ugpu_id (int): The UGPU ID (0 or 1).

    Returns:
        torch.cuda.MemPool: The CUDA memory pool/allocator for the specified UGPU and device.

    Raises:
        ValueError: If ugpu_id is not 0 or 1.
        RuntimeError: If UGPU localization is not supported or allocator is not available.
    """
    if ugpu_id not in [0, 1]:
        raise ValueError(f"ugpu_id must be 0 or 1, got {ugpu_id}")

    manager = get_ugpu_resource_manager()
    device_id = torch.cuda.current_device()

    if not manager.is_initialized(device_id):
        initialize_ugpu_resources()

    # Check if allocator was successfully created
    pool_key = (device_id, ugpu_id)
    if pool_key not in manager.mempools:
        raise RuntimeError(
            f"UGPU allocator for uGPU{ugpu_id} on device {device_id} is not available. "
            "The allocator may have failed to initialize during resource setup."
        )

    return manager.mempools[pool_key]

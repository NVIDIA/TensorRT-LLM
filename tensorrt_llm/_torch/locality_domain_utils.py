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
LOCALITY_DOMAIN Localization utilities for PyTorch backend.

Provides per-device global resources for LOCALITY_DOMAIN localization including:
- LocalizationHandle: Handle for LOCALITY_DOMAIN operations (one per GPU device)
- Streams: One for each LOCALITY_DOMAIN (locality domain 0 and locality domain 1) per GPU device
- Allocators: PyTorch CUDA allocators for each LOCALITY_DOMAIN per GPU device

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
    "get_locality_domain_stream",
    "get_locality_domain_mempool",
    "get_locality_domain_compute_sm_counts",
    "is_locality_domain_supported",
    "initialize_locality_domain_resources",
    "locality_domain_device",
    "get_current_locality_domain",
    "start_for_all_locality_domain",
    "end_for_all_locality_domain",
    "get_reserved_remainder_stream",
    "node_local_max_active_clusters",
]


class LocalityDomainResourceManager:
    """
    Manager class that holds all LOCALITY_DOMAIN-related global resources.

    This class centralizes LOCALITY_DOMAIN resources (streams, memory pools, events,
    allocators) and provides explicit control over their lifecycle,
    including manual cleanup/destruction.

    Usage:
        # Get the global instance
        manager = get_locality_domain_resource_manager()

        # Access resources
        streams = manager.streams
        mempools = manager.mempools

        # Manual cleanup when needed
        manager.cleanup()

        # Or reset the global manager entirely
        reset_locality_domain_resource_manager()
    """

    def __init__(self):
        # Per-device resources, Key: (device_id, locality_domain_id)
        self.streams: dict[tuple[int, int], torch.cuda.Stream] = {}
        self.mempools: dict[tuple[int, int], torch.cuda.MemPool] = {}
        # Per-device compute topology, Key: (device_id, locality_domain_id),
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
        # Thread-local storage for current locality domain ID
        self.current_locality_domain = threading.local()
        # Thread-local storage for tracking if we're inside a mem_pool context
        # This prevents nested use_mem_pool calls which cause "already recording to mempool_id" error
        self.in_mem_pool_context = threading.local()

        # This hacky WAR is used to avoid crash during application exit.
        # Which is caused by cudaGraph may cause Custom Allocator's refcount not zero.
        # The WAR is to increase the locality domain resource's refcount to not release it by Python at exit.
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
        if hasattr(self.current_locality_domain, "id"):
            delattr(self.current_locality_domain, "id")
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
_locality_domain_resource_manager: LocalityDomainResourceManager | None = None
_manager_lock = threading.Lock()

# Guards resource initialization. Held across the whole check-create-mark sequence so
# concurrent callers cannot both create resources. One global lock rather than a per-device
# one: the allocators are shared by every device, so per-device locking would not serialize
# their creation. Initialization runs once per device at startup, so the contention is
# irrelevant. This must not be _manager_lock: the initialization path calls
# get_locality_domain_resource_manager(), which takes that lock, and threading.Lock is not
# reentrant.
_init_lock = threading.Lock()


def get_locality_domain_resource_manager() -> LocalityDomainResourceManager:
    """
    Get the global LOCALITY_DOMAIN resource manager instance.

    Returns:
        LocalityDomainResourceManager: The global manager instance.
    """
    global _locality_domain_resource_manager
    if _locality_domain_resource_manager is None:
        with _manager_lock:
            if _locality_domain_resource_manager is None:
                _locality_domain_resource_manager = LocalityDomainResourceManager()
    return _locality_domain_resource_manager


def reset_locality_domain_resource_manager() -> None:
    """
    Reset the global LOCALITY_DOMAIN resource manager.

    This performs cleanup on the existing manager (if any) and then
    sets the global manager to None, so a fresh one will be created
    on next access.
    """
    global _locality_domain_resource_manager
    with _manager_lock:
        if _locality_domain_resource_manager is not None:
            _locality_domain_resource_manager.cleanup()
            _locality_domain_resource_manager = None


def cleanup_locality_domain_resources() -> None:
    """
    Cleanup all LOCALITY_DOMAIN resources.

    This clears all stored resources including streams, memory pools,
    events, and allocators. After calling this, resources will be
    lazily re-initialized on next use.
    """
    global _locality_domain_resource_manager
    with _manager_lock:
        if _locality_domain_resource_manager is not None:
            _locality_domain_resource_manager.cleanup()


def is_locality_domain_supported(device: int | None = None) -> bool:
    """
    Check if LOCALITY_DOMAIN localization is supported on this system.

    This issues a driver attribute query only. It creates no CUDA context and does not
    partition the device, so it is safe to call before the caller has selected its device.

    Args:
        device: CUDA device ordinal to query. Defaults to the current device.

    Returns:
        bool: True if LOCALITY_DOMAIN localization is supported, False otherwise.
    """
    try:
        if device is None:
            device = torch.cuda.current_device()
        return _tbr.device_supports_locality_domain(device)
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_locality_domain_enabled() -> bool:
    """
    Check if LOCALITY_DOMAIN localization is enabled on this system.
    """
    if os.getenv("DISABLE_LOCALITY_DOMAINS", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    if get_sm_version() != 107:
        return False
    return is_locality_domain_supported()


def get_current_locality_domain() -> int | None:
    """
    Get the current LOCALITY_DOMAIN ID for this thread.

    Returns:
        int | None: The current LOCALITY_DOMAIN ID (0 or 1), or None if not set.

    Example:
        >>> with locality_domain_device(0):
        ...     locality_domain_id = get_current_locality_domain()  # Returns 0
        >>> locality_domain_id = get_current_locality_domain()  # Returns None
    """
    manager = get_locality_domain_resource_manager()
    return getattr(manager.current_locality_domain, "id", None)


@contextmanager
def optional_locality_domain_mem_pool(use_locality_domain: bool = True):
    current_locality_domain = get_current_locality_domain()
    if not use_locality_domain or current_locality_domain is None:
        # No change on current allocator
        yield
    else:
        manager = get_locality_domain_resource_manager()
        # Check if we're already in a mem_pool context to prevent nested calls
        # PyTorch's use_mem_pool doesn't support re-entry and will raise
        # "RuntimeError: beginAllocateToPool: already recording to mempool_id"
        if getattr(manager.in_mem_pool_context, "active", False):
            # Already in a mem_pool context, just yield without entering again
            yield
        else:
            # enter torch.cuda.use_mem_pool
            assert isinstance(current_locality_domain, int)
            pool = get_locality_domain_mempool(current_locality_domain)
            manager.in_mem_pool_context.active = True
            try:
                with torch.cuda.use_mem_pool(pool):
                    yield
            finally:
                manager.in_mem_pool_context.active = False


@contextmanager
def locality_domain_device(locality_domain_id: int | None):
    """
    Context manager to set the current LOCALITY_DOMAIN device.

    This context manager allows you to specify which LOCALITY_DOMAIN should be used
    within its scope. It properly saves and restores the previous LOCALITY_DOMAIN ID,
    allowing for nested usage.

    If LOCALITY_DOMAIN is not enabled (is_locality_domain_enabled() returns False), this context
    manager becomes a no-op and the current LOCALITY_DOMAIN will remain None.

    Args:
        locality_domain_id: The LOCALITY_DOMAIN ID to use (0, 1, or None to disable LOCALITY_DOMAIN).

    Yields:
        None

    Raises:
        ValueError: If locality_domain_id is not 0, 1, or None.

    Example:
        >>> # Use locality domain 0
        >>> with locality_domain_device(0):
        ...     stream = get_locality_domain_stream(0)
        ...     # Operations here use locality domain 0
        >>> # Nested usage
        >>> with locality_domain_device(0):
        ...     print(get_current_locality_domain())  # 0
        ...     with locality_domain_device(1):
        ...         print(get_current_locality_domain())  # 1
        ...     print(get_current_locality_domain())  # 0
        >>> # Disable LOCALITY_DOMAIN
        >>> with locality_domain_device(None):
        ...     print(get_current_locality_domain())  # None
    """
    if locality_domain_id is not None and locality_domain_id not in [0, 1]:
        raise ValueError(f"locality_domain_id must be 0, 1, or None, got {locality_domain_id}")

    # If LOCALITY_DOMAIN is not enabled, do nothing and keep current LOCALITY_DOMAIN as None
    if not is_locality_domain_enabled():
        yield
        return

    # Save old value
    old_locality_domain_id = get_current_locality_domain()
    manager = get_locality_domain_resource_manager()

    try:
        # Set new value
        manager.current_locality_domain.id = locality_domain_id
        yield
    finally:
        # Restore old value
        manager.current_locality_domain.id = old_locality_domain_id


def initialize_locality_domain_allocators():
    """
    Initialize locality-domain allocators.
    - Two locality domain allocators, one for locality domain 0 and one for
      locality domain 1; all devices share the same allocator.
    """
    manager = get_locality_domain_resource_manager()
    if len(manager.allocators) > 0:
        assert len(manager.allocators) == 2
        return

    _create_locality_domain_allocators(manager)


def _create_locality_domain_allocators(manager: LocalityDomainResourceManager) -> None:
    """Create the process-wide allocators. Caller must hold _allocator_lock."""
    trtllm_dir = Path(trtllm.__file__).parent
    th_common_libname = "th_common"
    th_common_dir = trtllm_dir / "libs"
    th_common_so = th_common_dir / f"lib{th_common_libname}.so"
    locality_domain_allocator_holder0 = CUDAPluggableAllocator(
        th_common_so, "trtllm_locality_domain0_alloc", "trtllm_locality_domain0_free"
    )
    locality_domain_allocator_holder1 = CUDAPluggableAllocator(
        th_common_so, "trtllm_locality_domain1_alloc", "trtllm_locality_domain1_free"
    )
    locality_domain_allocator0 = locality_domain_allocator_holder0.allocator()
    locality_domain_allocator1 = locality_domain_allocator_holder1.allocator()
    manager.allocator_holders = [
        locality_domain_allocator_holder0,
        locality_domain_allocator_holder1,
    ]
    manager.allocators = [locality_domain_allocator0, locality_domain_allocator1]


def initialize_locality_domain_resources() -> None:
    """
    Initialize LOCALITY_DOMAIN resources for current device including:
    - Two CUDA streams (one for locality domain 0, one for locality domain 1)
    - Two CUDA locality domain MemPool (one for locality domain 0, one for locality domain 1)

    This function is idempotent - calling it multiple times for the same device is safe.
    Resources are only initialized on the first call for each device.

    Raises:
        RuntimeError: If LOCALITY_DOMAIN localization is not supported on this system.
    """
    manager = get_locality_domain_resource_manager()

    # Get current device
    device_id = torch.cuda.current_device()

    # Already initialized for this device
    if manager.is_initialized(device_id):
        return

    with _init_lock:
        # Re-check under the lock: another thread may have initialized this device
        # between the check above and acquiring the lock.
        if manager.is_initialized(device_id):
            return

        _initialize_locality_domain_resources_locked(manager, device_id)


def _initialize_locality_domain_resources_locked(
    manager: LocalityDomainResourceManager, device_id: int
) -> None:
    """Create this device's resources. Caller must hold the device's init lock."""
    initialize_locality_domain_allocators()

    # Create the LocalizationHandle for this device
    with torch.cuda.device(device_id):
        locality_domain_handle = _tbr.LocalizationHandle()

        if not locality_domain_handle.supports_localization():
            raise RuntimeError(
                f"LOCALITY_DOMAIN localization is not supported on device {device_id}. "
                "Please ensure you are running on a system with LOCALITY_DOMAIN support."
            )

        # Initialize resources for locality domain 0 and locality domain 1
        for locality_domain_id in [0, 1]:
            get_sm_counts = getattr(
                locality_domain_handle, "get_locality_domain_compute_sm_counts", None
            )
            sm_counts = get_sm_counts(locality_domain_id) if get_sm_counts else (0, 0)
            partition_sm_count, total_sm_count = (int(value) for value in sm_counts)
            manager.compute_sm_counts[(device_id, locality_domain_id)] = (
                (partition_sm_count, total_sm_count)
                if 0 < partition_sm_count <= total_sm_count
                else None
            )

            # Create LOCALITY_DOMAIN localized stream
            stream_ptr = locality_domain_handle.create_localized_stream(locality_domain_id)
            # The C++ process-lifetime singleton owns the cached raw stream.
            manager.streams[(device_id, locality_domain_id)] = torch.cuda.ExternalStream(
                stream_ptr, device=device_id
            )

            locality_domain_mempool = torch.cuda.MemPool(manager.allocators[locality_domain_id])
            manager.mempools[(device_id, locality_domain_id)] = locality_domain_mempool

        # The strict public split creates a third, non-localized green context
        # from the remainder resource. Its stream is owned by the
        # process-lifetime C++ locality domain resource and borrowed by PyTorch.
        get_remainder_stream = getattr(
            locality_domain_handle,
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


def get_locality_domain_stream(locality_domain_id: int) -> torch.cuda.Stream:
    """
    Get the CUDA stream for the specified LOCALITY_DOMAIN on a specific device.

    Args:
        locality_domain_id (int): The LOCALITY_DOMAIN ID (0 or 1).

    Returns:
        torch.cuda.Stream: The CUDA stream for the specified LOCALITY_DOMAIN and device.

    Raises:
        ValueError: If locality_domain_id is not 0 or 1.
        RuntimeError: If LOCALITY_DOMAIN localization is not supported.
    """
    if locality_domain_id not in [0, 1]:
        raise ValueError(f"locality_domain_id must be 0 or 1, got {locality_domain_id}")

    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()

    if not manager.is_initialized(device_id):
        initialize_locality_domain_resources()

    return manager.streams[(device_id, locality_domain_id)]


def get_locality_domain_compute_sm_counts(locality_domain_id: int) -> tuple[int, int] | None:
    """Return the actual compute-partition and full-device SM counts."""
    if locality_domain_id not in [0, 1]:
        raise ValueError(f"locality_domain_id must be 0 or 1, got {locality_domain_id}")

    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_locality_domain_resources()
    return manager.compute_sm_counts.get((device_id, locality_domain_id))


def node_local_max_active_clusters(max_active_full_device: int) -> int | None:
    """Scale a full-device cluster limit to the active locality domain partition.

    The cached counts come from the validated public CUDA resource split, so
    strict partitions scale to the locality-domain SM count while balanced
    partitions include their backfilled SMs.
    """
    locality_domain_id = get_current_locality_domain()
    if locality_domain_id is None:
        return None

    sm_counts = get_locality_domain_compute_sm_counts(locality_domain_id)
    if sm_counts is None:
        return None
    partition_sm_count, total_sm_count = sm_counts
    if max_active_full_device <= 0 or not 0 < partition_sm_count <= total_sm_count:
        return None
    return max(1, max_active_full_device * partition_sm_count // total_sm_count)


def get_reserved_remainder_stream() -> torch.cuda.Stream | None:
    """Return the strict split's non-localized remainder stream, if present."""
    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_locality_domain_resources()
    return manager.reserved_remainder_streams.get(device_id)


def start_for_all_locality_domain():
    """
    Start locality domain work, will wait on current stream.
    """
    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()
    if not manager.is_initialized(device_id):
        initialize_locality_domain_resources()
    base_event = manager.events[device_id][2]
    current_stream = torch.cuda.current_stream()
    current_stream.record_event(base_event)
    for locality_domain_id in [0, 1]:
        locality_domain_stream = get_locality_domain_stream(locality_domain_id)
        locality_domain_stream.wait_event(base_event)


def end_for_all_locality_domain():
    """
    End locality domain work, will record event.
    """
    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()
    current_stream = torch.cuda.current_stream()
    for locality_domain_id in [0, 1]:
        locality_domain_stream = get_locality_domain_stream(locality_domain_id)
        locality_domain_event = manager.events[device_id][locality_domain_id]
        locality_domain_stream.record_event(locality_domain_event)

    for locality_domain_id in [0, 1]:
        locality_domain_event = manager.events[device_id][locality_domain_id]
        current_stream.wait_event(locality_domain_event)


def get_locality_domain_mempool(locality_domain_id: int) -> torch.cuda.MemPool:
    """
    Get the CUDA memory pool for the specified LOCALITY_DOMAIN on a specific device.

    Args:
        locality_domain_id (int): The LOCALITY_DOMAIN ID (0 or 1).

    Returns:
        torch.cuda.MemPool: The CUDA memory pool/allocator for the specified LOCALITY_DOMAIN and device.

    Raises:
        ValueError: If locality_domain_id is not 0 or 1.
        RuntimeError: If LOCALITY_DOMAIN localization is not supported or allocator is not available.
    """
    if locality_domain_id not in [0, 1]:
        raise ValueError(f"locality_domain_id must be 0 or 1, got {locality_domain_id}")

    manager = get_locality_domain_resource_manager()
    device_id = torch.cuda.current_device()

    if not manager.is_initialized(device_id):
        initialize_locality_domain_resources()

    # Check if allocator was successfully created
    pool_key = (device_id, locality_domain_id)
    if pool_key not in manager.mempools:
        raise RuntimeError(
            f"locality-domain allocator for locality domain {locality_domain_id} "
            f"on device {device_id} is not available. The allocator may have "
            "failed to initialize during resource setup."
        )

    return manager.mempools[pool_key]

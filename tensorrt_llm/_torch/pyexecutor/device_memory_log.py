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
"""Periodic device and host memory snapshot for the executor iteration log.

The snapshot never synchronizes the device: ``cudaMemGetInfo`` and the PyTorch
caching-allocator counters are host-side queries, and ``/proc/meminfo`` is a
small read.

Device memory is not the only limit: every device virtual-memory mapping
(``cuMemMap`` + ``cuMemSetAccess`` of local or imported peer memory) takes an
entry in a driver table shared by all processes and GPUs on a node, and new
mappings fail with CUDA_ERROR_OUT_OF_MEMORY once it is full even if
``device_free`` is large. ``torch_mappings`` counts PyTorch's live device
allocation calls (one per expandable-segment page).
"""

import os

import torch

DEVICE_MEM_LOG_INTERVAL_ENV = "TLLM_DEVICE_MEM_LOG_INTERVAL"
# Off by default: a snapshot costs ~0.2-0.9 s on a busy GEN rank (PyTorch allocator stats over thousands of
# expandable-segment mappings) and stalls every in-flight request; enable it only for memory debugging.
DEFAULT_DEVICE_MEM_LOG_INTERVAL = 0

_GIB = float(1 << 30)


def device_mem_log_interval() -> int:
    """Iterations between snapshots; 0 or a negative value disables them."""
    raw = os.environ.get(DEVICE_MEM_LOG_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_DEVICE_MEM_LOG_INTERVAL
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_DEVICE_MEM_LOG_INTERVAL


def host_mem_available_bytes(path: str = "/proc/meminfo") -> int:
    """MemAvailable of the host in bytes, or -1 if unavailable."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return -1


def device_memory_snapshot(device: int | None = None) -> str:
    """One-line summary of device, PyTorch and host memory state.

    Args:
        device: CUDA device index; the current device if None.

    Returns:
        ``key = value`` pairs. ``non_torch_used`` is device memory in use that
        PyTorch has not reserved (KV cache pools, communication buffers, driver
        allocations); growth there is what a PyTorch cap derived from the free
        memory at setup cannot absorb.
    """
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    non_torch = total - free - reserved
    stats = torch.cuda.memory_stats(device)
    # Live cudaMalloc / cuMemMap calls of the caching allocator.
    torch_mappings = stats.get("num_device_alloc", 0) - stats.get("num_device_free", 0)
    host_avail = host_mem_available_bytes()
    host_avail_str = f"{host_avail / _GIB:.1f}GiB" if host_avail >= 0 else "N/A"
    return (
        f"device_free = {free / _GIB:.2f}GiB, "
        f"device_total = {total / _GIB:.2f}GiB, "
        f"torch_reserved = {reserved / _GIB:.2f}GiB, "
        f"torch_max_reserved = {max_reserved / _GIB:.2f}GiB, "
        f"torch_allocated = {allocated / _GIB:.2f}GiB, "
        f"non_torch_used = {non_torch / _GIB:.2f}GiB, "
        f"torch_mappings = {torch_mappings}, "
        f"host_mem_available = {host_avail_str}"
    )

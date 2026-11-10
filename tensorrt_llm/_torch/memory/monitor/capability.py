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
"""Allocator capability probe.

Which numbers mean anything depends on the allocator backend, so the probe
ships with the frame rather than after it. Under
``PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`` many ``memory_stats`` fields are
permanently zero; reporting those zeros as measurements would be worse than
reporting nothing.

The probe records configuration rather than judging it. One combination is
worth recording explicitly: TensorRT-LLM's own guidance recommends
``expandable_segments:True``, while the sleep/virtual-memory and GMS paths
install ``MemPool``\\ s backed by pluggable allocators. The two coexist
without error, but expandable segments are silently disabled inside such a
pool (its segments report ``is_expandable=False``). That is a
silent downgrade, not a conflict, so the probe reports the facts and does not
warn.
"""

import os
from dataclasses import dataclass

import torch

__all__ = [
    "NVML_UNAVAILABLE",
    "Capability",
    "metadata_supported",
    "nvml_process_usage",
    "probe_capability",
    "tagging_supported",
]

#: ``PYTORCH_CUDA_ALLOC_CONF`` wins over ``PYTORCH_ALLOC_CONF``; the two are
#: not merged (first match wins in c10's AllocatorConfig).
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


@dataclass(frozen=True)
class Capability:
    """What this process can actually measure."""

    allocator_backend: str
    alloc_conf: str
    alloc_conf_var: str | None
    expandable_segments: bool
    metadata_supported: bool
    nvml_available: bool

    def describe(self) -> str:
        parts = [
            f"backend={self.allocator_backend}",
            f"expandable_segments={'on' if self.expandable_segments else 'off'}",
            f"metadata={'yes' if self.metadata_supported else 'no'}",
            f"nvml={'yes' if self.nvml_available else 'no'}",
        ]
        if self.alloc_conf:
            parts.append(f"{self.alloc_conf_var}={self.alloc_conf!r}")
        return " ".join(parts)


def metadata_supported() -> bool:
    """Whether the allocator keeps the metadata a tag is stored in.

    A process-wide pluggable allocator answers by raising rather than
    returning False; that is read as unsupported.
    """
    probe = getattr(torch.cuda.memory, "_memory_metadata_supported", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except RuntimeError:
        return False


def tagging_supported() -> bool:
    """Whether a tag set here would actually be recorded.

    Returns False rather than raising, but never reports a tag total of zero
    on an allocator that drops tags -- the caller is expected to render the
    whole section as unsupported.
    """
    memory = torch.cuda.memory
    if not metadata_supported():
        return False
    if not hasattr(memory, "_set_memory_metadata"):
        return False
    if not hasattr(memory, "_record_memory_history"):
        return False
    # Anything but the native caching allocator accepts the call and
    # discards the tag.
    return torch.cuda.get_allocator_backend() == "native"


def _alloc_conf() -> tuple[str, str | None]:
    for var in _ALLOC_CONF_VARS:
        value = os.environ.get(var)
        if value:
            return value, var
    return "", None


#: Why :func:`nvml_process_usage` has no figure when NVML cannot be queried.
NVML_UNAVAILABLE = "NVML unavailable"

#: NVML answered, but not for this process.
_NOT_LISTED = (
    "this process is not in NVML's process list -- in a container without the "
    "host PID namespace NVML reports host PIDs"
)


def nvml_process_usage(device_index: int) -> tuple[int | None, str]:
    """This process's device memory according to NVML, and why not if not.

    ``None`` when NVML cannot answer, with :data:`NVML_UNAVAILABLE` or a
    more specific reason, since the failures call for different fixes.
    Falling back to ``total - free`` would silently attribute colocated
    processes' memory to this one.
    """
    unavailable = (None, NVML_UNAVAILABLE)
    try:
        # pynvml is optional; without it the report says NVML is unavailable.
        import pynvml

        from tensorrt_llm import profiler
    except ImportError:
        return unavailable
    pynvml_context = getattr(profiler, "pynvml_context", None)
    if pynvml_context is None:
        return unavailable
    try:
        # By UUID, not by index: torch numbers only the devices
        # CUDA_VISIBLE_DEVICES exposes, NVML numbers every device on the
        # host, and the two disagree whenever the variable remaps them.
        uuid = f"GPU-{torch.cuda.get_device_properties(device_index).uuid}"
    except (AssertionError, RuntimeError):
        return unavailable
    try:
        with pynvml_context():
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            pid = os.getpid()
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle):
                if proc.pid == pid:
                    used = proc.usedGpuMemory
                    # NVML reports None/0 for this field on some platforms;
                    # treat that as "unknown" rather than "nothing".
                    if used:
                        return int(used), ""
                    return None, "NVML reports no per-process usage on this platform"
    except pynvml.NVMLError:
        return unavailable
    return None, _NOT_LISTED


def _nvml_available() -> bool:
    """Whether NVML initialises and reports at least one device.

    An importable helper is not enough: on some hosts it imports while every
    query fails.
    """
    try:
        # pynvml is optional; without it the report says NVML is unavailable.
        import pynvml

        from tensorrt_llm import profiler
    except ImportError:
        return False
    context = getattr(profiler, "pynvml_context", None)
    if context is None:
        return False
    try:
        with context():
            return pynvml.nvmlDeviceGetCount() > 0
    except pynvml.NVMLError:
        return False


def probe_capability(nvml_available: bool | None = None) -> Capability:
    """Probe the allocator without perturbing it.

    Performs no allocation, no cache release and no synchronisation: a
    diagnostic that changes the thing it measures is worse than no
    diagnostic. ``nvml_available`` lets a caller that has just queried NVML
    pass the answer instead of opening a second NVML session.
    """
    conf, conf_var = _alloc_conf()
    return Capability(
        allocator_backend=torch.cuda.get_allocator_backend(),
        alloc_conf=conf,
        alloc_conf_var=conf_var,
        expandable_segments="expandable_segments:True" in conf,
        metadata_supported=metadata_supported(),
        nvml_available=_nvml_available() if nvml_available is None else nvml_available,
    )

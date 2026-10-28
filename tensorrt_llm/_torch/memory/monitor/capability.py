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
install ``MemPool``\\ s backed by pluggable allocators. Measured on torch 2.14,
the two coexist without error, but expandable segments are silently disabled
inside such a pool (its segments report ``is_expandable=False``). That is a
silent downgrade, not a conflict, so the probe reports the facts and does not
warn.
"""

import os
from dataclasses import dataclass, field

import torch

__all__ = ["Capability", "probe_capability"]

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
    history_api_present: bool
    snapshot_api_present: bool
    oom_observer_present: bool
    nvml_available: bool

    #: ``memory_stats`` keys that exist but are not meaningful on this
    #: backend. Consumers must render these as ``unsupported``, never as 0.
    unsupported_stats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def stats_trustworthy(self) -> bool:
        return self.allocator_backend == "native"

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


def _alloc_conf() -> tuple[str, str | None]:
    for var in _ALLOC_CONF_VARS:
        value = os.environ.get(var)
        if value:
            return value, var
    return "", None


def _nvml_available() -> bool:
    try:
        from tensorrt_llm import profiler
    except ImportError:
        return False
    return bool(getattr(profiler, "pynvml_context", None))


def probe_capability() -> Capability:
    """Probe the allocator without perturbing it.

    Performs no allocation, no cache release and no synchronisation: a
    diagnostic that changes the thing it measures is worse than no
    diagnostic.
    """
    conf, conf_var = _alloc_conf()
    backend = torch.cuda.get_allocator_backend()

    memory = torch.cuda.memory
    metadata_supported = False
    probe = getattr(memory, "_memory_metadata_supported", None)
    if probe is not None:
        try:
            metadata_supported = bool(probe())
        except RuntimeError:
            # A pluggable allocator installed process-wide answers by
            # raising rather than returning False.
            metadata_supported = False

    # Under cudaMallocAsync torch keeps these keys but never populates them.
    unsupported: tuple[str, ...] = ()
    if backend != "native":
        unsupported = (
            "reserved_bytes.all.current",
            "inactive_split_bytes.all.current",
            "active_bytes.all.current",
        )

    return Capability(
        allocator_backend=backend,
        alloc_conf=conf,
        alloc_conf_var=conf_var,
        expandable_segments="expandable_segments:True" in conf,
        metadata_supported=metadata_supported,
        history_api_present=hasattr(memory, "_record_memory_history"),
        snapshot_api_present=hasattr(memory, "_snapshot"),
        oom_observer_present=hasattr(torch._C, "_cuda_attach_out_of_memory_observer"),
        nvml_available=_nvml_available(),
        unsupported_stats=unsupported,
    )

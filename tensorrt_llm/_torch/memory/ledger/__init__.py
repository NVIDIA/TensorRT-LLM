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
"""GPU memory ledger: where did the device memory go?

Enabled with ``TLLM_MEM_LEDGER=1``; off by default and zero-cost when off.

Usage::

    from tensorrt_llm._torch.memory.ledger import capture, render_capture

    snapshot = capture(tag="after_weights")
    if snapshot is not None:
        print(render_capture(snapshot))

The ledger only reads. It never calls ``empty_cache()``, never synchronises,
and never resets peak stats -- ``_util.configure_kv_cache_capacity`` already
resets the global peak, and a second resetter would silently corrupt its
measurements.
"""

import os
from dataclasses import dataclass

import torch

from .capability import Capability, probe_capability
from .frame import DeviceReading, Frame, TorchReading, build_frame
from .report import render, summarize_by_category
from .schema import Basis, Category, Entry, Placement, Source

__all__ = [
    "Basis",
    "Capability",
    "Category",
    "Entry",
    "LedgerCapture",
    "Placement",
    "Source",
    "capture",
    "enabled",
    "render_capture",
    "summarize_by_category",
]

ENV_VAR = "TLLM_MEM_LEDGER"
RANK_ENV_VAR = "TLLM_MEM_LEDGER_RANKS"


@dataclass(frozen=True)
class LedgerCapture:
    """One resolved snapshot plus the context needed to read it."""

    frame: Frame
    capability: Capability
    tag: str
    trigger: str
    pid: int
    rank: int | None
    device_index: int


def enabled(rank: int | None = None) -> bool:
    """Whether the ledger should run in this process.

    ``TLLM_MEM_LEDGER_RANKS`` optionally restricts collection to a
    comma-separated rank list, so a multi-rank job can pay the cost on one
    rank only.
    """
    if os.environ.get(ENV_VAR, "") != "1":
        return False
    ranks = os.environ.get(RANK_ENV_VAR, "")
    if not ranks or rank is None:
        return True
    try:
        wanted = {int(part) for part in ranks.split(",") if part.strip()}
    except ValueError:
        return True
    return rank in wanted


def _read_device(device_index: int) -> DeviceReading:
    free, total = torch.cuda.mem_get_info(device_index)
    return DeviceReading(total=total, free=free, process_used=_nvml_process_used(device_index))


def _nvml_process_used(device_index: int) -> int | None:
    """This process's device memory according to NVML.

    ``None`` when NVML cannot answer. Falling back to ``total - free`` would
    silently attribute colocated processes' memory to us; TensorRT-LLM
    supports several colocated configurations, so that is a real hazard
    rather than a theoretical one.
    """
    try:
        from tensorrt_llm import profiler
    except ImportError:
        return None
    pynvml_context = getattr(profiler, "pynvml_context", None)
    if pynvml_context is None:
        return None
    try:
        import pynvml
    except ImportError:
        return None
    try:
        with pynvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pid = os.getpid()
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle):
                if proc.pid == pid:
                    used = proc.usedGpuMemory
                    # NVML reports None/0 for this field on some platforms;
                    # treat that as "unknown" rather than "nothing".
                    return int(used) if used else None
    except Exception:  # noqa: BLE001 - NVML raises many unrelated types
        return None
    return None


def _read_torch(device_index: int) -> TorchReading:
    stats = torch.cuda.memory_stats(device_index)
    live = stats.get("allocated_bytes.all.current")
    if live is None:
        live = torch.cuda.memory_allocated(device_index)
    return TorchReading(reserved=torch.cuda.memory_reserved(device_index), live=live)


#: Tags the executor sets during startup, mapped onto ledger categories.
#: A tag with no mapping is still reported, under its own name, rather than
#: dropped -- an unmapped tag is a gap in this table, not absent memory.
_TAG_CATEGORIES = {
    "WEIGHTS": Category.WEIGHTS,
    "SAMPLER": Category.SAMPLER,
}


def _tag_entries() -> list[Entry]:
    """Ledger entries for the frozen startup tags.

    Returns nothing when tagging is unsupported or the startup window never
    ran, so the torch-side rows stay absent rather than appearing as zeros.

    ``untagged`` is deliberately not emitted here: the frame already derives
    an ``unattributed_live`` row from ``torch_live`` minus the tagged total,
    and emitting both would double count.
    """
    from .tagging import UNTAGGED, get_frozen_map, live_bytes_by_tag

    by_tag = live_bytes_by_tag(get_frozen_map())
    entries: list[Entry] = []
    for label, size in sorted(by_tag.items()):
        if label == UNTAGGED or not size:
            continue
        entries.append(
            Entry(
                category=_TAG_CATEGORIES.get(label, Category.TAGGED_OTHER),
                size_bytes=size,
                source=Source.TAG,
                placement=Placement.TORCH,
                basis=Basis.LOGICAL,
                owner=label,
            )
        )
    return entries


def capture(
    *,
    tag: str = "",
    trigger: str = "",
    entries: list[Entry] | None = None,
    rank: int | None = None,
    device_index: int | None = None,
) -> LedgerCapture | None:
    """Collect a snapshot, or ``None`` when the ledger is disabled.

    ``entries`` carries provider contributions. Step 1 ships no providers, so
    the default report shows the frame with everything in
    ``unattributed_live`` and ``nontorch_residual`` -- which is the honest
    picture before attribution exists.
    """
    if not enabled(rank):
        return None
    if not torch.cuda.is_available():
        return None

    if device_index is None:
        device_index = torch.cuda.current_device()

    if entries is None:
        from .providers import collect_all

        entries = collect_all() + _tag_entries()

    return LedgerCapture(
        frame=build_frame(_read_device(device_index), _read_torch(device_index), entries),
        capability=probe_capability(),
        tag=tag,
        trigger=trigger,
        pid=os.getpid(),
        rank=rank,
        device_index=device_index,
    )


def render_capture(snapshot: LedgerCapture) -> str:
    """Render a capture produced by :func:`capture`."""
    return render(
        snapshot.frame,
        snapshot.capability,
        tag=snapshot.tag,
        trigger=snapshot.trigger,
        pid=snapshot.pid,
        rank=snapshot.rank,
        device_index=snapshot.device_index,
    )

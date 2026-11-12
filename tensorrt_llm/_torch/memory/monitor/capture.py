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
"""Taking a snapshot of this process's GPU memory and rendering it.

One allocator snapshot serves the PyTorch reading, the split by pool and the
tags, so the figures in one report describe one moment. Reading memory never
changes it: no ``empty_cache()``, no synchronisation, no peak-stat reset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from . import settings, startup
from .capability import NVML_UNAVAILABLE, Capability, nvml_process_usage, probe_capability
from .checks import CheckResult, check_kv_pool_conservation, check_tag_coverage
from .frame import DeviceReading, Frame, TorchReading, build_frame
from .names import CUDA_GRAPHS, UNTAGGED, WEIGHTS
from .pools import pool_usage
from .providers import collect_all
from .report import render
from .schema import Basis, Category, Entry, Group, Placement, Source
from .summary import group_totals
from .tagging import get_frozen_map, live_attribution, live_bytes_by_tag

__all__ = ["MonitorCapture", "capture", "read_device", "read_torch", "render_capture"]


@dataclass(frozen=True)
class MonitorCapture:
    """One resolved snapshot plus the context needed to read it."""

    frame: Frame
    capability: Capability
    tag: str
    pid: int
    rank: int | None
    device_index: int
    checks: tuple[CheckResult, ...] = ()
    #: One line saying what happened, shown above the summary (e.g. the
    #: failed allocation on OOM).
    headline: str = ""
    #: Group totals at the post-warmup baseline, when one was recorded.
    baseline: dict[Group, int | None] | None = None
    #: Advice that overrides the summary's own, when the cause is known.
    suggestion: str = ""
    #: Whether the summary may give tuning advice; off for a failure that
    #: is not about memory.
    advise: bool = True
    #: Whether to list memory taken by each startup step.
    startup_steps: bool = False
    #: Untagged PyTorch memory by allocating code; empty unless stack
    #: recording (``TLLM_MEM_MONITOR_STACKS=1``) is on.
    unattributed_sites: dict[str, int] | None = None


def read_device(device_index: int) -> DeviceReading:
    """Device totals and this process's usage according to NVML."""
    free, total = torch.cuda.mem_get_info(device_index)
    used, reason = nvml_process_usage(device_index)
    return DeviceReading(
        total=total, free=free, process_used=used, process_missing_reason=reason or ""
    )


def read_torch(device_index: int, segments: list[dict] | None = None) -> TorchReading:
    """Read the caching allocator, refusing a reserved figure it cannot mean.

    Whether ``reserved`` is populated depends on the backend *and* the torch
    version -- on 2.14 ``cudaMallocAsync`` reports it, older versions leave
    it at zero -- so it is judged from the data rather than from the backend
    name. Reserved below live is impossible for a populated reading, and is
    taken to mean "not reported".

    With ``segments`` (this device's allocator snapshot), ``reserved`` and
    ``live`` are taken from it as well: the same snapshot splits memory by
    pool and by tag, and figures read at a different moment would drift from
    those splits, so the groups would no longer add up.
    """
    stats = torch.cuda.memory_stats(device_index)
    live = stats.get("allocated_bytes.all.current")
    if live is None:
        live = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    if reserved < live:
        return TorchReading(reserved=None, live=live)
    if segments is None:
        return TorchReading(reserved=reserved, live=live)
    pools = tuple(pool_usage(segments))
    # A pool sleep mode has released is not resident; it is kept in
    # ``pools`` so the report can say what is asleep.
    resident = [p for p in pools if not p.released]
    return TorchReading(
        reserved=sum(p.reserved for p in resident),
        live=sum(p.allocated for p in resident),
        pools=pools,
    )


def _segments(device_index: int) -> list[dict] | None:
    """This device's allocator segments, or ``None`` if they cannot be read."""
    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except (RuntimeError, KeyError):
        return None
    return [s for s in segments if s.get("device", device_index) == device_index]


#: Labels attached during startup, by the category they report under.
#: ``WEIGHTS`` comes from claiming the model; the rest are the executor's
#: creation steps (``ExecutorMemoryType`` values), each of which tags what
#: it allocates. A step whose allocations are temporary maps to the same
#: category as its permanent counterpart, so anything it leaks lands there.
#: A label with no mapping is still reported, under its own name, rather
#: than dropped -- an unmapped label is a gap in this table, not absent
#: memory.
_TAG_CATEGORIES: dict[str, Category] = {
    WEIGHTS: Category.WEIGHTS,
    CUDA_GRAPHS: Category.CUDA_GRAPHS,
    "model": Category.MODEL_RUNTIME,
    "model_extra": Category.MODEL_RUNTIME,
    "sampler": Category.SAMPLER,
    "drafter": Category.SPECULATIVE,
    "draft_model": Category.SPECULATIVE,
    "spec_resource_manager": Category.SPECULATIVE,
    "guided_decoder": Category.GUIDED_DECODER,
    "kv_cache": Category.KV_CACHE_AUX,
    "_no_capture_init_kv_cache": Category.KV_CACHE_AUX,
    "executor_extra": Category.EXECUTOR_RUNTIME,
    "_no_capture_init_extra_resources": Category.EXECUTOR_RUNTIME,
}


def _tag_entries(by_tag: dict[str, int] | None = None) -> list[Entry]:
    """Monitor entries for the startup tags and claimed modules.

    Returns nothing when nothing was attributed, so the torch-side rows stay
    absent rather than appearing as zeros.

    ``untagged`` is deliberately not emitted here: the frame already derives
    an ``unattributed_live`` row from ``torch_live`` minus the tagged total,
    and emitting both would double count.
    """
    if by_tag is None:
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
                basis=Basis.LOGICAL,
                owner="in use by graphs" if label == CUDA_GRAPHS else startup.step_name(label),
                placement=(
                    Placement.TORCH_GRAPH_POOL if label == CUDA_GRAPHS else Placement.TORCH_DEFAULT
                ),
            )
        )
    return entries


def _check_failed(name: str, exc: Exception) -> CheckResult:
    return CheckResult(
        name=name,
        passed=True,
        detail=f"could not run ({type(exc).__name__}: {exc})",
        informational=True,
    )


def _run_checks(
    by_tag: dict[str, int],
    torch_live: int,
    pool_rows: dict[str, list[dict]] | None = None,
) -> tuple[CheckResult, ...]:
    """Conservation checks over this reading; a check that cannot run is listed, not raised.

    Each compares figures that come from independent sources, so a failure
    means the sources disagree -- which is worth more than any single number
    in the report. Check C runs on ``pool_rows``, the pool statistics each
    KV cache provider read for this same report, so the check and the KV
    row describe one reading even when two reports run at once.
    """
    results: list[CheckResult] = []
    try:
        results.extend(check_tag_coverage(by_tag, torch_live))
    except (KeyError, TypeError, ValueError) as exc:
        results.append(_check_failed("B:tag_coverage", exc))

    for name, rows in (pool_rows or {}).items():
        try:
            results.extend(check_kv_pool_conservation(rows, name=name))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(_check_failed(f"C:{name}", exc))
    return tuple(results)


def capture(
    *,
    tag: str = "",
    rank: int | None = None,
    device_index: int | None = None,
    headline: str = "",
    record_baseline: bool = False,
    suggestion: str = "",
    advise: bool = True,
    startup_steps: bool = False,
    force: bool = False,
) -> MonitorCapture | None:
    """Collect a snapshot, or ``None`` when the monitor is disabled.

    ``force`` collects even then, for a report somebody asked for. Without
    the monitor on during startup it has no startup steps or tags, but every
    component, pool and the per-process total are still read.

    Every available provider is read and the startup tags and claimed
    modules are resolved. ``rank`` defaults to :func:`current_rank` and labels the report, so
    interleaved logs from several ranks can be told apart.
    """
    if rank is None:
        rank = settings.current_rank()
    if not force and not settings.full_mode(rank):
        return None
    if not torch.cuda.is_available():
        return None

    if device_index is None:
        device_index = torch.cuda.current_device()

    # One allocator snapshot serves the torch reading, the pool split and the
    # tags: a snapshot costs milliseconds, and two would disagree.
    segments = _segments(device_index)
    torch_reading = read_torch(device_index, segments)
    # One tag reading feeds the entries, check B and the stack breakdown;
    # reading twice would let the allocator move between them.
    by_tag, sites = live_attribution(get_frozen_map(), segments)
    pool_rows: dict[str, list[dict]] = {}
    entries = collect_all(pool_rows=pool_rows) + _tag_entries(by_tag)
    checks = _run_checks(by_tag, torch_reading.live, pool_rows)

    frame = build_frame(read_device(device_index), torch_reading, entries)
    baseline = None if record_baseline else startup.baseline()
    if record_baseline:
        startup.record_baseline(group_totals(frame).sizes)

    return MonitorCapture(
        frame=frame,
        capability=probe_capability(
            nvml_available=frame.device.process_missing_reason != NVML_UNAVAILABLE
        ),
        tag=tag,
        pid=os.getpid(),
        rank=rank,
        device_index=device_index,
        checks=checks,
        headline=headline,
        baseline=baseline,
        suggestion=suggestion,
        advise=advise,
        startup_steps=startup_steps,
        unattributed_sites=sites or None,
    )


def render_capture(snapshot: MonitorCapture) -> str:
    """Render a capture produced by :func:`capture`."""
    return render(
        snapshot.frame,
        snapshot.capability,
        tag=snapshot.tag,
        pid=snapshot.pid,
        rank=snapshot.rank,
        device_index=snapshot.device_index,
        checks=list(snapshot.checks),
        headline=snapshot.headline,
        baseline=snapshot.baseline,
        advice=_advice() if snapshot.advise else {},
        suggestion=snapshot.suggestion,
        startup_steps=snapshot.startup_steps,
        unattributed_sites=snapshot.unattributed_sites,
    )


#: Summary group -> the executor creation step whose tuning advice applies.
#: The advice text itself is the executor's (``_ExecutorMemoryMonitor``),
#: handed over at startup, so the two never drift apart.
_ADVICE_STEP: dict[Group, str] = {
    Group.KV_CACHE: "kv_cache",
    Group.WEIGHTS: "model",
    Group.WORKSPACE: "executor_extra",
    Group.TORCH_UNATTRIBUTED: "executor_extra",
}


def _advice() -> dict[Group, str]:
    result = {}
    for group, step in _ADVICE_STEP.items():
        text = startup.advice_for_stage(step)
        if text:
            result[group] = text
    return result

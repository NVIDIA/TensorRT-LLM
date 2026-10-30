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
"""What executor startup already measures, handed to the monitor.

Startup keeps its own accounts, and the report reuses them rather than
measuring again:

* ``_ExecutorMemoryMonitor`` in ``py_executor_creator`` records free device
  memory around each creation step and knows which step failed. It also
  owns the tuning advice for each step.
* KV cache sizing profiles a forward pass and turns the result into a plan:
  a peak for everything outside the KV cache, with the KV cache sized into
  what is left.

Those producers push into this module; nothing here reaches back into them.

The executor's own step figures are device-wide (``total - free``), so on a
shared GPU another process freeing memory mid-step makes the step look
cheaper -- two runs of the same model reported the weight-loading step as
1.08 GiB and 0.06 GiB. Each step is therefore also measured per process
through NVML, and that figure is the one shown whenever it is available.
The KV sizing plan is device-wide too, so it is compared only with
device-wide usage.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

__all__ = [
    "MemoryPlan",
    "Reading",
    "RegionSample",
    "region",
    "residual_breakdown",
    "begin_stage",
    "StageSample",
    "advice_for_stage",
    "clear",
    "failed_stage",
    "memory_plan",
    "record_failed_stage",
    "record_memory_plan",
    "record_stage",
    "set_stage_advice",
    "stages",
    "step_name",
]

_GIB = float(1 << 30)


@dataclass(frozen=True)
class StageSample:
    stage: str
    name: str
    free_before: int
    free_after: int | None  # None for the step that failed
    #: This process's growth over the step, from NVML; ``None`` if unknown.
    process_taken: int | None = None
    #: Growth that neither PyTorch nor any component accounts for -- the
    #: step's share of the "CUDA and library overhead" row.
    residual_taken: int | None = None

    @property
    def device_taken(self) -> int | None:
        return None if self.free_after is None else self.free_before - self.free_after


@dataclass(frozen=True)
class MemoryPlan:
    """KV cache sizing's plan, device-wide."""

    #: Peak for everything except the KV cache, reserves included.
    non_kv_peak: int
    #: PyTorch allocations made by the profiled forward pass, if profiled.
    activation: int | None


@dataclass(frozen=True)
class Reading:
    """This process's memory, split the way the report splits it."""

    process: int  # NVML, this process
    torch: int  # memory_reserved()
    known: int  # what components account for outside PyTorch

    @property
    def residual(self) -> int:
        return self.process - self.torch - self.known


@dataclass(frozen=True)
class RegionSample:
    """Unexplained growth across a named piece of code, e.g. a library's init."""

    name: str
    stage: str | None
    residual_taken: int
    #: Recorded before the first step's reading, which therefore includes it.
    before_first: bool = False


_stages: list[StageSample] = []
_regions: list[RegionSample] = []
#: Residual taken by regions nested in each open region, innermost last, so
#: a region inside another is not counted twice.
_open_regions: list[int] = []
_first: Reading | None = None
_active_stage: str | None = None
_failed: StageSample | None = None
_names: dict[str, str] = {}
_advice: dict[str, str] = {}
_plan: MemoryPlan | None = None


def _on() -> bool:
    from . import enabled

    return enabled()


def _process_used() -> int | None:
    import torch

    from . import _nvml_process_used

    try:
        return _nvml_process_used(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def _known_outside_torch() -> int:
    from .providers import collect_all
    from .schema import Placement

    return sum(
        e.size_bytes
        for e in collect_all()
        if e.placement is Placement.NON_TORCH and e.counts_toward_total and e.size_bytes
    )


def _reading() -> Reading | None:
    import torch

    process = _process_used()
    if process is None:
        return None
    try:
        return Reading(process, torch.cuda.memory_reserved(), _known_outside_torch())
    except Exception:  # noqa: BLE001 - never break startup over a reading
        return None


def begin_stage(stage: str | None = None) -> Reading | None:
    """Read this process's memory as a step starts; pass it to the end call.

    The first reading is kept as well: whatever is unexplained before the
    first step -- the CUDA context, libraries initialised on import -- is
    the earliest share of the overhead row.
    """
    global _first, _active_stage
    if not _on():
        return None
    reading = _reading()
    if _first is None:
        _first = reading
    _active_stage = stage
    return reading


@contextlib.contextmanager
def region(name: str):
    """Attribute unexplained memory that appears inside this block to ``name``.

    For code that allocates outside PyTorch and keeps no account of its own,
    such as a library creating its communicators. A no-op when the monitor
    is off.
    """
    if not _on():
        yield
        return
    before = _reading()
    _open_regions.append(0)
    try:
        yield
    finally:
        nested = _open_regions.pop()
        after = _reading() if before is not None else None
        if after is not None:
            taken = after.residual - before.residual
            _regions.append(
                RegionSample(name, _active_stage, taken - nested, before_first=_first is None)
            )
            if _open_regions:
                _open_regions[-1] += taken


def set_stage_advice(names: dict[str, str], advice: dict[str, str]) -> None:
    """Adopt the executor's step names and tuning advice, keyed by step value."""
    _names.update(names)
    _advice.update(advice)


def record_stage(
    stage: str, free_before: int, free_after: int, process_before: Reading | None = None
) -> None:
    global _active_stage
    if not _on():
        return
    after = _reading() if process_before is not None else None
    process_taken = residual_taken = None
    if after is not None:
        process_taken = after.process - process_before.process
        residual_taken = after.residual - process_before.residual
    _stages.append(
        StageSample(
            stage, _names.get(stage, stage), free_before, free_after, process_taken, residual_taken
        )
    )
    _active_stage = None


def record_failed_stage(
    stage: str, free_before: int, process_before: Reading | None = None
) -> None:
    global _failed, _active_stage
    if _on():
        _failed = StageSample(stage, _names.get(stage, stage), free_before, None)
        _active_stage = None


def record_memory_plan(non_kv_peak: int, activation: int | None) -> None:
    global _plan
    if _on():
        _plan = MemoryPlan(non_kv_peak=int(non_kv_peak), activation=activation or None)


def stages() -> tuple[StageSample, ...]:
    return tuple(_stages)


def failed_stage() -> StageSample | None:
    return _failed


def memory_plan() -> MemoryPlan | None:
    return _plan


def step_name(stage: str) -> str:
    """The executor's own name for a creation step, or ``stage`` itself."""
    return _names.get(stage, stage)


def advice_for_stage(stage: str) -> str | None:
    return _advice.get(stage)


def clear() -> None:
    """Forget everything recorded. For tests."""
    global _failed, _plan, _first, _active_stage
    _stages.clear()
    _regions.clear()
    _open_regions.clear()
    _failed = None
    _plan = None
    _first = None
    _active_stage = None
    _names.clear()
    _advice.clear()


def render_stage_lines() -> list[str]:
    """Memory taken by each startup step, and the step that failed, if any."""
    if not _stages and _failed is None:
        return []
    per_process = bool(_stages) and all(s.process_taken is not None for s in _stages)
    if per_process:
        out = ["  Memory taken by each startup step (this process):"]
    else:
        out = [
            "  Memory taken by each startup step (device-wide; another process "
            "freeing or taking memory distorts these):"
        ]
    for sample in _stages:
        taken = sample.process_taken if per_process else sample.device_taken
        # Value first: step names vary too much in length to align after them.
        out.append(f"    {taken / _GIB:+8.2f} GiB  {sample.name}")
    if _failed is not None:
        out.append(
            f"      FAILED      {_failed.name} "
            f"(free device memory before it: {_failed.free_before / _GIB:.2f} GiB)"
        )
    return out


#: Differences below this are sampling noise between two device-wide reads,
#: not a workload outgrowing its plan.
_PLAN_TOLERANCE = 256 << 20


def render_plan_lines(
    device_used: int, kv_capacity: int | None, reusable: int | None = None
) -> list[str]:
    """Compare device usage now against the plan KV cache sizing made.

    ``reusable`` is memory the PyTorch allocator holds but is not using.
    Sizing measured after emptying that cache, and the workload can reuse it,
    so it is not counted against the plan.
    """
    if _plan is None:
        return []
    out = []
    planned = _plan.non_kv_peak + (kv_capacity or 0)
    in_use = device_used - (reusable or 0)
    over = in_use - planned
    kv = f" + KV cache {kv_capacity / _GIB:.2f} GiB" if kv_capacity else ""
    cache = f", excluding {reusable / _GIB:.2f} GiB of free PyTorch cache" if reusable else ""
    out.append(
        f"  Startup plan: peak outside the KV cache {_plan.non_kv_peak / _GIB:.2f} GiB{kv} "
        f"= {planned / _GIB:.2f} GiB; device now {in_use / _GIB:.2f} GiB{cache} ({over / _GIB:+.2f})."
    )
    if _plan.activation is not None:
        out.append(
            f"    of which the profiled forward pass allocated {_plan.activation / _GIB:.2f} GiB "
            "in PyTorch."
        )
    if over > _PLAN_TOLERANCE:
        out.append(
            f"  Usage exceeds the startup plan by {over / _GIB:.2f} GiB: the workload needs more "
            "than profiling measured, or another process grew."
        )
    return out


def residual_breakdown(current_residual: int) -> list[tuple[str, int]]:
    """Split today's unexplained memory by when, and where, it appeared.

    The parts sum to ``current_residual`` exactly: whatever the recorded
    steps and regions do not cover is reported as having appeared after
    startup (or, if negative, as released since). Each part is measured when
    it happened, so a part freed later shows up as a negative remainder
    rather than silently overstating what is held now.
    """
    if _first is None:
        return []
    # A region that ran before the first step is inside that first reading;
    # it is listed under its own name and taken out of the first part.
    early = sum(r.residual_taken for r in _regions if r.before_first)
    parts = [("before the first startup step", _first.residual - early)]
    for region_sample in _regions:
        parts.append((region_sample.name, region_sample.residual_taken))
    for sample in _stages:
        if sample.residual_taken is None:
            continue
        inside = sum(r.residual_taken for r in _regions if r.stage == sample.stage)
        parts.append(
            (f"{sample.name} (other)" if inside else sample.name, sample.residual_taken - inside)
        )
    remainder = current_residual - sum(size for _, size in parts)
    parts.append(("since startup" if remainder >= 0 else "released since startup", remainder))
    # A region entered more than once -- one workspace per mapping, say --
    # is one part; listing it twice would let a caller keying by name drop
    # all but one.
    merged: dict[str, int] = {}
    for name, size in parts:
        merged[name] = merged.get(name, 0) + size
    return list(merged.items())

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
"""GPU memory monitor: where did the device memory go?

Enabled with ``TLLM_MEM_MONITOR=1``; off by default and zero-cost when off.

Usage::

    from tensorrt_llm._torch.memory.monitor import capture, render_capture

    snapshot = capture(tag="after_weights")
    if snapshot is not None:
        print(render_capture(snapshot))

The monitor only reads. It never calls ``empty_cache()``, never synchronises,
and never resets peak stats -- ``_util.configure_kv_cache_capacity`` already
resets the global peak, and a second resetter would silently corrupt its
measurements.
"""

import os
from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger

from .capability import Capability, probe_capability
from .checks import CheckResult
from .frame import DeviceReading, Frame, TorchReading, build_frame
from .report import render, summarize_by_category
from .schema import Basis, Category, Entry, Placement, Source

__all__ = [
    "Basis",
    "Capability",
    "Category",
    "Entry",
    "MonitorCapture",
    "Placement",
    "Source",
    "capture",
    "clear_baseline",
    "current_rank",
    "enabled",
    "render_capture",
    "summarize_by_category",
]

ENV_VAR = "TLLM_MEM_MONITOR"
RANK_ENV_VAR = "TLLM_MEM_MONITOR_RANKS"


@dataclass(frozen=True)
class MonitorCapture:
    """One resolved snapshot plus the context needed to read it."""

    frame: Frame
    capability: Capability
    tag: str
    trigger: str
    pid: int
    rank: int | None
    device_index: int
    checks: "tuple[CheckResult, ...]" = ()
    #: One line saying what happened, shown above the summary (e.g. the
    #: failed allocation on OOM).
    headline: str = ""
    #: Group totals at the post-warmup baseline, when one was recorded.
    baseline: "dict | None" = None
    #: Advice that overrides the summary's own, when the cause is known.
    suggestion: str = ""
    #: Whether to list memory taken by each startup step.
    startup_steps: bool = False
    #: Untagged PyTorch memory by allocating code; empty unless stack
    #: recording (``TLLM_MEM_MONITOR_STACKS=1``) is on.
    unattributed_sites: "dict | None" = None


def current_rank() -> int | None:
    """This process's rank, resolved the way the executor resolves it.

    Uses the same ``mpi_rank()`` that sets ``Mapping.rank``, which also
    covers the Ray and torch.distributed launchers. ``None`` only when that
    cannot answer at all.
    """
    try:
        from tensorrt_llm._utils import mpi_rank

        return int(mpi_rank())
    except Exception:  # noqa: BLE001 - launch environments vary widely
        return None


def enabled(rank: int | None = None) -> bool:
    """Whether the monitor should run in this process.

    ``TLLM_MEM_MONITOR_RANKS`` optionally restricts collection to a
    comma-separated rank list, so a multi-rank job can pay the cost on one
    rank only. ``rank`` defaults to :func:`current_rank`; when no rank can be
    determined the filter is not applied, so the report is produced rather
    than silently skipped.
    """
    if os.environ.get(ENV_VAR, "") != "1":
        return False
    ranks = os.environ.get(RANK_ENV_VAR, "")
    if not ranks:
        return True
    if rank is None:
        rank = current_rank()
    if rank is None:
        return True
    try:
        wanted = {int(part) for part in ranks.split(",") if part.strip()}
    except ValueError:
        return True
    return rank in wanted


def _read_device(device_index: int) -> DeviceReading:
    free, total = torch.cuda.mem_get_info(device_index)
    used, reason = _nvml_process_lookup(device_index)
    return DeviceReading(
        total=total, free=free, process_used=used, process_missing_reason=reason or ""
    )


def _nvml_process_used(device_index: int) -> int | None:
    """This process's device memory according to NVML, or ``None``."""
    return _nvml_process_lookup(device_index)[0]


#: NVML answered, but not for this process.
_NOT_LISTED = (
    "this process is not in NVML's process list -- in a container without the "
    "host PID namespace NVML reports host PIDs"
)


def _nvml_process_lookup(device_index: int) -> tuple[int | None, str]:
    """This process's device memory according to NVML, and why not if not.

    ``None`` when NVML cannot answer. Falling back to ``total - free`` would
    silently attribute colocated processes' memory to us; TensorRT-LLM
    supports several colocated configurations, so that is a real hazard
    rather than a theoretical one. The reason says which of the different
    failures it was, since they call for different fixes.
    """
    unavailable = (None, "NVML unavailable")
    try:
        from tensorrt_llm import profiler
    except ImportError:
        return unavailable
    pynvml_context = getattr(profiler, "pynvml_context", None)
    if pynvml_context is None:
        return unavailable
    try:
        import pynvml
    except ImportError:
        return unavailable
    try:
        # By UUID, not by index: torch numbers only the devices
        # CUDA_VISIBLE_DEVICES exposes, NVML numbers every device on the
        # host, and the two disagree whenever the variable remaps them.
        # Measured with three devices visible and CUDA_VISIBLE_DEVICES=2:
        # torch device 0 is NVML device 2, and index 0 is another GPU.
        uuid = f"GPU-{torch.cuda.get_device_properties(device_index).uuid}"
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001 - NVML raises many unrelated types
        return unavailable
    return None, _NOT_LISTED


def _read_torch(device_index: int, segments: list | None = None) -> TorchReading:
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
    from .pools import pool_usage

    pools = tuple(pool_usage(segments))
    return TorchReading(
        reserved=sum(p.reserved for p in pools),
        live=sum(p.allocated for p in pools),
        pools=pools,
    )


def _segments(device_index: int) -> list | None:
    """This device's allocator segments, or ``None`` if they cannot be read."""
    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except Exception:  # noqa: BLE001
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
_TAG_CATEGORIES = {
    "WEIGHTS": Category.WEIGHTS,
    "CUDA_GRAPHS": Category.CUDA_GRAPHS,
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
    from .startup import step_name
    from .tagging import UNTAGGED, get_frozen_map, live_bytes_by_tag

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
                owner="in use by graphs" if label == "CUDA_GRAPHS" else step_name(label),
                placement=(
                    Placement.TORCH_GRAPH_POOL
                    if label == "CUDA_GRAPHS"
                    else Placement.TORCH_DEFAULT
                ),
            )
        )
    return entries


def _run_checks(by_tag: dict[str, int], torch_live: int) -> tuple:
    """Conservation checks over this reading, never raising.

    Each compares figures that come from independent sources, so a failure
    means the sources disagree -- which is worth more than any single number
    in the report.
    """
    from .checks import check_kv_pool_conservation, check_tag_coverage
    from .providers import default_providers
    from .providers.kv_cache import KvCacheProvider, kv_pool_group_stats

    results = []
    try:
        results.extend(check_tag_coverage(by_tag, torch_live))
    except Exception as exc:  # noqa: BLE001 - a check must not break the report
        logger.debug(f"memory monitor: tag check failed ({type(exc).__name__}: {exc})")

    for provider in default_providers():
        if not isinstance(provider, KvCacheProvider):
            continue
        manager = provider._manager
        if manager is None or not hasattr(manager, "_get_storage_statistics"):
            continue
        try:
            for result in check_kv_pool_conservation(kv_pool_group_stats(manager)):
                results.append(
                    type(result)(
                        name=result.name.replace("C:kv_pool_group", f"C:{provider.name}"),
                        passed=result.passed,
                        detail=result.detail,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"memory monitor: KV check failed ({type(exc).__name__}: {exc})")
    return tuple(results)


#: Group totals recorded once startup (including warmup) completes. Later
#: reports show each group's change against it.
_baseline: dict | None = None


def capture(
    *,
    tag: str = "",
    trigger: str = "",
    entries: list[Entry] | None = None,
    rank: int | None = None,
    device_index: int | None = None,
    headline: str = "",
    record_baseline: bool = False,
    suggestion: str = "",
    startup_steps: bool = False,
    force: bool = False,
) -> MonitorCapture | None:
    """Collect a snapshot, or ``None`` when the monitor is disabled.

    ``force`` collects even then, for a report somebody asked for. Without
    the monitor on during startup it has no startup steps or tags, but every
    component, pool and the per-process total are still read.

    ``entries`` overrides what is collected; by default every available
    provider is read and the startup tags and claimed modules are resolved.
    ``rank`` defaults to :func:`current_rank` and labels the report, so
    interleaved logs from several ranks can be told apart.
    """
    if rank is None:
        rank = current_rank()
    if not force and not enabled(rank):
        return None
    if not torch.cuda.is_available():
        return None

    if device_index is None:
        device_index = torch.cuda.current_device()

    # One allocator snapshot serves the torch reading, the pool split and the
    # tags: a snapshot costs milliseconds, and two would disagree.
    segments = _segments(device_index)
    torch_reading = _read_torch(device_index, segments)
    checks: tuple = ()
    sites: dict = {}
    if entries is None:
        from .providers import collect_all
        from .tagging import get_frozen_map, live_attribution

        # One tag reading feeds the entries, check B and the stack breakdown;
        # reading twice would let the allocator move between them.
        by_tag, sites = live_attribution(get_frozen_map(), segments)
        entries = collect_all() + _tag_entries(by_tag)
        checks = _run_checks(by_tag, torch_reading.live)

    global _baseline
    frame = build_frame(_read_device(device_index), torch_reading, entries)
    baseline = None if record_baseline else _baseline
    if record_baseline:
        from .summary import group_totals

        _baseline = dict(group_totals(frame).sizes)

    return MonitorCapture(
        frame=frame,
        capability=probe_capability(),
        tag=tag,
        trigger=trigger,
        pid=os.getpid(),
        rank=rank,
        device_index=device_index,
        checks=checks,
        headline=headline,
        baseline=baseline,
        suggestion=suggestion,
        startup_steps=startup_steps,
        unattributed_sites=sites or None,
    )


def clear_baseline() -> None:
    """Forget the post-warmup baseline. For tests."""
    global _baseline
    _baseline = None


def render_capture(snapshot: MonitorCapture) -> str:
    """Render a capture produced by :func:`capture`."""
    return render(
        snapshot.frame,
        snapshot.capability,
        tag=snapshot.tag,
        trigger=snapshot.trigger,
        pid=snapshot.pid,
        rank=snapshot.rank,
        device_index=snapshot.device_index,
        checks=list(snapshot.checks),
        headline=snapshot.headline,
        baseline=snapshot.baseline,
        advice=_advice(),
        suggestion=snapshot.suggestion,
        startup_steps=snapshot.startup_steps,
        unattributed_sites=snapshot.unattributed_sites,
    )


#: Summary group -> the executor creation step whose tuning advice applies.
#: The advice text itself is the executor's (``_ExecutorMemoryMonitor``),
#: handed over at startup, so the two never drift apart.
_ADVICE_STEP = {
    "KV_CACHE": "kv_cache",
    "WEIGHTS": "model",
    "WORKSPACE": "executor_extra",
    "TORCH_UNATTRIBUTED": "executor_extra",
}


def _advice() -> dict:
    from .startup import advice_for_stage
    from .summary import Group

    result = {}
    for group_name, step in _ADVICE_STEP.items():
        text = advice_for_stage(step)
        if text:
            result[Group[group_name]] = text
    return result

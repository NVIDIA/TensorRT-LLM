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

``TLLM_MEM_MONITOR`` selects a mode (see :func:`mode`). By default only an
out-of-memory error produces a report, and nothing runs until one happens;
``TLLM_MEM_MONITOR=1`` also records startup and reports when it completes.

Usage::

    from tensorrt_llm._torch.memory.monitor import capture, render_capture

    snapshot = capture(tag="after_weights")
    if snapshot is not None:
        print(render_capture(snapshot))

Reading memory never changes it: the monitor never calls ``empty_cache()``,
never synchronises, and never resets peak stats --
``_util.configure_kv_cache_capacity`` resets the global peak, and a second
resetter would corrupt its measurements. In full mode it does configure
PyTorch's allocation-history recording during startup (see ``tagging``).
"""

import math
import os
import re
from dataclasses import dataclass

import torch
from strenum import StrEnum

from tensorrt_llm.logger import logger

from .capability import NVML_UNAVAILABLE, Capability, nvml_process_usage, probe_capability
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
    "configure",
    "current_rank",
    "full_mode",
    "full_mode_requested",
    "Mode",
    "mode",
    "oom_reports_enabled",
    "render_capture",
    "setting",
    "stacks_requested",
    "summarize_by_category",
]

ENV_VAR = "TLLM_MEM_MONITOR"
RANK_ENV_VAR = "TLLM_MEM_MONITOR_RANKS"
STACKS_ENV_VAR = "TLLM_MEM_MONITOR_STACKS"


class Mode(StrEnum):
    """``TLLM_MEM_MONITOR`` modes; see :func:`mode`."""

    OFF = "off"
    OOM = "oom"
    FULL = "full"


OFF, OOM, FULL = Mode.OFF, Mode.OOM, Mode.FULL
_MODE_VALUES = {
    "0": OFF,
    "off": OFF,
    "false": OFF,
    "no": OFF,
    "oom": OOM,
    "1": FULL,
    "full": FULL,
    "true": FULL,
    "on": FULL,
    "yes": FULL,
}
#: Unset: report out-of-memory errors, which costs nothing until one happens.
DEFAULT_MODE = OOM
_ON_VALUES = frozenset({"1", "on", "true", "yes"})
_OFF_VALUES = frozenset({"", "0", "off", "false", "no"})
_RANK_TOKEN = re.compile(r"(\d+)(?:-(\d+))?")
#: ``(variable, value)`` pairs already warned about.
_warned: set[tuple[str, str]] = set()
#: Settings from ``memory_monitor_config``, by the variable they stand for.
_configured: dict[str, str] = {}


def configure(config) -> None:
    """Take settings from ``TorchLlmArgs.memory_monitor_config``.

    ``None`` clears them. Each setting is read as the value of the
    environment variable it corresponds to, so both pass the same checks;
    a variable that is set in the environment overrides the configuration.
    """
    from .triggers import DIR_VAR, OOM_INTERVAL_VAR, PERIODIC_INTERVAL_VAR

    _configured.clear()
    if config is None:
        return
    _configured[ENV_VAR] = config.mode
    _configured[STACKS_ENV_VAR] = "1" if config.record_stacks else "0"
    if config.ranks is not None:
        _configured[RANK_ENV_VAR] = ",".join(str(r) for r in config.ranks)
    if config.periodic_report_interval is not None:
        _configured[PERIODIC_INTERVAL_VAR] = repr(float(config.periodic_report_interval))
    _configured[OOM_INTERVAL_VAR] = repr(float(config.oom_report_interval))
    if config.report_dir is not None:
        _configured[DIR_VAR] = config.report_dir


def setting(var: str) -> str | None:
    """A monitor setting: the environment variable if set, else the configuration."""
    value = os.environ.get(var)
    if value is not None:
        return value
    return _configured.get(var)


def warn_once(var: str, raw: str, problem: str) -> None:
    """Log one warning per invalid ``var=raw`` setting for the process."""
    if (var, raw) in _warned:
        return
    _warned.add((var, raw))
    logger.warning(f"memory monitor: {var}={raw!r} {problem}")


def seconds_setting(var: str, default: float) -> float:
    """A non-negative number of seconds from ``var``; ``default`` if unset or invalid."""
    raw = (setting(var) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        warn_once(var, raw, f"is not a non-negative number of seconds; using {default:g}")
        return default
    return value


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
    except (ImportError, RuntimeError, ValueError):
        # mpi4py errors derive from RuntimeError.
        return None


def mode() -> str:
    """The monitor's mode from ``TLLM_MEM_MONITOR``.

    * ``0`` (also ``off``, ``false``, ``no``) -- off: no reports are
      produced on their own.
    * ``oom`` (the default, also when unset) -- a report when memory runs out,
      and nothing else. No work is done until an allocation fails, so the
      default costs nothing while serving; the report has every row but no
      startup-step breakdown and no comparison with warmup, which need
      recording during startup.
    * ``1`` (also ``full``, ``on``, ``true``, ``yes``) -- full: also records
      startup, reports when it completes, and enables the periodic report,
      stack mode, every-rank gathering under MPI and ``trtllm-serve``'s
      ``/memory_report``.

    An unrecognised value falls back to the default, with one warning.
    """
    raw = (setting(ENV_VAR) or "").strip().lower()
    if not raw:
        return DEFAULT_MODE
    if raw in _MODE_VALUES:
        return _MODE_VALUES[raw]
    warn_once(ENV_VAR, raw, f"is not one of 0, oom, 1 (or off, full); using {DEFAULT_MODE}")
    return DEFAULT_MODE


def full_mode_requested() -> bool:
    """Whether full mode is set, regardless of the rank filter.

    For setup every rank must agree on, such as a collective communicator
    duplication, and for the ``trtllm-serve`` frontend, which is rank 0 and
    serves every rank's report; :func:`full_mode` can differ between ranks.
    """
    return mode() == FULL


def full_mode(rank: int | None = None) -> bool:
    """Whether the full monitor runs in this process (``TLLM_MEM_MONITOR=1``).

    ``TLLM_MEM_MONITOR_RANKS`` optionally restricts it to some ranks, so a
    multi-rank job can pay the cost on one rank only. ``rank`` defaults to
    :func:`current_rank`; when no rank can be determined the filter is not
    applied, so the report is produced rather than silently skipped.
    """
    return mode() == FULL and _rank_selected(rank)


def oom_reports_enabled(rank: int | None = None) -> bool:
    """Whether out-of-memory errors are reported: the default and full modes."""
    return mode() != OFF and _rank_selected(rank)


def stacks_requested() -> bool:
    """Whether ``TLLM_MEM_MONITOR_STACKS`` asks for allocation stacks."""
    raw = (setting(STACKS_ENV_VAR) or "").strip().lower()
    if raw in _ON_VALUES:
        return True
    if raw not in _OFF_VALUES:
        warn_once(STACKS_ENV_VAR, raw, "is not 0 or 1; stacks are not recorded")
    return False


def selected_ranks() -> frozenset[int] | None:
    """Ranks ``TLLM_MEM_MONITOR_RANKS`` selects, or ``None`` for every rank.

    Accepts ranks and inclusive ranges separated by commas, e.g. ``0,4`` or
    ``0-3,8``. A value that selects nothing or cannot be parsed is reported
    once and then ignored, so a typo costs the filter rather than every
    report.
    """
    raw = setting(RANK_ENV_VAR)
    if raw is None or not raw.strip():
        if raw is not None and raw != "":
            warn_once(RANK_ENV_VAR, raw, "selects no rank; reporting on every rank")
        return None
    wanted: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        match = _RANK_TOKEN.fullmatch(token)
        low = int(match.group(1)) if match else 0
        high = int(match.group(2)) if match and match.group(2) else low
        if match is None or high < low:
            warn_once(
                RANK_ENV_VAR,
                raw,
                "is not a comma-separated list of ranks or ranges such as 0,4 or 0-3; "
                "reporting on every rank",
            )
            return None
        wanted.update(range(low, high + 1))
    if not wanted:
        warn_once(RANK_ENV_VAR, raw, "selects no rank; reporting on every rank")
        return None
    return frozenset(wanted)


def _rank_selected(rank: int | None) -> bool:
    wanted = selected_ranks()
    if wanted is None:
        return True
    if rank is None:
        rank = current_rank()
    return rank is None or rank in wanted


def _read_device(device_index: int) -> DeviceReading:
    free, total = torch.cuda.mem_get_info(device_index)
    used, reason = nvml_process_usage(device_index)
    return DeviceReading(
        total=total, free=free, process_used=used, process_missing_reason=reason or ""
    )


def _nvml_process_used(device_index: int) -> int | None:
    """This process's device memory according to NVML, or ``None``."""
    return nvml_process_usage(device_index)[0]


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
    # A pool sleep mode has released is not resident; it is kept in
    # ``pools`` so the report can say what is asleep.
    resident = [p for p in pools if not p.released]
    return TorchReading(
        reserved=sum(p.reserved for p in resident),
        live=sum(p.allocated for p in resident),
        pools=pools,
    )


def _segments(device_index: int) -> list | None:
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


def _check_failed(name: str, exc: Exception) -> CheckResult:
    return CheckResult(
        name=name,
        passed=True,
        detail=f"could not run ({type(exc).__name__}: {exc})",
        informational=True,
    )


def _run_checks(by_tag: dict[str, int], torch_live: int) -> tuple:
    """Conservation checks over this reading; a check that cannot run is listed, not raised.

    Each compares figures that come from independent sources, so a failure
    means the sources disagree -- which is worth more than any single number
    in the report. Check C reads the pool statistics the KV cache provider
    collected for this same report, so the check and the KV row describe
    one reading.
    """
    from .checks import check_kv_pool_conservation, check_tag_coverage
    from .providers import default_providers
    from .providers.kv_cache import KvCacheProvider

    results = []
    try:
        results.extend(check_tag_coverage(by_tag, torch_live))
    except (KeyError, TypeError, ValueError) as exc:
        results.append(_check_failed("B:tag_coverage", exc))

    for provider in default_providers():
        if not isinstance(provider, KvCacheProvider):
            continue
        try:
            rows = provider.last_pool_group_rows
            if rows is None:
                continue
            results.extend(check_kv_pool_conservation(rows, name=provider.name))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            results.append(_check_failed(f"C:{provider.name}", exc))
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
    if not force and not full_mode(rank):
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
        capability=probe_capability(
            nvml_available=frame.device.process_missing_reason != NVML_UNAVAILABLE
        ),
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
    """Forget the post-warmup baseline, as each executor's startup does."""
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

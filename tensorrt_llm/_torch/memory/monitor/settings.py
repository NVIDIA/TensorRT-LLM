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
"""The monitor's settings: mode, rank filter, intervals and output directory.

Each setting comes from ``TorchLlmArgs.memory_monitor_config`` or from the
``TLLM_MEM_MONITOR*`` environment variable that corresponds to it; a
variable that is set overrides the configuration. Both are parsed by the
same code, and an invalid value is reported once and then replaced by the
default rather than raising.
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING

from strenum import StrEnum

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

__all__ = [
    "DIR_VAR",
    "ENV_VAR",
    "OOM_INTERVAL_VAR",
    "PERIODIC_INTERVAL_VAR",
    "RANK_ENV_VAR",
    "STACKS_ENV_VAR",
    "Mode",
    "configure",
    "current_rank",
    "full_mode",
    "full_mode_requested",
    "mode",
    "oom_reports_enabled",
    "rank_selected",
    "seconds_setting",
    "selected_ranks",
    "setting",
    "stacks_requested",
    "warn_once",
]

ENV_VAR = "TLLM_MEM_MONITOR"
RANK_ENV_VAR = "TLLM_MEM_MONITOR_RANKS"
STACKS_ENV_VAR = "TLLM_MEM_MONITOR_STACKS"
#: Report every this many seconds while serving, in full mode.
PERIODIC_INTERVAL_VAR = "TLLM_MEM_MONITOR_INTERVAL"
#: Minimum seconds between two out-of-memory reports.
OOM_INTERVAL_VAR = "TLLM_MEM_MONITOR_OOM_INTERVAL"
#: Directory to which every report is also written, one file per report.
DIR_VAR = "TLLM_MEM_MONITOR_DIR"


class Mode(StrEnum):
    """``TLLM_MEM_MONITOR`` modes; see :func:`mode`."""

    OFF = "off"
    OOM = "oom"
    FULL = "full"


OFF, OOM, FULL = Mode.OFF, Mode.OOM, Mode.FULL
_MODE_VALUES: dict[str, Mode] = {
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
DEFAULT_MODE: Mode = OOM
_ON_VALUES: frozenset[str] = frozenset({"1", "on", "true", "yes"})
_OFF_VALUES: frozenset[str] = frozenset({"", "0", "off", "false", "no"})
_RANK_TOKEN: re.Pattern[str] = re.compile(r"(\d+)(?:-(\d+))?")
#: ``(variable, value)`` pairs already warned about.
_warned: set[tuple[str, str]] = set()
#: Settings from ``memory_monitor_config``, by the variable they stand for.
_configured: dict[str, str] = {}


def configure(config: MemoryMonitorConfig | None) -> None:
    """Take settings from ``TorchLlmArgs.memory_monitor_config``.

    ``None`` clears them. Each setting is read as the value of the
    environment variable it corresponds to, so both pass the same checks;
    a variable that is set in the environment overrides the configuration.
    """
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


def current_rank() -> int | None:
    """This process's rank, resolved the way the executor resolves it.

    Uses the same ``mpi_rank()`` that sets ``Mapping.rank``, which also
    covers the Ray and torch.distributed launchers. ``None`` only when that
    cannot answer at all.
    """
    try:
        return int(mpi_rank())
    except (RuntimeError, ValueError):
        # mpi4py errors derive from RuntimeError.
        return None


def mode() -> Mode:
    """The monitor's mode from ``TLLM_MEM_MONITOR`` or the configuration.

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
    """Whether the full monitor runs in this process.

    ``TLLM_MEM_MONITOR_RANKS`` optionally restricts it to some ranks, so a
    multi-rank job can pay the cost on one rank only. ``rank`` defaults to
    :func:`current_rank`; when no rank can be determined the filter is not
    applied, so the report is produced rather than silently skipped.
    """
    return mode() == FULL and rank_selected(rank)


def oom_reports_enabled(rank: int | None = None) -> bool:
    """Whether out-of-memory errors are reported: the default and full modes."""
    return mode() != OFF and rank_selected(rank)


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


def rank_selected(rank: int | None) -> bool:
    """Whether ``rank`` (default: this process's) passes the rank filter."""
    wanted = selected_ranks()
    if wanted is None:
        return True
    if rank is None:
        rank = current_rank()
    return rank is None or rank in wanted

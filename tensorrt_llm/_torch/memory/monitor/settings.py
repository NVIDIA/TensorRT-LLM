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
variable set to a non-empty value overrides the configuration, with one
warning when the two disagree. An empty variable counts as unset. Both are
parsed by the same code, and an invalid value is reported once and then
replaced by the default rather than raising.

The settings are process-wide. The first executor's configuration stays in
force while it lives: a second executor created in the same process runs
with it, and a configuration that differs is reported once and not applied.
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
    "MAX_RANK",
    "MAX_SECONDS",
    "OOM_INTERVAL_VAR",
    "PERIODIC_INTERVAL_VAR",
    "RANK_ENV_VAR",
    "STACKS_ENV_VAR",
    "Mode",
    "check_rank_filter",
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
    """Monitor modes (``memory_monitor_config.mode``, ``TLLM_MEM_MONITOR``); see :func:`mode`."""

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
#: Longest interval a setting may give: a week, far below ``threading.TIMEOUT_MAX``.
MAX_SECONDS = 7 * 24 * 3600.0
#: Highest rank a filter may name. Bounds the set a range expands to.
MAX_RANK = (1 << 16) - 1
#: Settings from ``memory_monitor_config``, by the variable they stand for.
_configured: dict[str, str] = {}
#: The ``memory_monitor_config`` field each variable stands for.
_FIELDS: dict[str, str] = {
    ENV_VAR: "mode",
    RANK_ENV_VAR: "ranks",
    STACKS_ENV_VAR: "record_stacks",
    PERIODIC_INTERVAL_VAR: "periodic_report_interval",
    OOM_INTERVAL_VAR: "oom_report_interval",
    DIR_VAR: "report_dir",
}
#: The last rank filter parsed: its raw value and what it selects.
_parsed_ranks: tuple[str, frozenset[int] | None] | None = None
#: A rank filter that selects none of the job's ranks, and is ignored.
_ranks_outside_job: str | None = None
#: ``(variable, value, configured)`` overrides already compared, so each
#: override of a configuration is reported once.
_checked_overrides: set[tuple[str, str, str]] = set()


def _settings_of(config: MemoryMonitorConfig | None) -> dict[str, str]:
    """``config`` as the environment variable values it stands for."""
    if config is None:
        return {}
    values = {
        ENV_VAR: str(config.mode),
        STACKS_ENV_VAR: "1" if config.record_stacks else "0",
        OOM_INTERVAL_VAR: repr(float(config.oom_report_interval)),
    }
    if config.ranks is not None:
        values[RANK_ENV_VAR] = ",".join(str(r) for r in config.ranks)
    if config.periodic_report_interval is not None:
        values[PERIODIC_INTERVAL_VAR] = repr(float(config.periodic_report_interval))
    if config.report_dir is not None:
        values[DIR_VAR] = config.report_dir
    return values


def _any_executor_alive() -> bool:
    """Whether an executor is alive in this process, counting one still being built."""
    # Imported here: the providers import the communicators, which import
    # this package while it is still initialising.
    from . import providers

    return providers.current_executor() is not None or providers.has_other_live_executors()


def configure(config: MemoryMonitorConfig | None) -> None:
    """Take settings from ``TorchLlmArgs.memory_monitor_config``.

    Call it before the executor's :func:`providers.begin_executor`. ``None``
    stands for the defaults. Applied only when no other executor is alive in
    this process: the settings are process-wide, and replacing them would
    change the mode, rank filter and report directory of an executor that is
    already serving. A configuration that differs from the one in force is
    then reported once and ignored.

    Each setting is read as the value of the environment variable it
    corresponds to, so both pass the same checks; a variable set to a
    non-empty value overrides the configuration.
    """
    global _ranks_outside_job
    values = _settings_of(config)
    if _any_executor_alive():
        if values != _configured:
            logger.warning_once(
                "memory monitor: another executor in this process is alive, so this "
                "executor's memory_monitor_config is not applied; the monitor keeps "
                f"the first executor's settings {_configured or 'defaults'!r}",
                key=("memory monitor: configure", tuple(sorted(values.items()))),
            )
        return
    _configured.clear()
    _configured.update(values)
    _ranks_outside_job = None
    _checked_overrides.clear()


def setting(var: str) -> str | None:
    """A monitor setting: the environment variable if non-empty, else the configuration.

    An empty variable counts as unset: launchers that export every variable
    (``docker -e VAR=``, ``export VAR=${VAR:-}``) must not silently replace
    the configuration with the defaults.
    """
    value = os.environ.get(var)
    configured = _configured.get(var)
    if not value:
        return configured
    if configured is not None and (var, value, configured) not in _checked_overrides:
        _checked_overrides.add((var, value, configured))
        if _normalized(var, value) != _normalized(var, configured):
            logger.warning(
                f"memory monitor: {var}={value!r} overrides memory_monitor_config."
                f"{_FIELDS.get(var, var)}={configured!r}"
            )
    return value


def _normalized(var: str, raw: str) -> object:
    """``raw`` in a form two spellings of one value compare equal in."""
    text = raw.strip().lower()
    if var == ENV_VAR:
        return _MODE_VALUES.get(text, text)
    if var == STACKS_ENV_VAR:
        return text in _ON_VALUES if text in _ON_VALUES | _OFF_VALUES else text
    if var in (PERIODIC_INTERVAL_VAR, OOM_INTERVAL_VAR):
        try:
            return float(text)
        except ValueError:
            return text
    if var == RANK_ENV_VAR:
        parsed = _parse_ranks(raw)
        return parsed if parsed else text
    return raw


def warn_once(var: str, raw: str, problem: str) -> None:
    """Log one warning per invalid ``var=raw`` setting and problem for the process."""
    logger.warning_once(
        f"memory monitor: {var}={raw!r} {problem}",
        key=("memory monitor", var, raw, problem),
    )


def seconds_setting(var: str, default: float) -> float:
    """Seconds from ``var``, up to :data:`MAX_SECONDS`; ``default`` if unset or invalid."""
    raw = (setting(var) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0 <= value <= MAX_SECONDS:
        warn_once(
            var,
            raw,
            f"is not a number of seconds from 0 to {MAX_SECONDS:g}; using {default:g}",
        )
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
    """The monitor's mode from ``memory_monitor_config.mode`` or ``TLLM_MEM_MONITOR``.

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

    ``memory_monitor_config.ranks`` (``TLLM_MEM_MONITOR_RANKS``) optionally
    restricts it to some ranks, so a
    multi-rank job can pay the cost on one rank only. ``rank`` defaults to
    :func:`current_rank`; when no rank can be determined the filter is not
    applied, so the report is produced rather than silently skipped.
    """
    return mode() == FULL and rank_selected(rank)


def oom_reports_enabled(rank: int | None = None) -> bool:
    """Whether out-of-memory errors are reported: the default and full modes."""
    return mode() != OFF and rank_selected(rank)


def stacks_requested() -> bool:
    """Whether ``record_stacks`` (``TLLM_MEM_MONITOR_STACKS``) asks for allocation stacks."""
    raw = (setting(STACKS_ENV_VAR) or "").strip().lower()
    if raw in _ON_VALUES:
        return True
    if raw not in _OFF_VALUES:
        warn_once(STACKS_ENV_VAR, raw, "is not 0 or 1; stacks are not recorded")
    return False


_UNPARSEABLE_RANKS = (
    f"is not a comma-separated list of ranks or ranges from 0 to {MAX_RANK}, such as "
    "0,4 or 0-3; reporting on every rank"
)


def _parse_ranks(raw: str) -> frozenset[int] | None:
    """The ranks ``raw`` names, or ``None`` if it cannot be parsed."""
    wanted: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        match = _RANK_TOKEN.fullmatch(token)
        if match is None:
            return None
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        if high < low or high > MAX_RANK:
            return None
        wanted.update(range(low, high + 1))
    return frozenset(wanted)


def selected_ranks() -> frozenset[int] | None:
    """Ranks ``TLLM_MEM_MONITOR_RANKS`` selects, or ``None`` for every rank.

    Accepts ranks and inclusive ranges from 0 to :data:`MAX_RANK` separated
    by commas, e.g. ``0,4`` or ``0-3,8``. A value that selects nothing,
    cannot be parsed, or selects none of the job's ranks
    (:func:`check_rank_filter`) is reported once and then ignored, so a typo
    costs the filter rather than every report. Each value is parsed once.
    """
    global _parsed_ranks
    raw = setting(RANK_ENV_VAR)
    if raw is None or raw == _ranks_outside_job:
        return None
    if _parsed_ranks is not None and _parsed_ranks[0] == raw:
        return _parsed_ranks[1]
    wanted = _parse_ranks(raw)
    if wanted is None:
        warn_once(RANK_ENV_VAR, raw, _UNPARSEABLE_RANKS)
    elif not wanted:
        warn_once(RANK_ENV_VAR, raw, "selects no rank; reporting on every rank")
        wanted = None
    _parsed_ranks = (raw, wanted)
    return wanted


def check_rank_filter(world_size: int) -> None:
    """Ignore a rank filter that selects none of the job's ``world_size`` ranks.

    Without this, a filter such as ``8`` in an eight-rank job would turn
    every report off on every rank without a word. Reported once.
    """
    global _ranks_outside_job
    wanted = selected_ranks()
    if wanted is None or any(rank < world_size for rank in wanted):
        return
    raw = setting(RANK_ENV_VAR) or ""
    warn_once(
        RANK_ENV_VAR,
        raw,
        f"selects none of this job's ranks 0-{world_size - 1}; reporting on every rank",
    )
    _ranks_outside_job = raw


def rank_selected(rank: int | None) -> bool:
    """Whether ``rank`` (default: this process's) passes the rank filter."""
    wanted = selected_ranks()
    if wanted is None:
        return True
    if rank is None:
        rank = current_rank()
    return rank is None or rank in wanted

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
variable set to a non-blank value overrides the configuration. A blank
variable counts as unset.

The environment is read once. ``TorchLlmArgs`` folds it into
``memory_monitor_config`` when the arguments are validated
(:func:`resolve_environment`), so every rank of a job gets the same settings
with the arguments, and the executor applies them with
``configure(..., read_environment=False)``. Code that configures the monitor
without ``TorchLlmArgs`` has the environment read when it calls
:func:`configure`; before any configuration, settings come from the
environment directly. An invalid value is reported once and ignored rather
than raising.

The settings are process-wide and belong to the one executor the monitor
follows (see :func:`providers.begin_executor`): while it lives, another
executor's configuration is not applied.
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
    "DEFAULT_MODE",
    "DEFAULT_OOM_INTERVAL_S",
    "DIR_VAR",
    "ENV_VAR",
    "MAX_RANK",
    "MAX_SECONDS",
    "MIN_PERIODIC_INTERVAL_S",
    "OOM_INTERVAL_VAR",
    "PERIODIC_INTERVAL_VAR",
    "RANK_ENV_VAR",
    "STACKS_ENV_VAR",
    "Mode",
    "check_rank_filter",
    "configure",
    "current_rank",
    "full_mode",
    "full_mode_in",
    "full_mode_requested",
    "mode",
    "oom_reports_enabled",
    "rank_selected",
    "reset",
    "resolve_environment",
    "seconds_setting",
    "selected_ranks",
    "setting",
    "setting_label",
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
#: Directory to which every report except on-demand ones is also written, one file per report.
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
#: Unset: report the out-of-memory errors the executor's error handling sees.
DEFAULT_MODE: Mode = OOM
_ON_VALUES: frozenset[str] = frozenset({"1", "on", "true", "yes"})
_OFF_VALUES: frozenset[str] = frozenset({"", "0", "off", "false", "no"})
_RANK_TOKEN: re.Pattern[str] = re.compile(r"(\d+)(?:-(\d+))?")
#: Longest interval a setting may give: a week, far below ``threading.TIMEOUT_MAX``.
MAX_SECONDS = 7 * 24 * 3600.0
#: Highest rank a filter may name. Bounds the set a range expands to.
MAX_RANK = (1 << 16) - 1
#: Shortest periodic interval; each report takes an allocator snapshot.
MIN_PERIODIC_INTERVAL_S = 1.0
#: Minimum seconds between two out-of-memory reports when none is configured.
DEFAULT_OOM_INTERVAL_S = 60.0
#: The ``memory_monitor_config`` field each variable stands for.
_FIELDS: dict[str, str] = {
    ENV_VAR: "mode",
    RANK_ENV_VAR: "ranks",
    STACKS_ENV_VAR: "record_stacks",
    PERIODIC_INTERVAL_VAR: "periodic_report_interval_s",
    OOM_INTERVAL_VAR: "oom_report_interval_s",
    DIR_VAR: "report_dir",
}
#: The configured settings, as the values of the variables they stand for;
#: ``None`` until :func:`configure`, while settings come from the environment.
_configured: dict[str, str] | None = None
#: Variables whose configured value came from the environment, for messages.
_from_environment: frozenset[str] = frozenset()
#: The last rank filter parsed: its raw value and what it selects.
_parsed_ranks: tuple[str, frozenset[int] | None] | None = None
#: A rank filter that selects none of the job's ranks, and is ignored.
_ranks_outside_job: str | None = None
#: Stands for an environment value that is invalid and ignored.
_INVALID = object()


def _settings_of(config: MemoryMonitorConfig | None) -> dict[str, str]:
    """``config`` as the environment variable values it stands for."""
    if config is None:
        return {}
    values = {
        ENV_VAR: str(config.mode),
        STACKS_ENV_VAR: "1" if config.record_stacks else "0",
        OOM_INTERVAL_VAR: repr(float(config.oom_report_interval_s)),
    }
    if config.ranks is not None:
        values[RANK_ENV_VAR] = ",".join(str(r) for r in config.ranks)
    if config.periodic_report_interval_s is not None:
        values[PERIODIC_INTERVAL_VAR] = repr(float(config.periodic_report_interval_s))
    if config.report_dir is not None:
        values[DIR_VAR] = config.report_dir
    return values


def _environment(var: str) -> str | None:
    """``var`` from the environment; ``None`` when unset or blank.

    A blank variable counts as unset: launchers that export every variable
    (``docker -e VAR=``, ``export VAR=${VAR:-}``) must not silently replace
    the configuration with the defaults.
    """
    value = os.environ.get(var)
    return value if value is not None and value.strip() else None


def _warn_environment(var: str, raw: str, problem: str) -> None:
    logger.warning_once(
        f"memory monitor: {var}={raw!r} {problem}", key=("memory monitor", var, raw, problem)
    )


def _environment_value(var: str, raw: str, world_size: int | None) -> object:
    """The ``memory_monitor_config`` value ``var=raw`` stands for, or :data:`_INVALID`."""
    text = raw.strip().lower()
    if var == ENV_VAR:
        if text in _MODE_VALUES:
            return str(_MODE_VALUES[text])
        _warn_environment(var, raw, "is not one of 0, oom, 1 (or off, full); ignored")
        return _INVALID
    if var == STACKS_ENV_VAR:
        if text in _ON_VALUES | _OFF_VALUES:
            return text in _ON_VALUES
        _warn_environment(var, raw, "is not 0 or 1; ignored")
        return _INVALID
    if var == RANK_ENV_VAR:
        wanted = _parse_ranks(raw)
        if wanted is None:
            _warn_environment(var, raw, _UNPARSEABLE_RANKS)
            return _INVALID
        if not wanted:
            _warn_environment(var, raw, "selects no rank; ignored")
            return _INVALID
        if world_size is None:
            return sorted(wanted)
        inside = sorted(rank for rank in wanted if rank < world_size)
        if not inside:
            _warn_environment(
                var, raw, f"selects none of this job's ranks 0-{world_size - 1}; ignored"
            )
            return _INVALID
        if len(inside) < len(wanted):
            _warn_environment(
                var,
                raw,
                f"names ranks beyond this job's ranks 0-{world_size - 1}; using {inside}",
            )
        return inside
    if var in (PERIODIC_INTERVAL_VAR, OOM_INTERVAL_VAR):
        try:
            seconds = float(text)
        except ValueError:
            seconds = math.nan
        if not 0 <= seconds <= MAX_SECONDS:
            _warn_environment(
                var, raw, f"is not a number of seconds from 0 to {MAX_SECONDS:g}; ignored"
            )
            return _INVALID
        if var == OOM_INTERVAL_VAR:
            return seconds
        if seconds == 0:
            return None  # no periodic report
        if seconds < MIN_PERIODIC_INTERVAL_S:
            _warn_environment(
                var,
                raw,
                f"is below the minimum of {MIN_PERIODIC_INTERVAL_S:g} s; using the minimum",
            )
            return MIN_PERIODIC_INTERVAL_S
        return seconds
    return raw


def resolve_environment(
    config: MemoryMonitorConfig | None, world_size: int | None = None
) -> MemoryMonitorConfig | None:
    """``config`` with the ``TLLM_MEM_MONITOR*`` variables set in this process applied.

    A non-blank variable replaces the field it stands for; one that is
    invalid, or a rank filter that selects none of the job's
    ``world_size`` ranks, is reported once and ignored. Replacing a field
    the configuration set explicitly to another value is reported once.
    Returns ``config`` itself when no variable is set, and ``None`` when
    neither is.
    """
    # Imported here: the arguments import this package's dependencies.
    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

    values: dict[str, object] = {}
    for var, field in _FIELDS.items():
        raw = _environment(var)
        if raw is None:
            continue
        value = _environment_value(var, raw, world_size)
        if value is _INVALID:
            continue
        values[field] = value
        if config is not None and field in config.model_fields_set:
            configured = getattr(config, field)
            if configured != value:
                logger.warning_once(
                    f"memory monitor: {var}={raw!r} overrides "
                    f"memory_monitor_config.{field}={configured!r}",
                    key=("memory monitor: override", var, raw, repr(configured)),
                )
    if not values:
        return config
    explicit = {} if config is None else config.model_dump(exclude_unset=True)
    resolved = MemoryMonitorConfig(**{**explicit, **values})
    resolved._from_environment = frozenset(values)
    return resolved


def full_mode_in(config: MemoryMonitorConfig | None) -> bool:
    """Whether ``config``, as carried by ``TorchLlmArgs``, selects full mode.

    The same answer on every rank of a job: the arguments carry the
    environment's overrides already. For setup every rank must agree on,
    such as a collective communicator duplication, and for the
    ``trtllm-serve`` frontend.
    """
    return config is not None and config.mode == FULL


def configure(
    config: MemoryMonitorConfig | None,
    *,
    read_environment: bool = True,
    world_size: int | None = None,
) -> None:
    """Take the monitor's settings from ``config`` (``None``: the defaults).

    Call it before the executor's :func:`providers.begin_executor`. Not
    applied while a monitored executor is alive in this process: the
    settings are process-wide and belong to that executor.

    ``read_environment=False`` is for a ``config`` that ``TorchLlmArgs``
    already resolved: the environment is not read again. Otherwise the
    ``TLLM_MEM_MONITOR*`` variables are applied now
    (:func:`resolve_environment`), and with ``world_size``, the job's rank
    count, a rank filter that selects none of the job's ranks is reported
    and ignored before startup is recorded. Either way the environment is
    not read after this call.
    """
    global _configured, _from_environment, _parsed_ranks, _ranks_outside_job
    # Imported here: the providers import the communicators, which import
    # this package while it is still initialising.
    from . import providers

    if providers.monitor_held():
        return
    if read_environment:
        config = resolve_environment(config, world_size)
    _configured = _settings_of(config)
    from_environment = getattr(config, "_from_environment", frozenset())
    _from_environment = frozenset(
        var for var, field in _FIELDS.items() if field in from_environment
    )
    _parsed_ranks = None
    _ranks_outside_job = None


def reset() -> None:
    """Forget the configuration: settings come from the environment again."""
    global _configured, _from_environment, _parsed_ranks, _ranks_outside_job
    _configured = None
    _from_environment = frozenset()
    _parsed_ranks = None
    _ranks_outside_job = None


def setting(var: str) -> str | None:
    """A monitor setting, as the value of the variable ``var`` it stands for.

    The configured value once :func:`configure` ran; before that, ``var``
    from the environment, a blank value counting as unset.
    """
    if _configured is None:
        return _environment(var)
    return _configured.get(var)


def setting_label(var: str) -> str:
    """How messages name the setting ``var``: where its value came from."""
    if _configured is None or var in _from_environment or var not in _FIELDS:
        return var
    return f"memory_monitor_config.{_FIELDS[var]}"


def warn_once(var: str, raw: str, problem: str) -> None:
    """Log one warning per invalid ``var=raw`` setting and problem for the process."""
    logger.warning_once(
        f"memory monitor: {setting_label(var)}={raw!r} {problem}",
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
    * ``oom`` (the default, also when unset) -- a report when the executor's
      own error handling sees an out-of-memory error: a failed request, a
      crashed executor loop or a failed wakeup. Nothing is recorded during
      startup or serving and no process-wide hook is installed; the report
      has every row but no startup-step breakdown and no comparison with
      warmup, which need recording during startup.
    * ``1`` (also ``full``, ``on``, ``true``, ``yes``) -- full: also records
      startup, reports when it completes, reports out-of-memory errors seen
      anywhere in the process (logged errors, dying threads, PyTorch's OOM
      observer), and enables the periodic report, stack mode, every-rank
      gathering under MPI and ``trtllm-serve``'s ``/memory_report``.

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

    :func:`full_mode` can differ between ranks; for a decision every rank
    must share, use :func:`full_mode_in` with the arguments' configuration.
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
    "0,4 or 0-3; ignored"
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

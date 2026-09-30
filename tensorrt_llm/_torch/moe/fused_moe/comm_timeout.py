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
"""Phase-aware device-side timeouts for MoE communication kernels.

Several MoE communication kernels wait for their peers inside the kernel and trap once a peer is
late by more than a device-side timeout. The trap is a sticky CUDA error, so every waiting rank
dies. During warmup, one-time per-rank work (JIT compilation, autotuning, module loading) can
delay a healthy rank by minutes, and no collective separates that work from the next launch.

This module relaxes the timeout of every registered backend while the process is in
``ExecutionPhase.WARMUP`` and applies the serving budget otherwise. Backends register a
``MoECommTimeoutSink`` when they are constructed.

Environment variables (integer seconds in ``1..86400``):
    TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC: Warmup budget for every backend; defaults to 1800.
    TRTLLM_MOE_COMM_TIMEOUT_SEC: Serving budget for every backend; when unset, each backend
        keeps its native timeout.
"""

import dataclasses
import os
import threading
import weakref
from collections.abc import Callable, Mapping
from typing import Protocol

from tensorrt_llm.logger import logger

from ...execution_phase import (
    ExecutionPhase,
    ExecutionPhaseListener,
    add_execution_phase_listener,
    get_execution_phase,
)

__all__ = [
    "DEFAULT_WARMUP_TIMEOUT_SEC",
    "MAX_TIMEOUT_SEC",
    "SERVING_TIMEOUT_ENV",
    "WARMUP_TIMEOUT_ENV",
    "MoECommTimeoutBudgets",
    "MoECommTimeoutPolicy",
    "MoECommTimeoutSink",
    "get_moe_comm_timeout_budgets",
    "register_moe_comm_timeout_sink",
    "resolve_moe_comm_timeout_budgets",
    "unregister_moe_comm_timeout_sink",
]

WARMUP_TIMEOUT_ENV = "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC"
SERVING_TIMEOUT_ENV = "TRTLLM_MOE_COMM_TIMEOUT_SEC"
DEFAULT_WARMUP_TIMEOUT_SEC = 1800
MAX_TIMEOUT_SEC = 24 * 60 * 60

# The replacements broaden the scope: the deprecated names used to configure NVLinkOneSided only.
_DEPRECATED_ENV_ALIASES: dict[str, str] = {
    WARMUP_TIMEOUT_ENV: "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC",
    SERVING_TIMEOUT_ENV: "TRTLLM_MOE_A2A_TIMEOUT_SEC",
}


class MoECommTimeoutSink(Protocol):
    """A MoE communication backend whose kernels wait for peers under a device-side timeout."""

    name: str
    """Backend name used in logs."""

    def set_timeout_seconds(self, seconds: int | None) -> None:
        """Apply a timeout to the launches issued after this call.

        Args:
            seconds: Timeout in seconds, or ``None`` to restore the backend's native default.
        """
        ...


@dataclasses.dataclass(frozen=True)
class MoECommTimeoutBudgets:
    """Timeouts per execution phase.

    Attributes:
        warmup_seconds: Timeout applied to every backend during warmup.
        serving_seconds: Timeout applied to every backend while serving; ``None`` keeps each
            backend's native default.
    """

    warmup_seconds: int
    serving_seconds: int | None

    def for_phase(self, phase: ExecutionPhase) -> int | None:
        """Return the timeout for ``phase``; ``None`` selects native defaults."""
        if phase is ExecutionPhase.WARMUP:
            return self.warmup_seconds
        return self.serving_seconds


def resolve_moe_comm_timeout_budgets(environ: Mapping[str, str]) -> MoECommTimeoutBudgets:
    """Resolve the timeout budgets from environment variables.

    Args:
        environ: The environment to read, usually ``os.environ``.

    Returns:
        The budgets for both phases.

    Raises:
        ValueError: If a value is not an integer in ``1..86400``, a deprecated alias conflicts
            with its replacement, or the warmup budget is shorter than the serving budget.
    """
    warmup_seconds = _read_timeout_seconds(environ, WARMUP_TIMEOUT_ENV)
    serving_seconds = _read_timeout_seconds(environ, SERVING_TIMEOUT_ENV)
    if warmup_seconds is None:
        warmup_seconds = DEFAULT_WARMUP_TIMEOUT_SEC
    if serving_seconds is not None and warmup_seconds < serving_seconds:
        raise ValueError(
            f"{WARMUP_TIMEOUT_ENV} resolves to {warmup_seconds} s, which is shorter than "
            f"{SERVING_TIMEOUT_ENV}={serving_seconds} s; warmup needs at least the serving budget"
        )
    return MoECommTimeoutBudgets(warmup_seconds=warmup_seconds, serving_seconds=serving_seconds)


class MoECommTimeoutPolicy:
    """Pushes the timeout of the current execution phase to the registered sinks.

    Budgets are resolved from the environment on first use and then stay fixed. Sinks are held
    weakly; their owners keep them alive and unregister them before releasing the resources the
    sinks control.

    Args:
        environ: The environment holding the budget variables.
        get_phase: Returns the current execution phase.
        add_phase_listener: Subscribes a callback to execution-phase transitions.
    """

    def __init__(
        self,
        environ: Mapping[str, str],
        get_phase: Callable[[], ExecutionPhase],
        add_phase_listener: Callable[[ExecutionPhaseListener], None],
    ) -> None:
        self._environ = environ
        self._get_phase = get_phase
        self._add_phase_listener = add_phase_listener
        # Phase transitions call ``_on_phase_change`` under the execution-phase lock and then take
        # ``_lock``. Subscribing takes the execution-phase lock, so it must not run under
        # ``_lock``; it has its own lock instead.
        self._lock = threading.Lock()
        self._subscribe_lock = threading.Lock()
        self._subscribed = False
        self._budgets: MoECommTimeoutBudgets | None = None
        self._sinks: weakref.WeakSet[MoECommTimeoutSink] = weakref.WeakSet()

    def budgets(self) -> MoECommTimeoutBudgets:
        """Return the budgets, resolving them from the environment on first use.

        Raises:
            ValueError: If the budget environment variables are invalid.
        """
        with self._lock:
            return self._resolve_budgets()

    def register(self, sink: MoECommTimeoutSink) -> None:
        """Apply the current phase's timeout to ``sink`` and keep it updated on transitions.

        A sink whose setter raises is not registered, and the error propagates.

        Args:
            sink: The backend to control.

        Raises:
            ValueError: If the budget environment variables are invalid.
        """
        self._subscribe()
        with self._lock:
            budgets = self._resolve_budgets()
            sink.set_timeout_seconds(budgets.for_phase(self._get_phase()))
            self._sinks.add(sink)

    def unregister(self, sink: MoECommTimeoutSink) -> None:
        """Stop updating ``sink``; unknown sinks are ignored.

        Args:
            sink: The backend to release.
        """
        with self._lock:
            self._sinks.discard(sink)

    def _subscribe(self) -> None:
        with self._subscribe_lock:
            if not self._subscribed:
                self._add_phase_listener(self._on_phase_change)
                self._subscribed = True

    def _resolve_budgets(self) -> MoECommTimeoutBudgets:
        if self._budgets is None:
            self._budgets = resolve_moe_comm_timeout_budgets(self._environ)
            serving = self._budgets.serving_seconds
            logger.info(
                f"MoE communication timeouts: warmup={self._budgets.warmup_seconds} s, "
                f"serving={'native defaults' if serving is None else f'{serving} s'}"
            )
        return self._budgets

    def _on_phase_change(self, phase: ExecutionPhase) -> None:
        with self._lock:
            seconds = self._resolve_budgets().for_phase(phase)
            sinks = list(self._sinks)
            for sink in sinks:
                sink.set_timeout_seconds(seconds)
        if sinks:
            applied = "native defaults" if seconds is None else f"{seconds} s"
            names = ", ".join(sink.name for sink in sinks)
            logger.info(f"MoE communication timeouts for {phase.value}: {applied} ({names})")


def _read_timeout_seconds(environ: Mapping[str, str], name: str) -> int | None:
    value = _parse_timeout_seconds(environ, name)
    alias = _DEPRECATED_ENV_ALIASES[name]
    alias_value = _parse_timeout_seconds(environ, alias)
    if alias_value is None:
        return value
    logger.warning_once(
        f"{alias} is deprecated; use {name}, which applies to every MoE communication kernel.",
        key=f"deprecated_env_{alias}",
    )
    if value is not None and value != alias_value:
        raise ValueError(f"{name}={value} conflicts with the deprecated {alias}={alias_value}")
    return alias_value


def _parse_timeout_seconds(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    error = ValueError(
        f"{name}={raw!r} must be an integer number of seconds in 1..{MAX_TIMEOUT_SEC}"
    )
    try:
        seconds = int(raw)
    except ValueError:
        raise error from None
    if not 1 <= seconds <= MAX_TIMEOUT_SEC:
        raise error
    return seconds


_DEFAULT_POLICY = MoECommTimeoutPolicy(
    environ=os.environ,
    get_phase=get_execution_phase,
    add_phase_listener=add_execution_phase_listener,
)


def get_moe_comm_timeout_budgets() -> MoECommTimeoutBudgets:
    """Return the process-wide budgets, resolving them from ``os.environ`` on first use.

    Backend selection calls this before trying any backend, because selection treats a
    constructor error as "backend unavailable" and would otherwise hide an invalid variable.

    Raises:
        ValueError: If the budget environment variables are invalid.
    """
    return _DEFAULT_POLICY.budgets()


def register_moe_comm_timeout_sink(sink: MoECommTimeoutSink) -> None:
    """Register ``sink`` with the process-wide policy; see ``MoECommTimeoutPolicy.register``.

    Args:
        sink: The backend to control.
    """
    _DEFAULT_POLICY.register(sink)


def unregister_moe_comm_timeout_sink(sink: MoECommTimeoutSink) -> None:
    """Unregister ``sink`` from the process-wide policy.

    Args:
        sink: The backend to release.
    """
    _DEFAULT_POLICY.unregister(sink)

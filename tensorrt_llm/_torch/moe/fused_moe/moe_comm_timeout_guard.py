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
"""Warmup-aware device-side timeouts for MoE communication kernels.

Several MoE communication kernels wait for their peers inside the kernel and trap once a peer is
late by more than a device-side timeout. The trap is a sticky CUDA error, so every waiting rank
dies. During warmup, one-time per-rank work (JIT compilation, autotuning, module loading) can
delay a healthy rank by minutes, and no collective separates that work from the next launch.

The model engine reports each warmup transition here. The guard applies the warmup timeout to
every registered backend while the engine warms up and the serving timeout otherwise. Each
backend registers a ``MoECommTimeoutProxy`` when it is constructed.

Environment variables (integer seconds in ``1..86400``):
    TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC: Warmup timeout for every backend; defaults to 1800.
    TRTLLM_MOE_COMM_TIMEOUT_SEC: Serving timeout for every backend; when unset, each backend
        keeps its native timeout.
"""

import contextlib
import dataclasses
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from typing import Protocol

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "DEFAULT_WARMUP_TIMEOUT_SEC",
    "MAX_TIMEOUT_SEC",
    "SERVING_TIMEOUT_ENV",
    "WARMUP_TIMEOUT_ENV",
    "MoECommTimeoutBudgets",
    "MoECommTimeoutGuard",
    "MoECommTimeoutProxy",
    "get_moe_comm_timeout_budgets",
    "moe_comm_serving_timeouts",
    "register_moe_comm_timeout_proxy",
    "resolve_moe_comm_timeout_budgets",
    "set_moe_comm_warmup",
    "unregister_moe_comm_timeout_proxy",
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


class MoECommTimeoutProxy(Protocol):
    """Stands in for the device-side timeout of one MoE communication backend."""

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
    """Timeouts for warmup and for serving.

    Attributes:
        warmup_seconds: Timeout applied to every backend during warmup.
        serving_seconds: Timeout applied to every backend while serving; ``None`` keeps each
            backend's native default.
    """

    warmup_seconds: int
    serving_seconds: int | None

    def select(self, in_warmup: bool) -> int | None:
        """Return the warmup or serving timeout; ``None`` selects native defaults."""
        return self.warmup_seconds if in_warmup else self.serving_seconds


def resolve_moe_comm_timeout_budgets(environ: Mapping[str, str]) -> MoECommTimeoutBudgets:
    """Resolve the timeout budgets from environment variables.

    Args:
        environ: The environment to read, usually ``os.environ``.

    Returns:
        The budgets for warmup and serving.

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


def _is_capturing_cuda_graph() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class MoECommTimeoutGuard:
    """Applies the warmup or serving timeout to every registered proxy.

    The guard starts in serving. Budgets are resolved from the environment the first time a proxy
    needs one and then stay fixed. Proxies are held weakly; their owners keep them alive and
    unregister them before releasing the resources the proxies control.

    Args:
        environ: The environment holding the budget variables.
        is_capturing: Returns whether the current CUDA stream is capturing a graph.
    """

    def __init__(
        self,
        environ: Mapping[str, str],
        is_capturing: Callable[[], bool] = _is_capturing_cuda_graph,
    ) -> None:
        self._environ = environ
        self._is_capturing = is_capturing
        self._lock = threading.Lock()
        self._in_warmup = False
        self._budgets: MoECommTimeoutBudgets | None = None
        self._proxies: weakref.WeakSet[MoECommTimeoutProxy] = weakref.WeakSet()

    @property
    def in_warmup(self) -> bool:
        """Whether the warmup timeout is in effect."""
        return self._in_warmup

    def budgets(self) -> MoECommTimeoutBudgets:
        """Return the budgets, resolving them from the environment on first use.

        Raises:
            ValueError: If the budget environment variables are invalid.
        """
        with self._lock:
            return self._resolve_budgets()

    def register(self, proxy: MoECommTimeoutProxy) -> None:
        """Apply the current timeout to ``proxy`` and keep it updated on warmup transitions.

        A proxy whose setter raises is not registered, and the error propagates.

        Args:
            proxy: The backend to control.

        Raises:
            ValueError: If the budget environment variables are invalid.
        """
        with self._lock:
            proxy.set_timeout_seconds(self._resolve_budgets().select(self._in_warmup))
            self._proxies.add(proxy)

    def unregister(self, proxy: MoECommTimeoutProxy) -> None:
        """Stop updating ``proxy``; unknown proxies are ignored.

        Args:
            proxy: The backend to release.
        """
        with self._lock:
            self._proxies.discard(proxy)

    def set_warmup(self, in_warmup: bool) -> None:
        """Apply the warmup or serving timeout to every registered proxy.

        Args:
            in_warmup: Whether the engine enters warmup.

        Raises:
            RuntimeError: If the current CUDA stream is capturing a graph. Proxies may issue
                host-synchronizing CUDA calls, which are illegal during capture.
            ValueError: If the budget environment variables are invalid.
        """
        if self._is_capturing():
            raise RuntimeError(
                "Cannot switch MoE communication timeouts while a CUDA graph is being captured"
            )
        with self._lock:
            if in_warmup == self._in_warmup:
                return
            self._in_warmup = in_warmup
            proxies = list(self._proxies)
            if not proxies:
                return
            seconds = self._resolve_budgets().select(in_warmup)
            for proxy in proxies:
                proxy.set_timeout_seconds(seconds)
        applied = "native defaults" if seconds is None else f"{seconds} s"
        names = ", ".join(proxy.name for proxy in proxies)
        stage = "warmup" if in_warmup else "serving"
        logger.info(f"MoE communication timeouts for {stage}: {applied} ({names})")

    @contextlib.contextmanager
    def serving_timeouts(self) -> Iterator[None]:
        """Apply the serving timeout in the enclosed block, then restore the previous timeout.

        A captured CUDA graph keeps the timeouts that were current at capture, and serving
        replays it, so the pass that captures uses the serving timeout even during warmup.
        """
        previous = self._in_warmup
        self.set_warmup(False)
        try:
            yield
        finally:
            self.set_warmup(previous)

    def _resolve_budgets(self) -> MoECommTimeoutBudgets:
        if self._budgets is None:
            self._budgets = resolve_moe_comm_timeout_budgets(self._environ)
            serving = self._budgets.serving_seconds
            logger.info(
                f"MoE communication timeouts: warmup={self._budgets.warmup_seconds} s, "
                f"serving={'native defaults' if serving is None else f'{serving} s'}"
            )
        return self._budgets


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


_DEFAULT_GUARD = MoECommTimeoutGuard(environ=os.environ)


def get_moe_comm_timeout_budgets() -> MoECommTimeoutBudgets:
    """Return the process-wide budgets, resolving them from ``os.environ`` on first use.

    Backend selection calls this before trying any backend, because selection treats a
    constructor error as "backend unavailable" and would otherwise hide an invalid variable.

    Raises:
        ValueError: If the budget environment variables are invalid.
    """
    return _DEFAULT_GUARD.budgets()


def register_moe_comm_timeout_proxy(proxy: MoECommTimeoutProxy) -> None:
    """Register ``proxy`` with the process-wide guard; see ``MoECommTimeoutGuard.register``.

    Args:
        proxy: The backend to control.
    """
    _DEFAULT_GUARD.register(proxy)


def unregister_moe_comm_timeout_proxy(proxy: MoECommTimeoutProxy) -> None:
    """Unregister ``proxy`` from the process-wide guard.

    Args:
        proxy: The backend to release.
    """
    _DEFAULT_GUARD.unregister(proxy)


def set_moe_comm_warmup(in_warmup: bool) -> None:
    """Switch the process-wide guard; see ``MoECommTimeoutGuard.set_warmup``.

    Args:
        in_warmup: Whether the engine enters warmup.
    """
    _DEFAULT_GUARD.set_warmup(in_warmup)


@contextlib.contextmanager
def moe_comm_serving_timeouts() -> Iterator[None]:
    """Apply the serving timeouts in the enclosed block.

    See ``MoECommTimeoutGuard.serving_timeouts``.
    """
    with _DEFAULT_GUARD.serving_timeouts():
        yield

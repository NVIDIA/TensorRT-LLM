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
"""Monitor providers.

A provider reads a component that already keeps its own accounting. The
contract is deliberately narrow:

``available()``
    ``False`` when the host object cannot be found. A provider that cannot
    look must not report ``0`` -- that would make "this component holds
    nothing" indistinguishable from "we failed to measure it".
``collect()``
    Entries, each declaring its :class:`~..schema.Basis`.

Collection never raises into the caller: a diagnostic that breaks the process
it is diagnosing is worse than a missing line.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tensorrt_llm.logger import logger

from .. import pools
from ..schema import GROUP_OF, Basis, Category, Entry, Placement, Source
from .comm import IpcMemoryProvider, McastAllReduceProvider, NcclWindowProvider, SymmMemProvider
from .misc import (
    DeepEpProvider,
    DwdpLocalWeightsProvider,
    DwdpPagePoolProvider,
    MemoryCountersProvider,
)
from .mnnvl import MnnvlProvider
from .native import NativeMemoryProvider

__all__ = [
    "DeepEpProvider",
    "DwdpLocalWeightsProvider",
    "DwdpPagePoolProvider",
    "ExecutorToken",
    "IpcMemoryProvider",
    "McastAllReduceProvider",
    "MemoryCountersProvider",
    "MnnvlProvider",
    "MonitorProvider",
    "NOT_MONITORED",
    "NativeMemoryProvider",
    "NcclWindowProvider",
    "SymmMemProvider",
    "begin_executor",
    "clear_registered_providers",
    "current_executor",
    "collect_all",
    "default_providers",
    "end_executor",
    "executor_monitored",
    "monitor_held",
    "register_provider",
    "unmonitored_executor_current",
    "unregister_providers_of",
]


@runtime_checkable
class MonitorProvider(Protocol):
    name: str
    category: Category

    def available(self) -> bool: ...

    def collect(self) -> list[Entry]: ...


#: Providers whose host object cannot be reached from here, with the
#: executor that registered them. The KV cache manager lives in a
#: ``ResourceManager`` owned by the executor and a
#: ``SymmetricMemoryAllReduce`` is a model submodule; neither has a
#: process-wide accessor, so the code that builds them registers them
#: instead of this module guessing where to look.
_registered: list[tuple[object | None, MonitorProvider]] = []

#: Why an executor gets no monitoring: the monitor follows one executor per process.
NOT_MONITORED = (
    "not monitored: another executor in this process holds the GPU memory monitor, "
    "which follows one executor per process"
)


class ExecutorToken:
    """An executor's registration with the monitor, from :func:`begin_executor`.

    ``monitored`` says whether the monitor follows this executor. Only one
    executor per process is monitored: the monitor's startup records,
    warmup baseline, device and settings describe a single executor, and a
    second executor in the same process would otherwise mix its memory into
    the first one's reports.
    """

    __slots__ = ("monitored",)

    def __init__(self, monitored: bool) -> None:
        self.monitored = monitored

    def __repr__(self) -> str:
        return f"ExecutorToken(monitored={self.monitored})"


#: The executor being built; registrations made meanwhile belong to it.
_current_executor: ExecutorToken | None = None
#: The executor the monitor follows, from its creation to its shutdown.
_monitored: ExecutorToken | None = None


def begin_executor() -> ExecutorToken:
    """Start a new executor's registrations; returns its token.

    The executor is monitored unless another monitored executor is still
    alive in this process; one that is not is told so once, in the log.
    Providers registered while it is being built belong to it. Pass the
    token to :func:`end_executor` at shutdown.
    """
    global _current_executor, _monitored
    token = ExecutorToken(monitored=_monitored is None)
    if token.monitored:
        _monitored = token
    else:
        logger.warning(
            "memory monitor: another executor in this process is monitored, so this "
            "executor is not: it gets no out-of-memory reports, startup record or "
            "memory report of its own"
        )
    _current_executor = token
    return token


def current_executor() -> ExecutorToken | None:
    """The token of the executor being built, if any."""
    return _current_executor


def executor_monitored(token: ExecutorToken | None = None) -> bool:
    """Whether the monitor follows ``token``'s executor (default: the one being built)."""
    if token is None:
        token = _current_executor
    return token is not None and token is _monitored


def unmonitored_executor_current() -> bool:
    """Whether the executor being built, or last built, is one the monitor does not follow.

    Its startup is not recorded, its providers are not registered and its
    creation does not suspend reporting for the monitored executor. False
    when no executor was begun, so code run outside an executor is
    monitored as before.
    """
    return _current_executor is not None and _current_executor is not _monitored


def monitor_held() -> bool:
    """Whether a monitored executor is alive in this process."""
    return _monitored is not None


def end_executor(token: ExecutorToken) -> None:
    """Release everything the monitor holds for ``token``'s executor, at its shutdown.

    For the monitored executor: its out-of-memory reporting, its startup
    records and stack recording, its registered providers and the
    sleep-mode release state its pools had; the next executor created in
    this process is then monitored. Safe to call more than once and for an
    executor that is not monitored.
    """
    global _current_executor, _monitored
    # Imported here: both modules build on this package.
    from .. import tagging, triggers

    if _current_executor is token:
        _current_executor = None
    if token is not _monitored:
        return
    triggers.release_oom_reporter(owner=token)
    tagging.forget_startup_of(token)
    _registered[:] = [(owner, p) for owner, p in _registered if owner is not token]
    _monitored = None
    pools.forget_released_tags()


def register_provider(provider: MonitorProvider) -> None:
    """Add a provider built by code that owns the component it reads.

    Registering the same ``name`` twice for one executor replaces the
    earlier one rather than reporting the component twice: a rebuilt KV
    cache manager supersedes the one it replaced. A provider registered
    while an executor that is not monitored is being built is dropped.
    """
    if unmonitored_executor_current():
        return
    owner = _current_executor
    _registered[:] = [
        (o, p) for o, p in _registered if not (o is owner and p.name == provider.name)
    ]
    _registered.append((owner, provider))


def unregister_providers_of(component: object) -> None:
    """Drop every provider that reads ``component``, at its release.

    For a component released while its executor lives on, such as the KV
    cache the sizing pass builds and tears down. A provider says what it
    reads through ``describes(component)``; one without it is kept.
    """
    _registered[:] = [
        (o, p)
        for o, p in _registered
        if not (callable(getattr(p, "describes", None)) and p.describes(component))
    ]


def clear_registered_providers() -> None:
    """Drop every registration and executor token, of every executor."""
    global _current_executor, _monitored
    _registered.clear()
    _current_executor = None
    _monitored = None


def default_providers() -> list[MonitorProvider]:
    """Providers that can find their host object without cooperation.

    Includes anything handed to :func:`register_provider`, which is how
    components that cannot be found from here get in.
    """
    return [
        MnnvlProvider(),
        McastAllReduceProvider(),
        SymmMemProvider(),
        IpcMemoryProvider(),
        NcclWindowProvider(),
        NativeMemoryProvider(),
        DeepEpProvider(),
        DwdpPagePoolProvider(),
        DwdpLocalWeightsProvider(),
        MemoryCountersProvider(),
        *(p for _, p in _registered),
    ]


def _failed_entry(provider: MonitorProvider, exc: Exception) -> Entry:
    """A provider that raised, listed as unmeasured rather than left out."""
    category = provider.category
    if GROUP_OF.get(category) is None:
        category = Category.TAGGED_OTHER
    return Entry(
        category=category,
        size_bytes=None,
        source=Source.COMPONENT,
        placement=Placement.NON_TORCH,
        basis=Basis.PHYSICAL_BACKING,
        owner=provider.name,
        note=f"provider failed: {type(exc).__name__}: {exc}",
    )


def collect_all(
    providers: list[MonitorProvider] | None = None,
    pool_rows: dict[str, list[dict]] | None = None,
) -> list[Entry]:
    """Collect from every available provider, tolerating individual failures.

    A provider that raises is reported as unmeasured, so its memory shows up
    as a named gap in the overhead row instead of silently disappearing.

    ``pool_rows``, when given, receives the KV pool-group rows each KV cache
    provider read for its entries, keyed by provider name, so a check on
    them describes this collection and no other.
    """
    entries: list[Entry] = []
    for provider in default_providers() if providers is None else providers:
        try:
            if not provider.available():
                continue
            read_pool_rows = getattr(provider, "collect_with_pool_rows", None)
            if pool_rows is not None and read_pool_rows is not None:
                collected, rows = read_pool_rows()
                if rows is not None:
                    pool_rows[provider.name] = rows
                entries.extend(collected)
                continue
            entries.extend(provider.collect())
        except Exception as exc:  # one provider must not break the report
            # TensorRT-LLM's logger is not the stdlib one and takes no
            # exc_info; formatting here keeps the handler itself from
            # raising, which would defeat the point of catching.
            logger.debug(
                f"memory monitor: provider {provider.name!r} failed ({type(exc).__name__}: {exc})"
            )
            entries.append(_failed_entry(provider, exc))
    return entries

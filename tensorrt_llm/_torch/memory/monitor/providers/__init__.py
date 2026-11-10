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
    "IpcMemoryProvider",
    "McastAllReduceProvider",
    "MemoryCountersProvider",
    "MnnvlProvider",
    "MonitorProvider",
    "NativeMemoryProvider",
    "NcclWindowProvider",
    "SymmMemProvider",
    "begin_executor",
    "clear_registered_providers",
    "collect_all",
    "default_providers",
    "end_executor",
    "register_provider",
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
#: The executor being built; registrations made meanwhile belong to it.
_current_executor: object | None = None


def begin_executor() -> object:
    """Start a new executor's registrations; returns its token.

    Providers registered from now on belong to this executor, so a second
    executor in the same process adds its own KV cache instead of replacing
    the first one's. Pass the token to :func:`end_executor` at shutdown.
    """
    global _current_executor
    _current_executor = object()
    return _current_executor


def end_executor(token: object) -> None:
    """Drop what the executor ``token`` registered, at its shutdown.

    Once no executor has registrations left, sleep-mode release state is
    forgotten too: the pools it described are gone, and a new executor's
    pools under the same tag would otherwise look released.
    """
    global _current_executor
    _registered[:] = [(owner, p) for owner, p in _registered if owner is not token]
    if _current_executor is token:
        _current_executor = None
    if not any(owner is not None for owner, _ in _registered):
        pools.forget_released_tags()


def register_provider(provider: MonitorProvider) -> None:
    """Add a provider built by code that owns the component it reads.

    Registering the same ``name`` twice for one executor replaces the
    earlier one rather than reporting the component twice: a rebuilt KV
    cache manager supersedes the one it replaced. Another executor's
    provider of the same name is kept.
    """
    owner = _current_executor
    _registered[:] = [
        (o, p) for o, p in _registered if not (o is owner and p.name == provider.name)
    ]
    _registered.append((owner, provider))


def clear_registered_providers() -> None:
    """Drop every registration, of every executor."""
    global _current_executor
    _registered.clear()
    _current_executor = None


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
    category = getattr(provider, "category", None)
    if GROUP_OF.get(category) is None:
        category = Category.TAGGED_OTHER
    return Entry(
        category=category,
        size_bytes=None,
        source=Source.COMPONENT,
        placement=Placement.NON_TORCH,
        basis=Basis.PHYSICAL_BACKING,
        owner=getattr(provider, "name", type(provider).__name__),
        note=f"provider failed: {type(exc).__name__}: {exc}",
    )


def collect_all(providers: list[MonitorProvider] | None = None) -> list[Entry]:
    """Collect from every available provider, tolerating individual failures.

    A provider that raises is reported as unmeasured, so its memory shows up
    as a named gap in the overhead row instead of silently disappearing.
    """
    entries: list[Entry] = []
    for provider in default_providers() if providers is None else providers:
        try:
            if not provider.available():
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

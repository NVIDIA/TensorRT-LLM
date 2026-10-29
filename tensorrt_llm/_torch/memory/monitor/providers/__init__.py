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

from typing import Protocol, runtime_checkable

from tensorrt_llm.logger import logger

from ..schema import Category, Entry
from .comm import McastAllReduceProvider, SymmMemProvider
from .misc import DeepEpProvider, DwdpPagePoolProvider, MemoryCountersProvider
from .mnnvl import MnnvlProvider

__all__ = [
    "DeepEpProvider",
    "McastAllReduceProvider",
    "SymmMemProvider",
    "clear_registered_providers",
    "DwdpPagePoolProvider",
    "MonitorProvider",
    "MemoryCountersProvider",
    "MnnvlProvider",
    "collect_all",
    "default_providers",
    "register_provider",
]


@runtime_checkable
class MonitorProvider(Protocol):
    name: str
    category: Category

    def available(self) -> bool: ...

    def collect(self) -> list[Entry]: ...


#: Providers whose host object cannot be reached from here. The KV cache
#: manager lives in a ``ResourceManager`` owned by the executor and a
#: ``SymmetricMemoryAllReduce`` is a model submodule; neither has a
#: process-wide accessor, so the code that builds them registers them
#: instead of this module guessing where to look.
_registered: list["MonitorProvider"] = []


def register_provider(provider: "MonitorProvider") -> None:
    """Add a provider built by code that owns the component it reads.

    Registering the same ``name`` twice replaces the earlier one rather than
    reporting the component twice: a rebuilt KV cache manager supersedes the
    one it replaced.
    """
    global _registered
    _registered = [p for p in _registered if p.name != provider.name]
    _registered.append(provider)


def clear_registered_providers() -> None:
    """Drop every registration. For tests, and for executor teardown."""
    _registered.clear()


def default_providers() -> list[MonitorProvider]:
    """Providers that can find their host object without cooperation.

    Includes anything handed to :func:`register_provider`, which is how
    components that cannot be found from here get in.
    """
    return [
        MnnvlProvider(),
        McastAllReduceProvider(),
        SymmMemProvider(),
        DeepEpProvider(),
        DwdpPagePoolProvider(),
        MemoryCountersProvider(),
        *_registered,
    ]


def collect_all(providers: list[MonitorProvider] | None = None) -> list[Entry]:
    """Collect from every available provider, tolerating individual failures."""
    entries: list[Entry] = []
    for provider in default_providers() if providers is None else providers:
        try:
            if not provider.available():
                continue
            entries.extend(provider.collect())
        except Exception as exc:  # noqa: BLE001 - never break the caller
            # TensorRT-LLM's logger is not the stdlib one and takes no
            # exc_info; formatting here keeps the handler itself from
            # raising, which would defeat the point of catching.
            logger.debug(
                f"memory monitor: provider {provider.name!r} failed "
                f"({type(exc).__name__}: {exc}); skipped"
            )
    return entries

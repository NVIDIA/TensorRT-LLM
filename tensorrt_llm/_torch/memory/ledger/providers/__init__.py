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
"""Ledger providers.

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
from .misc import DeepEpProvider, DwdpPagePoolProvider, MemoryCountersProvider
from .mnnvl import MnnvlProvider

__all__ = [
    "DeepEpProvider",
    "DwdpPagePoolProvider",
    "LedgerProvider",
    "MemoryCountersProvider",
    "MnnvlProvider",
    "collect_all",
    "default_providers",
]


@runtime_checkable
class LedgerProvider(Protocol):
    name: str
    category: Category

    def available(self) -> bool: ...

    def collect(self) -> list[Entry]: ...


def default_providers() -> list[LedgerProvider]:
    """Providers that can find their host object without cooperation.

    Absent by design: ``SymmetricMemoryAllReduce`` has no process-wide
    handle -- each instance is a submodule of the model -- so it needs either
    a registration call at construction or a model reference to walk. It is
    not discoverable from here, and guessing zero would be a lie.
    """
    return [
        MnnvlProvider(),
        DeepEpProvider(),
        DwdpPagePoolProvider(),
        MemoryCountersProvider(),
    ]


def collect_all(providers: list[LedgerProvider] | None = None) -> list[Entry]:
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
                f"memory ledger: provider {provider.name!r} failed "
                f"({type(exc).__name__}: {exc}); skipped"
            )
    return entries

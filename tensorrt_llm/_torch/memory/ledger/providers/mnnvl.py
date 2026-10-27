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
"""MNNVL provider.

Two things make a naive rollup wrong, and both have bitten us:

**Filtering on ``MAPPED`` hides memory that is definitely resident.** Backing
is created *after* the record moves to ``RESTORING``
(``mnnvl_memory.py``: state set, then ``_create_and_map_handles``, and only
after the frontend protocol completes does it become ``MAPPED``). ``PREPARING``
likewise still holds its handles -- the unmap happens on the way to
``UNMAPPED``. So a ``MAPPED``-only filter reports zero precisely during a
failed restore or an OOM, which is when the number matters most.

**``comm_size`` describes a virtual address window, not bytes.** Each record
has exactly one local ``cuMemCreate`` of ``aligned_size``; every peer's handle
is imported and costs no local memory. Multiplying by ``comm_size`` inflates
the figure by the number of peers. Worse, several records share one address
reservation (refcounted by ``start_address``), so even summing the windows
double counts unless deduplicated.
"""

from ..schema import Basis, Category, Entry, Placement, Source

__all__ = ["MnnvlProvider"]


def _walk_classes(root: type) -> list[type]:
    """``root`` and every subclass, breadth first.

    Subclasses matter: ``__init_subclass__`` gives each one its *own*
    ``allocated_map``, so walking only the base class silently misses
    ``CftMnnvlMemory`` and ``HelixCpMnnvlMemory``.
    """
    seen: list[type] = []
    stack = [root]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


class MnnvlProvider:
    """Local MNNVL backing, split by how certain we are that it is resident."""

    name = "mnnvl"
    category = Category.COMM_MNNVL

    def __init__(self) -> None:
        self._memory_cls: type | None = None
        self._state_cls: type | None = None

    def _resolve(self) -> bool:
        if self._memory_cls is not None:
            return True
        try:
            from tensorrt_llm._torch.distributed.mnnvl_memory import (
                MnnvlMemory,
                _MnnvlAllocationState,
            )
        except ImportError:
            return False
        self._memory_cls = MnnvlMemory
        self._state_cls = _MnnvlAllocationState
        return True

    def available(self) -> bool:
        """Whether MNNVL state can be read at all.

        A process with MNNVL present but no allocations is *available* and
        reports zero; a process where the module cannot be imported is
        unavailable and reports nothing. Collapsing the two would make
        "no MNNVL here" indistinguishable from "we failed to look".
        """
        return self._resolve()

    def _maps(self) -> list[dict]:
        maps = []
        for cls in _walk_classes(self._memory_cls):
            # __dict__ rather than getattr: a class that somehow lacked its
            # own map would otherwise fall through to the parent's and get
            # counted twice.
            amap = cls.__dict__.get("allocated_map")
            if amap:
                maps.append(amap)
        return maps

    def collect(self) -> list[Entry]:
        if not self._resolve():
            return []
        states = self._state_cls

        # Holds physical backing right now.
        resident_states = {
            states.MAPPED,
            states.RESTORING,  # handles already created and mapped
            states.PREPARING,  # not yet unmapped
        }

        resident = 0
        uncertain = 0
        va_windows: dict[tuple[int, int], int] = {}

        for amap in self._maps():
            for record in list(amap.values()):
                size = record.aligned_size  # the one local cuMemCreate
                if record.state in resident_states:
                    resident += size
                elif record.state is states.BROKEN:
                    # A teardown that raised. The handles may or may not
                    # survive; reporting 0 would assert something we do not
                    # know.
                    uncertain += size
                # UNMAPPED is the only state that is definitely released.

                va_windows[(id(amap), record.start_address)] = record.comm_size * record.rank_stride

        entries = [
            Entry(
                category=Category.COMM_MNNVL,
                size_bytes=resident,
                source=Source.LEDGER,
                placement=Placement.NON_TORCH,
                basis=Basis.PHYSICAL_BACKING,
                owner="MnnvlMemory",
                note="local backing (mapped/restoring/preparing)",
            ),
        ]
        if uncertain:
            entries.append(
                Entry(
                    category=Category.COMM_MNNVL,
                    size_bytes=None,
                    source=Source.LEDGER,
                    placement=Placement.NON_TORCH,
                    basis=Basis.PHYSICAL_BACKING,
                    owner="MnnvlMemory",
                    note=(
                        f"{uncertain} bytes in BROKEN records; residency "
                        "unknown, not asserted to be zero"
                    ),
                )
            )
        if va_windows:
            entries.append(
                Entry(
                    category=Category.COMM_MNNVL,
                    size_bytes=sum(va_windows.values()),
                    source=Source.LEDGER,
                    placement=Placement.NON_TORCH,
                    basis=Basis.LOGICAL,
                    owner="MnnvlMemory",
                    note="VA reservation, deduplicated by start_address",
                    counts_toward_total=False,
                )
            )
        return entries

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
"""Provider tests for the GPU memory ledger.

The MNNVL cases are the point of this file. Backing is created *after* the
record moves to ``RESTORING``, so a ``MAPPED``-only rollup reports zero
exactly when a restore has failed or memory ran out -- the moment the number
is worth having. These tests encode that by constructing records in each
state; narrowing the filter back to ``MAPPED`` turns them red.

CPU only: everything is driven through fakes.
"""

import sys
import types

import pytest

from tensorrt_llm._torch.memory.ledger.providers.misc import (
    DwdpPagePoolProvider,
    MemoryCountersProvider,
)
from tensorrt_llm._torch.memory.ledger.providers.mnnvl import MnnvlProvider
from tensorrt_llm._torch.memory.ledger.schema import Basis, Category, Source

MB = 1 << 20


class _State:
    """Stand-in for ``_MnnvlAllocationState``."""

    MAPPED = "mapped"
    PREPARING = "preparing"
    UNMAPPED = "unmapped"
    RESTORING = "restoring"
    BROKEN = "broken"


class _Record:
    def __init__(self, state, aligned_size=MB, comm_size=4, rank_stride=0, start_address=0x1000):
        self.state = state
        self.aligned_size = aligned_size
        self.comm_size = comm_size
        self.rank_stride = rank_stride
        self.start_address = start_address


def _install_fake_mnnvl(monkeypatch, base_map, subclass_map=None):
    """Install a fake ``mnnvl_memory`` module mirroring the real layout.

    The real classes hold ``allocated_map`` per class via
    ``__init_subclass__``; subclasses therefore have their own maps.
    """

    class MnnvlMemory:
        allocated_map = base_map

    created = [MnnvlMemory]
    if subclass_map is not None:

        class CftMnnvlMemory(MnnvlMemory):
            allocated_map = subclass_map

        created.append(CftMnnvlMemory)

    module = types.ModuleType("tensorrt_llm._torch.distributed.mnnvl_memory")
    module.MnnvlMemory = MnnvlMemory
    module._MnnvlAllocationState = _State
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed.mnnvl_memory", module)
    return created


def _by_basis(entries, basis):
    return [e for e in entries if e.basis is basis]


class TestMnnvlResidency:
    @pytest.mark.parametrize(
        "state",
        [
            _State.MAPPED,
            _State.RESTORING,
            _State.PREPARING,
        ],
    )
    def test_states_holding_backing_are_counted(self, monkeypatch, state):
        # RESTORING: handles are created after the state is set and before
        # the frontend protocol completes.
        # PREPARING: the unmap has not happened yet.
        _install_fake_mnnvl(monkeypatch, {0: _Record(state, aligned_size=8 * MB)})
        entries = _by_basis(MnnvlProvider().collect(), Basis.PHYSICAL_BACKING)
        resident = [e for e in entries if e.size_bytes is not None]
        assert resident, f"state {state} reported nothing"
        assert resident[0].size_bytes == 8 * MB, (
            f"state {state} must count toward resident backing; reporting 0 "
            "here hides memory during a failed restore or an OOM"
        )

    def test_unmapped_is_the_only_released_state(self, monkeypatch):
        _install_fake_mnnvl(monkeypatch, {0: _Record(_State.UNMAPPED, aligned_size=8 * MB)})
        entries = _by_basis(MnnvlProvider().collect(), Basis.PHYSICAL_BACKING)
        assert entries[0].size_bytes == 0

    def test_broken_is_unknown_not_zero(self, monkeypatch):
        # A teardown that raised. Whether the handles survived is unknown,
        # and asserting zero would claim knowledge we do not have.
        _install_fake_mnnvl(monkeypatch, {0: _Record(_State.BROKEN, aligned_size=8 * MB)})
        entries = MnnvlProvider().collect()
        unknown = [e for e in entries if e.size_bytes is None]
        assert unknown, "BROKEN records must surface as unknown"
        assert "BROKEN" in (unknown[0].note or "")

    def test_mixed_states(self, monkeypatch):
        _install_fake_mnnvl(
            monkeypatch,
            {
                0: _Record(_State.MAPPED, aligned_size=4 * MB),
                1: _Record(_State.RESTORING, aligned_size=2 * MB),
                2: _Record(_State.UNMAPPED, aligned_size=16 * MB),
            },
        )
        entries = _by_basis(MnnvlProvider().collect(), Basis.PHYSICAL_BACKING)
        resident = [e for e in entries if e.size_bytes is not None][0]
        assert resident.size_bytes == 6 * MB


class TestMnnvlVirtualAddress:
    def test_local_bytes_not_multiplied_by_peers(self, monkeypatch):
        # One local cuMemCreate of aligned_size; peers' handles are imported
        # and cost no local memory.
        _install_fake_mnnvl(
            monkeypatch, {0: _Record(_State.MAPPED, aligned_size=100 * MB, comm_size=4)}
        )
        resident = _by_basis(MnnvlProvider().collect(), Basis.PHYSICAL_BACKING)[0]
        assert resident.size_bytes == 100 * MB
        assert resident.size_bytes != 400 * MB

    def test_va_reported_but_never_summed(self, monkeypatch):
        _install_fake_mnnvl(
            monkeypatch,
            {0: _Record(_State.MAPPED, aligned_size=MB, comm_size=4, rank_stride=64 * MB)},
        )
        va = _by_basis(MnnvlProvider().collect(), Basis.LOGICAL)
        assert va and va[0].counts_toward_total is False

    def test_shared_va_window_deduplicated(self, monkeypatch):
        # Several records share one address reservation, refcounted by
        # start_address; summing per record would double count.
        shared = dict(comm_size=4, rank_stride=64 * MB, start_address=0xABC000)
        _install_fake_mnnvl(
            monkeypatch,
            {
                0: _Record(_State.MAPPED, aligned_size=MB, **shared),
                1: _Record(_State.MAPPED, aligned_size=MB, **shared),
            },
        )
        va = _by_basis(MnnvlProvider().collect(), Basis.LOGICAL)[0]
        assert va.size_bytes == 4 * 64 * MB


class TestMnnvlSubclasses:
    def test_subclass_maps_are_included(self, monkeypatch):
        # __init_subclass__ gives each subclass its own allocated_map;
        # walking only the base silently misses CftMnnvlMemory and
        # HelixCpMnnvlMemory.
        _install_fake_mnnvl(
            monkeypatch,
            base_map={0: _Record(_State.MAPPED, aligned_size=MB)},
            subclass_map={0: _Record(_State.MAPPED, aligned_size=2 * MB)},
        )
        resident = _by_basis(MnnvlProvider().collect(), Basis.PHYSICAL_BACKING)[0]
        assert resident.size_bytes == 3 * MB

    def test_unavailable_when_module_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed.mnnvl_memory", None)
        provider = MnnvlProvider()
        assert provider.available() is False
        assert provider.collect() == []


class TestDwdpRefusesConfiguredCapacity:
    """``num_pages()`` keeps reporting capacity after the pages are gone."""

    class _LegacyPool:
        """Only the pre-statistics API, as on current main."""

        page_size = 16 * MB
        num_slots = 2

        def num_pages(self, slot):
            return 4  # configured; unchanged by release()

    def test_reports_unavailable_without_a_physical_accessor(self, monkeypatch):
        provider = DwdpPagePoolProvider()
        monkeypatch.setattr(provider, "_pool", lambda: self._LegacyPool())
        entry = provider.collect()[0]
        # 8 pages * 16 MiB = 128 MiB would be invented out of nothing.
        assert entry.size_bytes is None
        assert entry.basis is Basis.CONFIGURED_CAPACITY

    def test_uses_held_pages_once_available(self, monkeypatch):
        class _Pool:
            page_size = 16 * MB
            num_slots = 2

            def configured_pages(self, slot):
                return 4

            def held_pages(self, slot):
                return 0  # everything released

        provider = DwdpPagePoolProvider()
        monkeypatch.setattr(provider, "_pool", lambda: _Pool())
        entry = provider.collect()[0]
        assert entry.size_bytes == 0
        assert entry.basis is Basis.PHYSICAL_BACKING


class TestMemoryCountersIsDiagnosticOnly:
    def test_never_counts_toward_total(self, monkeypatch):
        # Measured: 512 MiB inside a virtual_memory.scope raises both
        # MemoryCounters.gpu and torch_reserved by 512 MiB while the device
        # consumes 512 MiB once. Summing them doubles the figure.
        provider = MemoryCountersProvider()
        monkeypatch.setattr(provider, "available", lambda: True)
        fake = types.SimpleNamespace(
            instance=staticmethod(lambda: types.SimpleNamespace(gpu=512 * MB))
        )
        module = types.ModuleType("tensorrt_llm.bindings")
        module.MemoryCounters = fake
        monkeypatch.setitem(sys.modules, "tensorrt_llm.bindings", module)

        entry = provider.collect()[0]
        assert entry.counts_toward_total is False
        assert entry.source is Source.LEDGER
        assert entry.category is Category.CPP_BUFFERS


class TestCollectAllIsFaultTolerant:
    def test_failing_provider_does_not_break_collection(self):
        from tensorrt_llm._torch.memory.ledger.providers import collect_all
        from tensorrt_llm._torch.memory.ledger.schema import Entry, Placement

        class _Boom:
            name = "boom"
            category = Category.COMM_MNNVL

            def available(self):
                return True

            def collect(self):
                raise RuntimeError("provider exploded")

        class _Fine:
            name = "fine"
            category = Category.COMM_DEEPEP

            def available(self):
                return True

            def collect(self):
                return [
                    Entry(
                        category=Category.COMM_DEEPEP,
                        size_bytes=MB,
                        source=Source.LEDGER,
                        placement=Placement.NON_TORCH,
                        basis=Basis.REQUESTED,
                    )
                ]

        entries = collect_all([_Boom(), _Fine()])
        assert len(entries) == 1
        assert entries[0].size_bytes == MB

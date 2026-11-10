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
"""Provider tests for the GPU memory monitor.

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
from memory_monitor_test_utils import MB

from tensorrt_llm._torch.memory.monitor.providers.misc import (
    DwdpPagePoolProvider,
    MemoryCountersProvider,
)
from tensorrt_llm._torch.memory.monitor.providers.mnnvl import MnnvlProvider
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Placement, Source

pytestmark = pytest.mark.cpu_only


class _State:
    """Stand-in for ``MnnvlAllocationState``."""

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
    module.MnnvlAllocationState = _State
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed.mnnvl_memory", module)
    return created


def _by_basis(entries, basis):
    return [e for e in entries if e.basis is basis]


def test_mnnvl_counts_every_state_that_holds_backing(monkeypatch):
    def collect(records):
        _install_fake_mnnvl(monkeypatch, records)
        return MnnvlProvider().collect()

    # RESTORING: handles are created after the state is set and before
    # the frontend protocol completes.
    # PREPARING: the unmap has not happened yet.
    for state in (_State.MAPPED, _State.RESTORING, _State.PREPARING):
        entries = _by_basis(
            collect({0: _Record(state, aligned_size=8 * MB)}), Basis.PHYSICAL_BACKING
        )
        resident = [e for e in entries if e.size_bytes is not None]
        assert resident, f"state {state} reported nothing"
        assert resident[0].size_bytes == 8 * MB, (
            f"state {state} must count toward resident backing; reporting 0 "
            "here hides memory during a failed restore or an OOM"
        )

    entries = _by_basis(
        collect({0: _Record(_State.UNMAPPED, aligned_size=8 * MB)}), Basis.PHYSICAL_BACKING
    )
    assert entries[0].size_bytes == 0, "UNMAPPED is the only released state"

    # A teardown that raised. Whether the handles survived is unknown,
    # and asserting zero would claim knowledge we do not have.
    unknown = [
        e for e in collect({0: _Record(_State.BROKEN, aligned_size=8 * MB)}) if e.size_bytes is None
    ]
    assert unknown, "BROKEN records must surface as unknown"
    assert "BROKEN" in (unknown[0].note or "")

    entries = _by_basis(
        collect(
            {
                0: _Record(_State.MAPPED, aligned_size=4 * MB),
                1: _Record(_State.RESTORING, aligned_size=2 * MB),
                2: _Record(_State.UNMAPPED, aligned_size=16 * MB),
            }
        ),
        Basis.PHYSICAL_BACKING,
    )
    assert [e for e in entries if e.size_bytes is not None][0].size_bytes == 6 * MB, "mixed states"


def test_mnnvl_reads_local_backing_and_address_space_without_double_counting(monkeypatch):
    def collect(base_map, subclass_map=None):
        _install_fake_mnnvl(monkeypatch, base_map, subclass_map)
        return MnnvlProvider().collect()

    # One local cuMemCreate of aligned_size; peers' handles are imported
    # and cost no local memory.
    entries = collect({0: _Record(_State.MAPPED, aligned_size=100 * MB, comm_size=4)})
    resident = _by_basis(entries, Basis.PHYSICAL_BACKING)[0]
    assert resident.size_bytes == 100 * MB, "local bytes are not multiplied by peers"

    entries = collect(
        {0: _Record(_State.MAPPED, aligned_size=MB, comm_size=4, rank_stride=64 * MB)}
    )
    va = _by_basis(entries, Basis.LOGICAL)
    assert va and va[0].counts_toward_total is False, "VA is reported but never summed"

    # Several records share one address reservation, refcounted by
    # start_address; summing per record would double count.
    shared = dict(comm_size=4, rank_stride=64 * MB, start_address=0xABC000)
    entries = collect(
        {
            0: _Record(_State.MAPPED, aligned_size=MB, **shared),
            1: _Record(_State.MAPPED, aligned_size=MB, **shared),
        }
    )
    assert _by_basis(entries, Basis.LOGICAL)[0].size_bytes == 4 * 64 * MB, "shared VA window"

    # __init_subclass__ gives each subclass its own allocated_map;
    # walking only the base silently misses CftMnnvlMemory and
    # HelixCpMnnvlMemory.
    entries = collect(
        base_map={0: _Record(_State.MAPPED, aligned_size=MB)},
        subclass_map={0: _Record(_State.MAPPED, aligned_size=2 * MB)},
    )
    assert _by_basis(entries, Basis.PHYSICAL_BACKING)[0].size_bytes == 3 * MB, "subclass maps"

    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed.mnnvl_memory", None)
    provider = MnnvlProvider()
    assert provider.available() is False and provider.collect() == [], "module missing"


class _DwdpPool:
    page_size = 16 * MB
    num_slots = 2

    def num_pages(self, slot):
        return 4  # configured; unchanged by release()

    def held_pages(self, slot):
        return 0  # everything released


def test_dwdp_page_pool_reads_held_pages_and_lists_failed_releases_apart(monkeypatch):
    """``num_pages()`` keeps reporting capacity after the pages are gone."""
    import tensorrt_llm._torch.modules.dwdp.page_pool as page_pool

    monkeypatch.setattr(page_pool, "_unreleased_bytes", 0)
    pool = _DwdpPool()
    provider = DwdpPagePoolProvider()
    monkeypatch.setattr(provider, "_pool", lambda: pool)
    (entry,) = provider.collect()
    # 8 configured pages * 16 MiB = 128 MiB would be invented out of nothing.
    assert entry.size_bytes == 0
    assert entry.basis is Basis.PHYSICAL_BACKING

    monkeypatch.setattr(page_pool, "_unreleased_bytes", 3 * MB)
    held, failed = provider.collect()
    assert failed.size_bytes == 3 * MB and failed.counts_toward_total is False, (
        "pages that failed to release may or may not be resident, so they are not summed"
    )


def test_dwdp_local_weight_shards_are_weights(monkeypatch):
    """Each rank's own DWDP shards live in fabric handles outside PyTorch."""
    from tensorrt_llm._torch.memory.monitor.providers import misc
    from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager

    provider = misc.DwdpLocalWeightsProvider()
    monkeypatch.setattr(misc, "_dwdp_manager", lambda: None)
    monkeypatch.setattr(dense_manager, "get_global_dense_dwdp_manager", lambda: None)
    assert not provider.available() and provider.collect() == [], "no DWDP, no row"

    experts = types.SimpleNamespace(
        weight_manager=types.SimpleNamespace(local_weight_bytes=96 * MB)
    )
    dense = types.SimpleNamespace(dense=types.SimpleNamespace(local_store_bytes=32 * MB))
    monkeypatch.setattr(misc, "_dwdp_manager", lambda: experts)
    monkeypatch.setattr(dense_manager, "get_global_dense_dwdp_manager", lambda: dense)
    entries = {e.owner: e for e in provider.collect()}
    assert entries["DWDP expert shards"].size_bytes == 96 * MB
    assert entries["DWDP dense shards"].size_bytes == 32 * MB
    for entry in entries.values():
        assert entry.category is Category.WEIGHTS and entry.placement is Placement.NON_TORCH
        assert entry.basis is Basis.PHYSICAL_BACKING and entry.counts_toward_total


def test_dwdp_local_bytes_exclude_imported_peer_handles():
    from tensorrt_llm._torch.modules.dwdp.transport import DWDPTransport

    transport = object.__new__(DWDPTransport)
    transport._released = False
    # Only this rank's chunks are in the handle set; imported handles are not.
    transport._handle_set = types.SimpleNamespace(sizes={(0, "w1"): 64 * MB, (1, "w1"): 32 * MB})
    transport._imported_handles = [11, 12, 13]
    assert transport.local_bytes == 96 * MB
    transport._released = True
    assert transport.local_bytes == 0, "released"


class TestMemoryCountersIsDiagnosticOnly:
    def test_never_counts_toward_total(self, monkeypatch):
        # An allocation inside a virtual_memory.scope raises both
        # MemoryCounters.gpu and torch_reserved by its size while the device
        # consumes it once, so summing the two would double it.
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
        assert entry.source is Source.COMPONENT
        assert entry.category is Category.CPP_BUFFERS


class TestCollectAllIsFaultTolerant:
    def test_failing_provider_does_not_break_collection(self):
        from tensorrt_llm._torch.memory.monitor.providers import collect_all
        from tensorrt_llm._torch.memory.monitor.schema import Entry, Placement

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
                        source=Source.COMPONENT,
                        placement=Placement.NON_TORCH,
                        basis=Basis.REQUESTED,
                    )
                ]

        entries = collect_all([_Boom(), _Fine()])
        fine, failed = sorted(entries, key=lambda e: e.size_bytes is None)
        assert fine.size_bytes == MB, "the other providers still report"
        # The failure is listed as unmeasured under the provider's name, not dropped.
        assert failed.size_bytes is None and failed.owner == "boom"
        assert failed.category is Category.COMM_MNNVL and "provider exploded" in failed.note


def test_registrations_belong_to_their_executor():
    from tensorrt_llm._torch.memory.monitor import pools
    from tensorrt_llm._torch.memory.monitor import providers as reg

    class _Kv:
        name = "kv_cache.main"
        category = Category.KV_CACHE_POOL

        def __init__(self, size):
            self.size = size

    def kv_sizes():
        return sorted(p.size for p in reg.default_providers() if isinstance(p, _Kv))

    reg.clear_registered_providers()
    pools.forget_released_tags()
    try:
        first = reg.begin_executor()
        reg.register_provider(_Kv(10))
        reg.register_provider(_Kv(20))  # rebuilt by the same executor: replaces
        assert kv_sizes() == [20]

        second = reg.begin_executor()
        reg.register_provider(_Kv(30))  # another executor's KV cache is kept
        assert kv_sizes() == [20, 30]

        pools.mark_released(["kv_cache"])
        reg.end_executor(first)
        assert kv_sizes() == [30]
        assert pools.is_tag_released("kv_cache"), "the second executor still sleeps"

        reg.end_executor(second)
        assert kv_sizes() == []
        # Nothing left that released memory: a new executor starts awake.
        assert not pools.is_tag_released("kv_cache")
    finally:
        reg.clear_registered_providers()
        pools.forget_released_tags()

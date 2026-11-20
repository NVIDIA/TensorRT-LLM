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
"""Tests for MembershipMnnvlMemory.remap and HostExchangeMnnvlMemory against a fake CUDA driver."""

import gc
import itertools
import pickle
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.mnnvl_memory as mnnvl
from tensorrt_llm._torch.distributed.membership import StoreGroup
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory
from tensorrt_llm._torch.distributed.mnnvl_host_exchange import HostExchangeMnnvlMemory
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMappingError, MnnvlPeerHandleError
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FABRIC = "fabric"
_SIZE = 1 << 21
_STRIDE = mnnvl.MnnvlMemory.fabric_page_size
_BASE = 1 << 40


class _FakeDriver:
    """Fake the CUDA driver calls of MnnvlMemory and record the handle mapped at each address."""

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_FABRIC=_FABRIC)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    def __init__(self):
        self._ids = itertools.count(100)
        self._bases = itertools.count(1)
        self.mapped = {}  # address -> handle
        self.released = []
        self.reserved = []  # (base, size)
        self.fail_create = False
        self.fail_import = False

    CUmemAccessDesc = staticmethod(SimpleNamespace)
    CUdeviceptr = staticmethod(int)

    def cuCtxGetDevice(self):
        return 0, 0

    def cuMemAddressReserve(self, size, alignment, addr, flags):
        base = next(self._bases) << 44
        self.reserved.append((base, size))
        return 0, base

    def cuMemAddressFree(self, address, size):
        self.reserved.remove((address, size))
        return 0

    def cuMemCreate(self, size, prop, flags):
        if self.fail_create:
            return 2, None
        return 0, ("local", next(self._ids))

    def cuMemExportToShareableHandle(self, handle, handle_type, flags):
        return 0, SimpleNamespace(data=("export", handle))

    def cuMemImportFromShareableHandle(self, data, handle_type):
        if self.fail_import:
            return 2, None
        return 0, ("import", data)

    def cuMemMap(self, address, size, offset, handle, flags):
        assert address not in self.mapped, "address already mapped"
        self.mapped[address] = handle
        return 0

    def cuMemSetAccess(self, address, size, descs, count):
        return 0

    def cuMemUnmap(self, address, size):
        if self.mapped.pop(address, None) is None:
            return 1
        return 0

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return 0

    def cuMemsetD8(self, address, value, size):
        assert address in self.mapped, "memset of an unmapped row"
        return 0


class _RemapMemory(MembershipMnnvlMemory):
    pass


@pytest.fixture
def driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(mnnvl, "cuda", driver)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialized", True)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(lambda cls, dev_id: SimpleNamespace(requestedHandleTypes=_FABRIC, location=0)),
    )
    return driver


@pytest.fixture
def memory(driver):
    """Return rank 1's view of a 4-rank allocation with every row mapped."""
    memory = object.__new__(_RemapMemory)
    memory.ptr, memory.segment_size, memory.rank_stride = _BASE, _SIZE, _STRIDE
    handles = [("old", rank) for rank in range(4)]
    for rank, handle in enumerate(handles):
        driver.mapped[_BASE + rank * _STRIDE] = handle
    _RemapMemory.allocated_map[_BASE] = mnnvl._MnnvlAllocationRecord(
        comm=None,
        comm_size=4,
        comm_rank=1,
        comm_membership=tuple(range(4)),
        aligned_size=_SIZE,
        mem_handles=list(handles),
        start_address=_BASE,
        rank_stride=_STRIDE,
        address_offset=0,
    )
    yield memory
    # The fake allocation owns no address range, so keep it from being closed when collected.
    del memory.ptr
    _RemapMemory.allocated_map.clear()


class _Comm:
    """A ``StoreGroup``-like communicator of ``membership`` that records what it sends."""

    def __init__(self, gather, index=1, membership=range(4)):
        self.membership = tuple(membership)
        self.exchange_failed = False
        self.calls = []
        self._gather = gather
        self._index = index

    def Get_rank(self):
        return self._index

    def Get_size(self):
        return len(self.membership)

    def allgather(self, value):
        self.calls.append(value)
        self.exchange_failed = False
        try:
            return self._gather(value)
        except BaseException:
            self.exchange_failed = True
            raise


def _unreachable(value):
    raise AssertionError("exchanged over the wrong communicator")


def _gather_without(absent, size=4, index=1):
    return _Comm(lambda value: [None if rank in absent else value for rank in range(size)], index)


def _rows(*ranks, base=_BASE):
    return [base + rank * _STRIDE for rank in ranks]


_ALL, _SURVIVORS, _OWN_ROW_ONLY = frozenset(range(4)), frozenset({0, 1, 3}), frozenset({2})


def test_remap_needs_the_allocation_members(memory, driver):
    """Check that a remap over a communicator of other ranks maps nothing."""
    with pytest.raises(RuntimeError, match="cannot remap it as rank 1 of ranks"):
        memory.remap(_Comm(_unreachable, membership=(1, 0, 2, 3)), _ALL)
    assert sorted(driver.mapped) == _rows(0, 1, 2, 3)


def test_remap_maps_active_rows_in_place(memory, driver):
    memory.remap(_gather_without({2}), _ALL)

    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))
    assert memory.mapped
    assert sorted(driver.mapped) == _rows(0, 1, 3)
    assert driver.mapped[_BASE + _STRIDE][0] == "local"
    assert all(handle[0] != "old" for handle in driver.mapped.values())


@pytest.mark.parametrize("fresh_pages", [False, True])
def test_replacement_maps_own_row_until_admission(driver, monkeypatch, fresh_pages):
    """Check that a replacement maps only its own row until its admission maps every active row."""
    monkeypatch.setattr(
        _RemapMemory, "get_allocation_granularity", classmethod(lambda cls, dev_id: _SIZE)
    )
    group = StoreGroup(None, "workspace", 2, range(4), active=[2])
    monkeypatch.setattr(_RemapMemory, "get_comm", classmethod(lambda cls, mapping: group))
    memory = _RemapMemory(Mapping(world_size=4, rank=2, tp_size=4, moe_ep_size=4), _SIZE)
    try:
        rows = [memory.ptr + rank * memory.rank_stride for rank in range(4)]
        assert list(driver.mapped) == [rows[2]]
        record = _RemapMemory.allocated_map[memory.ptr]
        assert (record.comm_size, record.comm_rank, record.comm_membership) == (4, 2, (0, 1, 2, 3))
        own = driver.mapped[rows[2]]
        memory.remap(_gather_without(set(), index=2), _ALL if fresh_pages else frozenset())
        assert (driver.mapped[rows[2]] is own) is not fresh_pages
        assert sorted(driver.mapped) == rows
    finally:
        memory.checkpoint_prepare()
        _RemapMemory.allocated_map.clear()
        _RemapMemory.address_refcnt.clear()
        del memory.ptr


def test_release_skips_unmapped_absent_rows(memory, driver):
    memory.remap(_gather_without({2}), _ALL)
    memory.remap(_gather_without({0, 2}), _ALL)
    assert sorted(driver.mapped) == _rows(1, 3)


def test_absent_rows_on_scratch_until_admission(memory, driver, monkeypatch):
    """Check that absent rows share a scratch allocation that an admission replaces."""
    resets = []
    monkeypatch.setattr(_RemapMemory, "absent_rows_on_scratch", True)
    monkeypatch.setattr(
        _RemapMemory, "_reset_own_row", lambda self: resets.append(driver.mapped[_BASE + _STRIDE])
    )
    memory.remap(_gather_without({2, 3}), _ALL)
    own, scratch = driver.mapped[_BASE + _STRIDE], driver.mapped[_BASE + 2 * _STRIDE]
    assert driver.mapped[_BASE + 3 * _STRIDE] is scratch
    assert scratch[0] == "local"
    assert scratch != own
    assert resets == [own]
    assert memory.rows_needing_fresh_pages(_gather_without({2, 3})) == frozenset()

    kept = [driver.mapped[address] for address in _rows(0, 1)]
    driver.released.clear()
    memory.remap(_gather_without({3}), frozenset())
    assert [driver.mapped[address] for address in _rows(0, 1)] == kept
    assert driver.mapped[_BASE + 2 * _STRIDE][0] == "import"
    assert driver.mapped[_BASE + 3 * _STRIDE][0] == "local"
    assert driver.mapped[_BASE + 3 * _STRIDE] not in (scratch, own)
    assert driver.released == [scratch]
    assert resets == [own]


def test_failed_remap_is_retryable(memory, driver):
    driver.fail_create = True
    with pytest.raises(RuntimeError, match="handle export failed"):
        memory.remap(_gather_without({2}), _ALL)
    assert driver.mapped == {}
    assert not memory.mapped
    driver.fail_create = False
    memory.remap(_gather_without({2}), _ALL)
    assert memory.mapped


def test_admission_maps_only_joining_row(memory, driver):
    """Check that an admission creates and releases nothing and imports only the joining row."""
    memory.remap(_gather_without({2}), _ALL)
    kept = dict(driver.mapped)
    driver.released.clear()
    driver.fail_create = True
    memory.remap(_gather_without(set()), frozenset())
    assert sorted(driver.mapped) == _rows(0, 1, 2, 3)
    assert all(driver.mapped[address] is handle for address, handle in kept.items())
    (own,) = _rows(1)
    assert driver.mapped[_BASE + 2 * _STRIDE] == ("import", ("export", kept[own]))
    assert driver.released == []
    assert memory.mapped


def test_admission_refuses_row_of_departed_rank(memory, driver):
    """Check that an admission refuses to keep the pages of rank 3, which is no longer active."""
    with pytest.raises(RuntimeError, match="Rank 3 left while this rank maps its pages"):
        memory.remap(_gather_without({3}), frozenset())
    assert driver.mapped == {}
    assert not memory.mapped
    memory.remap(_gather_without({3}), _ALL)
    assert sorted(driver.mapped) == _rows(0, 1, 2)


@pytest.mark.parametrize(
    "absent, peers, fresh_pages",
    [
        # Every process that maps the pages is active, so there is nothing to remap.
        (set(), {0: _ALL, 2: _ALL, 3: _ALL}, False),
        # Rank 3 failed, and its process may still write into the pages.
        (set(), {0: _ALL, 2: _ALL}, True),
        # Rank 2's process was replaced before the survivors remapped. Its new process maps only its
        # own row, and its old one may still write into the pages the others map.
        (set(), {0: _ALL, 2: _OWN_ROW_ONLY, 3: _ALL}, True),
        # Rank 2's replacement joins the survivors, which alone map their pages since rank 2 failed.
        ({2}, {0: _SURVIVORS, 2: _OWN_ROW_ONLY, 3: _SURVIVORS}, False),
        # An earlier admission left the pages unmapped on rank 3.
        ({2}, {0: _SURVIVORS, 2: _OWN_ROW_ONLY, 3: frozenset()}, True),
    ],
    ids=["unchanged", "failover", "replaced-holder", "admission", "unmapped-on-a-peer"],
)
def test_fresh_pages_if_former_member_maps(memory, driver, absent, peers, fresh_pages):
    """Check that fresh pages are needed only when a process that maps them may not be active."""
    if absent:
        memory.remap(_gather_without(absent), _ALL)
    comm = _Comm(lambda value: [value if rank == 1 else peers.get(rank) for rank in range(4)])
    assert memory.rows_needing_fresh_pages(comm) == (_ALL if fresh_pages else frozenset())
    assert comm.calls == [_ALL - absent]


class _PeerFailure(Exception):
    pass


def _gather_outcomes(outcomes, absent, size=4):
    """Return a gather of handles without ``absent``, then of ``outcomes`` or raising them."""

    def gather(value):
        if len(comm.calls) == 1:
            return [None if rank in absent else value for rank in range(size)]
        if isinstance(outcomes, Exception):
            raise outcomes
        return [value if rank == 1 else outcomes.get(rank) for rank in range(size)]

    comm = _Comm(gather)
    return comm


@pytest.mark.parametrize("admission", [False, True], ids=["failover", "admission"])
def test_peer_mapping_failure_unmaps_this_rank(memory, driver, admission):
    """Check that a peer's failed mapping makes every active rank raise and release its pages."""
    absent = set() if admission else {2}
    if not admission:
        memory.checkpoint_prepare()
        driver.released.clear()
    with pytest.raises(MnnvlMappingError, match="rank 3: RuntimeError") as raised:
        memory.remap(
            _gather_outcomes({3: ("RuntimeError: map failed", [])}, absent=absent),
            frozenset() if admission else _ALL,
        )
    # The error names the failed rank, which alone reports the failure in a membership change.
    assert raised.value.ranks == (3,)
    assert driver.mapped == {}
    assert not memory.mapped
    if admission:
        assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))
        assert memory.rows_needing_fresh_pages(_gather_without(absent)) == _ALL
        with pytest.raises(RuntimeError, match="row of an unmapped MNNVL allocation needs fresh"):
            memory.remap(_gather_without(absent), frozenset())
    else:
        # The fresh local handle and the imported ones were released.
        assert len(driver.released) == 3
    memory.remap(_gather_without(absent), _ALL)
    assert sorted(driver.mapped) == _rows(*(rank for rank in range(4) if rank not in absent))


def test_failed_import_reported_to_peers(memory, driver):
    """Check global rank 5, communicator rank 1 of ranks 4 to 7, when it cannot import a handle.

    It fails on the first peer it imports, global rank 4. The failure and the CUDA result reach
    every active rank by global rank.
    """
    _RemapMemory.allocated_map[_BASE].comm_membership = (4, 5, 6, 7)
    memory.checkpoint_prepare()
    driver.fail_import = True
    reported = []

    def gather(value):
        reported.append(value)
        return [value if rank in (4, 5, 7) else None for rank in range(4, 8)]

    with pytest.raises(MnnvlPeerHandleError, match="rank 5: MnnvlPeerHandleError") as raised:
        memory.remap(_Comm(gather, membership=range(4, 8)), frozenset(range(4, 8)))
    assert reported[-1][1] == [4]
    assert raised.value.ranks == (4,)
    assert reported[-1][0] == "MnnvlPeerHandleError: 2"
    assert driver.mapped == {}
    assert not memory.mapped


def test_failed_export_reported_by_global_rank(memory, driver):
    """Check that global rank 5 that cannot create its pages is named by global rank."""
    _RemapMemory.allocated_map[_BASE].comm_membership = (4, 5, 6, 7)
    memory.checkpoint_prepare()
    driver.fail_create = True

    def gather(value):
        if len(comm.calls) == 1:
            return [value if index == 1 else {**value, "error": None} for index in range(4)]
        return [value if index == 1 else () for index in range(4)]

    comm = _Comm(gather, membership=range(4, 8))
    with pytest.raises(MnnvlMappingError, match="rank 5: MnnvlMappingError") as raised:
        memory.remap(comm, frozenset(range(4, 8)))
    assert raised.value.ranks == (5,)
    assert "before mapping:\nrank 5: RuntimeError" in comm.calls[-1][0]
    assert driver.mapped == {}
    assert not memory.mapped


@pytest.fixture(params=["remap", "new_rows"])
def map_rows(request, driver):
    """Return a call that remaps or creates rank 1's rows over a comm, and whether any stay mapped."""
    if request.param == "remap":
        memory = request.getfixturevalue("memory")
        memory.checkpoint_prepare()
        return lambda comm: memory.remap(comm, _ALL), lambda: memory.mapped
    request.getfixturevalue("rows_driver")
    return (
        lambda comm: HostExchangeMnnvlMemory(comm, _SIZE),
        lambda: bool(
            HostExchangeMnnvlMemory.allocated_map or HostExchangeMnnvlMemory.address_refcnt
        ),
    )


def test_only_failed_exporter_is_named(map_rows, driver):
    """Check that every active rank names only rank 3 when rank 3 could not create its pages."""
    map_with, mapped = map_rows

    def gather(value):
        if len(comm.calls) == 1:
            failed = {**value, "error": "RuntimeError: 2", "handle": None}
            return [None if rank == 2 else failed if rank == 3 else value for rank in range(4)]
        theirs = ("MnnvlMappingError: rank 3: RuntimeError: 2", [])
        return [None if rank == 2 else theirs if rank == 3 else value for rank in range(4)]

    comm = _Comm(gather)
    with pytest.raises(MnnvlMappingError, match="rank 3: MnnvlMappingError") as raised:
        map_with(comm)
    assert raised.value.ranks == (3,)
    assert comm.calls[-1] == ()
    assert driver.mapped == {}
    assert driver.reserved == []
    assert not mapped()


def test_peer_import_failure_fails_every_rank(map_rows, driver):
    """Check that rank 1, which mapped every row, raises when rank 0 could not import a handle."""
    map_with, mapped = map_rows

    def gather(value):
        if len(comm.calls) == 1:
            return [value if rank in (0, 1, 3) else None for rank in range(4)]
        failed = ("MnnvlPeerHandleError: rank 3's handle", [3])
        return [failed if rank == 0 else value if rank in (1, 3) else None for rank in range(4)]

    comm = _Comm(gather)
    with pytest.raises(MnnvlPeerHandleError, match="rank 0: MnnvlPeerHandleError") as raised:
        map_with(comm)
    assert comm.calls[-1] == ()
    assert raised.value.ranks == (3,)
    assert driver.mapped == {}
    assert driver.reserved == []
    assert not mapped()


def test_bug_in_mapping_skips_outcome_exchange(memory, driver, monkeypatch):
    """Check that an error other than RuntimeError raises at once, leaving nothing mapped."""
    memory.checkpoint_prepare()

    def failing_map(address, size, offset, handle, flags):
        raise ValueError("bad mapping")

    monkeypatch.setattr(driver, "cuMemMap", failing_map)
    comm = _gather_without({2})
    with pytest.raises(ValueError, match="bad mapping"):
        memory.remap(comm, _ALL)
    assert len(comm.calls) == 1
    assert driver.mapped == {}
    assert not memory.mapped


@pytest.mark.parametrize("failed_exchange, released", [(1, 1), (2, 3)], ids=["handles", "outcomes"])
def test_failed_exchange_leaves_unmapped(memory, driver, failed_exchange, released):
    """Check that a failed exchange releases the new pages and skips any later exchange."""
    memory.checkpoint_prepare()
    driver.released.clear()

    def gather(value):
        if len(comm.calls) == failed_exchange:
            raise _PeerFailure()
        return [None if rank == 2 else value for rank in range(4)]

    comm = _Comm(gather)
    with pytest.raises(_PeerFailure):
        memory.remap(comm, _ALL)
    assert len(comm.calls) == failed_exchange
    assert driver.mapped == {}
    assert not memory.mapped
    assert len(driver.released) == released
    memory.remap(_gather_without({2}), _ALL)
    assert memory.mapped


def test_failed_cleanup_breaks_allocation(memory, driver, monkeypatch):
    """Check that the remap's error reaches the caller when releasing its pages also fails."""
    memory.checkpoint_prepare()
    errors = []
    monkeypatch.setattr(mnnvl.logger, "error", errors.append)
    monkeypatch.setattr(driver, "cuMemUnmap", lambda address, size: 2)

    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}), _ALL)

    assert len(errors) == 1
    assert "failed MNNVL remap" in errors[0]
    assert _RemapMemory.allocated_map[_BASE].state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="Cannot remap MNNVL allocation in broken state"):
        memory.remap(_gather_without({2}), _ALL)


@pytest.fixture
def rows_driver(driver, monkeypatch):
    monkeypatch.setattr(HostExchangeMnnvlMemory, "allocation_granularity", _SIZE)
    yield driver
    HostExchangeMnnvlMemory.allocated_map.clear()
    HostExchangeMnnvlMemory.address_refcnt.clear()


def test_absent_rows_share_one_scratch_allocation(rows_driver):
    """Check that rows map back to back, with absent ranks 2 and 3 on one scratch allocation."""
    comm = _gather_without({2, 3})
    rows = HostExchangeMnnvlMemory(comm, _SIZE - 100)

    assert len(comm.calls) == 2
    assert comm.calls[1] == ()
    assert rows.stride == _SIZE
    assert rows.row(3) == rows.ptr + 3 * _SIZE
    assert rows_driver.reserved == [(rows.ptr, 4 * _SIZE)]
    handles = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
    assert [handle[0] for handle in handles] == ["import", "local", "local", "local"]
    assert handles[2] is handles[3]
    assert handles[2] != handles[1]

    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    assert sorted(rows_driver.released) == sorted(handles[:3])
    assert HostExchangeMnnvlMemory.allocated_map == {}


def test_membership_change_remaps_rows_in_place(rows_driver, monkeypatch):
    """Check that rows move to fresh pages in place, zeroed before the outcome exchange."""
    events = []
    monkeypatch.setattr(
        rows_driver,
        "cuMemsetD8",
        lambda address, value, size: events.append(("zero", address)) or 0,
    )

    def gather_among(active):
        def gather(value):
            events.append(("outcome" if value == () else "handle", value))
            return [value if rank in active else None for rank in range(4)]

        return _Comm(gather)

    rows = HostExchangeMnnvlMemory(gather_among({0, 1, 2, 3}), _SIZE)
    ptr = rows.ptr
    for active in ({0, 1, 2}, {0, 1, 2, 3}):
        before = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
        events.clear()
        rows.remap(gather_among(active), _ALL)
        after = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]

        assert rows.ptr == ptr
        assert rows_driver.reserved == [(ptr, 4 * _SIZE)]
        assert [event for event, _ in events] == ["handle", "zero", "outcome"]
        assert events[1] == ("zero", rows.row(1))
        assert all(handle in rows_driver.released for handle in before)
        assert not set(after) & set(before)
        assert after[3][0] == ("import" if 3 in active else "local")
    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []


@pytest.mark.parametrize("failure", ["import", "memset"])
def test_failed_rows_leave_nothing_mapped(rows_driver, monkeypatch, failure):
    """Check that failed rows leave nothing mapped and still join the outcome exchange."""
    comm = _gather_without({2})
    if failure == "import":
        imports = []

        def import_handle(data, handle_type):
            imports.append(data)
            return (2, None) if len(imports) == 2 else (0, ("import", data))

        monkeypatch.setattr(rows_driver, "cuMemImportFromShareableHandle", import_handle)
        expected = pytest.raises(MnnvlPeerHandleError, match="rank 1: MnnvlPeerHandleError: 2")
    else:
        monkeypatch.setattr(rows_driver, "cuMemsetD8", lambda address, value, size: 2)
        expected = pytest.raises(RuntimeError)

    with expected as raised:
        HostExchangeMnnvlMemory(comm, _SIZE)
    if failure == "import":
        assert raised.value.ranks == (3,)

    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    # This rank created its own row and the scratch row of rank 2, and imported one or two rows.
    released = rows_driver.released
    assert len(released) == 2 + (1 if failure == "import" else 2)
    assert len({id(handle) for handle in released}) == len(released)
    assert HostExchangeMnnvlMemory.allocated_map == {}
    assert HostExchangeMnnvlMemory.address_refcnt == {}
    assert len(comm.calls) == 2
    assert comm.calls[1] is not None


def test_collecting_failed_rows_keeps_retry_open(rows_driver, monkeypatch):
    """Check that collecting failed rows leaves a retry at the same address mapped and open."""
    base = 1 << 44

    def reserve(size, alignment, addr, flags):
        rows_driver.reserved.append((base, size))
        return 0, base

    monkeypatch.setattr(rows_driver, "cuMemAddressReserve", reserve)
    comm = _gather_without({2, 3})
    rows_driver.fail_import = True
    try:
        HostExchangeMnnvlMemory(comm, _SIZE)
    except MnnvlPeerHandleError:
        # The failed mapping released its own row's handle and the unmapped scratch handle.
        assert rows_driver.released == [("local", 100), ("local", 101)]
        # The error's traceback keeps the failed object alive during the retry.
        rows_driver.fail_import = False
        rows = HostExchangeMnnvlMemory(comm, _SIZE)
    else:
        pytest.fail("HostExchangeMnnvlMemory did not raise for a handle it could not import")
    assert rows.ptr == base
    gc.collect()

    assert base in HostExchangeMnnvlMemory.allocated_map
    assert all(rows.row(rank) in rows_driver.mapped for rank in range(4))
    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []


def test_failed_scratch_release_not_repeated(rows_driver, monkeypatch):
    """Check that closing rows releases a shared scratch allocation once even when that fails."""
    rows = HostExchangeMnnvlMemory(_gather_without({2, 3}), _SIZE)
    scratch = rows_driver.mapped[rows.row(2)]
    attempts = []

    def release(handle):
        attempts.append(handle)
        return 2 if handle is scratch else 0

    monkeypatch.setattr(rows_driver, "cuMemRelease", release)
    with pytest.raises(RuntimeError):
        rows.close()

    assert rows_driver.mapped == {}
    assert len(attempts) == 3
    assert sum(handle is scratch for handle in attempts) == 1
    # Neither a second close nor collection releases the rows again.
    rows.close()
    del rows
    gc.collect()
    assert len(attempts) == 3


def test_rows_require_fabric_handles(rows_driver, monkeypatch):
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(
            lambda cls, dev_id: SimpleNamespace(requestedHandleTypes="posix_fd", location=0)
        ),
    )
    comm = _gather_without({2, 3})
    with pytest.raises(RuntimeError, match="requires fabric memory handles"):
        HostExchangeMnnvlMemory(comm, _SIZE)
    assert comm.calls == []
    assert rows_driver.reserved == []


def test_failed_communicator_query_reserves_nothing(rows_driver, monkeypatch):
    """Check that rows query the communicator before they reserve an address range."""
    comm = _gather_without({2, 3})

    def get_rank():
        raise RuntimeError("communicator closed")

    monkeypatch.setattr(comm, "Get_rank", get_rank)
    with pytest.raises(RuntimeError, match="communicator closed"):
        HostExchangeMnnvlMemory(comm, _SIZE)
    assert rows_driver.reserved == []


@pytest.mark.parametrize(
    "error",
    [MnnvlMappingError([3, 1], "rank 3: export failed"), MnnvlPeerHandleError([5], "rank 5")],
    ids=["mapping", "peer_handle"],
)
def test_mapping_errors_survive_pickling(error):
    """Check that a mapping error reaches a process pool's caller with its ranks and message."""
    copy = pickle.loads(pickle.dumps(error))  # nosec B301
    assert type(copy) is type(error)
    assert copy.ranks == error.ranks
    assert str(copy) == str(error)

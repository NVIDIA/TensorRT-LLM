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
"""Tests that compare MnnvlDist's host collectives over HostExchange with MPIDist's."""

import contextlib
import math
import pickle
import sys
import threading
import time
import traceback

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_no_mnnvl_fabric_handles

import tensorrt_llm
from tensorrt_llm._torch.distributed.communicator import MPIDist, ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import HostExchange, MnnvlDist
from tensorrt_llm._torch.distributed.membership import latest_epoch, next_epoch
from tensorrt_llm._torch.distributed.mnnvl_host_exchange import (
    HostExchangeMnnvlMemory,
    MnnvlHostExchangeGroup,
)
from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)


def _in_mask(mask, rank):
    return bool(mask & (1 << rank))


class _MpiGroup:
    """The MPI world as a membership change's communicator, which has collectives only.

    A rank that left takes part in its gathers with None, as an inactive member's value is None in
    a ``StoreGroup``. The MNNVL handle exchange uses point-to-point requests on a communicator that
    has them, which a rank that left would not match.
    """

    def __init__(self):
        self._comm = mpi_comm()

    def Get_rank(self):
        return self._comm.Get_rank()

    def Get_size(self):
        return self._comm.Get_size()

    def allgather(self, value):
        return self._comm.allgather(value)


def _dists(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=tp, tp_size=tp, moe_ep_size=tp, rank=rank, enable_attention_dp=True
    )
    MPI.COMM_WORLD.barrier()
    return MnnvlDist(mapping), MPIDist(mapping), rank


def _collectives(tp):
    """Check that every host collective returns what MPIDist returns."""
    mnnvl, mpi, rank = _dists(tp)
    MnnvlDist.check_requirements(mnnvl.mapping)
    assert mnnvl.supports_single_round_broadcast
    small = {"rank": rank, "load": [rank, 2 * rank]}
    # Larger than one exchange slot, so it takes several rounds. Its length differs across ranks.
    large = bytes(range(256)) * (3000 + 500 * rank)
    for obj in (small, large, None, b""):
        assert mnnvl.allgather(obj) == mpi.allgather(obj)
        assert mnnvl.tp_allgather(obj) == mpi.tp_allgather(obj)
        assert mnnvl.tp_gather(obj, root=1) == mpi.tp_gather(obj, root=1)
        assert mnnvl.broadcast(obj, root=tp - 1) == mpi.broadcast(obj, root=tp - 1)
        assert mnnvl.broadcast_or_none(obj if rank == 0 else None) == mpi.broadcast_or_none(
            obj if rank == 0 else None
        )
    for op in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN):
        assert mnnvl.allreduce(rank + 1, op) == mpi.allreduce(rank + 1, op)
    assert mnnvl.tp_allreduce([rank], ReduceOp.SUM) == mpi.tp_allreduce([rank], ReduceOp.SUM)
    values = [rank, -rank, 2**40 + rank]
    assert np.array_equal(mnnvl.tp_allgather_int64(values), mpi.tp_allgather_int64(values))
    # Rows that arrive with their headers, that need a second poll, and that take several rounds.
    slot_values = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES // 8
    for n in (0, 1, MnnvlHostExchangeGroup._HEADER_POLL_PACKETS, slot_values, 2 * slot_values + 3):
        row = np.arange(n, dtype=np.int64) * (rank + 1) - 2**40
        assert np.array_equal(mnnvl.tp_allgather_int64(row), mpi.tp_allgather_int64(row)), n
    assert np.array_equal(
        mnnvl.broadcast_int64(values, root=1), mpi.broadcast_int64(values, root=1)
    )
    # With every rank active, a vote's gather has every row.
    assert np.array_equal(
        mnnvl.tp_allgather_int64(values, active_only=True), mpi.tp_allgather_int64(values)
    )
    mnnvl.barrier()


def _stress(tp):
    """Check many gathers of random sizes, up to three rounds each, byte for byte."""
    mnnvl, _, rank = _dists(tp)
    rng = np.random.default_rng(1234)
    slot = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
    for step in range(2000):
        # Every rank draws the same sizes.
        sizes = rng.integers(0, 3 * slot, size=tp)
        mine = np.random.default_rng(step * tp + rank).bytes(int(sizes[rank]))
        gathered = mnnvl.allgather(mine)
        for r, payload in enumerate(gathered):
            assert payload == np.random.default_rng(step * tp + r).bytes(int(sizes[r])), (step, r)
        # The int64 rows have one size per step and take up to three rounds.
        n = int(sizes[0]) // 8
        rows = mnnvl.tp_allgather_int64(
            np.random.default_rng(step * tp + rank).integers(-(2**62), 2**62, n)
        )
        for r in range(tp):
            assert np.array_equal(
                rows[r], np.random.default_rng(step * tp + r).integers(-(2**62), 2**62, n)
            ), (step, r)


def _epoch_wrap(tp):
    """Check that the epoch wraps to 1 and that rounds keep alternating slot sets across it."""
    mnnvl, _, rank = _dists(tp)
    exchange = mnnvl.host_exchange
    # Start a few exchanges before the epoch wraps to 1.
    exchange.admit(exchange.active_mask, 0xFFFFFFFE - 3)
    epochs = []
    # Consecutive single-round exchanges use alternate slot sets.
    for _ in range(8):
        mnnvl.allgather(rank)
        epochs.append(exchange.epoch)
    assert 1 in epochs, epochs
    assert all((a ^ b) & 1 for a, b in zip(epochs, epochs[1:])), epochs
    exchange.admit(exchange.active_mask, 0xFFFFFFFE - 3)
    half_slot = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES // 2
    for step in range(16):
        # Each payload takes two rounds.
        large = bytes([step, rank]) * (half_slot + 100 * rank)
        assert mnnvl.allgather(large) == [
            bytes([step, r]) * (half_slot + 100 * r) for r in range(tp)
        ]


@contextlib.contextmanager
def _woken(exchange):
    """Wake the exchange's waits every millisecond, as membership records would wake them."""
    done = threading.Event()

    def wake():
        while not done.wait(1e-3):
            exchange.wake()

    thread = threading.Thread(target=wake, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


def _install_on_interrupt(exchange, mpi, interrupted, active_mask, before=None):
    """Install ``active_mask`` when a wake finds ``interrupted``, and return each change's epochs.

    MPI stands in for the membership transaction that tells every rank the new mask and epoch.
    """
    changes = []

    def change():
        if before is not None:
            before()
        stopped = mpi.allgather(exchange.stopped_epoch)
        epoch = next_epoch(latest_epoch([e for e in stopped if e is not None]))
        exchange.install(_MpiGroup(), active_mask, epoch)
        exchange.set_interrupt_handler(lambda: False)
        changes.append((stopped, epoch))
        # Commit the change.
        mpi.barrier()

    exchange.set_interrupt_handler(interrupted, change)
    return changes


def _leave(mpi):
    """Match the survivors' gather of stopped epochs, their rebuild, and the commit."""
    mpi.allgather(None)
    _skip_rebuild(mpi)
    mpi.barrier()


def _membership(tp):
    """Let rank 1 and the last rank leave in turn, and admit each again."""
    mnnvl, mpi, rank = _dists(tp)
    exchange = mnnvl.host_exchange
    all_ranks = (1 << tp) - 1
    slot = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
    payloads = (
        lambda r: {"rank": r},
        lambda r: bytes([r]) * (100 + r),
        lambda r: bytes([r]) * (slot + 100 * r),  # several rounds
    )
    for leaver in sorted({1, tp - 1}):
        mnnvl.barrier()
        survivors = all_ranks & ~(1 << leaver)
        active = [r for r in range(tp) if _in_mask(survivors, r)]
        if rank == leaver:
            _leave(mpi)
        else:
            ptr = exchange.group.rows.ptr
            _install_on_interrupt(
                exchange,
                mpi,
                lambda leaver=leaver: _in_mask(exchange.waiting_for, leaver),
                survivors,
            )
            # The leaver never sends, so the round runs again without it.
            with _woken(exchange):
                assert mnnvl.allgather(rank) == [r if r in active else None for r in range(tp)]
            assert exchange.group.rows.ptr == ptr
            for obj_of in payloads:
                expected = [obj_of(r) if r in active else None for r in range(tp)]
                assert mnnvl.allgather(obj_of(rank)) == expected
                assert mnnvl.tp_allgather(obj_of(rank)) == expected
                assert mnnvl.tp_gather(obj_of(rank), root=0) == (expected if rank == 0 else None)
                for root in active:
                    assert mnnvl.broadcast(obj_of(rank), root=root) == expected[root]
                assert mnnvl.broadcast_or_none(obj_of(rank) if rank == 0 else None) == expected[0]
                assert mnnvl.broadcast_or_none(None) is None
            assert mnnvl.allreduce(rank + 1, ReduceOp.SUM) == sum(r + 1 for r in active)
            assert mnnvl.tp_allreduce(rank + 1, ReduceOp.MAX) == active[-1] + 1
            for n in (3, slot // 8 + 5):
                gathered = mnnvl.tp_allgather_int64(np.full(n, rank + 1))
                assert np.array_equal(
                    gathered, [np.full(n, r + 1 if r in active else 0) for r in range(tp)]
                ), n
                assert np.array_equal(
                    mnnvl.tp_allgather_int64(np.full(n, rank + 1), active_only=True),
                    [np.full(n, r + 1) for r in active],
                )
            assert np.array_equal(mnnvl.broadcast_int64([7, rank], root=0), [7, 0])
            mnnvl.barrier()

        # The rank that left still maps its old rows, so the admission maps fresh rows.
        epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
        _rebuild(exchange, _MpiGroup(), survivors, epoch)
        if rank != leaver:
            assert mnnvl.allgather(rank) == [r if r in active else None for r in range(tp)]
        exchange.admit(all_ranks, next_epoch(mpi.broadcast(exchange.epoch, root=0)))
        assert mnnvl.allgather(rank) == list(range(tp))


def _rebuild(exchange, comm, active_mask, epoch):
    """Map the exchange's rows among the active members of ``comm`` as an INSTALL does."""
    exchange.group.rows.remap(comm, exchange.group.rows.rows_needing_fresh_pages(comm))
    exchange.admit(active_mask, epoch)


def _skip_rebuild(mpi):
    """Match the gathers of fresh rows, handles, and outcomes of a rebuild as a rank that left."""
    for _ in range(3):
        mpi.allgather(None)


def _admission(tp):
    """Let the last rank leave and a replacement that maps only its own row join, keeping pages."""
    from tensorrt_llm._torch.distributed.membership import StoreGroup

    mnnvl, mpi, rank = _dists(tp)
    leaver = tp - 1
    survivors = ((1 << tp) - 1) & ~(1 << leaver)
    exchange = mnnvl.host_exchange
    mnnvl.barrier()
    epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
    if rank == leaver:
        _skip_rebuild(mpi)
        exchange.group.rows.close()
        # A second close does nothing.
        exchange.group.rows.close()
        slot = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
        isolated = StoreGroup(None, "rows", rank, range(tp), active=[rank])
        rows = HostExchangeMnnvlMemory(isolated, MnnvlHostExchangeGroup.buffer_bytes(tp, slot))
        exchange = mnnvl._host_exchange = HostExchange(MnnvlHostExchangeGroup(rank, tp, rows, slot))
    else:
        _rebuild(exchange, _MpiGroup(), survivors, epoch)
        assert mnnvl.allgather(rank) == [r if _in_mask(survivors, r) else None for r in range(tp)]
    mpi.barrier()

    own = np.zeros((2 * tp, 1 + exchange._words), dtype=np.uint64)
    exchange.group._copy_to_host(exchange.group.rows.row(rank), own)
    epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
    _rebuild(exchange, _MpiGroup(), (1 << tp) - 1, epoch)
    kept = np.zeros_like(own)
    exchange.group._copy_to_host(exchange.group.rows.row(rank), kept)
    # No rank writes into a row before its owner has read it.
    mpi.barrier()
    # The admission kept every row's pages, so the survivors' last packets are still there.
    assert np.array_equal(kept, own)
    assert rank == leaver or own.any()
    assert exchange.group.rows.rows_needing_fresh_pages(_MpiGroup()) == frozenset()
    assert mnnvl.allgather(rank) == list(range(tp))
    assert mnnvl.allreduce(rank + 1, ReduceOp.SUM) == sum(r + 1 for r in range(tp))


def _write_slot(exchange, payload, dest, sender, epoch, header_only=False):
    """Write ``sender``'s packets of ``epoch``, or only the header, into rank ``dest``'s row."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import _check_cu_result, cuda

    tag = np.uint64(epoch) << np.uint64(32)
    words = np.frombuffer(payload + b"\0" * (-len(payload) % 4), dtype=np.uint32)
    packets = np.empty(1 + words.size, dtype=np.uint64)
    packets[0] = tag | np.uint64(len(payload))
    packets[1:] = tag | words.astype(np.uint64)
    if header_only:
        packets = packets[:1]
    slot = (epoch & 1) * exchange.group._slot_set_bytes + sender * exchange.group._slot_bytes
    _check_cu_result(
        cuda.cuMemcpyHtoD(exchange.group.rows.row(dest) + slot, packets.ctypes.data, packets.nbytes)
    )
    torch.cuda.synchronize()


def _membership_change(tp):
    """Fail the last rank, just before the epoch wraps, after it sent a round to rank 0 only.

    Rank 0 finishes that round and stops in the next. The others stop in that round and finish it
    with the packets rank 0 received. Every survivor gets the same results.
    """
    mnnvl, mpi, rank = _dists(tp)
    failed = tp - 1
    survivors = ((1 << tp) - 1) & ~(1 << failed)
    exchange = mnnvl.host_exchange
    exchange.admit(exchange.active_mask, 0xFFFFFFFE - 1)
    mnnvl.barrier()
    if rank == failed:
        _write_slot(exchange, pickle.dumps(rank), 0, rank, next_epoch(exchange.epoch))
        MPI.COMM_WORLD.send(None, dest=0)
        _leave(mpi)
        return

    if rank == 0:
        # Rank 0 sends only once the failed rank's packets reached it.
        MPI.COMM_WORLD.recv(source=failed)
    changes = _install_on_interrupt(
        exchange, mpi, lambda: _in_mask(exchange.waiting_for, failed), survivors
    )
    with _woken(exchange):
        # Rank 0 received the failed rank's packets, so the first round has every payload.
        assert mnnvl.allgather(rank) == list(range(tp))
        assert mnnvl.allgather(10 + rank) == [
            10 + r if _in_mask(survivors, r) else None for r in range(tp)
        ]
    assert mnnvl.allreduce(1, ReduceOp.SUM) == tp - 1
    ((stopped, epoch),) = changes
    # Rank 0 stopped after the wrap, in the round that runs again.
    assert (next_epoch(stopped[rank]) == epoch) == (rank == 0), (rank, stopped, epoch)


def _change_without_loss(tp):
    """Check that every rank gets the same results from a change that loses no rank.

    The others stop while they wait for rank 0, which sends only once they stopped. Rank 0 finishes
    that round from their packets and stops in the next round, which runs again.
    """
    mnnvl, mpi, rank = _dists(tp)
    exchange = mnnvl.host_exchange
    mnnvl.barrier()
    changes = _install_on_interrupt(
        exchange,
        mpi,
        lambda: exchange.waiting_for != 0,
        (1 << tp) - 1,
        before=None if rank == 0 else lambda: MPI.COMM_WORLD.send(None, dest=0),
    )
    if rank == 0:
        for r in range(1, tp):
            MPI.COMM_WORLD.recv(source=r)
    with _woken(exchange):
        assert mnnvl.allgather(rank) == list(range(tp))
        assert mnnvl.allgather(10 + rank) == [10 + r for r in range(tp)]
    ((stopped, _),) = changes
    # Rank 0 stopped one round after the others, in the round that runs again.
    assert stopped[1:] == [stopped[1]] * (tp - 1), stopped
    assert stopped[0] == next_epoch(stopped[1]), stopped


def _wait_polling(tp):
    """Check that a wait on a late peer polls back to back for its spin, then sleeps between polls.

    A step wait as long as a prefill imbalance between ranks (0.1 s) never sleeps, a step wait
    longer than the spin sleeps after it, and an idle wait sleeps almost at once. A wait that polls
    back to back keeps its thread on the CPU and one that sleeps leaves it, so the check compares
    the thread's CPU time with the wait's duration. Every rank runs every wait before any rank
    asserts, so a failed check does not leave the other ranks waiting in the next collective.
    """
    mnnvl, _, rank = _dists(tp)
    spin_s = MnnvlHostExchangeGroup._SPIN_S
    failed = []
    # How late rank 0 is, whether the other ranks wait as idle ranks, and whether they spin.
    for late_s, idle, spins in (
        (0.1, False, True),
        (spin_s + 1.0, False, False),
        (1.0, True, False),
    ):
        mnnvl.barrier()
        MPI.COMM_WORLD.barrier()
        started_s, started_cpu_s = time.monotonic(), time.thread_time()
        if rank == 0:
            time.sleep(late_s)
        if idle:
            received = mnnvl.broadcast_int64([7], root=0, prefer_cpu=True).tolist()
            expected = [7]
        else:
            received, expected = mnnvl.allgather(rank), list(range(tp))
        if received != expected:
            failed.append((late_s, received))
        if rank != 0:
            cpu_s, wait_s = time.thread_time() - started_cpu_s, time.monotonic() - started_s
            if spins:
                polled_as_expected = cpu_s > wait_s / 2
            elif idle:
                polled_as_expected = cpu_s < wait_s / 2
            else:
                polled_as_expected = cpu_s - spin_s < (wait_s - spin_s) / 4
            if not polled_as_expected:
                failed.append((late_s, cpu_s, wait_s))
    MPI.COMM_WORLD.barrier()
    assert not failed, failed


def _payload_rows(tp):
    """Check that every row gets every header but only readers' rows get a long payload."""
    mnnvl, mpi, rank = _dists(tp)

    def payload_of(r):
        return bytes([1 + r]) * (3000 + r)

    payload = payload_of(rank)
    # This is the first exchange, so the rows hold nothing else.
    assert mnnvl.tp_gather(payload, root=0) == mpi.tp_gather(payload, root=0)
    exchange = mnnvl.host_exchange
    row = np.zeros((tp, 1 + exchange._words), dtype=np.uint64)
    exchange.group._copy_to_host(
        exchange.group.rows.row(rank) + (exchange.epoch & 1) * exchange.group._slot_set_bytes, row
    )
    epochs = row >> np.uint64(32)
    assert (epochs[:, 0] == exchange.epoch).all()
    if rank == 0:
        sent = min(1 + math.ceil(len(pickle.dumps(payload_of(r))) / 4) for r in range(tp))
        assert (epochs[:, 1:sent] == exchange.epoch).all()
    else:
        assert not row[:, 1:].any()
    # The readers wait for a payload wider than their header poll.
    large = bytes(range(256)) * 12
    assert mnnvl.broadcast(large if rank == 0 else None, root=0) == large


class _DeviceRows:
    """Hold the rows of a ``HostExchange`` whose ranks are all in this process, in device memory."""

    def __init__(self, nranks, nbytes):
        self._memory = torch.zeros(nranks * nbytes, dtype=torch.uint8, device="cuda")
        self.stride = nbytes

    def row(self, rank):
        return self._memory.data_ptr() + rank * self.stride


def _local_exchange(nranks):
    """Return rank 0 of a ``nranks``-rank exchange whose other ranks never send unless told to."""
    torch.cuda.set_device(tensorrt_llm.mpi_rank())
    slot = HostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
    rows = _DeviceRows(nranks, MnnvlHostExchangeGroup.buffer_bytes(nranks, slot))
    return HostExchange(MnnvlHostExchangeGroup(0, nranks, rows, slot))


def _wait_until(condition, timeout_s=10.0):
    """Poll ``condition`` until it holds."""
    deadline_s = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline_s, "timed out"
        time.sleep(1e-3)


def _polled_again(exchange, checks):
    """Wake the running wait twice, each time until ``interrupted`` appended to ``checks``."""
    for _ in range(2):
        count = len(checks)
        exchange.wake()
        _wait_until(lambda count=count: len(checks) > count)


def _bursty_thread(tp):
    """Check that a thread with bursts of Python work delays an exchange by one GIL handover at most.

    A wait that released the GIL around each driver call would hand it over three times or more.
    Each handover waits for the switch interval, which the check makes long against the
    scheduling delays of a shared node.
    """
    exchange = _local_exchange(1)
    default_switch_s, switch_s = sys.getswitchinterval(), 50e-3
    sys.setswitchinterval(switch_s)
    stop, bursting = threading.Event(), threading.Event()

    def bursts():
        while not stop.is_set():
            bursting.set()
            end_s = time.perf_counter() + 10 * switch_s
            while time.perf_counter() < end_s:
                pass
            bursting.clear()
            time.sleep(10 * switch_s)

    in_burst, between = [], []
    thread = threading.Thread(target=bursts, daemon=True)
    thread.start()
    try:
        step = 0
        while min(len(in_burst), len(between)) <= 20:
            assert thread.is_alive()
            # The executor waits for the GPU between collectives.
            time.sleep(1e-3)
            durations = in_burst if bursting.is_set() else between
            started_s = time.perf_counter()
            assert exchange.exchange(b"%d" % step) == [b"%d" % step]
            durations.append(time.perf_counter() - started_s)
            step += 1
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(default_switch_s)
    assert np.median(in_burst) < 2 * switch_s, sorted(in_burst)
    assert np.median(between) < switch_s / 2, sorted(between)


class _Stopped(Exception):
    """Raised by ``on_interrupt`` to end the exchange that a membership change stopped."""


def _abort_wait(tp):
    """Check that a wake ends a wait for a silent rank once ``interrupted`` holds, and only then.

    ``waiting_for`` names only the silent rank, not rank 2, whose wide payload arrived. The wait
    reads the wake word after every poll, so the stopping wake ends it within a poll or a sleep.
    """
    exchange = _local_exchange(3)
    epoch = next_epoch(exchange.epoch)
    _write_slot(exchange, bytes(4096), 0, 2, epoch)
    changing = threading.Event()
    checks, stopped_at_check, waiting_for, woken_s, stopped_s = [], [], [], [], []

    def interrupted():
        checks.append(changing.is_set())
        return changing.is_set()

    def on_interrupt():
        stopped_s.append(time.monotonic())
        stopped_at_check.append(len(checks))
        raise _Stopped

    exchange.set_interrupt_handler(interrupted, on_interrupt)

    def membership_client():
        _wait_until(lambda: exchange.waiting_for == 0b010)
        # Two records that do not stop the exchange, then one that does.
        _polled_again(exchange, checks)
        waiting_for.append(exchange.waiting_for)
        changing.set()
        woken_s.append(time.monotonic())
        exchange.wake()

    client = threading.Thread(target=membership_client, daemon=True)
    client.start()
    with pytest.raises(_Stopped):
        exchange.exchange(b"r0")
    client.join()
    assert waiting_for == [0b010]
    assert checks == [False, False, False, True]
    # The wait stopped once, at the first check that held, well within a step.
    assert stopped_at_check == [4]
    assert stopped_s[0] - woken_s[0] < 0.1, stopped_s[0] - woken_s[0]
    # The wakes that did not stop the wait did not restart its round either.
    assert exchange.stopped_epoch == epoch
    assert exchange.waiting_for == 0


def _stale_slots(tp):
    """Check that a wait takes only packets of the current epoch after a wrap, from whole slots."""
    exchange = _local_exchange(2)
    old_epoch, epoch = 0xFFFFFFFD, 1
    assert next_epoch(next_epoch(old_epoch)) == epoch
    exchange.admit(exchange.active_mask, epoch)
    _write_slot(exchange, b"new payload", 0, 1, old_epoch)
    checks = []

    def interrupted():
        checks.append(None)
        return False

    exchange.set_interrupt_handler(interrupted)
    result = []
    waiter = threading.Thread(target=lambda: result.append(exchange.exchange(b"r0")), daemon=True)
    waiter.start()
    _wait_until(lambda: exchange.waiting_for == 0b10)
    _polled_again(exchange, checks)
    assert waiter.is_alive()
    _write_slot(exchange, b"new payload", 0, 1, epoch, header_only=True)
    _polled_again(exchange, checks)
    assert waiter.is_alive()
    assert exchange.waiting_for == 0b10
    _write_slot(exchange, b"new payload", 0, 1, epoch)
    waiter.join(10.0)
    assert result == [[b"r0", b"new payload"]]
    assert exchange.epoch == epoch


def _arrived_slots(tp):
    """Check that a wait does not copy again a slot whose packets arrived."""
    exchange = _local_exchange(3)
    epoch = next_epoch(exchange.epoch)
    _write_slot(exchange, b"first", 0, 1, epoch)
    result = []
    waiter = threading.Thread(target=lambda: result.append(exchange.exchange(b"r0")), daemon=True)
    waiter.start()
    _wait_until(lambda: exchange.waiting_for == 0b100)
    _write_slot(exchange, b"later", 0, 1, epoch)
    _write_slot(exchange, b"r2", 0, 2, epoch)
    waiter.join(10.0)
    assert result == [[b"r0", b"first", b"r2"]]


def _worker(fn, tp):
    try:
        result = fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True if result is None else result


CASES = {
    "collectives": _collectives,
    "stress": _stress,
    "epoch_wrap": _epoch_wrap,
    "wait_polling": _wait_polling,
    "payload_rows": _payload_rows,
    "membership": _membership,
    "membership_change": _membership_change,
    "change_without_loss": _change_without_loss,
    "admission": _admission,
}
# membership_change needs a rank that finishes the round, one that does not, and a failed one.
PARAMS = [(case, tp) for case in CASES for tp in (2, 4) if case != "membership_change" or tp > 2]
# These cases run every rank of their exchange in one process, on one GPU.
CASES.update(
    bursty_thread=_bursty_thread,
    abort_wait=_abort_wait,
    stale_slots=_stale_slots,
    arrived_slots=_arrived_slots,
)
PARAMS += [(case, 1) for case in ("bursty_thread", "abort_wait", "stale_slots", "arrived_slots")]


@pytest.mark.parametrize("case,mpi_pool_executor", PARAMS, indirect=["mpi_pool_executor"])
def test_mnnvl_dist(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    # The one-rank cases keep the exchange in device memory. MnnvlDist shares it as MNNVL memory.
    if tp > 1:
        skip_no_mnnvl_fabric_handles(tp)
    elif not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    results = list(mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp))
    assert all(results)

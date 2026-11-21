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
"""Tests for the host collectives over MNNVL memory, ``HostExchange`` and ``MnnvlDist``.

The ranks run in threads of one process and exchange over rows of host memory.
"""

import ctypes
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.lib.stride_tricks import as_strided

from tensorrt_llm._torch.distributed import host_exchange, mnnvl_host_exchange, mnnvl_memory
from tensorrt_llm._torch.distributed.host_exchange import HostExchange, MnnvlDist
from tensorrt_llm._torch.distributed.membership import (
    Authority,
    RankReport,
    Reply,
    ResumeState,
    rank_mask,
)
from tensorrt_llm._torch.distributed.mnnvl_host_exchange import MnnvlHostExchangeGroup
from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
JOIN_TIMEOUT_S = 30.0


# The host exchange under a membership change.


class _HostCopies:
    """Run the copies and the native wait of HostExchange between host addresses, each whole."""

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemorytype = SimpleNamespace(CU_MEMORYTYPE_DEVICE="device", CU_MEMORYTYPE_HOST="host")

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def CUDA_MEMCPY2D():
        return SimpleNamespace()

    def _copy(self, dst, dst_pitch, src, src_pitch, width, height):
        with self._lock:
            for row in range(height):
                ctypes.memmove(dst + row * dst_pitch, src + row * src_pitch, width)

    def cuMemcpy2DAsync(self, copy, stream):
        src = copy.srcHost if copy.srcMemoryType == "host" else copy.srcDevice
        dst = copy.dstHost if copy.dstMemoryType == "host" else copy.dstDevice
        self._copy(dst, copy.dstPitch, src, copy.srcPitch, copy.WidthInBytes, copy.Height)
        return (0,)

    def wait_for_slots(self, stream, sends, slots, received, active, readers, epoch, *rest):
        """Wait as ``wait_for_slots`` does, polling every slot whole."""
        header_packets, stale, wake, spin_s = rest
        for copy in sends:
            self._copy(*copy)
        pitch = received.strides[0]
        while True:
            if slots is not None:
                self._copy(received.ctypes.data, pitch, slots, pitch, pitch, len(received))
            headers = received[:, 0] >> np.uint64(32) == epoch
            for rank, slot in enumerate(received):
                length = int(slot[0]) & 0xFFFFFFFF
                packets = 1 + min(len(slot) - 1, -(-length // 4)) if readers[rank] else 1
                epochs = slot[:packets] >> np.uint64(32)
                stale[rank] = active[rank] and bool((epochs != epoch).any())
            if slots is not None and not headers[active].all():
                # While a header is missing, the wait names only the ranks whose header is missing.
                stale[:] = active & ~headers
            if slots is None or not stale.any():
                return not stale.any()
            if wake[0]:
                return False
            time.sleep(1e-4)


class _HostRows:
    """Map HostExchangeMnnvlMemory rows in shared host memory as rank ``rank`` sees them."""

    def __init__(self, memory, stride, rank):
        self._memory, self.stride, self.rank = memory, stride, rank
        self.ptr = memory.ctypes.data

    def row(self, rank):
        return self.ptr + rank * self.stride

    def rows_needing_fresh_pages(self, gather):
        """Return every row, as after a failure."""
        return frozenset(range(len(self._memory) // self.stride))

    def remap(self, gather, fresh_rows):
        """Zero this rank's row if it gets fresh pages. The memory is shared, so no gather runs."""
        if self.rank in fresh_rows:
            self._memory[self.rank * self.stride : (self.rank + 1) * self.stride] = 0


_SLOT_BYTES = 64


def _use_host_copies(monkeypatch):
    """Run the copies and the native wait of the host exchange between host addresses."""
    copies = _HostCopies()
    monkeypatch.setattr(mnnvl_memory, "cuda", copies)
    monkeypatch.setattr(mnnvl_host_exchange, "wait_for_slots", copies.wait_for_slots)
    monkeypatch.setattr(mnnvl_host_exchange, "prefer_pinned", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda: SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
    )


def _host_exchange_group(monkeypatch, nranks, slot_bytes):
    """Return a HostExchange for each of ``nranks`` ranks over shared host rows."""
    _use_host_copies(monkeypatch)
    stride = MnnvlHostExchangeGroup.buffer_bytes(nranks, slot_bytes)
    memory = np.zeros(nranks * stride, dtype=np.uint8)
    return [
        HostExchange(
            MnnvlHostExchangeGroup(rank, nranks, _HostRows(memory, stride, rank), slot_bytes)
        )
        for rank in range(nranks)
    ]


def _run_ranks(exchanges, calls):
    """Run ``calls[rank](exchanges[rank])`` on a thread per rank, waking the waits every 1 ms."""
    results, errors = {}, []
    done = threading.Event()

    def run(rank):
        try:
            results[rank] = calls[rank](exchanges[rank])
        except BaseException as error:  # surface thread failures in the test
            errors.append(error)

    def wake():
        while not done.wait(1e-3):
            for exchange in exchanges:
                exchange.wake()

    ranks = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(len(calls))]
    waker = threading.Thread(target=wake, daemon=True)
    for thread in [*ranks, waker]:
        thread.start()
    for thread in ranks:
        thread.join(JOIN_TIMEOUT_S)
    done.set()
    waker.join()
    assert not errors, errors
    assert not any(thread.is_alive() for thread in ranks)
    return results


class _Replaced(Exception):
    """Raised where a newer membership change ends a rank's wait."""


@pytest.mark.timeout(60)
def test_superseded_install_gives_same_results(monkeypatch):
    """Check that every survivor gets the same results when a newer change supersedes an install.

    Rank 2 fails while ranks 0 to 2 remap the rows without rank 3, and rank 0 stops in that remap.
    """
    exchanges = _host_exchange_group(monkeypatch, WORLD, _SLOT_BYTES)
    authority = Authority(WORLD)
    authority.tick(0.0, {r: RankReport(started=True) for r in range(WORLD)}, (1 << WORLD) - 1)
    connected = {1: 0b0111, 2: 0b0011}
    stopped = {1: {}, 2: {}}
    epochs = {}

    def decide(generation):
        ranks = [r for r in range(WORLD) if connected[generation] >> r & 1]
        authority.tick(generation, {r: RankReport() for r in ranks}, connected[generation])
        ready = {
            r: RankReport(
                reply=Reply.READY,
                reply_generation=generation,
                paused_at=ResumeState(epoch=stopped[generation][r]),
            )
            for r in ranks
        }
        epochs[generation] = authority.tick(generation, ready, connected[generation]).resume.epoch

    ready = {1: threading.Barrier(3, action=lambda: decide(1), timeout=20)}
    ready[2] = threading.Barrier(2, action=lambda: decide(2), timeout=20)
    committed = threading.Barrier(2, timeout=20)
    generations = {0: [1, 2], 1: [1, 2], 2: [1]}
    remaps = []

    def remap_of_rank_0(gather, fresh_rows):
        remaps.append(fresh_rows)
        if len(remaps) == 1:
            raise _Replaced
        _HostRows.remap(exchanges[0].group.rows, gather, fresh_rows)

    exchanges[0].group.rows.remap = remap_of_rank_0

    def change(exchange):
        # Run the membership changes this rank takes part in, from READY to the commit.
        while True:
            generation = generations[exchange.rank].pop(0)
            stopped[generation][exchange.rank] = exchange.stopped_epoch
            ready[generation].wait()
            if exchange.rank == 2:
                raise _Replaced  # Its process ends.
            try:
                exchange.install(None, connected[generation], epochs[generation])
            except _Replaced:
                continue
            if generation == 2:
                committed.wait()
                exchange.set_interrupt_handler(lambda: False)
            return

    # Each change reaches a rank while it waits for a failed rank.
    for exchange in exchanges[:3]:
        exchange.set_interrupt_handler(
            lambda exchange=exchange: bool(exchange.waiting_for & 0b1100),
            lambda exchange=exchange: change(exchange),
        )

    def survivor(exchange):
        return exchange.exchange(b"r%d" % exchange.rank), exchange.exchange(b"s%d" % exchange.rank)

    def fails(exchange):
        with pytest.raises(_Replaced):
            survivor(exchange)

    results = _run_ranks(exchanges[:3], [survivor, survivor, fails])
    assert epochs == {1: 2, 2: 2}
    assert results[0] == ([b"r0", b"r1", None, None], [b"s0", b"s1", None, None])
    assert results[1] == results[0]


def test_stop_before_round_without_install_runs_it(monkeypatch):
    """Check that a rank that stops before a round and installs nothing runs the round once."""
    exchanges = _host_exchange_group(monkeypatch, 2, _SLOT_BYTES)
    stops = [True]
    exchanges[0].set_interrupt_handler(lambda: bool(stops) and stops.pop())

    def exchange_once(exchange):
        return exchange.exchange(b"r%d" % exchange.rank)

    results = _run_ranks(exchanges, [exchange_once, exchange_once])
    assert results[0] == [b"r0", b"r1"]
    assert results[1] == results[0]
    assert exchanges[0].stopped_epoch == 1


@pytest.mark.timeout(60)
def test_stop_in_wait_without_install_keeps_round(monkeypatch):
    """Check that a rank that stops while it waits and installs nothing keeps its sent round.

    Rank 0 stops once while it waits for rank 1, which starts its rounds only after that stop.
    """
    exchanges = _host_exchange_group(monkeypatch, 2, _SLOT_BYTES)
    stops = [True]
    exchanges[0].set_interrupt_handler(
        lambda: bool(exchanges[0].waiting_for & 0b10) and bool(stops) and stops.pop()
    )

    def two_rounds(exchange):
        if exchange.rank == 1:
            deadline_s = time.monotonic() + JOIN_TIMEOUT_S
            while stops and time.monotonic() < deadline_s:
                time.sleep(1e-4)
        return exchange.exchange(b"a%d" % exchange.rank), exchange.exchange(b"b%d" % exchange.rank)

    results = _run_ranks(exchanges, [two_rounds, two_rounds])
    assert not stops
    assert results[0] == ([b"a0", b"a1"], [b"b0", b"b1"])
    assert results[1] == results[0]


@pytest.mark.timeout(60)
def test_waiting_for_during_flag_rewrites(monkeypatch):
    """Check that ``waiting_for`` names only real ranks while the wait rewrites its flags.

    The native wait rewrites the flags on every poll without the GIL. Here numpy rewrites them as
    often, by copying into a view that repeats them, which also runs without the GIL. Before each
    read the test frees index arrays of -1, which numpy reuses for its next small results. So an
    index that np.flatnonzero counted but never wrote is out of range instead of a stale rank.
    """
    nranks, repeats = 72, 1 << 16
    exchange = _host_exchange_group(monkeypatch, nranks, _SLOT_BYTES)[0]
    polls = np.zeros((repeats, nranks), dtype=bool)
    polls[::2] = True
    rewriting = threading.Event()

    def rewrite_flags(
        stream,
        sends,
        slots,
        received,
        active,
        readers,
        epoch,
        header_packets,
        waiting_for,
        wake,
        spin_s,
    ):
        repeated = as_strided(waiting_for, (repeats, nranks), (0, waiting_for.strides[0]))
        rewriting.set()
        while not wake[0]:
            np.copyto(repeated, polls)
        return False

    def stop():
        raise _Replaced

    monkeypatch.setattr(mnnvl_host_exchange, "wait_for_slots", rewrite_flags)
    stopping = threading.Event()
    exchange.set_interrupt_handler(stopping.is_set, stop)
    errors = []

    def wait():
        try:
            exchange.exchange(b"")
        except _Replaced as error:
            errors.append(error)

    def read():
        for size in range(1, nranks + 1):
            np.full(size, -1, dtype=np.intp)
        return exchange.waiting_for

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    try:
        assert rewriting.wait(JOIN_TIMEOUT_S)
        # Read until the reads saw the flags change, which a contended node can delay.
        deadline_s = time.monotonic() + JOIN_TIMEOUT_S
        masks = {read() for _ in range(2000)}
        while len(masks) == 1 and time.monotonic() < deadline_s:
            masks |= {read() for _ in range(2000)}
    finally:
        stopping.set()
        exchange.wake()
        waiter.join(JOIN_TIMEOUT_S)
    assert not waiter.is_alive()
    assert len(errors) == 1
    assert len(masks) > 1
    assert max(masks) >> nranks == 0


def test_joining_rank_installs_rows(monkeypatch):
    """Check that a joining rank maps the rows at INSTALL without entering the exchange."""
    joining = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)[2]
    joining.group.rows._memory[:] = 1
    joining.install(None, 0b011, 7)
    assert joining.active_mask == 0b011
    assert joining.epoch == 6
    row = joining.group.rows._memory[2 * joining.group.rows.stride : 3 * joining.group.rows.stride]
    assert not row.any()


def _slot(epoch, payload, length=None, words=16):
    """Return the packets of a slot that carries ``payload`` in round ``epoch``."""
    packets = np.zeros(1 + words, dtype=np.uint64)
    packets[0] = len(payload) if length is None else length
    data = np.frombuffer(payload + b"\0" * (-len(payload) % 4), dtype=np.uint32)
    packets[1 : 1 + data.size] = data
    return packets | (np.uint64(epoch) << np.uint64(32))


class _LocalGroup:
    """A ``HostExchangeGroup`` whose ranks are threads of this process, with no MNNVL memory."""

    def __init__(self, rank, nranks, slots, words=16):
        self.rank, self.nranks, self.words = rank, nranks, words
        # (epoch, sender) -> the sender's packets, shared by every rank.
        self._slots = slots
        self._staged = None
        self._cancelled = threading.Event()
        self._waiting = 0
        self.installs = []

    def start_round(self, epoch, words, length, recipients):
        self._staged = (epoch, _slot(epoch, words.tobytes(), length, self.words))

    def _rows(self, epoch):
        empty = np.zeros(1 + self.words, dtype=np.uint64)
        return np.stack([self._slots.get((epoch, r), empty) for r in range(self.nranks)])

    def wait(self, epoch, active, readers, idle=False, received=None):
        if self._staged is not None:
            self._slots[self._staged[0], self.rank] = self._staged[1]
            self._staged = None
        if received is not None:
            return received
        while True:
            lacking = [int(r) for r in np.flatnonzero(active) if (epoch, r) not in self._slots]
            self._waiting = rank_mask(lacking, self.nranks)
            if not lacking:
                return self._rows(epoch)
            if self._cancelled.is_set():
                self._cancelled.clear()
                return None
            time.sleep(1e-4)

    def cancel(self):
        self._cancelled.set()

    @property
    def waiting_for(self):
        return self._waiting

    def end_wait(self):
        self._waiting = 0

    def install(self, comm, active_mask, incomplete=None):
        self.installs.append((active_mask, incomplete))
        return None if incomplete is None else self._rows(incomplete[0])


@pytest.mark.timeout(60)
def test_waited_epochs_name_lacking_ranks():
    """Report the running round's epoch with each rank that the round lacks packets from."""
    slots = {}
    exchanges = [HostExchange(_LocalGroup(rank, 3, slots)) for rank in range(3)]
    seen = []

    def first(exchange):
        exchange.exchange(b"r0")
        return exchange.epoch

    def late(exchange):
        while not seen:
            if exchange.rank == 1 and exchanges[0].waiting_for == 0b110:
                seen.append(exchanges[0].waited_epochs)
            time.sleep(1e-4)
        return exchange.exchange(b"late")

    results = _run_ranks(exchanges, [first, late, late])
    assert results[0] > 0
    assert seen == [((1, results[0]), (2, results[0]))]


@pytest.mark.timeout(60)
def test_exchange_over_any_group():
    """Check a round, a cancel and an install over a group that has only the interface."""
    slots = {}
    exchanges = [HostExchange(_LocalGroup(rank, 3, slots)) for rank in range(3)]

    def survivor(exchange):
        first = exchange.exchange(b"r%d" % exchange.rank)
        # Rank 2 sends no second round. A wake finds the survivors waiting for it.
        exchange.set_interrupt_handler(
            lambda: bool(exchange.waiting_for & 0b100),
            lambda: exchange.install(None, 0b011, exchange.stopped_epoch),
        )
        return first, exchange.exchange(b"s%d" % exchange.rank)

    def fails(exchange):
        return exchange.exchange(b"r2")

    results = _run_ranks(exchanges, [survivor, survivor, fails])
    assert results[2] == [b"r0", b"r1", b"r2"]
    for rank in range(2):
        assert results[rank] == ([b"r0", b"r1", b"r2"], [b"s0", b"s1", None])
        assert exchanges[rank].group.installs == [(0b011, None)]
        assert exchanges[rank].active_mask == 0b011
    assert exchanges[2].group.installs == []


@pytest.mark.timeout(60)
def test_lagging_rank_takes_completed_round():
    """Check that a rank one round behind takes the round that its install completed."""
    slots = {}
    exchange = HostExchange(_LocalGroup(0, 2, slots))

    def change():
        # Rank 1 finished round 1 and moved on to round 2 before the change.
        exchange.set_interrupt_handler(lambda: False)
        slots[1, 1], slots[3, 1] = _slot(1, b"b"), _slot(3, b"d")
        exchange.install(None, 0b11, 3)

    exchange.set_interrupt_handler(lambda: bool(exchange.waiting_for & 0b10), change)
    results = _run_ranks([exchange], [lambda e: (e.exchange(b"a"), e.exchange(b"c"))])
    assert results[0] == ([b"a", b"b"], [b"c", b"d"])
    ((mask, (epoch, readers, active)),) = exchange.group.installs
    assert (mask, epoch, readers.tolist(), active.tolist()) == (0b11, 1, [True] * 2, [True] * 2)
    assert exchange.epoch == 3


@pytest.mark.timeout(60)
def test_completed_round_counts_header_of_missing_payload(monkeypatch):
    """Check that a rank runs as many rounds as its peers after completing a round from their rows.

    In a gather to rank 0, rank 2's payload takes two rounds. Rank 2 failed after its header
    reached rank 1 but before its payload reached rank 0. Rank 1 finished round 1 from the
    headers and stopped in round 2, so rank 0 must run round 2 as well.
    """
    exchange = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)[0]
    group, rows = exchange.group, exchange.group.rows

    def write(row, epoch, sender, packets):
        slot = row * rows.stride + (epoch & 1) * group._slot_set_bytes + sender * group._slot_bytes
        rows._memory[slot : slot + packets.nbytes] = packets.view(np.uint8)

    write(0, 1, 1, _slot(1, b"p1"))
    write(1, 1, 2, _slot(1, b"", length=2 * _SLOT_BYTES)[:1])

    def change():
        exchange.set_interrupt_handler(lambda: False)
        exchange.install(None, 0b011, 3)
        # Rank 1's slot of round 2, which carries no payload, sent over the new rows.
        write(0, 3, 1, _slot(3, b""))

    exchange.set_interrupt_handler(lambda: bool(exchange.waiting_for & 0b100), change)
    results = _run_ranks([exchange], [lambda e: e.exchange(b"x0", recipients=(0,))])
    assert results[0] == [b"x0", b"p1", None]
    assert exchange.epoch == 3


def test_retried_install_keeps_completed_round(monkeypatch):
    """Check that a retried install takes the round that a failed install completed.

    The failed remap may release the peers' rows, so the retry must not read them again.
    """
    exchange = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)[0]
    group, rows = exchange.group, exchange.group.rows
    exchange.admit(0b111, 2)
    exchange.stopped_epoch = exchange.epoch
    exchange._round_readers = 0b111
    # Rank 1 received rank 2's slot of round 1; rank 2 then failed.
    slot = rows.stride + group._slot_set_bytes + 2 * group._slot_bytes
    sent = _slot(1, b"zz")
    rows._memory[slot : slot + sent.nbytes] = sent.view(np.uint8)
    remap = rows.remap

    def remap_fails_once(gather, fresh_rows):
        rows.remap = remap
        rows._memory[rows.stride : 2 * rows.stride] = 0  # The peers' rows are released.
        raise _Replaced

    rows.remap = remap_fails_once
    with pytest.raises(_Replaced):
        exchange.install(None, 0b011, 3)
    exchange.install(None, 0b011, 3)
    np.testing.assert_array_equal(exchange._finished_round[2, : sent.size], sent)


@pytest.mark.parametrize("stopped_epoch", [1, 2], ids=["in-round-1", "after-round-1"])
def test_install_refused_two_rounds_behind(stopped_epoch):
    """Check that a rank more than one round behind the latest active rank cannot install.

    The rank stopped in round 1 or just after it, and the latest active rank finished round 3. The
    peers' rows hold only one round, so the rank could not complete every round it missed.
    """
    exchange = HostExchange(_LocalGroup(0, 2, {}))
    exchange.admit(0b11, 2)
    exchange.stopped_epoch = stopped_epoch
    with pytest.raises(RuntimeError, match="more than one round before the latest active rank"):
        exchange.install(None, 0b11, 4)
    assert exchange.group.installs == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("seed", range(4))
def test_long_payloads_reach_only_recipients(monkeypatch, seed):
    """Check random payloads and readers, and that a long payload skips non-recipients' rows."""
    nranks, slot_bytes = 4, 4096
    exchanges = _host_exchange_group(monkeypatch, nranks, slot_bytes)
    rng = np.random.default_rng(seed)
    readers = [
        sorted(rng.choice(nranks, rng.integers(nranks + 1), replace=False).tolist())
        for _ in range(nranks)
    ]
    recipients = [[q for q in range(nranks) if r in readers[q]] for r in range(nranks)]
    payloads = [rng.bytes(int(rng.integers(3 * slot_bytes))) for _ in range(nranks)]

    def send(exchange):
        rank = exchange.rank
        return exchange.exchange(payloads[rank], readers[rank], recipients[rank])

    results = _run_ranks(exchanges, [send] * nranks)
    for q in range(nranks):
        assert results[q] == [payloads[r] if r in readers[q] else None for r in range(nranks)]
    rows = exchanges[0].group.rows._memory.view(np.uint64).reshape(nranks, 2, nranks, -1)
    header_poll = 1 + MnnvlHostExchangeGroup._HEADER_POLL_PACKETS
    for r in range(nranks):
        others = [q for q in range(nranks) if q not in recipients[r]]
        assert not rows[others, :, r, header_poll:].any()


def _row_chunk_by_slicing(rows, size, k):
    """Return round ``k`` of ``size``-byte rows by slicing the packets' lower halves in numpy."""
    words = rows.shape[1] - 1
    out = np.zeros((len(rows), -(-size // 4)), dtype=np.uint32)
    chunk_words = -(-min(4 * words, size - 4 * words * k) // 4)
    out[:, words * k : words * k + chunk_words] = rows.view(np.uint32)[
        :, 2 : 2 + 2 * chunk_words : 2
    ]
    return out.view(np.uint8)[:, :size]


def test_unpack_payloads_into_matches_slicing():
    """Check the native unpacking into rows against numpy slicing on random packets."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        nranks, words = int(rng.integers(1, 9)), int(rng.integers(1, 40))
        rows = rng.integers(2**64, size=(nranks, 1 + words), dtype=np.uint64)
        size = int(rng.integers(1, 12 * words))
        k = int(rng.integers(-(-size // (4 * words))))
        ranks = sorted(rng.choice(nranks, rng.integers(1, nranks + 1), replace=False).tolist())
        # Round 0 carries the whole row size in the header, later rounds their chunk size.
        length = size if k == 0 else min(4 * words, size - 4 * words * k)
        lengths = np.full(nranks, length, dtype=np.uint32)
        out = np.zeros((nranks, size), dtype=np.uint8)
        host_exchange.unpack_payloads_into(rows, lengths, ranks, out, 4 * words * k)
        expected = np.zeros_like(out)
        expected[ranks] = _row_chunk_by_slicing(rows, size, k)[ranks]
        np.testing.assert_array_equal(out, expected)


@pytest.mark.timeout(60)
def test_allgather_array_zeroes_absent_ranks(monkeypatch):
    """Check rows that span several rounds, and a zero row or none for an absent rank."""
    exchanges = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)
    for exchange in exchanges:
        exchange.active_mask = 0b011
    rows = [np.arange(20, dtype=np.int64) * 7 + 50 * r + 1 for r in range(3)]

    def gather(exchange):
        row = rows[exchange.rank]
        return exchange.allgather_array(row), exchange.allgather_array(row, active_only=True)

    results = _run_ranks(exchanges[:2], [gather, gather])
    expected = np.stack([rows[0], rows[1], np.zeros(20, dtype=np.int64)])
    for rank in range(2):
        every, active = results[rank]
        assert every.dtype == active.dtype == np.int64
        np.testing.assert_array_equal(every, expected)
        np.testing.assert_array_equal(active, expected[:2])


@pytest.mark.parametrize("rank, local", [(0, 4), (3, 4), (5, 2)])
def test_node_ranks_follow_gpus_per_node(rank, local):
    dist = MnnvlDist(Mapping(world_size=6, rank=rank, gpus_per_node=4, tp_size=6))
    assert dist.local_world_size == local


@pytest.mark.parametrize(
    "lacks, requirement",
    [
        (None, None),
        ("tp", "tensor_parallel_size equal to the world size"),
        ("cp", "context_parallel_size=1"),
        ("mnnvl", "multi-node NVLink (MNNVL) domain"),
        ("fabric", "an aarch64 host (such as a GB200 node), on which MNNVL memory is shared"),
        ("mpi", "the MPI orchestrator (TLLM_DISABLE_MPI unset)"),
    ],
)
def test_requirements_name_what_is_missing(monkeypatch, lacks, requirement):
    """Check that each unmet requirement is named with how to turn fault tolerance off."""
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1" if lacks == "mpi" else "0")
    monkeypatch.setattr(
        mnnvl_memory.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: lacks != "mnnvl")
    )
    monkeypatch.setattr(
        mnnvl_host_exchange.HostExchangeMnnvlMemory,
        "uses_fabric_handles",
        classmethod(lambda cls, dev_id: lacks != "fabric"),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mapping = SimpleNamespace(
        world_size=WORLD,
        enable_attention_dp=True,
        pp_size=1,
        cp_size=2 if lacks == "cp" else 1,
        tp_size=WORLD // 2 if lacks in ("tp", "cp") else WORLD,
    )
    if requirement is None:
        MnnvlDist.check_requirements(mapping)
        return
    with pytest.raises(ValueError, match=re.escape(requirement)) as error:
        MnnvlDist.check_requirements(mapping)
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


def test_cp_collectives_unavailable():
    """MnnvlDist has no context-parallel communication."""
    dist = MnnvlDist(Mapping(world_size=2, rank=0, tp_size=2))
    with pytest.raises(NotImplementedError, match="no context-parallel group"):
        dist.cp_allgather(0)
    with pytest.raises(NotImplementedError, match="no context-parallel group"):
        dist.cp_broadcast(0)

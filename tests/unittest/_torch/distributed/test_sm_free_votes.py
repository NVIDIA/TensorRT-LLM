# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM-free transport of TorchDist's object collectives: the round protocol.

The CUDA side needs GPUs. Here one thread per rank runs a vote group's ``SmFreeVoteStaging`` over
a fake fabric: every rank's symmetric buffer is host memory, puts go through the multicast mapping
or peer by peer, and sequence-word waits block on a condition. In the asynchronous mode every
rank's operations run in order on a worker thread of their own, a little late, and read their
source bytes only when they run -- like copies on a CUDA stream -- so a missing ``sync`` or fence
before the host rewrites a buffer shows up as a corrupted vote. That checks how payloads are cut
into rounds, where every rank's bytes land, what each rank reads back, and the TorchDist object
collectives built on top.
"""

import contextlib
import ctypes
import queue
import random
import threading
import time

import pytest
import torch

from tensorrt_llm._torch.distributed import communicator as comm
from tensorrt_llm._torch.distributed import sm_free_votes as sfv

pytestmark = pytest.mark.cpu_only


class _Event:
    """Host stand-in for torch.cuda.Event."""

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class _SharedFabric:
    """Every rank's symmetric buffer in host memory, shared by the rank threads."""

    def __init__(self, world: int, nbytes: int, multicast: bool):
        self.world = world
        self.multicast = multicast
        self.bufs = [torch.zeros(nbytes, dtype=torch.uint8) for _ in range(world)]
        self.cond = threading.Condition()
        self.log = [[] for _ in range(world)]


class _ThreadEvent:
    def __init__(self, event: threading.Event):
        self._event = event

    def synchronize(self):
        assert self._event.wait(timeout=60), "fence never reached"


class _RankFabric:
    """One rank's view: the operations SmFreeVoteStaging issues.

    Synchronous: every operation runs when it is issued. Asynchronous: operations queue for a
    worker thread (the rank's "stream"), which runs them in order after a random delay and reads
    each put's source bytes only then.
    """

    def __init__(self, shared: _SharedFabric, rank: int, async_ops: bool = False):
        self.shared = shared
        self.rank = rank
        self.multicast = shared.multicast
        self.errors = []
        self._queue = None
        self._thread = None
        if async_ops:
            self._queue = queue.Queue()
            self._rng = random.Random(rank)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        while True:
            op = self._queue.get()
            try:
                if op is None:
                    return
                time.sleep(self._rng.random() * 0.0005)
                op()
            except BaseException as e:  # noqa: BLE001 -- surfaced by _run
                self.errors.append(e)
                with self.shared.cond:
                    self.shared.cond.notify_all()
            finally:
                self._queue.task_done()

    def _submit(self, op) -> None:
        if self._queue is None:
            op()
        else:
            self._queue.put(op)

    def stop(self) -> None:
        if self._queue is not None:
            self._queue.put(None)
            self._thread.join(timeout=30)
            assert not self._thread.is_alive(), f"rank {self.rank} worker did not stop"

    def connect(self) -> None:
        pass

    def _targets(self):
        if self.multicast:
            return range(self.shared.world)
        return [p for p in range(self.shared.world) if p != self.rank]

    def _put(self, offset: int, src: int, nbytes: int) -> None:
        def op():
            data = ctypes.string_at(src, nbytes)  # the source as of now, like a stream copy
            with self.shared.cond:
                for peer in self._targets():
                    ctypes.memmove(self.shared.bufs[peer].data_ptr() + offset, data, nbytes)
                self.shared.cond.notify_all()

        self._submit(op)

    def put(self, offset: int, src: int, nbytes: int) -> None:
        self.shared.log[self.rank].append(("put", nbytes))
        self._put(offset, src, nbytes)

    def put_word(self, offset: int, src: int, nbytes: int) -> None:
        assert nbytes == 8
        self._put(offset, src, nbytes)

    def wait_words(self, offset: int, value: int) -> None:
        words = self.shared.bufs[self.rank][offset : offset + 8 * self.shared.world].view(
            torch.int64
        )
        peers = [p for p in range(self.shared.world) if p != self.rank]

        def op():
            with self.shared.cond:
                ok = self.shared.cond.wait_for(
                    lambda: all(int(words[p]) >= value for p in peers), timeout=30
                )
            assert ok, f"rank {self.rank} timed out waiting for sequence {value}"

        self._submit(op)

    def read(self, dst: int, offset: int, nbytes: int) -> None:
        self.shared.log[self.rank].append(("read", nbytes))
        src = self.shared.bufs[self.rank].data_ptr() + offset
        self._submit(lambda: ctypes.memmove(dst, src, nbytes))

    def sync(self) -> None:
        if self._queue is not None:
            self._queue.join()
        assert not self.errors, self.errors

    def fence(self):
        if self._queue is None:
            return _Event()
        event = threading.Event()
        self._submit(event.set)
        return _ThreadEvent(event)


class _Group:
    def __init__(self, world: int, rank: int):
        self._world, self._rank = world, rank

    def size(self) -> int:
        return self._world

    def rank(self) -> int:
        return self._rank


def _run(world, chunk, body, multicast=True, monkeypatch=None, async_ops=False):
    """Run ``body(rank, staging)`` on one thread per rank; returns the results and op log."""
    words = -(-8 * world // 128) * 128
    shared = _SharedFabric(world, words + 2 * world * chunk, multicast)
    fabrics = [_RankFabric(shared, r, async_ops) for r in range(world)]
    results, errors = [None] * world, []
    local = threading.local()
    if monkeypatch is not None:
        monkeypatch.setattr(comm.torch.distributed, "get_rank", lambda *a, **k: local.rank)
        monkeypatch.setattr(comm.torch.distributed, "get_group_rank", lambda pg, r: r)

    def thread_body(rank):
        local.rank = rank
        try:
            staging = sfv.SmFreeVoteStaging(
                _Group(world, rank), None, chunk_bytes=chunk, fabric=fabrics[rank]
            )
            results[rank] = body(rank, staging)
            fabrics[rank].sync()  # a root's last rounds may still be queued
        except BaseException as e:  # noqa: BLE001 -- surface any rank's failure
            errors.append(e)
            with shared.cond:
                shared.cond.notify_all()

    threads = [threading.Thread(target=thread_body, args=(r,)) for r in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for fabric in fabrics:
        fabric.stop()
    assert not errors, errors
    assert not [e for fabric in fabrics for e in fabric.errors]
    return results, shared.log


def _payload(rank: int, nbytes: int) -> torch.Tensor:
    return ((torch.arange(nbytes, dtype=torch.int64) * 31 + rank * 7) % 251).to(torch.uint8)


@pytest.mark.parametrize("multicast", [True, False], ids=["multicast", "unicast"])
@pytest.mark.parametrize(
    "dtype,values", [(torch.int64, [0, 5, 2**40, -3]), (torch.float64, [0.5, -1.25, 3.0, 1e300])]
)
def test_one_word_per_rank(multicast, dtype, values):
    results, _ = _run(
        4, 16, lambda rank, st: st.allgather_words(values[rank], dtype), multicast=multicast
    )
    for got in results:
        assert got.dtype == dtype and got.tolist() == values


@pytest.mark.parametrize("async_ops", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("multicast", [True, False], ids=["multicast", "unicast"])
@pytest.mark.parametrize("max_size", [1, 15, 16, 17, 16 * 3 + 5])
def test_payloads_come_back_whole_in_rounds_of_one_chunk(multicast, max_size, async_ops):
    world, chunk = 3, 16
    sizes = [max_size, max(0, max_size - 4), max_size // 2]

    def body(rank, st):
        mine = _payload(rank, sizes[rank])
        return st.allgather_bytes(mine if sizes[rank] else None, sizes[rank], max_size)

    results, log = _run(world, chunk, body, multicast=multicast, async_ops=async_ops)
    for flat in results:
        assert flat.numel() == world * max_size
        for r in range(world):
            got = flat[r * max_size : r * max_size + sizes[r]]
            assert torch.equal(got, _payload(r, sizes[r]))
    rounds = -(-max_size // chunk)
    for rank_log in log:
        assert [nbytes for op, nbytes in rank_log if op == "put"] == [
            min(chunk, max_size - i * chunk) for i in range(rounds)
        ]


def test_without_read_back_the_rounds_are_the_same():
    world, chunk, max_size = 2, 8, 20

    def body(read_back):
        return lambda rank, st: st.allgather_bytes(
            _payload(rank, max_size), max_size, max_size, read_back=read_back
        )

    _, read = _run(world, chunk, body(True))
    results, skipped = _run(world, chunk, body(False))
    assert results == [None] * world
    assert [[e for e in log if e[0] == "put"] for log in skipped] == [
        [e for e in log if e[0] == "put"] for log in read
    ]
    assert all(op != "read" for log in skipped for op, _ in log)


@pytest.mark.parametrize("async_ops", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("multicast", [True, False], ids=["multicast", "unicast"])
@pytest.mark.parametrize("world,root", [(4, 0), (4, 3), (3, 1)])
@pytest.mark.parametrize("nbytes", [0, 1, 63, 64, 65, 64 * 3 + 2])
def test_broadcast_reaches_every_rank(multicast, world, root, nbytes, async_ops):
    chunk = 64 // world  # a round carries world * chunk bytes
    data = _payload(root, nbytes).numpy().tobytes()

    def body(rank, st):
        return st.broadcast_bytes(data if rank == root else None, root)

    results, log = _run(world, chunk, body, multicast=multicast, async_ops=async_ops)
    for rank, got in enumerate(results):
        assert got is None if rank == root else got == data
    per_round = world * chunk
    root_puts = [nbytes for op, nbytes in log[root] if op == "put"]
    assert root_puts == [8] + [min(per_round, nbytes - o) for o in range(0, nbytes, per_round)]
    assert all(op != "put" for r, rank_log in enumerate(log) if r != root for op, _ in rank_log)


@pytest.mark.parametrize("async_ops", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("multicast", [True, False], ids=["multicast", "unicast"])
def test_a_mixed_stream_of_votes_stays_consistent(multicast, async_ops):
    world, chunk = 4, 8
    rng = random.Random(0)
    plan = [
        (
            rng.choice(["words", "bytes", "bcast"]),
            rng.randrange(world),
            rng.randrange(3 * world * chunk),
        )
        for _ in range(60)
    ]
    width = 2 * chunk + 3

    def body(rank, st):
        seen = []
        for i, (kind, root, n) in enumerate(plan):
            if kind == "words":
                seen.append(st.allgather_words(i * 10 + rank, torch.int64).tolist())
            elif kind == "bytes":
                size = (n + rank) % width
                flat = st.allgather_bytes(_payload(rank + i, size), size, width).view(world, width)
                seen.append([flat[r, : (n + r) % width].tolist() for r in range(world)])
            else:
                data = _payload(root + i, n).numpy().tobytes()
                got = st.broadcast_bytes(data if rank == root else None, root)
                seen.append(rank == root or got == data)
        return seen

    results, _ = _run(world, chunk, body, multicast=multicast, async_ops=async_ops)
    for seen in results:
        for i, (kind, _, n) in enumerate(plan):
            if kind == "words":
                assert seen[i] == [i * 10 + r for r in range(world)]
            elif kind == "bytes":
                assert seen[i] == [_payload(r + i, (n + r) % width).tolist() for r in range(world)]
            else:
                assert seen[i] is True


def test_self_test_runs_every_kind_of_round():
    world, chunk = 2, 64

    def body(rank, st):
        for issue in st.self_test_rounds():
            issue()

    _, log = _run(world, chunk, body)
    assert [nbytes for op, nbytes in log[0] if op == "put"] == [8, chunk, world * chunk]
    assert [nbytes for op, nbytes in log[1] if op == "put"] == [8, chunk]
    assert [nbytes for op, nbytes in log[1] if op == "read"] == [
        8 * world,
        world * chunk,
        world * chunk,
    ]


@pytest.mark.parametrize(
    "values,op,expected",
    [
        ([2.5, -1.0, 5.0], "SUM", 6.5),
        ([2.5, -1.0, -3.0], "PRODUCT", 7.5),
        ([2.5, -1.0, 5.0], "MIN", -1.0),
        ([2.5, -1.0, 5.0], "MAX", 5.0),
    ],
)
def test_host_reduce(values, op, expected):
    assert sfv.host_reduce(values, op) == expected


def test_host_reduce_refuses_bitwise_ops():
    with pytest.raises(ValueError):
        sfv.host_reduce([1.0], "BAND")


def _torchdist(staging: sfv.SmFreeVoteStaging, pg) -> comm.TorchDist:
    # TorchDist's SM-free paths over one group's staging; __init__ needs Ray.
    td = comm.TorchDist.__new__(comm.TorchDist)
    td._cuda_obj_lock = threading.Lock()
    td._cuda_obj_sm_free = {pg: staging}
    td._cuda_obj_stream_ctx = contextlib.nullcontext
    return td


@pytest.mark.parametrize("async_ops", [False, True], ids=["sync", "async"])
def test_torchdist_gathers_and_reductions_on_the_sm_free_transport(monkeypatch, async_ops):
    world, chunk = 3, 16  # pickles above 16 bytes take several rounds
    pg = _Group(world, 0)  # what the TorchDist paths read of a group: size()

    def body(rank, st):
        td = _torchdist(st, pg)
        return [
            td._cuda_allgather_object(("a", rank), pg),
            td._cuda_allgather_object([], pg),
            td._cuda_allgather_object([] if rank == 1 else bytes(40 + rank), pg),
            td._cuda_gather_object(("g", rank), pg, 1),
            [t.tolist() for t in td._cuda_allgather_cpu_tensor(torch.tensor([rank, -rank]), pg)],
            [t.tolist() for t in td._cuda_allgather_cpu_tensor(torch.tensor([rank % 2 == 0]), pg)],
            td._cuda_allreduce_scalar(rank + 1, comm.ReduceOp.MAX, pg),
            td._cuda_allreduce_scalar(0.5, comm.ReduceOp.SUM, pg),
        ]

    results, _ = _run(world, chunk, body, monkeypatch=monkeypatch, async_ops=async_ops)
    for rank, got in enumerate(results):
        assert got[0] == [("a", r) for r in range(world)]
        assert got[1] == [[] for _ in range(world)]
        assert got[2] == [[] if r == 1 else bytes(40 + r) for r in range(world)]
        assert got[3] == ([("g", r) for r in range(world)] if rank == 1 else None)
        assert got[4] == [[r, -r] for r in range(world)]
        assert got[5] == [[r % 2 == 0] for r in range(world)]
        assert got[6] == world and isinstance(got[6], int)
        assert got[7] == 1.5 and isinstance(got[7], float)


def test_torchdist_refuses_a_foreign_reduce_op_on_the_sm_free_transport(monkeypatch):
    def body(rank, st):
        with pytest.raises(TypeError, match="expected tensorrt_llm ReduceOp"):
            pg = _Group(2, 0)
            _torchdist(st, pg)._cuda_allreduce_scalar(1, 3, pg)

    _run(2, 16, body, monkeypatch=monkeypatch)


@pytest.mark.parametrize("async_ops", [False, True], ids=["sync", "async"])
def test_torchdist_broadcasts_on_the_sm_free_transport(monkeypatch, async_ops):
    world, chunk = 3, 16  # a round carries 48 bytes
    pg = _Group(world, 0)  # what the TorchDist paths read of a group: size()
    payloads = [("small", 1), bytes(100), bytes(range(200))]

    def body(rank, st):
        td = _torchdist(st, pg)
        got = [
            td._cuda_broadcast_object(obj if rank == i % world else None, pg, i % world)
            for i, obj in enumerate(payloads)
        ]
        got.append(td._cuda_broadcast_object(None, pg, 2, allow_none=True))
        got.append(
            td._cuda_broadcast_object({"k": 1} if rank == 0 else None, pg, 0, allow_none=True)
        )
        return got

    results, _ = _run(world, chunk, body, monkeypatch=monkeypatch, async_ops=async_ops)
    for got in results:
        assert got[:3] == payloads
        assert got[3] is None
        assert got[4] == {"k": 1}

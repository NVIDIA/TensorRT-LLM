# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process fake of the executor's ``dist`` object for multi-rank coordinator tests.

A ``FakeDistGroup`` is a world of ``world_size`` ranks split into TP groups of
``tp_size`` consecutive ranks, one thread per rank. Each collective is a
``threading.Barrier`` rendezvous over its group: the call blocks until every
rank of the group has entered a collective, checks that all of them entered
the same one, then returns the gathered or reduced payloads. The fake
verifies the protocol -- which collectives each rank enters, how often, in
what order and with what payload -- not the blocking semantics of a real
communication backend.

Barrier timeouts are wall-clock inside ``threading`` and unaffected by tests
that patch ``time.monotonic``.
"""

import copy
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.distributed.communicator import ReduceOp


class FakeDistTimeout(AssertionError):
    """A rank waited for a collective that some peer never entered."""


class FakeDistMismatch(AssertionError):
    """Ranks of one group entered different collectives at the same step."""


class _PeerFailed(Exception):
    """A pending collective was released because another rank raised."""


_REDUCERS = {ReduceOp.SUM: sum, ReduceOp.MAX: max, ReduceOp.MIN: min}


class _Rendezvous:
    """Barrier plus per-step payload slots for one group of ranks."""

    def __init__(
        self,
        name: str,
        ranks: Sequence[int],
        timeout_s: float,
        last_call: Callable[[int], str],
    ) -> None:
        self.name = name
        self.ranks = tuple(ranks)
        self._timeout_s = timeout_s
        self._last_call = last_call
        self._barrier = threading.Barrier(len(self.ranks), timeout=timeout_s)
        self._lock = threading.Lock()
        # step -> {rank: (collective, payload)}; a slot is written once, never overwritten.
        self._steps: Dict[int, Dict[int, Tuple[str, Any]]] = {}
        self._next_step = {rank: 0 for rank in self.ranks}
        self._abort_reason: Optional[str] = None

    def exchange(self, rank: int, collective: str, payload: Any) -> List[Any]:
        """Enter ``collective`` with ``payload``; return every rank's payload in rank order.

        ``payload`` is stored as is, so callers pass a value they will not
        mutate. ``collective`` is the label every rank of the group must match.
        """
        with self._lock:
            step = self._next_step[rank]
            self._next_step[rank] = step + 1
            self._steps.setdefault(step, {})[rank] = (collective, payload)
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            raise self._broken(rank, step) from None
        with self._lock:
            arrivals = dict(self._steps[step])
        entered = {peer: name for peer, (name, _) in arrivals.items()}
        if len(set(entered.values())) > 1:
            raise FakeDistMismatch(
                f"{self.name}, step {step}: ranks entered different collectives: "
                + ", ".join(f"rank {peer}: {name}" for peer, name in sorted(entered.items()))
            )
        return [arrivals[peer][1] for peer in self.ranks]

    def abort(self, reason: str) -> None:
        """Release every waiting rank; they raise ``_PeerFailed`` instead of timing out."""
        with self._lock:
            if self._abort_reason is None:
                self._abort_reason = reason
        self._barrier.abort()

    def _broken(self, rank: int, step: int) -> Exception:
        with self._lock:
            reason = self._abort_reason
            arrivals = dict(self._steps.get(step, {}))
        if reason is not None:
            return _PeerFailed(reason)
        arrived = ", ".join(f"rank {peer} ({name})" for peer, (name, _) in sorted(arrivals.items()))
        missing = ", ".join(
            f"rank {peer} (last call: {self._last_call(peer)})"
            for peer in self.ranks
            if peer not in arrivals
        )
        return FakeDistTimeout(
            f"{self.name}, step {step}: rank {rank} waited {self._timeout_s}s for a collective. "
            f"Arrived: {arrived}. Missing: {missing}."
        )


class FakeDistRank:
    """The ``dist`` object one rank hands to its coordinator.

    Models only what the coordinator uses: a single pipeline / context-parallel
    stage, TP collectives over the rank's TP group and ``allreduce`` over the
    world. Every call is appended to ``calls`` as ``(collective, payload)``
    with the payload as the caller passed it, including calls a single-rank
    group answers locally.
    """

    pp_size = 1
    cp_size = 1

    def __init__(self, group: "FakeDistGroup", rank: int) -> None:
        self._group = group
        self.rank = rank
        self.world_size = group.world_size
        self.tp_size = group.tp_size
        self.tp_rank = rank % group.tp_size
        self.calls: List[Tuple[str, Any]] = []

    def tp_allgather(self, obj, *, small_payload: bool = False) -> list:
        return self._gather(self._group.tp_rendezvous(self.rank), "tp_allgather", obj)

    def tp_allgather_int64(self, values) -> np.ndarray:
        gathered = self._gather(self._group.tp_rendezvous(self.rank), "tp_allgather_int64", values)
        rows = [np.asarray(row, dtype=np.int64).reshape(-1) for row in gathered]
        if len({row.size for row in rows}) > 1:
            raise FakeDistMismatch(
                f"tp_allgather_int64: ranks passed vectors of different lengths "
                f"{[row.size for row in rows]}"
            )
        return np.stack(rows)

    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return self._reduce(self._group.tp_rendezvous(self.rank), "tp_allreduce", obj, op)

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return self._reduce(self._group.world_rendezvous(), "allreduce", obj, op)

    def _gather(
        self, rendezvous: _Rendezvous, collective: str, payload: Any, label: Optional[str] = None
    ) -> list:
        """Record the call and exchange a snapshot of ``payload``; return one copy per rank.

        The snapshot is taken before waiting: a sender may mutate its input as
        soon as the call returns, while a peer may not have read the slot yet.
        Receivers get their own copies, as they would after unpickling.
        ``label`` is what peers must agree on; it defaults to ``collective``.
        """
        snapshot = copy.deepcopy(payload)
        self.calls.append((collective, snapshot))
        if len(rendezvous.ranks) == 1:
            return [copy.deepcopy(snapshot)]
        gathered = rendezvous.exchange(self.rank, label or collective, snapshot)
        return [copy.deepcopy(item) for item in gathered]

    def _reduce(self, rendezvous: _Rendezvous, collective: str, payload: Any, op: ReduceOp):
        # Peers must agree on the operation, not just on entering a reduce.
        op = ReduceOp(op)
        values = self._gather(rendezvous, collective, payload, label=f"{collective}[{op.name}]")
        if len(values) == 1:
            return values[0]
        reducer = _REDUCERS.get(op)
        if reducer is None:
            raise NotImplementedError(f"FakeDist does not model {op!r}")
        return reducer(values)


class FakeDistGroup:
    """``world_size`` fake ranks in TP groups of ``tp_size`` consecutive ranks."""

    def __init__(self, world_size: int, tp_size: int, timeout_s: float = 5.0) -> None:
        if world_size % tp_size:
            raise ValueError(f"tp_size {tp_size} must divide world_size {world_size}")
        self.world_size = world_size
        self.tp_size = tp_size
        self._ranks = [FakeDistRank(self, rank) for rank in range(world_size)]
        self._world = _Rendezvous("world", range(world_size), timeout_s, self._last_call)
        self._tp_groups = [
            _Rendezvous(
                f"TP group {index}",
                range(index * tp_size, (index + 1) * tp_size),
                timeout_s,
                self._last_call,
            )
            for index in range(world_size // tp_size)
        ]

    def rank(self, rank: int) -> FakeDistRank:
        return self._ranks[rank]

    def tp_rendezvous(self, rank: int) -> _Rendezvous:
        return self._tp_groups[rank // self.tp_size]

    def world_rendezvous(self) -> _Rendezvous:
        return self._world

    def run(self, fn: Callable[[int], Any]) -> List[Any]:
        """Run ``fn(rank)`` on one thread per rank; return the results in rank order.

        The first exception raised on any rank is re-raised here. Peers blocked
        in a collective at that moment are released instead of timing out.
        """
        results: List[Any] = [None] * self.world_size
        failures: List[Tuple[int, Exception]] = []
        lock = threading.Lock()

        def worker(rank: int) -> None:
            try:
                results[rank] = fn(rank)
            except Exception as error:  # re-raised on the caller's thread below
                with lock:
                    failures.append((rank, error))
                for rendezvous in (self._world, *self._tp_groups):
                    rendezvous.abort(f"rank {rank} raised {type(error).__name__}")

        threads = [
            threading.Thread(target=worker, args=(rank,), name=f"fake-dist-rank-{rank}")
            for rank in range(self.world_size)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            causes = [failure for failure in failures if not isinstance(failure[1], _PeerFailed)]
            rank, error = (causes or failures)[0]
            if hasattr(error, "add_note"):
                error.add_note(f"raised on fake rank {rank}")
            raise error
        return results

    def _last_call(self, rank: int) -> str:
        calls = self._ranks[rank].calls
        return calls[-1][0] if calls else "none"

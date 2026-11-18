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
"""Record of the startup collectives for a replacement rank to replay, as it starts alone.

A replacement rank starts while the other ranks serve, so it cannot run the startup collectives
with them. Yet it must reach the same results, such as the KV cache capacity and the clock offset,
so that its CUDA graphs and buffers match theirs. Rank 0 records the result of each host collective
of ``MnnvlDist`` in call order. A replacement replays the record instead of running the collectives.
At each call it checks that its own payload and reduction match its predecessor's and fails with
``REPLACEMENT_CANNOT_START`` at the first divergence.

The callers reach the record through ``Distributed``, whose other implementations run every
collective. ``Distributed.rank_varying`` marks gathers whose payloads differ by process.
``reduce_op`` marks min or max reductions. A replacement's own value below a recorded minimum (a
capacity) stops it. Its own value above a recorded maximum (a failed status) takes part in the
result, so that it fails with its own error. The KV cache estimation, which a replacement skips,
runs between :meth:`StartupRecord.pause` and :meth:`StartupRecord.resume`, and so does the autotuner
pass that only a replacement runs. The startup group (``mpi_comm()``) is not recorded. A
replacement's holds only itself and raises on a collective whose result needs the other ranks'
values.
"""

import functools
import pickle  # nosec B403
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

# The exit status of a replacement whose startup failure any other replacement would repeat.
REPLACEMENT_CANNOT_START = 78
# The frames of a diverging call that its failure shows.
_DIVERGING_CALL_FRAMES = 10


class ReplacementCannotStart(SystemExit):
    """A ``SystemExit`` that ends a replacement rank with ``REPLACEMENT_CANNOT_START``."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.code = REPLACEMENT_CANNOT_START


def _caller_stack() -> str:
    """Return the innermost frames of the call stack outside this module."""
    frames = [frame for frame in traceback.extract_stack() if frame.filename != __file__]
    return "".join(traceback.format_list(frames[-_DIVERGING_CALL_FRAMES:])).rstrip()


def diverged(reason: str) -> ReplacementCannotStart:
    """Return the failure of a replacement whose startup diverges at the caller, with its stack."""
    return ReplacementCannotStart(
        f"{reason}. The diverging call (most recent call last):\n{_caller_stack()}"
    )


class RecordedExchange(NamedTuple):
    """A startup collective's :meth:`StartupRecord.rank_varying` block, every rank's payload and
    its reduction (``"min"``, ``"max"`` or None)."""

    rank_varying: Optional[str]
    payloads: List[Optional[bytes]]
    reduction: Optional[str]


def _describe(index: int, rank_varying: Optional[str]) -> str:
    collective = f"host collective {index}"
    return collective if rank_varying is None else f"{collective} ({rank_varying})"


def _describe_reduction(reduction: Optional[str]) -> str:
    return "no min or max reduction" if reduction is None else f"a {reduction} reduction"


class StartupRecord:
    """The startup collectives' results as rank 0 got them, for a replacement rank to replay.

    A replacement must make the same collectives in the same order. Its payload must equal its
    predecessor's, except in a :meth:`rank_varying` block and for min or max reductions. A reduction
    of a measured capacity takes the original ranks' result. The replacement's own capacity must
    still cover it.
    """

    def __init__(self, report_progress: Callable[[], None] = lambda: None) -> None:
        """Create an empty record.

        Args:
            report_progress: Called after each startup step of this rank, such as a replayed
                collective, an autotuner warmup step or a loaded checkpoint file.
        """
        self.values: Dict[str, List[Any]] = {}
        self.exchanges: List[RecordedExchange] = []
        # This rank's profiling cache where the recorded collectives start, at the end of its KV cache
        # estimation. A replacement takes its predecessor's.
        self.tuned_tactics: Dict[Any, Any] = {}
        self._recording = False
        self._replacement = False
        self._replaying = False
        self._taken: Dict[str, int] = {}
        self._replayed_exchanges = 0
        self.report_progress = report_progress
        self._pauses = 0
        self._rank_varying: Optional[str] = None
        # The agreement whose gather runs on a recording rank.
        self._gathering: Optional[str] = None

    @property
    def active(self) -> bool:
        return (self._recording or self._replaying) and not self._pauses

    @property
    def replaying(self) -> bool:
        return self._replaying and not self._pauses

    @property
    def gathers_rank_varying(self) -> bool:
        """Return whether the running collectives gather per-process values (``rank_varying``)."""
        return self._rank_varying is not None

    @property
    def replacement(self) -> bool:
        """Return whether this process replaces a rank of a running deployment."""
        return self._replacement

    def record(self) -> None:
        """Record the startup collectives on rank 0 until :meth:`close`, from an empty record."""
        self.values, self.exchanges = {}, []
        self._pauses = 0
        self._recording = True

    def replay(self, published: Dict[str, Any]) -> None:
        """Replay the record that rank 0 published until :meth:`close`."""
        self.values, self.exchanges = published["values"], published["exchanges"]
        self._replacement = self._replaying = True

    def published(self) -> Dict[str, Any]:
        return {"values": self.values, "exchanges": self.exchanges}

    def pause(self) -> None:
        """Run the startup collectives without recording or replaying them until :meth:`resume`.

        Only the values of ``agree(..., per_process=True)`` are still recorded and replayed.
        """
        self._pauses += 1

    def resume(self) -> None:
        """End the innermost :meth:`pause`, and on an original rank keep the tactics tuned in it.

        Raises:
            RuntimeError: No :meth:`pause` is open.
        """
        if not self._pauses:
            raise RuntimeError("resume() without a matching pause()")
        self._pauses -= 1
        if not self._replacement and not self._pauses:
            # Local: keeps the autotuner out of the authority and the node supervisor.
            from ..autotuner import AutoTuner

            self.tuned_tactics = dict(AutoTuner.get().profiling_cache.cache)

    @contextmanager
    def rank_varying(self, name: str) -> Iterator[None]:
        """Mark the startup collectives in the ``with`` block as ``name``, of per-process payloads.

        Such a payload is a per-rank measurement or a process identity, so a replacement's payload
        may differ from its predecessor's. The replay still returns the recorded payloads, its
        predecessor's included.
        """
        outer, self._rank_varying = self._rank_varying, name
        try:
            yield
        finally:
            self._rank_varying = outer

    def close(self) -> None:
        """Stop recording or replaying at the end of this rank's startup.

        Raises:
            ReplacementCannotStart: A replacement rank made fewer startup collectives than the
                original ranks.
        """
        if self._replacement:
            counts = {name: len(values) for name, values in self.values.items()}
            taken = {name: self._taken.get(name, 0) for name in counts}
            if taken != counts:
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank takes the startup values {taken} times, "
                    f"the original ranks {counts} times"
                )
            if self._replayed_exchanges != len(self.exchanges):
                index = self._replayed_exchanges
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank makes {index} host collectives, the "
                    f"original ranks {len(self.exchanges)}: the next one is "
                    f"{_describe(index, self.exchanges[index].rank_varying)}"
                )
        self._recording = self._replaying = False

    def agree(
        self,
        name: str,
        gather: Callable[[], Any],
        per_process: bool = False,
        own: Optional[Tuple[int, Any]] = None,
    ) -> Any:
        """Return the result of ``gather``, or rank 0's recorded result while a replacement replays.

        ``per_process`` records and replays the value even while paused, for an agreement that each
        process makes once, whichever executor it starts first.

        ``own`` is ``(index, value)`` when ``gather`` returns every member's input by index and this
        process's input is ``value``. A replacement's ``value`` must then equal its predecessor's
        recorded one, except in a :meth:`rank_varying` block.

        Raises:
            ReplacementCannotStart: A replacement takes the value more often than the original
                ranks, or its own input differs from its predecessor's.
        """
        recorded_now = not self._pauses or per_process
        if self._replaying and recorded_now:
            recorded = self.values.get(name, [])
            index = self._taken.get(name, 0)
            if index == len(recorded):
                raise diverged(
                    f"The startup of this replacement rank takes the startup value {name} "
                    f"{index + 1} times, the original ranks {len(recorded)} times"
                )
            self._taken[name] = index + 1
            if own is not None and self._rank_varying is None:
                member, value = own
                predecessor = recorded[index][member]
                if predecessor is not None and predecessor != value:
                    raise diverged(
                        f"The input of this replacement rank to the startup value {name} is "
                        f"{value!r}, its predecessor's {predecessor!r}"
                    )
            self.report_progress()
            return recorded[index]
        if not self._recording or not recorded_now:
            return gather()
        if self._gathering is not None:
            # A replacement replays the outer value without running the gather, so it would never
            # take this one.
            raise RuntimeError(
                f"a startup agreement's gather must not take other startup values "
                f"({self._gathering} takes {name})"
            )
        # The gather's own collectives are not recorded, as the replay returns the value instead.
        self._pauses += 1
        self._gathering = name
        try:
            value = gather()
        finally:
            self._pauses -= 1
            self._gathering = None
        self.values.setdefault(name, []).append(value)
        return value

    def agree_rank_capacity(
        self, name: str, rank: int, capacity: int, gather: Callable[[], Sequence[int]]
    ) -> int:
        """Return the ``capacity`` of rank ``rank``, or its predecessor's on a replacement rank.

        ``gather`` returns every rank's ``capacity``. A replacement's own ``capacity`` below its
        predecessor's raises RuntimeError, which a restart may not repeat once that memory is free.
        """
        recorded = self.agree(name, gather)[rank]
        if recorded > capacity:
            raise RuntimeError(
                f"The {name} of this replacement rank is {capacity}, below its predecessor's "
                f"{recorded}"
            )
        return recorded

    def exchange(
        self,
        rank: int,
        payload: bytes,
        readers: Optional[Sequence[int]],
        run: Callable[[], List[Optional[bytes]]],
        reduce_op: Optional[Callable[[Any, Any], Any]] = None,
        decode: Callable[[bytes], Any] = pickle.loads,
    ) -> List[Optional[bytes]]:
        """Return every rank's pickled payload, from ``run()``, or on a replacement from the record.

        ``reduce_op`` is ``min`` or ``max`` for a reduction by it, else None. A replacement's own
        value above the original ranks' maximum, such as a failed status, takes part in the result,
        so that the replacement fails with its own error. ``decode`` turns a payload into the value
        that the replay compares, because equal values can pickle differently.

        Raises:
            ReplacementCannotStart: The collective or its payload differs from the original ranks'
                at that call.
            RuntimeError: This replacement's value in a reduction by ``min`` lies below the
                original ranks' result.
        """
        rank_varying = self._rank_varying
        reduction = None if reduce_op is None else reduce_op.__name__
        if self._recording:
            payloads = run()
            recorded = list(payloads)
            recorded[rank] = payload
            self.exchanges.append(RecordedExchange(rank_varying, recorded, reduction))
            return payloads
        index = self._replayed_exchanges
        collective = _describe(index, rank_varying)
        if index == len(self.exchanges):
            raise diverged(
                f"The startup of this replacement rank makes {collective}, one more than the "
                "original ranks made"
            )
        recorded = self.exchanges[index]
        if recorded.rank_varying != rank_varying:
            raise diverged(
                f"The startup of this replacement rank makes {collective}, the original ranks' "
                f"{_describe(index, recorded.rank_varying)}"
            )
        if recorded.reduction != reduction:
            raise diverged(
                f"The startup of this replacement rank makes {collective} as "
                f"{_describe_reduction(reduction)}, the original ranks as "
                f"{_describe_reduction(recorded.reduction)}"
            )
        payloads = recorded.payloads
        predecessor = payloads[rank]
        if (
            predecessor is not None
            and rank_varying is None
            # Only this deployment's ranks write the rendezvous store the record comes from.
            and decode(predecessor) != decode(payload)
        ):
            if reduce_op is None:
                raise diverged(
                    f"The payload of this replacement rank to {collective} of its startup differs "
                    "from the original rank's"
                )
            if self._check_reduction(collective, reduce_op, decode(payload), payloads, decode):
                payloads = [payload if r == rank else p for r, p in enumerate(payloads)]
        readers = range(len(payloads)) if readers is None else readers
        unread = [r for r in readers if payloads[r] is None]
        if unread:
            raise diverged(
                f"The {collective} of this replacement rank's startup reads the payloads of ranks "
                f"{unread}, which rank 0 did not receive"
            )
        self._replayed_exchanges += 1
        self.report_progress()
        return [p if r in readers else None for r, p in enumerate(payloads)]

    @staticmethod
    def _check_reduction(
        collective: str,
        reduce_op: Callable[[Any, Any], Any],
        own: Any,
        payloads: List[Optional[bytes]],
        decode: Callable[[bytes], Any],
    ) -> bool:
        """Return whether this replacement's own value changes the original ranks' maximum.

        Raises:
            RuntimeError: This replacement's value lowers the original ranks' minimum.
        """
        values = [decode(p) for p in payloads if p is not None]
        result = functools.reduce(reduce_op, values)
        if reduce_op(own, result) == result:
            return False
        if reduce_op is max:
            return True
        raise RuntimeError(
            f"The value of this replacement rank to {collective} of its startup is {own}, "
            f"beyond the original ranks' {reduce_op.__name__} {result}. The call (most recent "
            f"call last):\n{_caller_stack()}"
        )

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
"""Host collectives for Wide-EP fault tolerance.

``MnnvlDist`` implements the attention-DP host collectives of ``Distributed`` without MPI, so they
keep working after a rank fails or a replacement joins. Each collective is one or more rounds of
``HostExchange``, which runs every round over a ``HostExchangeGroup``. The group moves one slot
per rank; the exchange owns the epochs, the active ranks and the agreement on a round that a
membership change interrupted. ``MnnvlHostExchangeGroup`` moves the slots through MNNVL memory.
"""

from __future__ import annotations

import functools
import pickle  # nosec B403
from typing import Callable, Iterator, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm._wide_ep_fault_tolerance import check_fault_tolerance_requirements
from tensorrt_llm.bindings.internal.runtime import unpack_payloads, unpack_payloads_into
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from .communicator import Distributed, ReduceOp, _check_reduce_op
from .membership import StoreGroup, all_ranks, has_rank, next_epoch, rank_mask
from .mnnvl_host_exchange import MnnvlHostExchangeGroup


class HostExchangeGroup(Protocol):
    """Transport of ``HostExchange``: rounds that move one slot per rank among the active ranks.

    A round returns one row of packets per sender, in the slot layout of ``mnnvl_host_exchange``. A
    backend may build them on receipt.
    """

    rank: int
    nranks: int
    # Payload packets per slot.
    words: int

    @classmethod
    def requirements(cls) -> list[tuple[bool, str]]:
        """Return the platform requirements as ``(met, description)`` pairs."""

    @classmethod
    def open(cls, mapping: Mapping, slot_payload_bytes: int) -> tuple[HostExchangeGroup, int]:
        """Open the TP group's transport collectively, and return it with the active rank mask."""

    def start_round(
        self, epoch: int, words: np.ndarray, length: int, recipients: Optional[int]
    ) -> None:
        """Stage this rank's slot of round ``epoch``, its payload for ``recipients`` (None: all)."""

    def wait(
        self,
        epoch: int,
        active: np.ndarray,
        readers: np.ndarray,
        idle: bool = False,
        received: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Send the staged slot and wait for the ``active`` ranks' headers and ``readers``' slots.

        Return the packets, or None once ``cancel`` ended the wait or ``received`` lacks a slot.
        """

    def cancel(self) -> None:
        """Make a running wait return, leaving the group usable. Safe from any thread."""

    @property
    def waiting_for(self) -> int:
        """Return the mask of ranks the current wait still lacks packets from. Thread-safe."""

    def end_wait(self) -> None:
        """Forget the ranks that the last wait lacked packets from."""

    def install(
        self,
        comm: StoreGroup,
        active_mask: int,
        incomplete: Optional[tuple[int, np.ndarray, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """Switch to the ranks in ``active_mask``, collectively over the active members of ``comm``.

        Afterwards no write of a former member reaches memory that an active rank reads. With
        ``incomplete``, the ``(epoch, readers, active)`` of a round another rank finished, return it
        completed. A retry after an install that raised returns the same round, which the group keeps
        until an install succeeds.
        """


class HostExchange:
    """All-to-all of byte strings between the host processes of a TP group.

    Every active rank gets the same result across a membership change. If a rank fails in a round
    that a survivor already finished, the other survivors complete it from the slots the survivors
    received. Otherwise the survivors run the round again without it.
    """

    # Sized so that a step's request broadcast and response gather usually take one round.
    DEFAULT_SLOT_PAYLOAD_BYTES = 64 * 1024

    def __init__(self, group: HostExchangeGroup):
        self.group = group
        self.rank = group.rank
        self.nranks = group.nranks
        self._words = group.words
        # The round counter. It wraps and skips 0, because the rows start zeroed.
        self._epoch = 0
        self.active_mask = all_ranks(self.nranks)
        self._interrupted: Callable[[], bool] = lambda: False
        self._on_interrupt: Optional[Callable[[], None]] = None
        self.stopped_epoch = 0
        # Active ranks still missing from a round that install() completed. The round's result
        # treats them as inactive.
        self._missing = 0
        # Readers whose payload this collective lacks. Their payloads come back as None.
        self._failed = 0
        # This collective's readers. Completing a round needs their whole slots and only the
        # headers from other ranks.
        self._round_readers = 0
        # Packets that install() completed from the peers' rows. The round takes them instead of
        # polling.
        self._finished_round: Optional[np.ndarray] = None
        # The epoch to continue from once the round takes _finished_round.
        self._epoch_after_round: Optional[int] = None
        # Cached _ranks_of results, keyed by mask.
        self._ranks_by_mask: dict[int, tuple[np.ndarray, list[int]]] = {}

    @classmethod
    def create(cls, mapping, group_type: type[HostExchangeGroup]) -> "HostExchange":
        """Create the exchange over ``group_type``, collectively over the TP group's ranks."""
        group, active_mask = group_type.open(mapping, cls.DEFAULT_SLOT_PAYLOAD_BYTES)
        exchange = cls(group)
        exchange.active_mask = active_mask
        return exchange

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_interrupt_handler(
        self, interrupted: Callable[[], bool], on_interrupt: Optional[Callable[[], None]] = None
    ) -> None:
        """Set the callbacks that stop the exchange for a membership change that needs this rank.

        ``interrupted`` reports such a change. The exchange checks it before each round and, during
        a wait, after each ``wake()``. When it holds, the exchange sets ``stopped_epoch`` to the
        current or next round's epoch and calls ``on_interrupt``, which runs the change and calls
        ``install()``. The exchange then takes the round that ``install()`` completed from the
        peers' rows, or runs the round again.
        """
        self._interrupted = interrupted
        self._on_interrupt = on_interrupt

    def admit(self, active_mask: int, epoch: int) -> None:
        """Take part with the ranks in ``active_mask``, starting with the round of ``epoch``."""
        self.active_mask = active_mask
        # Each round advances the epoch before it sends.
        self._epoch = epoch - 1

    def install(self, comm: StoreGroup, active_mask: int, epoch: int) -> None:
        """Switch to the ranks in ``active_mask`` while stopped, collectively over ``comm``.

        ``epoch`` is the first round after the change.
        On an active rank it shows how far this rank stopped behind the latest active rank. At
        ``stopped_epoch`` or the epoch after it, this rank runs the round again. Two epochs after
        it, another rank finished this rank's round, and this rank completes it from the peers'
        rows.

        Raises:
            RuntimeError: This rank stopped more than one round before the latest active rank.
        """
        incomplete = None
        if has_rank(active_mask, self.rank):
            if epoch in (self.stopped_epoch, next_epoch(self.stopped_epoch)):
                # An earlier attempt of this change may have finished the round with a since-failed
                # rank's packets.
                self._finished_round = self._epoch_after_round = None
            elif self._finished_round is None:
                if (
                    next_epoch(next_epoch(self.stopped_epoch)) != epoch
                    or self.stopped_epoch != self._epoch
                ):
                    raise RuntimeError(
                        f"host exchange stopped at epoch {self.stopped_epoch}, more than one round "
                        f"before the latest active rank (epoch {epoch} is next)"
                    )
                incomplete = (
                    self._epoch,
                    self._ranks_of(self._round_readers)[0],
                    self._ranks_of(self.active_mask)[0],
                )
        round_epoch = self._epoch
        completed = self.group.install(comm, active_mask, incomplete)
        if completed is not None:
            self._finished_round = completed
        self.admit(active_mask, epoch)
        if self._finished_round is not None:
            # Take the completed round at its own epoch, then continue from the installed epoch.
            self._epoch_after_round, self._epoch = self._epoch, round_epoch

    def exchange(
        self,
        payload: bytes,
        readers: Optional[Sequence[int]] = None,
        recipients: Optional[Sequence[int]] = None,
        idle: bool = False,
    ) -> list[Optional[bytes]]:
        """Send ``payload`` to every rank and return every rank's payload, in rank order.

        Args:
            payload: The bytes this rank sends.
            readers: The ranks whose payloads this rank needs (default all). The others come back as
                None, as do ranks outside ``active_mask`` and ranks whose payload a membership
                change left incomplete.
            recipients: The ranks that need this rank's payload (default all). It must include every
                rank that names this rank among its ``readers``.
            idle: Sleep between polls sooner, for waits that may last until the next request.
        """
        reader_mask = self.active_mask
        if readers is not None:
            reader_mask &= rank_mask(readers, self.nranks)
        if recipients is not None:
            recipients = rank_mask(recipients, self.nranks)
        ranks = self._ranks_of(reader_mask)[1]
        rounds = self._rounds(payload, reader_mask, recipients, idle)
        lengths, rows = next(rounds)
        received = self._chunks(rows, lengths, ranks)
        # A payload longer than a slot takes several rounds. Join each rank's chunks in order.
        remaining_chunks = [self._chunks(rows, lengths, ranks) for lengths, rows in rounds]
        if remaining_chunks:
            received = [b"".join(parts) for parts in zip(received, *remaining_chunks)]
        payloads = [None] * self.nranks
        for r, received_payload in zip(ranks, received):
            if not has_rank(self._failed, r):
                payloads[r] = received_payload
        return payloads

    def allgather_array(self, row: np.ndarray, active_only: bool = False) -> np.ndarray:
        """Return every rank's ``row`` as ``[nranks, row.size]``, with zero rows for inactive ranks.

        With ``active_only``, return only the rows that this exchange received.

        Raises:
            ValueError: The ranks passed rows of different sizes.
        """
        row = np.ascontiguousarray(row).reshape(-1)
        data = row.view(np.uint8)
        readers = self.active_mask
        ranks = self._ranks_of(readers)[1]
        out = np.zeros((self.nranks, data.size), dtype=np.uint8)
        step = 4 * self._words
        for k, (lengths, rows) in enumerate(self._rounds(data.tobytes(), readers)):
            if k == 0 and (lengths[self._ranks_of(readers & ~self._failed)[0]] != data.size).any():
                raise ValueError(
                    f"allgather_array rows differ in size across ranks: {lengths.tolist()}"
                )
            if rows is not None:
                unpack_payloads_into(rows, lengths, ranks, out, step * k)
        kept = readers & ~self._failed
        if active_only:
            out = out[self._ranks_of(kept)[0]]
        elif kept != all_ranks(self.nranks):
            out[~self._ranks_of(kept)[0]] = 0
        return out.view(row.dtype)

    def _ranks_of(self, mask: int) -> tuple[np.ndarray, list[int]]:
        """Return ``mask`` as a bool array and as a sorted rank list, cached per mask."""
        ranks = self._ranks_by_mask.get(mask)
        if ranks is None:
            in_mask = np.array([has_rank(mask, r) for r in range(self.nranks)], dtype=bool)
            ranks = self._ranks_by_mask[mask] = (in_mask, np.flatnonzero(in_mask).tolist())
        return ranks

    def _chunks(
        self, rows: Optional[np.ndarray], lengths: np.ndarray, ranks: list[int]
    ) -> list[bytes]:
        """Return the payload bytes of ``ranks`` in one round's packets."""
        if rows is None:
            return [b""] * len(ranks)
        return unpack_payloads(rows, lengths, ranks)

    def _rounds(
        self, data: bytes, readers: int, recipients: Optional[int] = None, idle: bool = False
    ) -> Iterator[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yield header lengths and packets per round, sending one slot of ``data`` per round."""
        self._round_readers = readers
        self._failed = 0
        step = 4 * self._words
        lengths, rows = self._round(data[:step], len(data), readers, recipients, idle)
        yield lengths, rows
        for k in range(1, ceil_div(int(lengths.max()), step)):
            chunk = data[step * k : step * (k + 1)]
            yield self._round(chunk, len(chunk), readers, recipients, idle)

    def _round(
        self,
        chunk: bytes,
        length: int,
        readers: int,
        recipients: Optional[int] = None,
        idle: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Exchange up to one slot per rank, running the round again after a membership change."""
        result = None
        while result is None:
            result = self._try_round(chunk, length, readers, recipients, idle)
        return result

    def _try_round(
        self, chunk: bytes, length: int, readers: int, recipients: Optional[int], idle: bool
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Return every rank's header length and the packets received, or None to run it again.

        The packets are None if no reader sent a payload.
        """
        if self._interrupted():
            self._stop(next_epoch(self._epoch))
        self._epoch = next_epoch(self._epoch)
        self._missing = 0
        words = np.frombuffer(chunk + b"\0" * (-len(chunk) % 4), dtype=np.uint32)
        self.group.start_round(self._epoch, words, length, recipients)
        round_mask = self.active_mask
        rows = self._wait(round_mask, readers & round_mask, idle)
        if rows is None:
            return None
        active = round_mask & ~self._missing
        lengths = self._header_lengths(rows, active)
        payload_bytes = int(lengths[self._ranks_of(readers & active)[0]].max(initial=0))
        packets = 1 + min(self._words, ceil_div(payload_bytes, 4))
        self._failed |= readers & (~round_mask | self._missing)
        if self._epoch_after_round is not None:
            self._epoch, self._epoch_after_round = self._epoch_after_round, None
            self._finished_round = None
        return lengths, (rows[:, :packets] if packets > 1 else None)

    def _header_lengths(self, headers: np.ndarray, active: int) -> np.ndarray:
        """Return the payload length in each rank's header, or 0 for a rank outside ``active``."""
        return headers.view(np.uint32)[:, 0] * self._ranks_of(active)[0]

    @property
    def waiting_for(self) -> int:
        """Return the mask of ranks the current wait still lacks packets from. Thread-safe."""
        return self.group.waiting_for

    @property
    def waited_epochs(self) -> tuple[tuple[int, int], ...]:
        """Return (rank, epoch) for each rank the current wait lacks packets from."""
        waiting_for, epoch = self.waiting_for, self._epoch
        return tuple(
            (rank, epoch) for rank in range(waiting_for.bit_length()) if has_rank(waiting_for, rank)
        )

    def wake(self) -> None:
        """Make a running wait return and check for a membership change. Safe from any thread."""
        self.group.cancel()

    def _stop(self, epoch: int) -> None:
        """Record ``epoch`` as ``stopped_epoch`` and call the interrupt handler."""
        self.group.end_wait()
        self.stopped_epoch = epoch
        if self._on_interrupt is not None:
            self._on_interrupt()

    def _take_finished_round(self, active: int, readers: int) -> np.ndarray:
        """Send the staged slot and return the round that install() finished, noting missing ranks."""
        rows = self._finished_round
        if (
            self.group.wait(
                self._epoch, self._ranks_of(active)[0], self._ranks_of(readers)[0], received=rows
            )
            is None
        ):
            self._missing |= self.group.waiting_for
            self.group.end_wait()
        return rows

    def _wait(self, active: int, readers: int, idle: bool) -> Optional[np.ndarray]:
        """Send the staged slot and wait for the ``active`` ranks' headers and ``readers``' payloads.

        Return the received slots, or None if a membership change restarts the round.
        """
        if self._finished_round is not None:
            return self._take_finished_round(active, readers)
        in_active, in_readers = self._ranks_of(active)[0], self._ranks_of(readers)[0]
        while (rows := self.group.wait(self._epoch, in_active, in_readers, idle)) is None:
            if self._interrupted():
                self._stop(self._epoch)
                if self._finished_round is not None:
                    return self._take_finished_round(active, readers)
                return None
        return rows


class MnnvlDist(Distributed):
    """``Distributed`` whose host collectives go through MNNVL memory, over the active ranks.

    Object gathers return None for inactive ranks, reductions skip them, and ``tp_allgather_int64``
    gives them zero rows. A gather with ``active_only`` returns the active ranks' values only. On a
    replacement rank the startup collectives return the original ranks' results from the startup
    record. It has no pipeline-parallel, context-parallel or point-to-point communication.
    """

    _REDUCE = {
        ReduceOp.SUM: lambda a, b: a + b,
        ReduceOp.PRODUCT: lambda a, b: a * b,
        ReduceOp.MIN: min,
        ReduceOp.MAX: max,
        ReduceOp.BAND: lambda a, b: a & b,
        ReduceOp.BOR: lambda a, b: a | b,
        ReduceOp.BXOR: lambda a, b: a ^ b,
    }

    # broadcast_or_none is a single host collective, so the request broadcaster skips its
    # request-count probe.
    supports_single_round_broadcast = True

    # The transport of the host exchange.
    group_type: type[HostExchangeGroup] = MnnvlHostExchangeGroup

    def __init__(self, mapping: Mapping, host_exchange_owner: Optional[MnnvlDist] = None):
        """Create the communicator of ``mapping``.

        Args:
            host_exchange_owner: The communicator whose host exchange this one uses, by default
                itself.
        """
        super().__init__(mapping)
        self._host_exchange = None
        self._host_exchange_owner = self if host_exchange_owner is None else host_exchange_owner

    @staticmethod
    def check_requirements(mapping: Mapping) -> None:
        """Raise ValueError naming every unmet ``MnnvlDist`` requirement."""
        checks = [
            (
                not mpi_disabled(),
                "the MPI orchestrator (TLLM_DISABLE_MPI unset), because a replacement rank cannot "
                "join a torch.distributed process group",
            ),
            (
                mapping.enable_attention_dp,
                "attention DP (enable_attention_dp=True), because a failed rank must take only its "
                "own requests with it",
            ),
            (
                mapping.pp_size == 1,
                "pipeline_parallel_size=1, because MnnvlDist has no pipeline-parallel group",
            ),
            (
                mapping.cp_size == 1,
                "context_parallel_size=1, because MnnvlDist has no context-parallel group",
            ),
            (
                mapping.tp_size == mapping.world_size,
                "tensor_parallel_size equal to the world size, because the host collectives run "
                "over a TP group that holds every rank",
            ),
        ]
        if all(met for met, _ in checks):
            checks += MnnvlDist.group_type.requirements()
        check_fault_tolerance_requirements(checks)

    @property
    def local_world_size(self):
        """Return the number of ranks on this node that share its free memory as this rank starts.

        An original rank counts all ranks on the node. A replacement rank counts only the node's
        ranks outside the membership, because the active ranks already hold their memory.
        ``local_mpi_size()`` still counts all ranks on the node.
        """
        mapping = self.mapping
        node_ranks = [
            rank
            for rank in range(self.world_size)
            if mapping.get_node_rank(rank) == mapping.node_rank
        ]
        startup_record = Distributed.startup_record
        if startup_record is not None and startup_record.replacement:
            # The startup module imports this one.
            from .membership.startup import membership_client

            record = membership_client().record
            node_ranks = [rank for rank in node_ranks if not record.is_active(rank)]
        return len(node_ranks)

    @property
    def host_exchange(self):
        """Return the ``HostExchange``, created collectively on first use.

        Raises:
            RuntimeError: The exchange this communicator shares serves another TP group.
        """
        if self._host_exchange is None:
            owner = self._host_exchange_owner
            exchange = (
                HostExchange.create(self.mapping, self.group_type)
                if owner is self
                else owner.host_exchange
            )
            if (exchange.rank, exchange.nranks) != (self.mapping.tp_rank, self.tp_size):
                raise RuntimeError(
                    f"Host exchange for TP rank {exchange.rank}/{exchange.nranks} cannot serve "
                    f"TP rank {self.mapping.tp_rank}/{self.tp_size}"
                )
            self._host_exchange = exchange
        return self._host_exchange

    @property
    def waited_epochs(self) -> tuple[tuple[int, int], ...]:
        """Return (rank, epoch) for each rank a running host collective lacks packets from.

        Before the first host collective, return an empty tuple without creating the exchange.
        """
        exchange = self._host_exchange_owner._host_exchange
        return () if exchange is None else exchange.waited_epochs

    def _exchange(
        self, obj, readers=None, recipients=None, idle=False, reduce_op=None, active_only=False
    ) -> list:
        """Exchange ``obj`` as ``HostExchange.exchange`` does, and unpickle the payloads.

        ``reduce_op`` (``min`` or ``max``) lets a replacement's startup replay check its own value
        against the original ranks' reduced result. ``active_only`` drops inactive ranks.
        """
        host_exchange = self.host_exchange
        payload = pickle.dumps(obj)

        def run():
            return host_exchange.exchange(payload, readers, recipients, idle)

        record = Distributed.startup_record
        if record is not None and record.active:
            payloads = record.exchange(self.mapping.tp_rank, payload, readers, run, reduce_op)
        else:
            payloads = run()
        return [
            None if payload is None else pickle.loads(payload)  # nosec B301
            for payload in payloads
            if payload is not None or not active_only
        ]

    def broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, prefer_cpu=False):
        # prefer_cpu marks a wait that may last until the next request arrives, so it runs as an
        # idle wait.
        if self.mapping.tp_rank == root:
            self._exchange(obj, readers=(), idle=prefer_cpu)
            return obj
        return self._exchange(None, readers=(root,), recipients=(), idle=prefer_cpu)[root]

    def broadcast_or_none(self, obj, root=0):
        return self.broadcast(obj, root)

    def allgather(self, obj):
        return self._exchange(obj)

    def barrier(self):
        self._exchange(None, readers=(), recipients=())

    def tp_barrier(self):
        self.barrier()

    def tp_allgather(
        self,
        obj,
        chunk_size: int = 4 * 1024 * 1024,
        *,
        small_payload: bool = False,
        active_only: bool = False,
    ):
        return self._exchange(obj, active_only=active_only)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        if self.mapping.tp_rank != root:
            self._exchange(obj, readers=(), recipients=(root,))
            return None
        return self._exchange(obj, recipients=(root,))

    def tp_broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, **kwargs):
        return self.broadcast(obj, root)

    def cp_broadcast(self, obj, root=0, **kwargs):
        raise NotImplementedError("MnnvlDist has no context-parallel group")

    def cp_allgather(self, obj, *, small_payload: bool = False):
        raise NotImplementedError("MnnvlDist has no context-parallel group")

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        _check_reduce_op(op)
        reduce_op = self._REDUCE[op] if op in (ReduceOp.MIN, ReduceOp.MAX) else None
        values = self._exchange(obj, reduce_op=reduce_op, active_only=True)
        return functools.reduce(self._REDUCE[op], values)

    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return self.allreduce(obj, op)

    def tp_allgather_int64(self, values: ArrayLike, *, active_only: bool = False) -> np.ndarray:
        host_exchange = self.host_exchange
        values = np.asarray(values, dtype=np.int64)
        record = Distributed.startup_record
        if record is None or not record.active:
            return host_exchange.allgather_array(values, active_only)
        # The original ranks were all active at their startup.
        rows = record.exchange(
            self.mapping.tp_rank,
            values.tobytes(),
            None,
            lambda: [row.tobytes() for row in host_exchange.allgather_array(values)],
            decode=lambda row: np.frombuffer(row, dtype=np.int64).tolist(),
        )
        return np.stack([np.frombuffer(row, dtype=np.int64) for row in rows])

    def broadcast_int64(
        self, values: ArrayLike, root: int = 0, *, prefer_cpu: bool = False
    ) -> np.ndarray:
        row = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
        return self.broadcast(row, root, prefer_cpu=prefer_cpu)

    def tp_cp_broadcast_int64(self, values: ArrayLike, root: int = 0) -> np.ndarray:
        return self.broadcast_int64(values, root)


class MnnvlDistFactory:
    """Create a process's ``MnnvlDist`` communicators for ``Distributed.get``.

    They share the first one's host exchange, so a membership change reaches all of them.
    """

    def __init__(self) -> None:
        self._first: Optional[MnnvlDist] = None

    def __call__(self, mapping: Mapping) -> MnnvlDist:
        """Return a ``MnnvlDist`` for ``mapping`` after checking its requirements."""
        MnnvlDist.check_requirements(mapping)
        dist = MnnvlDist(mapping, self._first)
        if self._first is None:
            self._first = dist
        return dist

    @property
    def waited_epochs(self) -> tuple[tuple[int, int], ...]:
        """Return (rank, epoch) for each rank a running host collective lacks packets from."""
        return () if self._first is None else self._first.waited_epochs

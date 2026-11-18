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
"""Rendezvous store communicators for Wide-EP fault tolerance."""

import hashlib
import pickle  # nosec B403
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

from tensorrt_llm._torch.modules.dwdp.rendezvous import StoreComm
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST

from ..communicator import Distributed
from ..startup_record import diverged
from .records import MAX_RANKS, REPORT_INTERVAL_S, rank_mask

if TYPE_CHECKING:
    from torch.distributed import Store


LIVENESS_KEY = "liveness"
RECORD_KEY = "record"
TRANSACTION_KEY = "last_transaction"
# The Unix time in milliseconds since which the active ranks have waited on a rank that makes no
# progress, or empty while no rank stalls them.
STALLED_SINCE_KEY = "stalled_since"


# Ranks per store counter in an arrival mask. The store's counters are signed 64-bit integers.
_RANKS_PER_COUNTER = 63


def rendezvous_host_port(address: str) -> Tuple[str, int]:
    """Split a rendezvous address ``host:port`` into its host and port."""
    host, port = address.rsplit(":", 1)
    return host, int(port)


def wait_for_keys(
    store: "Store", keys: List[str], check_s: float, check: Callable[[], None]
) -> List[bytes]:
    """Wait until ``keys`` are set and return their values.

    Call ``check`` every ``check_s`` while waiting. ``check`` may raise to stop waiting.
    """
    from torch.distributed import DistStoreError

    while True:
        try:
            store.wait(keys, timedelta(seconds=check_s))
            break
        except DistStoreError:
            check()
    return store.multi_get(keys)


class StoreGroup(StoreComm):
    """A subset of an mpi4py communicator of the global ranks ``ranks``, over the rendezvous store.

    Only the members in ``active`` take part, and the others' values are None. While a call waits,
    ``on_wait`` gets a mask of active members whose values are missing, and may raise to stop
    waiting. A replacement rank's ``startup`` group holds only itself until the change that admits
    it. Until its startup ends, a collective on that group that needs the other members' values
    raises, unless it gathers per-process values (``Distributed.rank_varying``).
    """

    def __init__(
        self,
        store: "Store",
        name: str,
        rank: int,
        ranks: Sequence[int],
        on_wait: Callable[[int], None] = lambda waiting: None,
        active: Optional[Sequence[int]] = None,
        gpus_per_node: int = 1,
        startup: bool = False,
    ):
        """Build the group's communicator over ``store``.

        Args:
            store: The rendezvous store client that holds the group's values.
            name: The key prefix of the group's calls, the same on every member.
            rank: This process's global rank, one of ``ranks``.
            ranks: The group's global ranks, in communicator rank order.
            on_wait: Called while a call waits, with a mask of the active members whose values
                are missing, and with 0 once the call has them.
            active: The members that take part, by default every member.
            gpus_per_node: The ranks per node, which ``Split_type`` groups by.
            startup: Whether this is the world communicator of a rank's startup, on which a
                replacement may not gather the original ranks' values.
        """
        self.membership = tuple(ranks)
        super().__init__(
            store,
            name,
            self.membership.index(rank),
            len(self.membership),
            timedelta(seconds=REPORT_INTERVAL_S),
        )
        self.rank = rank
        self.exchange_failed = False
        self._on_wait = on_wait
        self._active = tuple(r for r in self.membership if active is None or r in active)
        self._gpus_per_node = gpus_per_node
        self._startup = startup
        self._subgroups = 0

    def allgather(self, value: Any) -> List[Any]:
        """Return every member's value by communicator rank, with None for an inactive member.

        ``exchange_failed`` tells afterwards whether the exchange itself raised.

        Raises:
            ReplacementCannotStart: A replacement's startup gathers the original ranks' values.
        """
        self._check_alone_at_startup("allgather")
        return self._gather_values(value)

    def _gather_values(self, value: Any) -> List[Any]:
        from torch.distributed import DistError

        self.exchange_failed = False
        try:
            if self._active == (self.rank,):
                self._on_wait(0)
                value = pickle.loads(pickle.dumps(value))  # nosec B301
                return [value if r == self.rank else None for r in self.membership]
            try:
                return super().allgather(value)
            except DistError as error:
                raise RuntimeError(
                    f"Rank {self.rank} lost the rendezvous store, which the membership authority "
                    "hosts"
                ) from error
        except BaseException:
            self.exchange_failed = True
            raise

    def barrier(self) -> None:
        """Wait for the active members.

        A barrier gathers no values, so a replacement's startup may call it.
        """
        self._gather_values(None)

    Barrier = barrier

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """Return ``obj`` from communicator rank ``root``, or None if that rank is not active."""
        self._check_alone_at_startup("bcast")
        return self._gather_values(obj if self.Get_rank() == root else None)[root]

    def allreduce(self, obj: Any, op: Any = None) -> Any:
        """Reduce the active members' ``obj`` by ``MPI.SUM`` (default), ``MIN`` or ``MAX``."""
        from mpi4py import MPI

        self._check_alone_at_startup("allreduce")
        op = MPI.SUM if op is None else op
        for mpi_op, reduce in ((MPI.SUM, sum), (MPI.MIN, min), (MPI.MAX, max)):
            if op == mpi_op:
                return reduce(value for value in self._gather_values(obj) if value is not None)
        raise NotImplementedError(f"StoreGroup reduces by MPI.SUM, MPI.MIN or MPI.MAX, not {op}")

    def Free(self) -> None:
        """Do nothing. The group's exchanges share this process's store client."""

    def close(self, active: Sequence[int]) -> None:
        """Delete the keys of this group's calls, once the ranks ``active`` made their last call.

        Each call deletes this member's value of the call before it, so this member deletes its
        values and the arrival counters of its last two calls. It also deletes the keys of the
        members not in ``active``. Such a member failed and left the keys of its last two calls,
        which are within one call of this member's last call, as a member enters a call only after
        every active member set the call before it. Only a group that no rank in ``active`` built,
        having skipped its change, keeps keys: its failed members' first call values and counter.
        """
        self._delete_calls(self._seq - 2, self._seq, [self.Get_rank()], counters=True)
        failed_indices = [
            i for i, r in enumerate(self.membership) if r in self._active and r not in active
        ]
        if failed_indices:
            self._delete_calls(self._seq - 3, self._seq + 1, failed_indices, counters=True)

    def Abort(self, errorcode: int = 0) -> None:
        """End this process with ``errorcode``. The other members keep running."""
        # The startup module imports this one.
        from .startup import end_process

        end_process(self.rank, errorcode, "MPI_Abort on its communicator")

    def _check_alone_at_startup(self, collective: str) -> None:
        """Raise if a replacement's startup makes ``collective`` on its startup group.

        That group holds only the replacement, so the result would miss the original ranks' values.
        """
        if not self._startup or self._active != (self.rank,) or len(self.membership) == 1:
            return
        record = Distributed.startup_record
        if record is not None and record.replaying and not record.gathers_rank_varying:
            raise diverged(
                f"A replacement's startup calls {collective} on mpi_comm(), which needs the "
                "original ranks' values. Use a Distributed collective or Distributed.rank_varying"
            )

    def subgroup(self, ranks: Sequence[int]) -> "StoreGroup":
        """Return the subgroup of ``ranks``, which must include this rank.

        A replacement's subgroups hold only itself. Their exchanges are never startup collectives.
        """
        ranks = tuple(ranks)
        digest = hashlib.sha256(repr(ranks).encode()).hexdigest()[:16]
        self._subgroups += 1
        return StoreGroup(
            self._store,
            f"{self._name}/{self._subgroups}-{digest}",
            self.rank,
            ranks,
            self._on_wait,
            self._active,
            self._gpus_per_node,
        )

    def Split(self, color: int = 0, key: int = 0) -> "StoreGroup":
        """Split as ``MPI.Comm.Split`` does, a replacement taking the colors from the record.

        Raises:
            ReplacementCannotStart: A replacement's ``color`` or ``key`` differs from its
                predecessor's.
        """

        def gather():
            return self.allgather((color, key))

        # A process splits at whichever startup step first needs the communicator, even while the
        # startup record is paused, such as in the KV cache estimation that a replacement skips.
        splits = Distributed.agree_per_process(
            "startup_group_split", gather, own=(self.Get_rank(), (color, key))
        )
        ranks = sorted(
            (split[1], index, rank)
            for index, (rank, split) in enumerate(zip(self.membership, splits))
            if split[0] == color
        )
        return self.subgroup(rank for _, _, rank in ranks)

    def Split_type(self, split_type: int, key: int = 0, info: Any = None) -> "StoreGroup":
        """Return the members on this rank's node, supporting only the node split types."""
        from mpi4py import MPI

        if split_type not in (MPI.COMM_TYPE_SHARED, OMPI_COMM_TYPE_HOST):
            raise NotImplementedError(
                f"StoreGroup splits by node only, not by split type {split_type}"
            )
        node = self.rank // self._gpus_per_node
        return self.subgroup(r for r in self.membership if r // self._gpus_per_node == node)

    def _gather(self, prefix: str) -> List[Optional[bytes]]:
        """Return the active members' payloads for call ``prefix``, None for the others."""
        self._store.add(self._counter(prefix, self.rank), 1 << self.rank % _RANKS_PER_COUNTER)
        index = {r: i for i, r in enumerate(self.membership)}
        try:
            values = self._wait(prefix, [f"{prefix}/{index[r]}" for r in self._active])
        finally:
            self._on_wait(0)
        self._delete_calls(
            self._seq - 2, self._seq - 1, [self.Get_rank()], counters=self.rank == min(self._active)
        )
        by_rank = dict(zip(self._active, values))
        return [by_rank.get(r) for r in self.membership]

    def _delete_calls(self, first: int, end: int, indices: List[int], counters: bool) -> None:
        """Delete the values of the members ``indices`` in calls ``first`` to ``end`` (exclusive).

        Also delete the calls' arrival counters if ``counters``.
        """
        for seq in range(max(0, first), end):
            prefix = f"{self._name}/{seq}"
            for index in indices:
                self._store.delete_key(f"{prefix}/{index}")
            if counters:
                for key in self._counters(prefix):
                    self._store.delete_key(key)

    def _wait(self, prefix: str, keys: List[str]) -> List[bytes]:
        active_mask = rank_mask(self._active, MAX_RANKS)

        def check() -> None:
            arrived = sum(
                self._store.add(key, 0) << int(key.rsplit("/", 1)[1]) * _RANKS_PER_COUNTER
                for key in self._counters(prefix)
            )
            self._on_wait(active_mask & ~arrived)

        return wait_for_keys(self._store, keys, REPORT_INTERVAL_S, check)

    def _counters(self, prefix: str) -> List[str]:
        return sorted({self._counter(prefix, r) for r in self._active})

    @staticmethod
    def _counter(prefix: str, rank: int) -> str:
        return f"{prefix}/arrived/{rank // _RANKS_PER_COUNTER}"

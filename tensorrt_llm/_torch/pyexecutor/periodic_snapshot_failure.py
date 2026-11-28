# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side rendezvous for terminal periodic-snapshot TP errors."""

from __future__ import annotations

import time
from datetime import timedelta

import torch.distributed as dist

from ..distributed.communicator import MPI, Distributed, MPIDist, TorchDist

_SETUP_TIMEOUT = timedelta(seconds=30)


def _wait_for_mpi(request: MPI.Request, deadline: float) -> None:
    while time.monotonic() < deadline:
        if request.Test():
            return
        time.sleep(0.001)
    # A pending MPI nonblocking collective cannot be cancelled. The caller
    # must fail stop without freeing the group or returning to normal work.
    raise TimeoutError("Periodic snapshot TP failure control timed out")


class PeriodicSnapshotFailureGroup:
    """Keep failure rendezvous separate from scheduling and model collectives."""

    def __init__(self, group: dist.ProcessGroup | MPI.Comm, *, mpi: bool):
        self._group = group
        self._mpi = mpi

    @staticmethod
    def supported(communicator: Distributed) -> bool:
        # Torch's new_group is world-collective. Periodic TP supports one
        # entire worker world (which may be an MPI pool's private world).
        if not isinstance(communicator, (MPIDist, TorchDist)):
            return False
        if communicator.world_size != communicator.tp_size:
            return False
        if isinstance(communicator, MPIDist):
            return (
                MPI is not None
                and MPI.Is_initialized()
                and not MPI.Is_finalized()
                and MPI.Query_thread() == MPI.THREAD_MULTIPLE
            )
        return dist.is_gloo_available() and dist.get_world_size() == communicator.tp_size

    @classmethod
    def create(cls, communicator: Distributed) -> PeriodicSnapshotFailureGroup:
        """Called by every rank after the TP capability vote."""
        if isinstance(communicator, MPIDist):
            deadline = time.monotonic() + _SETUP_TIMEOUT.total_seconds()
            group, request = communicator.tp_comm.Idup()
            _wait_for_mpi(request, deadline)
            return cls(group, mpi=True)
        return cls(dist.new_group(backend="gloo", timeout=_SETUP_TIMEOUT), mpi=False)

    def arrive(self, deadline: float) -> None:
        """Confirm that all ranks reached this terminal phase before the deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Periodic snapshot TP failure control timed out")
        if self._mpi:
            _wait_for_mpi(self._group.Ibarrier(), deadline)
        else:
            dist.monitored_barrier(
                group=self._group, timeout=timedelta(seconds=remaining), wait_all_ranks=True
            )

    def close(self) -> None:
        """Release the group once all submitted rendezvous have completed."""
        if self._mpi:
            self._group.Free()
        else:
            dist.destroy_process_group(self._group)

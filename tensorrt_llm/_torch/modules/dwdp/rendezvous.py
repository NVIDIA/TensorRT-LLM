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
"""Launch-agnostic rendezvous for a DWDP group.

Serves DWDP's ``allgather`` / ``Barrier`` over a ``torch.distributed.TCPStore`` so each context
worker can be launched independently (for example one ``trtllm-serve`` process per worker, each
with a size-1 MPI world) instead of as ranks of one MPI world. Setting the first two variables
selects it; with neither set DWDP uses the MPI sub-communicator as before.

* ``TRTLLM_DWDP_RANK`` -- this worker's DWDP rank across all groups, ``0 .. num_groups * dwdp_size - 1``.
* ``TRTLLM_DWDP_MASTER_ADDR`` -- host of DWDP rank 0, which serves the store.
* ``TRTLLM_DWDP_MASTER_PORT`` -- store port, default ``29600``.
* ``TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S`` -- collective timeout, default ``1800``; must cover peers' model load.
"""

from __future__ import annotations

import os
import pickle  # nosec B403
from datetime import timedelta
from typing import Any, List, Optional

from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.logger import logger

DWDP_RANK_ENV = "TRTLLM_DWDP_RANK"
DWDP_MASTER_ADDR_ENV = "TRTLLM_DWDP_MASTER_ADDR"
DWDP_MASTER_PORT_ENV = "TRTLLM_DWDP_MASTER_PORT"
DWDP_TIMEOUT_ENV = "TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S"

DEFAULT_MASTER_PORT = 29600
DEFAULT_TIMEOUT_S = 1800


def store_rendezvous_enabled() -> bool:
    """Whether this launch asked DWDP to rendezvous over a TCPStore.

    Raises if only one of ``TRTLLM_DWDP_RANK`` / ``TRTLLM_DWDP_MASTER_ADDR`` is set: a rank taken
    from the environment must never be paired with the MPI sub-communicator, whose ranks it
    need not match.
    """
    has_rank = bool(os.environ.get(DWDP_RANK_ENV))
    has_addr = bool(os.environ.get(DWDP_MASTER_ADDR_ENV))
    if has_rank != has_addr:
        missing = DWDP_MASTER_ADDR_ENV if has_rank else DWDP_RANK_ENV
        raise ValueError(
            f"DWDP store rendezvous needs both {DWDP_RANK_ENV} and "
            f"{DWDP_MASTER_ADDR_ENV}; {missing} is not set. Set both to launch "
            f"context workers independently, or neither to use MPI."
        )
    return has_rank


def dwdp_global_rank() -> int:
    """This worker's DWDP rank across all groups.

    ``TRTLLM_DWDP_RANK`` under a store rendezvous, otherwise the global MPI rank.
    """
    if not store_rendezvous_enabled():
        return global_mpi_rank()
    rank = int(os.environ[DWDP_RANK_ENV])
    if rank < 0:
        raise ValueError(f"{DWDP_RANK_ENV} must be non-negative, got {rank}")
    return rank


class StoreDwdpGroup:
    """The ``allgather`` / ``Barrier`` / ``Free`` subset of an mpi4py communicator, over a TCPStore.

    Global DWDP rank 0 hosts the store for every group; keys are namespaced by group id, so
    groups never see each other's payloads.
    """

    def __init__(
        self,
        global_rank: int,
        dwdp_size: int,
        master_addr: str,
        master_port: int = DEFAULT_MASTER_PORT,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        store: Optional[Any] = None,
    ) -> None:
        if dwdp_size <= 0:
            raise ValueError(f"dwdp_size must be positive, got {dwdp_size}")
        if global_rank < 0:
            raise ValueError(f"global_rank must be non-negative, got {global_rank}")

        self._size = dwdp_size
        self._rank = global_rank % dwdp_size
        self._group_id = global_rank // dwdp_size
        self._seq = 0
        self._timeout = timedelta(seconds=timeout_s)

        if store is None:
            from torch.distributed import TCPStore

            # Clients retry until the master's server is up, bounded by the timeout.
            store = TCPStore(
                host_name=master_addr,
                port=master_port,
                is_master=global_rank == 0,
                timeout=self._timeout,
                wait_for_workers=False,
            )
        self._store = store

        # Two workers launched with the same TRTLLM_DWDP_RANK would overwrite each other's
        # payloads and silently pair the wrong expert slices; MPI rules this out by
        # construction, so check it here. ``add`` is atomic, so exactly one of them proceeds.
        joins = self._store.add(f"dwdp/{self._group_id}/join/{self._rank}", 1)
        if joins != 1:
            self._store = None
            raise RuntimeError(
                f"DWDP rank {global_rank} joined the store rendezvous {joins} times; "
                f"every context worker needs a distinct {DWDP_RANK_ENV}"
            )

        logger.info(
            f"[DWDP] store rendezvous ready: global_rank={global_rank} "
            f"group={self._group_id} rank_in_group={self._rank}/{dwdp_size} "
            f"master={master_addr}:{master_port}"
        )

    @classmethod
    def from_env(cls, global_rank: int, dwdp_size: int) -> "StoreDwdpGroup":
        """Build the group from ``TRTLLM_DWDP_*``; see the module docstring."""
        master_addr = os.environ.get(DWDP_MASTER_ADDR_ENV)
        if not master_addr:
            raise RuntimeError(
                f"{DWDP_MASTER_ADDR_ENV} must name the host of DWDP rank 0 to "
                f"rendezvous over a store"
            )
        return cls(
            global_rank=global_rank,
            dwdp_size=dwdp_size,
            master_addr=master_addr,
            master_port=int(os.environ.get(DWDP_MASTER_PORT_ENV, DEFAULT_MASTER_PORT)),
            timeout_s=int(os.environ.get(DWDP_TIMEOUT_ENV, DEFAULT_TIMEOUT_S)),
        )

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def allgather(self, obj: Any) -> List[Any]:
        """Gather one picklable object per rank, ordered by rank in the group."""
        if self._store is None:
            raise RuntimeError("allgather on a freed DWDP group")

        # Keys are unique per (group, call); every rank must issue the same sequence of collectives.
        prefix = f"dwdp/{self._group_id}/{self._seq}/"
        self._seq += 1
        keys = [f"{prefix}{r}" for r in range(self._size)]

        self._store.set(keys[self._rank], pickle.dumps(obj))
        self._store.wait(keys, self._timeout)
        # Trust boundary: the store is served by DWDP rank 0 and written only by the
        # context workers of this launch; its port must stay on the trusted cluster
        # network, as for the MPI communicator this replaces.
        return [pickle.loads(self._store.get(key)) for key in keys]  # nosec B301

    def Barrier(self) -> None:
        """Match ``MPI_Barrier``; waiting for every key is the synchronization."""
        self.allgather(None)

    def Free(self) -> None:
        """Match ``MPI_Comm_free``: drop the store. Idempotent.

        ``TCPStore`` has no explicit close; releasing the last reference closes this worker's
        client connection and, on DWDP rank 0, stops the server along with its keys.
        """
        store, self._store = self._store, None
        del store

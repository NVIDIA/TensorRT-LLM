# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
worker can be an independent ``trtllm-serve`` process; set the first two variables to select it.

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
    """Whether this launch asked DWDP to rendezvous over a TCPStore."""
    return bool(os.environ.get(DWDP_RANK_ENV)) and bool(os.environ.get(DWDP_MASTER_ADDR_ENV))


def dwdp_global_rank() -> int:
    """This worker's DWDP rank across all groups: ``TRTLLM_DWDP_RANK`` if set, else the MPI rank."""
    rank_env = os.environ.get(DWDP_RANK_ENV)
    if not rank_env:
        return global_mpi_rank()
    rank = int(rank_env)
    if rank < 0:
        raise ValueError(f"{DWDP_RANK_ENV} must be non-negative, got {rank}")
    return rank


class StoreDwdpGroup:
    """The ``allgather`` / ``Barrier`` subset of an mpi4py communicator, backed by a TCPStore."""

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

            # Global rank 0 hosts the store for all groups; clients retry until it is up.
            store = TCPStore(
                host_name=master_addr,
                port=master_port,
                is_master=global_rank == 0,
                timeout=self._timeout,
                wait_for_workers=False,
            )
        self._store = store

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
        return [pickle.loads(self._store.get(key)) for key in keys]  # nosec B301

    def Barrier(self) -> None:
        """Match ``MPI_Barrier``; waiting for every key is the synchronization."""
        self.allgather(None)

    def Free(self) -> None:
        """Match ``MPI_Comm_free``. Keys are tiny and die with the host process."""
        self._store = None

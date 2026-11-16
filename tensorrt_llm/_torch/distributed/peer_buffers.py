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
"""One buffer per rank of a tensor-parallel group, writable by every rank of the group.

``peer_ptrs[r]`` is rank ``r``'s buffer in this rank's address space, so a low-latency exchange
written against peer pointers runs the same within a node (CUDA IPC) and across the nodes of one
NVLink domain (unicast fabric memory: these exchanges use only the peer mappings, so they take no
NVLS multicast object, a rack-wide NVSwitch resource).
"""

from __future__ import annotations

import enum
from typing import Optional, Sequence

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


class PeerTransport(enum.Enum):
    """How the ranks of a group map each other's buffers."""

    IPC = "ipc"
    MNNVL = "mnnvl"


class PeerBuffers:
    """``nbytes`` per rank, zero-filled on every rank before :meth:`create` returns on any rank.

    The ranks agree on setup failures: then :meth:`create` returns None on every rank.
    """

    def __init__(
        self,
        local_ptr: int,
        peer_ptrs: Sequence[int],
        nbytes: int,
        transport: PeerTransport,
        keepalive: object,
    ):
        self.local_ptr = local_ptr
        self.peer_ptrs = list(peer_ptrs)
        self.nbytes = nbytes
        self.transport = transport
        # The allocation and its peer mappings live as long as these buffers.
        self._keepalive = keepalive

    @staticmethod
    def _group_transport(mapping: Mapping) -> Optional[PeerTransport]:
        """IPC within a node, MNNVL across the nodes of one NVLink domain, else None."""
        from tensorrt_llm._ipc_utils import can_access_peer

        from .ops import MNNVLAllReduce

        if can_access_peer(mapping):
            return PeerTransport.IPC
        if mapping.is_multi_node() and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16):
            return PeerTransport.MNNVL
        return None

    @classmethod
    def create(
        cls, mapping: Mapping, nbytes: int, transport: Optional[PeerTransport] = None
    ) -> Optional["PeerBuffers"]:
        """The buffers of ``mapping``'s TP group, or None where the group has no peer transport or
        their setup failed on any rank.

        Collective over the TP group: every rank calls it at the same point. ``transport``
        overrides the group's own (IPC within a node, MNNVL across nodes).
        """
        if transport is None:
            transport = cls._group_transport(mapping)
        if transport is PeerTransport.IPC:
            return cls._create_ipc(mapping, nbytes)
        if transport is PeerTransport.MNNVL:
            return cls._create_mnnvl(mapping, nbytes)
        return None

    @classmethod
    def _create_ipc(cls, mapping: Mapping, nbytes: int) -> Optional["PeerBuffers"]:
        from tensorrt_llm._ipc_utils import (
            IpcMemory,
            _ipc_mem_handle_from_bytes,
            _ipc_mem_handle_to_bytes,
            _raise_if_error,
            cudart,
        )

        from .communicator import Distributed

        if mapping.tp_size > mapping.gpus_per_node:
            return None
        dist = Distributed.get(mapping)
        # A holder only: when collected it frees peer_ptrs[tp_rank] and closes the other peer_ptrs.
        ipc = IpcMemory(mapping, nbytes, False)
        ipc.open_ipc = True
        handle = None
        try:
            # 2 MiB-aligned like IpcMemory: smaller allocations can share an IPC handle.
            size = (nbytes + (1 << 21) - 1) // (1 << 21) * (1 << 21)
            error, ptr = cudart.cudaMalloc(size)
            _raise_if_error(error)
            ipc.local_ptr = ipc.peer_ptrs[mapping.tp_rank] = ptr
            _raise_if_error(cudart.cudaMemset(ptr, 0, size)[0])
            error, local_handle = cudart.cudaIpcGetMemHandle(ptr)
            _raise_if_error(error)
            handle = _ipc_mem_handle_to_bytes(local_handle)
        except RuntimeError as e:
            logger.warning(f"IPC peer buffers failed: {e}")
        # A rank whose allocation failed sends None in place of its handle: then no rank opens any.
        handles = dist.tp_allgather(handle)
        if None in handles:
            return None
        ok = True
        try:
            for r, peer_handle in enumerate(handles):
                if r != mapping.tp_rank:
                    error, ptr = cudart.cudaIpcOpenMemHandle(
                        _ipc_mem_handle_from_bytes(peer_handle),
                        cudart.cudaIpcMemLazyEnablePeerAccess,
                    )
                    _raise_if_error(error)
                    ipc.peer_ptrs[r] = ptr
        except RuntimeError as e:
            logger.warning(f"IPC peer buffers failed: {e}")
            ok = False
        # Opening a peer handle can fail on one rank only: every rank falls back together.
        if not all(dist.tp_allgather(ok)):
            return None
        # No rank writes before every rank's allocation is zeroed.
        torch.cuda.synchronize()
        dist.tp_barrier()
        return cls(ipc.local_ptr, ipc.peer_ptrs, nbytes, PeerTransport.IPC, ipc)

    @classmethod
    def _create_mnnvl(cls, mapping: Mapping, nbytes: int) -> Optional["PeerBuffers"]:
        from .mnnvl_memory import TpMnnvlMemory

        # An mpi4py communicator or, without MPI, a ProcessGroupComm: only the calls both have.
        comm = TpMnnvlMemory.get_comm(mapping)
        try:
            mem = TpMnnvlMemory(mapping, nbytes)
            mem.as_torch_strided_tensor(torch.uint8)[mapping.tp_rank].zero_()
        except RuntimeError as e:
            logger.warning(f"MNNVL peer buffers failed: {e}")
            mem = None
        # Importing or mapping a peer handle can fail on one rank only: every rank falls back together.
        if not all(comm.allgather(mem is not None)):
            return None
        torch.cuda.synchronize()
        # Every rank zeroed its buffer before any rank writes into it.
        comm.barrier()
        peers = [mem.ptr + r * mem.rank_stride for r in range(mapping.tp_size)]
        return cls(peers[mapping.tp_rank], peers, nbytes, PeerTransport.MNNVL, mem)


__all__ = ["PeerBuffers", "PeerTransport"]

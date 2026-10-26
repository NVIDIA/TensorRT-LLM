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
NVLink domain (fabric memory, the allocation MNNVL all-reduce uses).
"""

from __future__ import annotations

import enum
from typing import Optional, Sequence

import torch

from tensorrt_llm.mapping import Mapping


class PeerTransport(enum.Enum):
    """How the ranks of a group map each other's buffers."""

    IPC = "ipc"
    MNNVL = "mnnvl"


class PeerBuffers:
    """``nbytes`` per rank, zero-filled on every rank before :meth:`create` returns on any rank."""

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
        """The buffers of ``mapping``'s TP group, or None where the group has no peer transport.

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
        from tensorrt_llm._ipc_utils import IpcMemory

        from .communicator import Distributed

        ipc = IpcMemory(mapping, nbytes, True)
        if not ipc.open_ipc:
            return None
        # IpcMemory zero-fills its allocation; no rank writes before every rank has.
        torch.cuda.synchronize()
        Distributed.get(mapping).tp_barrier()
        return cls(ipc.local_ptr, ipc.peer_ptrs, nbytes, PeerTransport.IPC, ipc)

    @classmethod
    def _create_mnnvl(cls, mapping: Mapping, nbytes: int) -> "PeerBuffers":
        from .ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        comm = _get_mnnvl_workspace_comm(mapping)
        handle, views, error = None, None, None
        try:
            handle = _make_mnnvl_mcast_buffer(comm, nbytes, mapping, True)
            views = [
                handle.get_uc_buffer(r, (nbytes,), torch.uint8, 0) for r in range(mapping.tp_size)
            ]
            views[mapping.tp_rank].zero_()
            torch.cuda.synchronize()
        except Exception as e:  # converged below, so no rank waits on a failed peer
            error = e
        # Collective: every rank zeroed its buffer before any rank writes into it.
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            raise RuntimeError("MNNVL peer buffers failed on a TP rank") from error
        return cls(
            views[mapping.tp_rank].data_ptr(),
            [view.data_ptr() for view in views],
            nbytes,
            PeerTransport.MNNVL,
            (handle, comm),
        )


__all__ = ["PeerBuffers", "PeerTransport"]

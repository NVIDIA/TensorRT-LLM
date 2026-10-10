# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned state of the MNNVL one-shot collectives: one multicast workspace of a TP group.

A state type, not an entry: it launches nothing per call. Its constructor is collective and eager, the
target builds one in ``post_load_weights`` (before any CUDA-graph capture) and passes it to every MNNVL
one-shot op of that group (``comm/mnnvl_allreduce_attn_res``, and ``trtllm::mnnvl_fusion_allreduce`` given
the same ``comm_buffer`` / ``buffer_flags``). The contract is the ``## State`` section of
``mnnvl_allreduce_attn_res.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

NUM_LAMPORT_BUFFERS = 3
FLAG_WORDS = 9
"""``buffer_flags``: [current buffer, dirty buffer, bytes per buffer, dirty stages, bytes to clear x 4,
access count]."""


def mnnvl_tp_group_comm(mapping):
    """A new communicator of exactly mapping.tp_group (its rank i = TP rank i) for a caller-owned
    MNNVL state's create(); only the group's ranks take part (MPI_Comm_create_group). The caller
    frees it. Under Ray the TP ProcessGroup (c10d's).
    """
    from tensorrt_llm._utils import mpi_comm, mpi_disabled

    if mpi_disabled():
        pg = mapping.tp_group_pg
        assert pg is not None, "TP ProcessGroup not initialised"
        return pg
    session = mpi_comm()
    session_group = session.Get_group()
    group = session_group.Incl(mapping.tp_group)
    session_group.Free()
    comm = session.Create_group(group)
    group.Free()
    return comm


@dataclass(eq=False)
class MnnvlWorkspace:
    """One TP group's MNNVL Lamport workspace: three buffers of ``buffer_bytes`` behind one multicast mapping, and
    the flag words that rotate them. Every MNNVL call on it takes the next buffer, so all of a group's ranks must
    make the same calls on it in the same order (see the contract's ``## State``)."""

    lamport: torch.Tensor
    """fp32 [3 * buffer_bytes / 4]: this rank's unicast view of the three buffers (every word -0.0 when armed)."""
    buffer_flags: torch.Tensor
    """uint32 [9], see ``FLAG_WORDS``; read and advanced by every call."""
    buffer_bytes: int
    rank: int
    world_size: int
    handle: Any
    """The ``McastGPUBuffer`` that owns the memory; the workspace is valid while this object lives."""
    comm: Any
    """The TP-group communicator the handles were exchanged over."""

    def comm_buffer(self, dtype: torch.dtype) -> torch.Tensor:
        """The three buffers as the ops take them: ``dtype`` [3, buffer_bytes / itemsize] (a view, no copy)."""
        return self.lamport.view(dtype).view(NUM_LAMPORT_BUFFERS, -1)

    def max_one_shot_tokens(self, hidden: int, dtype: torch.dtype = torch.bfloat16) -> int:
        """The most tokens of ``hidden`` columns a one-shot call fits in one buffer."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        return self.buffer_bytes // (hidden * self.world_size * itemsize)

    @classmethod
    def create(
        cls, mapping, buffer_bytes: int, fabric_handle: Optional[bool] = None
    ) -> "MnnvlWorkspace":
        """Allocate and arm a workspace for ``mapping``'s TP group. Collective over that group only: every rank of
        the group calls it at the same point, eagerly (not under CUDA-graph capture); ranks outside it take no part.

        Failure model: before allocating, the ranks agree that each of them can (not capturing, a valid
        ``buffer_bytes``, the three buffers within its device's free memory); if one cannot, every rank raises
        ``RuntimeError``, none allocates, and under MPI each frees the communicator it made for the call. A failure
        that returns from the allocation is agreed and handled the same way. A rank that fails inside the
        allocation's handle exchange can leave its peers waiting in that exchange: that failure is not turned into
        an error on the other ranks.

        ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file
        descriptor; default ``mapping.is_multi_node()``."""
        from tensorrt_llm._torch.distributed.ops import (
            _initialize_allreduce_mnnvl_protocol,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )
        from tensorrt_llm._utils import mpi_disabled

        use_fabric = mapping.is_multi_node() if fabric_handle is None else bool(fabric_handle)
        total = NUM_LAMPORT_BUFFERS * buffer_bytes
        comm = mnnvl_tp_group_comm(mapping)
        # Every condition one rank alone can fail is checked before the allocation, and the ranks agree on it: a
        # rank failing inside the allocation would leave its peers in the handle exchange.
        problem: Optional[str] = None
        if torch.cuda.is_current_stream_capturing():
            problem = "it is collective and allocates: call it before capture"
        elif buffer_bytes <= 0 or buffer_bytes % 16:
            problem = f"buffer_bytes must be a positive multiple of 16, got {buffer_bytes}"
        else:
            free_bytes, _ = torch.cuda.mem_get_info(_mnnvl_device_index(mapping))
            if free_bytes < total:
                problem = f"its {total} bytes exceed the {free_bytes} free on this rank's device"
        if not _mnnvl_workspace_all_succeeded(comm, problem is None):
            # Every rank of the group takes this path: free the MPI communicator made above (a ProcessGroup is
            # c10d's).
            if not mpi_disabled():
                comm.Free()
            raise RuntimeError(
                f"MnnvlWorkspace.create: not every rank can allocate ({problem or 'another rank cannot'})"
            )
        error: Optional[Exception] = None
        workspace = None
        try:
            handle = _make_mnnvl_mcast_buffer(comm, total, mapping, use_fabric)
            lamport = handle.get_uc_buffer(mapping.tp_rank, (total // 4,), torch.float32, 0)
            flags = torch.zeros(FLAG_WORDS, dtype=torch.uint32, device=lamport.device)
            workspace = cls(
                lamport=lamport,
                buffer_flags=flags,
                buffer_bytes=buffer_bytes,
                rank=mapping.tp_rank,
                world_size=mapping.tp_size,
                handle=handle,
                comm=comm,
            )
        except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
            error = exc
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            # Every rank takes this path too. A handle only borrows the communicator and makes no MPI call when
            # destroyed, so it may outlive the communicator.
            if not mpi_disabled():
                comm.Free()
            raise RuntimeError("MnnvlWorkspace: allocation failed on at least one rank") from error
        # Arms every buffer word and the flags; also the barrier after which a peer may push into this rank.
        _initialize_allreduce_mnnvl_protocol(
            dict(
                uc_buffer=workspace.lamport,
                buffer_flags=workspace.buffer_flags,
                buffer_size_bytes=buffer_bytes,
                comm=comm,
            )
        )
        return workspace

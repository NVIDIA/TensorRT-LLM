# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multicast (MNNVL) memory of a TP group for the Kimi K3 family: the communicator, the allocation, the checks
around it, and the Lamport workspace the MNNVL one-shot collectives take.

Not an operation: nothing here launches a kernel per call. The family's ops and kernels that run collectives over
multicast memory build their state from these functions, once, eagerly, before any CUDA-graph capture:

* ``mnnvl_tp_group_comm``: the communicator the handles are exchanged over. Under MPI a new communicator of exactly
  the TP group's ranks (``MPI_Comm_create_group``), ranked by TP rank; the caller owns it and frees it with
  ``free_group_comm``. Without MPI (Ray) the mapping's TP process group, which needs a CPU (gloo) backend.
  ``MnnvlWorkspace`` and the family kernels' multicast states (``k3_fused_moe``, ``k3_sandwich``) use it.
* ``group_comm`` and ``use_fabric_handle``: the communicator and handle rule the built-in MNNVL all-reduce uses (a
  split of the session communicator, collective over every rank of the session; fabric handles across nodes or with
  ``TRTLLM_FORCE_MNNVL_AR=1``). Target A's ``k3_markov`` and ``k3_spec_accept`` workspaces use them.
* ``make_mcast_buffer``: the ``McastGPUBuffer`` that owns the memory. ``get_uc_buffer`` gives this rank's view,
  ``get_mc_buffer`` the multicast view. The memory lives as long as the handle.
* ``all_ranks_succeeded``: the agreement every step of a collective setup needs. A rank that skips a step while
  its peers go ahead leaves them waiting in a handle exchange that never completes.
* ``device_index``: the device the buffers live on.
* ``arm_lamport_buffers``: the reset of a three-buffer Lamport workspace (every word -0.0, the flag words
  initialized), the layout the MNNVL all-reduce kernels read.
* ``MnnvlWorkspace``: a caller-owned Lamport workspace of one TP group, built from the steps above. The target
  builds one in ``post_load_weights`` and passes it to every MNNVL one-shot op of that group
  (``ops/comm/mnnvl_allreduce_attn_res``, ``ops/comm/mnnvl_fusion_allreduce``, ``ops/comm/mnnvl_allgather_split``).
* ``shared_lamport_workspace``: the one workspace per mapping the built-in MNNVL all-reduce owns, grown on demand.
  A caller that passes its buffers to an MNNVL all-reduce kernel shares that kernel's Lamport rotation with every
  other user of the mapping.
* ``mnnvl_supported`` and ``one_shot_threshold_bytes``: the built-in all-reduce's decision to use MNNVL for a
  mapping, and the message size up to which its MNNVL kernel runs one-shot.

Every step that allocates or exchanges is collective over the TP group: every rank of the group calls it at the
same point, in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

NUM_LAMPORT_BUFFERS = 3
"""The MNNVL all-reduce workspace's buffer count: each call takes the next one and clears the one before."""
FLAG_WORDS = 9
"""``buffer_flags``: [current buffer, dirty buffer, bytes per buffer, dirty stages, bytes to clear x 4,
access count]."""


def mnnvl_tp_group_comm(mapping):
    """A new communicator of exactly mapping.tp_group (its rank i = TP rank i) for a caller-owned
    MNNVL state's create(); only the group's ranks take part (MPI_Comm_create_group). The caller
    frees it (``free_group_comm``). Under Ray the TP ProcessGroup (c10d's).
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


def group_comm(mapping) -> Any:
    """The communicator of ``mapping``'s TP group as the built-in MNNVL all-reduce makes it, rank i = TP rank i.

    Under MPI a new split of the session communicator (collective over the whole session: every rank calls it),
    owned by the caller (``free_group_comm``). Without MPI the mapping's TP process group (c10d's).
    """
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

    return _get_mnnvl_workspace_comm(mapping)


def use_fabric_handle(mapping) -> bool:
    """The built-in MNNVL all-reduce's handle rule for ``mapping``: fabric handles when the TP group spans nodes, or
    with ``TRTLLM_FORCE_MNNVL_AR=1``; else POSIX file descriptors. ``MnnvlWorkspace.create`` defaults to
    ``mapping.is_multi_node()`` instead."""
    from tensorrt_llm._torch.distributed.ops import use_mnnvl_fabric_handle

    return use_mnnvl_fabric_handle(mapping)


def free_group_comm(comm) -> None:
    """Free a communicator from ``mnnvl_tp_group_comm`` or ``group_comm`` under MPI. A process group belongs to c10d:
    nothing to free."""
    from tensorrt_llm._utils import mpi_disabled

    if not mpi_disabled():
        comm.Free()


def device_index(mapping) -> int:
    """The CUDA device of this rank's multicast buffers: the mapping's local rank under MPI, the current device
    without MPI (Ray workers select their device themselves)."""
    from tensorrt_llm._torch.distributed.ops import _mnnvl_device_index

    return _mnnvl_device_index(mapping)


def all_ranks_succeeded(comm, local_success: bool) -> bool:
    """Whether every rank of ``comm`` (from ``mnnvl_tp_group_comm``) passed ``local_success`` true.

    Collective over ``comm``. Under MPI an integer all-reduce; without MPI an all-reduce on the process group's CPU
    backend, which also works while the model is built under meta-device initialization.
    """
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return _mnnvl_workspace_all_succeeded(comm, local_success)


def make_mcast_buffer(comm, size_bytes: int, mapping, fabric_handle: bool) -> Any:
    """Allocate ``size_bytes`` per rank of multicast memory over ``mapping``'s TP group and exchange the handles
    over ``comm`` (from ``mnnvl_tp_group_comm``). Returns the ``McastGPUBuffer``: ``get_uc_buffer(tp_rank, shape,
    dtype, offset)`` is this rank's view, ``get_mc_buffer(shape, dtype, offset)`` the multicast view.
    ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file descriptor.

    Collective over the TP group. A rank that fails inside the exchange can leave its peers waiting there: agree
    beforehand (``all_ranks_succeeded``) on every condition one rank alone can fail.
    """
    from tensorrt_llm._torch.distributed.ops import _make_mnnvl_mcast_buffer

    return _make_mnnvl_mcast_buffer(comm, size_bytes, mapping, fabric_handle)


def arm_lamport_buffers(
    uc_buffer: torch.Tensor, buffer_flags: torch.Tensor, buffer_bytes: int, comm
) -> None:
    """Reset a three-buffer Lamport workspace for the MNNVL all-reduce kernels: every word of ``uc_buffer`` (this
    rank's view of ``NUM_LAMPORT_BUFFERS`` buffers of ``buffer_bytes``) to -0.0, the ``FLAG_WORDS`` words of
    ``buffer_flags`` to [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0]; then synchronize the device.

    Collective over ``comm``: it also fences the setup, so no rank writes into a peer's buffers before the peer has
    reset them. Raises on every rank if the reset failed on any.
    """
    from tensorrt_llm._torch.distributed.ops import _initialize_allreduce_mnnvl_protocol

    _initialize_allreduce_mnnvl_protocol(
        dict(
            uc_buffer=uc_buffer,
            buffer_flags=buffer_flags,
            buffer_size_bytes=buffer_bytes,
            comm=comm,
        )
    )


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
            free_bytes, _ = torch.cuda.mem_get_info(device_index(mapping))
            if free_bytes < total:
                problem = f"its {total} bytes exceed the {free_bytes} free on this rank's device"
        if not all_ranks_succeeded(comm, problem is None):
            # Every rank of the group takes this path: free the MPI communicator made above (a ProcessGroup is
            # c10d's).
            free_group_comm(comm)
            raise RuntimeError(
                f"MnnvlWorkspace.create: not every rank can allocate ({problem or 'another rank cannot'})"
            )
        error: Optional[Exception] = None
        workspace = None
        try:
            handle = make_mcast_buffer(comm, total, mapping, use_fabric)
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
        if not all_ranks_succeeded(comm, error is None):
            # Every rank takes this path too. A handle only borrows the communicator and makes no MPI call when
            # destroyed, so it may outlive the communicator.
            free_group_comm(comm)
            raise RuntimeError("MnnvlWorkspace: allocation failed on at least one rank") from error
        # Arms every buffer word and the flags; also the barrier after which a peer may push into this rank.
        arm_lamport_buffers(workspace.lamport, workspace.buffer_flags, buffer_bytes, comm)
        return workspace


@dataclass(frozen=True, eq=False)
class SharedLamportWorkspace:
    """A view of the mapping's shared MNNVL all-reduce workspace, as the MNNVL all-reduce kernels take it."""

    uc_buffer: torch.Tensor
    """fp32 [NUM_LAMPORT_BUFFERS * buffer_bytes / 4]: this rank's view of the three buffers."""
    buffer_flags: torch.Tensor
    """uint32 [FLAG_WORDS]: the rotation's flag words, read and advanced by every call."""
    buffer_bytes: int
    """Bytes of one Lamport buffer."""
    world_size: int
    """The TP group's size: the ranks the multicast memory spans."""

    def comm_buffer(self, dtype: torch.dtype) -> torch.Tensor:
        """The three buffers as the kernels take them: ``dtype`` [3, buffer_bytes / itemsize] (a view)."""
        return self.uc_buffer.view(dtype).view(NUM_LAMPORT_BUFFERS, -1)


def shared_lamport_workspace(
    mapping, dtype: torch.dtype, buffer_bytes: Optional[int] = None
) -> SharedLamportWorkspace:
    """The mapping's shared MNNVL all-reduce workspace, created or grown so one buffer holds ``buffer_bytes``.

    The built-in MNNVL all-reduce creates and grows it, over its own communicator (a split of the session
    communicator) and with its own handle rule (fabric handles across nodes or with ``TRTLLM_FORCE_MNNVL_AR=1``).
    The first call for a mapping creates it at ``max(1024 * 8192 * itemsize(dtype), buffer_bytes)``; a later call
    that needs more replaces it with one rounded up to 8 MiB. Creating or growing is collective over the TP group
    and cannot run under CUDA-graph capture: make the largest call of each shape once, eagerly, first. A workspace
    a captured graph used stays alive after it is replaced. A call that needs no growth is local.
    """
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping, dtype, buffer_size_bytes=buffer_bytes
    )
    return SharedLamportWorkspace(
        uc_buffer=workspace["uc_buffer"],
        buffer_flags=workspace["buffer_flags"],
        buffer_bytes=workspace["buffer_size_bytes"],
        world_size=mapping.tp_size,
    )


def has_shared_lamport_workspace(mapping) -> bool:
    """Whether ``mapping``'s shared MNNVL all-reduce workspace exists (some MNNVL all-reduce of it was built)."""
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    return MNNVLAllReduce.allreduce_mnnvl_workspaces.get(mapping) is not None


def mnnvl_supported(mapping, dtype: torch.dtype, explicitly_requested: bool = False) -> bool:
    """Whether the built-in all-reduce runs ``mapping``'s TP group over MNNVL: a supported dtype (fp16, bf16, fp32),
    no context parallelism, an aarch64 host whose driver supports multicast memory, and either a group that spans
    nodes or ``explicitly_requested`` (the MNNVL strategy named rather than AUTO). ``TLLM_TEST_MNNVL=1`` forces it
    on."""
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    return MNNVLAllReduce.is_mnnvl(mapping, dtype, explicitly_requested=explicitly_requested)


def one_shot_threshold_bytes() -> int:
    """The message size (``tokens * hidden * ranks * itemsize``) up to which the built-in MNNVL all-reduce runs
    one-shot: 1 MiB, or ``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES`` read once at import."""
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

    return _MNNVL_ONE_SHOT_THRESHOLD_BYTES

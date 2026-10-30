# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workspace the three TP4 Lamport all-reduce entries share.

`cute_dsl_allreduce`, `cute_dsl_allreduce_add_rmsnorm` and
`cute_dsl_allreduce_add_rmsnorm_quant_nvfp4` are one transport with three
epilogues. They read and write the same symmetric-memory peer buffers and
rotate the same device slot counter, so every collective of a forward goes
through one `LamportWorkspace`, built once at load by `build_workspace`.

The kernel reads its slot from the counter in device memory and advances it
itself, which is what lets one CUDA graph capture every collective of a
forward and still rotate slots on replay. The peer buffers are three slots x
world x the token budget, so every row count up to the budget fits.

Not an entry: it holds what the entries are handed, and no entry owns it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cutlass
import torch

from ...kernels.comm import tp4_allreduce_fused_rms as _ar

_PING_PONG = 3
_SIGNAL_STATE = 6
_BF16_VEC = 8
# In-band readiness: bitwise bf16 -0.0 marks a slot not yet arrived, as TRT-LLM's own
# Lamport backend does. Separate ready flags double the peer traffic.
_NEG_ZERO_BF16 = -32768
# PDL release points. The module default, `strict`, hangs here: it waits at entry for a
# producer release the flashinfer and torch kernels ahead of it never issue. `entry` on the
# fused ops gives the dependent kv_a projection the allreduce's NVLink wait to prefetch its
# weight slice into L2.
_PDL_POLICY = "output_safe"
_FUSED_PDL_POLICY = "entry"
_FUSED_QUANT_PDL_POLICY = "entry"
_FUSED_OPTIONS = dict(
    all_to_all_cluster_reduction=True,
    last_cta_signal_advance=True,
    elide_self_rank=True,
    rotate_peer_order=True,
    rotated_poll_order=True,
)
#: Above this many rows the add+RMSNorm entry runs two-shot: one-shot has every rank pull
#: the whole message from every peer. Below it two-shot measured slower. The choice is a
#: function of the row count alone, and every rank derives the row count from the same
#: batch, so all ranks take the same side.
TWOSHOT_ABOVE = 3072


def ensure_process_group(rank: int, world: int) -> None:
    """Bring up `torch.distributed` for this rank if nothing has yet.

    `symm_mem.rendezvous` needs a process group. The engine's ranks are
    MPI-spawned and TRT-LLM's collectives use their own transport, so there may
    be none. Three cases, in order:

    * A group is already up (the engine brought one up, or a test launcher
      did): it has to be this rank in a group of `world`, and is used as is.
    * The rendezvous address is in the environment (`MASTER_ADDR`,
      `MASTER_PORT`, `RANK`, `WORLD_SIZE`): the group comes up from it.
      MPI is not touched.
    * Otherwise (the engine under MPI): rank 0 draws a free port and the
      address is broadcast over `MPI.COMM_WORLD`, as TRT-LLM's MegaMoE path
      does. Only this case needs MPI, so only it checks that `COMM_WORLD` is
      the TP group.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        assert dist.get_world_size() == world and dist.get_rank() == rank, (
            f"the process group is rank {dist.get_rank()} of {dist.get_world_size()}; "
            f"the collective was built for rank {rank} of {world}"
        )
        return
    if not all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == world and comm.Get_rank() == rank, (comm.Get_size(), world)
        addr = None
        if rank == 0:
            import socket

            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            # The bare hostname does not resolve on every node; its address does.
            addr = (socket.gethostbyname(socket.gethostname()), port)
        host, port = comm.bcast(addr, root=0)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )


@dataclass(frozen=True)
class LamportWorkspace:
    """One set of peer buffers and compiled kernels serving every collective in the forward.

    Every compiled kernel is symbolic in the row count, so one compile each covers every
    row count up to `max_rows`; the kernel reads the active rows off its input's shape.
    `state` is the device slot counter: every launch of any of the four kernels reads and
    advances it on the device, so the workspace's tensors are shared state even though the
    dataclass is frozen. One workspace per rank per process.
    """

    hidden: int
    eps: float
    max_rows: int
    rank: int
    world: int
    #: This rank's `[3, world, max_rows * hidden]` bf16 symmetric buffer and its
    #: `[3, world, max_rows * hidden / 8]` int32 flags.
    buffer: torch.Tensor
    flags: torch.Tensor
    #: Every rank's buffer and flags, mapped into this rank, in rank order.
    peer_buffers: list
    peer_flags: list
    #: `[6]` int32: entry counter, current slot, per-slot vector counts, release epoch.
    state: torch.Tensor
    #: The multicast address of `buffer`, or 0 when any rank has none.
    multicast_base: int
    pure: object
    fused: object
    fused_quant: object
    fused_twoshot: object


def build_workspace(
    hidden: int, eps: float, max_rows: int, rank: int, world: int
) -> LamportWorkspace:
    """Bring up the group, pin the peer buffers, agree multicast, compile the kernels.

    Collective: every rank of the group calls it, with the same `hidden`, `eps` and
    `max_rows`. Allocates `3 * world * max_rows * hidden` bf16 plus a quarter of that
    in int32 flags of symmetric memory per rank, and compiles four CuTe DSL kernels
    (pure, fused, fused+quant, fused two-shot) for this rank on the current device.
    Launches nothing beyond the buffer fills and the one-element multicast agreement.

    The multicast decision is agreed across the group: a rank compiling the multicast
    variant while a peer did not would wait on a store that never lands.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    ensure_process_group(rank, world)
    dev = torch.device("cuda", torch.cuda.current_device())
    numel = max_rows * hidden
    buffer = symm_mem.empty((_PING_PONG, world, numel), dtype=torch.bfloat16, device=dev)
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO_BF16)
    flags = symm_mem.empty(
        (_PING_PONG, world, numel // _BF16_VEC), dtype=torch.int32, device=dev
    ).zero_()
    group = dist.group.WORLD
    name = getattr(group, "group_name", group)
    bh = symm_mem.rendezvous(buffer, group=name)
    fh = symm_mem.rendezvous(flags, group=name)
    peer_buffers = [bh.get_buffer(peer, buffer.shape, buffer.dtype) for peer in range(world)]
    peer_flags = [fh.get_buffer(peer, flags.shape, flags.dtype) for peer in range(world)]
    state = torch.zeros(_SIGNAL_STATE, dtype=torch.int32, device=dev)
    try:
        mc = int(getattr(bh, "multicast_ptr", 0) or 0)
    except Exception:  # noqa: BLE001 -- no multicast on this fabric is an answer
        mc = 0
    # Agreed over the process group that `rendezvous` just used, so it needs no MPI:
    # the minimum of every rank's "has multicast" flag.
    agreed = torch.tensor([1 if mc != 0 else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
    use_mc = bool(agreed.item())

    common = dict(use_ready_flags=False, use_pdl=True)
    pure = _ar.compile_pure_for_rank(
        rank, world, dev.index, hidden, pdl_release_policy=_PDL_POLICY, **common
    )
    fused = _ar.compile_fused_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        pdl_release_policy=_FUSED_PDL_POLICY,
        prefetch_fused_inputs=True,
        multicast_publish=use_mc,
        **_FUSED_OPTIONS,
        **common,
    )
    # Same transport, slots and signal state as the fused op; only the epilogue differs,
    # so the two interleave.
    fused_quant = _ar.compile_fused_out_quant_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        pdl_release_policy=_FUSED_QUANT_PDL_POLICY,
        prefetch_fused_inputs=True,
        multicast_publish=use_mc,
        **_FUSED_OPTIONS,
        **common,
    )
    fused_twoshot = _ar.compile_fused_twoshot_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        use_pdl=True,
        pdl_release_policy=_PDL_POLICY,
    )
    return LamportWorkspace(
        hidden=hidden,
        eps=eps,
        max_rows=max_rows,
        rank=rank,
        world=world,
        buffer=buffer,
        flags=flags,
        peer_buffers=peer_buffers,
        peer_flags=peer_flags,
        state=state,
        multicast_base=mc if use_mc else 0,
        pure=pure,
        fused=fused,
        fused_quant=fused_quant,
        fused_twoshot=fused_twoshot,
    )


def tail(workspace: LamportWorkspace, fused: bool) -> list:
    """The workspace operands every launch ends with. `signal` is dead with device signal
    state on, so it is 0; only the fused signatures carry `multicast_base`."""
    out = [workspace.peer_buffers, workspace.peer_flags, workspace.state]
    if fused:
        out.append(cutlass.Int64(workspace.multicast_base))
    out.append(cutlass.Int64(0))
    out.append(torch.cuda.current_stream().cuda_stream)
    return out


def check_rows(rows: int, workspace: LamportWorkspace) -> None:
    """The kernel takes the row count from its input, so a call above the token budget the
    peer buffers were sized for writes past every peer's buffer."""
    assert rows <= workspace.max_rows, (
        f"{rows} rows exceed the collective's budget of {workspace.max_rows}"
    )

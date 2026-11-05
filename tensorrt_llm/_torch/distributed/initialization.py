# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize the default PyTorch process group for TRT-LLM workers."""

from __future__ import annotations

import ipaddress
import os
import socket

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout

from tensorrt_llm.logger import logger

__all__ = [
    "init_torch_distributed",
    "ensure_cutedsl_torch_distributed",
    "ensure_deepgemm_torch_distributed",
]


def init_torch_distributed(
    *,
    backend: str,
    rank: int,
    world_size: int,
    store: dist.Store | None = None,
    init_method: str | None = None,
    device_id: torch.device | None = None,
) -> None:
    """Initialize a worker's default PyTorch process group with its launch settings.

    All participating ranks must call this before using PyTorch collectives.
    The caller decides whether initialization is needed and owns the group's
    lifetime and cleanup. PyTorch handles initialization errors.

    Args:
        backend: PyTorch backend specification, such as ``cuda:nccl,cpu:gloo``.
        rank: Rank in the default group.
        world_size: Number of ranks in the default group.
        store: Launcher-provided rendezvous store, mutually exclusive with init_method.
        init_method: Rendezvous URL; PyTorch defaults to ``env://`` without a store.
        device_id: CUDA device already assigned by the worker, when eager NCCL setup is needed.
    """
    # Pass only the rendezvous arguments that the caller set, so that each
    # caller makes the same init_process_group() call as it would directly.
    rendezvous = {}
    if store is not None:
        rendezvous["store"] = store
    if init_method is not None:
        rendezvous["init_method"] = init_method
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=device_id,
        **rendezvous,
    )


def ensure_cutedsl_torch_distributed(ep_size: int) -> None:
    """Bootstrap a torch.distributed NCCL WORLD group under the MPI orchestrator.

    The MPI orchestrator (trtllm-bench / mpirun) never calls
    ``init_process_group``, but the symmetric-memory rendezvous needs a
    ProcessGroup; for pure DEP a node-local NCCL WORLD suffices. No-op
    under Ray or single-rank. Failures propagate: a rank that retried
    ``init_process_group`` alone would never rendezvous with its peers.

    Args:
        ep_size: Expert-parallel world size; must match the MPI world before bootstrap.
    """
    if not dist.is_available() or dist.is_initialized():
        return
    from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size

    try:
        world = mpi_world_size()
        rank = mpi_rank()
    except Exception as e:  # not under MPI either -> leave uninitialized
        logger.debug(
            f"[MegaMoECuteDsl] MPI rank query failed ({e!r}); skipping torch.distributed bootstrap."
        )
        return
    if world <= 1:
        return
    # Check pure DEP BEFORE any collective: under PP/CP hybrids some ranks
    # never construct a MegaMoE module, so the bcast/init below would hang
    # instead of failing loud. Rank-identical inputs -> lockstep raise.
    if world != ep_size:
        raise ValueError(
            f"MegaMoECuteDsl requires pure DEP (MPI WORLD == EP) to "
            f"bootstrap torch.distributed: got MPI world={world}, "
            f"EP={ep_size}."
        )

    # Rank 0 hosts the rendezvous store on a port the OS assigns when the
    # store binds, then broadcasts where it is. Binding the store itself
    # leaves no window between choosing a port and binding it, which
    # drawing a free port first and letting init_process_group bind it
    # later does. MASTER_ADDR / MASTER_PORT are neither read nor written:
    # an MPI worker outlives the LLM that set them (worker shutdown
    # destroys the process group but not the environment), so the next
    # LLM in a reused worker pool would get the previous port. The bcast
    # is unconditional and also carries rank 0's failure, so every rank
    # raises together instead of waiting for a store that never came up.
    comm = mpi_comm()
    store = None
    payload = None
    if rank == 0:
        loopback = False
        try:
            try:
                addr = socket.gethostbyname(socket.gethostname())
            except OSError:
                # Unresolvable hostname (minimal containers / broken
                # DNS).
                addr = "127.0.0.1"
            # A resolvable hostname can also be loopback: Debian and
            # Ubuntu map it to 127.0.1.1 in /etc/hosts. Loopback reaches
            # rank 0 only from its own host; that is checked below on
            # every rank, after the bcast, so no rank skips a collective
            # its peers wait in.
            loopback = ipaddress.ip_address(addr).is_loopback
            store = dist.TCPStore(
                addr,
                0,
                world,
                is_master=True,
                timeout=default_pg_timeout,
                wait_for_workers=False,
            )
            payload = (addr, store.port, loopback, None)
        except RuntimeError as e:
            payload = (None, None, False, repr(e))
    host, port, loopback, error = comm.bcast(payload, root=0)
    if error is not None:
        raise RuntimeError(
            f"[MegaMoECuteDsl] rank 0 could not host the torch.distributed "
            f"rendezvous store: {error}"
        )
    if loopback:
        # Every rank received the same flag, so all of them enter this
        # allgather. A rank on another host would dial its own loopback
        # and time out alone; instead all ranks raise together here.
        hostnames = comm.allgather(socket.gethostname())
        if any(h != hostnames[0] for h in hostnames):
            raise RuntimeError(
                f"[MegaMoECuteDsl] rank 0's hostname {hostnames[0]!r} "
                f"does not resolve to an address other hosts can reach "
                f"(got loopback {host}), so it can only serve the "
                f"torch.distributed rendezvous on loopback, but the MPI "
                f"ranks span several hosts ({sorted(set(hostnames))}). "
                f"Map rank 0's hostname to a routable address."
            )
    if store is None:
        store = dist.TCPStore(host, port, world, is_master=False, timeout=default_pg_timeout)
    # The executor already bound this rank's device before the model was
    # built (base_worker: global_mpi_rank() % device_count). Re-deriving it
    # here could pick a different GPU (e.g. a disagg ctx/gen pair sharing a
    # node) and would move only the layers built after this point, so bind
    # NCCL to the current device instead of selecting one.
    device_id = None
    if torch.cuda.is_available():
        device_id = torch.device("cuda", torch.cuda.current_device())
    logger.info(
        f"[MegaMoECuteDsl] torch.distributed not initialized under MPI; "
        f"bootstrapping NCCL WORLD group (rank={rank}/{world}, "
        f"device={device_id}, store={host}:{port}) for the EP rendezvous."
    )
    init_torch_distributed(
        backend="cuda:nccl,cpu:gloo",
        store=store,
        rank=rank,
        world_size=world,
        device_id=device_id,
    )


def ensure_deepgemm_torch_distributed() -> None:
    """Initialize torch.distributed from mpi4py when running under TRT-LLM's MPI executor.

    Safe to call collectively: every MPI rank reaches MegaMoE's
    ``__init__`` synchronously during model build. Rank 0 picks a
    free port and bcasts (host, port) so single- and multi-node both
    work; the retry on RuntimeError absorbs the close()→bind() race
    on busy hosts. ``device_id`` uses intra-node local rank because
    ``global_rank % device_count`` is wrong on multi-node launches.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if world_size <= 1:
        return

    try:
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = local_comm.Get_rank()
    except Exception:
        local_rank = int(
            os.environ.get(
                "OMPI_COMM_WORLD_LOCAL_RANK",
                os.environ.get(
                    "MV2_COMM_WORLD_LOCAL_RANK", rank % max(1, torch.cuda.device_count())
                ),
            )
        )

    def _pick_rendezvous() -> tuple[str, int] | None:
        """Let rank 0 select the endpoint that will be broadcast to MPI peers."""
        if rank == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            host = socket.gethostbyname(socket.gethostname())
            return (host, port)
        return None

    # Respect launcher env vars only when they describe this MPI world.
    # A disaggregated launcher may export its outer context/generation
    # world into each server's independent model MPI world.
    launcher_env_matches_mpi = False
    launcher_port = None
    if all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        try:
            # Parse the port here too: accepting the launcher env on
            # RANK/WORLD_SIZE alone and only converting MASTER_PORT later
            # turns a non-numeric port into an uncaught ValueError instead
            # of falling back to the MPI rendezvous.
            launcher_port = int(os.environ["MASTER_PORT"])
            launcher_env_matches_mpi = (
                int(os.environ["RANK"]) == rank and int(os.environ["WORLD_SIZE"]) == world_size
            )
        except ValueError:
            launcher_env_matches_mpi = False

    if not launcher_env_matches_mpi:
        host, port = comm.bcast(_pick_rendezvous(), root=0)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
    else:
        host = os.environ["MASTER_ADDR"]
        port = launcher_port

    device_id = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_id = torch.device("cuda", local_rank % torch.cuda.device_count())
    try:
        init_torch_distributed(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device_id,
        )
    except RuntimeError:
        # close()→bind() race: redraw a port and retry once.
        new_host, new_port = comm.bcast(_pick_rendezvous(), root=0)
        os.environ["MASTER_ADDR"] = new_host
        os.environ["MASTER_PORT"] = str(new_port)
        host, port = new_host, new_port
        init_torch_distributed(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device_id,
        )
    logger.info(
        f"[MegaMoE] Initialized torch.distributed from mpi4py "
        f"(rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, master={host}:{port}, backend=nccl)"
    )

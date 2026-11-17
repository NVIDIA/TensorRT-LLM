# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collective acceptance workspace allocation; each target retains its type and cache."""

from __future__ import annotations

from typing import Callable, TypeVar

import torch

from tensorrt_llm.mapping import Mapping

_WorkspaceT = TypeVar("_WorkspaceT")


def create_accept_workspace(
    mapping: Mapping,
    device: torch.device | str,
    slot_rows: int,
    workspace_factory: Callable[..., _WorkspaceT],
) -> _WorkspaceT:
    """Exchange slots ([2 halves][slot_rows rows][world] uint64) behind one multicast mapping over the TP group, the
    local key accumulators [slot_rows] and the (finished CTAs, call) counters. Collective over the TP group; call
    outside CUDA-graph capture, in the same order on every rank."""
    words = 2 * slot_rows * max(mapping.tp_size, 1)
    if mapping.tp_size > 1:
        from tensorrt_llm._torch.distributed.ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )
        from tensorrt_llm._utils import mpi_comm

        session = mpi_comm()
        session_group = session.Get_group()
        group = session_group.Incl(mapping.tp_group)
        session_group.Free()
        comm = session.Create_group(group)
        group.Free()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            comm.Free()
            raise RuntimeError("exchange workspace creation failed on another rank")
        handle = _make_mnnvl_mcast_buffer(
            comm, max(words * 8, 4096), mapping, mapping.is_multi_node()
        )
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int64, 0)
        mc = handle.get_mc_buffer((words,), torch.int64, 0)
        uc.zero_()
        torch.cuda.synchronize()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("exchange workspace creation failed on another rank")
        mc_ptr, rank = (mc.data_ptr(), mapping.tp_rank)
    else:
        uc = torch.zeros(words, dtype=torch.int64, device=device)
        mc_ptr, rank, handle, comm = (uc.data_ptr(), 0, None, None)
    return workspace_factory(
        uc=uc,
        mc_ptr=mc_ptr,
        keys=torch.zeros(slot_rows, dtype=torch.int64, device=uc.device),
        counters=torch.zeros(2, dtype=torch.int32, device=uc.device),
        rank=rank,
        world_size=max(mapping.tp_size, 1),
        handle=handle,
        comm=comm,
    )

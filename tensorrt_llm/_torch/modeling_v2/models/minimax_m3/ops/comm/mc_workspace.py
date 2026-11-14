# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multicast workspace storage shared by the sm_107 TP topologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

EMPTY_WORD = -2147483648


@dataclass(eq=False)
class McAllReduceWorkspace:
    """Two halves of [max_tokens][world][H] BF16 (int32 words) per rank behind one multicast mapping, plus one call
    counter per CTA per cluster size. Create collectively and eagerly on every TP rank."""

    uc: torch.Tensor
    mc: torch.Tensor
    flags: dict
    rank: int
    world_size: int
    hidden: int
    handle: Any
    comm: Any
    max_tokens: int = 8

    @classmethod
    def create(cls, mapping, hidden: int, max_tokens: int = 8) -> "McAllReduceWorkspace":
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
        words = 4 * max_tokens * mapping.tp_size * (hidden // 2)  # two exchange sets of two halves
        ok = not torch.cuda.is_current_stream_capturing()
        if not _mnnvl_workspace_all_succeeded(comm, ok):
            comm.Free()
            raise RuntimeError(
                "McAllReduceWorkspace.create must run outside CUDA-graph capture on every rank"
            )
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, mapping.is_multi_node())
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        uc.fill_(EMPTY_WORD)
        torch.cuda.synchronize()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("McAllReduceWorkspace.create failed on another rank")
        return cls(
            uc=uc,
            mc=mc,
            flags={},
            rank=mapping.tp_rank,
            world_size=mapping.tp_size,
            hidden=hidden,
            handle=handle,
            comm=comm,
            max_tokens=max_tokens,
        )

    def flags_for(self, cluster: int):
        """This cluster size's per-CTA call counters, or None when they do not exist yet during a capture.
        get_workspace creates the cluster-8 counters with the workspace, so for cluster 8 (the only size built)
        that None never happens and cannot differ between ranks."""
        flags = self.flags.get(cluster)
        if flags is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            flags = torch.zeros(self.max_tokens * cluster, dtype=torch.int32, device=self.uc.device)
            self.flags[cluster] = flags
        return flags

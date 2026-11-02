# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shard geometry and exchange workspace of the key-sharded decode indexer top-k.

The decode indexer is replicated over the tensor-parallel group: without sharding every rank
computes the same FP8 MQA logits over the whole context and the same top-2048. Instead rank ``r``
scores only its contiguous shard of the request's indexer keys (``glm_dsa_hist_shard_meta`` builds
the shard's context length and block table), and ``glm_dsa_hist_topk`` finds every row's exact
top-2048 over all shards by exchanging key histograms through a private multicast workspace; every
rank ends with the same rows. See ``cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from tensorrt_llm.mapping import Mapping

# Constants of glmDsaHistTopK.h.
MAX_RANKS = 16
SHARD_ALIGN = 128
MAX_SHARD = 98304
MAX_ROWS = 8
TOPK = 2048
WORKSPACE_WORDS = 3 * (3 * 8 * 16 * 1088 + 8 * 2048 + 8 * 4096 * 2)
STATE_WORDS = 4 + 8 * 4


def shard_size(kv_len: int, group_size: int) -> int:
    """Keys per rank of a request of ``kv_len`` indexer keys: ceil(kv_len / group_size), rounded
    up to SHARD_ALIGN. Rank ``r`` holds the keys [r * S, r * S + n_r), n_r = clamp(kv_len - r * S,
    0, S)."""
    per_rank = -(-max(kv_len, 0) // group_size)
    return -(-per_rank // SHARD_ALIGN) * SHARD_ALIGN


def shard_capacity(max_seq_len: int, group_size: int) -> int:
    """Largest shard of a request of at most ``max_seq_len`` indexer keys."""
    return shard_size(max_seq_len, group_size)


@dataclass
class HistTopkWorkspace:
    """The exchange workspace and device state of ``glm_dsa_hist_topk`` for one TP group.

    ``comm_buffer`` is this rank's unicast view of ``WORKSPACE_WORDS`` int32 words of a
    McastGPUBuffer, all -1 at creation; ``state`` is ``STATE_WORDS`` int32 words, zero at creation.
    Both are private to the op and live as long as the engine: captured CUDA graphs launch on
    them. ``handle`` keeps the multicast allocation alive.
    """

    group_size: int
    handle: Any
    comm_buffer: torch.Tensor
    state: torch.Tensor
    comm: Any


def hist_topk_supported(mapping: Mapping) -> bool:
    """Whether the key-sharded top-k can serve this mapping: a pure tensor-parallel group of 2 to
    16 ranks without attention DP or context parallelism, on an MNNVL fabric."""
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    return (
        2 <= mapping.tp_size <= MAX_RANKS
        and mapping.world_size == mapping.tp_size
        and not mapping.has_cp()
        and not mapping.enable_attention_dp
        and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)
    )


def create_hist_topk_workspace(mapping: Mapping, comm: Optional[Any] = None) -> HistTopkWorkspace:
    """Allocate the workspace on every rank of ``mapping``'s TP group (a collective call).

    ``comm`` is the TP-group communicator the multicast handles are exchanged over: the MNNVL
    all-reduce's (an MPI split, or the TP ProcessGroup without MPI) when None. The outcome is
    converged over the group, so every rank raises or none does.
    """
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_device_index,
        _mnnvl_workspace_all_succeeded,
    )

    if comm is None:
        comm = _get_mnnvl_workspace_comm(mapping)
    error: Optional[Exception] = None
    workspace = None
    try:
        handle = _make_mnnvl_mcast_buffer(
            comm, WORKSPACE_WORDS * 4, mapping, use_fabric_handle=mapping.is_multi_node()
        )
        comm_buffer = handle.get_uc_buffer(mapping.tp_rank, (WORKSPACE_WORDS,), torch.int32, 0)
        with torch.inference_mode():
            comm_buffer.fill_(-1)
            state = torch.zeros(
                STATE_WORDS,
                dtype=torch.int32,
                device=torch.device("cuda", _mnnvl_device_index(mapping)),
            )
        torch.cuda.synchronize()
        workspace = HistTopkWorkspace(mapping.tp_size, handle, comm_buffer, state, comm)
    except Exception as e:
        # Any failure is converged over the group and re-raised on every rank below: a rank that
        # raised alone would leave its peers waiting in the next exchange.
        error = e
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError(
            "GLM DSA histogram top-k workspace construction failed on at least one rank"
        ) from error
    return workspace


_WORKSPACES: Dict[Mapping, HistTopkWorkspace] = {}


def get_hist_topk_workspace(mapping: Mapping) -> HistTopkWorkspace:
    """The workspace of ``mapping``'s TP group, created on the first call (collective: every rank
    of the group makes its first call in the same place, outside CUDA graph capture)."""
    workspace = _WORKSPACES.get(mapping)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the GLM DSA histogram top-k workspace is created outside capture")
        workspace = _WORKSPACES[mapping] = create_hist_topk_workspace(mapping)
    return workspace

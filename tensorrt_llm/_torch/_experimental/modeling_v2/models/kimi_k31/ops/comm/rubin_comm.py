# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builders of the K3.1 Rubin targets' collective resources: a runtime helper, not an OpWrapper.

A target calls them once in ``post_load_weights``, outside CUDA graph capture, at the same point
on every rank of the group (each one is collective). It holds the results in its runtime
resources and passes them to the entries as arguments:

* :func:`rubin_all_reduce`: the engine ``AllReduce`` of a TP group, which ``allreduce_module``
  takes;
* :func:`rubin_latent_ar_workspace`: the routed latent all-reduce's own MNNVL Lamport workspace,
  which ``rubin_latent_ar_norm_row`` takes.

Every group comes from the mapping the caller passes: on helix8 that is the mapping whose TP
group is the 8 CP ranks (``Mapping.repurpose_helix_cp_to_tp``), never the helix mapping itself,
whose TP group is one rank.
"""

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm._torch.distributed.ops import (
    AllReduce,
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.mapping import Mapping

from .rubin_push_allreduce import use_fabric_handle

#: The 9-word state of a Lamport workspace: current index, dirty index, bytes per buffer, dirty
#: stages, bytes to clear (x4), arrivals.
_FLAGS = (0, 2, None, 0, 0, 0, 0, 0, 0)


def _check_group(mapping: Mapping, group_size: int) -> None:
    """Fail unless ``mapping``'s TP group is the layout's group of ``group_size`` ranks."""
    if mapping.tp_size != group_size:
        raise ValueError(
            f"the TP group of this mapping has {mapping.tp_size} ranks, the layout reduces over "
            f"{group_size}: pass the mapping whose TP group is the layout's group"
        )


def rubin_all_reduce(mapping: Mapping, strategy: Any, group_size: int) -> AllReduce:
    """The engine ``AllReduce`` of ``mapping``'s TP group (``group_size`` ranks), in BF16."""
    _check_group(mapping, group_size)
    return AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)


def rubin_latent_ar_workspace(
    mapping: Mapping,
    group_size: int,
    hidden_size: int,
    max_tokens: int,
    oneshot_max_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """The routed latent all-reduce's own MNNVL Lamport workspace, for up to ``max_tokens`` rows.

    Three Lamport buffers and the 9-word state, laid out as the shared MNNVL workspace
    (``get_or_scale_allreduce_mnnvl_workspace``), each buffer large enough for the two-shot sizes
    of up to ``max_tokens`` rows and for every rank's rows in one buffer up to
    ``oneshot_max_bytes`` (rows x ``hidden_size`` x ranks x 2 bytes). The buffers start
    Lamport-clean (-0.0). Collective over the TP group.

    Returns a dict: ``handle`` (keeps the buffers alive), ``buffer`` (the local unicast view
    ``[3, n]``), ``mc`` (the multicast view), ``flags``, ``max_tokens``, ``buffer_bytes`` (bytes
    of one buffer), ``rank``, ``world_size``.
    """
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    _check_group(mapping, group_size)
    # The shared workspace's communicator is split by the same group.
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)["comm"]
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    if oneshot_max_bytes:
        row_bytes = hidden_size * mapping.tp_size * dtype.itemsize
        need = max(need, min(max_tokens, oneshot_max_bytes // row_bytes) * row_bytes)
    buffer_bytes = -(-need // (1 << 20)) << 20
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle(mapping),
        comm.py2f(),
    )
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    flags = torch.tensor(
        [buffer_bytes if v is None else v for v in _FLAGS],
        dtype=torch.uint32,
        device=buffer.device,
    )
    return {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        "max_tokens": max_tokens,
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }

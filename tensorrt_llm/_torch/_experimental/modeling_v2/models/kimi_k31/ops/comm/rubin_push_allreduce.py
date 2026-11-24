# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pushed MNNVL all-reduce of the K3.1 Rubin targets: a builder and helpers, not OpWrappers.

A producer may store its bf16 output straight into the Lamport buffers of the MNNVL all-reduce
that follows it; the consumer's kernel, or a finish kernel, then completes the all-reduce bitwise
like the C++ kernel the call replaces. Every kernel launch goes through the entries of
``ops/comm/rubin_lamport_push.py``.

The shared ``AllReduce`` / ``MNNVLAllReduce`` are not edited, so these are free functions on
the shared MNNVL workspace (same dict keys, same 9-word ``buffer_flags`` layout), with its
one-shot threshold ``_MNNVL_ONE_SHOT_THRESHOLD_BYTES``.

The targets' attention all-reduce builds its workspace once, with
:func:`rubin_lamport_push_workspace` (a runtime-helper builder, called while the runtime is
built): the step then pushes and finishes on it with no per-call state.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    AllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.mapping import Mapping

from ._rubin_comm_common import reduce_chunk

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace, required_bytes, row_push_desc

# Pushed all-reduces whose consumer finishes them (the attn_res kernel reduces the slots itself)
# use the one-shot transport up to this many bytes (num_tokens * hidden_dim * group_size * 2), the
# two-shot transport above: the transport of the C++ kernel the call replaces.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = int(_MNNVL_ONE_SHOT_THRESHOLD_BYTES)


def _use_pdl() -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return get_env_enable_pdl()


def use_fabric_handle(mapping: Mapping) -> bool:
    """Whether the family's own multicast buffers use fabric handles: across nodes.

    The shared workspace also takes them on one node under ``TRTLLM_FORCE_MNNVL_AR=1``; POSIX FD
    handles are equally valid on one node.
    """
    return mapping.is_multi_node()


@dataclasses.dataclass(frozen=True)
class LamportPushWs:
    """The pushed all-reduce of one TP group (:func:`rubin_lamport_push_workspace`).

    ``view`` is the ``LamportWorkspace`` of the shared MNNVL workspace, sized for the calls the
    builder was asked for. ``workspace`` is that workspace: held here, its buffers outlive a later
    replacement by a larger engine call. ``pdl``: the producer kernels launch with PDL.
    """

    view: object
    workspace: dict
    hidden: int
    pdl: bool

    def transport(self, rows: int) -> Tuple[bool, int]:
        """``(one_shot, finish_chunk)`` of a call of ``rows`` rows that its consumer finishes:
        the one-shot transport up to ``_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES``, and the ranks per
        partial sum of the C++ kernel the shared threshold selects."""
        nbytes = rows * self.hidden * self.view.world * 2
        one_shot = nbytes <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
        return one_shot, reduce_chunk(self.view.world, nbytes <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES)


def rubin_lamport_push_workspace(
    all_reduce: AllReduce, max_rows: int, hidden: int
) -> Optional[LamportPushWs]:
    """Build the pushed all-reduce of ``all_reduce`` for calls of 1 to ``max_rows`` rows.

    Sizes the shared MNNVL workspace for the largest of those calls (each on the transport
    :meth:`LamportPushWs.transport` picks) and builds its device view. Returns None where
    ``all_reduce`` has no MNNVL bf16 path, uses symmetric memory, spans one rank, or the calls
    do not fit a 32-bit size: the caller binds the plain all-reduce. Collective over the group
    when the workspace grows, so every rank calls it at the same point, outside graph capture.
    """
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if (
        mnnvl is None
        or all_reduce.mapping.tp_size == 1
        or getattr(all_reduce, "symm_mem_allreduce", None) is not None
        or mnnvl.dtype != torch.bfloat16
        or max_rows <= 0
        or hidden % 8
    ):
        return None
    world = mnnvl.mapping.tp_size
    need = 0
    for rows in range(1, max_rows + 1):
        one_shot = rows * hidden * world * 2 <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
        need = max(
            need,
            mnnvl.get_required_workspace_size(rows, hidden, world, mnnvl.dtype),
            required_bytes(rows, hidden, world, one_shot),
        )
    if need >= 2**32 - 1:
        return None
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, mnnvl.dtype, buffer_size_bytes=need
    )
    view = LamportWorkspace.from_mnnvl(workspace, world, mnnvl.mapping.tp_rank)
    return LamportPushWs(view, workspace, hidden, _use_pdl())


def rubin_lamport_row_push_desc(push: LamportPushWs) -> torch.Tensor:
    """The device descriptor a GEMM's Lamport row store reads to push into ``push``'s calls
    (``rubin_lamport_ar.row_push_desc``). Built eagerly, once per workspace, while the runtime is
    built."""
    return row_push_desc(push.view)


__all__ = [
    "LamportPushWs",
    "rubin_lamport_push_workspace",
    "rubin_lamport_row_push_desc",
    "use_fabric_handle",
]

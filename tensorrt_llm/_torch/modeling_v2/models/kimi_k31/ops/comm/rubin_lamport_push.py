# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin pushed MNNVL all-reduce: the push and finish kernels.

A producer stores its bf16 partial into the Lamport slots of the next all-reduce on an MNNVL
workspace (``rubin_lamport_push``, or a GEMM's push epilogue) and ``rubin_lamport_finish``
completes a one-shot call that carries all-gather rows, bit for bit like the C++ MNNVL all-reduce
followed by the add of the all-gathered rows. The kernels are
``kernels/comm/rubin_lamport_ar.py``; ``__call__`` calls their launchers directly. The runtime
builder of the push workspace (:func:`rubin_lamport_push_workspace`, re-exported by
``ops/comm/rubin_push_allreduce.py``) is here too: it sizes the shared MNNVL workspace the
entries run on.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, OpWrapper

from ._rubin_comm_common import (
    LAMPORT_NOTE,
    check_rows,
    lamport_allreduce_reference,
    reduce_chunk,
    sanitize_bf16,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace, required_bytes
    from ...kernels.comm.rubin_lamport_ar import run_lamport_finish as _launch_finish
    from ...kernels.comm.rubin_lamport_ar import run_lamport_push as _launch_push


class RubinLamportPush(OpWrapper):
    """Push this rank's bf16 ``x`` [m, hidden] into the Lamport slots of the next call on ``ws``
    (a ``LamportWorkspace`` of the kernel module): one-shot (every rank's buffer, multicast) or
    two-shot (the owner ``t % world`` of each row). Returns None; the finish reads the slots."""

    ARCHS: frozenset[Arch] = frozenset()
    note = LAMPORT_NOTE

    def __call__(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> None:
        _launch_push(ws, x, one_shot, use_pdl)

    def reference(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> torch.Tensor:
        """The payload of this rank's slots: ``x`` with ``-0`` sent as ``+0``. A finish of the
        call is :func:`lamport_allreduce_reference` of every rank's payload."""
        return sanitize_bf16(x)

    def is_valid(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> None:
        check_rows("x", ws, x)


class RubinLamportFinish(OpWrapper):
    """Complete the one-shot call every rank pushed, whose all-gather rows ``y`` the ranks
    filled, into ``out`` (contiguous bf16 [m, hidden]): ``out`` is ``bf16(y + bf16(sum))``, the
    sum in the C++ one-shot kernel's order. Returns ``out``."""

    ARCHS: frozenset[Arch] = frozenset()
    note = LAMPORT_NOTE

    def __call__(self, ws: LamportWorkspace, out: torch.Tensor, use_pdl: bool) -> torch.Tensor:
        return _launch_finish(ws, out, use_pdl=use_pdl)

    def reference(
        self, inputs_in_rank_order: Sequence[torch.Tensor], ag_rows: torch.Tensor
    ) -> torch.Tensor:
        """What every rank gets back. The collective exception to mirroring ``__call__``: it
        takes every rank's pushed partial in rank order and the all-gather rows."""
        chunk = reduce_chunk(len(inputs_in_rank_order), True)
        return lamport_allreduce_reference(inputs_in_rank_order, chunk, ag_rows)

    def is_valid(self, ws: LamportWorkspace, out: torch.Tensor, use_pdl: bool) -> None:
        check_rows("out", ws, out)


rubin_lamport_push = RubinLamportPush()
rubin_lamport_finish = RubinLamportFinish()


# Pushed all-reduces whose consumer finishes them (the attn_res kernel reduces the slots itself)
# use the one-shot transport up to this many bytes (num_tokens * hidden_dim * group_size * 2), the
# two-shot transport above: the transport of the C++ kernel the call replaces.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = int(_MNNVL_ONE_SHOT_THRESHOLD_BYTES)


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
    return LamportPushWs(view, workspace, hidden, get_env_enable_pdl())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks over multi-node NVLink, then add the residual and RMS-normalize."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.functional import AllReduceFusionOp

from .._op import Arch, Cell, OpWrapper
from .allreduce import allreduce as _allreduce

#: The one fusion this entry certifies.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The op's one-shot / two-shot switch, in bytes of `rows * hidden * world`.
#: Mirrors `kOneShotSizeThreshold` in thop/allreduceOp.cpp; the workspace size
#: below depends on which kernel runs.
_ONE_SHOT_BYTES = 64 * 1024 * 8 * 2


def workspace_bytes(rows: int, hidden: int, world: int, dtype: torch.dtype) -> int:
    """Bytes of one Lamport buffer a call over `rows` tokens needs.

    The same arithmetic as `MNNVLAllReduce.get_required_workspace_size`: the
    one-shot kernel stages every rank's rows, the two-shot kernel two stages of
    a world-aligned share.
    """
    elem = torch.empty((), dtype=dtype).element_size()
    if rows * hidden * world * elem <= _ONE_SHOT_BYTES:
        return rows * hidden * world * elem
    return 2 * -(-rows // world) * world * hidden * elem


def workspace_bytes_up_to(max_rows: int, hidden: int, world: int, dtype: torch.dtype) -> int:
    """Bytes of one Lamport buffer that every call over 1..`max_rows` tokens fits in.

    `workspace_bytes` is not monotonic in `rows`: past the one-shot threshold
    the two-shot kernel needs less. So the largest one-shot call can need more
    than `max_rows` itself, e.g. 22 rows need 1,013,760 bytes at gpt-oss TP8
    while 64 rows need 737,280. Each kernel's need grows with `rows`, so the
    larger of the two kernels' largest calls covers the range.
    """
    elem = torch.empty((), dtype=dtype).element_size()
    one_shot_rows = min(max_rows, _ONE_SHOT_BYTES // (hidden * world * elem))
    return max(
        workspace_bytes(max_rows, hidden, world, dtype),
        workspace_bytes(one_shot_rows, hidden, world, dtype),
    )


def mnnvl_workspace_args(mapping, max_rows: int, hidden: int, dtype: torch.dtype) -> dict:
    """Build (or reuse) this mapping's workspace for every call of 1..`max_rows` tokens.

    Returns the communicator side of the call. A target calls this once in
    `__init__` and passes `**args` at every call site.

    Collective over the TP group: every rank must call it, in the same order.
    """
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping,
        dtype,
        buffer_size_bytes=workspace_bytes_up_to(max_rows, hidden, mapping.tp_size, dtype),
    )
    return dict(
        comm_buffer=workspace["uc_buffer"].view(dtype).view(3, -1),
        buffer_flags=workspace["buffer_flags"],
        world=mapping.tp_size,
    )


class MnnvlAllreduce(OpWrapper):
    """Sum `input` over the TP group through a multicast NVLink workspace,
    add `residual`, and RMS-normalize the sum with `norm_weight`.

    Returns fresh `(norm_out, residual_out)` and leaves `input` and `residual`
    untouched, the same contract as `comm/allreduce` with `RESIDUAL_RMS_NORM`,
    so a target can call either one in the same slot.

    This is the kernel the built-in `AllReduce` module runs when its TP group
    spans nodes (`MNNVLAllReduce`): on GB300, across trays of one NVLink
    domain. `comm/allreduce`'s IPC workspace cannot reach another tray, and
    there its AUTO falls back to NCCL with the norm unfused.

    The workspace is this op's fixture, built by `mnnvl_workspace_args` for a
    row budget. The kernel does not check the budget; `is_valid` does.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The gpt-oss default target at attention TP8 over two GB300 trays. It
    # calls this twice per layer, after o_proj and after the MoE, on the full
    # token set every rank holds.
    _GPT_OSS = dict(hidden=2880, dtype=torch.bfloat16, eps=1e-5, world=8)
    # The deepseek default target at attention TP8, the same two calls per
    # layer at deepseek's hidden width.
    _DEEPSEEK = dict(hidden=7168, dtype=torch.bfloat16, eps=1e-6, world=8)

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token: the one-shot kernel", spec=dict(rows=1, **_GPT_OSS)),
        Cell(
            why="a decode batch: past the one-shot threshold, the two-shot kernel",
            spec=dict(rows=64, **_GPT_OSS),
        ),
        Cell(why="a full 8192-token prefill chunk", spec=dict(rows=8192, **_GPT_OSS)),
        Cell(
            why="deepseek, decode, one token: the one-shot kernel", spec=dict(rows=1, **_DEEPSEEK)
        ),
        Cell(
            why="deepseek, a decode batch: the two-shot kernel at hidden 7168",
            spec=dict(rows=16, **_DEEPSEEK),
        ),
        Cell(why="deepseek, a full 8192-token prefill chunk", spec=dict(rows=8192, **_DEEPSEEK)),
    )

    note = """
    Only `RESIDUAL_RMS_NORM` is certified. The op also has a plain sum and
    FP8 / NVFP4 quantizing fusions; none is reachable through this entry.

    The workspace is shared per mapping with the built-in module, through
    `get_or_scale_allreduce_mnnvl_workspace`. Under MPI its communicator is a
    split of the session communicator; with MPI disabled it is the mapping's
    TP process group, which needs a CPU (gloo) backend for the setup
    handshake.

    Building the workspace is a collective over the TP group: every rank must
    call `mnnvl_workspace_args`, in the same order.
    """

    def __call__(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        world: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`world` is not passed to the op, which reads it from the workspace;
        it is here so `is_valid` can size the call against the buffer."""
        norm_out, residual_out = torch.ops.trtllm.mnnvl_fusion_allreduce(
            input,
            norm_weight,
            residual,
            eps,
            comm_buffer,
            buffer_flags,
            True,
            None,
            RESIDUAL_RMS_NORM,
        )
        return norm_out, residual_out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The same arithmetic as `comm/allreduce`'s reference: fp32 sum, the
        residual added to it, the norm read from that fp32 value.

        The documented collective exception to mirroring `__call__`: it takes
        every rank's `input`, in rank order.
        """
        return _allreduce.reference(
            inputs_in_rank_order, residual, norm_weight, eps, RESIDUAL_RMS_NORM
        )

    def is_valid(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        world: int,
    ) -> None:
        assert residual.shape == input.shape, (residual.shape, input.shape)
        assert norm_weight.shape == (input.shape[-1],), (norm_weight.shape, input.shape)
        # The kernel indexes the Lamport buffer by the call's size and never
        # checks it against the buffer it was given.
        hidden = input.shape[-1]
        need = workspace_bytes(input.numel() // hidden, hidden, world, input.dtype)
        have = comm_buffer.shape[-1] * comm_buffer.element_size()
        assert need <= have, (
            f"{input.numel() // hidden} rows need {need} bytes of workspace, the buffer "
            f"holds {have}; build it with mnnvl_workspace_args and a larger max_rows"
        )


mnnvl_allreduce = MnnvlAllreduce()

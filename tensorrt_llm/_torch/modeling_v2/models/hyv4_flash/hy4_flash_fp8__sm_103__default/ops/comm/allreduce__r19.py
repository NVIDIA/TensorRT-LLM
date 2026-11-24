# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The plain sum across ranks at this target's width, via the trtllm allreduce op.

Target copy of ``catalog/comm/allreduce.py`` with cells for ``op=NONE``, the form the
row-parallel o_proj and MoE partials call.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

#: `residual_out = sum(input) + residual`, `norm_out = rmsnorm(residual_out) * norm_weight`.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The form this target calls: the sum alone, one output tensor.
PLAIN = int(AllReduceFusionOp.NONE)

#: One-shot or two-shot over the IPC workspace inside one NVLink domain, NCCL when
#: the message outgrows the workspace or the ranks cannot reach each other's memory.
AUTO = int(AllReduceStrategy.AUTO)

#: Strategies that never read the IPC workspace.
_NO_WORKSPACE = frozenset(
    {
        int(AllReduceStrategy.NCCL),
        int(AllReduceStrategy.NCCL_SYMMETRIC),
        int(AllReduceStrategy.UB),
    }
)


def allreduce_group_args(mapping, strategy: int = AUTO) -> dict:
    """The communicator arguments for this rank's TP group; build once, pass as `**args`.

    Mirrors the router in `_torch/distributed/ops.py`: under MPI the plain symbol is
    called and `process_group` stays None; with MPI disabled the `_pg` symbol needs the
    TP process group and this rank's global rank.
    """
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm._utils import mpi_disabled

    workspace = None if strategy in _NO_WORKSPACE else get_allreduce_workspace(mapping)
    process_group, rank = None, None
    if mpi_disabled():
        process_group = mapping.tp_group_pg.boxed()
        rank = torch.distributed.get_rank()
    return dict(
        workspace=workspace,
        group=list(mapping.tp_group),
        strategy=strategy,
        process_group=process_group,
        rank=rank,
    )


class Allreduce(OpWrapper):
    """Sum `input` over `group`; with `op=RESIDUAL_RMS_NORM` also add `residual` and RMS-normalize.

    Returns fresh tensors: `(norm_out, residual_out)` for `RESIDUAL_RMS_NORM`, the one
    summed tensor for `NONE`. This is the raw op, not the `AllReduce` module, so no MNNVL
    path: across trays it falls back to NCCL.
    """

    ARCHS = frozenset({Arch.SM_103})

    _HYV4 = dict(hidden=3072, dtype=torch.bfloat16, eps=1e-5, op=PLAIN, strategy=AUTO)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the ctx engine's decode, one token, world 2",
            spec=dict(world=2, rows=1, **_HYV4),
        ),
        Cell(
            why="the ctx engine's widest decode batch, world 2",
            spec=dict(world=2, rows=64, **_HYV4),
        ),
        Cell(
            why="a 5120-token prefill chunk, world 2",
            spec=dict(world=2, rows=5120, **_HYV4),
        ),
        Cell(
            why="the ctx engine's packed prefill, 8192 tokens, world 2",
            spec=dict(world=2, rows=8192, **_HYV4),
        ),
        Cell(why="decode, one token, world 4", spec=dict(world=4, rows=1, **_HYV4)),
        Cell(why="a decode batch, world 4", spec=dict(world=4, rows=64, **_HYV4)),
        Cell(
            why="a 5120-token prefill chunk, world 4",
            spec=dict(world=4, rows=5120, **_HYV4),
        ),
        Cell(
            why="decode, one token, world 8 over two trays",
            spec=dict(world=8, rows=1, **_HYV4),
        ),
        Cell(
            why="a decode batch, world 8 over two trays",
            spec=dict(world=8, rows=64, **_HYV4),
        ),
        Cell(
            why="a 5120-token prefill chunk, world 8 over two trays",
            spec=dict(world=8, rows=5120, **_HYV4),
        ),
    )

    def __call__(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        process_group: Optional[object] = None,
        rank: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """`process_group` None runs the MPI symbol; a boxed c10d ProcessGroup runs `allreduce_pg`."""
        if process_group is None:
            outs = torch.ops.trtllm.allreduce(
                input,
                residual,
                norm_weight,
                scale,
                bias,
                workspace,
                group,
                strategy,
                op,
                eps,
                trigger_completion_at_end,
            )
        else:
            outs = torch.ops.trtllm.allreduce_pg(
                input,
                residual,
                norm_weight,
                scale,
                bias,
                workspace,
                group,
                rank,
                process_group,
                strategy,
                op,
                eps,
                trigger_completion_at_end,
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        op: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """What every rank gets back, from every rank's `input` in rank order; fp32 sum, rounded once."""
        acc = torch.zeros(inputs_in_rank_order[0].shape, dtype=torch.float32, device="cuda")
        for x in inputs_in_rank_order:
            acc += x.float()
        dtype = inputs_in_rank_order[0].dtype
        if op == int(AllReduceFusionOp.NONE):
            return acc.to(dtype)
        assert op == RESIDUAL_RMS_NORM, f"reference covers NONE and RESIDUAL_RMS_NORM, not {op}"
        h = acc + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * norm_weight.float()).to(dtype), h.to(dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        process_group: Optional[object] = None,
        rank: Optional[int] = None,
    ) -> None:
        # The kernels take raw pointers and read every operand as packed rows.
        for name, tensor in (("input", input), ("residual", residual)):
            if tensor is not None:
                assert tensor.is_contiguous(), f"{name} must be contiguous"
        if op == RESIDUAL_RMS_NORM:
            assert residual is not None and norm_weight is not None, (
                "RESIDUAL_RMS_NORM reads residual and norm_weight; a None pointer is not rejected"
            )
            assert residual.shape == input.shape, (residual.shape, input.shape)
            assert norm_weight.shape == (input.shape[-1],), (norm_weight.shape, input.shape)
        assert strategy in _NO_WORKSPACE or workspace is not None, (
            f"strategy {strategy} reads the IPC workspace; build one with allreduce_group_args"
        )


allreduce = Allreduce()

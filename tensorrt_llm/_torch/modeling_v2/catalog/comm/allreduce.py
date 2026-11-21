# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks, then add the residual and RMS-normalize, via the trtllm allreduce op."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

from .._op import Arch, Cell, OpWrapper

#: The fusion this entry certifies: `residual_out = sum(input) + residual`,
#: `norm_out = rmsnorm(residual_out) * norm_weight`.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The strategy the default targets pass. The op picks a kernel by message
#: size and topology: one-shot or two-shot over the IPC workspace inside one
#: NVLink domain, NCCL plus a separate norm kernel when the message outgrows
#: the workspace or the ranks cannot reach each other's memory.
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
    """The communicator side of the call for this rank's TP group.

    A target calls this once in `__init__` and passes `**args` at every call
    site; the values do not change for the life of the mapping.

    The IPC workspace is this op's own fixture: one per mapping, shared with
    every other user of that mapping in the process, allocated on first use.
    It exists only for strategies that read it.

    Mirrors the router in `_torch/distributed/ops.py`: under MPI the plain
    symbol is called and `process_group` stays None; with MPI disabled the
    `_pg` symbol needs the TP process group and this rank's global rank.
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
    """Sum `input` over every rank in `group`; with `op=RESIDUAL_RMS_NORM` also
    add `residual` and RMS-normalize the sum.

    Returns fresh tensors and leaves `input` and `residual` untouched -- unlike
    `flashinfer_fused_add_rmsnorm`, which writes both of its arguments. For
    `RESIDUAL_RMS_NORM` the return value is `(norm_out, residual_out)`, the
    order a call site reads as `x, residual = op(...)`; for `NONE` it is the
    one summed tensor.

    This is the raw op, not the `AllReduce` module. The module also tries
    MNNVL and symmetric-memory kernels before it reaches this op, and runs an
    autotuner over its strategies. A target that calls this entry therefore
    does not get those paths: on GB200/GB300 the module's AUTO prefers MNNVL,
    and across trays this op falls back to NCCL with an unfused norm.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The gpt-oss default target at attention TP2, TP4 and TP8, inside one
    # node: up to four ranks in a GB200/GB300 tray, eight in a B200/B300 node.
    # It calls this twice per layer, after o_proj and after the MoE, on the
    # full token set every rank holds.
    _GPT_OSS = dict(
        hidden=2880, dtype=torch.bfloat16, eps=1e-5, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )
    # The deepseek default target at attention TP2, TP4 and TP8: the same two
    # calls per layer at deepseek's hidden width.
    _DEEPSEEK = dict(
        hidden=7168, dtype=torch.bfloat16, eps=1e-6, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token, tp4", spec=dict(world=4, rows=1, **_GPT_OSS)),
        Cell(why="a decode batch, tp4", spec=dict(world=4, rows=64, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp4: the message outgrows the one-shot kernel",
            spec=dict(world=4, rows=2048, **_GPT_OSS),
        ),
        Cell(
            why="a full 8192-token prefill chunk, tp4: the widest error measured",
            spec=dict(world=4, rows=8192, **_GPT_OSS),
        ),
        Cell(why="decode, one token, tp2", spec=dict(world=2, rows=1, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp2",
            spec=dict(world=2, rows=2048, **_GPT_OSS),
        ),
        Cell(why="deepseek, decode, one token, tp4", spec=dict(world=4, rows=1, **_DEEPSEEK)),
        Cell(why="deepseek, a decode batch, tp4", spec=dict(world=4, rows=64, **_DEEPSEEK)),
        Cell(why="deepseek, prefill chunk, tp4", spec=dict(world=4, rows=2048, **_DEEPSEEK)),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp4",
            spec=dict(world=4, rows=8192, **_DEEPSEEK),
        ),
        Cell(why="deepseek, decode, one token, tp2", spec=dict(world=2, rows=1, **_DEEPSEEK)),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp2",
            spec=dict(world=2, rows=8192, **_DEEPSEEK),
        ),
        # Eight ranks in one node is a B200/B300 layout only: a GB200/GB300
        # tp8 spans two trays and takes `comm/mnnvl_allreduce` instead.
        Cell(why="decode, one token, tp8 in one node", spec=dict(world=8, rows=1, **_GPT_OSS)),
        Cell(why="a decode batch, tp8 in one node", spec=dict(world=8, rows=64, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=2048, **_GPT_OSS),
        ),
        Cell(
            why="a full 8192-token prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=8192, **_GPT_OSS),
        ),
        Cell(
            why="deepseek, decode, one token, tp8 in one node",
            spec=dict(world=8, rows=1, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, a decode batch, tp8 in one node",
            spec=dict(world=8, rows=64, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=2048, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=8192, **_DEEPSEEK),
        ),
    )

    note = """
    Only `RESIDUAL_RMS_NORM` is certified: it is the one fusion the targets
    call. `reference` also computes `NONE`; the quantizing fusions (FP8, NVFP4
    outputs) are not modelled.

    The cells run the `_pg` symbol, because the certification matrix starts its
    ranks without MPI. A served engine under `mpirun` or
    `trtllm-llmapi-launch` runs the MPI symbol. Both dispatch to the same
    `AllreduceOp` in C++; only the communicator differs.

    `trigger_completion_at_end` only changes when the next kernel may launch
    under PDL. It is not a cell axis.
    """

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
        """`process_group` selects the symbol: None runs the MPI one, a boxed
        c10d ProcessGroup runs `allreduce_pg`, which also needs `rank`.
        `allreduce_group_args` builds the communicator arguments for a mapping.
        """
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
        """What every rank should get back.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so this takes every
        rank's `input`, in rank order.

        Summed in fp32 and rounded once. For `RESIDUAL_RMS_NORM` the residual
        is added to the fp32 sum, the norm reads that fp32 value, and both
        outputs are rounded to the input dtype at the end.
        """
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
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; the kernels read it as packed rows"
                )
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The plain sum across ranks at this target's width, via the trtllm allreduce op.

Local variant of ``catalog/comm/allreduce.py`` (``__r19``) owned by the
Hy4-Flash unified target. Same op call, same reference and same guards as the
catalog entry; the cells are the surface this target calls it with: ``op=NONE``
(the plain sum, no fusion) at hidden 3072, on the 1-row, 64-row and 5120-row
ends of a world-2 group (both ranks in one node), a world-4 group and a
world-8 group over two trays. World 2 is the disaggregated deployment's
context engine (``tensor_parallel_size`` 2, ``moe_expert_parallel_size`` 2);
its token budget ``max_num_tokens`` 8192 is the widest message it forms, so
the world-2 group carries an 8192-row cell beside the 5120-row prefill.

The catalog entry certifies only ``RESIDUAL_RMS_NORM``, at the gpt-oss and
deepseek hidden widths; no cell of it runs the plain form. This variant is the
one the target imports for the row-parallel o_proj and MoE output partials, so
the plain form is certified where this target uses it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

#: The fusion this entry certifies: `residual_out = sum(input) + residual`,
#: `norm_out = rmsnorm(residual_out) * norm_weight`.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The form this target calls: the sum alone, one output tensor.
PLAIN = int(AllReduceFusionOp.NONE)

#: The strategy the target passes. The op picks a kernel by message size and
#: topology: one-shot or two-shot over the IPC workspace inside one NVLink
#: domain, NCCL when the message outgrows the workspace or the ranks cannot
#: reach each other's memory.
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

    ARCHS = frozenset({Arch.SM_103})
    # Driven on the run's sm_103 device only; sm_100 is not measured here.

    # The target's calls: after o_proj and after the MoE, on the token set
    # every rank holds, at hidden 3072. `eps` is unread for NONE.
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

    note = """
    Only `NONE` is certified here: it is the form the target's o_proj and MoE
    partials call. `reference` and `is_valid` still cover `RESIDUAL_RMS_NORM`
    because this is a copy of the catalog entry; no cell runs it.

    The world-2 group's two ranks share a node, so its cells run the IPC
    workspace path (one-shot or two-shot, AUTO by message size), and even its
    8192-row message stays inside the 64 MiB workspace; the world-8 group over
    two trays opens no IPC handles and takes the NCCL fallback. Rows 8192 is
    the widest message this target forms: the engine's token budget is
    `BaseLlmArgs.max_num_tokens` 8192, defaulted in `llmapi/llm_args.py` and
    set explicitly by the disaggregated context config. Not certified: a
    packed prefill of 8192 rows at world 4 or 8 -- those groups' cells stop at
    5120.

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

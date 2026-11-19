# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sequence-parallel prefill collectives: row all-gather and row reduce-scatter over the TP group.

A sequence-parallel prefill keeps the residual stream on ``rows`` rows per rank (rank ``r`` owns rows
``[r * rows, (r + 1) * rows)`` of the step's tokens, zero-padded to ``rows * tp``). Each layer gathers the rows
its column-parallel projections read and reduce-scatters the row-parallel partials it would otherwise all-reduce.
Both are the engine's grouped NCCL collectives (`tensorrt_llm._torch.distributed.ops.allgather` / `reducescatter`,
one grouped call for a list of tensors).
"""

from typing import List, Sequence

import torch

from tensorrt_llm._torch.distributed.ops import allgather, reducescatter
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

#: Kimi K3.1 hidden size, MoE latent width and router width.
_HIDDEN = 7168
_LATENT = 3584
_EXPERTS = 896
#: Rows per rank of a 16384-token TEP8 prefill step, and of the smallest sequence-parallel step (8192 tokens).
_ROWS_16K_TP8 = 2048
_ROWS_8K_TP8 = 1024


class SpGatherRows(OpWrapper):
    """All-gather every TP rank's ``[rows, ...]`` tensors along dim 0 into ``[rows * tp, ...]``, in rank order.

    ``inputs`` is a list of tensors (one grouped NCCL call); every rank passes the same list of shapes. Returns new
    tensors, identical on every rank of the group; the inputs are not written. A one-rank group returns the inputs.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3.1 SP attention input: normed hidden rows of a 16k TEP8 step",
            spec=dict(world=4, shapes=((_ROWS_16K_TP8, _HIDDEN),), dtypes=(torch.bfloat16,)),
        ),
        Cell(
            why="K3.1 SP MoE input: normed hidden + latent + fp32 router logits (one grouped call)",
            spec=dict(
                world=4,
                shapes=((_ROWS_8K_TP8, _HIDDEN), (_ROWS_8K_TP8, _LATENT), (_ROWS_8K_TP8, _EXPERTS)),
                dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
            ),
        ),
        Cell(
            why="K3.1 SP final norm output rows (uneven token count padded to rows * tp)",
            spec=dict(world=4, shapes=((2049, _HIDDEN),), dtypes=(torch.bfloat16,)),
        ),
    )
    note = """
    Engine-function-internal: calls `tensorrt_llm._torch.distributed.ops.allgather(inputs, mapping, dim=0)`, the
    NCCL all-gather (grouped for a list). A gather is a copy: bit-exact against the rank-order concatenation.
    Not driven: groups wider than one GB300 tray (TEP8 spans two; the collective is the same NCCL call), the
    MPI-disabled `_pg` path, CUDA-graph capture (sequence-parallel steps run eagerly).
    """

    def __call__(self, inputs: List[torch.Tensor], mapping: Mapping) -> List[torch.Tensor]:
        if mapping.tp_size == 1:
            return list(inputs)
        return allgather(list(inputs), mapping, dim=0)

    def reference(
        self, inputs_in_rank_order: Sequence[Sequence[torch.Tensor]], mapping: Mapping = None
    ) -> List[torch.Tensor]:
        """Every rank's list, in rank order (the collective exception to mirroring ``__call__``); ``mapping``
        is ignored."""
        num = len(inputs_in_rank_order[0])
        return [torch.cat([r[i] for r in inputs_in_rank_order], dim=0) for i in range(num)]


class SpReduceScatterRows(OpWrapper):
    """Sum the TP ranks' ``[rows * tp, ...]`` partials and keep this rank's ``[rows, ...]`` slice
    (rows ``[tp_rank * rows, (tp_rank + 1) * rows)``), for a list of tensors in one grouped call.

    Every rank passes the same list of shapes; dim 0 must be a multiple of the TP size. A one-rank group returns
    the inputs.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3.1 SP attention o_proj partials of a 16k TEP8 step",
            spec=dict(world=4, shapes=((_ROWS_16K_TP8 * 4, _HIDDEN),), dtypes=(torch.bfloat16,)),
        ),
        Cell(
            why="K3.1 SP MoE routed latent + shared expert partials (one grouped call)",
            spec=dict(
                world=4,
                shapes=((_ROWS_8K_TP8 * 4, _LATENT), (_ROWS_8K_TP8 * 4, _HIDDEN)),
                dtypes=(torch.bfloat16, torch.bfloat16),
            ),
        ),
    )
    note = """
    Engine-function-internal: calls `tensorrt_llm._torch.distributed.ops.reducescatter(inputs, mapping, dim=0)`,
    the NCCL reduce-scatter (grouped for a list) with BF16 accumulation in NCCL's ring order. Not bitwise to the
    all-reduce it replaces in a non-sequence-parallel forward (other reduction order and rounding); the test gates
    it against an fp32 sum within a BF16 band. Not driven: groups wider than one tray, CUDA-graph capture.
    """

    def __call__(self, inputs: List[torch.Tensor], mapping: Mapping) -> List[torch.Tensor]:
        if mapping.tp_size == 1:
            return list(inputs)
        return reducescatter(list(inputs), mapping, dim=0)

    def reference(
        self, inputs_in_rank_order: Sequence[Sequence[torch.Tensor]], mapping: Mapping
    ) -> List[torch.Tensor]:
        """Every rank's list, in rank order (the collective exception); returns this ``mapping.tp_rank``'s
        slice of the fp32 sum, rounded once to the input dtype."""
        num = len(inputs_in_rank_order[0])
        out = []
        for i in range(num):
            total = torch.stack([r[i].float() for r in inputs_in_rank_order]).sum(0)
            rows = total.shape[0] // mapping.tp_size
            out.append(
                total[mapping.tp_rank * rows : (mapping.tp_rank + 1) * rows].to(
                    inputs_in_rank_order[0][i].dtype
                )
            )
        return out


sp_gather_rows = SpGatherRows()
sp_reduce_scatter_rows = SpReduceScatterRows()

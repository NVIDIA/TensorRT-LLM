# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built-in TP all-reduce, fused with the residual add and the next RMSNorm."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams


class M3AllReduce(OpWrapper):
    """``comm(x, all_reduce_params=params)``: the runtime's ``AllReduce`` over the TP group. ``params`` None: the
    plain sum over the ranks (the drafter's o_proj and down projection). With ``RESIDUAL_RMS_NORM`` params:
    ``updated = residual + sum_ranks(x)``, ``normed = rmsnorm(updated) * norm_weight``;
    returns (normed, updated). The strategy (``AllReduceStrategy.AUTO``: one-shot / two-shot Lamport, NCCL) is the
    communicator's per call.

    The layer boundaries the multicast all-reduce does not take run it: prefill and mixed steps, and generation steps
    beyond the multicast kernel's rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # No cell names a world size: the result does not depend on it, and the certifying matrix runs at every world
    # size one node can form (the target's two-tray group of 8 cannot be formed on one node).
    CELLS: tuple[Cell, ...] = (
        Cell(why="a prefill chunk", spec=dict(m=2048, hidden=6144)),
        Cell(why="a 64-row generation step", spec=dict(m=64, hidden=6144)),
        Cell(why="plain sum of eight drafter rows", spec=dict(m=8, hidden=6144, plain=True)),
    )

    note = """
    The reduction order across ranks is the strategy's; `reference` sums the ranks' rows in fp32 in rank order, then
    adds the residual and normalizes (one bf16 rounding of the sum and one of the norm). A strategy may round the sum
    to bf16 before the residual add, so the residual sum agrees within one bf16 ulp of its operands' magnitude.
    """

    def __call__(self, x: torch.Tensor, comm: AllReduce, params: Optional[AllReduceParams]):
        return comm(x, all_reduce_params=params)

    def reference(self, xs, comm: AllReduce, params: Optional[AllReduceParams]):
        """``xs``: every rank's ``x`` in rank order (a collective's reference takes the group's inputs)."""
        total = torch.stack([x.float() for x in xs]).sum(0)
        if params is None:
            return total.to(xs[0].dtype)
        updated = (params.residual.float() + total).to(xs[0].dtype)
        h = updated.float()
        normed = (
            h
            * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + params.eps)
            * params.norm_weight.float()
        )
        return normed.to(xs[0].dtype), updated


m3_allreduce = M3AllReduce()

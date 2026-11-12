# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across the TP group through a multicast NVLink (MNNVL) workspace, nothing fused.

This target's variant of `catalog/comm/mnnvl_allreduce` (which certifies only the residual +
RMSNorm fusion) for the decode-sized calls of `tp_sum`: the op's plain sum (fusion `NONE`)."""

from __future__ import annotations

from typing import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allreduce import (
    mnnvl_workspace_args,
    workspace_bytes,
)
from tensorrt_llm.functional import AllReduceFusionOp

NONE = int(AllReduceFusionOp.NONE)

HIDDEN = 2560

__all__ = ["mnnvl_tp_sum", "mnnvl_workspace_args"]

_TP_SUM = dict(hidden=HIDDEN, dtype=torch.bfloat16)


class MnnvlTpSum(OpWrapper):
    """Sum `input` (`[rows, HIDDEN]` bf16) over the TP group; returns a fresh tensor. The
    workspace is built once by `mnnvl_workspace_args` (a TP collective) for a row budget the
    kernel does not check; `is_valid` does."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"tp_sum, {why}, tp{world}", spec=dict(world=world, rows=rows, **_TP_SUM))
        for world, rows, why in (
            (4, 1, "one draft row"),
            (4, 32, "an MTP-3 verify of 8 requests"),
            (2, 4, "an MTP-3 verify of 1 request"),
        )
    )

    def __call__(
        self, input: torch.Tensor, comm_buffer: torch.Tensor, buffer_flags: torch.Tensor, world: int
    ) -> torch.Tensor:
        """`world` is not passed to the op, which reads it from the workspace; it is here so
        `is_valid` can size the call against the buffer."""
        return torch.ops.trtllm.mnnvl_fusion_allreduce(
            input, None, None, 1e-6, comm_buffer, buffer_flags, False, None, NONE
        )[0]

    def reference(self, inputs_in_rank_order: Sequence[torch.Tensor]) -> torch.Tensor:
        """The fp32 sum of every rank's `input`, in rank order, rounded once. The documented
        collective exception to mirroring `__call__`."""
        total = torch.zeros_like(inputs_in_rank_order[0], dtype=torch.float32)
        for x in inputs_in_rank_order:
            total += x.float()
        return total.to(inputs_in_rank_order[0].dtype)

    def is_valid(
        self, input: torch.Tensor, comm_buffer: torch.Tensor, buffer_flags: torch.Tensor, world: int
    ) -> None:
        hidden = input.shape[-1]
        need = workspace_bytes(input.numel() // hidden, hidden, world, input.dtype)
        have = comm_buffer.shape[-1] * comm_buffer.element_size()
        assert need <= have, (need, have)


mnnvl_tp_sum = MnnvlTpSum()

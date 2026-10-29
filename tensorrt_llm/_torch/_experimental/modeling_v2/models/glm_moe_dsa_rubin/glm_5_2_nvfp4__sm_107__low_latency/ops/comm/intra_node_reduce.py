# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 residual/RMSNorm, MoE finalize and LINEAR FP4 on one retained IPC workspace.

All calls on an instance must be stream ordered, including engine collective fallbacks.
The shared workspace's prior dirty range may be larger than the current rows. The native
kernels clear that complete range and preserve the shared rotation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

from ._ipc_workspace import (
    IpcWorkspace,
    bind_ipc_workspace,
    residual_norm_reference,
    validate_ipc_operands,
)
from .intra_node_moe_finalize import intra_node_moe_finalize
from .pre_moe_fp4_linear import pre_moe_fp4_linear


class GlmIntraNodeReduce(OpWrapper):
    """BF16 residual/RMSNorm over rank-ordered partials on the bound TP4 IPC workspace."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(why=f"TP4 verify/draft rows={rows}", spec=dict(rows=rows, hidden=6144, world=4))
        for rows in (1, 2, 3, 6, 12, 18)
    )
    note = "Opaque peer payload supports eager/CUDA graph only; actual flags/layout are mutated."

    def __call__(
        self,
        input,
        residual,
        gamma,
        epsilon,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
        trigger_completion_at_end,
    ):
        return torch.ops.trtllm.glm_intra_node_reduce(
            input,
            residual,
            gamma,
            epsilon,
            table,
            flags,
            layout,
            rank,
            nranks,
            buffer_bytes,
            trigger_completion_at_end,
        )

    def reference(self, per_rank_inputs, residual, gamma, epsilon, *, fp32_acc=True):
        total = per_rank_inputs[0].float()
        for value in per_rank_inputs[1:]:
            total = (
                total + value.float() if fp32_acc else (total.to(torch.bfloat16) + value).float()
            )
        return residual_norm_reference(total, residual, gamma, epsilon)

    def is_valid(
        self,
        input,
        residual,
        gamma,
        epsilon,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
        trigger_completion_at_end,
    ):
        validate_ipc_operands(residual, gamma, table, flags, layout, rank, nranks, buffer_bytes)
        assert input.shape == residual.shape and input.dtype == residual.dtype
        assert input.is_contiguous() and input.device == residual.device
        assert input.data_ptr() % 16 == residual.data_ptr() % 16 == gamma.data_ptr() % 16 == 0


intra_node_reduce = GlmIntraNodeReduce()


@dataclass(frozen=True)
class IpcCollectives:
    workspace: IpcWorkspace

    @classmethod
    def bind(
        cls,
        mapping: Mapping,
        workspaces: Sequence[torch.Tensor | None],
        *,
        max_rows: int = 24,
        hidden: int = 6144,
    ) -> tuple[IpcCollectives | None, str]:
        workspace, reason = bind_ipc_workspace(
            mapping, workspaces, max_rows=max_rows, hidden=hidden
        )
        return (cls(workspace) if workspace is not None else None), reason

    def supported(self, rows: int, *, finalize: bool = False) -> bool:
        if torch.compiler.is_compiling():
            return False
        allowed = self.workspace.finalize_rows if finalize else self.workspace.plain_rows
        return rows in allowed

    def _check(self, rows: int, *, finalize: bool = False):
        if not self.supported(rows, finalize=finalize):
            raise ValueError(
                "unsupported owned IPC call; select the engine path before writing state"
            )

    def plain_reduce(
        self,
        norm,
        partial: torch.Tensor,
        residual: torch.Tensor,
        *,
        trigger_completion_at_end: bool = True,
    ):
        """FP32 rank sum (unless explicitly opted out), BF16 residual add, RMSNorm."""
        self._check(partial.shape[0])
        return intra_node_reduce(
            partial,
            residual,
            norm.weight,
            norm.variance_epsilon,
            *self.workspace.args(),
            trigger_completion_at_end,
        )

    def finalize(
        self,
        fc2: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_rows: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        norm,
    ):
        """Local top-k in FP32, one BF16 rounding, rank-order BF16 sum, residual/RMSNorm."""
        self._check(residual.shape[0], finalize=True)
        return intra_node_moe_finalize(
            fc2,
            expert_weights,
            permuted_rows,
            shared,
            residual,
            norm.weight,
            norm.variance_epsilon,
            *self.workspace.args(),
        )

    def premoe(self, partial: torch.Tensor, residual: torch.Tensor, norm, scale: torch.Tensor):
        """Return norm/residual plus packed FP4 and LINEAR SF using FC31 input scale.

        ``scale`` is backend.fc31_input_scale: one CUDA float32 value. Packed output is
        uint8 [M,3072], scale factors uint8 [M,384]. The consumer must bypass requantization.
        ``residual`` must be ready before the producer's PDL trigger: this kernel reads it
        before its dependency wait. Newly produced residuals need an ordinary stream dependency.
        """
        self._check(partial.shape[0])
        return pre_moe_fp4_linear(
            partial, residual, norm.weight, norm.variance_epsilon, scale, *self.workspace.args()
        )

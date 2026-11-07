# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local top-k FP32 finalize followed by rank-order BF16 IPC sum and residual/RMSNorm."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._ipc_workspace import residual_norm_reference, validate_ipc_operands


class GlmIntraNodeMoeFinalize(OpWrapper):
    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(
            why=f"TP4 routed verify rows={rows}",
            spec=dict(rows=rows, hidden=6144, world=4, top_k=8),
        )
        for rows in (1, 3, 6, 12, 18)
    )
    note = (
        "Uses real IPC flags/layout and preserves BF16 rank addition; no functionalization support."
    )

    def __call__(
        self,
        fc2,
        weights,
        indices,
        shared,
        residual,
        gamma,
        epsilon,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
    ):
        return torch.ops.trtllm.glm_intra_node_moe_finalize(
            fc2,
            weights,
            indices,
            shared,
            residual,
            gamma,
            epsilon,
            table,
            flags,
            layout,
            rank,
            nranks,
            buffer_bytes,
        )

    def reference(
        self,
        per_rank_fc2,
        per_rank_weights,
        per_rank_indices,
        per_rank_shared,
        residual,
        gamma,
        epsilon,
    ):
        local = []
        for fc2, weights, indices, shared in zip(
            per_rank_fc2, per_rank_weights, per_rank_indices, per_rank_shared
        ):
            acc = torch.zeros_like(residual, dtype=torch.float32)
            for slot in range(indices.shape[1]):
                ids = indices[:, slot].long()
                rows = fc2[ids.clamp_min(0)].float()
                rows = torch.where((ids >= 0).unsqueeze(-1), rows, 0)
                acc = acc + rows * weights[:, slot].float().unsqueeze(-1)
            local.append((acc + shared.float()).to(torch.bfloat16))
        total = local[0]
        for value in local[1:]:
            total = total + value
        return residual_norm_reference(total, residual, gamma, epsilon)

    def is_valid(
        self,
        fc2,
        weights,
        indices,
        shared,
        residual,
        gamma,
        epsilon,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
    ):
        validate_ipc_operands(residual, gamma, table, flags, layout, rank, nranks, buffer_bytes)
        assert fc2.ndim == 2 and fc2.shape[1] == residual.shape[1]
        assert indices.ndim == 2 and indices.shape[0] == residual.shape[0]
        assert 1 <= indices.shape[1] <= 8 and indices.dtype == torch.int32
        assert weights.shape == indices.shape and weights.dtype == torch.bfloat16
        assert shared.shape == residual.shape and shared.dtype == fc2.dtype == torch.bfloat16
        for value in (fc2, weights, indices, shared):
            assert value.is_contiguous() and value.device == residual.device
        assert ((indices >= -1) & (indices < fc2.shape[0])).all().item()


intra_node_moe_finalize = GlmIntraNodeMoeFinalize()

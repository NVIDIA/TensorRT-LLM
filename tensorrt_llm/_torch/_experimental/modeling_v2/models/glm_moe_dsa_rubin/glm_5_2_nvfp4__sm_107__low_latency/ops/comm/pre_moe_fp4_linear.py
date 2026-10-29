# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared native residual/RMSNorm/FP4 kernel with an owned LINEAR-scale consumer contract."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._ipc_workspace import residual_norm_reference, validate_ipc_operands


class GlmPreMoeFp4Linear(OpWrapper):
    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(why=f"TP4 target pre-MoE FP4 rows={rows}", spec=dict(rows=rows, hidden=6144, world=4))
        for rows in (1, 3, 18)
    )
    note = "scale is FC31 input scale; the bound table indirectly writes the real flags/layout operands."

    def __call__(
        self,
        input,
        residual,
        gamma,
        epsilon,
        scale,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
    ):
        return torch.ops.trtllm.glm_pre_moe_fp4_linear(
            input, residual, gamma, epsilon, scale, table, flags, layout, rank, nranks, buffer_bytes
        )

    def reference(self, per_rank_inputs, residual, gamma, epsilon, scale, *, fp32_acc=True):
        total = per_rank_inputs[0].float()
        for value in per_rank_inputs[1:]:
            total = (
                total + value.float() if fp32_acc else (total.to(torch.bfloat16) + value).float()
            )
        norm, res = residual_norm_reference(total, residual, gamma, epsilon)
        # Quantization's shared plain-Torch oracle, not the native op under test.
        from ..quantization.fp4_quantize import fp4_quantize

        packed, sf = fp4_quantize.reference(norm, scale, 16, False, False)
        return norm, res, packed, sf

    def is_valid(
        self,
        input,
        residual,
        gamma,
        epsilon,
        scale,
        table,
        flags,
        layout,
        rank,
        nranks,
        buffer_bytes,
    ):
        validate_ipc_operands(residual, gamma, table, flags, layout, rank, nranks, buffer_bytes)
        assert input.shape == residual.shape and input.dtype == residual.dtype
        assert input.is_contiguous() and input.device == residual.device
        assert scale.dtype == torch.float32 and scale.numel() == 1 and scale.is_contiguous()
        assert scale.device == residual.device


pre_moe_fp4_linear = GlmPreMoeFp4Linear()

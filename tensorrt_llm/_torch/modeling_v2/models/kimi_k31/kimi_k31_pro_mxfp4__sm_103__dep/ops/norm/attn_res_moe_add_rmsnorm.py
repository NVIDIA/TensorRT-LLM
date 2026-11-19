# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE residual update + attention-residual mix + RMSNorm through the family's PDL kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm import _attn_res_common as _c

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import (
        attn_res as _k3_attn_res,
    )


class K31AttnResMoeAddRmsnorm(OpWrapper):
    """``(updated, out)``: ``updated = bf16(prefix + bf16(routed + shared))``, then ``out`` as in
    ``k31_attn_res_rmsnorm`` with ``updated`` as the layer row, all ``[M, 7168]`` BF16.

    ``updated`` has the bits of the two torch adds it replaces (``prefix + (routed + shared)``:
    the MoE block's sum of its routed and shared outputs and the decoder layer's residual add,
    both round-to-nearest FP32 adds rounded to BF16), so the two add kernels and the re-read of
    their results leave the step. ``out`` is the mix + RMSNorm of ``k31_attn_res_rmsnorm``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _c.cells(add=True)

    note = """
    The next layer's pre-attention mix in a decode / verify step: the layer's MoE output stays as
    its routed and shared halves and this op adds both to the running residual. Same scope as
    k31_attn_res_rmsnorm (generation servers, up to 192 rows per rank with attention TP and 256
    with attention DP); the add is bitwise, the mix + RMSNorm output is one BF16 step from the
    engine attn_res ops in a small fraction of the elements.
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k3_attn_res.k31_attn_res_moe_add_rmsnorm(
            prefix, routed, shared, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )

    def reference(
        self,
        prefix: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated = prefix + (routed + shared)
        out = _c.reference(updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)
        return updated, out

    def is_valid(
        self,
        prefix: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        _c.check_operands(
            prefix, block, res_weight, rms_weight, out_weight, addends=(routed, shared)
        )


k31_attn_res_moe_add_rmsnorm = K31AttnResMoeAddRmsnorm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 residual add + attention-residual mix + RMSNorm through the family's PDL kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _attn_res_common as _c

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import (
        attn_res as _k3_attn_res,
    )


class K31AttnResAddRmsnorm(OpWrapper):
    """``(updated, out)``: ``updated = bf16(prefix + addend)``, then ``out`` as in
    ``k31_attn_res_rmsnorm`` with ``updated`` as the layer row, both ``[M, 7168]`` BF16.

    ``updated`` is bit-identical to ``trtllm::attn_res_add_rmsnorm_fwd``'s; ``out`` can differ from
    its output by one BF16 rounding step in a small fraction of the elements (statistics summed in
    another order).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = _c.cells(add=True)

    note = """
    Same scope as k31_attn_res_rmsnorm: disaggregated generation servers, at most
    min(max_num_tokens, 192) rows per rank with attention TP and 256 with attention DP.
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k3_attn_res.k31_attn_res_add_rmsnorm(
            prefix, addend, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )

    def reference(
        self,
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated = (prefix.float() + addend.float()).to(torch.bfloat16)
        out = _c.reference(updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)
        return updated, out

    def is_valid(
        self,
        prefix: torch.Tensor,
        addend: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        _c.check_operands(prefix, block, res_weight, rms_weight, out_weight, addends=(addend,))


k31_attn_res_add_rmsnorm = K31AttnResAddRmsnorm()

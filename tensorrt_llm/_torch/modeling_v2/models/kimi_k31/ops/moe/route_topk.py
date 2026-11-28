# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routing top-16 with BF16 weights, through the family's routing-tail kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import (
        route_tail as _k3_route_tail,
    )

#: Experts per token the kernel is built for (Kimi K3.1).
TOP_K = 16


def _dep_rows(rows: int, why: str, scale: float = 1.0) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, scale=scale))


class K31RouteTopk(OpWrapper):
    """Top-16 of 896 experts per row by ``sigmoid(logit) + bias``; BF16 combine weights.

    Selection uses ``sigmoid(router_logits) + e_score_correction_bias``; the weights are the
    unbiased ``sigmoid`` scores of the selected experts, renormalized to sum to 1 and scaled by
    ``routed_scaling_factor``, then rounded to BF16. This is ``noaux_tc_op`` with one expert group
    (``num_expert_group == topk_group == 1``) followed by the cast of its FP32 weights to BF16 that
    the routed-expert scheduler applies for a backend that reads BF16 routing scales (TRTLLM-Gen),
    bit for bit.

    Returns ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16])``, newly allocated: the order of
    ``noaux_tc``'s result, so the separated routing can hand it out as is.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # The default target's attention-DP MoE routes 65..2048 rows of a rank through this kernel
    # (below 65 rows it runs the fused routing + quantization op). The kernel's own range is
    # 1..2048 rows, in CTAs of 8 (up to 1024 rows) or 16 rows.
    CELLS: tuple[Cell, ...] = (
        _dep_rows(1, "one row: the smallest grid (one 8-row CTA, seven idle rows)"),
        _dep_rows(65, "first row count the attention-DP MoE routes here"),
        _dep_rows(448, "DEP16 at 28 sequences x 16 rows: a whole number of 8-row CTAs"),
        _dep_rows(1024, "last row count of 8-row CTAs"),
        _dep_rows(1025, "first row count of 16-row CTAs, a partial last CTA"),
        _dep_rows(2048, "largest launch: 128 CTAs of 16 rows"),
        _dep_rows(
            300,
            "a scale other than 1: the FP64 renormalization instead of the unit-scale FP32 path",
            scale=2.5,
        ),
    )

    note = """
    The Kimi K3.1 checkpoint routes with one expert group and routed_scaling_factor 1.0. The
    kernel serves exactly 896 experts and top-16 (compile-time constants); other routers keep
    noaux_tc_op. Logits must be FP32 and the bias FP32 (the kernel reads both as dense,
    16-byte-aligned buffers and raises otherwise). Ties in the selection score go to the lower
    expert id, as in noaux_tc_op. The kernel keeps no state between launches, so a captured
    CUDA graph replays it as is.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k3_route_tail.k31_route_topk(
            router_logits, e_score_correction_bias, routed_scaling_factor
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native torch, in ``noaux_tc_op``'s arithmetic (tanh-form sigmoid, FP64 renormalization
        rounded to FP32), then the BF16 cast."""
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        choice = scores + e_score_correction_bias.float()
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
        weights = torch.gather(scores, 1, ids)
        total = weights.sum(-1, keepdim=True).double()
        weights = weights.double() / (total + 1e-20) * routed_scaling_factor
        return ids.to(torch.int32), weights.float().to(torch.bfloat16)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> None:
        """Nothing to add: the op itself raises (ValueError) on every input it would answer
        wrongly -- a dtype other than FP32, a width other than 896, a strided or misaligned
        tensor, a bias of another size, a row count outside 1..2048 -- which the test drives."""
        return None


k31_route_topk = K31RouteTopk()

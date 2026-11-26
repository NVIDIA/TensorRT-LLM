# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's MoE router: fp32 sigmoid scores, bias-corrected top-k selection, renormalized
weights.

    logits  = x @ W^T                         fp32 (bf16 operands, fp32 accumulation)
    scores  = sigmoid(logits)
    experts = top_k(scores + bias)            the bias steers selection only
    weights = scores[experts] / (sum(scores[experts]) + 1e-20) * route_scale
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

_INKLING2 = dict(hidden=4608, experts=384, top_k=8, route_scale=8.0)


class Inkling2Router(OpWrapper):
    """`x` `[T, H]` bf16, `weight` `[E, H]` bf16, `bias` `[E]` fp32 -> (`topk_ids` `[T, k]` int32,
    `topk_weights` `[T, k]` fp32). The ids are global expert ids, in no particular order."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="Inkling-2-Small, one decode token", spec=dict(_INKLING2, tokens=1)),
        Cell(why="Inkling-2-Small, 257 tokens", spec=dict(_INKLING2, tokens=257)),
    )

    note = """
    The selection adds the correction bias to the sigmoid scores, but the weights come from
    the uncorrected scores, renormalized over the chosen experts and multiplied by
    route_scale (8). A weight taken from the biased scores, or a softmax router, still sums
    to route_scale; only the per-expert weights show the difference.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        top_k: int,
        route_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = cublas_mm(x, weight.t(), None, torch.float32)
        scores = torch.sigmoid(logits)
        ids = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * route_scale
        return ids.to(torch.int32), weights

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        top_k: int,
        route_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 logits from a true fp32 product; selection and weights as the op."""
        with true_fp32_matmul():
            logits = x.float() @ weight.float().t()
        scores = torch.sigmoid(logits)
        ids = torch.topk(scores + bias.float(), top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * route_scale
        return ids.to(torch.int32), weights

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        top_k: int,
        route_scale: float,
    ) -> None:
        # A bf16 bias rounds the correction and changes which experts win; an fp32 weight
        # would need a different GEMM. Neither raises.
        assert bias.dtype == torch.float32, "the correction bias is fp32"
        assert x.dtype == weight.dtype == torch.bfloat16, "x and the router weight are bf16"
        assert weight.shape == (bias.shape[0], x.shape[1]), "weight must be [experts, hidden]"


inkling2_router = Inkling2Router()

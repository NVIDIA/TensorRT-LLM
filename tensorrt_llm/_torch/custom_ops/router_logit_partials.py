# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE router logits as K-chunk partials, for a consumer that sums them while it reads them.

For mid-size token counts a router GEMM is latency-bound, and cuBLAS runs it as a split-K GEMM plus
a separate reduce launch. Here the GEMM writes ``S`` K-chunk partials in one strided-batched launch,
and the routing launch (``dsv4_route_nvfp4_quant``) sums them in the fixed order ``s = 0 .. S-1``.
The logits change only in summation order; the sum is deterministic.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

# Token counts served by partials (up to the hi GEN batch, 320 requests x 4 rows); the router GEMM keeps its
# other paths outside them.
MIN_ROWS = 65
MAX_ROWS = 1280


class RouterLogitPartials:
    """FP32 router logits of ``M`` tokens over ``N`` experts as ``S`` K-chunk partials.

    ``parts`` is ``[S, M, N]`` (rows may be a sliced view). It supports only the row slicing and
    splitting a MoE scheduler applies to logits; ``dsv4_route_nvfp4_quant`` is its one reader.
    """

    __slots__ = ("parts",)

    def __init__(self, parts: torch.Tensor):
        if parts.dim() != 3 or parts.dtype != torch.float32 or parts.stride(2) != 1:
            raise ValueError(
                f"expected FP32 [S, M, N] partials with unit expert stride, got "
                f"{parts.dtype} {tuple(parts.shape)} {parts.stride()}"
            )
        self.parts = parts

    def __getitem__(self, rows: slice) -> "RouterLogitPartials":
        if not isinstance(rows, slice):
            raise TypeError("router logit partials take row slices only")
        return RouterLogitPartials(self.parts[:, rows])

    def split(self, sizes: Sequence[int]) -> List["RouterLogitPartials"]:
        return [RouterLogitPartials(p) for p in self.parts.split(list(sizes), dim=1)]


def num_splits(num_tokens: int, num_experts: int) -> int:
    """K chunks: the fastest strided-batched GEMM plus route+quant on GB300 for K = 5120; fewer as the logits
    grow, since route+quant reads all ``S x M x N`` partials."""
    num_logits = num_tokens * num_experts
    return 8 if num_logits <= 256 * 384 else 4 if num_logits <= 768 * 384 else 2


def applies(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``router_logit_partials`` serves ``x @ weight.T``."""
    num_tokens, hidden = x.shape
    return (
        MIN_ROWS <= num_tokens <= MAX_ROWS
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.stride(1) == 1
        and weight.is_contiguous()
        and weight.shape[1] == hidden
        and hidden % num_splits(num_tokens, weight.shape[0]) == 0
    )


def router_logit_partials(x: torch.Tensor, weight: torch.Tensor) -> RouterLogitPartials:
    """``x @ weight.T`` (BF16 ``[M, K]`` and ``[N, K]``) as FP32 K-chunk partials, one launch."""
    num_tokens, hidden = x.shape
    num_experts = weight.shape[0]
    splits = num_splits(num_tokens, num_experts)
    chunk = hidden // splits
    a = x.as_strided((splits, num_tokens, chunk), (chunk, x.stride(0), 1))
    b = weight.as_strided((splits, chunk, num_experts), (chunk, 1, hidden))
    return RouterLogitPartials(torch.bmm(a, b, out_dtype=torch.float32))

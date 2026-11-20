# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROUTER_BF16S (K5 U4): FP32-grade MiniMax-M3 router logits from BF16 rows by one cuBLAS BF16 GEMM against the FP32
router weight split into BF16 terms stacked along N.

``split_router_weight_bf16`` (the stock gate's split) lays the ``terms`` BF16 terms of the FP32 weight ``W`` [E, H]
side by side along K, so the stock GEMM reads ``cat([x] * terms, -1)``: ``terms`` copies of the activation. Here the
same terms are stacked along N, ``wn`` [terms * E, H] term-major (rows ``t * E .. t * E + E - 1`` hold term ``t``),
so the GEMM reads ``x`` once and writes ``terms`` partial logits per expert, which one reduce adds:

    logits = cublas_mm(x, wn.t(), None, float32).view(M, terms, E).sum(1)

BF16 x BF16 products are exact in FP32 and cuBLAS accumulates in FP32 (CUBLAS_COMPUTE_32F), so the weight carries 16
(two terms) or 24 (three terms) significant bits against TF32's 11: closer to the FP32 / FP64 product than a TF32
``F.linear`` on the widened rows. NOT bitwise against either the TF32 router or the stock K-stacked gate (the FP32
accumulation order differs). A target owns the switch, the terms count and the engage rules.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.models.modeling_minimaxm3 import split_router_weight_bf16

#: The terms counts a target may pick: 2 (16 weight bits) or 3 (24 bits, the stock gate's ROUTER_BF16_SPLIT_TERMS).
TERMS_CHOICES = (2, 3)
DEFAULT_TERMS = 2


def parse_terms(value: str, name: str = "terms") -> int:
    """The terms count from an environment value (empty = DEFAULT_TERMS); anything outside TERMS_CHOICES raises,
    naming ``name``, so a typo fails at import instead of silently running another split."""
    word = value.strip()
    if not word:
        return DEFAULT_TERMS
    if word not in tuple(str(t) for t in TERMS_CHOICES):
        raise ValueError(f"{name}={value!r}: expected one of {TERMS_CHOICES}")
    return int(word)


def n_stacked_split(weight: torch.Tensor, terms: int) -> torch.Tensor:
    """``weight`` [E, H] FP32 as ``terms`` BF16 terms stacked along N: a contiguous [terms * E, H] BF16 tensor whose
    rows ``t * E .. t * E + E - 1`` are term ``t`` of ``split_router_weight_bf16(weight, terms)`` (the same values)."""
    if terms not in TERMS_CHOICES:
        raise ValueError(f"terms {terms}: expected one of {TERMS_CHOICES}")
    k_stacked = split_router_weight_bf16(weight, terms)  # [E, terms * H]
    experts, hidden = weight.shape
    return (
        k_stacked.view(experts, terms, hidden)
        .transpose(0, 1)
        .reshape(terms * experts, hidden)
        .contiguous()
    )


def logits(x: torch.Tensor, wn: torch.Tensor, terms: int) -> torch.Tensor:
    """FP32 router logits [M, E] (contiguous, fresh) of BF16 rows ``x`` [M, H] (contiguous) and ``wn`` from
    :func:`n_stacked_split`: one cuBLAS GEMM ([M, terms * E] FP32) and one reduce over the terms."""
    out = torch.ops.trtllm.cublas_mm(x, wn.t(), bias=None, out_dtype=torch.float32)
    return out.view(x.shape[0], terms, wn.shape[0] // terms).sum(dim=1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lossless BF16 fragment layout and draft-head reference arithmetic."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, Optional, Protocol

import torch


class DraftHeadTarget(Protocol):
    from_fragments: Callable[..., torch.Tensor]
    decode_coded: Callable[..., torch.Tensor]
    head_weight: Callable[..., torch.Tensor]
    true_fp32_matmul: Callable[..., AbstractContextManager[None]]


def bf16_fragments(state: DraftHeadTarget, /, weight: torch.Tensor) -> torch.Tensor:
    """BF16 [N, K] -> the same values in A-fragment order [N / 16, K / 16, 256].

    One lane's 16 bytes of a 16-row tile's k16 step are the m16n8k16 A fragment: element (row h * 8 + g, k ks * 16 +
    hk * 8 + jj * 2 + e) goes to [tile][ks][g][jj][hk][h][e] (lane = g * 4 + jj).
    """
    n, k = weight.shape
    assert n % 16 == 0 and k % 16 == 0 and (weight.dtype == torch.bfloat16), (n, k, weight.dtype)
    q = weight.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return q.view(n // 16, k // 16, 256)


def from_fragments(state: DraftHeadTarget, /, wf: torch.Tensor) -> torch.Tensor:
    """The inverse of ``bf16_fragments``: [N / 16, K / 16, 256] -> [N, K]."""
    tiles, ks, _ = wf.shape
    return (
        wf.view(tiles, ks, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(tiles * 16, ks * 16)
    )


def head_weight(
    state: DraftHeadTarget,
    /,
    wf: torch.Tensor,
    k: int,
    exc_off: Optional[torch.Tensor] = None,
    exc_pos: Optional[torch.Tensor] = None,
    exc_w: Optional[torch.Tensor] = None,
    exp_base: int = 0,
) -> torch.Tensor:
    """The BF16 head shard [N, K] the kernel multiplies by: the fragment-ordered copy, or (``exc_off`` given) the
    12-bit coded stream with its exception lists decoded."""
    if exc_off is None:
        return state.from_fragments(wf)
    n = (exc_off.numel() - 1) * 16
    return state.decode_coded(wf, exp_base, exc_off, exc_pos, exc_w, n, k)


def shard_logits(
    state: DraftHeadTarget,
    /,
    x: torch.Tensor,
    wf: torch.Tensor,
    exc_off: Optional[torch.Tensor] = None,
    exc_pos: Optional[torch.Tensor] = None,
    exc_w: Optional[torch.Tensor] = None,
    exp_base: int = 0,
) -> torch.Tensor:
    """The in-tree head's BF16 logits of the rows x [M, K] against the shard (``head_weight``): the fp32 product,
    rounded once to BF16 (the BF16 GEMM's output), as fp32."""
    w = state.head_weight(wf, x.shape[1], exc_off, exc_pos, exc_w, exp_base)
    with state.true_fp32_matmul():
        logits = x.float() @ w.float().t()
    return logits.to(torch.bfloat16).float()

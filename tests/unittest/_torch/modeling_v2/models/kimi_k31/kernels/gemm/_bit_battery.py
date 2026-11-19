# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bit battery the Kimi K3.1 bitwise GEMM tests compare a kernel's output with a reference GEMM's on."""

from typing import List

#: Inputs per bit check (:func:`bit_battery`): random ones, then four of each structured kind.
BIT_INPUTS = 32


def bit_battery(
    m: int, n: int, k: int, device="cuda", seed: int = 0, count: int = BIT_INPUTS
) -> List[tuple]:
    """``count`` BF16 ``(a [m, k], weight [n, k])`` pairs on which a GEMM's bits are compared with the reference's.

    Two kernels that sum K in different orders give FP32 sums a few ulps apart, which the BF16 rounding hides on most
    elements, so random inputs alone can pass by luck when few elements are compared. The last 16 pairs are structured
    to expose the order: rows that all equal one row; magnitudes mixed over 2^-12..2^12 (activations) and
    2^-6..2^6 (weights); exact cancellation (the second half of K, or a permutation of it, repeats the first with the
    weight's sign flipped, so the output is the summation's own rounding error); and a few large spikes in small
    activations. The first ``count - 16`` pairs are random, half with weights of the checkpoint's scale (N(0, 0.02))
    and half normalized by sqrt(K)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    def bf16(t):
        return t.to(torch.bfloat16)

    def uniform(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g, device=device)

    pairs = []
    for i in range(count - 16):
        w = randn(n, k) * 0.02 if i % 2 else randn(n, k) / k**0.5
        pairs.append((bf16(randn(m, k)), bf16(w)))
    for _ in range(4):  # every row the same
        pairs.append((bf16(randn(1, k).expand(m, k).contiguous()), bf16(randn(n, k) / k**0.5)))
    for _ in range(4):  # mixed magnitudes
        a = randn(m, k) * torch.exp2(uniform(-12.0, 12.0, m, k).round())
        w = randn(n, k) / k**0.5 * torch.exp2(uniform(-6.0, 6.0, n, k).round())
        pairs.append((bf16(a), bf16(w)))
    half = k // 2
    for i in range(4):  # exact cancellation: halves, or a permutation of the second half
        u, v = bf16(randn(m, half)), bf16(randn(n, half) / k**0.5)
        perm = (
            torch.randperm(half, generator=g, device=device)
            if i % 2
            else torch.arange(half, device=device)
        )
        a = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
        w = torch.zeros(n, k, dtype=torch.bfloat16, device=device)
        a[:, :half], a[:, half : 2 * half] = u, u[:, perm]
        w[:, :half], w[:, half : 2 * half] = v, -v[:, perm]
        pairs.append((a, w))
    for _ in range(4):  # a few large spikes in small activations
        a = randn(m, k) * 1e-2
        cols = torch.randint(0, k, (m, 4), generator=g, device=device)
        a.scatter_(1, cols, randn(m, 4).sign() * 1e3)
        pairs.append((bf16(a), bf16(randn(n, k) / k**0.5)))
    return pairs

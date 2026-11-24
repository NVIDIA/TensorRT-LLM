# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bit battery (``_bit_battery``) on CPU: it catches a GEMM that sums K in another order, which the bitwise GEMM
tests rely on."""

import pytest

__extra_import_path__ = ["."]
from _bit_battery import bit_battery  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.cpu_only


def _bf16_sum(a, w, chunks):
    """``a @ w.T`` accumulated in FP32 over the K chunks in the given order, rounded to BF16 once."""
    import torch

    acc = torch.zeros(a.shape[0], w.shape[0])
    for c in chunks:
        acc += a[:, c].float() @ w[:, c].float().t()
    return acc.bfloat16()


def test_bit_battery_exposes_another_summation_order() -> None:
    """A GEMM that sums K's chunks in reverse order: the battery's structured inputs (exact cancellation) catch it on
    every pair, where a random input at one row may round to the same BF16 values."""
    import torch

    m, n, k = 1, 64, 1536
    battery = bit_battery(m, n, k, device="cpu", seed=3)
    assert len(battery) >= 32
    for a, w in battery:
        assert a.shape == (m, k) and w.shape == (n, k)
        assert a.dtype == w.dtype == torch.bfloat16
    forward = [slice(i, i + 64) for i in range(0, k, 64)]
    differs = [
        not torch.equal(_bf16_sum(a, w, forward), _bf16_sum(a, w, forward[::-1]))
        for a, w in battery
    ]
    assert all(differs[-8:-4]), differs[-8:-4]
    # The same order on the same pair gives the same bits (the check compares nothing else).
    a, w = battery[0]
    assert torch.equal(_bf16_sum(a, w, forward), _bf16_sum(a, w, forward))

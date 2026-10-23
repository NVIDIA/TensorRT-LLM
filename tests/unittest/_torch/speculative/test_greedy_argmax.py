# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`greedy_argmax` must equal `torch.argmax(logits, dim=-1)` element for element."""

import pytest
import torch

from tensorrt_llm._torch.speculative.greedy_argmax import MIN_NUMEL, eligible, greedy_argmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 248320


def _check(x: torch.Tensor, out_dtype: torch.dtype = torch.int64) -> None:
    assert eligible(x)
    got = greedy_argmax(x, out_dtype)
    ref = torch.argmax(x, dim=-1).to(out_dtype)
    assert got.dtype == out_dtype
    assert torch.equal(got, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [17, 128, 256, 512, 1024])
def test_random(rows, dtype):
    torch.manual_seed(rows)
    _check(torch.randn(rows, VOCAB, device="cuda").to(dtype))


@pytest.mark.parametrize("cols", [VOCAB + 1, 4097, 151937, 1 << 20])
def test_odd_vocab(cols):
    rows = max(1, MIN_NUMEL // cols + 1)
    torch.manual_seed(cols)
    x = torch.randn(rows, cols, device="cuda")
    _check(x)
    _check(x, torch.int32)


@pytest.mark.parametrize("rows", [64, 256, 1024])
def test_ties_and_special_values(rows):
    torch.manual_seed(rows)
    # bf16-valued fp32 logits, as the widened head output: ties are frequent.
    x = torch.randn(rows, VOCAB, device="cuda").bfloat16().float()
    r = torch.arange(rows, device="cuda")
    # A repeated row maximum at several positions, across chunk boundaries.
    for c in (7, 4095, 4096, 31040, 31041, 124159, VOCAB - 1):
        x[r % 4 == 0, c] = 100.0
    # +0.0 / -0.0 ties as the maximum.
    x[r % 4 == 1] = -1.0
    x[r % 4 == 1, 9000] = -0.0
    x[r % 4 == 1, 100] = 0.0
    x[r % 4 == 1, 200000] = -0.0
    # All -inf; -inf with one finite value late in the row.
    x[r % 8 == 2] = float("-inf")
    x[r % 8 == 6] = float("-inf")
    x[r % 8 == 6, VOCAB - 3] = -1e30
    # NaN: the first NaN wins over +inf and larger indices.
    x[r % 4 == 3, 50] = float("inf")
    x[r % 4 == 3, 70000] = float("nan")
    x[r % 8 == 3, 3] = float("nan")
    x[r % 4 == 3, VOCAB - 1] = float("nan")
    _check(x)
    _check(x.bfloat16())
    # One row entirely NaN.
    x[0] = float("nan")
    _check(x)


def test_small_input_falls_back():
    x = torch.randn(2, 1000, device="cuda")
    assert not eligible(x)
    assert torch.equal(greedy_argmax(x), torch.argmax(x, dim=-1))


def test_strided_rows():
    torch.manual_seed(1)
    base = torch.randn(256, VOCAB + 64, device="cuda")
    _check(base[:, :VOCAB])


def test_cuda_graph():
    torch.manual_seed(2)
    x = torch.randn(256, VOCAB, device="cuda")
    out = greedy_argmax(x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = greedy_argmax(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.argmax(x, dim=-1))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PAD_FILL on a GPU: ``pad_fill.pad_rows_fill`` leaves every row as the stock padding fills leave it, bit for bit
(the one-hot logits fill, the zero fill and the uniform probs fill; eager and inside a CUDA graph), and declines inputs
outside its contract."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    pad_fill,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

VOCAB = 200064
# A one-hot padding logit: finite in every logits dtype and large enough that softmax of the row is one-hot.
ONEHOT = 1.0e4


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _mask(rows, kind, gen):
    if kind == "none":
        return torch.zeros(rows, dtype=torch.bool, device="cuda")
    if kind == "all":
        return torch.ones(rows, dtype=torch.bool, device="cuda")
    return torch.rand(rows, generator=gen, device="cuda") < 0.25


def _stock_logits(x, mask, col0):
    y = x.masked_fill(mask.unsqueeze(-1), 0.0)
    if col0 is not None:
        y[:, 0].masked_fill_(mask, col0)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [8, 96, 128, 384])
@pytest.mark.parametrize("kind", ["none", "random", "all"])
def test_the_fills_are_bitwise_the_stock_fills(dtype, rows, kind):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3.0).to(dtype)
    mask = _mask(rows, kind, gen)
    for col0 in (ONEHOT, None):
        y = x.clone()
        assert pad_fill.pad_rows_fill(y, mask, 0.0, col0) is y
        assert torch.equal(_bits(y), _bits(_stock_logits(x, mask, col0)))
    y = x.clone()
    pad_fill.pad_rows_fill(y, mask, 1.0 / VOCAB)
    assert torch.equal(_bits(y), _bits(x.clone().masked_fill_(mask.unsqueeze(-1), 1.0 / VOCAB)))


def test_the_fill_replays_inside_a_cuda_graph():
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows = 96
    static_x = torch.empty(rows, VOCAB, device="cuda")
    static_mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
    pad_fill.pad_rows_fill(static_x, static_mask, 0.0, ONEHOT)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pad_fill.pad_rows_fill(static_x, static_mask, 0.0, ONEHOT)
    for _ in range(3):
        x = torch.randn(rows, VOCAB, generator=gen, device="cuda")
        mask = _mask(rows, "random", gen)
        static_x.copy_(x)
        static_mask.copy_(mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(static_x), _bits(_stock_logits(x, mask, ONEHOT)))


def test_inputs_outside_the_contract_are_declined():
    x = torch.randn(8, 64, device="cuda")
    mask = torch.zeros(8, dtype=torch.bool, device="cuda")
    assert pad_fill.pad_rows_fill(x[:, ::2], mask, 0.0) is None  # not contiguous
    assert pad_fill.pad_rows_fill(x.to(torch.int32), mask, 0.0) is None  # dtype
    assert pad_fill.pad_rows_fill(x, mask.to(torch.uint8), 0.0) is None  # mask dtype
    assert pad_fill.pad_rows_fill(x, mask[:4], 0.0) is None  # mask length
    assert pad_fill.pad_rows_fill(x.cpu(), mask.cpu(), 0.0) is None  # not on CUDA
    assert pad_fill.pad_rows_fill(x.view(2, 4, 64), mask, 0.0) is None  # not 2-D

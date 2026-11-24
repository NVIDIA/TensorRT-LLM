# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed latent gathers against an E2M1/E4M3 torch decode, including graph replay."""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.nvfp4_mla_gather import (  # noqa: E501
    nvfp4_mla_context_kv_cache_gather,
    nvfp4_mla_kv_cache_gather,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    skip_pre_blackwell,
]


def _inputs(cell):
    torch.manual_seed(52)
    page, rows = cell.spec["page"], cell.spec["rows"]
    tokens = 8 * 2 * page
    data = torch.randint(0, 256, (tokens, 288), dtype=torch.uint8, device="cuda")
    scales = (torch.randint(1, 5, (tokens, 36), device="cuda").float() / 4).to(torch.float8_e4m3fn)
    pointers = torch.tensor([[[data.data_ptr(), scales.data_ptr()], [0, 0]]], dtype=torch.int64)
    mapping = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    dequant = torch.tensor([0.125], device="cuda")
    topk = torch.randint(0, page + 5, (rows, 64), dtype=torch.int32, device="cuda")
    topk[:, -1] = -1
    return page, data, scales, pointers, mapping, dequant, topk


@pytest.mark.parametrize("cell", nvfp4_mla_kv_cache_gather.CELLS, ids=lambda c: c.why)
def test_generation(cell):
    page, data, scales, pointers, mapping, dequant, local = _inputs(cell)
    # Layer 1 of a two-layer pool; reverse the first two logical pages.
    indices = ((1 - local // page) * 2 + 1) * page + local % page
    indices[local < 0] = -1
    indices[:, -2] = data.shape[0] + 7
    output = torch.empty((*indices.shape, 576), dtype=torch.float8_e4m3fn, device="cuda")
    compact = torch.empty_like(indices)
    args = (pointers, mapping, indices, output, compact, dequant, 1, 0, data.shape[0])
    nvfp4_mla_kv_cache_gather(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvfp4_mla_kv_cache_gather(*args)
    graph.replay()
    expected = nvfp4_mla_kv_cache_gather.reference(data, scales, indices, dequant)
    valid = (indices >= 0) & (indices < data.shape[0])
    torch.testing.assert_close(output.float()[valid], expected.float()[valid], rtol=0, atol=0)
    offsets = torch.arange(indices.numel(), device="cuda").view_as(indices)
    torch.testing.assert_close(compact[valid].long(), offsets[valid])
    assert torch.all(compact[~valid] == -1)


@pytest.mark.parametrize("cell", nvfp4_mla_context_kv_cache_gather.CELLS, ids=lambda c: c.why)
def test_context(cell):
    page, data, scales, pointers, mapping, dequant, local = _inputs(cell)
    rows = local.shape[0]
    requests = torch.arange(rows, dtype=torch.int32, device="cuda") % 2
    blocks = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([0, page + 5, 2 * (page + 5)], dtype=torch.int64, device="cuda")
    output = torch.empty(
        (max(2 * (page + 5), 64), 1, 576), dtype=torch.float8_e4m3fn, device="cuda"
    )
    compact = torch.empty_like(local)
    nvfp4_mla_context_kv_cache_gather(
        pointers,
        mapping,
        local,
        requests,
        blocks,
        lengths,
        output,
        compact,
        dequant,
        1,
        2 * (page + 5),
        page,
        2 * page,
        1,
        0,
        data.shape[0],
    )
    indices = (
        blocks[requests[:, None].long(), local.clamp_min(0).long() // page] * 2 + 1
    ) * page + local.clamp_min(0) % page
    indices[local < 0] = -1
    expected = nvfp4_mla_context_kv_cache_gather.reference(data, scales, indices, dequant)
    valid = local >= 0
    actual = output.float().view(-1, 576)[compact[valid].long()]
    torch.testing.assert_close(actual, expected.float()[valid], rtol=0, atol=0)
    assert torch.all(compact[~valid] == -1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.block_sparse_attention import (
    block_sparse_attention,
    is_cute_supported,
)


def _inputs(cell):
    spec = cell.spec
    batch, heads, dim = 1, 2, 128
    counts, sizes = spec["counts"], spec["sizes"]
    rng = torch.Generator(device="cuda").manual_seed(81)

    def tensor(tiles):
        return torch.randn(
            batch, heads, len(tiles) * 64, dim, dtype=spec["dtype"], device="cuda", generator=rng
        )

    q, k, v = tensor(counts), tensor(sizes), tensor(sizes)
    query = torch.arange(len(counts), device="cuda")[:, None]
    head = torch.arange(heads, device="cuda")[:, None, None]
    route = torch.arange(max(counts), device="cuda")
    indices = ((head + query + route) % len(sizes)).to(torch.int32).unsqueeze(0)
    num = (
        torch.tensor(counts, device="cuda", dtype=torch.int32).expand(batch, heads, -1).contiguous()
    )
    lengths = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    return q, k, v, indices, num, lengths, spec["scale"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CuTe sparse attention needs CUDA")
@pytest.mark.parametrize("cell", block_sparse_attention.CELLS, ids=lambda cell: cell.why)
def test_block_sparse_attention(cell):
    inputs = _inputs(cell)
    if not is_cute_supported(inputs[0]):
        pytest.skip("requires the Blackwell CuTe sparse attention kernel")
    block_sparse_attention.is_valid(*inputs)
    out, lse = block_sparse_attention(*inputs)
    expected_out, expected_lse = block_sparse_attention.reference(*inputs)
    torch.testing.assert_close(out, expected_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses CUDA input buffers")
def test_paired_query_tiles_require_equal_counts():
    inputs = _inputs(block_sparse_attention.CELLS[0])
    counts = inputs[4]
    counts[0, 0, 1] = 1
    with pytest.raises(AssertionError, match="paired query tiles"):
        block_sparse_attention.is_valid(*inputs)

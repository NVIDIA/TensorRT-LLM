# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SOL predictor kernels (block pooling, block statistics, exact-block selection)."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.kernels import (
    _block_pool_torch,
    _block_statistics_torch,
    _select_exact_blocks_torch,
    block_pool,
    block_statistics,
    select_exact_blocks,
)

_CPU_ONLY = pytest.mark.cpu_only
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
BLOCK = 64


def _pooled_shape(x: torch.Tensor, block_size: int) -> tuple[int, int, int, int]:
    batch, seq_len, heads, head_dim = x.shape
    return batch, (seq_len + block_size - 1) // block_size, heads, head_dim


def _unpack(bits: torch.Tensor, num_kv_blocks: int) -> torch.Tensor:
    words = bits.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    return (
        ((words.unsqueeze(-1) >> shifts) & 1)
        .bool()
        .reshape(*bits.shape[:-1], -1)[..., :num_kv_blocks]
    )


@_CPU_ONLY
def test_block_pool_torch_fallback_means_valid_tokens_only() -> None:
    x = torch.zeros((1, 70, 1, 4), dtype=torch.bfloat16)
    x[0, :64] = 2.0
    x[0, 64:70] = 3.0
    out = torch.empty(_pooled_shape(x, BLOCK), dtype=torch.float32)
    block_pool(x, out, block_size=BLOCK, reduce="mean")
    assert torch.equal(out[0, 0], torch.full((1, 4), 2.0))
    assert torch.equal(out[0, 1], torch.full((1, 4), 3.0))
    total = torch.empty_like(out)
    block_pool(x, total, block_size=BLOCK, reduce="sum")
    assert torch.equal(total[0, 1], torch.full((1, 4), 18.0))


@_CPU_ONLY
def test_block_pool_rejects_mismatched_output() -> None:
    x = torch.zeros((1, 70, 1, 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out"):
        block_pool(x, torch.empty((1, 3, 1, 4)), block_size=BLOCK, reduce="mean")
    with pytest.raises(ValueError, match="reduce"):
        block_pool(x, torch.empty(_pooled_shape(x, BLOCK)), block_size=BLOCK, reduce="max")


@_CPU_ONLY
def test_select_exact_blocks_torch_fallback_packs_bit_r_of_word_w() -> None:
    blocks = 35
    centroid = torch.zeros((1, blocks, 1, 8), dtype=torch.float32)
    k_summary = torch.zeros((1, blocks, 1, 8), dtype=torch.bfloat16)
    k_summary[0, 33, 0, 0] = 1.0
    centroid[0, 0, 0, 0] = 1.0
    k_mean = torch.zeros((1, 1, 8))
    k_var = torch.zeros((1, 1, 8))
    bits = torch.empty((1, 1, blocks, 2), dtype=torch.uint32)
    select_exact_blocks(centroid, k_summary, k_mean, k_var, bits, tau=0.5, sm_scale=1.0)
    exact = _unpack(bits, blocks)
    # Row 0 scores 1.0 against block 33 only (threshold 0.5 * sqrt(1e-6)) plus its local band.
    expected = torch.zeros(blocks, dtype=torch.bool)
    expected[[0, 1, 33]] = True
    assert torch.equal(exact[0, 0, 0], expected)
    # Bit 1 of word 1 is block 33.
    assert int(bits[0, 0, 0, 1]) == 2
    # Rows without scores keep only the local band.
    assert torch.equal(exact[0, 0, 17].nonzero().flatten(), torch.tensor([16, 17, 18]))


@_REQUIRES_CUDA
@pytest.mark.parametrize("seq_len", [64, 257, 4097])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_block_pool_matches_torch_fallback(seq_len: int, out_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn((2, seq_len, 3, 128), device="cuda", dtype=torch.bfloat16)
    for reduce in ("mean", "sum"):
        out = torch.empty(_pooled_shape(x, BLOCK), dtype=out_dtype, device="cuda")
        block_pool(x, out, block_size=BLOCK, reduce=reduce)
        expected = torch.empty_like(out)
        _block_pool_torch(x, expected, block_size=BLOCK, reduce=reduce)
        tolerance = (
            {"rtol": 1e-5, "atol": 1e-5}
            if out_dtype == torch.float32
            else {"rtol": 1e-2, "atol": 1e-2}
        )
        torch.testing.assert_close(out, expected, **tolerance)


@_REQUIRES_CUDA
def test_block_pool_accepts_strided_batch_and_token_dims() -> None:
    torch.manual_seed(1)
    full = torch.randn((2, 130, 2, 3, 128), device="cuda", dtype=torch.bfloat16)
    x = full[:, :, 1]  # heads/head_dim contiguous, token stride wider than a row
    out = torch.empty(_pooled_shape(x, BLOCK), dtype=torch.float32, device="cuda")
    block_pool(x, out, block_size=BLOCK, reduce="mean")
    expected = torch.empty_like(out)
    _block_pool_torch(x.contiguous(), expected, block_size=BLOCK, reduce="mean")
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@_REQUIRES_CUDA
def test_block_statistics_matches_torch_fallback() -> None:
    torch.manual_seed(2)
    k_summary = torch.randn((2, 1182, 3, 128), device="cuda", dtype=torch.bfloat16)
    mean = torch.empty((2, 3, 128), device="cuda", dtype=torch.float32)
    var = torch.empty_like(mean)
    block_statistics(k_summary, mean, var)
    expected_mean = torch.empty_like(mean)
    expected_var = torch.empty_like(var)
    _block_statistics_torch(k_summary, expected_mean, expected_var)
    torch.testing.assert_close(mean, expected_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(var, expected_var, rtol=1e-4, atol=1e-6)
    assert bool((var >= 0).all())


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_blocks", [5, 258])
def test_select_exact_blocks_matches_fallback_and_clears_padding_bits(num_blocks: int) -> None:
    torch.manual_seed(3)
    batch, heads, dim = 2, 3, 128
    centroid = torch.randn((batch, num_blocks, heads, dim), device="cuda") * 0.125
    k_summary = (torch.randn((batch, num_blocks, heads, dim), device="cuda") * 0.125).to(
        torch.bfloat16
    )
    k_mean = torch.empty((batch, heads, dim), device="cuda")
    k_var = torch.empty_like(k_mean)
    block_statistics(k_summary, k_mean, k_var)
    words = (num_blocks + 31) // 32
    bits = torch.empty((batch, heads, num_blocks, words), device="cuda", dtype=torch.uint32)
    expected = torch.empty_like(bits)
    for tau in (0.75, -1.0e6, 1.0e6):
        select_exact_blocks(centroid, k_summary, k_mean, k_var, bits, tau=tau, sm_scale=0.125)
        _select_exact_blocks_torch(
            centroid, k_summary, k_mean, k_var, expected, tau=tau, sm_scale=0.125
        )
        assert torch.equal(bits, expected), f"tau={tau}"
    padding = words * 32 - num_blocks
    if padding:
        assert int(bits[..., -1].to(torch.int64).max()) < (1 << (32 - padding))
    exact = _unpack(bits, num_blocks)
    ids = torch.arange(num_blocks, device="cuda")
    assert bool(exact[..., (ids[:, None] - ids[None, :]).abs() <= 1].all())


@_REQUIRES_CUDA
def test_kernels_replay_inside_cuda_graph() -> None:
    torch.manual_seed(4)
    q = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    shape = _pooled_shape(q, BLOCK)
    centroid = torch.empty(shape, device="cuda", dtype=torch.float32)
    k_summary = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    mean = torch.empty((1, 2, 128), device="cuda")
    var = torch.empty_like(mean)
    bits = torch.empty((1, 2, shape[1], 1), device="cuda", dtype=torch.uint32)

    def run() -> None:
        block_pool(q, centroid, block_size=BLOCK, reduce="mean")
        block_pool(k, k_summary, block_size=BLOCK, reduce="mean")
        block_statistics(k_summary, mean, var)
        select_exact_blocks(centroid, k_summary, mean, var, bits, tau=0.5, sm_scale=0.125)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    graph.replay()
    torch.cuda.synchronize()

    expected_centroid = torch.empty_like(centroid)
    expected_summary = torch.empty_like(k_summary)
    _block_pool_torch(q, expected_centroid, block_size=BLOCK, reduce="mean")
    _block_pool_torch(k, expected_summary, block_size=BLOCK, reduce="mean")
    expected_mean = torch.empty_like(mean)
    expected_var = torch.empty_like(var)
    _block_statistics_torch(expected_summary, expected_mean, expected_var)
    expected_bits = torch.empty_like(bits)
    _select_exact_blocks_torch(
        expected_centroid,
        expected_summary,
        expected_mean,
        expected_var,
        expected_bits,
        tau=0.5,
        sm_scale=0.125,
    )
    torch.testing.assert_close(centroid, expected_centroid, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_summary, expected_summary, rtol=1e-2, atol=1e-2)
    assert torch.equal(bits, expected_bits)

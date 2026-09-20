# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SOL predictor kernels (block pooling, block thresholds, exact-block selection)."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.kernels import (
    _LOG2_E,
    _block_pool_torch,
    _block_thresholds_torch,
    _select_exact_blocks_torch,
    block_pool,
    block_thresholds,
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
    threshold = torch.full((1, 1, blocks), 5.0e-4, dtype=torch.float32)
    bits = torch.empty((1, 1, blocks, 2), dtype=torch.uint32)
    select_exact_blocks(centroid, k_summary, threshold, bits, sm_scale=1.0)
    exact = _unpack(bits, blocks)
    # Row 0 scores log2(e) against block 33 only, above the 5e-4 threshold, plus its local band.
    expected = torch.zeros(blocks, dtype=torch.bool)
    expected[[0, 1, 33]] = True
    assert torch.equal(exact[0, 0, 0], expected)
    # Bit 1 of word 1 is block 33.
    assert int(bits[0, 0, 0, 1]) == 2
    # Rows without scores keep only the local band.
    assert torch.equal(exact[0, 0, 17].nonzero().flatten(), torch.tensor([16, 17, 18]))


def _threshold_oracle(
    centroid: torch.Tensor,
    k_summary: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
    thresh_type: str,
) -> torch.Tensor:
    c = centroid.double().permute(0, 2, 1, 3)
    keys = k_summary.double().permute(0, 2, 1, 3)
    k_mean = keys.mean(dim=2)
    mean = torch.einsum("bhqd,bhd->bhq", c, k_mean)
    if thresh_type == "diag":
        var = torch.einsum("bhqd,bhd->bhq", c.square(), keys.var(dim=2, unbiased=False))
    else:
        centered = keys - k_mean.unsqueeze(2)
        covariance = torch.matmul(centered.transpose(-1, -2), centered) / keys.shape[2]
        var = torch.einsum("bhqd,bhqd->bhq", torch.matmul(c, covariance), c)
    log2_scale = sm_scale * _LOG2_E
    return mean * log2_scale + tau * torch.sqrt(var * log2_scale * log2_scale + 1.0e-6)


@_CPU_ONLY
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
def test_block_thresholds_match_fp64_oracle(thresh_type: str) -> None:
    torch.manual_seed(7)
    centroid = torch.randn((2, 9, 3, 16), dtype=torch.float32)
    k_summary = torch.randn((2, 9, 3, 16)).to(torch.bfloat16)
    threshold = block_thresholds(
        centroid, k_summary, tau=0.75, sm_scale=0.25, thresh_type=thresh_type
    )
    assert threshold.shape == (2, 3, 9) and threshold.dtype == torch.float32
    assert threshold.is_contiguous()
    expected = _threshold_oracle(
        centroid, k_summary, tau=0.75, sm_scale=0.25, thresh_type=thresh_type
    )
    torch.testing.assert_close(threshold.double(), expected, rtol=1e-4, atol=1e-5)


@_CPU_ONLY
def test_block_thresholds_exact_exceeds_diag_for_correlated_keys() -> None:
    """A rank-one key distribution has a full-covariance variance the diagonal policy underestimates."""
    torch.manual_seed(8)
    centroid = torch.ones((1, 4, 1, 16), dtype=torch.float32)
    k_summary = torch.randn((1, 12, 1, 1)).expand(-1, -1, -1, 16).contiguous().to(torch.bfloat16)
    diag = block_thresholds(centroid, k_summary, tau=1.0, sm_scale=1.0, thresh_type="diag")
    exact = block_thresholds(centroid, k_summary, tau=1.0, sm_scale=1.0, thresh_type="exact")
    assert torch.all(exact > diag)
    with pytest.raises(ValueError, match="thresh_type"):
        block_thresholds(centroid, k_summary, tau=1.0, sm_scale=1.0, thresh_type="full")


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
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
@pytest.mark.parametrize("num_blocks", [5, 258])
def test_block_thresholds_kernels_match_torch_fallback(thresh_type: str, num_blocks: int) -> None:
    torch.manual_seed(5)
    batch, heads, dim = 2, 3, 128
    centroid = torch.randn((batch, 7, heads, dim), device="cuda") * 0.125
    k_summary = (torch.randn((batch, num_blocks, heads, dim), device="cuda") * 0.125).to(
        torch.bfloat16
    )
    for tau in (0.75, -0.5):
        threshold = block_thresholds(
            centroid, k_summary, tau=tau, sm_scale=0.125, thresh_type=thresh_type
        )
        expected = _block_thresholds_torch(
            centroid, k_summary, tau=tau, sm_scale=0.125, thresh_type=thresh_type
        )
        assert threshold.shape == (batch, heads, 7) and threshold.is_contiguous()
        torch.testing.assert_close(threshold, expected, rtol=1e-4, atol=1e-5)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_blocks", [5, 258])
def test_select_exact_blocks_matches_fallback_and_clears_padding_bits(num_blocks: int) -> None:
    torch.manual_seed(3)
    batch, heads, dim = 2, 3, 128
    centroid = torch.randn((batch, num_blocks, heads, dim), device="cuda") * 0.125
    k_summary = (torch.randn((batch, num_blocks, heads, dim), device="cuda") * 0.125).to(
        torch.bfloat16
    )
    words = (num_blocks + 31) // 32
    bits = torch.empty((batch, heads, num_blocks, words), device="cuda", dtype=torch.uint32)
    expected = torch.empty_like(bits)
    for tau, thresh_type in ((0.75, "diag"), (0.75, "exact"), (-1.0e6, "diag"), (1.0e6, "exact")):
        threshold = block_thresholds(
            centroid, k_summary, tau=tau, sm_scale=0.125, thresh_type=thresh_type
        )
        select_exact_blocks(centroid, k_summary, threshold, bits, sm_scale=0.125)
        _select_exact_blocks_torch(centroid, k_summary, threshold, expected, sm_scale=0.125)
        assert torch.equal(bits, expected), f"tau={tau} thresh_type={thresh_type}"
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
    bits = torch.empty((1, 2, shape[1], 1), device="cuda", dtype=torch.uint32)

    def run() -> None:
        block_pool(q, centroid, block_size=BLOCK, reduce="mean")
        block_pool(k, k_summary, block_size=BLOCK, reduce="mean")
        threshold = block_thresholds(
            centroid, k_summary, tau=0.5, sm_scale=0.125, thresh_type="exact"
        )
        select_exact_blocks(centroid, k_summary, threshold, bits, sm_scale=0.125)

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
    expected_threshold = block_thresholds(
        expected_centroid, expected_summary, tau=0.5, sm_scale=0.125, thresh_type="exact"
    )
    expected_bits = torch.empty_like(bits)
    _select_exact_blocks_torch(
        expected_centroid, expected_summary, expected_threshold, expected_bits, sm_scale=0.125
    )
    torch.testing.assert_close(centroid, expected_centroid, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_summary, expected_summary, rtol=1e-2, atol=1e-2)
    assert torch.equal(bits, expected_bits)

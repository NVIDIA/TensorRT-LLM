# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity of the Sage quantizers used by PrimTS block-sparse attention with PyTorch references.

Scale slots are located with the PrimTS flat-layout helpers that the attention kernel uses to
read them.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.prims_ts.sage import (
    sage_flat_scale_numel,
    sage_flat_scale_slot,
)
from tensorrt_llm._torch.attention.kernels.sage_quant import SageQuantized, sage_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


# --------------------------------------------------------------------------- summaries
def _summary_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, smooth_k: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(43)
    k = torch.randn(shape, device="cuda", generator=generator)
    v = torch.randn(shape, device="cuda", generator=generator)
    magnitudes = torch.logspace(-5, 2, shape[1], device="cuda").view(1, -1, 1, 1)
    k, v = (k * magnitudes).to(dtype), (v * magnitudes).to(dtype)
    # Raw V has larger extrema than its block means, and the attention epilogue
    # dequantizes exact and mean V with the same per-sequence channel scales.
    v_scale = v.float().abs().amax(dim=1).clamp_min(1e-3) * 2 / 448.0
    k_mean = k.float().mean(dim=1) if smooth_k else None
    return k, v, v_scale, k_mean


def _quantize_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    k_mean: torch.Tensor | None,
    quant_dtype: torch.dtype,
    v_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``[batch, blocks, heads, head_dim]`` block means as the attention core does:
    every mean is a key of its sequence with a scale of its own, reusing the given statistics."""
    batch, blocks = k.shape[:2]
    cu_seqlens = torch.arange(0, (batch + 1) * blocks, blocks, dtype=torch.int32, device=k.device)
    stats = SageQuantized(None, None, None, None, None, v_scale, k_mean, v_mean)
    actual = sage_quantize(
        None,
        k.flatten(0, 1),
        v.flatten(0, 1),
        None,
        cu_seqlens,
        k_block_size=1,
        qk_quant_dtype=quant_dtype,
        stats=stats,
    )
    return actual.k.view(k.shape), actual.v.view(v.shape), actual.k_scale


def _assert_quantized_close(
    actual: torch.Tensor, normalized: torch.Tensor, quant_dtype: torch.dtype
) -> None:
    """Quantized values round the saturated FP32 reference within half a grid step."""
    limit = 127.0 if quant_dtype == torch.int8 else 448.0
    expected = normalized.clamp(-limit, limit)
    if quant_dtype == torch.int8:
        tolerance = torch.full_like(expected, 0.51)
    else:
        # E4M3 keeps 3 mantissa bits: half a step is 2^-4 of the value.
        tolerance = expected.abs() * 0.0626 + 2**-9
    assert torch.all((actual.float() - expected).abs() <= tolerance)


def _assert_summaries_match_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    k_mean: torch.Tensor | None,
    quant_dtype: torch.dtype,
    v_mean: torch.Tensor | None = None,
) -> None:
    batch, blocks, heads, _ = k.shape
    divisor = 126.9 if quant_dtype == torch.int8 else 448.0
    k_ref = k.float() if k_mean is None else k.float() - k_mean[:, None]
    maximum = k_ref.abs().amax(dim=-1).clamp_min(1e-3)
    scales = maximum / torch.full_like(maximum, divisor)
    slots = sage_flat_scale_slot(
        torch.arange(batch, device=k.device)[:, None],
        torch.arange(blocks, device=k.device)[None, :],
        blocks,
        0,
    ).flatten()

    actual_k, actual_v, actual_scales = actual
    assert actual_k.shape == k.shape and actual_v.shape == v.shape
    assert actual_scales.shape == (heads, sage_flat_scale_numel(batch, blocks, 1))
    torch.testing.assert_close(
        actual_scales[:, slots], scales.permute(2, 0, 1).reshape(heads, -1), rtol=1e-6, atol=0
    )
    _assert_quantized_close(actual_k, k_ref / scales.unsqueeze(-1), quant_dtype)
    v_ref = v.float() if v_mean is None else v.float() - v_mean[:, None]
    _assert_quantized_close(actual_v, v_ref / v_scale[:, None], torch.float8_e4m3fn)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("shape", [(2, 9, 3, 64), (1, 129, 2, 128)])
@pytest.mark.parametrize("smooth", [False, True])
def test_quantize_sage_summaries_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype, quant_dtype: torch.dtype, smooth: bool
) -> None:
    k, v, v_scale, k_mean = _summary_inputs(shape, dtype, smooth_k=smooth)
    v_mean = v.float().mean(dim=1) if smooth else None
    actual = _quantize_summaries(k, v, v_scale, k_mean, quant_dtype, v_mean)
    assert [result.dtype for result in actual] == [quant_dtype, torch.float8_e4m3fn, torch.float32]
    _assert_summaries_match_reference(actual, k, v, v_scale, k_mean, quant_dtype, v_mean)


@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantize_sage_summaries_zero_blocks_use_scale_floor(quant_dtype: torch.dtype) -> None:
    k = torch.zeros(2, 9, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v_scale = torch.full((2, 2, 64), 1e-3 / 448.0, device="cuda")
    actual = _quantize_summaries(k, v, v_scale, None, quant_dtype)
    _assert_summaries_match_reference(actual, k, v, v_scale, None, quant_dtype)


def test_quantize_sage_summaries_rejects_mismatched_statistics() -> None:
    k, v, v_scale, _ = _summary_inputs((2, 9, 3, 64), torch.bfloat16)
    cases = (
        ((k, v[:1], v_scale, None), "key and value must have the same shape"),
        ((k, v, v_scale[:1], None), "v_scale must be"),
        ((k, v, v_scale, v_scale[:, :1]), "k_mean must be"),
    )
    for args, message in cases:
        with pytest.raises(RuntimeError, match=message):
            _quantize_summaries(*args, torch.int8)
    stats = SageQuantized(None, None, None, None, None, v_scale)
    cu_seqlens = _cu_seqlens([9, 9])
    keys = k.flatten(0, 1)
    with pytest.raises(ValueError, match="only K and V are quantized"):
        sage_quantize(
            keys,
            keys,
            v.flatten(0, 1),
            cu_seqlens,
            cu_seqlens,
            k_block_size=1,
            qk_quant_dtype=torch.int8,
            stats=stats,
        )


@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantize_sage_summaries_under_compile_and_cuda_graph(quant_dtype: torch.dtype) -> None:
    k, v, v_scale, k_mean = _summary_inputs((2, 9, 2, 64), torch.bfloat16, smooth_k=True)

    def quantize(
        key: torch.Tensor, value: torch.Tensor, value_scale: torch.Tensor, key_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _quantize_summaries(key, value, value_scale, key_mean, quant_dtype)

    _assert_summaries_match_reference(
        torch.compile(quantize, fullgraph=True)(k, v, v_scale, k_mean),
        k,
        v,
        v_scale,
        k_mean,
        quant_dtype,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize(k, v, v_scale, k_mean)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantize(k, v, v_scale, k_mean)
    k.mul_(2)
    v.mul_(2)
    graph.replay()
    _assert_summaries_match_reference(captured, k, v, v_scale, k_mean, quant_dtype)


# --------------------------------------------------------------------------- Q/K/V
def _ragged(lengths: list[int], heads: int, head_dim: int, generator: torch.Generator):
    """Token-major BF16 tensor with per-sequence channel offsets, so that K smoothing matters."""
    tokens = sum(lengths)
    x = torch.randn(tokens, heads, head_dim, device="cuda", generator=generator)
    offsets = 4 * torch.randn(len(lengths), heads, head_dim, device="cuda", generator=generator)
    sequence = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda"), torch.tensor(lengths, device="cuda")
    )
    return (x + offsets[sequence]).to(torch.bfloat16)


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda").int()


def _check_block_quantized(
    quant: torch.Tensor,
    scale: torch.Tensor,
    source: torch.Tensor,
    lengths: list[int],
    block_size: int,
    quant_dtype: torch.dtype,
) -> None:
    """Scales equal the reference block maxima; values dequantize within one grid step."""
    divisor = 126.9 if quant_dtype == torch.int8 else 448.0
    batch = len(lengths)
    expected_numel = sage_flat_scale_numel(1, source.shape[0], block_size) + batch - 1
    assert scale.shape == (source.shape[1], expected_numel)
    begin = 0
    for b, length in enumerate(lengths):
        seq = source[begin : begin + length].float()
        blocks = -(-length // block_size)
        padded = torch.zeros(blocks * block_size, *seq.shape[1:], device="cuda")
        padded[:length] = seq
        maximum = padded.view(blocks, block_size, *seq.shape[1:]).abs().amax(dim=(1, 3))
        maximum = maximum.clamp_min(1e-3)
        expected = maximum / torch.full_like(maximum, divisor)
        # Sequence b's scales start at cu_seqlens[b] // block_size + b.
        slots = begin // block_size + sage_flat_scale_slot(
            b, torch.arange(blocks, device="cuda"), 0, 0
        )
        torch.testing.assert_close(scale[:, slots], expected.T, rtol=1e-6, atol=0)

        token_scale = expected.repeat_interleave(block_size, dim=0)[:length].unsqueeze(-1)
        _assert_quantized_close(quant[begin : begin + length], seq / token_scale, quant_dtype)
        begin += length


@pytest.mark.parametrize("block_size", [1, 4, 16, 32])
@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
# Every smoothing combination is its own kernel; each also takes one of the head dimensions.
@pytest.mark.parametrize(
    ("smooth_k", "smooth_v", "head_dim"),
    [(False, False, 64), (True, False, 128), (False, True, 64), (True, True, 128)],
)
def test_sage_quantize_matches_reference(
    block_size: int, quant_dtype: torch.dtype, smooth_k: bool, smooth_v: bool, head_dim: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Ragged sequences with partial final blocks; GQA with 2 query heads per KV head.
    lengths = [301, 77, 520]
    q = _ragged(lengths, 4, head_dim, generator)
    k = _ragged(lengths, 2, head_dim, generator)
    v = _ragged(lengths, 2, head_dim, generator)
    cu_seqlens = _cu_seqlens(lengths)
    token_lengths = torch.tensor(lengths, device="cuda")

    actual = sage_quantize(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        q_block_size=block_size,
        k_block_size=block_size,
        qk_quant_dtype=quant_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
    )

    if smooth_k:
        k_mean = torch.stack([seq.float().mean(dim=0) for seq in torch.split(k, lengths)])
        torch.testing.assert_close(actual.k_mean, k_mean, rtol=1e-5, atol=1e-5)
        k_ref = k.float() - torch.repeat_interleave(actual.k_mean, token_lengths, dim=0)
    else:
        assert actual.k_mean is None
        k_ref = k.float()
    _check_block_quantized(actual.q, actual.q_scale, q, lengths, block_size, quant_dtype)
    _check_block_quantized(actual.k, actual.k_scale, k_ref, lengths, block_size, quant_dtype)

    # V: one FP8 scale per channel of every sequence, from the maxima of V or of V minus its mean.
    if smooth_v:
        v_mean = torch.stack([seq.float().mean(dim=0) for seq in torch.split(v, lengths)])
        torch.testing.assert_close(actual.v_mean, v_mean, rtol=1e-5, atol=1e-5)
        v_ref = v.float() - torch.repeat_interleave(actual.v_mean, token_lengths, dim=0)
        v_max = torch.stack([seq.abs().amax(dim=0) for seq in torch.split(v_ref, lengths)])
        v_scale = v_max.clamp_min(1e-3) / 448.0
        torch.testing.assert_close(actual.v_scale, v_scale, rtol=1e-6, atol=0)
    else:
        assert actual.v_mean is None
        v_ref = v.float()
        v_max = torch.stack([seq.abs().amax(dim=0) for seq in torch.split(v, lengths)])
        floor = torch.tensor(1e-3, dtype=torch.bfloat16, device="cuda")
        v_scale = torch.maximum(v_max, floor).float() * torch.tensor(1 / 448.0, device="cuda")
        torch.testing.assert_close(actual.v_scale, v_scale, rtol=0, atol=0)
    token_v_scale = torch.repeat_interleave(v_scale, token_lengths, dim=0)
    error = (actual.v.float() * token_v_scale - v_ref).abs()
    assert torch.all(error <= v_ref.abs() * 0.07 + token_v_scale * 2**-6)


def test_sage_quantize_mixed_block_sizes_and_query_lengths() -> None:
    generator = torch.Generator(device="cuda").manual_seed(11)
    q_lengths, kv_lengths = [64, 200], [256, 90]
    q = _ragged(q_lengths, 2, 128, generator)
    k = _ragged(kv_lengths, 2, 128, generator)
    v = _ragged(kv_lengths, 2, 128, generator)
    actual = sage_quantize(
        q,
        k,
        v,
        _cu_seqlens(q_lengths),
        _cu_seqlens(kv_lengths),
        q_block_size=1,
        k_block_size=32,
        qk_quant_dtype=torch.float8_e4m3fn,
    )
    _check_block_quantized(actual.q, actual.q_scale, q, q_lengths, 1, torch.float8_e4m3fn)
    _check_block_quantized(actual.k, actual.k_scale, k, kv_lengths, 32, torch.float8_e4m3fn)
    assert actual.v_scale.shape == (2, 2, 128)


def test_sage_quantize_v_extrema_order_negative_zero() -> None:
    # Every channel is -0.0 except for a -5 in the first token: across the warps that merge the
    # V extrema, -0.0 must win the maximum and lose the minimum to -5.
    x = torch.zeros(4096, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(x, -0.0)
    v[0] = -5.0
    cu_seqlens = _cu_seqlens([4096])
    actual = sage_quantize(
        x,
        x,
        v,
        cu_seqlens,
        cu_seqlens,
        q_block_size=1,
        k_block_size=16,
        qk_quant_dtype=torch.float8_e4m3fn,
        smooth_v=True,
    )
    v_ref = v.float() - v.float().mean(dim=0)
    expected = v_ref.abs().amax(dim=0).clamp_min(1e-3) / 448.0
    torch.testing.assert_close(actual.v_scale[0], expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
def test_sage_quantize_smoothing_is_bitwise_reproducible(quant_dtype: torch.dtype) -> None:
    # Long sequences spread every mean over the most warps; summed with float atomics, the means
    # and with them the quantized K and V would change from call to call.
    generator = torch.Generator(device="cuda").manual_seed(5)
    lengths = [9000, 4097, 7001]
    q = _ragged(lengths, 4, 128, generator)
    k = _ragged(lengths, 4, 128, generator)
    v = _ragged(lengths, 4, 128, generator)
    cu_seqlens = _cu_seqlens(lengths)

    def quantize():
        return sage_quantize(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            q_block_size=1,
            k_block_size=16,
            qk_quant_dtype=quant_dtype,
            smooth_k=True,
            smooth_v=True,
        )

    # Q does not use the means; the K scales are checked through K, as their buffer also holds
    # slots that are never written.
    first = quantize()
    for _ in range(4):
        again = quantize()
        for name in ("k", "v", "v_scale", "k_mean", "v_mean"):
            expected, actual = getattr(first, name), getattr(again, name)
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), name


def test_sage_quantize_rejects_unsupported_block_sizes() -> None:
    x = torch.zeros(16, 2, 128, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = _cu_seqlens([16])
    with pytest.raises(RuntimeError, match="must be 1, 4, 16, or 32"):
        sage_quantize(
            x,
            x,
            x,
            cu_seqlens,
            cu_seqlens,
            q_block_size=3,
            k_block_size=16,
            qk_quant_dtype=torch.int8,
        )

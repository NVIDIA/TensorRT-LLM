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
"""Exact parity of the Sage K/V summary quantizer with a PyTorch reference.

Scale slots are located with the PrimTS flat-layout helpers that the attention
kernel uses to read them.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.prims_ts.sage import flat_scale_numel, flat_scale_slot
from tensorrt_llm._torch.attention.kernels.sage_quant import quantize_sage_summaries

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _inputs(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(43)
    k = torch.randn(shape, device="cuda", generator=generator)
    v = torch.randn(shape, device="cuda", generator=generator)
    magnitudes = torch.logspace(-5, 2, shape[1], device="cuda").view(1, -1, 1, 1)
    k, v = (k * magnitudes).to(dtype), (v * magnitudes).to(dtype)
    # Raw V has larger extrema than its block means, and the attention epilogue
    # dequantizes exact and mean V with the same channel scales.
    v_scale = v.float().abs().amax(dim=(0, 1)).clamp_min(1e-3) * 2 / 448.0
    return k, v, v_scale


def _assert_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    quant_dtype: torch.dtype,
) -> None:
    batch, blocks, heads, _ = k.shape
    divisor = 126.9 if quant_dtype == torch.int8 else 448.0
    maximum = k.float().abs().amax(dim=-1).clamp_min(1e-3)
    # Tensor division keeps FP32 round-to-nearest; dividing by a Python scalar
    # multiplies by its rounded reciprocal and can move an FP8 rounding tie.
    scales = maximum / torch.full_like(maximum, divisor)
    normalized = k.float() / scales.unsqueeze(-1)
    if quant_dtype == torch.int8:
        quant_k = normalized.round().clamp(-127, 127).to(quant_dtype)
    else:
        quant_k = normalized.clamp(-448, 448).to(quant_dtype)
    quant_v = (v.float() / v_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    slots = flat_scale_slot(
        torch.arange(batch, device=k.device)[:, None],
        torch.arange(blocks, device=k.device)[None, :],
        blocks,
        0,
    ).flatten()

    actual_k, actual_v, actual_scales = actual
    assert actual_scales.shape == (heads, flat_scale_numel(batch, blocks, 1))
    torch.testing.assert_close(actual_k.float(), quant_k.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_v.float(), quant_v.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales[:, slots], scales.permute(2, 0, 1).reshape(heads, -1), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("shape", [(2, 9, 3, 64), (3, 5, 2, 7), (1, 129, 2, 128)])
# A 100x smaller V scale saturates the quantized V means at the E4M3 maximum.
@pytest.mark.parametrize("v_scale_divisor", [1.0, 100.0])
def test_quantize_sage_summaries_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype, quant_dtype: torch.dtype, v_scale_divisor: float
) -> None:
    k, v, v_scale = _inputs(shape, dtype)
    v_scale = v_scale / v_scale_divisor
    actual = quantize_sage_summaries(k, v, v_scale, quant_dtype)
    assert [result.dtype for result in actual] == [quant_dtype, torch.float8_e4m3fn, torch.float32]
    _assert_matches_reference(actual, k, v, v_scale, quant_dtype)


@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantize_sage_summaries_zero_blocks_use_scale_floor(quant_dtype: torch.dtype) -> None:
    k = torch.zeros(2, 9, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v_scale = torch.full((2, 64), 1e-3 / 448.0, device="cuda")
    actual = quantize_sage_summaries(k, v, v_scale, quant_dtype)
    _assert_matches_reference(actual, k, v, v_scale, quant_dtype)


def test_quantize_sage_summaries_rejects_layouts_the_kernel_cannot_index() -> None:
    k, v, v_scale = _inputs((2, 9, 3, 64), torch.bfloat16)
    cases = (
        ((k, v[:, :1], v_scale), "matching shape, dtype, and device"),
        ((k, v, v_scale[:1]), "FP32 with shape"),
        ((k.transpose(0, 1), v.transpose(0, 1), v_scale), "contiguous"),
    )
    for args, message in cases:
        with pytest.raises(ValueError, match=message):
            quantize_sage_summaries(*args, torch.int8)


@pytest.mark.parametrize("quant_dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantize_sage_summaries_under_compile_and_cuda_graph(quant_dtype: torch.dtype) -> None:
    k, v, v_scale = _inputs((2, 9, 2, 64), torch.bfloat16)

    def quantize(
        key: torch.Tensor, value: torch.Tensor, value_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quantize_sage_summaries(key, value, value_scale, quant_dtype)

    _assert_matches_reference(
        torch.compile(quantize, fullgraph=True)(k, v, v_scale), k, v, v_scale, quant_dtype
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize(k, v, v_scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantize(k, v, v_scale)
    k.mul_(2)
    v.mul_(2)
    graph.replay()
    _assert_matches_reference(captured, k, v, v_scale, quant_dtype)

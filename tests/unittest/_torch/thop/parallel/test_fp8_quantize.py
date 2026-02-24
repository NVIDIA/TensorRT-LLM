# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

import pytest
import torch
from parameterized import parameterized
from utils.util import (getSMVersion, skip_pre_blackwell_unittest,
                        unittest_name_func)

from tensorrt_llm.quantization.utils.fp8_utils import \
    per_token_quant_and_transform


def _dequant_fp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1

    if m_tile == 1:
        assert n % 16 == 0, "n must be divisible by 16"
        total_blocks = math.ceil(n / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_col = block * 128
            end_col = min(start_col + 128, n)
            output[:, start_col:
                   end_col] = input[:, start_col:end_col] * scale[:,
                                                                  block].view(
                                                                      -1, 1)

    elif n_tile == 1:
        assert m % 16 == 0, "m must be divisible by 16"
        total_blocks = math.ceil(m / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_row = block * 128
            end_row = min(start_row + 128, m)
            output[start_row:end_row, :] = input[start_row:end_row, :] * scale[
                block, :]
    else:
        assert n % 16 == 0, "n must be divisible by 16"
        assert m % 16 == 0, "m must be divisible by 16"
        n_blocks = math.ceil(n / 128)
        m_blocks = math.ceil(m / 128)
        for i in range(n_blocks):
            for j in range(m_blocks):
                start_row = j * 128
                end_row = min(start_row + 128, m)
                start_col = i * 128
                end_col = min(start_col + 128, n)
                output[start_row:end_row,
                       start_col:end_col] = input[start_row:end_row,
                                                  start_col:end_col] * scale[j,
                                                                             i]
    return output


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 90,
    reason="Only test on Blackwell and Hopper",
)
@pytest.mark.parametrize("k", [576, 256, 32])
@pytest.mark.parametrize(
    "m",
    [4, 16, 256],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_quantize_blackwell(dtype, m, k):
    torch.random.manual_seed(0)
    # TODO: make sure there is no padding for now
    assert m % 4 == 0, "Disable padding for now"
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    fp8_a, fp8_a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
    fp8_a_scale = fp8_a_scale.view(-1,
                                   fp8_a.shape[0])  # transpose the scale view
    a_dequant = _dequant_fp8(fp8_a, fp8_a_scale, True, False, True)

    torch.testing.assert_close(a_dequant.cpu().to(torch.float32),
                               a.cpu().to(torch.float32),
                               atol=1e-1,
                               rtol=1e-1)


def mxfp8_quantize_check_accuracy(a, b, atol, rtol, percent):
    if torch.any(torch.isnan(a)):
        raise Exception("NaN in a")
    if torch.any(torch.isnan(b)):
        raise Exception("NaN in b")
    assert a.shape == b.shape
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if mismatch_percent > 1 - percent:
        raise Exception("Mismatch percentage is %f for rtol %f" %
                        (mismatch_percent, rtol))


@parameterized.expand(list([[1, 1024, torch.half, True],
                            [2, 512, torch.bfloat16, True],
                            [16, 512, torch.half, True],
                            [16, 512, torch.half, False],
                            [1024, 512, torch.half, False],
                            [1024, 512, torch.half, False]]),
                      name_func=unittest_name_func)
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_torch_host(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).cpu().contiguous()

    a_fp8, a_sf = torch.ops.tensorrt_llm.quantize_mxe4m3_host(
        a, is_sf_swizzled_layout)

    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.view(torch.uint8), a_sf.view(torch.uint8), is_sf_swizzled_layout)

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt, a, 8, 0, 0.999)


@parameterized.expand(list([[1, 1024, torch.half, True],
                            [2, 512, torch.bfloat16, True],
                            [16, 512, torch.half, True],
                            [16, 512, torch.half, False],
                            [1024, 512, torch.half, False],
                            [1024, 512, torch.half, False]]),
                      name_func=unittest_name_func)
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_torch_device(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) *
         16).to(dtype).cuda().contiguous()

    # Quantize it on device.
    a_fp8, a_sf = torch.ops.trtllm.mxfp8_quantize(a, is_sf_swizzled_layout, 32)

    # Dequantize it on host.
    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8), is_sf_swizzled_layout)

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt.cpu().to(torch.float32),
                                  a.cpu().to(torch.float32), 8, 0, 0.999)


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [1568])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("alignment", [64, 128])
@skip_pre_blackwell_unittest
def test_mxfp8_quantize_alignment_torch_device(m, k, dtype,
                                               is_sf_swizzled_layout,
                                               alignment):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) *
         16).to(dtype).cuda().contiguous()
    padded_k = ((k + alignment - 1) // alignment) * alignment

    # Quantize it on device.
    a_fp8, a_sf = torch.ops.trtllm.mxfp8_quantize(a, is_sf_swizzled_layout,
                                                  alignment)
    assert a_fp8.shape[1] == padded_k

    # Dequantize it on host.
    a_pt = torch.ops.tensorrt_llm.dequantize_mxe4m3_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8), is_sf_swizzled_layout)

    # Check if the bits of paddings are zero.
    paddings = a_fp8.view(torch.int8)[:, k:]
    assert torch.all(paddings == 0), "Paddings should be zero"

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt[:, :k].cpu().to(torch.float32),
                                  a.cpu().to(torch.float32), 8, 0, 0.999)


@pytest.mark.skipif(
    getSMVersion() != 103 and getSMVersion() != 90,
    reason="Only test on Blackwell and Hopper",
)
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("m", [4, 16, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_quantize_ue8m0_vs_triton(dtype, m, k):
    """Test fp8_quantize_1x128 with use_ue8m0=True against Triton reference."""
    torch.random.manual_seed(42)

    # Ensure alignment requirements
    assert m % 4 == 0, "m must be divisible by 4"
    assert k % 128 == 0, "k must be divisible by 128"

    # Create input tensor
    input_tensor = torch.randn((m, k), device='cuda', dtype=dtype)

    # CUDA version with UE8M0
    from tensorrt_llm.quantization.utils import fp8_utils
    cuda_fp8, cuda_scale_float = fp8_utils.fp8_quantize_1x128_sf_transpose(
        input_tensor, use_ue8m0=True)

    # Triton reference with UE8M0
    triton_fp8, triton_scale_int32 = per_token_quant_and_transform(
        input_tensor.clone(), quant_group_size=128, scale_ue8m0=True)

    # Convert CUDA float32 scale to int32 packed format for comparison
    cuda_scale_int32 = fp8_utils.transform_sf_into_required_layout(
        sf=cuda_scale_float,
        mn=m,
        k=k,
        recipe=(1, 1, 128),
        num_groups=None,
        is_sfa=False)

    # Compare quantized FP8 values (should match within FP8 precision)
    torch.testing.assert_close(
        cuda_fp8.to(torch.float32),
        triton_fp8.to(torch.float32),
        atol=1.0,
        rtol=0.01,
        msg=f"FP8 quantized values mismatch for shape ({m}, {k})")

    # Compare packed int32 scales by decoding to float
    def decode_ue8m0_int32_to_float(int32_tensor):
        """Decode int32 packed UE8M0 scales to float32. Formula: value = 2^(exponent - 127)
        """
        # Extract 4 bytes from each int32
        b0 = (int32_tensor >> 0) & 0xFF
        b1 = (int32_tensor >> 8) & 0xFF
        b2 = (int32_tensor >> 16) & 0xFF
        b3 = (int32_tensor >> 24) & 0xFF

        # Decode: value = 2^(exponent - 127)
        s0 = torch.pow(2.0, b0.float() - 127.0)
        s1 = torch.pow(2.0, b1.float() - 127.0)
        s2 = torch.pow(2.0, b2.float() - 127.0)
        s3 = torch.pow(2.0, b3.float() - 127.0)

        return torch.stack([s0, s1, s2, s3], dim=-1)

    cuda_scales_decoded = decode_ue8m0_int32_to_float(cuda_scale_int32)
    triton_scales_decoded = decode_ue8m0_int32_to_float(triton_scale_int32)

    # Compare: values should match, or both be tiny (< 1e-10 means zero-block)
    torch.testing.assert_close(
        cuda_scales_decoded,
        triton_scales_decoded,
        atol=1e-10,
        rtol=0.01,
        msg=f"UE8M0 decoded scales mismatch for shape ({m}, {k})")

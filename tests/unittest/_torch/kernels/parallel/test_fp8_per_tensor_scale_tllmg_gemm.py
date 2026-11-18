# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from _torch.helpers import calc_diff
from utils.util import getSMVersion, isSM100Family

from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a)


@pytest.mark.skipif(
    not isSM100Family(),
    reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(2048, 5120), (1024, 5120)],
)
@pytest.mark.parametrize(
    "m",
    [1, 4, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "inference_mode",
    ["low-latency", "throughput"],
)
def test_fp8_block_scale_gemm(dtype, m, k, n, inference_mode):
    if inference_mode == "low-latency" and dtype == torch.bfloat16:
        pytest.skip("https://nvbugs/5328141")

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=torch.float)
    b = torch.randn((n, k), device='cuda', dtype=torch.float)

    # Get max abs value of a, b, and c
    a_abs_max = a.abs().max()
    b_abs_max = b.abs().max()

    a_scale = 448.0 / a_abs_max
    b_scale = 448.0 / b_abs_max
    global_scale = 1.0 / (a_scale * b_scale)

    # Quantize a, b
    act_a_fp8 = (a * a_scale).to(torch.float8_e4m3fn)
    act_b_fp8 = (b * b_scale).to(torch.float8_e4m3fn)

    act_a_quant_float = act_a_fp8.to(torch.float)
    act_b_quant_float = act_b_fp8.to(torch.float)

    # Run reference implementation
    output_expected = act_a_quant_float @ act_b_quant_float.t()

    # Scale output
    output_expected /= (a_scale * b_scale)

    c_abs_max = output_expected.abs().max()
    c_scale = 448.0 / c_abs_max

    if inference_mode == "low-latency":
        # FIXME: this should not be hardcoded and set from the kernel runner.
        epilogue_tile_m = 128
        act_b_fp8 = shuffle_matrix_a(act_b_fp8.view(torch.uint8),
                                     epilogue_tile_m).view(torch.float8_e4m3fn)

    if dtype == torch.float8_e4m3fn:
        global_scale *= c_scale
        output_expected = output_expected * c_scale

    # Convert to output dtype
    output_expected = output_expected.to(dtype)

    # Run ops
    output = torch.ops.trtllm.fp8_per_tensor_scaling_tllmg_gemm(
        act_a_fp8.contiguous(),
        act_b_fp8.contiguous(),
        global_scale,
        out_dtype=dtype,
        low_latency_kernel=(inference_mode == "low-latency"))

    # Some compare ops are not implemented for float8_e4m3fn
    output = output.to(torch.float)
    output_expected = output_expected.to(torch.float)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    getSMVersion() < 100 or getSMVersion() >= 110,
    reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(5120, 512), (5120, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [1, 4, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "inference_mode",
    ["low-latency", "throughput"],
)
def test_fp8_block_scale_gemm_gated_silu(dtype, m, k, n, inference_mode):
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=torch.float)
    w1 = torch.randn((n, k), device='cuda', dtype=torch.float)
    w2 = torch.randn((n, k), device='cuda', dtype=torch.float)

    # Get max abs value of a, b, and c
    a_abs_max = a.abs().max()
    b_abs_max = max(w1.abs().max(), w2.abs().max())

    a_scale = 448.0 / a_abs_max
    b_scale = 448.0 / b_abs_max

    # Quantize a, b
    act_a_fp8 = (a * a_scale).to(torch.float8_e4m3fn)
    act_w1_fp8 = (w1 * b_scale).to(torch.float8_e4m3fn)
    act_w2_fp8 = (w2 * b_scale).to(torch.float8_e4m3fn)
    act_a_quant_float = act_a_fp8.to(torch.float)
    act_w1_quant_float = act_w1_fp8.to(torch.float)
    act_w2_quant_float = act_w2_fp8.to(torch.float)

    # Run reference implementation
    out1 = act_a_quant_float @ act_w1_quant_float.t()
    out2 = act_a_quant_float @ act_w2_quant_float.t()
    # Scale output
    out1 = out1 / (a_scale * b_scale)
    out2 = out2 / (a_scale * b_scale)
    # Run silu
    out2 = torch.nn.functional.silu(out2)
    # Scale output
    output_expected = out1 * out2

    c_abs_max = output_expected.abs().max()
    c_scale = 448.0 / c_abs_max

    act_b_fp8 = torch.stack([act_w1_fp8,
                             act_w2_fp8]).reshape(2 * n,
                                                  k).view(torch.float8_e4m3fn)
    act_b_fp8 = reorder_rows_for_gated_act_gemm(act_b_fp8.view(
        torch.uint8)).view(torch.float8_e4m3fn)
    if inference_mode == "low-latency":
        # FIXME: this should not be hardcoded and set from the kernel runner.
        epilogue_tile_m = 128
        act_b_fp8 = shuffle_matrix_a(act_b_fp8.view(torch.uint8),
                                     epilogue_tile_m).view(torch.float8_e4m3fn)

    global_scale = 1.0 / (a_scale * b_scale)
    global_scale_gate = 1.0 / (a_scale * b_scale)
    if dtype == torch.float8_e4m3fn:
        global_scale *= c_scale
        output_expected = output_expected * c_scale

    # Convert to output dtype
    output_expected = output_expected.to(dtype)

    # Run ops
    output = torch.ops.trtllm.fp8_per_tensor_scaling_tllmg_gemm(
        act_a_fp8.contiguous(),
        act_b_fp8.contiguous(),
        global_scale,
        global_scale_gate=global_scale_gate,
        out_dtype=dtype,
        low_latency_kernel=(inference_mode == "low-latency"),
        gated_silu=True)

    # Some compare ops are not implemented for float8_e4m3fn
    output = output.to(torch.float)
    output_expected = output_expected.to(torch.float)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-2, rtol=1e-2)

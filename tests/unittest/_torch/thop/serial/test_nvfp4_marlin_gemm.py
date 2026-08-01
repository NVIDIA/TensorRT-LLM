# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Standalone test for the PyTorch ``torch.ops.trtllm.marlin_nvfp4_gemm`` custom op.

Relocated out of the deleted ``tests/unittest/trt/functional/test_fp4_gemm.py``: this
case validates a live PyTorch-backend custom op against a ``torch.matmul`` reference and
never built a TensorRT engine, so it is kept (not removed) with the TensorRT backend.
"""

import unittest
from itertools import product

import torch
from parameterized import parameterized
from utils.util import skip_non_hopper_unittest, unittest_name_func

import tensorrt_llm  # noqa: F401  # registers torch.ops.tensorrt_llm / torch.ops.trtllm ops


def float_tensor_to_e2m1_and_ufp8_scale(
    float_tensor: torch.Tensor, sf_vec_size, ufp8_type: int = 1, is_sf_swizzled_layout: bool = True
):
    value_e2m1, scale_ufp8, rep_float = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
        float_tensor, sf_vec_size, ufp8_type, is_sf_swizzled_layout
    )
    return value_e2m1, scale_ufp8, rep_float


def e2m1_and_ufp8_scale_to_float_tensor(
    e2m1_tensor: torch.Tensor, ufp8_scale_tensor: torch.Tensor, sf_vec_size, ufp8_type: int = 1
):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float(
        e2m1_tensor, ufp8_scale_tensor, sf_vec_size, ufp8_type
    )
    return float_tensor


def random_fp4_tensor_and_sf(shape, sf_vec_size):
    assert shape[-1] % sf_vec_size == 0
    float_tensor = torch.randn(shape, dtype=torch.float32)
    e2m1_tensor, e8m0_sf_tensor, repr_float_tensor = float_tensor_to_e2m1_and_ufp8_scale(
        float_tensor, sf_vec_size
    )
    represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor(
        e2m1_tensor, e8m0_sf_tensor, sf_vec_size
    )
    assert torch.equal(repr_float_tensor, represented_float_tensor)
    return e2m1_tensor, e8m0_sf_tensor, represented_float_tensor


class TestNvfp4MarlinGemm(unittest.TestCase):
    @parameterized.expand(
        list(product([1024, 2048], [1024, 2048], [1, 8, 128], [16], [1.0, 2.0], ["nvfp4", "bf16"])),
        name_func=unittest_name_func,
    )
    @skip_non_hopper_unittest
    def test_nvfp4_marlin_gemm(
        self, input_dim, output_dim, batch_size, sf_vec_size, alpha, act_dtype
    ):
        from tensorrt_llm.quantization.utils import marlin_utils

        torch.random.manual_seed(0)

        is_bf16_act = act_dtype == "bf16"

        if is_bf16_act:
            # BF16 activations — no FP4 quantization needed
            input_bf16 = torch.randn((batch_size, input_dim), dtype=torch.bfloat16).cuda()
            input_fp32 = input_bf16.float().cpu()
        else:
            # FP4 activations with swizzled scales (for act dequant kernel)
            input_e2m1, input_e4m3_scale, input_fp32 = random_fp4_tensor_and_sf(
                (batch_size, input_dim), sf_vec_size
            )

        # FP4 weights with UN-swizzled scales (for Marlin processing)
        weights_e2m1, weights_e4m3_scale_raw, weights_fp32 = float_tensor_to_e2m1_and_ufp8_scale(
            torch.randn((output_dim, input_dim), dtype=torch.float32),
            sf_vec_size,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        )

        weights_fp32_transposed = torch.transpose(weights_fp32, 0, 1)
        alpha_tensor = torch.FloatTensor([alpha]).cuda()

        ref_output_fp32 = torch.matmul(input_fp32, weights_fp32_transposed)
        if not is_bf16_act:
            ref_output_fp32 *= alpha

        # Step 1: Repack weights to Marlin tiled format
        qweight_int32 = weights_e2m1.cuda().view(torch.int32).T.contiguous()
        perm = torch.empty(0, dtype=torch.int32, device="cuda")
        marlin_weight = torch.ops.trtllm.gptq_marlin_repack(
            b_q_weight=qweight_int32,
            perm=perm,
            size_k=input_dim,
            size_n=output_dim,
            num_bits=4,
            is_a_8bit=False,
        )

        # Step 2: Process weight scales for Marlin kernel
        num_groups = input_dim // sf_vec_size
        scale_fp8 = weights_e4m3_scale_raw.cuda().view(torch.float8_e4m3fn)
        scale_2d = scale_fp8.reshape(output_dim, num_groups).T.contiguous()
        marlin_scale = marlin_utils.marlin_permute_scales(
            scale_2d.to(torch.half), input_dim, output_dim, group_size=sf_vec_size
        )
        marlin_scale = marlin_utils.nvfp4_marlin_process_scales(marlin_scale)

        # Step 3: Process global scale (includes exponent bias correction)
        weight_global_scale = marlin_utils.nvfp4_marlin_process_global_scale(
            torch.tensor(1.0, dtype=torch.bfloat16, device="cuda")
        )

        if is_bf16_act:
            # BF16 path: pass BF16 activations directly, dummy scale_a/alpha
            mat_a = input_bf16
            scale_a = torch.zeros(1, dtype=torch.uint8, device="cuda")
            alpha_arg = torch.ones(1, dtype=torch.float32, device="cuda")
        else:
            # FP4 path: unswizzle activation scales for the dequant kernel
            mat_a = input_e2m1.cuda()
            m_padded = (batch_size + 128 - 1) // 128 * 128
            act_sf_gpu = input_e4m3_scale.cuda()
            scale_a = torch.ops.trtllm.block_scale_interleave_reverse(
                act_sf_gpu.view(m_padded, -1)
            ).flatten()
            alpha_arg = alpha_tensor

        output = torch.ops.trtllm.marlin_nvfp4_gemm(
            mat_a,
            marlin_weight,
            scale_a=scale_a,
            scale_b=marlin_scale,
            alpha=alpha_arg,
            weight_global_scale=weight_global_scale,
            bias=None,
            out_dtype=torch.bfloat16,
            size_n=output_dim,
            size_k=input_dim,
        )

        output_cpu_float = output.float().cpu()
        assert torch.allclose(output_cpu_float, ref_output_fp32, atol=0.75, rtol=0.02)


if __name__ == "__main__":
    unittest.main()

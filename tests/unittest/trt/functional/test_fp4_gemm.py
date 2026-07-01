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
import copy
import unittest
from itertools import product

import tensorrt as trt
import torch
from parameterized import parameterized
from utils.util import (skip_non_hopper_unittest, skip_pre_blackwell_unittest,
                        unittest_name_func)

import tensorrt_llm
from tensorrt_llm import Tensor


# ufp8_type: 0 for ue8m0, 1 for ue4m3
def float_tensor_to_e2m1_and_ufp8_scale(float_tensor: torch.Tensor,
                                        sf_vec_size,
                                        ufp8_type: int = 1,
                                        is_sf_swizzled_layout: bool = True):
    value_e2m1, scale_ufp8, rep_float = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
        float_tensor, sf_vec_size, ufp8_type, is_sf_swizzled_layout)
    return value_e2m1, scale_ufp8, rep_float


# ufp8_type: 0 for ue8m0, 1 for ue4m3
def half_tensor_to_e2m1_and_ufp8_scale(half_tensor: torch.Tensor,
                                       sf_scale_tensor: torch.Tensor,
                                       sf_vec_size,
                                       ufp8_type: int = 1):
    value_e2m1, scale_ufp8 = torch.ops.tensorrt_llm.half_to_e2m1_and_ufp8sf_scale(
        half_tensor, sf_scale_tensor, sf_vec_size, ufp8_type)
    return value_e2m1, scale_ufp8


def e2m1_and_ufp8_scale_to_float_tensor(e2m1_tensor: torch.Tensor,
                                        ufp8_scale_tensor: torch.Tensor,
                                        sf_vec_size,
                                        ufp8_type: int = 1):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float(
        e2m1_tensor, ufp8_scale_tensor, sf_vec_size, ufp8_type)
    return float_tensor


# Used by the (fp16 -> int4) quant layer + int4 gemm network.
def e2m1_and_ufp8_scale_to_float_tensor_v2(e2m1_tensor: torch.Tensor,
                                           ufp8_scale_tensor: torch.Tensor,
                                           global_scale_tensor: torch.Tensor,
                                           sf_vec_size,
                                           ufp8_type: int = 1,
                                           is_sf_swizzled_layout: bool = True):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size,
        ufp8_type, is_sf_swizzled_layout)
    return float_tensor


def random_fp4_tensor_and_sf(shape, sf_vec_size):
    assert shape[-1] % sf_vec_size == 0
    float_tensor = torch.randn(shape, dtype=torch.float32)
    e2m1_tensor, e8m0_sf_tensor, repr_float_tensor = float_tensor_to_e2m1_and_ufp8_scale(
        float_tensor, sf_vec_size)
    represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor(
        e2m1_tensor, e8m0_sf_tensor, sf_vec_size)
    assert torch.equal(repr_float_tensor, represented_float_tensor)
    return e2m1_tensor, e8m0_sf_tensor, represented_float_tensor


def random_fp4_tensor_and_sf_v2(shape, sf_vec_size):
    assert shape[-1] % sf_vec_size == 0
    float_tensor = torch.randn(shape, dtype=torch.float32)
    half_tensor = float_tensor.to(torch.float16).cuda()

    # global scale trick for int4 quantization.
    alpha = 448.0 / (torch.max(float_tensor) / 6.0)
    sf_scale_tensor = torch.FloatTensor([alpha]).cuda()
    gemm_alpha_tensor = torch.FloatTensor([1.0 / alpha])

    # device tensor
    e2m1_tensor, e8m0_sf_tensor = half_tensor_to_e2m1_and_ufp8_scale(
        half_tensor, sf_scale_tensor, sf_vec_size)
    # host tensor
    represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor_v2(
        e2m1_tensor.cpu(), e8m0_sf_tensor.cpu(), gemm_alpha_tensor, sf_vec_size)
    return e2m1_tensor.cpu(), e8m0_sf_tensor.cpu(
    ), represented_float_tensor, alpha


def ones_fp4_tensor_and_sf(shape, sf_vec_size):
    assert shape[-1] % sf_vec_size == 0
    float_tensor = torch.ones(shape, dtype=torch.float32)
    e2m1_tensor, e8m0_sf_tensor, repr_float_tensor = float_tensor_to_e2m1_and_ufp8_scale(
        float_tensor, sf_vec_size)
    represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor(
        e2m1_tensor, e8m0_sf_tensor, sf_vec_size)
    assert torch.equal(repr_float_tensor, represented_float_tensor)
    return e2m1_tensor, e8m0_sf_tensor, represented_float_tensor


def get_fp4_shapes(base_shape, sf_vec_size):
    int8_container_shape = copy.deepcopy(base_shape)
    int8_container_shape[-1] = int8_container_shape[-1] // 2
    fp8_sf_shape = copy.deepcopy(base_shape)
    fp8_sf_shape[-1] = fp8_sf_shape[-1] // sf_vec_size
    return int8_container_shape, fp8_sf_shape


trtllm_packed_input_type_str = 'int64'
trtllm_scale_type_str = 'int32'


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    @parameterized.expand(
        list(
            product(
                [1024, 2048],
                [1024, 2048],
                [1, 8, 128, 1023],
                #product([256], [256], [1],
                ['float16'],
                [16],
                [1.0, 2.0])),
        name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_gemm(self, input_dim, output_dim, batch_size, output_dtype,
                      sf_vec_size, alpha):
        torch.random.manual_seed(0)

        input_e2m1, input_e8m0_scale, input_fp32 = random_fp4_tensor_and_sf(
            (batch_size, input_dim), sf_vec_size)
        weights_e2m1, weights_e8m0_scale, weights_fp32 = random_fp4_tensor_and_sf(
            (output_dim, input_dim), sf_vec_size)

        weights_fp32_transposed = torch.transpose(weights_fp32, 0, 1)
        alpha_tensor = torch.FloatTensor([alpha])
        alpha_tensor_gpu = alpha_tensor.cuda()

        ref_output_fp32 = torch.matmul(input_fp32, weights_fp32_transposed)
        ref_output_fp32 *= alpha
        #print(f"input_fp32={input_fp32}")
        #print(f"weights_fp32={weights_fp32}")
        #print(f"ref_output_fp32={ref_output_fp32}")
        ref_e2m1, ref_e8m0_scale, ref_rep_fp32 = float_tensor_to_e2m1_and_ufp8_scale(
            ref_output_fp32, sf_vec_size)
        #print(f"ref_rep_fp32={ref_rep_fp32}")

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.gemm_plugin = 'nvfp4'

        input_e2m1_trt = input_e2m1.cuda()
        input_e8m0_scale_trt = input_e8m0_scale.cuda().view(torch.float8_e4m3fn)
        weights_e2m1_trt = weights_e2m1.cuda()
        weights_e8m0_scale_trt = weights_e8m0_scale.cuda().view(
            torch.float8_e4m3fn)

        #print(f"input_e2m1_trt={input_e2m1_trt}")
        print(f"input_e8m0_scale_trt={input_e8m0_scale_trt}")
        #print(f'input_e8m0_scale_trt.shape={input_e8m0_scale_trt.shape}')
        #print(f"weights_e2m1_trt={weights_e2m1_trt}")
        #print(f"weights_e8m0_scale_trt={weights_e8m0_scale_trt}")

        output_trt = None
        output_sf_trt = None

        if output_dtype == 'int4':
            output_trt = torch.zeros_like(
                ref_e2m1,
                device='cuda',
                dtype=tensorrt_llm.str_dtype_to_torch(output_dtype))
            output_sf_trt = torch.zeros_like(ref_e8m0_scale, device='cuda')
        else:
            output_trt = torch.zeros_like(
                ref_rep_fp32,
                dtype=tensorrt_llm.str_dtype_to_torch(output_dtype),
                device='cuda')

        with tensorrt_llm.net_guard(net):
            input_tensor = Tensor(name='input',
                                  shape=(batch_size, input_dim),
                                  dtype=trt.fp4)
            input_sf_tensor = Tensor(name='input_sf',
                                     shape=input_e8m0_scale_trt.shape,
                                     dtype=trt.fp8)
            weights_tensor = Tensor(name='weights',
                                    shape=(output_dim, input_dim),
                                    dtype=trt.fp4)
            weights_sf_tensor = Tensor(name='weights_sf',
                                       shape=weights_e8m0_scale_trt.shape,
                                       dtype=trt.fp8)
            global_sf_tensor = Tensor(
                name='global_sf',
                shape=alpha_tensor.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            outputs = tensorrt_llm.quantization.functional.fp4_gemm(
                input_tensor,
                input_sf_tensor,
                weights_tensor,
                weights_sf_tensor,
                global_sf_tensor,
                output_dtype,
            )
            if output_dtype == 'int4':
                net._mark_output(
                    outputs[0],
                    'output',
                    dtype=tensorrt_llm.str_dtype_to_trt(output_dtype))
                net._mark_output(
                    outputs[1],
                    'output_sf',
                    dtype=tensorrt_llm.str_dtype_to_trt(trtllm_scale_type_str))
            else:
                net._mark_output(
                    outputs,
                    'output',
                    dtype=tensorrt_llm.str_dtype_to_trt(output_dtype))

            inputs = {
                'input': input_e2m1_trt,
                'input_sf': input_e8m0_scale_trt,
                'weights': weights_e2m1_trt,
                'weights_sf': weights_e8m0_scale_trt,
                'global_sf': alpha_tensor_gpu,
            }
            outputs = {'output': output_trt}
            if output_dtype == 'int4':
                outputs['output_sf'] = output_sf_trt

        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=output_dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        output_trt_cpu_float = output_trt.float().cpu()
        #print(f'output_trt={output_trt_cpu_float}')
        #print(f'ref_output_fp32={ref_output_fp32}')

        if output_dtype == 'int4':
            output_trt_fp32 = e2m1_and_ufp8_scale_to_float_tensor(
                output_trt.cpu(), output_sf_trt.cpu(), sf_vec_size)
            assert torch.allclose(output_trt_fp32, ref_output_fp32)
        else:
            assert torch.allclose(output_trt_cpu_float,
                                  ref_output_fp32,
                                  atol=1e-3,
                                  rtol=1e-3)

    @parameterized.expand(
        list(product([1024, 2048], [1024, 2048], [1, 8, 128, 1023], [16])) +
        list(product([8192, 10240, 28672], [28672, 10240, 8192], [1, 20],
                     [16])),
        name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_input_quant_and_fp4_gemm(self, input_dim, output_dim, batch_size,
                                      sf_vec_size):
        torch.random.manual_seed(0)
        torch.set_printoptions(threshold=64)
        output_dtype = 'float16'

        input_fp32 = torch.randn((batch_size, input_dim), dtype=torch.float32)
        ref_gemm_input_e2m1, ref_gemm_input_ufp8_scale, repr_float_tensor = float_tensor_to_e2m1_and_ufp8_scale(
            input_fp32, sf_vec_size)
        weights_e2m1, weight_ufp8_scale, weights_fp32, weights_alpha = random_fp4_tensor_and_sf_v2(
            (output_dim, input_dim), sf_vec_size)

        input_fp16 = input_fp32.to(torch.float16).cuda()

        # global scale trick for int4 quantization.
        alpha = 448.0 / (torch.max(input_fp32) / 6.0)

        weights_fp32_transposed = torch.transpose(weights_fp32, 0, 1)
        sf_scale_tensor = torch.FloatTensor([alpha]).cuda()
        act_unscale_tensor = torch.FloatTensor([1.0 / alpha]).cuda()
        gemm_alpha_tensor = torch.FloatTensor([1.0 / (alpha * weights_alpha)
                                               ]).cuda()

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.gemm_plugin = 'nvfp4'

        weights_e2m1_trt = weights_e2m1.cuda()
        weight_ufp8_scale_trt = weight_ufp8_scale.cuda().view(
            torch.float8_e4m3fn)

        output_trt = None
        gemm_quantized_input = None
        gemm_input_sf_tensor = None

        output_trt = torch.zeros(
            (batch_size, output_dim),
            dtype=tensorrt_llm.str_dtype_to_torch(output_dtype),
            device='cuda')
        gemm_quantized_input = torch.zeros_like(ref_gemm_input_e2m1,
                                                device='cuda')
        gemm_input_sf_tensor = torch.zeros_like(ref_gemm_input_ufp8_scale,
                                                device='cuda')

        with tensorrt_llm.net_guard(net):
            input_tensor = Tensor(
                name='input',
                shape=input_fp16.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("float16"))

            weights_tensor = Tensor(name='weights',
                                    shape=(output_dim, input_dim),
                                    dtype=trt.fp4)
            weights_sf_tensor = Tensor(name='weights_sf',
                                       shape=weight_ufp8_scale_trt.shape,
                                       dtype=trt.fp8)
            sf_scale_tensor_trt = Tensor(
                name='sf_scale',
                shape=gemm_alpha_tensor.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            gemm_alpha_tensor_trt = Tensor(
                name='gemm_alpha',
                shape=gemm_alpha_tensor.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))

            quantized_input, input_sf_tensor = tensorrt_llm.quantization.functional.quantize_to_fp4_tensor(
                input_tensor, sf_scale_tensor_trt)

            # mark the intermediate results.
            net._mark_output(quantized_input,
                             'quantized_gemm_input',
                             dtype=trt.fp4)
            net._mark_output(input_sf_tensor,
                             'gemm_input_sf_tensor',
                             dtype=trt.fp8)

            outputs = tensorrt_llm.quantization.functional.fp4_gemm(
                quantized_input,
                input_sf_tensor,
                weights_tensor,
                weights_sf_tensor,
                gemm_alpha_tensor_trt,
                output_dtype,
            )

            net._mark_output(outputs,
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(output_dtype))

            inputs = {
                'input': input_fp16,
                'weights': weights_e2m1_trt,
                'weights_sf': weight_ufp8_scale_trt,
                'sf_scale': sf_scale_tensor,
                'gemm_alpha': gemm_alpha_tensor,
            }
            outputs = {
                'quantized_gemm_input': gemm_quantized_input,
                'gemm_input_sf_tensor': gemm_input_sf_tensor,
                'output': output_trt
            }

        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=output_dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        output_trt_cpu_float = output_trt.float().cpu()

        # Simulate the quantization workflow for reference gemm.
        represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor_v2(
            gemm_quantized_input.cpu(), gemm_input_sf_tensor.cpu(),
            act_unscale_tensor.cpu(), sf_vec_size)
        ref_output_fp32 = torch.matmul(represented_float_tensor,
                                       weights_fp32_transposed)
        assert torch.allclose(output_trt_cpu_float,
                              ref_output_fp32,
                              atol=1e-3,
                              rtol=1e-3)

    @parameterized.expand(list(
        product([1024, 2048], [1024, 2048], [1, 8, 128], [16], [1.0, 2.0],
                ['nvfp4', 'bf16'])),
                          name_func=unittest_name_func)
    @skip_non_hopper_unittest
    def test_nvfp4_marlin_gemm(self, input_dim, output_dim, batch_size,
                               sf_vec_size, alpha, act_dtype):
        from tensorrt_llm.quantization.utils import marlin_utils

        torch.random.manual_seed(0)

        is_bf16_act = (act_dtype == 'bf16')

        if is_bf16_act:
            # BF16 activations — no FP4 quantization needed
            input_bf16 = torch.randn((batch_size, input_dim),
                                     dtype=torch.bfloat16).cuda()
            input_fp32 = input_bf16.float().cpu()
        else:
            # FP4 activations with swizzled scales (for act dequant kernel)
            input_e2m1, input_e4m3_scale, input_fp32 = random_fp4_tensor_and_sf(
                (batch_size, input_dim), sf_vec_size)

        # FP4 weights with UN-swizzled scales (for Marlin processing)
        weights_e2m1, weights_e4m3_scale_raw, weights_fp32 = \
            float_tensor_to_e2m1_and_ufp8_scale(
                torch.randn((output_dim, input_dim), dtype=torch.float32),
                sf_vec_size, ufp8_type=1, is_sf_swizzled_layout=False)

        weights_fp32_transposed = torch.transpose(weights_fp32, 0, 1)
        alpha_tensor = torch.FloatTensor([alpha]).cuda()

        ref_output_fp32 = torch.matmul(input_fp32, weights_fp32_transposed)
        if not is_bf16_act:
            ref_output_fp32 *= alpha

        # Step 1: Repack weights to Marlin tiled format
        qweight_int32 = weights_e2m1.cuda().view(torch.int32).T.contiguous()
        perm = torch.empty(0, dtype=torch.int32, device='cuda')
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
            scale_2d.to(torch.half),
            input_dim,
            output_dim,
            group_size=sf_vec_size)
        marlin_scale = marlin_utils.nvfp4_marlin_process_scales(marlin_scale)

        # Step 3: Process global scale (includes exponent bias correction)
        weight_global_scale = marlin_utils.nvfp4_marlin_process_global_scale(
            torch.tensor(1.0, dtype=torch.bfloat16, device='cuda'))

        if is_bf16_act:
            # BF16 path: pass BF16 activations directly, dummy scale_a/alpha
            mat_a = input_bf16
            scale_a = torch.zeros(1, dtype=torch.uint8, device='cuda')
            alpha_arg = torch.ones(1, dtype=torch.float32, device='cuda')
        else:
            # FP4 path: unswizzle activation scales for the dequant kernel
            mat_a = input_e2m1.cuda()
            m_padded = (batch_size + 128 - 1) // 128 * 128
            act_sf_gpu = input_e4m3_scale.cuda()
            scale_a = torch.ops.trtllm.block_scale_interleave_reverse(
                act_sf_gpu.view(m_padded, -1)).flatten()
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
        assert torch.allclose(output_cpu_float,
                              ref_output_fp32,
                              atol=0.75,
                              rtol=0.02)

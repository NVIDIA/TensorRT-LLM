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
import unittest

import tensorrt as trt
import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import (dequantize, quantize,
                                                  quantize_per_token)
from tensorrt_llm.quantization.layers import quantize_tensor

from . import _utils


class TestQuantizationFunctional(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', True), ('float16', True),
                           ('float32', False), ('float16', False),
                           ('bfloat16', True), ('bfloat16', False)],
                          name_func=unittest_name_func)
    def test_quantize_tensor(self, dtype, use_plugin):
        x_data = torch.randn(
            (1, 2, 2, 4),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")
        scaling_factor_data = torch.tensor(0.4, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        network = builder.create_network()
        if use_plugin:
            network.plugin_config.quantize_tensor_plugin = True

        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = quantize_tensor(x, scaling_factor)
            output.mark_output('output', trt.int8)

        session = create_session(builder, network, precision=dtype, int8=True)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        scaling_factor_data = scaling_factor_data.cuda()
        quantized = (x_data * scaling_factor_data).round().clip(
            -128, 127).to(dtype=torch.int8)
        torch.testing.assert_close(quantized, outputs['output'])

    def test_quantize_per_tensor(self):
        dtype = "float32"
        x_data = torch.randn(
            (1, 2, 2, 4),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")
        scaling_factor_data = torch.tensor(0.4, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = quantize(x, scaling_factor, 'int8')
            output.mark_output('output', trt.int8)

        session = create_session(builder, network, precision=dtype, int8=True)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)
        scaling_factor_data = scaling_factor_data.cuda()
        ref = torch.quantize_per_tensor(x_data, scaling_factor_data, 0,
                                        torch.qint8)

        torch.testing.assert_close(ref.int_repr(), outputs['output'])

    def test_quantize_per_channel(self):
        dtype = 'float32'
        x_data = torch.randn((4, 2, 4, 8), dtype=torch.float32, device="cuda")
        scaling_factor_data = torch.tensor((0.4, 0.3), dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())

            output = quantize(x, scaling_factor, 'int8', 1)
            output.mark_output('output', trt.int8)

        session = create_session(builder, network, precision=dtype, int8=True)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)
        scaling_factor_data = scaling_factor_data.cuda()
        ref = torch.quantize_per_channel(x_data, scaling_factor_data,
                                         torch.tensor([0, 0], device="cuda"), 1,
                                         torch.qint8)

        torch.testing.assert_close(ref.int_repr(), outputs['output'])

    @parameterized.expand([
        ('float32', False, False),
        ('float32', True, True),
        ('float32', True, False),
        ('float16', True, True),
        ('float16', True, False),
        ('bfloat16', True, True),
        ('bfloat16', True, False),
    ],
                          name_func=unittest_name_func)
    def test_quantize_per_token(self, dtype, use_plugin, sum_per_token):
        x_data = torch.randn(
            (4, 2, 4, 8),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        if use_plugin:
            network.plugin_config.quantize_per_token_plugin = True

        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            if sum_per_token:
                output, scale, sums = quantize_per_token(x, sum_per_token=True)
                sums.mark_output('sums', trt.float32)
            else:
                output, scale = quantize_per_token(x)

            output.mark_output('output', trt.int8)
            scale.mark_output('scale', dtype)

        session = create_session(builder, network, precision=dtype, int8=True)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        ref, ref_scale = _utils.gt_quantize_per_token(x_data)
        if sum_per_token:
            ref_sum = x_data.float().sum(dim=-1, keepdim=True)

        scale_shape = list(x_data.shape)
        scale_shape[-1] = 1
        ref_scale = ref_scale.reshape(scale_shape)

        torch.testing.assert_close(ref, outputs['output'], atol=1, rtol=1e-1)

        torch.testing.assert_close(ref_scale.float(),
                                   outputs['scale'].float(),
                                   atol=1e-2,
                                   rtol=0)
        if sum_per_token:
            torch.testing.assert_close(ref_sum, outputs['sums'])

    def test_dequantize(self):
        dtype = 'int8'
        x_data = torch.quantize_per_tensor(
            torch.tensor([-1.0, 0.0, 1.0, 2.0],
                         dtype=torch.float32,
                         device="cuda"), 0.1, 0, torch.qint8)

        scaling_factor_data = torch.tensor(0.1, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = dequantize(x, scaling_factor, output_type='float32')
            output.mark_output('output')

        session = create_session(builder, network, precision=float, int8=True)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        ref = torch.dequantize(x_data)

        torch.testing.assert_close(ref, outputs['output'])


if __name__ == '__main__':
    unittest.main()

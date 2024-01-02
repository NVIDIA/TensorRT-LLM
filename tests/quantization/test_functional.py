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

import _utils
import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import (dequantize, quantize,
                                                  quantize_per_token)
from tensorrt_llm.quantization.layers import quantize_tensor


class TestQuantization(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', True), ('float16', True),
                           ('float32', False)])
    def test_quantize_tensor(self, dtype, use_plugin):
        x_data = torch.randn(
            (1, 2, 2, 4), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        scaling_factor_data = torch.tensor(0.4, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        if use_plugin:
            net.plugin_config.set_quantize_tensor_plugin()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = quantize_tensor(x, scaling_factor)
            net._mark_output(output, 'output', trt.int8)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        quantized = (x_data.cuda() * scaling_factor_data.cuda()).round().clip(
            -128, 127).to(dtype=torch.int8)
        np.testing.assert_allclose(quantized.cpu().numpy(), outputs['output'])

    def test_quantize_per_tensor(self):
        dtype = "float32"
        x_data = torch.randn(
            (1, 2, 2, 4), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        scaling_factor_data = torch.tensor(0.4, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = quantize(x, scaling_factor, 'int8')
            net._mark_output(output, 'output', trt.int8)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.quantize_per_tensor(x_data, scaling_factor_data, 0,
                                        torch.qint8)

        np.testing.assert_allclose(ref.int_repr().cpu().numpy(),
                                   outputs['output'])

    def test_quantize_per_channel(self):
        dtype = 'float32'
        x_data = torch.randn((4, 2, 4, 8), dtype=torch.float32)
        scaling_factor_data = torch.tensor((0.4, 0.3), dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())

            output = quantize(x, scaling_factor, 'int8', 1)
            net._mark_output(output, 'output', trt.int8)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.quantize_per_channel(x_data, scaling_factor_data,
                                         torch.tensor([0, 0]), 1, torch.qint8)

        np.testing.assert_allclose(ref.int_repr().cpu().numpy(),
                                   outputs['output'])

    @parameterized.expand([('float32', True), ('float16', True),
                           ('float32', False)])
    def test_quantize_per_token(self, dtype, use_plugin):
        x_data = torch.randn(
            (4, 2, 4, 8), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        if use_plugin:
            net.plugin_config.set_quantize_per_token_plugin()

        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output, scale = quantize_per_token(x)
            net._mark_output(output, 'output', trt.int8)
            net._mark_output(scale, 'scale',
                             tensorrt_llm.str_dtype_to_trt(dtype))

        for l in net.trt_network:
            if l.get_output(0).dtype == tensorrt_llm._utils.str_dtype_to_trt(
                    "int8"):
                l.get_output(0).set_dynamic_range(-127, 127)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)

        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref, ref_scale = _utils.gt_quantize_per_token(x_data)
        scale_shape = list(x_data.shape)
        scale_shape[-1] = 1
        ref_scale = ref_scale.reshape(scale_shape)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

        np.testing.assert_allclose(ref_scale.cpu().numpy(),
                                   outputs['scale'],
                                   atol=1e-2)

    def test_dequantize(self):
        dtype = 'int8'
        quantized_torch_tensor = torch.quantize_per_tensor(
            torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32), 0.1, 0,
            torch.qint8)
        quantized_data = quantized_torch_tensor.int_repr()

        scaling_factor_data = torch.tensor(0.1, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=quantized_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            scaling_factor = tensorrt_llm.constant(scaling_factor_data.numpy())
            output = dequantize(x, scaling_factor).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={'x': quantized_data.cpu().numpy()})

        ref = torch.dequantize(quantized_torch_tensor)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])


if __name__ == '__main__':
    unittest.main()

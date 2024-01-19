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

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm.quantization.functional import smooth_quant_layer_norm


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = [('float16', False, True), ('float16', True, True),
                      ('float32', False, True), ('float32', True, True),
                      ('float16', True, False)]
        return [i + (True, )
                for i in test_cases] + [i + (False, ) for i in test_cases]

    @parameterized.expand(load_test_cases)
    def test_smooth_quant_layer_norm_plugin(self, dtype, dynamic_act_scaling,
                                            elementwise_affine,
                                            remove_batch_dim):
        torch.manual_seed(1997)
        # test data
        hidden_size = 1024
        x_data = torch.randn(
            (8, 128, hidden_size) if not remove_batch_dim else
            (8 * 128, hidden_size),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()
        eps = 1e-5

        m = torch.nn.LayerNorm(
            hidden_size,
            eps=eps,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            elementwise_affine=elementwise_affine).cuda()

        # Scale to int
        scale_data = torch.randint(2, 32, (1, ), dtype=torch.float32).cuda()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_layernorm_quantization_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            weight = None
            bias = None
            if elementwise_affine:
                gamma_data = m.weight.detach().cpu()
                beta_data = m.bias.detach().cpu()
                weight = Parameter(gamma_data.cpu().numpy()).value
                bias = Parameter(beta_data.cpu().numpy()).value
            scale = Parameter(scale_data.cpu().numpy()).value

            output = smooth_quant_layer_norm(
                x,
                hidden_size,
                weight,
                bias,
                scale,
                eps,
                dynamic_act_scaling=dynamic_act_scaling)

            if dynamic_act_scaling:
                output, dynamic_scales = output
                dynamic_scales = dynamic_scales.trt_tensor
                dynamic_scales.name = 'dynamic_scales'
                network.mark_output(dynamic_scales)
                dynamic_scales.dtype = tensorrt_llm.str_dtype_to_trt('float32')

            output = output.trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt('int8')

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(int8=True,
                                fp16=(dtype == 'float16'),
                                precision_constraints="obey"))
        assert build_engine is not None, "Build engine failed"
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu().numpy()})

        def cast_to_int8_with_sat(tensor):
            return tensor.round().clip(-128, 127).to(dtype=torch.int8)

        # pytorch run
        with torch.no_grad():
            ref = m(x_data).to(dtype=torch.float32)
            if dynamic_act_scaling:
                abs_max_f, _ = ref.abs().max(dim=-1, keepdim=True)
                dynamic_scale = abs_max_f / 127.0
                ref_quantized = cast_to_int8_with_sat(ref * (127.0 / abs_max_f))
            else:
                ref_quantized = cast_to_int8_with_sat(ref * scale_data)

        # compare diff of quantized output
        # Set absolute tolerance to 1 to mitigate some rounding error
        np.testing.assert_allclose(ref_quantized.cpu().numpy(),
                                   outputs['output'],
                                   atol=1,
                                   rtol=0)

        # compare diff of dynamic activation scales
        if dynamic_act_scaling:
            np.testing.assert_allclose(dynamic_scale.cpu().numpy(),
                                       outputs['dynamic_scales'],
                                       atol=1e-2)

    def test_sq_layer_norm_no_plugin(self):
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            tensorrt_llm.default_trtnet()
            # Get output tensor for SQ gemm
            with self.assertRaisesRegex(
                    TypeError,
                    "Smooth Quant Layer Norm is only supported with plugin"):
                smooth_quant_layer_norm(None, 0, None, None, None, 0)

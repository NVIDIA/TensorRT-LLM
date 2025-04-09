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

import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.quantization.functional import smooth_quant_layer_norm


class TestSmoothQuantLayerNorm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1997)
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = [('float16', False, True), ('float16', True, True),
                      ('bfloat16', False, True), ('bfloat16', True, True),
                      ('float32', False, True), ('float32', True, True),
                      ('float16', True, False)]
        test_cases = [i + (True, ) for i in test_cases
                      ] + [i + (False, ) for i in test_cases]
        return [i + (True, )
                for i in test_cases] + [i + (False, ) for i in test_cases]

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_smooth_quant_layer_norm(self, dtype, dynamic_act_scaling,
                                     elementwise_affine, remove_batch_dim,
                                     use_plugin):
        # test data
        hidden_size = 1024
        x_data = torch.randn(
            (8, 128, hidden_size) if not remove_batch_dim else
            (8 * 128, hidden_size),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")
        eps = 1e-5

        m = torch.nn.LayerNorm(
            hidden_size,
            eps=eps,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            elementwise_affine=elementwise_affine,
            device="cuda")

        # Scale to int
        scale_data = torch.randint(2,
                                   32, (1, ),
                                   dtype=torch.float32,
                                   device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        if use_plugin:
            network.plugin_config.layernorm_quantization_plugin = dtype
        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            weight = None
            bias = None
            if elementwise_affine:
                gamma_data = m.weight.detach().cpu()
                beta_data = m.bias.detach().cpu()
                weight = Parameter(torch_to_numpy(gamma_data)).value
                bias = Parameter(torch_to_numpy(beta_data)).value
            scale = Parameter(torch_to_numpy(scale_data)).value

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
                dynamic_scales.mark_output('dynamic_scales', 'float32')

            output.mark_output('output', 'int8')

        session = create_session(builder, network, precision=dtype, int8=True)
        inputs = {
            'x': x_data,
        }

        outputs = run_session(session, inputs)

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
        torch.testing.assert_close(ref_quantized,
                                   outputs['output'],
                                   atol=1,
                                   rtol=0)

        # compare diff of dynamic activation scales
        if dynamic_act_scaling:
            torch.testing.assert_close(dynamic_scale,
                                       outputs['dynamic_scales'],
                                       atol=1e-2,
                                       rtol=1e-2)


if __name__ == '__main__':
    unittest.main()

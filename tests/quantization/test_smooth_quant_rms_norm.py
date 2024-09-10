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
import os
import sys
import unittest

import torch
from parameterized import parameterized
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm.quantization.functional import smooth_quant_rms_norm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import (create_session, run_session, skip_bf16_pre_ampere,
                        unittest_name_func)


class TestSmoothQuantRmsNorm(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float16', False), ('float16', True),
                           ('bfloat16', False), ('bfloat16', True),
                           ('float32', False), ('float32', True)],
                          name_func=unittest_name_func)
    def test_smooth_quant_rms_norm_plugin(self, dtype, dynamic_act_scaling):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        test_shape = [2, 5, 10, 10]

        x_data = torch.randn(
            *test_shape,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        m = LlamaRMSNorm(
            test_shape[-1]).cuda()  # LlamaRMSNorm only supports last dim

        scale_data = torch.randint(2,
                                   32, (1, ),
                                   dtype=torch.float32,
                                   device="cuda")

        with torch.no_grad():

            def cast_to_int8_with_sat(tensor):
                return tensor.round().clip(-128, 127).to(dtype=torch.int8)

            # pytorch run
            with torch.no_grad():
                ref = m(x_data).to(dtype=torch.float32)
                if dynamic_act_scaling:
                    abs_max_f, _ = ref.abs().max(dim=-1, keepdim=True)
                    dynamic_scale = abs_max_f / 127.0
                    ref_quantized = cast_to_int8_with_sat(ref *
                                                          (127.0 / abs_max_f))
                else:
                    ref_quantized = cast_to_int8_with_sat(ref * scale_data)

        # construct trt network
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        network = builder.create_network()
        network.plugin_config.rmsnorm_quantization_plugin = dtype
        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = smooth_quant_rms_norm(
                x,
                test_shape[-1],
                weight=tensorrt_llm.constant(m.weight.detach().cpu().numpy()),
                scale=Parameter(scale_data.cpu().numpy()).value,
                eps=m.variance_epsilon,
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

    def test_sq_rms_norm_no_plugin(self):
        # Create builder
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        # Create empty network
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            # SQ Rmsnorm ootb should fail.
            with self.assertRaisesRegex(
                    TypeError,
                    "Smooth Quant Rms Norm is only supported with plugin"):
                smooth_quant_rms_norm(None, 0, None, None, None, 0)


if __name__ == '__main__':
    unittest.main()

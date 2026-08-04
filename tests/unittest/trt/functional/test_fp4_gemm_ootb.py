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
from itertools import product

import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from parameterized import parameterized
from utils.util import skip_pre_blackwell_unittest, unittest_name_func

import tensorrt_llm
import tensorrt_llm.quantization.functional
import tensorrt_llm.quantization.layers
from tensorrt_llm import Tensor


def random_quantized_tensor(shape, dtype, block_size):
    raw = torch.rand(shape, dtype=dtype)
    quantized, block_sf, global_sf = NVFP4QTensor.quantize(
        raw, block_size=block_size)
    return raw, quantized, block_sf, global_sf


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")

    @parameterized.expand(
        list(
            product([1, 4, 32, 128, 1023], [512, 1024, 2048], ["float16"],
                    [16])),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_nvfp4_qdq(self, batch_size, hidden_size, input_dtype, block_size):
        torch_dtype = tensorrt_llm.str_dtype_to_torch(input_dtype)
        raw, quantized, block_sf, global_sf = random_quantized_tensor(
            (batch_size, hidden_size), torch_dtype, block_size)

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            input = Tensor(shape=(batch_size, hidden_size),
                           dtype=input_dtype,
                           name="input")
            global_sf_tensor = tensorrt_llm.functional.constant(
                global_sf.cpu().numpy())

            quantized_tensor, block_sf_tensor = (
                tensorrt_llm.quantization.functional.dynamic_quantize(
                    input, global_sf_tensor, block_size=block_size))
            dequantized_tensor = tensorrt_llm.quantization.functional.block_double_dequantize(
                quantized_tensor,
                block_sf_tensor,
                global_sf_tensor,
                dtype="float32")
            output = dequantized_tensor.cast(input_dtype)
            output.mark_output("output")

        output_buffer = torch.zeros_like(raw)
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=input_dtype)
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(
            inputs={"input": raw},
            outputs={"output": output_buffer},
            stream=stream.cuda_stream,
        )
        torch.cuda.synchronize()

        ref_dequantized = quantized.dequantize(
            torch_dtype,
            scale=block_sf.float(),
            double_scale=global_sf,
            block_sizes=[16],
        )

        assert torch.allclose(output_buffer, ref_dequantized)

    @parameterized.expand(
        list(
            product([1, 16, 128, 1023], [512, 1024], [256, 2048], ["float16"],
                    [16])),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_nvfp4_gemm_ootb(self, batch_size, input_hidden_size,
                             output_hidden_size, input_dtype, block_size):
        torch_dtype = tensorrt_llm.str_dtype_to_torch(input_dtype)
        input_raw, input_quantized, input_block_sf, input_global_sf = (
            random_quantized_tensor((batch_size, input_hidden_size),
                                    torch_dtype, block_size))
        weight_raw, weight_quantized, weight_block_sf, weight_global_sf = (
            random_quantized_tensor((output_hidden_size, input_hidden_size),
                                    torch_dtype, block_size))
        bias_raw = torch.rand(output_hidden_size, dtype=torch_dtype)

        linear = tensorrt_llm.quantization.layers.FP4Linear(input_hidden_size,
                                                            output_hidden_size,
                                                            dtype=input_dtype)
        linear.weight.value = weight_quantized._quantized_data
        linear.weights_block_scaling_factor.value = weight_block_sf
        linear.weights_global_scaling_factor.value = weight_global_sf
        linear.activation_global_scaling_factor.value = input_global_sf
        linear.bias.value = bias_raw

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            input = Tensor(shape=(batch_size, input_hidden_size),
                           dtype=input_dtype,
                           name="input")
            output = linear(input)
            output.mark_output("output")

        output_buffer = torch.zeros((batch_size, output_hidden_size),
                                    dtype=torch_dtype)
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=input_dtype)
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(
            inputs={"input": input_raw},
            outputs={"output": output_buffer},
            stream=stream.cuda_stream,
        )
        torch.cuda.synchronize()

        ref_input = input_quantized.dequantize(
            torch_dtype,
            scale=input_block_sf.float(),
            double_scale=input_global_sf,
            block_sizes=[16],
        )

        ref_weight = weight_quantized.dequantize(
            torch_dtype,
            scale=weight_block_sf.float(),
            double_scale=weight_global_sf,
            block_sizes=[16],
        )

        ref_output = torch.nn.functional.linear(ref_input, ref_weight, bias_raw)
        assert torch.allclose(output_buffer, ref_output, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()

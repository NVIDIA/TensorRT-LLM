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
from parameterized import parameterized
from utils.util import (create_session, run_session,
                        skip_pre_blackwell_unittest, unittest_name_func)

import tensorrt_llm
from tensorrt_llm import Tensor


def float_tensor_to_e2m1_and_ufp8_scale(float_tensor: torch.Tensor,
                                        sf_vec_size,
                                        ufp8_type: int = 1):
    value_e2m1, scale_ufp8, rep_float = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
        float_tensor, sf_vec_size, ufp8_type)
    return value_e2m1, scale_ufp8, rep_float


def e2m1_and_ufp8_scale_to_float_tensor(e2m1_tensor: torch.Tensor,
                                        ufp8_scale_tensor: torch.Tensor,
                                        sf_vec_size,
                                        ufp8_type: int = 1):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float(
        e2m1_tensor, ufp8_scale_tensor, sf_vec_size, ufp8_type)
    return float_tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(list(product([4, 8], [16, 2048], ["fp8", "float16"])),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4quant(self, M, N, input_type):
        torch.random.manual_seed(0)
        shape = [M, N]
        sf_vec_size = 16
        input_type = "fp8"

        float_tensor = torch.randn(shape, dtype=torch.float32)
        e2m1_tensor, e8m0_sf_tensor, repr_float_tensor = float_tensor_to_e2m1_and_ufp8_scale(
            float_tensor, sf_vec_size)
        represented_float_tensor_ref = e2m1_and_ufp8_scale_to_float_tensor(
            e2m1_tensor, e8m0_sf_tensor, sf_vec_size)
        assert torch.equal(repr_float_tensor, represented_float_tensor_ref)

        fp8_tensor, _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            float_tensor)

        global_scale_factor = 448.0 / (float_tensor.abs().max() /
                                       6.0).to("cuda")
        global_scale_factor = global_scale_factor.reshape(1)
        if input_type == "fp8":
            plugin_input = fp8_tensor.to("cuda").view(torch.float8_e4m3fn)
        elif input_type == "float16":
            plugin_input = float_tensor.to(torch.float16).to("cuda")

        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            input_tensor = Tensor(
                name="input",
                shape=shape,
                dtype=tensorrt_llm.str_dtype_to_trt(input_type))
            sf_scale_tensor_trt = Tensor(
                name="sf_scale",
                shape=(1, ),
                dtype=tensorrt_llm.str_dtype_to_trt("float32"))

            quantized_input, input_sf_tensor = tensorrt_llm.quantization.functional.quantize_to_fp4_tensor(
                input_tensor, sf_scale_tensor_trt)

            net._mark_output(quantized_input,
                             'quant_tensor',
                             dtype=tensorrt_llm.str_dtype_to_trt("int64"))
            net._mark_output(input_sf_tensor,
                             'sf_tensor',
                             dtype=tensorrt_llm.str_dtype_to_trt("int32"))

        inputs = {
            'input': plugin_input,
            'sf_scale': global_scale_factor,
        }
        session = create_session(builder, net, precision="float16")
        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        plugin_output_quant_tensor = torch.tensor(
            outputs["quant_tensor"].untyped_storage(),
            dtype=torch.int8).reshape(e2m1_tensor.shape)
        plugin_output_sf_tensor = torch.tensor(
            outputs["sf_tensor"].untyped_storage(),
            dtype=torch.uint8).reshape(e8m0_sf_tensor.shape)
        represented_float_tensor = e2m1_and_ufp8_scale_to_float_tensor(
            plugin_output_quant_tensor, e8m0_sf_tensor, sf_vec_size)

        cos_similarity_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        res = cos_similarity_func(represented_float_tensor, float_tensor)
        assert res.max() > 0.95

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

import _utils

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import constant
from tensorrt_llm.quantization.functional import weight_only_quant_matmul

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestWeightOnlyQuantMatmul(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _unconvert_weights(self, weights, scales, dtype, wTypeId):
        if wTypeId == 1 or wTypeId == 2:
            pass
        else:
            assert (False)
        torch_dtype = _utils.woq_torch_dtype(dtype)
        # Init operands for multiplication in int32
        mat1 = torch.eye(weights.shape[0], dtype=torch_dtype)

        return self._run_matmul(mat1, weights, scales, dtype, wTypeId, True)

    def _run_matmul(self, mat1, processed_torch_weights, torch_weight_scales,
                    dtype, wTypeId, use_plugin):
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        # Allow WQ plugin of dtype type
        if use_plugin:
            net.plugin_config.set_weight_only_quant_matmul_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            # Init TensorRT-LLM tensor for mat1
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
            # Init TensorRT-LLM tensor for weight
            weights = constant(processed_torch_weights.numpy())
            # Init TensorRT-LLM tensor for per channel scaling
            scale = constant(torch_weight_scales.numpy())
            # Get output tensor for WOQ Matmul
            output = weight_only_quant_matmul(x, weights, scale,
                                              wTypeId).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm._utils.str_dtype_to_trt(dtype)

        # Build engine consisting of only WOQ Matmul
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=True,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        # Infer engine
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': mat1.numpy(),
            })

        return torch.tensor(outputs['output'])

    def _woq_matmul(self, m, n, k, dtype, wTypeId, use_plugin=True):
        # Init operands for multiplication in int32
        mat1 = _utils.woq_gen_weights(m, k, dtype) * 200.0
        weight = _utils.woq_gen_weights(k, n, dtype)

        ref_torch_weights, processed_torch_weights, torch_weight_scales = _utils.woq_conversion(
            weight, wTypeId)
        if wTypeId == 2 and use_plugin:
            ref_torch_weights = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(
                ref_torch_weights)
        if not use_plugin:
            processed_torch_weights = ref_torch_weights

        output = self._run_matmul(mat1, processed_torch_weights,
                                  torch_weight_scales, dtype, wTypeId,
                                  use_plugin)

        ref = _utils.woq_gt_matmul(m, mat1, ref_torch_weights,
                                   torch_weight_scales, dtype)

        _utils.woq_assert_colwise_near_eq(ref, output, wTypeId)
        '''
        ref = ref.cpu().flatten()
        diff = abs(ref - output)

        max_diff = diff.max()
        ref_value_of_max_diff = ref[diff == max_diff]
        out_value_of_max_diff = output[diff == max_diff]
        print("###############\nmax diff is {} form {} vs {}\n###############\n\n".format(max_diff, out_value_of_max_diff, ref_value_of_max_diff))
        '''

    @parameterized.expand([
        (1, 1024, 4096, 'float16', 1, True),
        (1, 1024, 4096, 'float16', 1, False),
        (128, 6144, 12288, 'float16', 1, True),  #FP16 * INT8
        (1, 1024, 4096, 'float16', 2, True),
        (128, 6144, 12288, 'float16', 2, True),
    ])  #FP16 * INT4
    def test_matmul(self, m, n, k, dtype, wTypeId, use_plugin):
        self._woq_matmul(m, n, k, dtype, wTypeId, use_plugin)

    @parameterized.expand([
        (1024, 4096, 'float16', 1), (4096, 512, 'float16', 1),
        (1024, 4096, 'float16', 2), (4096, 512, 'float16', 2)
    ])
    def test_conversion(self, n, k, dtype, wTypeId):
        weight_ref = _utils.woq_gen_weights(n, k, dtype)
        ref_int, perm_int, scale = _utils.woq_conversion(weight_ref, wTypeId)
        weight_act = self._unconvert_weights(perm_int, scale, dtype, wTypeId)

        _utils.woq_assert_colwise_near_eq(weight_ref, weight_act, wTypeId)


if __name__ == '__main__':
    unittest.main()

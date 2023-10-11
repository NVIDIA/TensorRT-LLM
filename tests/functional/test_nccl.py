# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm as tllm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import allreduce


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


class TestCommunicationPlugin(unittest.TestCase):

    def setUp(self):
        tllm.logger.set_level('error')
        self.world_size = tllm.mpi_world_size()
        self.rank = tllm.mpi_rank()

        torch.cuda.set_device(self.rank)
        self.reference_tensors = [
            torch.full([200000], i + 1, dtype=torch.float32, device="cuda")
            for i in range(self.world_size)
        ]
        self.group = [i for i in range(self.world_size)]

    @parameterized.expand(list(
        product(["bfloat16", 'float16', "float32"], [True, False],
                [256, 200000])),
                          name_func=custom_name_func)
    def test_nccl_allreduce(self, dtype: str, use_custom_all_reduce: bool,
                            size: int):
        if self.world_size == 1:
            pytest.skip()

        torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
        allreduce_ref = torch.zeros(self.reference_tensors[0][:size].shape,
                                    dtype=torch_dtype,
                                    device="cuda")
        for i in range(self.world_size):
            allreduce_ref = allreduce_ref + self.reference_tensors[i][:size].to(
                torch_dtype)

        builder = tllm.Builder()
        net = builder.create_network()
        net.plugin_config.set_nccl_plugin(dtype, use_custom_all_reduce)

        input = self.reference_tensors[self.rank][:size]
        with tllm.net_guard(net):
            network = tllm.default_trtnet()

            x = Tensor(name='x',
                       shape=input.shape,
                       dtype=tllm.str_dtype_to_trt(dtype))
            output = allreduce(x, self.group).trt_tensor

            output.name = 'output'
            output.dtype = tllm.str_dtype_to_trt(dtype)
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config=CreateConfig(
                                             fp16=(dtype == 'float16'),
                                             bf16=(dtype == 'bfloat16'),
                                             precision_constraints='obey',
                                         ))

        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': input.to(torch_dtype)})

        self.assertTrue(torch.allclose(outputs['output'], allreduce_ref.cpu()))

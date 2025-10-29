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

import pytest

# isort: off
import torch
# isort: on

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm.functional import (AllReduceParams, AllReduceStrategy,
                                     allreduce)
from tensorrt_llm.plugin.plugin import current_all_reduce_helper


class TestCommunicationPlugin(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')
        self.world_size = tensorrt_llm.mpi_world_size()
        self.rank = tensorrt_llm.mpi_rank()

        torch.cuda.set_device(self.rank)
        cudart.cudaSetDevice(self.rank)

        self.reference_tensors = [
            torch.full([10000000], i + 1, dtype=torch.float32, device="cuda")
            for i in range(self.world_size)
        ]
        self.mapping = Mapping(self.world_size,
                               self.rank,
                               self.world_size,
                               tp_size=self.world_size)

    @parameterized.expand(list(
        product(["bfloat16", 'float16', "float32"], [
            AllReduceStrategy.NCCL, AllReduceStrategy.ONESHOT,
            AllReduceStrategy.TWOSHOT
        ], [64 * 70000, 64 * 70, 64])),
                          name_func=unittest_name_func)
    def test_allreduce(self, dtype: str, strategy: AllReduceStrategy,
                       size: int):

        if self.world_size == 1:
            pytest.skip("Skip single GPU NCCL")

        workspace = None

        torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        dtype_size = torch.finfo(torch_dtype).bits // 8

        allreduce_ref = torch.zeros(self.reference_tensors[0][:size].shape,
                                    dtype=torch_dtype,
                                    device="cuda")
        for i in range(self.world_size):
            allreduce_ref = allreduce_ref + self.reference_tensors[i][:size].to(
                torch_dtype)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.set_nccl_plugin(dtype)
        _, workspace = current_all_reduce_helper().allocate_workspace(
            self.mapping, size * dtype_size)

        input = self.reference_tensors[self.rank][:size].to(torch_dtype)
        inner_loop = 5

        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=input.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            current_all_reduce_helper().set_workspace_tensor(self.mapping)

            current = x
            for i in range(inner_loop):
                current = allreduce(
                    current,
                    self.mapping.tp_group,
                    all_reduce_params=AllReduceParams(strategy=strategy))

            current.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': input, 'all_reduce_workspace': workspace}
        outputs = run_session(session, inputs)

        # compare diff
        torch.testing.assert_close(outputs['output'],
                                   (self.mapping.tp_size**(inner_loop - 1)) *
                                   allreduce_ref)

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
from cuda import cudart
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._ipc_utils import peer_access
from tensorrt_llm.functional import AllReduceStrategy, allreduce
from tensorrt_llm.plugin.plugin import (current_all_reduce_helper,
                                        init_all_reduce_helper)


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
        cudart.cudaSetDevice(self.rank)

        self.reference_tensors = [
            torch.full([10000000], i + 1, dtype=torch.float32, device="cuda")
            for i in range(self.world_size)
        ]
        self.mapping = Mapping(self.world_size, self.rank, self.world_size,
                               self.world_size)

    @parameterized.expand(list(
        product(["bfloat16", 'float16', "float32"], [
            AllReduceStrategy.RING, AllReduceStrategy.ONESHOT,
            AllReduceStrategy.TWOSHOT
        ], [64 * 70000, 64 * 70, 64])),
                          name_func=custom_name_func)
    def test_nccl_allreduce(self, dtype: str, strategy: AllReduceStrategy,
                            size: int):
        if self.world_size == 1:
            pytest.skip()

        workspace = None

        torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
        dtype_size = torch.finfo(torch_dtype).bits // 8
        init_all_reduce_helper()
        _, workspace = current_all_reduce_helper().allocate_workspace(
            self.mapping, size * dtype_size)
        print(workspace)

        allreduce_ref = torch.zeros(self.reference_tensors[0][:size].shape,
                                    dtype=torch_dtype,
                                    device="cuda")
        for i in range(self.world_size):
            allreduce_ref = allreduce_ref + self.reference_tensors[i][:size].to(
                torch_dtype)

        builder = tllm.Builder()
        net = builder.create_network()
        net.plugin_config.set_nccl_plugin(dtype)

        input = self.reference_tensors[self.rank][:size].to(torch_dtype)
        inner_loop = 5

        with peer_access(self.mapping):
            with tllm.net_guard(net):
                network = tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))
                current_all_reduce_helper().set_workspace_tensor(self.mapping)

                current = x
                for i in range(inner_loop):
                    current = allreduce(current, self.mapping.tp_group,
                                        strategy)
                output = current.trt_tensor

                output.name = 'output'
                output.dtype = tllm.str_dtype_to_trt(dtype)
                network.mark_output(output)

            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(
                    fp16=(dtype == 'float16'),
                    bf16=(dtype == 'bfloat16'),
                    precision_constraints='obey',
                ))

            output = torch.zeros_like(input)

            stream = torch.cuda.current_stream()
            feed_dict = {'x': input, 'all_reduce_workspace': workspace}

            session = tllm.runtime.Session.from_engine(build_engine())
            session.run(inputs=feed_dict,
                        outputs={"output": output},
                        stream=stream.cuda_stream)
            torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(output.cpu(),
                           (self.mapping.tp_size**(inner_loop - 1)) *
                           allreduce_ref.cpu()))


if __name__ == "__main__":
    unittest.main()

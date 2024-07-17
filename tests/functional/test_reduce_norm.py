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
import os
import sys

from cuda import cudart
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm._ipc_utils import peer_access
from tensorrt_llm.functional import (AllReduceConfig, AllReduceFusionOp,
                                     AllReduceFusionParams, AllReduceStrategy,
                                     allreduce)
from tensorrt_llm.plugin.plugin import current_all_reduce_helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import unittest_name_func


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


class TestCommunicationPlugin(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20240603)
        torch.cuda.manual_seed(20240603)
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
        product(['bfloat16', 'float16'], [
            AllReduceStrategy.NCCL, AllReduceStrategy.ONESHOT,
            AllReduceStrategy.TWOSHOT
        ], [AllReduceConfig(0)], [1, 4, 16, 64], [4096, 8192, 12288])),
                          name_func=unittest_name_func)
    def test_allreduce(self, dtype: str, strategy: AllReduceStrategy,
                       config: AllReduceConfig, token_num: int,
                       hidden_size: int):
        if self.world_size == 1:
            pytest.skip("Skip single GPU NCCL")
        if strategy == AllReduceStrategy.NCCL and config != AllReduceConfig(0):
            pytest.skip("NCCL with specific config discarded")
        size = token_num * hidden_size
        workspace = None
        torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
        dtype_size = torch.finfo(torch_dtype).bits // 8
        allreduce_ref = torch.zeros(self.reference_tensors[0][:size].shape,
                                    dtype=torch_dtype,
                                    device="cuda").reshape(
                                        token_num, hidden_size)
        residual = torch.rand(allreduce_ref.shape,
                              dtype=torch_dtype,
                              device="cuda")
        weight = torch.rand((1, hidden_size), dtype=torch_dtype, device="cuda")
        bias = torch.rand((1, hidden_size), dtype=torch_dtype, device="cuda")
        eps = 1e-6

        for i in range(self.world_size):
            allreduce_ref = allreduce_ref + self.reference_tensors[i][:size].to(
                torch_dtype).reshape(token_num, hidden_size)
        allreduce_ref = allreduce_ref + bias + residual
        allreduce_ref = rms_norm(allreduce_ref, weight, eps)

        builder = tllm.Builder()
        net = builder.create_network()
        net.plugin_config.set_nccl_plugin(dtype, use_custom_all_reduce=True)
        _, workspace = current_all_reduce_helper().allocate_workspace(
            self.mapping, size * dtype_size)

        input = self.reference_tensors[self.rank][:size].to(
            torch_dtype).reshape(token_num, hidden_size)

        with peer_access(self.mapping):
            with tllm.net_guard(net):
                network = tllm.default_trtnet()

                x = Tensor(name='x',
                           shape=input.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))
                y = Tensor(name='y',
                           shape=bias.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))
                z = Tensor(name='z',
                           shape=residual.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))
                w = Tensor(name='w',
                           shape=weight.shape,
                           dtype=tllm.str_dtype_to_trt(dtype))
                current_all_reduce_helper().set_workspace_tensor(self.mapping)

                current = x
                current, z = allreduce(
                    current,
                    self.mapping.tp_group,
                    strategy,
                    config,
                    reduce_fusion_params=AllReduceFusionParams(
                        AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        bias=y,
                        residual=z,
                        norm_weight=w,
                        eps=eps))
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
            feed_dict = {
                'x': input,
                'y': bias,
                'z': residual,
                'w': weight,
                'all_reduce_workspace': workspace
            }

            session = tllm.runtime.Session.from_engine(build_engine())
            session.run(inputs=feed_dict,
                        outputs={"output": output},
                        stream=stream.cuda_stream)
            torch.cuda.synchronize()

        close = torch.isclose(allreduce_ref, output, rtol=1e-2, atol=1e-3)
        if not torch.all(close):
            not_close_a = allreduce_ref[~close]
            not_close_b = output[~close]
            print("rank {}, \n{}\n{}".format(self.rank, allreduce_ref, output))
            print("mismatch value:")
            print("ref:", not_close_a)
            print("output:", not_close_b)

        self.assertTrue(
            torch.allclose(output.cpu(),
                           allreduce_ref.cpu(),
                           rtol=1e-2,
                           atol=1e-3))


if __name__ == "__main__":
    unittest.main()

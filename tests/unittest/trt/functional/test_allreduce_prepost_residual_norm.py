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

import tensorrt_llm as tllm
from tensorrt_llm import Mapping, Tensor
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AllReduceStrategy, allreduce)
from tensorrt_llm.plugin.plugin import (current_all_reduce_helper,
                                        init_all_reduce_helper)


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
        self.mapping = Mapping(self.world_size,
                               self.rank,
                               self.world_size,
                               tp_size=self.world_size)

    @parameterized.expand(list(
        product(['float16', 'bfloat16'],
                [AllReduceStrategy.ONESHOT, AllReduceStrategy.NCCL],
                [1, 4, 16, 64], [4096, 8192])),
                          name_func=unittest_name_func)
    def test_allreduce(self, dtype: str, strategy: AllReduceStrategy,
                       token_num: int, hidden_size: int):
        if self.world_size == 1:
            pytest.skip("Skip single GPU NCCL")
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
        weight_preresidual = torch.rand((1, hidden_size),
                                        dtype=torch_dtype,
                                        device="cuda")
        bias = torch.rand((1, hidden_size), dtype=torch_dtype, device="cuda")
        eps = 1e-6

        for i in range(self.world_size):
            allreduce_ref = allreduce_ref + self.reference_tensors[i][:size].to(
                torch_dtype).reshape(token_num, hidden_size)
        allreduce_ref = allreduce_ref + bias
        allreduce_ref = rms_norm(allreduce_ref, weight_preresidual, eps)
        allreduce_ref = allreduce_ref + residual
        allreduce_ref = rms_norm(allreduce_ref, weight, eps)

        builder = tllm.Builder()
        net = builder.create_network()
        net.plugin_config.set_nccl_plugin(dtype)
        init_all_reduce_helper()
        _, workspace = current_all_reduce_helper().allocate_workspace(
            self.mapping, size * dtype_size)

        input = self.reference_tensors[self.rank][:size].to(
            torch_dtype).reshape(token_num, hidden_size)

        with tllm.net_guard(net):
            tllm.default_trtnet()

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
            w_preresidual = Tensor(name='w_preresidual',
                                   shape=weight_preresidual.shape,
                                   dtype=tllm.str_dtype_to_trt(dtype))
            current_all_reduce_helper().set_workspace_tensor(self.mapping)

            current = x
            current, z = allreduce(
                current,
                self.mapping.tp_group,
                all_reduce_params=AllReduceParams(
                    strategy=strategy,
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM,
                    bias=y,
                    residual=z,
                    norm_weight=w,
                    norm_pre_residual_weight=w_preresidual,
                    eps=eps),
            )
            current.mark_output('output', dtype)

        feed_dict = {
            'x': input,
            'y': bias,
            'z': residual,
            'w': weight,
            'w_preresidual': weight_preresidual,
            'all_reduce_workspace': workspace
        }
        session = create_session(builder, net, precision=dtype)
        outputs = run_session(session, feed_dict)

        rtol = 1e-2
        atol = 1e-2
        if dtype == 'bfloat16':
            rtol *= 3
            atol *= 3

        close = torch.isclose(allreduce_ref,
                              outputs['output'],
                              rtol=rtol,
                              atol=atol)
        if not torch.all(close):
            not_close_a = allreduce_ref[~close]
            not_close_b = outputs['output'][~close]
            print("rank {}, \n{}\n{}".format(self.rank, allreduce_ref,
                                             outputs['output']))
            print("mismatch value:")
            print("ref:", not_close_a)
            print("output:", not_close_b)

        self.assertTrue(
            torch.allclose(outputs['output'].cpu(),
                           allreduce_ref.cpu(),
                           rtol=rtol,
                           atol=atol))


if __name__ == "__main__":
    unittest.main()

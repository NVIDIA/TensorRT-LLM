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
from tensorrt_llm.functional import (allgather, allreduce, concat, recv,
                                     reduce_scatter, send)
from tensorrt_llm.plugin.plugin import (current_all_reduce_helper,
                                        init_all_reduce_helper)


def forward_allreduce(x: Tensor, y: Tensor, mapping: Mapping) -> Tensor:
    current = x
    if mapping.tp_size > 1 and mapping.tp_group is not None:
        current = allreduce(current, mapping.tp_group)
    current = current + y
    return current


def forward_reduce_scatter(x: Tensor, y: Tensor, mapping: Mapping,
                           hidden_size: int) -> Tensor:
    if mapping.tp_rank == 0:
        current = x + y
    else:
        current = x + 0
    # reshape to (-1)
    current = current.view(concat([-1]))
    if mapping.tp_size > 1 and mapping.tp_group is not None:
        current = reduce_scatter(current, mapping.tp_group)
    # reshape to (-1, hidden_size // tp_size)
    current = current.view(concat([-1, hidden_size // mapping.tp_size]))
    return current


class TestPPReduceScatter(unittest.TestCase):

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

    @parameterized.expand(list(
        product(['bfloat16', 'float16', 'float32'], [1, 4, 16, 64],
                [4096, 8192, 12288], [2, 4, 8])),
                          name_func=unittest_name_func)
    def test_pp_reduce_scatter(self, dtype: str, token_num: int,
                               hidden_size: int, pp_size: int):
        if self.world_size == 1 or pp_size > self.world_size:
            pytest.skip("Skip single GPU and pp_size > world_size case")
        tp_size = self.world_size // pp_size
        mapping = Mapping(self.world_size,
                          self.rank,
                          self.world_size,
                          tp_size=tp_size,
                          pp_size=pp_size)

        size = token_num * hidden_size  # tensor size
        torch_dtype = tllm._utils.str_dtype_to_torch(dtype)
        dtype_size = torch.finfo(torch_dtype).bits // 8
        input = self.reference_tensors[self.rank][:size].to(
            torch_dtype).reshape(token_num, hidden_size)
        residual = torch.rand(input.shape, dtype=torch_dtype, device="cuda")
        input_recv = torch.zeros(torch.Size([token_num,
                                             hidden_size // tp_size]),
                                 dtype=torch_dtype,
                                 device="cuda")

        builder = tllm.Builder()
        net_ref = builder.create_network()
        net = builder.create_network()
        init_all_reduce_helper()
        _, workspace = current_all_reduce_helper().allocate_workspace(
            mapping, size * dtype_size)

        with tllm.net_guard(net_ref):
            x = Tensor(name='x',
                       shape=input.shape,
                       dtype=tllm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=residual.shape,
                       dtype=tllm.str_dtype_to_trt(dtype))
            current_all_reduce_helper().set_workspace_tensor(mapping)

            if not mapping.is_first_pp_rank():
                net_ref_input = x
                net_ref_input = recv(net_ref_input, mapping.prev_pp_rank())
            else:
                net_ref_input = x

            if not mapping.is_last_pp_rank():
                output_ref = forward_allreduce(net_ref_input, y, mapping)
                output_ref = send(output_ref, mapping.next_pp_rank())
            else:
                output_ref = forward_allreduce(net_ref_input, y, mapping)

            output_ref.mark_output('output', dtype)

        with tllm.net_guard(net):
            x = Tensor(name='x',
                       shape=input.shape,
                       dtype=tllm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=residual.shape,
                       dtype=tllm.str_dtype_to_trt(dtype))
            x_recv = Tensor(name='x_recv',
                            shape=torch.Size(
                                [token_num, hidden_size // mapping.tp_size]),
                            dtype=tllm.str_dtype_to_trt(dtype))
            current_all_reduce_helper().set_workspace_tensor(mapping)

            if not mapping.is_first_pp_rank():
                net_input = x_recv
                net_input = recv(net_input, mapping.prev_pp_rank())
                net_input = allgather(net_input, mapping.tp_group, gather_dim=0)
                # reshape to (-1, hidden_size)
                net_input = net_input.view(concat([-1, hidden_size]))
            else:
                net_input = x

            if not mapping.is_last_pp_rank():
                output = forward_reduce_scatter(net_input, y, mapping,
                                                hidden_size)
                output = send(output, mapping.next_pp_rank())
            else:
                output = forward_allreduce(net_input, y, mapping)

            output.mark_output('output', dtype)

        feed_dict_ref = {
            'x': input,
            'y': residual,
            'all_reduce_workspace': workspace
        }

        feed_dict = {
            'x': input,
            'y': residual,
            'x_recv': input_recv,
            'all_reduce_workspace': workspace
        }

        # trt run
        session_ref = create_session(builder, net_ref, precision=dtype)
        outputs_ref = run_session(session_ref, feed_dict_ref)

        session = create_session(builder, net, precision=dtype)
        outputs = run_session(session, feed_dict)

        # compare diff
        if mapping.is_last_pp_rank():
            torch.testing.assert_allclose(outputs['output'],
                                          outputs_ref['output'],
                                          atol=1e-5,
                                          rtol=1e-2)


if __name__ == "__main__":
    unittest.main()

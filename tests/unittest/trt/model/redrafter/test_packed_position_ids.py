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

import torch
from utils.util import create_session, run_session, set_input_shapes

import tensorrt_llm
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    def test_packed_position_ids(self):
        bs = 2
        nb = 3
        bl = 4
        max_gen = nb * (bl - 1) + 1
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        active_indices = torch.tensor(
            [[0, 1, 2, 3, 6, 7, 11, 0, 1], [0, 1, 2, 3, 5, 6, 7, 10, 11]],
            dtype=torch.int32) % bl
        total_lengths = torch.tensor([7, 9], dtype=torch.int32)
        total_gen_len = total_lengths.sum()
        max_tl = total_lengths.max()
        indices = torch.arange(max_tl, dtype=torch.int32)
        position_ids_base = torch.tensor([3, 10], dtype=torch.int32)

        # ref outputs
        ref_packed_position_ids = torch.tensor(
            [3, 4, 5, 6, 5, 6, 6, 10, 11, 12, 13, 11, 12, 13, 12, 13],
            dtype=torch.int32)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            active_indices_t = Tensor(
                name='ai',
                shape=[-1, -1],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            indices_t = Tensor(name='i',
                               shape=[-1],
                               dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            total_lengths_t = Tensor(
                name='tl',
                shape=[-1],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            position_ids_base_t = Tensor(
                name='pib',
                shape=[-1],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            output = tensorrt_llm.models.redrafter.redrafter_helper._get_packed_position_ids(
                active_indices_t,
                indices_t,
                total_lengths_t,
                position_ids_base_t,
            )
            output.mark_output('packed_position_ids')
            # save onnx
            # model_path = 'packed_position.onnx'
            # to_onnx(net.trt_network, model_path)

        # trt run
        # needs profile for dynamic shape
        profile = builder.trt_builder.create_optimization_profile()
        set_input_shapes(profile, active_indices_t, [1, 0], [16, max_gen // 2],
                         [32, max_gen])
        set_input_shapes(profile, indices_t, [0], [max_gen // 2], [max_gen])
        set_input_shapes(profile, total_lengths_t, [1], [16], [32])
        set_input_shapes(profile, position_ids_base_t, [1], [16], [32])
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            'ai': active_indices,
            'i': indices,
            'tl': total_lengths,
            'pib': position_ids_base,
        }
        outputs = {
            "packed_position_ids": torch.empty((bs * nb * bl, ),
                                               dtype=torch.int32),
        }
        outputs = run_session(session, inputs, outputs)
        torch.testing.assert_close(
            outputs['packed_position_ids'][:total_gen_len],
            ref_packed_position_ids,
            rtol=0,
            atol=0)
        torch.set_default_device(old_device)
        return

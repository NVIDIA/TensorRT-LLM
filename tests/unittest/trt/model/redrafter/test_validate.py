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

    def test_validate(self):
        bs = 2
        nb = 3
        bl = 4
        V = 4
        S = max(7, 9)
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(0)
        greedy_search = True
        draft_probs = torch.rand((bs, nb, bl - 1, V), dtype=torch.float32)
        draft_tokens = torch.tensor([[
            [91, 92, 93, 95],
            [91, 92, 94, 96],
            [91, 92, 93, 97],
        ], [
            [93, 94, 95, 92],
            [93, 95, 96, 93],
            [93, 94, 97, 96],
        ]],
                                    dtype=torch.int32) % V
        draft_tokens = torch.randint(10, size=(bs, nb, bl),
                                     dtype=torch.int32) % V
        draft_indices = torch.tensor([[
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [0, 1, 2, 6],
        ], [
            [0, 1, 2, 3],
            [0, 4, 5, 6],
            [0, 1, 7, 8],
        ]],
                                     dtype=torch.int32) % S
        draft_indices = torch.randint(10, size=(bs, nb, bl),
                                      dtype=torch.int32) % S
        flattened_logits = torch.rand((bs, S, V), dtype=torch.float32)
        rand_data = torch.rand((bs, nb, bl - 1), dtype=torch.float32)

        # ref outputs
        ref_max = torch.tensor([0, 1], dtype=torch.int32)
        ref_beam = torch.tensor([0, 0], dtype=torch.int32)
        ref_probs = torch.tensor([[
            [[-0., -50000., -50000., -50000.], [-0., -50000., -50000., -50000.],
             [-50000., -50000., -0., -50000.]],
            [[-50000., -0., -50000., -50000.], [-50000., -50000., -50000., -0.],
             [-50000., -50000., -50000., -0.]],
            [[-0., -50000., -50000., -50000.], [-50000., -50000., -0., -50000.],
             [-50000., -0., -50000., -50000.]]
        ],
                                  [[[-50000., -0., -50000., -50000.],
                                    [-0., -50000., -50000., -50000.],
                                    [-0., -50000., -50000., -50000.]],
                                   [[-50000., -50000., -0., -50000.],
                                    [-50000., -50000., -0., -50000.],
                                    [-0., -50000., -50000., -50000.]],
                                   [[-50000., -0., -50000., -50000.],
                                    [-50000., -50000., -0., -50000.],
                                    [-50000., -50000., -0., -50000.]]]],
                                 dtype=torch.float32)
        ref_last_probs = torch.tensor([[[-50000., -50000., -50000., -0.],
                                        [-50000., -50000., -0., -50000.],
                                        [-50000., -50000., -50000., -0.]],
                                       [[-50000., -50000., -0., -50000.],
                                        [-0., -50000., -50000., -50000.],
                                        [-50000., -0., -50000., -50000.]]],
                                      dtype=torch.float32)

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            draft_probs_t = Tensor(name='draft_probs',
                                   shape=[-1] + list(draft_probs.shape[1:]),
                                   dtype=tensorrt_llm.torch_dtype_to_trt(
                                       draft_probs.dtype))
            draft_tokens_t = Tensor(name='draft_tokens',
                                    shape=[-1] + list(draft_tokens.shape[1:]),
                                    dtype=tensorrt_llm.torch_dtype_to_trt(
                                        draft_tokens.dtype))
            draft_indices_t = Tensor(name='draft_indices',
                                     shape=[-1] + list(draft_indices.shape[1:]),
                                     dtype=tensorrt_llm.torch_dtype_to_trt(
                                         draft_indices.dtype))
            flattened_logits_t = Tensor(
                name='flattened_logits',
                shape=[-1, -1] + list(flattened_logits.shape[2:]),
                dtype=tensorrt_llm.torch_dtype_to_trt(flattened_logits.dtype))
            rand_data_t = Tensor(name='rand_data',
                                 shape=[-1] + list(rand_data.shape[1:]),
                                 dtype=tensorrt_llm.torch_dtype_to_trt(
                                     rand_data.dtype))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._validate_draft_tokens(
                draft_probs_t, draft_tokens_t, draft_indices_t,
                flattened_logits_t, nb, bl, greedy_search, rand_data_t)
            outputs[0].mark_output('max_num_accept_tokens')
            outputs[1].mark_output('beam_index')
            outputs[2].mark_output('base_log_probs')
            outputs[3].mark_output('last_base_log_probs')

        # trt run
        profile = builder.trt_builder.create_optimization_profile()
        set_input_shapes(profile, draft_probs_t, [0, nb, bl - 1, V],
                         [16, nb, bl - 1, V], [32, nb, bl - 1, V])
        set_input_shapes(profile, draft_indices_t, [0, nb, bl], [16, nb, bl],
                         [32, nb, bl])
        set_input_shapes(profile, draft_tokens_t, [0, nb, bl], [16, nb, bl],
                         [32, nb, bl])
        set_input_shapes(profile, rand_data_t, [0, nb, bl - 1],
                         [16, nb, bl - 1], [32, nb, bl - 1])
        set_input_shapes(profile, flattened_logits_t, [1, 1, V], [16, 8, V],
                         [32, 16, V])
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            'draft_probs': draft_probs,
            'draft_tokens': draft_tokens,
            'draft_indices': draft_indices,
            'rand_data': rand_data,
            'flattened_logits': flattened_logits,
        }
        outputs = run_session(session, inputs)

        # compare diff
        torch.testing.assert_close(ref_max,
                                   outputs['max_num_accept_tokens'],
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(ref_beam,
                                   outputs['beam_index'],
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(ref_probs,
                                   outputs['base_log_probs'],
                                   atol=0.01,
                                   rtol=0.1)
        torch.testing.assert_close(ref_last_probs,
                                   outputs['last_base_log_probs'],
                                   atol=0.01,
                                   rtol=0.1)
        torch.set_default_device(old_device)
        return

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
from utils.util import create_session, run_session

import tensorrt_llm
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor

# from tensorrt_llm.tools.onnx_utils import to_onnx

T, F = True, False


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def test_beams2tree(self):
        # test data
        bs = 2
        nb = 3
        bl = 4
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        beams = torch.tensor(
            [  # Assuming a batch of two sequences, each has 3 beams of 4 tokens.
                [
                    [91, 92, 93, 95],
                    [91, 92, 94, 96],
                    [91, 92, 93, 97],
                ],
                [
                    [93, 94, 95, 92],
                    [93, 95, 96, 93],
                    [93, 94, 97, 96],
                ],
            ],
            dtype=torch.int32,
        )
        position_ids_base = torch.tensor([3, 10], dtype=torch.int32)
        assert beams.shape == (bs, nb, bl)
        assert position_ids_base.shape == (bs, )
        # ref outputs
        ref_active_flat_tokens = torch.tensor(
            [91, 92, 93, 95, 94, 96, 97, 93, 94, 95, 92, 95, 96, 93, 97, 96],
            dtype=torch.int32)
        ref_active_token_indices = torch.tensor(
            [[0, 1, 2, 3, 6, 7, 11, 0, 1], [0, 1, 2, 3, 5, 6, 7, 10, 11]],
            dtype=torch.int32)
        ref_draft_token_indices = torch.tensor(
            [
                [
                    [0, 1, 2, 3],
                    [0, 1, 4, 5],
                    [0, 1, 2, 6],
                ],
                [
                    [0, 1, 2, 3],
                    [0, 4, 5, 6],
                    [0, 1, 7, 8],
                ],
            ],
            dtype=torch.int32,
        )
        ref_total_lengths = torch.tensor([7, 9], dtype=torch.int32)
        ref_max_len = ref_total_lengths.max().int()
        ref_total_gen_len = ref_total_lengths.sum().int()
        ref_position_offsets = ref_active_token_indices % bl
        position_ids = ref_position_offsets + position_ids_base.unsqueeze(1)
        ref_packed_position_ids = torch.concat(
            [position_ids[b, :ref_total_lengths[b]] for b in range(bs)]).int()
        ref_mask = torch.tensor([[[T, F, F, F, F, F, F, F, F],
                                  [T, T, F, F, F, F, F, F, F],
                                  [T, T, T, F, F, F, F, F, F],
                                  [T, T, T, T, F, F, F, F, F],
                                  [T, T, F, F, T, F, F, F, F],
                                  [T, T, F, F, T, T, F, F, F],
                                  [T, T, T, F, F, F, T, F, F],
                                  [T, F, F, F, F, F, F, F, F],
                                  [T, T, F, F, F, F, F, F, F]],
                                 [[T, F, F, F, F, F, F, F, F],
                                  [T, T, F, F, F, F, F, F, F],
                                  [T, T, T, F, F, F, F, F, F],
                                  [T, T, T, T, F, F, F, F, F],
                                  [T, F, F, F, T, F, F, F, F],
                                  [T, F, F, F, T, T, F, F, F],
                                  [T, F, F, F, T, T, T, F, F],
                                  [T, T, F, F, F, F, F, T, F],
                                  [T, T, F, F, F, F, F, T, T]]])

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            beams_t = Tensor(name='beams',
                             shape=beams.shape,
                             dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            position_ids_base_t = Tensor(
                name='position_ids_base',
                shape=position_ids_base.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._beams2tree(
                beams_t, nb, bl, position_ids_base_t)
            outputs[0].mark_output('active_tokens_flattened')
            outputs[1].mark_output('draft_token_indices')
            outputs[2].mark_output('spec_decoding_mask')
            outputs[3].mark_output('position_offsets')
            outputs[4].mark_output('packed_position_ids')
            outputs[5].mark_output('total_lengths')
            outputs[6].mark_output('max_gen_len')
            outputs[7].mark_output('total_gen_len')
            # save onnx
            # model_path = 'b2t.onnx'
            # to_onnx(network.trt_network, model_path)
        session = create_session(
            builder,
            network,
            precision='float32',
        )
        inputs = {
            'beams': beams,
            'position_ids_base': position_ids_base,
        }
        outputs = {
            "active_tokens_flattened":
            torch.empty((bs * nb * bl, ), dtype=torch.int32),
            "draft_token_indices":
            torch.empty((bs, nb, bl), dtype=torch.int32),
            "spec_decoding_mask":
            torch.empty((bs * nb * bl * nb * bl, ), dtype=torch.bool),
            "position_offsets":
            torch.empty((bs * nb * bl, ), dtype=torch.int32),
            "packed_position_ids":
            torch.empty((bs * nb * bl, ), dtype=torch.int32),
            "total_lengths":
            torch.empty((bs, ), dtype=torch.int32),
            "max_len":
            torch.empty((), dtype=torch.int32),
            "total_gen_len":
            torch.empty((), dtype=torch.int32),
        }
        outputs = run_session(session, inputs, outputs)
        # # compare diff
        torch.testing.assert_close(outputs['max_gen_len'],
                                   ref_max_len,
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(outputs['total_gen_len'],
                                   ref_total_gen_len,
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(outputs['total_lengths'],
                                   ref_total_lengths,
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(
            outputs['active_tokens_flattened'][:ref_total_gen_len],
            ref_active_flat_tokens,
            rtol=0,
            atol=0)
        torch.testing.assert_close(outputs['draft_token_indices'],
                                   ref_draft_token_indices,
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(
            outputs["spec_decoding_mask"][:bs * ref_max_len * ref_max_len].view(
                bs, ref_max_len, ref_max_len),
            ref_mask,
            rtol=0,
            atol=0)
        torch.testing.assert_close(
            outputs['position_offsets'][:bs * ref_max_len].view(
                bs, ref_max_len),
            ref_position_offsets,
            rtol=0,
            atol=0)
        torch.testing.assert_close(
            outputs['packed_position_ids'][:ref_total_gen_len],
            ref_packed_position_ids,
            rtol=0,
            atol=0)
        torch.set_default_device(old_device)
        return

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

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from utils.util import (create_session, run_session, set_input_shapes,
                        unittest_name_func)

import tensorrt_llm
import tensorrt_llm.models.eagle
from tensorrt_llm import Tensor
from tensorrt_llm.layers import AttentionParams


def pack_mask(bool_mask):
    num_tokens = len(bool_mask)
    N = len(bool_mask[0])
    num_ints = -(-N // 32)  # Equivalent to ceil(N / 32)
    bitmask = np.zeros((num_tokens, num_ints), dtype=np.int32)

    for ii in range(num_tokens):
        for jj in range(N):
            if bool_mask[ii][jj]:
                int_index = jj // 32
                bit_index = jj % 32
                bitmask[ii, int_index] |= (1 << bit_index)
    return torch.from_numpy(bitmask).to('cuda')


class TestEaglePrepareDrafterInputsPlugin(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    def load_test_cases():
        test_cases = []
        ################# CASE 0 ##########################
        # BS=1, layer_idx=0 (Ctx Eagle Net), gen request.
        # EAGLE-1.
        layer_idx = 0
        num_layers = 3
        max_non_leaves_per_layer = 2
        sequence_lengths = torch.tensor([6], dtype=torch.int32, device="cuda")
        context_lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([0, 1, 2, 3, 4],
                                 dtype=torch.int32,
                                 device="cuda")
        chunked_context_next_tokens = torch.tensor([-1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[0, 1, 5, -1]],
                                          dtype=torch.int32,
                                          device="cuda")
        accepted_lens = torch.tensor([3], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([1], dtype=torch.int32, device="cuda")
        next_draft_tokens = torch.tensor([[0, 0, 0, 0]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        prev_draft_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
        # Next path is the same as prev path.
        prev_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        # Irrelevant for level=0 plugin
        hidden_size_batch_level_starts = torch.tensor([0, 0, 0, 0],
                                                      dtype=torch.int32,
                                                      device="cuda")
        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.tensor([0, 1, 2, 3, 4],
                                        dtype=torch.int32,
                                        device="cuda")

        input_spec_decoding_generation_lengths = torch.tensor([0],
                                                              dtype=torch.int32,
                                                              device="cuda")
        # Refs
        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_sequence_lengths = torch.tensor([4],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([3],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([0, 1, 5],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([1, 2, 3],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 2, 4, 0],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([3],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([1],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([3, 1],
                                              dtype=torch.int32,
                                              device="cuda")

        ref_spec_decoding_generation_lengths = None
        ref_spec_decoding_position_offsets = None
        ref_spec_decoding_packed_mask = None

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 1 ##########################
        # BS=1, layer_idx=0 (Ctx Eagle Net), ctx request.
        # EAGLE-1.
        layer_idx = 0
        sequence_lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        context_lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        chunked_context_next_tokens = torch.tensor([-1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[4, -1, -1, -1]],
                                          dtype=torch.int32,
                                          device="cuda")
        accepted_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
        next_draft_tokens = torch.tensor([[0, 0, 0, 0]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        prev_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        # Next path is the same as prev path.
        prev_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.empty((0, ), dtype=torch.int32)

        input_spec_decoding_generation_lengths = torch.empty((0, ),
                                                             dtype=torch.int32)

        # Refs
        ref_sequence_lengths = torch.tensor([4],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([4],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([1, 2, 3, 4],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([0, 1, 2, 3],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 1, 2, 3],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([4],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([1],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([4, 1],
                                              dtype=torch.int32,
                                              device="cuda")
        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = None
        ref_spec_decoding_position_offsets = None
        ref_spec_decoding_packed_mask = None

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 2 ##########################
        # BS=2, layer_idx=0 (Ctx Eagle Net), 2 gen requests.
        # EAGLE-1.
        layer_idx = 0
        sequence_lengths = torch.tensor([6, 8],
                                        dtype=torch.int32,
                                        device="cuda")
        context_lengths = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([0, 1, 2, 10, 11, 12, 13],
                                 dtype=torch.int32,
                                 device="cuda")
        chunked_context_next_tokens = torch.tensor([-1, -1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[0, 5, -1, -1], [10, 11, 20, -1]],
                                          dtype=torch.int32,
                                          device="cuda")
        accepted_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0, 1],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_tokens = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, 3, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        prev_draft_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        # Next path is the same as prev path.
        prev_draft_paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, 3, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.zeros((7, ), dtype=torch.int32)

        input_spec_decoding_generation_lengths = torch.zeros((2, ),
                                                             dtype=torch.int32)

        # Refs
        ref_sequence_lengths = torch.tensor([5, 7],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([2, 3],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([0, 5, 10, 11, 20],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([3, 4, 4, 5, 6],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 1, 3, 5, 6, 0, 0, 0],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([5],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([2],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([2, 5, 1, 1],
                                              dtype=torch.int32,
                                              device="cuda")
        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1, 2],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = None
        ref_spec_decoding_position_offsets = None
        ref_spec_decoding_packed_mask = None

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 3 ##########################
        # BS=2, layer_idx=0 (Ctx Eagle Net), 1 ctx and 1 gen requests.
        # EAGLE-1.
        layer_idx = 0
        sequence_lengths = torch.tensor([3, 8],
                                        dtype=torch.int32,
                                        device="cuda")
        context_lengths = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([0, 1, 2, 10, 11, 12, 13],
                                 dtype=torch.int32,
                                 device="cuda")
        chunked_context_next_tokens = torch.tensor([-1, -1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[3, -1, -1, -1], [10, 11, 20, -1]],
                                          dtype=torch.int32,
                                          device="cuda")
        accepted_lens = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0, 1],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_tokens = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, 3, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        prev_draft_lens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        # Next path is the same as prev path.
        prev_draft_paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, 3, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.zeros((4, ), dtype=torch.int32)

        input_spec_decoding_generation_lengths = torch.zeros((1, ),
                                                             dtype=torch.int32)

        # Refs
        ref_sequence_lengths = torch.tensor([3, 7],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([3, 3],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([1, 2, 3, 10, 11, 20],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([0, 1, 2, 4, 5, 6],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 1, 2, 3, 5, 6, 0],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([6],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([2],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([3, 6, 1, 1],
                                              dtype=torch.int32,
                                              device="cuda")
        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1, 2],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = None
        ref_spec_decoding_position_offsets = None
        ref_spec_decoding_packed_mask = None

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 4 ##########################
        # BS=1, layer_idx=1 (Gen Eagle Net).
        # EAGLE-1.
        layer_idx = 1
        # seq len of the EagleNet0
        sequence_lengths = torch.tensor([6], dtype=torch.int32, device="cuda")
        context_lengths = torch.tensor([3], dtype=torch.int32, device="cuda")
        next_draft_tokens = torch.tensor([[11, 12, 13, 14, 15, 16, 17]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, 3, 6], [0, 1, 4, -1], [0, 2, 5, 7], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Not relevant here
        input_ids = torch.tensor([0], dtype=torch.int32,
                                 device="cuda")  # Not relevant here
        chunked_context_next_tokens = torch.tensor([-1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[3, -1, -1, -1]],
                                          dtype=torch.int32,
                                          device="cuda")  # Not relevant here
        accepted_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
        prev_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        prev_draft_paths = torch.tensor(
            [[[0, 1, 3, 6], [0, 1, 4, -1], [0, 2, 5, 7], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        hidden_size_batch_level_starts = torch.tensor([0, 1, 0, 0],
                                                      dtype=torch.int32,
                                                      device="cuda")

        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.zeros((4, ), dtype=torch.int32)

        input_spec_decoding_generation_lengths = torch.zeros((1, ),
                                                             dtype=torch.int32)

        # Refs
        ref_sequence_lengths = torch.tensor([8],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([3],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([11, 12],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([6], dtype=torch.int32, device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 0],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([2],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([2],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([1, 2],
                                              dtype=torch.int32,
                                              device="cuda")

        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1, 3],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = torch.tensor([2],
                                                            dtype=torch.int32,
                                                            device="cuda")
        ref_spec_decoding_position_offsets = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        masks = [[True, False], [False, True]]
        ref_spec_decoding_packed_mask = pack_mask(masks)

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 5 ##########################
        # BS=1, layer_idx=2 (Gen Eagle Net).
        # EAGLE-1.
        layer_idx = 2
        # Same inputs as test Case 4
        hidden_size_batch_level_starts = torch.tensor([0, 2, 4, 0],
                                                      dtype=torch.int32,
                                                      device="cuda")

        # Refs
        ref_sequence_lengths = torch.tensor([10],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([3],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([11, 12, 13, 15],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([6], dtype=torch.int32, device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 0, 2, 3],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([4],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([2],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([3, 4],
                                              dtype=torch.int32,
                                              device="cuda")

        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 2, 4, 6],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = torch.tensor([4],
                                                            dtype=torch.int32,
                                                            device="cuda")
        ref_spec_decoding_position_offsets = torch.tensor(
            [[0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
        # yapf: disable
        masks = [[True, False, False, False],
                 [False, True, False, False],
                 [True, False, True, False],
                 [False, True, False, True]]
        # yapf: enable
        ref_spec_decoding_packed_mask = pack_mask(masks)

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 6 ##########################
        # BS=2, layer_idx=2 (Gen Eagle Net).
        # EAGLE-1.
        layer_idx = 2
        # seq len of the EagleNet0
        sequence_lengths = torch.tensor([6, 8],
                                        dtype=torch.int32,
                                        device="cuda")
        context_lengths = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
        next_draft_tokens = torch.tensor(
            [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, -1]],
            dtype=torch.int32,
            device="cuda")
        next_draft_lens = torch.tensor([7, 6], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, 3, 6], [0, 1, 4, -1], [0, 2, 5, 7], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]],
             [[0, 1, 5, 6], [0, 2, -1, -1], [0, 3, -1, -1], [0, 4, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Not relevant here
        input_ids = torch.tensor([0], dtype=torch.int32,
                                 device="cuda")  # Not relevant here
        chunked_context_next_tokens = torch.tensor([-1, -1],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[3, -1, -1, -1], [3, -1, -1, -1]],
                                          dtype=torch.int32,
                                          device="cuda")  # Not relevant here
        accepted_lens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0, 0],
                                         dtype=torch.int32,
                                         device="cuda")
        prev_draft_lens = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        prev_draft_paths = torch.tensor(
            [[[0, 1, 3, 6], [0, 1, 4, -1], [0, 2, 5, 7], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]],
             [[0, 1, 5, 6], [0, 2, -1, -1], [0, 3, -1, -1], [0, 4, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        hidden_size_batch_level_starts = torch.tensor([0, 1, 3, 5, 7, 0, 0],
                                                      dtype=torch.int32,
                                                      device="cuda")

        # Refs
        ref_sequence_lengths = torch.tensor([10, 10],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([3, 5],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([11, 12, 13, 15, 21, 25],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([6, 8],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 0, 3, 4, 1, 5, 0, 0],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([6],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([3],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([3, 4, 6, 1],
                                              dtype=torch.int32,
                                              device="cuda")
        ref_out_hidden_size_batch_level_starts = torch.tensor(
            [0, 1, 3, 5, 7, 9, 11], dtype=torch.int32, device="cuda")

        ref_spec_decoding_generation_lengths = torch.tensor([4, 2],
                                                            dtype=torch.int32,
                                                            device="cuda")
        ref_spec_decoding_position_offsets = torch.tensor(
            [[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda")
        # yapf: disable
        masks = [[True, False, False, False],
                 [False, True, False, False],
                 [True, False, True, False],
                 [False, True, False, True],
                 [True, False, False, False],
                 [True, True, False, False]]
        # yapf: enable
        ref_spec_decoding_packed_mask = pack_mask(masks)

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        ################# CASE 7 ##########################
        # BS=1, layer_idx=0 (Ctx Eagle Net), chunked ctx request.
        # EAGLE-1.
        layer_idx = 0
        sequence_lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        context_lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        chunked_context_next_tokens = torch.tensor([5],
                                                   dtype=torch.int32,
                                                   device="cuda")
        accepted_token_ids = torch.tensor([[4, -1, -1, -1]],
                                          dtype=torch.int32,
                                          device="cuda")
        accepted_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
        accepted_path_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
        next_draft_tokens = torch.tensor([[0, 0, 0, 0]],
                                         dtype=torch.int32,
                                         device="cuda")
        next_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        next_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")
        prev_draft_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        # Next path is the same as prev path.
        prev_draft_paths = torch.tensor(
            [[[0, 1, 3, -1], [0, 2, 4, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        # Content of these two tensors is irrelevant. Only shape
        input_gen_tokens = torch.empty((0, ), dtype=torch.int32)

        input_spec_decoding_generation_lengths = torch.empty((0, ),
                                                             dtype=torch.int32)

        # Refs
        ref_sequence_lengths = torch.tensor([4],
                                            dtype=torch.int32,
                                            device="cuda")
        ref_context_lengths = torch.tensor([4],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_output_ids = torch.tensor([1, 2, 3, 5],
                                      dtype=torch.int32,
                                      device="cuda")
        ref_position_ids = torch.tensor([0, 1, 2, 3],
                                        dtype=torch.int32,
                                        device="cuda")
        ref_hidden_states_indices = torch.tensor([0, 1, 2, 3],
                                                 dtype=torch.int32,
                                                 device="cuda")
        ref_num_output_tokens = torch.tensor([4],
                                             dtype=torch.int32,
                                             device="cuda")
        ref_num_last_token_indices = torch.tensor([1],
                                                  dtype=torch.int32,
                                                  device="cuda")
        ref_last_token_indices = torch.tensor([4, 1],
                                              dtype=torch.int32,
                                              device="cuda")
        ref_out_hidden_size_batch_level_starts = torch.tensor([0, 1],
                                                              dtype=torch.int32,
                                                              device="cuda")

        ref_spec_decoding_generation_lengths = None
        ref_spec_decoding_position_offsets = None
        ref_spec_decoding_packed_mask = None

        test_cases += [[
            layer_idx, num_layers, max_non_leaves_per_layer, sequence_lengths,
            context_lengths, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths,
            ref_sequence_lengths, ref_context_lengths,
            ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts
        ]]

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_prepare_draft_inputs_plugin(
            self, layer_idx, num_layers, max_non_leaves_per_layer,
            sequence_lengths, context_lengths, input_ids,
            chunked_context_next_tokens, accepted_token_ids, accepted_lens,
            accepted_path_ids, next_draft_tokens, next_draft_lens,
            next_draft_paths, prev_draft_lens, prev_draft_paths,
            hidden_size_batch_level_starts, input_gen_tokens,
            input_spec_decoding_generation_lengths, ref_sequence_lengths,
            ref_context_lengths, ref_spec_decoding_generation_lengths,
            ref_spec_decoding_position_offsets, ref_spec_decoding_packed_mask,
            ref_output_ids, ref_position_ids, ref_hidden_states_indices,
            ref_last_token_indices, ref_num_output_tokens,
            ref_num_last_token_indices, ref_out_hidden_size_batch_level_starts):

        print_tensors = False
        # Few sanity checks
        batch_size = sequence_lengths.shape[0]
        max_decoding_tokens = prev_draft_paths.shape[1]
        max_path_len = prev_draft_paths.shape[2]

        assert num_layers + 1 == max_path_len
        # assert max_non_leaves_per_layer * num_layers + 1 <= max_decoding_tokens
        assert sequence_lengths.shape[0] == batch_size

        assert context_lengths.shape[0] == batch_size

        assert chunked_context_next_tokens.shape[0] == batch_size

        assert accepted_token_ids.shape[0] == batch_size
        assert accepted_token_ids.shape[1] == max_path_len

        assert accepted_lens.shape[0] == batch_size

        assert accepted_path_ids.shape[0] == batch_size

        assert next_draft_tokens.shape[0] == batch_size
        assert next_draft_tokens.shape[1] == max_decoding_tokens - 1

        assert next_draft_lens.shape[0] == batch_size

        assert prev_draft_lens.shape[0] == batch_size

        assert next_draft_paths.shape[0] == batch_size
        assert next_draft_paths.shape[1] == max_decoding_tokens
        assert next_draft_paths.shape[2] == max_path_len

        assert prev_draft_paths.shape[0] == batch_size
        assert prev_draft_paths.shape[1] == max_decoding_tokens
        assert prev_draft_paths.shape[2] == max_path_len

        assert ref_sequence_lengths.shape[0] == batch_size
        assert ref_context_lengths.shape[0] == batch_size

        if layer_idx > 0:
            assert ref_spec_decoding_generation_lengths.shape[0] == batch_size

            assert ref_spec_decoding_position_offsets.shape[0] == batch_size
            assert ref_spec_decoding_position_offsets.shape[
                1] == max_decoding_tokens

            assert ref_spec_decoding_packed_mask.shape[
                0] == ref_num_output_tokens[0]

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            sequence_lengths_t = Tensor(name='sequence_lengths',
                                        dtype=trt.int32,
                                        shape=sequence_lengths.shape)
            context_lengths_t = Tensor(name='context_lengths',
                                       dtype=trt.int32,
                                       shape=context_lengths.shape)
            input_ids_t = Tensor(name='input_ids',
                                 dtype=trt.int32,
                                 shape=input_ids.shape)
            chunked_context_next_tokens_t = Tensor(
                name='chunked_context_next_tokens',
                dtype=trt.int32,
                shape=chunked_context_next_tokens.shape)
            accepted_token_ids_t = Tensor(name='accepted_token_ids',
                                          dtype=trt.int32,
                                          shape=accepted_token_ids.shape)
            accepted_lens_t = Tensor(name='accepted_lens',
                                     dtype=trt.int32,
                                     shape=accepted_lens.shape)
            accepted_path_ids_t = Tensor(name='accepted_path_ids',
                                         dtype=trt.int32,
                                         shape=accepted_path_ids.shape)
            next_draft_tokens_t = Tensor(name='next_draft_tokens',
                                         dtype=trt.int32,
                                         shape=next_draft_tokens.shape)
            next_draft_lens_t = Tensor(name='next_draft_lens',
                                       dtype=trt.int32,
                                       shape=next_draft_lens.shape)
            next_draft_paths_t = Tensor(name='next_draft_paths',
                                        dtype=trt.int32,
                                        shape=next_draft_paths.shape)
            prev_draft_lens_t = Tensor(name='prev_draft_lens',
                                       dtype=trt.int32,
                                       shape=prev_draft_lens.shape)
            prev_draft_paths_t = Tensor(name='prev_draft_paths',
                                        dtype=trt.int32,
                                        shape=prev_draft_paths.shape)
            hidden_size_batch_level_starts_t = Tensor(
                name='hidden_size_batch_level_starts',
                dtype=trt.int32,
                shape=hidden_size_batch_level_starts.shape)
            input_gen_tokens_t = Tensor(name='input_gen_tokens',
                                        dtype=trt.int32,
                                        shape=[-1])
            input_spec_decoding_generation_lengths_t = Tensor(
                name='input_spec_decoding_generation_lengths',
                dtype=trt.int32,
                shape=[-1])

            attention_params = AttentionParams()
            attention_params.sequence_length = sequence_lengths_t
            attention_params.context_lengths = context_lengths_t

            output = tensorrt_llm.models.eagle.model.eagle_prepare_drafter_inputs_plugin(
                layer_idx, num_layers, max_non_leaves_per_layer,
                attention_params, input_ids_t, chunked_context_next_tokens_t,
                accepted_token_ids_t, accepted_lens_t, accepted_path_ids_t,
                next_draft_tokens_t, next_draft_lens_t, next_draft_paths_t,
                prev_draft_lens_t, prev_draft_paths_t,
                hidden_size_batch_level_starts_t, input_gen_tokens_t,
                input_spec_decoding_generation_lengths_t)

            output[0].mark_output('output_sequence_lengths')
            output[1].mark_output('output_context_lengths')
            output[2].mark_output('spec_decoding_generation_lengths')
            output[3].mark_output('spec_decoding_position_offsets')
            output[4].mark_output('spec_decoding_packed_mask')
            output[5].mark_output('output_ids')
            output[6].mark_output('output_position_ids')
            output[7].mark_output('hidden_states_indices')
            output[8].mark_output('last_token_indices')
            output[9].mark_output('num_last_token_indices')
            output[10].mark_output('out_hidden_size_batch_level_starts')

        def get_ranges(shape):
            return [*shape], [*shape], [*shape]

        profile = builder.trt_builder.create_optimization_profile()
        set_input_shapes(profile, input_gen_tokens_t, [0], [1],
                         [max_decoding_tokens * batch_size])
        set_input_shapes(profile, input_spec_decoding_generation_lengths_t, [0],
                         [1], [batch_size])

        # trt run
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            "sequence_lengths": sequence_lengths,
            "context_lengths": context_lengths,
            "input_ids": input_ids,
            "chunked_context_next_tokens": chunked_context_next_tokens,
            "accepted_token_ids": accepted_token_ids,
            "accepted_lens": accepted_lens,
            "accepted_path_ids": accepted_path_ids,
            "next_draft_tokens": next_draft_tokens,
            "next_draft_lens": next_draft_lens,
            "next_draft_paths": next_draft_paths,
            "prev_draft_lens": prev_draft_lens,
            "prev_draft_paths": prev_draft_paths,
            "hidden_size_batch_level_starts": hidden_size_batch_level_starts
        }

        override_shapes = {}
        override_types = {}
        if input_spec_decoding_generation_lengths.shape[0] != 0:
            inputs["input_gen_tokens"] = input_gen_tokens
            inputs[
                "input_spec_decoding_generation_lengths"] = input_spec_decoding_generation_lengths
        else:
            # When tensors are empty, use this WAR to avoid setting nullptr to the engine
            inputs["input_gen_tokens"] = 4
            inputs["input_spec_decoding_generation_lengths"] = 4
            override_shapes = {
                "input_gen_tokens": (0, ),
                "input_spec_decoding_generation_lengths": (0, )
            }
            override_types = {
                "input_gen_tokens": torch.int32,
                "input_spec_decoding_generation_lengths": torch.int32
            }

        outputs = run_session(session,
                              inputs,
                              override_shapes=override_shapes,
                              override_types=override_types)

        if print_tensors:
            print("output_sequence_lengths", outputs['output_sequence_lengths'])
            print("output_context_lengths", outputs['output_context_lengths'])
            print("output_ids", outputs['output_ids'])
            print("output_position_ids", outputs['output_position_ids'])
            print("hidden_states_indices", outputs['hidden_states_indices'],
                  ref_hidden_states_indices)
            print("last_token_indices", outputs['last_token_indices'])
            # print("num_last_token_indices", outputs['num_last_token_indices'])
            print("out_hidden_size_batch_level_starts",
                  outputs['out_hidden_size_batch_level_starts'])
            print("num_last_token_indices", outputs['num_last_token_indices'])

        torch.testing.assert_close(ref_sequence_lengths,
                                   outputs['output_sequence_lengths'],
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(ref_context_lengths,
                                   outputs['output_context_lengths'],
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(
            ref_output_ids,
            outputs['output_ids'][:ref_num_output_tokens[0]],
            rtol=0,
            atol=0)
        num_position_ids = ref_num_output_tokens[
            0] if layer_idx == 0 else batch_size
        torch.testing.assert_close(
            ref_position_ids,
            outputs['output_position_ids'][:num_position_ids],
            rtol=0,
            atol=0)
        torch.testing.assert_close(ref_hidden_states_indices,
                                   outputs['hidden_states_indices'],
                                   rtol=0,
                                   atol=0)

        torch.testing.assert_close(ref_num_last_token_indices,
                                   outputs['num_last_token_indices'],
                                   rtol=0,
                                   atol=0)

        torch.testing.assert_close(ref_last_token_indices,
                                   outputs['last_token_indices'],
                                   rtol=0,
                                   atol=0)

        torch.testing.assert_close(
            ref_out_hidden_size_batch_level_starts,
            outputs['out_hidden_size_batch_level_starts']
            [:ref_out_hidden_size_batch_level_starts.shape[0]],
            rtol=0,
            atol=0)

        # Do not need spec decoding data when layer idx is 0 (EagleNet is ctx).
        if layer_idx > 0:
            out_mask_shape = outputs['spec_decoding_packed_mask'].shape
            out_mask = outputs['spec_decoding_packed_mask'].reshape(
                out_mask_shape[0] * out_mask_shape[1],
                -1)[:ref_num_output_tokens[0]]

            if print_tensors:
                print("spec_decoding_generation_lengths",
                      outputs['spec_decoding_generation_lengths'])
                print("spec_decoding_position_offsets",
                      outputs['spec_decoding_position_offsets'])
                print("spec_decoding_packed_mask", out_mask)
                print("ref_spec_decoding_packed_mask",
                      ref_spec_decoding_packed_mask)

            torch.testing.assert_close(
                ref_spec_decoding_generation_lengths,
                outputs['spec_decoding_generation_lengths'],
                rtol=0,
                atol=0)

            for bi in range(batch_size):
                torch.testing.assert_close(
                    ref_spec_decoding_position_offsets[
                        bi, :ref_spec_decoding_generation_lengths[bi]],
                    outputs['spec_decoding_position_offsets'][
                        bi, :ref_spec_decoding_generation_lengths[bi]],
                    rtol=0,
                    atol=0)

            torch.testing.assert_close(ref_spec_decoding_packed_mask,
                                       out_mask,
                                       rtol=0,
                                       atol=0)

if __name__ == "__main__":
    unittest.main()

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

import tensorrt as trt
import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
import tensorrt_llm.models.eagle
from tensorrt_llm import Tensor


def logsoftmax(input_logits):
    m = torch.nn.LogSoftmax(dim=-1)
    return m(input_logits)


def generate_ref_eagle2(layerIdx, batch_size, input_logits,
                        dynamic_tree_max_topK, input_prev_paths,
                        input_prev_scores, input_draft_token_ids,
                        input_all_layers_scores,
                        input_all_layers_draft_token_ids,
                        input_all_layers_draft_token_ids_predecessor):

    # input_logits: [batch_size * dynamic_tree_max_topK, vocab_size_padded]
    # input_prev_paths: [batch_size, max_decoding_tokens, max_path_len]
    # input_prev_scores: [batch_size, max_decoding_draft_tokens]
    # input_draft_token_ids: [batch_size, max_decoding_draft_tokens]
    # input_all_layers_scores: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
    # input_all_layers_draft_token_ids: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
    # input_all_layers_draft_token_ids_predecessor: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

    ref_return_draft_token_ids = []
    ref_return_current_scores = []
    ref_return_output_all_layers_scores = []
    ref_return_output_all_layers_draft_token_ids = []

    ref_return_next_expand_indices = []
    ref_return_output_all_layers_draft_token_ids_predecessor = []

    for bix in range(batch_size):
        logits = input_logits[
            bix * dynamic_tree_max_topK:(bix + 1) *
            dynamic_tree_max_topK]  # shape [dynamic_tree_max_topK, vocab_size_padded]

        # Reference the official implementation: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py#L704
        last_p = logsoftmax(logits)
        top = torch.topk(last_p, dynamic_tree_max_topK, dim=-1)

        topk_index, topk_p = top.indices, top.values  # both shape [dynamic_tree_max_topK, dynamic_tree_max_topK]

        # print(f"input_prev_scores[bix, :dynamic_tree_max_topK]: {input_prev_scores[bix, :dynamic_tree_max_topK]}")
        prev_scores = input_prev_scores[
            bix][:dynamic_tree_max_topK]  # [dynamic_tree_max_topK]
        cu_scores = topk_p + prev_scores[:,
                                         None]  # shape [dynamic_tree_max_topK, dynamic_tree_max_topK]

        topk_cs = torch.topk(cu_scores.view(-1), dynamic_tree_max_topK, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values  # both shape [dynamic_tree_max_topK]

        # We sort here to match our implement. We need to ensure that the expand tokenIds index are increase from left to right
        topk_cs_index, topk_cs_sort_idx = torch.sort(topk_cs_index,
                                                     descending=False)
        topk_cs_p = topk_cs_p[topk_cs_sort_idx]

        next_scores = topk_cs_p
        output_ids = topk_index.view(-1)[topk_cs_index]

        # Concat with input
        ## draft token ids
        # only slice meaningful values
        cur_input_draft_token_ids = input_draft_token_ids[
            bix][:layerIdx * dynamic_tree_max_topK]

        ref_return_draft_token_ids.append(
            torch.cat((cur_input_draft_token_ids, output_ids), dim=0))

        nun_all_layers_scores = (
            layerIdx - 1
        ) * dynamic_tree_max_topK * dynamic_tree_max_topK + dynamic_tree_max_topK
        ## all layers scores
        prev_input_all_layers_scores = input_all_layers_scores[bix].view(
            -1)[:nun_all_layers_scores]
        ref_return_output_all_layers_scores.append(
            torch.cat((prev_input_all_layers_scores, cu_scores.view(-1)),
                      dim=0))

        ## all layers draft tokens
        prev_input_all_layers_draft_token_ids = input_all_layers_draft_token_ids[
            bix].view(-1)[:nun_all_layers_scores]
        ref_return_output_all_layers_draft_token_ids.append(
            torch.cat(
                (prev_input_all_layers_draft_token_ids, topk_index.view(-1)),
                dim=0))

        ## current scores
        ref_return_current_scores.append(next_scores)

        ## next expand indices
        start_offset = (
            layerIdx - 1
        ) * dynamic_tree_max_topK * dynamic_tree_max_topK + dynamic_tree_max_topK + 1
        ref_return_next_expand_indices.append(topk_cs_index + start_offset)

        ## all layers draft tokenids predecessor
        assert (len(topk_cs_index) == dynamic_tree_max_topK)
        cur_layer_predecessor = (topk_cs_index +
                                 start_offset) // dynamic_tree_max_topK
        cur_layer_predecessor = cur_layer_predecessor.repeat_interleave(
            dynamic_tree_max_topK)
        prev_input_all_layers_draft_token_ids_predecessor = input_all_layers_draft_token_ids_predecessor[
            bix].view(-1)[:nun_all_layers_scores]
        ref_return_output_all_layers_draft_token_ids_predecessor.append(
            torch.cat((prev_input_all_layers_draft_token_ids_predecessor,
                       cur_layer_predecessor),
                      dim=0))



    return ref_return_draft_token_ids, ref_return_current_scores, ref_return_next_expand_indices, \
        ref_return_output_all_layers_scores, ref_return_output_all_layers_draft_token_ids, ref_return_output_all_layers_draft_token_ids_predecessor


class TestEagleDecodeDraftTokensPlugin(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def load_test_cases():
        test_cases = []

        ################# Eagle-1 test cases ##########################
        ################# CASE 0 ##########################
        # BS=1, topK sampling
        # 1 input logits, from node "0"
        # layer_id = 0
        # logits_data_type = float32
        batch_size = 1
        layerId = 0
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, 3, -100],  # Top3 id = 6, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [1, 8, 4]

        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([0], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 1 ##########################
        # BS=2, topK sampling
        # 2 input logits, from req0 node "0" and req1 node "0"
        # layer_id = 0
        # logits_data_type = float32
        batch_size = 2
        layerId = 0
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, -100, -100],  # Top2 id = 3, 2
                [-100, 3, -100, 2, -100, 1, -100, -100],  # Top3 id = 1, 3, 5
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([2],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, 4, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [2, 5, 4]

        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1], [-1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([0, 0],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[3, 2, -1, -1], [1, 3, 5, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [2, 3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 2 ##########################
        # BS=1, topK sampling
        # 2 input loigts, from req0 node "1" and "3"
        # layer_id = 1
        # logits_data_type = float32

        batch_size = 1
        layerId = 1
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, -100, 1, -100, -100, -100, -100],  # Top1 id = 3
                [-100, 1, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([2],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [1, 8, 4]

        input_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([3], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, 3, 1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [5], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 3 ##########################
        # BS=2, topK sampling
        # 1 input loigts, from req1, node "3"
        # layer_id = 1
        # logits_data_type = float32

        batch_size = 1
        layerId = 1
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, -100, -100, -100, 1, -100, -100],  # Top1 id = 5
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, 4, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [2, 5, 4]

        input_draft_token_ids = torch.tensor(
            [[2, 1, -1, -1], [1, 2, 3, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([2, 3],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[2, 1, -1, -1], [1, 2, 3, 5]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [2, 4], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 4 ##########################
        # BS=2, topK sampling
        # 2 input logits, from req0 node "4" and req1 node "4"
        # layer_id = 2
        # logits_data_type = float32

        batch_size = 2
        layerId = 2
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, -100, -100],  # Top2 id = 3, 2
                [-100, -100, -100, -100, 1, 2, -100, -100],  # Top2 id = 5, 4
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([2],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]],
             [[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [2, 8, 4]

        input_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, -1, -1], [1, 2, 3, 4, 5, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([5, 5],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 3, 2], [1, 2, 3, 4, 5, 5, 4]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [7, 7], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 5 ##########################
        # BS=1, topK sampling
        # 1 input logits, from req0 node "0"
        # layer_id = 0
        # logits_data_type = float16

        batch_size = 1
        layerId = 0
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float16
        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, 3, -100],  # Top3 id = 6, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [1, 8, 4]

        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([0], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 6 ##########################
        # BS=1, topK sampling
        # 5 input logits, only the 1st is valid, from req0 node "0"
        # layer_id = 0
        # logits_data_type = float16

        batch_size = 1
        layerId = 0
        dynamic_tree_max_topK_t = -1
        num_eagle_layers = 4
        max_decoding_draft_tokens = 7

        logits_data_type = torch.float16
        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, 3, -100],  # Top3 id = 6, 3, 2
                [1, 2, -100, -100, -100, -100, -100, 3],  # Top3 id = 7, 1, 0
                [1, 2, -100, -100, -100, -100, -100, 3],  # Top3 id = 7, 1, 0
                [1, 2, -100, -100, -100, -100, -100, 3],  # Top3 id = 7, 1, 0
                [1, 2, -100, -100, -100, -100, -100, 3],  # Top3 id = 7, 1, 0
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, 4, 6], [0, 1, 4, 7], [0, 2, -1, -1], [0, 3, 5, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len] -> [1, 8, 4]

        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_draft_lens = torch.tensor([0], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]

        top_k_sampling = True
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        # Eagle-2 related inputs/outputs, useless for Eagle-1
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_current_expand_indices = torch.full(
            (batch_size, max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_path = paths  # Same as the input
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        # ################# Eagle-2 test cases ##########################
        # ################# CASE 0: test the first layer ##########################
        # BS=1, topK sampling
        # 1 input logits, from node "0"
        # layerId = 0
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 1
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 0

        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, 3, -100],  # Top3 id = 6, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.full(
            (batch_size, max_decoding_tokens, max_path_len),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([0], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[0, -1, -1, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        ref_return_output_path = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        # For the layerIdx = 0
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        pp = log_softmax(logits)
        top_k_result = torch.topk(input=pp, k=dynamic_tree_max_topK_t, dim=-1)

        ref_return_current_scores = top_k_result.values
        ref_return_next_expand_indices = torch.tensor(
            [[1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_all_layers_scores = top_k_result.values
        ref_return_output_all_layers_draft_token_ids = top_k_result.indices
        ref_return_output_all_layers_draft_token_ids_predecessor = torch.tensor(
            [[0, 0, 0]], dtype=torch.int32, device="cuda"
        )  # Actual shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
        # Since we will save this value continuously

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 1: test the first layer ##########################
        # BS=2, topK sampling

        # In this test, in the second sampling, each node will has 1 leaf
        # The input path is:
        # [
        #   [0, 1, -1, -1],
        #   [0, 2, -1, -1],
        #   [0, 3, -1, -1]
        #   [-1, -1, -1, -1],
        #   ...
        # ]
        # The output path is:
        # [
        #   [0, 1, 4, -1],
        #   [0, 2, 5, -1],
        #   [0, 3, 6, -1],
        #   [-1, -1, -1, -1],
        #   ...
        # ]

        # 2 input logits, from node "0"
        # layerId = 0
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 2
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 0

        logits = torch.tensor(
            [
                [-100, -100, 1, 2, -100, -100, 3, -100],  # Top3 id = 6, 3, 2
                [-100, 10, 1, -100, -100, 20, -100, -100],  # Top3 id = 5, 1, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([2],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.full(
            (batch_size, max_decoding_tokens, max_path_len),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([0, 0],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[0, -1, -1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1], [5, 1, 2, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3, 3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        ref_return_output_path = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        # For the layerIdx = 0
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        pp = log_softmax(logits)
        top_k_result = torch.topk(input=pp, k=dynamic_tree_max_topK_t, dim=-1)

        ref_return_current_scores = top_k_result.values
        ref_return_next_expand_indices = torch.tensor(
            [[1, 2, 3, -1, -1, -1, -1], [1, 2, 3, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_output_all_layers_scores = top_k_result.values
        ref_return_output_all_layers_draft_token_ids = top_k_result.indices
        ref_return_output_all_layers_draft_token_ids_predecessor = torch.tensor(
            [[0, 0, 0], [0, 0, 0]], dtype=torch.int32, device="cuda"
        )  # Actual shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
        # Since we will save this value continuously

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 2: test the internal layer ##########################
        # BS=1, topK sampling
        # In this case, new selected draft tokens comes from node_2, node_2, and node_3
        # 3 input logits, from node node_1, node_2 and node_3, respectively
        # layerId = 1
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 1
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 1

        logits = torch.tensor(
            [
                [-10, 14, 13, -10, -10, -10, -10, -10, 15, -10, -10, -10
                 ],  # Top3 id = 8, 1, 2
                [-10, -10, 10, 11, -10, -10, 12, -10, -10, -10, -10, -10
                 ],  # Top3 id = 6, 3, 2
                [-10, 16, -10, -10, 17, -10, -10, 18, -10, -10, -10, -10
                 ],  # Top3 id = 7, 4, 1
            ],
            dtype=logits_data_type,
            device="cuda"
        )  # shape: [batch_size * dynamic_tree_max_topK, vocab_size_padded]

        num_last_token_indices = torch.tensor([3],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([3], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.tensor(
            [[1.1, 5.2, 3.3, -1, -1, -1, -1]],
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.tensor(
            [
                # batchIdx = 0
                [
                    [1.1, 5.2, 3.3] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.tensor(
            [
                # batchIdx = 0
                [
                    [6, 3, 2] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.tensor(
            [
                # batchIdx = 0
                [
                    [0, 0, 0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_output_path = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, 4, -1], [0, 2, 5, -1], [0, 3, 6, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        ref_return_draft_token_ids, ref_return_current_scores, ref_return_next_expand_indices, \
        ref_return_output_all_layers_scores, ref_return_output_all_layers_draft_token_ids, ref_return_output_all_layers_draft_token_ids_predecessor \
            = generate_ref_eagle2(
            layerIdx = layerId,
            batch_size = batch_size,
            input_logits = logits,
            dynamic_tree_max_topK = dynamic_tree_max_topK_t,
            input_prev_paths = paths,
            input_prev_scores = input_prev_scores,
            input_draft_token_ids = input_draft_token_ids,
            input_all_layers_scores = input_all_layers_scores,
            input_all_layers_draft_token_ids = input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor = input_all_layers_draft_token_ids_predecessor
        )

        ref_return_draft_len = torch.tensor(
            [6], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 3: test the internal layer ##########################
        # BS=2, topK sampling
        # For bs=0, the new expand nodes are from node_1, node_2, and node_3, respectively
        # For bs=1, the new expand nodes are all from node_1
        # 6 input logits, 3 from bs0, and 3 from bs1. And for each request, these 3 logits are from node_1, node_2 and node_3, respectively
        # layerId = 1
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 2
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 1

        logits = torch.tensor(
            [
                [-10, 14, 13, -10, -10, -10, -10, -10, 15, -10, -10, -10
                 ],  # Top3 id = 8, 1, 2
                [-10, -10, 10, 11, -10, -10, 12, -10, -10, -10, -10, -10
                 ],  # Top3 id = 6, 3, 2
                [-10, 16, -10, -10, 17, -10, -10, 18, -10, -10, -10, -10
                 ],  # Top3 id = 7, 4, 1
                [-10, 26, -10, 27, 28, -10, -10, -10, -10, -10, -10, -10
                 ],  # Top3 id = 4, 3, 1
                [-10, 24, 23, -10, -10, 25, -10, -10, -10, -10, -10, -10
                 ],  # Top3 id = 5, 1, 2
                [-10, -10, 20, 21, -10, -10, -10, -10, -10, -10, 22, -10
                 ],  # Top3 id = 10, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda"
        )  # shape: [batch_size * dynamic_tree_max_topK, vocab_size_padded]

        num_last_token_indices = torch.tensor([6],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]],
             [[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1], [5, 1, 2, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([3, 3],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.tensor(
            [[1.0, 1.0, 1.0, -1, -1, -1, -1], [14.4, 5.5, 6.6, -1, -1, -1, -1]],
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[1, 2, 3, -1, -1, -1, -1], [1, 2, 3, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.tensor(
            [
                # batchIdx = 0
                [
                    [1.0, 1.0, 1.0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ],
                # batchIdx = 1
                [
                    [14.4, 5.5, 6.6] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.tensor(
            [
                # batchIdx = 0
                [
                    [6, 3, 2] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ],

                # batchIdx = 1
                [
                    [5, 1, 2] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.tensor(
            [
                # batchIdx = 0
                [
                    [0, 0, 0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ],

                # batchIdx = 1
                [
                    [0, 0, 0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_output_path = torch.tensor(
            [
                [
                    [0, 1, 4, -1], [0, 2, 5, -1], [0, 3, 6, -1],
                    [-1, -1, -1, -1
                     ], [-1, -1, -1, -1
                         ], [-1, -1, -1, -1
                             ], [-1, -1, -1, -1
                                 ], [-1, -1, -1, -1]
                ],  # the new expand nodes are from node_1, node_2, and node_3, respectively
                [[0, 1, 4, -1], [0, 1, 5, -1], [0, 1, 6, -1], [0, 2, -1, -1],
                 [0, 3, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
                 [-1, -1, -1, -1]]
            ],  # the new expand nodes are all from node_1
            dtype=torch.int32,
            device="cuda")

        ref_return_draft_token_ids, ref_return_current_scores, ref_return_next_expand_indices, \
        ref_return_output_all_layers_scores, ref_return_output_all_layers_draft_token_ids, ref_return_output_all_layers_draft_token_ids_predecessor \
            = generate_ref_eagle2(
            layerIdx = layerId,
            batch_size = batch_size,
            input_logits = logits,
            dynamic_tree_max_topK = dynamic_tree_max_topK_t,
            input_prev_paths = paths,
            input_prev_scores = input_prev_scores,
            input_draft_token_ids = input_draft_token_ids,
            input_all_layers_scores = input_all_layers_scores,
            input_all_layers_draft_token_ids = input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor = input_all_layers_draft_token_ids_predecessor
        )

        ref_return_draft_len = torch.tensor(
            [6, 6], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 4: test the internal layer ##########################
        # In this test, in the second sampling, node 1 will has 2 leaves, and node 3 will has 1 leaf
        # The input path is:
        # [
        #   [0, 1, -1, -1],
        #   [0, 2, -1, -1],
        #   [0, 3, -1, -1]
        #   [-1, -1, -1, -1],
        #   ...
        # ]
        # The output path is:
        # [
        #   [0, 1, 4, -1],
        #   [0, 1, 5, -1],
        #   [0, 2, -1, -1],
        #   [0, 3, 6, -1],
        #   [-1, -1, -1, -1],
        #   ...
        # ]

        # BS=1, topK sampling
        # 3 input logits, from node_1, node_2, and node_3, respectively.
        # layerId = 1
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 1
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 1

        logits = torch.tensor(
            [
                [-1, 11.9, 7, -1, -1, -1, -1, -1, 12, -1, -1, -1
                 ],  # Top3 id = 8, 1, 2
                [-1, -1, 19.4, 19.5, -1, -1, 20, -1, -1, -1, -1, -1
                 ],  # Top3 id = 6, 3, 2
                [-1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1, -1
                 ],  # Top3 id = 7, 4, 1
            ],
            dtype=logits_data_type,
            device="cuda"
        )  # shape: [batch_size * dynamic_tree_max_topK, vocab_size_padded]

        num_last_token_indices = torch.tensor([3],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([3], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.tensor(
            [[1.0, 1.0, 1.0, -1, -1, -1, -1]],
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.tensor(
            [
                # batchIdx = 0
                [
                    [1.0, 1.0, 1.0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.tensor(
            [
                # batchIdx = 0
                [
                    [6, 3, 2] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.tensor(
            [
                # batchIdx = 0
                [
                    [0, 0, 0] + [-1] *
                    (max_decoding_draft_tokens * max_decoding_draft_tokens -
                     dynamic_tree_max_topK_t),  # layerIdx = 0
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 1
                    [-1] * (max_decoding_draft_tokens *
                            max_decoding_draft_tokens),  # layerIdx = 2
                ]
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_output_path = torch.tensor(
            [[[0, 1, 4, -1], [0, 1, 5, -1], [0, 2, -1, -1], [0, 3, 6, -1],
              [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda")

        ref_return_draft_token_ids, ref_return_current_scores, ref_return_next_expand_indices, \
        ref_return_output_all_layers_scores, ref_return_output_all_layers_draft_token_ids, ref_return_output_all_layers_draft_token_ids_predecessor \
            = generate_ref_eagle2(
            layerIdx = layerId,
            batch_size = batch_size,
            input_logits = logits,
            dynamic_tree_max_topK = dynamic_tree_max_topK_t,
            input_prev_paths = paths,
            input_prev_scores = input_prev_scores,
            input_draft_token_ids = input_draft_token_ids,
            input_all_layers_scores = input_all_layers_scores,
            input_all_layers_draft_token_ids = input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor = input_all_layers_draft_token_ids_predecessor
        )

        ref_return_draft_len = torch.tensor(
            [6], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 5: test the last layer ##########################
        # BS=1, topK sampling
        # 3 input logits
        # layerId = 2, which is the last layer
        # logits_data_type = float32

        # The input paths
        # [
        #   [0, 1, 4, -1],
        #   [0, 1, 5, -1],
        #   [0, 1, 6, -1],
        #   [0, 2, -1, -1],
        #   [0, 3, -1, -1],
        #   [-1, -1, -1, -1]
        # ]
        # Three input logits are from node_4, node_5, and node_6
        # We set the node_1 to node_6 have large scores, so they will be selected in the final tree

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 1
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 2

        logits = torch.tensor(
            [
                [-1, 11.9, 7, -1, -1, -1, -1, -1, 12, -1, -1, -1
                 ],  # Top3 id = 8, 1, 2
                [-1, -1, 19.4, 19.5, -1, -1, 20, -1, -1, -1, -1, -1
                 ],  # Top3 id = 6, 3, 2
                [-1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1, -1
                 ],  # Top3 id = 7, 4, 1
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([3],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[
                [0, 1, 4, -1],
                [0, 1, 5, -1],
                [0, 1, 6, -1],
                [0, 2, -1, -1],
                [0, 3, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([6], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.tensor(
            [[10, 10, 10, -1, -1, -1, -1]], dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[4, 5, 6, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.tensor(
            [[
                [10, 10, 10, 10, 10, 10, -10, -10, -10, -10, -10, -10] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ]],
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.tensor(
            [[
                [1, 2, 3, 4, 5, 6, 11, 11, 11, 11, 11, 11] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.tensor(
            [[
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [7], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        ref_return_output_path = torch.tensor(
            [[[0, 1, 4, -1], [0, 1, 5, -1], [0, 1, 6, 7], [0, 2, -1, -1],
              [0, 3, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        # For the last layer, we do not need to check these outputs
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 6: test the last layer ##########################
        # batch_size = 2
        logits_data_type = torch.float32
        max_decoding_draft_tokens = 7
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 4
        num_eagle_layers = 3
        batch_size = 2
        dynamic_tree_max_topK_t = 3
        top_k_sampling = True
        layerId = 2

        logits = torch.tensor(
            [
                [-1, 11.9, 7, -1, -1, -1, -1, -1, 12, -1, -1, -1
                 ],  # Top3 id = 8, 1, 2
                [-1, -1, 19.4, 19.5, -1, -1, 20, -1, -1, -1, -1, -1
                 ],  # Top3 id = 6, 3, 2
                [-1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1, -1
                 ],  # Top3 id = 7, 4, 1
                [-1, 11.9, -1, 7, 12, -1, -1, -1, -1, -1, -1, -1
                 ],  # Top3 id = 4, 3, 1
                [-1, 19.5, 19.4, -1, -1, 20, -1, -1, -1, -1, -1, -1
                 ],  # Top3 id = 5, 1, 2
                [-1, -1, 3, 4, -1, -1, -1, -1, -1, -1, 5, -1
                 ],  # Top3 id = 10, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([6],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[
                [0, 1, 4, -1],
                [0, 1, 5, -1],
                [0, 1, 6, -1],
                [0, 2, -1, -1],
                [0, 3, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
             [
                 [0, 1, 4, -1],
                 [0, 1, 5, -1],
                 [0, 2, 6, -1],
                 [0, 3, -1, -1],
                 [-1, -1, -1, -1],
                 [-1, -1, -1, -1],
                 [-1, -1, -1, -1],
                 [-1, -1, -1, -1],
             ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, -1], [6, 5, 4, 3, 2, 1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([6, 6],
                                        dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.tensor(
            [[10, 10, 10, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1]],
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        # The index will take all the draft tokens into consideration, even they are not selected.
        # But they will be sampled at the last layer.
        # As to the bi=0, the index '8' is actually correspond to the '6' in the input paths.
        input_current_expand_indices = torch.tensor(
            [[4, 5, 6, -1, -1, -1, -1], [4, 5, 8, -1, -1, -1, -1]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.tensor(
            [[
                [10, 10, 10, 10, 10, 10, -10, -10, -10, -10, -10, -10] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ],
             [
                 [16, 15, 14, 13, 12, -10, -10, 11, -10, -10, -10, -10] + [-1] *
                 (max_decoding_draft_tokens * max_decoding_draft_tokens -
                  dynamic_tree_max_topK_t -
                  dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
             ]],
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.tensor(
            [[
                [1, 2, 3, 4, 5, 6, 11, 11, 11, 11, 11, 11] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ],
             [
                 [6, 5, 4, 3, 2, 11, 11, 1, 11, 11, 11, 11] + [-1] *
                 (max_decoding_draft_tokens * max_decoding_draft_tokens -
                  dynamic_tree_max_topK_t -
                  dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
             ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.tensor(
            [[
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] + [-1] *
                (max_decoding_draft_tokens * max_decoding_draft_tokens -
                 dynamic_tree_max_topK_t -
                 dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
            ],
             [
                 [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] + [-1] *
                 (max_decoding_draft_tokens * max_decoding_draft_tokens -
                  dynamic_tree_max_topK_t -
                  dynamic_tree_max_topK_t * dynamic_tree_max_topK_t),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
                 [-1] * (max_decoding_draft_tokens * max_decoding_draft_tokens),
             ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7], [6, 5, 4, 3, 2, 1, 10]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [7, 7], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        ref_return_output_path = torch.tensor(
            [
                [[0, 1, 4, -1], [0, 1, 5, -1], [0, 1, 6, 7], [0, 2, -1, -1],
                 [0, 3, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
                 [-1, -1, -1, -1]],
                [[0, 1, 4, -1], [0, 1, 5, -1], [0, 2, 6, 7], [0, 3, -1, -1],
                 [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1],
                 [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        # For the last layer, we do not need to check these outputs
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]

        ################# CASE 7: test the fist, but also the last layer ##########################
        # BS=1, topK sampling
        # 1 input logits
        # layerId = 0, which is the first layer, but also the last layer
        # logits_data_type = float32

        logits_data_type = torch.float32
        max_decoding_draft_tokens = 4
        max_decoding_tokens = max_decoding_draft_tokens + 1
        max_path_len = 2
        num_eagle_layers = 1
        batch_size = 1
        dynamic_tree_max_topK_t = 4
        top_k_sampling = True
        layerId = 0

        logits = torch.tensor(
            [
                [-1, -1, 2, -1, -1, 5, -1, 4, 3, -1, -1, -1
                 ],  # Top4 id = 5, 7, 8, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        num_last_token_indices = torch.tensor([1],
                                              dtype=torch.int32,
                                              device="cuda")  # shape: [1]

        paths = torch.tensor(
            [[
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        use_dynamic_tree = torch.tensor([1], dtype=torch.int32,
                                        device="cpu")  # shape: [1]
        dynamic_tree_max_topK = torch.tensor(dynamic_tree_max_topK_t,
                                             dtype=torch.int32,
                                             device="cpu")  # shape: [1]
        input_draft_token_ids = torch.tensor(
            [[-1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]
        input_draft_lens = torch.tensor([0], dtype=torch.int32,
                                        device="cuda")  # shape: [batch_size]
        input_prev_scores = torch.full(
            (batch_size, max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_current_expand_indices = torch.tensor(
            [[0, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        input_all_layers_scores = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            float('-inf'),
            dtype=torch.float32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        input_all_layers_draft_token_ids_predecessor = torch.full(
            (batch_size, num_eagle_layers,
             max_decoding_draft_tokens * max_decoding_draft_tokens),
            -1,
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]

        ref_return_draft_token_ids = torch.tensor(
            [[5, 7, 8, 2]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [4], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        ref_return_output_path = torch.tensor(
            [[[0, 1, -1, -1], [0, 2, -1, -1], [0, 3, -1, -1], [0, 4, -1, -1],
              [-1, -1, -1, -1]]],
            dtype=torch.int32,
            device="cuda"
        )  # shape: [batch_size, max_decoding_tokens, max_path_len]

        # For the last layer, we do not need to check these outputs
        ref_return_current_scores = None
        ref_return_next_expand_indices = None
        ref_return_output_all_layers_scores = None
        ref_return_output_all_layers_draft_token_ids = None
        ref_return_output_all_layers_draft_token_ids_predecessor = None

        test_cases += [[
            logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor
        ]]
        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_sample_draft_tokens_plugin(
            self, logits, num_last_token_indices, paths, use_dynamic_tree,
            dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
            input_prev_scores, input_current_expand_indices,
            input_all_layers_scores, input_all_layers_draft_token_ids,
            input_all_layers_draft_token_ids_predecessor, top_k_sampling,
            num_eagle_layers, layerId, ref_return_draft_token_ids,
            ref_return_draft_len, ref_return_output_path,
            ref_return_current_scores, ref_return_next_expand_indices,
            ref_return_output_all_layers_scores,
            ref_return_output_all_layers_draft_token_ids,
            ref_return_output_all_layers_draft_token_ids_predecessor):

        # test data
        torch.get_default_device()
        torch.set_default_device("cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            logits_t = Tensor(name='logits',
                              dtype=tensorrt_llm.torch_dtype_to_trt(
                                  logits.dtype),
                              shape=logits.shape)
            num_last_token_indices_t = Tensor(
                name='num_last_token_indices',
                dtype=tensorrt_llm.torch_dtype_to_trt(
                    num_last_token_indices.dtype),
                shape=num_last_token_indices.shape)
            paths_t = Tensor(name='paths', dtype=trt.int32, shape=paths.shape)
            use_dynamic_tree_t = Tensor(name='use_dynamic_tree',
                                        dtype=trt.int32,
                                        shape=use_dynamic_tree.shape)
            dynamic_tree_max_topK_t = Tensor(name='dynamic_tree_max_topK',
                                             dtype=trt.int32,
                                             shape=dynamic_tree_max_topK.shape)
            input_draft_token_ids_t = Tensor(name='input_draft_token_ids',
                                             dtype=trt.int32,
                                             shape=input_draft_token_ids.shape)
            input_draft_lens_t = Tensor(name='input_draft_lens',
                                        dtype=trt.int32,
                                        shape=input_draft_lens.shape)
            input_prev_scores_t = Tensor(name='input_prev_scores',
                                         dtype=trt.float32,
                                         shape=input_prev_scores.shape)
            input_current_expand_indices_t = Tensor(
                name='input_current_expand_indices',
                dtype=trt.int32,
                shape=input_current_expand_indices.shape)
            input_all_layers_scores_t = Tensor(
                name='input_all_layers_scores',
                dtype=tensorrt_llm.torch_dtype_to_trt(
                    input_all_layers_scores.dtype),
                shape=input_all_layers_scores.shape)
            input_all_layers_draft_token_ids_t = Tensor(
                name='input_all_layers_draft_token_ids',
                dtype=trt.int32,
                shape=input_all_layers_draft_token_ids.shape)
            input_all_layers_draft_token_ids_predecessor_t = Tensor(
                name='input_all_layers_draft_token_ids_predecessor',
                dtype=trt.int32,
                shape=input_all_layers_draft_token_ids_predecessor.shape)

            output = tensorrt_llm.models.eagle.model.eagle_draft_decoder_plugin(
                layer_idx=layerId,
                num_eagle_layers=num_eagle_layers,
                top_k_sampling=top_k_sampling,
                logits=logits_t,
                num_last_token_indices=num_last_token_indices_t,
                input_paths=paths_t,
                use_dynamic_tree=use_dynamic_tree_t,
                dynamic_tree_max_topK=dynamic_tree_max_topK_t,
                input_draft_token_ids=input_draft_token_ids_t,
                input_draft_lens=input_draft_lens_t,
                input_prev_scores=input_prev_scores_t,
                input_current_expand_indices=input_current_expand_indices_t,
                input_all_layers_scores=input_all_layers_scores_t,
                input_all_layers_draft_token_ids=
                input_all_layers_draft_token_ids_t,
                input_all_layers_draft_token_ids_predecessor=
                input_all_layers_draft_token_ids_predecessor_t)

            output_draft_token_ids, output_draft_lens, output_paths, output_current_scores, output_next_expand_indices, \
                output_all_layers_scores, output_all_layers_draft_token_ids, output_all_layers_draft_token_ids_predecessor = output

            output_draft_token_ids.mark_output('output_draft_token_ids')
            output_draft_lens.mark_output('output_draft_lens')
            output_paths.mark_output('output_paths')
            output_current_scores.mark_output('output_current_scores')
            output_next_expand_indices.mark_output('output_next_expand_indices')
            output_all_layers_scores.mark_output('output_all_layers_scores')
            output_all_layers_draft_token_ids.mark_output(
                'output_all_layers_draft_token_ids')
            output_all_layers_draft_token_ids_predecessor.mark_output(
                'output_all_layers_draft_token_ids_predecessor')

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "logits":
            logits,
            "num_last_token_indices":
            num_last_token_indices,
            "paths":
            paths,
            "use_dynamic_tree":
            use_dynamic_tree,
            "dynamic_tree_max_topK":
            dynamic_tree_max_topK,
            "input_draft_token_ids":
            input_draft_token_ids,
            "input_draft_lens":
            input_draft_lens,
            "input_prev_scores":
            input_prev_scores,
            "input_current_expand_indices":
            input_current_expand_indices,
            "input_all_layers_scores":
            input_all_layers_scores,
            "input_all_layers_draft_token_ids":
            input_all_layers_draft_token_ids,
            "input_all_layers_draft_token_ids_predecessor":
            input_all_layers_draft_token_ids_predecessor
        }
        outputs = run_session(session, inputs)

        output_draft_token_ids = outputs['output_draft_token_ids']
        output_draft_lens = outputs['output_draft_lens']
        output_paths = outputs['output_paths']
        output_current_scores = outputs['output_current_scores']
        output_next_expand_indices = outputs['output_next_expand_indices']
        output_all_layers_scores = outputs['output_all_layers_scores']
        output_all_layers_draft_token_ids = outputs[
            'output_all_layers_draft_token_ids']
        output_all_layers_draft_token_ids_predecessor = outputs[
            'output_all_layers_draft_token_ids_predecessor']

        # Check output
        batch_size = paths.shape[0]
        for bix in range(batch_size):
            # 1) Check output length
            self.assertEqual(ref_return_draft_len[bix], output_draft_lens[bix])

            # 2) Check output token
            for jj in range(output_draft_lens[bix]):
                self.assertEqual(ref_return_draft_token_ids[bix][jj],
                                 output_draft_token_ids[bix][jj])

            # 3) Check output paths
            max_decoding_tokens = output_paths.shape[1]
            output_paths.shape[2]
            for jj in range(max_decoding_tokens):
                for kk in range(layerId + 1):  # '+1' is because the root node
                    self.assertEqual(ref_return_output_path[bix][jj][kk],
                                     output_paths[bix][jj][kk])

            # For eagle-2
            if use_dynamic_tree:
                num_all_layers_draft_tokens = (
                    layerId - 1
                ) * dynamic_tree_max_topK * dynamic_tree_max_topK + dynamic_tree_max_topK

                if layerId != num_eagle_layers - 1:
                    # Only check these output for internal layers
                    # 4) Check output current scores, check shape: [batch_size, dynamic_tree_max_topK]
                    for jj in range(dynamic_tree_max_topK):
                        self.assertAlmostEqual(
                            ref_return_current_scores[bix][jj],
                            output_current_scores[bix][jj],
                            delta=0.1)

                    # 5) Check output next expand indices, check shape: [batch_size, dynamic_tree_max_topK]
                    for jj in range(dynamic_tree_max_topK):
                        self.assertEqual(
                            ref_return_next_expand_indices[bix][jj],
                            output_next_expand_indices[bix][jj])

                    # 6) Check output all layers scores
                    cur_output_all_layers_scores = output_all_layers_scores[
                        bix].view(-1)
                    for jj in range(num_all_layers_draft_tokens):
                        self.assertAlmostEqual(
                            ref_return_output_all_layers_scores[bix][jj],
                            cur_output_all_layers_scores[jj],
                            delta=0.1)

                    # 7) Check output all layers draft token ids
                    cur_output_all_layers_draft_token_ids = output_all_layers_draft_token_ids[
                        bix].view(-1)
                    for jj in range(num_all_layers_draft_tokens):
                        self.assertEqual(
                            ref_return_output_all_layers_draft_token_ids[bix]
                            [jj], cur_output_all_layers_draft_token_ids[jj])

                    # 8) Check output all layers draft token ids predecessor
                    cur_output_all_layers_draft_token_ids_predecessor = output_all_layers_draft_token_ids_predecessor[
                        bix].view(-1)
                    for jj in range(num_all_layers_draft_tokens):
                        self.assertEqual(
                            ref_return_output_all_layers_draft_token_ids_predecessor[
                                bix][jj],
                            cur_output_all_layers_draft_token_ids_predecessor[
                                jj])


if __name__ == "__main__":
    unittest.main()

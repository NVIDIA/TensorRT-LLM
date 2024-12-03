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
import os
import sys
import unittest

import tensorrt as trt
import torch
from parameterized import parameterized

import tensorrt_llm
import tensorrt_llm.models.eagle
from tensorrt_llm import Tensor
from tensorrt_llm.models.eagle.model import TreeParams

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from utils.util import create_session, run_session, unittest_name_func


class TestEagleDecodeDraftTokensPlugin(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def load_test_cases():
        test_cases = []

        ################# CASE 0 ##########################
        # BS=1, topK sampling
        # 1 input logits, from node "0"
        # layerId = 0
        # logits_data_type = float32

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 0, 1, -100, -100, 2, -100],  # Top3 id = 6, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 0
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        ################# CASE 1 ##########################
        # BS=2, topK sampling
        # 2 input logits, from req0 node "0" and req1 node "0"
        # layerId = 0
        # logits_data_type = float32

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 0, 1, -100, -100, -100, -100],  # Top2 id = 3, 2
                [-100, 3, -100, 2, -100, 1, -100, -100],  # Top3 id = 1, 3, 5
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0, 0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 0
        ref_return_draft_token_ids = torch.tensor(
            [[3, 2, -1, -1], [1, 3, 5, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [2, 3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        ################# CASE 2 ##########################
        # BS=1, topK sampling
        # 2 input loigts, from req0 node "1" and "3"
        # layerId = 1
        # logits_data_type = float32

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, -100, 1, -100, -100, -100, -100],  # Top1 id = 3
                [-100, 1, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0, 0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 1
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, 3, 1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [5], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        ################# CASE 3 ##########################
        # BS=2, topK sampling
        # 1 input loigts, from req1, node "3"
        # layerId = 1
        # logits_data_type = float32

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, -100, -100, -100, 1, -100, -100],  # Top1 id = 5
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 1
        ref_return_draft_token_ids = torch.tensor(
            [[2, 1, -1, -1], [1, 2, 3, 5]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [2, 4], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        ################# CASE 4 ##########################
        # BS=2, topK sampling
        # 2 input logits, from req0 node "4" and req1 node "4"
        # layerId = 2
        # logits_data_type = float32

        logits_data_type = torch.float32
        logits = torch.tensor(
            [
                [-100, -100, 0, 1, -100, -100, -100, -100],  # Top2 id = 3, 2
                [-100, -100, -100, -100, 0, 1, -100, -100],  # Top2 id = 5, 4
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0, 0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 2
        ref_return_draft_token_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 3, 2], [1, 2, 3, 4, 5, 5, 4]],
            dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [7, 7], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        ################# CASE 5 ##########################
        # BS=1, topK sampling
        # 1 input logits, from req0 node "0"
        # layerId = 0
        # logits_data_type = float16

        logits_data_type = torch.float16
        logits = torch.tensor(
            [
                [-100, -100, 0, 1, -100, -100, 2, -100],  # Top3 id = 6, 3, 2
            ],
            dtype=logits_data_type,
            device="cuda")  # shape: [num_tokens, vocab_size_padded]

        rand_sample = torch.tensor([0], dtype=torch.float32,
                                   device="cuda")  # shape: [num_tokens]

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

        topKSampling = True
        layerId = 0
        ref_return_draft_token_ids = torch.tensor(
            [[6, 3, 2, -1, -1, -1, -1]], dtype=torch.int32,
            device="cuda")  # shape: [batch_size, max_decoding_draft_tokens]

        ref_return_draft_len = torch.tensor(
            [3], dtype=torch.int32, device="cuda")  # shape: [batch_size]

        test_cases += [[
            logits, rand_sample, paths, input_draft_token_ids, input_draft_lens,
            topKSampling, layerId, ref_return_draft_token_ids,
            ref_return_draft_len
        ]]

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_sample_draft_tokens_plugin(self, logits, rand_sample, paths,
                                        input_draft_token_ids, input_draft_lens,
                                        topKSampling, layerId,
                                        ref_return_draft_token_ids,
                                        ref_return_draft_len):
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
            rand_sample_t = Tensor(name='rand_sample',
                                   dtype=trt.float32,
                                   shape=rand_sample.shape)
            paths_t = Tensor(name='paths', dtype=trt.int32, shape=paths.shape)
            input_draft_token_ids_t = Tensor(name='input_draft_token_ids',
                                             dtype=trt.int32,
                                             shape=input_draft_token_ids.shape)
            input_draft_lens_t = Tensor(name='input_draft_lens',
                                        dtype=trt.int32,
                                        shape=input_draft_lens.shape)

            output = tensorrt_llm.models.eagle.model.eagle_draft_decoder_plugin(
                layer_idx=layerId,
                top_k_sampling=topKSampling,
                logits=logits_t,
                rand_sample=rand_sample_t,
                tree_params=TreeParams(paths=paths_t),
                input_draft_token_ids=input_draft_token_ids_t,
                input_draft_lens=input_draft_lens_t)

            output_draft_token_ids, output_draft_lens = output

            output_draft_token_ids.mark_output('output_draft_token_ids')
            output_draft_lens.mark_output('output_draft_lens')

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "logits": logits,
            "rand_sample": rand_sample,
            "paths": paths,
            "input_draft_token_ids": input_draft_token_ids,
            "input_draft_lens": input_draft_lens
        }
        outputs = run_session(session, inputs)

        output_draft_token_ids = outputs['output_draft_token_ids']
        output_draft_lens = outputs['output_draft_lens']

        # Check output
        batch_size = paths.shape[0]
        for i in range(batch_size):
            # Check output length
            self.assertEqual(ref_return_draft_len[i], output_draft_lens[i])

            # Check output token
            for j in range(output_draft_lens[i]):
                self.assertEqual(ref_return_draft_token_ids[i][j],
                                 output_draft_token_ids[i][j])


if __name__ == "__main__":
    unittest.main()

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
from tensorrt_llm.models.eagle.model import TreeParams


class TestEagleSampleAcceptDraftTokensPlugin(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    def load_test_cases():
        test_cases = []

        logits = torch.tensor(
            [
                [0, -100, -100, -100, -100, -100, -100, -100
                 ],  # t0: Top1 id = 0
                [-100, 0, -100, -100, -100, -100, -100, -100
                 ],  # t1: Top1 id = 1
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t2: Top1 id = 2
                [-100, -100, -100, 0, -100, -100, -100, -100
                 ],  # t3: Top1 id = 3
                [-100, -100, -100, 0, -100, -100, -100, -100
                 ],  # t4: Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t5: Top1 id = 2
                [-100, 0, -100, -100, -100, -100, -100, -100
                 ],  # t6: Top1 id = 1
                [0, -100, -100, -100, -100, -100, -100, -100]  # t7: Top1 id = 0
            ],
            dtype=torch.float32,
            device="cuda")
        ################# CASE 0 ##########################
        # BS=1, greedy sampling, gen request
        # 7 draft tokens
        draft_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor([[0.0 for _ in range(8)]],
                                            dtype=torch.float,
                                            device="cuda")
        posterior_alpha = torch.tensor([0.1], dtype=torch.float, device="cuda")
        posterior_threshold = torch.tensor([0.1],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, 6],  # Draft seq [0, 3, 5],  Target seq [0, 1, 3, 1]
                [0, 1, 4, 7],  # Draft seq [0, 3, 6],  Target seq [0, 1, 3, 0]
                [0, 2, -1, -1],  # Draft seq [1],        Target seq [0, 2]
                [0, 3, 5, -1],  # Draft seq [2, 4],     Target seq [0, 3, 2]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([1], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[0, 1]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 1 ##########################
        # BS=1, greedy sampling, gen request
        # 7 draft tokens
        logits = torch.tensor(
            [
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t0: Top1 id = 0
                [-100, 0, -100, -100, -100, -100, -100, -100
                 ],  # t1: Top1 id = 1
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t2: Top1 id = 2
                [-100, -100, -100, -100, -100, 0, -100, -100
                 ],  # t3: Top1 id = 5
                [-100, -100, -100, 0, -100, -100, -100, -100
                 ],  # t4: Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t5: Top1 id = 2
                [-100, -100, -100, -100, -100, -100, 0, -100
                 ],  # t6: Top1 id = 6
                [-100, -100, -100, -100, -100, -100, -100, 0]  # t7: Top1 id = 7
            ],
            dtype=torch.float32,
            device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, -1],  # Draft seq [0, 3],  Target seq [2, 1, 3]
                [0, 2, 5, -1],  # Draft seq [1, 4],  Target seq [2, 2, 2]
                [0, 3, 6, 7],  # Draft seq [2, 5, 6],  Target seq [2, 5, 6, 7]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([1], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[2, 5, 6, 7]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([4],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([2], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 2 ##########################
        # BS=2, greedy sampling, gen request
        # 3 draft tokens
        logits = torch.tensor(
            [
                [0, -100, -100, -100, -100, -100, -100, -100
                 ],  # t0: Top1 id = 0
                [-100, 0, -100, -100, -100, -100, -100, -100
                 ],  # t1: Top1 id = 1
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t2: Top1 id = 2
                [-100, -100, -100, 0, -100, -100, -100, -100
                 ],  # t3: Top1 id = 3
                [-100, -100, -100, 0, -100, -100, -100, -100
                 ],  # t4: Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100
                 ],  # t5: Top1 id = 2
                [-100, 0, -100, -100, -100, -100, -100, -100
                 ],  # t6: Top1 id = 1
                [0, -100, -100, -100, -100, -100, -100, -100]  # t7: Top1 id = 0
            ],
            dtype=torch.float32,
            device="cuda")
        draft_tokens = torch.tensor([[0, 1, -1, -1], [2, 3, 4, 5]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.0, 0.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.0 for _ in range(5)], [0.0 for _ in range(5)]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.1, 0.1],
                                       dtype=torch.float,
                                       device="cuda")
        posterior_threshold = torch.tensor([0.1, 0.1],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [
                [
                    [0, 1, -1, -1],  # Draft seq [0],  Target seq [0, 1]
                    [0, 2, -1, -1],  # Draft seq [1],  Target seq [0, 2]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ],
                [
                    [0, 1, -1, -1],  # Draft seq [2],    Target seq [3, 3]
                    [0, 2, -1, -1],  # Draft seq [3],    Target seq [3, 2]
                    [0, 3, 4, -1],  # Draft seq [4, 5], Target seq [3, 1, 0]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ]
            ],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([1], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[0, 1, -1, -1], [3, 2, -1, -1]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2, 2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0, 1],
                                          dtype=torch.int32,
                                          device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 3 ##########################
        # BS=2, greedy sampling, 2 ctx request, 1 gen request
        draft_tokens = torch.tensor(
            [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 1, 2, 3, 4]],
            dtype=torch.int32,
            device="cuda")
        draft_lens = torch.tensor([0, 0, 5], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.0, 0.0, 0.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.0
              for _ in range(6)], [0.0
                                   for _ in range(6)], [0.0 for _ in range(6)]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.1, 0.1, 0.1],
                                       dtype=torch.float,
                                       device="cuda")
        posterior_threshold = torch.tensor([0.1, 0.1, 0.1],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [
                [
                    [0, 1, 2, -1],  # Draft seq [],  Target seq [0]
                    [0, 1, 3, -1],  # Draft seq [],  Target seq [0]
                    [0, 1, 4, -1],  # Draft seq [],  Target seq [0]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ],
                [
                    [0, 1, 2, -1],  # Draft seq [],  Target seq [1]
                    [0, 1, 3, -1],  # Draft seq [],  Target seq [1]
                    [0, 1, 4, -1],  # Draft seq [],  Target seq [1]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ],
                [
                    [0, 1, -1, -1],  # Draft seq [0],       Target seq [2, 3]
                    [0, 2, -1, -1],  # Draft seq [1],       Target seq [2, 3]
                    [0, 3, 4, 5
                     ],  # Draft seq [2, 3, 4], Target seq [2, 2, 1, 0]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ]
            ],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([1], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor(
            [[0, -1, -1, -1], [1, -1, -1, -1], [2, 2, -1, -1]],
            dtype=torch.int32,
            device="cuda")
        ref_num_accepted_tokens = torch.tensor([1, 1, 2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0, 0, 2],
                                          dtype=torch.int32,
                                          device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 4 ##########################
        # BS=1, typical sampling, temperature = 0.f -- same as greedy sampling, gen request
        # 7 draft tokens
        draft_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor([[0.0 for _ in range(8)]],
                                            dtype=torch.float,
                                            device="cuda")
        posterior_alpha = torch.tensor([0.1], dtype=torch.float, device="cuda")
        posterior_threshold = torch.tensor([0.1],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, 6],  # Draft seq [0, 3, 5],  Target seq [0, 1, 3, 1]
                [0, 1, 4, 7],  # Draft seq [0, 3, 6],  Target seq [0, 1, 3, 0]
                [0, 2, -1, -1],  # Draft seq [1],        Target seq [0, 2]
                [0, 3, 5, -1],  # Draft seq [2, 4],     Target seq [0, 3, 2]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([0], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[0, 1]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 5 ##########################
        # BS=1, typical sampling, temperature = 1.f, gen request
        # Very conservative posterior_threshold
        # 7 draft tokens
        logits = torch.tensor(
            [
                [-0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9, -1e9, -1e9
                 ],  # t0: Top4 id = 0, 1, 2, 3
                [-1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9, -1e9
                 ],  # t1: Top4 id = 1, 2, 3,4
                [-1e9, -1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9
                 ],  # t2: Top5 id = 2, 3, 4, 5
                [-1e9, -1e9, -1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9
                 ],  # t3: Top4 id = 3, 4, 5, 6
                [-1e9, -1e9, -1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9
                 ],  # t4: Top4 id = 3, 4, 5, 6
                [-1e9, -1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9
                 ],  # t5: Top4 id = 2, 3, 4, 5
                [-1e9, -0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9, -1e9
                 ],  # t6: Top4 id = 2, 3, 4, 5
                [-0.9163, -1.2040, -1.6094, -2.3026, -1e9, -1e9, -1e9, -1e9
                 ]  # t7: Top1 id = 0, 1, 2, 3
            ],
            dtype=torch.float32,
            device="cuda")

        draft_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([1.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.1, 0.8, 0.5, 0.5, 0.1, 0.2, 0.8, 0.1]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.9999],
                                       dtype=torch.float,
                                       device="cuda")
        posterior_threshold = torch.tensor([0.99],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, 6],  # Draft seq [0, 3, 5],  Target seq [0, 2, 3, 1]
                [0, 1, 4, 7],  # Draft seq [0, 3, 6],  Target seq [0, 2, 3, 2]
                [0, 2, -1, -1],  # Draft seq [1],        Target seq [0, 3]
                [0, 3, 5, -1],  # Draft seq [2, 4],     Target seq [0, 4, 2]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([0], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[0, 1]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 6 ##########################
        # BS=1, typical sampling, temperature = 1.f, gen request
        # Small posterior_threshold
        # 7 draft tokens

        draft_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([1.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.1, 0.8, 0.5, 0.5, 0.1, 0.2, 0.8, 0.1]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.3], dtype=torch.float, device="cuda")
        posterior_threshold = torch.tensor([0.09],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, 6],  # Draft seq [0, 3, 5],  Target seq [0, 2, 3, 1]
                [0, 1, 4, 7],  # Draft seq [0, 3, 6],  Target seq [0, 2, 3, 2]
                [0, 2, -1, -1],  # Draft seq [1],        Target seq [0, 3]
                [0, 3, 5, -1],  # Draft seq [2, 4],     Target seq [0, 4, 2]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([0], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[0, 2]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([0], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################# CASE 7 ##########################
        # BS=1, typical sampling, temperature = 0.8f, gen request
        # Small posterior_threshold
        # 7 draft tokens

        draft_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.8],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.95, 0.8, 0.5, 0.7, 0.1, 0.2, 0.8, 0.1]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.3], dtype=torch.float, device="cuda")
        posterior_threshold = torch.tensor([0.09],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [[
                [0, 1, 4, 6],  # Draft seq [0, 3, 5],  Target seq [0, 2, 3, 1]
                [0, 1, 4, 7],  # Draft seq [0, 3, 6],  Target seq [0, 2, 3, 2]
                [0, 2, -1, -1],  # Draft seq [1],        Target seq [0, 3]
                [0, 3, 5, -1],  # Draft seq [2, 4],     Target seq [2, 4, 2]
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ]],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([0], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")
        ref_accepted_tokens = torch.tensor([[2, 4, 2]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([3],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([3], dtype=torch.int32, device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        ################ CASE 8 ##########################
        # BS=2, typical sampling, gen request
        draft_tokens = torch.tensor([[0, 1, -1, -1], [2, 3, 4, 5]],
                                    dtype=torch.int32,
                                    device="cuda")
        draft_lens = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
        eagle_temperature = torch.tensor([0.8, 1.0],
                                         dtype=torch.float,
                                         device="cuda")
        rand_data_validation = torch.tensor(
            [[0.8, 0.1, 0.95, 0.0, 0.0], [0.5, 0.1, 0.5, 0.2, 0.7]],
            dtype=torch.float,
            device="cuda")
        posterior_alpha = torch.tensor([0.3, 0.999],
                                       dtype=torch.float,
                                       device="cuda")
        posterior_threshold = torch.tensor([0.09, 0.99],
                                           dtype=torch.float,
                                           device="cuda")
        paths = torch.tensor(
            [
                [
                    [0, 1, -1, -1],  # Draft seq [0],  Target seq [1, 1]
                    [0, 2, -1, -1],  # Draft seq [1],  Target seq [1, 4]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ],
                [
                    [0, 1, -1, -1],  # Draft seq [2],    Target seq [3, 3]
                    [0, 2, -1, -1],  # Draft seq [3],    Target seq [3, 2]
                    [0, 3, 4, -1],  # Draft seq [4, 5], Target seq [3, 1, 0]
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1]
                ]
            ],
            dtype=torch.int32,
            device="cuda")
        greedy_sampling = torch.tensor([0], dtype=torch.int32, device="cpu")
        use_dynamic_tree = torch.tensor([0], dtype=torch.int32, device="cpu")

        ref_accepted_tokens = torch.tensor([[1, 4, -1, -1], [3, 2, -1, -1]],
                                           dtype=torch.int32,
                                           device="cuda")
        ref_num_accepted_tokens = torch.tensor([2, 2],
                                               dtype=torch.int32,
                                               device="cuda")
        ref_accepted_paths = torch.tensor([1, 1],
                                          dtype=torch.int32,
                                          device="cuda")

        test_cases += [[
            logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths
        ]]

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_sample_accept_draft_tokens_plugin(
            self, logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold, paths,
            greedy_sampling, use_dynamic_tree, ref_accepted_tokens,
            ref_num_accepted_tokens, ref_accepted_paths):
        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            logits_t = Tensor(name='logits',
                              dtype=tensorrt_llm.torch_dtype_to_trt(
                                  logits.dtype),
                              shape=logits.shape)
            draft_tokens_t = Tensor(name='draft_tokens',
                                    dtype=trt.int32,
                                    shape=draft_tokens.shape)
            draft_lens_t = Tensor(name='draft_lens',
                                  dtype=trt.int32,
                                  shape=draft_lens.shape)
            eagle_temperature_t = Tensor(name='eagle_temperature',
                                         dtype=trt.float32,
                                         shape=eagle_temperature.shape)
            rand_data_validation_t = Tensor(name='rand_data_validation',
                                            dtype=trt.float32,
                                            shape=rand_data_validation.shape)
            posterior_alpha_t = Tensor(name='posterior_alpha',
                                       dtype=trt.float32,
                                       shape=posterior_alpha.shape)
            posterior_threshold_t = Tensor(name='posterior_threshold',
                                           dtype=trt.float32,
                                           shape=posterior_threshold.shape)
            paths_t = Tensor(name='paths', dtype=trt.int32, shape=paths.shape)
            greedy_sampling_t = Tensor(name='greedy_sampling',
                                       dtype=trt.int32,
                                       shape=greedy_sampling.shape)
            use_dynamic_tree_t = Tensor(name='use_dynamic_tree',
                                        dtype=trt.int32,
                                        shape=use_dynamic_tree.shape)

            output = tensorrt_llm.models.eagle.model.eagle_sample_and_accept_draft_plugin(
                logits_t,
                draft_tokens_t,
                draft_lens_t,
                eagle_temperature_t,
                rand_data_validation_t,
                posterior_alpha_t,
                posterior_threshold_t,
                TreeParams(paths=paths_t),
                greedy_sampling=greedy_sampling_t,
                use_dynamic_tree=use_dynamic_tree_t)
            accepted_tokens, num_accepted_tokens, accepted_paths, next_draft_tokens, next_draft_lens, next_draft_paths, hidden_size_batch_level_starts = output

            accepted_tokens.mark_output('accepted_tokens')
            num_accepted_tokens.mark_output('num_accepted_tokens')
            accepted_paths.mark_output('accepted_paths')
            next_draft_tokens.mark_output('next_draft_tokens')
            next_draft_lens.mark_output('next_draft_lens')
            next_draft_paths.mark_output('next_draft_paths')
            hidden_size_batch_level_starts.mark_output(
                'hidden_size_batch_level_starts')

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "logits": logits,
            "draft_tokens": draft_tokens,
            "draft_lens": draft_lens,
            "eagle_temperature": eagle_temperature,
            "rand_data_validation": rand_data_validation,
            "posterior_alpha": posterior_alpha,
            "posterior_threshold": posterior_threshold,
            "paths": paths,
            "greedy_sampling": greedy_sampling,
            "use_dynamic_tree": use_dynamic_tree
        }
        outputs = run_session(session, inputs)

        batch_size = ref_accepted_tokens.shape[0]
        torch.testing.assert_close(ref_num_accepted_tokens,
                                   outputs["num_accepted_tokens"],
                                   rtol=0,
                                   atol=0)
        for bi in range(batch_size):
            torch.testing.assert_close(
                ref_accepted_tokens[bi][:ref_num_accepted_tokens[bi]],
                outputs["accepted_tokens"][bi][:ref_num_accepted_tokens[bi]],
                rtol=0,
                atol=0)
        torch.testing.assert_close(ref_accepted_paths,
                                   outputs["accepted_paths"],
                                   rtol=0,
                                   atol=0)

        torch.testing.assert_close(paths,
                                   outputs["next_draft_paths"],
                                   rtol=0,
                                   atol=0)

        self.assertEqual(outputs["next_draft_tokens"].shape, draft_tokens.shape)
        self.assertEqual(outputs["next_draft_lens"].shape, draft_lens.shape)
        self.assertEqual(outputs["hidden_size_batch_level_starts"].shape[0],
                         batch_size * (paths.shape[2] - 1) + 1)

if __name__ == "__main__":
    unittest.main()

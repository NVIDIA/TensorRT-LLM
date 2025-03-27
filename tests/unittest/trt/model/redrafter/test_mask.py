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

T, F = True, False


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def test_get_mask(self):
        # test data
        bs = 2
        nb = 3
        bl = 4
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        draft_token_indices = torch.tensor(
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
        assert draft_token_indices.shape == (bs, nb, bl)

        active_token_indices = torch.tensor(
            [[0, 1, 2, 3, 6, 7, 11, 0, 1], [0, 1, 2, 3, 5, 6, 7, 10, 11]],
            dtype=torch.int32,
        )

        # ref output
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
            draft_token_indices_trt = Tensor(
                name="draft_token_indices",
                shape=draft_token_indices.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            )
            active_token_indices_trt = Tensor(
                name="active_token_indices",
                shape=active_token_indices.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            )
            outputs = tensorrt_llm.models.redrafter.redrafter_helper._get_mask(
                draft_token_indices_trt, active_token_indices_trt, nb, bl)
            outputs.mark_output('spec_decoding_mask')

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "draft_token_indices": draft_token_indices,
            "active_token_indices": active_token_indices
        }
        outputs = run_session(session, inputs)

        torch.testing.assert_close(ref_mask,
                                   outputs["spec_decoding_mask"],
                                   rtol=0,
                                   atol=0)
        torch.set_default_device(old_device)
        return

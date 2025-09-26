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
from parameterized import parameterized
from utils.util import create_session, run_session

import tensorrt_llm
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor

REFS_0 = torch.tensor([3, 2, 1, 4], dtype=torch.int32, device="cuda")

REFS_1 = torch.tensor([[3, 2, 4, 1], [1, 8, 1, 3], [1, 7, 6, 4], [7, 8, 8, 4]],
                      dtype=torch.int32,
                      device="cuda")

REFS_2 = torch.tensor(
    [[[5, 4, 3, 7], [7, 7, 9, 6], [7, 8, 8, 4], [0, 2, 2, 2]],
     [[1, 5, 5, 0], [5, 7, 7, 5], [9, 4, 7, 4], [1, 0, 0, 8]],
     [[4, 8, 0, 8], [3, 4, 0, 2], [0, 9, 1, 3], [5, 6, 5, 2]],
     [[7, 6, 7, 5], [9, 7, 8, 1], [6, 8, 9, 0], [6, 1, 1, 2]]],
    dtype=torch.int32,
    device="cuda")


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

########################################################################################################################

    @parameterized.expand([
        ((4, 4), REFS_0),
        ((4, 4, 4), REFS_1),
        ((4, 4, 4, 4), REFS_2),
    ])
    def test_batch_index_select(self, shape, ref_res) -> None:
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(7)
        x_data = torch.randint(10, size=shape, dtype=torch.int32)
        indices = torch.randint(shape[1], size=(shape[0], ), dtype=torch.int32)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x_trt = Tensor(name="x",
                           shape=x_data.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt("int32"))
            indices_trt = Tensor(name="indices",
                                 shape=indices.shape,
                                 dtype=tensorrt_llm.str_dtype_to_trt("int32"))

            output = tensorrt_llm.models.redrafter.redrafter_helper._batch_index_select(
                x_trt, indices_trt)
            output.mark_output("output")

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "x": x_data,
            "indices": indices,
        }
        outputs = run_session(session, inputs)

        # compare diff
        torch.testing.assert_close(ref_res, outputs["output"])
        torch.set_default_device(old_device)
        return


########################################################################################################################

    def test_prepare_next_input(self) -> None:
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(17)
        # test data
        batch_size, num_candidates, candidate_len, vocab_size, hidden_size = 2, 4, 4, 1, 1
        draft_log_probs = torch.rand(
            [batch_size, num_candidates, candidate_len, vocab_size],
            dtype=torch.float32)
        base_log_probs = torch.rand(
            [batch_size, num_candidates, candidate_len, vocab_size],
            dtype=torch.float32)
        last_base_log_probs = torch.rand(
            [batch_size, num_candidates, vocab_size], dtype=torch.float32)
        beam_index = torch.randint(0,
                                   num_candidates, (batch_size, ),
                                   dtype=torch.int32)
        num_accept_tokens = torch.randint(0,
                                          candidate_len, (batch_size, ),
                                          dtype=torch.int32)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            draft_log_probs_trt = Tensor(
                name="draft_log_probs",
                shape=draft_log_probs.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("float32"),
            )
            base_log_probs_trt = Tensor(
                name="base_log_probs",
                shape=base_log_probs.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("float32"),
            )
            last_base_log_probs_trt = Tensor(
                name="last_base_log_probs",
                shape=last_base_log_probs.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("float32"),
            )
            beam_index_trt = Tensor(
                name="beam_index",
                shape=beam_index.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            )
            num_accept_tokens_trt = Tensor(
                name="num_accept_tokens",
                shape=num_accept_tokens.shape,
                dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            )
            probs = tensorrt_llm.models.redrafter.redrafter_helper._prepare_drafter_input(
                draft_log_probs_trt,
                base_log_probs_trt,
                last_base_log_probs_trt,
                beam_index_trt,
                num_accept_tokens_trt,
            )
            probs.mark_output("probs")

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {
            "draft_log_probs": draft_log_probs,
            "base_log_probs": base_log_probs,
            "last_base_log_probs": last_base_log_probs,
            "beam_index": beam_index,
            "num_accept_tokens": num_accept_tokens,
        }
        outputs = run_session(session, inputs)

        ref_probs = torch.tensor([[0.1245], [0.3713]], dtype=torch.float32)

        # compare diff
        torch.testing.assert_close(ref_probs,
                                   outputs["probs"],
                                   atol=1e-4,
                                   rtol=0.1)
        torch.set_default_device(old_device)
        return

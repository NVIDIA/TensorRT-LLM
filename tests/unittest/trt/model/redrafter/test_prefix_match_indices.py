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


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    def test_prefix_match_indices(self):
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
            dtype=torch.int,
        )
        assert beams.shape == (bs, nb, bl)

        # ref output
        ref_prefix_match_indices = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]],
             [[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 2, 2]]],
            dtype=torch.int,
        )
        assert ref_prefix_match_indices.shape == (bs, nb, bl)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            beams_t = Tensor(name='beams',
                             shape=beams.shape,
                             dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._get_prefix_match_indices(
                beams_t, bl)
            outputs.mark_output('prefix_match_indices')

        # trt run
        session = create_session(builder, network, precision='float32')
        inputs = {"beams": beams}
        outputs = run_session(session, inputs)

        # compare diff
        torch.testing.assert_close(ref_prefix_match_indices,
                                   outputs['prefix_match_indices'],
                                   rtol=0,
                                   atol=0)
        torch.set_default_device(old_device)
        return

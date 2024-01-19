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

from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.models import GPTLMHeadModel


class TestModelDtype(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([(GPTLMHeadModel, 'float32'),
                           (GPTLMHeadModel, 'bfloat16'),
                           (GPTLMHeadModel, 'float16')])
    def test_model_dtype(self, model_cls, dtype):
        ''' Every parameter in the model should have the same dtype as the model initialized to
        '''
        tiny_model = model_cls(num_layers=6,
                               num_heads=4,
                               hidden_size=128,
                               vocab_size=128,
                               hidden_act='relu',
                               max_position_embeddings=128,
                               dtype=dtype)
        for p in tiny_model.parameter():
            self.assertEqual(p.raw_value.dtype, str_dtype_to_np(dtype))


if __name__ == '__main__':
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm.llmapi.model_support_matrix import (
    KEY_MODEL_MATRIX,
    MULTIMODAL_MATRIX,
    Feature,
    SupportStatus,
    get_status,
)


class TestGetStatusAmbiguityHandling(unittest.TestCase):
    def test_llama4_in_both_matrices_returns_none(self):
        self.assertIn("Llama4ForConditionalGeneration", KEY_MODEL_MATRIX)
        self.assertIn("Llama4ForConditionalGeneration", MULTIMODAL_MATRIX)
        status = get_status(
            "Llama4ForConditionalGeneration",
            Feature.CHUNKED_PREFILL,
        )
        self.assertIsNone(status)

    def test_model_only_in_key_model_matrix(self):
        self.assertIn("DeepseekV3ForCausalLM", KEY_MODEL_MATRIX)
        self.assertNotIn("DeepseekV3ForCausalLM", MULTIMODAL_MATRIX)
        status = get_status("DeepseekV3ForCausalLM", Feature.CHUNKED_PREFILL)
        self.assertEqual(status, SupportStatus.YES)

    def test_model_only_in_multimodal_matrix(self):
        self.assertIn("Qwen2VLForConditionalGeneration", MULTIMODAL_MATRIX)
        self.assertNotIn("Qwen2VLForConditionalGeneration", KEY_MODEL_MATRIX)
        status = get_status("Qwen2VLForConditionalGeneration", Feature.CHUNKED_PREFILL)
        self.assertEqual(status, SupportStatus.YES)

    def test_unknown_model_returns_none(self):
        status = get_status("UnknownModelForCausalLM", Feature.CHUNKED_PREFILL)
        self.assertIsNone(status)

    def test_gptoss_chunked_prefill_is_yes(self):
        self.assertIn("GptOssForCausalLM", KEY_MODEL_MATRIX)
        self.assertNotIn("GptOssForCausalLM", MULTIMODAL_MATRIX)
        status = get_status("GptOssForCausalLM", Feature.CHUNKED_PREFILL)
        self.assertEqual(status, SupportStatus.YES)

    def test_feature_not_in_matrix_returns_none(self):
        status = get_status("DeepseekV3ForCausalLM", Feature.MODALITY)
        self.assertIsNone(status)


if __name__ == "__main__":
    unittest.main()

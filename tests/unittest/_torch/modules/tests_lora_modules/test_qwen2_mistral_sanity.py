# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Sanity tests for Qwen2 and Mistral LoRA support.

Both decoder layers used to leave `GatedMLP.layer_idx` unset, and Mistral
additionally dropped `lora_params` before calling its MLP.

These adapters deliberately target *only* the MLP modules. Attention LoRA works
on both models either way, so an adapter that also targeted attention would
still change the output and the tests would pass even with the MLP path broken.
With MLP-only adapters, a `lora_params` that never reaches the MLP means no
LoRA is applied at all, which the output comparison catches.
"""

import os

import pytest
from _torch.modules.tests_lora_modules.lora_sanity_utils import (
    MLP_LORA_MODULES,
    MLP_TRTLLM_MODULES,
    run_lora_test,
)
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_40gb


class TestQwen2LoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen2-0.5B-Instruct"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_qwen2_bf16_lora(self):
        run_lora_test(
            self.model_path,
            MLP_LORA_MODULES,
            MLP_TRTLLM_MODULES,
        )


@skip_gpu_memory_less_than_40gb
class TestMistralLoRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/mistral-7b-v0.1"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def test_mistral_bf16_lora(self):
        run_lora_test(
            self.model_path,
            MLP_LORA_MODULES,
            MLP_TRTLLM_MODULES,
        )

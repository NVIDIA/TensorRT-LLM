# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tensorrt_llm.llmapi import MTPDecodingConfig

from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .._common import _MODEL_NAME, _MODEL_PATH, _llm, _sm100_tp4


@_sm100_tp4
class TestModelingV2Glm53FlashNvfp4Sm100Default(LlmapiAccuracyTestHarness):
    """The split target at attention TP4 with expert-parallel MoE (EP4)."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    def test_gsm8k(self) -> None:
        with _llm(self.MODEL_PATH) as llm:
            GSM8K(self.MODEL_NAME).evaluate(llm)

    def test_gsm8k_mtp_vanilla(self) -> None:
        """Vanilla one-model MTP, which the fused target declines: the engine's MTP worker drives this target."""
        spec = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True)
        with _llm(self.MODEL_PATH, spec) as llm:
            GSM8K(self.MODEL_NAME).evaluate(llm)

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
import pytest

from tensorrt_llm.llmapi import LLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_ada
from .accuracy_core import MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @skip_pre_ada
    def test_fp8_rowwise(self):
        quant_config = QuantConfig(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-7B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    def test_weight_only(self):
        quant_config = QuantConfig(QuantAlgo.W8A16)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


class TestQwen2_5_0_5BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-0.5B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_5_1_5BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-1.5B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_weight_only(self):
        quant_config = QuantConfig(QuantAlgo.W8A16)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_5_7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-7B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

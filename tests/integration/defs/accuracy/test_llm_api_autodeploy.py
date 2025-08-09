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

from tensorrt_llm import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @pytest.mark.skip_less_device_memory(32000)
    def test_auto_dtype(self):
        with AutoDeployLLM(self.MODEL_PATH) as llm:
            # task = CnnDailymail(self.MODEL_NAME)
            # task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    def test_nvfp4(self):
        model_path = f"{llm_models_root()}/nvfp4-quantized/Meta-Llama-3.1-8B"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.parametrize("stream_interval", [4, 64],
                             ids=["stream_interval_4", "stream_interval_64"])
    def test_nvfp4_streaming(self, stream_interval):
        # When stream_interval < TLLM_STREAM_INTERVAL_THRESHOLD, hf incremental detokenization is used.
        # When stream_interval >= TLLM_STREAM_INTERVAL_THRESHOLD, trtllm implemented incremental detokenization is used.
        # The behavior is due to perf considerations, while both paths need to be tested.
        with LLM(f"{llm_models_root()}/nvfp4-quantized/Meta-Llama-3.1-8B",
                 stream_interval=stream_interval) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.stream_interval == stream_interval
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm, streaming=True)

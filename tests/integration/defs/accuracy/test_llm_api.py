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

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import (EagleDecodingConfig,
                                 ExtendedRuntimePerfKnobConfig, KvCacheConfig,
                                 SamplingParams)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_post_blackwell, skip_pre_ada
from .accuracy_core import (GSM8K, MMLU, CnnDailymail, JsonModeEval,
                            LlmapiAccuracyTestHarness)


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise(self):
        quant_config = QuantConfig(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device(2)
    def test_cp2(self):
        with LLM(self.MODEL_PATH, context_parallel_size=2) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    def test_tp2cp2(self):
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=2,
                 context_parallel_size=2) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("backend", ["xgrammar"])
    def test_guided_decoding(self, backend: str):
        llm = LLM(self.MODEL_PATH, guided_decoding_backend=backend)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("backend", ["xgrammar"])
    def test_guided_decoding_4gpus(self, backend: str):
        llm = LLM(self.MODEL_PATH,
                  guided_decoding_backend=backend,
                  tensor_parallel_size=2,
                  pipeline_parallel_size=2)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    def test_gather_generation_logits_cuda_graph(self):
        """RCCA: https://nvbugs/5365525"""
        extended_runtime_perf_knob_config = ExtendedRuntimePerfKnobConfig(
            cuda_graph_mode=True, cuda_graph_cache_size=1)
        llm = LLM(
            self.MODEL_PATH,
            gather_generation_logits=True,
            extended_runtime_perf_knob_config=extended_runtime_perf_knob_config)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    def test_logprobs(self):
        sampling_config = SamplingParams(logprobs=2)
        llm = LLM(self.MODEL_PATH, gather_generation_logits=True)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_config,
                          extra_acc_spec="logprobs=2")


class TestLlama3_2_1B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_post_blackwell
    def test_smooth_quant(self):
        quant_config = QuantConfig(
            QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_post_blackwell
    def test_smooth_quant_ootb(self):
        quant_config = QuantConfig(QuantAlgo.W8A8_SQ_PER_CHANNEL)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_post_blackwell
    def test_int4_awq(self):
        quant_config = QuantConfig(QuantAlgo.W4A16_AWQ)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_post_blackwell
    def test_int4_awq_int8_kv_cache(self):
        quant_config = QuantConfig(QuantAlgo.W4A16_AWQ)
        kv_cache_config = KvCacheConfig(quant_algo=QuantAlgo.INT8)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 kv_cache_config=kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        kv_cache_config = KvCacheConfig(quant_algo=QuantAlgo.FP8)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 kv_cache_config=kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device(2)
    def test_fp8_pp2(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        kv_cache_config = KvCacheConfig(quant_algo=QuantAlgo.FP8)
        with LLM(self.MODEL_PATH,
                 pipeline_parallel_size=2,
                 quant_config=quant_config,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=64) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise(self):
        quant_config = QuantConfig(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral7B_0_3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_PATH = f"{llm_models_root()}/Mistral-7B-Instruct-v0.3"

    @skip_post_blackwell
    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize("quant", ['int4', 'int4_awq', 'int8_awq'])
    def test_quant_tp4(self, quant):
        if quant == 'int4':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
        elif quant == 'int4_awq':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        elif quant == 'int8_awq':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistralNemo12B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-Nemo-12b-Base"
    MODEL_PATH = f"{llm_models_root()}/Mistral-Nemo-Base-2407"

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_auto_dtype_tp2(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 tensor_parallel_size=2,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(80000)
    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8,
                                   kv_cache_quant_algo=QuantAlgo.FP8)
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral_NeMo_Minitron_8B_Instruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Mistral-NeMo-Minitron-8B-Instruct"

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"

    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    def test_smooth_quant_tp2pp2(self):
        quant_config = QuantConfig(
            quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 tensor_parallel_size=2,
                 pipeline_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1"

    @skip_post_blackwell
    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(80000)
    def test_awq_tp2(self):
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestPhi4MiniInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-4-mini-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-4-mini-instruct"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
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

    @skip_post_blackwell
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

    @skip_post_blackwell
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

    @pytest.mark.skip(reason="https://nvbugs/5280461")
    @skip_pre_ada
    def test_fp8_kvcache(self):
        "RCCA: https://nvbugs/5065080"
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                                   kv_cache_quant_algo=QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestEagleVicuna_7B_v1_3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "lmsys/vicuna-7b-v1.3"
    MODEL_PATH = f"{llm_models_root()}/vicuna-7b-v1.3"

    speculative_config = EagleDecodingConfig(
        max_draft_len=63,
        speculative_model_dir=f"{llm_models_root()}/EAGLE-Vicuna-7B-v1.3",
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )

    def test_auto_dtype(self):
        with LLM(
                self.MODEL_PATH,
                max_batch_size=8,  # Spec-dec use case less than bs=8
                speculative_config=self.speculative_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestEagle2Vicuna_7B_v1_3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "lmsys/vicuna-7b-v1.3"
    MODEL_PATH = f"{llm_models_root()}/vicuna-7b-v1.3"

    speculative_config = EagleDecodingConfig(
        max_draft_len=63,
        speculative_model_dir=f"{llm_models_root()}/EAGLE-Vicuna-7B-v1.3",
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=10)

    def test_auto_dtype(self):
        with LLM(
                self.MODEL_PATH,
                max_batch_size=8,  # Spec-dec use case less than bs=8
                speculative_config=self.speculative_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestStarCoder2_7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "bigcode/starcoder2-7b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-7b"
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    @pytest.mark.skip_less_device_memory(70000)
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH, kv_cache_config=self.kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device_memory(70000)
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 kv_cache_config=self.kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestCodestral_22B_V01(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Codestral-22B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Codestral-22B-v0.1"
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH, kv_cache_config=self.kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device_memory(80000)
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 kv_cache_config=self.kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

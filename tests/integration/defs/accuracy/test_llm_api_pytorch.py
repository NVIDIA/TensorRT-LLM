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

from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (llm_models_root, parametrize_with_ids, skip_pre_ada,
                        skip_pre_blackwell, skip_pre_hopper)
from .accuracy_core import MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @pytest.mark.skip_less_device_memory(32000)
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    def test_nvfp4(self):
        model_path = f"{llm_models_root()}/nvfp4-quantized/Meta-Llama-3.1-8B"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_bfloat16(self, attn_backend, torch_compile):
        if torch_compile:
            pytest.skip("https://nvbugs/5216737")
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
        )
        llm = LLM(self.MODEL_PATH, pytorch_backend_config=pytorch_config)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2)],
                             ids=["tp4", "tp2pp2"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, attn_backend,
                            torch_compile):
        if torch_compile and pp_size > 1:
            pytest.skip(
                "Pipeline parallel with torch.compile is not supported yet.\n"
                "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
                "discarded at graph breaks.")
        if torch_compile:
            pytest.skip("https://nvbugs/5216737")
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
        )
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  pytorch_backend_config=pytorch_config)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    def test_fp8(self, fp8kv, attn_backend, torch_compile):
        if torch_compile:
            pytest.skip("https://nvbugs/5216737")
        quant_config = QuantConfig(QuantAlgo.FP8)
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
        )
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"
        llm = LLM(
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
            quant_config=quant_config,
            pytorch_backend_config=pytorch_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2)],
                             ids=["tp4", "tp2pp2"])
    def test_fp8_4gpus(self, tp_size, pp_size, fp8kv, attn_backend,
                       torch_compile):
        if torch_compile:
            pytest.skip("https://nvbugs/5216737")
        if torch_compile and pp_size > 1:
            pytest.skip(
                "Pipeline parallel with torch.compile is not supported yet.\n"
                "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
                "discarded at graph breaks.")
        quant_config = QuantConfig(QuantAlgo.FP8)
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
        )
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"
        llm = LLM(
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            quant_config=quant_config,
            pytorch_backend_config=pytorch_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama3_3_70BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["H100", "B200"])
    def test_fp8_tp4(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8"
        with LLM(model_path, tensor_parallel_size=4) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["B200"])
    def test_nvfp4_tp4(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4"
        with LLM(model_path, tensor_parallel_size=4) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/mistral-7b-v0.1"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
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

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_device_not_contain(["H100", "B200"])
    def test_fp8_tp2(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp8"
        with LLM(model_path, tensor_parallel_size=2) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_device_not_contain(["B200"])
    def test_nvfp4_tp2(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp4"
        with LLM(model_path, tensor_parallel_size=2) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


# This class has extensively parameterized test methods, which yield totally 200 test cases.
# This is because this model requires high test coverage over the feature combinations.
# Normally we should not parameterize test methods so extensively -- just test on the typical/important feature combinations.
class TestDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn",
                          [None, pytest.param(2, marks=skip_pre_hopper)])
    def test_bfloat16(self, mtp_nextn, attention_dp, cuda_graph,
                      overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        if mtp_nextn is not None and mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        else:
            mtp_config = None
        llm = LLM(self.MODEL_PATH,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn",
                          [None, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                            attention_dp, cuda_graph, overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        if mtp_nextn is not None and mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        else:
            mtp_config = None
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_device_not_contain(["H100"])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    @parametrize_with_ids("mtp_nextn", [None, 2])
    def test_fp8_block_scales(self, mtp_nextn, attention_dp, cuda_graph,
                              overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        if mtp_nextn is not None and mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        else:
            mtp_config = None
        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["H100"])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    @parametrize_with_ids("mtp_nextn", [None, 2])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_fp8_block_scales_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                                    attention_dp, cuda_graph,
                                    overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        if mtp_nextn is not None and mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        else:
            mtp_config = None
        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    def test_nvfp4(self, attention_dp, cuda_graph, overlap_scheduler):
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp)
        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_nvfp4_4gpus(self, tp_size, pp_size, ep_size, attention_dp,
                         cuda_graph, overlap_scheduler):
        pytorch_config = PyTorchConfig(
            enable_overlap_scheduler=overlap_scheduler,
            use_cuda_graph=cuda_graph)
        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp)
        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMinitron4BBaseInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/nemotron/nemotron-mini-4b-instruct_vfp8-fp8-bf16-export"

    @skip_pre_ada
    def test_fp8_prequantized(self):
        with LLM(self.MODEL_PATH) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronNas(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"

    @pytest.mark.skip_less_device(8)
    def test_auto_dtype_tp8(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        pytorch_config = PyTorchConfig(enable_overlap_scheduler=True)

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 pytorch_backend_config=pytorch_config) as llm:

            task = CnnDailymail(self.MODEL_NAME)
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

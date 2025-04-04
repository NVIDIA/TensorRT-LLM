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
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (llm_models_root, parametrize_with_ids,
                        skip_pre_blackwell, skip_pre_hopper)
from .accuracy_core import MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

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
                                                         (2, 2, 1)],
                             ids=["tp4", "ep4", "tp2pp2"])
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
                                                         (2, 2, 1)],
                             ids=["tp4", "ep4", "tp2pp2"])
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
                                                         (2, 2, 1)],
                             ids=["tp4", "ep4", "tp2pp2"])
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

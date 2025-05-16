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
from .accuracy_core import (GSM8K, MMLU, CnnDailymail, GPQADiamond,
                            LlmapiAccuracyTestHarness)


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
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_chunked_prefill(self, attn_backend):
        pytest.skip("https://nvbugspro.nvidia.com/bug/5285881")
        pytorch_config = PyTorchConfig(attn_backend=attn_backend, )
        llm = LLM(self.MODEL_PATH,
                  enable_chunked_prefill=True,
                  max_num_tokens=64,
                  pytorch_backend_config=pytorch_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(32000)
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_bfloat16(self, attn_backend, torch_compile):
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        llm = LLM(self.MODEL_PATH, pytorch_backend_config=pytorch_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2), (1, 4)],
                             ids=["tp4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, attn_backend,
                            torch_compile):
        if torch_compile and pp_size > 1:
            pytest.skip(
                "Pipeline parallel with torch.compile is not supported yet.\n"
                "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
                "discarded at graph breaks.")
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  pytorch_backend_config=pytorch_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    def test_fp8(self, fp8kv, attn_backend, torch_compile):
        quant_config = QuantConfig(QuantAlgo.FP8)
        pytorch_config = PyTorchConfig(
            torch_compile_enabled=torch_compile,
            cuda_graph_padding_enabled=torch_compile,
            cuda_graph_batch_sizes=[4],
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
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
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2), (1, 4)],
                             ids=["tp4", "tp2pp2", "pp4"])
    def test_fp8_4gpus(self, tp_size, pp_size, fp8kv, attn_backend,
                       torch_compile):
        if pp_size > 1 and torch_compile:
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
            disable_overlap_scheduler=torch_compile,
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
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama3_3_70BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["H100", "H200", "B200"])
    def test_fp8_tp4(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8"
        with LLM(model_path, tensor_parallel_size=4) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["B200"])
    def test_nvfp4_tp4(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4"
        with LLM(model_path, tensor_parallel_size=4) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))


class TestLlama4MaverickInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Maverick-17B-128E-Instruct"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 use_cuda_graph=cuda_graph) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_fp8_prequantized(self, cuda_graph, tp_size, pp_size, ep_size):
        pytorch_config = PyTorchConfig(use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        pytorch_config.kv_cache_dtype = "fp8"

        model_path = f"{llm_models_root()}/llama4-models/Llama-4-Maverick-17B-128E-Instruct-FP8"
        with LLM(model_path,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 pytorch_backend_config=pytorch_config,
                 quant_config=quant_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama4ScoutInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 use_cuda_graph=cuda_graph) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/mistral-7b-v0.1"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestGemma3_1BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-1b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

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
    @pytest.mark.skip_device_not_contain(["H100", "H200", "B200"])
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
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    def test_bfloat16(self, mtp_nextn, attention_dp, cuda_graph,
                      overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                            attention_dp, cuda_graph, overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_device_not_contain(["H100", "H200"])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_fp8_block_scales(self, mtp_nextn, fp8kv, attention_dp, cuda_graph,
                              overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                        enable_block_reuse=not fp8kv)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)

        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            # No need to run MMLU for fp8kv
            if not fp8kv:
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_device_not_contain(["H100"])
    def test_fp8_block_scales_cuda_graph_padding(self):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        pytorch_config = PyTorchConfig(disable_overlap_scheduler=False,
                                       use_cuda_graph=True,
                                       cuda_graph_max_batch_size=512,
                                       cuda_graph_padding_enabled=True)
        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["H100", "H200"])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (False, True, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_fp8_block_scales_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                                    fp8kv, attention_dp, cuda_graph,
                                    overlap_scheduler):
        # OOM on H100 with default free_gpu_memory_fraction=0.9
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                        enable_block_reuse=not fp8kv)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)

        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            # No need to run MMLU for fp8kv
            if not fp8kv:
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, True, True, True)])
    def test_nvfp4(self, fp8kv, attention_dp, cuda_graph, overlap_scheduler):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9,
                                        enable_block_reuse=not fp8kv)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.NVFP4
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp)

        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            # No need to run MMLU for fp8kv
            if not fp8kv:
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, True, True, True)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_nvfp4_4gpus(self, fp8kv, attention_dp, cuda_graph,
                         overlap_scheduler, tp_size, pp_size, ep_size):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9,
                                        enable_block_reuse=not fp8kv)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.NVFP4
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        llm = LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp)

        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            # No need to run MMLU for fp8kv
            if not fp8kv:
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (False, True, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("quant_dtype", [
        pytest.param("none", marks=skip_pre_hopper),
        pytest.param("fp8", marks=skip_pre_hopper),
        pytest.param("nvfp4", marks=skip_pre_blackwell)
    ])
    def test_no_kv_cache_reuse(self, quant_dtype, mtp_nextn, fp8kv,
                               attention_dp, cuda_graph, overlap_scheduler):
        if quant_dtype == "nvfp4" and mtp_nextn > 0:
            pytest.skip("MTP is not supported for NVFP4")

        model_path = self.MODEL_PATH
        if quant_dtype == "fp8":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/fp8"
        elif quant_dtype == "nvfp4":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only"

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6,
                                        enable_block_reuse=False)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        if quant_dtype == "none":
            assert not fp8kv
            quant_config = None
        else:
            quant_config = QuantConfig()
            if quant_dtype == "fp8":
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            elif quant_dtype == "nvfp4":
                quant_config.quant_algo = QuantAlgo.NVFP4
            if fp8kv:
                quant_config.kv_cache_quant_algo = QuantAlgo.FP8
                pytorch_config.kv_cache_dtype = "fp8"

        llm = LLM(model_path,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)

        if quant_dtype == "fp8":
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        elif quant_dtype == "nvfp4":
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            # No need to run MMLU for fp8kv
            if not fp8kv:
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestDeepSeekR1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-R1"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1"

    @pytest.mark.skip_less_device(8)
    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,batch_size,moe_backend",
        [
            (8, 1, 4, 3, False, False, True, True, 1, "CUTLASS"),
            #TODO: enable mtp after bug fix
            (8, 1, 4, 0, False, False, True, True, 1, "TRTLLM"),
            (8, 1, 8, 0, True, True, True, True, 24, "CUTLASS"),
            (8, 1, 1, 0, True, True, True, True, 24, "CUTLASS"),
        ],
        ids=["latency", "latency_trtllmgen", "throughput", "throughput_tp8"])
    def test_nvfp4_8gpus(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                         attention_dp, cuda_graph, overlap_scheduler,
                         batch_size, moe_backend):

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=not fp8kv)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.NVFP4
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                  batch_size=batch_size,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config,
                  moe_backend=moe_backend)
        assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))

    @pytest.mark.skip_less_device(8)
    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,batch_size",
        [(8, 1, 4, 3, False, False, True, True, 1),
         (8, 1, 8, 0, True, True, True, True, 24)],
        ids=["latency", "throughput"])
    def test_fp8_blockscale(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                            attention_dp, cuda_graph, overlap_scheduler,
                            batch_size):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            pytorch_config.kv_cache_dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1",
                  batch_size=batch_size,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  pytorch_backend_config=pytorch_config,
                  quant_config=quant_config,
                  enable_attention_dp=attention_dp,
                  speculative_config=mtp_config)
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        if fp8kv:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
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
        pytorch_config = PyTorchConfig()

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 pytorch_backend_config=pytorch_config) as llm:

            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronSuper(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    MODEL_PATH = f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1"

    @pytest.mark.skip_less_device(2)
    def test_auto_dtype_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))


class TestNemotronNano(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    MODEL_PATH = f"{llm_models_root()}/Llama-3.1-Nemotron-Nano-8B-v1"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronH(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
    MODEL_PATH = f"{llm_models_root()}/Nemotron-H-8B-Base-8K"

    @pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5264431")
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
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


class TestQwen3_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-8B"

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, False, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler):
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        llm = LLM(f"{llm_models_root()}/Qwen3/Qwen3-8B-FP8",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_30B_A3B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-30B-A3B"

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, False, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler):
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        llm = LLM(f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B-FP8",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, True, True, True)],
        ids=["latency"])
    def test_fp8(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                 overlap_scheduler):
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        llm = LLM(
            f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_fp8_hf",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            pytorch_backend_config=pytorch_config,
            enable_attention_dp=attention_dp)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, True, True, True)],
        ids=["latency"])
    def test_nvfp4(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                   overlap_scheduler):
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        llm = LLM(
            f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            pytorch_backend_config=pytorch_config,
            enable_attention_dp=attention_dp)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_32B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-32B"

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, False, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler):
        pytorch_config = PyTorchConfig(
            disable_overlap_scheduler=not overlap_scheduler,
            use_cuda_graph=cuda_graph)

        llm = LLM(f"{llm_models_root()}/Qwen3/Qwen3-32B-FP8",
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  pytorch_backend_config=pytorch_config,
                  enable_attention_dp=attention_dp)
        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

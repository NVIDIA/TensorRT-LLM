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
from test_common.llm_data import hf_id_to_local_model_dir, llm_models_root

from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

from .accuracy_core import GSM8K, MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = hf_id_to_local_model_dir(MODEL_NAME)

    def get_default_kwargs(self, enable_chunked_prefill=False):
        config = {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "max_batch_size": 512,
            # 131072 is the max seq len for the model
            "max_seq_len": 8192,
            # max num tokens is derived in the build_config, which is not used by AutoDeploy llmargs.
            # Set it explicitly here to 8192 which is the default in build_config.
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "transforms": {
                "resize_kv_cache": {
                    "free_mem_ratio": 0.7
                },
                "compile_model": {
                    "backend":
                    "torch-opt",
                    "cuda_graph_batch_sizes":
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                },
            },
        }
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            config[
                "max_num_tokens"] = 512  # NOTE: must be > max(attn_page_size, max_batch_size)
        return config

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    @pytest.mark.parametrize("enable_chunked_prefill", [False, True])
    def test_auto_dtype(self, world_size, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           world_size=world_size,
                           **kwargs) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)


class TestNemotronH(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
    MODEL_PATH = f"{llm_models_root()}/Nemotron-H-8B-Base-8K"

    def get_default_kwargs(self, enable_chunked_prefill=False):
        config = {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            # SSMs do not support cache reuse.
            "kv_cache_config": {
                "enable_block_reuse": False
            },
            # Keep max_batch_size as in the PyTorch test to avoid OOM
            "max_batch_size": 128,
            # Model context length is 8K
            "max_seq_len": 8192,
            # Set explicitly to match default build_config behavior
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "transforms": {
                "resize_kv_cache": {
                    "free_mem_ratio": 0.7
                },
                "compile_model": {
                    "backend": "torch-cudagraph",
                    "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                },
            },
        }
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            config[
                "max_num_tokens"] = 512  # NOTE: must be > max(attn_page_size, max_batch_size)
        return config

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [False, True])
    def test_auto_dtype(self, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronMOE(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-MOE"
    MODEL_PATH_BF16 = f"{llm_models_root()}/Nemotron-Nano-3-30B-A3.5B-dev-1024"
    MODEL_PATH_FP8 = f"{llm_models_root()}/Nemotron-Nano-3-30B-A3.5B-FP8-KVFP8-dev"
    MODEL_PATH_NVFP4 = f"{llm_models_root()}/Nemotron-3-Nano-30B-A3B-NVFP4"

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            # SSMs do not support cache reuse.
            "kv_cache_config": {
                "enable_block_reuse": False
            },
            # Keep max_batch_size as in the PyTorch test to avoid OOM
            "max_batch_size": 128,
            # Model context length is 8K
            "enable_chunked_prefill": True,
            "max_seq_len": 8192,
            # Set explicitly to match default build_config behavior
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "compile_backend": "torch-cudagraph",
            "free_mem_ratio": 0.7,
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "transforms": {
                "detect_sharding": {
                    "sharding_source": ['factory', 'heuristic'],
                    "sharding_dims": ['ep', 'bmm'],
                },
                "multi_stream_moe": {
                    "stage": "compile",
                    "enabled": True,
                },
                # NOTE: some accuracy benchmarks may require fp32 precision for mamba cache
                # "insert_cached_ssm_attention": {
                #     "cache_config": {
                #         "mamba_dtype": "float32",
                #     },
                # },
            }
        }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(32000)
    def test_bf16(self):
        kwargs = self.get_default_kwargs()
        # TODO: multi-stream MOE seems to increase the memory usage
        kwargs["max_batch_size"] = 32
        kwargs["free_mem_ratio"] = 0.4
        with AutoDeployLLM(model=self.MODEL_PATH_BF16,
                           tokenizer=self.MODEL_PATH_BF16,
                           **kwargs) as llm:
            sampling_params = self.get_default_sampling_params()
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(32000)
    def test_fp8(self):
        kwargs = self.get_default_kwargs()
        kwargs["max_batch_size"] = 64
        with AutoDeployLLM(model=self.MODEL_PATH_FP8,
                           tokenizer=self.MODEL_PATH_FP8,
                           **kwargs) as llm:
            # Manually set quant_config for FP8 model to get the accuracy threshold
            llm.args.quant_config.quant_algo = QuantAlgo.FP8
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            sampling_params = self.get_default_sampling_params()
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip(reason="NVFP4 model is not in the CI yet")
    def test_nvfp4(self):
        kwargs = self.get_default_kwargs()
        with AutoDeployLLM(model=self.MODEL_PATH_NVFP4,
                           tokenizer=self.MODEL_PATH_NVFP4,
                           **kwargs) as llm:
            # Manually set quant_config for NVFP4 model to get the accuracy threshold
            llm.args.quant_config.quant_algo = QuantAlgo.NVFP4
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            sampling_params = self.get_default_sampling_params()

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronSuperV3(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for Nemotron Super V3.

    Runs the model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K
    """

    MODEL_NAME = "nvidia/Nemotron-Super-V3"
    MODEL_PATH_BF16 = f"{llm_models_root()}/Nemotron-Super-3-120B-A12B-dev"
    # Set minimum possible seq len + small buffer, for test speed & memory usage
    MAX_SEQ_LEN = max(MMLU.MAX_INPUT_LEN + MMLU.MAX_OUTPUT_LEN,
                      GSM8K.MAX_INPUT_LEN + GSM8K.MAX_OUTPUT_LEN)

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "skip_loading_weights": False,
            "compile_backend": "torch-cudagraph",
            "free_mem_ratio": 0.9,
            "max_batch_size": 128,
            "max_seq_len": self.MAX_SEQ_LEN,
            "max_num_tokens": self.MAX_SEQ_LEN,
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "transforms": {
                "detect_sharding": {
                    "sharding_source": ['factory', 'heuristic'],
                    "sharding_dims": ['ep', 'bmm'],
                },
            }
        }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    # 180GB works, might be able to go lower
    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.skip_less_device(4)
    def test_bf16(self):
        kwargs = self.get_default_kwargs()
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH_BF16,
                           tokenizer=self.MODEL_PATH_BF16,
                           world_size=4,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

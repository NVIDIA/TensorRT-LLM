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

from pathlib import Path

import pytest
import torch
import yaml
from defs.conftest import get_llm_root, get_sm_version, skip_pre_blackwell
from test_common.llm_data import hf_id_to_local_model_dir, llm_models_root

from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

from ..conftest import get_device_count, llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, MMLU, CnnDailymail, LlmapiAccuracyTestHarness

_AD_CONFIGS_DIR = (Path(get_llm_root()) / 'examples' / 'auto_deploy' /
                   'model_registry' / 'configs')


def _load_ad_config(config_name):
    """Load a YAML config from the AutoDeploy model registry configs directory."""
    with open(_AD_CONFIGS_DIR / config_name) as f:
        return yaml.safe_load(f)


# Mapping from model_id to (quant_algo, kv_cache_quant_algo) for accuracy
# threshold lookups. AutoDeploy infers quantization from the checkpoint, but
# quant_config must still be set explicitly so that the accuracy harness can
# look up the correct per-quantization thresholds.
_QUANT_ALGO_BY_MODEL_ID = {
    "fp8": (QuantAlgo.FP8, QuantAlgo.FP8),
    "nvfp4": (QuantAlgo.NVFP4, QuantAlgo.FP8),
}


def _set_quant_config(llm, model_id: str) -> None:
    """Set quant_config on *llm* based on *model_id* so the accuracy harness
    can resolve the correct thresholds."""
    if model_id in _QUANT_ALGO_BY_MODEL_ID:
        quant_algo, kv_cache_quant_algo = _QUANT_ALGO_BY_MODEL_ID[model_id]
        llm.args.quant_config.quant_algo = quant_algo
        llm.args.quant_config.kv_cache_quant_algo = kv_cache_quant_algo


def print_memory_usage(label: str):
    """Print detailed memory usage for all CUDA devices."""
    print(f"\n{'=' * 60}")
    print(f"Memory Usage: {label}")
    print(f"{'=' * 60}")
    num_devices = torch.cuda.device_count()
    for device_id in range(num_devices):
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device_id) / 1024**3
        cached_available = reserved - allocated  # Available in PyTorch's cache
        free, total = torch.cuda.mem_get_info(device_id)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        used_gb = total_gb - free_gb
        print(f"  Device {device_id}:")
        print(f"    Allocated:       {allocated:.2f} GB")
        print(f"    Reserved:        {reserved:.2f} GB")
        print(f"    Peak Allocated:  {peak_allocated:.2f} GB")
        print(f"    Peak Reserved:   {peak_reserved:.2f} GB")
        print(
            f"    Available:       {cached_available:.2f} GB (in PyTorch cache) | {free_gb:.2f} GB (on GPU)"
        )
        print(f"    GPU Total:       {used_gb:.2f} / {total_gb:.2f} GB")
    print(f"{'=' * 60}\n")


def low_memory_overrides(config,
                         max_batch_size=32,
                         free_gpu_memory_fraction=0.4,
                         max_seq_len=8192,
                         max_num_tokens=8192,
                         cuda_graph_batch_sizes=None):
    """Update and return config that reduce memory footprint for unquantized (bf16) runs."""
    if cuda_graph_batch_sizes is None:
        cuda_graph_batch_sizes = [
            s for s in [1, 2, 4, 8, 16, 32, 64, 128] if s <= max_batch_size
        ]
    config.update({
        "max_batch_size": max_batch_size,
        "max_seq_len": max_seq_len,
        "max_num_tokens": max_num_tokens,
        "cuda_graph_batch_sizes": cuda_graph_batch_sizes,
    })
    kv_cache_config = config.setdefault("kv_cache_config", {})
    kv_cache_config["free_gpu_memory_fraction"] = free_gpu_memory_fraction
    return config


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = hf_id_to_local_model_dir(MODEL_NAME)

    # Configuration presets for different attention backends
    ATTN_BACKEND_CONFIGS = {
        "flashinfer": {
            "max_batch_size": 512,
            "max_seq_len": 8192,
            "compile_backend": "torch-cudagraph",
        },
        "trtllm": {
            "max_batch_size": 512,
            "max_seq_len": 8192,
            "compile_backend": "torch-cudagraph",
        },
        "torch": {
            "max_batch_size": 128,
            "max_seq_len": 2048,
            "compile_backend": "torch-simple",
        },
    }

    def get_default_kwargs(self,
                           enable_chunked_prefill=False,
                           attn_backend="flashinfer"):
        backend_cfg = self.ATTN_BACKEND_CONFIGS[attn_backend]

        config = {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "attn_backend": attn_backend,
            "max_batch_size": backend_cfg["max_batch_size"],
            # 131072 is the max seq len for the model
            "max_seq_len": backend_cfg["max_seq_len"],
            # max num tokens is derived in the build_config, which is not used by AutoDeploy llmargs.
            # Set it explicitly here to 8192 which is the default in build_config.
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.7
            },
            "transforms": {
                "compile_model": {
                    "backend":
                    backend_cfg["compile_backend"],
                    "cuda_graph_batch_sizes":
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                },
            },
        }
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            # NOTE: must be > max(tokens_per_block, max_batch_size)
            config["max_num_tokens"] = 512
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
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm", "torch"])
    def test_auto_dtype(self, world_size, enable_chunked_prefill, attn_backend):
        kwargs = self.get_default_kwargs(enable_chunked_prefill, attn_backend)
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           world_size=world_size,
                           **kwargs) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            if attn_backend != "torch":
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm, sampling_params=sampling_params)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_attention_dp(self, world_size):
        """Test attention data parallelism mode where TP sharding is disabled."""
        kwargs = self.get_default_kwargs(enable_chunked_prefill=True)
        # Enable attention DP - this disables TP sharding
        kwargs["transforms"]["detect_sharding"] = {"enable_attention_dp": True}
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

    def get_default_kwargs(self,
                           enable_chunked_prefill=False,
                           attn_backend="flashinfer"):
        config = {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "attn_backend": attn_backend,
            # SSMs do not support cache reuse.
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.7
            },
            # Keep max_batch_size as in the PyTorch test to avoid OOM
            "max_batch_size": 128,
            # Model context length is 8K
            "max_seq_len": 8192,
            # Set explicitly to match default build_config behavior
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "transforms": {
                "compile_model": {
                    "backend": "torch-cudagraph",
                    "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                },
            },
        }
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            config[
                "max_num_tokens"] = 512  # NOTE: must be > max(tokens_per_block, max_batch_size)
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
    @pytest.mark.parametrize("ssm_backend", ["triton_ssm", "flashinfer_ssm"])
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
    def test_auto_dtype(self, enable_chunked_prefill, ssm_backend,
                        attn_backend):
        kwargs = self.get_default_kwargs(enable_chunked_prefill, attn_backend)
        kwargs.setdefault("transforms", {})
        insert_ssm_cfg = {"backend": ssm_backend}
        if ssm_backend == "flashinfer_ssm":
            insert_ssm_cfg["cache_config"] = {"mamba_dtype": "bfloat16"}
        kwargs["transforms"]["insert_cached_ssm_attention"] = insert_ssm_cfg
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronV2(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    _MODEL_PATH_BASE = f"{llm_models_root()}/NVIDIA-Nemotron-Nano-9B-v2"
    MODEL_PATH = _MODEL_PATH_BASE
    MODEL_PATH_FP8 = f"{_MODEL_PATH_BASE}-FP8"
    MODEL_PATH_NVFP4 = f"{_MODEL_PATH_BASE}-NVFP4"

    def get_default_kwargs(self, enable_chunked_prefill=False):
        config = _load_ad_config('nemotron-nano-9b-v2.yaml')
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            # NOTE: must be > max(tokens_per_block, max_batch_size)
            config["max_num_tokens"] = 512
        return config

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    def evaluate_tasks(self, llm, sampling_params):
        task = MMLU(self.MODEL_NAME)
        task.evaluate(llm, sampling_params=sampling_params)
        task = GSM8K(self.MODEL_NAME)
        task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [True, False])
    def test_auto_dtype(self, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        kwargs.setdefault("transforms", {})
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           **kwargs) as llm:
            self.evaluate_tasks(llm, sampling_params)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [True])
    def test_fp8(self, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        kwargs.setdefault("transforms", {})
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH_FP8,
                           tokenizer=self.MODEL_PATH_FP8,
                           **kwargs) as llm:
            # Manually set quant_config for FP8 model to get the accuracy threshold
            llm.args.quant_config.quant_algo = QuantAlgo.FP8
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            self.evaluate_tasks(llm, sampling_params)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [True])
    def test_nvfp4(self, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        kwargs.setdefault("transforms", {})
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH_NVFP4,
                           tokenizer=self.MODEL_PATH_NVFP4,
                           **kwargs) as llm:
            # Manually set quant_config for NVFP4 model to get the accuracy threshold
            llm.args.quant_config.quant_algo = QuantAlgo.NVFP4
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            self.evaluate_tasks(llm, sampling_params)


class TestNemotronNanoV3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-MOE"

    CONFIG_YAML = str(
        Path(get_llm_root()) / "examples" / "auto_deploy" / "nano_v3.yaml")
    MODEL_PATHS = {
        "bf16": f"{llm_models_root()}/Nemotron-Nano-3-30B-A3.5B-dev-1024",
        "fp8": f"{llm_models_root()}/Nemotron-Nano-3-30B-A3.5B-FP8-KVFP8-dev",
        "nvfp4": f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    @pytest.mark.parametrize("model_id", ["bf16", "fp8", "nvfp4"])
    def test_accuracy(self, model_id, world_size, attn_backend):
        if model_id == "nvfp4" and get_sm_version() < 100:
            pytest.skip("NVFP4 requires Blackwell or later")
        if world_size > get_device_count():
            pytest.skip(f"Not enough devices for world_size={world_size}")
        model_path = self.MODEL_PATHS[model_id]
        kwargs = {}
        if model_id == "bf16":
            low_memory_overrides(kwargs)
        kwargs["attn_backend"] = attn_backend

        with AutoDeployLLM(model=model_path,
                           tokenizer=model_path,
                           world_size=world_size,
                           yaml_extra=[self.CONFIG_YAML],
                           trust_remote_code=True,
                           **kwargs) as llm:
            _set_quant_config(llm, model_id)

            sampling_params = self.get_default_sampling_params()
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronSuperV3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-012726"
    CONFIG_YAML = str(
        Path(get_llm_root()) / "examples" / "auto_deploy" / "super_v3.yaml")
    MODEL_PATHS = {
        "bf16":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-BF16-BF16KV-012726"),
        "fp8":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-FP8-FP8KV-012726"),
        "nvfp4":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-NVFP4-FP8KV-012726"),
    }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
    @pytest.mark.parametrize("enable_attention_dp", [False, True],
                             ids=["attn_dp_off", "attn_dp_on"])
    @pytest.mark.parametrize("world_size", [1, 4, 8])
    @pytest.mark.parametrize("model_id", ["bf16", "fp8", "nvfp4"])
    def test_accuracy(self, model_id, world_size, enable_attention_dp,
                      attn_backend):
        if model_id == "nvfp4":
            pytest.skip("NVFP4 not yet supported for Super V3")
        if get_device_count() < world_size:
            pytest.skip(f"Not enough devices for world_size={world_size}")
        # bf16 120B model requires at least 4 GPUs
        if model_id == "bf16" and world_size < 4:
            pytest.skip("bf16 Super V3 requires at least 4 GPUs")

        model_path = self.MODEL_PATHS[model_id]
        kwargs = {}
        if model_id == "bf16":
            low_memory_overrides(kwargs)
        kwargs["attn_backend"] = attn_backend
        if enable_attention_dp:
            kwargs.setdefault("transforms", {})["detect_sharding"] = {
                "enable_attention_dp": True
            }

        print_memory_usage("test start")
        with AutoDeployLLM(model=model_path,
                           tokenizer=model_path,
                           world_size=world_size,
                           yaml_extra=[self.CONFIG_YAML],
                           trust_remote_code=True,
                           **kwargs) as llm:
            _set_quant_config(llm, model_id)

            print_memory_usage("after engine build")

            sampling_params = self.get_default_sampling_params()
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

        print_memory_usage("after evaluation")


class TestGLM4Flash(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for GLM-4.7-Flash variants"""

    MODEL_NAME = "GLM-4.7-Flash"
    MODEL_PATH_BF16 = hf_id_to_local_model_dir("zai-org/GLM-4.7-Flash")
    MODEL_PATH_NVFP4 = hf_id_to_local_model_dir("DeepInfra/GLM-4.7-Flash-NVFP4")

    # Set minimum possible seq len + small buffer, for test speed & memory usage
    MAX_SEQ_LEN = max(MMLU.MAX_INPUT_LEN + MMLU.MAX_OUTPUT_LEN,
                      GSM8K.MAX_INPUT_LEN + GSM8K.MAX_OUTPUT_LEN)
    MAX_NUM_TOKENS = MAX_SEQ_LEN

    def get_default_kwargs(self,
                           enable_chunked_prefill=False,
                           attn_backend="flashinfer"):
        config = {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "attn_backend": attn_backend,
            "compile_backend": "torch-cudagraph",
            "max_batch_size": 128,
            "max_seq_len": self.MAX_SEQ_LEN,
            "max_num_tokens": self.MAX_NUM_TOKENS,
            "skip_loading_weights": False,
            "disable_overlap_scheduler": False,
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.88
            },
            "model_kwargs": {
                "torch_dtype": "bfloat16"
            },
            "transforms": {
                "fuse_nvfp4_moe": {
                    "allow_different_input_scales": True,
                },
                "multi_stream_moe": {
                    "stage": "compile",
                    "enabled": True,
                },
                "multi_stream_mla_attn": {
                    "stage": "compile",
                    "enabled": True,
                },
            }
        }
        if enable_chunked_prefill:
            config["enable_chunked_prefill"] = True
            config[
                "max_num_tokens"] = 512  # NOTE: must be > max(tokens_per_block, max_batch_size)
        return config

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [True, False])
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
    def test_auto_dtype(self, enable_chunked_prefill, attn_backend):
        kwargs = self.get_default_kwargs(enable_chunked_prefill, attn_backend)
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH_BF16,
                           tokenizer=self.MODEL_PATH_BF16,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("enable_chunked_prefill", [True, False])
    def test_nvfp4(self, enable_chunked_prefill):
        kwargs = self.get_default_kwargs(enable_chunked_prefill)
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH_NVFP4,
                           tokenizer=self.MODEL_PATH_NVFP4,
                           **kwargs) as llm:
            # Manually set quant_config for NVFP4 model to get the accuracy threshold
            llm.args.quant_config.quant_algo = QuantAlgo.NVFP4
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3NextInstruct(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for Qwen3-Next Instruct via AutoDeploy.

    Runs the model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K.
    Configuration derived from examples/auto_deploy/model_registry/configs/qwen3Next.yaml.
    """

    MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "skip_loading_weights": False,
            "enable_chunked_prefill": True,
            "max_batch_size": 64,
            "max_seq_len": 4096,
            "max_num_tokens": 4096,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.7,
            },
            "transforms": {
                "export_to_gm": {
                    "num_moe_experts_for_export": 2,
                },
                "detect_sharding": {
                    "sharding_source": ['factory', 'heuristic'],
                    "sharding_dims": ['ep', 'bmm'],
                },
            },
        }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize("world_size", [1, 4])
    def test_auto_dtype(self, world_size):
        if get_device_count() < world_size:
            pytest.skip("Not enough devices for world size, skipping test")
        kwargs = self.get_default_kwargs()
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_NAME,
                           tokenizer=self.MODEL_NAME,
                           world_size=world_size,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_5_MoE(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for Qwen3.5-397B-A17B via AutoDeploy.

    Runs the model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K.
    Configuration derived from examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml.
    """

    MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
    MAX_SEQ_LEN = max(MMLU.MAX_INPUT_LEN + MMLU.MAX_OUTPUT_LEN,
                      GSM8K.MAX_INPUT_LEN + GSM8K.MAX_OUTPUT_LEN)

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
            "compile_backend": "torch-cudagraph",
            "max_batch_size": 128,
            "max_seq_len": self.MAX_SEQ_LEN,
            "max_num_tokens": self.MAX_SEQ_LEN,
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.5,
                "tokens_per_block": 64,
            },
            "model_kwargs": {
                "torch_dtype": "bfloat16",
            },
            "transforms": {
                "export_to_gm": {
                    "num_moe_experts_for_export": 2,
                },
            },
        }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize("world_size", [8])
    def test_bf16(self, world_size):
        if get_device_count() < world_size:
            pytest.skip("Not enough devices for world size, skipping test")
        kwargs = self.get_default_kwargs()
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_NAME,
                           tokenizer=self.MODEL_NAME,
                           dtype="bfloat16",
                           world_size=world_size,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

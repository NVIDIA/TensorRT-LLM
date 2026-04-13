# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from defs.conftest import (get_llm_root, get_sm_version, skip_pre_blackwell,
                           skip_pre_hopper)
from test_common.llm_data import hf_id_to_local_model_dir, llm_models_root

from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.llmapi import Eagle3DecodingConfig, MTPDecodingConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

from ..conftest import get_device_count, llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, MMLU, CnnDailymail, LlmapiAccuracyTestHarness

_AD_CONFIGS_DIR = (Path(get_llm_root()) / 'examples' / 'auto_deploy' /
                   'model_registry' / 'configs')
_AD_MODEL_REGISTRY_DIR = Path(
    get_llm_root()) / 'examples' / 'auto_deploy' / 'model_registry'


def _load_ad_config(config_name):
    """Load a YAML config from the AutoDeploy model registry configs directory."""
    with open(_AD_CONFIGS_DIR / config_name) as f:
        return yaml.safe_load(f)


def _get_registry_yaml_extra(model_name: str) -> tuple[list[str], int]:
    """Return (yaml_extra paths, world_size) from the AutoDeploy model registry."""
    with open(_AD_MODEL_REGISTRY_DIR / "models.yaml") as f:
        registry = yaml.safe_load(f)
    for entry in registry["models"]:
        if entry["name"] != model_name:
            continue
        config_dir = _AD_MODEL_REGISTRY_DIR / "configs"
        paths = [str(config_dir / cfg) for cfg in entry["yaml_extra"]]
        world_size = 1
        for cfg in entry["yaml_extra"]:
            cfg_name = str(cfg)
            if "world_size_" in cfg_name and cfg_name.endswith(".yaml"):
                world_size = int(
                    cfg_name.replace("world_size_", "").replace(".yaml", ""))
        return paths, world_size
    raise ValueError(f"Model '{model_name}' not found in model registry")


def _set_quant_config(llm, model_id: str) -> None:
    """Set quant_config on *llm* based on *model_id* so the accuracy harness
    can resolve the correct thresholds."""
    QUANT_ALGO_BY_MODEL_ID = {
        "fp8": {
            "weights": QuantAlgo.FP8,
            "kv_cache": QuantAlgo.FP8
        },
        "nvfp4": {
            "weights": QuantAlgo.NVFP4,
            "kv_cache": QuantAlgo.FP8
        },
    }

    if model_id in QUANT_ALGO_BY_MODEL_ID:
        llm.args.quant_config.quant_algo = QUANT_ALGO_BY_MODEL_ID[model_id][
            "weights"]
        llm.args.quant_config.kv_cache_quant_algo = QUANT_ALGO_BY_MODEL_ID[
            model_id]["kv_cache"]
    else:
        print("Could not match quantization scheme, using default values")


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


def _check_acceptance_rate_stats(stats, min_acceptance_rate: float) -> None:
    total_drafted = 0
    total_accepted = 0
    num_spec_iterations = 0

    for stat in stats:
        spec_stats = stat.get("specDecodingStats", {})
        num_draft = spec_stats.get("numDraftTokens", 0)
        num_accepted = spec_stats.get("numAcceptedTokens", 0)
        if num_draft <= 0:
            continue

        num_spec_iterations += 1
        total_drafted += num_draft
        total_accepted += num_accepted

    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    print("Spec dec acceptance rate: "
          f"{accept_rate:.2%} ({total_accepted}/{total_drafted} tokens across "
          f"{num_spec_iterations} speculative iterations)")

    assert accept_rate >= min_acceptance_rate, (
        f"Acceptance rate {accept_rate:.2%} below threshold {min_acceptance_rate:.0%}"
    )


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
        "cuda_graph_config": {
            "batch_sizes": cuda_graph_batch_sizes
        },
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
        "triton_paged": {
            "max_batch_size": 128,
            "max_seq_len": 8192,
            "compile_backend": "torch-cudagraph",
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
                "fuse_silu_mul": {
                    "enabled": True,
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

    def check_acceptance_rate(self, llm, min_acceptance_rate: float):
        """Check speculative decoding acceptance rate for the current run."""
        _check_acceptance_rate_stats(llm.get_stats(), min_acceptance_rate)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    @pytest.mark.parametrize("enable_chunked_prefill", [False, True])
    @pytest.mark.parametrize("attn_backend",
                             ["flashinfer", "trtllm", "torch", "triton_paged"])
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


class TestLlama3_1_8B_Instruct_Eagle3(LlmapiAccuracyTestHarness):
    """Accuracy test for Eagle3 one-model speculative decoding with AutoDeploy."""

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = hf_id_to_local_model_dir(MODEL_NAME)
    EAGLE_MODEL_PATH = hf_id_to_local_model_dir(
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")

    def get_default_kwargs(self):
        speculative_config = Eagle3DecodingConfig(
            max_draft_len=3,
            speculative_model=self.EAGLE_MODEL_PATH,
            eagle3_one_model=True,
            eagle3_layers_to_capture={1, 15, 28},
        )
        return {
            "compile_backend": "torch-simple",
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "max_batch_size": 128,
            "max_seq_len": 8192,
            "max_num_tokens": 8192,
            "skip_loading_weights": False,
            "disable_overlap_scheduler": False,
            "enable_iter_perf_stats": True,  # Enable stats for acceptance rate
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.7
            },
            "speculative_config": speculative_config,
        }

    def get_default_sampling_params(self):
        return SamplingParams(
            max_tokens=GSM8K.MAX_OUTPUT_LEN,  # 256 tokens
            truncate_prompt_tokens=GSM8K.MAX_INPUT_LEN,
        )

    def check_acceptance_rate(self, llm, min_acceptance_rate: float):
        """Check speculative decoding acceptance rate.

        Args:
            llm: The LLM instance with enable_iter_perf_stats=True.
            min_acceptance_rate: Minimum acceptance rate threshold (default 7%).
        """
        _check_acceptance_rate_stats(llm.get_stats(), min_acceptance_rate)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(32000)
    def test_eagle3_one_model(self):
        """Test Eagle3 one-model speculative decoding accuracy on GSM8K."""
        kwargs = self.get_default_kwargs()

        with AutoDeployLLM(
                model=self.MODEL_PATH,
                tokenizer=self.MODEL_PATH,
                **kwargs,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

            self.check_acceptance_rate(llm, min_acceptance_rate=0.25)


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
    MODEL_NAME = "nvidia/Nemotron-3-Nano"

    CONFIG_YAML = str(
        Path(get_llm_root()) / "examples" / "auto_deploy" / "nano_v3.yaml")
    MODEL_PATHS = {
        "bf16":
        hf_id_to_local_model_dir("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
        "fp8":
        hf_id_to_local_model_dir("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"),
        "nvfp4":
        hf_id_to_local_model_dir("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"),
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
    MODEL_NAME = "nvidia/Nemotron-Super-V3"
    CONFIG_YAML = str(
        Path(get_llm_root()) / "examples" / "auto_deploy" / "super_v3.yaml")
    MODEL_PATHS = {
        "bf16":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        "fp8":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"),
        "nvfp4":
        hf_id_to_local_model_dir(
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
    }

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    def check_acceptance_rate(self, llm, min_acceptance_rate: float):
        """Check speculative decoding acceptance rate for the current run."""
        _check_acceptance_rate_stats(llm.get_stats(), min_acceptance_rate)

    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
    @pytest.mark.parametrize("enable_attention_dp", [False, True],
                             ids=["attn_dp_off", "attn_dp_on"])
    @pytest.mark.parametrize("world_size", [1, 4, 8])
    @pytest.mark.parametrize("model_id", ["bf16", "fp8", "nvfp4"])
    def test_accuracy(self, model_id, world_size, enable_attention_dp,
                      attn_backend):
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
        kwargs.setdefault("transforms", {}).setdefault(
            "detect_sharding", {})["enable_attention_dp"] = enable_attention_dp

        print_memory_usage("test start")
        with AutoDeployLLM(model=model_path,
                           tokenizer=model_path,
                           world_size=world_size,
                           yaml_extra=[self.CONFIG_YAML],
                           trust_remote_code=True,
                           **kwargs) as llm:
            _set_quant_config(llm, model_id)
            # the nvfp4 model is mixed precision, should be tested against higher thresholds
            if model_id == "nvfp4":
                llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
            print_memory_usage("after engine build")

            sampling_params = self.get_default_sampling_params()
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

        print_memory_usage("after evaluation")

    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.parametrize("world_size", [4, 8])
    def test_mtp(self, world_size):
        if get_device_count() < world_size:
            pytest.skip(f"Not enough devices for world_size={world_size}")

        model_path = self.MODEL_PATHS["bf16"]
        kwargs = {}
        low_memory_overrides(kwargs)
        kwargs["compile_backend"] = "torch-simple"
        kwargs["attn_backend"] = "flashinfer"
        kwargs["speculative_config"] = MTPDecodingConfig(
            num_nextn_predict_layers=6,
            mtp_eagle_one_model=True,
            speculative_model=model_path,
        )
        kwargs["transforms"] = {
            "insert_cached_ssm_attention": {
                "backend": "triton_ssm"
            },
            "insert_cached_causal_conv": {
                "backend": "triton_causal_conv"
            },
        }

        print(
            f"SuperV3 MTP params: world_size={world_size}, model_path={model_path}"
        )
        print(f"kwargs: {kwargs}")

        print_memory_usage("test start")
        with AutoDeployLLM(
                model=model_path,
                tokenizer=model_path,
                world_size=world_size,
                enable_iter_perf_stats=True,
                **kwargs,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            self.check_acceptance_rate(llm, min_acceptance_rate=0.45)

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
            "cuda_graph_config": {
                "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128]
            },
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.8
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
            config["transforms"]["compile_model"] = {
                "piecewise_enabled": True,
            }
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
                    "sharding_dims": ['tp', 'ep', 'bmm'],
                    # NOTE: sharding_source applies only to TP sharding
                    "sharding_source": ['factory', 'heuristic'],
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


class TestQwen3_5_397B_MoE(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for Qwen3.5-397B-A17B via AutoDeploy.

    Runs the model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K.
    Configuration derived from examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml.
    """

    MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
    MODEL_NAME_SMALL = "Qwen/Qwen3.5-35B-A3B"
    EXTRA_EVALUATOR_KWARGS = dict(chat_template_kwargs=dict(
        enable_thinking=False))

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
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
        model_path = hf_id_to_local_model_dir(self.MODEL_NAME)
        yaml_paths, registry_world_size = _get_registry_yaml_extra(
            self.MODEL_NAME)
        assert registry_world_size == world_size
        with AutoDeployLLM(model=model_path,
                           tokenizer=model_path,
                           dtype="bfloat16",
                           world_size=world_size,
                           yaml_extra=yaml_paths,
                           **kwargs) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @staticmethod
    def _load_small_config():
        config = _load_ad_config('qwen3.5_moe_35b.yaml')
        world_size = config.pop('world_size', 1)
        return config, world_size


class TestMiniMaxM2(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for MiniMax M2.

    Runs the model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K.
    """

    MODEL_NAME = "MiniMaxAI/MiniMax-M2"
    # Set minimum possible seq len + small buffer, for test speed & memory usage
    MAX_SEQ_LEN = max(MMLU.MAX_INPUT_LEN + MMLU.MAX_OUTPUT_LEN,
                      GSM8K.MAX_INPUT_LEN + GSM8K.MAX_OUTPUT_LEN)

    def get_default_kwargs(self):
        return {
            "skip_tokenizer_init":
            False,
            "trust_remote_code":
            True,
            "skip_loading_weights":
            False,
            "compile_backend":
            "torch-cudagraph",
            "free_mem_ratio":
            0.88,
            "max_batch_size":
            64,
            "max_seq_len":
            self.MAX_SEQ_LEN,
            "max_num_tokens":
            self.MAX_SEQ_LEN,
            "enable_chunked_prefill":
            True,
            "cuda_graph_batch_sizes":
            [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384],
            "model_kwargs": {
                "torch_dtype": "bfloat16",
            },
        }


class TestKimiK2_5(LlmapiAccuracyTestHarness):
    """Accuracy regression tests for Kimi-K2.5 (moonshotai/Kimi-K2.5) via AutoDeploy.

    Runs the NVFP4 model via AutoDeploy and verifies benchmark performance on MMLU and GSM8K.
    Configuration from examples/auto_deploy/model_registry/configs/kimi_k2.yaml.
    """

    MODEL_NAME = "moonshotai/Kimi-K2.5"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K2.5-NVFP4"
    CONFIG_YAML = str(_AD_CONFIGS_DIR / "kimi_k2.yaml")

    def get_default_sampling_params(self):
        eos_id = -1
        beam_width = 1
        return SamplingParams(end_id=eos_id,
                              pad_id=eos_id,
                              n=beam_width,
                              use_beam_search=beam_width > 1)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(120000)
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "ep_size,attention_dp",
        [(1, False), (1, True), (8, False), (8, True)],
        ids=["tp8", "tp8_attn_dp", "ep8", "dep8"],
    )
    def test_nvfp4(self, ep_size, attention_dp):
        if get_device_count() < 8:
            pytest.skip("Not enough devices for world size 8, skipping test")
        config = _load_ad_config("kimi_k2.yaml")
        config["world_size"] = 8
        kwargs = {k: v for k, v in config.items() if k != "world_size"}
        kwargs.setdefault("transforms", {})["detect_sharding"] = {
            "enable_attention_dp": attention_dp,
            "dist_mapping": {
                "tp": 8,
                "moe_ep": ep_size
            },
        }
        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_PATH,
                           tokenizer=self.MODEL_PATH,
                           world_size=8,
                           yaml_extra=[self.CONFIG_YAML],
                           trust_remote_code=True,
                           **kwargs) as llm:
            _set_quant_config(llm, "nvfp4")
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestGemma4MoE(LlmapiAccuracyTestHarness):
    """Bench-run coverage for Gemma4 MoE via AutoDeploy."""

    MODEL_NAME = "google/gemma-4-26B-A4B-it"
    EXTRA_EVALUATOR_KWARGS = {
        "apply_chat_template": True,
    }

    def get_default_sampling_params(self):
        return SamplingParams(end_id=None,
                              pad_id=None,
                              n=1,
                              use_beam_search=False)

    @pytest.mark.skip_less_device_memory(80000)
    def test_bf16(self):
        yaml_paths, registry_world_size = _get_registry_yaml_extra(
            self.MODEL_NAME)
        if get_device_count() < registry_world_size:
            pytest.skip("Not enough devices for world size, skipping test")

        sampling_params = self.get_default_sampling_params()
        with AutoDeployLLM(model=self.MODEL_NAME,
                           tokenizer=self.MODEL_NAME,
                           world_size=registry_world_size,
                           yaml_extra=yaml_paths) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_params,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


class TestModelRegistryAccuracy(LlmapiAccuracyTestHarness):
    """Accuracy tests for models from the AutoDeploy model registry.

    Config = yaml_extra (merged) + config_overrides.
    Model paths are resolved via hf_id_to_local_model_dir.
    """
    # Aliases for models that have different names in the registry and the reference accuracy files.
    MODEL_REFERENCE_ALIASES = {
        "nvidia/Llama-3.1-8B-Instruct-FP8": "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/Llama-3.1-8B-Instruct-NVFP4":
        "meta-llama/Llama-3.1-8B-Instruct",
    }

    # Each param: (model_name, config_overrides, tasks). Marks skip when machine lacks GPUs/memory.
    MODEL_REGISTRY_ACCURACY_PARAMS = [
        pytest.param("meta-llama/Llama-3.1-8B-Instruct",
                     {"cuda_graph_config": {
                         "max_batch_size": 8
                     }}, [MMLU, GSM8K],
                     id="meta-llama_Llama-3.1-8B-Instruct"),
        pytest.param("nvidia/Llama-3.1-8B-Instruct-FP8", {}, [MMLU, GSM8K],
                     id="nvidia_Llama-3.1-8B-Instruct-FP8"),
        pytest.param("nvidia/Llama-3.1-8B-Instruct-NVFP4", {}, [MMLU, GSM8K],
                     id="nvidia_Llama-3.1-8B-Instruct-NVFP4"),
        pytest.param("google/gemma-3-1b-it",
                     {"cuda_graph_config": {
                         "max_batch_size": 8
                     }}, [MMLU, GSM8K],
                     id="google_gemma-3-1b-it"),
        pytest.param("mistralai/Ministral-8B-Instruct-2410",
                     {"cuda_graph_config": {
                         "max_batch_size": 8
                     }}, [MMLU, GSM8K],
                     id="mistralai_Ministral-8B-Instruct-2410"),
        pytest.param("mistralai/Codestral-22B-v0.1",
                     {"cuda_graph_config": {
                         "max_batch_size": 8
                     }}, [MMLU, GSM8K],
                     id="mistralai_Codestral-22B-v0.1"),
        pytest.param("nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                     {"cuda_graph_config": {
                         "max_batch_size": 8
                     }}, [MMLU, GSM8K],
                     id="nvidia_Llama-3.1-Nemotron-Nano-8B-v1"),
        pytest.param(
            "Qwen/QwQ-32B",
            {"cuda_graph_config": {
                "max_batch_size": 8
            }},
            [MMLU],
            marks=pytest.mark.skip_less_device_memory(80000),
            id="Qwen_QwQ-32B",
        ),
        pytest.param(
            "meta-llama/Llama-3.3-70B-Instruct",
            {},
            [MMLU, GSM8K],
            marks=pytest.mark.skip_less_device_memory(80000),
            id="meta-llama_Llama-3.3-70B-Instruct",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-0528",
            {},
            [MMLU, GSM8K],
            marks=(
                skip_pre_blackwell,
                pytest.mark.skip_less_device(8),
                pytest.mark.skip_less_device_memory(120000),
            ),
            id="deepseek-ai_DeepSeek-R1-0528",
        ),
    ]

    def get_default_sampling_params(self):
        # Use end_id=None so _setup runs and tokenizes stop sequences (e.g. GSM8K).
        return SamplingParams(end_id=None,
                              pad_id=None,
                              n=1,
                              use_beam_search=False)

    @pytest.mark.skip_less_device_memory(32000)
    @pytest.mark.parametrize("accuracy_check", [False, True])
    @pytest.mark.parametrize("model_name,config_overrides,tasks",
                             MODEL_REGISTRY_ACCURACY_PARAMS)
    def test_autodeploy_from_registry(self, model_name, config_overrides, tasks,
                                      accuracy_check):
        model_path = hf_id_to_local_model_dir(model_name)
        yaml_paths, registry_world_size = _get_registry_yaml_extra(model_name)
        effective_world_size = config_overrides.get("world_size",
                                                    registry_world_size)
        if get_device_count() < effective_world_size:
            pytest.skip("Not enough devices for world size, skipping test")
        sampling_params = self.get_default_sampling_params()

        with AutoDeployLLM(model=model_path,
                           tokenizer=model_path,
                           yaml_extra=yaml_paths,
                           **config_overrides) as llm:
            if accuracy_check:
                if "NVFP4" in model_name:
                    _set_quant_config(llm, "nvfp4")
                elif "FP8" in model_name:
                    _set_quant_config(llm, "fp8")
                elif model_name in {
                        "deepseek-ai/DeepSeek-R1",
                        "deepseek-ai/DeepSeek-R1-0528",
                }:
                    llm.args.quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                reference_model_name = self.MODEL_REFERENCE_ALIASES.get(
                    model_name, model_name)
                for task_cls in tasks:
                    task = task_cls(reference_model_name)
                    try:
                        evaluate_kwargs = {
                            "sampling_params": sampling_params
                        } if task_cls is MMLU else {}
                        task.evaluate(llm, **evaluate_kwargs)
                    except (AssertionError, RuntimeError, ValueError) as e:
                        raise type(e)(f"[{task_cls.__name__}] {e}") from None

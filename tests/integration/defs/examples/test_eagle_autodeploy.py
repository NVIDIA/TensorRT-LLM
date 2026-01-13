#!/usr/bin/env python3
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
"""Test loading Eagle model as ADEngine in AutoDeploy.

This test wraps the EagleModel in a HuggingFace-compatible interface
and runs it through the full AutoDeploy pipeline.

Usage:
    pytest test_eagle_autodeploy.py -v

Environment:
    LLM_MODELS_ROOT: Path to LLM models directory containing EAGLE3-LLaMA3.1-Instruct-8B
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from build_and_run_ad import ExperimentConfig, main
from defs.conftest import llm_models_root
from transformers import AutoConfig

# Import Eagle model from AutoDeploy's custom models
# This import triggers the registration of EagleConfig and EagleModelForCausalLM
from tensorrt_llm._torch.auto_deploy.models.custom import EagleModelForCausalLM
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleConfig
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm.llmapi import KvCacheConfig

EAGLE_MODEL_SUBPATH = "EAGLE3-LLaMA3.1-Instruct-8B"
LLAMA_MODEL_SUBPATH = "llama-3.1-model/Llama-3.1-8B-Instruct"


def get_model_paths():
    """Get model paths using llm_models_root()."""
    models_root = llm_models_root()
    eagle_model = os.path.join(models_root, EAGLE_MODEL_SUBPATH)
    llama_model = os.path.join(models_root, LLAMA_MODEL_SUBPATH)

    print(f"Eagle model path: {eagle_model}")
    print(f"Llama model path: {llama_model}")
    return eagle_model, llama_model


def get_eagle_config(eagle_model_path: str) -> EagleConfig:
    """Load Eagle config from checkpoint."""
    config_path = Path(eagle_model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Eagle config not found at {config_path}")

    print(f"Loading Eagle config from: {config_path}")

    # Load the base config
    base_config = AutoConfig.from_pretrained(config_path, attn_implementation="eager")

    # Convert to EagleConfig
    config = EagleConfig(
        draft_vocab_size=getattr(base_config, "draft_vocab_size", None),
        **{k: v for k, v in base_config.to_dict().items() if k != "draft_vocab_size"},
    )

    print(f"Created EagleConfig with num_hidden_layers={config.num_hidden_layers}")
    return config


def test_eagle_model_standalone_forward():
    """Test EagleModelForCausalLM forward pass with mock hidden states."""
    print("\n" + "=" * 80)
    print("Test: EagleModelForCausalLM standalone forward pass")
    print("=" * 80)

    eagle_model_path, _ = get_model_paths()

    if not Path(eagle_model_path).exists():
        pytest.skip(f"Eagle model not found at {eagle_model_path}")

    config = get_eagle_config(eagle_model_path)
    print(f"Config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EagleModelForCausalLM(config)
    model = model.to(device)
    model = model.to(torch.bfloat16)  # Use bf16 for AutoDeploy compatibility
    model.eval()

    # Create mock inputs
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Run forward pass
    with torch.inference_mode():
        output = model(input_ids=input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output.logits.shape}")

    # Verify output shape
    draft_vocab = getattr(config, "draft_vocab_size", config.vocab_size)
    expected_shape = (batch_size, seq_len, draft_vocab)
    assert output.logits.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.logits.shape}"
    )

    print("✅ Forward pass successful!")


def test_eagle_autodeploy_registration():
    """Test that EagleModelForCausalLM is properly registered with AutoDeploy factory."""
    print("\n" + "=" * 80)
    print("Test: EagleModelForCausalLM AutoDeploy registration")
    print("=" * 80)

    # The import of modeling_eagle.py should have auto-registered the model
    assert "EagleConfig" in AutoModelForCausalLMFactory._custom_model_mapping
    assert AutoModelForCausalLMFactory._custom_model_mapping["EagleConfig"] == EagleModelForCausalLM
    print("✅ EagleModelForCausalLM is registered with AutoDeploy factory!")


def test_eagle_autodeploy_full_pipeline():
    """Test full AutoDeploy pipeline with Eagle model.

    This test:
    1. Sets model_type="eagle3" in config.json to trigger EagleConfig
    2. Uses explicit tokenizer path pointing to Llama model
    3. Runs through the full AutoDeploy pipeline
    """
    print("\n" + "=" * 80)
    print("Test: Full AutoDeploy pipeline with Eagle")
    print("=" * 80)

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    eagle_model_path, llama_model_path = get_model_paths()

    if not Path(eagle_model_path).exists():
        pytest.skip(f"Eagle model not found at {eagle_model_path}")

    if not Path(llama_model_path).exists():
        pytest.skip(f"Llama model not found at {llama_model_path}")

    # Create a temporary directory with model_type="eagle3"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "eagle_model"
        tmp_model_path.mkdir()

        # Copy/symlink all files from original Eagle model
        eagle_path = Path(eagle_model_path)
        for item in eagle_path.iterdir():
            if item.name == "config.json":
                # Modify config.json to set model_type="eagle3"
                with open(item, "r") as f:
                    config_dict = json.load(f)
                config_dict["model_type"] = "eagle3"
                config_dict["num_hidden_layers"] = 1
                config_dict["torch_dtype"] = "bfloat16"  # Use bf16 for AutoDeploy compatibility
                with open(tmp_model_path / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                print("✅ Created config.json with model_type='eagle3'")
            else:
                # Symlink other files to save space
                (tmp_model_path / item.name).symlink_to(item)

        print(f"✅ Created temporary model directory: {tmp_model_path}")

        # Configure KV cache
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.01)

        # Configure AutoDeploy LLM arguments
        llm_args = {
            "model": str(tmp_model_path),
            "tokenizer": llama_model_path,  # Eagle shares tokenizer with target Llama model
            "skip_loading_weights": False,
            "speculative_config": None,
            "runtime": "trtllm",
            "world_size": 1,
            "kv_cache_config": kv_cache_config,
            "disable_overlap_scheduler": True,
            "max_num_tokens": 64,
            "dtype": "bfloat16",  # Use bf16 for AutoDeploy's custom kernels
        }

        # Configure experiment with a simple prompt
        experiment_config = {
            "args": llm_args,
            "benchmark": {"enabled": False},
            "prompt": {
                "batch_size": 1,
                "queries": ["Hello, how are you?"],
            },
        }

        # Create ExperimentConfig
        cfg = ExperimentConfig(**experiment_config)

        # Add sampling parameters
        cfg.prompt.sp_kwargs = {
            "max_tokens": 10,
            "top_k": None,
            "temperature": 0.0,
            "seed": 42,
        }

        print("Running AutoDeploy pipeline via ExperimentConfig...")

        # Run the experiment
        result = main(cfg)

        print("✅ AutoDeploy pipeline completed!")
        print(f"   - Result keys: {list(result.keys())}")
        if "prompts_and_outputs" in result:
            for prompt, output in result["prompts_and_outputs"]:
                print(f"   - Prompt: {prompt[:50]}...")
                print(f"   - Output: {output[:50]}..." if output else "   - Output: (empty)")

    print("✅ Full AutoDeploy pipeline test passed!")

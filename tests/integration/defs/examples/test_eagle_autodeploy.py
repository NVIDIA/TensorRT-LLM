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
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleConfig, EagleModel
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


def test_eagle_model_with_weights():
    """Test EagleModel forward pass with loaded weights.

    This test mirrors the main() function from run_eagle_model.py to validate
    that modeling_eagle.py is correct. It:
    1. Loads the Eagle config from checkpoint (using AutoConfig, not EagleConfig)
    2. Loads weights from pytorch_model.bin
    3. Runs forward pass with mock hidden states
    4. Verifies output shape matches expected draft vocabulary size
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel forward pass with loaded weights")
    print("=" * 80)

    eagle_model_path, _ = get_model_paths()
    eagle_path = Path(eagle_model_path)

    if not eagle_path.exists():
        pytest.skip(f"Eagle model not found at {eagle_model_path}")

    # 1. Setup Device & Dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # 2. Load Config (use AutoConfig like run_eagle_model.py, not EagleConfig)
    config_path = eagle_path / "config.json"
    print(f"Loading config from: {config_path}")
    config = AutoConfig.from_pretrained(config_path, attn_implementation="eager")

    # Patch num_hidden_layers to 1 (Eagle has single transformer layer)
    if config.num_hidden_layers > 1:
        print(f"ℹ️  Patching config: num_hidden_layers {config.num_hidden_layers} -> 1")
        config.num_hidden_layers = 1

    print(f"Config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")

    # 3. Load Weights
    bin_path = eagle_path / "pytorch_model.bin"
    if not bin_path.exists():
        pytest.skip(f"Weights not found at {bin_path}")

    print(f"Loading weights from: {bin_path}")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # 4. Create Model and Load Weights
    model = EagleModel(config)

    # Load weights (similar to EagleModel.load_weights in run_eagle_model.py)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    for missing_key in missing:
        if missing_key == "embed_tokens.weight":
            print(
                "Embed tokens weight is missing. This is expected - "
                "it should be loaded from the target model."
            )
        else:
            print(f"⚠️ Unexpected key missing in loaded weights: {missing_key}")

    for unexpected_key in unexpected:
        print(f"⚠️ Unexpected key: {unexpected_key}")

    model.to(device, dtype=dtype)
    model.eval()

    # 5. Create Mock Inputs
    batch_size = 1
    seq_len = 8
    hidden_dim = config.hidden_size

    # Mock hidden states (3 layers concatenated from target model)
    mock_hidden_states = torch.randn(
        (batch_size, seq_len, hidden_dim * 3), device=device, dtype=dtype
    )

    # Mock input IDs (using original vocab size before draft vocab patching)
    original_vocab_size = model._original_vocab_size
    input_ids = torch.randint(
        0, original_vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )

    # Position IDs
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    print("Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {mock_hidden_states.shape}")

    # 6. Run Forward Pass
    with torch.inference_mode():
        output_logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=mock_hidden_states,
        )

    print(f"Output shape: {output_logits.shape}")

    # 7. Verify Output Shape
    draft_vocab = config.draft_vocab_size or config.vocab_size
    expected_shape = (batch_size, seq_len, draft_vocab)
    assert output_logits.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output_logits.shape}"
    )

    print("✅ Forward pass with weights successful!")


def test_eagle_model_torch_export():
    """Test that EagleModel can be exported with torch.export.

    This validates that the model architecture is compatible with
    torch.export for potential TensorRT compilation.
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel torch.export")
    print("=" * 80)

    eagle_model_path, _ = get_model_paths()
    eagle_path = Path(eagle_model_path)

    if not eagle_path.exists():
        pytest.skip(f"Eagle model not found at {eagle_model_path}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Load config (use AutoConfig like run_eagle_model.py, not EagleConfig)
    config_path = eagle_path / "config.json"
    config = AutoConfig.from_pretrained(config_path, attn_implementation="eager")

    # Patch num_hidden_layers to 1 (Eagle has single transformer layer)
    if config.num_hidden_layers > 1:
        config.num_hidden_layers = 1

    model = EagleModel(config)

    # Load weights if available
    bin_path = eagle_path / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.to(device, dtype=dtype)
    model.eval()

    # Create inputs for export
    batch_size = 1
    seq_len = 8
    hidden_dim = config.hidden_size

    input_ids = torch.randint(
        0, model._original_vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    mock_hidden_states = torch.randn(
        (batch_size, seq_len, hidden_dim * 3), device=device, dtype=dtype
    )
    mock_attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)

    print("Export input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {mock_hidden_states.shape}")
    print(f"  attention_mask: {mock_attention_mask.shape}")

    example_args = (
        input_ids,
        position_ids,
        mock_hidden_states,
        mock_attention_mask,
    )

    # Attempt torch.export
    try:
        exported_program = torch.export.export(model, args=example_args)
        print("✅ torch.export successful!")
        print("Graph module code preview (first 20 lines):")
        code_lines = exported_program.graph_module.code.split("\n")[:20]
        print("\n".join(code_lines))
    except Exception as e:
        pytest.fail(f"torch.export failed: {e}")

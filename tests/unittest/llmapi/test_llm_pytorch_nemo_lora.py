"""Tests for NeMo LoRA checkpoint loading in PyTorch workflow.

This file contains fast unit tests that do not require full model initialization.
For integration tests that require full model loading and GPU inference, see:
    tests/integration/defs/llmapi/test_llm_pytorch_nemo_lora.py

Unit tests here should run in seconds, not minutes.
"""

import json
import tarfile
import tempfile
from pathlib import Path

import pytest
import torch

from tensorrt_llm.lora_manager import LoraConfig


def create_mock_nemo_lora_checkpoint(
    lora_dir: Path,
    hidden_size: int = 4096,
    num_layers: int = 32,
    lora_rank: int = 8,
    tp_size: int = 1,
) -> Path:
    """Create a minimal NeMo LoRA checkpoint for testing.

    This creates a .nemo tarfile with the expected structure:
    - model_weights.ckpt containing attn_qkv adapter weights
    - model_config.yaml with basic configuration

    Args:
        lora_dir: Directory to create the checkpoint in
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        lora_rank: LoRA rank
        tp_size: Tensor parallelism size

    Returns:
        Path to the created .nemo file
    """
    # Create temporary directory for checkpoint contents
    temp_dir = lora_dir / "temp_nemo"
    temp_dir.mkdir(exist_ok=True)

    # Create LoRA weights dict
    weights_dict = {}

    for layer_idx in range(num_layers):
        # NeMo uses this key format for QKV adapters
        key_prefix = f"model.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter"

        # Create linear_in weights [lora_rank, hidden_size]
        linear_in_key = f"{key_prefix}.linear_in.weight"
        weights_dict[linear_in_key] = torch.zeros(lora_rank,
                                                  hidden_size,
                                                  dtype=torch.float16)

        # Create linear_out weights [3 * hidden_size, lora_rank] for QKV combined
        linear_out_key = f"{key_prefix}.linear_out.weight"
        weights_dict[linear_out_key] = torch.zeros(3 * hidden_size,
                                                   lora_rank,
                                                   dtype=torch.float16)

    # Save checkpoint
    ckpt_path = temp_dir / "model_weights.ckpt"
    torch.save(weights_dict, ckpt_path)

    # Create minimal config
    config = {
        "precision": "fp16",
        "trainer": {
            "num_nodes": 1,
            "devices": tp_size,
        },
        "model": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
        "lora": {
            "rank": lora_rank,
            "target_modules": ["attn_qkv"],
        }
    }

    config_path = temp_dir / "model_config.yaml"
    # Using JSON for simplicity since YAML parsing isn't critical for the test
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # Create .nemo tarfile
    nemo_path = lora_dir / "test_lora.nemo"
    with tarfile.open(nemo_path, 'w') as tar:
        tar.add(ckpt_path, arcname="model_weights.ckpt")
        tar.add(config_path, arcname="model_config.yaml")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir)

    return nemo_path


# Test data for parametrized tests
NEMO_LORA_UNIT_TEST_PARAMS = [
    # (hidden_size, num_layers, lora_rank, description)
    (2048, 16, 8, "small_model_rank_8"),
    (4096, 32, 16, "large_model_rank_16"),
    (1024, 12, 4, "tiny_model_rank_4"),
]

LORA_RANK_CONFIGS = [
    # (lora_rank, max_lora_rank, description)
    (8, 8, "rank_8"),
    (16, 16, "rank_16"),
    (4, 8, "rank_4_max_8"),
]


class TestNemoLoraUnit:
    """Unit tests for NeMo LoRA loading without full model initialization."""

    @pytest.mark.parametrize("hidden_size,num_layers,lora_rank,description",
                             NEMO_LORA_UNIT_TEST_PARAMS)
    def test_nemo_lora_loader_creation(self, hidden_size, num_layers, lora_rank,
                                       description):
        """Test NemoLoraLoader creation with different model configurations."""
        from tensorrt_llm.lora_manager import NemoLoraLoader

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo checkpoint
            nemo_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lora_rank=lora_rank,
            )

            # Test NemoLoraLoader directly
            loader = NemoLoraLoader([str(nemo_path)])
            assert loader.is_valid, f"NemoLoraLoader failed to validate {nemo_path} for {description}"
            assert loader.lora_target_modules == [
                "attn_qkv"
            ], f"Expected attn_qkv modules for {description}"

    @pytest.mark.parametrize("lora_rank,max_lora_rank,description",
                             LORA_RANK_CONFIGS)
    def test_load_torch_nemo_lora_function(self, lora_rank, max_lora_rank,
                                           description):
        """Test load_torch_nemo_lora function with different LoRA rank configurations."""
        from tensorrt_llm.lora_manager import load_torch_nemo_lora

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo checkpoint
            nemo_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,
                num_layers=16,
                lora_rank=lora_rank,
            )

            # Test load_torch_nemo_lora
            lora_config = LoraConfig(
                lora_dir=[str(nemo_path)],
                lora_ckpt_source="nemo",
                max_lora_rank=max_lora_rank,
            )

            # This should not raise an error
            load_torch_nemo_lora(lora_config)

            # Verify configuration was set correctly
            assert lora_config.lora_target_modules == [
                "attn_qkv"
            ], f"Expected attn_qkv modules for {description}"
            assert lora_config.trtllm_modules_to_hf_modules == {
                "attn_qkv": "attn_qkv"
            }, f"Expected correct module mapping for {description}"

    def test_nemo_lora_unsupported_modules_validation(self):
        """Test validation of unsupported modules in NeMo LoRA."""
        from tensorrt_llm.lora_manager import load_torch_nemo_lora

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo checkpoint
            nemo_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,
                num_layers=16,
                lora_rank=8,
            )

            # Test validation: should fail with unsupported modules
            invalid_config = LoraConfig(
                lora_dir=[str(nemo_path)],
                lora_ckpt_source="nemo",
                lora_target_modules=["attn_qkv", "mlp_h_to_4h"
                                     ],  # mlp_h_to_4h not supported
                max_lora_rank=8,
            )

            with pytest.raises(ValueError, match="NeMo LoRA only supports"):
                load_torch_nemo_lora(invalid_config)

    def test_nemo_lora_empty_target_modules(self):
        """Test NeMo LoRA with empty target modules list."""
        from tensorrt_llm.lora_manager import load_torch_nemo_lora

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo checkpoint
            nemo_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,
                num_layers=16,
                lora_rank=8,
            )

            # Test with empty target modules - should auto-detect
            lora_config = LoraConfig(
                lora_dir=[str(nemo_path)],
                lora_ckpt_source="nemo",
                max_lora_rank=8,
            )

            load_torch_nemo_lora(lora_config)

            # Should auto-detect and set attn_qkv
            assert lora_config.lora_target_modules == ["attn_qkv"]

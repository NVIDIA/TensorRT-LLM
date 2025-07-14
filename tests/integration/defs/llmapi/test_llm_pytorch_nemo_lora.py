"""Integration tests for NeMo LoRA checkpoint loading in PyTorch workflow."""

import json
import tarfile
import tempfile
from pathlib import Path

import pytest
import torch
from defs.conftest import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


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
LORA_RANK_CONFIGS = [
    # (lora_rank, max_lora_rank, description)
    (8, 8, "rank_8"),
    (16, 16, "rank_16"),
    (4, 8, "rank_4_max_8"),
]


class TestNemoLoraIntegration:
    """Integration tests for NeMo LoRA with full model initialization."""

    @pytest.mark.parametrize("lora_rank,max_lora_rank,description",
                             LORA_RANK_CONFIGS)
    def test_llama_nemo_lora_inference(self, lora_rank, max_lora_rank,
                                       description):
        """Test NeMo LoRA inference with Llama model using different LoRA ranks."""
        model_dir = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B/"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo LoRA checkpoint
            nemo_lora_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,  # Llama 3.2 1B hidden size
                num_layers=16,  # Llama 3.2 1B layer count
                lora_rank=lora_rank,
            )

            # Configure LoRA with nemo source
            lora_config = LoraConfig(
                lora_dir=[str(nemo_lora_path)],
                lora_ckpt_source="nemo",
                lora_target_modules=["attn_qkv"],
                max_lora_rank=max_lora_rank,
            )

            # Create LLM instance with LoRA
            llm = LLM(
                model=model_dir,
                lora_config=lora_config,
                backend="pytorch",
            )

            try:
                # Test inference with LoRA
                prompts = ["Hello, how are you?"]
                sampling_params = SamplingParams(max_tokens=10)

                outputs = llm.generate(prompts, sampling_params)

                # Basic validation - should generate something
                assert len(outputs) == 1, f"Expected 1 output for {description}"
                assert len(outputs[0].outputs[0].text
                           ) > 0, f"Expected non-empty text for {description}"

                print(
                    f"[{description}] Generated text: {outputs[0].outputs[0].text}"
                )
            finally:
                # Ensure proper cleanup
                del llm
                import gc
                gc.collect()

    @pytest.mark.parametrize("prompt,max_tokens,description", [
        ("Hello, how are you?", 10, "greeting_short"),
        ("The weather today is", 20, "weather_medium"),
        ("Tell me about", 15, "question_medium"),
    ])
    def test_llama_nemo_lora_different_prompts(self, prompt, max_tokens,
                                               description):
        """Test NeMo LoRA with different prompts and generation lengths."""
        model_dir = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B/"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo LoRA checkpoint
            nemo_lora_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,
                num_layers=16,
                lora_rank=8,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                lora_dir=[str(nemo_lora_path)],
                lora_ckpt_source="nemo",
                lora_target_modules=["attn_qkv"],
                max_lora_rank=8,
            )

            # Create LLM instance with LoRA
            llm = LLM(
                model=model_dir,
                lora_config=lora_config,
                backend="pytorch",
            )

            try:
                # Test inference with different prompts
                prompts = [prompt]
                sampling_params = SamplingParams(max_tokens=max_tokens)

                outputs = llm.generate(prompts, sampling_params)

                # Validation
                assert len(outputs) == 1, f"Expected 1 output for {description}"
                generated_text = outputs[0].outputs[0].text
                assert len(generated_text
                           ) > 0, f"Expected non-empty text for {description}"

                # Basic sanity check - generated text should have reasonable length
                assert len(
                    generated_text.split()
                ) <= max_tokens + 5, f"Generated text too long for {description}"

                print(
                    f"[{description}] Prompt: '{prompt}' -> Generated: '{generated_text}'"
                )
            finally:
                # Ensure proper cleanup
                del llm
                import gc
                gc.collect()


class TestNemoLoraTensorParallel:
    """Tests for NeMo LoRA with tensor parallelism."""

    @pytest.mark.parametrize("tp_size,description", [
        (2, "tp_2"),
        (4, "tp_4"),
    ])
    def test_llama_nemo_lora_tensor_parallel(self, tp_size, description):
        """Test NeMo LoRA loading with different tensor parallelism sizes."""
        import torch
        if torch.cuda.device_count() < tp_size:
            pytest.skip(f"Test requires at least {tp_size} GPUs")

        model_dir = f"{llm_models_root()}/llama-models-v3/llama-3.2-1b-hf"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock NeMo LoRA checkpoint with specified TP size
            nemo_lora_path = create_mock_nemo_lora_checkpoint(
                temp_path,
                hidden_size=2048,
                num_layers=16,
                lora_rank=8,
                tp_size=tp_size,
            )

            # Configure LoRA
            lora_config = LoraConfig(
                lora_dir=[str(nemo_lora_path)],
                lora_ckpt_source="nemo",
                lora_target_modules=["attn_qkv"],
                max_lora_rank=8,
            )

            # Create LLM instance with tensor parallelism
            llm = LLM(
                model=model_dir,
                lora_config=lora_config,
                backend="pytorch",
                tensor_parallel_size=tp_size,
            )

            try:
                # Test inference
                prompts = ["The weather today is"]
                sampling_params = SamplingParams(max_tokens=20)

                outputs = llm.generate(prompts, sampling_params)

                assert len(outputs) == 1, f"Expected 1 output for {description}"
                assert len(outputs[0].outputs[0].text
                           ) > 0, f"Expected non-empty text for {description}"

                print(
                    f"[{description}] Generated text: {outputs[0].outputs[0].text}"
                )
            finally:
                # Ensure proper cleanup
                del llm
                import gc
                gc.collect()

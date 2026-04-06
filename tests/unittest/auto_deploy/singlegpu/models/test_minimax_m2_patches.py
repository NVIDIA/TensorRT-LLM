"""Test the MiniMax-M2 MoE forward used for export compatibility.

This verifies that the AutoDeploy MoE implementation in
``MiniMaxM2SparseMoeBlock.forward`` produces numerically equivalent hidden
states to the original HuggingFace implementation when applied to the same
module weights and inputs.
"""

import types

import pytest
import torch
from test_common.llm_data import hf_id_to_local_model_dir
from transformers import AutoConfig, AutoModelForCausalLM

# Import custom_ops to register torch.ops.auto_deploy.torch_moe
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_minimax_m2 import (
    MiniMaxM2SparseMoeBlock,
)


def _load_minimax_m2_moe_layer(model_name_or_path):
    """Load the MoE layer from MiniMax-M2 model with a minimal configuration.

    We create a small model to keep tests fast while still exercising the
    MoE routing and computation logic.

    Parameters:
        model_name_or_path (str): Path or name of the pretrained model.

    Returns:
        module: The MiniMaxM2SparseMoeBlock layer.
    """
    try:
        # Load only the model configuration
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Configure minimal model for fast testing
        config.num_hidden_layers = 1
        config.use_cache = False
        config.hidden_size = 16  # Small hidden size
        config.intermediate_size = 8  # For MLP within experts
        config.mlp_intermediate_size = 32
        config.num_local_experts = 4  # Small number of experts
        config.num_experts_per_tok = 2  # Top-k experts
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.router_jitter_noise = 0.0  # Disable jitter for deterministic tests

        # Build the model architecture (no weights loaded)
        # Note: Importing minimax_m2 module auto-patches from_config, so the
        # instance's forward is already patched. But the CLASS method is still
        # the original HF implementation, which we use as reference.
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        # Access the MoE layer
        layer_name = "model.layers.0.block_sparse_moe"
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            print(f"Layer '{layer_name}' not found in the model.")
        else:
            print(f"Successfully extracted layer '{layer_name}'.")
        return module
    except Exception as e:
        print(f"Error extracting layer: {e}")
        return None


def test_minimax_m2_moe_patch():
    """Test that the AutoDeploy MiniMaxM2SparseMoeBlock forward matches HF."""
    # Resolve model path at test time (not collection time) to avoid ValueError
    try:
        model_name = hf_id_to_local_model_dir("MiniMaxAI/MiniMax-M2")
    except (ValueError, FileNotFoundError):
        pytest.skip("MiniMax-M2 model not available locally")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Get HF MoE module from the remote-code model implementation.
    module = _load_minimax_m2_moe_layer(model_name)
    assert module is not None, "Failed to load MiniMax-M2 MoE layer"

    # Convert module to bfloat16 to match input dtype
    module = module.to(torch.bfloat16)

    # Create test input - same input will be used for both original and patched
    # hidden_size=16 matches the config in _load_minimax_m2_moe_layer
    hidden_size = 16
    inputs = torch.randn(2, 6, hidden_size, dtype=torch.bfloat16)

    # The class method on the loaded module is the original HF implementation.
    original_class_forward = type(module).forward

    # Generate reference output using the original HF class method.
    with torch.no_grad():
        ref_output, _ = original_class_forward(module, inputs)

    # Replace the instance forward with the AutoDeploy implementation.
    module.forward = types.MethodType(MiniMaxM2SparseMoeBlock.forward, module)

    # Generate test output using the AutoDeploy implementation.
    with torch.no_grad():
        test_output = module(inputs)

    # Final hidden states should be very close
    # (small tolerance for different computation order in torch_moe)
    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-3,
        rtol=1e-3,
        msg="Output mismatch between original and patched MoE",
    )

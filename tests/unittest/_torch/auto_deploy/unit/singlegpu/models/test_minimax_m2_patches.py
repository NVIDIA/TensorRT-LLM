"""Testing module patches that enable export of MiniMax-M2 model.

This test verifies that the patched MiniMaxM2SparseMoeBlock forward function
produces identical outputs to the original HuggingFace implementation.
"""

import types

import pytest
import torch
from test_common.llm_data import hf_id_to_local_model_dir
from transformers import AutoConfig, AutoModelForCausalLM

# Import custom_ops to register torch.ops.auto_deploy.torch_moe
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.minimax_m2 import minimax_m2_moe


def _load_minimax_m2_moe_layer(model_name_or_path):
    """
    Loads the MoE layer from MiniMax-M2 model with a minimal configuration.

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


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(
            hf_id_to_local_model_dir("MiniMaxAI/MiniMax-M2"),
        ),
    ],
)
def test_minimax_m2_moe_patch(model_name):
    """
    Test that the patched MiniMaxM2SparseMoeBlock forward produces the same
    output as the original HuggingFace implementation.

    The patch rewrites the forward to use torch.ops.auto_deploy.torch_moe
    for torch.export compatibility while maintaining numerical equivalence.

    Since importing minimax_m2.py auto-patches module instances, we use the
    CLASS method (type(module).forward) as the original HF reference.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Get MoE module (instance is already patched by import side-effect)
    module = _load_minimax_m2_moe_layer(model_name)
    assert module is not None, "Failed to load MiniMax-M2 MoE layer"

    # Convert module to bfloat16 to match input dtype
    module = module.to(torch.bfloat16)

    # Create test input - same input will be used for both original and patched
    # hidden_size=16 matches the config in _load_minimax_m2_moe_layer
    hidden_size = 16
    inputs = torch.randn(2, 6, hidden_size, dtype=torch.bfloat16)

    # The CLASS method is still the original HuggingFace implementation
    # (the auto-patch only patches instance methods, not the class)
    original_class_forward = type(module).forward

    # Generate reference output using original HF class method
    # Uses: same module weights, same input tensor
    with torch.no_grad():
        ref_output, ref_router_logits = original_class_forward(module, inputs)

    # The instance forward is already patched by the import side-effect,
    # but let's be explicit and apply our patch function directly
    module.forward = types.MethodType(minimax_m2_moe, module)

    # Generate test output using patched implementation
    # Uses: same module weights, same input tensor
    with torch.no_grad():
        test_output, test_router_logits = module(inputs)

    # Verify outputs match
    # Router logits should be identical (same computation path)
    torch.testing.assert_close(
        ref_router_logits,
        test_router_logits,
        atol=1e-5,
        rtol=1e-5,
        msg="Router logits mismatch between original and patched MoE",
    )

    # Final hidden states should be very close
    # (small tolerance for different computation order in torch_moe)
    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-3,
        rtol=1e-3,
        msg="Output mismatch between original and patched MoE",
    )

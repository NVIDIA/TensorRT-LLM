"""Testing module patches that enable export of GLM4-MoE model."""

import types

import pytest
import torch
from _model_test_utils import get_small_model_kwargs
from transformers import AutoConfig, AutoModelForCausalLM

# Import to register torch_moe custom op
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.torch_moe import torch_moe  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.glm4_moe import glm4_moe_forward


def _load_moe_layer_from_glm4():
    """
    Loads the GLM4-MoE MoE layer with a small config for testing.

    Returns:
        module: The Glm4MoeMoE module
    """
    try:
        model_id = "zai-org/GLM-4.7"

        # Load the model configuration
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # Apply shared small-config overrides (reusable across unit tests)
        for k, v in get_small_model_kwargs(model_id).items():
            setattr(config, k, v)

        # Build the model architecture (no weights loaded)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        # Access the MoE layer (layer 0 will be MoE since first_k_dense_replace=0)
        moe_layer = model.model.layers[0].mlp
        print(f"Successfully extracted MoE layer: {type(moe_layer).__name__}")
        return moe_layer
    except Exception as e:
        print(f"Error extracting layer: {e}")
        raise


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        pytest.param(1, 4, id="batch1_seq4"),
        pytest.param(2, 8, id="batch2_seq8"),
    ],
)
def test_glm4_moe_patch(batch_size, seq_len):
    """Test that the GLM4-MoE patch produces numerically identical results."""
    # Get the MoE module
    module = _load_moe_layer_from_glm4()
    hidden_size = 32  # Matches config

    # Create test input
    inputs = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Generate reference output (without patch)
    with torch.inference_mode():
        ref_output = module(inputs)

    # Patch the layer
    module.forward = types.MethodType(glm4_moe_forward, module)

    # Generate patched output
    with torch.inference_mode():
        patched_output = module(inputs)

    # Verify outputs are numerically close
    torch.testing.assert_close(
        ref_output,
        patched_output,
        rtol=1e-5,
        atol=1e-5,
        msg=lambda m: f"GLM4-MoE patch output mismatch:\n{m}",
    )

"""Testing module patches that enable export of Qwen3Next MoE.

This test verifies that the patched Qwen3NextSparseMoeBlock forward function
produces identical outputs to the original HuggingFace implementation.
"""

import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Register torch.ops.auto_deploy.torch_moe
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.qwen3_next import _forward_moe


def _load_qwen3_next_moe_layer():
    """Build a tiny Qwen3Next model and extract the MoE block.

    We create a small model to keep tests fast while still exercising the
    MoE routing, expert computation, shared expert, and sigmoid gating logic.

    Returns:
        module: The Qwen3NextSparseMoeBlock layer.
    """
    try:
        config = AutoConfig.for_model("qwen3_next")

        # Minimal dimensions for fast testing
        config.num_hidden_layers = 1
        config.use_cache = False
        config.hidden_size = 16
        config.intermediate_size = 8  # dense MLP intermediate
        config.moe_intermediate_size = 8  # routed expert intermediate
        config.shared_expert_intermediate_size = 8
        config.num_experts = 4
        config.num_experts_per_tok = 2
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.head_dim = 8  # 2 heads * 8 = 16 = hidden_size
        config.decoder_sparse_step = 1  # every layer is MoE
        config.layer_types = ["full_attention"]  # 1 layer, full attention type
        config.norm_topk_prob = True

        # Linear attention params (unused for full_attention layers but required by config)
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
        config.linear_key_head_dim = 8
        config.linear_value_head_dim = 8
        config.linear_conv_kernel_dim = 4

        # Build the model architecture (no weights loaded -- random init)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        # The MoE block is at model.layers.0.mlp
        layer_name = "model.layers.0.mlp"
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            print(f"Layer '{layer_name}' not found in the model.")
        else:
            print(f"Successfully extracted layer '{layer_name}'.")
        return module
    except Exception as e:
        print(f"Error extracting layer: {e}")
        return None


def test_qwen3_next_moe_patch():
    """Verify the patched Qwen3NextSparseMoeBlock produces the same output
    as the original HuggingFace implementation.

    The patch rewrites the forward to use torch.ops.auto_deploy.torch_moe
    for torch.export compatibility while maintaining numerical equivalence.
    """
    torch.manual_seed(42)

    module = _load_qwen3_next_moe_layer()
    assert module is not None, "Failed to load Qwen3Next MoE layer"

    # Convert module to bfloat16 to match inference dtype
    module = module.to(torch.bfloat16)

    # Create test input: (batch_size=2, seq_len=6, hidden_size=16)
    hidden_size = 16
    inputs = torch.randn(2, 6, hidden_size, dtype=torch.bfloat16)

    # Reference: original HF forward returns (hidden_states, router_logits)
    with torch.no_grad():
        ref_output, ref_router_logits = type(module).forward(module, inputs)

    # Patched: our _forward_moe also returns (hidden_states, router_logits)
    module.forward = types.MethodType(_forward_moe, module)
    with torch.no_grad():
        test_output, test_router_logits = module(inputs)

    # Router logits should be identical (same computation path)
    torch.testing.assert_close(
        ref_router_logits,
        test_router_logits,
        atol=1e-5,
        rtol=1e-5,
        msg="Router logits mismatch between original and patched Qwen3Next MoE",
    )

    # Final hidden states should be very close
    # (small tolerance for different computation order in torch_moe)
    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-3,
        rtol=1e-3,
        msg="Output mismatch between original and patched Qwen3Next MoE",
    )

"""Testing GDN (Gated Delta Net) patches for Qwen3Next.

This test verifies that:
1. The `torch_gated_delta_rule` custom op produces the same output as the HF
   `torch_chunk_gated_delta_rule` reference implementation.
2. The patched `Qwen3NextGatedDeltaNet.forward` produces the same output as the
   original HuggingFace implementation.

Reference HF modeling file (v4.57.1):
  https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py
"""

import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule as hf_torch_chunk_gated_delta_rule,
)

# Register all auto_deploy custom ops (torch_gated_delta_rule, torch_causal_conv1d, torch_l2norm)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.qwen3_next import _patched_gdn_forward


def test_torch_gated_delta_rule_op():
    """Verify the `torch_gated_delta_rule` custom op produces the same output
    as the HF `torch_chunk_gated_delta_rule` function.

    Both operate on pure-torch math (no FLA kernels). We compare with
    `use_qk_l2norm_in_kernel=False` so L2 norm is excluded from both paths.
    """
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 128
    num_heads = 4
    k_head_dim = 16
    v_head_dim = 16

    # Inputs in [B, S, H, D] layout (bsnd convention)
    q = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.float32)
    g = -torch.rand(batch_size, seq_len, num_heads, dtype=torch.float32)  # negative (decay)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32))

    # L2 normalize Q and K (as our patched forward does externally)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    # Reference: HF torch implementation (no l2norm inside, since we did it externally)
    with torch.no_grad():
        ref_output, _ = hf_torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False
        )

    # Test: our custom op
    with torch.no_grad():
        test_output = torch.ops.auto_deploy.torch_gated_delta_rule(q, k, v, g, beta)

    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-4,
        rtol=1e-4,
        msg="Output mismatch between HF torch_chunk_gated_delta_rule and auto_deploy::torch_gated_delta_rule",
    )


def test_torch_gated_delta_rule_op_bfloat16():
    """Verify the custom op works correctly with bfloat16 inputs."""
    torch.manual_seed(123)

    batch_size = 1
    seq_len = 64
    num_heads = 2
    k_head_dim = 8
    v_head_dim = 8

    q = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.bfloat16)
    g = -torch.rand(batch_size, seq_len, num_heads, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, dtype=torch.bfloat16))

    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    with torch.no_grad():
        ref_output, _ = hf_torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False
        )

    with torch.no_grad():
        test_output = torch.ops.auto_deploy.torch_gated_delta_rule(q, k, v, g, beta)

    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-2,
        rtol=1e-2,
        msg="Output mismatch for bfloat16 between HF and auto_deploy gated delta rule",
    )


def _load_qwen3_next_gdn_layer():
    """Build a tiny Qwen3Next model with a linear_attention layer and extract the GDN block.

    Returns:
        module: The Qwen3NextGatedDeltaNet layer.
    """
    config = AutoConfig.for_model("qwen3_next")

    # Minimal dimensions for fast testing
    config.num_hidden_layers = 1
    config.use_cache = False
    config.hidden_size = 32
    config.intermediate_size = 16
    config.moe_intermediate_size = 16
    config.shared_expert_intermediate_size = 16
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 8
    config.decoder_sparse_step = 1
    config.norm_topk_prob = True

    # Use a single linear_attention layer to get the GDN block
    config.layer_types = ["linear_attention"]

    # Linear attention params
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.linear_conv_kernel_dim = 4

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    # The GDN block is at model.layers.0.linear_attn
    layer_name = "model.layers.0.linear_attn"
    module = dict(model.named_modules()).get(layer_name)
    assert module is not None, f"Layer '{layer_name}' not found in the model"
    assert isinstance(module, Qwen3NextGatedDeltaNet), (
        f"Expected Qwen3NextGatedDeltaNet, got {type(module)}"
    )
    return module


def _force_torch_fallbacks(module):
    """Force the GDN module to use pure-torch fallbacks instead of FLA/causal_conv1d kernels.

    This ensures the reference forward uses the same algorithmic path as our
    patched forward, so we can compare with tight tolerances.
    """
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule as hf_chunk,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_recurrent_gated_delta_rule as hf_recurrent,
    )

    module.causal_conv1d_fn = None
    module.chunk_gated_delta_rule = hf_chunk
    module.recurrent_gated_delta_rule = hf_recurrent


def test_qwen3_next_gdn_patch():
    """Verify the patched Qwen3NextGatedDeltaNet.forward produces the same
    output as the original HuggingFace implementation.

    The patch replaces the forward with autodeploy custom ops
    (torch_causal_conv1d, torch_l2norm, torch_gated_delta_rule) while
    maintaining numerical equivalence.
    """
    torch.manual_seed(42)

    module = _load_qwen3_next_gdn_layer()

    # Force torch fallbacks for the reference path so both sides use pure torch
    _force_torch_fallbacks(module)

    # Convert to bfloat16 to match typical inference dtype
    module = module.to(torch.bfloat16)

    hidden_size = 32
    inputs = torch.randn(2, 16, hidden_size, dtype=torch.bfloat16)

    # Reference: original HF forward (with torch fallbacks, no cache)
    with torch.no_grad():
        ref_output = type(module).forward(module, inputs)

    # Patched: our _patched_gdn_forward
    module.forward = types.MethodType(_patched_gdn_forward, module)
    with torch.no_grad():
        test_output = module(inputs)

    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-2,
        rtol=1e-2,
        msg="Output mismatch between original and patched Qwen3Next GDN forward",
    )


def test_qwen3_next_gdn_patch_float32():
    """Same as above but in float32 for tighter tolerance checks."""
    torch.manual_seed(42)

    module = _load_qwen3_next_gdn_layer()
    _force_torch_fallbacks(module)

    # Stay in float32 for tighter tolerances
    module = module.to(torch.float32)

    hidden_size = 32
    inputs = torch.randn(2, 16, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        ref_output = type(module).forward(module, inputs)

    module.forward = types.MethodType(_patched_gdn_forward, module)
    with torch.no_grad():
        test_output = module(inputs)

    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-4,
        rtol=1e-4,
        msg="Output mismatch (float32) between original and patched Qwen3Next GDN forward",
    )

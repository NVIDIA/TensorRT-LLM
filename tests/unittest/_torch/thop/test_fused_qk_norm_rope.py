import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding


def torch_ref_rms_norm_rope(qkv,
                            position_ids,
                            num_heads_q,
                            num_heads_k,
                            num_heads_v,
                            head_dim,
                            eps=1e-5,
                            base=10000.0):
    """
    PyTorch reference implementation of RMSNorm+RoPE for verification.

    Uses TensorRT-LLM's own RMSNorm and RotaryEmbedding modules to ensure consistency
    with the expected behavior of the fused kernel.

    Args:
        qkv: Combined QKV tensor of shape [num_tokens, hidden_size]
        position_ids: Position IDs for RoPE of shape [num_tokens]
        num_heads_q: Number of query heads
        num_heads_k: Number of key heads
        num_heads_v: Number of value heads (unused for normalization/RoPE but needed for tensor splitting)
        head_dim: Dimension of each head
        eps: Epsilon value for RMS normalization
        base: Base value for RoPE calculations

    Returns:
        Combined tensor with Q and K parts normalized and RoPE applied
    """
    # Get input shape information
    num_tokens = qkv.shape[0]
    hidden_size = qkv.shape[1]

    # Calculate dimensions for Q, K, V segments
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim

    # Verify dimensions match
    assert hidden_size == q_size + k_size + v_size, f"Hidden size {hidden_size} doesn't match Q+K+V dimensions {q_size + k_size + v_size}"

    # Split the tensor into Q, K, V parts
    q = qkv[:, :q_size]
    k = qkv[:, q_size:q_size + k_size]
    v = qkv[:, q_size + k_size:]

    # Create and apply RMSNorm modules
    # No trainable parameters needed, just pure normalization
    q_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)

    # Initialize weights to 1.0 (no scaling, just normalization)
    q_norm.weight.data.fill_(1.0)
    k_norm.weight.data.fill_(1.0)

    # Apply RMSNorm to Q and K
    q_normalized = q_norm(q.reshape(num_tokens * num_heads_q,
                                    head_dim)).reshape(num_tokens, q_size)
    k_normalized = k_norm(k.reshape(num_tokens * num_heads_k,
                                    head_dim)).reshape(num_tokens, k_size)

    # Create and apply RotaryEmbedding module
    rope_params = RopeParams(
        dim=head_dim,  # Set the rotary dimension to match the head dimension
        theta=base,  # Base value for RoPE calculations
        max_positions=8192  # Large enough for any reasonable sequence length
    )
    # TODO: support is_neox=True
    rotary_emb = RotaryEmbedding(rope_params=rope_params,
                                 head_dim=head_dim,
                                 is_neox=False).to(qkv.device)

    # Apply RoPE to the normalized Q and K
    [q_rope, k_rope] = rotary_emb(position_ids,
                                  [q_normalized, k_normalized])


    # Combine Q, K, V back together
    result = torch.cat([q_rope, k_rope, v], dim=1)

    return result


head_dims = [64, 128]
# (Q heads, K heads, V heads)
num_heads_groups = [
    (16, 8, 8),  # Qwen3-0.6B, Qwen3-1.7B
    (32, 8, 8),  # Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B
    (40, 8, 8),  # Qwen3-14B
    (64, 8, 8)   # Qwen3-32B, Qwen3-235B-A22B
]
num_tokens_list = [3, 8, 32, 256]
dtypes = [torch.bfloat16]  # TODO: support float16


@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads_group", num_heads_groups)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("dtype", dtypes)
def test_fused_qk_norm_rope(head_dim, num_heads_group, num_tokens, dtype):
    """
    Test the fused QK RMSNorm + RoPE operation with various configurations.

    This test verifies that the fused kernel correctly applies:
    1. RMSNorm to both query (Q) and key (K) portions of the QKV tensor
    2. Rotary Position Embeddings (RoPE) to the normalized Q and K
    3. Leaves the value (V) portion unchanged

    Args:
        head_dim: Dimension of each attention head
        num_heads_group: Tuple of (num_heads_q, num_heads_k, num_heads_v)
        num_tokens: Number of tokens to process
        dtype: Data type (float16 or bfloat16)
    """
    device = "cuda"
    torch_dtype = dtype

    # Unpack head counts
    num_heads_q, num_heads_k, num_heads_v = num_heads_group

    # Calculate total hidden dimension
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    # Generate random inputs directly as 2D [num_tokens, hidden_size]
    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch_dtype, device=device)
    qkv_copy = qkv.clone()

    # Generate position IDs with +100 offset to test decoding scenarios
    position_ids = torch.arange(num_tokens, dtype=torch.int32,
                                device=device) + 100

    # Set RMSNorm and RoPE parameters
    eps = 1e-5
    base = 10000.0

    # Run the custom fusedQKNormRope operation
    torch.ops.trtllm.fused_qk_norm_rope(qkv, position_ids, num_heads_q,
                                        num_heads_k, num_heads_v, head_dim, eps,
                                        base)
    # Check for CUDA errors
    torch.cuda.synchronize()
    output = qkv  # This op is inplace

    # Compute reference output using TensorRT-LLM modules
    ref_output = torch_ref_rms_norm_rope(qkv_copy, position_ids, num_heads_q,
                                         num_heads_k, num_heads_v, head_dim,
                                         eps, base)

    # Set tolerance based on dtype - bfloat16 needs higher tolerance
    atol = 1e-2 if dtype == torch.float16 else 1e-1  # Higher tolerance for bfloat16

    # breakpoint()
    # Split output into q,k,v components based on head counts
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    num_heads_v * head_dim

    q = output[2, :q_size]
    k = output[2, q_size:q_size + k_size]
    v = output[2, q_size + k_size:]

    q_ref = ref_output[2, :q_size]
    k_ref = ref_output[2, q_size:q_size + k_size]
    v_ref = ref_output[2, q_size + k_size:]

    q_orig = qkv_copy[2, :q_size]
    k_orig = qkv_copy[2, q_size:q_size + k_size]
    v_orig = qkv_copy[2, q_size + k_size:]

    print("Query:", q)
    print("Query Ref:", q_ref)
    print("Query Orig:", q_orig)
    print("Key:", k)
    print("Key Ref:", k_ref)
    print("Key Orig:", k_orig)
    print("Value:", v)
    print("Value Ref:", v_ref)
    print("Value Orig:", v_orig)

    # Compare outputs from custom kernel vs reference implementation
    torch.testing.assert_close(
        output,
        ref_output,
        rtol=1e-2,
        atol=atol,
        msg=
        f"Failed with head_dim={head_dim}, num_heads=({num_heads_q},{num_heads_k},{num_heads_v}), num_tokens={num_tokens}, dtype={dtype}"
    )

    print("Passed")

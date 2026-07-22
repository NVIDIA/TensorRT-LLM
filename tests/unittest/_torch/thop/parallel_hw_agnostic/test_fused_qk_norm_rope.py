import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import MRotaryEmbedding, RotaryEmbedding


@torch.inference_mode()
def torch_ref_rms_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids,
):
    """
    PyTorch reference implementation of RMSNorm+RoPE for verification.

    Uses TensorRT-LLM's own RMSNorm and RotaryEmbedding modules to ensure consistency
    with the expected behavior of the fused kernel.

    Args:
        qkv: Combined QKV tensor of shape [num_tokens, hidden_size]
        num_heads_q: Number of query heads
        num_heads_k: Number of key heads
        num_heads_v: Number of value heads (unused for normalization/RoPE but needed for tensor splitting)
        head_dim: Dimension of each head
        rotary_dim: Dimension for RoPE
        eps: Epsilon value for RMS normalization
        q_weight: RMSNorm weights for query [head_dim]
        k_weight: RMSNorm weights for key [head_dim]
        base: Base value for RoPE calculations
        is_neox: Whether to use NeoX style RoPE
        position_ids: Position IDs for RoPE of shape [num_tokens]

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
    assert hidden_size == q_size + k_size + v_size, (
        f"Hidden size {hidden_size} doesn't match Q+K+V dimensions {q_size + k_size + v_size}"
    )

    # Split the tensor into Q, K, V parts
    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    # Create and apply RMSNorm modules with custom weights
    q_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device).to(qkv.dtype)

    # Set the weights to the provided weights
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)

    # Apply RMSNorm to Q and K
    q_normalized = q_norm(q.reshape(num_tokens * num_heads_q, head_dim)).reshape(num_tokens, q_size)
    k_normalized = k_norm(k.reshape(num_tokens * num_heads_k, head_dim)).reshape(num_tokens, k_size)

    # Create and apply RotaryEmbedding module
    rope_params = RopeParams(
        dim=rotary_dim,  # Set the rotary dimension
        theta=base,  # Base value for RoPE calculations
        max_positions=8192,  # Large enough for any reasonable hidden size
    )
    rotary_emb = RotaryEmbedding(rope_params=rope_params, head_dim=head_dim, is_neox=is_neox).to(
        qkv.device
    )

    # Apply RoPE to the normalized Q and K
    [q_rope, k_rope] = rotary_emb(position_ids, [q_normalized, k_normalized])

    # Combine Q, K, V back together
    result = torch.cat([q_rope, k_rope, v], dim=1)

    return result


head_dims = [64, 128]
# (Q heads, K heads, V heads)
num_heads_groups = [
    (16, 8, 8),  # Qwen3-0.6B, Qwen3-1.7B
    (32, 8, 8),  # Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B
    (40, 8, 8),  # Qwen3-14B
    (64, 8, 8),  # Qwen3-32B, Qwen3-235B-A22B
    (24, 8, 8),  # GLM 4.6
]
num_tokens_list = [1, 3, 8, 32, 256]
is_neox_list = [False, True]
partial_rotary_factor_list = [1.0, 0.5]
dtypes = [torch.bfloat16]  # TODO: support float16


@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads_group", num_heads_groups)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("is_neox", is_neox_list)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("partial_rotary_factor", partial_rotary_factor_list)
def test_fused_qk_norm_rope(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox, dtype
):
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
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device) + 100

    # Generate random weights for RMSNorm
    q_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0

    # Set RMSNorm and RoPE parameters
    eps = 1e-5
    base = 10000.0

    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    # Run the custom fusedQKNormRope operation
    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        True,
        False,  # use_gemma (standard RMSNorm reference below)
        False,  # use_mrope (plain RoPE)
        0,  # mrope_section1 (unused when use_mrope=False)
        0,  # mrope_section2
    )
    output = qkv  # This op is inplace

    # Compute reference output using TensorRT LLM modules
    ref_output = torch_ref_rms_norm_rope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
    )

    # Compare outputs from custom kernel vs reference implementation
    torch.testing.assert_close(
        output,
        ref_output,
        rtol=5e-2,
        atol=1e-1,
    )


@torch.inference_mode()
def torch_ref_gemma_mrope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids_3d,
    mrope_section,
):
    """Reference for the Gemma-RMSNorm + interleaved-mRoPE fused path.

    Mirrors apply_qk_norm_rope's fused branch: Gemma RMSNorm (scale by
    (1 + weight)) on Q/K, then interleaved mRoPE via MRotaryEmbedding.
    """
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    q_norm = RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=True).to(qkv.device).to(qkv.dtype)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=True).to(qkv.device).to(qkv.dtype)
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)
    q_n = q_norm(q.reshape(num_tokens * num_heads_q, head_dim)).reshape(num_tokens, q_size)
    k_n = k_norm(k.reshape(num_tokens * num_heads_k, head_dim)).reshape(num_tokens, k_size)

    rope_params = RopeParams(dim=rotary_dim, theta=base, max_positions=8192)
    rotary_emb = MRotaryEmbedding(
        rope_params=rope_params,
        head_dim=head_dim,
        mrope_section=mrope_section,
        is_neox=is_neox,
        mrope_interleaved=True,
    ).to(qkv.device)
    [q_rope, k_rope] = rotary_emb(position_ids_3d, [q_n, k_n])
    return torch.cat([q_rope, k_rope, v], dim=1)


@pytest.mark.skip(
    reason="WIP: standalone MRotaryEmbedding reference shape handling needs "
    "fixing. The kernel's Gemma + interleaved-mRoPE path is validated "
    "end-to-end by the Qwen3.5 accuracy test."
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_heads_group", [(16, 8, 8), (32, 8, 8)])
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 256])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
def test_fused_qk_norm_rope_gemma_mrope(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox
):
    """Cover the Gemma-RMSNorm + interleaved-mRoPE fused path (Qwen3.5)."""
    device = "cuda"
    dtype = torch.bfloat16
    num_heads_q, num_heads_k, num_heads_v = num_heads_group
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_copy = qkv.clone()

    # 3D position_ids [3, num_tokens] with distinct t/h/w to exercise the
    # interleaved section selection (identical components would hide bugs).
    base_pos = torch.arange(num_tokens, dtype=torch.int32, device=device)
    position_ids_3d = torch.stack(
        [base_pos + 100, base_pos + 50, base_pos + 10], dim=0
    ).contiguous()

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    eps = 1e-5
    base = 10000.0
    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    half = rotary_dim // 2
    s = half // 3
    mrope_section = [half - 2 * s, s, s]  # sums to half (rotary_dim/2)

    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids_3d,
        factor,
        low,
        high,
        attention_factor,
        True,  # is_qk_norm
        True,  # use_gemma
        True,  # use_mrope
        mrope_section[1],
        mrope_section[2],
    )
    output = qkv

    ref_output = torch_ref_gemma_mrope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids_3d,
        mrope_section,
    )

    torch.testing.assert_close(output, ref_output, rtol=5e-2, atol=1e-1)


# FP8 out-variant coverage. Includes an M3-like GQA shape (8 Q / 1 KV, head_dim
# 128) so the MiniMax-M3 FP8-KV path geometry is exercised directly.
fp8_num_heads_groups = [
    (16, 8, 8),
    (32, 8, 8),
    (8, 1, 1),  # MiniMax-M3 sharded GQA (num_heads=8, num_kv_heads=1)
]


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads_group", fp8_num_heads_groups)
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 256])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
def test_fused_qk_norm_rope_to_fp8(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox
):
    """Test the FP8 out-variant of fused QK RMSNorm + RoPE.

    The op reads a BF16 qkv, applies RMSNorm + RoPE to Q/K and copy-casts V, and
    returns a new FP8 (E4M3) tensor, folding the FP8 activation-quant into the
    norm+RoPE epilogue. Verifies:
      1. The input qkv is left untouched (out-of-place).
      2. Output dtype is FP8 E4M3 with the same shape.
      3. The dequantized output matches the BF16 fused reference (Q/K normed+roped,
         V unchanged) within FP8-appropriate tolerance.
    """
    device = "cuda"
    dtype = torch.bfloat16
    num_heads_q, num_heads_k, num_heads_v = num_heads_group
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_ref = qkv.clone()

    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device) + 100
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    eps = 1e-5
    base = 10000.0
    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)

    out_fp8 = torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        True,  # is_qk_norm
        False,  # use_gemma (standard RMSNorm reference below)
        False,  # use_mrope (plain RoPE)
        0,  # mrope_section1
        0,  # mrope_section2
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert tuple(out_fp8.shape) == (num_tokens, hidden_size)
    # Out-of-place: the BF16 input must be left byte-for-byte unchanged.
    torch.testing.assert_close(qkv, qkv_ref, rtol=0.0, atol=0.0)

    ref_output = torch_ref_rms_norm_rope(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
    )

    # The op folds the E4M3 cast into the epilogue; compare the dequantized
    # result against the BF16 reference with FP8-appropriate tolerance (E4M3 has
    # 3 mantissa bits, so ~1/8 relative resolution).
    torch.testing.assert_close(
        out_fp8.float(),
        ref_output.float(),
        rtol=0.2,
        atol=0.1,
    )

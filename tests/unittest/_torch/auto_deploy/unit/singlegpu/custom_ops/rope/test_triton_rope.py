from typing import Optional, Tuple

import pytest
import torch
from _custom_op_utils import torch_rope_reference

# Import after we've imported torch (to ensure custom ops are registered)
from tensorrt_llm._torch.auto_deploy.custom_ops import triton_rope  # noqa: F401


def _precompute_cos_sin_cache(
    max_seq_len: int, head_dim: int, rope_theta: float = 10000.0, dtype: torch.dtype = torch.float16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin cache for RoPE (DeepSeek-style)."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [max_seq_len, head_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, head_dim]
    cos_cache = emb.cos().to(dtype)
    sin_cache = emb.sin().to(dtype)
    return cos_cache, sin_cache


def _precompute_freqs_cis(
    seq_len: int, head_dim: int, rope_theta: Optional[float] = None
) -> torch.Tensor:
    if rope_theta is None:
        rope_theta = 1e4
    freqs = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # cos and sin (real and img) are packed
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


@pytest.mark.parametrize("d_head", [16, 96])
def test_rope(d_head):
    SEQ_LEN = 4
    N_ELEM = d_head

    input_position = torch.tensor([10], dtype=torch.int32, device="cuda")
    freqs_cis = _precompute_freqs_cis(1024, N_ELEM)
    print(freqs_cis.shape)

    x = torch.randn((1, SEQ_LEN, 8, N_ELEM), dtype=torch.float16)
    y_ref = torch_rope_reference(x, freqs_cis, input_position)
    freqs_cis = freqs_cis.to("cuda")
    x_reshaped = x.unflatten(-1, (N_ELEM // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()
    y = torch.ops.auto_deploy.triton_rope_with_input_pos(
        x_reshaped.to("cuda"), freqs_cis, input_position, "bsnd"
    )
    y_reshaped = y.unflatten(-1, (2, N_ELEM // 2)).transpose(-2, -1).flatten(-2).contiguous()
    assert torch.allclose(y_ref.cpu(), y_reshaped.cpu(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("d_head", [16, 96])
def test_rope_flattened(d_head):
    SEQ_LENS = [4, 16, 28]
    N_ELEM = d_head

    freqs_cis = _precompute_freqs_cis(1024, N_ELEM).to("cuda")

    input_position = torch.tensor([0] * len(SEQ_LENS), device="cuda")
    x = []
    y_ref = []
    for i, s in enumerate(SEQ_LENS):
        tmp = torch.randn((1, s, 8, N_ELEM), dtype=torch.float16, device="cuda")
        y_ref.append(torch_rope_reference(tmp, freqs_cis, input_position[i]))
        x.append(tmp)
    x = torch.cat(x, 1).squeeze()  # [B*S,...]
    y_ref = torch.cat(y_ref, 1).squeeze()

    x_reshaped = x.unflatten(-1, (N_ELEM // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()

    seq_lens = torch.tensor(SEQ_LENS, device="cuda")
    seq_start_indices = torch.zeros(len(SEQ_LENS), dtype=torch.int32, device="cuda")
    seq_start_indices[1:] = torch.cumsum(seq_lens[:-1], 0)

    y = torch.ops.auto_deploy.triton_rope_on_flattened_inputs(
        x_reshaped.to("cuda"), freqs_cis, input_position, seq_lens, seq_start_indices
    )
    y_reshaped = y.unflatten(-1, (2, N_ELEM // 2)).transpose(-2, -1).flatten(-2).contiguous()

    assert torch.allclose(y_ref.cpu(), y_reshaped.cpu(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_q_heads,num_k_heads,head_dim",
    [
        (1, 4, 8, 8, 64),  # Standard case
        (2, 16, 20, 1, 128),  # GLM-4 style: non-power-of-2 heads, MQA
        (1, 1, 8, 8, 64),  # Single token (decode)
        (4, 32, 16, 2, 96),  # GQA with non-standard head_dim
    ],
)
def test_triton_rope_on_interleaved_qk_inputs(
    batch_size: int, seq_len: int, num_q_heads: int, num_k_heads: int, head_dim: int
):
    """
    Test that triton_rope_on_interleaved_qk_inputs produces the same output as
    the PyTorch reference (index + torch_rope_with_qk_interleaving).
    """
    device = "cuda"
    dtype = torch.bfloat16
    max_seq_len = 1024

    # Create random inputs with interleaved layout [B, S, H, D]
    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_k_heads, head_dim, device=device, dtype=dtype)

    # Precompute cos/sin cache
    cos_cache, sin_cache = _precompute_cos_sin_cache(max_seq_len, head_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    # Random position IDs (not necessarily sequential)
    position_ids = torch.randint(0, max_seq_len - seq_len, (batch_size,), device=device)
    position_ids = position_ids.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)
    # position_ids: [B, S]

    # ========== PyTorch Reference ==========
    # Step 1: Index cos/sin with position_ids
    cos_indexed = cos_cache[position_ids]  # [B, S, D]
    sin_indexed = sin_cache[position_ids]  # [B, S, D]

    # Step 2: Apply PyTorch rope with qk interleaving
    # unsqueeze_dim=2 for [B, S, H, D] layout
    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos_indexed, sin_indexed, unsqueeze_dim=2
    )

    # ========== Triton Implementation ==========
    q_triton, k_triton = torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs(
        q, k, cos_cache, sin_cache, position_ids
    )

    # ========== Compare Outputs ==========
    # Use relative tolerance for bf16
    atol = 1e-2
    rtol = 1e-2

    assert torch.allclose(q_ref, q_triton, atol=atol, rtol=rtol), (
        f"Q mismatch: max diff = {(q_ref - q_triton).abs().max().item()}"
    )
    assert torch.allclose(k_ref, k_triton, atol=atol, rtol=rtol), (
        f"K mismatch: max diff = {(k_ref - k_triton).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rope_interleaved_dtype_consistency(dtype):
    """Test that the Triton kernel works correctly with different dtypes."""
    device = "cuda"
    batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64
    max_seq_len = 1024

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    cos_cache, sin_cache = _precompute_cos_sin_cache(max_seq_len, head_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # PyTorch reference
    cos_indexed = cos_cache[position_ids]
    sin_indexed = sin_cache[position_ids]
    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos_indexed, sin_indexed, unsqueeze_dim=2
    )

    # Triton
    q_triton, k_triton = torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs(
        q, k, cos_cache, sin_cache, position_ids
    )

    # Verify outputs match
    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(q_ref, q_triton, atol=atol, rtol=rtol)
    assert torch.allclose(k_ref, k_triton, atol=atol, rtol=rtol)

    # Verify output dtype is preserved
    assert q_triton.dtype == dtype
    assert k_triton.dtype == dtype

from typing import Tuple

import flashinfer
import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401

torch.manual_seed(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@pytest.mark.parametrize("head_dim", [64, 256])  # head_dim must be a multiple of 64
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 1e-4, 1e-4),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],  # q/k must be in half precision
)
def test_flashinfer_custom_op_and_hf_impl(dtype, atol, rtol, head_dim):
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3

    # Prepare rotary embedding values.
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
    cos_vals = torch.cos(angles)  # [seq_len, head_dim//2]
    sin_vals = torch.sin(angles)  # [seq_len, head_dim//2]

    # For direct FlashInfer call: non-interleaved cache [seq_len, head_dim] (concatenated).
    cos_sin_cache = torch.cat([cos_vals, sin_vals], dim=1)
    # For HF and the custom op: duplicated layout [seq_len, head_dim].
    cos_new = torch.cat([cos_vals, cos_vals], dim=-1)
    sin_new = torch.cat([sin_vals, sin_vals], dim=-1)

    query = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)

    # Direct FlashInfer kernel call.
    query_flat = query.view(batch * seq_len, n_head * head_dim)
    key_flat = key.view(batch * seq_len, n_head * head_dim)
    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])
    q_flash, k_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, query_flat, key_flat, head_dim, cos_sin_cache, is_neox=True
    )
    q_flash = q_flash.view(batch, seq_len, n_head, head_dim)
    k_flash = k_flash.view(batch, seq_len, n_head, head_dim)

    # HF implementation using apply_rotary_pos_emb.
    # HF expects [batch, n_head, seq_len, head_dim] for unsqueeze_dim=1
    q_for_hf = query.transpose(1, 2).clone()
    k_for_hf = key.transpose(1, 2).clone()
    cos_expand = cos_new.unsqueeze(0).expand(batch, -1, -1)  # [batch, seq_len, head_dim]
    sin_expand = sin_new.unsqueeze(0).expand(batch, -1, -1)  # [batch, seq_len, head_dim]
    q_hf, k_hf = apply_rotary_pos_emb(q_for_hf, k_for_hf, cos_expand, sin_expand, unsqueeze_dim=1)

    # Convert outputs to [batch, seq_len, n_head, head_dim]
    q_hf = q_hf.transpose(1, 2).to(dtype)
    k_hf = k_hf.transpose(1, 2).to(dtype)

    # Custom op call
    custom_q, custom_k = torch.ops.rope.flashinfer(query, key, positions, cos_sin_cache, True)

    torch.testing.assert_close(q_hf, q_flash, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_hf, k_flash, rtol=rtol, atol=atol)
    torch.testing.assert_close(q_hf, custom_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_hf, custom_k, rtol=rtol, atol=atol)


# Version 2: complex multiplication approach
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # Expected shape: (B, seq, head_dim//2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@pytest.mark.parametrize("head_dim", [64, 256])  # Must be a multiple of 64
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 1e-5, 1e-5),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],  # q/k must be in half precision
)
def test_flashinfer_custom_op_and_complex_impl(dtype, atol, rtol, head_dim):
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3

    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(0)  # shape: (seq_len, head_dim//2)
    freqs_cis = torch.polar(torch.ones((seq_len, head_dim // 2), device=device), angles)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)  # shape: (B, seq, head_dim//2)

    query = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)

    out_q_v2, out_k_v2 = apply_rotary_emb(query, key, freqs_cis)

    cos_from_freqs = torch.real(freqs_cis)  # (B, seq, head_dim//2)
    sin_from_freqs = torch.imag(freqs_cis)  # (B, seq, head_dim//2)
    cos_sin_cache = torch.cat([cos_from_freqs, sin_from_freqs], dim=-1)[0]  # (seq, head_dim))

    # q/k of llama4 rope is interleaved
    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])
    custom_q, custom_k = torch.ops.rope.flashinfer(query, key, positions, cos_sin_cache, False)

    torch.testing.assert_close(out_q_v2, custom_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k_v2, custom_k, rtol=rtol, atol=atol)


# Copy of TritonWithFlattenedInputs._precompute_freqs_cis
def precompute_freqs_cis_interleaved(
    seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Precompute interleaved cosine and sine frequency cache for rotary position embeddings (RoPE).

    Returns a tensor of shape [seq_len, head_dim//2, 2], where the last dimension
    alternates [cos, sin] values for each rotary frequency.
        cache[s, i, 0] == cos(position=s · inv_freq[i])
        cache[s, i, 1] == sin(position=s · inv_freq[i]).
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device)
    angles = t.unsqueeze(1) * inv_freq.unsqueeze(0)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype)


@pytest.mark.parametrize("layout", ["bsnd", "bnsd"])
@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-4, 1e-4),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],
)
def test_triton_custom_op_and_hf_impl(layout, head_dim, dtype, atol, rtol):
    device = "cuda"
    batch, seq_len, n_head = 2, 4, 3

    # build cache and per-batch zero positions
    cosin_cache = precompute_freqs_cis_interleaved(seq_len, head_dim, dtype, device)  # [S, D/2, 2]
    positions = torch.zeros(batch, dtype=torch.int32, device=device)

    if layout == "bsnd":
        q = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
        unsq = 2
    else:  # "bnsd"
        q = torch.randn(batch, n_head, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, n_head, seq_len, head_dim, dtype=dtype, device=device)
        unsq = 1

    # build HF float32 cos/sin full tensors
    cos_f32 = cosin_cache[..., 0].to(torch.float32)  # [S, H/2]
    sin_f32 = cosin_cache[..., 1].to(torch.float32)  # [S, H/2]
    cos_full = torch.cat([cos_f32, cos_f32], dim=1)  # [S, H]
    sin_full = torch.cat([sin_f32, sin_f32], dim=1)  # [S, H]
    cos_exp = cos_full.unsqueeze(0).expand(batch, -1, -1)  # [B, S, H]
    sin_exp = sin_full.unsqueeze(0).expand(batch, -1, -1)  # [B, S, H]

    # HF reference in float32, then cast back
    q_f32, k_f32 = apply_rotary_pos_emb(
        q.to(torch.float32), k.to(torch.float32), cos_exp, sin_exp, unsqueeze_dim=unsq
    )
    q_hf = q_f32.to(dtype)
    k_hf = k_f32.to(dtype)

    q_out = torch.ops.rope.apply_rope_with_input_pos(q, cosin_cache, positions, layout)
    k_out = torch.ops.rope.apply_rope_with_input_pos(k, cosin_cache, positions, layout)

    torch.testing.assert_close(q_hf, q_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_hf, k_out, atol=atol, rtol=rtol)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L339
def apply_rotary_pos_emb_ds(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings by interleaving Q/K ,
    indexing cos/sin tables with position_ids, and returning rotated q, k.
    cos:  [seq_len, head_dim]
    sin:  [seq_len, head_dim]
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def inverse_interleave_permute_for_rotary(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    x = x.view(b, h, s, 2, d // 2)
    x = x.transpose(4, 3)
    return x.reshape(b, h, s, d)


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ds_impl_and_hf_impl(dtype, head_dim):
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
    cos_vals = torch.cos(angles)  # [seq_len, head_dim//2]
    sin_vals = torch.sin(angles)  # [seq_len, head_dim//2]
    # duplicate to shape [seq_len, head_dim]
    cos_new = torch.cat([cos_vals, cos_vals], dim=-1)
    sin_new = torch.cat([sin_vals, sin_vals], dim=-1)

    query = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)

    # HF torch expects inputs of shape [B, N, S, D]
    q_for_hf = query.transpose(1, 2).clone()
    k_for_hf = key.transpose(1, 2).clone()
    cos_expand = cos_new.unsqueeze(0).expand(batch, -1, -1)  # [batch, seq_len, head_dim]
    sin_expand = sin_new.unsqueeze(0).expand(batch, -1, -1)  # [batch, seq_len, head_dim]
    q_rotated_hf, k_rotated_hf = apply_rotary_pos_emb(
        q_for_hf, k_for_hf, cos_expand, sin_expand, unsqueeze_dim=1
    )
    q_rotated_hf = q_rotated_hf.transpose(1, 2).to(torch.float32)
    k_rotated_hf = k_rotated_hf.transpose(1, 2).to(torch.float32)

    q_for_hf2 = inverse_interleave_permute_for_rotary(q_for_hf.clone())
    k_for_hf2 = inverse_interleave_permute_for_rotary(k_for_hf.clone())

    q_rotated_hf2, k_rotated_hf2 = apply_rotary_pos_emb_ds(
        q_for_hf2, k_for_hf2, cos_new, sin_new, position_ids, unsqueeze_dim=1
    )

    atol = 1e-3 if dtype == torch.float16 else 1e-2
    rtol = 1e-3 if dtype == torch.float16 else 1e-2

    torch.testing.assert_close(q_rotated_hf2.transpose(1, 2), q_rotated_hf, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_rotated_hf2.transpose(1, 2), k_rotated_hf, rtol=rtol, atol=atol)

import flashinfer
import pytest
import torch
from _model_test_utils import (
    apply_rotary_pos_emb_complex,
    apply_rotary_pos_emb_ds,
    apply_rotary_pos_emb_explicit,
)

import tensorrt_llm._torch.auto_deploy  # noqa: F401

torch.manual_seed(1234)


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
    """
    Verify FlashInfer's Neox RoPE kernel against HF's apply_rotary_pos_emb:
    - Q/K: [B, S, N, D] non-interleaved half-precision.
    - cos_sin_cache: [S, D] = [cos||sin] concatenated.
    - HF path: Q/K → [B, N, S, D], cos_new/sin_new: [S, D] duplicated, then broadcast to [B, S, D].
    """
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
    cos_sin_cache_expand = (
        cos_sin_cache.unsqueeze(0).expand(batch, -1, -1).contiguous().view(batch * seq_len, -1)
    )  # [batch * seq_len, head_dim]
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
    q_hf, k_hf = apply_rotary_pos_emb_explicit(
        q_for_hf, k_for_hf, cos_expand, sin_expand, unsqueeze_dim=1
    )

    # Convert outputs to [batch, seq_len, n_head, head_dim]
    q_hf = q_hf.transpose(1, 2).to(dtype)
    k_hf = k_hf.transpose(1, 2).to(dtype)

    # Custom op call
    positions_flat = torch.arange(batch * seq_len, device=device)
    custom_q, custom_k = torch.ops.auto_deploy.flashinfer_rope(
        query, key, positions_flat, cos_sin_cache_expand, True
    )

    torch.testing.assert_close(q_hf, q_flash, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_hf, k_flash, rtol=rtol, atol=atol)
    torch.testing.assert_close(q_hf, custom_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_hf, custom_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize("head_dim", [64, 256])  # Must be a multiple of 64
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 1e-4, 1e-4),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],  # q/k must be in half precision
)
def test_flashinfer_custom_op_and_complex_impl(dtype, atol, rtol, head_dim):
    """
    Check FlashInfer's RoPE matches the complex-multiplication approach:
    - Q/K: [B, S, N, D] non-interleaved half-precision.
    - freqs_cis: [B, S, D/2] complex polar values.
    - flashinfer uses cos_sin_cache: [S, D] interleaved from real/imag of freqs_cis.
    """
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

    out_q_v2, out_k_v2 = apply_rotary_pos_emb_complex(query, key, freqs_cis)

    cos_from_freqs = torch.real(freqs_cis)  # (B, seq, head_dim//2)
    sin_from_freqs = torch.imag(freqs_cis)  # (B, seq, head_dim//2)
    cos_sin_cache = torch.cat([cos_from_freqs, sin_from_freqs], dim=-1)[0]  # (seq, head_dim))
    cos_sin_cache_expand = (
        cos_sin_cache.unsqueeze(0).expand(batch, -1, -1).contiguous().view(batch * seq_len, -1)
    )  # [batch * seq_len, head_dim]

    # q/k of llama4 rope is interleaved
    positions_flat = torch.arange(batch * seq_len, device=device)
    custom_q, custom_k = torch.ops.auto_deploy.flashinfer_rope(
        query, key, positions_flat, cos_sin_cache_expand, False
    )

    torch.testing.assert_close(out_q_v2, custom_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k_v2, custom_k, rtol=rtol, atol=atol)


# Copy of TritonAttention._precompute_freqs_cis
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
    """
    Validate custom Triton apply_rope_with_input_pos against HF's apply_rotary_pos_emb:
    - Q/K: layout 'bsnd'→[B,S,N,D] or 'bnsd'→[B,N,S,D], non-interleaved half-precision.
    - cosin_cache: [S, D/2, 2] interleaved [cos,sin].
    - HF path: cos_full/sin_full: [S, D] then expanded to [B, S, D].
    """
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
    q_f32, k_f32 = apply_rotary_pos_emb_explicit(
        q.to(torch.float32), k.to(torch.float32), cos_exp, sin_exp, unsqueeze_dim=unsq
    )
    q_hf = q_f32.to(dtype)
    k_hf = k_f32.to(dtype)

    q_out = torch.ops.auto_deploy.triton_rope_with_input_pos(q, cosin_cache, positions, layout)
    k_out = torch.ops.auto_deploy.triton_rope_with_input_pos(k, cosin_cache, positions, layout)

    torch.testing.assert_close(q_hf, q_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_hf, k_out, atol=atol, rtol=rtol)


def inverse_interleave_permute_for_rotary(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    x = x.view(b, h, s, 2, d // 2)
    x = x.transpose(4, 3)
    return x.reshape(b, h, s, d)


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-4, 1e-4),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],
)
def test_ds_impl_and_hf_impl(dtype, head_dim, atol, rtol):
    """
    Ensure Deepseek's interleaved-Q/K RoPE matches HF apply_rotary_pos_emb:
    - DS Q/K: [B, N, S, D] channel-interleaved in last dim.
    - cos_new/sin_new: [S, D] duplicated real values.
    - HF path: Q/K → [B,N,S,D], cos_expand/sin_expand: [B,S,D], unsqueezed at dim=1.
    """
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
    q_rotated_hf, k_rotated_hf = apply_rotary_pos_emb_explicit(
        q_for_hf, k_for_hf, cos_expand, sin_expand, unsqueeze_dim=1
    )
    q_rotated_hf = q_rotated_hf.transpose(1, 2).to(torch.float32)
    k_rotated_hf = k_rotated_hf.transpose(1, 2).to(torch.float32)

    q_for_hf2 = inverse_interleave_permute_for_rotary(q_for_hf.clone())
    k_for_hf2 = inverse_interleave_permute_for_rotary(k_for_hf.clone())

    # adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L134
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_ds = emb.cos()
    sin_ds = emb.sin()

    torch.testing.assert_close(cos_ds, cos_new, rtol=rtol, atol=atol)
    torch.testing.assert_close(sin_ds, sin_new, rtol=rtol, atol=atol)

    q_rotated_hf2, k_rotated_hf2 = apply_rotary_pos_emb_ds(
        q_for_hf2, k_for_hf2, cos_new, sin_new, position_ids, unsqueeze_dim=1
    )

    torch.testing.assert_close(q_rotated_hf2.transpose(1, 2), q_rotated_hf, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_rotated_hf2.transpose(1, 2), k_rotated_hf, rtol=rtol, atol=atol)


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.bfloat16, 1e-4, 1e-4),
        (torch.float16, 5e-4, 5e-4),
    ],
    ids=["bfloat16", "float16"],
)
def test_flashinfer_custom_op_strided_interleaved(dtype, atol, rtol, head_dim):
    """
    Verify FlashInfer's RoPE handles non-contiguous (strided) q/k inputs
    with is_neox=False (interleaved mode), matching contiguous-input results
    and complex-multiplication reference.
    """
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3
    extra_dim = 16

    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(0)
    freqs_cis = torch.polar(torch.ones((seq_len, head_dim // 2), device=device), angles)
    freqs_cis_batch = freqs_cis.unsqueeze(0).expand(batch, -1, -1)

    # Complex-frequency cos_sin_cache: [cos_from_freqs || sin_from_freqs]
    cos_from_freqs = torch.real(freqs_cis)
    sin_from_freqs = torch.imag(freqs_cis)
    cos_sin_cache = torch.cat([cos_from_freqs, sin_from_freqs], dim=-1)  # [seq, head_dim]
    cos_sin_cache_expand = (
        cos_sin_cache.unsqueeze(0).expand(batch, -1, -1).contiguous().view(batch * seq_len, -1)
    )

    # Create strided tensors via split
    full_q = torch.randn(batch, seq_len, n_head, head_dim + extra_dim, dtype=dtype, device=device)
    full_k = torch.randn(batch, seq_len, n_head, head_dim + extra_dim, dtype=dtype, device=device)
    q_strided, _ = torch.split(full_q, [head_dim, extra_dim], dim=-1)
    k_strided, _ = torch.split(full_k, [head_dim, extra_dim], dim=-1)

    assert not q_strided.is_contiguous()
    assert not k_strided.is_contiguous()

    q_contig = q_strided.contiguous()
    k_contig = k_strided.contiguous()

    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])

    # Strided path (is_neox=False)
    custom_q_strided, custom_k_strided = torch.ops.auto_deploy.flashinfer_rope(
        q_strided, k_strided, positions, cos_sin_cache_expand, False
    )

    # Contiguous path
    custom_q_contig, custom_k_contig = torch.ops.auto_deploy.flashinfer_rope(
        q_contig, k_contig, positions, cos_sin_cache_expand, False
    )

    # Complex reference
    out_q_ref, out_k_ref = apply_rotary_pos_emb_complex(q_contig, k_contig, freqs_cis_batch)

    # Strided matches contiguous
    torch.testing.assert_close(custom_q_strided, custom_q_contig, rtol=rtol, atol=atol)
    torch.testing.assert_close(custom_k_strided, custom_k_contig, rtol=rtol, atol=atol)
    # Both match complex reference
    torch.testing.assert_close(custom_q_strided, out_q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(custom_k_strided, out_k_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("has_bias", [True, False])
def test_rope_deinterleave_load_hook(has_bias):
    """
    Test _rope_deinterleave_load_hook permutes weights correctly:
    - q_b_proj: nope portion unchanged, rope portion de-interleaved by perm
    - kv_a_proj: first kv_lora_rank rows unchanged, last qk_rope_head_dim rows permuted
    - bias (when present): same split+permute pattern as kv_a_proj weight
    """
    from tensorrt_llm._torch.auto_deploy.models.custom.mla_rope_utils import (
        _rope_deinterleave_load_hook,
    )

    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    num_heads = 4
    kv_lora_rank = 512
    q_lora_rank = 128
    hidden_size = 256
    num_layers = 1

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])

    state_dict = {}
    layer_prefix = "model.layers.0.self_attn."

    # q_b_proj.weight: [num_heads * qk_head_dim, q_lora_rank]
    q_weight = torch.randn(num_heads * qk_head_dim, q_lora_rank)
    state_dict[layer_prefix + "q_b_proj.weight"] = q_weight.clone()

    # kv_a_proj_with_mqa.weight: [kv_lora_rank + qk_rope_head_dim, hidden_size]
    kv_weight = torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size)
    state_dict[layer_prefix + "kv_a_proj_with_mqa.weight"] = kv_weight.clone()

    if has_bias:
        kv_bias = torch.randn(kv_lora_rank + qk_rope_head_dim)
        state_dict[layer_prefix + "kv_a_proj_with_mqa.bias"] = kv_bias.clone()

    _rope_deinterleave_load_hook(
        state_dict,
        "",
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        num_layers=num_layers,
    )

    # Check q_b_proj: nope portion unchanged, rope portion permuted
    q_out = state_dict[layer_prefix + "q_b_proj.weight"]
    q_out_3d = q_out.view(num_heads, qk_head_dim, -1)
    q_orig_3d = q_weight.view(num_heads, qk_head_dim, -1)

    torch.testing.assert_close(
        q_out_3d[:, :qk_nope_head_dim, :], q_orig_3d[:, :qk_nope_head_dim, :]
    )
    torch.testing.assert_close(
        q_out_3d[:, qk_nope_head_dim:, :], q_orig_3d[:, qk_nope_head_dim:, :][:, perm, :]
    )

    # Check kv_a_proj: first kv_lora_rank rows unchanged, last qk_rope_head_dim rows permuted
    kv_out = state_dict[layer_prefix + "kv_a_proj_with_mqa.weight"]
    torch.testing.assert_close(kv_out[:kv_lora_rank, :], kv_weight[:kv_lora_rank, :])
    torch.testing.assert_close(kv_out[kv_lora_rank:, :], kv_weight[kv_lora_rank:, :][perm, :])

    if has_bias:
        kv_bias_out = state_dict[layer_prefix + "kv_a_proj_with_mqa.bias"]
        torch.testing.assert_close(kv_bias_out[:kv_lora_rank], kv_bias[:kv_lora_rank])
        torch.testing.assert_close(kv_bias_out[kv_lora_rank:], kv_bias[kv_lora_rank:][perm])

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
def test_flashinfer_and_custom_rope_ops(dtype, atol, rtol, head_dim):
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
def test_flashinfer_complex_rotary(dtype, atol, rtol, head_dim):
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

import flashinfer
import pytest
import torch

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


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# An older version with position_ids and interleaves input internally
def apply_rotary_pos_emb_w_pos_ids(q, k, cos, sin, position_ids, unsqueeze_dim=1):
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


@pytest.mark.parametrize(
    "head_dim", [64, 256]
)  # Flashinfer op requires head_dim to be a multiple of 64
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16]
)  # Flashinfer op requires Q/K tensors to be half precision
def test_rope_ops(dtype, head_dim):
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3

    positions = torch.cat(
        [torch.arange(seq_len, device=device) for _ in range(batch)]
    )  # For FlashInfer
    position_ids = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    )  # For HF

    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
    cos_vals = torch.cos(angles)  # [seq_len, head_dim//2]
    sin_vals = torch.sin(angles)  # [seq_len, head_dim//2]

    # Prepare cache for FlashInfer op: non-interleaved concatenation [cos_vals, sin_vals]
    cos_sin_cache = torch.cat([cos_vals, sin_vals], dim=1)

    # Prepare schedule for HF implementations: duplicate to shape [seq_len, head_dim]
    cos_new = torch.cat([cos_vals, cos_vals], dim=-1)
    sin_new = torch.cat([sin_vals, sin_vals], dim=-1)

    # Generate random query and key tensors.
    query = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)

    # FlashInfer expects flattened inputs of shape [B*S, N*D]
    query_flat = query.view(batch * seq_len, n_head * head_dim)
    key_flat = key.view(batch * seq_len, n_head * head_dim)
    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, query_flat, key_flat, head_dim, cos_sin_cache, is_neox=True
    )
    query_rotated_flash = query_rotated_flash.view(batch, seq_len, n_head, head_dim)
    key_rotated_flash = key_rotated_flash.view(batch, seq_len, n_head, head_dim)

    # HF torch expects inputs of shape [B, N, S, D]
    q_for_hf = query.transpose(1, 2).clone()  # shape: [batch, n_head, seq_len, head_dim]
    k_for_hf = key.transpose(1, 2).clone()
    q_rotated_hf, k_rotated_hf = apply_rotary_pos_emb(
        q_for_hf, k_for_hf, cos_new, sin_new, unsqueeze_dim=0
    )
    q_rotated_hf = q_rotated_hf.transpose(1, 2).to(torch.float32)
    k_rotated_hf = k_rotated_hf.transpose(1, 2).to(torch.float32)

    # HF implementation using positional IDs
    q_for_hf2 = inverse_interleave_permute_for_rotary(q_for_hf.clone())
    k_for_hf2 = inverse_interleave_permute_for_rotary(k_for_hf.clone())

    q_rotated_hf2, k_rotated_hf2 = apply_rotary_pos_emb_w_pos_ids(
        q_for_hf2, k_for_hf2, cos_new, sin_new, position_ids, unsqueeze_dim=1
    )

    atol = 1e-3 if dtype == torch.float16 else 1e-2
    rtol = 1e-3 if dtype == torch.float16 else 1e-2

    torch.testing.assert_close(
        q_rotated_hf, query_rotated_flash.to(torch.float32), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        k_rotated_hf, key_rotated_flash.to(torch.float32), rtol=rtol, atol=atol
    )

    torch.testing.assert_close(q_rotated_hf2.transpose(1, 2), q_rotated_hf, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_rotated_hf2.transpose(1, 2), k_rotated_hf, rtol=rtol, atol=atol)

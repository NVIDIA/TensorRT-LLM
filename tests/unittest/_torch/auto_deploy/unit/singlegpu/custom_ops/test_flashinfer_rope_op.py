import flashinfer
import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.torch_attention import apply_rotary_pos_emb

torch.manual_seed(0)


@pytest.mark.parametrize(
    "head_dim", [64, 256]
)  # Flashinfer op requires head_dim to be multiple of 64
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16]
)  # Flashinfer op requires Q K tensors to be half precision
@pytest.mark.parametrize("random_input_pos", [False, True])
def test_rope_ops(random_input_pos, dtype, head_dim):
    device = "cuda"
    batch = 2
    seq_len = 4
    n_head = 3

    if random_input_pos:
        input_pos = torch.randint(low=0, high=100, size=(batch,), device=device, dtype=torch.int32)
        # For FlashInfer, positions for every token: each sequence's token positions are offset by input_pos.
        positions_list = [
            input_pos[i].item() + torch.arange(seq_len, device=device) for i in range(batch)
        ]
        positions = torch.cat(positions_list)
        # Maximum sequence length for the frequency cache must cover the highest token index.
        max_seq_len = input_pos.max().item() + seq_len
    else:
        input_pos = torch.zeros(batch, dtype=torch.int32, device=device)
        # For FlashInfer: positions for every token, shape: (batch * seq_len,)
        positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])
        max_seq_len = seq_len

    # For the flattened op: each sequence has a length and a start index.
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    seq_start_indices = torch.arange(
        0, batch * seq_len, step=seq_len, dtype=torch.int32, device=device
    )

    # --- Precompute cosine-sine cache ---
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions_range = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = positions_range.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: [max_seq_len, head_dim/2]
    cos_vals = torch.cos(angles)  # Shape: [max_seq_len, head_dim/2]
    sin_vals = torch.sin(angles)  # Shape: [max_seq_len, head_dim/2]

    # For the custom op: create interleaved frequencies [cos0, sin0, cos1, sin1, ...]
    freqs_cis = torch.empty(max_seq_len, head_dim, dtype=torch.float32, device=device)
    freqs_cis[:, 0::2] = cos_vals
    freqs_cis[:, 1::2] = sin_vals

    # For FlashInfer: create non-interleaved cos_sin_cache (first half cosines, second half sines)
    cos_sin_cache = torch.cat([cos_vals, sin_vals], dim=1)

    # --- Create dummy query and key tensors ---
    query = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, n_head, head_dim, dtype=dtype, device=device)

    query_rotated_custom = torch.ops.rope.apply_rope_with_input_pos(
        query, freqs_cis, input_pos, "bsnd"
    )
    key_rotated_custom = torch.ops.rope.apply_rope_with_input_pos(key, freqs_cis, input_pos, "bsnd")

    query_flattened = query.view(batch * seq_len, n_head, head_dim)
    query_rotated_flat = torch.ops.rope.apply_rope_on_flattened_inputs(
        query_flattened, freqs_cis, input_pos, seq_lens, seq_start_indices
    )
    query_rotated_flat = query_rotated_flat.view(batch, seq_len, n_head, head_dim)
    key_flattened = key.view(batch * seq_len, n_head, head_dim)
    key_rotated_flat = torch.ops.rope.apply_rope_on_flattened_inputs(
        key_flattened, freqs_cis, input_pos, seq_lens, seq_start_indices
    )
    key_rotated_flat = key_rotated_flat.view(batch, seq_len, n_head, head_dim)

    # FlashInfer expects flattened tensors of shape [B*S, N*D]
    query_flat = query.view(batch * seq_len, n_head * head_dim)
    key_flat = key.view(batch * seq_len, n_head * head_dim)
    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, query_flat, key_flat, head_dim, cos_sin_cache, is_neox=True
    )
    query_rotated_flash = query_rotated_flash.view(batch, seq_len, n_head, head_dim)
    key_rotated_flash = key_rotated_flash.view(batch, seq_len, n_head, head_dim)

    # --- Compare outputs ---
    q_custom = query_rotated_custom.to(torch.float32)
    k_custom = key_rotated_custom.to(torch.float32)
    q_flat = query_rotated_flat.to(torch.float32)
    k_flat = key_rotated_flat.to(torch.float32)
    q_flash = query_rotated_flash.to(torch.float32)
    k_flash = key_rotated_flash.to(torch.float32)

    atol = 1e-3
    rtol = 1e-3

    assert torch.allclose(q_custom, q_flat, atol=atol, rtol=rtol), (
        "Custom op and flattened op differ for query."
    )
    assert torch.allclose(q_custom, q_flash, atol=atol, rtol=rtol), (
        "Custom op and FlashInfer op differ for query."
    )
    assert torch.allclose(k_custom, k_flat, atol=atol, rtol=rtol), (
        "Custom op and flattened op differ for key."
    )
    assert torch.allclose(k_custom, k_flash, atol=atol, rtol=rtol), (
        "Custom op and FlashInfer op differ for key."
    )

    if not random_input_pos:
        # Calculate and compare with plain function results
        q_for_emb = query.transpose(1, 2).clone()  # Shape: [B, N, S, D]
        k_for_emb = key.transpose(1, 2).clone()
        q_rotated_emb, k_rotated_emb = apply_rotary_pos_emb(q_for_emb, k_for_emb, seq_len, head_dim)
        # Transpose back to [B, S, N, D]
        q_rotated_emb = q_rotated_emb.transpose(1, 2)
        k_rotated_emb = k_rotated_emb.transpose(1, 2)

        q_emb = q_rotated_emb.to(torch.float32)
        k_emb = k_rotated_emb.to(torch.float32)
        assert torch.allclose(q_custom, q_emb, atol=atol, rtol=rtol), (
            "Custom op and plain function differ for query."
        )
        assert torch.allclose(k_custom, k_emb, atol=atol, rtol=rtol), (
            "Custom op and plain function differ for key."
        )

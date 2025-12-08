import math

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.kernel import (
    triton_bmm,
    triton_rocket_paged_kt_cache_bmm,
)


def pytorch_reference_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cu_seqlens: torch.Tensor,
    k_cu_seqlens: torch.Tensor,
    batch_size: int,
    sm_scale: float = None,
    causal: bool = False,
) -> torch.Tensor:
    num_q_heads, total_q_tokens, head_dim = q.shape
    num_k_heads, total_k_tokens, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Compute q_len_per_seq
    q_len_per_seq = total_q_tokens // batch_size

    scores = torch.full(
        (num_q_heads, q_len_per_seq, total_k_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # Process each batch
    for batch_idx in range(batch_size):
        q_start = q_cu_seqlens[batch_idx].item()
        q_end = q_cu_seqlens[batch_idx + 1].item()
        k_start = k_cu_seqlens[batch_idx].item()
        k_end = k_cu_seqlens[batch_idx + 1].item()

        q_seqlen = q_end - q_start
        k_seqlen = k_end - k_start

        if q_seqlen <= 0 or k_seqlen <= 0:
            continue

        q_batch = q[:, q_start:q_end, :]  # [num_q_heads, q_seqlen, head_dim]

        num_heads_per_kv = num_q_heads // num_k_heads

        for head_idx in range(num_q_heads):
            k_head_idx = head_idx // num_heads_per_kv
            k_batch = k[k_head_idx, k_start:k_end, :]  # [k_seqlen, head_dim]

            qk = torch.matmul(q_batch[head_idx], k_batch.T) * sm_scale

            if causal:
                causal_mask = torch.triu(
                    torch.ones(q_seqlen, k_seqlen, device=q.device, dtype=torch.bool), diagonal=1
                )
                qk = qk.masked_fill(causal_mask, float("-inf"))

            scores[head_idx, :q_seqlen, k_start:k_end] = qk

    return scores


def create_kt_cache_from_k(
    k: torch.Tensor,
    kv_lens: torch.Tensor,
    kt_page_size: int,
    tokens_per_block: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create paged KT cache tensor from continuous K tensor.

    Args:
        k: Key tensor [num_gen_tokens, num_kv_heads * head_dim]
        kv_lens: Sequence lengths [num_gen_tokens]
        kt_page_size: Page size for KT tokens
        tokens_per_block: Tokens per cache block
        num_kv_heads: Number of KV heads
        head_dim: Head dimension

    Returns:
        kt_cache_tensor: KT cache [num_blocks, tokens_per_block, num_kv_heads, 2*head_dim]
        kt_cache_block_offsets: Block offsets [num_gen_tokens, max_kt_blocks_per_seq]
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
    """
    num_gen_tokens = k.shape[0]
    device = k.device
    dtype = k.dtype

    # Calculate number of kt tokens per sequence
    num_kt_tokens_per_seq = [
        (kv_len.item() + kt_page_size - 1) // kt_page_size for kv_len in kv_lens
    ]
    max_kt_tokens = max(num_kt_tokens_per_seq)
    max_kt_blocks_per_seq = (max_kt_tokens + tokens_per_block - 1) // tokens_per_block

    # Calculate total number of blocks needed
    total_blocks_needed = sum(
        (kt_tokens + tokens_per_block - 1) // tokens_per_block
        for kt_tokens in num_kt_tokens_per_seq
    )

    # Create KT cache tensor
    kt_cache_tensor = torch.zeros(
        (total_blocks_needed, tokens_per_block, num_kv_heads, 2 * head_dim),
        device=device,
        dtype=dtype,
    )

    # Create block offsets tensor
    kt_cache_block_offsets = torch.full(
        (num_gen_tokens, max_kt_blocks_per_seq), -1, dtype=torch.int32, device=device
    )

    # Fill KT cache and block offsets
    current_block_idx = 0

    for seq_idx in range(num_gen_tokens):
        kv_len = kv_lens[seq_idx].item()
        num_kt_tokens = num_kt_tokens_per_seq[seq_idx]

        # Reshape k for this sequence: [num_kv_heads, head_dim]
        k_seq = k[seq_idx].view(num_kv_heads, head_dim)

        # Process each kt token (page)
        for kt_idx in range(num_kt_tokens):
            page_start = kt_idx * kt_page_size

            # For simplicity, we use the first token in the page as representative
            # In real usage, this would be min/max over the page
            # Here we just replicate the first token's value for testing
            token_idx = page_start
            if token_idx < kv_len:
                k_val = k_seq  # [num_kv_heads, head_dim]

                # Store k_min and k_max (for testing, we use same value)
                kt_min = k_val
                kt_max = k_val

                # Determine which block this kt token belongs to
                block_offset = kt_idx // tokens_per_block
                token_offset_in_block = kt_idx % tokens_per_block

                # Assign block index if not already assigned
                if kt_cache_block_offsets[seq_idx, block_offset] < 0:
                    kt_cache_block_offsets[seq_idx, block_offset] = current_block_idx
                    current_block_idx += 1

                block_idx = kt_cache_block_offsets[seq_idx, block_offset].item()

                # Store in cache: [block, token_in_block, head, 2*head_dim]
                kt_cache_tensor[block_idx, token_offset_in_block, :, :head_dim] = kt_min
                kt_cache_tensor[block_idx, token_offset_in_block, :, head_dim:] = kt_max

    return kt_cache_tensor, kt_cache_block_offsets, max_kt_blocks_per_seq


def pytorch_reference_paged_kt_cache_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    kv_lens: torch.Tensor,
    kt_page_size: int,
    sm_scale: float = None,
) -> torch.Tensor:
    num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim = q.shape
    device = q.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_kt_tokens = max((kv_len.item() + kt_page_size - 1) // kt_page_size for kv_len in kv_lens)
    total_kt_tokens = num_gen_tokens * max_kt_tokens

    # Output shape matches kernel: [num_kv_heads, num_heads_per_kv, total_kt_tokens]
    scores = torch.zeros(
        (num_kv_heads, num_heads_per_kv, total_kt_tokens), dtype=torch.float32, device=device
    )

    # Process each generation token
    for batch_idx in range(num_gen_tokens):
        kv_len = kv_lens[batch_idx].item()
        num_kt_tokens = (kv_len + kt_page_size - 1) // kt_page_size

        q_batch = q[batch_idx]  # [num_kv_heads, num_heads_per_kv, head_dim]
        k_batch = k[batch_idx].view(num_kv_heads, head_dim)  # [num_kv_heads, head_dim]

        output_offset = batch_idx * max_kt_tokens

        for kv_head_idx in range(num_kv_heads):
            q_heads = q_batch[kv_head_idx]  # [num_heads_per_kv, head_dim]
            q_sum = q_heads.sum(dim=0)  # [head_dim]
            dim_pos_vec = q_sum > 0  # [head_dim], boolean mask

            k_vec = k_batch[kv_head_idx]  # [head_dim]

            # Select k_max where dim_pos > 0, k_min otherwise
            # For simplicity in test, we use k as both min and max
            k_selected = torch.where(dim_pos_vec, k_vec, k_vec)

            for q_head_idx in range(num_heads_per_kv):
                q_vec = q_batch[kv_head_idx, q_head_idx]  # [head_dim]

                # Compute score for each kt token (simplified)
                for kt_idx in range(num_kt_tokens):
                    score = torch.dot(q_vec, k_selected) * sm_scale
                    scores[kv_head_idx, q_head_idx, output_offset + kt_idx] = score

    return scores


@pytest.mark.parametrize(
    "batch_size,q_len_per_seq,k_lens,num_q_heads,num_kv_heads,head_dim,causal",
    [
        # Single batch
        (1, 32, [128], 8, 8, 128, False),
        (1, 32, [128], 8, 8, 128, True),
        # Multiple batches with different k_len
        (3, 32, [64, 128, 256], 8, 8, 128, False),
        (4, 16, [100, 200, 150, 300], 32, 8, 128, False),
        # Edge cases
        (2, 1, [10, 20], 8, 8, 128, False),  # q_len=1
        (2, 64, [64, 128], 16, 4, 64, True),  # Different head_dim with causal
    ],
)
def test_triton_bmm(batch_size, q_len_per_seq, k_lens, num_q_heads, num_kv_heads, head_dim, causal):
    device = torch.device("cuda")
    dtype = torch.float32

    total_q_tokens = batch_size * q_len_per_seq
    q_cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * q_len_per_seq

    k_lens_tensor = torch.tensor(k_lens, dtype=torch.int32, device=device)
    k_cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(k_lens_tensor, dim=0)]
    )
    total_k_tokens = k_cu_seqlens[-1].item()

    q = torch.randn((num_q_heads, total_q_tokens, head_dim), dtype=dtype, device=device)
    k = torch.randn((num_kv_heads, total_k_tokens, head_dim), dtype=dtype, device=device)

    triton_scores = triton_bmm(
        q=q,
        k=k,
        q_cu_seqlens=q_cu_seqlens,
        k_cu_seqlens=k_cu_seqlens,
        batch_size=batch_size,
        sm_scale=None,
        causal=causal,
    )

    reference_scores = pytorch_reference_bmm(
        q=q,
        k=k,
        q_cu_seqlens=q_cu_seqlens,
        k_cu_seqlens=k_cu_seqlens,
        batch_size=batch_size,
        sm_scale=None,
        causal=causal,
    )

    # Compare results
    # Handle -inf values separately
    triton_finite = torch.isfinite(triton_scores)
    reference_finite = torch.isfinite(reference_scores)

    # Check that inf/finite masks match
    assert torch.all(triton_finite == reference_finite), (
        "Finite/infinite mask mismatch between Triton and reference"
    )

    # Compare finite values
    if triton_finite.any():
        max_diff = torch.max(
            torch.abs(triton_scores[triton_finite] - reference_scores[reference_finite])
        ).item()

        print(f"Max absolute difference: {max_diff:.6f}")

        assert max_diff < 0.01, f"Max difference {max_diff} exceeds threshold"


@pytest.mark.parametrize(
    "batch_size,kv_lens,num_kv_heads,num_heads_per_kv,head_dim,kt_page_size,tokens_per_block",
    [
        # Single batch
        (1, [128], 8, 4, 128, 3, 64),
        # Multiple batches with different kv_len
        (3, [100, 200, 150], 8, 4, 128, 3, 64),
    ],
)
def test_triton_rocket_paged_kt_cache_bmm(
    batch_size, kv_lens, num_kv_heads, num_heads_per_kv, head_dim, kt_page_size, tokens_per_block
):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_gen_tokens = batch_size

    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # Create Q tensor: [num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim]
    q = torch.randn(
        (num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim), dtype=dtype, device=device
    )

    # Create K tensor for reference: [num_gen_tokens, num_kv_heads * head_dim]
    k = torch.randn((num_gen_tokens, num_kv_heads * head_dim), dtype=dtype, device=device)

    # Create paged KT cache
    kt_cache_tensor, kt_cache_block_offsets, max_kt_blocks_per_seq = create_kt_cache_from_k(
        k=k,
        kv_lens=kv_lens_tensor,
        kt_page_size=kt_page_size,
        tokens_per_block=tokens_per_block,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Calculate output offsets
    max_kt_tokens = max((kv_len + kt_page_size - 1) // kt_page_size for kv_len in kv_lens)
    total_kt_tokens = num_gen_tokens * max_kt_tokens
    output_offsets = (
        torch.arange(0, num_gen_tokens + 1, device=device, dtype=torch.int32) * max_kt_tokens
    )

    triton_scores = triton_rocket_paged_kt_cache_bmm(
        q=q,
        kt_cache_tensor=kt_cache_tensor,
        kt_cache_block_offsets=kt_cache_block_offsets,
        kv_lens=kv_lens_tensor,
        output_offsets=output_offsets,
        kt_page_size=kt_page_size,
        tokens_per_block=tokens_per_block,
        max_kt_blocks_per_seq=max_kt_blocks_per_seq,
        total_kt_tokens=total_kt_tokens,
        sm_scale=None,
    )

    reference_scores = pytorch_reference_paged_kt_cache_bmm(
        q=q,
        k=k,
        kv_lens=kv_lens_tensor,
        kt_page_size=kt_page_size,
        sm_scale=None,
    )

    # Compare results
    # Only compare non-zero entries (valid kt tokens)
    mask = torch.zeros_like(triton_scores, dtype=torch.bool)
    for batch_idx in range(num_gen_tokens):
        kv_len = kv_lens[batch_idx]
        num_kt_tokens = (kv_len + kt_page_size - 1) // kt_page_size
        offset = batch_idx * max_kt_tokens
        mask[:, :, offset : offset + num_kt_tokens] = True

    triton_valid = triton_scores[mask]
    reference_valid = reference_scores[mask]

    max_diff = torch.max(torch.abs(triton_valid - reference_valid)).item()

    print(f"Max absolute difference: {max_diff:.6f}")

    assert max_diff < 0.05, f"Max difference {max_diff} exceeds threshold"


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing Triton BMM Kernel")
    print("=" * 80)

    # Test triton_bmm
    print("\n--- Single batch, non-causal ---")
    test_triton_bmm(1, 32, [128], 8, 8, 128, False)

    print("\n--- Single batch, causal ---")
    test_triton_bmm(1, 32, [128], 8, 8, 128, True)

    print("\n--- Multiple batches, different k_len ---")
    test_triton_bmm(3, 32, [64, 128, 256], 8, 8, 128, False)

    print("\n" + "=" * 80)
    print("Testing Triton Rocket Paged KT Cache BMM Kernel")
    print("=" * 80)

    # Test triton_rocket_paged_kt_cache_bmm
    print("\n--- Single batch ---")
    test_triton_rocket_paged_kt_cache_bmm(1, [128], 8, 4, 128, 3, 64)

    print("\n--- Multiple batches, different kv_len ---")
    test_triton_rocket_paged_kt_cache_bmm(3, [100, 200, 150], 8, 4, 128, 3, 64)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

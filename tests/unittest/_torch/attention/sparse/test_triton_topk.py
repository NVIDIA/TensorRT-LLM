import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.kernel import triton_topk


def pytorch_reference_topk(
    input_tensor: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """
    Args:
        input_tensor: Input scores [num_kv_heads, sum(kv_lens)]
        kv_offsets: KV offsets [batch_size + 1]
        kv_lens: KV lengths [batch_size]
        topk: TopK parameter

    Returns:
        output_indices: Padded indices [num_kv_heads, batch_size, topk]
    """
    num_kv_heads = input_tensor.shape[0]
    batch_size = len(kv_lens)
    device = input_tensor.device

    # Compute max sequence length for padding
    max_seq_len = kv_lens.max().item()

    # Ensure padding size >= topk
    pad_size = max(max_seq_len, topk)

    # Create padded tensor [num_kv_heads, batch_size, pad_size]
    padded_tensor = torch.full(
        (num_kv_heads, batch_size, pad_size), float("-inf"), dtype=input_tensor.dtype, device=device
    )

    # Fill in actual values
    for batch_idx in range(batch_size):
        start = kv_offsets[batch_idx].item()
        end = kv_offsets[batch_idx + 1].item()
        seq_len = kv_lens[batch_idx].item()

        for head_idx in range(num_kv_heads):
            padded_tensor[head_idx, batch_idx, :seq_len] = input_tensor[head_idx, start:end]

    # Perform batch topk: [num_kv_heads, batch_size, pad_size] -> [num_kv_heads, batch_size, topk]
    topk_values, topk_indices = torch.topk(
        padded_tensor,
        k=topk,
        dim=-1,
        largest=True,
    )

    # Mask out invalid indices based on each batch's seq_len
    seq_lens_expanded = kv_lens.to(device).unsqueeze(0).unsqueeze(-1)  # [1, batch_size, 1]
    # topk_indices: [num_kv_heads, batch_size, topk]
    mask = topk_indices >= seq_lens_expanded
    topk_indices.masked_fill_(mask, -1)

    return topk_indices


def triton_topk_wrapper(
    input_tensor: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """
    Args:
        input_tensor: Input scores [num_kv_heads, sum(kv_lens)]
        kv_offsets: KV offsets [batch_size + 1]
        kv_lens: KV lengths [batch_size]
        topk: TopK parameter

    Returns:
        output_indices: Padded indices [num_kv_heads, batch_size, topk]
    """
    num_kv_heads = input_tensor.shape[0]
    batch_size = len(kv_lens)
    device = input_tensor.device

    sparse_lens = torch.tensor(
        [min(topk, seq_len.item()) for seq_len in kv_lens], dtype=torch.int32, device=device
    )
    sparse_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(sparse_lens, dim=0)]
    ).to(device)
    total_sparse_indices = sparse_offsets[-1].item()

    output_indices_flat = triton_topk(
        input_tensor, kv_offsets, sparse_offsets, total_sparse_indices, topk
    )

    # Convert flat format to padded format [num_kv_heads, batch_size, topk]
    output_indices_padded = torch.full(
        (num_kv_heads, batch_size, topk), -1, dtype=torch.int32, device=device
    )

    for batch_idx in range(batch_size):
        start = sparse_offsets[batch_idx].item()
        end = sparse_offsets[batch_idx + 1].item()
        actual_len = end - start

        for head_idx in range(num_kv_heads):
            output_indices_padded[head_idx, batch_idx, :actual_len] = output_indices_flat[
                head_idx, start:end
            ]

    return output_indices_padded


def compute_overlap_ratio(
    triton_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    kv_lens: torch.Tensor,
) -> float:
    """
    Args:
        triton_indices: Triton topk results [num_kv_heads, batch_size, topk]
        reference_indices: Reference topk results [num_kv_heads, batch_size, topk]
        kv_lens: KV lengths [batch_size]

    Returns:
        Average overlap ratio across all batches and heads
    """
    num_kv_heads = triton_indices.shape[0]
    batch_size = triton_indices.shape[1]

    overlap_ratios = []

    # Compare batch by batch
    for batch_idx in range(batch_size):
        for head_idx in range(num_kv_heads):
            # Extract indices for this batch and head
            triton_batch = triton_indices[head_idx, batch_idx, :].cpu().tolist()
            reference_batch = reference_indices[head_idx, batch_idx, :].cpu().tolist()

            # Filter out -1 (invalid/padding indices)
            triton_set = set([x for x in triton_batch if x >= 0])
            reference_set = set([x for x in reference_batch if x >= 0])

            if len(reference_set) > 0:
                overlap = len(triton_set & reference_set)
                overlap_ratio = overlap / len(reference_set)
                overlap_ratios.append(overlap_ratio)

    if len(overlap_ratios) == 0:
        return 1.0

    return sum(overlap_ratios) / len(overlap_ratios)


@pytest.mark.parametrize(
    "batch_size,seq_lens,num_kv_heads,topk",
    [
        # Single batch, seq_len > topk
        (1, [3000], 1, 2048),
        # Single batch, seq_len < topk
        (1, [1000], 8, 2048),
        # Multiple batches, mixed seq_len (some < topk, some > topk)
        (6, [50, 150, 80, 300, 100, 256], 8, 128),
        (6, [1000, 2500, 1500, 3000, 1800, 4000], 8, 2048),
    ],
)
def test_topk_kernel(batch_size, seq_lens, num_kv_heads, topk):
    device = torch.device("cuda")
    dtype = torch.float32

    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    kv_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(kv_lens, dim=0)]
    ).to(device)

    total_tokens = kv_offsets[-1].item()

    input_tensor = torch.randn((num_kv_heads, total_tokens), dtype=dtype, device=device)

    triton_output = triton_topk_wrapper(
        input_tensor=input_tensor,
        kv_offsets=kv_offsets,
        kv_lens=kv_lens,
        topk=topk,
    )

    reference_output = pytorch_reference_topk(
        input_tensor=input_tensor,
        kv_offsets=kv_offsets,
        kv_lens=kv_lens,
        topk=topk,
    )

    overlap_ratio = compute_overlap_ratio(
        triton_output,
        reference_output,
        kv_lens,
    )

    min_threshold = 0.99

    print(f"overlap_ratio: {overlap_ratio}")

    assert overlap_ratio >= min_threshold, (
        f"Overlap ratio {overlap_ratio:.4f} is too low (< {min_threshold})"
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing Triton TopK Kernel")
    print("=" * 80)

    # Single batch tests
    print("\n--- Single batch, seq_len > topk ---")
    test_topk_kernel(1, [3000], 8, 2048)

    print("\n--- Single batch, seq_len < topk ---")
    test_topk_kernel(1, [1000], 8, 2048)

    print("\n--- Multiple batches, mixed seq_len ---")
    test_topk_kernel(6, [50, 150, 80, 300, 100, 256], 8, 128)
    test_topk_kernel(6, [1000, 2500, 1500, 3000, 1800, 4000], 8, 2048)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

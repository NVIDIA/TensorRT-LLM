import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.kernel import triton_gather_k_cache


def reference_gather_k_cache(
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    k_token_end: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference: vectorized flat-index gather.

    Mirrors the original ``_gather_k_cache_for_chunk`` logic from dsa.py but
    operates on the flattened cache directly (no ``_unravel_indices`` needed).
    """
    num_k_tokens = k_token_end - k_token_start
    device = k_cache.device
    scale_bytes = 4

    if num_k_tokens == 0:
        return (
            torch.empty(0, head_dim, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    k_cache_flat = k_cache.reshape(-1)

    fp8_bases = slot_mapping_fp8[k_token_start:k_token_end]
    byte_offsets_fp8 = torch.arange(head_dim, device=device, dtype=torch.int64)
    gather_fp8 = fp8_bases.unsqueeze(1) + byte_offsets_fp8.unsqueeze(0)
    out_fp8 = k_cache_flat[gather_fp8]

    scale_bases = slot_mapping_scale[k_token_start:k_token_end]
    byte_offsets_scale = torch.arange(scale_bytes, device=device, dtype=torch.int64)
    gather_scale = scale_bases.unsqueeze(1) + byte_offsets_scale.unsqueeze(0)
    out_scale = k_cache_flat[gather_scale]

    k_fp8 = out_fp8.view(torch.float8_e4m3fn)
    k_scale = out_scale.view(torch.float32).view(num_k_tokens, 1)
    return k_fp8, k_scale


def _create_cache_and_mappings(
    total_kv_len: int,
    head_dim: int,
    num_cache_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random k_cache with valid, non-overlapping slot mappings.

    Each token occupies ``head_dim + 4`` bytes in the flat cache (FP8 data
    followed by scale data).  Token positions are shuffled so that the
    gathered regions are non-contiguous, stressing the kernel's indexing.
    """
    scale_bytes = 4
    bytes_per_token = head_dim + scale_bytes

    num_cols = (total_kv_len * bytes_per_token + num_cache_rows - 1) // num_cache_rows
    k_cache = torch.randint(0, 256, (num_cache_rows, num_cols), dtype=torch.uint8, device=device)

    perm = torch.randperm(total_kv_len, device=device)
    region_starts = perm.to(torch.int64) * bytes_per_token
    slot_mapping_fp8 = region_starts
    slot_mapping_scale = region_starts + head_dim

    return k_cache, slot_mapping_fp8, slot_mapping_scale


@pytest.mark.parametrize(
    "total_kv_len,k_token_start,k_token_end,head_dim,num_cache_rows",
    [
        # Gather all tokens
        (64, 0, 64, 128, 16),
        # Sub-range from the middle
        (128, 32, 96, 128, 32),
        # Single token
        (10, 3, 4, 128, 4),
        # Smaller head dim
        (64, 0, 64, 64, 16),
        # Larger test
        (512, 100, 400, 128, 64),
        # Non-power-of-2 token count (exercises BLOCK_TOKENS tail masking)
        (100, 10, 47, 128, 32),
        # Very small
        (3, 0, 3, 128, 1),
    ],
)
def test_triton_gather_k_cache(
    total_kv_len,
    k_token_start,
    k_token_end,
    head_dim,
    num_cache_rows,
):
    device = torch.device("cuda")

    k_cache, slot_mapping_fp8, slot_mapping_scale = _create_cache_and_mappings(
        total_kv_len, head_dim, num_cache_rows, device
    )

    triton_fp8, triton_scale = triton_gather_k_cache(
        k_cache,
        slot_mapping_fp8,
        slot_mapping_scale,
        k_token_start,
        k_token_end,
        head_dim,
    )

    ref_fp8, ref_scale = reference_gather_k_cache(
        k_cache,
        slot_mapping_fp8,
        slot_mapping_scale,
        k_token_start,
        k_token_end,
        head_dim,
    )

    assert triton_fp8.shape == ref_fp8.shape
    assert triton_scale.shape == ref_scale.shape
    assert torch.equal(triton_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), "FP8 data mismatch"
    assert torch.equal(triton_scale.view(torch.uint8), ref_scale.view(torch.uint8)), (
        "Scale data mismatch"
    )


def test_triton_gather_k_cache_empty():
    """Zero-length chunk should return correctly shaped empty tensors."""
    device = torch.device("cuda")
    head_dim = 128
    k_cache = torch.randint(0, 256, (4, 256), dtype=torch.uint8, device=device)
    slot_mapping_fp8 = torch.zeros(10, dtype=torch.int64, device=device)
    slot_mapping_scale = torch.zeros(10, dtype=torch.int64, device=device)

    k_fp8, k_scale = triton_gather_k_cache(
        k_cache,
        slot_mapping_fp8,
        slot_mapping_scale,
        k_token_start=5,
        k_token_end=5,
        head_dim=head_dim,
    )

    assert k_fp8.shape == (0, head_dim)
    assert k_scale.shape == (0, 1)
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert k_scale.dtype == torch.float32

import pytest
import torch
from utils.util import getSMVersion, skip_pre_hopper

# Import tensorrt_llm to load custom CUDA operators (indexer_topk_decode, indexer_topk_prefill)
import tensorrt_llm  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for indexer_topk tests", allow_module_level=True)


def _prefill_param_values():
    """
    Decide parameter coverage based on GPU architecture (SM version).

    - pre-Hopper (SM < 90): skip via @skip_pre_hopper
    - Hopper (SM == 90): reduced coverage
    - Blackwell (SM >= 100): full coverage
    """
    sm = getSMVersion()
    if sm >= 100:  # Blackwell family
        return [1, 32], [4096, 8192, 32768]
    # Hopper (and other >= 90 but < 100, if any): reduced coverage
    return [1, 4], [4096, 8192, 32768]


_PREFILL_BATCH_SIZES, _PREFILL_NUM_TOKENS = _prefill_param_values()


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Tensor of shape (num_rows,) indicating the start position of each row
        row_ends: Tensor of shape (num_rows,) indicating the end position (exclusive) of each row
        dtype: Data type for the logits tensor
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (num_rows, max_row_length) with random values and -inf padding
    """
    torch.manual_seed(seed)
    num_rows = row_starts.shape[0]
    max_len = int(row_ends.max().item())

    # Generate random logits in range [0, 1)
    logits = torch.rand(num_rows, max_len, dtype=dtype, device="cuda")

    # Vectorized masking: set positions outside [row_start, row_end) to -inf
    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)  # (1, max_len)
    mask_lo = col_indices < row_starts.unsqueeze(1)  # positions before row_start
    mask_hi = col_indices >= row_ends.unsqueeze(1)  # positions at or after row_end
    mask = mask_lo | mask_hi  # positions outside valid range
    logits[mask] = float("-inf")

    return logits


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Handles different shapes and -1 placeholders in cuda_indices.

    Args:
        logits: Input logits tensor [num_rows, vocab_size]
        cuda_indices: CUDA implementation output [num_rows, cuda_k], may contain -1
        torch_indices: PyTorch reference output [num_rows, torch_k], may contain -1
        row_starts: Start positions for each row [num_rows]
        row_ends: End positions for each row [num_rows]
        top_k: Target top-k value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Handle potentially different k values
    cuda_indices.shape[1]
    torch_indices.shape[1]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    # For each row, compare only the valid indices (non -1)
    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        # Get valid indices from both implementations (filter out -1)
        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        # Filter out -1 (invalid) indices
        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

        # Check if the number of valid indices matches
        if cuda_valid.shape[0] != torch_valid.shape[0]:
            print(
                f"Row {row_idx}: Different number of valid indices - "
                f"CUDA: {cuda_valid.shape[0]}, PyTorch: {torch_valid.shape[0]}"
            )
            return False

        if cuda_valid.shape[0] != expected_valid:
            print(
                f"Row {row_idx}: Expected {expected_valid} valid indices, got {cuda_valid.shape[0]}"
            )
            return False

        # If no valid indices, continue
        if cuda_valid.shape[0] == 0:
            continue

        # Gather the corresponding logit values
        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        # Adjust indices to absolute positions (add row_start offset)
        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        # Get logit values for the selected indices
        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        # Sort both value arrays in descending order
        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        # Compare sorted values
        if not torch.allclose(
            cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance
        ):
            # Additional debug: check if sets are identical
            cuda_set = set(cuda_valid.cpu().tolist())
            torch_set = set(torch_valid.cpu().tolist())
            if cuda_set != torch_set:
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True


def generate_seq_lens(batch_size, min_long_seq, num_tokens):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    is_long = torch.rand(batch_size, device="cuda") < 0.9
    num_long = is_long.sum().item()
    if num_long > 0:
        seq_lens[is_long] = torch.randint(
            min_long_seq, num_tokens, (num_long,), dtype=torch.int32, device="cuda"
        )

    num_short = (~is_long).sum().item()
    if num_short > 0:
        seq_lens[~is_long] = torch.randint(
            1, min_long_seq, (num_short,), dtype=torch.int32, device="cuda"
        )
    return seq_lens


@pytest.mark.parametrize("batch_size", [1, 64, 512, 2048])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    # Set input data
    num_gen_tokens = batch_size * next_n  # Use the same variable name as dsa.py
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    # Create output tensors
    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    # Run CUDA implementation
    torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, next_n, index_topk)

    torch.cuda.synchronize()

    # Run reference implementation
    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match torch.topk"


@skip_pre_hopper
@pytest.mark.parametrize("batch_size", _PREFILL_BATCH_SIZES)
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", _PREFILL_NUM_TOKENS)
def test_indexer_topk_prefill(batch_size, index_topk, num_tokens):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    # gen random input for the sequence length
    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    num_gen_tokens = seq_lens.sum()

    # gen the row_starts and row_ends (from 1 to ...)
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(1, seq_lens.max() + 1, dtype=torch.int32, device="cuda")
    row_ends = row_indices.expand(seq_lens.size(0), -1)[
        row_indices.expand(seq_lens.size(0), -1) <= seq_lens.unsqueeze(1)
    ].contiguous()

    # gen logits
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    # Create output tensors
    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    # Run CUDA implementation
    torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    torch.cuda.synchronize()

    # Run reference implementation
    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match torch.topk"

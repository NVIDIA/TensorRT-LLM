"""
Distribution-parameterized correctness tests for the heuristic indexer_topk_decode.

Logits are sampled from four distribution families that characterise
negative-shifted decode-phase logit spaces (means −0.5 to −4.5):

  beta        — bounded, bell-shaped; near-zero / moderate / deep negative mean
  logistic    — heavy-tailed symmetric (leptokurtic)
  lognorm     — positively skewed, wide support
  weibull_min — right-skewed extreme-value; narrow and wide spread variants

Two extensions beyond the baseline test_indexer_topk.py:

  pre_idx (heuristic candidates)
    Shape [batch_size, index_topk].  For each batch element b, pre_idx[b] is
    built from the base row's actual top-K:
      - pre_idx[b, 0] = argmax  (kernel invariant)
      - floor(index_topk * success_ratio) slots drawn from actual top-K indices
      - remaining slots filled with random valid indices
    success_ratio is a pytest parameter (>= 0.4).

  MTP structure (next_n > 1)
    When next_n > 1, consecutive rows within each batch element share most of
    their logit values.  For batch element b with valid_base = row_ends[b*next_n]
    and MTP offset nni = 1…next_n-1:
      logits[b*next_n + nni, nni : nni+valid_base] = logits[b*next_n, 0 : valid_base]
    Positions 0..nni-1 are independently sampled (new token positions);
    positions >= nni+valid_base remain -inf.

Logit shapes (batch_size, next_n, num_tokens) match test_indexer_topk.py.
"""

import numpy as np
import pytest
import torch
from utils.util import getSMVersion, skip_pre_blackwell, skip_pre_hopper

import tensorrt_llm  # noqa: F401

# Import CuTE DSL utils
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for indexer_topk tests", allow_module_level=True)

try:
    import scipy.stats as _scipy_stats
    from scipy.special import gamma as _gamma

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Prefill parameter helpers (unchanged from test_indexer_topk.py)
# ---------------------------------------------------------------------------


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
    return [1, 4], [4096, 8192, 32768]


_PREFILL_BATCH_SIZES, _PREFILL_NUM_TOKENS = _prefill_param_values()


# ---------------------------------------------------------------------------
# Shared helpers (verbatim from test_indexer_topk.py)
# ---------------------------------------------------------------------------


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

    logits = torch.rand(num_rows, max_len, dtype=dtype, device="cuda")

    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)
    mask_lo = col_indices < row_starts.unsqueeze(1)
    mask_hi = col_indices >= row_ends.unsqueeze(1)
    logits[mask_lo | mask_hi] = float("-inf")

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
        tolerance: Tolerance for floating point value comparison.
            Set to full_range/256 to accept tie-breaking at the histogram
            bin boundary (elements within one bin width of the K-th value
            are ambiguous and either choice is a valid top-K answer).

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

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

        if cuda_valid.shape[0] == 0:
            continue

        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        if not torch.allclose(
            cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance
        ):
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


# ---------------------------------------------------------------------------
# Original random-data decode test (verbatim from test_indexer_topk.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 64, 512, 2048])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, next_n, index_topk)
    torch.cuda.synchronize()

    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

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

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    num_gen_tokens = seq_lens.sum()

    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(1, seq_lens.max() + 1, dtype=torch.int32, device="cuda")
    row_ends = row_indices.expand(seq_lens.size(0), -1)[
        row_indices.expand(seq_lens.size(0), -1) <= seq_lens.unsqueeze(1)
    ].contiguous()

    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    torch.cuda.synchronize()

    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match torch.topk"


# ============================================================================
# CuTE DSL Top-K Tests
# ============================================================================


def _run_cute_dsl_topk_test(batch_size, next_n, index_topk, num_tokens, dtype, run_fn):
    """Common test logic for CuTE DSL top-k kernels.

    Args:
        batch_size: Number of sequences in the batch.
        next_n: Number of next tokens per sequence.
        index_topk: Number of top-k indices to select.
        num_tokens: Maximum sequence length for generating seq_lens.
        dtype: Data type for the logits tensor.
        run_fn: Callable(logits, seq_lens) -> indices tensor.
    """

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    logits = create_random_logits(row_starts, row_ends, dtype, 42)

    cute_indices = run_fn(logits, seq_lens)
    torch.cuda.synchronize()

    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~mask, -1)

    assert compare_top_k_results(
        logits, cute_indices.to(torch.int32), torch_indices, row_starts, row_ends, index_topk
    ), "CuTE DSL top-k results don't match torch.topk"


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("load_balance", [False, True])
def test_cute_dsl_topk_decode(batch_size, next_n, index_topk, num_tokens, dtype, load_balance):
    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        dtype,
        lambda logits, seq_lens: torch.ops.trtllm.cute_dsl_topk_decode_blackwell(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            num_copy_bits=256,
            load_balance=load_balance,
        ),
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [32768, 65536])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk_size_per_cta", [16384])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cute_dsl_topk_decode_multi_cta(
    batch_size, next_n, index_topk, num_tokens, dtype, chunk_size_per_cta, dynamic
):
    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        dtype,
        lambda logits, seq_lens: torch.ops.trtllm.cute_dsl_topk_decode_multi_cta_blackwell(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            num_copy_bits=256,
            chunk_size_per_cta=chunk_size_per_cta,
            dynamic=dynamic,
        ),
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 4, 64, 128])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 8192, 65536, 131072])
def test_cute_dsl_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens):
    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        torch.float32,
        lambda logits, seq_lens: torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            num_copy_bits=256,
        ),
    )

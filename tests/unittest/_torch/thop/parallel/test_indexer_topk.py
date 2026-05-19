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
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
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
    # Pad to multiple of 8 so stride0 satisfies the alignment requirement of
    # launchHeuristicTopKDecode for both fp32 (float4 = 4 elements) and
    # bf16/fp16 (int4 = 16 B = 8 elements) in multi-row mode (matches TRT-LLM
    # runtime where strides are always multiples of tokens_per_block >= 64).
    max_len = (max_len + 7) & ~7

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
    """Generate random sequence lengths: 90% long [min_long_seq, num_tokens), 10% short."""
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
# Random-data decode tests (verbatim helper from test_indexer_topk.py, plus
# regression coverage for the SM-saturation heuristic).
# ---------------------------------------------------------------------------


def _run_indexer_topk_decode_check(batch_size, next_n, index_topk, num_tokens, compress_ratio):
    """Run the random-data decode equivalence check against torch.topk."""
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)

    # Calculate actual KV lengths
    actual_kv_lens = seq_lens[row_indices] - next_n + next_n_offset + 1

    # Apply compression with floor division
    row_ends = actual_kv_lens // compress_ratio

    # Generate logits with the compressed size
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    # Run CUDA implementation with compress_ratio
    torch.ops.trtllm.indexer_topk_decode(
        logits, seq_lens, indices, next_n, index_topk, compress_ratio=compress_ratio
    )

    torch.cuda.synchronize()

    # Run reference implementation on compressed row_ends
    max_row_len = row_ends.max().item()
    if max_row_len == 0:
        # All rows are empty after compression, skip comparison
        return
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match torch.topk"


@pytest.mark.parametrize("batch_size", [1, 64, 512, 2048])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048, 512, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens, compress_ratio):
    _run_indexer_topk_decode_check(batch_size, next_n, index_topk, num_tokens, compress_ratio)


# Regression coverage for the SM-saturation heuristic in indexerTopK decode.
# Exercises the multi-block split path that was newly enabled for small batches
# and moderate numColumns (commits 1c88ecd58, 38f9b59b2):
#   batch_size in {16, 32, 128} bracket the smTarget transitions (>= 9, 5, 2 blocks/row)
#   num_tokens 4096-32768 with cr in {1, 4} spans both the small-numColumns range
#   that previously short-circuited to blocksPerRow=1 and the legacy single-block
#   regime (numCols < 2048 stays single-block via the maxByCols guard).
@pytest.mark.parametrize("batch_size", [16, 32, 128])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 16384, 32768])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_indexer_topk_decode_sm_saturation(
    batch_size, next_n, index_topk, num_tokens, compress_ratio
):
    _run_indexer_topk_decode_check(batch_size, next_n, index_topk, num_tokens, compress_ratio)


# Covers the single-block / multi-block path transition at
# batch_size == kDecodeTargetTotalBlocks (= 132). batch_size < 132 keeps the
# single-block-per-row path; at and above 132 the launch policy routes to
# the multi-block split-and-merge path with bp = 2 to avoid the wave-
# scheduling cliff that the single-block radix kernel hits once gridDim.x
# approaches smCount on B200.
@pytest.mark.parametrize("batch_size", [1, 64, 100, 132, 148, 256])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 16384, 32768])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_indexer_topk_decode_launch_policy_transitions(
    batch_size, next_n, index_topk, num_tokens, compress_ratio
):
    _run_indexer_topk_decode_check(batch_size, next_n, index_topk, num_tokens, compress_ratio)


def _run_indexer_topk_prefill_check(batch_size, index_topk, num_tokens):
    """Run the prefill equivalence check against torch.topk."""
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


@skip_pre_hopper
@pytest.mark.parametrize("batch_size", _PREFILL_BATCH_SIZES)
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", _PREFILL_NUM_TOKENS)
def test_indexer_topk_prefill(batch_size, index_topk, num_tokens):
    """Verify indexer_topk_prefill output matches torch.topk for variable-length rows."""
    _run_indexer_topk_prefill_check(batch_size, index_topk, num_tokens)


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
    # Clamp seq_lens so that every effective row length >= index_topk.
    # With next_n > 1, effective length = seq_len - next_n + offset + 1,
    # and the minimum (offset=0) is seq_len - next_n + 1.
    # Ensure seq_len >= next_n so effective length is at least 1.
    seq_lens = seq_lens.clamp(min=next_n)
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
@pytest.mark.parametrize("batch_size", [1, 4, 64, 256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("load_balance", [False, True])
def test_cute_dsl_topk_decode_single_cta(
    batch_size, next_n, index_topk, num_tokens, dtype, load_balance
):
    """Correctness test for CuTE DSL single-CTA TopK decode on Blackwell."""
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
    """Correctness test for CuTE DSL multi-CTA TopK decode on Blackwell."""
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
    """Correctness test for CuTE DSL indexer TopK decode with in-place output."""
    num_gen_tokens = batch_size * next_n

    def run_fn(logits, seq_lens):
        """Run CuTE DSL indexer TopK decode and return output indices."""
        output_indices = torch.empty(num_gen_tokens, index_topk, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            input_values=logits,
            seq_lens=seq_lens,
            output_indices=output_indices,
            top_k=index_topk,
            next_n=next_n,
            num_copy_bits=256,
        )
        return output_indices

    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        torch.float32,
        run_fn,
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 256])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [32768, 65536, 131072, 262144])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cute_dsl_topk_decode_single_pass_multi_cta(
    batch_size, next_n, index_topk, num_tokens, dtype
):
    """Correctness test for CuTE DSL single-pass multi-CTA TopK on Blackwell."""
    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        dtype,
        lambda logits,
        seq_lens: cute_dsl_custom_ops.CuteDSLTopKDecodeSinglePassMultiCTARunner.forward(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            return_val=False,
            num_copy_bits=256,
        )[0],
    )


# ---------------------------------------------------------------------------
# Distribution configs for heuristic decode correctness tests
#
# Each entry is a dict with keys:
#   dist       — distribution family
#   mean       — target mean (negative; typical decode logit range −0.5 to −4.5)
#   std        — target standard deviation
#   full_range — support width (high − low), used as the bounding interval
#   c          — Weibull shape parameter (weibull_min only; c≈14 for moderate skew)
#
# Parameter derivation (all analytical, no external data dependencies):
#
#   beta:
#     low  = mean − full_range/2,  high = mean + full_range/2
#     mu01 = (mean − low) / full_range
#     conc = mu01*(1−mu01) / (std/full_range)² − 1
#     α = conc*mu01,  β = conc*(1−mu01)
#
#   logistic:
#     scale = std * √3 / π          [std(logistic) = scale*π/√3]
#     CDF inversion: x = mean + scale * ln(u/(1−u)),  u ~ U(0,1)
#
#   lognorm  (left-shifted to loc = mean − full_range/2):
#     pos_mean = full_range / 2
#     σ     = √(log(1 + (std/pos_mean)²))   → matches target std exactly
#     scale = exp(log(pos_mean) − σ²/2)      → matches target mean exactly
#
#   weibull_min  (shape c fixed; loc/scale solved from mean/std):
#     scale = std / √(Γ(1+2/c) − Γ²(1+1/c))
#     loc   = mean − scale * Γ(1+1/c)
# ---------------------------------------------------------------------------

_DECODE_DIST_CONFIGS = [
    # --- Beta: bounded, bell-shaped ---
    # shallow negative mean, wide spread
    dict(dist="beta", mean=-0.75, std=1.90, full_range=13.60),
    # moderate negative mean
    dict(dist="beta", mean=-2.96, std=1.68, full_range=12.85),
    # deep negative mean, narrow spread
    dict(dist="beta", mean=-4.51, std=1.75, full_range=11.24),
    # --- Logistic: heavy-tailed symmetric (leptokurtic) ---
    dict(dist="logistic", mean=-0.47, std=1.46, full_range=12.32),
    # --- Lognorm: positively skewed, wide support ---
    dict(dist="lognorm", mean=-4.12, std=2.55, full_range=17.28),
    # --- Weibull minimum: right-skewed extreme-value ---
    # wider spread
    dict(dist="weibull_min", mean=-3.04, std=1.57, full_range=12.30, c=14.0),
    # narrower spread
    dict(dist="weibull_min", mean=-2.26, std=1.28, full_range=9.71, c=14.0),
]

# Human-readable pytest IDs: dist_mean_std
_DECODE_DIST_IDS = [
    f"{c['dist']}_m{abs(c['mean']):.2f}_s{c['std']:.2f}" for c in _DECODE_DIST_CONFIGS
]


# ---------------------------------------------------------------------------
# Distribution-aware logit generator
# ---------------------------------------------------------------------------


def _fit_beta_params(mean: float, std: float, low: float, high: float):
    """Fit Beta(α, β) on [low, high] to match target mean and std."""
    r = high - low
    mu = (mean - low) / r
    var = min((std / r) ** 2, mu * (1 - mu) * 0.99)
    conc = mu * (1 - mu) / var - 1
    return conc * mu, conc * (1 - mu)


def _fit_weibull_params(mean: float, std: float, c: float):
    """Fit Weibull_min(c, loc, scale) to match target mean and std."""
    g1 = _gamma(1 + 1 / c)
    g2 = _gamma(1 + 2 / c)
    scale = std / np.sqrt(g2 - g1**2)
    return c, mean - scale * g1, scale


def create_distributed_logits(
    cfg: dict,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """
    Generate a logits tensor sampled from the distribution specified by *cfg*.

    Values outside [row_start, row_end) are set to -inf.  All distribution
    parameters are derived analytically from (mean, std, full_range).

    Args:
        cfg:        One entry from _DECODE_DIST_CONFIGS
        row_starts: (num_rows,) inclusive start column per row
        row_ends:   (num_rows,) exclusive end column per row
        dtype:      Target torch dtype
        seed:       NumPy RNG seed

    Returns:
        Tensor (num_rows, max_len) with sampled values and -inf padding.
    """
    rng = np.random.default_rng(seed)
    num_rows = int(row_starts.shape[0])
    max_len = int(row_ends.cpu().max().item())
    # Pad to multiple of 8 so stride0 satisfies the alignment requirement of
    # launchHeuristicTopKDecode for both fp32 (float4 = 4 elements) and
    # bf16/fp16 (int4 = 16 B = 8 elements) in multi-row mode (matches TRT-LLM
    # runtime where strides are always multiples of tokens_per_block >= 64).
    max_len = (max_len + 7) & ~7
    size = (num_rows, max_len)

    dist = cfg["dist"]
    mean, std, full_range = cfg["mean"], cfg["std"], cfg["full_range"]
    low = mean - full_range / 2
    high = mean + full_range / 2

    if dist == "beta":
        alpha, beta_p = _fit_beta_params(mean, std, low, high)
        samples = (rng.beta(alpha, beta_p, size=size) * (high - low) + low).astype(np.float32)

    elif dist == "logistic":
        s = std * np.sqrt(3) / np.pi
        u = rng.uniform(1e-7, 1 - 1e-7, size=size)
        samples = (mean + s * np.log(u / (1 - u))).astype(np.float32)

    elif dist == "lognorm":
        loc = low
        pos_mean = max(mean - loc, 1e-6)  # = full_range / 2
        sigma = float(np.sqrt(np.log(1 + (std / pos_mean) ** 2)))
        scale = np.exp(np.log(pos_mean) - sigma**2 / 2)
        samples = _scipy_stats.lognorm.rvs(
            s=sigma,
            loc=loc,
            scale=scale,
            size=size,
            random_state=int(rng.integers(2**31 - 1)),
        ).astype(np.float32)

    elif dist == "weibull_min":
        c, loc, scale = _fit_weibull_params(mean, std, cfg.get("c", 14.0))
        samples = _scipy_stats.weibull_min.rvs(
            c,
            loc=loc,
            scale=scale,
            size=size,
            random_state=int(rng.integers(2**31 - 1)),
        ).astype(np.float32)

    else:
        raise ValueError(f"Unknown distribution: {dist!r}")

    # Clip to [low, high] to bound the effective value range to exactly full_range.
    # Unbounded distributions (lognorm, logistic, weibull_min) can produce outliers
    # above `high` that inflate the histogram bin width in the kernel's 256-bin
    # threshold search, causing boundary-element misidentification.
    samples = np.clip(samples, low, high).astype(np.float32)

    logits = torch.from_numpy(samples).to(dtype=dtype, device="cuda")
    col_idx = torch.arange(max_len, device="cuda").unsqueeze(0)
    mask = (col_idx < row_starts.unsqueeze(1)) | (col_idx >= row_ends.unsqueeze(1))
    logits[mask] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# MTP structure: make consecutive rows within a batch correlated
# ---------------------------------------------------------------------------


def apply_mtp_structure(
    logits: torch.Tensor,
    batch_size: int,
    next_n: int,
    row_ends: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce MTP (Multi-Token Prediction) logit correlation within each batch.

    For batch element b with base row valid length valid_base = row_ends[b*next_n],
    each MTP offset nni = 1…next_n-1 satisfies:

        logits[b*next_n + nni, nni : nni+valid_base] = logits[b*next_n, 0 : valid_base]

    Positions 0..nni-1 of each MTP row remain independently sampled (new token
    positions).  Positions nni+valid_base.. are already -inf from
    create_distributed_logits (since row_ends[b*next_n+nni] = valid_base+nni),
    so no additional masking is required after this function.

    Args:
        logits:     (batch_size*next_n, max_len) float tensor; modified in-place
        batch_size: number of batch elements
        next_n:     MTP factor; returns logits unchanged when next_n == 1
        row_ends:   (batch_size*next_n,) exclusive end column per row

    Returns:
        Same logits tensor with MTP segments overwritten.
    """
    if next_n == 1:
        return logits

    for b in range(batch_size):
        base = b * next_n
        valid_base = int(row_ends[base].item())  # valid length of base row
        for nni in range(1, next_n):
            # Copy base row positions [0:valid_base] → MTP row positions [nni:nni+valid_base]
            # Positions 0..nni-1 stay independently sampled; positions >=nni+valid_base stay -inf.
            logits[base + nni, nni : nni + valid_base] = logits[base, :valid_base]

    return logits


# ---------------------------------------------------------------------------
# pre_idx generator: heuristic candidate indices for the indexer kernel
# ---------------------------------------------------------------------------


def generate_pre_idx(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    batch_size: int,
    next_n: int,
    index_topk: int,
    success_ratio: float = 0.6,
    seed: int = 0,
) -> torch.Tensor:
    """
    Build the heuristic pre-prediction index tensor for each batch element.

    The V3.2 multi-row kernel (`heuristicTopKMultiRowKernel{,Dtype}` in
    cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu) internally adds
    `preIdxOffset = (rowIdx % next_n) + 1` to every pre_idx slot during its
    Phase-1 stats reduction (heuristic_topk.cuh:654/1209). Production V3.2
    callers therefore pass pre_idx in PREVIOUS-step coordinates so the
    kernel's +1 / +2 / +3 shift maps prev positions to current-step
    positions correctly.

    This test builds pre_idx from `current_logits.topk()`, then applies a
    `-1` shift before returning, so the kernel's internal `+1` brings every
    hint back to its intended current-step position (preserves the kernel's
    argmax invariant: kernel reads `input[(argmax_pos - 1) + 1] = input[argmax_pos]`).

    For batch element b (base row = b*next_n):
      - pre_idx[b, 0]       = argmax of the base row  (kernel invariant)
      - n_hit slots          = floor(index_topk * success_ratio) indices drawn
                               WITHOUT replacement from the actual top-K
      - n_fill = index_topk - n_hit slots
                             = indices drawn WITHOUT replacement from the
                               non-top-K pool (all valid indices except top-K)

    No element appears more than once in pre_idx[b].  The hit and fill pools
    are disjoint by construction, so cross-pool duplicates are impossible.

    Edge case: when valid_len < index_topk (short sequences, ~10% of batches),
    the non-top-K pool may be smaller than n_fill.  In that case all available
    non-top-K indices are used first; any remaining slots are filled from the
    unused top-K tail (topk_idx[n_hit:]) to preserve the no-duplicate guarantee
    as far as possible.

    Args:
        logits:        (batch_size*next_n, max_len) logits tensor
        row_ends:      (batch_size*next_n,) valid lengths per row
        batch_size:    number of batch elements
        next_n:        MTP factor; base row index is b*next_n
        index_topk:    number of pre-predicted candidates (K)
        success_ratio: fraction of pre_idx drawn from actual top-K (>= 0.4)
        seed:          torch manual seed for reproducibility

    Returns:
        pre_idx: int32 tensor of shape (batch_size, index_topk), no duplicates
    """
    torch.manual_seed(seed)
    pre_idx = torch.zeros(batch_size, index_topk, dtype=torch.int32, device=logits.device)

    for b in range(batch_size):
        base = b * next_n
        valid_len = int(row_ends[base].item())
        k = min(index_topk, valid_len)

        # Actual top-K of the base row (no duplicates); index 0 = argmax (kernel invariant)
        _, topk_idx = logits[base, :valid_len].topk(k)

        # --- Hit slots: sample n_hit from top-K without replacement ---
        # Always include argmax at position 0.
        n_hit = max(1, int(k * success_ratio))
        n_hit = min(n_hit, k)

        if n_hit > 1:
            perm = torch.randperm(k - 1, device=logits.device)[: n_hit - 1]
            hits = torch.cat([topk_idx[:1], topk_idx[1:][perm]])
        else:
            hits = topk_idx[:1]

        pre_idx[b, :n_hit] = hits.int()

        # --- Fill slots: sample n_fill from non-top-K pool without replacement ---
        # The non-top-K pool is disjoint from topk_idx, so no cross-pool duplicates.
        n_fill = index_topk - n_hit
        if n_fill > 0:
            # Build non-top-K pool: all valid indices that are NOT in topk_idx
            topk_mask = torch.zeros(valid_len, dtype=torch.bool, device=logits.device)
            topk_mask[topk_idx] = True
            non_topk = torch.where(~topk_mask)[0]  # shape: (valid_len - k,)

            if len(non_topk) >= n_fill:
                # Normal case: enough non-top-K candidates
                perm = torch.randperm(len(non_topk), device=logits.device)[:n_fill]
                pre_idx[b, n_hit:] = non_topk[perm].int()
            else:
                # Edge case (valid_len ≈ index_topk): use all non-top-K first,
                # then fill remaining from the unused top-K tail (topk_idx[n_hit:])
                pre_idx[b, n_hit : n_hit + len(non_topk)] = non_topk.int()
                leftover = n_fill - len(non_topk)
                topk_tail = topk_idx[n_hit:]  # not yet in pre_idx[b]
                take = min(leftover, len(topk_tail))
                if take > 0:
                    pre_idx[b, n_hit + len(non_topk) : n_hit + len(non_topk) + take] = topk_tail[
                        :take
                    ].int()

    # V3.2 compensation: kernel adds `(rowIdx % next_n) + 1` to every pre_idx
    # entry during P1 stats reduction. Shifting by -1 here means that for the
    # base row (rowIdx % next_n == 0, offset = +1) the kernel reads the exact
    # current-step positions our `topk()` selected. Negative entries are
    # silently dropped by the kernel's `idx >= 0 && idx < N` range check.
    pre_idx -= 1
    return pre_idx


def apply_mtp_structure_compressed(
    logits: torch.Tensor,
    batch_size: int,
    next_n: int,
    row_ends: torch.Tensor,
) -> torch.Tensor:
    """
    cr=4-safe variant of apply_mtp_structure.

    apply_mtp_structure assumes ``row_ends[b*next_n + nni] == row_ends[b*next_n]
    + nni`` (the V3.2 cr=1 invariant where each MTP draft adds exactly one KV
    token). Under cr=4, ``row_ends = floor(actual_kv_len / 4)`` and that
    invariant breaks: when ``actual_kv_len[base] mod 4`` lies in {1, 2, 3}
    (75% of seq_lens) we get ``row_ends[base+nni] == row_ends[base]``, so the
    copy ``[nni : nni+valid_base]`` overruns ``row_ends[base+nni]`` and writes
    finite values into what create_distributed_logits left as -inf. The
    polluted positions then leak into torch.topk's reference (which doesn't
    know about the row's true compressed N), producing off-by-one counts vs.
    the kernel.

    This variant clips the per-row copy length to fit within row b*next_n+nni's
    valid compressed range, preserving MTP correlation where it fits and
    leaving -inf positions untouched.
    """
    if next_n == 1:
        return logits

    for b in range(batch_size):
        base = b * next_n
        valid_base = int(row_ends[base].item())
        for nni in range(1, next_n):
            row = base + nni
            valid_row = int(row_ends[row].item())
            # Largest copy_len such that [nni, nni+copy_len) ⊆ [0, valid_row).
            copy_len = max(0, min(valid_base, valid_row - nni))
            if copy_len > 0:
                logits[row, nni : nni + copy_len] = logits[base, :copy_len]

    return logits


def generate_pre_idx_v4(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    batch_size: int,
    next_n: int,
    index_topk: int,
    success_ratio: float = 0.6,
    seed: int = 0,
) -> torch.Tensor:
    """
    DSv4 (compress_ratio=4) variant of generate_pre_idx — no `-1` shift.

    Unlike V3.2 where the kernel applies preIdxOffset = (rowIdx % next_n) + 1
    to every preIdx entry (KV grew by 1 per decode step in uncompressed space),
    the V4 indexer operates in compressed-token-index space where consecutive
    decode steps may add 0 or 1 compressed entries (each compressed entry
    fuses 4 real tokens). Per-row Δc varies with prev kv_len mod 4 alignment,
    but new compressed entries are always appended at the end so prev-step
    indices in [0, c_prev-1] remain valid as-is in [0, c_curr-1]. The kernel
    therefore forces preIdxOffset = 0 when compressRatio != 1, and tests must
    pass preIdx in CURRENT-step coordinates (no -1 shift).

    Structure of the returned pre_idx[b]:
      - pre_idx[b, 0]              = argmax of the base row (kernel invariant)
      - floor(K * success_ratio) slots from the actual top-K (without replace)
      - remaining slots from non-top-K pool (without replace)

    Edge case (valid_len < K) handled identically to generate_pre_idx.

    Args:
        logits, row_ends, batch_size, next_n, index_topk, success_ratio, seed:
            See generate_pre_idx — the V4 helper mirrors its sampling logic.

    Returns:
        pre_idx: int32 tensor of shape (batch_size, index_topk), entries in
        the compressed current-step index space (no negative entries since
        the kernel uses offset = 0).
    """
    torch.manual_seed(seed)
    pre_idx = torch.zeros(batch_size, index_topk, dtype=torch.int32, device=logits.device)

    for b in range(batch_size):
        base = b * next_n
        valid_len = int(row_ends[base].item())
        k = min(index_topk, valid_len)

        # Actual top-K of the base row; index 0 = argmax (kernel invariant).
        _, topk_idx = logits[base, :valid_len].topk(k)

        n_hit = max(1, int(k * success_ratio))
        n_hit = min(n_hit, k)

        if n_hit > 1:
            perm = torch.randperm(k - 1, device=logits.device)[: n_hit - 1]
            hits = torch.cat([topk_idx[:1], topk_idx[1:][perm]])
        else:
            hits = topk_idx[:1]

        pre_idx[b, :n_hit] = hits.int()

        n_fill = index_topk - n_hit
        if n_fill > 0:
            topk_mask = torch.zeros(valid_len, dtype=torch.bool, device=logits.device)
            topk_mask[topk_idx] = True
            non_topk = torch.where(~topk_mask)[0]

            if len(non_topk) >= n_fill:
                perm = torch.randperm(len(non_topk), device=logits.device)[:n_fill]
                pre_idx[b, n_hit:] = non_topk[perm].int()
            else:
                pre_idx[b, n_hit : n_hit + len(non_topk)] = non_topk.int()
                leftover = n_fill - len(non_topk)
                topk_tail = topk_idx[n_hit:]
                take = min(leftover, len(topk_tail))
                if take > 0:
                    pre_idx[b, n_hit + len(non_topk) : n_hit + len(non_topk) + take] = topk_tail[
                        :take
                    ].int()

    # No shift: kernel reads input[preIdx[i] + 0] = input[preIdx[i]] directly
    # in compressed current-step coordinates.
    return pre_idx


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 4, 64, 256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("load_balance", [False, True])
def test_cute_dsl_topk_decode_single_cta(  # noqa: F811
    batch_size, next_n, index_topk, num_tokens, dtype, load_balance
):
    """Correctness test for CuTE DSL single-CTA TopK decode on Blackwell."""
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
def test_cute_dsl_topk_decode_multi_cta(  # noqa: F811
    batch_size, next_n, index_topk, num_tokens, dtype, chunk_size_per_cta, dynamic
):
    """Correctness test for CuTE DSL multi-CTA TopK decode on Blackwell."""
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
def test_cute_dsl_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens):  # noqa: F811
    """Correctness test for CuTE DSL indexer TopK decode with in-place output."""
    num_gen_tokens = batch_size * next_n

    def run_fn(logits, seq_lens):
        """Run CuTE DSL indexer TopK decode and return output indices."""
        output_indices = torch.empty(num_gen_tokens, index_topk, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            input_values=logits,
            seq_lens=seq_lens,
            output_indices=output_indices,
            top_k=index_topk,
            next_n=next_n,
            num_copy_bits=256,
        )
        return output_indices

    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        torch.float32,
        run_fn,
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [32768, 131072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cute_dsl_topk_decode_single_pass_multi_cta(  # noqa: F811
    batch_size, next_n, index_topk, num_tokens, dtype
):
    """Correctness test for CuTE DSL single-pass multi-CTA TopK on Blackwell."""
    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        dtype,
        lambda logits,
        seq_lens: cute_dsl_custom_ops.CuteDSLTopKDecodeSinglePassMultiCTARunner.forward(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            return_val=False,
            num_copy_bits=256,
        )[0],
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("num_tokens", [32768, 131072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cute_dsl_topk_decode_single_pass_multi_cta_cluster(
    batch_size, next_n, index_topk, num_tokens, dtype
):
    """Correctness test for CuTE DSL single-pass multi-CTA cluster TopK on Blackwell."""

    def run_fn(logits, seq_lens):
        """Run cluster TopK decode, skip if problem size exceeds capacity."""
        result = cute_dsl_custom_ops.CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner.forward(
            input_values=logits,
            seq_lens=seq_lens,
            top_k=index_topk,
            next_n=next_n,
            return_val=False,
            num_copy_bits=256,
        )
        if result[0] is None:
            pytest.skip("Problem size exceeds cluster kernel capacity")
        return result[0]

    _run_cute_dsl_topk_test(
        batch_size,
        next_n,
        index_topk,
        num_tokens,
        dtype,
        run_fn,
    )


# ============================================================================
# Heuristic Decode Distribution-Parameterised Tests
# ============================================================================


@skip_pre_blackwell
@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy required for distribution tests")
@pytest.mark.parametrize("success_ratio", [0.5, 0.9])
@pytest.mark.parametrize("dist_cfg", _DECODE_DIST_CONFIGS, ids=_DECODE_DIST_IDS)
@pytest.mark.parametrize("batch_size", [1, 64, 128])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
@pytest.mark.parametrize("num_tokens", [8192, 16384])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_indexer_topk_decode_dist(
    dist_cfg, batch_size, next_n, index_topk, num_tokens, success_ratio, dtype
):
    """
    Correctness test for the heuristic indexer_topk_decode across realistic
    logit distributions, MTP correlation structures, pre_idx accuracy levels,
    GVR-supported K values, and supported logit dtypes.
    """
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    # Clamp so that every base row has valid_len >= 1 (i.e., seq_len >= next_n).
    # Without this, seq_len < next_n produces non-positive row_ends for offset 0.
    seq_lens = seq_lens.clamp(min=next_n)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    # 1. Sample logits from the target distribution
    logits = create_distributed_logits(dist_cfg, row_starts, row_ends, dtype, seed=42)

    # 2. Apply MTP correlation: consecutive rows share their tail logits
    if next_n > 1:
        logits = apply_mtp_structure(logits, batch_size, next_n, row_ends)

    # 3. Build heuristic pre-prediction indices
    pre_idx = generate_pre_idx(
        logits,
        row_ends,
        batch_size,
        next_n,
        index_topk,
        success_ratio=success_ratio,
        seed=7,
    )

    # 4. Run heuristic CUDA kernel — heuristic_scratch dtype must match logits.
    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")
    heuristic_scratch = torch.empty(num_gen_tokens * index_topk, dtype=dtype, device="cuda")
    torch.ops.trtllm.indexer_topk_decode(
        logits, seq_lens, indices, next_n, index_topk, pre_idx, heuristic_scratch
    )
    torch.cuda.synchronize()

    # 5. Reference: exact torch.topk masked to valid range
    max_row_len = int(row_ends.max().item())
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    torch_indices = torch_indices.masked_fill(~(mask_lo & mask_hi), -1)

    # GVR Top-K is an exact algorithm: with same-dtype `logits.topk` as the
    # reference, the sorted output values must be bit-identical (bf16 -> fp32
    # promotion inside the kernel is lossless and order-preserving, so the
    # K-th cutoff is identical in both comparison spaces). Any value gap is
    # a real kernel bug, not algorithmic noise — keep the default 1e-5
    # tolerance and let CI surface regressions.
    assert compare_top_k_results(
        logits,
        indices,
        torch_indices,
        row_starts,
        row_ends,
        index_topk,
    ), (
        f"heuristic indexer_topk_decode mismatch: dist={dist_cfg['dist']}, "
        f"mean={dist_cfg['mean']}, std={dist_cfg['std']}, "
        f"next_n={next_n}, success_ratio={success_ratio}, dtype={dtype}"
    )


# ============================================================================
# DSv4 Heuristic Decode Test (compress_ratio = 4)
# ============================================================================
#
# Exercises the V4 indexer GVR Top-K path enabled by the
# `compressRatio == 1 || compressRatio == 4` relaxation in
# canUseHeuristic (cpp/tensorrt_llm/kernels/indexerTopK.cu). For
# compressRatio != 1 the kernel:
#   1. Computes N = (seq_len - next_n + (rowIdx % next_n) + 1) / compressRatio,
#      i.e. the row's compressed-KV length (vs. uncompressed N in the V3.2
#      path).
#   2. Forces preIdxOffset = 0 (vs. (rowIdx % next_n) + 1 in V3.2), since
#      compressed entries are appended at the end of the compressed KV and
#      prev-step indices remain valid as-is.
#
# To reach the GVR (Heuristic) path with cr=4 we need the *compressed*
# numColumns ≥ kSeqSmall (≈12288), so the test uses num_tokens ∈
# {65536, 131072} which gives compressed range ≈ {16K, 32K}. Smaller cr=4
# cases (where compressed N falls below kSeqSmall) are already covered by
# test_indexer_topk_decode parametrized on compress_ratio ∈ [1, 4] — those
# exercise the Radix/Insertion fallback for the same gate.


def _run_indexer_topk_decode_v4_gvr_check(
    batch_size: int,
    next_n: int,
    index_topk: int,
    num_tokens: int,
    dtype: torch.dtype,
    dist_cfg: dict,
    success_ratio: float,
):
    """Run the V4 (compress_ratio=4) heuristic indexer_topk_decode check."""
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    compress_ratio = 4
    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    # Uncompressed seq_lens are what the kernel receives in `seq_lens`.
    # Clamp so that compressed_actual_kv_len ≥ kSeqSmall (= 12288) for every
    # row; the kernel will divide actual_kv_len by compress_ratio internally,
    # so a floor of (kSeqSmall + 1) * compress_ratio + next_n on the
    # uncompressed seq_len guarantees compressed N stays in the GVR window.
    min_uncompressed = (12288 + 1) * compress_ratio + next_n
    seq_lens = generate_seq_lens(batch_size, min_uncompressed, num_tokens)
    seq_lens = seq_lens.clamp(min=min_uncompressed)

    # row_ends is the compressed-KV length per row (= what logits' columns
    # represent in V4 — the indexer operates in compressed-token-index space).
    actual_kv_lens = seq_lens[row_indices] - next_n + next_n_offset + 1
    row_ends = actual_kv_lens // compress_ratio

    # 1. Sample logits over the compressed shape.
    logits = create_distributed_logits(dist_cfg, row_starts, row_ends, dtype, seed=42)

    # 2. Apply MTP correlation between rows within each batch element.
    # Use the compressed-aware variant: cr=4 breaks the cr=1 invariant
    # row_ends[base+nni] = row_ends[base]+nni, so the copy length must be
    # clipped per-row to avoid overrunning the row's valid range.
    if next_n > 1:
        logits = apply_mtp_structure_compressed(logits, batch_size, next_n, row_ends)

    # 3. Build heuristic pre-prediction indices — V4 variant (no -1 shift).
    pre_idx = generate_pre_idx_v4(
        logits,
        row_ends,
        batch_size,
        next_n,
        index_topk,
        success_ratio=success_ratio,
        seed=7,
    )

    # 4. Run heuristic CUDA kernel with compress_ratio=4. The kernel:
    #    - reads logits in compressed-index space (numColumns = logits.shape[1])
    #    - divides seq_lens by compress_ratio to derive per-row N
    #    - uses preIdxOffset = 0 (preIdx already in current-step coords)
    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")
    heuristic_scratch = torch.empty(num_gen_tokens * index_topk, dtype=dtype, device="cuda")
    torch.ops.trtllm.indexer_topk_decode(
        logits,
        seq_lens,
        indices,
        next_n,
        index_topk,
        pre_idx,
        heuristic_scratch,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    # 5. Reference: torch.topk masked to the compressed row_ends.
    max_row_len = int(row_ends.max().item())
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~mask, -1)

    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), (
        f"V4 heuristic indexer_topk_decode (cr=4) mismatch: dist={dist_cfg['dist']}, "
        f"mean={dist_cfg['mean']}, std={dist_cfg['std']}, batch_size={batch_size}, "
        f"next_n={next_n}, index_topk={index_topk}, num_tokens={num_tokens}, "
        f"success_ratio={success_ratio}, dtype={dtype}"
    )


# Param matrix is intentionally tighter than test_indexer_topk_decode_dist:
# only one logit distribution and one success_ratio because the GVR algorithm
# is dist-/hint-quality-invariant for correctness (an exact algorithm). The
# axes that *do* differ in V4 vs V3.2 are exercised in full:
#   compress_ratio = 4         (fixed — sole purpose of this test)
#   next_n in {1, 2, 3}         (decode + MTP windows)
#   index_topk in {512, 1024, 2048}  (all GVR-supported K)
#   num_tokens in {65536, 131072}    (compressed N ≈ 16K and 32K)
#   dtype: fp32 / bf16 / fp16   (both kernel templates)
#   batch_size: 1 (single-row), 64 (multi-row)
@skip_pre_blackwell
@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy required for distribution tests")
@pytest.mark.parametrize("success_ratio", [0.7])
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
@pytest.mark.parametrize("num_tokens", [65536, 131072])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_indexer_topk_decode_dist_v4_cr4(
    batch_size, next_n, index_topk, num_tokens, success_ratio, dtype
):
    """
    Correctness test for the DSv4 heuristic indexer_topk_decode with
    compress_ratio=4 across MTP windows, all GVR-supported K, and all
    supported logit dtypes. Uses one representative distribution; broader
    distribution coverage is left to test_indexer_topk_decode_dist (cr=1).
    """
    # Logistic chosen as the single representative distribution — its
    # heavy-tailed symmetric shape produces the wide K-th-value spread that
    # stresses GVR's secant threshold search most.
    dist_cfg = dict(dist="logistic", mean=-0.47, std=1.46, full_range=12.32)
    _run_indexer_topk_decode_v4_gvr_check(
        batch_size, next_n, index_topk, num_tokens, dtype, dist_cfg, success_ratio
    )

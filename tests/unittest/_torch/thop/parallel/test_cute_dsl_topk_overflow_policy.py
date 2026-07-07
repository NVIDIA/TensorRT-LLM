# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for overflow policies in the DSL radix top-k kernels.

Overflow policies:
  GMEM_SPILL    – spill threshold-bucket overflow to a pre-allocated GMEM buffer (exact)
  TRUNCATE      – discard overflow elements; histogram stays consistent     (non-exact)
  REREAD_ALWAYS – first pass builds histogram only; second GMEM scan fills s_input_idx (exact)
  REREAD        – optimistic SMEM collection; falls back to GMEM re-scan if overflow occurs (exact)

Coverage (overflow scenarios only — all policies are identical when num_cols <= smem_input_size):
  * decode single-CTA: overflow num_tokens, large and small batch, multiple dtypes/shapes
  * decode REREAD: enable_reread=True (num_tokens > smem_size) but did_overflow=False at runtime
  * prefill: overflow with zero and non-zero row_starts
  * GMEM_SPILL / REREAD_ALWAYS / REREAD verified with compare_top_k_results (exact match)
  * TRUNCATE verified with compare_truncate_result (valid-subset check)

SMEM overflow thresholds (large_occupancy path, B200):
  bf16 / fp16, Uint16 index, num_buffer=1: smem_input_size = 16384
  fp32,        Uint16 index, num_buffer=2: smem_input_size =  8192
  Overflow occurs when num_cols > smem_input_size.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLTopKPrefillSingleCTARunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.filtered_top_k_decode_varlen import (
    cute_dsl_topk_wrapper,
    generate_seq_lens,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.filtered_top_k_varlen_util import (
    compare_top_k_results,
    create_random_logits,
)

# ── reference comparison helpers ──────────────────────────────────────────────


def _build_torch_ref(logits, row_starts, row_ends, top_k):
    """Exact-copy of the reference computation in existing tests."""
    max_row_len = int(row_ends.max().item())
    torch_indices = logits.topk(min(top_k, max_row_len), dim=-1)[1]
    mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~mask, -1)
    return torch_indices


def _compare_truncate_result(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
) -> bool:
    """TRUNCATE correctness check.

    TRUNCATE is non-exact: it returns top-K of the first smem_size elements
    (memory order) in the threshold-coarse bin, NOT the global top-K.  With
    adversarial data layout the returned values can legitimately be below the
    global k-th largest, so comparing against torch.topk would be incorrect.

    We verify instead:
      1. Each row returns exactly min(top_k, row_len) valid (non -1) elements.
      2. All returned indices are within [row_start, row_end).
      3. No duplicate indices within a row.

    cuda_indices must contain absolute indices into logits (not row-local).
    """
    invalid_mask = cuda_indices == -1  # [num_rows, K]
    valid_counts = (~invalid_mask).sum(dim=1)  # [num_rows]

    # 1. count check
    row_lens = row_ends - row_starts
    expected = torch.minimum(row_lens, row_lens.new_full((), top_k))
    bad_rows = (valid_counts != expected).nonzero(as_tuple=True)[0]
    if bad_rows.numel() > 0:
        r = bad_rows[0].item()
        print(
            f"TRUNCATE Row {r}: expected {expected[r].item()} valid elements, got {valid_counts[r].item()}"
        )
        return False

    # 2. bounds check: safe-clamp -1 slots to 0 before comparison, then mask them out
    safe = cuda_indices.clone()
    safe[invalid_mask] = 0
    out_of_bounds = (
        (safe < row_starts.unsqueeze(1)) | (safe >= row_ends.unsqueeze(1))
    ) & ~invalid_mask
    bad_rows = out_of_bounds.any(dim=1).nonzero(as_tuple=True)[0]
    if bad_rows.numel() > 0:
        r = bad_rows[0].item()
        bad_vals = safe[r][out_of_bounds[r]]
        print(
            f"TRUNCATE Row {r}: index out of [row_start={row_starts[r].item()}, "
            f"row_end={row_ends[r].item()}): {bad_vals[:4].tolist()}"
        )
        return False

    # 3. no-duplicate check: replace invalid with INT_MAX so they sort to the end,
    # then compare adjacent entries that are not sentinel.
    sentinel = torch.iinfo(torch.int32).max
    safe_for_dup = cuda_indices.clone()
    safe_for_dup[invalid_mask] = sentinel
    sorted_dup = safe_for_dup.sort(dim=1).values  # [num_rows, K]
    dups = (sorted_dup[:, 1:] == sorted_dup[:, :-1]) & (sorted_dup[:, 1:] != sentinel)
    bad_rows = dups.any(dim=1).nonzero(as_tuple=True)[0]
    if bad_rows.numel() > 0:
        r = bad_rows[0].item()
        print(f"TRUNCATE Row {r}: duplicate indices found")
        return False

    return True


# ── decode kernel run via wrapper ─────────────────────────────────────────────


def _run_decode_kernel(logits, seq_lens, top_k, next_n, overflow_policy):
    """Run decode kernel via cute_dsl_topk_wrapper with given overflow_policy."""
    indices, _ = cute_dsl_topk_wrapper(
        logits,
        seq_lens,
        top_k,
        next_n,
        return_val=False,
        overflow_policy=overflow_policy,
    )
    return indices


# ── shared test bodies ────────────────────────────────────────────────────────


def _run_decode_policy_test(policy, batch_size, next_n, top_k, num_tokens, dtype, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_gen_tokens = batch_size * next_n
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    seq_lens = generate_seq_lens(batch_size, top_k, num_tokens)
    seq_lens = seq_lens.clamp(min=next_n)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    logits = create_random_logits(row_starts, row_ends, dtype, seed)

    cuda_indices = _run_decode_kernel(logits, seq_lens, top_k, next_n, policy)
    torch.cuda.synchronize()

    cuda_indices = cuda_indices.to(torch.int32)

    if policy == "TRUNCATE":
        assert _compare_truncate_result(logits, cuda_indices, row_starts, row_ends, top_k), (
            f"TRUNCATE decode: invalid results (policy={policy})"
        )
    else:
        torch_indices = _build_torch_ref(logits, row_starts, row_ends, top_k)
        assert compare_top_k_results(
            logits, cuda_indices, torch_indices, row_starts, row_ends, top_k
        ), f"Decode results mismatch vs torch.topk (policy={policy})"


def _run_prefill_policy_test(
    policy, batch_size, top_k, num_tokens, dtype, row_start_offset=0, seed=77
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_lens = generate_seq_lens(batch_size, top_k, num_tokens)
    num_rows = int(seq_lens.sum().item())

    row_indices = torch.arange(1, seq_lens.max() + 1, dtype=torch.int32, device="cuda")
    row_lengths = row_indices.expand(seq_lens.size(0), -1)[
        row_indices.expand(seq_lens.size(0), -1) <= seq_lens.unsqueeze(1)
    ].contiguous()

    row_starts = torch.full((num_rows,), row_start_offset, dtype=torch.int32, device="cuda")
    row_ends = (row_starts + row_lengths).contiguous()

    logits = create_random_logits(row_starts, row_ends, dtype, seed)

    cuda_local, _ = CuteDSLTopKPrefillSingleCTARunner.forward(
        logits,
        row_starts,
        row_ends,
        top_k,
        return_val=False,
        overflow_policy=policy,
    )

    if policy == "TRUNCATE":
        # TRUNCATE returns LOCAL indices; convert reference to LOCAL too.
        max_row_end = int(row_ends.max().item())
        torch_abs = logits.topk(min(top_k, max_row_end), dim=-1)[1].to(torch.int32)
        torch_local = torch_abs - row_starts.unsqueeze(1)
        valid = (torch_abs >= row_starts.unsqueeze(1)) & (torch_abs < row_ends.unsqueeze(1))
        torch_local = torch_local.masked_fill(~valid, -1)

        # For TRUNCATE with local indices: build absolute cuda_indices for comparison.
        cuda_abs = cuda_local.clone()
        not_neg1 = cuda_abs != -1
        cuda_abs[not_neg1] = (
            cuda_abs[not_neg1] + row_starts.unsqueeze(1).expand_as(cuda_abs)[not_neg1]
        )
        # compare_truncate expects absolute indices in logits space and row_starts=0 for absolute check.
        # Easier: rebuild logits for absolute-index comparison using row_starts.
        assert _compare_truncate_result(logits, cuda_abs, row_starts, row_ends, top_k), (
            f"TRUNCATE prefill: invalid results (policy={policy})"
        )
    else:
        # Exact policy: compare LOCAL indices using compare_top_k_results.
        max_row_end = int(row_ends.max().item())
        torch_abs = logits.topk(min(top_k, max_row_end), dim=-1)[1].to(torch.int32)
        torch_local = torch_abs - row_starts.unsqueeze(1)
        valid = (torch_abs >= row_starts.unsqueeze(1)) & (torch_abs < row_ends.unsqueeze(1))
        torch_local = torch_local.masked_fill(~valid, -1)
        assert compare_top_k_results(
            logits, cuda_local, torch_local, row_starts, row_ends, top_k
        ), f"Prefill results mismatch vs torch.topk (policy={policy})"


# ═══════════════════════════════════════════════════════════════════════════════
# Parametrized tests
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_POLICIES = ["GMEM_SPILL", "TRUNCATE", "REREAD_ALWAYS", "REREAD"]

# ----------------------------------------------------------------------------
# Decode single-CTA — overflow (num_tokens > smem_input_size)
# bf16: smem_size=16384 → overflow at 32768
# fp32: smem_size=8192  → overflow at 16384
# Use large batch_size (256) to trigger large_occupancy path.
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("overflow_policy", _ALL_POLICIES)
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "num_tokens, dtype",
    [
        (32768, torch.bfloat16),  # bf16: 32768 > smem_size 16384 → overflow
        (65536, torch.bfloat16),  # bf16: 65536 >> smem_size
        (16384, torch.float32),  # fp32: 16384 > smem_size  8192 → overflow
        (32768, torch.float32),  # fp32: 32768 >> smem_size
    ],
)
def test_decode_overflow_policy_overflow(
    overflow_policy, batch_size, next_n, top_k, num_tokens, dtype
):
    """Decode single-CTA: all policies, large num_tokens (SMEM overflow triggered)."""
    _run_decode_policy_test(overflow_policy, batch_size, next_n, top_k, num_tokens, dtype)


# ----------------------------------------------------------------------------
# Decode — REREAD policy: enable_reread=True but did_overflow=False at runtime
#
# REREAD has two compile-time conditions:
#   enable_reread=True  requires num_tokens > filtered_topk_smem_input_size
#                       (large_occ bf16: 16384, fp32: 8192)
#   did_overflow=False  at runtime: threshold-bin count fits in SMEM
#
# With uniform random data and top_k=512, the threshold bin holds
#   ~num_tokens / 256 ≈ 128 elements (bf16@32768) or 64 elements (fp32@16384),
# both far below smem_input_size → did_overflow=False → SMEM-refinement path.
#
# This test exercises the REREAD path that test_decode_overflow_policy_no_overflow
# cannot cover (those use num_tokens ≤ smem_size → enable_reread=False).
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("batch_size", [1, 64, 256])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    "num_tokens, dtype",
    [
        (32768, torch.bfloat16),  # large_occ bf16: enable_reread=True, threshold-bin ~128 << 16384
        (16384, torch.float32),  # large_occ fp32: enable_reread=True, threshold-bin ~64  <<  8192
    ],
)
def test_decode_reread_no_smem_overflow(batch_size, next_n, top_k, num_tokens, dtype):
    """REREAD: enable_reread=True (num_tokens > smem_size) but runtime did_overflow=False.

    Exercises the SMEM-based refinement path inside the REREAD policy.
    """
    _run_decode_policy_test("REREAD", batch_size, next_n, top_k, num_tokens, dtype)


# ----------------------------------------------------------------------------
# Decode — small batch (no large_occupancy), overflow path uses base-class smem_size
# Base-class: bf16 Uint16 num_buffer=1 → max_smem=64K → smem_size=min(64K, max_cols)
# Only overflows at very large num_tokens (> 65536); use 131072.
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("overflow_policy", _ALL_POLICIES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    "num_tokens, dtype",
    [
        (131072, torch.bfloat16),
        (131072, torch.float32),
    ],
)
def test_decode_overflow_policy_small_batch_overflow(
    overflow_policy, batch_size, next_n, top_k, num_tokens, dtype
):
    """Decode small batch (no large_occupancy): overflow at very large num_tokens."""
    _run_decode_policy_test(overflow_policy, batch_size, next_n, top_k, num_tokens, dtype)


# ----------------------------------------------------------------------------
# Prefill — overflow (num_tokens > smem_input_size)
# Prefill large_occupancy smem_input_size: bf16 Uint16 num_buffer=1 → 16384
#                                           fp32 Uint16 num_buffer=2 →  8192
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTE DSL not available")
@skip_pre_blackwell
@pytest.mark.parametrize("overflow_policy", _ALL_POLICIES)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "num_tokens, dtype",
    [
        (20480, torch.bfloat16),  # bf16: 20480 > smem_size 16384 → overflow
        (12288, torch.float32),  # fp32: 12288 > smem_size  8192 → overflow
    ],
)
@pytest.mark.parametrize("row_start_offset", [256])
def test_prefill_overflow_policy_overflow(
    overflow_policy, batch_size, top_k, num_tokens, dtype, row_start_offset
):
    """Prefill: all policies, large num_tokens (SMEM overflow triggered)."""
    # Skip combinations where the logits tensor would exceed ~80 GB.
    # logits shape = (sum_seq_lens, num_tokens); sum_seq_lens ≈ batch_size * num_tokens / 2.
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4
    est_gb = batch_size * (num_tokens // 2) * num_tokens * dtype_bytes / (1024**3)
    if est_gb > 80:
        pytest.skip(
            f"Estimated logits tensor ~{est_gb:.0f} GB exceeds memory budget; "
            f"use smaller batch_size or num_tokens"
        )
    _run_prefill_policy_test(
        overflow_policy,
        batch_size,
        top_k,
        num_tokens,
        dtype,
        row_start_offset=row_start_offset,
    )

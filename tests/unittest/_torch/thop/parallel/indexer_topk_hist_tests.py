# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
Correctness worker for the env-gated ported coarse-histogram indexer top-k
(``TRTLLM_DSA_TOPK_HIST``).

This module is NOT auto-collected (no ``test_`` filename prefix). It is launched
in a fresh subprocess by ``test_indexer_topk_hist.py`` with the env set, because
the C++ gate reads ``TRTLLM_DSA_TOPK_HIST`` exactly once per process
(``std::call_once`` in ``indexerTopK.cu``) and so cannot be toggled mid-pytest.

Every shape here is deliberately inside the ported kernel's supported set
(``index_topk`` in {512,1024,2048}, ``num_rows = batch*next_n <= 512``, unit
inner stride, ``compress_ratio`` in {1,4}), so the gate WILL route through the
port rather than silently falling back to the stock split/merge kernel. Both
paths are validated against ``torch.topk`` via the shared ``compare_top_k_results``
(value-set equivalence with the histogram-bin-boundary tie tolerance).
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for indexer_topk tests", allow_module_level=True)

if os.environ.get("TRTLLM_DSA_TOPK_HIST") != "1":
    # Guard against accidental direct invocation: without the gate set this file
    # would just re-test the stock path. The driver test always sets it.
    pytest.skip(
        "indexer_topk_hist_tests must run with TRTLLM_DSA_TOPK_HIST=1 "
        "(driven by test_indexer_topk_hist.py)",
        allow_module_level=True,
    )

# Reuse the stock indexer-topk test's helpers verbatim (same directory; pytest's
# prepend import mode puts this dir on sys.path).
from test_indexer_topk import (  # noqa: E402
    _build_decode_inputs,
    _build_radix_aux_buffers,
    _run_indexer_topk_decode_check,
    compare_top_k_results,
)

# Random-seq-len shapes: cover the short-row / Register4 tiers plus batch,
# next_n and compress_ratio variety. index_topk restricted to the supported
# {512,1024,2048}; num_rows = batch*next_n kept <= 512 (kClusterMaxBatch).
_HIST_RANDOM_SHAPES = [
    # (batch_size, next_n, index_topk, num_tokens, compress_ratio)
    (1, 1, 2048, 4096, 1),
    (1, 6, 2048, 8192, 1),  # MTP-5 production row count (6 rows)
    (64, 1, 512, 8192, 4),  # compress_ratio == 4
    (256, 1, 1024, 4096, 1),
    (512, 1, 2048, 4096, 4),  # max supported batch (num_rows == kClusterMaxBatch)
]

# Deterministic full-length rows so row_end is fixed and each dispatch tier is
# hit on purpose. Small batches (num_rows <= 15) use cluster_floor = 32768.
# Tiers (rowEnd ~ num_tokens / compress_ratio):
#   <= 16384         -> Register4 single block
#   16384..32768     -> Streaming single block
#   > 32768          -> 8-block DSMEM cluster
#   >> cluster_floor -> cluster + single-CTA exact-radix OVERFLOW fallback,
#                       the path that fixes the >=256k tie-buffer-truncation bug.
_HIST_FULLLEN_SHAPES = [
    # (batch_size, next_n, index_topk, num_tokens, compress_ratio)
    (1, 6, 2048, 8192, 1),  # Register4
    (1, 6, 2048, 16384, 1),  # Register4 boundary
    (1, 6, 2048, 32768, 1),  # Streaming
    (1, 6, 2048, 65536, 1),  # cluster
    (1, 6, 2048, 131072, 1),  # cluster + overflow fallback
    (1, 6, 2048, 262144, 1),  # cluster + overflow (the fixed >=256k bug)
    (1, 6, 2048, 364544, 1),  # ~workload p99 input length
    (2, 6, 2048, 262144, 1),  # multi-row overflow
    (1, 6, 2048, 262144, 4),  # compress_ratio == 4, large context
    (1, 1, 2048, 1048576, 1),  # max_seq_len extreme overflow
]


def _run_fulllen_check(batch_size, next_n, index_topk, num_tokens, compress_ratio):
    """Full-length-row decode equivalence check vs torch.topk.

    Forces every row to span the whole window (via the stock test's
    ``_build_decode_inputs``) so the on-device row_end deterministically lands
    in the intended dispatch tier, then compares the ported kernel's selection
    against torch.topk by value-set."""
    logits, seq_lens, row_starts, row_ends = _build_decode_inputs(
        batch_size, next_n, index_topk, num_tokens, compress_ratio
    )
    num_rows = logits.shape[0]

    indices = torch.empty((num_rows, index_topk), dtype=torch.int32, device="cuda")
    # The cpp op requires caller-owned Radix aux scratch when blocks_per_row > 1
    # (large num_columns); the ported path ignores it. Supplying it is harmless
    # and keeps the op unconditionally accepted. See test_indexer_topk.py.
    aux_indices, aux_logits = _build_radix_aux_buffers(num_rows, index_topk)

    torch.ops.trtllm.indexer_topk_decode(
        logits,
        seq_lens,
        indices,
        next_n,
        index_topk,
        compress_ratio=compress_ratio,
        radix_aux_indices=aux_indices,
        radix_aux_logits=aux_logits,
    )
    torch.cuda.synchronize()

    max_row_len = int(row_ends.max().item())
    if max_row_len == 0:
        return
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~mask, -1)

    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), (
        "ported histogram top-k mismatch vs torch.topk at full-length shape "
        f"{(batch_size, next_n, index_topk, num_tokens, compress_ratio)}"
    )


@pytest.mark.parametrize(
    "batch_size,next_n,index_topk,num_tokens,compress_ratio", _HIST_RANDOM_SHAPES
)
def test_hist_decode_random(batch_size, next_n, index_topk, num_tokens, compress_ratio):
    """Random-data decode equivalence, exercising the ported kernel's short-row
    and Register4 tiers across batch / next_n / compress_ratio."""
    _run_indexer_topk_decode_check(batch_size, next_n, index_topk, num_tokens, compress_ratio)


@pytest.mark.parametrize(
    "batch_size,next_n,index_topk,num_tokens,compress_ratio", _HIST_FULLLEN_SHAPES
)
def test_hist_decode_fulllen_dispatch_tiers(
    batch_size, next_n, index_topk, num_tokens, compress_ratio
):
    """Deterministic full-length rows across every dispatch tier, including the
    >=256k cluster overflow fallback (regression for the tie-buffer-truncation
    bug)."""
    _run_fulllen_check(batch_size, next_n, index_topk, num_tokens, compress_ratio)


def _run_peaked_check(num_tokens, index_topk):
    """Peaked-row regression. A row of distinct fp32 values in a tiny range collapses
    >kMaxNumTie candidates into one fp16 coarse bin. Before the fix the Register4/Streaming
    tiers truncated to an arbitrary subset (~1500-1900 of 2048 wrong); every tier must now
    recover the exact top-K via the exact-radix overflow fallback."""
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    num_rows, next_n = 1, 1
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    max_len = (num_tokens + 7) & ~7
    logits = torch.full((num_rows, max_len), float("-inf"), device="cuda", dtype=torch.float32)
    logits[0, :num_tokens] = 0.5 + 1e-3 * torch.rand(num_tokens, device="cuda", dtype=torch.float32)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")

    indices = torch.empty((num_rows, index_topk), dtype=torch.int32, device="cuda")
    aux_indices, aux_logits = _build_radix_aux_buffers(num_rows, index_topk)
    torch.ops.trtllm.indexer_topk_decode(
        logits,
        seq_lens,
        indices,
        next_n,
        index_topk,
        compress_ratio=1,
        radix_aux_indices=aux_indices,
        radix_aux_logits=aux_logits,
    )
    torch.cuda.synchronize()

    torch_indices = logits.topk(min(index_topk, num_tokens), dim=-1)[1]
    mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~mask, -1)
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), f"peaked-row histogram top-k mismatch vs torch.topk at N={num_tokens} K={index_topk}"


# num_tokens spans Register4 (<=16384) / Streaming (<=32768) / Cluster (>32768).
@pytest.mark.parametrize("num_tokens", [8192, 16384, 32768, 65536, 262144])
@pytest.mark.parametrize("index_topk", [512, 2048])
def test_hist_decode_peaked_overflow(num_tokens, index_topk):
    """Regression for the tie-buffer-truncation blocker across all three tiers:
    a peaked distribution forces the overflow path on every tier."""
    _run_peaked_check(num_tokens, index_topk)


# The exact single-block shapes the pre-fix port returned WRONG: on a
# peaked row it truncated the tie buffer to an arbitrary 2048-subset and
# mis-selected the following counts vs torch.topk (num_tokens, index_topk, #wrong):
_HIST_BLOCKER1_FAIL_SHAPES = [
    (8192, 2048),  # Register4 -- was 1520 / 2048 wrong
    (16384, 2048),  # Register4 -- was 1777 / 2048 wrong
    (32768, 2048),  # Streaming -- was 1916 / 2048 wrong
]


@pytest.mark.parametrize("num_tokens,index_topk", _HIST_BLOCKER1_FAIL_SHAPES)
def test_hist_decode_blocker1_regression(num_tokens, index_topk):
    """Named guard for the confirmed blocker-#1 failures: the single-block
    Register4/Streaming tiers must now return the exact top-K on the very shapes
    that previously mis-selected ~1500-1900 of 2048 indices."""
    _run_peaked_check(num_tokens, index_topk)


# ===========================================================================
# Concurrency regressions for the two decode-hang fixes (indexerTopKHist.cu):
#   Fix A -- hoist pdlWaitPrimary() before the seqLens read (per-rank rowEnd race).
#   Fix B -- cluster.sync() (not __syncthreads()) at the cluster overflow guard
#            (histogram/tie union cross-rank TOCTOU).
#
# WHY THE CORRECTNESS TESTS ABOVE MISSED THE HANG: they exercise the cluster path
# (num_tokens >> 65536) but on a QUIET GPU, so a row's 8 cluster blocks advance
# near-lockstep and the Fix-B race window (a fast peer's remote tie-merge clobbering
# rank 0's histogram[threshold_bin] before rank 0 reads it) never opens; and they
# launch the kernel standalone with no PDL predecessor mutating seqLens (Fix A).
# Both bugs are timing/contention-dependent, so a correctness-only quiet-GPU run is
# green even when the serving stack deadlocks (HangDetector -> MPI_Abort).
#
# NOTE on Fix A: the cross-grid PDL/seqLens visibility race needs a predecessor grid
# mutating kv_lens_cuda concurrently under overlap-scheduler+MTP -- a serving-stack
# condition not reconstructible deterministically in a standalone kernel unit test.
# It is covered at the e2e/integration level. The tests below target Fix B, which
# is unit-reachable, plus a compute-sanitizer pass that flags BOTH barrier-divergence
# signatures structurally.
# ===========================================================================


def _cluster_stress_inputs(num_tokens=131072):
    # small batch => cluster_floor 32768 (the tail/low-load exposure shape the hang
    # preferred); long context => the DSMEM 8-block cluster path with the overflow guard.
    return _build_decode_inputs(1, 6, 2048, num_tokens, 1)


@pytest.mark.timeout(240)
def test_hist_decode_cluster_deadlock_stress():
    """Behavioral regression for the Fix-B cluster deadlock. Repeatedly launch the
    cluster tier while a concurrent stream floods the SMs to skew inter-block progress
    and OPEN the union-TOCTOU race window. If the overflow guard regresses to a
    block-local __syncthreads(), a subset of a cluster diverges on cluster.sync() and
    the device deadlocks -> the periodic torch.cuda.synchronize() below (and the
    driver's subprocess timeout) never returns. Correctness is re-verified each round."""
    logits, seq_lens, row_starts, row_ends = _cluster_stress_inputs()
    num_rows = logits.shape[0]
    indices = torch.empty((num_rows, 2048), dtype=torch.int32, device="cuda")
    aux_indices, aux_logits = _build_radix_aux_buffers(num_rows, 2048)

    max_row_len = int(row_ends.max().item())
    torch_indices = logits.topk(min(2048, max_row_len), dim=-1)[1]
    _mask = (torch_indices >= 0) & ((torch_indices - (row_ends - row_starts)[:, None]) < 0)
    torch_indices = torch_indices.masked_fill(~_mask, -1)

    contention = torch.cuda.Stream()
    flood = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    iters = int(os.environ.get("TRTLLM_HIST_STRESS_ITERS", "600"))
    for it in range(iters):
        with torch.cuda.stream(contention):
            for _ in range(3):
                flood = torch.mm(flood, flood)  # keep SMs busy against the topk cluster
        torch.ops.trtllm.indexer_topk_decode(
            logits,
            seq_lens,
            indices,
            6,
            2048,
            compress_ratio=1,
            radix_aux_indices=aux_indices,
            radix_aux_logits=aux_logits,
        )
        if it % 50 == 0:
            torch.cuda.synchronize()  # a Fix-B regression deadlocks HERE -> timeout
            assert compare_top_k_results(
                logits, indices, torch_indices, row_starts, row_ends, 2048
            ), f"cluster-path top-k mismatch under contention at iter {it}"
    torch.cuda.synchronize()


def test_hist_decode_cluster_min_for_sanitizer():
    """One small, fast cluster-path launch (rowEnd 65536 > cluster_floor) used as the
    target for the compute-sanitizer synccheck/racecheck driver pass -- kept minimal so
    the (instrumented, slow) sanitizer run stays bounded."""
    _run_fulllen_check(1, 6, 2048, 65536, 1)

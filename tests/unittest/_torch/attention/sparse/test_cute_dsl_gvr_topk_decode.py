# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute import runtime as _crt

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_dispatch as _tier_dispatch,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import (
    GvrTopKKernel as _GvrTopKKernel,
)
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL GVR Top-K only supports SM 100/103, got SM {get_sm_version()}",
)


@pytest.fixture(autouse=True)
def _tiers_off(monkeypatch):
    """This module tests the IN-TREE kernel contract. Its op-level cases
    (fp32 shapes within the tier envelope) would otherwise be routed to the
    GVR tiers by the dispatcher — which also ignores the explicit
    ``cluster_size`` these tests parametrize over — so pin the op to the
    in-tree path. The GVR tiers are covered by
    ``test_cute_dsl_gvr_topk_tiers.py`` (which flips the routing the other
    way: fallback bands off so every test reaches a GVR tier)."""
    monkeypatch.setenv("TRTLLM_GVR_TIERS_DISABLE", "1")
    _tier_dispatch._reset_env_cache()
    yield
    monkeypatch.delenv("TRTLLM_GVR_TIERS_DISABLE", raising=False)
    _tier_dispatch._reset_env_cache()


def _make_inputs_impl(
    num_rows: int,
    N: int,
    top_k: int,
    dtype: torch.dtype,
    next_n: int,
    seed: int,
    compress_ratio: int = 1,
    preidx_hit_rate: float = 0.0,
    varlen: bool = False,
    seq_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, pre_idx, seq_lens) for the op.

    ``logits`` lives in compressed-token-index space (``N = N_uncompressed /
    compress_ratio``). ``seq_lens`` is in UNCOMPRESSED space — kernel divides
    by ``compress_ratio`` internally. ``pre_idx[..., 0]`` is the per-group
    argmax (indexer invariant).

    ``preidx_hit_rate`` controls how many ``pre_idx[..., 1:]`` slots are
    real ``torch.topk`` indices vs random fillers. 0.0 = current worst-case
    (only slot 0 meaningful, rest = junk arange); 0.3-0.8 = realistic
    production (V3.2 ~40%, V4 Pro ~75%) where the kernel's Guess phase
    short-circuits. Always preserves the ``pre_idx[..., 0] = argmax``
    invariant on slot 0.

    ``varlen=False``: ``seq_lens = N * cr`` uniformly across groups.
    ``varlen=True``: per-group seq_lens drawn uniformly in
    ``[top_k*cr + next_n, N*cr]`` so the kernel's per-row N_eff varies.
    Argmax / ref_topk are computed over the smallest group's N_eff so
    they're guaranteed in-range for every row.

    ``seq_lens``: optional pre-built seq_lens tensor (uncompressed space).
    When provided, overrides ``varlen`` and the internal seq_lens generation.
    Argmax is still computed over ``min(seq_lens)`` so pre_idx[..., 0]
    is in-range for every row.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    logits = logits_f32.to(dtype)

    num_groups = num_rows // next_n

    # seq_lens in UNCOMPRESSED space. Kernel divides by cr internally.
    if seq_lens is not None:
        pass  # use caller-provided seq_lens as-is
    elif varlen:
        lo = top_k * compress_ratio + next_n  # ensures N_eff >= top_k
        seq_lens = torch.randint(
            lo, N * compress_ratio + 1, (num_groups,), dtype=torch.int32, device="cuda"
        )
    else:
        seq_lens_val = N * compress_ratio
        seq_lens = torch.full((num_groups,), seq_lens_val, dtype=torch.int32, device="cuda")

    # Smallest per-row N_eff across all groups — safe upper bound for the
    # argmax/topk scan range (every row's N_eff >= this value).
    min_seq_lens = int(seq_lens.min().item())
    effective_len = (min_seq_lens - next_n + 1) // compress_ratio

    group_logits = logits[::next_n, :effective_len]
    argmax_idx = group_logits.argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx

    if preidx_hit_rate <= 0.0:
        # Worst-case: only slot 0 is meaningful, rest are junk arange.
        pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    else:
        # Realistic: mix ``preidx_hit_rate`` real torch.topk indices with
        # random in-range fillers. Tests the Guess-phase short-circuit
        # path (production V3.2 ~40%, V4 Pro ~75%).
        ref_topk = group_logits.topk(top_k, dim=-1).indices.int()
        keep_mask = torch.rand(ref_topk.shape, device="cuda") < preidx_hit_rate
        random_fill = torch.randint(
            0, effective_len, ref_topk.shape, device="cuda", dtype=torch.int32
        )
        guess = torch.where(keep_mask, ref_topk, random_fill)
        guess[:, 0] = argmax_idx
        pre_idx[:, :] = guess

    return logits, pre_idx, seq_lens


# Module-level input memoization. ``_make_inputs_impl`` is fully deterministic
# (seed-keyed RNG), and the parametrized sweeps below request the same
# (shape, dtype, hit-rate, ...) combination once per cluster_size / dispatch
# variant — regenerating logits + the reference topk dominated suite
# wall-clock, not the kernel under test. Cached tensors are returned WITHOUT
# cloning under a strict read-only convention: the op writes only
# ``out_indices`` (allocated fresh by every test), never its inputs.
_inputs_cache: dict = {}
# Reference top-K values memoized per cached-inputs identity (see
# ``_tie_aware_check``). Keyed on object ids, which is safe only because the
# keying tensors are pinned for the process lifetime by ``_inputs_cache``.
_ref_vals_cache: dict = {}


def _gvr_check(check, out_indices, logits, seq_lens, top_k, next_n, compress_ratio=1):
    """Delegate to the shared conftest checker (``tie_aware_check`` fixture),
    attaching this module's reference-values memo only when (logits,
    seq_lens) are pinned for the process lifetime by ``_inputs_cache``
    (id-keyed caching on transient tensors would alias after GC)."""
    ref = (
        _ref_vals_cache
        if any(logits is v[0] and seq_lens is v[2] for v in _inputs_cache.values())
        else None
    )
    check(out_indices, logits, seq_lens, top_k, next_n, compress_ratio, ref_vals_cache=ref)


def _make_inputs(
    num_rows: int,
    N: int,
    top_k: int,
    dtype: torch.dtype,
    next_n: int,
    seed: int,
    compress_ratio: int = 1,
    preidx_hit_rate: float = 0.0,
    varlen: bool = False,
    seq_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoizing wrapper around ``_make_inputs_impl`` (same signature).

    Caller-provided ``seq_lens`` bypasses the cache (the tensor identity is
    not part of a hashable key).
    """
    if seq_lens is not None:
        return _make_inputs_impl(
            num_rows,
            N,
            top_k,
            dtype,
            next_n,
            seed,
            compress_ratio=compress_ratio,
            preidx_hit_rate=preidx_hit_rate,
            varlen=varlen,
            seq_lens=seq_lens,
        )
    key = (num_rows, N, top_k, dtype, next_n, seed, compress_ratio, preidx_hit_rate, varlen)
    if key not in _inputs_cache:
        _inputs_cache[key] = _make_inputs_impl(
            num_rows,
            N,
            top_k,
            dtype,
            next_n,
            seed,
            compress_ratio=compress_ratio,
            preidx_hit_rate=preidx_hit_rate,
            varlen=varlen,
        )
    return _inputs_cache[key]


# ---------------------------------------------------------------------------
# Compile-cost-aware covering design for the op-level sweep.
#
# The cuTe DSL variant (== JIT compile) key is
#   (dtype, top_k, next_n, compress_ratio, cluster_size,
#    pick_tuning(dtype, num_rows, N // cluster_size))
# so ``dtype/K``, ``next_n``, ``cr``, ``cluster_size`` and ``N`` each multiply
# the number of codegen'd kernels, while ``varlen``, ``batch_size`` and
# ``preidx_hit_rate`` are RUNTIME-only axes that reuse an already-compiled
# variant. The old full 8-way cross was 512 cases / 64 compiles and cuTe DSL
# codegen (10-35 s per first-seen variant) dominated the file's CI wall clock.
#
# Compile cells below keep, at 16 compiles instead of 64:
#   * (dtype, K) x (next_n, cr) FULLY crossed -- the per-row scan range
#     ``N_eff = (seq_len - next_n + nn + 1) // cr`` is the one genuine 2-way
#     interaction in the index math, and it is dtype/K-independent only in
#     theory (K sets the SMEM layout the row indices are written through);
#   * (N, cluster_size) laid out as a Latin square over those 16 cells, so
#     (dtype, K) x (N, cs) and (next_n, cr) x (N, cs) are ALSO fully crossed
#     (16/16 pairs each). N is what selects the T=512/1024 + 256-bit-load
#     tuning bucket; cs selects single-CTA vs DSMEM cluster.
# Only 3-way and higher interactions are dropped.
_DECODE_COMPILE_CELLS = [
    # (dtype, top_k, next_n, compress_ratio, N, cluster_size)
    (torch.bfloat16, 512, 1, 1, 4096, 1),
    (torch.bfloat16, 512, 1, 4, 4096, 4),
    (torch.bfloat16, 512, 2, 1, 65536, 1),
    (torch.bfloat16, 512, 2, 4, 65536, 4),
    (torch.bfloat16, 1024, 1, 1, 4096, 4),
    (torch.bfloat16, 1024, 1, 4, 65536, 1),
    (torch.bfloat16, 1024, 2, 1, 65536, 4),
    (torch.bfloat16, 1024, 2, 4, 4096, 1),
    (torch.float16, 1024, 1, 1, 65536, 1),
    (torch.float16, 1024, 1, 4, 65536, 4),
    (torch.float16, 1024, 2, 1, 4096, 1),
    (torch.float16, 1024, 2, 4, 4096, 4),
    (torch.float32, 2048, 1, 1, 65536, 4),
    (torch.float32, 2048, 1, 4, 4096, 1),
    (torch.float32, 2048, 2, 1, 4096, 4),
    (torch.float32, 2048, 2, 4, 65536, 1),
]
# Runtime-only axes, crossed against EVERY compile cell at zero codegen cost.
# (varlen, preidx_hit_rate) is fully crossed; batch_size covers the
# single-row grid and the batched grid. varlen needs batch_size >= 2.
_DECODE_RUNTIME_CELLS = [
    (False, 1, 0.0),  # single row, worst-case hint (only argmax is real)
    (False, 32, 0.5),  # batched, production-like hint overlap
    (True, 32, 0.0),  # ragged rows, worst-case hint
    (True, 32, 0.5),  # ragged rows, production-like hint
]


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k,next_n,compress_ratio,N,cluster_size",
    _DECODE_COMPILE_CELLS,
)
@pytest.mark.parametrize("varlen,batch_size,preidx_hit_rate", _DECODE_RUNTIME_CELLS)
def test_cute_dsl_gvr_topk_decode(
    dtype,
    top_k,
    N,
    varlen,
    next_n,
    batch_size,
    compress_ratio,
    preidx_hit_rate,
    cluster_size,
    tie_aware_check,
):
    """Compare custom op output against torch.topk reference (tie-aware).

    ``preidx_hit_rate=0.0`` exercises the worst-case (only argmax slot is
    a real topK index); ``0.5`` matches realistic production preIdx
    overlap with topK (V3.2 ~40%, V4 Pro ~75%) and exercises the
    kernel's Guess-phase short-circuit path.

    ``varlen=False`` uses uniform seq_lens=N*cr across the batch;
    ``varlen=True`` draws per-row seq_lens uniformly in [N/2, N]*cr.

    The LJF host-side dispatch order (``order_row``) is covered by the
    dedicated ``test_cute_dsl_gvr_topk_decode_seqlen_sorted`` below on
    representative cells instead of doubling this whole sweep.
    """
    if N - next_n + 1 < top_k:
        pytest.skip(f"N_eff < top_k ({N - next_n + 1} < {top_k}) is a degenerate path")
    if varlen and batch_size < 2:
        pytest.skip("varlen with batch_size<2 collapses to fixed")

    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        next_n,
        seed=42,
        compress_ratio=compress_ratio,
        preidx_hit_rate=preidx_hit_rate,
        varlen=varlen,
    )

    out_indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        cluster_size=cluster_size,
    )
    torch.cuda.synchronize()

    _gvr_check(
        tie_aware_check, out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio
    )


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size,varlen,next_n,compress_ratio,cluster_size",
    [
        # Representative cells for the LJF host-side dispatch order
        # (previously a full extra dimension on the sweep above): varlen
        # batches so the argsort is a real permutation, both SMEM layout
        # endpoints (bf16/K=512, fp32/K=2048), next_n=2 for the
        # ``order_row[req] * next_n + nn`` row expansion, cr=4 for the
        # order_row + compressed seq_lens interaction, cluster and
        # single-CTA paths, and a batch_size=1 trivial-permutation smoke.
        (torch.bfloat16, 512, 65536, 32, True, 1, 1, 4),
        (torch.float32, 2048, 65536, 32, True, 2, 1, 1),
        (torch.bfloat16, 1024, 4096, 32, True, 2, 4, 1),
        (torch.float16, 1024, 65536, 1, False, 1, 1, 1),
    ],
)
def test_cute_dsl_gvr_topk_decode_seqlen_sorted(
    dtype, top_k, N, batch_size, varlen, next_n, compress_ratio, cluster_size, tie_aware_check
):
    """LJF host-side dispatch order: ``order_row`` = descending argsort of
    ``seq_lens`` passed through the custom op.

    The kernel must produce the same per-row top-K as the unsorted launch
    (rows are still written back at their original positions, since the
    kernel uses ``row_idx = order_row[req] * next_n + nn`` for both reads
    and writes). The reference comparison is unchanged — it asserts that
    each row's output is a valid top-K of that row's masked logits.
    """
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        next_n,
        seed=42,
        compress_ratio=compress_ratio,
        preidx_hit_rate=0.5,
        varlen=varlen,
    )
    out_indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    # LJF dispatch order — request-level descending argsort of seq_lens.
    order_row = torch.argsort(seq_lens, descending=True, stable=False).to(torch.int32)

    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        cluster_size=cluster_size,
        order_row=order_row,
    )
    torch.cuda.synchronize()

    _gvr_check(
        tie_aware_check, out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio
    )


# ===========================================================================
# Degenerate-hint bracket-rewrite race regression tests.
#
# Phase 1 gathers the row values at ``pre_idx``; when every gathered value
# lands on one value (all-zeros pre_idx — the documented production cold
# start — or a hint whose gathered values fall in one tie class), the
# kernel rewrites the Phase-1 bracket to a synthetic one under a guarded
# barrier. Without a barrier between the bracket reads and that rewrite, a
# straggler warp can read the already-rewritten bracket, evaluate the
# degenerate condition differently, and skip the guarded barrier — barrier
# divergence: wrong index sets, unwritten output slots, out-of-range
# indices, intermittent illegal memory access. The same read-vs-rewrite
# pattern (and fix) applies to the leader-published ``done`` flag in the
# Phase-2 refine/collapse loops and the Phase-4 bin-search publish. The
# corruption is a timing race, so the tests launch repeatedly and poison
# the output buffer before every launch (an unwritten slot then fails the
# checker's range assertion); detection is probabilistic pre-fix, while
# the fixed kernel must pass every launch. The junk-arange hints used
# elsewhere in this module gather the argmax column plus ``top_k - 1``
# distinct arange columns, whose randn values essentially never all
# collide, so the existing sweeps do not arm the degenerate rewrite —
# which is why they could not catch this.
# ===========================================================================


@skip_not_sm100
def test_cute_dsl_gvr_topk_decode_all_zeros_pre_idx(tie_aware_check):
    """All-zeros ``pre_idx`` (production cold start) must not corrupt output.

    With ``pre_idx = 0`` every row gathers ``top_k`` copies of ``row[1]``
    (``pre_idx_offset = 1`` at cr=1), so every CTA takes the degenerate-hint
    bracket rewrite — the race site. Pre-fix this shape returned wrong sets,
    unwritten output slots, and out-of-range indices on most launches.
    """
    top_k = 1024
    num_rows, N = 4096, 4096
    torch.manual_seed(7)
    logits = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    pre_idx = torch.zeros(num_rows, top_k, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((num_rows,), N, dtype=torch.int32, device="cuda")
    ref_cache: dict = {}

    for _ in range(40):
        # Poison: any output slot the kernel fails to write stays negative
        # and trips the checker's range assertion.
        out_indices = torch.full((num_rows, top_k), -777777, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            logits,
            pre_idx,
            seq_lens,
            out_indices,
            top_k=top_k,
            next_n=1,
            compress_ratio=1,
            cluster_size=1,
        )
        torch.cuda.synchronize()
        tie_aware_check(
            out_indices,
            logits,
            seq_lens,
            top_k,
            1,
            compress_ratio=1,
            ref_vals_cache=ref_cache,
        )


@skip_not_sm100
@pytest.mark.parametrize("scenario", ["quantized", "plateau_wider_than_kc"])
def test_cute_dsl_gvr_topk_decode_tie_degenerate_hint(scenario, tie_aware_check):
    """Tie-heavy rows + argmax-only hint over zeros.

    ``pre_idx`` carries only the argmax invariant (slot 0) over zeros — a
    synthetic low-information hint (production keeps the full previous
    top-k after the first step, and starts fully zeroed before it); on
    tie-heavy rows the two gathered columns
    collide bitwise (~32/1024 rows in the quantized scenario; every row in
    the plateau scenario), arming the degenerate-hint bracket rewrite even
    though a real hint is present.

    ``quantized``: logits quantized to 0.25 steps (~60 distinct values),
    the shape class that crashed pre-fix with an intermittent illegal
    memory access. ``plateau_wider_than_kc``: two-valued rows whose 0.0 tie
    class is far wider than the candidate buffer ``kC``, pinning the
    Phase-2 plateau-collapse terminal + Phase-4 plateau fill (no threshold
    count can land in [K, kC]). Both scenarios share one compile variant.
    """
    top_k = 512
    num_rows, N = 1024, 16384
    if scenario == "quantized":
        torch.manual_seed(11)
        logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 8.0).round() / 4.0
    else:
        n_win = 100
        logits = torch.zeros(num_rows, N, dtype=torch.float32, device="cuda")
        # Winners beyond column top_k+1 at stride >= 2: every hint-gathered
        # column (offset +1) stays on the 0.0 plateau.
        stride = (N - top_k - 2) // n_win
        win_cols = top_k + 2 + stride * torch.arange(n_win, device="cuda")
        logits[:, win_cols] = 1.0
    pre_idx = torch.zeros(num_rows, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = logits.argmax(dim=-1).int()
    seq_lens = torch.full((num_rows,), N, dtype=torch.int32, device="cuda")
    ref_cache: dict = {}

    for _ in range(60):
        out_indices = torch.full((num_rows, top_k), -777777, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            logits,
            pre_idx,
            seq_lens,
            out_indices,
            top_k=top_k,
            next_n=1,
            compress_ratio=1,
            cluster_size=1,
        )
        torch.cuda.synchronize()
        tie_aware_check(
            out_indices,
            logits,
            seq_lens,
            top_k,
            1,
            compress_ratio=1,
            ref_vals_cache=ref_cache,
        )


# ===========================================================================
# GVR top-K multi-CTA short-row degrade boundary tests.
#
# For cluster_size > 1 each row is owned by a cluster of CTAs.  When the
# actual row length N_eff fits within a single CTA's static slice
# (N_eff <= ceil(buffer_N / cluster_size)), the cluster degrades: CTA 0
# scans the row solo (no cluster sync) and the other CTAs exit early.
# When N_eff > max_slice_len all CTAs cooperate via DSMEM.
# ===========================================================================


@skip_not_sm100
@pytest.mark.parametrize("cluster_size", [2, 4])
@pytest.mark.parametrize(
    "dtype,top_k",
    [(torch.bfloat16, 512), (torch.float32, 2048)],
)
def test_cute_dsl_gvr_topk_multi_cta_shortrow_degrade_boundary(
    dtype, top_k, cluster_size, tie_aware_check
):
    """GVR top-K multi-CTA short-row degrade: correctness at the cluster transition boundary.

    When ``cluster_size > 1``, each row is dispatched to a cluster of
    ``cluster_size`` CTAs.  Whether the cluster cooperates depends on the
    actual row length N_eff relative to ``max_slice_len = ceil(buffer_N /
    cluster_size)`` — the per-CTA design slice width:

    * N_eff <= max_slice_len (short row): all tokens fit in CTA 0's static
      slice; CTA 0 scans solo (``do_cluster_sync=False``), the other
      ``cluster_size - 1`` CTAs exit early.  This avoids wasted mbarrier
      overhead when the cluster would add no parallelism.

    * N_eff > max_slice_len (long row): tokens span multiple CTAs; all
      ``cluster_size`` CTAs cooperate via DSMEM (``do_cluster_sync=True``).

    This test pins N_eff at max_slice_len − 1 / max_slice_len /
    max_slice_len + 1 to verify correctness exactly at the boundary, and
    adds a mixed batch with alternating short and long rows.
    """
    next_n = 1
    compress_ratio = 1
    # Choose N so that per_cta_design (the kernel's ceil(N/cs)) is:
    #   (a) a multiple of vec_size (128-bit load width) — CTA k starts at
    #       k*per_cta_design, which must be vec_size-aligned to avoid
    #       cudaErrorMisalignedAddress on global vector loads.
    #   (b) >= 2*top_k — GVR histogram uses 256 bins over the value range;
    #       when N_eff barely exceeds top_k (ratio ~1) the per-bin count is
    #       too coarse for the threshold bucket to converge, causing -1 outputs.
    #       A ratio of 2 (per_cta_design = 2*top_k) matches the smallest
    #       N tested in test_cute_dsl_gvr_topk_decode (N=4096, K=2048).
    vec_size = 16 // dtype.itemsize  # 128-bit = 16 bytes; bf16→8, fp32→4
    per_cta_design = ((top_k * 2) + vec_size - 1) // vec_size * vec_size
    N = per_cta_design * cluster_size
    max_slice_len = (N + cluster_size - 1) // cluster_size  # == per_cta_design

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    for n_eff, case_name in [
        (max_slice_len - 1, "degrade_below"),  # CTA 0 solo, N_eff < max_slice_len
        (max_slice_len, "degrade_exact"),  # CTA 0 solo, fills slice exactly
        (max_slice_len + 1, "coop_one_extra"),  # CTA 1 gets 1 element
    ]:
        batch_size = 8
        logits = torch.randn(batch_size, N, dtype=dtype, device="cuda") * 2.0
        seq_lens = torch.full((batch_size,), n_eff, dtype=torch.int32, device="cuda")
        # Junk-arange pre_idx (same pattern as _make_inputs with preidx_hit_rate=0):
        # slot 0 = per-row argmax; slots 1..K-1 = arange 1..K-1.  This avoids
        # duplicate-0 degenerate pre_idx that forces the kernel to find all K
        # indices from refinement alone.
        pre_idx = (
            torch.arange(top_k, dtype=torch.int32, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        pre_idx[:, 0] = logits[:, :n_eff].argmax(dim=-1).int()

        out_indices = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            logits,
            pre_idx,
            seq_lens,
            out_indices,
            top_k=top_k,
            next_n=next_n,
            compress_ratio=compress_ratio,
            cluster_size=cluster_size,
        )
        torch.cuda.synchronize()
        _gvr_check(
            tie_aware_check,
            out_indices,
            logits,
            seq_lens,
            top_k,
            next_n,
            compress_ratio=compress_ratio,
        )

    # Mixed batch: alternating degrade (even rows) and co-op (odd rows).
    batch_size = 8
    logits = torch.randn(batch_size, N, dtype=dtype, device="cuda") * 2.0
    n_eff_short = max_slice_len - 1  # degrade
    n_eff_long = max_slice_len + 1  # co-op
    seq_lens = torch.tensor(
        [n_eff_short if i % 2 == 0 else n_eff_long for i in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    pre_idx = (
        torch.arange(top_k, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    for i in range(batch_size):
        n_eff_i = int(seq_lens[i].item())
        pre_idx[i, 0] = int(logits[i, :n_eff_i].argmax().item())

    out_indices = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        cluster_size=cluster_size,
    )
    torch.cuda.synchronize()
    _gvr_check(
        tie_aware_check, out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio
    )


# ===========================================================================
# Load-Balance (hybrid multi-CTA + single-CTA) tests.
#
# The LB kernel adds a prepare step that classifies requests as long
# (seq_len > long_threshold) vs short and dispatches each cluster
# of 4 CTAs into either:
#   - long branch: 4 CTAs cooperatively process 1 long row (cs=4 path)
#   - short branch: 4 CTAs each process 1 short row independently (cs=1)
#
# Tests at three layers:
#   1. ``test_lb_prepare_partition`` — drive prepare alone, validate
#      counters + order_row against a numpy reference.
#   2. ``test_lb_main_branches`` — force each branch (all_long / all_short
#      / mixed) and verify the produced indices are correct.
#   3. ``test_lb_vs_reference`` — sweep matching the GVR cs=1 UT params;
#      compare against the same tie-aware torch.topk reference.
# ===========================================================================


@skip_not_sm100
@pytest.mark.parametrize("B", [1, 8, 32, 128, 256, 1024])
@pytest.mark.parametrize(
    "ratio",
    [0.0, 0.25, 0.5, 0.75, 1.0],
    ids=["all_short", "1/4_long", "half_long", "3/4_long", "all_long"],
)
def test_lb_prepare_partition(B, ratio):
    """Prepare kernel: counters + order_row match a numpy partition reference.

    Builds synthetic seq_lens with a controlled long/short ratio, shuffles
    them, then verifies the kernel partitions the request_ids into
    [long...][short...] correctly.
    """
    long_threshold = 64 * 1024
    torch.manual_seed(B * 1000 + int(ratio * 100))
    n_long_expect = round(B * ratio)
    seq_lens = torch.empty(B, dtype=torch.int32, device="cuda")
    seq_lens[:n_long_expect] = long_threshold * 2
    seq_lens[n_long_expect:] = long_threshold // 2
    perm = torch.randperm(B, device="cuda")
    seq_lens = seq_lens[perm]

    is_long = (seq_lens > long_threshold).cpu().numpy()
    ref_n_long = int(is_long.sum())
    ref_n_short = B - ref_n_long

    max_batch_size = 1024
    order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_lb_prepare(
        seq_lens, order_row, counters, max_batch_size, long_threshold
    )
    n_long = int(counters[0].item())
    n_short = int(counters[1].item())
    assert n_long == ref_n_long, f"n_long mismatch: {n_long} vs {ref_n_long}"
    assert n_short == ref_n_short, f"n_short mismatch: {n_short} vs {ref_n_short}"

    out_ids = order_row[: n_long + n_short].cpu().numpy()
    long_part = set(int(x) for x in out_ids[:n_long])
    short_part = set(int(x) for x in out_ids[n_long:])
    ref_long_set = set(int(i) for i in range(B) if is_long[i])
    ref_short_set = set(int(i) for i in range(B) if not is_long[i])
    assert long_part == ref_long_set, (
        f"long set mismatch: missing={ref_long_set - long_part}, extra={long_part - ref_long_set}"
    )
    assert short_part == ref_short_set, (
        f"short set mismatch: missing={ref_short_set - short_part}, "
        f"extra={short_part - ref_short_set}"
    )


# N picked so all rows fall clearly below / above the 64K long_threshold.
# fp32/K=2048 carries the (scenario x next_n) cross; bf16/K=1024 pins the
# other SMEM layout on one case per branch. batch_size is runtime-only.
# 10 cases / 6 compiles, down from 24 cases / 8 compiles.
_LB_BRANCH_CELLS = [
    # (dtype, top_k, scenario, N, seq_lens_mode, batch_size, next_n)
    (torch.float32, 2048, "all_short", 8 * 1024, "uniform", 4, 1),
    (torch.float32, 2048, "all_short", 8 * 1024, "uniform", 32, 2),
    (torch.float32, 2048, "all_long", 128 * 1024, "uniform", 4, 1),
    (torch.float32, 2048, "all_long", 128 * 1024, "uniform", 32, 2),
    (torch.float32, 2048, "mixed_half", 128 * 1024, "half_short_half_long", 4, 1),
    (torch.float32, 2048, "mixed_half", 128 * 1024, "half_short_half_long", 32, 2),
    # One bf16 cell pins the other SMEM layout; mixed_half covers the long
    # branch AND both branches in one launch.
    (torch.bfloat16, 1024, "mixed_half", 128 * 1024, "half_short_half_long", 32, 1),
]


@skip_not_sm100
@pytest.mark.parametrize("dtype,top_k,scenario,N,seq_lens_mode,batch_size,next_n", _LB_BRANCH_CELLS)
def test_lb_main_branches(
    dtype, top_k, scenario, N, seq_lens_mode, batch_size, next_n, tie_aware_check
):
    """Each LB branch (all_long / all_short / mixed) produces correct top-K.

    For ``mixed_half`` half the rows are forced to be short (seq_len < threshold)
    and half long, exercising both branches inside the same launch.

    ``next_n>1`` exercises the request-level → row-level expansion
    (``order_row[req] * next_n + nn``) in both branches: long branch's
    cluster CTAs all read the same request and slice it, while short
    branch's CTAs each handle a different (req, nn) row pair. A
    mis-indexed expansion would show up as out-of-range writes caught
    by ``_tie_aware_check``.
    """
    num_rows = batch_size * next_n
    num_groups = batch_size  # batch_size groups of next_n rows each
    # For half_short_half_long, build seq_lens first so _make_inputs computes
    # argmax over min(seq_lens)=8K, keeping pre_idx[..., 0] in-range for all rows.
    if seq_lens_mode == "half_short_half_long":
        seq_lens_override = torch.empty(num_groups, dtype=torch.int32, device="cuda")
        seq_lens_override[: batch_size // 2] = 8 * 1024  # short half
        seq_lens_override[batch_size // 2 :] = 128 * 1024  # long half
    else:
        seq_lens_override = None
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        next_n,
        seed=42,
        compress_ratio=1,
        preidx_hit_rate=0.5,
        varlen=False,
        seq_lens=seq_lens_override,
    )

    max_batch_size = 1024
    long_threshold = 64 * 1024
    order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_lb_prepare(
        seq_lens, order_row, counters, max_batch_size, long_threshold
    )
    out_indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        order_row=order_row,
        counters=counters,
        max_batch_size=max_batch_size,
    )
    torch.cuda.synchronize()

    n_long = int(counters[0].item())
    if scenario == "all_long":
        assert n_long == batch_size, f"expected {batch_size} long, got {n_long}"
    elif scenario == "all_short":
        assert n_long == 0, f"expected 0 long, got {n_long}"
    elif scenario == "mixed_half":
        assert n_long == batch_size - batch_size // 2

    _gvr_check(tie_aware_check, out_indices, logits, seq_lens, top_k, next_n, compress_ratio=1)


# LB dispatch (prepare partition + long/short branch selection) is
# dtype-insensitive by construction, so the fp32/K=2048 arm carries the
# (next_n x cr) cross and bf16/K=512 only pins the other SMEM layout.
# N below/above the 64K long_threshold selects the branch AND the
# 256-bit-load/mbpm tuning bucket, so both stay represented.
# The LB kernel is a DORMANT capability -- ``indexer.py`` passes ``order_row``
# but never ``counters``/``max_batch_size``, so no production shape reaches
# this path -- which is why it is covered at smoke depth (7 compiles / 14
# cases) instead of mirroring the in-tree kernel's production dtype x K map
# (was 16 compiles / 128 cases = 15% of this file's CI wall clock).
_LB_COMPILE_CELLS = [
    # (dtype, top_k, N, next_n, compress_ratio)
    # (next_n x cr) fully crossed on the short branch, and re-crossed on the
    # long branch at the two diagonal corners so the >64K tuning bucket sees
    # both the trivial (1, 1) and the fully-shifted (2, 4) index math.
    (torch.float32, 2048, 8 * 1024, 1, 1),
    (torch.float32, 2048, 8 * 1024, 1, 4),
    (torch.float32, 2048, 8 * 1024, 2, 1),
    (torch.float32, 2048, 8 * 1024, 2, 4),
    (torch.float32, 2048, 128 * 1024, 1, 1),
    (torch.float32, 2048, 128 * 1024, 2, 4),
    (torch.bfloat16, 512, 128 * 1024, 1, 1),
]
# Runtime-only for LB as well (varlen / batch_size / hint overlap never enter
# the compile key), so the (varlen x preidx_hit_rate) cross is kept intact at
# every LB compile cell: only compile-axis COMBINATIONS were dropped above, no
# runtime behaviour lost.
_LB_RUNTIME_CELLS = [
    (False, 4, 0.0),  # uniform seq_lens, worst-case hint
    (False, 32, 0.5),  # uniform, production-like hint overlap
    (True, 32, 0.0),  # ragged rows, worst-case hint
    (True, 32, 0.5),  # ragged rows, production-like hint
]


@skip_not_sm100
@pytest.mark.parametrize("dtype,top_k,N,next_n,compress_ratio", _LB_COMPILE_CELLS)
@pytest.mark.parametrize("varlen,batch_size,preidx_hit_rate", _LB_RUNTIME_CELLS)
def test_lb_vs_reference(
    dtype,
    top_k,
    N,
    varlen,
    next_n,
    batch_size,
    compress_ratio,
    preidx_hit_rate,
    tie_aware_check,
):
    """LB kernel output matches torch.topk tie-aware reference across the
    same param sweep used by the single-CTA UT."""
    if N - next_n + 1 < top_k:
        pytest.skip(f"N_eff < top_k ({N - next_n + 1} < {top_k}) is degenerate")
    if varlen and batch_size < 2:
        pytest.skip("varlen with batch_size<2 collapses to fixed")

    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        next_n,
        seed=42,
        compress_ratio=compress_ratio,
        preidx_hit_rate=preidx_hit_rate,
        varlen=varlen,
    )
    max_batch_size = 1024
    long_threshold = 64 * 1024
    order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_lb_prepare(
        seq_lens,
        order_row,
        counters,
        max_batch_size,
        long_threshold,
        compress_ratio,
    )
    out_indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        order_row=order_row,
        counters=counters,
        max_batch_size=max_batch_size,
    )
    torch.cuda.synchronize()
    _gvr_check(
        tie_aware_check,
        out_indices,
        logits,
        seq_lens,
        top_k,
        next_n,
        compress_ratio=compress_ratio,
    )


# ===========================================================================
# R0 histogram-ladder admission equivalence tests.
#
# ``enable_r0=True`` (the GvrTopKKernel default) replaces the Phase-2 secant
# threshold search with a single-pass multi-threshold "rung ladder" admission
# seeded by a 256-bin histogram over the prev-topK gathered values. This must
# select the SAME top-K as the retained secant baseline (``enable_r0=False``).
#
# top-K is order-independent, so correctness is checked by INDEX SET (not
# position): for continuous fp32 logits (tie-free with probability 1) the R0
# and base index sets must be identical; for bf16/fp16 boundary value-ties can
# make two equally-valid selections differ in index, so there the guarantee is
# value-set (multiset) equality against the tie-aware torch.topk reference.
#
# The custom op does not plumb ``enable_r0`` (activation / dispatch land in a
# follow-up PR), so these tests drive ``GvrTopKKernel`` directly. This is also
# the only remaining coverage of the secant fallback path, since every op-level
# test above now inherits the ``enable_r0=True`` default.
# ===========================================================================

_R0_DT = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}
# Compiled-kernel cache keyed on (enable_r0, dtype, top_k, cluster_size, T,
# min_blocks_per_mp). Shapes (num_rows / N / batch) are symbolic, so one
# compile covers every N and batch_size within a bucket (mirrors the runner).
_r0_kernel_cache: dict = {}


def _compile_gvr_direct(kernel):
    """Compile a ``GvrTopKKernel`` with symbolic shapes, mirroring the
    production runner's fake-tensor construction (128-bit loads, no
    ``order_row`` / ``output_values``)."""
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_f = _crt.make_fake_compact_tensor(
        kernel.dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=16
    )
    pi_f = _crt.make_fake_compact_tensor(
        cutlass.Int32, (n_batch, kernel.top_k), stride_order=(1, 0), assumed_align=16
    )
    sl_f = _crt.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
    oi_f = _crt.make_fake_compact_tensor(
        cutlass.Int32, (n_rows, kernel.top_k), stride_order=(1, 0), assumed_align=16
    )
    fs = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    # __call__(input, pre_idx, seq_lens, output_values, output_indices, order_row, stream)
    return cute.compile(
        kernel, in_f, pi_f, sl_f, None, oi_f, None, stream=fs, options="--enable-tvm-ffi"
    )


def _run_gvr_direct(logits, pre_idx, seq_lens, top_k, enable_r0, cluster_size):
    """Drive ``GvrTopKKernel`` directly (bypassing the custom op, which does
    not expose ``enable_r0``). Fixed at ``next_n=1``, ``compress_ratio=1``,
    128-bit loads. When ``enable_r0=True`` the ctor auto-derives the shipped
    R0 config (r0_qfracs=M2D, cs-aware p1b_cache, K512 kC-diet, P4
    rank-scatter) — i.e. the exact default arm. Returns int32
    ``[num_rows, top_k]`` indices."""
    num_rows, N = logits.shape
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_threads = 1024 if (num_rows <= num_sms and N >= 65536) else 512
    min_blocks_per_mp = 1 if num_rows <= num_sms else 3
    key = (enable_r0, logits.dtype, top_k, cluster_size, num_threads, min_blocks_per_mp)
    if key not in _r0_kernel_cache:
        kernel = _GvrTopKKernel(
            dtype=_R0_DT[logits.dtype],
            top_k=top_k,
            next_n=1,
            num_threads=num_threads,
            compress_ratio=1,
            use_256bit_load=False,
            min_blocks_per_mp=min_blocks_per_mp,
            cluster_size=cluster_size,
            return_output_values=False,
            enable_r0=enable_r0,
        )
        _r0_kernel_cache[key] = _compile_gvr_direct(kernel)
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    _r0_kernel_cache[key](logits, pre_idx, seq_lens, None, out, None)
    torch.cuda.synchronize()
    return out


def _assert_index_sets_equal_tie_aware(out_base, out_r0, logits):
    """Assert two arms' top-K index sets match, modulo boundary value-ties.

    fp32 randn logits DO collide bit-exactly at these sample counts; when the
    duplicated value sits on the top-K boundary, each arm may legitimately
    keep a different member of the tie class. Indices in the symmetric
    difference must all carry the row's boundary (minimum kept) value —
    anything else is a genuine divergence.
    """
    base_sorted, _ = out_base.sort(dim=-1)
    r0_sorted, _ = out_r0.sort(dim=-1)
    mismatch = (base_sorted != r0_sorted).any(dim=-1)
    for bad in mismatch.nonzero().flatten().tolist():
        base_set = set(out_base[bad].tolist())
        r0_set = set(out_r0[bad].tolist())
        diff = sorted(base_set.symmetric_difference(r0_set))
        row_vals = logits[bad].float()
        kth = row_vals[out_base[bad].long()].min()
        diff_vals = row_vals[torch.tensor(diff, device=logits.device, dtype=torch.long)]
        if not bool((diff_vals == kth).all().item()):
            raise AssertionError(
                f"row={bad}: R0 index set != secant-base index set beyond a "
                f"boundary value-tie (kth={kth.item()}, "
                f"diff={[(i, row_vals[i].item()) for i in diff]}, "
                f"base={sorted(base_set)}, r0={sorted(r0_set)})"
            )


def _make_r0_pre_idx(logits, top_k, hint, seed):
    """Build ``pre_idx`` in the kernel's native cr=1 convention: the kernel
    reads ``logits[pre_idx + 1]``, so store ``true_index - 1``.

    ``hint='real'`` seeds a warm hint (near-topK indices) so R0's admission
    ladder hits on the first pass; ``hint='rand'`` seeds a cold hint (random
    in-range indices) that misses admission and forces the R0-miss inline
    log-falsi (R1) + fb_fix fallback."""
    num_rows, N = logits.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    if hint == "real":
        noised = logits.float() + 0.15 * torch.randn(num_rows, N, generator=g, device="cuda")
        pre = noised.topk(top_k, dim=1).indices.int()
    else:
        pre = torch.randint(0, N, (num_rows, top_k), generator=g, device="cuda").int()
    return (pre - 1).contiguous()


# Every case here compiles TWO kernels (R0 arm + secant arm), so this sweep
# is the second-most codegen-dense in the file. The compile key of
# ``_run_gvr_direct`` is (enable_r0, dtype, top_k, cluster_size, T, mbpm)
# with T selected by N (>= 65536 -> 1024) and mbpm pinned at 1 for BS <= SMs
# -- so ``hint`` and ``batch_size`` are runtime-only. The cells below keep
# (dtype, K) x cluster_size fully crossed and alternate N so both T buckets
# are exercised for every (dtype, K): 9 cells / 18 compiles, down from
# 4 x 2 x 3 = 24 - 4 skipped = 20 cells / 40 compiles.
_R0_EQ_CELLS = [
    # (dtype, top_k, N, cluster_size)
    (torch.bfloat16, 512, 8192, 1),
    (torch.bfloat16, 512, 65536, 4),
    (torch.bfloat16, 1024, 65536, 1),
    (torch.bfloat16, 1024, 8192, 4),
    (torch.float16, 1024, 8192, 1),
    (torch.float16, 1024, 65536, 4),
    (torch.float32, 2048, 65536, 1),
    (torch.float32, 2048, 8192, 4),
    # cs=8 is the runner's tiny-grid large-N pick (7-peer DSMEM aggregation);
    # large-N only, and dtype-insensitive -> one cell.
    (torch.float32, 2048, 65536, 8),
]


@skip_not_sm100
@pytest.mark.parametrize("dtype,top_k,N,cluster_size", _R0_EQ_CELLS)
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("hint", ["real", "rand"])
def test_cute_dsl_gvr_topk_decode_r0_equivalence(
    dtype, top_k, N, batch_size, hint, cluster_size, tie_aware_check
):
    """R0 admission (``enable_r0=True``, the new default) selects the same
    top-K as the secant baseline (``enable_r0=False``), by index set.

    ``hint='real'`` exercises the R0 admission-hit fast path; ``hint='rand'``
    forces the R0-miss log-falsi (R1) + fb_fix fallback. ``cluster_size=4``
    confirms R0 gates to single-CTA and the ``None`` R0 buffers propagate
    cleanly through the cluster path; ``cluster_size=8`` covers the runner's
    tiny-grid large-N pick (BS<=4, N>=128K -> cs=8): 7-peer DSMEM
    aggregation, large-N only (per-CTA slice too short below 64K).
    """
    if N < top_k * 2:
        pytest.skip(f"N ({N}) < 2*top_k ({2 * top_k}): GVR histogram bucket too coarse")
    if cluster_size == 8 and N < 65536:
        pytest.skip("cs=8 is a large-N production config (runner picks it only at N >= 131072)")

    num_rows = batch_size  # next_n = 1
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    logits = (torch.randn(num_rows, N, device="cuda") * 2.0).to(dtype).contiguous()
    seq_lens = torch.full((num_rows,), N, dtype=torch.int32, device="cuda")
    pre_idx = _make_r0_pre_idx(logits, top_k, hint, seed=1)

    out_base = _run_gvr_direct(
        logits, pre_idx, seq_lens, top_k, enable_r0=False, cluster_size=cluster_size
    )
    out_r0 = _run_gvr_direct(
        logits, pre_idx, seq_lens, top_k, enable_r0=True, cluster_size=cluster_size
    )

    # 1. Both arms independently produce a valid top-K (tie-aware value set).
    _gvr_check(tie_aware_check, out_base, logits, seq_lens, top_k, next_n=1, compress_ratio=1)
    _gvr_check(tie_aware_check, out_r0, logits, seq_lens, top_k, next_n=1, compress_ratio=1)

    # 2. Equivalence. fp32 logits are ALMOST tie-free, so R0 and base must
    #    return the identical index set (order-independent) — but randn
    #    quantized to fp32 does collide (bs=16, N=8192, seed 0:
    #    logits[3,2956] == logits[3,4949] bit-exactly, straddling the K=2048
    #    boundary), and a boundary value-tie makes the arms' distinct index
    #    picks equally valid. Where the sets differ, require every differing
    #    index to carry the row's boundary (k-th) value; anything else is a
    #    real divergence. bf16/fp16 boundary ties are common, so equivalence
    #    there is the value-set equality already established in step 1
    #    (both == torch.topk reference).
    if dtype == torch.float32:
        _assert_index_sets_equal_tie_aware(out_base, out_r0, logits)


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size,cluster_size",
    [
        # Multi-wave single-CTA grids (batch_size > num_sms): exercises the
        # occupancy regime where rows alone oversubscribe the device.
        (torch.bfloat16, 512, 16384, 256, 1),
        (torch.float32, 2048, 65536, 256, 1),
        # Multi-wave cluster grid (batch_size * cs > num_sms): DSMEM handoff
        # correctness across wave boundaries.
        (torch.bfloat16, 512, 65536, 64, 4),
    ],
)
@pytest.mark.parametrize("hint", ["real", "rand"])
def test_cute_dsl_gvr_topk_decode_r0_equivalence_bigbs(
    dtype, top_k, N, batch_size, hint, cluster_size, tie_aware_check
):
    """Big-batch R0-vs-secant equivalence: multi-wave grids only.

    The main equivalence grid tops out at batch_size=16 (single wave).
    These cells lock R0 + cluster correctness when the grid spans several
    waves — the throughput-bound regime of the BS-scaling study."""
    num_rows = batch_size
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    logits = (torch.randn(num_rows, N, device="cuda") * 2.0).to(dtype).contiguous()
    seq_lens = torch.full((num_rows,), N, dtype=torch.int32, device="cuda")
    pre_idx = _make_r0_pre_idx(logits, top_k, hint, seed=1)

    out_base = _run_gvr_direct(
        logits, pre_idx, seq_lens, top_k, enable_r0=False, cluster_size=cluster_size
    )
    out_r0 = _run_gvr_direct(
        logits, pre_idx, seq_lens, top_k, enable_r0=True, cluster_size=cluster_size
    )
    _gvr_check(tie_aware_check, out_base, logits, seq_lens, top_k, next_n=1, compress_ratio=1)
    _gvr_check(tie_aware_check, out_r0, logits, seq_lens, top_k, next_n=1, compress_ratio=1)
    if dtype == torch.float32:
        _assert_index_sets_equal_tie_aware(out_base, out_r0, logits)


@skip_not_sm100
@pytest.mark.parametrize(
    "top_k,N,cluster_size",
    [(512, 16384, 1), (1024, 131072, 4), (2048, 131072, 4)],
)
@pytest.mark.parametrize("band", ["sub_resolution", "one_ulp"])
def test_cute_dsl_gvr_topk_decode_p4_exact_tail_ties(top_k, N, cluster_size, band):
    """fp32 near-tie adversarial exactness (``p4_exact_tail``).

    The P4 rank-scatter fine recursion resolves candidate values to
    range/(kNumBins*256); distinct values spaced below that which straddle
    the top-K boundary land in ONE fine bin and were previously kept in
    arrival order (observed on real DSv4-Pro 512k-ISL captures as |miss|=1
    with dv ~ 3e-6). ``sub_resolution`` plants ~2.4k distinct values spaced
    5e-8 around the boundary; ``one_ulp`` plants a two-value bitwise plateau
    (``nextafter`` pairs). Both stay within the kC candidate budget (tie
    sets wider than kC are outside the kernel's contract). The default fp32
    kernel must return the exact top-K value set; natural random data never
    triggers this, so the adversarial construction is the only regression
    coverage."""
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    logits = (torch.randn(1, N, device="cuda") * 2.0).float().contiguous()
    boundary = torch.topk(logits[0], top_k).values[top_k - 1].item()
    n_tie = 2400 if band == "sub_resolution" else 2000
    plant = torch.randperm(N)[:n_tie]
    if band == "sub_resolution":
        tie_vals = (
            boundary + (torch.arange(n_tie, dtype=torch.float32, device="cuda") - n_tie // 2) * 5e-8
        )
    else:
        tie_vals = torch.full((n_tie,), boundary, device="cuda")
        tie_vals[::2] = torch.nextafter(tie_vals[::2], torch.tensor(float("inf"), device="cuda"))
    logits[0, plant] = tie_vals
    seq_lens = torch.full((1,), N, dtype=torch.int32, device="cuda")
    pre_idx = _make_r0_pre_idx(logits, top_k, "real", seed=4)

    out = _run_gvr_direct(
        logits, pre_idx, seq_lens, top_k, enable_r0=True, cluster_size=cluster_size
    )

    # Value-multiset exactness (the boundary index set is not unique under
    # bitwise plateaus, so indices are compared through their values).
    sel = logits[0][out[0].long()].sort().values
    ref = torch.topk(logits[0], top_k).values.sort().values
    torch.testing.assert_close(sel, ref, rtol=0.0, atol=0.0)


@skip_not_sm100
def test_cute_dsl_gvr_topk_decode_pick_config_policy():
    """``pick_config`` returns the runner-equivalent launch shapes.

    Locks the (BS, N) -> cluster_size map and the BS-aware occupancy knobs
    (the 2026-07-15 big-BS triage: a config frozen at the BS=1 optimum is
    geomean 2.27x slower than these picks at BS in {64, 256, 1024})."""
    sms = 148  # policy is expressed against a fixed SM count for determinism
    pc = _GvrTopKKernel.pick_config

    # cluster_size policy: N<64K -> 1; tiny grid large-N -> 8; single-wave
    # -> 4/2; multi-wave -> 1.
    assert pc(torch.float32, 1, 32768, num_sms=sms)["cluster_size"] == 1
    assert pc(torch.float32, 2, 131072, num_sms=sms)["cluster_size"] == 8
    assert pc(torch.float32, 16, 65536, num_sms=sms)["cluster_size"] == 4
    assert pc(torch.float32, 64, 65536, num_sms=sms)["cluster_size"] == 2
    assert pc(torch.float32, 256, 65536, num_sms=sms)["cluster_size"] == 1

    # Occupancy knobs at multi-wave BS: T=512 + mbpm>=2 (NOT the BS=1
    # frozen T=1024/mbpm=1 that loses 2.3-6x at big BS).
    big = pc(torch.float32, 1024, 65536, num_sms=sms)
    assert big["num_threads"] == 512 and big["min_blocks_per_mp"] == 2
    big16 = pc(torch.bfloat16, 1024, 65536, num_sms=sms)
    assert big16["num_threads"] == 512 and big16["min_blocks_per_mp"] == 3

    # Graph-capture contract: max_seq_len (peak N) overrides the capture N.
    cap = pc(torch.bfloat16, 1, 8192, max_seq_len=131072, num_sms=sms)
    assert cap["cluster_size"] == 8  # picked for the replay shape


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        (torch.float32, 2048, 32768, 1),  # cs=1 small-N
        (torch.bfloat16, 512, 65536, 16),  # cs=4 single-wave
        (torch.float32, 1024, 131072, 2),  # cs=8 tiny grid large-N
        (torch.bfloat16, 1024, 65536, 256),  # cs=1 multi-wave big-BS
    ],
)
def test_cute_dsl_gvr_topk_decode_launch_autoconfig(dtype, top_k, N, batch_size, tie_aware_check):
    """``GvrTopKKernel.launch`` (pick_config + variant cache) produces a
    valid top-K at every launch-shape regime the policy can pick, including
    cluster_size=8. Direct-drive users get production-equivalent shapes."""
    num_rows = batch_size
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    logits = (torch.randn(num_rows, N, device="cuda") * 2.0).to(dtype).contiguous()
    seq_lens = torch.full((num_rows,), N, dtype=torch.int32, device="cuda")
    pre_idx = _make_r0_pre_idx(logits, top_k, "real", seed=1)
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    _GvrTopKKernel.launch(logits, pre_idx, seq_lens, out, top_k)
    torch.cuda.synchronize()
    _gvr_check(tie_aware_check, out, logits, seq_lens, top_k, next_n=1, compress_ratio=1)

    # Override path: forcing the secant arm through launch() must also be a
    # valid top-K, and on fp32 the same index set up to boundary value-ties.
    # Restricted to fp32 because on half precision the secant arm only
    # re-checked the tie-aware value set the R0-equivalence sweep already
    # covers, at the cost of one more compiled kernel per cell.
    if dtype != torch.float32:
        return
    out_sec = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    _GvrTopKKernel.launch(logits, pre_idx, seq_lens, out_sec, top_k, enable_r0=False)
    torch.cuda.synchronize()
    _gvr_check(tie_aware_check, out_sec, logits, seq_lens, top_k, next_n=1, compress_ratio=1)
    # fp32 randn does collide bit-exactly at these sample counts, and a
    # collision on the top-K boundary makes the two arms' distinct picks
    # equally valid -- so compare the index sets tie-aware, not with
    # torch.equal (same reasoning as the R0-equivalence sweep above).
    _assert_index_sets_equal_tie_aware(out, out_sec, logits)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_dsl_gvr_topk_decode_p4_exact_tail_16bit(dtype):
    """16-bit exact-tail adversarial: two DISTINCT half-precision values in
    ONE fine bin straddling the K boundary (window-relative binning cannot
    separate them under a wide Phase-2 bracket — e.g. fp16 1.0 vs 1.25
    under a [0, 65504]-scale bracket). The 16-bit default stays OFF (the
    ambiguity gate fires on virtually every 16-bit input, measured gm
    1.29-1.36x envelope cost, while typical 16-bit inputs are value-exact
    without the tail); this covers the explicit OPT-IN: with
    ``p4_exact_tail=True`` the tail radix must keep every 1.25 above
    every 1.0."""
    torch.manual_seed(11)
    top_k, n = 1024, 32768
    bs = 2
    lo = torch.full((bs, n), -1.0, dtype=dtype, device="cuda")
    # wide bracket anchors (force a wide Phase-2 window)
    lo[:, :8] = torch.tensor(
        [60000.0, 40000.0, 20000.0, 10000.0, 5000.0, 2500.0, 1200.0, 600.0], device="cuda"
    ).to(dtype)
    # boundary: (top_k - 8 - 512) high-tie values 1.25 and 1024 low-tie 1.0
    n_hi, n_lo = top_k - 8 - 512, 1024
    perm = torch.randperm(n - 8, device="cuda") + 8
    hi_pos, lo_pos = perm[:n_hi], perm[n_hi : n_hi + n_lo]
    for r in range(bs):
        lo[r, hi_pos] = torch.tensor(1.25, dtype=dtype, device="cuda")
        lo[r, lo_pos] = torch.tensor(1.0, dtype=dtype, device="cuda")
    seq_lens = torch.full((bs,), n, dtype=torch.int32, device="cuda")
    pre = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    pre[:, 0] = lo.float().argmax(dim=-1).int()
    pre[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(bs, top_k, dtype=torch.int32, device="cuda")
    _GvrTopKKernel.launch(lo, pre, seq_lens, out, top_k, compress_ratio=1, p4_exact_tail=True)
    torch.cuda.synchronize()
    sel = torch.gather(lo.float(), -1, out.long())
    # every 1.25 must be selected before any 1.0 fills the remainder
    assert int((sel == 1.25).sum()) == n_hi * bs, (
        f"exact-tail 16-bit: {(sel == 1.25).sum()} of {n_hi * bs} high-tie values selected"
    )
    ref = torch.topk(lo.float(), top_k, dim=-1).values.sort(-1).values
    got = sel.sort(-1).values
    assert torch.equal(ref, got)


def test_cute_dsl_gvr_topk_decode_pick_policy_single_source():
    """The production runner's tuning adapter must agree with the kernel's
    pick_cluster_size/pick_tuning single source across a shape sweep
    (guards the de-duplicated launch-shape policy against drift)."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLGvrTopKDecodeRunner as R

    num_sms = 148
    for torch_dtype in (torch.float32, torch.bfloat16, torch.float16):
        for num_rows in (1, 4, 32, 148, 300, 512):
            for n in (4096, 16384, 65536, 131072, 262144):
                for msl in (None, 262144):
                    cs = _GvrTopKKernel.pick_cluster_size(
                        num_rows, msl if msl is not None else n, num_sms
                    )
                    cfg = _GvrTopKKernel.pick_config(
                        torch_dtype, num_rows, n, max_seq_len=msl, num_sms=num_sms
                    )
                    assert cfg["cluster_size"] == cs
                    tuning = R._pick_tuning(
                        torch_dtype,
                        num_rows,
                        (msl if msl is not None else n) // cs,
                        num_sms,
                        msl,
                        0,
                    )
                    assert tuning["num_threads_per_block"] == cfg["num_threads"]
                    assert tuning["use_256bit_load"] == cfg["use_256bit_load"]
                    assert tuning["min_blocks_per_mp"] == cfg["min_blocks_per_mp"]
                    assert (
                        tuning["enable_warp_parallel_reduce"] == cfg["enable_warp_parallel_reduce"]
                    )


# Every (dtype, variant) pair is its own compiled kernel. The plateau
# terminal is a value-space property (bitwise-equal plateau wider than kC),
# so fp32 carries all five admission/emit routes and fp16 only pins the
# half-precision bracket on the two extremes: the shipped default
# (rank_scatter_cs1) and the secant fallback (base_r0off).
# base_r0off exercises the classic secant admission (the exact fallback the
# production path takes when the R0 ladder misses): its refine budget can
# run out while the bracket is still wide, which is a different route into
# the plateau terminal than the R0 ladder's.
_PLATEAU_CELLS = [
    (torch.float32, "rank_scatter_cs1"),
    (torch.float32, "rank_scatter_cs4"),
    (torch.float32, "snap_cs1"),
    (torch.float32, "base_r0off"),
    (torch.float32, "base_r0off_cs4"),
    (torch.float16, "rank_scatter_cs1"),
    (torch.float16, "base_r0off"),
]


@skip_not_sm100
@pytest.mark.parametrize("dtype,variant", _PLATEAU_CELLS)
def test_cute_dsl_gvr_topk_decode_plateau_terminal(dtype, variant):
    """Adversarial plateau terminal (done == 3): a bitwise-equal plateau
    WIDER than the candidate buffer (kC) straddling the K boundary. Any
    threshold either overflows the buffer (>= plateau) or undershoots K
    (> plateau), so the Phase-2 bracket collapses to adjacent floats.
    The plateau terminal must emit the sure winners plus plateau members
    (exact tie-aware) instead of the old -1 pad / unfilled tail."""
    torch.manual_seed(23)
    top_k, n, bs = 1024, 32768, 2
    n_hi, n_plateau = 512, 8500  # kC = 6144 for K=1024; 8500 > kC
    lo = torch.full((bs, n), -1.0, dtype=dtype, device="cuda")
    for r in range(bs):
        perm = torch.randperm(n, device="cuda")
        hi = perm[:n_hi]
        pl = perm[n_hi : n_hi + n_plateau]
        lo[r, hi] = (5.0 + torch.arange(n_hi, device="cuda").float() * 0.01).to(dtype)
        lo[r, pl] = torch.tensor(1.0, dtype=dtype, device="cuda")
    seq_lens = torch.full((bs,), n, dtype=torch.int32, device="cuda")
    pre = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    pre[:, 0] = lo.float().argmax(dim=-1).int()
    pre[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(bs, top_k, dtype=torch.int32, device="cuda")
    overrides = {}
    if variant == "rank_scatter_cs4":
        overrides["cluster_size"] = 4
    elif variant == "snap_cs1":
        overrides["enable_p4_rank_scatter"] = False
    elif variant == "base_r0off":
        overrides["enable_r0"] = False
    elif variant == "base_r0off_cs4":
        overrides["enable_r0"] = False
        overrides["cluster_size"] = 4
    _GvrTopKKernel.launch(lo, pre, seq_lens, out, top_k, compress_ratio=1, **overrides)
    torch.cuda.synchronize()
    assert int((out < 0).sum()) == 0, (
        f"{int((out < 0).sum())} unfilled/-1 slots in the plateau terminal"
    )
    # in-range, unique
    assert out.max() < n and out.min() >= 0
    for r in range(bs):
        assert out[r].unique().numel() == top_k
    sel = torch.gather(lo.float(), -1, out.long())
    assert int((sel >= 5.0).sum()) == n_hi * bs, "missing sure winners"
    assert int((sel == 1.0).sum()) == (top_k - n_hi) * bs, "remaining slots must be plateau members"
    ref = torch.topk(lo.float(), top_k, dim=-1).values.sort(-1).values
    assert torch.equal(sel.sort(-1).values, ref)


# ============================================================================
# GVR non-converged threshold-search repair regressions (CuTe DSL counterpart
# of #17550): hostile/degenerate hints and ReLU-sparse tie plateaus used to
# yield a silently wrong top-K (row[0:K] or -1-padded rows). The tests force
# the in-tree kernel via TRTLLM_GVR_TIERS_DISABLE; the tiers are covered by
# test_cute_dsl_gvr_topk_tiers.py.
# ============================================================================


@pytest.fixture
def _intree_only(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
        gvr_topk_decode_dispatch as _disp,
    )

    monkeypatch.setenv("TRTLLM_GVR_TIERS_DISABLE", "1")
    _disp._reset_env_cache()
    yield
    monkeypatch.delenv("TRTLLM_GVR_TIERS_DISABLE", raising=False)
    _disp._reset_env_cache()


def _assert_exact_topk(out, logits, seq_lens, top_k, next_n, compress_ratio):
    """Self-contained per-row exactness: no -1 beyond the legal short-row
    pad, K distinct in-range indices, and a tie-aware value-multiset match
    against torch.topk over the row's N_eff prefix."""
    for r in range(out.shape[0]):
        seq = int(seq_lens[r // next_n])
        n_eff = min((seq - next_n + (r % next_n) + 1) // compress_ratio, logits.shape[1])
        k_eff = min(top_k, n_eff)
        idx = out[r].long()
        n_neg = int((idx < 0).sum())
        assert n_neg == top_k - k_eff, f"row {r}: {n_neg} -1 slots (legal pad = {top_k - k_eff})"
        sel = idx[idx >= 0]
        assert sel.numel() == k_eff, f"row {r}: {sel.numel()} valid indices, expected {k_eff}"
        assert bool((sel < n_eff).all()), f"row {r}: out-of-range index (n_eff={n_eff})"
        assert int(sel.unique().numel()) == k_eff, f"row {r}: duplicated indices"
        row = logits[r, :n_eff].float()
        assert torch.equal(
            logits[r].float()[sel].sort().values, row.topk(k_eff).values.sort().values
        ), f"row {r}: selected values differ from torch.topk"


@skip_not_sm100
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("hint", ["bottom_k", "uniform", "random"])
def test_cute_dsl_gvr_topk_decode_hostile_hint(top_k, hint, _intree_only):
    """A hint pointing away from the true top-K must not change the result;
    ``uniform`` covers the degenerate bracket that used to emit row[0:K]."""
    N, cr = 65536, 4
    g = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.randn(1, N, generator=g, dtype=torch.float32, device="cuda")
    flat = logits[0]
    if hint == "bottom_k":
        pre = flat.topk(top_k, largest=False).indices.to(torch.int32)
    elif hint == "uniform":
        pre = torch.full((top_k,), N // 2, dtype=torch.int32, device="cuda")
    else:
        pre = torch.randint(0, N, (top_k,), generator=g, device="cuda", dtype=torch.int32)
    pre = pre.view(1, top_k).contiguous()
    seq_lens = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
    out = torch.full((1, top_k), -1, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre, seq_lens, out, top_k=top_k, next_n=1, compress_ratio=cr
    )
    torch.cuda.synchronize()
    _assert_exact_topk(out, logits, seq_lens, top_k, 1, cr)


@skip_not_sm100
@pytest.mark.parametrize("n_pos", [3, 100, 1000])
def test_cute_dsl_gvr_topk_decode_relu_sparse_plateau(n_pos, _intree_only):
    """ReLU-sparse row (n_pos positives + exact-0.0 plateau wider than kC):
    the fail-soft used to return (top_k - n_pos) trailing -1 slots."""
    top_k, N, cr = 2048, 32768, 1
    g = torch.Generator(device="cuda").manual_seed(61)
    row = torch.zeros(N, dtype=torch.float32, device="cuda")
    row[torch.randperm(N, generator=g, device="cuda")[:n_pos]] = (
        torch.rand(n_pos, generator=g, device="cuda") + 1.0
    )
    logits = row.view(1, N).contiguous()
    pre = row.topk(top_k).indices.to(torch.int32).view(1, top_k).contiguous()
    seq_lens = torch.full((1,), N, dtype=torch.int32, device="cuda")
    out = torch.full((1, top_k), -1, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre, seq_lens, out, top_k=top_k, next_n=1, compress_ratio=cr
    )
    torch.cuda.synchronize()
    _assert_exact_topk(out, logits, seq_lens, top_k, 1, cr)


@skip_not_sm100
@pytest.mark.parametrize("next_n", [2, 4])
@pytest.mark.parametrize("hint", ["bottom_k", "uniform"])
def test_cute_dsl_gvr_topk_decode_mtp_hostile_hint(next_n, hint, _intree_only):
    """Hostile/degenerate hints under MTP row geometry (next_n > 1):
    request-level hint sharing plus the per-row N_eff arithmetic must stay
    exact when the repair path fires on every row."""
    top_k, N, cr, n_req = 512, 65536, 4, 2
    g = torch.Generator(device="cuda").manual_seed(7)
    num_rows = n_req * next_n
    logits = torch.randn(num_rows, N, generator=g, dtype=torch.float32, device="cuda")
    if hint == "bottom_k":
        pre1 = logits[0].topk(top_k, largest=False).indices.to(torch.int32)
    else:
        pre1 = torch.full((top_k,), N // 2, dtype=torch.int32, device="cuda")
    pre = pre1.view(1, top_k).expand(n_req, -1).contiguous()
    # exercise the mod-cr boundary: per-request kv_len differs by one token
    seq_lens = torch.tensor([N * cr, N * cr - 1], dtype=torch.int32, device="cuda")
    out = torch.full((num_rows, top_k), -1, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre, seq_lens, out, top_k=top_k, next_n=next_n, compress_ratio=cr
    )
    torch.cuda.synchronize()
    _assert_exact_topk(out, logits, seq_lens, top_k, next_n, cr)

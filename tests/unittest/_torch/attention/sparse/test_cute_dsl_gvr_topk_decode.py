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
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import (
    GvrTopKKernel as _GvrTopKKernel,
)
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL GVR Top-K only supports SM 100/103, got SM {get_sm_version()}",
)


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
        for j in range(1, top_k):
            pre_idx[:, j] = j
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


def _tie_aware_check(
    out_indices: torch.Tensor,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int = 1,
) -> None:
    """Vectorized multi-row tie-aware correctness check with strict sort+allclose.

    Per row r: scan range is ``logits[r, :N_eff(r)]`` where N_eff mirrors
    the kernel's exact formula (see ``GvrTopKKernel.gvr_topk_kernel``):

        actual_kv_len = seq_lens[r // next_n] - next_n + (r % next_n) + 1
        N_eff = actual_kv_len // compress_ratio   # cr=1 is identity

    Reference ``torch.topk`` is masked to this range so the reference and
    kernel scan exactly the same columns under any (next_n, cr) combo
    (including next_n>=3 + cr>=2 where the floor-division makes per-row
    N_eff vary within a group).

    All checks (out-of-range, duplicates, n_below, sort+allclose) run as
    batched GPU ops; only assertion-failure diagnostics fall back to host.
    """
    num_rows, top_k_out = out_indices.shape
    assert top_k_out == top_k
    device = logits.device
    logits_f32 = logits.to(torch.float32)
    N = logits.shape[1]

    # Per-row N_eff mirroring the kernel formula. seq_lens is per-group
    # (length num_rows // next_n); broadcast across the next_n rows of
    # each group before computing actual_kv_len // cr.
    row_idx = torch.arange(num_rows, device=device)
    group_idx = row_idx // next_n
    ofs = row_idx % next_n
    seq_lens_per_row = seq_lens.to(device=device, dtype=torch.long)[group_idx]
    actual_kv_len = seq_lens_per_row - next_n + ofs + 1
    N_eff = actual_kv_len // compress_ratio  # [num_rows]

    # Mask logits beyond per-row N_eff to -inf so torch.topk ignores tails.
    col_idx = torch.arange(N, device=device)
    in_range_mask = col_idx[None, :] < N_eff[:, None]  # [num_rows, N]
    masked_logits = torch.where(in_range_mask, logits_f32, float("-inf"))

    # Reference per-row top-K, sorted descending.
    ref_vals, _ = torch.topk(masked_logits, k=top_k, largest=True, sorted=True, dim=-1)

    # ---- 1. Out-of-range / -1 placeholder check (single fused mask) ----
    out_of_range = (out_indices < 0) | (out_indices >= N_eff[:, None])
    if bool(out_of_range.any().item()):
        bad_row = int(out_of_range.any(dim=1).int().argmax().item())
        bad_indices = out_indices[bad_row].cpu().tolist()
        raise AssertionError(
            f"row={bad_row}: kernel returned out-of-range index "
            f"(N_eff={int(N_eff[bad_row].item())}, indices={bad_indices})"
        )

    # ---- 2. Duplicate-index check (sort each row, scan consecutive eq) ----
    sorted_idx, _ = out_indices.sort(dim=-1)
    has_dup = (sorted_idx[:, 1:] == sorted_idx[:, :-1]).any(dim=-1)
    if bool(has_dup.any().item()):
        bad_row = int(has_dup.int().argmax().item())
        raise AssertionError(
            f"row={bad_row}: kernel returned duplicate indices: "
            f"{out_indices[bad_row].cpu().tolist()}"
        )

    # ---- 3. Gather selected values (safe — already in range) ----
    sel_vals = torch.gather(logits_f32, dim=-1, index=out_indices.long())

    # ---- 4. n_below check vs per-row K-th value ----
    kth_vals = ref_vals[:, -1:]  # [num_rows, 1]
    n_below_per_row = (sel_vals < kth_vals).sum(dim=-1)
    if bool((n_below_per_row > 0).any().item()):
        bad_row = int(n_below_per_row.argmax().item())
        n_below = int(n_below_per_row[bad_row].item())
        kth = float(kth_vals[bad_row, 0].item())
        raise AssertionError(
            f"row={bad_row}: {n_below} selected values < Kth-rank value ({kth:.6f})"
        )

    # ---- 5. Strict: sorted-value multiset == torch.topk reference ----
    sel_sorted, _ = sel_vals.sort(dim=-1, descending=True)
    diff = (sel_sorted - ref_vals).abs()
    if not bool(torch.allclose(sel_sorted, ref_vals, rtol=1e-5, atol=1e-5)):
        per_row_max = diff.max(dim=-1).values
        bad_row = int(per_row_max.argmax().item())
        max_diff = float(per_row_max[bad_row].item())
        raise AssertionError(f"row={bad_row}: sorted-value mismatch — max diff {max_diff:.4e}")


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        # Production cells: (bf16, K=512/1024) and (fp32, K=2048) match
        # the deployed K -> dtype mapping. (fp16, K=1024) is added to keep
        # the fp16 convert-to-fp32 tail path under test even though it is
        # not a current production cell.
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 65536])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("preidx_hit_rate", [0.0, 0.5])
@pytest.mark.parametrize("cluster_size", [1, 4])
@pytest.mark.parametrize("seqlen_sorted", [False, True])
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
    seqlen_sorted,
):
    """Compare custom op output against torch.topk reference (tie-aware).

    ``preidx_hit_rate=0.0`` exercises the worst-case (only argmax slot is
    a real topK index); ``0.5`` matches realistic production preIdx
    overlap with topK (V3.2 ~40%, V4 Pro ~75%) and exercises the
    kernel's Guess-phase short-circuit path.

    ``varlen=False`` uses uniform seq_lens=N*cr across the batch;
    ``varlen=True`` draws per-row seq_lens uniformly in [N/2, N]*cr.

    ``seqlen_sorted=True`` exercises the LJF host-side dispatch order:
    we build ``order_row`` as a descending argsort over ``seq_lens`` and
    pass it through the custom op. The
    kernel must produce the same per-row top-K (rows are still written
    back at their original positions, since the kernel uses
    ``row_idx = order_row[req] * next_n + nn`` for both reads and
    writes). The reference comparison is unchanged — it asserts that
    each row's output is a valid top-K of that row's masked logits.
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

    # LJF dispatch order — request-level descending argsort of seq_lens.
    order_row = (
        torch.argsort(seq_lens, descending=True, stable=False).to(torch.int32)
        if seqlen_sorted
        else None
    )

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

    _tie_aware_check(out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio)


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
def test_cute_dsl_gvr_topk_multi_cta_shortrow_degrade_boundary(dtype, top_k, cluster_size):
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
        _tie_aware_check(
            out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio
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
    _tie_aware_check(out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio)


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


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k",
    [(torch.bfloat16, 1024), (torch.float32, 2048)],
)
@pytest.mark.parametrize(
    "scenario,N,seq_lens_mode",
    [
        # N picked so all rows fall clearly below / above the 64K long_threshold.
        ("all_short", 8 * 1024, "uniform"),
        ("all_long", 128 * 1024, "uniform"),
        ("mixed_half", 128 * 1024, "half_short_half_long"),
    ],
)
@pytest.mark.parametrize("batch_size", [4, 32])
@pytest.mark.parametrize("next_n", [1, 2])
def test_lb_main_branches(dtype, top_k, scenario, N, seq_lens_mode, batch_size, next_n):
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

    _tie_aware_check(out_indices, logits, seq_lens, top_k, next_n, compress_ratio=1)


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [8 * 1024, 128 * 1024])  # below / above 64K threshold
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("batch_size", [4, 32])
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("preidx_hit_rate", [0.0, 0.5])
def test_lb_vs_reference(
    dtype,
    top_k,
    N,
    varlen,
    next_n,
    batch_size,
    compress_ratio,
    preidx_hit_rate,
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
    _tie_aware_check(
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
    return (pre - 1).clamp(min=0).contiguous()


@skip_not_sm100
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [8192, 65536])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("hint", ["real", "rand"])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_cute_dsl_gvr_topk_decode_r0_equivalence(dtype, top_k, N, batch_size, hint, cluster_size):
    """R0 admission (``enable_r0=True``, the new default) selects the same
    top-K as the secant baseline (``enable_r0=False``), by index set.

    ``hint='real'`` exercises the R0 admission-hit fast path; ``hint='rand'``
    forces the R0-miss log-falsi (R1) + fb_fix fallback. ``cluster_size=4``
    confirms R0 gates to single-CTA and the ``None`` R0 buffers propagate
    cleanly through the cluster path.
    """
    if N < top_k * 2:
        pytest.skip(f"N ({N}) < 2*top_k ({2 * top_k}): GVR histogram bucket too coarse")

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
    _tie_aware_check(out_base, logits, seq_lens, top_k, next_n=1, compress_ratio=1)
    _tie_aware_check(out_r0, logits, seq_lens, top_k, next_n=1, compress_ratio=1)

    # 2. Equivalence. fp32 logits are tie-free w.p. 1 → the top-K index set is
    #    unique, so R0 and base must return the identical set (order-independent,
    #    compared as sorted indices). For bf16/fp16 boundary value-ties permit
    #    distinct valid index sets, so equivalence there is the value-set
    #    equality already established in step 1 (both == torch.topk reference).
    if dtype == torch.float32:
        base_sorted, _ = out_base.sort(dim=-1)
        r0_sorted, _ = out_r0.sort(dim=-1)
        mismatch = (base_sorted != r0_sorted).any(dim=-1)
        if bool(mismatch.any().item()):
            bad = int(mismatch.int().argmax().item())
            raise AssertionError(
                f"row={bad}: R0 index set != secant-base index set "
                f"(base={sorted(out_base[bad].tolist())}, "
                f"r0={sorted(out_r0[bad].tolist())})"
            )

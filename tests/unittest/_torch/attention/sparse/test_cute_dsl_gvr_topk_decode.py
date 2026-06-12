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

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401
from tensorrt_llm._utils import get_sm_version

# Standalone LB wrappers live in run_gvr_topk.py alongside the single-CTA
# wrapper so tests and bench scripts share the same compile-time caching.
_RUN_GVR_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
    ),
    "tests",
    "scripts",
    "cute_dsl_kernels",
    "top_k",
)
if _RUN_GVR_DIR not in sys.path:
    sys.path.insert(0, _RUN_GVR_DIR)
from run_gvr_topk import (  # noqa: E402
    gvr_topk_lb_decode,
    gvr_topk_lb_prepare,
    gvr_topk_sort_prepare,
)

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
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    logits = logits_f32.to(dtype)

    num_groups = num_rows // next_n

    # seq_lens in UNCOMPRESSED space. Kernel divides by cr internally.
    if varlen:
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
    we build ``order_row`` via :func:`gvr_topk_sort_prepare` (descending
    argsort over ``seq_lens``) and pass it through the custom op. The
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

    # LJF dispatch order — request-level (length == batch_size).
    # gvr_topk_sort_prepare returns int32 on the same device as seq_lens.
    order_row = gvr_topk_sort_prepare(seq_lens) if seqlen_sorted else None

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
# Load-Balance (Idea C) tests.
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
    n_long_expect = int(round(B * ratio))
    seq_lens = torch.empty(B, dtype=torch.int32, device="cuda")
    seq_lens[:n_long_expect] = long_threshold * 2
    seq_lens[n_long_expect:] = long_threshold // 2
    perm = torch.randperm(B, device="cuda")
    seq_lens = seq_lens[perm]

    is_long = (seq_lens > long_threshold).cpu().numpy()
    ref_n_long = int(is_long.sum())
    ref_n_short = B - ref_n_long

    max_batch_size = 1024
    order_row, counters = gvr_topk_lb_prepare(seq_lens, max_batch_size, long_threshold)
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
    "scenario,N,override",
    [
        # N picked so all rows fall clearly below / above the 64K long_threshold.
        ("all_short", 8 * 1024, None),
        ("all_long", 128 * 1024, None),
        ("mixed_half", 128 * 1024, "half"),  # 50/50 long/short via override
    ],
)
@pytest.mark.parametrize("batch_size", [4, 32])
def test_lb_main_branches(dtype, top_k, scenario, N, override, batch_size):
    """Each LB branch (all_long / all_short / mixed) produces correct top-K.

    For ``mixed_half`` half the rows are forced to be short (seq_len < threshold)
    and half long, exercising both branches inside the same launch.
    """
    next_n = 1
    num_rows = batch_size * next_n
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
    )
    if override == "half":
        seq_lens = seq_lens.clone()
        seq_lens[: batch_size // 2] = 8 * 1024  # short half (< 64K threshold)
        seq_lens[batch_size // 2 :] = 128 * 1024  # long half  (> 64K threshold)

    max_batch_size = 1024
    long_threshold = 64 * 1024
    order_row, counters = gvr_topk_lb_prepare(seq_lens, max_batch_size, long_threshold)
    _, out_indices = gvr_topk_lb_decode(
        logits,
        pre_idx,
        seq_lens,
        order_row,
        counters,
        top_k,
        next_n,
        max_batch_size=max_batch_size,
        long_threshold=long_threshold,
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
    order_row, counters = gvr_topk_lb_prepare(seq_lens, max_batch_size, long_threshold)
    _, out_indices = gvr_topk_lb_decode(
        logits,
        pre_idx,
        seq_lens,
        order_row,
        counters,
        top_k,
        next_n,
        compress_ratio=compress_ratio,
        max_batch_size=max_batch_size,
        long_threshold=long_threshold,
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

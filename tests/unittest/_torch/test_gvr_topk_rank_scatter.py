# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness tests for the GVR Top-K Phase-4 rank-scatter variant.

Exercises ``enable_p4_rank_scatter`` / ``enable_p4_rank_scatter_exact`` on the
two paths the load-balance kernel uses:

  * single-CTA-per-row :class:`GvrTopKKernel` (``cluster_size=1``)
  * hybrid :class:`GvrTopKLBKernel` — short rows take the single-CTA branch,
    long rows (``seq_len > long_threshold``) take the cluster leader-only
    rank-scatter branch.

Both must return the exact top-K. Correctness is checked with a per-row,
tie-aware set/value comparison against a masked ``torch.topk`` reference: the
selected indices must be in range, unique, exactly ``top_k`` of them, and the
multiset of selected values must equal the true top-K values (so boundary ties
are accepted regardless of which tied index a kernel picks).

These are pure cuTe DSL kernels; the test only needs a Blackwell GPU
(sm_100/sm_103) and ``cutlass`` — no engine build or model weights.
"""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute  # noqa: E402

from tensorrt_llm._utils import get_sm_version  # noqa: E402

# Importing the kernels triggers no heavy init beyond cuTe DSL itself.
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import (  # noqa: E402
    GvrTopKKernel, )
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_load_balance import (  # noqa: E402
    GvrTopKLBKernel, GvrTopKLBPrepareKernel)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="requires a CUDA device"),
    pytest.mark.skipif(get_sm_version() < 100,
                       reason="GVR cuTe DSL Top-K targets Blackwell (sm_100+)"),
]

_DTYPE_TORCH_TO_CUTE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}

# LB prepare classifies rows with (seq_len / compress_ratio) > long_threshold
# as "long" (cluster branch). 64K matches the production default.
_LONG_THRESHOLD = 64 * 1024


# ---------------------------------------------------------------------------
# Input builder + tie-aware reference (mirrors the PR #15304 driver helper).
# ---------------------------------------------------------------------------
def _make_varlen_inputs(seq_lens_host, top_k, dtype, seed=42, compress_ratio=1):
    """Build (logits, pre_idx, seq_lens) for a variable-length, next_n=1 batch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_rows = len(seq_lens_host)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    n_raw = int(max(seq_lens_host)) // compress_ratio
    # Pad widest row to a multiple of 16 elems so every row start is aligned
    # for vectorized loads; columns past a row's N_eff are never scanned.
    n_cols = ((n_raw + 15) // 16) * 16
    logits = (torch.randn(num_rows, n_cols, dtype=torch.float32, device="cuda")
              * 2.0).to(dtype)
    # pre_idx is the temporal hint (prev-step top-K); seed[:,0] with the argmax
    # and fill the rest with a cheap ramp — GVR uses it only to seed Phase-2.
    argmax_idx = logits.argmax(dim=-1).int()
    pre_idx = torch.zeros(num_rows, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j
    return logits, pre_idx, seq_lens


def _assert_tie_aware_correct(kernel_idxs, logits, seq_lens, top_k,
                              compress_ratio=1):
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    for row in range(kernel_idxs.shape[0]):
        actual_kv_len = int(seq_lens_host[row])
        n_eff = actual_kv_len // compress_ratio
        if n_eff < top_k:
            continue
        row_logits = logits_f32[row, :n_eff]
        topk_vals, _ = torch.topk(row_logits, k=top_k, largest=True, sorted=True)
        kth_value = topk_vals[-1].item()
        sel = [int(i) for i in kernel_idxs[row].cpu().tolist() if i >= 0]
        assert all(i < n_eff for i in sel), f"row {row}: out-of-range index"
        assert len(set(sel)) == len(sel), f"row {row}: duplicate indices"
        assert len(sel) == top_k, f"row {row}: got {len(sel)} != {top_k} indices"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device,
                                           dtype=torch.long)]
        assert int((sel_vals < kth_value).sum().item()) == 0, (
            f"row {row}: a selected value is below the Kth-rank value")
        sel_sorted, _ = sel_vals.sort(descending=True)
        assert torch.allclose(sel_sorted, topk_vals, rtol=1e-5, atol=1e-5), (
            f"row {row}: selected value multiset != true top-K")


# ---------------------------------------------------------------------------
# Single-CTA path (cluster_size=1).
# ---------------------------------------------------------------------------
def _run_single_cta(logits, pre_idx, seq_lens, top_k, rank_scatter, exact):
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]
    num_rows = logits.shape[0]
    out_indices = torch.empty((num_rows, top_k), dtype=torch.int32,
                              device=logits.device)
    kernel = GvrTopKKernel(
        dtype=cute_dtype, top_k=top_k, next_n=1, num_threads=512,
        use_256bit_load=False, compress_ratio=1, return_output_values=False,
        cluster_size=1,
        enable_p4_rank_scatter=rank_scatter,
        enable_p4_rank_scatter_exact=exact,
    )
    n_rows_s, n_cols_s = cute.sym_int(), cute.sym_int()
    fl = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (n_rows_s, n_cols_s), stride_order=(1, 0), assumed_align=16)
    fp = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(), top_k), stride_order=(1, 0), assumed_align=16)
    fs = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(),), stride_order=(0,))
    fo = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(), top_k), stride_order=(1, 0), assumed_align=16)
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kernel, fl, fp, fs, None, fo, None, stream=st,
                            options="--enable-tvm-ffi")
    compiled(logits, pre_idx, seq_lens, None, out_indices, None)
    torch.cuda.synchronize()
    return out_indices


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_single_cta_rank_scatter_exact(dtype, top_k):
    """op#7: single-CTA-per-row GVR with exact rank-scatter P4 is exact."""
    n = 16384
    logits, pre_idx, seq_lens = _make_varlen_inputs([n], top_k, dtype, seed=42)
    out = _run_single_cta(logits, pre_idx, seq_lens, top_k,
                          rank_scatter=True, exact=True)
    _assert_tie_aware_correct(out, logits, seq_lens, top_k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_cta_rank_scatter_matches_snap_baseline(dtype):
    """rank-scatter and the default histogram-snap P4 must both be exact and
    agree on the selected value set (algorithm cross-check)."""
    top_k, n = 512, 8192
    logits, pre_idx, seq_lens = _make_varlen_inputs([n], top_k, dtype, seed=7)
    out_snap = _run_single_cta(logits, pre_idx, seq_lens, top_k,
                               rank_scatter=False, exact=False)
    out_rs = _run_single_cta(logits, pre_idx, seq_lens, top_k,
                             rank_scatter=True, exact=True)
    _assert_tie_aware_correct(out_snap, logits, seq_lens, top_k)
    _assert_tie_aware_correct(out_rs, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Hybrid load-balance path (op#8 cluster leader + op#7 short branch).
# ---------------------------------------------------------------------------
def _run_lb(logits, pre_idx, seq_lens, top_k, cluster_size, rank_scatter, exact,
            max_batch_size=64):
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]
    num_rows, n_cols = logits.shape
    device = logits.device

    prep = GvrTopKLBPrepareKernel(long_threshold=_LONG_THRESHOLD,
                                  num_threads=max_batch_size)
    fs = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_rows,),
                                               stride_order=(0,))
    fr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (max_batch_size,),
                                               stride_order=(0,))
    fc = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    prep_compiled = cute.compile(prep, fs, fr, fc, cutlass.Int32(0), stream=st,
                                 options="--enable-tvm-ffi")
    order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
    counters = torch.zeros(2, dtype=torch.int32, device=device)
    prep_compiled(seq_lens, order_row, counters, cutlass.Int32(num_rows))

    kernel = GvrTopKLBKernel(
        dtype=cute_dtype, top_k=top_k, next_n=1, num_threads=512,
        compress_ratio=1, return_output_values=False, cluster_size=cluster_size,
        max_batch_size=max_batch_size,
        enable_p4_rank_scatter=rank_scatter,
        enable_p4_rank_scatter_exact=exact,
    )
    fl = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (num_rows, n_cols), stride_order=(1, 0), assumed_align=16)
    fp = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16)
    fs2 = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_rows,),
                                                stride_order=(0,))
    fo = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16)
    fr2 = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (max_batch_size,),
                                                stride_order=(0,))
    fc2 = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    main_compiled = cute.compile(kernel, fl, fp, fs2, None, fo, fr2, fc2, stream=st,
                                 options="--enable-tvm-ffi")
    out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    main_compiled(logits, pre_idx, seq_lens, None, out_indices, order_row, counters)
    torch.cuda.synchronize()
    return out_indices, counters


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_lb_hybrid_rank_scatter_exact(dtype):
    """op#8 + op#7: hybrid load-balance kernel with exact rank-scatter P4 is
    exact across both the long (cluster) and short (single-CTA) branches."""
    top_k = 512
    # one long row (>64K -> cluster branch) + two short rows (single-CTA branch)
    seq_lens_host = [80000, 4096, 8192]
    logits, pre_idx, seq_lens = _make_varlen_inputs(seq_lens_host, top_k, dtype,
                                                    seed=123)
    out, counters = _run_lb(logits, pre_idx, seq_lens, top_k, cluster_size=4,
                            rank_scatter=True, exact=True)
    # prepare must have classified exactly 1 long + 2 short (both branches live)
    assert counters.cpu().tolist() == [1, 2]
    _assert_tie_aware_correct(out, logits, seq_lens, top_k)

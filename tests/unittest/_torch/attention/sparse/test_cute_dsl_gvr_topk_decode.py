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

from typing import Tuple

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, pre_idx, seq_lens) for the op.

    ``logits`` lives in compressed-token-index space (``N = N_uncompressed /
    compress_ratio``). ``seq_lens`` is in UNCOMPRESSED space — kernel divides
    by ``compress_ratio`` internally. ``pre_idx[..., 0]`` is the per-group
    argmax (indexer invariant).
    """
    torch.manual_seed(seed)
    device = "cuda"
    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device=device) * 2.0
    logits = logits_f32.to(dtype)

    num_groups = num_rows // next_n
    argmax_idx = logits[::next_n].argmax(dim=-1).int()  # one per group
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device=device)
    pre_idx[:, 0] = argmax_idx
    # Fill remaining columns with arbitrary in-range indices (kernel only
    # treats slot 0 as the argmax hint; the rest contribute to the
    # guess-set merge but don't have an invariant).
    for j in range(1, top_k):
        pre_idx[:, j] = j

    # seq_lens is uncompressed; kernel divides by cr internally.
    # ``seq_lens = N * cr`` makes the kernel's
    # ``N_kernel = (seq_lens - next_n + ofs + 1) // cr`` match ref's
    # ``N_eff = N - next_n + ofs + 1`` for every row in a next_n in {1, 2}
    # group. (For cr=1 this reduces to seq_lens = N.) NOTE: next_n >= 3
    # + cr >= 2 has an unavoidable floor-division mismatch across rows in
    # the same group; not exercised by the current sweep.
    seq_lens_val = N * compress_ratio
    seq_lens = torch.full((num_groups,), seq_lens_val, dtype=torch.int32, device=device)
    return logits, pre_idx, seq_lens


def _tie_aware_check(
    out_indices: torch.Tensor,
    logits: torch.Tensor,
    top_k: int,
    next_n: int,
) -> None:
    """Vectorized multi-row tie-aware correctness check with strict sort+allclose.

    Per row r: scan range is ``logits[r, :N_eff]`` where
    ``N_eff = logits.shape[1] - next_n + (r % next_n) + 1``. Reference
    ``torch.topk`` is masked to this range so next_n>1 doesn't produce
    false negatives from columns the kernel never reads.

    All checks (out-of-range, duplicates, n_below, sort+allclose) run as
    batched GPU ops; only assertion-failure diagnostics fall back to host.
    """
    num_rows, top_k_out = out_indices.shape
    assert top_k_out == top_k
    device = logits.device
    logits_f32 = logits.to(torch.float32)
    N = logits.shape[1]

    # Per-row N_eff (length-num_rows), depends only on (row % next_n).
    row_idx = torch.arange(num_rows, device=device)
    N_eff = N - next_n + (row_idx % next_n) + 1  # [num_rows]

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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("N", [4096, 65536])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_cute_dsl_gvr_topk_decode(dtype, top_k, N, next_n, batch_size, compress_ratio):
    """Compare custom op output against torch.topk reference (tie-aware)."""
    if N - next_n + 1 < top_k:
        pytest.skip(
            f"N_eff < top_k ({N - next_n + 1} < {top_k}) is a degenerate path not exercised here"
        )

    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        next_n,
        seed=42,
        compress_ratio=compress_ratio,
    )

    out_values = torch.empty(num_rows, top_k, dtype=dtype, device="cuda")
    out_indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_values,
        out_indices,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    _tie_aware_check(out_indices, logits, top_k, next_n)

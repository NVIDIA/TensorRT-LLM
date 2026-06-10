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
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
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
):
    """Compare custom op output against torch.topk reference (tie-aware).

    ``preidx_hit_rate=0.0`` exercises the worst-case (only argmax slot is
    a real topK index); ``0.5`` matches realistic production preIdx
    overlap with topK (V3.2 ~40%, V4 Pro ~75%) and exercises the
    kernel's Guess-phase short-circuit path.

    ``varlen=False`` uses uniform seq_lens=N*cr across the batch;
    ``varlen=True`` draws per-row seq_lens uniformly in [N/2, N]*cr.
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

    _tie_aware_check(out_indices, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio)

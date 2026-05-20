# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Correctness tests for the cuTe DSL GVR Top-K kernel.

Compares the kernel output against ``torch.topk`` across the supported
(dtype, K) specializations using tie-aware set equality.
"""

from typing import Tuple

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import gvr_topk_decode


def _make_inputs(
    N: int, top_k: int, dtype: torch.dtype, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, pre_idx, seq_lens) for a single-row test.

    ``pre_idx.shape[1] == top_k`` per CUDA invariant
    (heuristic_topk.cuh:810: `preIdxCount == topK` is a dispatch precondition).
    """
    torch.manual_seed(seed)
    device = "cuda"
    logits_f32 = torch.randn(1, N, dtype=torch.float32, device=device) * 2.0
    logits = logits_f32.to(dtype)
    argmax_idx = int(logits[0].argmax().item())
    pre_idx_list = [argmax_idx] + [i for i in range(top_k - 1)]
    pre_idx = torch.tensor(pre_idx_list, dtype=torch.int32, device=device).view(1, top_k)
    seq_lens = torch.tensor([N], dtype=torch.int32, device=device)
    return logits, pre_idx, seq_lens


def _tie_aware_correct(
    kernel_idxs: torch.Tensor, logits: torch.Tensor, top_k: int, effective_len: int
) -> Tuple[bool, int]:
    """Check that every selected index resolves to a value >= the K-th-rank value
    in ``logits[0, :effective_len]``, and that the returned set has no duplicates
    and is full.

    Set-equality is too strict when many candidates share the same value
    (common for bf16/fp16 with large vocab).

    ``effective_len`` is the per-row scan length the kernel actually uses:
        N_eff = seq_lens[0] - next_n + (row_idx % next_n) + 1
    For row 0 (the only one this test inspects), ``N_eff = seq_lens[0] - next_n + 1``.
    Reference torch.topk must be masked to this range — otherwise next_n>1
    produces false negatives from columns the kernel never reads.
    """
    logits_f32 = logits[0, :effective_len].to(torch.float32)
    topk_vals, _ = torch.topk(logits_f32, k=top_k, largest=True, sorted=True)
    kth_value = topk_vals[-1].item()
    sel = [int(i) for i in kernel_idxs[0].cpu().tolist() if i >= 0]
    if any(i >= effective_len for i in sel):
        # Kernel selected an out-of-range index — definite failure.
        return False, sum(1 for i in sel if i >= effective_len)
    vals = logits_f32[torch.tensor(sel, device=logits.device, dtype=torch.long)]
    n_below = int((vals < kth_value).sum().item())
    n_dups = len(sel) - len(set(sel))
    ok = (n_below == 0) and (n_dups == 0) and (len(sel) == top_k)
    return ok, n_below + n_dups + max(0, top_k - len(sel))


_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_TOPKS = [512, 1024, 2048]
_NS = [4096, 8192, 16384, 65536]
_SEEDS = [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("top_k", _TOPKS)
@pytest.mark.parametrize("N", _NS)
@pytest.mark.parametrize("seed", _SEEDS)
# limin expanded next_n test. originally, it is [1]
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
def test_gvr_topk_decode_correctness(
    dtype: torch.dtype, top_k: int, N: int, seed: int, next_n: int
) -> None:
    # Kernel scans `N_eff = seq_lens[0] - next_n + (row_idx % next_n) + 1`
    # columns for row 0 (= seq_lens[0] - next_n + 1). The degenerate
    # `N_eff <= top_k` path is a separate code branch — skip here.
    effective_len = N - next_n + 1
    if effective_len < top_k:
        pytest.skip("N_eff < top_k is degenerate; the kernel requires N_eff >= top_k")
    logits, pre_idx, seq_lens = _make_inputs(N, top_k, dtype, seed)
    _, out_idxs = gvr_topk_decode(logits, pre_idx, seq_lens, top_k, next_n=next_n)
    torch.cuda.synchronize()
    ok, err = _tie_aware_correct(out_idxs, logits, top_k, effective_len)
    assert ok, (
        f"dtype={dtype} K={top_k} N={N} seed={seed} next_n={next_n}: "
        f"tie-aware check failed (err_count={err})"
    )

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
"""Deterministic self-sampling GVR top-K: fixed selected set, fixed order, optional ascending rows."""

import pytest
import torch

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling_host as ss
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="self-sampling GVR kernels target sm_100a",
)

# (batch, n, k): one shape per kernel family (reg, reg_clus, clus, main R=1, main SPLIT)
_SHAPES = [
    (1, 8192, 512),
    (64, 32768, 1024),
    (64, 131072, 512),
    (128, 32768, 2048),
    (1, 524288, 1024),
    (256, 32783, 1024),
    (512, 32783, 2048),
    (256, 131087, 2048),
    (512, 131075, 1024),
    (4, 32771, 2048),  # reg_clus with K=2048
]
_PATTERNS = ["unique", "quantized", "equal", "boundary", "clustered"]


def _row(n: int, k: int, pattern: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(20261011 + n)
    base = torch.randperm(n, generator=gen).float() / n
    if pattern == "unique":
        return base
    if pattern == "quantized":
        return base.bfloat16().float()
    if pattern == "equal":
        return torch.ones(n)
    if pattern == "clustered":
        # Concentrate high scores in the chunks owned by a single warp.
        indices = torch.arange(n)
        return base + 2 * ((indices // 4) % 512 < 32).float()
    perm = torch.randperm(n, generator=gen)  # 64 ties straddling the boundary, 16 of them selected
    row = torch.zeros(n)
    row[perm[: k - 16]] = 2
    row[perm[k - 16 : k + 48]] = 1
    return row


def _oracle(row: torch.Tensor, k: int) -> set:
    """{score > tau} + the smallest indices among {score == tau}: stable sort by (-score, index)."""
    order = torch.sort(-row, stable=True).indices
    return set(order[:k].tolist())


def _run(scores, k, n, **kw):
    b = scores.shape[0]
    if scores.shape[1] % 16:
        scores = torch.nn.functional.pad(scores, (0, -scores.shape[1] % 16))
    lengths = torch.full((b,), n, dtype=torch.int32, device="cuda")
    out = torch.empty((b, k), dtype=torch.int32, device="cuda")
    ss.run_varlen(scores, lengths, out, max_seq_len=n, **kw)
    torch.cuda.synchronize()
    return out


@skip_not_sm100
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("b,n,k", _SHAPES)
def test_deterministic_set_and_order(b, n, k, pattern):
    row = _row(n, k, pattern)
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    oracle = _oracle(row, k)
    outs = [_run(scores, k, n, deterministic=True) for _ in range(4)]
    for out in outs:
        assert torch.equal(out, outs[0])  # fixed order across launches
        for r in (0, b - 1):
            assert set(out[r].tolist()) == oracle  # fixed selected set, smaller index wins ties


@skip_not_sm100
@pytest.mark.parametrize("b,n,k", _SHAPES[::2])
def test_deterministic_sorted(b, n, k):
    row = _row(n, k, "quantized")
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    out = _run(scores, k, n, deterministic=True, sorting="ascending")
    assert torch.equal(out, out.sort(dim=1).values)
    assert set(out[0].tolist()) == _oracle(row, k)


@skip_not_sm100
@pytest.mark.parametrize("rows,n,k", [(1, 8192, 512), (149, 32768, 1024), (297, 32768, 2048)])
def test_deterministic_prefill(rows, n, k):
    """Prefill windows [ks, ke) per row: oracle set in the local frame, fixed order, ascending option."""
    gen = torch.Generator().manual_seed(20261012 + n)
    scores = torch.randperm(rows * n, generator=gen).float().view(rows, n).bfloat16().float().cuda()
    ks = torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32)
    ke = n - torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    runs = []
    for _ in range(3):
        ss.run_prefill(scores, ks.cuda(), ke.cuda(), out, deterministic=True)
        torch.cuda.synchronize()
        runs.append(out.clone())
    assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2])
    for r in (0, rows - 1):
        row = scores[r, int(ks[r]) : int(ke[r])].cpu()
        assert set(runs[0][r].tolist()) == _oracle(row, k)  # local frame: column - ks
    ss.run_prefill(scores, ks.cuda(), ke.cuda(), out, deterministic=True, sorting="ascending")
    torch.cuda.synchronize()
    assert torch.equal(out, out.sort(dim=1).values)
    assert torch.equal(out.sort(dim=1).values, runs[0].sort(dim=1).values)


@skip_not_sm100
def test_default_path_is_the_original_kernel():
    b, n, k = _SHAPES[2]
    row = _row(n, k, "unique")
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    out = _run(scores, k, n)
    assert set(out[0].tolist()) == _oracle(row, k)  # exact set on tie-free data
    with pytest.raises(ValueError):
        _run(scores, k, n, sorting="ascending")  # sorting needs deterministic=True

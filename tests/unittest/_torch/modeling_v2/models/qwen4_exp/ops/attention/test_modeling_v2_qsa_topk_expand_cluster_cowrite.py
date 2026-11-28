# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`qsa_topk_expand_cluster(..., cowrite=True)`: the cluster top-k's selection stored through a
shared position -> block map, coalesced.

* **GPU, every cell** (the op's own cells, which the FP8 target's attention-op test holds to the
  reference), eagerly and replayed from a CUDA graph over fresh scores: `topk_indices` and the
  token selection with `cowrite` are bitwise the launch without it, and the reference's.
* **CPU**: the reference ignores `cowrite`; `is_valid` refuses a block budget past the map.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_topk_expand_cluster__decode as op,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the cluster top-k is served on SM10x",
)
K, CR, TOKEN_TOPK = 512, 4, 2048


def _operands(spec, seed: int, device):
    """One cell's arguments (the FP8 target's attention-op test's builder): garbage and a NaN
    past each row's window, which no form may read."""
    g = torch.Generator(device=device).manual_seed(seed)
    W = spec["width"]
    if "requests" in spec:
        pos = torch.cat(
            [torch.arange(s, s + c, dtype=torch.int64, device=device) for s, c in spec["requests"]]
        )
        seqlen = torch.cat(
            [torch.full((c,), s + c, dtype=torch.int32, device=device) for s, c in spec["requests"]]
        )
    else:
        pos = torch.tensor(spec["pos"], dtype=torch.int64, device=device)
        seqlen = (pos + 1).to(torch.int32)
    rows = pos.numel()
    scores = torch.randn(rows, W, generator=g, device=device).abs() * 3.0
    if spec.get("relu_sparse"):
        scores = torch.relu(torch.randn(rows, W, generator=g, device=device) - 2.2)
        scores[:, ::7] = -0.0
    if spec.get("relu_sum"):
        scores = torch.zeros(rows, W, device=device)
        for _ in range(4):
            scores += torch.relu(torch.randn(rows, W, generator=g, device=device))
        scores *= 0.0884
    if spec.get("equal"):
        scores = torch.full((rows, W), 0.5, device=device)
    if spec.get("lognormal"):
        scores = torch.exp(1.74 + 0.2 * torch.randn(rows, W, generator=g, device=device))
        scores = scores.to(torch.bfloat16).float()
        scores[::2, 5] = 0.0
    ends = ((pos + 1) // CR).to(torch.int32)
    cols = torch.arange(W, device=device)
    scores = torch.where(cols[None, :] >= ends[:, None], 1.0e30, scores)
    scores = torch.where(cols[None, :] == ends[:, None] + 3, float("nan"), scores)
    return scores, ends, pos, seqlen


def _knobs(spec) -> dict:
    return {k: spec[k] for k in ("solo_blocks", "pairs") if k in spec}


@_GPU
@pytest.mark.parametrize("cell", op.qsa_topk_expand_cluster.CELLS, ids=lambda c: c.why[:40])
def test_cowrite_is_bitwise_the_scattered_write(cell):
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31
    scores, ends, pos, seqlen = _operands(spec, seed, device)
    rows = scores.shape[0]
    outs = []
    for cowrite in (False, True):
        topk = torch.full((rows, K), -7, dtype=torch.int32, device=device)
        sel = op.qsa_topk_expand_cluster(
            scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK, cowrite=cowrite, **_knobs(spec)
        )
        outs.append((topk, sel))
    torch.cuda.synchronize()
    (t0, s0), (t1, s1) = outs
    assert torch.equal(t1, t0), "block list"
    assert torch.equal(s1, s0), "token selection"
    want_topk, want_sel = op.qsa_topk_expand_cluster.reference(
        scores, ends, pos, seqlen, t0.clone(), CR, TOKEN_TOPK
    )
    assert torch.equal(t1, want_topk) and torch.equal(s1, want_sel)

    # Captured once, replayed over fresh scores in the static buffer.
    s_scores = scores.clone()
    topk = torch.full((rows, K), -7, dtype=torch.int32, device=device)
    out = torch.empty_like(s0)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op.qsa_topk_expand_cluster(
            s_scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK, out=out, cowrite=True, **_knobs(spec)
        )
    torch.cuda.current_stream().wait_stream(stream)
    for k in range(2):
        fresh, _, _, _ = _operands(spec, seed + 1 + k, device)
        s_scores.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        want_topk, want_sel = op.qsa_topk_expand_cluster.reference(
            s_scores, ends, pos, seqlen, topk.clone(), CR, TOKEN_TOPK
        )
        assert torch.equal(topk, want_topk) and torch.equal(out, want_sel)


@pytest.mark.cpu_only
def test_reference_ignores_cowrite_and_is_valid_bounds_the_map():
    rows, W = 2, 4100
    scores = torch.rand(rows, W)
    ends = torch.tensor([W, 300], dtype=torch.int32)
    pos = (ends.to(torch.int64) * CR - 1).clamp(min=0)
    lens = (pos + 1).to(torch.int32)
    topk = torch.full((rows, K), -7, dtype=torch.int32)
    a = op.qsa_topk_expand_cluster.reference(scores, ends, pos, lens, topk, CR, TOKEN_TOPK)
    b = op.qsa_topk_expand_cluster.reference(
        scores, ends, pos, lens, topk, CR, TOKEN_TOPK, cowrite=True
    )
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    op.qsa_topk_expand_cluster.is_valid(scores, ends, pos, lens, topk, CR, TOKEN_TOPK, cowrite=True)
    wide = torch.full((rows, 8192), -7, dtype=torch.int32)
    with pytest.raises(AssertionError):
        op.qsa_topk_expand_cluster.is_valid(
            scores, ends, pos, lens, wide, CR, 8192 * CR, cowrite=True
        )

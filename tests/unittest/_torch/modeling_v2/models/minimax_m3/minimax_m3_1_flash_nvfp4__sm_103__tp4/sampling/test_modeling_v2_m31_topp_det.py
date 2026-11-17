# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET: the spec sampler's top-p renorm as
``family_ops().minimax_m31_topp_renorm_det`` through ``ops/sampling/m31_topp_det``, installed as the sampler's
top-p renorm override.

On rows with top_p < 1 it must equal FlashInfer's ``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` bit for
bit (the microbenchmark generators: randn / peaked / ties / near_uniform / all_equal rows);
rows with top_p >= 1 keep every token. The shared sampler hook must be byte-for-byte stock while no override is set, and
fall back to the stock call whenever the override declines.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_103__tp4.ops.sampling import (  # noqa: E501
        m31_topp_det as entry,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

FLT_MIN = torch.finfo(torch.float32).tiny
op = entry.m31_topp_det


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("TOPP_DET is an sm_103 kernel")
    if entry.topp_det_op() is None:
        pytest.skip("minimax_m31_topp_renorm_det is not built")
    pytest.importorskip("flashinfer.sampling")


def _make(cls, rows, d, seed):
    """The microbenchmark row generators: softmax of the class's logits."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if cls == "randn":
        logits = torch.randn(rows, d, generator=g, device="cuda") * 3
    elif cls == "peaked":  # few dominant tokens, long tail (real-like draft rows)
        logits = torch.randn(rows, d, generator=g, device="cuda") * 2
        idx = torch.randint(0, d, (rows, 8), generator=g, device="cuda")
        logits.scatter_(1, idx, 14 + 3 * torch.rand(rows, 8, generator=g, device="cuda"))
    elif cls == "ties":
        logits = torch.round(torch.randn(rows, d, generator=g, device="cuda") * 2) / 2
    elif cls == "near_uniform":
        logits = torch.randn(rows, d, generator=g, device="cuda") * 1e-3
    elif cls == "all_equal":
        logits = torch.zeros(rows, d, device="cuda")
    else:
        raise ValueError(cls)
    return torch.softmax(logits, -1).contiguous()


def _top_p(mode, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    if mode == "p95":
        return torch.full((rows,), 0.95, device="cuda")
    if mode == "mixed":
        return (0.05 + 0.95 * torch.rand(rows, generator=g, device="cuda")).contiguous()
    raise ValueError(mode)


def _fi_det(probs, top_p):
    import flashinfer.sampling

    return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)


def _same(a, b, what):
    """Bit for bit."""
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"{what}: fp32 bits differ"


# Every class at p95; the mixed top-p mode on randn only.
# Boundary classes at top_p 0.95, plus a batch mixing top_p values.
_CLASS_MODES = [("randn", "p95"), ("ties", "p95"), ("all_equal", "p95"), ("randn", "mixed")]


@pytest.mark.parametrize("cls,mode", _CLASS_MODES, ids=[f"{c}-{m}" for c, m in _CLASS_MODES])
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_bitwise_flashinfer_det_below_one(cell, cls, mode) -> None:
    """Rows with top_p < 1 at each certified (rows, d): the override equals FlashInfer's det renorm bit for bit, and
    the explicit entry call gives the same bits again (deterministic)."""
    _gpu()
    rows, d = cell.spec["rows"], cell.spec["d"]
    for seed in range(2):
        probs = _make(cls, rows, d, 1000 * rows + seed)
        top_p = _top_p(mode, rows, seed)
        got = entry.top_p_renorm_det(probs, top_p)
        assert got is not None, "an in-contract input must be served"
        ref = _fi_det(probs, top_p)
        _same(got, ref, f"{cls} {mode} rows={rows} seed={seed}")
        again = op(
            probs,
            top_p,
            entry.prebuild(probs.device),
            entry.ctas_per_row(rows, entry.sm_count(probs.device)),
        )
        _same(again, got, "repeat")


@pytest.mark.parametrize("cls", ["randn", "ties", "all_equal"])
def test_keep_all_rows_in_a_mixed_batch(cls) -> None:
    """A batch of 8 rows, a third at top_p 1: the top_p 1 rows keep every element whose renormed value is a normal float
    (output nonzero <=> ftz(p * norm) != 0, the --use_fast_math semantics FlashInfer's apply shares) and sum to 1 within
    1e-5; the top_p < 1 rows are FlashInfer's det renorm bit for bit."""
    _gpu()
    rows, d = 8, 200064
    probs = _make(cls, rows, d, 77)
    top_p = _top_p("mixed", rows, 3)
    keep = torch.zeros(rows, dtype=torch.bool, device="cuda")
    keep[::3] = True
    top_p[keep] = 1.0
    got = entry.top_p_renorm_det(probs, top_p)
    assert got is not None
    ref = _fi_det(probs, top_p)
    _same(got[~keep], ref[~keep], f"{cls}: top_p < 1 rows")
    for r in keep.nonzero().flatten().tolist():
        p64 = probs[r].double()
        norm = 1.0 / float(p64[probs[r] >= FLT_MIN].sum())
        # kept <=> ftz(p * norm) != 0; elements within 1e-6 of the FLT_MIN boundary are left to the rounding
        live = p64 * norm >= FLT_MIN * (1 + 1e-6)
        dead = probs[r] < FLT_MIN
        assert bool((got[r][live] != 0).all()), f"{cls} row {r}: a kept token was dropped"
        assert bool((got[r][dead] == 0).all()), f"{cls} row {r}: a sub-FLT_MIN token is nonzero"
        total = float(got[r].double().sum())
        assert abs(total - 1.0) <= 1e-5, f"{cls} row {r}: sum {total}"


def test_graph_replay_equals_eager() -> None:
    """Captured in a CUDA graph (workspace prebuilt outside the capture), the replay gives the eager bits."""
    _gpu()
    probs = _make("peaked", 8, 200064, 5)
    top_p = _top_p("mixed", 8, 5)
    entry.prebuild(probs.device)
    ref = entry.top_p_renorm_det(probs, top_p)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        entry.top_p_renorm_det(probs, top_p)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.top_p_renorm_det(probs, top_p)
    assert out is not None, "a prebuilt workspace must serve a captured call"
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _same(out, ref, "graph replay")

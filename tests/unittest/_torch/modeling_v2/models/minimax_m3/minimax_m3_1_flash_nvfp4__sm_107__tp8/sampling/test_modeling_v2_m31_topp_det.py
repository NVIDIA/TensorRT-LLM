# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET: the spec sampler's top-p renorm as ``mv2_minimax_m3::minimax_m31_topp_renorm_det`` through
``ops/sampling/m31_topp_det``.

On rows with top_p < 1 it must equal FlashInfer's ``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` bit for
bit (randn / peaked / ties / near_uniform / all_equal rows); rows with top_p >= 1 keep every token. Outside its
contract the override declines (the sampler runs stock FlashInfer)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_topp_det as entry,
)

CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal")
FLT_MIN = torch.finfo(torch.float32).tiny
op = entry.m31_topp_det


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("TOPP_DET is built for sm_107 here")
    assert entry.topp_det_op() is not None, "this build has no minimax_m31_topp_renorm_det"
    pytest.importorskip("flashinfer.sampling")


def _make(cls, rows, d, seed):
    """The microbench row generators: softmax of the class's logits."""
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


@pytest.mark.parametrize("mode", ["p95", "mixed"])
@pytest.mark.parametrize("cls", CLASSES)
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


@pytest.mark.parametrize("cls", CLASSES)
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


def test_outside_the_contract_the_override_declines() -> None:
    """A float top_p, a non-fp32 / misaligned / non-contiguous probs, a d % 4 width, a mis-sized top_p or more rows
    than the workspace holds return None (the sampler runs stock FlashInfer)."""
    _gpu()
    d = 200064
    probs = _make("randn", 4, d, 1)
    top_p = torch.full((4,), 0.9, device="cuda")
    assert entry.top_p_renorm_det(probs, 0.9) is None
    assert entry.top_p_renorm_det(probs.half(), top_p) is None
    assert entry.top_p_renorm_det(probs, top_p.double()) is None
    assert entry.top_p_renorm_det(probs, top_p[:3]) is None
    assert entry.top_p_renorm_det(probs[:, : d - 2], top_p) is None
    assert entry.top_p_renorm_det(probs[:, : d - 2].contiguous(), top_p) is None
    flat = torch.empty(4 * d + 1, device="cuda")
    assert entry.top_p_renorm_det(flat[1:].view(4, d), top_p) is None
    wide = _make("randn", entry.MAX_ROWS + 1, d, 2)
    assert (
        entry.top_p_renorm_det(wide, torch.full((entry.MAX_ROWS + 1,), 0.9, device="cuda")) is None
    )
    assert entry.ctas_per_row(1, 148) == 16 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1

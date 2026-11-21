# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_ADD_RMSNORM (family ``minimax_m3/ops/add_rmsnorm``) on sm_107: each form of
``trtllm::minimax_m31_add_rmsnorm`` writes the served kernel's bytes (input, residual, MXFP8 rows, swizzled scales
with the zeroed padding rows, replicas): the CuTe DSL ``add_rmsnorm_mxfp8`` / ``add_rmsnorm_replicated`` (2 and 3
copies) / ``add_rmsnorm_replicated_mxfp8`` and FlashInfer's ``fused_add_rmsnorm`` (plain, also through the catalog
op), at row counts on both sides of a launch-shape boundary and of a 128-row scale tile, with zero, huge, tiny and
signed-zero rows and weights, and inside a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import add_rmsnorm as native
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the native add + RMSNorm op runs on sm_107 (Rubin) only",
)

H, EPS = 6144, 1e-6
ROWS = (1, 93, 130)
PLAIN_ROWS = (1, 93, 107, 130)
FORMS = {"mxfp8": (True, 0), "rep2": (False, 2), "rep3": (False, 3), "rep2_mxfp8": (True, 2)}


def _cute():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe import (
        add_rmsnorm_mxfp8 as cute,  # noqa: E501
    )

    return cute


def _scale_bytes(m):
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.mxfp8_rows import (
        scale_bytes,
    )

    return scale_bytes(m, H)


def _rows(m, gen):
    x = torch.randn(m, H, device="cuda", generator=gen) * 1.5
    r = torch.randn(m, H, device="cuda", generator=gen) * 3.0
    if m > 3:
        x[1].zero_()
        r[1].zero_()
        x[2] *= 1e4
        r[3] *= 1e-30
        x[3] *= 1e-30
        x[0, ::97] = 0.0
        r[0, 5::101] = -0.0
    return x.to(torch.bfloat16), r.to(torch.bfloat16)


def _weight(gen):
    w = torch.randn(H, device="cuda", generator=gen) * 0.3 + 1.0
    w[32:64] = 0.0
    w[100] = -0.0
    w[200:208] = 300.0
    return w.to(torch.bfloat16)


def _bufs(m, form):
    mx, n = FORMS[form]
    q = torch.zeros(m, H, dtype=torch.float8_e4m3fn, device="cuda") if mx else None
    sf = torch.full((_scale_bytes(m) + 64,), 0xA5, dtype=torch.uint8, device="cuda") if mx else None
    rep = torch.zeros(m, n * H, dtype=torch.bfloat16, device="cuda") if n else None
    return q, sf, rep


def _run(mod, form, x, r, w, q, sf, rep, pdl):
    if form == "mxfp8":
        mod.add_rmsnorm_mxfp8(x, r, w, EPS, q, sf, enable_pdl=pdl)
    elif form in ("rep2", "rep3"):
        mod.add_rmsnorm_replicated(x, r, w, EPS, rep, enable_pdl=pdl)
    else:
        mod.add_rmsnorm_replicated_mxfp8(x, r, w, EPS, rep, q, sf, enable_pdl=pdl)


def _same(a, b):
    return (
        a is None
        and b is None
        or torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


@pytest.mark.parametrize("form, pdl", [(form, True) for form in FORMS])
@pytest.mark.parametrize("rows", ROWS)
def test_each_form_writes_the_served_kernels_bytes(form, rows, pdl):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    w = _weight(gen)
    x, r = _rows(rows, gen)
    assert native.takes(x, r, w)
    outs = []
    for mod in (_cute(), native):
        xx, rr = x.clone(), r.clone()
        q, sf, rep = _bufs(rows, form)
        _run(mod, form, xx, rr, w, q, sf, rep, pdl)
        outs.append((xx, rr, q, sf, rep))
    for name, want, got in zip(("input", "residual", "q", "sf", "replicas"), *outs):
        assert _same(want, got), name


@pytest.mark.parametrize("rows", PLAIN_ROWS)
def test_the_plain_form_is_flashinfer_bit_for_bit(rows):
    gen = torch.Generator(device="cuda").manual_seed(500 + rows)
    w = _weight(gen)
    x, r = _rows(rows, gen)
    want_x, want_r = x.clone(), r.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(want_x, want_r, w, EPS)
    got_x, got_r = x.clone(), r.clone()
    assert native.fused_add_rmsnorm(got_x, got_r, w, EPS)
    assert _same(got_x, want_x) and _same(got_r, want_r)
    cat_x, cat_r = x.clone(), r.clone()
    flashinfer_fused_add_rmsnorm(cat_x, cat_r, w, EPS)  # the catalog op routes 6144-wide rows here
    assert _same(cat_x, want_x) and _same(cat_r, want_r)


def test_a_cuda_graph_replays_the_served_bytes():
    gen = torch.Generator(device="cuda").manual_seed(7)
    w = _weight(gen)
    x, r = _rows(96, gen)
    q, sf, rep = _bufs(96, "rep2_mxfp8")
    sx, sr = x.clone(), r.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run(native, "rep2_mxfp8", sx.clone(), sr.clone(), w, q, sf, rep, True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(native, "rep2_mxfp8", sx, sr, w, q, sf, rep, True)
    x2, r2 = _rows(96, gen)
    sx.copy_(x2)
    sr.copy_(r2)
    graph.replay()
    torch.cuda.synchronize()
    wq, wsf, wrep = _bufs(96, "rep2_mxfp8")
    wx, wr = x2.clone(), r2.clone()
    _run(_cute(), "rep2_mxfp8", wx, wr, w, wq, wsf, wrep, True)
    for name, want, got in zip(
        ("input", "residual", "q", "sf", "replicas"), (wx, wr, wq, wsf, wrep), (sx, sr, q, sf, rep)
    ):
        assert _same(want, got), name

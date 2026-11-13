# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_ADD_RMSNORM (family ``minimax_m3/ops/add_rmsnorm``) on sm_107: each form of
``trtllm::minimax_m31_add_rmsnorm`` writes the served kernel's bytes (input, residual, MXFP8 rows, swizzled scales
with the zeroed padding rows, replicas): the CuTe DSL ``add_rmsnorm_mxfp8`` / ``add_rmsnorm_replicated`` (2 and 3
copies) / ``add_rmsnorm_replicated_mxfp8`` and FlashInfer's ``fused_add_rmsnorm`` (plain, also through the catalog
op), at row counts on both sides of each launch-shape boundary (92 / 93, 106 / 107) and of a 128-row scale tile,
with zero, huge, tiny and signed-zero rows and weights, PDL on and off, and inside a CUDA graph; and the DEP target's
plain-norm install (``install_plain`` with tail.py's FlashInfer check) leaves RMSNorm.forward's results bit for bit
FlashInfer's at the DEP row counts, skips norms another install replaced or tail.py declines, and comes off cleanly;
and the NORM_L2PF prefetch of the consumer GEMM's weight + scales (forced at every launch shape, by the band
rule, and inside a CUDA graph) changes no output byte."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    add_rmsnorm as native,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the native add + RMSNorm op runs on sm_107 (Rubin) only",
)

H, EPS = 6144, 1e-6
ROWS = (1, 2, 50, 84, 92, 93, 96, 106, 107, 128, 130)
FORMS = {"mxfp8": (True, 0), "rep2": (False, 2), "rep3": (False, 3), "rep2_mxfp8": (True, 2)}


def _cute():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe import (
        add_rmsnorm_mxfp8 as cute,  # noqa: E501
    )

    return cute


def _scale_bytes(m):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.mxfp8_rows import (
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


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("pdl", [True, False])
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


@pytest.mark.parametrize("rows", ROWS)
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


def test_other_operands_take_the_served_kernels():
    gen = torch.Generator(device="cuda").manual_seed(9)
    w = _weight(gen)
    x, r = _rows(8, gen)
    assert not native.takes(
        x[:, :4096].contiguous(), r[:, :4096].contiguous(), w[:4096].contiguous()
    )
    assert not native.takes(x.float(), r.float(), w)
    assert not native.takes(x.t().contiguous().t(), r, w)
    narrow = [t[:, :4096].contiguous() for t in (x, r)]
    want, got = [t.clone() for t in narrow], [t.clone() for t in narrow]
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(*want, w[:4096].contiguous(), EPS)
    assert not native.fused_add_rmsnorm(*got, w[:4096].contiguous(), EPS)
    flashinfer_fused_add_rmsnorm(*got, w[:4096].contiguous(), EPS)
    assert _same(got[0], want[0]) and _same(got[1], want[1])


def _norm(gen, **kwargs):
    norm = RMSNorm(hidden_size=H, eps=EPS, dtype=torch.bfloat16, device="cuda", **kwargs)
    with torch.no_grad():
        norm.weight.copy_(_weight(gen))
    return norm


@pytest.mark.parametrize("rows", ROWS)
def test_the_dep_plain_norms_are_flashinfer_bit_for_bit(rows):
    gen = torch.Generator(device="cuda").manual_seed(700 + rows)
    norm, stock = _norm(gen), RMSNorm(hidden_size=H, eps=EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        stock.weight.copy_(norm.weight)
    assert native.install_plain([norm], tail.flashinfer_norm_decline_reason) == 1
    assert native.is_plain_site(norm) and not native.is_plain_site(stock)
    x, r = _rows(rows, gen)
    got = norm(x.clone(), r.clone())
    want = stock(x.clone(), r.clone())
    assert _same(got[0], want[0]) and _same(got[1], want[1])
    assert _same(norm(x.clone()), stock(x.clone()))  # no residual: the stock FlashInfer rmsnorm


def test_the_dep_install_skips_replaced_and_declined_norms_and_comes_off():
    gen = torch.Generator(device="cuda").manual_seed(3)
    replaced, gemma, plain = _norm(gen), _norm(gen, use_gemma=True), _norm(gen)
    narrow = RMSNorm(hidden_size=4096, eps=EPS, dtype=torch.bfloat16, device="cuda")
    replaced.forward = lambda *args, **kwargs: None
    assert tail.flashinfer_norm_decline_reason(gemma) is not None
    mods = [replaced, gemma, narrow, plain, torch.nn.Linear(2, 2)]
    assert native.install_plain(mods, tail.flashinfer_norm_decline_reason) == 1
    assert (
        native.is_plain_site(plain)
        and not native.is_plain_site(gemma)
        and not native.is_plain_site(narrow)
    )
    assert native.install_plain(mods, tail.flashinfer_norm_decline_reason) == 0  # once per instance
    native.uninstall_plain(mods)
    assert "forward" not in plain.__dict__ and "forward" in replaced.__dict__


# --- NORM_L2PF: the L2 prefetch of the consumer GEMM's weight never changes a byte ----------------------------------

L2PF_ROWS = (1, 64, 65, 88, 128, 129, 384)


def _qkv_weight():
    """A qkv_proj-sized MXFP8 weight [9856, 6144] and its swizzled scale bytes (the ranges the DEP site prefetches)."""
    return (
        torch.zeros(9856, H, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(9856 * (H // 32), dtype=torch.uint8, device="cuda"),
    )


def _mxfp8_call(x, r, w, pdl, prefetch):
    q, sf, _ = _bufs(x.shape[0], "mxfp8")
    native.add_rmsnorm_mxfp8(x, r, w, EPS, q, sf, enable_pdl=pdl, prefetch=prefetch)
    return q, sf


l2pf = pytest.mark.skipif(
    torch.cuda.is_available()
    and native.add_rmsnorm_op() is not None
    and not native.takes_prefetch(),
    reason="the op has no prefetch arguments",
)


@l2pf
@pytest.mark.parametrize("rows", L2PF_ROWS)
@pytest.mark.parametrize("fraction", [4, 1])
@pytest.mark.parametrize("pdl", [True, False])
def test_the_l2_prefetch_leaves_every_output_byte_unchanged(rows, fraction, pdl):
    """Forced at every launch shape (4- / 2-CTA clusters, one CTA), a quarter or all of both ranges."""
    gen = torch.Generator(device="cuda").manual_seed(900 + rows)
    w = _weight(gen)
    x, r = _rows(rows, gen)
    weight, scale = _qkv_weight()
    prefetch = tuple((t, (t.numel() * t.element_size() // fraction) & ~15) for t in (weight, scale))
    outs = []
    for pf in (None, prefetch):
        xx, rr = x.clone(), r.clone()
        q, sf = _mxfp8_call(xx, rr, w, pdl, pf)
        outs.append((xx, rr, q, sf))
    torch.cuda.synchronize()
    for name, want, got in zip(("input", "residual", "q", "sf"), *outs):
        assert _same(want, got), name


@l2pf
@pytest.mark.parametrize("rows", L2PF_ROWS)
def test_the_band_rule_feeds_the_op_and_changes_no_byte(rows):
    gen = torch.Generator(device="cuda").manual_seed(950 + rows)
    w = _weight(gen)
    x, r = _rows(rows, gen)
    weight, scale = _qkv_weight()
    prefetch = native.l2pf_ranges(rows, weight, scale)
    assert (prefetch is not None) == (65 <= rows <= 128)
    outs = []
    for pf in (None, prefetch):
        xx, rr = x.clone(), r.clone()
        outs.append((xx, rr, *_mxfp8_call(xx, rr, w, True, pf)))
    torch.cuda.synchronize()
    for name, want, got in zip(("input", "residual", "q", "sf"), *outs):
        assert _same(want, got), name


@l2pf
def test_a_cuda_graph_with_the_prefetch_replays_the_eager_bytes():
    gen = torch.Generator(device="cuda").manual_seed(88)
    w = _weight(gen)
    x, r = _rows(88, gen)
    weight, scale = _qkv_weight()
    prefetch = native.l2pf_ranges(88, weight, scale)
    q, sf, _ = _bufs(88, "mxfp8")
    sx, sr = x.clone(), r.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        native.add_rmsnorm_mxfp8(
            sx.clone(), sr.clone(), w, EPS, q, sf, enable_pdl=True, prefetch=prefetch
        )
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.add_rmsnorm_mxfp8(sx, sr, w, EPS, q, sf, enable_pdl=True, prefetch=prefetch)
    x2, r2 = _rows(88, gen)
    sx.copy_(x2)
    sr.copy_(r2)
    graph.replay()
    torch.cuda.synchronize()
    wx, wr = x2.clone(), r2.clone()
    wq, wsf = _mxfp8_call(wx, wr, w, True, None)
    torch.cuda.synchronize()
    for name, want, got in zip(
        ("input", "residual", "q", "sf"), (wx, wr, wq, wsf), (sx, sr, q, sf)
    ):
        assert _same(want, got), name


@l2pf
def test_the_op_rejects_a_range_past_its_tensor():
    gen = torch.Generator(device="cuda").manual_seed(3)
    w = _weight(gen)
    x, r = _rows(88, gen)
    weight, scale = _qkv_weight()
    q, sf, _ = _bufs(88, "mxfp8")
    too_many = weight.numel() + 16
    with pytest.raises(RuntimeError, match="prefetch0"):
        native.add_rmsnorm_op()(x, r, w, EPS, q, sf, None, True, weight, too_many, scale, 16)
    with pytest.raises(RuntimeError, match="prefetch1_bytes must be >= 0"):
        native.add_rmsnorm_op()(x, r, w, EPS, q, sf, None, True, weight, 16, scale, -16)

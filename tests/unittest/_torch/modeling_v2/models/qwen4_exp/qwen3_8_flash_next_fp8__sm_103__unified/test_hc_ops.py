# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's hyper-connection entries and their sites' paths."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_rstd import (  # noqa: E501
    qwen4_exp_hc_inject_rstd as rstd_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.hc.qwen4_exp_hc_decode_rows_tc__decode import (  # noqa: E501
    qwen4_exp_hc_decode_rows_first as first_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.hc.qwen4_exp_hc_decode_rows_tc__decode import (  # noqa: E501
    qwen4_exp_hc_decode_rows_mix as mix_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.hc.qwen4_exp_hc_decode_tc_first__decode import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first as op,
)

__extra_import_path__ = ["../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "the hyper-connection entries require a CUDA device"

DEV = "cuda"
HC, HIDDEN, RANK, FIRST_ROWS = 4, 2560, 320, 384
WIDTH = HC * HIDDEN
EPS = 1e-6
CORE = SimpleNamespace(hc_count=HC, hc_lowrank=RANK, hc_first_cols=352, eps=EPS)


def _cell_seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % (2**31)


def _ratio(out, ref) -> float:
    """Worst error in bands of one ulp plus `eps / 16` of the max (SiLU cancels near zero)."""
    eps = torch.finfo(ref.dtype).eps
    r = ref.double()
    return ((out.double() - r).abs() / (eps * r.abs() + eps / 16 * r.abs().max())).max().item()


def _site(rows, inject, seed):
    """One site's normed rows and weights at the checkpoint's statistics."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    parts = [torch.randn(RANK, WIDTH, generator=g, device=DEV) * 0.0296]
    if inject:
        parts.append(torch.randn(HC, WIDTH, generator=g, device=DEV) * 0.0107)
    w = torch.cat(parts)
    first = torch.cat([w, torch.zeros(FIRST_ROWS - w.shape[0], WIDTH, device=DEV)])
    up = torch.randn(WIDTH, RANK, generator=g, device=DEV) * 0.05
    gamma = torch.randn(WIDTH, generator=g, device=DEV) * 0.477
    x = torch.randn(rows, HC, HIDDEN, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.flatten(-2) * (1.0 + gamma)).to(torch.bfloat16)
    return normed, first.to(torch.bfloat16).t(), up.to(torch.bfloat16).t()


def _check_tc_first(cell):
    rows, inject = cell.spec["rows"], cell.spec.get("inject", True)
    normed, first, _ = _site(rows, inject, _cell_seed(cell))
    args = (normed, first, RANK, HC, inject)
    low, inj = op(*args)
    assert low.shape == (rows, RANK)
    for got, want in zip((low, inj), op.reference(*args)):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _ratio(got, want) <= 1.0, f"{_ratio(got, want):.4g}x the band"


def _mixer(monkeypatch, cap, normed, first, up, inject):
    calls = []
    entry = M.qwen4_exp_hc_decode_tc_first

    def counted(*args, **kwargs):
        calls.append(args[0].shape[0])
        return entry(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(M, "_HC_TC_FIRST_MAX_ROWS", cap)
        m.setattr(M, "qwen4_exp_hc_decode_tc_first", counted)
        mixed, inj = M._hc_mixer(CORE, normed, first, up, inject)
    return mixed, inj, calls


def _mixer_first_halves(rows, inject, monkeypatch):
    """`_hc_mixer`'s entry against its cuBLAS path: bf16 roundings of fp32 sums in other orders."""
    normed, first, up = _site(rows, inject, seed=100 + rows)
    tc = _mixer(monkeypatch, 640, normed, first, up, inject)
    cb = _mixer(monkeypatch, 0, normed, first, up, inject)
    assert tc[2] == [rows] and cb[2] == []
    assert_within_ulp(tc[0], cb[0], element_ulp=2.0, rms_ulp=0.25)
    if inject:
        assert_within_ulp(tc[1], cb[1], element_ulp=2.0, rms_ulp=1.0)
    else:
        assert tc[1] is None and cb[1] is None


def _operands(rows, seed):
    """One site in the forward's layouts; `gate` is column 1280 of a [rows, 1288] gate|up row."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    bf = torch.bfloat16

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(bf)

    w = torch.cat(
        [
            torch.randn(RANK, WIDTH, generator=g, device=DEV) * 0.0296,
            torch.randn(HC, WIDTH, generator=g, device=DEV) * 0.0107,
            torch.zeros(FIRST_ROWS - RANK - HC, WIDTH, device=DEV),
        ]
    )
    return dict(
        first=w.to(bf).t(),
        up=rnd(WIDTH, RANK, scale=0.05).t(),
        norm=rnd(WIDTH, scale=0.477),
        hc=rnd(rows, WIDTH, scale=2.0),
        block=rnd(rows, HIDDEN, scale=0.5),
        shared=rnd(rows, HIDDEN, scale=0.5),
        gate=rnd(rows, 1288)[:, 1280:1281],
        inject=(2.0 * torch.sigmoid(torch.randn(rows, HC, generator=g, device=DEV))).to(bf),
    )


def _site_inputs(o, kind):
    blk, inj = (None, None) if kind == "norm" else (o["block"], o["inject"])
    sh, gate = (o["shared"], o["gate"]) if kind == "combine" else (None, None)
    return blk, inj, sh, gate


def _first_args(o, site, cols):
    blk, inj, sh, gate = _site_inputs(o, site)
    return (o["hc"], blk, inj, o["norm"], o["first"], HC, cols, EPS, sh, gate)


def _check_rows_first(cell):
    rows, site, cols = cell.spec["rows"], cell.spec["site"], cell.spec["cols"]
    args = _first_args(_operands(rows, _cell_seed(cell)), site, cols)
    hc_out, low, inj, rstd = first_op(*args)
    ref = first_op.reference(*args)
    assert low.shape == (rows, RANK) and rstd.shape == (rows, HC) and rstd.dtype == torch.float32
    assert (inj is None) == (cols == RANK)
    if site == "norm":
        assert hc_out is args[0], "without a pending block the stream is the input"
    assert _ratio(hc_out, ref[0]) <= 1.0, "hc_out"
    assert _ratio(low, ref[1]) <= 1.0, f"low {_ratio(low, ref[1]):.3g}x the band"
    if inj is not None:
        assert _ratio(inj, ref[2]) <= 1.0, f"inject {_ratio(inj, ref[2]):.3g}x the band"
    assert ((rstd - ref[3]).abs() / ref[3]).max() <= 1e-6, "rstd in fp32"
    # The residual every later site reads: bitwise what `qwen4_exp_hc_inject_norm` writes.
    want, _ = qwen4_exp_hc_inject_norm(*args[:4], EPS, HC, *args[8:])
    assert torch.equal(hc_out, want)


def _check_rows_mix(cell):
    rows = cell.spec["rows"]
    o = _operands(rows, _cell_seed(cell))
    with validating(first_op):
        hc_out, low, _, rstd = first_op(*_first_args(o, "combine", RANK + HC))
    args = (low, rstd, o["up"], hc_out, o["norm"], HC)
    mixed = mix_op(*args)
    assert mixed.shape == (rows, HIDDEN) and mixed.is_contiguous()
    assert _ratio(mixed, mix_op.reference(*args)) <= 1.0


def _site_run(monkeypatch, lo, o, kind):
    """`_hc_site` with the two-launch range starting at `lo`; counts its first launches."""
    calls = []
    entry = M.qwen4_exp_hc_decode_rows_first

    def counted(*args, **kwargs):
        calls.append(args[0].shape[0])
        return entry(*args, **kwargs)

    blk, inj, sh, gate = _site_inputs(o, kind)
    extra = (sh, gate) if kind == "combine" else ()
    with monkeypatch.context() as m:
        m.setattr(M, "_HC_ROWS_TC_MIN_ROWS", lo)
        m.setattr(M, "qwen4_exp_hc_decode_rows_first", counted)
        hc, mixed, inject = M._hc_site(
            CORE, o["hc"], blk, inj, o["norm"], o["first"], o["up"], kind != "norm", *extra
        )
    return hc, mixed, inject, calls


def _rows_tc_sites(rows, kind, monkeypatch):
    """`_hc_site`'s two launches against `qwen4_exp_hc_inject_norm` + `_hc_mixer`: the residual
    bitwise, `mixed` within rounding since the grouped norm reads other bf16 operands."""
    o = _operands(rows, 100 + rows)
    two = _site_run(monkeypatch, 1, o, kind)
    old = _site_run(monkeypatch, 99, o, kind)
    assert two[3] == [rows] and old[3] == []
    assert torch.equal(two[0], old[0])
    assert_within_ulp(two[1], old[1], element_ulp=3.0, rms_ulp=1.0)
    if kind == "norm":
        assert two[2] is None and old[2] is None
    else:
        a, b = two[2].double(), old[2].double()
        step = torch.finfo(torch.bfloat16).eps * torch.maximum(a.abs(), b.abs())
        assert ((a - b).abs() <= step).all(), (a - b).abs().max().item()


def _mixer_rstd(rows, inject):
    """The prefill rstd site (normed stream formed on chip in fp32) against `_hc_mixer` on the
    stored bf16 stream; the planted |1 + gamma| = 7.6 column sets the 4-ulp element bound."""
    o = _operands(rows, 900 + rows)
    o["norm"][0], o["norm"][WIDTH // 2] = 6.625, -6.625  # this checkpoint's largest |gamma|
    hc, rstd = rstd_op(o["hc"], o["block"], o["inject"], EPS, HC)
    hc2, normed = qwen4_exp_hc_inject_norm(o["hc"], o["block"], o["inject"], o["norm"], EPS, HC)
    assert torch.equal(hc, hc2)
    w1 = {o["norm"].data_ptr(): (1.0 + o["norm"].float()).contiguous()}
    core = SimpleNamespace(**vars(CORE), _hc_norm_w1=w1)
    a = M._hc_mixer_rstd(core, hc, rstd, o["norm"], o["first"], o["up"], inject)
    b = M._hc_mixer(core, normed, o["first"], o["up"], inject)
    assert_within_ulp(a[0], b[0], element_ulp=4.0, rms_ulp=1.0)
    if inject:
        assert_within_ulp(a[1], b[1], element_ulp=2.0, rms_ulp=1.0)
    else:
        assert a[1] is None and b[1] is None


def _tp_partial():
    """`_tp_partial` defers the sum only where the next site's fused all-reduce can take it;
    `_hc_site` sums a partial it cannot fuse with `tp_sum` first, bitwise the summed call."""
    calls = []

    def tp_sum(x):
        calls.append(x.shape[0])
        return (x.float() * 0.75).to(x.dtype)

    lo = M._HC_AR_INJECT_MIN_ROWS
    x_lo = torch.zeros(lo, HIDDEN, dtype=torch.bfloat16, device=DEV)
    x16 = torch.zeros(16, HIDDEN, dtype=torch.bfloat16, device=DEV)
    off = SimpleNamespace(tp_sum=tp_sum)
    on = SimpleNamespace(hc_ar_inject=object(), tp_sum=tp_sum)
    assert not isinstance(M._tp_partial(off, x_lo), M._TpPartial)
    assert isinstance(M._tp_partial(on, x_lo), M._TpPartial)
    assert not isinstance(M._tp_partial(on, x16), M._TpPartial)
    assert calls == [lo, 16]
    o = _operands(lo, 77)
    core = SimpleNamespace(**vars(CORE), tp_sum=tp_sum)
    for sh in ((o["shared"], o["gate"]), ()):
        rest = (o["norm"], o["first"], o["up"], True, *sh)
        a = M._hc_site(core, o["hc"], M._TpPartial(o["block"]), o["inject"], *rest)
        b = M._hc_site(core, o["hc"], tp_sum(o["block"]), o["inject"], *rest)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def _inject_norm_launch(rows, kind):
    """`_hc_inject_norm_launch` against the family op: residual bitwise, normed stream within one
    rounding at the site's warps; all bitwise at eight, eager and replayed behind a producer."""

    def inputs(seed):
        o = _operands(rows, seed)
        blk, inj, sh, gate = _site_inputs(o, kind)
        return o, (o["hc"], blk, inj, o["norm"], EPS, HC, sh, gate)

    _, args = inputs(500 + rows)
    hc8, normed8 = qwen4_exp_hc_inject_norm(*args)
    for warps in (M._HC_INJECT_WARPS, 2):
        hc_w, normed_w = M._hc_inject_norm_launch(*args, warps)
        assert torch.equal(hc_w, hc8)
        assert_within_ulp(normed_w, normed8, element_ulp=1.0, rms_ulp=0.25)
    o, args = inputs(700 + rows)
    hc8, normed8 = qwen4_exp_hc_inject_norm(*args)
    hc_l, normed_l = M._hc_inject_norm_launch(*args, 8)
    assert torch.equal(hc_l, hc8) and torch.equal(normed_l, normed8)
    src = o["hc"].clone()
    hc_in = src.clone()
    M._hc_inject_norm_launch(hc_in, *args[1:], 8)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hc_in.copy_(src * 1.5)
        hc_g, normed_g = M._hc_inject_norm_launch(hc_in, *args[1:], 8)
    graph.replay()
    torch.cuda.synchronize()
    hc_e, normed_e = M._hc_inject_norm_launch(src * 1.5, *args[1:], 8)
    torch.cuda.synchronize()
    assert torch.equal(hc_g, hc_e) and torch.equal(normed_g, normed_e)


_CELL_CHECKS = ((op, _check_tc_first), (first_op, _check_rows_first), (mix_op, _check_rows_mix))
CASES = [(entry, check, cell) for entry, check in _CELL_CHECKS for cell in entry.CELLS]


@pytest.mark.parametrize(
    "entry, check, cell", CASES, ids=[f"{type(e).__name__}-{c.why[:40]}" for e, _, c in CASES]
)
def test_certified_cells(entry, check, cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(entry)
    check(cell)


_KINDS = ("combine", "inject", "norm")
#: `(path, rows, variant)`: `inject` for the mixers, the site kind for the decode sites.
# fmt: off
_SITE_CASES = (
    [("mixer_first_halves", r, i) for r in (4, 64, 288, 576) for i in (True, False)]
    + [("rows_tc_sites", r, k) for r in (1, 8, 16) for k in _KINDS]
    + [("mixer_rstd", r, i) for r in (3072, 8192) for i in (True, False)]
    + [("tp_partial", 0, None)]
    + [("inject_norm_launch", r, k) for r in (17, 192, 288) for k in _KINDS]
)
# fmt: on


@pytest.mark.parametrize("path, rows, variant", _SITE_CASES, ids=lambda v: str(v))
def test_hc_site_paths(path, rows, variant, monkeypatch) -> None:
    """Site paths no certified cell reaches, each against its alternative."""
    if path == "mixer_first_halves":
        _mixer_first_halves(rows, variant, monkeypatch)
    elif path == "rows_tc_sites":
        _rows_tc_sites(rows, variant, monkeypatch)
    elif path == "mixer_rstd":
        _mixer_rstd(rows, variant)
    elif path == "tp_partial":
        _tp_partial()
    else:
        _inject_norm_launch(rows, variant)

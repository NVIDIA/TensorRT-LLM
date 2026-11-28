# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The site-B fold into the output projection (`DP_OUT_PROJ_INJECT_ENV`, with
`DP_SITEB_PLAN32_ENV`; opt-in, CPU): the values, the first projection's plan arguments (shared
with `_hc_first`), the build (rows served, PLAN32, every decline), the arming of one output
projection by the trunk loop and the site behind a folded projection.

The kernels' numerics are covered by the ops' GPU tests: `mid_mm_inject` in
`ops/gemm/test_modeling_v2_mid_rows_mm.py`, the deferred-rstd first projection in
`hc/test_modeling_v2_qwen4_exp_hc_decode_tc_first__decode.py` and the up kernel's `rstd=` in
`ops/hc/test_modeling_v2_qwen4_exp_hc_up_gate_mean.py`. The build's own check of one fold
against the stock site on a rank's weights runs on a GPU only (`test_selfcheck_on_gpu`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, want",
    [(None, None), ("0", None), ("1", (17, 80)), ("all", (1, 144)), ("5:7", (20, 28))],
)
def test_wanted(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(M.DP_OUT_PROJ_INJECT_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_OUT_PROJ_INJECT_ENV, value)
    assert M._siteb_wanted(4) == want


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["2", "7:5", "a:b", "gdn"])
def test_wanted_rejects(monkeypatch, value):
    monkeypatch.setenv(M.DP_OUT_PROJ_INJECT_ENV, value)
    with pytest.raises(ValueError, match=M.DP_OUT_PROJ_INJECT_ENV):
        M._siteb_wanted(4)


def _hc_core(tiles=True, x2=False, lowrows=None):
    core = SimpleNamespace(hc_tiles=tiles, hc_first_x2=x2)
    if lowrows is not None:
        core.hc_rowsn_lo = lowrows
    return core


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "core, rows, inject, form",
    [
        (_hc_core(False), 48, True, {}),
        (_hc_core(), 4, True, {}),
        (_hc_core(), 20, True, {"tm": 64}),
        (_hc_core(), 64, False, {"tm": 64}),
        (_hc_core(), 80, True, {"rows_on_n": True}),
        (_hc_core(), 80, False, {}),
        (_hc_core(x2=True), 80, True, {"tm64x2": True}),
        (_hc_core(lowrows=8), 12, True, {"rows_on_n": True}),
        (_hc_core(x2=True, lowrows=8), 12, True, {"rows_on_n": True}),
        (_hc_core(), 112, True, {}),
    ],
)
def test_first_form(monkeypatch, core, rows, inject, form):
    """`_hc_first`'s plan arguments: what the site B fold compiles its first projection for."""
    assert M._hc_first_form(core, rows, inject) == form
    calls = []
    monkeypatch.setattr(M, "qwen4_exp_hc_decode_tc_first", lambda *a, **k: calls.append((a[4], k)))
    core.hc_lowrank, core.hc_count = 320, 4
    M._hc_first(core, torch.zeros(rows, 8), None, inject)
    assert calls == [(inject, form)]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows, release, tm", [(20, False, 64), (64, True, 64), (80, False, 64), (80, True, 128)]
)
def test_up_tile(rows, release, tm):
    assert M._hc_up_tile(_hc_core(), rows, release) == tm
    assert M._hc_up_tile(_hc_core(False), rows, release) == 128


P12 = M.MidPlan(12, 2, 8, 2)
P6 = M.MidPlan(6, 4, 10, 1)


def _core(gdn_rows=None, attn_rows=(), **over):
    gdn_rows = gdn_rows or {12: (32, P12), 20: (32, P12), 28: (32, P12), 48: (48, P6), 96: (96, P6)}
    layer = SimpleNamespace(mhc_first=torch.zeros(10240, 384), mhc_norm=torch.zeros(10240))
    core = SimpleNamespace(
        max_verify_rows=4,
        _out_proj={"gdn_out": dict(gdn_rows), "attn_o": {r: gdn_rows[r] for r in attn_rows}},
        hc_count=4,
        stream=10240,
        hidden=2560,
        hc_lowrank=320,
        eps=1e-6,
        hc_tiles=True,
        hc_first_x2=False,
        ple_layer=3,
        gdn_layers=[0, 1, 2],
        w={f"l{i}_out_w": torch.zeros(2560, 6144) for i in (0, 1, 2)},
        _decode=SimpleNamespace(_layers=[layer] * 4),
        dp_rank=0,
        dp_size=8,
    )
    for k, v in over.items():
        setattr(core, k, v)
    return core


def _build(monkeypatch, value, core, plan32=False, check=None, fail_rows=()):
    monkeypatch.setenv(M.DP_OUT_PROJ_INJECT_ENV, value)
    if plan32:
        monkeypatch.setenv(M.DP_SITEB_PLAN32_ENV, "1")
    else:
        monkeypatch.delenv(M.DP_SITEB_PLAN32_ENV, raising=False)
    compiled = []

    def inject(x, w, plan, hc, inj, norm, pdl=True, mpad=None, compile_only=False):
        if x.shape[0] in fail_rows:
            raise RuntimeError("no compile")
        compiled.append(("gemm", x.shape[0], mpad, plan))

    monkeypatch.setattr(M, "mid_mm_inject", inject)
    monkeypatch.setattr(
        M, "hc_tc_first_ss", lambda y, *a, **k: compiled.append(("first", y.shape[0], a[-1]))
    )
    monkeypatch.setattr(
        M, "hc_up_precompile", lambda *a, ss=False: compiled.append(("up", a[-1], ss))
    )
    checked = []
    monkeypatch.setattr(
        M, "_siteb_selfcheck", lambda core, st, L, rows, device: checked.append(rows) or check
    )
    return M._siteb_build(core, torch.device("cpu")), compiled, checked


@pytest.mark.cpu_only
def test_build_serves_the_values_rows(monkeypatch):
    st, compiled, checked = _build(monkeypatch, "1", _core(attn_rows=(48, 96)))
    # `1` = 17..80 rows: 20 / 28 / 48 of the table (12 and 96 are outside).
    assert sorted(st.gemm) == [20, 28, 48]
    assert st.gemm[20] == (32, P12) and st.gemm[48] == (48, P6)
    assert st.attn_rows == frozenset({48}) and st.skip == 2 and st.plan32 == ()
    assert st.first[20] == {"tm": 64} and st.first[48] == {"tm": 64}
    assert st.up[20] == 64
    # One check a GEMM plan, at its smallest row count.
    assert sorted(checked) == [20, 48]
    assert ("up", [(False, 64)], True) in compiled


@pytest.mark.cpu_only
def test_build_plan32_moves_the_12_way_rows(monkeypatch):
    st, compiled, checked = _build(monkeypatch, "all", _core(), plan32=True)
    assert sorted(st.gemm) == [12, 20, 28, 48, 96]
    assert st.plan32 == (12, 20, 28)
    assert all(st.gemm[r] == (32, P6) for r in (12, 20, 28))
    assert st.gemm[96] == (96, P6)
    # 12 rows: the rows-on-N first projection only with the low-rows floor; here a 64-row tile.
    assert st.first[12] == {"tm": 64} and st.first[96] == {"rows_on_n": True}
    assert sorted(checked) == [12, 48, 96]


@pytest.mark.cpu_only
def test_build_drops_a_row_count_that_does_not_compile(monkeypatch):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    st, _, _ = _build(monkeypatch, "1", _core(), fail_rows=(28,))
    assert sorted(st.gemm) == [20, 48]
    assert len(warned) == 1 and "28 rows" in warned[0]


@pytest.mark.cpu_only
def test_build_skips_rows_the_fold_cannot_take(monkeypatch):
    # 6 rows (not 4 a request) and 4 rows (the first kernel's 10-way K split).
    core = _core(gdn_rows={4: (32, P12), 6: (32, P12), 20: (32, P12)})
    st, _, _ = _build(monkeypatch, "all", core)
    assert sorted(st.gemm) == [20]


@pytest.mark.cpu_only
def test_build_declines(monkeypatch):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert _build(monkeypatch, "1", _core(_out_proj=None))[0] is None
    assert _build(monkeypatch, "1", _core(hc_count=2))[0] is None
    assert _build(monkeypatch, "5:5", _core(gdn_rows={48: (48, P6)}))[0] is None
    assert _build(monkeypatch, "1", _core(), check="the injected residual differs")[0] is None
    assert len(warned) == 4 and all(M.DP_OUT_PROJ_INJECT_ENV in w for w in warned)
    assert "residual differs" in warned[-1]
    warned.clear()
    monkeypatch.delenv(M.DP_OUT_PROJ_INJECT_ENV)
    monkeypatch.setenv(M.DP_SITEB_PLAN32_ENV, "1")
    assert M._siteb_build(_core(), torch.device("cpu")) is None
    assert len(warned) == 1 and M.DP_SITEB_PLAN32_ENV in warned[0]
    monkeypatch.delenv(M.DP_SITEB_PLAN32_ENV)
    warned.clear()
    assert M._siteb_build(_core(), torch.device("cpu")) is None and not warned


@pytest.mark.cpu_only
def test_build_checks_every_launch_shape(monkeypatch):
    # One GEMM plan, two first-projection kinds: under the low-rows floor 12 rows take the
    # rows-on-N kernel and 20 rows the 64-row tile, so both are checked.
    core = _core(gdn_rows={12: (32, P12), 20: (32, P12), 28: (32, P12)}, hc_rowsn_lo=8)
    st, _, checked = _build(monkeypatch, "all", core)
    assert st.first[12] == {"rows_on_n": True} and st.first[20] == {"tm": 64}
    assert sorted(checked) == [12, 20]


@pytest.mark.cpu_only
def test_build_declines_a_check_that_raises(monkeypatch):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    monkeypatch.setenv(M.DP_OUT_PROJ_INJECT_ENV, "1")
    monkeypatch.delenv(M.DP_SITEB_PLAN32_ENV, raising=False)
    monkeypatch.setattr(M, "mid_mm_inject", lambda *a, **k: None)
    monkeypatch.setattr(M, "hc_tc_first_ss", lambda *a, **k: None)
    monkeypatch.setattr(M, "hc_up_precompile", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("bad launch")

    monkeypatch.setattr(M, "_siteb_selfcheck", boom)
    assert M._siteb_build(_core(), torch.device("cpu")) is None
    assert len(warned) == 1 and "raised AssertionError" in warned[0]


def _st(gemm=(20, 48), attn=(48,), skip=2):
    return M._SiteB(
        gemm={r: (32, P12) for r in gemm},
        attn_rows=frozenset(attn),
        first={r: {"tm": 64} for r in gemm},
        up={r: 64 for r in gemm},
        skip=skip,
        plan32=(),
    )


def _arm_core(st):
    return SimpleNamespace(
        _siteb=st, is_attn=[False, True, False, True], stream=16, hc_count=4, _siteb_pending=None
    )


def _ops(rows, bad=None):
    t = {
        "hc": torch.zeros(rows, 16, dtype=torch.bfloat16),
        "inject": torch.zeros(rows, 4, dtype=torch.bfloat16),
        "norm": torch.zeros(16, dtype=torch.bfloat16),
    }
    if bad == "dtype":
        t["hc"] = t["hc"].float()
    if bad == "strided":
        t["inject"] = torch.zeros(rows, 8, dtype=torch.bfloat16)[:, ::2]
    return t


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "layer, rows, bad, site",
    [
        (0, 20, None, "gdn_out"),
        (1, 48, None, "attn_o"),
        (1, 20, None, None),  # the attention site does not take 20 rows
        (2, 20, None, None),  # the layer in front of the PLE layer
        (0, 28, None, None),  # not served
        (0, 20, "dtype", None),
        (0, 20, "strided", None),
    ],
)
def test_arm(layer, rows, bad, site):
    core = _arm_core(_st())
    t = _ops(rows, bad)
    L = SimpleNamespace(mhc_norm=t["norm"])
    armed = M._siteb_arm(core, L, layer, rows, t["hc"], t["inject"])
    assert armed == (site is not None)
    assert (core._siteb_pending is None) == (site is None)
    if site is not None:
        assert core._siteb_pending.site == site and core._siteb_pending.hc is t["hc"]
        M._siteb_disarm(core)
        assert core._siteb_pending is None
    assert not M._siteb_arm(_arm_core(None), L, layer, rows, t["hc"], t["inject"])


@pytest.mark.cpu_only
def test_out_proj_takes_the_armed_fold_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        M,
        "mid_mm_raw",
        lambda x, w, plan, pdl=True, mpad=None: calls.append(("raw", x.shape[0])) or "raw",
    )
    monkeypatch.setattr(
        M,
        "mid_mm_inject",
        lambda x, w, plan, hc, inj, norm, mpad=None: calls.append(("inject", mpad, plan))
        or ("hc", "y", "ss"),
    )
    weight = torch.zeros(2560, 6144, dtype=torch.bfloat16)
    st = _st()
    core = SimpleNamespace(
        _out_proj={"gdn_out": {20: (32, P6)}, "attn_o": {20: (32, P6)}},
        _siteb=st,
        _siteb_pending=None,
    )
    t = _ops(20)
    core._siteb_pending = M._SiteBArm("gdn_out", t["hc"], t["inject"], t["norm"])
    x = torch.zeros(20, 6144)
    # Another site's projection leaves the arm alone.
    assert M._out_proj(core, x, weight.t(), "attn_o", None) == "raw"
    assert core._siteb_pending is not None
    out = M._out_proj(core, x, weight.t(), "gdn_out", None)
    assert type(out) is M._SiteBFolded and out.hc == "hc" and core._siteb_pending is None
    # The fold's own plan (PLAN32 may differ from the table's), not the table's.
    assert calls == [("raw", 20), ("inject", 32, P12)]
    assert M._out_proj(core, x, weight.t(), "gdn_out", None) == "raw"


@pytest.mark.cpu_only
def test_site_behind_a_folded_projection(monkeypatch):
    calls = []
    monkeypatch.setattr(
        M,
        "qwen4_exp_hc_decode_tc_first_ss",
        lambda y, ss, first, rank, c, eps, **kw: calls.append(("first", kw)) or ("low", "inj", "r"),
    )
    monkeypatch.setattr(
        M,
        "qwen4_exp_hc_up_gate_mean",
        lambda low, up, y, c, tm=128, rstd=None: calls.append(("up", low, y, tm, rstd)) or "mixed",
    )
    core = SimpleNamespace(_siteb=_st(), hc_lowrank=320, hc_count=4, eps=1e-6, dp_size=8)
    folded = M._SiteBFolded(torch.zeros(20, 16), "y", "ss")
    L = SimpleNamespace(mhc_first="first", mhc_up="up")
    target = SimpleNamespace(core=core)
    hc, mixed, inject, fp4 = M.DecodeTarget._hc_moe_site(target, None, folded, None, L)
    assert hc is folded.hc and mixed == "mixed" and inject == "inj" and fp4 is None
    assert calls == [("first", {"tm": 64}), ("up", "low", "y", 64, "r")]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103: the fold's kernels",
)
@pytest.mark.parametrize("rows, plan32", [(20, False), (20, True), (48, False), (80, False)])
def test_selfcheck_on_gpu(rows, plan32):
    """The build's check on checkpoint-shaped random weights: the residual bitwise the inject
    kernel's, `mixed` / `inject` within rounding level of the stock site."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(rows)
    stream, rank = 10240, 320
    first = torch.zeros(384, stream, device=dev)
    first[: rank + 4] = torch.randn(rank + 4, stream, generator=g, device=dev) * 0.03
    L = SimpleNamespace(
        out=(torch.randn(2560, 6144, generator=g, device=dev) * 6144**-0.5).to(torch.bfloat16).t(),
        mhc_norm=(torch.randn(stream, generator=g, device=dev) * 0.5).to(torch.bfloat16),
        mhc_first=first.to(torch.bfloat16).t(),
        mhc_up=(torch.randn(stream, rank, generator=g, device=dev) * 0.068).to(torch.bfloat16).t(),
    )
    core = SimpleNamespace(
        hc_tiles=True, hc_first_x2=False, hc_lowrank=rank, hc_count=4, stream=stream, eps=1e-6
    )
    mpad, plan = M._out_proj_shape(rows)
    if plan32 and (mpad, plan) == M._SITEB_PLAN32[0]:
        mpad, plan = M._SITEB_PLAN32[1]
    st = M._SiteB(
        gemm={rows: (mpad, plan)},
        attn_rows=frozenset(),
        first={rows: M._hc_first_form(core, rows, True)},
        up={rows: M._hc_up_tile(core, rows, False)},
        skip=-1,
        plan32=(),
    )
    assert M._siteb_selfcheck(core, st, L, rows, dev) is None

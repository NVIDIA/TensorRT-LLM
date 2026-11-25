# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An attention-DP rank's hyper-connection decode forms (CPU, recorders).

* **Mixer plans** (`DP_HC_TILES_ENV`, `_hc_first` / `_hc_mixer`): the 64-row tile at 5..64 rows,
  the rows-on-N first kernel at 65..96 rows of a mixer with the injection, the up kernel on two
  64-row tiles at 65..128 rows unless it releases; nothing changes without `core.hc_tiles`.
* **Input projection behind a releasing site A** (`DP_INPROJ_RELEASE_ENV`): the plan table,
  the build (layers with another decode copy are skipped; one differing element in the
  self-check, or a kernel that raises, leaves the switch off), the site's release and the
  block's GEMM. A 64-row plan that does not compile leaves `DP_HC_TILES_ENV` off.

The kernels' bits are covered by the GPU tests of `qwen4_exp_hc_decode_tc_first__decode` and
`qwen4_exp_hc_up_gate_mean` (`tm=64`, `rows_on_n`, `release_early`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _first_recorder(monkeypatch):
    calls = []

    def first(normed, first_w, rank, hc_count, with_inject, **kw):
        calls.append(("first", with_inject, kw))
        return torch.zeros(normed.shape[0], rank), None

    def up(low, up_w, normed, hc_count, **kw):
        calls.append(("up", kw))
        return torch.zeros(normed.shape[0], 8)

    monkeypatch.setattr(M, "qwen4_exp_hc_decode_tc_first", first)
    monkeypatch.setattr(M, "qwen4_exp_hc_up_gate_mean", up)
    return calls


def _core(tiles=True):
    return SimpleNamespace(hc_tiles=tiles, hc_lowrank=320, hc_count=4)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tiles, rows, inject, first_kw",
    [
        (True, 4, True, {}),  # the 10-way split stays
        (True, 5, True, {"tm": 64}),
        (True, 64, False, {"tm": 64}),
        (True, 65, True, {"rows_on_n": True}),
        (True, 96, True, {"rows_on_n": True}),
        (True, 80, False, {}),  # rows on N serves a mixer with the injection only
        (True, 97, True, {}),
        (False, 36, True, {}),
        (False, 80, True, {}),
    ],
)
def test_first_kernel_plan(monkeypatch, tiles, rows, inject, first_kw):
    calls = _first_recorder(monkeypatch)
    M._hc_first(_core(tiles), torch.zeros(rows, 16), None, inject)
    assert calls == [("first", inject, first_kw)]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tiles, rows, release, up_kw",
    [
        (True, 5, False, {"release_early": False, "tm": 64}),
        (True, 64, True, {"release_early": True, "tm": 64}),
        (True, 65, False, {"release_early": False, "tm": 64}),  # two 64-row tiles
        (True, 128, False, {"release_early": False, "tm": 64}),
        (True, 96, True, {"release_early": True, "tm": 128}),  # a releasing kernel: one tile
        (True, 129, False, {}),
        (True, 4, False, {}),
        (False, 36, False, {}),
        (False, 36, True, {"release_early": True, "tm": 128}),
    ],
)
def test_up_kernel_plan(monkeypatch, tiles, rows, release, up_kw):
    calls = _first_recorder(monkeypatch)
    M._hc_mixer(_core(tiles), torch.zeros(rows, 16), None, None, True, release)
    assert calls[-1] == ("up", up_kw)


@pytest.mark.cpu_only
def test_hc_tiles_build_gating(monkeypatch):
    compiled = []
    monkeypatch.setattr(M, "hc_tc_first_precompile", lambda *a, **k: compiled.append("first"))
    monkeypatch.setattr(M, "hc_tc_first_precompile_rowsn", lambda *a, **k: compiled.append("rowsn"))
    monkeypatch.setattr(M, "hc_up_precompile", lambda *a, **k: compiled.append("up"))
    core = SimpleNamespace(
        stream=10240, hc_lowrank=320, hc_count=4, hidden=2560, dp_rank=0, dp_size=8
    )
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: False)
    assert not M._hc_tiles_build(core, torch.device("cpu")) and not compiled
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setenv(M.DP_HC_TILES_ENV, "0")
    assert not M._hc_tiles_build(core, torch.device("cpu")) and not compiled
    monkeypatch.delenv(M.DP_HC_TILES_ENV)
    assert M._hc_tiles_build(core, torch.device("cpu"))
    assert compiled == ["first", "rowsn", "up"]


@pytest.mark.cpu_only
def test_hc_tiles_build_declines_a_plan_that_does_not_compile(monkeypatch):
    """A default-on form: a compile that raises turns it off at this rank (one warning) instead
    of stopping the load, which would leave the group's later collective builds waiting."""
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M, "hc_tc_first_precompile", lambda *a, **k: None)

    def rowsn(*a, **k):
        raise RuntimeError("smem")

    monkeypatch.setattr(M, "hc_tc_first_precompile_rowsn", rowsn)
    monkeypatch.setattr(M, "hc_up_precompile", lambda *a, **k: None)
    core = SimpleNamespace(
        stream=10240, hc_lowrank=320, hc_count=4, hidden=2560, dp_rank=0, dp_size=8
    )
    assert M._hc_tiles_build(core, torch.device("cpu")) is False
    assert len(warned) == 1 and M.DP_HC_TILES_ENV in warned[0] and "smem" in warned[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "dp, attn_tp, one_rank, cap, expected",
    [
        (8, 1, False, (10, 3), True),  # dep8
        (4, 1, False, (10, 3), True),  # dep4
        (2, 1, False, (10, 3), False),  # dep2: never measured with these forms
        (4, 1, False, (10, 7), False),  # not measured on SM107
        (1, 4, False, (10, 3), False),  # tep4
        (1, 1, True, (10, 3), False),
    ],
)
def test_dp_decode_rank(monkeypatch, dp, attn_tp, one_rank, cap, expected):
    monkeypatch.setattr(M.torch.cuda, "get_device_capability", lambda device=None: cap)
    core = SimpleNamespace(one_rank=one_rank, dp_size=dp, attn_tp_size=attn_tp)
    assert M._dp_decode_rank(core, SimpleNamespace(type="cuda")) is expected
    assert not M._dp_decode_rank(core, SimpleNamespace(type="cpu"))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "geom, rows, mpad, plan",
    [
        ((16480, 2560), 28, 32, M.MidPlan(1, 8, 5, 2, 16)),
        ((16480, 2560), 48, 64, M.MidPlan(1, 8, 5, 2, 16)),
        ((16480, 2560), 65, 80, M.MidPlan(1, 4, 8, 4, 16)),  # the fine MMA N from 65 rows
        ((16480, 2560), 72, 80, M.MidPlan(1, 4, 8, 4, 16)),
        ((14080, 2560), 96, 96, M.MidPlan(1, 4, 8, 4, 16)),
        ((14080, 2560), 97, 128, M.MidPlan(1, 4, 6, 4, 16)),
        ((14080, 2560), 120, 128, M.MidPlan(1, 4, 6, 4, 16)),
        ((16480, 2560), 136, 144, M.MidPlan(1, 4, 5, 1, 16)),  # the fine MMA N
        ((16480, 2560), 224, 224, M.MidPlan(1, 4, 5, 1, 16)),
    ],
)
def test_inproj_plan(geom, rows, mpad, plan):
    assert M._inproj_plan(*geom, rows) == (mpad, plan)


@pytest.mark.cpu_only
def test_inproj_plan_unknown_geometry():
    assert M._inproj_plan(2560, 6144, 32) is None
    assert M._inproj_plan(16480, 2560, 257) is None


def _release_core(bad_rows=()):
    w = {
        "l0_in_proj_w": torch.zeros(16480, 2560, dtype=torch.bfloat16),
        "l1_attn_in_w": torch.zeros(14080, 2560, dtype=torch.bfloat16),
        "l2_in_proj_w": torch.zeros(16480, 2560, dtype=torch.bfloat16),
    }

    def stock(x, wt):
        out = torch.zeros(x.shape[0], wt.shape[1], dtype=torch.bfloat16)
        if x.shape[0] in bad_rows:
            out[0, 0] = 1.0
        return out

    return SimpleNamespace(
        num_layers=3,
        is_attn=[False, True, False],
        w=w,
        # Layer 2 has another decode copy of its input projection: it keeps it.
        _mid={"l2_in_proj_w": object()},
        _gemv8={},
        _gemv_nv={},
        max_verify_rows=4,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 2, 4, 7, 12, 64, 65]),
        decode_mm=stock,
        decode_in_mm=stock,
        hc_count=4,
        hidden=2560,
        hc_lowrank=320,
        dp_rank=0,
        dp_size=4,
    )


def _release_build(monkeypatch, core):
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M, "_DECODE_PROJ_PDL", True)
    monkeypatch.setattr(
        M, "_inproj_mm", lambda x, ent: torch.zeros(x.shape[0], ent.w.shape[0], dtype=x.dtype)
    )
    monkeypatch.setattr(M, "hc_up_precompile", lambda *a, **k: None)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a, **k: None)
    return M._inproj_release_build(core, torch.device("cpu"))


@pytest.mark.cpu_only
def test_inproj_release_build(monkeypatch):
    table = _release_build(monkeypatch, _release_core())
    assert sorted(table) == [0, 1]
    # Graph rows 4 x batch inside 17..256: 28, 48, 256 (4 x 64); 4 x 65 is past the range.
    assert sorted(table[0]) == [28, 48, 256] and sorted(table[1]) == [28, 48, 256]
    ent = table[1][48]
    assert (ent.layer, ent.mpad, ent.plan) == (1, 64, M.MidPlan(1, 8, 5, 2, 16))
    assert table[0][256].mpad == 256


@pytest.mark.cpu_only
def test_inproj_release_off_when_the_self_check_differs(monkeypatch):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert _release_build(monkeypatch, _release_core(bad_rows=(48,))) is None
    assert len(warned) == 1 and M.DP_INPROJ_RELEASE_ENV in warned[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("where", ["gemm", "up"])
def test_inproj_release_off_when_a_kernel_raises(monkeypatch, where):
    """A GEMM (or the releasing up kernel) that does not compile or run is declined like one
    that differs: the switch is off at this rank, with one warning, and the load goes on."""
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    core = _release_core()

    def boom(*a, **k):
        raise RuntimeError("no such plan")

    if where == "gemm":
        core.decode_in_mm = boom
        assert _release_build(monkeypatch, core) is None
    else:
        monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
        monkeypatch.setattr(M, "_DECODE_PROJ_PDL", True)
        monkeypatch.setattr(
            M, "_inproj_mm", lambda x, ent: torch.zeros(x.shape[0], ent.w.shape[0], dtype=x.dtype)
        )
        monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(M, "hc_up_precompile", boom)
        assert M._inproj_release_build(core, torch.device("cpu")) is None
    assert len(warned) == 1 and M.DP_INPROJ_RELEASE_ENV in warned[0]
    assert "no such plan" in warned[0]


@pytest.mark.cpu_only
def test_inproj_release_switch_off(monkeypatch):
    monkeypatch.setenv(M.DP_INPROJ_RELEASE_ENV, "0")
    assert _release_build(monkeypatch, _release_core()) is None


@pytest.mark.cpu_only
def test_inproj_release_at():
    ent = M._InProj(0, None, 32, M.MidPlan())
    core = SimpleNamespace(_inproj_release={0: {28: ent}})
    assert M._inproj_release_at(core, 0, 28) is ent
    assert M._inproj_release_at(core, 0, 32) is None
    assert M._inproj_release_at(core, 1, 28) is None
    assert M._inproj_release_at(SimpleNamespace(_inproj_release=None), 0, 28) is None


@pytest.mark.cpu_only
def test_hc_site_passes_release_to_the_mixer(monkeypatch):
    seen = []
    hc = torch.zeros(28, 8)
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a, **k: (hc, hc))
    monkeypatch.setattr(
        M,
        "_hc_mixer",
        lambda core, normed, first, up, wi, release=False: seen.append(release) or (hc, hc),
    )
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = SimpleNamespace(hc_decode_pairs=False, hc_lowrank=320, hc_count=4, eps=1e-6)
    target._hc_site(hc, None, None, None, None, None, True, release=True)
    target._hc_site(hc, None, None, None, None, None, True)
    assert seen == [True, False]
    target.core.hc_decode_pairs = True
    with pytest.raises(AssertionError, match="inject-norm route"):
        target._hc_site(torch.zeros(4, 8), None, None, None, None, None, True, release=True)


_SM103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the input projection's bitwise claim is measured on SM103 (GB300)",
)


@_SM103
@pytest.mark.parametrize("geom", sorted(M._INPROJ_RELEASE_PLANS), ids=str)
def test_inproj_release_gemm_is_bitwise_the_stock_entry(geom):
    """What the build's self-check requires, on random operands: at every plan of
    `_INPROJ_RELEASE_PLANS` (rows 17..256, both MMA N rules) the bf16 mid-rows GEMM gives the
    stock entry's bits (`_decode_projection` for the attention in-projection,
    `low_m_bf16_mm` for the GDN one, as `_dense_mm` calls them at an attention-DP rank). A
    failure here means the default-on `DP_INPROJ_RELEASE_ENV` turns itself off at build."""
    dev = torch.device("cuda")
    n, k = geom
    gen = torch.Generator(device=dev).manual_seed(n)
    w = (torch.randn(n, k, generator=gen, device=dev) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(256, k, generator=gen, device=dev).to(torch.bfloat16)
    attn = geom == (14080, 2560)
    for rows in (20, 28, 48, 64, 65, 72, 80, 96, 97, 120, 128, 144, 224, 256):
        mpad, plan = M._inproj_plan(n, k, rows)
        got = M._inproj_mm(x[:rows], M._InProj(0, w, mpad, plan))
        if attn:
            ref = M._decode_projection(x[:rows], w.t())
        else:
            ref = M.low_m_bf16_mm(x[:rows], w.t(), M._DECODE_PROJ_PDL)
        torch.cuda.synchronize()
        bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        assert bad == 0, f"{geom} at {rows} rows (MMA N {mpad}): {bad} elements differ"

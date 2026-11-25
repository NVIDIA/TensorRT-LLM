# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An attention-DP rank's output projections on the bf16 mid-rows GEMM (`DP_OUT_PROJ_BF16_ENV`,
opt-in, CPU): the values, the table, the build (what it serves per value, where it declines)
and the dispatch (`_out_proj`: the table's GEMM at its rows, the step's entry elsewhere).

The GEMM's numerics (bf16 operands, fp32 accumulation, one rounding) are covered by the
mid-rows op's GPU tests (`ops/gemm/test_modeling_v2_mid_rows_mm.py`, bf16 cells and `mpad`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, mode",
    [(None, ""), ("0", ""), ("1", "1"), ("on", "1"), ("gdn", "gdn"), ("ALL", "all")],
)
def test_mode(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv(M.DP_OUT_PROJ_BF16_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ENV, value)
    assert M._out_proj_mode() == mode


@pytest.mark.cpu_only
def test_mode_rejects_another_value(monkeypatch):
    monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ENV, "attn")
    with pytest.raises(ValueError, match=M.DP_OUT_PROJ_BF16_ENV):
        M._out_proj_mode()


@pytest.mark.cpu_only
def test_table_is_contiguous_and_padded():
    """Every row count from the first to the last is in exactly one entry, at an MMA N that
    holds it and is a multiple of 16."""
    prev = M._OUT_PROJ_TABLE[0][0] - 1
    for lo, hi, mpad, _ in M._OUT_PROJ_TABLE:
        assert lo == prev + 1 and lo <= hi <= mpad and mpad % 16 == 0
        prev = hi
    assert M._out_proj_shape(16) is None and M._out_proj_shape(145) is None
    assert M._out_proj_shape(48) == (48, M.MidPlan(6, 4, 10, 1))


def _core(**over):
    w = {f"l{i}_out_w": torch.zeros(2560, 6144, dtype=torch.bfloat16) for i in (0, 1)}
    w["l2_o_w"] = torch.zeros(2560, 6144, dtype=torch.bfloat16)
    core = SimpleNamespace(
        gdn_layers=[0, 1],
        attn_layers=[2],
        w=w,
        _mid={},
        max_verify_rows=4,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 4, 7, 12, 20, 24, 36, 40]),
        dp_rank=0,
        dp_size=8,
    )
    for k, v in over.items():
        setattr(core, k, v)
    return core


def _build(monkeypatch, value, core, dp=True):
    monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ENV, value)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    compiled = []
    monkeypatch.setattr(M, "mid_mm_raw", lambda x, w, plan, mpad=None, **k: compiled.append(mpad))
    return M._out_proj_build(core, torch.device("cpu")), compiled


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, attn_rows",
    [("1", [48, 80, 96]), ("gdn", []), ("all", [28, 48, 80, 96, 144])],
)
def test_build(monkeypatch, value, attn_rows):
    table, compiled = _build(monkeypatch, value, _core())
    # Graph rows 4 x {7, 12, 20, 24, 36} = 28, 48, 80, 96, 144 are in the table; 4 / 16 / 160
    # are not.
    assert sorted(table["gdn_out"]) == [28, 48, 80, 96, 144]
    assert sorted(table["attn_o"]) == attn_rows
    assert sorted(compiled) == sorted({m for m, _ in table["gdn_out"].values()})


@pytest.mark.cpu_only
def test_build_declines(monkeypatch):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert _build(monkeypatch, "1", _core(), dp=False)[0] is None
    held = _core(_mid={"l1_out_w": object()})  # another decode copy (the tp1 dense FP8 mode)
    assert _build(monkeypatch, "1", held)[0] is None
    assert len(warned) == 2 and all(M.DP_OUT_PROJ_BF16_ENV in w for w in warned)
    monkeypatch.delenv(M.DP_OUT_PROJ_BF16_ENV)
    assert M._out_proj_build(_core(), torch.device("cpu")) is None


@pytest.mark.cpu_only
def test_dispatch(monkeypatch):
    """Each call takes the expected projection and returns what that projection returned."""
    calls = []
    mid_out, dense_out = torch.full((48, 2560), 1.0), torch.full((48, 2560), 2.0)
    monkeypatch.setattr(
        M,
        "mid_mm_raw",
        lambda x, w, plan, pdl=True, mpad=None: calls.append(("mid", mpad, w.shape)) or mid_out,
    )
    weight = torch.zeros(2560, 6144, dtype=torch.bfloat16)
    core = SimpleNamespace(_out_proj={"gdn_out": {48: (48, M.MidPlan())}, "attn_o": {}})

    def dense():
        calls.append(("dense",))
        return dense_out

    got = [
        M._out_proj(core, torch.zeros(48, 6144), weight.t(), "gdn_out", dense),
        M._out_proj(core, torch.zeros(48, 6144), weight.t(), "attn_o", dense),
        M._out_proj(core, torch.zeros(44, 6144), weight.t(), "gdn_out", dense),
        M._out_proj(core, torch.zeros(48, 12288)[:, ::2], weight.t(), "gdn_out", dense),  # strided
        M._out_proj(
            SimpleNamespace(_out_proj=None), torch.zeros(48, 6144), weight.t(), "gdn_out", dense
        ),
    ]
    assert calls == [("mid", 48, (2560, 6144))] + [("dense",)] * 4
    assert got[0] is mid_out and all(g is dense_out for g in got[1:])


@pytest.mark.cpu_only
@pytest.mark.parametrize("value, rowslo", [(None, None), ("0", None), ("5", 5), ("16", 16)])
def test_rowslo_value(monkeypatch, value, rowslo):
    if value is None:
        monkeypatch.delenv(M.DP_OUT_PROJ_BF16_ROWSLO_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ROWSLO_ENV, value)
    assert M._out_proj_rowslo() == rowslo


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["17", "on", "-1"])
def test_rowslo_rejects_another_value(monkeypatch, value):
    monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ROWSLO_ENV, value)
    with pytest.raises(ValueError, match=M.DP_OUT_PROJ_BF16_ROWSLO_ENV):
        M._out_proj_rowslo()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, attn_rows", [("1", [48, 80, 96]), ("all", [8, 12, 16, 28, 48, 80, 96, 144])]
)
def test_build_with_rowslo(monkeypatch, value, attn_rows):
    """`DP_OUT_PROJ_BF16_ROWSLO_ENV=5`: graph rows 8 / 12 / 16 take the first entry too (the GDN
    site always, the attention site under `all`); 4 rows stay on the entry."""
    monkeypatch.setenv(M.DP_OUT_PROJ_BF16_ROWSLO_ENV, "5")
    core = _core(
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 2, 3, 4, 7, 12, 20, 24, 36])
    )
    table, _ = _build(monkeypatch, value, core)
    first = (M._OUT_PROJ_TABLE[0][2], M._OUT_PROJ_TABLE[0][3])
    assert sorted(table["gdn_out"]) == [8, 12, 16, 28, 48, 80, 96, 144]
    assert all(table["gdn_out"][r] == first for r in (8, 12, 16, 28))
    assert sorted(table["attn_o"]) == attn_rows


def _draft_core(**over):
    core = _core(
        mtp_enabled=True,
        mtp_layer_idx=3,
        _out_proj={"gdn_out": {}, "attn_o": {}},
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[1, 2, 3, 8, 32, 40]),
    )
    core.w["l3_mtp_o_w"] = torch.zeros(2560, 6144, dtype=torch.bfloat16)
    for k, v in over.items():
        setattr(core, k, v)
    return core


@pytest.mark.cpu_only
def test_draft_o_build(monkeypatch):
    monkeypatch.setenv(M.DP_DRAFT_O_MID_ENV, "1")
    compiled = []
    monkeypatch.setattr(M, "mid_mm_raw", lambda x, w, plan, mpad=None, **k: compiled.append(mpad))
    rows = M._draft_o_build(_draft_core(), torch.device("cpu"))
    first = (M._OUT_PROJ_TABLE[0][2], M._OUT_PROJ_TABLE[0][3])
    assert rows == {b: first for b in (1, 2, 3, 8, 32)} and compiled == [first[0]]
    monkeypatch.delenv(M.DP_DRAFT_O_MID_ENV)
    assert M._draft_o_build(_draft_core(), torch.device("cpu")) is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "over",
    [
        {"_out_proj": None},  # the output projections are not engaged
        {"mtp_enabled": False},
        {"model_config": SimpleNamespace(cuda_graph_batch_sizes=[40, 48])},
    ],
    ids=["no-out-proj", "no-draft", "no-rows"],
)
def test_draft_o_build_declines(monkeypatch, over):
    monkeypatch.setenv(M.DP_DRAFT_O_MID_ENV, "1")
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._draft_o_build(_draft_core(**over), torch.device("cpu")) is None
    assert len(warned) == 1 and M.DP_DRAFT_O_MID_ENV in warned[0]


@pytest.mark.cpu_only
def test_draft_o_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        M, "mid_mm_raw", lambda x, w, plan, pdl=True, mpad=None: calls.append(("mid", mpad))
    )
    monkeypatch.setattr(M, "cublas_mm", lambda a, b: calls.append(("cublas", a.shape[0])))
    weight_t = torch.zeros(2560, 6144, dtype=torch.bfloat16).t()
    core = SimpleNamespace(_draft_o={3: (32, M.MidPlan(12, 2, 8, 2))})
    M._draft_o(core, torch.zeros(3, 6144), weight_t)
    M._draft_o(core, torch.zeros(4, 6144), weight_t)
    M._draft_o(SimpleNamespace(_draft_o=None), torch.zeros(3, 6144), weight_t)
    assert calls == [("mid", 32), ("cublas", 4), ("cublas", 3)]


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the tcgen05 mid-rows GEMM serves SM10x",
)


@_GPU
@pytest.mark.parametrize(
    "lo, hi, mpad, plan", M._OUT_PROJ_TABLE, ids=[f"{e[0]}-{e[1]}" for e in M._OUT_PROJ_TABLE]
)
def test_table_plans_on_the_gemm(lo, hi, mpad, plan):
    """Every entry of the table compiles and runs on `mid_mm_raw` at its first and last row
    counts: within one bf16 rounding of the fp64 product (plus fp32 summation noise relative to
    sum |x w|), and bitwise the same plan at the op's own MMA N (`mpad` moves no value)."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(4100 + lo)
    n, k = M._OUT_PROJ_GEOM
    w = (torch.randn(n, k, generator=gen, device=dev) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(hi, k, generator=gen, device=dev).to(torch.bfloat16)
    for rows in sorted({lo, hi}):
        got = M.mid_mm_raw(x[:rows], w, plan, pdl=True, mpad=mpad)
        own = M.mid_mm_raw(x[:rows], w, plan, pdl=True)
        torch.cuda.synchronize()
        assert got.shape == (rows, n)
        assert torch.equal(got, own), f"MMA N {mpad} vs the op's at {rows} rows"
        ref = x[:rows].double() @ w.double().t()
        absref = x[:rows].double().abs() @ w.double().abs().t()
        ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
        excess = ((got.double() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
        assert excess.max().item() < 1e-5, f"{rows} rows: {excess.max().item():.2e}"


@_GPU
@pytest.mark.parametrize("rows", [1, 3, 5, 8, 12, 16])
def test_first_entry_below_its_rows(rows):
    """`DP_OUT_PROJ_BF16_ROWSLO_ENV` and `DP_DRAFT_O_MID_ENV` run the table's first entry below
    its first row count: bitwise the op's own MMA N there, and within one bf16 rounding of the
    fp64 product."""
    dev = "cuda"
    _, _, mpad, plan = M._OUT_PROJ_TABLE[0]
    gen = torch.Generator(device=dev).manual_seed(4300 + rows)
    n, k = M._OUT_PROJ_GEOM
    w = (torch.randn(n, k, generator=gen, device=dev) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(rows, k, generator=gen, device=dev).to(torch.bfloat16)
    got = M.mid_mm_raw(x, w, plan, pdl=True, mpad=mpad)
    own = M.mid_mm_raw(x, w, plan, pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(got, own)
    ref = x.double() @ w.double().t()
    absref = x.double().abs() @ w.double().abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    excess = ((got.double() - ref).abs() - 0.5 * ulp).clamp_min(0) / absref.clamp_min(1e-30)
    assert excess.max().item() < 1e-5

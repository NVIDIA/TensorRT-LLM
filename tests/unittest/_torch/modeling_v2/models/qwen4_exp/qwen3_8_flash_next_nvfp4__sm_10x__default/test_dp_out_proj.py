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
    calls = []
    monkeypatch.setattr(
        M,
        "mid_mm_raw",
        lambda x, w, plan, pdl=True, mpad=None: calls.append(("mid", mpad, w.shape)),
    )
    weight = torch.zeros(2560, 6144, dtype=torch.bfloat16)
    core = SimpleNamespace(_out_proj={"gdn_out": {48: (48, M.MidPlan())}, "attn_o": {}})

    def dense():
        calls.append(("dense",))

    M._out_proj(core, torch.zeros(48, 6144), weight.t(), "gdn_out", dense)
    M._out_proj(core, torch.zeros(48, 6144), weight.t(), "attn_o", dense)
    M._out_proj(core, torch.zeros(44, 6144), weight.t(), "gdn_out", dense)
    M._out_proj(core, torch.zeros(48, 12288)[:, ::2], weight.t(), "gdn_out", dense)  # strided
    M._out_proj(
        SimpleNamespace(_out_proj=None), torch.zeros(48, 6144), weight.t(), "gdn_out", dense
    )
    assert calls == [("mid", 48, (2560, 6144))] + [("dense",)] * 4

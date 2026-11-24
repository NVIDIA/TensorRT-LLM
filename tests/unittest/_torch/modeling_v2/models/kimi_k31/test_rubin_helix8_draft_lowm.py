# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter levers ``draft_lowm_gemm`` (+ ``draft_lowm_gemm_d_class``), ``draft_conv_norm`` and
``draft_window_mla`` on CPU.

The startup constants line states the four (the D-class switch as a detail of ``draft_lowm_gemm``, the CuTe DSL
window's row bands ``draft_window_mla_bands`` as one of ``draft_window_mla``), and the module's ``draft_step``
(read from modeling.py, run over stand-ins) takes the lever path exactly when a GEMM or norm lever is on: with all
off the drafter step is today's, kernel for kernel. ``draft_window_mla`` keeps ``draft_step`` and swaps only the
window, at the row counts of its bands, for a drafter of the measured block (k5).
``test_rubin_helix8_draft_lowm_sm107.py`` holds the lever path to today's bits on the GPU.
"""

from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace

import pytest

__extra_import_path__ = ["."]
from test_rubin_helix8_draft_window import _MODELING, _constants, _exec, _tree  # noqa: E402

pytestmark = pytest.mark.cpu_only

_LEVERS = ("draft_lowm_gemm", "draft_lowm_gemm_d_class", "draft_conv_norm", "draft_window_mla")


def test_the_levers_are_logged_generation_constants():
    tree = _tree(_MODELING)
    ns = _constants(tree)
    generation = ns["_CONSTANTS"]["generation"]
    ns.update(_exec(ns, tree, "constants_line"))
    line = ns["constants_line"]("generation")
    for key in _LEVERS:
        assert isinstance(generation[key], bool), key
    assert f"draft_conv_norm={generation['draft_conv_norm']}" in line
    assert (
        f"draft_window_mla={generation['draft_window_mla']}"
        f"(draft_window_mla_bands={generation['draft_window_mla_bands']})" in line
    )
    assert generation["draft_window_mla_bands"] == "1-30,70-120"
    bands = _bands(ns, tree)
    assert bands == tuple(tuple(map(int, b.split("-"))) for b in "1-30,70-120".split(","))
    assert (
        f"draft_lowm_gemm={generation['draft_lowm_gemm']}"
        f"(draft_lowm_gemm_d_class={generation['draft_lowm_gemm_d_class']})" in line
    )
    assert ns["_DRAFT_LOWM_GEMM"] is generation["draft_lowm_gemm"]
    assert ns["_DRAFT_CONV_NORM"] is generation["draft_conv_norm"]
    assert ns["_DRAFT_WINDOW_MLA"] is generation["draft_window_mla"]


@pytest.mark.parametrize("gemm,conv_norm,mla", list(itertools.product((False, True), repeat=3)))
def test_draft_step_takes_the_lever_path_exactly_when_a_gemm_or_norm_lever_is_on(
    gemm, conv_norm, mla
):
    """``draft_step`` hands the step to the lever path when a GEMM or norm lever is on; otherwise it runs today's
    kernels: it asks ``_draft_window_mla`` for the CuTe DSL window's buffers first and plans the Triton window
    exactly when there are none (always, with ``draft_window_mla`` off)."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    calls = []

    class _Stop(Exception):
        pass

    def plan(*args):
        calls.append("window_plan")
        raise _Stop

    def window_mla(d, rows, block, heads, page_size, device):
        calls.append(("window_mla", rows, block))
        if mla:
            raise _Stop
        return None

    ns.update(
        _DRAFT_LOWM_GEMM=gemm,
        _DRAFT_CONV_NORM=conv_norm,
        _DRAFT_WINDOW_MLA=mla,
        _draft_step_fused=lambda *args: calls.append("fused") or "fused",
        _draft_window_mla=window_mla,
        rubin_dflash2_window_plan=plan,
        _DRAFT_WINDOW=1024,
    )
    ns.update(_exec(ns, tree, "draft_step"))
    layer = SimpleNamespace(
        k_b=SimpleNamespace(shape=(96, 128, 512)), kv_a=SimpleNamespace(shape=(576, 7168))
    )
    DW = SimpleNamespace(layers=(layer,))
    d = SimpleNamespace(eps=1e-6, taps=2, groups=448, group=16, cp_size=8, cp_rank=0, plan=None)
    st = SimpleNamespace(
        rows=6, block=6, num_ctx=None, page_table=None, table_rows=None, page_size=64
    )
    rt = SimpleNamespace(draft=d)
    hidden = SimpleNamespace(device="cuda")
    if gemm or conv_norm:
        assert ns["draft_step"](DW, hidden, st, rt) == "fused"
        assert calls == ["fused"]
    else:
        with pytest.raises(_Stop):
            ns["draft_step"](DW, hidden, st, rt)
        assert calls == [("window_mla", 6, 6)] + ([] if mla else ["window_plan"])


def _bands(ns: dict, tree) -> tuple:
    """``_DRAFT_WINDOW_MLA_BANDS`` as the module parses it (its row-band parser over the constant)."""
    ns.update(_exec(ns, tree, "_moe_row_bands", "_moe_in_bands"))
    (assign,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and [getattr(t, "id", None) for t in n.targets] == ["_DRAFT_WINDOW_MLA_BANDS"]
    ]
    ns["_DRAFT_WINDOW_MLA_BANDS"] = eval(compile(ast.Expression(assign.value), "bands", "eval"), ns)
    return ns["_DRAFT_WINDOW_MLA_BANDS"]


def _window_mla_ns(capturing: bool, made: list) -> dict:
    tree = _tree(_MODELING)
    ns = _constants(tree)
    _bands(ns, tree)
    ns.update(
        _DRAFT_WINDOW_MLA=True,
        _DRAFT_WINDOW=1024,
        helix_window_workspace=lambda *args: made.append(args) or SimpleNamespace(args=args),
        torch=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)),
        logger=SimpleNamespace(warning_once=lambda *a, **k: None, info_once=lambda *a, **k: None),
    )
    ns.update(_exec(ns, tree, "_draft_window_mla"))
    return ns


@pytest.mark.parametrize("capturing", [False, True])
def test_the_cute_window_buffers_are_made_per_row_count_outside_capture(capturing):
    """Under ``draft_window_mla`` the drafter step takes the CuTe DSL window's buffers for its row count, made at
    the first step that is not being captured and kept; a capture without them runs the Triton window (None)."""
    made = []
    ns = _window_mla_ns(capturing, made)
    d = SimpleNamespace(cp_size=8, mla={}, kv_scale=1.0)
    rows = ns["_DRAFT_WINDOW_MLA_BANDS"][0][1]
    first = ns["_draft_window_mla"](d, rows, 6, 96, 64, "cuda")
    if capturing:
        assert first is None and not made and d.mla == {}
        return
    assert made == [(rows, 96, 64, 1024, 8, "cuda")]
    assert d.mla == {rows: first}
    assert ns["_draft_window_mla"](d, rows, 6, 96, 64, "cuda") is first
    assert len(made) == 1


@pytest.mark.parametrize("block", [2, 4, 5, 6, 8, 9])
def test_the_cute_window_runs_only_in_its_row_bands_at_the_measured_block(block):
    """``draft_window_mla_bands`` gates the CuTe DSL window by block rows of the block it was certified at
    (``_DRAFT_LOWM_BLOCK``, k5): in a band a row count gets its buffers, outside every band the Triton window
    runs (None) and no buffers are made; a drafter of another block runs the Triton window at every row count,
    and so does a history of another dequant scale."""
    made = []
    ns = _window_mla_ns(False, made)
    bands = ns["_DRAFT_WINDOW_MLA_BANDS"]
    assert ns["_DRAFT_LOWM_BLOCK"] == 6
    d = SimpleNamespace(cp_size=8, mla={}, kv_scale=1.0)
    rows = [block * b for b in range(1, 17)]
    inside = [r for r in rows if any(lo <= r <= hi for lo, hi in bands)] if block == 6 else []
    got = {r: ns["_draft_window_mla"](d, r, block, 96, 64, "cuda") for r in rows}
    assert [r for r, ws in got.items() if ws is not None] == inside
    assert sorted(d.mla) == inside == [a[0] for a in made]
    if block == 6:
        assert inside
    scaled = SimpleNamespace(cp_size=8, mla={}, kv_scale=0.5)
    assert all(ns["_draft_window_mla"](scaled, r, block, 96, 64, "cuda") is None for r in rows)
    assert scaled.mla == {}

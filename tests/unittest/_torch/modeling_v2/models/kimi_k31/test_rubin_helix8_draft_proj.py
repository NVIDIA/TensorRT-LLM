# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter projection levers on CPU: ``draft_q_a_kv_a_fused``, ``draft_conv_pdl`` and
``draft_norm_ctas``, and the block the low-M plans serve.

* The startup constants line states the three (``draft_norm_ctas`` as a detail of
  ``draft_conv_norm``), and the module aliases read them.
* ``draft_q_a_kv_a_fused``: the weight table stacks each drafter layer's q_a rows with its kv_a
  rows (``d_l{j}_q_a_kv_a``; ``d_kv_a`` keeps its own stack for the history projection, so the
  kv_a rows are held twice) and reads the same checkpoint keys; ``_draft_views`` hands q_a as a
  view of the stack; ``draft_step`` projects both in one GEMM on a step whose rows fall in
  ``draft_q_a_kv_a_bands`` (else in two GEMMs, q_a a view of the stack), the q_a norm reads the
  product's first q_lora columns and the block RoPE the rest, and returns the step's values without
  the lever (exact stand-ins).
* ``draft_conv_pdl``: ``draft_step`` launches its four convolutions a layer with the engine's PDL
  switch; off, without PDL.
* ``draft_lowm_gemm``'s plans serve the block their rows were measured at (k5, 6 queries a request);
  a drafter of another block projects on cuBLAS.
* ``draft_norm_ctas``: the conv + add + RMSNorm takes the cluster only for a drafter of that block; a
  drafter of another block normalizes a row a CTA.

``test_rubin_helix8_draft_lowm_sm107.py`` holds the levers to the drafter's bits on the GPU.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib
import math
from types import SimpleNamespace
from typing import NamedTuple, Optional

import pytest
import torch

__extra_import_path__ = ["."]
from test_rubin_helix8_draft_window import (  # noqa: E402
    _MODELING,
    _constants,
    _exec,
    _helpers,
    _tree,
)

pytestmark = pytest.mark.cpu_only

_WEIGHTS = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix8.weights"
)
_SWITCHES = ("draft_q_a_kv_a_fused", "draft_conv_pdl")


def test_the_levers_are_logged_generation_constants():
    tree = _tree(_MODELING)
    ns = _constants(tree)
    generation = ns["_CONSTANTS"]["generation"]
    ns.update(_exec(ns, tree, "constants_line"))
    line = ns["constants_line"]("generation")
    for key in _SWITCHES:
        assert isinstance(generation[key], bool), key
        assert f"{key}={generation[key]}" in line, key
    assert generation["draft_norm_ctas"] in (0, 7)
    assert (
        f"draft_conv_norm={generation['draft_conv_norm']}"
        f"(draft_norm_ctas={generation['draft_norm_ctas']})" in line
    )
    assert (
        f"draft_q_a_kv_a_fused={generation['draft_q_a_kv_a_fused']}"
        f"(draft_q_a_kv_a_bands={generation['draft_q_a_kv_a_bands']})" in line
    )
    # the bands parse (the module's own parser) and stay within the measured rows (b1..b13)
    _exec(ns, tree, "_moe_row_bands")
    bands = ns["_moe_row_bands"](generation["draft_q_a_kv_a_bands"], 1024)
    assert bands and all(lo >= 1 and hi <= 78 for lo, hi in bands)
    assert ns["_DRAFT_Q_A_KV_A"] is generation["draft_q_a_kv_a_fused"]
    assert ns["_DRAFT_CONV_PDL"] is generation["draft_conv_pdl"]
    assert ns["_DRAFT_NORM_CTAS"] == generation["draft_norm_ctas"]


# --- the weight table ---


def _dims(W, **levers):
    d = W.Dims(*([0] * len(W.Dims._fields)))
    d = d._replace(
        num_layers=2, hidden=7168, vocab=163840, dtype=torch.bfloat16, kda_tp=8, kda_tp_rank=3
    )
    return d._replace(helix_cp=8, helix_cp_rank=3, **levers)


@pytest.mark.parametrize("fused", [False, True], ids=["off", "fused"])
def test_the_drafter_table_stacks_q_a_with_kv_a_only_under_the_lever(fused):
    W = importlib.import_module(_WEIGHTS)
    d, off = _dims(W, draft_q_a_kv_a_fused=fused), _dims(W)
    rows = W.w3_rows(d)
    names = {r.name for r in rows}
    for j in range(5):
        assert (f"d_l{j}_q_a_kv_a" in names) == fused and (f"d_l{j}_q_a" in names) != fused
    assert W._read_keys(d) == W._read_keys(off)
    extra = sum(math.prod(r.shape(d)) for r in rows) - sum(
        math.prod(r.shape(off)) for r in W.w3_rows(off)
    )
    # Under the lever the kv_a rows are held twice: exactly five layers' kv_a rows more.
    assert extra == (5 * 576 * 7168 if fused else 0), extra
    if fused:
        (row,) = [r for r in rows if r.name == "d_l2_q_a_kv_a"]
        assert row.shape(d) == (1536 + 576, 7168)
        ((keys, _),) = row.sources(d, None)
        assert keys == (
            "layers.2.self_attn.q_a_proj.weight",
            "layers.2.self_attn.kv_a_proj_with_mqa.weight",
        )


def test_the_core_lays_out_the_drafter_by_the_constant():
    """The core replaces the table's drafter layout with the constant's (``dims(self)``)."""
    text = _MODELING.read_text()
    assert "self.dims = _weights.dims(self)._replace(draft_q_a_kv_a_fused=_DRAFT_Q_A_KV_A)" in text


# --- the views and draft_step over exact stand-ins ---

_L, _HEADS, _NOPE, _ROPE, _KVL, _V, _QL, _HID, _I, _TAPS = 5, 2, 3, 2, 4, 3, 5, 6, 4, 2
_ROWS, _BLOCK = 4, 2


def _ints(gen, *shape):
    """Small integers as float32: every product and sum below is exact."""
    return torch.randint(-3, 4, shape, generator=gen).float()


def _weights(fused: bool) -> dict:
    gen = torch.Generator().manual_seed(77)
    latent = _KVL + _ROPE
    w = {
        "d_fc": _ints(gen, _HID, 2 * _HID),
        "d_final_norm": _ints(gen, _HID),
        "d_kv_a": _ints(gen, _L * latent, _HID),
        "d_kv_a_norm": _ints(gen, _L, _KVL),
    }
    for j in range(_L):
        w.update(
            {
                f"d_l{j}_kv_b": _ints(gen, _HEADS * (_NOPE + _V), _KVL),
                f"d_l{j}_in_norm": _ints(gen, _HID),
                f"d_l{j}_post_norm": _ints(gen, _HID),
                f"d_l{j}_q_a_norm": _ints(gen, _QL),
                f"d_l{j}_q_b": _ints(gen, _HEADS * (_NOPE + _ROPE), _QL),
                f"d_l{j}_o": _ints(gen, _HID, _HEADS * _V),
                f"d_l{j}_gate_up": _ints(gen, 2 * _I, _HID),
                f"d_l{j}_down": _ints(gen, _HID, _I),
                f"d_l{j}_attn_conv_base": _ints(gen, 2, _TAPS, _HID),
                f"d_l{j}_attn_conv_proj": _ints(gen, 2 * _TAPS, _HID),
                f"d_l{j}_mlp_conv_base": _ints(gen, 2, _TAPS, _HID),
                f"d_l{j}_mlp_conv_proj": _ints(gen, 2 * _TAPS, _HID),
            }
        )
        q_a = _ints(gen, _QL, _HID)
        kv_a = w["d_kv_a"][j * latent : (j + 1) * latent]
        if fused:  # the table's stack: q_a's rows, then the layer's kv_a rows (a copy)
            w[f"d_l{j}_q_a_kv_a"] = torch.cat([q_a, kv_a])
        else:
            w[f"d_l{j}_q_a"] = q_a
    return w


def _views(fused: bool):
    """modeling.py's ``_draft_views`` (and the two view classes) over the table's tensors."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    ns.update(nn=SimpleNamespace(ParameterDict=dict), NamedTuple=NamedTuple, Optional=Optional)
    classes = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("_DraftLayerW", "_DraftW")
    ]
    assert len(classes) == 2
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    _exec(ns, tree, "_draft_views")
    return ns["_draft_views"](_weights(fused), _L, _NOPE, _KVL)


def test_the_drafter_views_read_the_stack():
    for fused in (False, True):
        for L in _views(fused).layers:
            assert (L.q_a_kv_a is not None) == fused
            if fused:
                ptr = L.q_a_kv_a.untyped_storage().data_ptr()
                assert L.q_a.untyped_storage().data_ptr() == ptr
                assert L.q_a.shape == (_QL, _HID)
                assert torch.equal(L.q_a_kv_a[_QL:], L.kv_a)


def _gemm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (a.double() @ w.double().t()).float()


def _draft(
    fused: bool, conv_pdl: bool = False, env_pdl: bool = True, bands=((1, 1024),), block=_BLOCK
) -> SimpleNamespace:
    """helix8's ``draft_step`` from modeling.py over exact stand-ins: the GEMMs' weights in order,
    what the q_a norm and the block RoPE read, the convolutions' PDL flags, the output."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    seen = SimpleNamespace(gemms=[], qa=[], kv=[], conv_pdl=[])
    DW = _views(fused)
    names = {}
    for j, L in enumerate(DW.layers):
        for f in L._fields:
            t = getattr(L, f)
            if isinstance(t, torch.Tensor):
                names.setdefault(id(t), f"{f}{j}")

    def rubin_linear(x, w, tf32):
        seen.gemms.append(names.get(id(w), "?"))
        return _gemm(x, w)

    def rmsnorm(x, w, eps):  # bounded, exact: the sign of each value times the weight
        if x.shape[-1] == _QL:
            seen.qa.append(x.clone())
        return torch.sign(x) * w

    def block_rope(q, query, kv, *args):
        seen.kv.append(kv.clone())
        query[..., _KVL:] = q[..., _NOPE:]

    def conv(x, coef, base, block, group, pdl=False):
        seen.conv_pdl.append(pdl)
        return x + coef.sum(-1).sum(-1)[:, None] * base[0]

    def bmm_out(a, b, out):
        out.copy_(torch.bmm(a, b))
        return out

    ns.update(
        _DRAFT_LOWM_GEMM=False,
        _DRAFT_CONV_NORM=False,
        _DRAFT_WINDOW_MLA=False,
        _DRAFT_CONV_PDL=conv_pdl,
        _DRAFT_Q_A_KV_A_BANDS=bands,
        _DRAFT_LOWM_BLOCK=block,
        _DRAFT_WINDOW=1024,
        _AR_NONE=0,
        get_env_enable_pdl=lambda: env_pdl,
        rubin_dflash2_window_plan=lambda *a: "plan",
        flashinfer_rmsnorm=rmsnorm,
        flashinfer_fused_add_rmsnorm=lambda a, res, w, eps: (torch.sign(a + res) * w, a + res),
        rubin_linear=rubin_linear,
        _gemm_pinned=lambda site, a, b, today, *args: today(),
        dflash2_grouped_conv=conv,
        bmm_out=bmm_out,
        dflash2_mla_block_rope=block_rope,
        rubin_dflash2_window_triton=lambda query, *a, **k: query[..., :_KVL].contiguous(),
        window_vec_io_aligned=lambda *a: True,
        rubin_dflash2_helix_combine=lambda p, s, m, h, v, zero_kv_mask=None: p.reshape(
            p.shape[0], h * v
        ),
        allreduce_module=lambda x, *a: x,
        flashinfer_silu_and_mul=lambda x: x[:, : x.shape[1] // 2] * x[:, x.shape[1] // 2 :],
        logger=SimpleNamespace(info_once=lambda msg, key=None: None),
    )
    _exec(ns, tree, "draft_step", *_helpers(tree, "draft_step", ns))
    st = SimpleNamespace(
        rows=_ROWS,
        block=_BLOCK,
        num_ctx=torch.tensor([3, 70]),
        page_table=None,
        table_rows=None,
        page_size=64,
        positions=torch.arange(_ROWS),
        pools=[torch.zeros(3, 1, 1, 64, _KVL + _ROPE) for _ in range(_L)],
    )
    d = SimpleNamespace(
        eps=1e-5,
        taps=_TAPS,
        groups=1,
        group=_HID,
        cp_size=8,
        cp_rank=3,
        plan=None,
        block_kv=torch.zeros(8, 1, _KVL + _ROPE),
        stats=torch.zeros(8, _HEADS, 2),
        softmax_scale=0.13,
        kv_scale=1.0,
        workspace=None,
        freqs=None,
        block_slots=torch.arange(8),
        block_offsets=torch.zeros(8, dtype=torch.int64),
        mapping=None,
    )
    x = _ints(torch.Generator().manual_seed(5), _ROWS, _HID)
    seen.out = ns["draft_step"](DW, x, st, SimpleNamespace(draft=d, all_reduce=None))
    return seen


def test_draft_step_projects_q_a_and_kv_a_in_one_gemm_under_the_lever():
    off, fused = _draft(False), _draft(True)
    assert off.gemms[:5] == ["attn_conv_proj0", "q_a0", "q_b0", "kv_a0", "o0"]
    assert fused.gemms[:4] == ["attn_conv_proj0", "q_a_kv_a0", "q_b0", "o0"]
    assert len(fused.gemms) == len(off.gemms) - 5
    # The q_a norm reads the stack's first q_lora columns, the block RoPE the rest: the values of
    # the two GEMMs, as views of the one product.
    for a, b in zip(off.qa, fused.qa):
        assert torch.equal(a, b)
    for a, b in zip(off.kv, fused.kv):
        assert torch.equal(a, b)
    assert len(fused.kv) == len(fused.qa) == _L
    assert torch.equal(off.out, fused.out)


@pytest.mark.parametrize("why", ["rows", "block"])
def test_a_step_outside_the_bands_projects_q_a_and_kv_a_in_two_gemms(why):
    """Rows outside draft_q_a_kv_a_bands, or a drafter of another block than the measured one, run
    q_a (a view of the stack) and kv_a in two GEMMs."""
    off = _draft(False)
    if why == "rows":
        outside = _draft(True, bands=((_ROWS + 1, 78),))
    else:
        outside = _draft(True, block=_BLOCK + 4)
    assert outside.gemms[:5] == ["attn_conv_proj0", "q_a0", "q_b0", "kv_a0", "o0"]
    assert len(outside.gemms) == len(off.gemms)
    assert torch.equal(off.out, outside.out)


@pytest.mark.parametrize("conv_pdl,env_pdl", [(False, True), (True, True), (True, False)])
def test_draft_step_launches_the_convolutions_with_pdl_under_the_lever(conv_pdl, env_pdl):
    run = _draft(False, conv_pdl=conv_pdl, env_pdl=env_pdl)
    assert run.conv_pdl == [conv_pdl and env_pdl] * (4 * _L)
    assert torch.equal(run.out, _draft(False).out)


@pytest.mark.parametrize("block", [2, 6, 8])
def test_the_low_m_plans_serve_only_the_measured_block(block):
    tree = _tree(_MODELING)
    ns = _constants(tree)
    plan = {(1, 1): ((6, "kernel"),)}
    seen = []
    ns.update(
        _DRAFT_LOWM_PLAN=plan,
        get_env_enable_pdl=lambda: True,
        k31_low_m_linear=lambda x, w, p, pdl: seen.append(p) or "out",
    )
    _exec(ns, tree, "_draft_lowm_plan", "_draft_linear")
    assert ns["_DRAFT_LOWM_BLOCK"] == 6
    assert ns["_draft_linear"]("x", "w", block) == "out"
    assert seen == [plan if block == 6 else None]


@pytest.mark.parametrize("block", [2, 6, 8])
@pytest.mark.parametrize("ctas", [0, 7])
def test_the_norm_cluster_serves_only_the_measured_block(block, ctas):
    """Under draft_conv_norm the conv + add + RMSNorm gets draft_norm_ctas for a drafter of the
    measured block (k5) and one CTA a row (0) for a drafter of any other block."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    seen = []
    ns.update(
        _DRAFT_CONV_NORM=True,
        _DRAFT_NORM_CTAS=ctas,
        dflash2_conv_add_rmsnorm=lambda *args: seen.append(args) or ("normed", "residual"),
    )
    _exec(ns, tree, "_draft_conv_add_norm")
    assert ns["_DRAFT_LOWM_BLOCK"] == 6
    args = ("a", "delta", "base", "residual", "norm", 1e-6, block, 16, True, False)
    assert ns["_draft_conv_add_norm"](*args) == ("normed", "residual")
    assert seen == [args + (ctas if block == 6 else 0,)]


@pytest.mark.parametrize("named", [True, False], ids=["plan_kernel", "no_plan_kernel"])
def test_a_pinned_site_runs_its_pin_where_the_plan_names_no_kernel(named):
    """Under draft_lowm_gemm a projection the plan names a kernel for runs it; else the site's
    ``_gemm_pinned`` call (``unplanned``), whose GEMM is today's."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    seen = []
    x, w = torch.zeros(54, 8), torch.zeros(16, 8)
    choice = ("splitkc",) if named else ("default",)
    ns.update(
        _DRAFT_LOWM_PLAN={(16, 8): ((60, choice),)},
        get_env_enable_pdl=lambda: True,
        plan_choice=lambda plan, m, n, k: (
            None if plan is None or plan[(n, k)][0][1][0] == "default" else plan[(n, k)][0][1]
        ),
        k31_low_m_linear=lambda *a: seen.append("low_m") or "low_m",
    )
    _exec(ns, tree, "_draft_lowm_plan", "_draft_linear")
    got = ns["_draft_linear"](x, w, 6, lambda: seen.append("pinned") or "pinned")
    assert got == ("low_m" if named else "pinned") and seen == [got]
    # another block: no plan, so the pinned call
    seen.clear()
    assert ns["_draft_linear"](x, w, 8, lambda: seen.append("pinned") or "pinned") == "pinned"

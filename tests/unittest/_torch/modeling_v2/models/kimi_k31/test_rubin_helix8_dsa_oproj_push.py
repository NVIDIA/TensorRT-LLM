# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's ``dsa_oproj_push`` (on as shipped): on reduce_norm's push-path rows within
``dsa_oproj_push_bands`` (1 to 72 rows), a DSA layer's o_proj runs as the BF16 push GEMM
(``bf16_gemm_push``, one wave of CTAs at every row count) where its output equals today's o_proj word for
word, checked once per process and row count; the mix after the layer finishes the call
(``rt.oproj_push.finish_norm``, the decoder's ``_Pushed`` branch).

CPU (``cpu_only``), from helix8's modeling.py with recording stand-ins: the defaults and the tactic
table; the push slot is built for either push lever and carries the DSA bands; ``_mla_output``
only pushes with the switch on and the push-path flag; ``_dsa_oproj_push``'s device check (first
call compares and runs today's GEMM, equal rows push from then on, unequal or uncheckable rows
never do, a capture before the check runs today's GEMM unrecorded, rows past the push kernel, the
push workspace or outside the bands run today's GEMM); the decoder's push-path flag; the README's
INFO line.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import NamedTuple, Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"
#: helix8's generation rows: b x (k5 + 1).
_HELIX8_ROWS = tuple(6 * b for b in range(1, 13))


def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
    return ns


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _define(ns: dict, tree: ast.Module, names) -> None:
    body = [copy.deepcopy(_function(tree, name)) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))


class _Pushed(NamedTuple):
    rows: int


_PUSH_FNS = ["_dsa_oproj_push", "_dsa_oproj_tactic", "_moe_in_bands"]


def _exec(names, equal=True, capturing=False, raises=None, **switches):
    """helix8's functions ``names`` with the module constants ``switches`` overridden and recording
    stand-ins: today's GEMM returns ones, the push kernel's values ones (``equal``) or twos."""
    tree = _tree()
    ns = _constants(tree)
    ns.update(switches)
    calls, log = [], _Log()

    def today(name, a, *args, **kwargs):
        calls.append((name, a.shape[0]))
        return torch.ones(a.shape[0], 4, dtype=torch.bfloat16)

    def values(x, w, out, tactic):
        calls.append(("values", x.shape[0], tactic))
        if raises:
            raise ValueError(raises)
        return out.fill_(1.0 if equal else 2.0)

    def push(x, w, shape, view, one_shot, tactic, pdl):
        calls.append(("push", x.shape[0], tuple(shape.shape), view, one_shot, tactic, pdl))

    ns.update(
        _gemm_pinned=lambda site, a, b, fn, **kw: today(f"pinned:{site}", a),
        rubin_linear=lambda a, w, bias: today("linear", a),
        k31_sigmoid_gate_mul=lambda out, g: out,
        bf16_gemm_push_values=values,
        bf16_gemm_push=push,
        logger=log,
        _Pushed=_Pushed,
        Optional=Optional,
        Any=object,
        _MlaW=object,
        _Rt=object,
    )
    ns["_DSA_OPROJ_PUSH_CHECKED"] = {}
    torch_cuda = SimpleNamespace(is_current_stream_capturing=lambda: capturing)
    torch_names = ("empty", "empty_like", "equal", "int16")
    ns["torch"] = SimpleNamespace(**{k: getattr(torch, k) for k in torch_names})
    ns["torch"].cuda = torch_cuda
    _define(ns, tree, names)
    return ns, calls, log


def _layer():
    return SimpleNamespace(o=torch.zeros(4, 3, dtype=torch.bfloat16), g=None)


def _rt(max_rows=128, one_shot=False, bands=((1, 72),)):
    ws = SimpleNamespace(view="lamport-view", pdl=True, transport=lambda rows: (one_shot, 2))
    return SimpleNamespace(oproj_push=SimpleNamespace(ws=ws, max_rows=max_rows, dsa_bands=bands))


def test_the_lever_is_on_as_shipped():
    ns = _constants(_tree())
    gen = ns["_CONSTANTS"]["generation"]
    # On in the shipped stack at every band it measured faster in at world 8 (6-72 rows).
    assert gen["dsa_oproj_push"] is True and ns["_DSA_OPROJ_PUSH"] is True
    assert gen["dsa_oproj_push_bands"] == "1-72"
    assert ns["_DETAILS"]["dsa_oproj_push_bands"] == "dsa_oproj_push"
    # The push path needs reduce_norm's Lamport workspace and PDL mix up to 72 rows.
    assert gen["lamport_push_allreduce"] and gen["ar_push_amid"] and gen["ar_finish_fused"]
    assert gen["attn_res_pdl"] == 2 and gen["attn_res_pdl_max_m"] >= ns["_DSA_OPROJ_PUSH_MAX_M"]


def test_the_tactic_table_covers_1_to_72_rows_in_one_wave():
    ns = _constants(_tree())
    table = ns["_DSA_OPROJ_PUSH_TACTICS"]
    assert ns["_DSA_OPROJ_PUSH_MAX_M"] == 72 == table[-1][0]
    bounds = [bound for bound, _ in table]
    assert bounds == sorted(bounds) and len(set(bounds)) == len(bounds)
    for _, (kind, mma_m, mma_n, split_k, stages) in table:
        # No K split (one FP32 order, as cuBLAS's o_proj); operands and staging tile fit.
        assert (kind, split_k) == ("splitk", 1) and mma_m in (64, 128) and mma_n in (8, 16, 32, 64)
        smem = (mma_m + mma_n) * 128 * 2 * stages + mma_m * mma_n * 2
        assert smem <= 232448
    _define(ns, _tree(), ["_dsa_oproj_tactic"])
    for rows in range(1, 73):
        _, mma_m, mma_n, _, _ = ns["_dsa_oproj_tactic"](rows)
        # One wave of CTAs on 212 SMs, one CTA per SM.
        assert (7168 // mma_m) * -(-rows // mma_n) <= 212, rows
        if rows <= 32:
            # The token tile of 1-32 rows holds them, on 64-row weight tiles.
            assert (mma_m, mma_n) == (64, 8 if rows <= 8 else 16 if rows <= 16 else 32), rows
    assert ns["_dsa_oproj_tactic"](73) is None


@pytest.mark.parametrize(
    "kda, dsa, built", [(False, False, False), (True, False, True), (False, True, True)]
)
def test_the_push_slot_is_built_for_either_push_lever(kda, dsa, built):
    tree = _tree()
    ns = _constants(tree)
    ns.update(
        rubin_lamport_row_push_desc=lambda push: "desc",
        _OprojPush=lambda **kw: SimpleNamespace(**kw),
        rubin_attn_res_lamport_add_rmsnorm=None,
        logger=None,
        Optional=Optional,
        Any=object,
        Callable=object,
        _MixW=object,
    )
    _define(ns, tree, ["_oproj_push_slot", "_moe_row_bands"])
    c = dict(
        attn_res_pdl=2,
        attn_res_pdl_max_m=128,
        kda_oproj_push=kda,
        dsa_oproj_push=dsa,
        dsa_oproj_push_bands="1-30,49-72",
    )
    slot = ns["_oproj_push_slot"](c, 1e-5, "push-ws")
    assert (slot is not None) is built
    if built:
        assert slot.ws == "push-ws" and slot.max_rows == 128 and slot.desc == "desc"
        assert slot.dsa_bands == (((1, 30), (49, 72)) if dsa else ())


@pytest.mark.parametrize("bands", ["0-12", "1-73", "30-12", "12-30,24-36", "12"])
def test_the_push_slot_rejects_bands_outside_1_to_72(bands):
    tree = _tree()
    ns = _constants(tree)
    ns.update(
        rubin_lamport_row_push_desc=lambda push: "desc",
        _OprojPush=lambda **kw: SimpleNamespace(**kw),
        Optional=Optional,
        Any=object,
        Callable=object,
        _MixW=object,
    )
    _define(ns, tree, ["_oproj_push_slot", "_moe_row_bands"])
    c = dict(
        attn_res_pdl=2, attn_res_pdl_max_m=128, dsa_oproj_push=True, dsa_oproj_push_bands=bands
    )
    with pytest.raises(ValueError, match="row bands"):
        ns["_oproj_push_slot"](c, 1e-5, "push-ws")


@pytest.mark.parametrize("switch", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("push", [False, True], ids=["plain_path", "push_path"])
@pytest.mark.parametrize("rows", [24, 54])
def test_mla_output_pushes_only_with_the_switch_on_the_push_path(switch, push, rows):
    ns, calls, _ = _exec(["_mla_output", *_PUSH_FNS], _DSA_OPROJ_PUSH=switch)
    L, rt, out = _layer(), _rt(), torch.zeros(rows, 3, dtype=torch.bfloat16)
    gate = (None, SimpleNamespace(wait=lambda: None))
    first = ns["_mla_output"](L, None, out, rows, gate, rt, push)
    second = ns["_mla_output"](L, None, out, rows, gate, rt, push)
    if switch and push:
        # The first call checks (and runs today's GEMM); the second pushes.
        assert isinstance(first, torch.Tensor) and isinstance(second, _Pushed)
        assert [c[0] for c in calls] == ["pinned:mla_o", "values", "pinned:mla_o", "push"]
    else:
        assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
        assert [c[0] for c in calls] == ["pinned:mla_o", "pinned:mla_o"]


@pytest.mark.parametrize("rows", [1, 8, 9, 16, 17, 32, 33, 64, 65, 72, *_HELIX8_ROWS])
def test_the_push_after_an_equal_check(rows):
    ns, calls, log = _exec(_PUSH_FNS)
    L, rt, gated = _layer(), _rt(one_shot=rows <= 8), torch.zeros(rows, 3, dtype=torch.bfloat16)
    assert ns["_dsa_oproj_push"](L, gated, rt) is None
    assert ns["_DSA_OPROJ_PUSH_CHECKED"] == {rows: True}
    assert log.lines == [("info", ns["_DSA_OPROJ_PUSH_LINE"].format(rows=rows))]
    assert ns["_dsa_oproj_push"](L, gated, rt) == _Pushed(rows)
    (_, _, tactic) = calls[1]
    assert tactic == ns["_dsa_oproj_tactic"](rows)
    assert tactic[0] == "splitk" and tactic[3] == 1, "no K split"
    push = calls[2]
    assert push == ("push", rows, (rows, 4), "lamport-view", rows <= 8, tactic, True)


@pytest.mark.parametrize("raises", [None, "unsupported rows"], ids=["unequal", "unsupported"])
def test_a_failed_check_keeps_todays_gemm(raises):
    ns, calls, log = _exec(_PUSH_FNS, equal=False, raises=raises)
    L, rt, gated = _layer(), _rt(), torch.zeros(54, 3, dtype=torch.bfloat16)
    for _ in range(3):
        assert ns["_dsa_oproj_push"](L, gated, rt) is None
    assert ns["_DSA_OPROJ_PUSH_CHECKED"] == {54: False}
    assert [c[0] for c in calls] == ["pinned:mla_o", "values"]
    ((level, line),) = log.lines
    assert level == "warning" and line.startswith("Kimi K3.1 Rubin DSA: o_proj push at 54 rows")


def test_a_capture_before_the_check_runs_todays_gemm_unrecorded():
    ns, calls, log = _exec(_PUSH_FNS, capturing=True)
    L, rt, gated = _layer(), _rt(), torch.zeros(24, 3, dtype=torch.bfloat16)
    assert ns["_dsa_oproj_push"](L, gated, rt) is None
    assert ns["_DSA_OPROJ_PUSH_CHECKED"] == {} and calls == [] and log.lines == []


@pytest.mark.parametrize(
    "rows, max_rows, has_slot, bands",
    [
        (73, 128, True, ((1, 72),)),  # past the push kernel's rows
        (96, 128, True, ((1, 72),)),
        (24, 16, True, ((1, 72),)),  # past the push workspace's rows
        (24, 128, False, ((1, 72),)),  # no push workspace
        (0, 128, True, ((1, 72),)),
        (36, 128, True, ((1, 30), (49, 72))),  # outside the bands
        (24, 128, True, ()),  # the lever off at the slot (only kda_oproj_push built it)
    ],
)
def test_rows_the_push_does_not_serve(rows, max_rows, has_slot, bands):
    ns, calls, _ = _exec(_PUSH_FNS)
    rt = _rt(max_rows, bands=bands) if has_slot else SimpleNamespace(oproj_push=None)
    gated = torch.zeros(rows, 3, dtype=torch.bfloat16)
    assert ns["_dsa_oproj_push"](_layer(), gated, rt) is None
    assert calls == [] and ns["_DSA_OPROJ_PUSH_CHECKED"] == {}


def test_the_decoder_passes_the_push_path_rule():
    """The MLA layers take the KDA verify's push-path rule: a non-boundary layer with snapshot rows."""
    (cls,) = [n for n in _tree().body if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget"]
    calls = [
        ast.unparse(n)
        for n in ast.walk(cls)
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ("mla_generate", "kda_verify")
    ]
    assert "mla_generate(L.mla, i, x, st, rt, not L.boundary and L.n_bank > 0)" in calls
    # The KDA verify's rule, which boundary_ar_push widens to a checked boundary.
    assert "kda_verify(L.kda, x, st, rt, push)" in calls
    assert "push = not L.boundary and L.n_bank > 0 or bmode == 'push'" in ast.unparse(cls)
    text = ast.unparse(_function(_tree(), "mla_generate"))
    assert "return _mla_output(L, x, out, n, gate, rt, push)" in text


def test_the_info_line():
    line = _constants(_tree())["_DSA_OPROJ_PUSH_LINE"]
    assert line.startswith("Kimi K3.1 Rubin DSA: ") and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()

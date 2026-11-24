# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's KDA o_proj hand-off levers: ``kda_oproj_push`` and the verify's early release of o_proj.

``kda_oproj_push``: on the rows ``_tp_reduce_slots``'s ``reduce_norm`` pushes with the copy kernel
(a non-boundary layer with snapshot rows, 1 to ``attn_res_pdl_max_m`` rows, the push workspace
built), o_proj's epilogue stores the partial into the Lamport slots
(``rubin_mxfp8_linear_prequantized_push``), ``kda_verify`` returns ``_Pushed`` and the decoder calls
``rt.oproj_push.finish_norm``: the mix call of ``reduce_norm``'s push path with the same arguments,
without the copy kernel. ``kda_decode_early_trigger`` with ``kda_verify_pdl`` has the verify kernel
release o_proj right after its grid wait. Both change scheduling and the store's destination only
(the GPU tests compare the bits:
``ops/gemm/test_rubin_mxfp8_linear_row_store.py``, ``ops/ssm/test_rubin_kda_mtp_verify.py`` and
``rubin_multirank/test_rubin_mxfp8_linear_push.py``).

CPU (``cpu_only``), from helix8's modeling.py over its constants:

* the keys are generation launch constants, and the module-level copies read them;
* ``finish_norm`` makes the mix call ``reduce_norm`` makes after its push, argument for argument,
  on both transports, and pushes nothing; it is absent without the key, the workspace or the PDL
  mix; it carries the workspace's row-push descriptor;
* ``kda_verify`` pushes and returns ``_Pushed`` only when the lever is on, the decoder asks and the
  rows fit, and logs the INFO line once it acts (a warning once if the GEMM had no push epilogue);
* the verify call takes its early release from ``_KDA_EARLY_TRIGGER`` and its PDL launch from
  ``_kda_verify_pdl``;
* the decoder asks for the push on non-boundary layers with snapshot rows and finishes a
  ``_Pushed`` partial through ``finish_norm``;
* the README carries the INFO line.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"

#: The keys and their values in helix8's generation constants.
_KEYS = {"kda_oproj_push": True}
#: Rows up to which the stand-in workspace takes the one-shot transport (two-shot above).
_ONE_SHOT_ROWS = 9


def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _constants(tree: ast.Module) -> dict:
    """Module-level constants that evaluate on their own, in source order."""
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


def _defs(tree: ast.Module, names: tuple, ns: dict) -> dict:
    """Execute the module-level definitions ``names`` of ``tree`` into ``ns``."""
    nodes = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert sorted(n.name for n in nodes) == sorted(names)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


class _Log:
    def __init__(self):
        self.once = []
        self.warned = []

    def info_once(self, msg, key=None):
        self.once.append(msg)

    def warning_once(self, msg, key=None):
        self.warned.append(msg)


def _push():
    """A stand-in push workspace of 8 ranks: one-shot up to _ONE_SHOT_ROWS rows, then two-shot."""
    view = SimpleNamespace(
        uc_local=11, flags=SimpleNamespace(data_ptr=lambda: 12), world=8, mc_base=13, rank=3
    )
    return SimpleNamespace(
        view=view,
        transport=lambda rows: (rows <= _ONE_SHOT_ROWS, 1 if rows <= _ONE_SHOT_ROWS else 2),
        pdl=True,
    )


def _gen(**over) -> dict:
    consts = dict(_constants(_tree())["_CONSTANTS"]["generation"])
    # kda_oproj_push alone: dsa_oproj_push also builds the slot (test_rubin_helix8_dsa_oproj_push).
    consts.update(dsa_oproj_push=False)
    consts.update(over)
    return consts


def _slots(consts: dict, push):
    """``_tp_reduce_slots`` and ``_oproj_push_slot`` over ``consts`` with recording stand-ins."""
    tree = _tree()
    ns = _constants(tree)
    calls, log = [], _Log()

    def mix_kernel(*args, **kwargs):
        calls.append(("finish", args, kwargs))
        return args[0], args[0]

    ns.update(
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        rubin_lamport_push=lambda ws, partial, one_shot, pdl: calls.append(("push", one_shot)),
        rubin_attn_res_lamport_add_rmsnorm=mix_kernel,
        rubin_lamport_row_push_desc=lambda ws: ("desc", ws),
        allreduce_module=lambda partial, *a: partial,
        logger=log,
    )
    _defs(tree, ("_MixW", "_OprojPush", "_tp_reduce_slots", "_oproj_push_slot"), ns)

    def add_norm(prefix, summed, bank, mw):
        calls.append(("add_norm",))
        return prefix, prefix

    tp = ns["_tp_reduce_slots"](consts, 1e-6, object(), push, {"add_norm": add_norm})
    return ns, tp, ns["_oproj_push_slot"](consts, 1e-6, push), calls, log


def _mix_inputs(rows: int, banked: int = 4):
    prefix = torch.zeros(rows, 8, dtype=torch.bfloat16)
    bank = torch.zeros(banked, rows, 8, dtype=torch.bfloat16)
    mw = SimpleNamespace(proj="proj", res_norm="res_norm", out_norm="out_norm")
    return prefix, bank, mw


def test_the_levers_are_launch_constants():
    ns = _constants(_tree())
    gen = ns["_CONSTANTS"]["generation"]
    for key, value in _KEYS.items():
        assert gen[key] is value, key
    assert ns["_KDA_OPROJ_PUSH"] is gen["kda_oproj_push"]
    assert gen["attn_res_pdl_max_m"] == 128
    # The push path is reduce_norm's: the workspace and the PDL mix it needs are on.
    assert gen["lamport_push_allreduce"] and gen["ar_push_amid"] and gen["ar_finish_fused"]
    assert gen["attn_res_pdl"] == 2 and gen["kda_oproj_quant_k32"]


@pytest.mark.parametrize("early_poll", [True, False])
def test_finish_norm_is_reduce_norms_mix_call_without_the_push(early_poll):
    consts = _gen(kda_oproj_push=True, oproj_ar_early_poll=early_poll)
    push = _push()
    ns, tp, slot, calls, log = _slots(consts, push)
    assert slot is not None and slot.max_rows == consts["attn_res_pdl_max_m"]
    assert slot.ws is push and slot.desc == ("desc", push)
    for rows in (1, _ONE_SHOT_ROWS, _ONE_SHOT_ROWS + 1, 54, 96, consts["attn_res_pdl_max_m"]):
        del calls[:]
        prefix, bank, mw = _mix_inputs(rows)
        tp["reduce_norm"](prefix, torch.zeros_like(prefix), bank, mw)
        assert [c[0] for c in calls] == ["push", "finish"], rows
        want = calls[1]
        del calls[:]
        slot.finish_norm(prefix, bank, mw)
        assert [c[0] for c in calls] == ["finish"], rows
        got = calls[0]
        assert len(got[1]) == len(want[1]) and got[2] == want[2], rows
        for a, b in zip(got[1], want[1]):
            assert a is b or a == b, rows
    line = ns["_OPROJ_EARLY_POLL_LINE"]
    assert set(log.once) == ({line} if early_poll else set())


def test_no_push_slot_without_the_key_the_workspace_or_the_pdl_mix():
    for consts, push in (
        (_gen(kda_oproj_push=False), _push()),
        (_gen(kda_oproj_push=True), None),
        (_gen(kda_oproj_push=True, attn_res_pdl=1), _push()),
    ):
        _, _, slot, _, _ = _slots(consts, push)
        assert slot is None


def _verify_ns(on: bool, slot: Optional[Any], fused: bool = True):
    """``kda_verify`` from modeling.py with recording stand-ins; ``slot`` is ``rt.oproj_push``."""
    tree = _tree()
    ns = _constants(tree)
    calls, log = [], _Log()
    ns.update(
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        logger=log,
        _KDA_OPROJ_K32=True,
        _KDA_OPROJ_PUSH=on,
    )
    _defs(tree, ("_Pushed", "_kda_split", "kda_verify"), ns)

    def core(L, x, st, rt, mx_out=False, mx=None):
        calls.append(("verify", mx_out))
        return torch.zeros(x.shape[0], 4), "scales"

    def gemm(codes, scales, w, s):
        calls.append(("gemm",))
        return "partial"

    def gemm_push(codes, scales, w, s, ws, desc, one_shot, use_pdl):
        calls.append(("gemm_push", ws, desc, one_shot, use_pdl))
        return fused

    ns.update(
        _kda_verify_core=core,
        rubin_mxfp8_linear_prequantized=gemm,
        rubin_mxfp8_linear_prequantized_push=gemm_push,
        rubin_mxfp8_linear=None,
    )
    rt = SimpleNamespace(oproj_push=slot)
    L = SimpleNamespace(o_w="o_w", o_s="o_s")
    return ns, rt, L, calls, log


@pytest.mark.parametrize("fused", [True, False])
def test_kda_verify_pushes_on_the_push_rows(fused):
    push = _push()
    slot = SimpleNamespace(ws=push, desc="desc", max_rows=128, finish_norm=None)
    ns, rt, L, calls, log = _verify_ns(True, slot, fused)
    for rows in (1, _ONE_SHOT_ROWS + 1, 54, 102, 108, 128):
        del calls[:]
        out = ns["kda_verify"](L, torch.zeros(rows, 8), None, rt, True)
        assert isinstance(out, ns["_Pushed"]) and out.rows == rows
        one_shot = rows <= _ONE_SHOT_ROWS
        assert calls == [("verify", True), ("gemm_push", push.view, "desc", one_shot, True)]
        # Without the decoder's push the same rows run the plain GEMM.
        del calls[:]
        assert ns["kda_verify"](L, torch.zeros(rows, 8), None, rt, False) == "partial"
        assert calls == [("verify", True), ("gemm",)]
    if fused:
        assert set(log.once) == {ns["_KDA_OPROJ_K32_LINE"], ns["_KDA_OPROJ_PUSH_LINE"]}
        assert log.warned == []
    else:
        assert set(log.once) == {ns["_KDA_OPROJ_K32_LINE"]}
        assert set(log.warned) == {ns["_KDA_OPROJ_PUSH_FALLBACK_LINE"]}


def test_kda_verify_keeps_the_plain_gemm_off_or_past_the_push_rows():
    slot = SimpleNamespace(ws=_push(), desc="desc", max_rows=128, finish_norm=None)
    for on, s in ((False, slot), (True, None)):
        ns, rt, L, calls, log = _verify_ns(on, s)
        for rows in (6, 54, 128):
            del calls[:]
            assert ns["kda_verify"](L, torch.zeros(rows, 8), None, rt, True) == "partial"
            assert calls == [("verify", True), ("gemm",)]
        assert ns["_KDA_OPROJ_PUSH_LINE"] not in log.once
    ns, rt, L, calls, log = _verify_ns(True, slot)
    assert ns["kda_verify"](L, torch.zeros(144, 8), None, rt, True) == "partial"
    assert calls == [("verify", True), ("gemm",)]


def test_the_verify_takes_the_early_release_constant():
    tree = _tree()
    (core,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_kda_verify_core"
    ]
    (call,) = [
        n
        for n in ast.walk(core)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "rubin_kda_mtp_verify"
    ]
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kwargs["early_trigger"] == "_KDA_EARLY_TRIGGER"
    assert kwargs["use_pdl"] == "pdl"


def test_the_decoder_finishes_a_pushed_partial_in_the_mix():
    tree = _tree()
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget"]
    # The layer loop (moe_shared_late runs it on its own stream on a band step).
    (fwd,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run_layers"]
    text = ast.unparse(fwd)
    # Non-boundary layers with snapshot rows; a boundary only under boundary_ar_push (its check).
    assert "push = not L.boundary and L.n_bank > 0 or bmode == 'push'" in text
    assert "o = kda_verify(L.kda, x, st, rt, push)" in text
    branches = [n for n in ast.walk(fwd) if isinstance(n, ast.If)]
    pushed = [b for b in branches if ast.unparse(b.test) == "isinstance(o, _Pushed)"]
    (pushed,) = [
        b
        for b in pushed
        if [ast.unparse(s) for s in b.orelse]
        == ["prefix, x = rt.reduce_norm(prefix, o, bank[:L.n_bank], L.mid)"]
    ]
    assert [ast.unparse(s) for s in pushed.body] == [
        "prefix, x = rt.oproj_push.finish_norm(prefix, bank[:L.n_bank], L.mid)"
    ]
    # The runtime builds the slot from the same constants and push workspace as reduce_norm.
    (build,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_runtime"
    ]
    build_text = ast.unparse(build)
    assert "oproj_push = _oproj_push_slot(c, core.eps, push)" in build_text
    assert "oproj_push=oproj_push" in build_text


def test_the_info_line():
    ns = _constants(_tree())
    line = ns["_KDA_OPROJ_PUSH_LINE"]
    assert line.startswith("Kimi K3.1 Rubin KDA: ") and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()

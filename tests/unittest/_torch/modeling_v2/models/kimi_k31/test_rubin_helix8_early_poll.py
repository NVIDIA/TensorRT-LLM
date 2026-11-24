# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's o_proj all-reduce with ``oproj_ar_early_poll``: the PDL mix after the pushed partials
finishes their all-reduce before its grid wait.

``_tp_reduce_slots``'s ``reduce_norm`` pushes a decode step's o_proj partial of at most
``attn_res_pdl_max_m`` rows into the Lamport push workspace and hands the finish to the mix kernel
(``rubin_attn_res_lamport_add_rmsnorm``). ``early_poll`` moves where that kernel waits for its grid:
it polls and sums the pushed slots in rank order and computes the mix before
``griddepcontrol.wait``, and stores after it. The finish and the mix are the same code either way,
so the outputs are those without it bit for bit (``rubin_multirank/test_rubin_rp_tail.py`` compares
both settings with the C++ all-reduce on the GPU, through tep8's slot).

CPU (``cpu_only``), from helix8's modeling.py over its constants:

* helix8's generation constants set ``oproj_ar_early_poll`` with the push workspace and the PDL
  mix it needs;
* ``reduce_norm`` passes ``early_poll`` to the mix kernel for 1 to ``attn_res_pdl_max_m`` rows, on
  both transports, and logs the README's INFO line through ``logger.info_once`` only then; above
  that row count, without snapshot rows or without the push workspace it runs the plain
  all-reduce and the add-and-mix slot and logs nothing; with the key off it passes False and logs
  nothing;
* helix8's ``_tp_reduce_slots`` is tep8's but for the docstring and the INFO statement, and both
  import the kernels it calls from the same modules.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_TEP8 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__tep8" / "modeling.py"
_MIX_OP = _FAMILY / "ops/norm/rubin_attn_res_lamport_add_rmsnorm.py"
_README = _FAMILY / "README.md"

#: Rows up to which the stand-in workspace takes the one-shot transport (two-shot above).
_ONE_SHOT_ROWS = 16
_LINE_PREFIX = "Kimi K3.1 Rubin o_proj all-reduce: "


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order (as the
    helix8 MoE front test reads them)."""
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


def _mix_params() -> tuple[list, list]:
    """The mix kernel entry's positional and required keyword-only parameters, read off its source
    (the optional ones, ``updated`` and ``replace``, keep their defaults on this path)."""
    for node in ast.walk(_tree(_MIX_OP)):
        if isinstance(node, ast.ClassDef) and node.name == "RubinAttnResLamportAddRmsnorm":
            (call,) = [
                f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == "__call__"
            ]
            kw = [
                a.arg
                for a, default in zip(call.args.kwonlyargs, call.args.kw_defaults)
                if default is None
            ]
            return [a.arg for a in call.args.args[1:]], kw
    raise AssertionError("RubinAttnResLamportAddRmsnorm.__call__ not found")


class _Log:
    def __init__(self):
        self.once = []

    def info_once(self, msg, key=None):
        self.once.append((msg, key))


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


def _slots(consts: dict, push):
    """helix8's ``_tp_reduce_slots`` from its modeling.py over its constants, with recording
    stand-ins for the push, the mix kernel, the all-reduce, the add-and-mix slot and the logger."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    calls, log = [], _Log()
    pos, kw = _mix_params()

    def mix_kernel(*args, **kwargs):
        assert len(args) == len(pos) and sorted(kwargs) == sorted(kw), (len(args), sorted(kwargs))
        calls.append(("finish", {**dict(zip(pos, args)), **kwargs}))
        return args[0], args[0]

    def push_kernel(ws, partial, one_shot, pdl):
        calls.append(("push", one_shot, pdl, ws is push.view))

    def all_reduce(partial, module, op, *rest):
        calls.append(("all_reduce", op))
        return partial

    ns.update(
        rubin_lamport_push=push_kernel,
        rubin_attn_res_lamport_add_rmsnorm=mix_kernel,
        allreduce_module=all_reduce,
        logger=log,
    )
    fn = copy.deepcopy(_function(tree, "_tp_reduce_slots"))
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)

    def add_norm(prefix, summed, bank, mw):
        calls.append(("add_norm",))
        return prefix, prefix

    slots = ns["_tp_reduce_slots"](consts, 1e-6, object(), push, {"add_norm": add_norm})
    return ns, slots, calls, log


def _run(slots, rows: int, banked: int):
    prefix = torch.zeros(rows, 8, dtype=torch.bfloat16)
    bank = torch.zeros(banked, rows, 8, dtype=torch.bfloat16)
    mw = SimpleNamespace(proj="proj", res_norm="res_norm", out_norm="out_norm")
    return slots["reduce_norm"](prefix, torch.zeros_like(prefix), bank, mw)


def test_helix8_sets_the_early_poll_with_the_push_and_the_pdl_mix():
    c = _constants(_tree(_MODELING))["_CONSTANTS"]["generation"]
    assert c["oproj_ar_early_poll"] is True
    # _build_runtime builds the push workspace only with these three; the PDL mix takes up to
    # attn_res_pdl_max_m rows (attn_res_pdl 2), a b16 k5 decode step's 96 among them.
    assert c["lamport_push_allreduce"] and c["ar_push_amid"] and c["ar_finish_fused"]
    assert c["attn_res_pdl"] == 2 and c["attn_res_pdl_max_m"] >= 96


@pytest.mark.parametrize("on", [True, False], ids=["early_poll", "key_off"])
def test_reduce_norm_finishes_before_the_grid_wait(on):
    consts = dict(_constants(_tree(_MODELING))["_CONSTANTS"]["generation"])
    consts["oproj_ar_early_poll"] = on
    ns, slots, calls, log = _slots(consts, _push())
    line = ns["_OPROJ_EARLY_POLL_LINE"]
    sizes = (1, _ONE_SHOT_ROWS, _ONE_SHOT_ROWS + 1, 96, consts["attn_res_pdl_max_m"])
    for rows in sizes:
        del calls[:]
        _run(slots, rows, 4)
        assert [c[0] for c in calls] == ["push", "finish"], (rows, calls)
        one_shot = rows <= _ONE_SHOT_ROWS
        assert calls[0][1:] == (one_shot, True, True)
        args = calls[1][1]
        assert args["early_poll"] is on, rows
        assert args["finish_world"] == 8 and args["finish_two_shot"] is (not one_shot)
        assert args["finish_chunk"] == (1 if one_shot else 2) and args["rank"] == 3
        assert (args["uc_local"], args["flags"], args["mc_base"]) == (11, 12, 13)
        assert (args["add_rn"], args["entry_trigger"], args["finish_ag"]) == (False, False, False)
        assert args["early_trigger"] is bool(consts["attn_res_early_trigger"])
        assert (args["res_weight"], args["rms_weight"], args["out_weight"]) == (
            "proj",
            "res_norm",
            "out_norm",
        )
    # The stand-in logger records every call; the engine's info_once logs the first per process.
    assert log.once == ([(line, line)] * len(sizes) if on else [])


def test_reduce_norm_without_the_early_poll_path():
    """Above attn_res_pdl_max_m rows, without snapshot rows or without the push workspace: the
    plain all-reduce and the add-and-mix slot, no finish in the mix and no INFO line."""
    consts = dict(_constants(_tree(_MODELING))["_CONSTANTS"]["generation"])
    max_m = consts["attn_res_pdl_max_m"]
    for push, rows, banked in ((_push(), max_m + 1, 4), (_push(), 96, 0), (None, 96, 4)):
        ns, slots, calls, log = _slots(consts, push)
        _run(slots, rows, banked)
        assert calls == [("all_reduce", ns["_AR_NONE"]), ("add_norm",)], (rows, banked, calls)
        assert log.once == []


def _strip(fn: ast.FunctionDef) -> str:
    """``fn`` without docstrings and without an ``if early_poll: logger.info_once(...)``."""
    fn = copy.deepcopy(fn)
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.FunctionDef)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]

    class _DropInfo(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            text = [ast.unparse(s) for s in node.body]
            if (
                ast.unparse(node.test) == "early_poll"
                and not node.orelse
                and text == ["logger.info_once(_OPROJ_EARLY_POLL_LINE, key=_OPROJ_EARLY_POLL_LINE)"]
            ):
                return None
            return node

    return ast.dump(_DropInfo().visit(fn))


def _imported_from(tree: ast.Module, names: tuple) -> dict:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) in names:
                    found[alias.asname or alias.name] = (node.level, node.module, alias.name)
    return found


def test_helix8_slot_is_tep8s():
    h8_tree, tep8_tree = _tree(_MODELING), _tree(_TEP8)
    h8, tep8 = _function(h8_tree, "_tp_reduce_slots"), _function(tep8_tree, "_tp_reduce_slots")
    assert _strip(h8) == _strip(tep8)
    info = [ast.unparse(n) for n in ast.walk(h8) if isinstance(n, ast.Call)]
    info = [t for t in info if t.startswith("logger.info")]
    assert info == ["logger.info_once(_OPROJ_EARLY_POLL_LINE, key=_OPROJ_EARLY_POLL_LINE)"]
    names = ("allreduce_module", "rubin_lamport_push", "rubin_attn_res_lamport_add_rmsnorm")
    h8_from, tep8_from = _imported_from(h8_tree, names), _imported_from(tep8_tree, names)
    assert set(h8_from) == set(names) and h8_from == tep8_from, (h8_from, tep8_from)


def test_the_info_line():
    line = _constants(_tree(_MODELING))["_OPROJ_EARLY_POLL_LINE"]
    assert line.startswith(_LINE_PREFIX) and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()

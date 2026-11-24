# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's mix-node levers on the CPU: ``mix_bank_in_mix``, ``draft_tap_stream``,
``boundary_ar_push`` and ``dense_ar_push`` (generation constants; the first and the third on).

Each moves graph nodes and keeps the bytes (the GPU tests compare them:
``ops/norm/test_rubin_attn_res_moe_add_rmsnorm.py``, the ``updated_out`` and ``replace`` cells of
``rubin_attn_res_lamport_add_rmsnorm`` in ``rubin_multirank/``). From helix8's modeling.py over
recording stand-ins:

* the keys are generation launch constants at their shipped states, and the module-level copy
  reads the table;
* ``_mix_in`` hands a block boundary's bank row to the MoE mix slot only when it is given one, and
  never on a tapped layer; the slots pass it to the PDL mixes as ``updated``;
* ``_mix_in`` sends a ``_DensePartial`` through ``reduce_norm`` (tapped: the all-reduce, then the
  add), the plain dense output through ``add_norm`` as before;
* ``dense_mlp`` returns the partial only under ``rt.dense_push``;
* ``_TapStream`` forks after the tapped rows, copies on its stream, keeps the rows alive there,
  and joins once; ``_capture`` copies on the current stream without it;
* ``_BoundaryPush``: the mix call of the replace finish, the shapes it serves, the device check's
  verdict agreed over the group, and no check inside a capture;
* ``boundary_ar_push`` without the KDA verify's o_proj epilogue push (``kda_oproj_push`` and
  ``kda_oproj_quant_k32``) refuses the import, ``dsa_oproj_push`` alone included;
* the decoder: the snapshot copy only where the mix did not store it, the boundary push only on a
  checked shape;
* the README carries the lines.
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
import test_rubin_helix8_moe_cp_tail as T  # noqa: E402 -- the mix slots' recording stand-ins
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"
_KEYS = ("mix_bank_in_mix", "draft_tap_stream", "boundary_ar_push", "dense_ar_push")
_LINES = ("_MIX_BANK_LINE", "_TAP_STREAM_LINE", "_BOUNDARY_PUSH_LINE", "_DENSE_PUSH_LINE")


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


def _defs(names: tuple, ns: dict) -> dict:
    """Execute the module-level definitions ``names`` of modeling.py into ``ns``."""
    nodes = [
        copy.deepcopy(n)
        for n in _tree().body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert sorted(n.name for n in nodes) == sorted(names)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _base_ns(log=None) -> dict:
    ns = _constants(_tree())
    ns.update(
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        logger=log or SimpleNamespace(info_once=lambda *a, **k: None),
    )
    return ns


#: The shipped stack's states: the bank snapshot by its producer and the boundary push on (both
#: faster at world 8), the tap stream (under its gate) and the dense push (under its gate) off.
_SHIPPED = {
    "mix_bank_in_mix": True,
    "draft_tap_stream": False,
    "boundary_ar_push": True,
    "dense_ar_push": False,
}


def test_the_levers_are_launch_constants_as_shipped():
    ns = _constants(_tree())
    gen = ns["_CONSTANTS"]["generation"]
    assert set(_SHIPPED) == set(_KEYS)
    for key in _KEYS:
        assert gen[key] is _SHIPPED[key], key
    assert ns["_MIX_BANK_IN_MIX"] is gen["mix_bank_in_mix"]
    # boundary_ar_push rides on kda_oproj_push's epilogue push and the PDL mix's rows.
    assert gen["kda_oproj_push"] and gen["attn_res_pdl"] == 2 and gen["kda_oproj_quant_k32"]


# --- _mix_in: the bank row, the dense partial, the tap ---


def _mix_ns() -> dict:
    ns = _base_ns()
    return _defs(("_MoeOut", "_MoeTail", "_DensePartial", "_capture", "_mix_in"), ns)


def _rt(calls: list, taps=None) -> SimpleNamespace:
    def slot(name, result):
        def call(*args):
            calls.append((name, args))
            return result

        return call

    return SimpleNamespace(
        attn_res_norm=slot("attn_res_norm", "n"),
        add_norm=slot("add_norm", ("u", "n")),
        reduce_norm=slot("reduce_norm", ("r", "n")),
        reduce=slot("reduce", 5.0),
        moe_norm=slot("moe_norm", ("u", "n")),
        moe_fold_norm=slot("moe_fold_norm", ("u", "n")),
        moe_norm_mx=slot("moe_norm_mx", ("u", "n")),
        moe_add=slot("moe_add", "u"),
        attn_res_mix=slot("attn_res_mix", "mix"),
        rms_norm=slot("rms_norm", "n"),
        taps=taps,
    )


@pytest.mark.parametrize("kda", [False, True], ids=["mla", "kda"])
def test_mix_in_hands_the_bank_row_to_the_moe_slot_only_when_given(kda):
    ns = _mix_ns()
    moe = ns["_MoeOut"]("r", "s")
    calls = []
    ns["_mix_in"](_rt(calls), "p", moe, "bank", "mw", None, kda)
    ns["_mix_in"](_rt(calls), "p", moe, "bank", "mw", None, kda, "row")
    name = "moe_norm_mx" if kda else "moe_fold_norm"
    assert calls == [(name, ("p", moe, "bank", "mw")), (name, ("p", moe, "bank", "mw", "row"))]
    # A tapped layer runs apart: no bank row reaches a slot.
    calls.clear()
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: calls.append(("tap", a))), 3)
    out = ns["_mix_in"](_rt(calls), "p", moe, "bank", SimpleNamespace(out_norm="o"), tap, kda, "x")
    assert [c[0] for c in calls] == ["moe_add", "attn_res_mix", "tap", "rms_norm"]
    assert out == ("u", "n")


def test_mix_in_sums_a_dense_partial_through_reduce_norm():
    ns = _mix_ns()
    part = ns["_DensePartial"](torch.ones(2, 4))
    calls = []
    assert ns["_mix_in"](_rt(calls), "p", part, "bank", "mw", None, True) == ("r", "n")
    assert calls == [("reduce_norm", ("p", part.partial, "bank", "mw"))]
    # The plain dense output (the lever off) keeps add_norm.
    calls.clear()
    out = torch.ones(2, 4)
    ns["_mix_in"](_rt(calls), "p", out, "bank", "mw", None, True)
    assert calls == [("add_norm", ("p", out, "bank", "mw"))]
    # Tapped: the all-reduce, then today's add.
    calls.clear()
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: calls.append(("tap", a))), 3)
    prefix = torch.zeros(2, 4)
    got, _ = ns["_mix_in"](_rt(calls), prefix, part, "bank", SimpleNamespace(out_norm="o"), tap)
    assert [c[0] for c in calls] == ["reduce", "attn_res_mix", "tap", "rms_norm"]
    assert torch.equal(got, prefix + 5.0)


# --- the slots: updated reaches the PDL mixes ---


@pytest.mark.parametrize("rows", [12, 24, 54, 66])
def test_the_tail_mixes_store_into_the_given_row(rows):
    calls, lines = [], []
    ns = T._slots_ns(calls, lines, set())
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True)
    slots = ns["_mix_slots"](gen, 1e-5)
    tail = T._tail(ns)
    prefix, bank, mw = T._inputs(rows, 3)
    row = torch.empty(rows, T.F._HIDDEN, dtype=torch.bfloat16)
    for slot in ("moe_norm", "moe_fold_norm", "moe_norm_mx"):
        calls.clear()
        got = slots[slot](prefix, tail, bank, mw, row)
        assert got[0] is row, slot
        assert ("updated", row) in calls, slot
        # Without a row: today's call, no updated keyword.
        calls.clear()
        slots[slot](prefix, tail, bank, mw)
        assert all(c[0] != "updated" for c in calls), slot


def test_the_moe_mixes_pass_the_row_to_their_kernels():
    """``_MoeOut`` rows on the PDL kernel: the fused MoE mixes take ``updated``; the persistent
    kernels do not (their caller copies)."""
    seen = []
    ns = T._slots_ns([], [], set())

    def kernel(name, result):
        def call(*args, **kwargs):
            seen.append((name, kwargs.get("updated")))
            return result

        return call

    ns["rubin_attn_res_moe_add_rmsnorm"] = kernel("pdl_moe", ("u", "n"))
    ns["rubin_attn_res_moe_add_rmsnorm_mx"] = kernel("pdl_moe_mx", ("u", "n", "c", "s"))
    gen = dict(ns["_CONSTANTS"]["generation"])
    slots = ns["_mix_slots"](gen, 1e-5)
    moe = ns["_MoeOut"]("r", "s")
    _, bank, mw = T._inputs(96, 3)
    prefix = torch.ones(96, T.F._HIDDEN, dtype=torch.bfloat16)
    slots["moe_norm"](prefix, moe, bank, mw, "row")
    slots["moe_norm_mx"](prefix, moe, bank, mw, "row")
    assert seen == [("pdl_moe", "row"), ("pdl_moe_mx", "row")]


# --- dense_mlp ---


def test_dense_mlp_returns_the_partial_only_under_dense_push():
    ns = _base_ns(SimpleNamespace(info_once=lambda *a, **k: None))
    ns.update(
        _gemm_pinned=lambda site, x, w, fn: fn(),
        cublas_mm=lambda a, b: "out",
        situ_and_mul=lambda gu, *a: "h",
        _MOE_SITU=(),
        allreduce_module=lambda out, ar, *a: ("summed", out),
        _MOE_AR_NONE=0,
    )
    _defs(("_DensePartial", "dense_mlp"), ns)
    L0 = SimpleNamespace(gu=torch.zeros(2, 2), dn=torch.zeros(2, 2))
    for push, ar, want in (
        (False, "ar", ("summed", "out")),
        (True, "ar", ns["_DensePartial"]("out")),
        (True, None, "out"),
        (False, None, "out"),
    ):
        rt = SimpleNamespace(moe_comm=SimpleNamespace(dense_all_reduce=ar), dense_push=push)
        assert ns["dense_mlp"](L0, "x", None, rt) == want


# --- _TapStream ---


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))


class _Event:
    def __init__(self, name, log, current):
        self.name, self.log, self.current = name, log, current

    def record(self):
        self.log.append(("record", self.name, self.current[0].name))


def _tap_ns(log):
    main = _Stream("main", log)
    current = [main]

    class _Ctx:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.prev = current[0]
            current[0] = self.stream

        def __exit__(self, *exc):
            current[0] = self.prev

    cuda = SimpleNamespace(
        current_stream=lambda: current[0],
        stream=_Ctx,
        Event=lambda: _Event("done", log, current),
        Stream=_Stream,
    )
    ns = _base_ns(SimpleNamespace(info_once=lambda m, key=None: log.append(("line", m))))
    ns["torch"] = SimpleNamespace(cuda=cuda, Tensor=torch.Tensor)
    _defs(("_TapStream", "_capture"), ns)
    return ns, main, current


def test_the_tap_stream_forks_copies_keeps_the_rows_and_joins_once():
    log = []
    ns, main, current = _tap_ns(log)
    taps = ns["_TapStream"](_Stream("tap", log))
    rows = SimpleNamespace(record_stream=lambda s: log.append(("keep", s.name)))
    meta = SimpleNamespace(
        maybe_capture_hidden_states=lambda i, r, res: log.append(("copy", i, current[0].name))
    )
    ns["_capture"](SimpleNamespace(taps=taps), (meta, 8), rows)
    assert log[:4] == [
        ("wait_stream", "tap", "main"),
        ("copy", 8, "tap"),
        ("record", "done", "tap"),
        ("keep", "tap"),
    ]
    assert taps.pending and log[4] == ("line", ns["_TAP_STREAM_LINE"])
    log.clear()
    aux = _Stream("aux", log)
    taps.join(aux)
    taps.join()  # joined already: nothing
    assert log == [("wait_event", "aux", "done")] and not taps.pending
    # Without the lever the copy runs on the current stream.
    log.clear()
    ns["_capture"](SimpleNamespace(taps=None), (meta, 9), rows)
    assert log == [("copy", 9, "main")]


# --- _BoundaryPush ---


class _Flags:
    """``partial.new_zeros(1, H)`` stand-in: the mismatch flag."""

    def __init__(self):
        self.value = 0.0

    def __setitem__(self, idx, value):
        self.value = value


def _boundary(votes: float, capturing: bool = False):
    calls, log = [], SimpleNamespace(info_once=lambda *a, **k: None)
    log.info = lambda m: calls.append(("info", m))
    log.warning = lambda m: calls.append(("warning", m))
    ns = _base_ns(log)

    def mix(*args, **kwargs):
        calls.append(("mix", args, kwargs))
        return torch.ones(args[0].shape[0], 2), torch.ones(args[0].shape[0], 2)

    def reduce(x):
        if isinstance(x, _Flags):
            calls.append(("vote", x.value))
            return torch.tensor([[votes]])
        calls.append(("reduce",))
        return torch.ones(x.shape[0], 2)

    ns.update(
        rubin_attn_res_lamport_add_rmsnorm=mix,
        rubin_lamport_push=lambda ws, part, one_shot, pdl: calls.append(("push", one_shot, pdl)),
    )
    ns["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing),
        equal=torch.equal,
        int16=torch.int16,
        Tensor=torch.Tensor,
    )
    _defs(("_MixW", "_BoundaryPush"), ns)
    view = SimpleNamespace(
        uc_local=11, flags=SimpleNamespace(data_ptr=lambda: 12), world=8, mc_base=13, rank=3
    )
    push = SimpleNamespace(view=view, transport=lambda rows: (rows <= 9, 8), pdl=True)
    consts = {"oproj_ar_early_poll": True, "attn_res_early_trigger": True}
    norm = lambda p, b, mw: torch.ones(p.shape[0], 2)  # noqa: E731
    bp = ns["_BoundaryPush"](push, 128, reduce, norm, consts, 1e-5)
    return ns, bp, calls


class _Partial:
    def __init__(self, rows):
        self.shape = (rows, 2)

    def contiguous(self):
        return self

    def new_zeros(self, *shape):
        return _Flags()


def test_the_boundary_finish_is_the_replace_mix_call():
    ns, bp, calls = _boundary(0.0)
    mw = ns["_MixW"]("proj", "res", "out")
    for rows, one_shot in ((6, True), (54, False)):
        calls.clear()
        bp.finish(torch.zeros(rows, 2), "bank", mw)
        ((name, args, kwargs),) = calls
        assert name == "mix" and args[1:7] == (11, 12, "bank", "proj", "res", "out")
        assert args[9:] == (8, 8, not one_shot, 13, 3, False, True, False, False)
        assert kwargs == dict(early_trigger=True, replace=True)


def test_the_boundary_push_serves_checked_shapes_only():
    _, bp, _ = _boundary(0.0)
    assert bp.mode(0, 2) is None and bp.mode(129, 2) is None
    assert bp.mode(54, 9) == "check"
    bp.verdict[(54, 9)] = True
    bp.verdict[(24, 9)] = False
    assert bp.mode(54, 9) == "push" and bp.mode(24, 9) is None
    _, captured, _ = _boundary(0.0, capturing=True)
    assert captured.mode(54, 9) is None  # a capture before the check keeps today's calls


@pytest.mark.parametrize("votes", [0.0, 1.0, 3.0])
def test_the_check_runs_both_paths_and_agrees_on_the_verdict(votes):
    ns, bp, calls = _boundary(votes)
    mw = ns["_MixW"]("proj", "res", "out")
    want = bp.check(_Partial(24), torch.zeros(24, 2), torch.zeros(4, 24, 2), mw)
    names = [c[0] for c in calls]
    assert names[:3] == ["reduce", "push", "mix"] and names[3] == "vote"
    assert calls[3][1] == 0.0  # this rank's words matched (all-ones stand-ins)
    assert bp.verdict == {(24, 5): votes == 0.0}
    assert names[4] == ("info" if votes == 0.0 else "warning")
    assert torch.equal(want[0], torch.ones(24, 2))


# --- the import check: the boundary push finishes the KDA verify's o_proj push ---


def _boundary_import_check(**values) -> None:
    """Run modeling.py's import-time check of ``boundary_ar_push`` with the generation constants
    ``values`` over the source's (``T._source``)."""
    tree = ast.parse(T._source(**values), filename=str(_MODELING))
    ns = T._constants(tree)
    (check,) = [
        n for n in tree.body if isinstance(n, ast.If) and "boundary_ar_push" in ast.unparse(n.test)
    ]
    module = ast.fix_missing_locations(ast.Module(body=[check], type_ignores=[]))
    exec(compile(module, str(_MODELING), "exec"), ns)


@pytest.mark.parametrize(
    "values",
    [
        {},
        dict(boundary_ar_push=False, kda_oproj_push=False),
        dict(boundary_ar_push=False, kda_oproj_quant_k32=False),
    ],
    ids=["shipped", "off_without_kda_push", "off_without_k32"],
)
def test_the_boundary_push_imports_with_the_kda_oproj_push(values):
    _boundary_import_check(**values)


@pytest.mark.parametrize(
    "values",
    [
        # dsa_oproj_push (on) alone builds the push slot; no KDA verify o_proj would push.
        dict(kda_oproj_push=False),
        dict(kda_oproj_push=False, dsa_oproj_push=False),
        dict(kda_oproj_quant_k32=False),
    ],
    ids=["dsa_push_alone", "no_push", "no_k32"],
)
def test_the_boundary_push_without_the_kda_oproj_push_refuses_to_import(values):
    with pytest.raises(NotImplementedError, match="boundary_ar_push needs kda_oproj_push"):
        _boundary_import_check(**values)


# --- the decoder ---


def test_the_decoder_copies_the_snapshot_only_where_the_mix_did_not_store_it():
    (cls,) = [n for n in _tree().body if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget"]
    # The embedding and the layer loop (moe_shared_late runs them on its own stream on a band step).
    (fwd,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run_layers"]
    text = ast.unparse(fwd)
    assert "snap = bank[L.n_bank] if L.boundary and _MIX_BANK_IN_MIX else None" in text
    assert "if snap is None or prefix.data_ptr() != snap.data_ptr():" in text
    assert "h = torch.index_select(embed, 0, input_ids, out=bank[0])" in text
    assert "bmode = bp.mode(prefix.shape[0], L.n_bank + 1) if L.boundary and bp else None" in text
    assert "prefix, x = bp.finish(prefix, bank[:L.n_bank + 1], L.mid)" in text
    assert "prefix, x = bp.check(o, prefix, bank[:L.n_bank + 1], L.mid)" in text
    # Off: today's embedding and copy.
    assert "nn.functional.embedding(input_ids, head.embed)" in text
    assert "bank[L.n_bank].copy_(prefix)" in text


def test_the_lines():
    ns = _constants(_tree())
    readme = _README.read_text()
    for name in _LINES:
        line = ns[name]
        assert line.startswith("Kimi K3.1 Rubin ") and "\n" not in line
        assert f"{line}\n" in readme, name

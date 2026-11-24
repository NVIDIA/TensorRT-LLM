# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's GEMM pins: BF16 generation GEMMs at a pinned cuBLASLt algorithm at measured row counts.

CPU (``cpu_only``), from modeling.py's source:

* each site group's switch (``gemm_pins_*``) is a generation constant, on (the tests
  below read the source with every group on, unless one is switched off); the tables pin only the
  measured row counts (6 rows a request: b2..b4, b9..b12, b21, b24), and each pin is a cell of the
  ``rubin_lt_gemm`` entry at the site's shape, with the call it replaces (B-class cells write its
  words, the A-class table's cells do not); a group switched off pins nothing; gemm_pins_a_class
  puts the A-class entries over the table's, for the groups that are on;
* every site calls ``_gemm_pinned`` where it called its GEMM, with that call as ``today`` and the
  same weight's ``.t()`` as ``b``;
* ``_gemm_pinned`` over recording stand-ins: an unpinned row count runs today's call only; the
  first call of a pinned site and row count outside capture runs today's call, then the entry into
  a new tensor of today's output layout, compares the words, returns today's output, records the
  outcome and logs it once; later calls run the entry alone, into a new ``[rows, N]`` tensor of the
  output dtype on the current stream (or into the given ``out``); a capture before the check runs
  today's call and records nothing; a pin whose words differ (a sign of zero included) or that the
  entry cannot run is logged as a WARNING, and today's call runs from then on; a later call the
  entry cannot run (another operand layout) runs today's call, with one WARNING; an A-class pin is
  checked against a band of today's largest magnitude instead (its own INFO line; outside the band
  or not finite is a WARNING);
* the binding loads only a cuBLAS 13 libcublasLt (its enums and struct sizes are 13's).
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import re
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import rubin_lt_pin
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_lt_gemm import (
    LtPin,
    LtUnavailable,
    rubin_lt_gemm,
)

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MODELING = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__helix8"
    / "modeling.py"
)
#: Each pinned site: (N, K) of its GEMM at helix8's share, the call it replaces (the entry cells'
#: ``today``), the functions that call it, and the callee of its ``today``.
_SITES = {
    "moe_latent_up": (
        (7168, 3584),
        "mm",
        ("_moe_latent_up", "_moe_tp_gen", "_moe_tp_gen_streams"),
        "_moe_latent_gemm",
    ),
    "moe_latent_down": (
        (3584, 7168),
        "mm",
        ("_moe_tp_gen", "_moe_tp_gen_streams", "_moe_tp_gen_tail", "_moe_tp_gen_streams_tail"),
        "_moe_latent_gemm",
    ),
    "moe_router": ((896, 7168), "router", ("_moe_router",), "dsv3_router_gemm_op"),
    "mla_qkv_a": ((2112, 7168), "mm", ("_mla_project",), "cublas_mm"),
    "mla_wuv": ((128, 512), "uv", ("_mla_expand",), "bmm_out"),
    "mla_o": ((7168, 1536), "linear", ("_mla_output", "_dsa_oproj_push"), "rubin_linear"),
    "mla_idx_wq_b": ((4096, 1536), "mm", ("_idx_queries",), "cublas_mm"),
    "dense_gu": ((8448, 7168), "mm", ("dense_mlp",), "cublas_mm"),
    "dense_dn": ((7168, 4224), "mm", ("dense_mlp",), "cublas_mm"),
    "draft_q_b": ((18432, 1536), "linear", ("draft_step", "_draft_step_fused"), "rubin_linear"),
    "draft_o": ((7168, 1536), "linear", ("draft_step", "_draft_step_fused"), "rubin_linear"),
    "draft_gate_up": ((3584, 7168), "linear", ("draft_step", "_draft_step_fused"), "rubin_linear"),
    "draft_down": ((7168, 1792), "linear", ("draft_step", "_draft_step_fused"), "rubin_linear"),
    "draft_attn_conv_proj": (
        (1792, 7168),
        "linear",
        ("draft_step", "_draft_step_fused"),
        "rubin_linear",
    ),
    "draft_mlp_conv_proj": (
        (1792, 7168),
        "linear",
        ("draft_step", "_draft_step_fused"),
        "rubin_linear",
    ),
}
_MEASURED_ROWS = (12, 18, 24, 54, 60, 66, 72, 126, 144)


def _tree(**switches) -> ast.Module:
    """modeling.py's source with every ``gemm_pins_*`` group on, then ``switches`` applied."""
    text = re.sub(r'"(gemm_pins_\w+)": False,', r'"\1": True,', _MODELING.read_text())
    for name, value in switches.items():
        old = f'"{name}": {not value},'
        assert text.count(old) == 1, f"{name} is not one {not value} constant of the source"
        text = text.replace(old, f'"{name}": {value},')
    return ast.parse(text, filename=str(_MODELING))


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


def _functions(tree: ast.Module, names) -> dict:
    """``names`` defined over ``tree``'s constants and ``namespace`` stand-ins (see ``_block``)."""
    wanted = set(names)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted}


def _block(tree: ast.Module, namespace: dict) -> dict:
    nodes = [
        copy.deepcopy(n) for n in _functions(tree, ("_gemm_pin_check", "_gemm_pinned")).values()
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ns = {**_constants(tree), "Callable": Callable, "Optional": Optional, "Any": Any, **namespace}
    exec(compile(module, str(_MODELING), "exec"), ns)
    return ns


#: The shipped switches: the B-class groups on, the A-class pins off.
_SHIPPED = {
    "gemm_pins_moe": True,
    "gemm_pins_mla": True,
    "gemm_pins_dense": True,
    "gemm_pins_draft": True,
    "gemm_pins_a_class": False,
}


def test_each_group_switch_is_a_generation_constant_as_shipped():
    ns = _constants(ast.parse(_MODELING.read_text(), filename=str(_MODELING)))
    generation = ns["_CONSTANTS"]["generation"]
    assert set(ns["_GEMM_PIN_GROUPS"]) | {"gemm_pins_a_class"} == set(_SHIPPED)
    assert {key: generation[key] for key in _SHIPPED} == _SHIPPED
    want = {
        (site, rows): config
        for site, pins in ns["_GEMM_PIN_TABLE"].items()
        for rows, config in pins.items()
    }
    if _SHIPPED["gemm_pins_a_class"]:
        want.update(
            {
                (site, rows): config
                for site, pins in ns["_GEMM_PIN_TABLE_A"].items()
                for rows, config in pins.items()
            }
        )
    assert ns["_GEMM_PINS"] == want
    generation = _constants(_tree())["_CONSTANTS"]["generation"]
    assert all(generation[key] is True for key in _SHIPPED)
    sites = [s for group in ns["_GEMM_PIN_GROUPS"].values() for s in group]
    assert sorted(sites) == sorted(ns["_GEMM_PIN_TABLE"]) == sorted(_SITES)


def test_the_table_pins_only_measured_row_counts():
    ns = _constants(_tree())
    assert set(ns["_GEMM_PIN_TABLE_A"]) <= set(ns["_GEMM_PIN_TABLE"])
    for table in (ns["_GEMM_PIN_TABLE"], ns["_GEMM_PIN_TABLE_A"]):
        for site, pins in table.items():
            assert set(pins) <= set(_MEASURED_ROWS), (site, sorted(pins))
            for rows, config in pins.items():
                assert len(config) == len(LtPin._fields) and all(isinstance(v, int) for v in config)
                assert config[0] == 66, f"{site} at {rows}: not an NVJET algorithm"
    a_class = {(site, rows) for site, pins in ns["_GEMM_PIN_TABLE_A"].items() for rows in pins}
    assert ns["_GEMM_PINS_A_CLASS"] == a_class
    assert ns["_GEMM_PINS"] == {
        **{
            (site, rows): config
            for site, pins in ns["_GEMM_PIN_TABLE"].items()
            for rows, config in pins.items()
        },
        **{(site, rows): ns["_GEMM_PIN_TABLE_A"][site][rows] for site, rows in a_class},
    }


def test_every_pin_is_a_cell_of_the_entry_at_its_sites_shape():
    ns = _constants(_tree())
    cells = {
        (c.spec["rows"], c.spec["n"], c.spec["k"], c.spec["pin"], c.spec["today"], c.spec["exact"])
        for c in rubin_lt_gemm.CELLS
    }
    for table, exact in ((ns["_GEMM_PIN_TABLE"], True), (ns["_GEMM_PIN_TABLE_A"], False)):
        for site, pins in table.items():
            (n, k), today = _SITES[site][:2]
            for rows, config in pins.items():
                assert (rows, n, k, LtPin(*config), today, exact) in cells, (site, rows, exact)


def test_a_class_pins_need_their_switch_and_their_group():
    off = _constants(_tree(gemm_pins_a_class=False))
    assert off["_GEMM_PINS_A_CLASS"] == frozenset()
    assert off["_GEMM_PINS"] == {
        (site, rows): config
        for site, pins in off["_GEMM_PIN_TABLE"].items()
        for rows, config in pins.items()
    }
    moe_off = _constants(_tree(gemm_pins_moe=False))
    assert not [key for key in moe_off["_GEMM_PINS_A_CLASS"] if key[0].startswith("moe_")]
    assert [key for key in moe_off["_GEMM_PINS_A_CLASS"] if key[0].startswith("draft_")]


@pytest.mark.parametrize(
    "group", ["gemm_pins_moe", "gemm_pins_mla", "gemm_pins_dense", "gemm_pins_draft"]
)
def test_a_group_switched_off_pins_none_of_its_sites(group):
    ns = _constants(_tree(**{group: False}))
    sites = set(ns["_GEMM_PIN_GROUPS"][group])
    assert not [key for key in ns["_GEMM_PINS"] if key[0] in sites]


def _calls(tree: ast.Module) -> dict:
    """site -> [(enclosing function, today's callee, a, b)] of every ``_gemm_pinned`` call."""
    found: dict = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_gemm_pinned"
            ):
                continue
            site, a, b, today = node.args[:4]
            assert isinstance(today, ast.Lambda) and isinstance(today.body, ast.Call), (
                f"{fn.name}: today is not a call"
            )
            call = today.body
            found.setdefault(site.value, []).append(
                (fn.name, ast.unparse(call.func), ast.unparse(a), ast.unparse(b), call)
            )
    return found


def test_every_site_calls_its_gemm_through_gemm_pinned():
    """Each site's ``_gemm_pinned`` call sits in the functions that ran the GEMM, with that GEMM as
    ``today``: on ``a`` and the weight whose ``.t()`` is ``b`` (the router's on that ``.t()``)."""
    calls = _calls(_tree())
    assert sorted(calls) == sorted(_SITES)
    for site, (_, _, functions, callee) in _SITES.items():
        assert sorted({c[0] for c in calls[site]}) == sorted(functions), site
        for fn, got, a, b, call in calls[site]:
            assert got == callee, (site, fn, got)
            args = [ast.unparse(x) for x in call.args]
            assert args[0] == a, (site, fn, args)
            assert b in (f"{args[1]}.t()", args[1]), (site, fn, b, args)


class _Recorder:
    """The stand-ins ``_gemm_pinned`` reaches: today's call, the entry, the logger, capture."""

    def __init__(self, rows: int, n: int, dtype=torch.bfloat16, differ: bool = False, fail=None):
        self.rows, self.n, self.dtype = rows, n, dtype
        #: ``fail``: the exception the entry raises (None: it runs).
        self.differ, self.fail = differ, fail
        self.capturing = False
        self.calls: list = []
        self.lines: list = []

    def today(self) -> torch.Tensor:
        out = torch.zeros(self.rows, self.n, dtype=self.dtype)
        self.calls.append(("today", out))
        return out

    def entry(self, a, b, out, pin):
        if self.fail is not None:
            raise self.fail
        out.fill_(-0.0 if self.differ else 0.0)
        self.calls.append(("entry", out, pin))
        return out

    def namespace(self) -> dict:
        logger = SimpleNamespace(
            info=lambda msg: self.lines.append(("info", msg)),
            warning=lambda msg: self.lines.append(("warning", msg)),
            warning_once=lambda msg, key: self.lines.append(("warning_once", msg, key)),
        )

        class _Torch:
            cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)

            def __getattr__(self, name):
                return getattr(torch, name)

        return dict(
            rubin_lt_gemm=self.entry, LtPin=LtPin, LtUnavailable=LtUnavailable, logger=logger,
            torch=_Torch(),
        )  # fmt: skip


def _run(ns: dict, rec: _Recorder, site: str, out=None, out_dtype=torch.bfloat16):
    a = torch.zeros(rec.rows, 8, dtype=torch.bfloat16)
    b = torch.zeros(rec.n, 8, dtype=torch.bfloat16).t()
    return ns["_gemm_pinned"](site, a, b, rec.today, out=out, out_dtype=out_dtype)


def test_an_unpinned_row_count_runs_todays_call_only():
    rec = _Recorder(48, 7168)
    ns = _block(_tree(), rec.namespace())
    got = _run(ns, rec, "moe_latent_up")
    assert [c[0] for c in rec.calls] == ["today"] and got is rec.calls[0][1]
    assert ns["_GEMM_PIN_CHECKED"] == {} and rec.lines == []


@pytest.mark.parametrize("site", sorted(_SITES))
def test_the_first_call_checks_then_the_pin_runs(site):
    ns0 = _constants(_tree())
    for (pinned_site, rows), config in ns0["_GEMM_PINS"].items():
        if pinned_site != site:
            continue
        n = _SITES[site][0][0]
        dtype = torch.float32 if _SITES[site][1] == "router" else torch.bfloat16
        rec = _Recorder(rows, n, dtype)
        ns = _block(_tree(), rec.namespace())
        first = _run(ns, rec, site, out_dtype=dtype)
        assert [c[0] for c in rec.calls] == ["today", "entry"]
        assert first is rec.calls[0][1], "the checking call returns today's output"
        assert rec.calls[1][2] == LtPin(*config)
        assert rec.calls[1][1].shape == (rows, n) and rec.calls[1][1].dtype == dtype
        assert ns["_GEMM_PIN_CHECKED"] == {(site, rows): True}
        pin = LtPin(*config).name()
        if (site, rows) in ns0["_GEMM_PINS_A_CLASS"]:
            tol = ns["_GEMM_PIN_A_TOL"]
            line = ns["_GEMM_PIN_LINE_A"].format(site=site, rows=rows, pin=pin, tol=tol)
        else:
            line = ns["_GEMM_PIN_LINE"].format(site=site, rows=rows, pin=pin)
        assert rec.lines == [("info", line)]
        second = _run(ns, rec, site, out_dtype=dtype)
        assert [c[0] for c in rec.calls] == ["today", "entry", "entry"]
        assert second is rec.calls[2][1] and second.shape == (rows, n) and second.dtype == dtype
        given = torch.empty(rows, n, dtype=dtype)
        assert _run(ns, rec, site, out=given, out_dtype=dtype) is given
        assert len(rec.lines) == 1, "logged once"


def test_a_capture_before_the_check_runs_todays_call_and_records_nothing():
    rec = _Recorder(144, 7168)
    rec.capturing = True
    ns = _block(_tree(), rec.namespace())
    _run(ns, rec, "moe_latent_up")
    assert [c[0] for c in rec.calls] == ["today"]
    assert ns["_GEMM_PIN_CHECKED"] == {} and rec.lines == []


_FAILS = {
    "differ": (None, "word for word"),
    "unavailable": (LtUnavailable("cuBLASLt's heuristic does not offer it"), "does not offer"),
    "layout": (ValueError("b has no unit stride"), "no unit stride"),
}


@pytest.mark.parametrize("why", sorted(_FAILS))
def test_a_failed_check_keeps_todays_call(why):
    fail, said = _FAILS[why]
    rec = _Recorder(144, 896, torch.float32, differ=why == "differ", fail=fail)
    ns = _block(_tree(), rec.namespace())
    first = _run(ns, rec, "moe_router", out_dtype=torch.float32)
    assert first is rec.calls[0][1]
    assert ns["_GEMM_PIN_CHECKED"] == {("moe_router", 144): False}
    ((level, line),) = rec.lines
    assert level == "warning" and line.startswith("Kimi K3.1 Rubin GEMM pins: moe_router at 144")
    assert said in line
    rec.calls.clear()
    _run(ns, rec, "moe_router", out_dtype=torch.float32)
    assert [c[0] for c in rec.calls] == ["today"]


@pytest.mark.parametrize("why", ["unavailable", "layout"])
def test_a_later_call_the_entry_cannot_run_runs_todays_call(why):
    """After a passed check, a call whose layout the entry has no plan for (a capture of other
    strides) or refuses runs today's call and logs one WARNING, instead of failing the step."""
    rec = _Recorder(144, 896, torch.float32)
    ns = _block(_tree(), rec.namespace())
    _run(ns, rec, "moe_router", out_dtype=torch.float32)
    assert ns["_GEMM_PIN_CHECKED"] == {("moe_router", 144): True}
    rec.calls.clear()
    rec.lines.clear()
    rec.fail = _FAILS[why][0]
    for _ in range(2):
        got = _run(ns, rec, "moe_router", out_dtype=torch.float32)
        assert got is rec.calls[-1][1] and rec.calls[-1][0] == "today"
    assert [c[0] for c in rec.calls] == ["today", "today"]
    levels = {line[0] for line in rec.lines}
    assert levels == {"warning_once"} and len({line[2] for line in rec.lines}) == 1
    assert rec.lines[0][1].startswith("Kimi K3.1 Rubin GEMM pins: moe_router at 144 rows: ")
    assert rec.lines[0][1].endswith("; today's GEMM runs") and _FAILS[why][1] in rec.lines[0][1]


class _BandRecorder(_Recorder):
    """Today's output is a ramp of largest magnitude 1; the entry writes it plus ``offset`` at one
    element (NaN: a NaN there)."""

    def __init__(self, rows: int, n: int, offset: float):
        super().__init__(rows, n)
        self.offset = offset

    def today(self) -> torch.Tensor:
        out = torch.linspace(-1.0, 1.0, self.rows * self.n).view(self.rows, self.n).to(self.dtype)
        self.calls.append(("today", out))
        return out

    def entry(self, a, b, out, pin):
        out.copy_(torch.linspace(-1.0, 1.0, self.rows * self.n).view(self.rows, self.n))
        out[0, 0] += self.offset
        self.calls.append(("entry", out, pin))
        return out


@pytest.mark.parametrize(
    "offset, passes", [(0.0, True), (2.0**-7, True), (2.0**-4, False), (float("nan"), False)]
)
def test_an_a_class_check_is_a_band_of_todays_largest_magnitude(offset, passes):
    ns0 = _constants(_tree())
    site, rows = sorted(ns0["_GEMM_PINS_A_CLASS"])[0]
    rec = _BandRecorder(rows, _SITES[site][0][0], offset)
    ns = _block(_tree(), rec.namespace())
    _run(ns, rec, site)
    assert ns["_GEMM_PIN_CHECKED"] == {(site, rows): passes}
    ((level, line),) = rec.lines
    assert level == ("info" if passes else "warning")
    assert ("A-class" in line) if passes else ("not within" in line)


class _Fn:
    """A ctypes function stand-in: settable ``restype`` / ``argtypes``, a fixed return value."""

    def __init__(self, value: int = 0):
        self.value = value

    def __call__(self, *args):
        return self.value


class _FakeLt:
    """A loaded libcublasLt stand-in reporting ``version``; every other symbol returns 0."""

    def __init__(self, version: int):
        self.cublasLtGetVersion = _Fn(version)

    def __getattr__(self, name):
        fn = _Fn()
        setattr(self, name, fn)
        return fn


@pytest.mark.parametrize("version, loads", [(120804, False), (130100, True), (140000, False)])
def test_the_binding_loads_only_cublas_13(monkeypatch, version, loads):
    """The binding's enums and struct sizes are cuBLAS 13's: another library is refused with
    ``LtUnavailable`` (the pin's check then logs a WARNING and today's GEMM runs)."""
    monkeypatch.setattr(rubin_lt_pin, "_LIB", [])
    monkeypatch.setattr(rubin_lt_pin.ctypes, "CDLL", lambda name: _FakeLt(version))
    if loads:
        assert isinstance(rubin_lt_pin._lib(), _FakeLt)
    else:
        with pytest.raises(LtUnavailable, match="is not cuBLAS 13"):
            rubin_lt_pin._lib()
        assert rubin_lt_pin._LIB == []

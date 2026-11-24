# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's ``moe_shared_late`` (a generation constant, on): the shared experts of a band step fork
after the routing tail, below every other stream of the step.

CPU (``cpu_only``), from helix8's modeling.py over its constants and the recording stand-ins of
``test_rubin_helix8_moe_front.py`` / ``test_rubin_helix8_moe_cp_tail.py``:

* the constant is on with its bands ``36-60`` (rows, so a draft length other than k5 keeps the
  same row band) and the startup constants line states both; off, no band is served;
* the bands must lie inside ``moe_cp_tail_bands``: a band row the column-parallel tail does not
  serve refuses the import;
* a CUDA graph decode step whose runtime carries the shared experts' stream (``rt.shared_stream``)
  runs the column-parallel tail with the shared fork recorded right after the routing tail instead
  of right before it, and the shared chain (the shared experts, the push into the tail's Lamport
  call, the join record) on that stream; every other op is today's, in today's order, every kernel
  touches only memory its stream may touch, every side stream is joined before the step returns,
  and the README's INFO line is logged once;
* without that stream every step keeps today's plan (the cp tail tests run with it unset);
* ``_moe_shared_late_rt`` moves every side stream of the runtime (every ``Stream`` field of
  ``_Rt`` and ``_MoeComm``) onto the raised streams and the shared experts onto the given stream;
  ``_late_streams`` makes one raised stream per side stream of the core plus the layers' own, at the
  highest priority torch offers;
* ``DecodeTarget.forward`` runs the layers through ``_on_stream`` on the late streams exactly on
  CUDA graph steps whose rows fall in the bands, with the switch on; ``_on_stream`` forks once and
  joins once.

Whether a replay keeps the priorities, and the timing and bytes on the GPU, are checked on a
world-8 GPU run, not here.
"""

from __future__ import annotations
import __future__

import ast
import copy
import re
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import pytest
import test_rubin_helix8_moe_cp_tail as T  # noqa: E402 -- the tail's stand-ins
import test_rubin_helix8_moe_front as F  # noqa: E402 -- the block's recording stand-ins
import torch

pytestmark = pytest.mark.cpu_only

_MODELING, _README = F._MODELING, F._README
_HIGH = -3  # the stand-in device's highest stream priority (CUDA's greatest priority is negative)


def _source(**values) -> str:
    """modeling.py's source with the generation constants ``values`` (strings quoted)."""
    text = _MODELING.read_text()
    for name, value in values.items():
        rendered = f'"{value}"' if isinstance(value, str) else repr(value)
        text, hits = re.subn(rf'(?m)^( {{16}}"{name}": )[^,\n]+,$', rf"\g<1>{rendered},", text)
        assert hits == 1, (name, hits)
    return text


def _tree(**values) -> ast.Module:
    """modeling.py's tree with the tail on (as shipped) and ``values`` applied."""
    return ast.parse(_source(**{"moe_cp_tail": True, **values}), filename=str(_MODELING))


def _on() -> ast.Module:
    return _tree(moe_shared_late=True)


def _top(tree: ast.Module, name: str) -> ast.stmt:
    (node,) = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    ]
    return node


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in _top(tree, cls).body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _exec(ns: dict, *nodes: ast.stmt) -> dict:
    module = ast.fix_missing_locations(ast.Module([copy.deepcopy(n) for n in nodes], []))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


# --- the constant and its bands ---


def test_the_lever_is_on_with_its_row_bands_and_the_constants_line_states_it():
    shipped = ast.parse(_MODELING.read_text())
    ns = T._constants(shipped)
    gen = ns["_CONSTANTS"]["generation"]
    # On in the shipped stack at the rows the world-8 gate kept (36-60).
    assert gen["moe_shared_late"] is True and gen["moe_shared_late_bands"] == "36-60"
    assert ns["_DETAILS"]["moe_shared_late_bands"] == "moe_shared_late"
    assert ns["_MOE_SHARED_LATE_BANDS"] == ((36, 60),)
    _exec(ns, _top(shipped, "constants_line"))
    assert "moe_shared_late=True(moe_shared_late_bands=36-60)" in ns["constants_line"]("generation")
    assert T._constants(_tree(moe_shared_late=False))["_MOE_SHARED_LATE_BANDS"] == ()


def _import_check(**values) -> None:
    """Run modeling.py's import-time check of moe_shared_late's bands with ``values``."""
    tree = _tree(**values)
    ns = T._constants(tree)
    _exec(ns, *F._definitions(tree, ("_moe_in_bands", "_moe_cp_tail_serves")))
    (check,) = [
        n for n in tree.body if isinstance(n, ast.If) and "_MOE_SHARED_LATE_BANDS" in ast.unparse(n)
    ]
    _exec(ns, check)


@pytest.mark.parametrize(
    "values",
    [
        dict(moe_shared_late=True),
        dict(moe_shared_late=True, moe_shared_late_bands="24-30,36-60"),
        dict(moe_shared_late=True, moe_shared_late_bands="48-48"),
        dict(moe_shared_late=False, moe_cp_tail_bands="1-30"),
    ],
    ids=["shipped_bands", "two_bands", "one_row", "off_on_a_narrow_tail"],
)
def test_bands_inside_the_tail_import(values):
    _import_check(**values)


@pytest.mark.parametrize(
    "values",
    [
        dict(moe_shared_late=True, moe_cp_tail_bands="1-30"),
        dict(moe_shared_late=True, moe_cp_tail_bands="1-30,49-60"),
        dict(moe_shared_late=True, moe_cp_tail=False),
    ],
    ids=["tail_1_30", "tail_gap", "tail_off"],
)
def test_bands_the_tail_does_not_serve_refuse_to_import(values):
    with pytest.raises(NotImplementedError, match="moe_shared_late_bands"):
        _import_check(**values)


@pytest.mark.parametrize(
    "rows, served", [(1, False), (30, False), (35, False), (36, True), (42, True), (60, True),
                     (61, False), (66, False), (72, False)]
)  # fmt: skip
def test_the_lever_serves_exactly_its_bands(rows, served):
    names = ("_moe_in_bands", "_moe_shared_late_serves")
    on = T._block(_on(), {}, names)
    off = T._block(_tree(moe_shared_late=False), {}, names)
    assert on["_moe_shared_late_serves"](rows) is served
    assert off["_moe_shared_late_serves"](rows) is False


# --- the decode step's MoE ---


class _Plan(F._Plan):
    """``F._Plan`` with the shared experts' own stream."""

    STREAMS = (*F._Plan.STREAMS, "shared")


def _run(rows: int, late: bool):
    """``(plan, log, block, seen, out)`` of one graph step's ``latent_moe`` (tail on, lever on)."""
    plan, seen = _Plan(), {}
    log = F._Logger(plan)
    block = T._block(
        _on(),
        {
            "torch": F._Torch(plan),
            "logger": log,
            **F._entries(plan),
            **T._tail_entries(plan, seen),
        },
    )
    layer, rt = F._layer(False), T._rt(plan)
    if late:
        rt.shared_stream = F._Stream("shared")
    x = torch.empty(rows, F._HIDDEN, dtype=torch.bfloat16)
    with F._Record(plan):
        out = block["latent_moe"](layer, F._LAYER, x, SimpleNamespace(cuda_graph=True), rt)
    return plan, log, block, seen, out


def _late_ops(today: list) -> list:
    """Today's ops with the shared fork moved right after the routing tail and the shared chain on
    its own stream, then the late line."""
    ops = []
    for op in today:
        if op == "record:shared_fork@main":
            continue
        ops.append(op.replace("@aux", "@shared"))
        if op == "route_tail@main":
            ops.append("record:shared_fork@main")
        if op == "record:shared_join@aux":
            ops.append("log")
    return ops


@pytest.mark.parametrize("rows", [36, 42, 48, 54, 60])
def test_a_band_step_forks_the_shared_experts_after_the_routing_tail(rows):
    today, _, _, today_seen, _ = _run(rows, late=False)
    plan, log, block, seen, out = _run(rows, late=True)
    assert isinstance(out, block["_MoeTail"])
    assert "record:shared_fork@main" in today.ops
    assert plan.ops == _late_ops(today.ops), plan.ops
    assert not plan.problems, "\n".join(plan.problems)
    # The shared chain waits for the routing tail: its stream's view of the current stream is past
    # the routing tail when the shared experts run.
    route_at = plan.kernel_at("route_tail")[0]
    shared = plan.kernel_at("shared")
    assert shared[1] == "shared" and shared[4]["main"] > route_at
    # Nothing runs on the aux stream; every side stream is joined before the step returns.
    assert not [k for k in plan.kernels if k[1] == "aux"]
    for side in ("latent", "prezero", "shared"):
        last = max(at for at, stream, *_ in plan.kernels if stream == side)
        assert plan.clock["main"].get(side, 0) > last, f"the {side} stream is not joined"
    # The same push of the same partial into the same call.
    assert seen["push"]["data"] is shared[3][1]
    assert seen["push"]["ws"] == today_seen["push"]["ws"] == "tail_ws"
    assert seen["push"]["flags"] == today_seen["push"]["flags"] == (True, True)
    assert seen["cp_tail"]["tactic"] == today_seen["cp_tail"]["tactic"] == f"tactic_{rows}"
    line = block["_MOE_SHARED_LATE_LINE"]
    assert (line, line) in log.once and len(log.once) == len(set(log.once))


@pytest.mark.parametrize("rows", [30, 36, 54, 66])
def test_without_the_shared_stream_a_step_keeps_todays_plan(rows):
    plan, log, block, _, _ = _run(rows, late=False)
    assert plan.ops == T._run(rows)[0].ops
    assert block["_MOE_SHARED_LATE_LINE"] not in [line for line, _ in log.once]


def test_the_readme_states_the_line():
    line = T._constants(_on())["_MOE_SHARED_LATE_LINE"]
    assert "\n" not in line and f"\n{line}\n" in _README.read_text()
    assert "`moe_shared_late` (a helix8 generation constant, on)" in _README.read_text()


# --- the streams and the step ---


class _CudaStream:
    def __init__(self, log: list, name: str, priority: int | None = None):
        self.log, self.name, self.priority = log, name, 0 if priority is None else priority
        self.passed_priority = priority

    def wait_stream(self, other: "_CudaStream") -> None:
        self.log.append(("wait", self.name, other.name))


class _Cuda:
    """``torch.cuda`` as ``_late_streams`` and ``_on_stream`` see it."""

    def __init__(self, capturing: bool = False):
        self.log: list = []
        self.made: list = []
        self.capturing = capturing
        self.main = self.current = _CudaStream(self.log, "main")
        cuda = self

        class Stream(_CudaStream):
            def __init__(self, priority: int | None = None):
                super().__init__(cuda.log, f"s{len(cuda.made)}", priority)
                cuda.made.append(self)

            @staticmethod
            def priority_range():
                return (0, _HIGH)

        self.Stream = Stream

    def current_stream(self):
        return self.current

    def stream(self, s):
        cuda = self

        class _Ctx:
            def __enter__(self):
                self.prev, cuda.current = cuda.current, s
                cuda.log.append(("enter", s.name))

            def __exit__(self, *exc):
                cuda.current = self.prev
                cuda.log.append(("exit", s.name))

        return _Ctx()

    def is_current_stream_capturing(self):
        return self.capturing


def _streams_ns(cuda: _Cuda) -> dict:
    tree = _on()
    ns = T._constants(tree)
    ns.update(torch=SimpleNamespace(cuda=cuda), NamedTuple=NamedTuple, Optional=Optional, Any=Any,
              Callable=Callable)  # fmt: skip
    names = ("_Streams", "_LateStreams", "_late_streams", "_on_stream", "_MoeComm", "_Rt")
    return _exec(ns, *(_top(tree, n) for n in names), _top(tree, "_moe_shared_late_rt"))


def test_late_streams_raise_every_side_stream_of_the_core_and_the_layers():
    cuda = _Cuda()
    ns = _streams_ns(cuda)
    late = ns["_late_streams"]()
    assert late._fields == ("fwd", "aux", "fork", "idx", "latent", "prezero")
    # The tap stream (draft_tap_stream) is the drafter's: it keeps the core's stream.
    assert set(late._fields) - {"fwd"} == set(ns["_Streams"]._fields) - {"tap"}
    assert list(late) == cuda.made and len({id(s) for s in late}) == 6
    assert all(s.priority == _HIGH and s.passed_priority == _HIGH for s in late)


def test_the_band_runtime_moves_every_side_stream_and_the_shared_experts():
    ns = _streams_ns(_Cuda())
    rt_cls, comm_cls = ns["_Rt"], ns["_MoeComm"]
    comm = comm_cls(*(f"comm.{f}" for f in comm_cls._fields))
    required = [f for f in rt_cls._fields if f not in rt_cls._field_defaults and f != "moe_comm"]
    rt = rt_cls(**{f: f"rt.{f}" for f in required}, moe_comm=comm)
    assert rt.shared_stream is None, "outside a band step the shared experts use rt.aux_stream"
    late = SimpleNamespace(**{f: f"late.{f}" for f in ("fwd", "aux", "fork", "idx", "latent",
                                                       "prezero")})  # fmt: skip
    got = ns["_moe_shared_late_rt"](rt, late, "core.aux")
    moved = {"aux_stream": "late.aux", "fork_stream": "late.fork", "idx_stream": "late.idx",
             "shared_stream": "core.aux"}  # fmt: skip
    for f in rt_cls._fields:
        if f != "moe_comm":
            assert getattr(got, f) == moved.get(f, getattr(rt, f)), f
    comm_moved = {"latent_stream": "late.latent", "prezero_stream": "late.prezero"}
    for f in comm_cls._fields:
        assert getattr(got.moe_comm, f) == comm_moved.get(f, getattr(comm, f)), f
    # Every stream field of the runtime is one the band runtime moves: a new side stream needs a
    # raised counterpart, or it would run below the layers' stream. The one exception is
    # ``taps`` (draft_tap_stream): the drafter joins the core runtime's tap stream, so a band step
    # keeps it (its copies stay at the default priority).
    tree = _on()
    for cls, names in (("_Rt", moved), ("_MoeComm", comm_moved)):
        streams = {
            n.target.id
            for n in _top(tree, cls).body
            if isinstance(n, ast.AnnAssign) and "Stream" in ast.unparse(n.annotation)
        }
        assert streams - {"taps"} == set(names), (cls, streams)
    assert got.taps == rt.taps


@pytest.mark.parametrize("capturing", [True, False], ids=["capture", "warmup"])
def test_on_stream_forks_once_and_joins_once(capturing):
    cuda = _Cuda(capturing=capturing)
    ns = _streams_ns(cuda)
    fwd = cuda.Stream(priority=_HIGH)
    fwd.name = "fwd"
    seen = []

    class _Out:
        def record_stream(self, s):
            cuda.log.append(("record_stream", s.name))

    def run(a, b):
        seen.append((cuda.current_stream().name, a, b))
        cuda.log.append(("run",))
        return _Out()

    out = ns["_on_stream"](fwd, run, 1, 2)
    assert isinstance(out, _Out) and seen == [("fwd", 1, 2)]
    want = [("wait", "fwd", "main"), ("enter", "fwd"), ("run",), ("exit", "fwd"),
            ("wait", "main", "fwd")]  # fmt: skip
    if not capturing:
        want.append(("record_stream", "main"))
    assert cuda.log == want and cuda.current_stream() is cuda.main


@pytest.mark.parametrize("on", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("cuda_graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("rows", [30, 36, 54, 60, 66])
def test_decode_target_runs_band_graph_steps_on_the_late_streams(rows, cuda_graph, on):
    tree = _on() if on else _tree()
    serves = T._block(tree, {}, ("_moe_in_bands", "_moe_shared_late_serves"))
    calls = []
    st = SimpleNamespace(cuda_graph=cuda_graph)
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(bind=lambda cache: calls.append(("bind",))),
                         ks_ws="ws")  # fmt: skip

    def on_stream(stream, run, *args):
        calls.append(("on_stream", stream))
        return run(*args)

    ns = {
        "_build_step_args": lambda md, spec, pools, decode: st,
        "_on_stream": on_stream,
        "_key_shard_ws": lambda *a: pytest.fail("the workspace exists"),
        "_moe_shared_late_serves": serves["_moe_shared_late_serves"],
        "_moe_shared_late_rt": lambda r, late, shared: ("late_rt", r, late, shared),
        "Any": Any,
    }
    _exec(ns, _method(tree, "DecodeTarget", "forward"))

    def run_layers(step, runtime, input_ids, inputs_embeds):
        calls.append(("run_layers", step, runtime, input_ids, inputs_embeds))
        return "hidden"

    late = SimpleNamespace(fwd="late.fwd") if on else None
    core = SimpleNamespace(late_streams=late, streams=SimpleNamespace(aux="core.aux"))
    target = SimpleNamespace(_rt=rt, core=core, _run_layers=run_layers)
    md = SimpleNamespace(kv_cache_manager="cache", key_shard_world=8)
    ids = torch.empty(rows, dtype=torch.int32)
    assert ns["forward"](target, md, ids, "pos", None, "spec") == "hidden"
    if on and cuda_graph and 36 <= rows <= 60:
        want_rt = ("late_rt", rt, late, "core.aux")
        want = [("bind",), ("on_stream", "late.fwd"), ("run_layers", st, want_rt, ids, None)]
        assert calls == want
    else:
        assert calls == [("bind",), ("run_layers", st, rt, ids, None)]


def test_the_core_makes_the_late_streams_only_with_the_lever_on():
    tree = _on()
    (core,) = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Helix8Core")
    ]
    init = ast.unparse(_method(tree, core.name, "__init__"))
    # Today's five side streams as before, then the raised set only when the lever has bands.
    assert "self.streams = _Streams(*(torch.cuda.Stream() for _ in _Streams._fields))" in init
    assert "self.late_streams = _late_streams() if _MOE_SHARED_LATE_BANDS else None" in init
    assert init.index("self.streams =") < init.index("self.late_streams =")

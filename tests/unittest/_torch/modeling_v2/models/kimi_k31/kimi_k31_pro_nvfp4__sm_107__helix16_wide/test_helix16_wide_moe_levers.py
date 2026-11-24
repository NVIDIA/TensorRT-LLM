# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE switches' paths (CPU only, on the sources, kernels stood in by recorders).

- The tuned tail: ``prepare_tuned_tail`` gives every published row count the shared-row push
  kernel and one ``Fc12TailFeed`` (two-shot above 16 rows); a tuned step runs the sort, the glue's
  join, FC12, the feed, the glue's release, the shared expert's join, then the tail kernel.
- The merged front: ``FrontOnce`` with the merged tensor-memory front compiles ``MoeFrontRows`` at
  N 64 and launches it, then the routing.
- The FP32 routing weights: ``RouteSituF32`` keeps the parent's final wait at 72-128 rows, the
  entry wait at 64.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import family_dir, function_node, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

VENDOR = "kernels/vendor/vllm_mr8/specialized_decode/cutedsl"
FAKE = "_k31_helix16_wide_fake_v3p"


@pytest.fixture(scope="module")
def policy():
    path = family_dir() / VENDOR / "common/policy.py"
    spec = importlib.util.spec_from_file_location("_k31_helix16_wide_policy_v3p", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packages(monkeypatch, modules):
    for name in (FAKE, f"{FAKE}.kernels", f"{FAKE}.kernels.vendor"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    loader = types.ModuleType(f"{FAKE}.kernels.vendor.vllm_mr8")
    loader.module = modules.__getitem__
    monkeypatch.setitem(sys.modules, loader.__name__, loader)


# --- The tuned tail ------------------------------------------------------------------------------


class _PreparedTail:
    """``tail_runtime.PreparedTail`` stood in: Rubin TP16's published (5-64) / two-shot (17-64)
    rows; its own push kernels where native's experts do not push (9-64)."""

    def __init__(self, op, norm, up, batches, placement, *, local_up=False):
        self.args = (op, norm, up, list(batches), placement, local_up)
        self.twoshots = {b for b in batches if b > 16}
        self.pushers = {b: ("push", b) for b in batches if b > 8}
        self.publish = {b: ("publish", b) for b in batches}


def test_the_tuned_tail_pushes_every_published_row(monkeypatch):
    compiled = []
    tail_runtime = SimpleNamespace(
        published=lambda tp, b, device: b > 4, PreparedTail=_PreparedTail,
        compile_push=lambda b, device, tp: ("compile_push", b, device, tp),
    )  # fmt: skip
    _packages(monkeypatch, {"cutedsl.moe.tail_runtime": tail_runtime})
    ns = {
        "__package__": f"{FAKE}.target", "SimpleNamespace": SimpleNamespace,
        "compile_fc12_feed": lambda b, twoshot, dev: compiled.append((b, twoshot)) or (b, twoshot),
    }  # fmt: skip
    source_objects(modeling_path(), ("prepare_tuned_tail",), ns)
    weights = SimpleNamespace(norm=torch.zeros(3584), up=torch.zeros((448, 3584)), eps=1e-6)
    rows = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64]
    tail, feeds = ns["prepare_tuned_tail"](weights, rows, "box", 7, 3, 92)
    published = [b for b in rows if b > 4]
    op, norm, up, batches, placement, local_up = tail.args
    assert (op.contract.tp_size, op.contract.rms_eps, op.rank) == (16, 1e-6, 7)
    assert norm is weights.norm and up is weights.up and local_up is True
    assert batches == published and placement == ("box", 3, 92)
    # Native's experts push the shared rows of 5-8 rows; FC12 does not: push kernels there too.
    assert sorted(tail.pushers) == published
    assert {b: tail.pushers[b] for b in (5, 6, 7, 8)} == {
        b: ("compile_push", b, None, 16) for b in (5, 6, 7, 8)
    }
    assert feeds == {b: (b, b > 16) for b in published}
    assert sorted(compiled) == sorted(feeds.values())
    with pytest.raises(ValueError):
        ns["prepare_tuned_tail"](weights, [1, 2, 4], "box", 7, 3, 92)


def test_a_tuned_step_feeds_then_joins_then_runs_the_tail(monkeypatch):
    log = []
    host = SimpleNamespace(stream=lambda device: "s", to_cute=lambda t: ("cute", t.shape[0]))
    _packages(monkeypatch, {"cutedsl.moe.prepared": SimpleNamespace(PreparedMoE=object)})
    ns = {
        "__package__": f"{FAKE}.target", "torch": torch, "contextlib": contextlib,
        "module": {"cutedsl.moe.prepared": SimpleNamespace(PreparedMoE=object),
                   "cutedsl.common.host": host}.__getitem__,
    }  # fmt: skip
    source_objects(modeling_path(), ("_MOE_TOP_K", "FRONT_ROWS", "Fc12MoE"), ns)
    moe = object.__new__(ns["Fc12MoE"])
    b = 32
    expanded, front = torch.zeros((64 * 16, 8)), tuple(torch.zeros((64, 8)) for _ in range(7))
    moe.state = SimpleNamespace(device="cpu", expanded=expanded, front=front)
    moe.partial = torch.zeros((64, 3584))
    moe.glue = SimpleNamespace(
        ones=torch.ones((64, 16)), counters="counters", join=lambda: log.append("glue join"),
        release=lambda layer, rows: log.append(("release", layer, rows)),
    )  # fmt: skip
    moe.experts = SimpleNamespace(
        sort=lambda ids, w: log.append("sort") or "tables",
        assemble=lambda *a: "operands", launch=lambda o, c: log.append(("fc12", c)),
    )  # fmt: skip
    moe.prefetch, moe.layer, moe.layer_idx = None, 4, 4

    def feed(*args):
        log.append(("feed", args))

    moe.tuned = SimpleNamespace(
        publish={b: ("slots", "peers", "slot")}, tail_box="box",
        tail_weights=SimpleNamespace(run=lambda *a: log.append(("tail", a))),
    )  # fmt: skip
    moe.feeds = {b: feed}
    weights = torch.zeros((b, 16))
    glued = ("x4", "sf", "s")
    moe.experts_and_tail(None, None, None, weights, b, lambda: log.append("join"), glued)
    kinds = [e if isinstance(e, str) else e[0] for e in log]
    assert kinds == ["sort", "glue join", "fc12", "feed", "release", "join", "tail"]
    assert log[2] == ("fc12", "counters")
    assert log[3] == ("feed", (("cute", b), "slots", "peers", "slot", "s"))
    assert log[4] == ("release", 4, b)
    box, rows, top, shared = log[6][1]
    assert box == "box" and top is weights
    assert rows.shape[0] == b * 16 and rows.data_ptr() == expanded.data_ptr()
    assert shared.shape[0] == b and shared.data_ptr() == front[5].data_ptr()


# --- The merged front ----------------------------------------------------------------------------


def test_front_once_builds_the_tmem_front(policy):
    log = []
    kernel, hidden = "tmem_merged", 384
    projection = SimpleNamespace(
        weight_map=lambda w, rows: SimpleNamespace(data_ptr=lambda: rows),
        matrix_view=lambda t: ("view", tuple(t.shape)),
    )  # fmt: skip
    front_module = SimpleNamespace(
        WIDTH=5376, front_geometry=lambda *a, **k: ("geometry",),
        compile_route=lambda *a, **k: "route",
    )  # fmt: skip
    host = SimpleNamespace(
        to_cute=lambda t: ("cute", tuple(t.shape)), stream=lambda d: "s",
        prepare_shared_carveout=lambda code, dev, value: code,
    )  # fmt: skip
    modules = {
        "cutedsl.moe.front": front_module, "cutedsl.common.host": host,
        "cutedsl.common.policy": policy, "cutedsl.gemm.projection": projection,
    }  # fmt: skip

    def compile_front_rows(*args):
        log.append(("compile", args))
        return lambda *a: log.append(("front", a))

    ns = {
        "module": modules.__getitem__, "torch": torch, "Int32": lambda v: v, "Int64": lambda v: v,
        "Float32": lambda v: ("f32", v), "MoeRouteLaunch": lambda **k: SimpleNamespace(**k),
        "moe_route": lambda launch, s: log.append(("route", launch.rows)),
        "compile_front_rows": compile_front_rows, "compile_front_once": None,
        "compile_route_f32": None, "Sequence": tuple,
    }  # fmt: skip
    names = ("FRONT_ROWS", "FRONT_ONCE", "FRONT_TMEM_N", "WIDE_FRONTS", "_MOE_LATENT")
    source_objects(modeling_path(), (*names, "_MOE_EXPERTS", "_MOE_SITU", "FrontOnce"), ns)
    weights = SimpleNamespace(
        front=torch.zeros((288, 8)), bias=torch.zeros(896), scale=2.8, renormalize=True,
        shared=SimpleNamespace(gate_up=SimpleNamespace(weight=torch.zeros((768, 8)))),
    )  # fmt: skip
    box = SimpleNamespace(buffer=torch.zeros((1, 128, 5376)), mc=9, rank=2)
    storage = tuple(torch.zeros((128, n)) for n in (896, 768, 16, 16, 384, 7168, 3584))
    rows = (72, 96, 104, 128)
    once = ns["FrontOnce"](weights, box=box, storage=storage, rows=rows, q=8, kernel=kernel)
    compiles = [args for kind, args in log if kind == "compile"]
    # M6: N 64 at every wide row (two N tiles), one compile.
    assert compiles == [(64, 56, 224, hidden, 64, 128, 384, None)] * 4
    assert sorted(once.tmem_codes) == [64] and once.label(72) == f"{kernel}:N64"
    log.clear()
    x = torch.zeros((104, 7168))
    once.route(x)
    (kind, args), route = log
    assert kind == "front" and route == ("route", 104)
    assert args[:3] == (("view", (288, 8)), ("view", (104, 7168)), ("cute", (128, 384)))
    assert args[3:7] == once.box and args[7:9] == (("f32", 4.0), ("f32", 25.0))
    assert args[9:] == (128, 64, "s")  # the packed rows' map, gate/up's, the stream


# --- The FP32 routing weights --------------------------------------------------------------------


def test_the_fp32_routing_keeps_the_final_wait(monkeypatch):
    """On ``FrontOnce``'s arguments (72-128 rows: lean, polled, not early) the parent's final wait is
    kept, so the routing's completion still implies the front's; at 64 rows (early "front") the
    kernel waits at entry."""
    front = family_dir() / VENDOR / "moe/front.py"
    # The parent's constructor and exits_unwaited (plain Python), under RouteSituF32's.
    methods = [
        ast.unparse(function_node(front, f"RouteSitu.{name}"))
        for name in ("__init__", "exits_unwaited")
    ]
    parent: dict = {}
    exec("class RouteSitu:\n" + "\n".join(textwrap.indent(m, "    ") for m in methods), parent)
    runtime = types.ModuleType("cutlass.cute.runtime")
    runtime.make_fake_stream = lambda: "stream"
    for name, module in (
        ("cutlass", types.ModuleType("cutlass")), ("cutlass.cute", types.ModuleType("cutlass.cute")),
        ("cutlass.cute.runtime", runtime),
    ):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)
    modules = {
        "cutedsl.moe.front": SimpleNamespace(RouteSitu=parent["RouteSitu"], WIDTH=5376),
        "cutedsl.common.host": SimpleNamespace(fake=lambda *a: ("fake", a)),
    }  # fmt: skip
    ns = {
        "module": modules.__getitem__, "cache": lambda f: f, "Float32": float, "Int32": int,
        "cute": SimpleNamespace(jit=lambda f: f, compile=lambda kernel, *args: kernel),
        "torch": SimpleNamespace(
            accelerator=SimpleNamespace(device_index=lambda d: contextlib.nullcontext())
        ),
    }  # fmt: skip
    source_objects(modeling_path(), ("RouteSituF32", "compile_route_f32"), ns)
    for rows in range(72, 129, 8):
        # FrontOnce's call: compile_route_f32(b, shared, renorm, dev, None, None, False, True, cap)
        kernel = ns["compile_route_f32"](rows, 384, True, 0, None, None, False, True, 128)
        flags = (kernel.lean, kernel.polled, kernel.early, kernel.halves)
        assert flags == (True, True, False, False)
        assert kernel.signal is True and kernel.exits_unwaited()
    kernel = ns["compile_route_f32"](64, 384, True, 0, True, "front", False, True, 64)
    assert kernel.early == "front" and not kernel.exits_unwaited()

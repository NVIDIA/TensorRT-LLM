# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step levers (CPU only, on the sources; kernels stood in by recorders, or run in Triton's
interpreter): N3's carved o-projection consumers, N1's replaced verify kernels, N6's drafter
tiles, M2's counting sort (tables against ``RubinMoeSort.reference``, the stream order), M6 with
the merged front (``step_levers``)."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import cache
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import pytest
import torch
from _helix16_wide_cpu import family_dir, function_node, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

VENDOR = family_dir() / "kernels/vendor/vllm_mr8/specialized_decode/cutedsl"
OPTION_NAMES = (
    "N1_KERNELS", "N3_CARVEOUT", "N3_CARVEOUT_ROWS", "DRAFT_TILE_REQUESTS", "DRAFT_TILE_WARPS",
    "DRAFT_TILE_CAP", "DRAFT_TILES", "_WIDE_SWITCHES", "record_switches", "wide_switches",
)  # fmt: skip


def _base_namespace(**extra) -> dict:
    ns = {
        "dataclass": dataclass, "fields": fields, "Dict": Dict, "Any": Any, "Optional": Optional,
        "Tuple": Tuple, "Iterable": Iterable, "contextmanager": contextmanager, "cache": cache,
        "torch": torch, "SimpleNamespace": SimpleNamespace,
    }  # fmt: skip
    ns.update(extra)
    return ns


# --- N3: the o-projection --------------------------------------------------------------------


class _Oproj:
    """``oproj.prepared`` stood in: ``WIDE_CHUNK`` as MR8 ships it, ``PreparedOproj`` recording
    its construction (and the table it saw), ``compile_oproj`` with its cache's ``__wrapped__``."""

    def __init__(self, raise_in_init=False):
        self.WIDE_CHUNK = {16: "65-80:80,192-256:96"}
        self.built, self.compiled = [], []
        oproj = self

        class PreparedOproj:
            def __init__(self, decoder, mailbox, batches, mla, **kwargs):
                oproj.built.append((decoder, mailbox, batches, mla, kwargs, dict(oproj.WIDE_CHUNK)))
                if raise_in_init:
                    raise RuntimeError("construction failed")
                self.device = SimpleNamespace(index=3)
                self.wide = {
                    b: (("producer", ("consumer", b)), "head", "gather", "tail", "rows", "tma")
                    for b in batches if b > 64
                }  # fmt: skip

        def compile_oproj(*args):
            raise AssertionError("the cached compile must not be used for the carved consumer")

        compile_oproj.__wrapped__ = lambda *args: oproj.compiled.append(args) or ("carved", args)
        self.PreparedOproj, self.compile_oproj = PreparedOproj, compile_oproj


@pytest.fixture(scope="module")
def policy():
    """MR8's ``common.policy`` (``batch_table``), loaded by path."""
    spec = importlib.util.spec_from_file_location("_k31_v4_policy", VENDOR / "common/policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def oproj_ns(policy):
    carved = []
    host = SimpleNamespace(prepare_shared_carveout=lambda code, i, c: carved.append((code, i, c)))
    oproj = _Oproj()
    modules = {"cutedsl.oproj.prepared": oproj, "cutedsl.common.host": host}
    modules["cutedsl.common.policy"] = policy
    ns = _base_namespace(module=modules.__getitem__)
    names = (*OPTION_NAMES, "compile_oproj_carved", "oproj_consumer_args",
             "carve_wide_oproj_consumers", "prepare_oproj")  # fmt: skip
    source_objects(modeling_path(), names, ns)
    ns["oproj"], ns["carved"], ns["modules"] = oproj, carved, modules
    return ns


def _decoder(write: bool = False):
    return SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=torch.empty(7168, 768))),
        prev_valid_blocks=5, is_block_write_layer=write,
    )  # fmt: skip


ROWS = (1, 8, 16, 64, 72, 80, 88, 96, 104, 112, 120, 128)


def test_n3_carveout_replaces_the_wide_consumers_with_their_own_functions(oproj_ns):
    decoder, mailbox, x = _decoder(write=True), SimpleNamespace(tp=16), torch.empty(128, 768)
    prepared = oproj_ns["prepare_oproj"](
        decoder, mailbox, ROWS, twoshot="ts", x_buffer=x
    )  # fmt: skip
    wide = [b for b in ROWS if b > 64]
    assert [args[2] for args in oproj_ns["oproj"].compiled] == wide
    for b in wide:
        producer, consumer = prepared.wide[b][0]
        assert producer == "producer" and consumer[0] == "carved"
        k, nb, delta = 768, 5 + 1, False  # a block-write layer: nb + 1, no delta
        shape = (k, b, 16, nb, delta, False, False, 128, True)
        assert consumer[1] == ("consumer", *shape, False, False, False, False, True)
        assert prepared.wide[b][1:] == ("head", "gather", "tail", "rows", "tma")
    assert [(i, c) for _, i, c in oproj_ns["carved"]] == [(3, 100)] * len(wide)
    assert prepared.wide.keys() == set(wide)
    record = {"carved_rows": tuple(wide)}
    assert oproj_ns["wide_switches"](prepared) == record


# --- N1: KDA verify ----------------------------------------------------------------------------


def test_n1_replaces_only_the_listed_verify_kernels():
    calls = []
    ns = _base_namespace(
        kda_verify_specs=lambda core, caches, r, t, pitch: ("specs", r, t, pitch),
        compile_kda_verify=lambda specs, *options: calls.append((specs, options)) or ("new", specs[1]),
    )  # fmt: skip
    source_objects(modeling_path(), ("_WIDE_SWITCHES", "record_switches", "wide_switches"), ns)
    source_objects(modeling_path(), ("N1_KERNELS", "FlatKda"), ns)
    kda = ns["FlatKda"].__new__(ns["FlatKda"])
    kda.caches, kda.n1_kernels = "caches", {}
    core = SimpleNamespace(heads=6, has_bias=False, device=SimpleNamespace(index=0))
    core.kernels = {(b, 8): ("old", b) for b in range(8, 17)} | {8: ("q1", 8)}
    kda.core = core
    policies = {**{(b, 8): (3216, 0) for b in range(8, 17)}, 8: (3216, 0)}
    kda.verify_kernels = {}
    kda._verify_override(policies)
    assert not calls and all(v[0] in ("old", "q1") for v in core.kernels.values())
    kda.verify_kernels = dict(ns["N1_KERNELS"])
    assert kda.verify_kernels == dict.fromkeys(range(11, 15), (1, 8, 2, 4))
    kda._verify_override(policies)
    assert [(s[1], o[-1]) for s, o in calls] == [(r, (1, 8, 2, 4)) for r in range(11, 15)]
    assert core.kernels[(12, 8)] == ("new", 12) and core.kernels[(10, 8)] == ("old", 10)
    assert core.kernels[(15, 8)] == ("old", 15) and core.kernels[8] == ("q1", 8)
    assert calls[0][1] == (6, 0, False, 0, 8, (1, 8, 2, 4))
    # the record names the kernels replaced, by (requests, q).
    record = ns["wide_switches"](kda)
    assert record["n1_kernels"] == {(r, 8): (1, 8, 2, 4) for r in range(11, 15)}
    assert record["n1"] == kda.verify_kernels
    assert ns["wide_switches"]()["kda"] == record


# --- N6: the drafter's ranked partial ----------------------------------------------------------


def test_n6_applies_in_the_headtp_drafter():
    """The decode drafter always runs the headtp layout (``DraftModel.bind_runtime`` builds only
    its runtime), so the draft stage records N6's choice with the layout and launches it."""
    bind = ast.unparse(function_node(modeling_path(), "DraftModel.bind_runtime"))
    assert "draft = self._headtp_runtime(" in bind
    build = function_node(modeling_path(), "build_flat_target")
    text = ast.unparse(build)
    assert "draft_attention" not in text and "_layout" not in text
    assert "record_switches('draft', draft_core, {'n6': choice, 'layout': layout})" in text
    assert "log_stage(plain.tp_rank, 'draft', values)" in text
    # The layout and N6's choice as the build computes them.
    picked = [
        n for n in build.body
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("layout", "choice") for t in n.targets)
    ]  # fmt: skip
    assert [ast.unparse(n.targets[0]) for n in picked] == ["layout", "choice"]
    ns = source_objects(modeling_path(), ("DRAFT_TILE_WARPS", "DRAFT_TILES", "DRAFT_TILE_CAP"), {})
    exec(compile(ast.Module(body=picked, type_ignores=[]), "build_flat_target", "exec"), ns)
    assert ns["layout"] == "headtp"
    assert ns["choice"] == (*ns["DRAFT_TILES"], True, ns["DRAFT_TILE_CAP"])
    assert ns["choice"][:2] == (64, 64) and ns["choice"][-2:] == (True, 64)  # 64x64, rolled, cap


def test_the_features_read_are_k31features_fields():
    """Every ``k31_features().<name>`` read in the target (``rubin_ull_features.active``) is a
    field or method of the family's ``K31Features``, and every name imported from the features
    module exists there."""
    features = family_dir() / "ops/common/rubin_ull_features.py"
    tree = ast.parse(features.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "K31Features")
    attrs = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    attrs |= {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    names |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name)}  # fmt: skip
    ours = ast.parse(modeling_path().read_text())
    imported = [
        a for n in ast.walk(ours)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("rubin_ull_features")
        for a in n.names
    ]  # fmt: skip
    assert imported and all(a.name in names for a in imported), [a.name for a in imported]
    alias = {a.asname or a.name for a in imported if a.name == "active"}
    assert alias == {"k31_features"}
    reads = {
        n.attr for n in ast.walk(ours)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id in alias
    }  # fmt: skip
    assert reads and reads <= attrs, sorted(reads - attrs)


# --- M2: the counting sort -------------------------------------------------------------------


_M2_CHECK = r"""
import ast, sys, textwrap
import torch, triton
import triton.language as tl
modeling, op, kind = sys.argv[1:4]
text = open(modeling).read()
names = {"SORT_ROWS", "SORT_BLOCK", "SORT_BINS", "SORT_TOKENS", "SORT_COUNT_WARPS", "_popc32",
         "_transpose_round", "_moe_sort_count", "sort_max_tiles", "MoeSortWindow"}
picked = []
for node in ast.parse(text).body:
    defined = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else {
        t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)}
    if defined & names:
        picked.append(node)
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
module = ast.fix_missing_locations(ast.Module(body=[future, *picked], type_ignores=[]))
ns = {"triton": triton, "tl": tl, "torch": torch, "_MOE_TOP_K": 16, "_MOE_TILE": 128,
      "Iterable": object, "Dict": dict, "Tuple": tuple, "__name__": "m2"}
exec(compile(module, modeling, "exec"), ns)
source = open(op).read()
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RubinMoeSort")
fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "reference")
ref = {"torch": torch}
exec("from __future__ import annotations\n" + textwrap.dedent(ast.get_source_segment(source, fn)), ref)
g = torch.Generator().manual_seed(7)
def routing(b):
    return torch.stack([torch.randperm(896, generator=g)[:16] for _ in range(b)])
# count: row counts over the windows, the window's edges, a skewed, an empty and a spread window,
# repeated experts in a row (the exact path).
cases = [(routing(b), 112 * w) for b, w in ((65, 0), (72, 7), (96, 3), (127, 5), (128, 1))]
for b in (72, 128):
    n = b * 16
    cases += [(torch.tensor([447, 448, 559, 560] * 4).repeat(b, 1), 448),
              (torch.arange(16).repeat(b, 1) + 448, 448), (torch.arange(16).repeat(b, 1), 448),
              ((torch.arange(16) * 7 + 448).repeat(b, 1), 448), (torch.full((b, 16), 453), 448)]
    tight = torch.full((n,), 448 + 111)
    tight[:111] = torch.arange(111) + 448
    cases.append((tight[torch.randperm(n, generator=g)].view(b, 16), 448))
    one = routing(b)
    one[3, 5] = one[3, 9]
    cases.append((one, 112 * (b % 8)))
runs = 0
for number, (ids, offset) in enumerate(cases):
    b = ids.shape[0]
    sorter = ns["MoeSortWindow"](torch.device("cpu"), [b], offset=offset, local=112)
    want = ref["reference"](None, ids, torch.ones(ids.shape), 896, 16, offset, 112, 128)
    for parity in ((1, 0, 3) if number % 16 == 0 else (1,)):
        # Each parity's own tables (3: parity 1's again), and no old contents reach the result:
        # the first run reads the warm-up's, the others poisoned ones.
        if parity != 1:
            for t in sorter.tables[parity % 2, b]:
                t.copy_(torch.randint(-2**31, 2**31 - 1, t.shape, generator=g))
        got = sorter.sort(ids.to(torch.int32).contiguous(), parity, pdl=False)
        assert all(x is y for x, y in zip(got, sorter.tables[parity % 2, b]))
        for name, x, y in zip(("t2e", "t2l", "e2p", "p2e", "total", "tiles"), got, want):
            assert x.dtype == torch.int32 and x.shape == y.shape, (b, offset, name)
            assert torch.equal(x, y.to(torch.int32)), (b, offset, parity, name)
        runs += 1
try:
    sorter.sort(ids.to(torch.int32).contiguous().to("meta"), 0, pdl=False)
    raise AssertionError("ids on another device were sorted")
except ValueError as error:
    assert "the sorter's tables on cpu" in str(error), error
print("M2 OK", kind, len(cases), runs)
"""


def test_m2_count_tables_are_the_references_byte_for_byte():
    """``count`` (``_moe_sort_count``) in Triton's interpreter, byte for byte
    ``RubinMoeSort.reference``'s (tables poisoned before the reruns of both parities)."""
    import subprocess

    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1")
    op = family_dir() / "ops/moe/rubin_moe_sort.py"
    run = subprocess.run(
        [sys.executable, "-I", "-c", _M2_CHECK, str(modeling_path()), str(op), "count"],
        capture_output=True, text=True, env=env, timeout=900,
    )  # fmt: skip
    if run.returncode != 0 and "interpret" in (run.stderr or "").lower() and "not" in run.stderr:
        pytest.skip(f"Triton's interpreter does not run here: {run.stderr[-300:]}")
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "M2 OK count 19 23"


class _Streams:
    """CUDA's streams and events stood in: each launch, record and wait noted on its stream."""

    def __init__(self):
        self.log: list = []
        self.current = "main"

    def current_stream(self, device=None):
        return _Stream(self.current, self)

    @contextlib.contextmanager
    def stream(self, s):
        before, self.current = self.current, s.name
        try:
            yield
        finally:
            self.current = before

    def note(self, *what):
        self.log.append((self.current, *what))


class _Stream:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def wait_event(self, event):
        self.cuda.log.append((self.name, "wait", event.name))


class _Event:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def record(self, stream=None):
        self.cuda.log.append(((stream.name if stream else self.cuda.current), "record", self.name))


def _wide_moe(sort_first: bool, sorter: bool):
    """``Fc12MoE.run_wide`` on stand-ins: the routing, the glue stream's operands (as
    ``Fc12Glue.operands``: the record unless ``recorded``), the side stream's down GEMMs, the
    sort (M2's kernel or ``moe_sort``), FC12 and the wide tail, each noted on its stream."""
    cuda = _Streams()
    prepared = SimpleNamespace(PreparedMoE=object, JOIN={16: ""})
    ns = {
        "torch": SimpleNamespace(cuda=cuda), "contextlib": contextlib, "Optional": Optional,
        "Tuple": Tuple, "module": {"cutedsl.moe.prepared": prepared}.__getitem__,
    }  # fmt: skip
    source_objects(modeling_path(), ("FRONT_ROWS", "Fc12MoE"), ns)
    moe = object.__new__(ns["Fc12MoE"])
    glue = SimpleNamespace(routed=_Event("glue.routed", cuda), ones=torch.ones((128, 16)))
    glue.counters = "counters"

    def operands(experts, latent, weights, b, scales, recorded=False):
        if not recorded:
            glue.routed.record(cuda.current_stream())
        with cuda.stream(_Stream("glue", cuda)):
            cuda.log.append(("glue", "wait", "glue.routed"))
            cuda.note("quantize", b)
        return "x4", "sf", "scales"

    glue.operands = operands
    glue.join = lambda: cuda.log.append(("main", "wait", "glue.done"))
    glue.release = lambda layer, b: cuda.note("release")
    tail = SimpleNamespace(run=lambda partial, b, join: (cuda.note("LatentRows"), join()))
    fronted = _Event("fronted", cuda)  # S2: FrontOnce.route records it before its routing launch
    once = SimpleNamespace(
        route=lambda x: (fronted.record(), cuda.note("route", x.shape[0]), "ids", "w", "lat")[2:]
    )
    experts = SimpleNamespace(
        sort=lambda ids, ones: cuda.note("moe_sort") or "engine tables",
        assemble=lambda *a: a, launch=lambda operands, counters: cuda.note("fc12", operands[2]),
    )  # fmt: skip
    moe.state = SimpleNamespace(
        device="cpu", stream=_Stream("side", cuda), begin=_Event("begin", cuda),
        shared_done=_Event("shared_done", cuda),
    )  # fmt: skip
    moe.wide = (None, None, tail, once)
    moe.glue, moe.experts, moe.fronted, moe.layer = glue, experts, fronted, 3
    moe.partial, moe.prefetch, moe.tuned, moe.feeds = torch.zeros((128, 4)), None, None, {}
    moe.route32, moe.route32_rows, moe.sort_first = None, frozenset(), sort_first
    moe.sorter = None
    if sorter:
        moe.sorter = SimpleNamespace(
            rows=(96,), sort=lambda ids, p: cuda.note("count", p) or "ours"
        )
    moe.shared_parts = lambda x, b: (None, lambda: cuda.note("down+push"))
    return moe, cuda


@pytest.mark.parametrize("sorter", [False, True])
def test_m2_stream_sorts_right_behind_the_routing(sorter):
    """M2 ``stream``: the glue stream forks at the routing (its event recorded before the sort),
    the sort is the main stream's next launch, FC12 waits for the glue stream as before; the side
    stream forks at the front kernel (S2: its event recorded inside ``FrontOnce.route``); the same
    launches on the same values as without it (the same DAG: each launch has the same
    predecessors). Off: order (the operands' fork, the side stream's, the sort)."""
    x = torch.zeros((96, 4))
    moe, cuda = _wide_moe(False, sorter)
    moe.run_wide(x, 96)
    off = list(cuda.log)
    sort = ("main", "count", 1) if sorter else ("main", "moe_sort")
    assert off[:5] == [("main", "record", "fronted"), ("main", "route", 96),
                       ("main", "record", "glue.routed"), ("glue", "wait", "glue.routed"),
                       ("glue", "quantize", 96)]  # fmt: skip
    assert off.index(("side", "wait", "fronted")) < off.index(("side", "down+push"))
    assert (
        off.index(("side", "down+push"))
        < off.index(sort)
        < off.index(("main", "wait", "glue.done"))
    )
    moe, cuda = _wide_moe(True, sorter)
    moe.run_wide(x, 96)
    on = list(cuda.log)
    assert on[:4] == [("main", "record", "fronted"), ("main", "route", 96),
                      ("main", "record", "glue.routed"), sort]  # fmt: skip
    assert on.count(("main", "record", "glue.routed")) == 1 and on.count(sort) == 1
    assert on.index(sort) < on.index(("glue", "wait", "glue.routed"))
    assert on.index(sort) < on.index(("side", "wait", "fronted"))
    fc12 = ("main", "fc12", "ours" if sorter else "engine tables")
    for log in (off, on):
        assert (
            log.index(("main", "wait", "glue.done"))
            < log.index(fc12)
            < log.index(("main", "LatentRows"))
        )
    assert sorted(map(repr, off)) == sorted(map(repr, on))  # the same launches, records, waits


def test_the_step_levers_follow_the_front(policy):
    """The fixed step levers; the merged front's N tiles (M6) only with that front."""
    ns = _base_namespace(module={"cutedsl.common.policy": policy}.__getitem__)
    names = ("FRONT_TMEM_N", "N1_KERNELS", "DRAFT_TILE_WARPS", "DRAFT_TILE_CAP", "DRAFT_TILES",
             "SORT_ROWS", "LATENT_LOADS_ROWS", "NORM_VOTE_ROWS", "row_spans", "step_levers")  # fmt: skip
    source_objects(modeling_path(), names, ns)
    on = ns["step_levers"](SimpleNamespace(front="tmem_merged"))
    off = ns["step_levers"](SimpleNamespace(front="once"))
    assert on["m6"] == dict.fromkeys(range(65, 129), 64) and off["m6"] == {}
    assert {k: v for k, v in on.items() if k != "m6"} == {k: v for k, v in off.items() if k != "m6"}
    assert on["n1"] == dict.fromkeys(range(11, 15), (1, 8, 2, 4)) and on["n3"] == "carveout"
    assert on["n6"] == (64, 64, 8, True, 64) and on["m2"] == ("count", False, True)
    assert (on["l1"], on["q1"], on["s2"]) == ("65-128", "65-128", "65-128.fit")
    assert (on["h8b"], on["vote"]) == ("113-128", "65-128")

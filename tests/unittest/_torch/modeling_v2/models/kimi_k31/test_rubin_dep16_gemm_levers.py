# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's decode levers of a generation server's graph steps (CPU, from source).

``_Levers`` (``rt.levers``) holds three switches checked here, set by the role's constants
(``_levers``): on for a generation server, off on a context server. Its three MXFP8 levers, off on
both roles, are ``test_rubin_dep16_mxfp8_levers.py``'s.

* ``dsa_rope_before_idx_join``: ``mla_generate`` issues W_UK and the RoPE kernel
  (``_mla_gen_query``) before the current stream joins the indexer's stream; the logits weights,
  the indexer, the attention, W_UV and the gate after it;
* ``dsa_logits_first`` (over the former): the indexer's query projection and logits weights on
  the aux stream, q_b, g_proj, W_UK and the RoPE kernel on the fork stream, the logits, Top-K and
  remap on the current stream before it joins the fork stream for the attention
  (``_mla_generate_logits_first``);
* ``draft_q_a_kv_a_fused``: the drafter table stacks each layer's q_a and kv_a rows
  (``d_l{j}_q_a_kv_a``; ``d_kv_a`` keeps its own stack of the five kv_a for the history
  projection, so the kv_a rows are held twice) and ``draft_step`` projects both in one GEMM.

The block functions run from modeling.py over recording stand-ins whose values follow their inputs
(small integers, so every product is exact): each setting's plan (kernels, streams, events in issue
order) is checked, every kernel reads what another stream wrote only after waiting for it, every
side stream is joined, and the lever's step returns the bits of the step without it.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, NamedTuple

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TARGET_DIR = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16"
_MODELING = _TARGET_DIR / "modeling.py"
_WEIGHTS = _TARGET_DIR / "weights.py"
_ROUTING = _FAMILY / "routing.py"

# mla_generate's lever off, and on (a generation server's constants: _levers("generation")).
_SETTINGS = {
    "off": (),
    "rope_first": ("dsa_rope_before_idx_join",),
    "logits_first": ("dsa_rope_before_idx_join", "dsa_logits_first"),
}


def _tree(path: Path = _MODELING) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _constants(tree: ast.Module, ns: dict) -> dict:
    """``tree``'s module-level assignments that evaluate on their own over ``ns``, in order."""
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = eval(compile(ast.Expression(node.value), "<c>", "eval"), ns)
        except Exception:
            continue
    return ns


def _exec(tree: ast.Module, names, ns: dict, path: Path = _MODELING) -> dict:
    """The module-level definitions of ``names`` of ``tree``, executed over ``ns``."""
    wanted, nodes = set(names), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            nodes.append(copy.deepcopy(node))
            wanted.discard(node.name)
    assert not wanted, f"{path.name} no longer defines {sorted(wanted)}"
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(path), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _function(name: str, tree: ast.Module | None = None) -> ast.FunctionDef:
    (fn,) = [n for n in (tree or _tree()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


class _Plan:
    """Issue order on the step's streams. ``clock[s][o]``: the point of stream ``o``'s order that
    stream ``s`` waited for; a kernel that reads memory another stream wrote must have waited
    past that write."""

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {}
        self.writer: dict = {}  # storage -> (stream, point)
        self.ops: list = []
        self.problems: list = []
        self.last: dict = {}

    def tick(self) -> int:
        self.now += 1
        return self.now

    def mark(self, stream: str) -> dict:
        return {**self.clock.get(stream, {}), stream: self.tick()}

    def wait(self, waiter: str, position: dict) -> None:
        mine = self.clock.setdefault(waiter, {})
        for s, at in position.items():
            if s != waiter:
                mine[s] = max(mine.get(s, 0), at)

    def kernel(self, name: str, reads=(), writes=()) -> None:
        at, s = self.tick(), self.stream
        for t in reads:
            src = self.writer.get(_ptr(t))
            if src is None or src[0] == s:
                continue
            if self.clock.get(s, {}).get(src[0], 0) < src[1]:
                self.problems.append(f"{name}@{s} reads what {src[0]} wrote at {src[1]}, unwaited")
        for t in writes:
            self.writer[_ptr(t)] = (s, at)
        self.last[s] = at
        self.ops.append(f"{name}@{s}")


class _Event:
    def __init__(self, plan: _Plan, name: str):
        self.plan, self.name, self.at = plan, name, None

    def record(self) -> None:
        self.at = self.plan.mark(self.plan.stream)
        self.plan.ops.append(f"record:{self.name}@{self.plan.stream}")

    def wait(self) -> None:
        assert self.at is not None, f"a wait on {self.name}, which nothing recorded"
        self.plan.wait(self.plan.stream, self.at)
        self.plan.ops.append(f"wait:{self.name}@{self.plan.stream}")


class _Stream:
    def __init__(self, plan: _Plan, name: str):
        self.plan, self.name = plan, name

    def wait_stream(self, other: "_Stream") -> None:
        self.plan.wait(self.name, self.plan.mark(other.name))
        self.plan.ops.append(f"wait_stream:{other.name}@{self.name}")


def _torch(plan: _Plan, main: _Stream) -> SimpleNamespace:
    """``torch`` as the block sees it on a capture: streams of the plan, allocation as torch's."""

    @contextmanager
    def stream(s: _Stream):
        outer, plan.stream = plan.stream, s.name
        try:
            yield
        finally:
            plan.stream = outer

    def cat(tensors, dim):
        out = torch.cat(tensors, dim)
        plan.kernel("latent_cat", tuple(tensors), (out,))
        return out

    cuda = SimpleNamespace(
        stream=stream, current_stream=lambda: main, is_current_stream_capturing=lambda: True
    )
    return SimpleNamespace(
        cuda=cuda,
        cat=cat,
        **{n: getattr(torch, n) for n in ("empty", "int32", "uint8", "float32", "Tensor")},
        uint32=torch.int64,
    )


class _Logger:
    def __init__(self, plan: _Plan):
        self.plan, self.once = plan, []

    def info_once(self, msg, *, key):
        self.once.append(key)
        self.plan.ops.append("log")


def _ints(gen: torch.Generator, *shape) -> torch.Tensor:
    """Small integers as float32: every product and sum below is exact."""
    return torch.randint(-3, 4, shape, generator=gen).float()


def _gemm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (a.double() @ w.double().t()).float()


# --- mla_generate on a decode graph step ---

_H, _NOPE, _ROPE, _KVL, _V, _QL, _HID, _IH, _ID = 2, 4, 2, 6, 3, 5, 9, 2, 4
_ROWS = 3


def _mla(plan: _Plan, levers: tuple, mx: bool = False) -> SimpleNamespace:
    """mla_generate's graph path from modeling.py over stand-ins, its layer and inputs. ``mx``
    (``mla_mxfp8``): g_proj and o_proj are ``_Mx`` weights whose stand-in GEMM is the same exact
    product, so every setting keeps the lever-off bits and only the entries differ."""
    tree = _tree()
    main = _Stream(plan, "main")
    seen: dict = {}
    gen = torch.Generator().manual_seed(31)
    q_b = _ints(gen, _H * (_NOPE + _ROPE), _QL)
    iwqb = _ints(gen, _IH * _ID, _QL)
    kv_b = _ints(gen, _H * (_NOPE + _V), _KVL)
    # [q_a | kv_a | k_pe] selects x's columns: q_a = x[:, :QL], kv_a = x[:, :KVL], k_pe the next
    # ROPE columns; with the norms' stand-ins below the inlined projection gives _mla_project's.
    sel = torch.zeros(_HID, _QL + _KVL + _ROPE)
    for col, row in enumerate([*range(_QL), *range(_KVL), *range(_KVL, _KVL + _ROPE)]):
        sel[row, col] = 1
    L = SimpleNamespace(
        q_b=q_b,
        qkv_a_t=sel,
        q_a_norm=torch.ones(_QL),
        kv_a_norm=torch.ones(_KVL),
        idx_wq_b_t=iwqb.t(),
        g=_ints(gen, _H * _V, _HID),
        o=_ints(gen, _HID, _H * _V),
        k_b=kv_b[: _H * _NOPE].view(_H, _NOPE, _KVL),
        v_b_t=kv_b[_H * _NOPE :].view(_H, _V, _KVL).transpose(1, 2),
    )
    index_cache = torch.zeros(_ROWS, _ID)
    latent_cache = torch.zeros(_ROWS, _KVL + _ROPE)
    x = _ints(gen, _ROWS, _HID)
    names = {id(L.q_b): "q_b", id(L.g): "g_proj", id(L.o): "o_proj"}

    def rubin_linear(a, w, tf32):
        out = _gemm(a, w)
        plan.kernel(names[id(w)], (a, w), (out,))
        return out

    def rubin_mxfp8_linear(a, w, s, pinned=False):
        # g_proj runs pinned, o_proj (K = 12288) on the shared low-M / mid-M GEMM
        assert pinned is (names[id(w)] == "g_proj"), (names[id(w)], pinned)
        assert s.dtype == torch.uint8
        out = _gemm(a, w)
        plan.kernel("mx:" + names[id(w)], (a, w, s), (out,))
        return out

    def cublas_mm(a, b):
        out = (a.double() @ b.double()).float()
        plan.kernel("qkv_a" if b is L.qkv_a_t else "idx_wq_b", (a, b), (out,))
        return out

    def flashinfer_rmsnorm(a, w, eps):
        q_a = w is L.q_a_norm
        out = a + 1 if q_a else a - 1
        plan.kernel("q_a_norm" if q_a else "kv_a_norm", (a, w), (out,))
        return out

    def fused_cat_fp8(pe, nope, ue8m0):
        # The kernel walks the flattened rows with one stride: the leading dims must collapse.
        for t in (pe, nope):
            assert t.stride(0) == t.stride(1) * t.shape[1], (t.shape, t.stride())
        rows = torch.cat([pe, nope], -1)
        seen.setdefault("iq", []).append(rows.clone())
        q8, qs = rows * 2, rows.abs().amax(-1, keepdim=True) + 1
        plan.kernel("fused_cat_fp8", (pe, nope), (q8, qs))
        return q8, qs

    def _idx_keys(L_, a):
        k8, ks, w = a[:, :_ID] * 1, a[:, :1] * 1, a[:, :_IH] * 1
        plan.kernel("idx_keys", (a,), (k8, ks, w))
        return k8, ks, w

    def _idx_insert(i, k8, ks, n, st, rt):
        index_cache.copy_(k8 + ks)
        plan.kernel("idx_insert", (k8, ks), (index_cache,))

    def _mla_project(L_, a, st, rt):
        qr = a[:, :_QL] + 1
        latent = torch.cat([a[:, :_KVL] - 1, a[:, _KVL : _KVL + _ROPE]], -1)
        plan.kernel("mla_project", (a,), (qr, latent))
        return qr, latent

    def rubin_idx_weights(w, qs, scale):
        out = w * qs[:, : w.shape[1]] * scale
        plan.kernel("idx_weights", (w, qs), (out,))
        return out

    def bmm_out(a, b, out):
        out.copy_(torch.bmm(a.double(), b.double()).float())
        plan.kernel("W_UK" if b.shape[1] == _NOPE else "W_UV", (a, b), (out,))
        return out

    def mla_rope_generation(**kw):
        fq, qpe, lat = kw["fused_q"], kw["q_pe"], kw["latent_cache"]
        fq[..., _KVL:] = qpe
        kw["quant_q_buffer"].copy_((fq.abs() * 3).clamp(max=255).to(kw["quant_q_buffer"].dtype))
        latent_cache.copy_(lat)
        plan.kernel("rope", (fq, qpe, lat), (fq, kw["quant_q_buffer"], latent_cache))

    def _idx_decode(i, q8, w, st, rt):
        topk = q8.sum((1, 2)) + w.sum(1) + index_cache.sum(1)
        plan.kernel("idx_decode", (q8, w, index_cache), (topk,))
        return topk

    def _idx_remap(i, topk, req, table, st, rt):
        out = topk + 1
        plan.kernel("idx_remap", (topk,), (out,))
        return out

    def thop_attention(**kw):
        q, idx, qq = kw["q"], kw["sparse_attn_indices"], kw["quant_q_buffer"]
        lat = q.view(_ROWS, _H, -1)[..., :_KVL] + idx.view(-1, 1, 1) + latent_cache[:, None, :_KVL]
        lat = lat + qq.float().view(_ROWS, _H, -1)[..., :_KVL]
        out = lat.reshape(_ROWS, -1).contiguous()
        plan.kernel("attention", (q, kw["q_pe"], idx, qq, latent_cache), (out,))
        return out

    def k31_sigmoid_gate_mul(out, g):
        res = out * g
        plan.kernel("gate", (out, g), (res,))
        return res

    log = _Logger(plan)
    ns = dict(
        torch=_torch(plan, main),
        logger=log,
        NamedTuple=NamedTuple,
        Any=Any,
        Optional=None,
        MappingProxyType=MappingProxyType,
        rubin_linear=rubin_linear,
        rubin_mxfp8_linear=rubin_mxfp8_linear,
        cublas_mm=cublas_mm,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
        fused_cat_fp8=fused_cat_fp8,
        _idx_keys=_idx_keys,
        _idx_insert=_idx_insert,
        _mla_project=_mla_project,
        rubin_idx_weights=rubin_idx_weights,
        bmm_out=bmm_out,
        mla_rope_generation=mla_rope_generation,
        _idx_decode=_idx_decode,
        _idx_remap=_idx_remap,
        thop_attention=thop_attention,
        k31_sigmoid_gate_mul=k31_sigmoid_gate_mul,
        _mla_window=lambda i, st, rt: 64,
        _MlaW=object,
        _Step=object,
        _Rt=object,
    )
    _constants(tree, ns)
    ns.update(
        _MLA_Q_LORA=_QL,
        _MLA_EPS=1e-6,
        _MLA_HEADS=_H,
        _MLA_NOPE=_NOPE,
        _MLA_ROPE=_ROPE,
        _MLA_QK=_NOPE + _ROPE,
        _MLA_KV_LORA=_KVL,
        _MLA_LATENT=_KVL + _ROPE,
        _MLA_V=_V,
        _MLA_OUT_HEADS=_H,
        _IDX_HEADS=_IH,
        _IDX_DIM=_ID,
        _IDX_HALF=_ID // 2,
        _IDX_WEIGHT_SCALE=0.5,
        _MLA_ATTN={},
        _MLA_ROPE_GEN={},
        _MLA_GENERATION=2,
    )
    _exec(
        tree,
        (
            "_Levers",
            "_Mx",
            "_dense",
            "_MlaQuery",
            "_mla_gen_heads",
            "_mla_gen_query",
            "_mla_generation",
            "_mla_attend",
            "_mla_absorb",
            "_mla_expand",
            "_idx_queries",
            "_mla_output",
            "mla_generate",
            "_mla_generate_logits_first",
        ),
        ns,
    )

    def ev(name: str) -> _Event:
        return _Event(plan, name)

    rt = SimpleNamespace(
        # _EV_ATTN's pair is named for its use after _mla_project's join (q_b's and g_proj's);
        # dsa_logits_first takes it for kv_a's norm on the aux stream (its fork and join).
        events=(
            (ev("q_ready"), ev("g_ready")),
            (ev("idx_fork"), ev("idx_join")),
            (ev("qr_ready"), ev("idx_ready")),
            (ev("latent_ready"), ev("rope_done")),
        ),
        aux_stream=_Stream(plan, "aux"),
        idx_stream=_Stream(plan, "idx"),
        fork_stream=_Stream(plan, "fork"),
        levers=ns["_Levers"](**dict.fromkeys(levers, True)),
        dsa_pools=SimpleNamespace(local={3: 0}, rope_positions=64),
        mla_rope=None,
    )
    if mx:  # the table's MXFP8 pairs: the stand-in GEMM reads the values as the weight itself
        L.g = ns["_Mx"](L.g, torch.zeros(1, dtype=torch.uint8))
        L.o = ns["_Mx"](L.o, torch.zeros(1, dtype=torch.uint8))
    st = SimpleNamespace(
        num_tokens=_ROWS,
        num_contexts=0,
        num_generations=1,
        cuda_graph=True,
        dsa=SimpleNamespace(rope_gen={}, sparse_gen={}, req_idx_gen=None, block_table_gen=None),
    )
    out = ns["mla_generate"](L, 3, x, st, rt)
    return SimpleNamespace(
        out=out, seen=seen, caches=(index_cache, latent_cache), log=log, ns=ns, plan=plan
    )


def _mla_run(setting: str) -> SimpleNamespace:
    return _mla(_Plan(), _SETTINGS[setting])


_IDX = [
    "record:idx_fork@main",
    "wait:idx_fork@idx",
    "idx_keys@idx",
    "idx_insert@idx",
    "record:idx_join@idx",
    "mla_project@main",
]
_TWO = [
    "wait_stream:main@fork",
    "q_b@fork",
    "record:q_ready@fork",
    "g_proj@fork",
    "record:g_ready@fork",
    "idx_wq_b@main",
    "fused_cat_fp8@main",
]
_QUERY = ["W_UK@main", "rope@main"]
_ATTEND = [
    "idx_decode@main",
    "idx_remap@main",
    "attention@main",
    "W_UV@main",
    "wait:g_ready@main",
    "gate@main",
    "o_proj@main",
]
_LOGS = ["log"] * 3
_PLANS = {
    "logits_first": [
        *_IDX[:-1],
        "qkv_a@main",
        "record:q_ready@main",
        "q_a_norm@main",
        "record:qr_ready@main",
        "wait:q_ready@aux",
        "kv_a_norm@aux",
        "record:g_ready@aux",
        "wait:qr_ready@aux",
        "idx_wq_b@aux",
        "fused_cat_fp8@aux",
        "wait:idx_join@aux",
        "idx_weights@aux",
        "record:idx_ready@aux",
        "wait:qr_ready@fork",
        "q_b@fork",
        "g_proj@fork",
        "wait:g_ready@main",
        "latent_cat@main",
        "record:latent_ready@main",
        "wait:latent_ready@fork",
        *[op.replace("@main", "@fork") for op in _QUERY],
        "record:rope_done@fork",
        "wait:idx_ready@main",
        "idx_decode@main",
        "idx_remap@main",
        "wait:rope_done@main",
        *_LOGS,
        "log",
        "attention@main",
        "W_UV@main",
        "wait:rope_done@main",
        "gate@main",
        "o_proj@main",
    ],
    "off": [
        *_IDX,
        *_TWO,
        "wait:idx_join@main",
        "idx_weights@main",
        "wait:q_ready@main",
        *_LOGS,
        *_QUERY,
        *_ATTEND,
    ],
    "rope_first": [
        *_IDX,
        *_TWO,
        "wait:q_ready@main",
        *_QUERY,
        "wait:idx_join@main",
        "idx_weights@main",
        *_LOGS,
        "log",
        *_ATTEND,
    ],
}


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_each_setting_takes_its_plan(setting):
    run = _mla_run(setting)
    assert run.plan.ops == _PLANS[setting], run.plan.ops


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_every_kernel_reads_after_its_writers_and_the_side_streams_are_joined(setting):
    run = _mla_run(setting)
    assert not run.plan.problems, run.plan.problems
    main = run.plan.clock.get("main", {})
    sides = {s for s in run.plan.last if s != "main"}
    assert sides >= {"idx", "fork"}, sides
    for side in sides:
        assert main.get(side, 0) >= run.plan.last[side], f"{side} not joined: {run.plan.ops}"


@pytest.mark.parametrize("setting", ["rope_first", "logits_first"])
def test_the_lever_returns_the_bits_of_the_step_without_it(setting):
    off, on = _mla_run("off"), _mla_run(setting)
    assert torch.equal(off.out, on.out)
    for a, b in zip(off.caches, on.caches):
        assert torch.equal(a, b)
    (iq_off,), (iq_on,) = off.seen["iq"], on.seen["iq"]
    assert torch.equal(iq_off, iq_on)


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_each_setting_logs_its_lines(setting):
    keys = _mla_run(setting).log.once
    ns = _mla_run("off").ns
    want = [ns["_IDX_FORK_LINE"], ns["_IDX_SHORT_LINE"], ns["_MLA_GATE_FORK_LINE"]]
    if "dsa_logits_first" in _SETTINGS[setting]:
        want.append(ns["_DSA_LOGITS_FIRST_LINE"])
    elif "dsa_rope_before_idx_join" in _SETTINGS[setting]:
        want.append(ns["_DSA_ROPE_FIRST_LINE"])
    assert keys == want, keys


def test_the_generation_call_runs_the_query_unless_handed_it():
    """``_mla_generation`` runs ``_mla_gen_query`` unless handed its result (``mq``), then the
    indexer and the attention; the prefill's generation rows take it without."""
    body = [ast.unparse(s) for s in _function("_mla_generation").body]
    assert "if mq is None:\n    mq = _mla_gen_query(L, i, q, latent, st, rt)" in body, body
    prefill = ast.unparse(_function("mla_prefill"))
    assert "_mla_generation(L, i, q[c:n], latent[c:n], q8, w, out[c:n], st, rt)" in prefill


# --- the levers' construction ---


def _levers_ns() -> dict:
    tree = _tree()
    ns = _constants(tree, dict(MappingProxyType=MappingProxyType))
    return _exec(tree, ("_Levers", "_levers"), {**ns, "NamedTuple": NamedTuple})


#: The MXFP8 levers (``test_rubin_dep16_mxfp8_levers.py``): off on both roles.
_MXFP8_LEVERS = ("mla_mxfp8", "moe_latent_mxfp8", "draft_mxfp8")
#: The 128-head sparse MLA decode (``test_rubin_dep16_mla_gen_pad128.py``): on for a generation
#: server.
_PAD_LEVERS = ("mla_gen_pad128",)


def test_a_generation_server_runs_the_three_levers_and_a_context_server_none():
    """A generation server runs this file's three levers (and the 128-head sparse MLA decode), not
    the MXFP8 levers, which its constants set False; a context server's constants name none of
    them."""
    ns = _levers_ns()
    levers, cls, consts = ns["_levers"], ns["_Levers"], ns["_CONSTANTS"]
    gen = levers("generation")
    on = {"dsa_rope_before_idx_join", "dsa_logits_first", "draft_q_a_kv_a_fused", *_PAD_LEVERS}
    assert set(cls._fields) == on | set(_MXFP8_LEVERS)
    assert {k for k in gen._fields if getattr(gen, k)} == on
    assert not any(levers("context")) and not any(cls())
    for key in cls._fields:
        assert consts["generation"][key] is (key in on) and key not in consts["context"], key
    # mla_generate's plan on a generation server is the "logits_first" setting's
    assert {k for k in gen._fields if getattr(gen, k)} >= set(_SETTINGS["logits_first"])


def test_the_levers_are_the_role_constants_and_read_no_variable():
    """``_levers`` reads the role's constants only; routing records no dep16 lever variable; the
    runtime binds the core's levers, which are bound before the weight table (it lays out the
    drafter's stacked rows)."""
    fn = _function("_levers")
    assert [a.arg for a in fn.args.args] == ["role"]
    src = ast.unparse(fn)
    assert "_CONSTANTS[role]" in src and "lever_env" not in src and "environ" not in src, src
    (env,) = [
        ast.literal_eval(n.value)
        for n in _tree(_ROUTING).body
        if isinstance(n, ast.Assign) and [getattr(t, "id", "") for t in n.targets] == ["LEVER_ENV"]
    ]
    assert not [v for v in env if "DEP16" in v], env
    build = ast.unparse(_function("_build_runtime"))
    assert "levers=core.levers" in build
    init = ast.unparse(_tree())
    assert init.index("self.levers = _levers(self.role)") < init.index(
        "self.dims = _weights.dims(self)"
    ), "the levers are bound before the weight table, which lays out the drafter's stacked rows"
    dims = ast.unparse(_function("dims", _tree(_WEIGHTS)))
    assert "draft_q_a_kv_a_fused=bool(getattr(levers, 'draft_q_a_kv_a_fused', False))" in dims


# --- the weight table and the views ---


def _weights_ns() -> dict:
    """weights.py's module body without its imports, over the names it imports."""
    tree = _tree(_WEIGHTS)
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    import glob as _glob
    import os as _os
    from contextlib import ExitStack as _ExitStack
    from typing import Iterator

    # A module of its own: dataclass() resolves the string annotations through sys.modules.
    mod = types.ModuleType("_dep16_weights_under_test")
    sys.modules[mod.__name__] = mod
    ns = mod.__dict__
    ns.update(
        torch=torch,
        nn=torch.nn,
        MappingProxyType=MappingProxyType,
        NamedTuple=NamedTuple,
        dataclass=dataclass,
        Any=Any,
        Callable=Callable,
        Iterator=Iterator,
        contextmanager=contextmanager,
        ExitStack=_ExitStack,
        glob=_glob,
        os=_os,
        safe_open=None,
        modelopt_nvfp4_mismatch=lambda cfg: None,
    )
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_WEIGHTS), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _dims(ns: dict, **levers) -> Any:
    values = dict(
        num_layers=8,
        hidden=7168,
        vocab=163840,
        dtype=torch.bfloat16,
        role="generation",
        kda_layers=(0, 1, 2, 4, 5, 6),
        mla_layers=(3, 7),
        moe_layers=tuple(range(1, 8)),
        dense_layers=(0,),
        attn_res_block=12,
        dp_size=16,
        kda_tp=1,
        kda_tp_rank=0,
        kda_heads=96,
        kda_head_dim=128,
        kda_conv=4,
        mla_heads=96,
        mla_o_heads=96,
        q_lora=1536,
        kv_lora=512,
        nope=128,
        rope=64,
        v_dim=128,
        helix_cp=1,
        helix_cp_rank=0,
        index_heads=32,
        index_dim=128,
        num_experts=896,
        moe_tp=1,
        moe_tp_rank=0,
        moe_ep=16,
        moe_ep_rank=0,
        local_experts=56,
        expert_offset=0,
        latent=3584,
        moe_inter=3072,
        shared_inter=6144,
        dense_inter=33792,
    )
    D = ns["Dims"]
    return D(**{f: values[f] for f in D._fields if f in values})._replace(**levers)


def _meta_elements(ns: dict, d: Any, rows) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in ns["declare"](d, rows).values())


@pytest.mark.parametrize("fused", [False, True], ids=["off", "fused"])
def test_the_drafter_table_stacks_q_a_with_kv_a_only_under_the_lever(fused):
    ns = _weights_ns()
    d, off = _dims(ns, draft_q_a_kv_a_fused=fused), _dims(ns)
    rows = ns["w3_rows"](d)
    names = {r.name for r in rows}
    for j in range(5):
        assert (f"d_l{j}_q_a_kv_a" in names) == fused and (f"d_l{j}_q_a" in names) != fused
    # Every checkpoint key is still read. Under the lever the kv_a rows are held twice (d_kv_a
    # stacks the five layers' for the history projection): exactly five layers' kv_a rows more.
    assert ns["_read_keys"](d) == ns["_read_keys"](off)
    extra = _meta_elements(ns, d, rows) - _meta_elements(ns, off, ns["w3_rows"](off))
    assert extra == (5 * 576 * 7168 if fused else 0), extra
    if fused:
        (row,) = [r for r in rows if r.name == "d_l2_q_a_kv_a"]
        assert row.shape(d) == (1536 + 576, 7168)
        ((keys, _),) = row.sources(d, None)
        assert keys == (
            "layers.2.self_attn.q_a_proj.weight",
            "layers.2.self_attn.kv_a_proj_with_mqa.weight",
        )


def test_the_drafter_views_read_the_stack():
    """``_draft_views`` hands q_a as a view of its layer's stack under the lever."""
    for fused in (False, True):
        views = _draft_views(fused)
        for L in views.layers:
            assert (L.q_a_kv_a is not None) == fused
            if fused:
                assert _ptr(L.q_a) == _ptr(L.q_a_kv_a) and L.q_a.shape == (_DQL, _DHID)
                assert torch.equal(L.q_a_kv_a[_DQL:], L.kv_a)


# --- draft_step ---

_DL, _DH, _DNOPE, _DROPE, _DKVL, _DV, _DQL, _DHID, _DI, _TAPS, _GROUPS = (
    2,
    2,
    3,
    2,
    4,
    3,
    5,
    6,
    4,
    2,
    3,
)
_DROWS, _BLOCK = 4, 2


def _draft_weights(fused: bool, mx: bool = False) -> dict:
    gen = torch.Generator().manual_seed(77)
    w = {
        "d_fc": _ints(gen, _DHID, 2 * _DHID),
        "d_final_norm": _ints(gen, _DHID),
        "d_kv_a": _ints(gen, _DL * (_DKVL + _DROPE), _DHID),
        "d_kv_a_norm": _ints(gen, _DL, _DKVL),
    }
    for j in range(_DL):
        w.update(
            {
                f"d_l{j}_kv_b": _ints(gen, _DH * (_DNOPE + _DV), _DKVL),
                f"d_l{j}_in_norm": _ints(gen, _DHID),
                f"d_l{j}_post_norm": _ints(gen, _DHID),
                f"d_l{j}_q_a_norm": _ints(gen, _DQL),
                f"d_l{j}_q_b": _ints(gen, _DH * (_DNOPE + _DROPE), _DQL),
                f"d_l{j}_o": _ints(gen, _DHID, _DH * _DV),
                f"d_l{j}_gate_up": _ints(gen, 2 * _DI, _DHID),
                f"d_l{j}_down": _ints(gen, _DHID, _DI),
                f"d_l{j}_attn_conv_base": _ints(gen, 2, _TAPS, _DHID),
                f"d_l{j}_attn_conv_proj": _ints(gen, 2 * _TAPS * _GROUPS, _DHID),
                f"d_l{j}_mlp_conv_base": _ints(gen, 2, _TAPS, _DHID),
                f"d_l{j}_mlp_conv_proj": _ints(gen, 2 * _TAPS * _GROUPS, _DHID),
            }
        )
        q_a = _ints(gen, _DQL, _DHID)
        latent = _DKVL + _DROPE
        kv_a = w["d_kv_a"][j * latent : (j + 1) * latent]
        if fused:  # the table's stack: q_a's rows, then the layer's kv_a rows (a copy)
            w[f"d_l{j}_q_a_kv_a"] = torch.cat([q_a, kv_a])
        else:
            w[f"d_l{j}_q_a"] = q_a
        if mx:  # draft_mxfp8: the table's {name}_w / {name}_s pairs (stand-in values: the weight)
            for n in ("o", "gate_up", "down"):
                w[f"d_l{j}_{n}_w"] = w.pop(f"d_l{j}_{n}")
                w[f"d_l{j}_{n}_s"] = torch.zeros(1, dtype=torch.uint8)
    return w


def _draft_views(fused: bool):
    tree = _tree()
    ns = _exec(
        tree,
        ("_Mx", "_mx_view", "_DraftLayerW", "_DraftW", "_draft_views"),
        {
            "NamedTuple": NamedTuple,
            "Any": Any,
            "Optional": None,
            "nn": SimpleNamespace(ParameterDict=dict),
        },
    )
    return ns["_draft_views"](_draft_weights(fused), _DL, _DNOPE, _DKVL)


def _draft_run(fused: bool, mx: bool = False) -> SimpleNamespace:
    """``draft_step`` from modeling.py over exact stand-ins: the GEMMs in order, what the q_a
    norm and the block RoPE read, the output. ``mx`` (``draft_mxfp8``): o_proj, gate_up and down
    are ``_Mx`` weights whose stand-in GEMM is the same exact product."""
    gemms, seen = [], {"qa": [], "kv": []}
    names = {}

    def rubin_linear(x, w, tf32):
        gemms.append(names[id(w)])
        return _gemm(x, w)

    def rubin_mxfp8_linear(x, w, s, pinned=False):
        # gate_up and down run pinned, o_proj (K = 12288) on the shared low-M / mid-M GEMM
        assert pinned is (names[id(w)] != "o"), (names[id(w)], pinned)
        assert s.dtype == torch.uint8
        gemms.append("mx:" + names[id(w)])
        return _gemm(x, w)

    def flashinfer_rmsnorm(x, w, eps):
        if x.shape[-1] == _DQL:
            seen["qa"].append(x.clone())
        return (x + w).clamp(-3, 3)

    def flashinfer_fused_add_rmsnorm(a, residual, w, eps):
        residual = residual + a
        return (residual + w).clamp(-3, 3), residual

    def dflash2_grouped_conv(x, coef, base, block, group):
        return (x + coef.sum((1, 2)).view(-1, 1) + base.sum(0)).clamp(-3, 3)

    def bmm_out(a, b, out):
        out.copy_(torch.bmm(a.double(), b.double()).float())
        return out

    def dflash2_mla_block_rope(q, query, kv, kv_norm, eps, freqs, positions, pool, pages, offsets):
        seen["kv"].append(kv.clone())
        query[..., _DKVL:] = (kv[:, None, _DKVL:] + q[..., _DNOPE:]).clamp(-3, 3)

    def rubin_dflash2_window_fp32(query, pool_rows, rows, scale, kv_lora, kv_scale):
        return query[..., :kv_lora].clamp(-3, 3)

    def flashinfer_silu_and_mul(a):
        return (a[:, : a.shape[1] // 2] * a[:, a.shape[1] // 2 :]).clamp(-3, 3)

    ns = _exec(
        _tree(),
        ("_Mx", "_mx_view", "_dense", "_DraftLayerW", "_DraftW", "_draft_views", "draft_step"),
        dict(
            torch=torch,
            Any=Any,
            NamedTuple=NamedTuple,
            Optional=None,
            nn=SimpleNamespace(ParameterDict=dict),
            rubin_linear=rubin_linear,
            rubin_mxfp8_linear=rubin_mxfp8_linear,
            flashinfer_rmsnorm=flashinfer_rmsnorm,
            flashinfer_fused_add_rmsnorm=flashinfer_fused_add_rmsnorm,
            dflash2_grouped_conv=dflash2_grouped_conv,
            bmm_out=bmm_out,
            dflash2_mla_block_rope=dflash2_mla_block_rope,
            rubin_dflash2_window_fp32=rubin_dflash2_window_fp32,
            flashinfer_silu_and_mul=flashinfer_silu_and_mul,
            dflash2_window_rows=lambda *a: (torch.zeros(_DROWS, 4, dtype=torch.int32), None),
            _DRAFT_WINDOW=4,
            _DraftStep=object,
            _Rt=object,
        ),
    )
    DW = ns["_draft_views"](_draft_weights(fused, mx), _DL, _DNOPE, _DKVL)
    for L in DW.layers:
        for f in ("attn_conv_proj", "q_a", "q_b", "kv_a", "o", "mlp_conv_proj", "gate_up", "down"):
            v = getattr(L, f)
            names[id(v.w if isinstance(v, ns["_Mx"]) else v)] = f
        if L.q_a_kv_a is not None:
            names[id(L.q_a_kv_a)] = "q_a_kv_a"
    st = SimpleNamespace(
        rows=_DROWS,
        block=_BLOCK,
        num_ctx=torch.tensor([3, 5]),
        page_tables=torch.zeros(2, 4, dtype=torch.long),
        page_size=4,
        positions=torch.arange(_DROWS),
        pools=[torch.zeros(4, 1, 1, 4, _DKVL + _DROPE) for _ in range(_DL)],
    )
    d = SimpleNamespace(
        eps=1e-5, taps=_TAPS, groups=_GROUPS, group=2, freqs=None, softmax_scale=1.0, kv_scale=1.0
    )
    gen = torch.Generator().manual_seed(9)
    h = ns["draft_step"](DW, _ints(gen, _DROWS, _DHID), st, SimpleNamespace(draft=d))
    return SimpleNamespace(gemms=gemms, seen=seen, out=h)


def test_the_drafter_projects_q_a_and_kv_a_in_one_gemm_under_the_lever():
    off, fused = _draft_run(False), _draft_run(True)
    layer = ["attn_conv_proj", "q_a", "q_b", "kv_a", "o", "mlp_conv_proj", "gate_up", "down"]
    assert off.gemms == layer * _DL, off.gemms
    one = ["attn_conv_proj", "q_a_kv_a", "q_b", "o", "mlp_conv_proj", "gate_up", "down"]
    assert fused.gemms == one * _DL, fused.gemms
    for key in ("qa", "kv"):
        assert len(off.seen[key]) == len(fused.seen[key]) == _DL
        for a, b in zip(off.seen[key], fused.seen[key]):
            assert torch.equal(a, b), key
    # the q_a norm and the block RoPE read row-strided views of the one product
    assert torch.equal(off.out, fused.out)

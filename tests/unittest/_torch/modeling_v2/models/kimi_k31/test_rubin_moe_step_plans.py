# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE blocks' step plans of tep8 and dep16 (CPU, from source).

tep8, decode steps (``latent_moe``): a step of up to ``_MOE_ULL_FRONT_MAX_M`` (16) rows runs the
sharded front (``rubin_ull_front_bf16``: this rank's tiles of latent_down, the routing and the
NVFP4 latent), FC12, the shared experts' partial pushed into the column-parallel tail's call,
then the tail (the latent all-reduce and norm, ``rubin_ull_cp_tail``). A step of up to
``_MOE_CP_TAIL_MAX_M`` (32) rows runs the replicated front (latent_down, the routing tail, the
quantization), FC12, the push and the tail. Both return ``_MoeTail``; a larger step returns
``_MoeOut`` (the two all-reduces and latent_up). On a CUDA graph's steps the shared experts run
on the aux stream between a fork and a join. The next mix finishes a ``_MoeTail``: ``_mix_in`` and
``_mix_out`` hand it to ``moe_norm`` / ``moe_add`` as they do a ``_MoeOut``, and those slots
finish the call in the mix kernel, or with the finish kernel and the add.

dep16, context steps (``_moe_dep_ctx``): when a rank holds no rows, every rank pads to the step's
largest count before the all-to-all. A rank with rows runs the router and latent_down GEMMs on its
own rows and pads their outputs with zero rows; a rank without rows runs them on zero rows. The
padded rows' outputs are dropped. The rows are routed by ``rubin_route_topk_wide`` under
``moe_route_topk_wide`` up to the entry's row bound, else by ``noaux_tc_op`` (``_moe_route_ctx``).
A context server's shared experts (``_moe_shared_ctx``) take
SiTU's MXFP8 form into the down GEMM (``shared_down_mx_handoff``): the gate_up GEMM,
``rubin_situ_and_mul_mxfp8`` and ``rubin_mxfp8_gemm``; a rank without rows keeps
``_moe_shared``. dep16 refuses at import a ``num_expert_group`` other than 1: its bucket routing
computes the top-k of one expert group.

dep16, the context experts (``_moe_cutedsl``): FC2 finalizes into the all-to-all's combine
payload, and only the rows it adds into, the received rows, are zeroed first
(``rubin_moe_zero_local_rows`` over the routing and this rank's window), on the memset stream
between the fork after the sort and the join before FC2; an eager step marks ``out`` and ``ids``
used there. The sort cuts ``moe_expert_tiling``'s row tile (256 on a context server), and FC1 and
FC2 pin that tiling's tactics.

The products' functions and constants run from their source; the entries they call, the streams
and the events are recording stand-ins.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TEP8 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__tep8" / "modeling.py"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
# Stand-in widths: hidden, latent, router, top-k, local experts (the plans do not depend on them).
_HIDDEN, _LATENT, _EXPERTS, _TOP_K, _LOCAL = 64, 32, 16, 4, 2


def _definitions(path: Path, names) -> list:
    """The module-level definitions and assignments of ``names`` in ``path``, in source order;
    each name must be defined exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(node)
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), (path.name, seen)
    return nodes


def _run(path: Path, nodes: list, ns: dict) -> dict:
    """Run ``nodes`` of ``path`` in ``ns``. ``compile`` inherits this file's future import, so
    annotations stay unevaluated, as in the product."""
    module = ast.fix_missing_locations(ast.Module(nodes, []))
    exec(compile(module, path.name, "exec"), ns)
    return ns


def _load(path: Path, names, ns: dict) -> dict:
    return _run(path, _definitions(path, names), ns)


class _Trace:
    """The calls a step makes, in order: ``(op, stream, detail)``."""

    def __init__(self):
        self.calls: list = []
        self.stream = "main"

    def __call__(self, op: str, detail: Any = None) -> None:
        self.calls.append((op, self.stream, detail))

    def plan(self) -> list:
        return [f"{op}@{stream}" for op, stream, _ in self.calls]

    def detail(self, op: str) -> Any:
        (detail,) = [d for o, _, d in self.calls if o == op]
        return detail


class _Stream(NamedTuple):
    name: str


class _OnStream:
    """``torch.cuda.stream(s)``: the calls made inside run on ``s``."""

    def __init__(self, trace: _Trace, stream: _Stream):
        self.trace, self.stream, self.outer = trace, stream, None

    def __enter__(self):
        self.outer, self.trace.stream = self.trace.stream, self.stream.name

    def __exit__(self, *exc):
        self.trace.stream = self.outer


class _Event:
    def __init__(self, trace: _Trace, name: str):
        self.trace, self.name = trace, name

    def record(self):
        self.trace(f"record:{self.name}")

    def wait(self):
        self.trace(f"wait:{self.name}")


class _Torch:
    """``torch`` as the products see it here: ``torch.cuda.stream`` is the recording one."""

    def __init__(self, trace: _Trace):
        self.cuda = SimpleNamespace(stream=lambda stream: _OnStream(trace, stream))

    def __getattr__(self, name: str):
        return getattr(torch, name)


class _Zeroed:
    """The int32 buffer of FC12's output and counters, zeroed beside the routing."""

    def __init__(self, trace: _Trace):
        self.trace = trace

    def zero_(self):
        self.trace("zero")
        return self


# --- tep8: the decode step's MoE and the next mix ---

_TEP8_NAMES = (
    "_K31",
    "_CONSTANTS",
    "_ATTN_RES_OP_MAX_TOKENS",
    "_FUSED_MAX_ROWS",
    "_PERSISTENT_MAX_CANDIDATES",
    "_ULL_TAIL_MAX_ROWS",
    "_ULL_FRONT_MAX_ROWS",
    "_MoeOut",
    "_MoeTail",
    "_mix_slots",
    "_MOE_TOP_K",
    "_MOE_GROUPS",
    "_MOE_SCALE",
    "_MOE_EPS",
    "_MOE_SF_VEC",
    "_MOE_TILE",
    "_MOE_AR_NONE",
    "_MOE_GEN",
    "_MOE_ROUTE_TAIL_MAX_M",
    "_MOE_ROUTE_APPROX",
    "_MOE_ROUTE_BANDS",
    "_MOE_ROUTER_EXACT_MAX_M",
    "_MOE_ROUTER_FIRST_MAX_M",
    "_MOE_OWN_WS_MAX_M",
    "_MOE_OWN_WS_MAX_BYTES",
    "_MOE_FC12_PREZERO",
    "_MOE_CP_TAIL_MAX_M",
    "_MOE_ULL_FRONT_MAX_M",
    "_moe_cublas_linear",
    "_moe_latent_gemm",
    "_moe_router",
    "_moe_router_gen",
    "_moe_route",
    "_moe_quantize",
    "_moe_one_shot",
    "_moe_tp_gen",
    "_moe_tp_gen_streams",
    "_moe_cp_tail",
    "_moe_tp_gen_tail",
    "_moe_tp_gen_streams_tail",
    "_moe_tp_gen_ull",
    "latent_moe",
    "_mix_in",
    "_mix_out",
)


def _tables(tag: str) -> tuple:
    """moe_sort's six tables, as stand-in objects that name their producer."""
    return tuple(SimpleNamespace(table=i, by=tag) for i in range(6))


def _tep8(trace: _Trace) -> dict:
    """tep8's MoE decode functions in a namespace of recording stand-ins."""

    def cublas_mm(a, b):
        trace("latent_down" if b.shape[1] == _LATENT else "latent_up", a.shape[0])
        return torch.zeros(a.shape[0], b.shape[1], dtype=torch.bfloat16)

    def dsv3_router_gemm_op(x, router_t, bias, dtype):
        trace("router_gemm", x.shape[0])
        return torch.zeros(x.shape[0], _EXPERTS)

    def rubin_router_logits(x, router):
        trace("router_exact", x.shape[0])
        return torch.zeros(x.shape[0], _EXPERTS)

    def _routed(rows: int) -> tuple:
        return torch.zeros(rows, _TOP_K, dtype=torch.int32), torch.zeros(rows, _TOP_K)

    def rubin_route_tail(logits, bias, scale, offset, num_local, tile, *, route_approx, **kw):
        tables = _tables("route_tail")
        trace("route_tail", tables)
        return (*_routed(logits.shape[0]), *tables)

    def noaux_tc_op(logits, bias, groups, topk_groups, top_k, scale):
        trace("noaux_tc", logits.shape[0])
        ids, weights = _routed(logits.shape[0])
        return weights, ids

    def fp4_quantize(x, scale, sf_vec, swizzled, linear):
        trace("fp4_quantize", x.shape[0])
        rows = x.shape[0]
        return (
            torch.zeros(rows, _LATENT // 2, dtype=torch.uint8),
            torch.zeros(rows * _LATENT // sf_vec, dtype=torch.uint8),
        )

    def rubin_ull_front_bf16(x, logits, latent_down, views, bias, scale, in_scale, *rest, zero):
        offset, num_local, ws, tactic = rest
        tables = _tables("front")
        trace("front", dict(down=latent_down, views=views, ws=ws, tactic=tactic, zero=zero))
        rows = x.shape[0]
        x4 = torch.zeros(rows, _LATENT // 2, dtype=torch.uint8)
        sf = torch.zeros(rows, _LATENT // 16, dtype=torch.uint8)
        return (*_routed(rows), *tables), x4, sf

    zeroed = []

    def _moe_fc12_zeroed(rows, num_local, hidden, device):
        zeroed.append((_Zeroed(trace), torch.zeros(rows, hidden), torch.zeros(64)))
        return zeroed[-1]

    def _moe_fc12(L, x4, sf, ids, weights, tables, zeroed_bufs):
        trace("fc12", dict(tables=tables, zeroed=zeroed_bufs))
        return torch.zeros(ids.shape[0], _LATENT, dtype=torch.bfloat16)

    def _moe_shared(L, x):
        out = torch.zeros(x.shape[0], _HIDDEN, dtype=torch.bfloat16)
        trace("shared", out)
        return out

    def _moe_tep_ctx(L, x, st, rt):
        trace("context_path", x.shape[0])
        return "context"

    def allreduce_module(x, all_reduce, fusion_op, norm_weight, eps):
        trace(f"allreduce{fusion_op}", x.shape[0])
        return torch.zeros_like(x)

    def rubin_latent_ar_norm_row(x, weight, eps, out, **kw):
        trace("latent_ar_norm", x.shape[0])

    def rubin_lamport_push(ws, data, one_shot, use_pdl):
        trace("lamport_push", dict(ws=ws, data=data, flags=(one_shot, use_pdl)))

    def rubin_ull_cp_tail(latent, latent_up, ws, tactic, *, c0):
        trace("cp_tail", dict(rows=latent.shape[0], up=latent_up, ws=ws, tactic=tactic, c0=c0))

    ns = dict(
        torch=_Torch(trace),
        Any=Any,
        NamedTuple=NamedTuple,
        MappingProxyType=MappingProxyType,
        cublas_mm=cublas_mm,
        dsv3_router_gemm_op=dsv3_router_gemm_op,
        rubin_router_logits=rubin_router_logits,
        rubin_route_tail=rubin_route_tail,
        noaux_tc_op=noaux_tc_op,
        fp4_quantize=fp4_quantize,
        rubin_ull_front_bf16=rubin_ull_front_bf16,
        _moe_fc12_zeroed=_moe_fc12_zeroed,
        _moe_fc12=_moe_fc12,
        _moe_shared=_moe_shared,
        _moe_tep_ctx=_moe_tep_ctx,
        allreduce_module=allreduce_module,
        rubin_latent_ar_norm_row=rubin_latent_ar_norm_row,
        rubin_lamport_push=rubin_lamport_push,
        rubin_ull_cp_tail=rubin_ull_cp_tail,
        zeroed=zeroed,
    )
    return _load(_TEP8, _TEP8_NAMES, ns)


def _tep8_layer() -> SimpleNamespace:
    """An MLA layer's MoE view (the router GEMM on every row count)."""
    return SimpleNamespace(
        after_kda=False,
        router=torch.zeros(_EXPERTS, _HIDDEN),
        router_bias=torch.zeros(_EXPERTS),
        latent_down=torch.arange(_LATENT * _HIDDEN, dtype=torch.float32).view(_LATENT, _HIDDEN),
        latent_up=torch.arange(_HIDDEN * _LATENT, dtype=torch.float32).view(_HIDDEN, _LATENT),
        latent_norm=torch.ones(_LATENT),
        fc31_in_scale=torch.ones(1),
        expert_offset=0,
        fc1_w=torch.zeros(_LOCAL, 1),
        fc2_w=torch.zeros(_LOCAL, _LATENT),
    )


def _tep8_rt(trace: _Trace) -> SimpleNamespace:
    events = {
        name: _Event(trace, name)
        for name in ("shared_fork", "shared_join", "latent_fork", "latent_ready", "prezero_ready")
    }
    latent_ws = dict(
        world_size=8,
        max_tokens=128,
        buffer_bytes=1 << 30,
        buffer=torch.zeros(1),
        mc=torch.zeros(1),
        flags=torch.zeros(1),
        rank=0,
    )
    return SimpleNamespace(
        aux_stream=_Stream("aux"),
        moe_comm=SimpleNamespace(
            all_reduce="tp_all_reduce",
            latent_ws=latent_ws,
            latent_stream=_Stream("latent"),
            prezero_stream=_Stream("prezero"),
            **events,
        ),
        ull_tail=SimpleNamespace(
            ws="tail_ws",
            c0=16,
            n=16,
            tactics=tuple(f"tail_tactic_{m}" for m in range(33)),
            cpp_one_shot=True,
            chunk=8,
        ),
        ull_front=SimpleNamespace(
            ws="front_ws",
            logits_views=("half_0", "half_1"),
            r0=8,
            n=8,
            tactics=tuple(f"front_tactic_{m}" for m in range(17)),
        ),
    )


# The decode plans, in launch order; ``op@stream``.
_FRONT_EAGER = [
    "router_gemm@main",
    "front@main",
    "fc12@main",
    "shared@main",
    "lamport_push@main",
    "latent_ar_norm@main",
    "cp_tail@main",
]
_FRONT_GRAPH = [
    "router_gemm@main",
    "record:shared_fork@main",
    "front@main",
    "fc12@main",
    "wait:shared_fork@aux",
    "shared@aux",
    "lamport_push@aux",
    "record:shared_join@aux",
    "latent_ar_norm@main",
    "cp_tail@main",
    "wait:shared_join@main",
]
_TAIL_EAGER = [
    "router_gemm@main",
    "latent_down@main",
    "route_tail@main",
    "fp4_quantize@main",
    "fc12@main",
    "shared@main",
    "lamport_push@main",
    "latent_ar_norm@main",
    "cp_tail@main",
]
# The replicated front of a CUDA graph's step: FC12's buffer zeroed on its stream, the router
# first, latent_down on its stream.
_GRAPH_FRONT = [
    "record:latent_fork@main",
    "wait:latent_fork@prezero",
    "zero@prezero",
    "record:prezero_ready@prezero",
    "router_gemm@main",
    "wait:latent_fork@latent",
    "latent_down@latent",
    "record:latent_ready@latent",
    "record:shared_fork@main",
    "route_tail@main",
    "wait:latent_ready@main",
    "fp4_quantize@main",
    "wait:prezero_ready@main",
    "fc12@main",
    "wait:shared_fork@aux",
    "shared@aux",
]
_TAIL_GRAPH = _GRAPH_FRONT + [
    "lamport_push@aux",
    "record:shared_join@aux",
    "latent_ar_norm@main",
    "cp_tail@main",
    "wait:shared_join@main",
]
_REPLICATED_EAGER = [
    "router_gemm@main",
    "latent_down@main",
    "route_tail@main",
    "fp4_quantize@main",
    "fc12@main",
    "shared@main",
    "allreduce0@main",
    "allreduce9@main",
    "latent_up@main",
]
_REPLICATED_GRAPH = _GRAPH_FRONT + [
    "allreduce0@aux",
    "record:shared_join@aux",
    "latent_ar_norm@main",
    "latent_up@main",
    "wait:shared_join@main",
]
_PLANS = {
    (16, False): _FRONT_EAGER,
    (16, True): _FRONT_GRAPH,
    (17, False): _TAIL_EAGER,
    (17, True): _TAIL_GRAPH,
    (32, False): _TAIL_EAGER,
    (32, True): _TAIL_GRAPH,
    (33, False): _REPLICATED_EAGER,
    (33, True): _REPLICATED_GRAPH,
}


def _decode(rows: int, graph: bool):
    trace = _Trace()
    ns = _tep8(trace)
    layer, rt = _tep8_layer(), _tep8_rt(trace)
    st = SimpleNamespace(num_contexts=0, cuda_graph=graph)
    out = ns["latent_moe"](layer, 3, torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16), st, rt)
    return ns, trace, layer, rt, out


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("rows", [16, 17, 32, 33])
def test_a_tep8_decode_step_takes_its_row_counts_plan(rows, graph):
    ns, trace, layer, rt, out = _decode(rows, graph)
    assert trace.plan() == _PLANS[(rows, graph)]
    if rows > 32:
        assert isinstance(out, ns["_MoeOut"]), type(out)
        return
    # The tail's call, which the next mix finishes.
    assert isinstance(out, ns["_MoeTail"]), type(out)
    assert out == (rt.ull_tail.ws, rt.ull_tail.cpp_one_shot, rt.ull_tail.chunk)
    # The shared experts' partial goes into the tail's call, one-shot with PDL.
    push = trace.detail("lamport_push")
    assert push["ws"] is rt.ull_tail.ws and push["flags"] == (True, True)
    assert push["data"] is trace.detail("shared")
    # The tail: this rank's rows of latent_up, the push GEMM's tactic for the row count.
    tail = trace.detail("cp_tail")
    assert tail["rows"] == rows and tail["ws"] is rt.ull_tail.ws and tail["c0"] == 16
    assert tail["tactic"] == f"tail_tactic_{rows}"
    assert torch.equal(tail["up"], layer.latent_up[16:32])
    fc12 = trace.detail("fc12")
    if rows <= 16:
        # The front: this rank's tiles of latent_down; it routes, sorts and zeroes FC12's buffer.
        front = trace.detail("front")
        assert front["ws"] is rt.ull_front.ws and front["views"] is rt.ull_front.logits_views
        assert front["tactic"] == f"front_tactic_{rows}"
        assert torch.equal(front["down"], layer.latent_down[8:16])
        assert front["zero"] is ns["zeroed"][-1][0]
        assert [t.by for t in fc12["tables"]] == ["front"] * 6
        assert fc12["zeroed"] is ns["zeroed"][-1]
    else:
        assert [t.by for t in fc12["tables"]] == ["route_tail"] * 6
        assert fc12["zeroed"] is (ns["zeroed"][-1] if graph else None)


@pytest.mark.parametrize("rows", [1, 16, 33])
def test_a_step_with_context_rows_takes_the_context_path(rows):
    trace = _Trace()
    ns = _tep8(trace)
    st = SimpleNamespace(num_contexts=1, cuda_graph=False)
    x = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16)
    assert ns["latent_moe"](_tep8_layer(), 3, x, st, _tep8_rt(trace)) == "context"
    assert trace.plan() == ["context_path@main"]


def _slot_rt(calls: list) -> SimpleNamespace:
    def slot(name, result):
        def call(*args):
            calls.append((name, args))
            return result

        return call

    return SimpleNamespace(
        attn_res_norm=slot("attn_res_norm", "normed"),
        add_norm=slot("add_norm", ("added", "normed")),
        moe_norm=slot("moe_norm", ("moe_added", "normed")),
        moe_add=slot("moe_add", torch.full((2, _HIDDEN), 5.0)),
        attn_res_mix=slot("attn_res_mix", "mix"),
        rms_norm=slot("rms_norm", "normed"),
    )


def _pending(ns: dict, kind: str):
    if kind == "tail":
        return ns["_MoeTail"]("tail_ws", True, 8)
    if kind == "moe_out":
        return ns["_MoeOut"](torch.ones(2, _HIDDEN), torch.ones(2, _HIDDEN))
    return torch.full((2, _HIDDEN), 2.0)  # layer 0's dense output


@pytest.mark.parametrize("tapped", [False, True], ids=["untapped", "tapped"])
@pytest.mark.parametrize("kind", ["tail", "moe_out", "dense"])
def test_the_next_mix_takes_a_tail_as_a_moe_output(kind, tapped):
    """``_mix_in`` / ``_mix_out`` hand a ``_MoeTail`` to the MoE slots, as a ``_MoeOut``."""
    ns = _tep8(_Trace())
    pending, prefix, bank = _pending(ns, kind), torch.ones(2, _HIDDEN), "bank"
    mw = SimpleNamespace(out_norm="out_norm")
    taps = []
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: taps.append(a)), 5)
    moe = kind != "dense"

    calls = []
    ns["_mix_in"](_slot_rt(calls), prefix, pending, bank, mw, tap if tapped else None)
    names = [name for name, _ in calls]
    if not tapped:
        assert names == (["moe_norm"] if moe else ["add_norm"])
        assert calls[0][1][1] is pending
    else:
        assert names == (["moe_add"] if moe else []) + ["attn_res_mix", "rms_norm"]
        assert taps == [(5, "mix", None)]
        if moe:
            assert calls[0][1][1] is pending

    calls = []
    ns["_mix_out"](_slot_rt(calls), prefix, pending, bank, SimpleNamespace(out=mw), None)
    assert [name for name, _ in calls] == (["moe_add"] if moe else []) + ["attn_res_norm"]
    if moe:
        assert calls[0][1][1] is pending
    else:
        assert torch.equal(calls[0][1][0], prefix + pending)


@pytest.mark.parametrize("sandwich", [True, False])
def test_the_moe_slots_finish_a_tails_call(sandwich):
    """``moe_norm`` on a ``_MoeTail``: with snapshots in the bank (up to the PDL rows) the mix
    kernel finishes the call (one-shot, its all-gather rows, the round-to-nearest add, the early
    poll as ``moe_tail_early_poll`` says); else, as ``moe_add``, the finish kernel then the add."""
    ns = _tep8(_Trace())
    calls = []
    finished = torch.full((8, _HIDDEN), 0.5, dtype=torch.bfloat16)

    def rubin_lamport_finish(ws, out, *flags):
        calls.append(("finish", ws, tuple(out.shape), out.dtype, flags))
        return out.copy_(finished)

    def rubin_attn_res_lamport_add_rmsnorm(prefix, uc, flags_ptr, bank, proj, res, out, *a, **kw):
        calls.append(("mix", prefix, uc, flags_ptr, bank, (proj, res, out), a, kw))
        return "updated", "normed"

    def flashinfer_rmsnorm(x, weight, eps):
        calls.append(("rmsnorm", x))
        return "normed"

    ns.update(
        rubin_lamport_finish=rubin_lamport_finish,
        rubin_attn_res_lamport_add_rmsnorm=rubin_attn_res_lamport_add_rmsnorm,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
    )
    generation = dict(ns["_CONSTANTS"]["generation"], moe_tail_early_poll=sandwich)
    assert generation["attn_res_pdl"] == 2 and generation["attn_res_early_trigger"]
    slots = ns["_mix_slots"](generation, 1e-5)
    flags = torch.zeros(4, dtype=torch.int32)
    ws = SimpleNamespace(uc_local=1234, flags=flags, mc_base=5678, rank=3, world=8)
    # Stand-in call fields: four ranks per partial sum, the two-shot C++ sums.
    tail = ns["_MoeTail"](ws, False, 4)
    mw = SimpleNamespace(proj="proj", res_norm="res_norm", out_norm="out_norm")
    prefix = torch.ones(8, _HIDDEN, dtype=torch.bfloat16)
    bank = torch.zeros(2, 8, _HIDDEN, dtype=torch.bfloat16)

    assert slots["moe_norm"](prefix, tail, bank, mw) == ("updated", "normed")
    ((op, got_prefix, uc, flags_ptr, got_bank, weights, args, kw),) = calls
    assert op == "mix" and got_prefix is prefix and got_bank is bank
    assert (uc, flags_ptr, weights) == (1234, flags.data_ptr(), ("proj", "res_norm", "out_norm"))
    # eps, eps, world, ranks per partial sum, two-shot finish, multicast base, rank,
    # round-to-nearest add, early poll, entry trigger, the all-gather rows.
    assert args == (1e-5, 1e-5, 8, 4, False, 5678, 3, True, sandwich, False, True)
    assert kw == {"early_trigger": True}

    want = (prefix.float() + finished.float()).to(torch.bfloat16)
    # one-shot, PDL, no wait first, the call's C++ sums, the all-gather rows.
    finish = ("finish", ws, (8, _HIDDEN), torch.bfloat16, (True, True, False, False, True))
    calls.clear()
    assert torch.equal(slots["moe_add"](prefix, tail), want)
    assert calls == [finish]
    calls.clear()
    updated, normed = slots["moe_norm"](prefix, tail, bank[:0], mw)
    assert torch.equal(updated, want) and normed == "normed"
    assert [c[0] for c in calls] == ["finish", "rmsnorm"] and calls[0] == finish


# --- dep16: the context step's MoE, and the expert-group refusal ---

_DEP16_NAMES = (
    "_K31",
    "_CONSTANTS",
    "_MOE_A2A_FORCE_CFT",
    "_MoeOut",
    "_MOE_TOP_K",
    "_MOE_GROUPS",
    "_MOE_SCALE",
    "_MOE_EPS",
    "_MOE_SF_VEC",
    "_MOE_A2A_INVALID_ID",
    "_MOE_SITU",
    "_MOE_CTX_SHARED_MX",
    "_MOE_CTX_ROUTE_WIDE",
    "_moe_cublas_linear",
    "_moe_latent_gemm",
    "_moe_router",
    "_moe_route",
    "_moe_route_ctx",
    "_moe_quantize",
    "_moe_shared_ctx",
    "_moe_routed",
    "_moe_dep_ctx",
)
# rubin_route_topk_wide's row bound (MAX_M, imported by dep16 as _MOE_CTX_ROUTE_MAX_M).
_ROUTE_WIDE_MAX_M = ast.literal_eval(
    next(
        node.value
        for node in ast.parse((_FAMILY / "ops/moe/rubin_route_topk_wide.py").read_text()).body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "MAX_M"
    )
)


def _dep16(trace: _Trace, seen: dict) -> dict:
    """dep16's context-step MoE in a namespace of recording stand-ins (GEMMs as matmuls)."""

    def dsv3_router_gemm_op(x, router_t, bias, dtype):
        trace("router", x.shape[0])
        return x.float() @ router_t.float()

    def cublas_mm(a, b):
        trace("latent_down" if b.shape[1] == _LATENT else "latent_up", a.shape[0])
        return (a.float() @ b.float()).to(torch.bfloat16)

    def noaux_tc_op(logits, bias, groups, topk_groups, top_k, scale):
        trace("noaux_tc", logits.shape[0])
        seen["logits"] = logits
        rows = logits.shape[0]
        return torch.zeros(rows, _TOP_K), torch.zeros(rows, _TOP_K, dtype=torch.int32)

    def rubin_route_topk_wide(logits, bias, scale):
        trace("route_wide", logits.shape[0])
        seen["logits"] = logits
        rows = logits.shape[0]
        return torch.zeros(rows, _TOP_K, dtype=torch.int32), torch.zeros(rows, _TOP_K)

    def fp4_quantize(x, scale, sf_vec, swizzled, linear):
        trace("fp4_quantize", x.shape[0])
        seen["latent_in"] = x
        rows = x.shape[0]
        return (
            torch.zeros(rows, _LATENT // 2, dtype=torch.uint8),
            torch.zeros(rows * _LATENT // sf_vec, dtype=torch.uint8),
        )

    def a2a_use_cft(can_use_cft, force_cft, max_batch, runtime_max):
        return False

    def rubin_moe_a2a_dispatch(x4, sf, ids, weights, a2a, runtime_max, cft, *rest):
        trace("a2a_dispatch", (x4.shape[0], runtime_max))
        out = torch.zeros(runtime_max, _LATENT, dtype=torch.bfloat16)
        return x4, sf, ids, weights, "a2a_state", out

    def _moe_cutedsl(L, x4, sf, ids, weights, out, st, rt):
        trace("experts", x4.shape[0])
        return out

    def rubin_moe_a2a_combine(out, rows, a2a, runtime_max, state, in_workspace, low, cft):
        trace("a2a_combine", (rows, runtime_max))
        seen["combine"] = (in_workspace, low, cft)
        # Row r of the combined output holds r, so the rows kept are visible.
        return torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, _LATENT).bfloat16()

    def flashinfer_rmsnorm(x, weight, eps):
        trace("rmsnorm", x.shape[0])
        seen["kept"] = x[:, 0].tolist()
        return x

    def _moe_shared(L, x):
        trace("shared", x.shape[0])
        return torch.zeros(x.shape[0], _HIDDEN, dtype=torch.bfloat16)

    def rubin_mxfp8_linear(x, w, s):
        trace("shared_gate_up", x.shape[0])
        seen["gate_up"] = (w, s)
        return torch.zeros(x.shape[0], 2 * _LATENT, dtype=torch.bfloat16)

    def rubin_situ_and_mul_mxfp8(x, beta, linear_beta):
        trace("situ_mxfp8", x.shape[0])
        seen["situ"] = (beta, linear_beta)
        return torch.zeros(x.shape[0], _LATENT, dtype=torch.uint8), "scales"

    def rubin_mxfp8_gemm(codes, w, scales, s):
        trace("shared_down_mxfp8", codes.shape[0])
        seen["down"] = (w, scales, s)
        return torch.zeros(codes.shape[0], _HIDDEN, dtype=torch.bfloat16)

    ns = dict(
        torch=torch,
        Any=Any,
        NamedTuple=NamedTuple,
        MappingProxyType=MappingProxyType,
        CFT_MAX_BATCH=64,
        _MOE_CTX_ROUTE_MAX_M=_ROUTE_WIDE_MAX_M,
        dsv3_router_gemm_op=dsv3_router_gemm_op,
        cublas_mm=cublas_mm,
        noaux_tc_op=noaux_tc_op,
        rubin_route_topk_wide=rubin_route_topk_wide,
        fp4_quantize=fp4_quantize,
        a2a_use_cft=a2a_use_cft,
        rubin_moe_a2a_dispatch=rubin_moe_a2a_dispatch,
        _moe_cutedsl=_moe_cutedsl,
        rubin_moe_a2a_combine=rubin_moe_a2a_combine,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
        _moe_shared=_moe_shared,
        rubin_mxfp8_linear=rubin_mxfp8_linear,
        rubin_situ_and_mul_mxfp8=rubin_situ_and_mul_mxfp8,
        rubin_mxfp8_gemm=rubin_mxfp8_gemm,
    )
    return _load(_DEP16, _DEP16_NAMES, ns)


class _NoPlan:
    """The replica plan of a layer without a load balancer: its waits are free."""

    def wait_for_routes(self) -> None:
        return None

    def wait_for_helper_weights(self) -> None:
        return None


# Rows per rank of a context step with ranks without rows, and of one without such ranks.
_WITH_EMPTY = [3, 0, 7, 1, 7, 2, 0, 5, 4, 6, 1, 2, 3, 7, 5, 4]
_NO_EMPTY = [3, 1, 7, 1, 7, 2, 2, 5, 4, 6, 1, 2, 3, 7, 5, 4]


@pytest.mark.parametrize(
    "counts, rank, gemm_rows, a2a_rows",
    [
        (_WITH_EMPTY, 0, 3, 7),  # rows: the GEMMs on them, the outputs padded to the largest
        (_WITH_EMPTY, 1, 7, 7),  # no rows: the GEMMs on the padded zero rows
        (_WITH_EMPTY, 2, 7, 7),  # the largest count: nothing to pad
        (_NO_EMPTY, 0, 3, 3),  # every rank has rows: no padding
    ],
    ids=["rows", "no-rows", "largest", "no-empty-rank"],
)
@pytest.mark.parametrize("low", [True, False], ids=["fp8-combine", "bf16-combine"])
def test_a_dep16_context_step_pads_after_the_gemms(counts, rank, gemm_rows, a2a_rows, low):
    trace, seen = _Trace(), {}
    ns = _dep16(trace, seen)
    rows, largest = counts[rank], max(counts)
    layer = SimpleNamespace(
        router=torch.ones(_EXPERTS, _HIDDEN),
        router_bias=None,
        latent_down=torch.ones(_LATENT, _HIDDEN),
        latent_up=torch.ones(_HIDDEN, _LATENT),
        latent_norm=torch.ones(_LATENT),
        fc31_in_scale=None,
        shared_gu_w="gu_w",
        shared_gu_s="gu_s",
        shared_dn_w="dn_w",
        shared_dn_s="dn_s",
        plan=lambda ids: (ids, _NoPlan()),
        record=lambda ids, slots: None,
        finish=lambda plan: None,
    )
    st = SimpleNamespace(all_rank_num_tokens=counts)
    rt = SimpleNamespace(a2a=SimpleNamespace(can_use_cft=True, use_low_precision_combine=low))
    out = ns["_moe_dep_ctx"](layer, torch.ones(rows, _HIDDEN, dtype=torch.bfloat16), st, rt)
    assert [(op, d) for op, _, d in trace.calls] == [
        ("router", gemm_rows),
        ("latent_down", gemm_rows),
        ("route_wide", a2a_rows),
        ("fp4_quantize", a2a_rows),
        ("a2a_dispatch", (a2a_rows, largest)),
        ("experts", a2a_rows),
        ("a2a_combine", (a2a_rows, largest)),
        ("rmsnorm", rows),
        ("latent_up", rows),
        *_shared_plan(rows),
    ]
    # This rank's rows carry the GEMMs' outputs, the padded rows zeros; its own rows are kept.
    for name in ("logits", "latent_in"):
        assert bool((seen[name][:rows] != 0).all()) and bool((seen[name][rows:] == 0).all()), name
    assert seen["kept"] == list(range(rows))
    # The combine reads the expert output in the workspace, on the wire the workspace was built for.
    assert seen["combine"] == (True, low, False)
    assert isinstance(out, ns["_MoeOut"]) and tuple(out.routed.shape) == (rows, _HIDDEN)
    assert tuple(out.shared.shape) == (rows, _HIDDEN)
    if rows:
        assert seen["gate_up"] == ("gu_w", "gu_s") and seen["situ"] == (4.0, 25.0)
        assert seen["down"] == ("dn_w", "scales", "dn_s"), "SiTU's codes and scales into down"


@pytest.mark.parametrize("target", ["dep16", "cep"])
def test_a_context_all_to_all_combines_on_the_configured_wire(target):
    """The context targets build the all-to-all workspace for
    ``moe_config.use_low_precision_moe_combine`` and combine on the workspace's wire: no module
    constant picks FP8 or BF16."""
    path = _FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{target}" / "modeling.py"
    tree = ast.parse(path.read_text())
    assert "_MOE_A2A_LOW_PRECISION" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def calls(fn: str, callee: str) -> list:
        (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn]
        return [
            n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == callee
        ]

    (build,) = calls("_build_runtime", "rubin_moe_a2a_workspace")
    wire = {k.arg: ast.unparse(k.value) for k in build.keywords}["use_low_precision_combine"]
    assert wire == "bool(cfg.use_low_precision_moe_combine)"
    (combine,) = calls("_moe_routed", "rubin_moe_a2a_combine")
    assert ast.unparse(combine.args[6]) == "rt.a2a.use_low_precision_combine"


def _shared_plan(rows: int) -> list:
    """A context step's shared experts: SiTU's MXFP8 form into the down GEMM, or ``_moe_shared``
    on a rank without rows."""
    if not rows:
        return [("shared", 0)]
    return [("shared_gate_up", rows), ("situ_mxfp8", rows), ("shared_down_mxfp8", rows)]


def test_the_dep16_shared_down_hand_off_follows_its_constant():
    trace, seen = _Trace(), {}
    ns = _dep16(trace, seen)
    assert ns["_MOE_CTX_SHARED_MX"] is ns["_CONSTANTS"]["context"]["shared_down_mx_handoff"]
    assert ns["_MOE_CTX_SHARED_MX"] is True
    x = torch.ones(5, _HIDDEN, dtype=torch.bfloat16)
    layer = SimpleNamespace(shared_gu_w=None, shared_gu_s=None, shared_dn_w=None, shared_dn_s=None)
    ns["_moe_shared_ctx"](layer, x)
    ns["_MOE_CTX_SHARED_MX"] = False
    ns["_moe_shared_ctx"](layer, x)
    assert [(op, d) for op, _, d in trace.calls] == [*_shared_plan(5), ("shared", 5)]


@pytest.mark.parametrize(
    "rows, on, route",
    [
        (16384, True, "route_wide"),  # a DEP16 / DEP12 / DEP8 context rank's chunk
        (1, True, "route_wide"),
        (_ROUTE_WIDE_MAX_M, True, "route_wide"),
        (_ROUTE_WIDE_MAX_M + 1, True, "noaux_tc"),  # past the entry's bound
        (0, True, "noaux_tc"),
        (16384, False, "noaux_tc"),  # the constant off: today's op
    ],
)
def test_the_dep16_context_route_follows_its_constant_and_bound(rows, on, route):
    """A context step routes by ``rubin_route_topk_wide`` under ``moe_route_topk_wide`` within the
    entry's rows, else by ``noaux_tc_op`` (``_moe_route``); either returns ``(ids, weights)``."""
    trace, seen = _Trace(), {}
    ns = _dep16(trace, seen)
    assert ns["_MOE_CTX_ROUTE_WIDE"] is ns["_CONSTANTS"]["context"]["moe_route_topk_wide"]
    assert ns["_MOE_CTX_ROUTE_WIDE"] is True
    assert "moe_route_topk_wide" not in ns["_CONSTANTS"]["generation"]
    ns["_MOE_CTX_ROUTE_WIDE"] = on
    logits = torch.zeros(rows, 1).expand(rows, _EXPERTS)
    ids, weights = ns["_moe_route_ctx"](SimpleNamespace(router_bias=None), logits)
    assert [(op, d) for op, _, d in trace.calls] == [(route, rows)]
    assert seen["logits"] is logits
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert tuple(ids.shape) == tuple(weights.shape) == (rows, _TOP_K)


class _Used:
    """A tensor the experts hand to the memset stream: ``record_stream`` is recorded."""

    def __init__(self, trace: _Trace, name: str):
        self.trace, self.name = trace, name

    def record_stream(self, stream: _Stream) -> None:
        self.trace(f"record_stream:{self.name}", stream.name)


# Stand-ins of the FC1 / FC2 tactics a context server pins at tile 256.
_FC1_T256, _FC2_T256 = "fc1 tile-256 tactic", "fc2 tile-256 tactic"


def _dep16_experts(trace: _Trace) -> dict:
    """dep16's context experts (``_moe_cutedsl``) in a namespace of recording stand-ins."""

    def rubin_moe_sort(ids, weights, num_experts, top_k, offset, local, tile):
        trace("sort", (offset, local, tile))
        return "t2e", "t2l", "e2p", "p2e", "padded", "n"

    def rubin_moe_zero_local_rows(out, ids, offset, local):
        trace("zero_local_rows", (out.name, ids.name, offset, local))

    def rubin_cutedsl_fc1(
        x4,
        w,
        sf,
        s,
        alpha,
        t2e,
        t2l,
        p2e,
        n,
        in_scale,
        experts,
        top_k,
        local,
        offset,
        tile,
        *rest,
        precomputed_tactic,
    ):
        trace("fc1", (experts, top_k, local, offset, tile, precomputed_tactic))
        return "h", "h_sf"

    def rubin_cutedsl_fc2_finalize(
        h,
        w,
        h_sf,
        s,
        alpha,
        out,
        t2e,
        t2l,
        p2e,
        n,
        weights,
        experts,
        top_k,
        local,
        offset,
        tile,
        *,
        precomputed_tactic,
    ):
        trace("fc2_finalize", (out.name, experts, top_k, local, offset, tile, precomputed_tactic))

    ns = dict(
        torch=_Torch(trace),
        MappingProxyType=MappingProxyType,
        rubin_moe_sort=rubin_moe_sort,
        rubin_moe_zero_local_rows=rubin_moe_zero_local_rows,
        rubin_cutedsl_fc1=rubin_cutedsl_fc1,
        rubin_cutedsl_fc2_finalize=rubin_cutedsl_fc2_finalize,
        FC1_TILE256_TACTIC=_FC1_T256,
        FC2_TILE256_TACTIC=_FC2_T256,
    )
    names = (
        "_K31",
        "_CONSTANTS",
        "_MOE_TILINGS",
        "_MOE_TOP_K",
        "_MOE_SITU",
        "_MOE_TILE",
        "_MOE_ACT_SITU",
        "_moe_sort",
        "_moe_cutedsl",
    )
    return _load(_DEP16, names, ns)


# (first expert, count) of an EP rank's window: dep16 rank 1; dep12 ranks 0, 8 and 11 (75 experts
# from 0, 74 from 600 and the last 74, to expert 895).
_WINDOWS = {
    "dep16-r1": (56, 56),
    "dep12-r0": (0, 75),
    "dep12-r8": (600, 74),
    "dep12-r11": (822, 74),
}


@pytest.mark.parametrize("window", list(_WINDOWS.values()), ids=list(_WINDOWS))
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_the_dep16_context_experts_zero_only_the_received_rows(graph, window):
    """The sort, the zeroing, FC1 and FC2 all take the window of the layer's rows: its first
    expert and its count (the rows' expert count), over all 896 experts and top 16."""
    offset, local = window
    trace = _Trace()
    ns = _dep16_experts(trace)
    comm = SimpleNamespace(
        memset_stream=_Stream("memset"),
        latent_fork=_Event(trace, "fork"),
        memset_ready=_Event(trace, "ready"),
    )
    layer = SimpleNamespace(
        fc1_w=torch.zeros(local, 1),
        fc1_s=None,
        fc31_alpha=None,
        fc2_in_scale=None,
        fc2_w=None,
        fc2_s=None,
        fc2_alpha=None,
        expert_offset=offset,
        num_experts=896,
    )
    out, ids = _Used(trace, "out"), _Used(trace, "ids")
    st = SimpleNamespace(cuda_graph=graph)
    got = ns["_moe_cutedsl"](
        layer, "x4", "sf", ids, "weights", out, st, SimpleNamespace(moe_comm=comm)
    )
    assert got is out
    used = [] if graph else ["record_stream:out@main", "record_stream:ids@main"]
    assert trace.plan() == [
        "sort@main",
        "record:fork@main",
        *used,
        "wait:fork@memset",
        "zero_local_rows@memset",
        "record:ready@memset",
        "fc1@main",
        "wait:ready@main",
        "fc2_finalize@main",
    ]
    tile, fc1_tactic, fc2_tactic = ns["_MOE_TILINGS"][
        ns["_CONSTANTS"]["context"]["moe_expert_tiling"]
    ]
    assert trace.detail("sort") == (offset, local, tile)
    assert trace.detail("zero_local_rows") == ("out", "ids", offset, local)
    assert trace.detail("fc1") == (896, 16, local, offset, tile, fc1_tactic)
    assert trace.detail("fc2_finalize") == ("out", 896, 16, local, offset, tile, fc2_tactic)
    if not graph:
        assert trace.detail("record_stream:out") == trace.detail("record_stream:ids") == "memset"


def test_the_dep16_context_experts_run_256_row_tiles_with_pinned_tactics():
    """moe_expert_tiling names the row tile the sort cuts and the tactics FC1 and FC2 pin; a
    context server runs 256-row tiles on the 2-CTA tactics, and t128_tuned is the AutoTuner's
    tile-128 path."""
    ns = _dep16_experts(_Trace())
    assert ns["_CONSTANTS"]["context"]["moe_expert_tiling"] == "t256_2cta_n256"
    assert dict(ns["_MOE_TILINGS"]) == {
        "t128_tuned": (128, None, None),
        "t256_2cta_n256": (256, _FC1_T256, _FC2_T256),
    }
    assert (ns["_MOE_TILE"], ns["_MOE_FC1_TACTIC"], ns["_MOE_FC2_TACTIC"]) == (
        256,
        _FC1_T256,
        _FC2_T256,
    )


def _expert_group_refusal() -> list:
    """dep16's ``_MOE_GROUPS`` assignment and the module-level checks that read it."""
    body = ast.parse(_DEP16.read_text()).body
    checks = [
        node
        for node in body
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Name) and n.id == "_MOE_GROUPS" for n in ast.walk(node.test))
    ]
    assert len(checks) == 1, "one import-time check reads _MOE_GROUPS"
    return _definitions(_DEP16, ("_MOE_GROUPS",)) + checks


def test_dep16_refuses_more_than_one_expert_group_at_import():
    """The bucket routing computes the top-k of one expert group (noaux_tc's with one group)."""
    k31 = _load(_DEP16, ("_K31",), dict(MappingProxyType=MappingProxyType))["_K31"]
    assert k31["num_expert_group"] == 1
    _run(_DEP16, _expert_group_refusal(), dict(_K31=k31))
    with pytest.raises(NotImplementedError, match="num_expert_group=8"):
        _run(_DEP16, _expert_group_refusal(), dict(_K31={**k31, "num_expert_group": 8}))

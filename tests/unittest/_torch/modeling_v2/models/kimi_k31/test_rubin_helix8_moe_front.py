# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's MoE front on CUDA graph steps: FC12's NVFP4 input is quantized on the latent stream,
beside the routing tail; under ``router_first_wide`` the router GEMM is issued first above 96 rows.

On a graph step ``_moe_tp_gen_streams`` forks ``latent_down`` onto ``moe_comm.latent_stream`` beside
the router GEMM and the routing tail. The NVFP4 quantization of its output (``_moe_quantize``) runs
on that stream too, right after it, so it does not wait for the routing tail, which it does not
read. FC12 waits for both: the routing in its own stream's order, the quantized input through
``latent_ready``.

CPU (``cpu_only``): helix8's MoE block runs from its modeling.py over recording stand-ins (the
entries, the streams and events, the allocations), on graph steps of 96 rows (the router GEMM after
the fork), 64 rows (the router GEMM first), 16 rows of a KDA layer (the exact router) and 300 rows
(``noaux_tc`` and moe_sort, the TP group's all-reduce):

* the latent stream runs latent_down, the quantization, then records ``latent_ready``, and never
  waits for the current stream after the fork: the routing tail is in no order before the
  quantization;
* FC12 runs on the current stream after the routing tail and after the wait on ``latent_ready``;
* every kernel touches memory allocated before the step, on its own stream, or on another stream
  before its own stream's last wait on that one (FC12 reads x4 and sf, which the latent stream
  allocates, after the current stream's wait);
* the same block with ``latent_ready`` recorded before the quantization is flagged;
* an eager step runs on the current stream alone, the quantization after the routing tail;
* a graph step logs the README's INFO line through ``logger.info_once``; an eager step does not;
* ``router_first_wide`` (a generation constant, on; the source's constant switched off here for the
  off path): a 144-row graph step (b24 k5) issues the router GEMM before the latent stream's
  kernels, as up to 64 rows, with the same checks of streams, waits and memory as above and its own
  README line once; 96 rows keep the latent fork first; off, 144 rows keep the 96-row order; eager
  steps do not change.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
from torch.overrides import TorchFunctionMode

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"

# K3.1's MoE at helix8's share (TP2 x EP4): hidden, latent, router width, top-k, local experts and
# their TP2 slice of the intermediate, the shared experts' TP8 share.
_HIDDEN, _LATENT, _EXPERTS, _TOP_K = 7168, 3584, 896, 16
_LOCAL, _INTER, _SHARED = 224, 1536, 768
_OFFSET = 672  # EP rank 3's window
_LAYER = 5

# The block's classes and functions, run from modeling.py over its module-level constants (those
# that evaluate on their own, in source order: a constant a later commit adds is picked up as well).
_BLOCK = (
    "_MoeOut",
    "_moe_cublas_linear",
    "_moe_latent_gemm",
    "_moe_in_bands",
    "_moe_latent_up",
    "_moe_router",
    "_moe_router_gen",
    "_moe_route",
    "_moe_quantize",
    "_moe_sort",
    "_moe_fc12_zeroed",
    "_moe_fc12",
    "_moe_one_shot",
    "_moe_tp_gen",
    "_moe_tp_gen_streams",
    "_moe_cp_tail_serves",
    "latent_moe",
)


def _tree(**switches) -> ast.Module:
    """modeling.py's tree; ``switches`` set generation constants (``name=True``) in its source.
    The column-parallel MoE tail (``moe_cp_tail``) is off unless a switch turns it on: this file
    checks today's replicated tail (test_rubin_helix8_moe_cp_tail.py checks the tail)."""
    switches = {"moe_cp_tail": False, **switches}
    text = _MODELING.read_text()
    for name, value in switches.items():
        if text.count(f'"{name}": {value},') == 1:
            continue
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
        names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
        if not names or not isinstance(target, (ast.Name, ast.Tuple)):
            continue
        # A value that reads an import or a later definition is not a constant the block reads.
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        if isinstance(target, ast.Name):
            ns[target.id] = value
        else:
            for name, item in zip(names, value):
                ns[name] = item
    return ns


def _definitions(tree: ast.Module, names) -> list:
    """The module-level definitions and assignments of ``names``, in source order; each name must
    be defined exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(copy.deepcopy(node))
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), seen
    return nodes


def _block(tree: ast.Module, namespace: dict) -> dict:
    """The block of ``tree`` executed over its constants and ``namespace``; annotations stay
    unevaluated."""
    module = ast.fix_missing_locations(ast.Module(body=_definitions(tree, _BLOCK), type_ignores=[]))
    ns = {**_constants(tree), "NamedTuple": NamedTuple, "Any": Any, **namespace}
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _tensors(obj):
    """The tensors in ``obj`` and its nested tuples, lists and dicts."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


def _memory(t: torch.Tensor) -> int:
    """The address of the allocation ``t`` views; 0 for an empty tensor."""
    return t.untyped_storage().data_ptr() if t.numel() else 0


class _Plan:
    """One step's host order on four streams: its allocations, event records and waits, kernels.

    ``clock[s][o]`` is the latest point of stream ``o``'s order that stream ``s`` waited for.
    Memory allocated on stream ``o`` at point ``a`` is memory the caching allocator may have freed
    in ``o``'s order, so a kernel of another stream ``s`` may touch it only with
    ``clock[s][o] > a``. Memory the step did not allocate (weights, inputs) is never handed out.
    ``ops`` is the step's order as ``op@stream``.
    """

    STREAMS = ("main", "latent", "prezero", "aux")

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {s: {} for s in self.STREAMS}
        self.homes: dict = {}  # allocation address -> (stream, point)
        self.kernels: list = []  # (point, stream, name, what it read and wrote, clock then)
        self.ops: list = []
        self.problems: list = []

    def tick(self) -> int:
        self.now += 1
        return self.now

    def position(self, stream: str) -> dict:
        """What a wait on an event recorded on ``stream`` now waits for."""
        return {**self.clock[stream], stream: self.tick()}

    def wait(self, waiter: str, position: dict) -> None:
        mine = self.clock[waiter]
        for stream, at in position.items():
            if stream != waiter:
                mine[stream] = max(mine.get(stream, 0), at)

    def allocate(self, t: torch.Tensor) -> None:
        self.homes[_memory(t)] = (self.stream, self.tick())

    def kernel(self, name: str, payload) -> None:
        at, stream = self.tick(), self.stream
        for t in _tensors(payload):
            home = self.homes.get(_memory(t))
            if home is None or home[0] == stream:
                continue
            waited = self.clock[stream].get(home[0], 0)
            if waited <= home[1]:
                self.problems.append(
                    f"{name} on the {stream} stream touches memory the {home[0]} stream "
                    f"allocated at {home[1]}, after the {stream} stream's last wait on it "
                    f"({waited})"
                )
        self.kernels.append((at, stream, name, payload, dict(self.clock[stream])))
        self.ops.append(f"{name}@{stream}")

    def kernel_at(self, name: str) -> tuple:
        (hit,) = [k for k in self.kernels if k[2] == name]
        return hit


# Torch calls that only allocate or view: no kernel reads their inputs.
_ALLOCATE_ONLY = frozenset({"empty", "empty_like", "empty_strided", "new_empty"})


class _Record(TorchFunctionMode):
    """The step's torch calls in the plan: a result in fresh memory is an allocation on the
    current stream; a call that fills fresh memory or writes in place is a kernel there."""

    def __init__(self, plan: _Plan):
        super().__init__()
        self.plan = plan

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        held = {_memory(t) for t in _tensors((args, kwargs))}
        out = func(*args, **kwargs)
        fresh = [t for t in _tensors(out) if _memory(t) and _memory(t) not in held]
        for t in fresh:
            self.plan.allocate(t)
        name = getattr(func, "__name__", "")
        in_place = name.endswith("_") and not name.endswith("__")
        if (fresh or in_place) and name not in _ALLOCATE_ONLY:
            self.plan.kernel(f"torch {name}", (args, kwargs, out))
        return out


class _Stream:
    def __init__(self, name: str):
        self.name = name


class _Event:
    """A CUDA event of the plan: recorded on the current stream, waited for by the current one."""

    def __init__(self, plan: _Plan, name: str):
        self.plan, self.name, self.at = plan, name, None

    def record(self) -> None:
        self.at = self.plan.position(self.plan.stream)
        self.plan.ops.append(f"record:{self.name}@{self.plan.stream}")

    def wait(self) -> None:
        assert self.at is not None, f"a wait on {self.name}, which nothing recorded"
        self.plan.wait(self.plan.stream, self.at)
        self.plan.ops.append(f"wait:{self.name}@{self.plan.stream}")


class _On:
    """``torch.cuda.stream(s)``: what runs inside is issued on ``s``."""

    def __init__(self, plan: _Plan, stream: _Stream):
        self.plan, self.stream, self.outer = plan, stream, None

    def __enter__(self):
        self.outer, self.plan.stream = self.plan.stream, self.stream.name

    def __exit__(self, *exc):
        self.plan.stream = self.outer


class _Torch:
    """``torch`` as the block sees it: ``torch.cuda.stream`` issues on the plan's streams."""

    def __init__(self, plan: _Plan):
        self.cuda = SimpleNamespace(stream=lambda stream: _On(plan, stream))

    def __getattr__(self, name):
        return getattr(torch, name)


class _Logger:
    """``logger`` as the block sees it: the ``info_once`` calls, in order."""

    def __init__(self, plan: _Plan):
        self.plan, self.once = plan, []

    def info_once(self, *msg, key):
        self.once.append((" ".join(map(str, msg)), key))
        self.plan.ops.append("log")


def _entries(plan: _Plan) -> dict:
    """The block's entries as stand-ins: each is a kernel on the current stream over what it is
    passed and returns, and returns a result of the entry's shape (no values)."""

    def cublas_mm(a, b):
        out = torch.empty(a.shape[0], b.shape[1], dtype=a.dtype)
        plan.kernel("latent_down" if b.shape[1] == _LATENT else "latent_up", (a, b, out))
        return out

    def dsv3_router_gemm_op(x, router_t, bias, dtype):
        out = torch.empty(x.shape[0], router_t.shape[1], dtype=dtype)
        plan.kernel("router_gemm", (x, router_t, out))
        return out

    def rubin_router_logits(x, router):
        out = torch.empty(x.shape[0], router.shape[0], dtype=torch.float32)
        plan.kernel("router_exact", (x, router, out))
        return out

    def _routed(rows: int) -> tuple:
        return (
            torch.empty(rows, _TOP_K, dtype=torch.int32),
            torch.empty(rows, _TOP_K, dtype=torch.float32),
        )

    def _six_tables(rows: int) -> tuple:
        i32 = torch.int32
        return tuple(torch.empty(n, dtype=i32) for n in (64, 64, rows * _TOP_K, 4096, 1, 1))

    def rubin_route_tail(logits, bias, scale, offset, num_local, tile, *, route_approx, **kw):
        out = (*_routed(logits.shape[0]), *_six_tables(logits.shape[0]))
        plan.kernel("route_tail", (logits, bias, out))
        return out

    def noaux_tc_op(logits, bias, groups, topk_groups, top_k, scale):
        ids, weights = _routed(logits.shape[0])
        plan.kernel("noaux_tc", (logits, bias, ids, weights))
        return weights, ids

    def rubin_moe_sort(ids, weights, num_experts, top_k, offset, num_local, tile):
        out = _six_tables(ids.shape[0])
        plan.kernel("moe_sort", (ids, weights, out))
        return out

    def fp4_quantize(x, scale, sf_vec, sf_use_ue8m0, swizzled):
        rows, cols = x.shape
        out = (
            torch.empty(rows, cols // 2, dtype=torch.uint8),
            torch.empty(rows * cols // sf_vec, dtype=torch.uint8),
        )
        plan.kernel("fp4_quantize", (x, scale, out))
        return out

    def rubin_fc12(*args, **kwargs):
        plan.kernel("fc12", (args, kwargs))

    def _moe_shared(L, x):
        out = torch.empty(x.shape[0], _HIDDEN, dtype=torch.bfloat16)
        plan.kernel("shared", (x, out))
        return out

    def allreduce_module(x, all_reduce, fusion_op, norm_weight, eps):
        out = torch.empty_like(x)
        plan.kernel(f"allreduce{fusion_op}", (x, out))
        return out

    def rubin_latent_ar_norm_row(x, weight, eps, out, **kw):
        plan.kernel("latent_ar_norm", (x, weight, out))

    def rubin_latent_up_resident(x, weight):
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype)
        plan.kernel("latent_up_resident", (x, weight, out))
        return out

    return dict(
        cublas_mm=cublas_mm,
        dsv3_router_gemm_op=dsv3_router_gemm_op,
        rubin_router_logits=rubin_router_logits,
        rubin_route_tail=rubin_route_tail,
        noaux_tc_op=noaux_tc_op,
        rubin_moe_sort=rubin_moe_sort,
        fp4_quantize=fp4_quantize,
        rubin_fc12=rubin_fc12,
        _moe_shared=_moe_shared,
        allreduce_module=allreduce_module,
        rubin_latent_ar_norm_row=rubin_latent_ar_norm_row,
        # The GEMM pins run today's call here (test_rubin_helix8_gemm_pins.py checks them).
        _gemm_pinned=lambda site, a, b, today, out=None, out_dtype=None: today(),
        rubin_latent_up_resident=rubin_latent_up_resident,
    )


def _shape_only(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    """A tensor of ``shape`` over one row of memory: the stand-ins read shapes only."""
    return torch.empty(shape[-1], dtype=dtype).expand(*shape)


def _layer(after_kda: bool) -> SimpleNamespace:
    """The MoE weight views the block reads, at helix8's share."""
    u8, f32 = torch.uint8, torch.float32
    return SimpleNamespace(
        router=_shape_only(_EXPERTS, _HIDDEN),
        router_bias=torch.empty(_EXPERTS, dtype=f32),
        latent_down=_shape_only(_LATENT, _HIDDEN),
        latent_norm=torch.empty(_LATENT, dtype=torch.bfloat16),
        latent_up=_shape_only(_HIDDEN, _LATENT),
        fc1_w=_shape_only(_LOCAL, 2 * _INTER, _LATENT // 2, dtype=u8),
        fc1_s=_shape_only(_LOCAL, 2 * _INTER, _LATENT // 16, dtype=u8),
        fc2_w=_shape_only(_LOCAL, _LATENT, _INTER // 2, dtype=u8),
        fc2_s=_shape_only(_LOCAL, _LATENT, _INTER // 16, dtype=u8),
        fc31_in_scale=torch.empty(1, dtype=f32),
        fc2_in_scale=torch.empty(1, dtype=f32),
        fc31_alpha=torch.empty(_LOCAL, dtype=f32),
        fc2_alpha=torch.empty(_LOCAL, dtype=f32),
        shared_gu_w=_shape_only(2 * _SHARED, _HIDDEN, dtype=u8),
        shared_dn_w=_shape_only(_HIDDEN, _SHARED, dtype=u8),
        expert_offset=_OFFSET,
        after_kda=after_kda,
    )


def _rt(plan: _Plan) -> SimpleNamespace:
    events = {
        name: _Event(plan, name)
        for name in ("shared_fork", "shared_join", "latent_fork", "latent_ready", "prezero_ready")
    }
    latent_ws = dict(
        world_size=8,
        max_tokens=128,
        buffer_bytes=1 << 30,
        buffer=torch.empty(1),
        mc=torch.empty(1),
        flags=torch.empty(1, dtype=torch.int32),
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
        # moe_shared_late's stream: None outside its band steps.
        shared_stream=None,
    )


def _run(rows: int, cuda_graph: bool = True, after_kda: bool = False, tree=None):
    """``(plan, logger, block)`` of one ``latent_moe`` call of ``tree`` (modeling.py by default)."""
    plan = _Plan()
    log = _Logger(plan)
    block = _block(tree or _tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan)})
    # Weights, inputs and workspaces exist before the step: built outside the recording.
    layer, rt, st = _layer(after_kda), _rt(plan), SimpleNamespace(cuda_graph=cuda_graph)
    x = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16)
    with _Record(plan):
        out = block["latent_moe"](layer, _LAYER, x, st, rt)
    assert tuple(out.routed.shape) == (rows, _HIDDEN) and tuple(out.shared.shape) == (rows, _HIDDEN)
    return plan, log, block


def _latent_ready_before_the_quantization(tree: ast.Module) -> ast.Module:
    """``tree`` with ``latent_ready`` recorded right after latent_down, before the quantization on
    the latent stream: FC12 would then read the quantized input without an edge to its kernel."""
    tree = copy.deepcopy(tree)
    (fn,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_moe_tp_gen_streams"
    ]
    (latent,) = [
        s
        for s in fn.body
        if isinstance(s, ast.With) and "c.latent_stream" in ast.unparse(s.items[0])
    ]
    texts = [ast.unparse(s) for s in latent.body]
    (q,) = [k for k, t in enumerate(texts) if "_moe_quantize(" in t]
    (r,) = [k for k, t in enumerate(texts) if t == "c.latent_ready.record()"]
    assert q < r, texts
    body = list(latent.body)
    body.insert(q, body.pop(r))
    latent.body = body
    return tree


# The graph step at 96 rows (b16 k5): the router GEMM after the fork.
_GRAPH_96 = [
    "record:latent_fork@main",
    "wait:latent_fork@prezero",
    "torch zero_@prezero",
    "record:prezero_ready@prezero",
    "wait:latent_fork@latent",
    "latent_down@latent",
    "fp4_quantize@latent",
    "record:latent_ready@latent",
    "router_gemm@main",
    "record:shared_fork@main",
    "route_tail@main",
    "wait:latent_ready@main",
    "wait:prezero_ready@main",
    "fc12@main",
    "log",
    "wait:shared_fork@aux",
    "shared@aux",
    "allreduce0@aux",
    "record:shared_join@aux",
    "latent_ar_norm@main",
    "latent_up@main",
    "wait:shared_join@main",
]
# The eager step: one stream, the quantization after the routing tail as before.
_EAGER_96 = [
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


def test_the_step_shapes_take_the_paths_this_file_checks():
    plan, _, block = _run(96)
    assert block["_MOE_ROUTER_FIRST_MAX_M"] == 64 and block["_MOE_ROUTER_EXACT_MAX_M"] == 16
    assert block["_MOE_ROUTE_TAIL_MAX_M"] == 256 and block["_MOE_OWN_WS_MAX_M"] == 128
    assert block["_MOE_FC12_PREZERO"]
    # router_first_wide is on: it covers 97 rows up to the routing tail's bound; off, none.
    assert block["_CONSTANTS"]["generation"]["router_first_wide"] is True
    assert block["_MOE_ROUTER_FIRST_WIDE"] == (97, 256)
    assert _constants(_tree(router_first_wide=False))["_MOE_ROUTER_FIRST_WIDE"] == (1, 0)


def test_a_96_row_graph_step_quantizes_on_the_latent_stream():
    plan, log, _ = _run(96)
    assert plan.ops == _GRAPH_96


@pytest.mark.parametrize(
    "rows, after_kda, router",
    [(96, False, "router_gemm"), (64, False, "router_gemm"), (16, True, "router_exact")],
    ids=["b16k5", "router_first", "exact_router"],
)
def test_the_quantization_does_not_wait_for_the_routing_tail(rows, after_kda, router):
    plan, _, _ = _run(rows, after_kda=after_kda)
    assert not plan.problems, "\n".join(plan.problems)
    q_at, q_stream, _, _, q_clock = plan.kernel_at("fp4_quantize")
    d_at, d_stream, _, _, _ = plan.kernel_at("latent_down")
    t_at, t_stream, _, _, _ = plan.kernel_at("route_tail")
    r_at, r_stream, _, _, _ = plan.kernel_at(router)
    f_at, f_stream, _, (args, kwargs), f_clock = plan.kernel_at("fc12")
    assert (q_stream, d_stream, t_stream, r_stream, f_stream) == (
        "latent",
        "latent",
        "main",
        "main",
        "main",
    )
    assert d_at < q_at, "the quantization runs before latent_down"
    # The latent stream waited for nothing of the current stream's after the fork.
    assert q_clock.get("main", 0) < min(t_at, r_at), q_clock
    # FC12 runs after the routing tail and after the quantization's record.
    assert t_at < f_at and f_clock.get("latent", 0) > q_at, f_clock
    # FC12 reads the quantization's outputs: allocated on the latent stream.
    x4, sf = args[0], args[2]
    assert {plan.homes[_memory(t)][0] for t in (x4, sf)} == {"latent"}


@pytest.mark.parametrize("rows", [96, 300], ids=["own_workspace", "tp_all_reduce"])
def test_a_graph_step_touches_memory_after_its_earlier_users(rows):
    plan, _, _ = _run(rows)
    assert not plan.problems, "\n".join(plan.problems)
    assert plan.ops.index("fp4_quantize@latent") < plan.ops.index("record:latent_ready@latent")
    last = {}
    for at, stream, *_ in plan.kernels:
        last[stream] = at
    for side in ("latent", "prezero", "aux"):
        assert side in last, f"nothing runs on the {side} stream"
        assert plan.clock["main"].get(side, 0) > last[side], f"the {side} stream is not joined"


def test_the_check_flags_latent_ready_recorded_before_the_quantization():
    plan, _, _ = _run(96, tree=_latent_ready_before_the_quantization(_tree()))
    flagged = [p for p in plan.problems if p.startswith("fc12 on the main stream")]
    assert flagged, plan.problems


def test_an_eager_step_runs_on_the_current_stream_alone():
    plan, log, _ = _run(96, cuda_graph=False)
    assert plan.ops == _EAGER_96
    assert not any(plan.clock.values()) and not plan.problems
    assert log.once == []


# The front of a graph step (to FC12) with the router GEMM issued first: up to 64 rows, and above
# 96 rows under router_first_wide.
_FRONT_ROUTER_FIRST = [
    "record:latent_fork@main",
    "wait:latent_fork@prezero",
    "torch zero_@prezero",
    "record:prezero_ready@prezero",
    "router_gemm@main",
    "wait:latent_fork@latent",
    "latent_down@latent",
    "fp4_quantize@latent",
    "record:latent_ready@latent",
    "record:shared_fork@main",
    "route_tail@main",
    "wait:latent_ready@main",
    "wait:prezero_ready@main",
    "fc12@main",
]


def _front(ops: list) -> list:
    """The ops of a graph step up to FC12."""
    return ops[: ops.index("fc12@main") + 1]


def test_off_a_144_row_graph_step_forks_latent_down_first():
    plan, log, block = _run(144, tree=_tree(router_first_wide=False))
    assert _front(plan.ops) == _front(_GRAPH_96)
    assert not plan.problems, "\n".join(plan.problems)
    assert [line for line, _ in log.once] == [block["_MOE_QUANT_BESIDE_LINE"]]


def test_router_first_wide_issues_the_router_gemm_first_at_144_rows():
    plan, log, block = _run(144)
    assert _front(plan.ops) == _FRONT_ROUTER_FIRST
    assert _front(_run(64)[0].ops) == _FRONT_ROUTER_FIRST
    assert not plan.problems, "\n".join(plan.problems)
    q_at, _, _, _, q_clock = plan.kernel_at("fp4_quantize")
    r_at, r_stream, _, _, _ = plan.kernel_at("router_gemm")
    t_at, _, _, _, _ = plan.kernel_at("route_tail")
    f_at, f_stream, _, (args, _), f_clock = plan.kernel_at("fc12")
    assert (r_stream, f_stream) == ("main", "main") and r_at < plan.kernel_at("latent_down")[0]
    # The latent stream waits for the fork only: not for the router GEMM issued before it, nor
    # for the routing tail.
    assert q_clock.get("main", 0) < r_at, q_clock
    assert t_at < f_at and f_clock.get("latent", 0) > q_at, f_clock
    assert {plan.homes[_memory(t)][0] for t in (args[0], args[2])} == {"latent"}
    for side in ("latent", "prezero", "aux"):
        last = max(at for at, stream, *_ in plan.kernels if stream == side)
        assert plan.clock["main"].get(side, 0) > last, f"the {side} stream is not joined"
    wide = block["_MOE_ROUTER_FIRST_WIDE_LINE"]
    assert log.once == [(block["_MOE_QUANT_BESIDE_LINE"],) * 2, (wide, wide)]
    assert wide.startswith("Kimi K3.1 Rubin MoE front: ") and "\n" not in wide
    assert f"\n{wide}\n" in _README.read_text()


def test_router_first_wide_keeps_96_rows_and_eager_steps():
    tree = _tree()
    plan, log, block = _run(96, tree=tree)
    assert plan.ops == _GRAPH_96 and not plan.problems
    assert block["_MOE_ROUTER_FIRST_WIDE_LINE"] not in [line for line, _ in log.once]
    plan, log, _ = _run(144, cuda_graph=False, tree=tree)
    assert plan.ops == _EAGER_96
    assert not any(plan.clock.values()) and log.once == []


def test_a_graph_step_logs_the_readme_line_once_per_process():
    plan, log, block = _run(96)
    line = block["_MOE_QUANT_BESIDE_LINE"]
    assert log.once == [(line, line)]
    assert line.startswith("Kimi K3.1 Rubin MoE front: ") and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()

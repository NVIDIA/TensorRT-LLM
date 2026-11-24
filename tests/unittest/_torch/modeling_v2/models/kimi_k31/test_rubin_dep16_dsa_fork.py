# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's DSA decode streams on CUDA graph steps (``dsa_idx_fork``, ``dsa_gate_early``,
``dsa_rope_before_idx_join``, ``dsa_logits_first``), on CPU.

On a graph step ``mla_generate`` runs the indexer's work on the layer input alone (the widening,
the TF32 ``[wk | weights_proj]``, the key norm and quantization, the index-key insert) on
``rt.idx_stream`` from the top of the layer, q_b and then ``g_proj(x)`` on ``rt.fork_stream``,
and everything else on the current stream, which waits for q before W_UK and the RoPE kernel (a
generation server issues those two before the join), joins the idx stream before the logits
weights take the query scales, and waits for g's event before the gate. Under
``dsa_logits_first`` (a generation server's) the indexer's query projection and the logits
weights run on the aux stream after kv_a's norm, W_UK and the RoPE kernel follow g_proj on the
fork stream, and the current stream runs the logits, the Top-K and the remap before it waits for
the fork stream right before the attention. The block runs from modeling.py over recording
stand-ins with a generation server's levers (``_levers``), and with ``dsa_logits_first`` off: every
kernel that touches memory another stream allocated is ordered after that stream's work on it,
the logits read the index keys after the insert, the graph step issues today's kernels (the
widening and the weights' two products by their one-kernel entries), and an eager step issues
today's kernels in today's order.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib.util
from collections import Counter
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch.overrides import TorchFunctionMode

pytestmark = pytest.mark.cpu_only

_TARGET = "kimi_k31_pro_nvfp4__sm_107__dep16"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MODELING = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31" / _TARGET / "modeling.py"

# K3.1's MLA and indexer under attention DP: all 96 heads attend and feed o_proj.
_HIDDEN, _Q_LORA, _KV_LORA, _ROPE, _NOPE, _V = 7168, 1536, 512, 64, 128, 128
_HEADS, _IDX_HEADS, _IDX_DIM = 96, 32, 128
_TOPK = 2048
_TOKENS_PER_BLOCK = 64
_LAYER = 3  # a DSA layer
# A verify step of three requests of four rows in a CUDA graph bucket of 16 rows.
_ROWS, _GEN, _NEXT_N = 16, 3, 4
_LIVE = _GEN * _NEXT_N
_MAX_LEN = 512
_BLOCK = (
    "mla_generate",
    "_mla_project",
    "_widen_fp32",
    "_idx_project",
    "_idx_keys",
    "_idx_queries",
    "_idx_insert",
    "_idx_logits_arena",
    "_idx_decode",
    "_idx_remap",
    "_mla_absorb",
    "_mla_expand",
    "_mla_window",
    "_Mx",
    "_dense",
    "_MlaQuery",
    "_mla_gen_heads",
    "_mla_gen_query",
    "_mla_generation",
    "_mla_attend",
    "_mla_generate_logits_first",
    "_mla_output",
    "_Levers",
    "_levers",
)
# Today's mla_generate (before dsa_idx_fork): every kernel on the current stream, the indexer
# after q_b, the gate projection before the gate.
_TODAY = """
def mla_generate(L, i, x, st, rt):
    n = st.num_tokens
    qr, latent = _mla_project(L, x, st, rt)
    q = rubin_linear(qr, L.q_b, False)
    q8, k8, ks, w = _idx_project(L, x, qr, st, rt)
    _idx_insert(i, k8[:n], ks[:n], n, st, rt)
    out = x.new_empty(x.shape[0], _MLA_OUT_HEADS * _MLA_V)
    _mla_generation(L, i, q[:n], latent[:n], q8, w, out[:n], st, rt)
    return _mla_output(L, x, out, n)
"""
_LINES = (
    "Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream beside [q_a | kv_a]",
    "Kimi K3.1 Rubin DSA indexer decode: (w * qs) * scale in one kernel",
    "Kimi K3.1 Rubin MLA: output gate projection on the fork stream after q_b",
    "Kimi K3.1 Rubin DSA: the indexer's logits and Top-K before W_UK and the RoPE kernel, which "
    "follow g_proj on the fork stream (dsa_logits_first)",
)
_ROPE_FIRST_LINE = (
    "Kimi K3.1 Rubin MLA: W_UK and the RoPE kernel before the indexer join "
    "(dsa_rope_before_idx_join)"
)


@functools.lru_cache(maxsize=None)
def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


@functools.lru_cache(maxsize=None)
def _constants() -> MappingProxyType:
    """modeling.py's module-level constants that evaluate on their own, in source order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in _tree().body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        ns[node.targets[0].id] = value
    return MappingProxyType(ns)


def _block(tree: ast.Module, namespace: dict, today: bool = False) -> dict:
    """The block's functions of ``tree`` over the constants and ``namespace``; ``today``: with
    today's ``mla_generate`` in place of the tree's."""
    body = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _BLOCK
    ]
    missing = set(_BLOCK) - {n.name for n in body}
    assert not missing, f"dep16's modeling.py defines no {sorted(missing)}"
    if today:
        body = [n for n in body if n.name != "mla_generate"] + ast.parse(_TODAY).body
    ns = {**_constants(), "NamedTuple": NamedTuple, **namespace}
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


def _memory(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr() if t.numel() else 0


class _Plan:
    """One step's host order on four streams: allocations, waits, kernels (helix8's recorder).

    ``clock[s][o]``: the latest point of stream ``o``'s order that stream ``s`` waited for. A
    kernel of stream ``s`` may touch memory stream ``o`` allocated at point ``a`` only with
    ``clock[s][o] > a``. ``seen[k]`` is the kernel's stream clock when it was issued.
    """

    STREAMS = ("main", "aux", "fork", "idx")

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {s: {} for s in self.STREAMS}
        self.homes: dict = {}
        self.kernels: list = []  # (point, stream, name, payload)
        self.seen: list = []
        self.problems: list = []

    def tick(self) -> int:
        self.now += 1
        return self.now

    def position(self, stream: str) -> dict:
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
        self.kernels.append((at, stream, name, payload))
        self.seen.append(dict(self.clock[stream]))

    def find(self, name: str, stream: str | None = None) -> list:
        return [
            k
            for k, (_, s, n, _) in enumerate(self.kernels)
            if n == name and (stream is None or s == stream)
        ]


_ALLOCATE_ONLY = frozenset({"empty", "empty_like", "empty_strided", "new_empty"})


class _Record(TorchFunctionMode):
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
    def __init__(self, plan: _Plan, name: str):
        self.plan, self.name = plan, name

    def wait_stream(self, other: "_Stream") -> None:
        self.plan.wait(self.name, self.plan.position(other.name))


class _Event:
    def __init__(self, plan: _Plan):
        self.plan, self.at = plan, None

    def record(self) -> None:
        self.at = self.plan.position(self.plan.stream)

    def wait(self) -> None:
        assert self.at is not None, "a wait on an event nothing recorded"
        self.plan.wait(self.plan.stream, self.at)


class _On:
    def __init__(self, plan: _Plan, stream: _Stream):
        self.plan, self.stream, self.outer = plan, stream, None

    def __enter__(self):
        self.outer, self.plan.stream = self.plan.stream, self.stream.name

    def __exit__(self, *exc):
        self.plan.stream = self.outer


class _Torch:
    def __init__(self, plan: _Plan, streams: dict):
        self.cuda = SimpleNamespace(
            current_stream=lambda: streams[plan.stream],
            stream=lambda stream: _On(plan, stream),
            is_current_stream_capturing=lambda: True,
        )

    def __getattr__(self, name):
        return getattr(torch, name)


def _entries(plan: _Plan) -> dict:
    """The block's entries as stand-ins: a kernel on the current stream over what each is passed
    and returns, with a result of the entry's shape."""

    def like(t):
        return torch.empty(t.shape, dtype=t.dtype)

    def remap(req, table, tok, *rest):
        return torch.empty(tok.shape, dtype=torch.int32)

    results = {
        "cublas_mm": lambda a, b: torch.empty(a.shape[0], b.shape[1], dtype=a.dtype),
        "rubin_linear": lambda x, w, tf32: torch.empty(x.shape[0], w.shape[0], dtype=x.dtype),
        "flashinfer_rmsnorm": lambda x, w, eps: like(x),
        "rubin_layer_norm": lambda x, w, b, eps: like(x),
        "rubin_upcast_fp32": lambda x: torch.empty(x.shape, dtype=torch.float32),
        "rubin_idx_weights": lambda w, qs, scale: torch.empty(w.shape, dtype=torch.float32),
        "fused_cat_fp8": lambda a, b, ue8m0: (
            torch.empty(*a.shape[:-1], a.shape[-1] + b.shape[-1], dtype=torch.float8_e4m3fn),
            torch.empty(*a.shape[:-1], 1, dtype=torch.float32),
        ),
        "indexer_k_cache_scatter_op": None,
        "bmm_out": None,
        "rubin_pad_heads": lambda dst, src, heads: dst,
        "mla_rope_generation": None,
        "rubin_fp8_paged_mqa_logits": lambda *args, pdl=False: args[-1],
        "rubin_gvr_topk_decode": None,
        "indexer_topk_decode": None,
        "convert_req_index_to_global": remap,
        "rubin_dsa_topk_remap": remap,
        "thop_attention": lambda **kw: kw["output"],
        "k31_sigmoid_gate_mul": lambda out, gate: like(out),
    }

    def entry(name, result):
        def run(*args, **kwargs):
            out = None if result is None else result(*args, **kwargs)
            plan.kernel(name, (args, kwargs, out))
            return out

        return run

    return {name: entry(name, result) for name, result in results.items()}


def _shape_only(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.empty(shape[-1], dtype=dtype).expand(*shape)


def _cpu_layer() -> SimpleNamespace:
    return SimpleNamespace(
        qkv_a_t=_shape_only(_Q_LORA + _KV_LORA + _ROPE, _HIDDEN).t(),
        q_a_norm=_shape_only(_Q_LORA),
        kv_a_norm=_shape_only(_KV_LORA),
        q_b=_shape_only(_HEADS * (_NOPE + _ROPE), _Q_LORA),
        k_b=_shape_only(_HEADS, _NOPE, _KV_LORA),
        v_b_t=_shape_only(_HEADS, _KV_LORA, _V),
        g=_shape_only(_HEADS * _V, _HIDDEN),
        o=_shape_only(_HIDDEN, _HEADS * _V),
        idx_wq_b_t=_shape_only(_IDX_HEADS * _IDX_DIM, _Q_LORA).t(),
        idx_wk_wp=_shape_only(_IDX_DIM + _IDX_HEADS, _HIDDEN, dtype=torch.float32),
        idx_k_norm_w=_shape_only(_IDX_DIM, dtype=torch.float32),
        idx_k_norm_b=_shape_only(_IDX_DIM, dtype=torch.float32),
    )


def _cpu_step(plan: _Plan, streams: dict, cuda_graph: bool):
    i32 = torch.int32
    dsa = SimpleNamespace(
        next_n=_NEXT_N,
        skip_gen=False,
        slot_fp8=torch.empty(_ROWS, dtype=torch.int64),
        slot_scale=torch.empty(_ROWS, dtype=torch.int64),
        ctx_lens=torch.empty(_GEN, dtype=i32),
        block_table=torch.empty(_GEN, 8, dtype=i32),
        sched=torch.empty(9, 2, dtype=i32),
        max_len=_MAX_LEN,
        gen_kv_lens=torch.empty(_GEN, dtype=i32),
        topk_out=torch.empty(_ROWS, _TOPK, dtype=i32),
        topk_dense=torch.empty(_ROWS, _TOPK, dtype=i32),
        radix_aux_indices=None,
        radix_aux_logits=None,
        req_idx_gen=torch.empty(_LIVE, dtype=i32),
        block_table_gen=torch.empty(_GEN, 8, dtype=i32),
        tokens_per_block=_TOKENS_PER_BLOCK,
        rope_gen=dict(sequence_length=torch.empty(_GEN, dtype=i32)),
        sparse_gen=dict(sequence_length=torch.empty(_GEN, dtype=i32)),
    )
    st = SimpleNamespace(
        num_tokens=_LIVE,
        num_contexts=0,
        num_ctx_tokens=0,
        num_generations=_GEN,
        cuda_graph=cuda_graph,
        dsa=dsa,
        md=SimpleNamespace(max_seq_len=4096, max_num_requests=_GEN, max_total_draft_tokens=3),
        cache=SimpleNamespace(max_attention_window_vec=[None]),
        position_ids=torch.empty(_ROWS, dtype=i32),
    )
    rt = SimpleNamespace(
        events=tuple((_Event(plan), _Event(plan)) for _ in range(4)),
        aux_stream=streams["aux"],
        fork_stream=streams["fork"],
        idx_stream=streams["idx"],
        dsa_pools=SimpleNamespace(
            idx_k_cache={_LAYER: torch.empty(16, _TOKENS_PER_BLOCK, 1, 132, dtype=torch.uint8)},
            pool_params={_LAYER: (1, 0)},
            local={_LAYER: 0},
            rope_positions=4096,
            gen_requests=_GEN,
            idx_logits=torch.empty(_ROWS, _MAX_LEN),
        ),
        mla_rope=torch.empty(1, 16),
        # the decode logits' PDL launch, as the generation constant sets it (dsa_idx_logits_pdl)
        idx_logits_pdl=_constants()["_CONSTANTS"]["generation"]["dsa_idx_logits_pdl"],
    )
    return st, rt


def _plan(tree=None, cuda_graph: bool = True, today: bool = False, logits_first: bool = True):
    """The plan of one ``mla_generate`` call of ``tree`` (modeling.py by default) and its log;
    ``logits_first``: the generation server's ``dsa_logits_first`` (on) or off."""
    plan = _Plan()
    streams = {s: _Stream(plan, s) for s in _Plan.STREAMS}
    logged: list = []
    logger = SimpleNamespace(info_once=lambda msg, key: logged.append((msg, key)))
    block = _block(
        tree or _tree(),
        {"torch": _Torch(plan, streams), "logger": logger, **_entries(plan)},
        today=today,
    )
    st, rt = _cpu_step(plan, streams, cuda_graph)
    # a generation server's, from its constants
    rt.levers = block["_levers"]("generation")._replace(dsa_logits_first=logits_first)
    layer, x = _cpu_layer(), torch.empty(_ROWS, _HIDDEN, dtype=torch.bfloat16)
    with _Record(plan):
        out = block["mla_generate"](layer, _LAYER, x, st, rt)
    assert tuple(out.shape) == (_ROWS, _HIDDEN)
    return plan, logged


def _issued_after(plan: _Plan, reader: int, writer: int) -> bool:
    """Kernel ``reader`` is ordered after kernel ``writer`` (same stream: later; else through a
    wait of the reader's stream past the writer's point)."""
    at_w, s_w = plan.kernels[writer][:2]
    at_r, s_r = plan.kernels[reader][:2]
    if s_w == s_r:
        return at_r > at_w
    return plan.seen[reader].get(s_w, 0) > at_w


def _joined(plan: _Plan) -> dict:
    last = {s: at for at, s, _, _ in plan.kernels}
    return {s: plan.clock["main"].get(s, 0) > at for s, at in last.items() if s != "main"}


def _signature(kernel) -> tuple:
    _, _, name, (args, kwargs, _) = kernel
    shapes = tuple((tuple(t.shape), str(t.dtype)) for t in _tensors((args, kwargs)))
    return name, shapes


def _without(tree: ast.Module, fn_name: str, call: str) -> ast.Module:
    """``tree`` with the statement ``call`` (e.g. ``join.wait()``) removed from ``fn_name``."""
    tree = copy.deepcopy(tree)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name]

    class Drop(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if ast.unparse(node) == call else node

    Drop().visit(fn)
    ast.fix_missing_locations(tree)
    assert ast.unparse(fn).count(call) == 0
    return tree


@pytest.mark.parametrize("logits_first", [True, False])
def test_a_graph_step_orders_every_read_after_its_writer(logits_first):
    plan, _ = _plan(logits_first=logits_first)
    assert not plan.problems, "\n".join(plan.problems)
    streams = {}
    for _, s, name, _ in plan.kernels:
        streams.setdefault(s, []).append(name)
    assert streams["idx"] == [
        "rubin_upcast_fp32",
        "rubin_linear",
        "torch to",
        "rubin_layer_norm",
        "fused_cat_fp8",
        "indexer_k_cache_scatter_op",
    ], streams["idx"]
    side = "aux" if logits_first else "main"  # the indexer's queries and the logits weights
    rope = "fork" if logits_first else "main"  # W_UK and the RoPE kernel
    if logits_first:
        assert streams["fork"] == [
            "rubin_linear",
            "rubin_linear",
            "bmm_out",
            "mla_rope_generation",
        ], streams["fork"]
        assert streams["aux"] == [
            "flashinfer_rmsnorm",
            "cublas_mm",
            "fused_cat_fp8",
            "rubin_idx_weights",
        ], streams["aux"]
    else:
        assert streams["fork"] == ["rubin_linear", "rubin_linear"], streams["fork"]
        assert streams["aux"] == ["flashinfer_rmsnorm"], streams["aux"]
    (insert,) = plan.find("indexer_k_cache_scatter_op", "idx")
    (logits,) = plan.find("rubin_fp8_paged_mqa_logits", "main")
    (weights,) = plan.find("rubin_idx_weights", side)
    q_b, g = plan.find("rubin_linear", "fork")
    w_uk = plan.find("bmm_out", rope)[0]
    (rope_kernel,) = plan.find("mla_rope_generation", rope)
    (attention,) = plan.find("thop_attention", "main")
    (gate,) = plan.find("k31_sigmoid_gate_mul", "main")
    (tf32,) = plan.find("rubin_linear", "idx")
    assert _issued_after(plan, logits, insert), "the logits may read the keys before the insert"
    assert _issued_after(plan, logits, weights), "the logits may read the weights before them"
    assert _issued_after(plan, weights, tf32), "the weights may read w before the projection"
    assert _issued_after(plan, w_uk, q_b), "W_UK may read q before q_b"
    assert _issued_after(plan, attention, rope_kernel), "the attention may run before the RoPE"
    assert _issued_after(plan, gate, g), "the gate may read g before g_proj"
    if logits_first:  # nothing on the current stream waits for W_UK before the logits
        assert not _issued_after(plan, logits, rope_kernel)
    assert plan.kernels[g][3][0][1].shape == (_HEADS * _V, _HIDDEN), "g_proj is not the 2nd"
    # The idx stream starts after the current stream's work before the layer, nothing else.
    first_idx = next(k for k, (_, s, _, _) in enumerate(plan.kernels) if s == "idx")
    assert plan.seen[first_idx].get("main", 0) > 0
    assert _joined(plan) == {"aux": True, "fork": True, "idx": True}


@pytest.mark.parametrize(
    "fn_name, call, logits_first",
    [
        ("mla_generate", "join.wait()", False),
        ("mla_generate", "q_ready.wait()", False),
        ("_mla_output", "ready.wait()", False),
        ("mla_generate", "fork.wait()", False),
        ("mla_generate", "fork.wait()", True),
        ("_mla_generate_logits_first", "kv_fork.wait()", True),
        ("_mla_generate_logits_first", "kv_done.wait()", True),
        ("_mla_generate_logits_first", "qr_ready.wait()", True),
        ("_mla_generate_logits_first", "idx_done.wait()", True),
        ("_mla_generate_logits_first", "latent_ready.wait()", True),
        ("_mla_generate_logits_first", "idx_ready.wait()", True),
        ("_mla_generate_logits_first", "rope_done.wait()", True),
    ],
)
def test_the_check_flags_a_missing_wait(fn_name, call, logits_first):
    plan, _ = _plan(_without(_tree(), fn_name, call), logits_first=logits_first)
    if call == "fork.wait()":
        # The idx stream would read x before the current stream wrote it: no wait on main.
        first_idx = next(k for k, (_, s, _, _) in enumerate(plan.kernels) if s == "idx")
        assert plan.seen[first_idx].get("main", 0) == 0
        return
    assert plan.problems or not all(_joined(plan).values()), "a missing wait went unseen"


@pytest.mark.parametrize("logits_first", [True, False])
def test_a_graph_step_issues_todays_kernels(logits_first):
    """The graph step's kernels are today's graph step's on the same inputs, with ``x.float()``
    as ``rubin_upcast_fp32`` and the weights' two products as ``rubin_idx_weights``."""
    lever, _ = _plan(logits_first=logits_first)
    today, _ = _plan(today=True)
    got = Counter(_signature(k) for k in lever.kernels)
    want = Counter(_signature(k) for k in today.kernels)
    widen = [k for k in want if k[0] == "torch float"]
    products = [k for k in want if k[0].startswith("torch ") and "mul" in k[0]]
    assert len(widen) == 1 and sum(want[k] for k in products) == 2, (widen, products)
    for k in widen + products:
        del want[k]
    assert got.pop(("rubin_upcast_fp32", (((_ROWS, _HIDDEN), "torch.bfloat16"),))) == 1
    (short,) = [k for k in got if k[0] == "rubin_idx_weights"]
    assert got.pop(short) == 1
    assert got == want, (got - want, want - got)


def test_an_eager_step_is_todays_step():
    lever, logged = _plan(cuda_graph=False)
    today, _ = _plan(cuda_graph=False, today=True)
    assert {s for _, s, _, _ in lever.kernels} == {"main"}
    assert [_signature(k) for k in lever.kernels] == [_signature(k) for k in today.kernels]
    # The plan's 16 rows are below mla_gen_pad128_min_rows: the decode keeps 96 heads.
    assert not lever.problems and not any(lever.clock.values()) and not logged


def test_the_constants_and_the_log_lines():
    consts = _constants()["_CONSTANTS"]
    assert consts["generation"]["dsa_idx_fork"] is True
    assert consts["generation"]["dsa_gate_early"] is True
    assert consts["generation"]["dsa_rope_before_idx_join"] is True
    assert consts["generation"]["dsa_logits_first"] is True
    assert "dsa_idx_fork" not in consts["context"] and "dsa_gate_early" not in consts["context"]
    assert "dsa_rope_before_idx_join" not in consts["context"]
    assert "dsa_logits_first" not in consts["context"]
    _, logged = _plan()
    assert [msg for msg, _ in logged] == list(_LINES)
    assert all(msg == key for msg, key in logged)
    _, logged = _plan(logits_first=False)
    assert [msg for msg, _ in logged] == [*_LINES[:3], _ROPE_FIRST_LINE]

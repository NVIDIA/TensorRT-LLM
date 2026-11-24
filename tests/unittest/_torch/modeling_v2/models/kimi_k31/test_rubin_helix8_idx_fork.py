# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's indexer fork on CUDA graph steps (``dsa_idx_fork``): the indexer's work on the layer
input alone runs on ``rt.idx_stream`` from the top of ``mla_generate``.

On a graph step ``mla_generate`` forks the idx stream from the current stream before
``[q_a | kv_a]``. There the indexer's ``[wk | weights_proj]`` projection (the input widened to
FP32, TF32), the key's BF16 rounding, LayerNorm and FP8 quantization, and the index-key insert run
beside ``[q_a | kv_a]``, its norms and the indexer's query projection, which stay on the current
stream. The current stream joins the idx stream right before the logits weights take the query
scales, the first read of anything the idx stream wrote. Same kernels on the same inputs: the same
bits (the fork_q test replays the block on sm_107, with a spin at the head of the idx stream).

CPU (``cpu_only``): helix8's MLA block runs from its modeling.py over recording stand-ins (the
entries, the four streams and their events, the allocations), on a graph step of each indexer path
(the dense Top-K, the key-sharded one with today's chain, and the key-sharded one through
``dsa_ks_select_pack``'s select / pack at a checked row count) with CUDA graph padding rows:

* the key and weights kernels and the insert run on the idx stream, which waits for the current
  stream once, at the fork, before any other kernel of the block; the query projection and its
  quantization run on the current stream;
* every kernel touches memory allocated before the step, on its own stream, or on another stream
  before its own stream's last wait on that one; the logits read the index keys after the join;
* the same block without the join, or forked before its input is written, is flagged;
* a graph step issues the kernels of an eager step, which runs on the current stream alone;
* with the select / pack, its one launch on the current stream after the join replaces the local
  Top-K, the pack, the merge and the remap, and the sparse attention takes its KV pool rows as the
  remap's: the step's Top-K rows themselves, int32 ``[live rows, 2048]``; the helix mask reads the
  counts it wrote;
* a graph step logs the INFO line once, an eager step not; the launch constants state the fork.
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

import pytest
import torch
from torch.overrides import TorchFunctionMode

_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MODELING = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31" / _TARGET / "modeling.py"

# K3.1's MLA and indexer at helix8's share: all 96 query heads attend, o_proj takes 12 of them.
_HIDDEN, _Q_LORA, _KV_LORA, _ROPE, _NOPE, _V = 7168, 1536, 512, 64, 128, 128
_HEADS, _O_HEADS, _IDX_HEADS, _IDX_DIM = 96, 12, 32, 128
_CP = _HEADS // _O_HEADS
_TOPK = 2048
_TOKENS_PER_BLOCK = 64
_LAYER = 3  # a DSA layer
# A decode step of six requests in a CUDA graph bucket of eight rows: two padding rows.
_ROWS, _LIVE = 8, 6
# The MLA block, run from modeling.py.
_BLOCK = (
    "mla_generate",
    "_mla_project",
    "_idx_keys",
    "_idx_queries",
    "_idx_insert",
    "_idx_fallback_logits",
    "_idx_decode",
    "_idx_remap",
    "_ks_select",
    "_ks_kv_rows",
    "_mla_absorb",
    "_mla_expand",
    "_mla_window",
    "_mla_helix_mask",
    "_mla_query",
    "_mla_generation",
    "_mla_output",
)
# The fallback indexer's entries: an eager step's, never a graph step's.
_EAGER_ONLY = (
    "fp8_paged_mqa_logits",
    "indexer_topk_decode",
    "rubin_gvr_topk_decode",
    "allgather_module",
    "rubin_merge_helix_topk",
)
# The indexer's work on the layer input alone, in issue order: the idx stream's on a graph step.
_KEY_KERNELS = (
    "rubin_upcast_fp32",
    "rubin_linear tf32",
    "torch to",
    "rubin_layer_norm",
    "fused_cat_fp8 key",
    "indexer_k_cache_scatter_op",
)
# The indexer's query projection and quantization: the current stream's.
_QUERY_KERNELS = ("cublas_mm idx_wq_b", "fused_cat_fp8 query")


# --- the block's streams, from source ---------------------------------------------------------------


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
        # A value that reads an import or a later definition is not a constant the block reads.
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        ns[node.targets[0].id] = value
    return MappingProxyType(ns)


def _block(tree: ast.Module, namespace: dict) -> dict:
    """The block's functions of ``tree``, executed over the constants and ``namespace``."""
    body = [
        copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _BLOCK
    ]
    missing = set(_BLOCK) - {n.name for n in body}
    assert not missing, f"helix8's modeling.py defines no {sorted(missing)}"
    ns = {**_constants(), **namespace}
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
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
    """One step's host order on four streams: its allocations, waits and kernels.

    ``clock[s][o]`` is the latest point of stream ``o``'s order that stream ``s`` waited for
    (waits and events carry it on). Memory allocated on stream ``o`` at point ``a`` is memory the
    caching allocator may have freed in ``o``'s order, so a kernel of another stream ``s`` may
    touch it only with ``clock[s][o] > a``: nothing else orders it after the memory's earlier
    users. Memory the step did not allocate (weights, caches, step metadata) is never handed out.
    Each kernel keeps what its stream had waited for when it was issued.
    """

    STREAMS = ("main", "aux", "fork", "idx")

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {s: {} for s in self.STREAMS}
        self.homes: dict = {}  # allocation address -> (stream, point)
        self.kernels: list = []  # (point, stream, name, what it was passed and returned, waited)
        self.problems: list = []

    def tick(self) -> int:
        self.now += 1
        return self.now

    def position(self, stream: str) -> dict:
        """What a wait on ``stream`` issued now waits for: its own work so far and its waits."""
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


# Torch calls that only allocate: no kernel reads their inputs.
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
    """A stream of the plan, with the CUDA stream's ``wait_stream``."""

    def __init__(self, plan: _Plan, name: str):
        self.plan, self.name = plan, name

    def wait_stream(self, other: _Stream) -> None:
        self.plan.wait(self.name, self.plan.position(other.name))


class _Event:
    """A CUDA event of the plan: recorded on the current stream, waited for by the current one."""

    def __init__(self, plan: _Plan):
        self.plan, self.at = plan, None

    def record(self) -> None:
        self.at = self.plan.position(self.plan.stream)

    def wait(self) -> None:
        assert self.at is not None, "a wait on an event nothing recorded"
        self.plan.wait(self.plan.stream, self.at)


class _On:
    """``torch.cuda.stream(s)``: what runs inside is issued on ``s``."""

    def __init__(self, plan: _Plan, stream: _Stream):
        self.plan, self.stream, self.outer = plan, stream, None

    def __enter__(self):
        self.outer, self.plan.stream = self.plan.stream, self.stream.name

    def __exit__(self, *exc):
        self.plan.stream = self.outer


class _Torch:
    """``torch`` as the block sees it: the plan's streams under ``torch.cuda``, in a capture."""

    def __init__(self, plan: _Plan, streams: dict):
        self.cuda = SimpleNamespace(
            current_stream=lambda: streams[plan.stream],
            stream=lambda stream: _On(plan, stream),
            is_current_stream_capturing=lambda: True,
        )

    def __getattr__(self, name):
        return getattr(torch, name)


class _Logger:
    """The block's logger: keeps each ``info_once`` call and the stream it was made on."""

    def __init__(self, plan: _Plan):
        self.plan, self.calls = plan, []

    def info_once(self, *msg, key):
        self.calls.append((" ".join(map(str, msg)), key, self.plan.stream))


def _entries(plan: _Plan, layer: SimpleNamespace) -> dict:
    """The block's entries as stand-ins: each is a kernel on the current stream over what it is
    passed and returns, and returns a result of the entry's shape (no values). The indexer's
    calls are named by what they compute."""

    def like(t):
        return torch.empty(t.shape, dtype=t.dtype)

    def remap(req, table, tok, *rest):
        return torch.empty(tok.shape, dtype=torch.int32)

    def kv_rows(block_table, req_idx, stride_rows, row_offset):
        """``key_shard_kv_rows``: the remap's arguments, no kernel (a tuple, so the plan sees its
        tensors where the select / pack reads them)."""
        return (block_table, req_idx, stride_rows, row_offset)

    def cat_name(pe, nope, ue8m0):
        return "fused_cat_fp8 query" if pe.dim() == 3 else "fused_cat_fp8 key"

    def mm_name(a, b):
        return "cublas_mm idx_wq_b" if _memory(b) == _memory(layer.idx_wq_b_t) else "cublas_mm"

    def linear_name(x, w, tf32):
        return "rubin_linear tf32" if tf32 else "rubin_linear"

    results = {
        "cublas_mm": (mm_name, lambda a, b: torch.empty(a.shape[0], b.shape[1], dtype=a.dtype)),
        "rubin_linear": (
            linear_name,
            lambda x, w, tf32: torch.empty(x.shape[0], w.shape[0], dtype=x.dtype),
        ),
        "flashinfer_rmsnorm": (None, lambda x, w, eps: like(x)),
        "rubin_layer_norm": (None, lambda x, w, b, eps: like(x)),
        "rubin_upcast_fp32": (None, lambda x: torch.empty(x.shape, dtype=torch.float32)),
        "rubin_idx_weights": (
            None,
            lambda w, qs, scale: torch.empty(w.shape, dtype=torch.float32),
        ),
        "fused_cat_fp8": (
            cat_name,
            lambda pe, nope, ue8m0: (
                torch.empty(
                    *pe.shape[:-1], pe.shape[-1] + nope.shape[-1], dtype=torch.float8_e4m3fn
                ),
                torch.empty(*pe.shape[:-1], 1, dtype=torch.float32),
            ),
        ),
        "indexer_k_cache_scatter_op": (None, None),
        "bmm_out": (None, None),
        "mla_rope_generation": (None, None),
        "rubin_fp8_paged_mqa_logits": (None, lambda *args, pdl=False: args[-1]),
        "rubin_topk_decode_cute": (None, None),
        "rubin_dsa_key_shard_pack": (None, None),
        "rubin_dsa_key_shard_merge": (None, None),
        "rubin_dsa_key_shard_select_pack": (None, None),
        "convert_req_index_to_global": (None, remap),
        "rubin_dsa_topk_remap": (None, remap),
        "thop_attention": (None, lambda **kw: kw["output"]),
        "rubin_helix_zero_or_empty_rows": (
            None,
            lambda bounds, counts: torch.empty(bounds.shape, dtype=torch.bool),
        ),
        "rubin_dflash2_helix_combine": (
            None,
            lambda v, stats, mapping, heads, dim, mask, compile_: (
                torch.empty(v.shape[0], heads // _CP * dim, dtype=v.dtype)
            ),
        ),
        "k31_sigmoid_gate_mul": (None, lambda out, gate: like(out)),
    }

    def entry(name, namer, result):
        def run(*args, **kwargs):
            out = None if result is None else result(*args, **kwargs)
            plan.kernel(name if namer is None else namer(*args, **kwargs), (args, kwargs, out))
            return out

        return run

    def eager_only(name):
        def run(*args, **kwargs):
            raise AssertionError(f"{name} is the eager fallback's, not a graph step's")

        return run

    return {
        **{name: entry(name, *spec) for name, spec in results.items()},
        **{name: eager_only(name) for name in _EAGER_ONLY},
        "key_shard_kv_rows": kv_rows,
        # The GEMM pins run today's call here (test_rubin_helix8_gemm_pins.py checks them).
        "_gemm_pinned": lambda site, a, b, today, out=None, out_dtype=None: today(),
    }


def _shape_only(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    """A tensor of ``shape`` over one row of memory: the stand-ins read shapes only."""
    return torch.empty(shape[-1], dtype=dtype).expand(*shape)


def _cpu_layer() -> SimpleNamespace:
    """The weight views the block reads, at helix8's shapes."""
    return SimpleNamespace(
        qkv_a_t=_shape_only(_Q_LORA + _KV_LORA + _ROPE, _HIDDEN).t(),
        q_a_norm=_shape_only(_Q_LORA),
        kv_a_norm=_shape_only(_KV_LORA),
        q_b=_shape_only(_HEADS * (_NOPE + _ROPE), _Q_LORA),
        k_b=_shape_only(_HEADS, _NOPE, _KV_LORA),
        v_b_t=_shape_only(_HEADS, _KV_LORA, _V),
        g=_shape_only(_O_HEADS * _V, _HIDDEN),
        o=_shape_only(_HIDDEN, _O_HEADS * _V),
        idx_wq_b_t=_shape_only(_IDX_HEADS * _IDX_DIM, _Q_LORA).t(),
        idx_wk_wp=_shape_only(_IDX_DIM + _IDX_HEADS, _HIDDEN, dtype=torch.float32),
        idx_k_norm_w=_shape_only(_IDX_DIM, dtype=torch.float32),
        idx_k_norm_b=_shape_only(_IDX_DIM, dtype=torch.float32),
    )


def _cpu_step(path: str, plan: _Plan, streams: dict, cuda_graph: bool):
    """``(st, rt)`` of a decode step of ``_LIVE`` requests on ``path``: the dense Top-K (the step
    skips the indexer) or the key-sharded Top-K, the two indexer paths of a graph step; the
    key-sharded one through today's chain (no select / pack runner) or, ``key_shard_select``,
    through dsa_ks_select_pack's runner (its check at ``_LIVE`` rows passed: ``_plan``)."""
    i32 = torch.int32
    key_shard = path in ("key_shard", "key_shard_select")
    dsa = SimpleNamespace(
        next_n=1,
        skip_gen=not key_shard,
        slot_fp8=torch.empty(_ROWS, dtype=torch.int64),
        slot_scale=torch.empty(_ROWS, dtype=torch.int64),
        ks_next_n=1 if key_shard else 0,
        ks_block_table=torch.empty(_LIVE, 4, dtype=i32),
        ks_ctx_lens=torch.empty(_LIVE, dtype=i32),
        ks_sched=torch.empty(16, 2, dtype=i32),
        ks_row_lens=torch.empty(_LIVE, dtype=i32),
        ks_local_len=256,
        row_bounds=None,
        counts_out=torch.empty(_ROWS, dtype=i32) if key_shard else None,
        topk_out=torch.empty(_ROWS, _TOPK, dtype=i32),
        topk_dense=torch.empty(_ROWS, _TOPK, dtype=i32),
        gen_kv_lens=torch.empty(_LIVE, dtype=i32),
        req_idx_gen=torch.empty(_LIVE, dtype=i32),
        block_table_gen=torch.empty(_LIVE, 8, dtype=i32),
        tokens_per_block=_TOKENS_PER_BLOCK,
        rope_gen=dict(sequence_length=torch.empty(_LIVE, dtype=i32)),
        sparse_gen=dict(sequence_length=torch.empty(_LIVE, dtype=i32)),
    )
    st = SimpleNamespace(
        num_tokens=_LIVE,
        num_contexts=0,
        num_generations=_LIVE,
        cuda_graph=cuda_graph,
        dsa=dsa,
        md=SimpleNamespace(max_seq_len=4096),
        cache=SimpleNamespace(max_attention_window_vec=[None]),
    )
    rt = SimpleNamespace(
        events=((_Event(plan), _Event(plan)), (_Event(plan), _Event(plan))),
        aux_stream=streams["aux"],
        fork_stream=streams["fork"],
        idx_stream=streams["idx"],
        dsa_pools=SimpleNamespace(
            idx_k_cache={_LAYER: torch.empty(16, _TOKENS_PER_BLOCK, 1, 132, dtype=torch.uint8)},
            pool_params={_LAYER: (1, 0)},
            local={_LAYER: 0},
            rope_positions=4096,
        ),
        mla_rope=torch.empty(1, 16),
        ks_ws=SimpleNamespace(
            local_logits=torch.empty(_ROWS, 256),
            local_topk=torch.empty(_ROWS, 64, dtype=i32),
            local_k=64,
        ),
        # dsa_ks_select_pack's runner: none on the key_shard path, so today's chain runs there.
        ks_sel=SimpleNamespace(runner="select / pack") if path == "key_shard_select" else None,
        cp_group=None,
        cp_mapping=None,
    )
    return st, rt


def _plan(path: str, tree: ast.Module | None = None, cuda_graph: bool = True):
    """The plan and the logger of one ``mla_generate`` call of ``tree`` (modeling.py by default)."""
    plan = _Plan()
    streams = {s: _Stream(plan, s) for s in _Plan.STREAMS}
    layer, logger = _cpu_layer(), _Logger(plan)
    namespace = {
        "torch": _Torch(plan, streams),
        "logger": logger,
        **_entries(plan, layer),
        # The select / pack's page (the entry's, imported by modeling.py) and its check at this
        # step's row count (no bounds) passed in this process.
        "_KS_SELECT_PAGE": _TOKENS_PER_BLOCK,
        "_KS_SELECT_CHECKED": {(_LIVE, False): True},
    }
    block = _block(tree or _tree(), namespace)
    st, rt = _cpu_step(path, plan, streams, cuda_graph)
    x = torch.empty(_ROWS, _HIDDEN, dtype=torch.bfloat16)
    with _Record(plan):
        out = block["mla_generate"](layer, _LAYER, x, st, rt)
    assert tuple(out.shape) == (_ROWS, _HIDDEN)
    return plan, logger


# --- planted faults ---------------------------------------------------------------------------------


def _mla_generate(tree: ast.Module) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mla_generate"]
    return fn


def _graph_branches(fn: ast.FunctionDef) -> list:
    """``mla_generate``'s two ``if st.cuda_graph:`` statements: the fork, then the join."""
    found = [s for s in fn.body if isinstance(s, ast.If) and ast.unparse(s.test) == "st.cuda_graph"]
    assert len(found) == 2, [ast.unparse(s.test) for s in found]
    return found


def _without_the_join(tree: ast.Module) -> ast.Module:
    """``tree`` whose current stream never waits for the idx stream."""
    tree = copy.deepcopy(tree)
    join = _graph_branches(_mla_generate(tree))[1]
    kept = [s for s in join.body if ast.unparse(s) != "join.wait()"]
    assert len(kept) == len(join.body) - 1, "mla_generate has no join.wait()"
    join.body = kept
    return tree


def _forked_before_the_input(tree: ast.Module) -> ast.Module:
    """``tree`` whose idx stream forks before the layer input is written: the current stream
    writes a new ``x`` after the fork's record."""
    tree = copy.deepcopy(tree)
    fork = _graph_branches(_mla_generate(tree))[0]
    texts = [ast.unparse(s) for s in fork.body]
    at = texts.index("fork.record()")
    fork.body.insert(at + 1, ast.parse("x = x + 0").body[0])
    return tree


# --- the tests --------------------------------------------------------------------------------------


def _on(plan: _Plan, names) -> dict:
    """Each named kernel's streams, in issue order."""
    found: dict = {}
    for _, stream, name, _, _ in plan.kernels:
        if name in names:
            found.setdefault(name, []).append(stream)
    return found


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", ["dense", "key_shard", "key_shard_select"])
def test_a_graph_step_runs_the_indexer_keys_on_the_idx_stream(path):
    plan, _ = _plan(path)
    assert not plan.problems, "\n".join(plan.problems)
    assert _on(plan, _KEY_KERNELS) == {k: ["idx"] for k in _KEY_KERNELS}
    assert _on(plan, _QUERY_KERNELS) == {k: ["main"] for k in _QUERY_KERNELS}
    idx = [(at, name) for at, s, name, _, _ in plan.kernels if s == "idx"]
    assert [name for _, name in idx] == list(_KEY_KERNELS), idx
    # The idx stream waits for the current stream once, before the block's first kernel there.
    first_main = min(at for at, s, _, _, _ in plan.kernels if s == "main")
    assert set(plan.clock["idx"]) == {"main"}, plan.clock["idx"]
    assert plan.clock["idx"]["main"] < first_main
    # The current stream joins the idx stream after its last kernel.
    assert plan.clock["main"].get("idx", 0) > idx[-1][0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", ["key_shard", "key_shard_select"])
def test_the_logits_read_the_index_keys_after_the_insert(path):
    plan, _ = _plan(path)
    (insert,) = [at for at, _, name, _, _ in plan.kernels if name == "indexer_k_cache_scatter_op"]
    (logits,) = [
        (s, waited)
        for _, s, name, _, waited in plan.kernels
        if name == "rubin_fp8_paged_mqa_logits"
    ]
    assert logits[0] == "main" and logits[1].get("idx", 0) > insert, logits


@pytest.mark.cpu_only
def test_the_check_flags_a_block_without_the_join():
    plan, _ = _plan("key_shard", _without_the_join(_tree()))
    flagged = [p for p in plan.problems if "the idx stream allocated" in p]
    assert flagged, plan.problems


@pytest.mark.cpu_only
def test_the_check_flags_a_fork_before_the_input_is_written():
    plan, _ = _plan("dense", _forked_before_the_input(_tree()))
    flagged = [p for p in plan.problems if p.startswith("rubin_upcast_fp32 on the idx stream")]
    assert flagged, plan.problems


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", ["dense", "key_shard", "key_shard_select"])
def test_a_graph_step_issues_the_kernels_of_an_eager_step(path):
    graph, _ = _plan(path)
    eager, _ = _plan(path, cuda_graph=False)
    assert {s for _, s, _, _, _ in eager.kernels} == {"main"}
    assert not any(eager.clock.values()) and not eager.problems
    names = [Counter(name for _, _, name, _, _ in p.kernels) for p in (graph, eager)]
    assert names[0] == names[1]


# Today's key-sharded Top-K after the logits: the local Top-K, the pack, the merge and the remap.
_TODAYS_KEY_SHARD = (
    "rubin_topk_decode_cute",
    "rubin_dsa_key_shard_pack",
    "rubin_dsa_key_shard_merge",
    "convert_req_index_to_global",
)


def _call(plan: _Plan, name: str) -> tuple:
    """The one kernel ``name`` of the plan: ``(point, stream, args, kwargs, waited)``."""
    (found,) = [(at, s, p, w) for at, s, n, p, w in plan.kernels if n == name]
    at, stream, (args, kwargs, _), waited = found
    return at, stream, args, kwargs, waited


@pytest.mark.cpu_only
@pytest.mark.parametrize("cuda_graph", [True, False], ids=["graph", "eager"])
def test_the_select_pack_hands_the_attention_kv_pool_rows(cuda_graph):
    plan, logger = _plan("key_shard_select", cuda_graph=cuda_graph)
    today, _ = _plan("key_shard", cuda_graph=cuda_graph)
    assert not plan.problems, "\n".join(plan.problems)
    assert not set(_TODAYS_KEY_SHARD) & {name for _, _, name, _, _ in plan.kernels}
    assert set(_TODAYS_KEY_SHARD) <= {name for _, _, name, _, _ in today.kernels}
    # One launch on the current stream, after the logits (and, on a graph step, after the join of
    # the idx stream that inserted the index keys).
    logits_at = _call(plan, "rubin_fp8_paged_mqa_logits")[0]
    insert_at = _call(plan, "indexer_k_cache_scatter_op")[0]
    at, stream, args, _, waited = _call(plan, "rubin_dsa_key_shard_select_pack")
    assert stream == "main" and at > logits_at
    if cuda_graph:
        assert waited.get("idx", 0) > insert_at
    sel, _, n, topk, counts, bounds, mask, kv = args
    assert sel.runner == "select / pack" and n == _LIVE and bounds is None and mask is None
    # Layer 3's pool parameters (scale 1, offset 0), in rows of a page.
    assert kv[2:] == (_TOKENS_PER_BLOCK, 0)
    # The sparse attention takes the select / pack's KV pool rows where today's step hands it the
    # remap's: the same dtype and shape, here the step's Top-K rows themselves.
    rows = _call(plan, "thop_attention")[3]["sparse_attn_indices"]
    todays = _call(today, "thop_attention")[3]["sparse_attn_indices"]
    assert rows.data_ptr() == topk.data_ptr() and rows.shape == topk.shape
    assert (rows.dtype, tuple(rows.shape)) == (todays.dtype, tuple(todays.shape))
    assert (rows.dtype, tuple(rows.shape)) == (torch.int32, (_LIVE, _TOPK))
    # The helix mask reads the counts the select / pack wrote, after it, on the same stream.
    mask_at, mask_stream, mask_args, _, _ = _call(plan, "rubin_helix_zero_or_empty_rows")
    assert mask_stream == "main" and mask_at > at and mask_args[1] is counts
    line = _constants()["_KS_SELECT_LINE"]
    assert [c for c in logger.calls if c[1] == line] == [(line, line, "main")]


@pytest.mark.cpu_only
def test_a_graph_step_logs_the_fork_and_the_constants_state_it():
    line = _constants()["_IDX_FORK_LINE"]
    assert line.startswith("Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream")
    _, graph = _plan("key_shard")
    _, eager = _plan("key_shard", cuda_graph=False)
    # This fork's line only: the indexer projection logs its own (README).
    fork = [[c for c in p.calls if c[1] == line] for p in (graph, eager)]
    assert fork == [[(line, line, "main")], []], fork
    assert _constants()["_CONSTANTS"]["generation"]["dsa_idx_fork"] is True

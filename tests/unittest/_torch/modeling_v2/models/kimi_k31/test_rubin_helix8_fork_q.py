# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream dependencies and CUDA graph buffer lifetimes in the specialized Helix8 target.

The CPU checks run on each indexer path of a graph step: the dense Top-K, the key-sharded one
with today's chain, and helix8's key-sharded one through ``dsa_ks_select_pack``'s select / pack
at a checked row count.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MODELING = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31" / _TARGET / "modeling.py"

# K3.1's MLA and indexer at helix8's share: all 96 query heads attend, o_proj takes 12 of them.
_HIDDEN, _Q_LORA, _KV_LORA, _ROPE, _NOPE, _V = 7168, 1536, 512, 64, 128, 128
_HEADS, _O_HEADS, _IDX_HEADS, _IDX_DIM = 96, 12, 32, 128
_LATENT = _KV_LORA + _ROPE
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
    "_mla_absorb",
    "_mla_expand",
    "_mla_window",
    "_mla_helix_mask",
    "_mla_query",
    "_mla_generation",
    "_mla_output",
)
# helix8's indexer work on the layer input alone and its query projection (dsa_idx_fork); a tree
# without them (the default target's copy of the block) runs its one projection in their place.
_IDX_SPLIT, _IDX_ONE = ("_idx_keys", "_idx_queries"), ("_idx_project",)
# helix8's dsa_ks_select_pack (the default target has none).
_KS_SELECT = ("_ks_select", "_ks_kv_rows")
# The fallback indexer's entries: an eager step's, never a graph step's.
_EAGER_ONLY = (
    "fp8_paged_mqa_logits",
    "rubin_fp4_paged_mqa_logits_deepgemm",
    "indexer_topk_decode",
    "rubin_gvr_topk_decode",
    "allgather_module",
    "rubin_merge_helix_topk",
)
# The RoPE kernel's outputs the attention reads.
_ROPE_BUFFERS = (
    "cu_q_seqlens",
    "cu_kv_seqlens",
    "fmha_scheduler_counter",
    "mla_bmm1_scale",
    "mla_bmm2_scale",
    "quant_q_buffer",
)


# --- CPU: the block's streams, from source ---


@functools.lru_cache(maxsize=None)
def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


@functools.lru_cache(maxsize=None)
def _constants(tree: ast.Module | None = None) -> MappingProxyType:
    """modeling.py's (or ``tree``'s) module-level constants that evaluate on their own, in source
    order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in (tree or _tree()).body:
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
    names = set(_BLOCK)
    if not any(isinstance(n, ast.FunctionDef) and n.name in _IDX_SPLIT for n in tree.body):
        names = names - set(_IDX_SPLIT) | set(_IDX_ONE)
    if any(isinstance(n, ast.FunctionDef) and n.name in _KS_SELECT for n in tree.body):
        names |= set(_KS_SELECT)
    body = [
        copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    missing = names - {n.name for n in body}
    assert not missing, f"helix8's modeling.py defines no {sorted(missing)}"
    ns = {**_constants(None if tree is _tree() else tree), **namespace}
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
    """One step's host order on three streams: its allocations, waits and kernels.

    ``clock[s][o]`` is the latest point of stream ``o``'s order that stream ``s`` waited for
    (waits and events carry it on). Memory allocated on stream ``o`` at point ``a`` is memory the
    caching allocator may have freed in ``o``'s order, so a kernel of another stream ``s`` may
    touch it only with ``clock[s][o] > a``: nothing else orders it after the memory's earlier
    users. Memory the step did not allocate (weights, caches, step metadata) is never handed out.
    """

    STREAMS = ("main", "aux", "fork", "idx")

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {s: {} for s in self.STREAMS}
        self.homes: dict = {}  # allocation address -> (stream, point)
        self.kernels: list = []  # (point, stream, name, what it was passed and returned)
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
        self.kernels.append((at, stream, name, payload))


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


def _entries(plan: _Plan) -> dict:
    """The block's entries as stand-ins: each is a kernel on the current stream over what it is
    passed and returns, and returns a result of the entry's shape (no values)."""

    def like(t):
        return torch.empty(t.shape, dtype=t.dtype)

    def remap(req, table, tok, *rest):
        return torch.empty(tok.shape, dtype=torch.int32)

    def fp4_codes(a, b):
        """The index-key FP4 quantization: codes and the exponent words of each half pair."""
        words = (*a.shape[:-1], 1) if a.dim() == 2 else a.shape[:-1]
        return torch.empty(a.shape, dtype=torch.uint8), torch.empty(words, dtype=torch.int32)

    results = {
        "cublas_mm": lambda a, b: torch.empty(a.shape[0], b.shape[1], dtype=a.dtype),
        "rubin_linear": lambda x, w, tf32: torch.empty(x.shape[0], w.shape[0], dtype=x.dtype),
        "flashinfer_rmsnorm": lambda x, w, eps: like(x),
        "rubin_layer_norm": lambda x, w, b, eps: like(x),
        "rubin_upcast_fp32": lambda x: torch.empty(x.shape, dtype=torch.float32),
        "rubin_idx_weights": lambda w, qs, scale: torch.empty(w.shape, dtype=torch.float32),
        "fused_cat_fp8": lambda pe, nope, ue8m0: (
            torch.empty(*pe.shape[:-1], pe.shape[-1] + nope.shape[-1], dtype=torch.float8_e4m3fn),
            torch.empty(*pe.shape[:-1], 1, dtype=torch.float32),
        ),
        "indexer_k_cache_scatter_op": None,
        "bmm_out": None,
        "mla_rope_generation": None,
        "rubin_fp8_paged_mqa_logits": lambda *args, pdl=False: args[-1],
        # The default target's FP4 index-key variants of the block.
        "rubin_idx_fp4_quantize": fp4_codes,
        "rubin_idx_k_cache_scatter_fp4": None,
        "rubin_fp4_paged_mqa_logits": lambda *args: args[-1],
        "rubin_topk_decode_cute": None,
        "rubin_dsa_key_shard_pack": None,
        "rubin_dsa_key_shard_merge": None,
        "rubin_dsa_key_shard_select_pack": None,
        "convert_req_index_to_global": remap,
        "rubin_dsa_topk_remap": remap,
        "thop_attention": lambda **kw: kw["output"],
        "rubin_helix_zero_or_empty_rows": lambda bounds, counts: torch.empty(
            bounds.shape, dtype=torch.bool
        ),
        "rubin_dflash2_helix_combine": lambda v, stats, mapping, heads, dim, mask, compile_: (
            torch.empty(v.shape[0], heads // _CP * dim, dtype=v.dtype)
        ),
        "k31_sigmoid_gate_mul": lambda out, gate: like(out),
    }

    def entry(name, result):
        def run(*args, **kwargs):
            out = None if result is None else result(*args, **kwargs)
            plan.kernel(name, (args, kwargs, out))
            return out

        return run

    def eager_only(name):
        def run(*args, **kwargs):
            raise AssertionError(f"{name} is the eager fallback's, not a graph step's")

        return run

    return {
        **{name: entry(name, result) for name, result in results.items()},
        **{name: eager_only(name) for name in _EAGER_ONLY},
        # key_shard_kv_rows: the remap's arguments, no kernel (a tuple, so the plan sees its
        # tensors where the select / pack reads them).
        "key_shard_kv_rows": lambda *args: args,
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


def _plan(path: str, tree: ast.Module | None = None, cuda_graph: bool = True) -> _Plan:
    """The plan of one ``mla_generate`` call of ``tree`` (modeling.py by default)."""
    plan = _Plan()
    streams = {s: _Stream(plan, s) for s in _Plan.STREAMS}
    logger = SimpleNamespace(info_once=lambda *msg, key: None)
    namespace = {
        "torch": _Torch(plan, streams),
        "logger": logger,
        **_entries(plan),
        # The select / pack's page (the entry's, imported by modeling.py) and its check at this
        # step's row count (no bounds) passed in this process.
        "_KS_SELECT_PAGE": _TOKENS_PER_BLOCK,
        "_KS_SELECT_CHECKED": {(_LIVE, False): True},
    }
    block = _block(tree or _tree(), namespace)
    st, rt = _cpu_step(path, plan, streams, cuda_graph)
    layer, x = _cpu_layer(), torch.empty(_ROWS, _HIDDEN, dtype=torch.bfloat16)
    with _Record(plan):
        out = block["mla_generate"](layer, _LAYER, x, st, rt)
    assert tuple(out.shape) == (_ROWS, _HIDDEN)
    return plan


def _buffers_after_the_indexer(tree: ast.Module) -> ast.Module:
    """``tree`` with ``mla_generate`` allocating the RoPE buffers right before the attention call
    that takes them, after the fork and the indexer: the order this test guards against."""
    tree = copy.deepcopy(tree)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mla_generate"]

    def assigned(stmt) -> set:
        if not isinstance(stmt, ast.Assign):
            return set()
        return {t.id for t in stmt.targets if isinstance(t, ast.Name)}

    moved = [s for s in fn.body if assigned(s) & {"seqs", "bufs"}]
    assert [assigned(s) for s in moved] == [{"seqs"}, {"bufs"}], "mla_generate allocates no bufs"
    rest = [s for s in fn.body if all(s is not m for m in moved)]
    (at,) = [k for k, s in enumerate(rest) if "_mla_generation(" in ast.unparse(s)]
    fn.body = rest[:at] + moved + rest[at:]
    return tree


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", ["dense", "key_shard", "key_shard_select"])
def test_a_graph_step_touches_memory_after_its_earlier_users(path):
    """helix8's MLA block, run from modeling.py over recording stand-ins (the entries, the streams
    and events, the allocations) on a graph step with CUDA graph padding rows, on each indexer path
    (the dense Top-K, the key-sharded one, and the key-sharded one through the select / pack):
    every kernel touches memory allocated before the step, on its own stream, or on another stream
    before its own stream's last wait on that one. The RoPE kernel runs on the fork stream over
    buffers the current stream allocated, and the current stream joins every side stream before
    the block returns."""
    plan = _plan(path)
    _check_graph_step(plan, ("aux", "fork", "idx"))
    selects = [s for _, s, name, _ in plan.kernels if name == "rubin_dsa_key_shard_select_pack"]
    assert selects == (["main"] if path == "key_shard_select" else [])


def _check_graph_step(plan: _Plan, sides: tuple = ("aux", "fork")) -> None:
    assert not plan.problems, "\n".join(plan.problems)
    rope = [(s, payload) for _, s, name, payload in plan.kernels if name == "mla_rope_generation"]
    assert len(rope) == 1, rope
    stream, (_, kwargs, _) = rope[0]
    assert stream == "fork", "the RoPE kernel does not run on the fork stream"
    homes = {name: plan.homes[_memory(kwargs[name])][0] for name in _ROPE_BUFFERS}
    assert homes == dict.fromkeys(_ROPE_BUFFERS, "main"), homes
    last = {s: at for at, s, _, _ in plan.kernels}
    for side in sides:
        assert side in last, f"nothing runs on the {side} stream"
        assert plan.clock["main"].get(side, 0) > last[side], f"the {side} stream is not joined"


@pytest.mark.cpu_only
def test_the_check_flags_rope_buffers_allocated_after_the_indexer():
    """The same block with the RoPE buffers allocated after the indexer is flagged."""
    plan = _plan("dense", _buffers_after_the_indexer(_tree()))
    flagged = [p for p in plan.problems if p.startswith("mla_rope_generation on the fork stream")]
    assert flagged, plan.problems


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", ["dense", "key_shard", "key_shard_select"])
def test_an_eager_step_runs_on_the_current_stream_alone(path):
    plan = _plan(path, cuda_graph=False)
    assert {s for _, s, _, _ in plan.kernels} == {"main"}
    assert not any(plan.clock.values()) and not plan.problems


# --- CPU: the default target's FP4 index-key variant of the block ---
# Its FP8 block is this target's code; routing refuses an NVFP4 cache on helix8 generation.


# --- one sm_107 GPU: capture, replay on new inputs, compare with the in-order step ---

_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="helix8's kernels run on sm_107",
)
# The requests' KV lengths, the new token included, around page boundaries.
_KV_LENS = (40, 64, 65, 128, 129, 250)
_MAX_SEQ = 512
_PAGES = _MAX_SEQ // _TOKENS_PER_BLOCK
_BLOCKS = len(_KV_LENS) * _PAGES + 2
# A delay's spin cycles: milliseconds, far longer than either stream's work after the fork.
_SPIN = 4_000_000


def _module():
    return importlib.import_module(f"{_PKG}.{_TARGET}.modeling")


def _gpu_layer(mod, gen: torch.Generator):
    """A DSA layer's weight views at helix8's share, seeded (the views ``_layer_views`` takes)."""

    def rnd(rows, cols, dtype=torch.bfloat16):
        return (torch.randn(rows, cols, generator=gen, device="cuda") * cols**-0.5).to(dtype)

    def norm(size, dtype=torch.bfloat16):
        return (1 + 0.1 * torch.randn(size, generator=gen, device="cuda")).to(dtype)

    qkv_a = rnd(_Q_LORA + _KV_LORA + _ROPE, _HIDDEN)
    q_b = rnd(_HEADS * (_NOPE + _ROPE), _Q_LORA)
    kv_b = rnd(_HEADS * (_NOPE + _V), _KV_LORA)  # every head's k_nope rows, then every head's v
    g, o = rnd(_O_HEADS * _V, _HIDDEN), rnd(_HIDDEN, _O_HEADS * _V)
    idx_wq_b = rnd(_IDX_HEADS * _IDX_DIM, _Q_LORA)
    return mod._MlaW(
        q_a_norm=norm(_Q_LORA),
        kv_a_norm=norm(_KV_LORA),
        q_b=q_b,
        g=g,
        o=o,
        idx_wk_wp=rnd(_IDX_DIM + _IDX_HEADS, _HIDDEN, torch.float32),
        idx_k_norm_w=norm(_IDX_DIM, torch.float32),
        idx_k_norm_b=0.1 * torch.randn(_IDX_DIM, generator=gen, device="cuda"),
        k_b=kv_b[: _HEADS * _NOPE].view(_HEADS, _NOPE, _KV_LORA),
        v_b_t=kv_b[_HEADS * _NOPE :].view(_HEADS, _V, _KV_LORA).transpose(1, 2),
        qkv_a_t=qkv_a.t(),
        idx_wq_b_t=idx_wq_b.t(),
    )


class _Metadata(SimpleNamespace):
    """What helix8's step projection reads of the family DSA metadata, for one rank's decode step:
    no context rows, no drafts, a dense Top-K (the step skips the indexer)."""

    def helix_spec_row_kv_lens(self, rows):
        return None  # no verify block

    def helix_sparse_row_lens_out(self, rows):
        return None  # no kept counts: the dense Top-K

    def get_indexer_max_seq_len(self):
        return _MAX_SEQ

    def get_empty(self, buffers, shape, cache_name, dtype, capture_graph):
        return torch.zeros(shape, dtype=dtype, device="cuda")

    def token_major_gen_view(self, token_sparse):
        return None  # one row per request already


class _Rig:
    """One rank's DSA layer, its caches and a decode step's state, all in fixed tensors: a
    captured step reads and writes the same storage at replay. The step state is the target's
    own projection (``_dsa_step_args``) of the metadata."""

    def __init__(self, mod):
        self.mod = mod
        dev = "cuda"
        self.layer = _gpu_layer(mod, torch.Generator(device=dev).manual_seed(0))
        ng = len(_KV_LENS)
        order = torch.randperm(_BLOCKS, generator=torch.Generator().manual_seed(1)).tolist()
        pages = [[order.pop() for _ in range(-(-kv // _TOKENS_PER_BLOCK))] for kv in _KV_LENS]
        table = torch.zeros(ng, _PAGES, dtype=torch.int32)
        for r, row in enumerate(pages):
            table[r, : len(row)] = torch.tensor(row, dtype=torch.int32)
        offsets = torch.zeros(1, _ROWS, 2, _PAGES, dtype=torch.int32)
        offsets[0, :ng, 0] = table
        offsets[0, :ng, 1] = table
        # The FP8 latent pool of the layer, and its index-key cache: a page holds 64 keys of 128
        # bytes, then their 64 FP32 scales.
        self.pool = torch.zeros(
            _BLOCKS, _TOKENS_PER_BLOCK, _LATENT, dtype=torch.float8_e4m3fn, device=dev
        )
        self.idx_cache = torch.zeros(
            _BLOCKS, _TOKENS_PER_BLOCK, 1, _IDX_DIM + 4, dtype=torch.uint8, device=dev
        )
        self.x = torch.zeros(_ROWS, _HIDDEN, dtype=torch.bfloat16, device=dev)
        kv = torch.tensor(_KV_LENS, dtype=torch.int32)
        pos = kv - 1  # the new token's position
        page = torch.tensor([row[p // _TOKENS_PER_BLOCK] for row, p in zip(pages, pos.tolist())])
        slot = pos.long() % _TOKENS_PER_BLOCK
        page_bytes = _TOKENS_PER_BLOCK * (_IDX_DIM + 4)
        slot_fp8 = torch.zeros(_ROWS, dtype=torch.int64)
        slot_scale = torch.zeros(_ROWS, dtype=torch.int64)
        slot_fp8[:ng] = page * page_bytes + slot * _IDX_DIM
        slot_scale[:ng] = page * page_bytes + _TOKENS_PER_BLOCK * _IDX_DIM + slot * 4
        topk = torch.full((_ROWS, _TOPK), -1, dtype=torch.int32)
        for r, n in enumerate(_KV_LENS):
            topk[r, :n] = torch.arange(n, dtype=torch.int32)
        positions = torch.zeros(_ROWS, dtype=torch.int32)
        positions[:ng] = pos
        req_idx = torch.zeros(_ROWS, dtype=torch.int32)
        req_idx[:ng] = torch.arange(ng, dtype=torch.int32)
        cache = SimpleNamespace(
            kv_cache_pool_pointers=torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64),
            kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
            tokens_per_block=_TOKENS_PER_BLOCK,
            max_attention_window_vec=[None],
        )
        md = _Metadata(
            is_ragged_verify=False,
            skip_indexer_for_gen_reqs=True,
            key_shard_next_n=0,
            key_shard_world=1,
            is_cuda_graph=True,
            cuda_graph_buffers=None,
            kv_cache_manager=cache,
            kv_lens_cuda_runtime=kv.to(dev),
            kv_lens_runtime=kv.clone(),
            gen_indexer_kv_lens_cuda_runtime=kv.to(dev),
            prompt_lens_cuda_runtime=pos.to(dev),
            prompt_lens_cpu_runtime=pos.clone(),
            host_total_kv_lens=torch.tensor([0, int(kv.sum())], dtype=torch.int32),
            host_request_types_runtime=torch.ones(ng, dtype=torch.int32),
            max_context_q_len_override=None,
            kv_cache_block_offsets=offsets.to(dev),
            host_kv_cache_pool_pointers=cache.kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=cache.kv_cache_pool_mapping,
            cache_indirection=None,
            block_ids_per_seq=None,
            tokens_per_block=_TOKENS_PER_BLOCK,
            max_num_requests=_ROWS,
            max_num_sequences=_ROWS,
            beam_width=1,
            effective_beam_width=1,
            use_paged_context_fmha=False,
            helix_position_offsets=positions.to(dev),
            helix_is_inactive_rank=torch.zeros(_ROWS, dtype=torch.bool, device=dev),
            _helix_spec_tokens_valid=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets_for_cpp=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            max_total_draft_tokens=None,
            force_prepare_spec_dec_tree_mask=False,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            num_sparse_topk=_TOPK,
            flash_mla_tile_scheduler_metadata=None,
            flash_mla_num_splits=None,
            max_context_length=_MAX_SEQ,
            max_seq_len=_MAX_SEQ,
            trtllm_gen_jit_warmup=False,
            is_cross=False,
            effective_workspace=torch.empty(64 << 20, dtype=torch.int8, device=dev),
            indexer_k_cache_block_offsets=table.to(dev),
            scheduler_metadata_buffer=torch.zeros(1, dtype=torch.int32, device=dev),
            topk_indices_buffer=topk.to(dev),
            radix_aux_indices=None,
            radix_aux_logits=None,
            req_idx_per_token=req_idx.to(dev),
            block_table=table.to(dev),
            slot_mapping_fp8=slot_fp8.to(dev),
            slot_mapping_scale=slot_scale.to(dev),
        )
        pools = SimpleNamespace(
            idx_k_cache={_LAYER: self.idx_cache},
            pool_params={_LAYER: (1, 0)},
            local={_LAYER: 0},
            rope_positions=2 * _MAX_SEQ,
            tokens_per_block=_TOKENS_PER_BLOCK,
        )
        self.rt = SimpleNamespace(
            dsa_pools=pools,
            mla_rope=mod._identity_rope_table(pools.rope_positions, _ROPE, dev),
            aux_stream=torch.cuda.Stream(),
            fork_stream=torch.cuda.Stream(),
            idx_stream=torch.cuda.Stream(),
            events=tuple((torch.cuda.Event(), torch.cuda.Event()) for _ in range(2)),
            ks_ws=None,
            cp_group=None,
            cp_mapping=None,
        )
        dsa = mod._dsa_step_args(md, pools, ng, ng)
        self.steps = {
            graph: mod._Step(
                md=md,
                kda_md=None,
                cache=cache,
                num_tokens=ng,
                num_contexts=0,
                num_ctx_tokens=0,
                num_generations=ng,
                rows_per_gen=1,
                spec_md=None,
                cuda_graph=graph,
                dsa=dsa,
            )
            for graph in (False, True)
        }

    def run(self, cuda_graph: bool) -> torch.Tensor:
        return self.mod.mla_generate(self.layer, _LAYER, self.x, self.steps[cuda_graph], self.rt)

    def inputs(self, seed: int) -> dict:
        """A step's data: the layer input and the caches' contents (the KV history)."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        pool = 0.5 * torch.randn(self.pool.shape, generator=gen, device="cuda")
        return dict(
            x=torch.randn(self.x.shape, generator=gen, device="cuda").to(torch.bfloat16),
            pool=pool.to(torch.float8_e4m3fn),
            idx_cache=torch.randint(
                0, 256, self.idx_cache.shape, generator=gen, device="cuda", dtype=torch.uint8
            ),
        )

    def load(self, inputs: dict) -> None:
        self.x.copy_(inputs["x"])
        self.pool.copy_(inputs["pool"])
        self.idx_cache.copy_(inputs["idx_cache"])


def _keep(kept: dict, name: str, t: torch.Tensor) -> None:
    """Copy ``t``'s bytes into a fixed tensor (made by the first, eager call)."""
    raw = t.view(torch.uint8)
    if name not in kept:
        kept[name] = torch.empty_like(raw)
    kept[name].copy_(raw)


def _one_rank_combine(partial_v, stats, mapping, heads, value_dim, zero_kv_mask, compile_=False):
    """The helix combine on one rank, which holds every key: its partial is the whole attention.
    It keeps rank 0's share of the heads, as the exchange leaves each rank its own."""
    rows, share = partial_v.shape[0], heads // _CP
    out = partial_v[:, :share].reshape(rows, share * value_dim)
    return out.masked_fill(zero_kv_mask.view(rows, 1), 0)


def _spy_on_the_attention(mod, kept: dict, monkeypatch) -> None:
    """Keep what the attention reads (the RoPE kernel's outputs among it) and what it writes."""
    inner = mod.thop_attention

    def attention(**kw):
        for name in ("q", "sparse_attn_indices", *_ROPE_BUFFERS):
            _keep(kept, f"reads {name}", kw[name])
        out = inner(**kw)
        _keep(kept, "writes output", out)
        _keep(kept, "writes softmax_stats_tensor", kw["softmax_stats_tensor"])
        return out

    monkeypatch.setattr(mod, "thop_attention", attention)


def _spy_on_the_indexer(mod, kept: dict, monkeypatch) -> None:
    """Keep the indexer's FP8 queries and logits weights, as the Top-K takes them."""
    inner = mod._idx_decode

    def idx_decode(i, q8, w, st, rt):
        _keep(kept, "indexer q8", q8)
        _keep(kept, "indexer weights", w)
        return inner(i, q8, w, st, rt)

    monkeypatch.setattr(mod, "_idx_decode", idx_decode)


def _delay(mod, rig: _Rig, where: str, monkeypatch) -> None:
    """A spin kernel right after a fork: on the current stream before the indexer's query
    projection (``main``), on the fork stream before q_b (``fork``), or on the idx stream before
    the indexer's key projection (``idx``)."""
    if where == "main":
        queries = mod._idx_queries

        def idx_queries(*args):
            torch.cuda._sleep(_SPIN)
            return queries(*args)

        monkeypatch.setattr(mod, "_idx_queries", idx_queries)
    elif where == "fork":
        linear = mod.rubin_linear

        def rubin_linear(*args):
            if torch.cuda.current_stream() == rig.rt.fork_stream:
                torch.cuda._sleep(_SPIN)
            return linear(*args)

        monkeypatch.setattr(mod, "rubin_linear", rubin_linear)
    elif where == "idx":
        keys = mod._idx_keys

        def idx_keys(*args):
            if torch.cuda.current_stream() == rig.rt.idx_stream:
                torch.cuda._sleep(_SPIN)
            return keys(*args)

        monkeypatch.setattr(mod, "_idx_keys", idx_keys)


def _snapshot(rig: _Rig, out: torch.Tensor, kept: dict) -> dict:
    snap = {"output": out.view(torch.uint8), "latent pool": rig.pool.view(torch.uint8)}
    snap["index-key cache"] = rig.idx_cache
    snap.update(kept)
    return {name: t.clone() for name, t in snap.items()}


@_ON_SM107
@pytest.mark.parametrize("delay", ["none", "main", "fork", "idx"])
def test_replay_on_new_inputs_equals_the_in_order_step(delay, monkeypatch):
    """``mla_generate`` of one DSA layer at helix8's shapes, every kernel real, on one rank's dense
    Top-K decode step (the helix combine is its one-rank identity). Captured in a CUDA graph on one
    set of inputs and replayed on two others, it equals the in-order step on those inputs bit for
    bit: the output, what the attention reads and writes, the indexer's FP8 queries and logits
    weights, the latent and index-key caches. A spin kernel right after a fork, on the current
    stream, at the head of the fork stream or at the head of the idx stream (dsa_idx_fork), puts
    the other streams' work first, so a missing edge between them corrupts every run instead of
    some."""
    mod = _module()
    rig = _Rig(mod)
    kept: dict = {}
    monkeypatch.setattr(mod, "rubin_dflash2_helix_combine", _one_rank_combine)
    _spy_on_the_attention(mod, kept, monkeypatch)
    _spy_on_the_indexer(mod, kept, monkeypatch)
    _delay(mod, rig, delay, monkeypatch)
    a, b, c = (rig.inputs(seed) for seed in (1, 2, 3))
    # Both plans once (kernel builds, the attention's workspace), then the capture on A.
    for cuda_graph in (False, True):
        rig.load(a)
        rig.run(cuda_graph)
    rig.load(a)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rig.run(True)
    torch.cuda.synchronize()
    for inputs in (b, c):
        rig.load(inputs)
        graph.replay()
        replayed = _snapshot(rig, out, kept)
        rig.load(inputs)
        in_order = rig.run(False)
        want = _snapshot(rig, in_order, kept)
        assert torch.isfinite(in_order.float()).all() and in_order.abs().sum() > 0
        differ = [name for name in want if not torch.equal(replayed[name], want[name])]
        assert not differ, f"the replayed step differs from the in-order step in {differ}"

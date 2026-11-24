# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's MoE front on a generation server's CUDA graph steps: latent_down and the bucket
quantization on the latent stream, beside the FP32 router and the routing.

On a graph step ``_moe_dep_gen_streams`` widens the router's input to FP32 (``_moe_router_input``:
``rubin_upcast_fp32`` on a contiguous input, ``x.float()``'s bits), then forks
``latent_down`` onto ``moe_comm.latent_stream``; on a one-chunk step of the bucket path the NVFP4
quantization into the chunk's MegaMoE bucket (``rubin_megamoe_quantize``) follows it there, and
``latent_ready`` is recorded after it. The current stream runs the router GEMM, the shared
experts' fork and the routing into the same bucket (``rubin_route_topk_out``), then waits for
``latent_ready`` right before MegaMoE (``rubin_megamoe_in_bucket``). A step of several chunks, a
row count of ``_MOE_ROUTE_SKIP_M`` and a rank without rows run the chunk code after that wait.

CPU (``cpu_only``): dep16's MoE block runs from its modeling.py over recording stand-ins (the
entries, the streams and events, the allocations) whose values follow their inputs, on one rank of
a step:

* each case's plan: the latent stream runs latent_down (and, on the bucket path, the
  quantization), then records ``latent_ready``; MegaMoE runs on the current stream after the wait
  on it;
* the latent stream forks after the router's cast and before the router GEMM (source order); in
  ``moe_widen_after_fork``'s rows (a generation constant, on at 48 rows) the cast follows the fork,
  still before the router GEMM, and bounds patched to every row count or to none change nothing
  else in the plan, the bucket or the outputs;
* both kinds of step widen the router's input with the entry, once, and hand the router GEMM its
  bits; a non-contiguous input keeps ``x.float()`` and logs nothing; an input the mix already
  widened (``xf``, the router_f32 lever) is the router's input and no widening runs;
* MegaMoE reads the bucket after both of its writers, which write disjoint rows of it;
* every kernel touches memory allocated before the step, on its own stream, or on another stream
  after its own stream's last wait on that one, and every side stream is joined at the end;
* the same block with ``latent_ready`` recorded before the quantization, or with the wait after
  MegaMoE's launch, is flagged;
* ``megamoe_quantize_pdl`` (a generation constant, on at 96 rows): a graph step of the bucket path
  whose rows lie in its bounds launches the quantization programmatically, any other step (other
  rows, the staging and chunk paths, an eager step) in stream order; bounds patched to every row
  count or to none change nothing else in the plan, the bucket or the outputs;
* a graph step returns, and leaves in the bucket, what the eager step does;
* an eager step runs on the current stream alone, in the order it had before;
* a graph step logs two INFO lines through ``logger.info_once`` (the router's widening, the
  latent stream); an eager step only the first;
* the runtime holds the latent stream and its ``latent_ready`` event;
* ``moe_router_tf32_splitk`` (a generation constant, on at 96 rows, where the runtime's check
  passed): a graph step of the bucket path whose rows lie in its bounds runs the router GEMM as
  the 8 K-slice partials of its logits and routes the bucket on their sum in slice order; any
  other step (other rows, the staging and chunk paths, an eager step, a runtime whose check
  failed) runs ``rubin_linear``; bounds patched to every row count or to none change nothing
  but that kernel, and nothing in the bucket or the outputs.
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
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_ROUTE = _FAMILY / "ops/moe/rubin_route_topk_out.py"

# Stand-in widths (the plans do not depend on them): hidden, latent, router width; K3.1's top-k
# and dep16's EP ranks.
_HIDDEN, _LATENT, _EXPERTS, _TOP_K, _RANKS = 64, 32, 48, 16, 16
_SF_VEC = 16
_LAYER = 5
# MegaMoE's bucket ladder: rows per rank of a launch.
_LADDER = (4, 32, 64, 96, 128)
# The INFO line a dep16 generation rank logs once, at its first CUDA graph step.
_LINE = (
    "Kimi K3.1 Rubin dep16 MoE front: latent_down and the bucket quantization on the latent "
    "stream beside the FP32 router and the routing"
)
# The INFO line a dep16 generation rank logs once, at its first widening of the router's input.
_WIDEN = "Kimi K3.1 Rubin dep16 MoE front: router input widened by rubin_upcast_fp32"

# The block's classes and functions, run from modeling.py over its module-level constants.
_BLOCK = (
    "_MoeOut",
    "_Mx",
    "_moe_cublas_linear",
    "_moe_latent_gemm",
    "_moe_latent_down",
    "_widen_fp32",
    "_moe_router_input",
    "_moe_router_fp32",
    "_moe_router_gen",
    "_moe_route",
    "_moe_quantize",
    "_moe_bucket_rows",
    "_moe_bucket_route",
    "_moe_bucket_route_partials",
    "_moe_bucket_quantize",
    "_moe_bucket_launch",
    "_moe_megamoe_chunk",
    "_moe_split",
    "_moe_chunk_plane",
    "_moe_routed_norm",
    "_moe_megamoe",
    "_moe_dep_gen_routed",
    "_moe_dep_gen",
    "_moe_dep_gen_streams",
    "latent_moe",
)

# Rows per rank of the steps below: one chunk of at most 96 rows per rank, and a step that a chunk
# size of 4 rows splits into four chunks.
_MIXED = [6, 96, 32, 0, 1, 31, 33, 64, 95, 2, 3, 4, 5, 7, 8, 9]
_CHUNKED = [5, 9, 1, 13, 7, 2, 11, 3, 8, 6, 4, 10, 12, 1, 9, 0]
# case -> (rows per rank, the rank run, rows per rank of a chunk)
_CASES = {
    "bucket": ([6] * _RANKS, 0, 96),  # one chunk into bucket 32: a routing tail [6, 32)
    "full_bucket": ([96] * _RANKS, 0, 96),  # 96 rows into bucket 96: no routing tail
    "mixed": (_MIXED, 0, 96),  # 6 rows into the chunk's bucket, 96 (another rank's rows)
    "staged": (_MIXED, 2, 96),  # 32 rows: the staging path
    "no_rows": (_MIXED, 3, 96),  # no rows: the staging path launches on none
    "chunks": (_CHUNKED, 3, 4),  # 13 rows in four chunks of 4, 3, 3, 3 rows
    "mid": ([48] * _RANKS, 0, 96),  # 48 rows into bucket 64: moe_widen_after_fork's rows
}
_BUCKET_CASES = ("bucket", "full_bucket", "mixed", "mid")


def _rows(case: str) -> int:
    counts, rank, _ = _CASES[case]
    return counts[rank]


def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


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


def _route_max() -> int:
    """``MAX_M`` of the bucket routing's entry (modeling.py's ``_MOE_ROUTE_MAX_M``)."""
    for node in ast.parse(_ROUTE.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            "MAX_M"
        ]:
            return ast.literal_eval(node.value)
    raise KeyError("MAX_M")


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
    ns = {
        **_constants(tree),
        "NamedTuple": NamedTuple,
        "Any": Any,
        "_MOE_ROUTE_MAX_M": _route_max(),
        **namespace,
    }
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


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
    """One step's host order on three streams: its allocations, event records and waits, kernels.

    ``clock[s][o]`` is the latest point of stream ``o``'s order that stream ``s`` waited for.
    Memory allocated on stream ``o`` at point ``a`` is memory the caching allocator may have freed
    in ``o``'s order, so a kernel of another stream ``s`` may touch it only with
    ``clock[s][o] > a``. Memory the step did not allocate (weights, inputs, MegaMoE's buckets) is
    never handed out. ``ops`` is the step's order as ``op@stream``.
    """

    STREAMS = ("main", "latent", "aux")

    def __init__(self):
        self.now = 0
        self.stream = "main"
        self.clock: dict = {s: {} for s in self.STREAMS}
        self.homes: dict = {}  # allocation address -> (stream, point)
        self.kernels: list = []  # (point, stream, name, what it read and wrote, clock then)
        self.ops: list = []
        self.problems: list = []
        self.quiet = 0  # inside a stand-in: its own torch calls are not the block's

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
        if _memory(t):
            self.homes[_memory(t)] = (self.stream, self.tick())

    def kernel(self, name: str, payload, out=()) -> None:
        """One kernel on the current stream over ``payload``; ``out``: the memory it allocates."""
        for t in _tensors(out):
            self.allocate(t)
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


# Torch calls that only allocate or view: no kernel reads their inputs.
_ALLOCATE_ONLY = frozenset({"empty", "empty_like", "empty_strided", "new_empty"})


class _Record(TorchFunctionMode):
    """The block's own torch calls in the plan: a result in fresh memory is an allocation on the
    current stream; a call that fills fresh memory or writes in place is a kernel there. A
    stand-in's own calls are not the block's: the stand-in records itself."""

    def __init__(self, plan: _Plan):
        super().__init__()
        self.plan = plan

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.plan.quiet:
            return func(*args, **kwargs)
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


def _topk(logits: torch.Tensor, bias: torch.Tensor, scale: float) -> tuple:
    """Stand-in routing of the rows: ``(ids int32, weights FP32)``, ``[rows, top-k]``."""
    weights, ids = (logits + bias).topk(_TOP_K, dim=-1)
    return ids.to(torch.int32), (weights * scale).float()


def _nvfp4(x: torch.Tensor, scale: torch.Tensor) -> tuple:
    """Stand-in NVFP4 bytes of the rows: values ``[rows, latent / 2]``, scales ``[rows, latent /
    16]``."""
    f, n = x.float() * scale.float(), x.shape[0]
    x4 = (f.view(n, _LATENT // 2, 2).sum(-1) * 4).round().remainder(256).to(torch.uint8)
    sf = (f.abs().view(n, _LATENT // _SF_VEC, _SF_VEC).amax(-1) * 8).round().clamp(0, 255)
    return x4, sf.to(torch.uint8)


def _experts(x4, sf, ids, weights) -> torch.Tensor:
    """Stand-in MegaMoE output ``[rows, latent]`` BF16 of the rows' NVFP4 input and routing."""
    v = x4.float().repeat_interleave(2, dim=1) + sf.float().repeat_interleave(_SF_VEC, dim=1)
    return (v + (ids.float() * weights).sum(-1, keepdim=True)).to(torch.bfloat16)


def _entries(plan: _Plan, seen: dict) -> dict:
    """The block's entries as stand-ins whose values follow their inputs: each is one kernel on the
    current stream over what it reads and writes. ``seen`` keeps the bucket writers' arguments."""

    def entry(fn):
        def run(*args, **kwargs):
            plan.quiet += 1
            try:
                return fn(*args, **kwargs)
            finally:
                plan.quiet -= 1

        return run

    @entry
    def cublas_mm(a, b):
        out = (a.float() @ b.float()).to(torch.bfloat16)
        plan.kernel("latent_down" if b.shape[1] == _LATENT else "latent_up", (a, b), out)
        return out

    @entry
    def rubin_upcast_fp32(x):
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        out = x.float()
        if x.numel():  # the entry launches nothing for an empty input
            plan.kernel("upcast_fp32", (x,), out)
        return out

    @entry
    def rubin_linear(x, w, tf32):
        assert x.dtype == torch.float32 and w.dtype == torch.float32 and not tf32
        out = x @ w.t()
        plan.kernel("router_gemm", (x, w), out)
        return out

    @entry
    def rubin_router_tf32_partials(x, w):
        """Stand-in partials: the logits in slice 0, zeros in the other seven (their sum in slice
        order is the logits)."""
        assert x.dtype == torch.float32 and w.dtype == torch.float32
        out = torch.zeros(8, x.shape[0], w.shape[0])
        out[0] = x @ w.t()
        plan.kernel("router_tf32", (x, w), out)
        return out

    @entry
    def rubin_route_topk_out(
        logits, bias, scale, topk_ids, topk_weights, *, route_approx=False, rows=None, tail=None
    ):
        if logits.dim() == 3:  # split-K partials: rows [0, rows) of their sum in slice order
            assert rows is not None and tail is not None
            summed = logits[0, :rows]
            for part in logits[1:]:
                summed = summed + part[:rows]
            seen.setdefault("route_tail", []).append(tail)
        else:
            assert rows is None and tail is None
            summed = logits
        assert route_approx and summed.shape[0] == topk_ids.shape[0] == topk_weights.shape[0]
        ids, weights = _topk(summed, bias, scale)
        topk_ids.copy_(ids)
        topk_weights.copy_(weights)
        seen.setdefault("route", []).append((topk_ids, topk_weights, plan.stream))
        plan.kernel("route_topk_out", (logits, bias, topk_ids, topk_weights))

    @entry
    def rubin_megamoe_quantize(
        x, scale, activation, activation_sf, topk_idx, topk_weights, pdl=False
    ):
        n = x.shape[0]
        activation[:n], activation_sf[:n] = _nvfp4(x, scale)
        topk_idx[n:] = -1
        topk_weights[n:] = 0
        args = (x, activation, activation_sf, topk_idx, topk_weights)
        seen.setdefault("quantize", []).append((*args, plan.stream))
        seen.setdefault("quantize_pdl", []).append(pdl)
        plan.kernel("megamoe_quantize", (*args, scale))

    @entry
    def rubin_megamoe_in_bucket(b, *weights_ws_rows):
        rows = weights_ws_rows[-1]
        # Past the rows the bucket holds the routing of no row: the quantization's tail write.
        assert bool((b.topk_idx[rows:] == -1).all()) and bool((b.topk_weights[rows:] == 0).all())
        bufs = (b.activation, b.activation_sf, b.topk_idx, b.topk_weights)
        out = _experts(*(t[:rows] for t in bufs))
        plan.kernel("megamoe_in_bucket", bufs, out)
        return out

    @entry
    def noaux_tc_op(logits, bias, groups, topk_groups, top_k, scale):
        assert groups == topk_groups == 1 and top_k == _TOP_K
        ids, weights = _topk(logits, bias, scale)
        plan.kernel("noaux_tc", (logits, bias), (ids, weights))
        return weights, ids

    @entry
    def fp4_quantize(x, scale, sf_vec, sf_use_ue8m0, swizzled):
        assert sf_vec == _SF_VEC and not sf_use_ue8m0 and not swizzled
        x4, sf = _nvfp4(x, scale)
        sf = sf.reshape(-1)  # the linear block scales, flat
        plan.kernel("fp4_quantize", (x, scale), (x4, sf))
        return x4, sf

    @entry
    def rubin_megamoe(x4, sf, ids, weights, *weights_ws_chunk_rows):
        rows = weights_ws_chunk_rows[-1]
        assert x4.shape[0] == sf.shape[0] == ids.shape[0] == weights.shape[0] == rows
        out = _experts(x4, sf, ids, weights)
        plan.kernel("megamoe", (x4, sf, ids, weights), out)
        return out

    @entry
    def flashinfer_rmsnorm(x, weight, eps):
        f = x.float()
        out = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        out = out.to(torch.bfloat16)
        plan.kernel("rmsnorm", (x, weight), out)
        return out

    @entry
    def _moe_shared(L, x, gate_up_reserve=0):
        out = (x.float() * 0.5 + 1.0).to(torch.bfloat16)
        plan.kernel("shared", (x,), out)
        return out

    def megamoe_bucket(ws, chunk_max, rows):
        """The chunk's bucket: the ladder's smallest of its largest row count (no kernel)."""
        return [b for b in ws.buckets if b.max_tokens >= chunk_max][0]

    def _moe_dep_ctx(L, x, st, rt):
        raise AssertionError("a generation server's step took the context path")

    return dict(
        cublas_mm=cublas_mm,
        rubin_upcast_fp32=rubin_upcast_fp32,
        rubin_linear=rubin_linear,
        rubin_router_tf32_partials=rubin_router_tf32_partials,
        rubin_route_topk_out=rubin_route_topk_out,
        rubin_megamoe_quantize=rubin_megamoe_quantize,
        rubin_megamoe_in_bucket=rubin_megamoe_in_bucket,
        noaux_tc_op=noaux_tc_op,
        fp4_quantize=fp4_quantize,
        rubin_megamoe=rubin_megamoe,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
        _moe_shared=_moe_shared,
        megamoe_bucket=megamoe_bucket,
        _moe_dep_ctx=_moe_dep_ctx,
    )


def _layer(gen: torch.Generator) -> SimpleNamespace:
    """The MoE weight views the block reads (the experts' own weights go to MegaMoE untouched)."""

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen).to(dtype)

    return SimpleNamespace(
        router=r(_EXPERTS, _HIDDEN, dtype=torch.float32),
        router_bias=r(_EXPERTS, dtype=torch.float32),
        latent_down=r(_LATENT, _HIDDEN),
        latent_up=r(_HIDDEN, _LATENT),
        latent_norm=r(_LATENT),
        fc31_in_scale=torch.ones(1),
        fc1_w=None,
        fc1_s=None,
        fc2_w=None,
        fc2_s=None,
        fc31_alpha=None,
        fc2_alpha=None,
        fc1_norm_const=None,
    )


def _workspace(rank: int, chunk_tokens: int) -> SimpleNamespace:
    """MegaMoE's buckets as one rank holds them, built before the step; the bytes nothing wrote
    hold a pattern no writer leaves."""

    def bucket(t: int) -> SimpleNamespace:
        return SimpleNamespace(
            max_tokens=t,
            activation=torch.full((t, _LATENT // 2), 0x5A, dtype=torch.uint8),
            activation_sf=torch.full((t, _LATENT // _SF_VEC), 0x5A, dtype=torch.uint8),
            topk_idx=torch.full((t, _TOP_K), 7, dtype=torch.int32),
            topk_weights=torch.full((t, _TOP_K), 0.5),
        )

    return SimpleNamespace(
        ep_rank=rank, chunk_tokens=chunk_tokens, buckets=tuple(bucket(t) for t in _LADDER)
    )


def _rt(plan: _Plan, ws: SimpleNamespace, moe_router_tf32: bool = True) -> SimpleNamespace:
    """The runtime one rank's step reads; ``moe_router_tf32``: router_tf32_splitk's check passed
    (as on an SM107 server with K3.1's BF16 router weights)."""
    events = {
        name: _Event(plan, name)
        for name in ("shared_fork", "shared_join", "latent_fork", "latent_ready", "memset_ready")
    }
    return SimpleNamespace(
        aux_stream=_Stream("aux"),
        moe_comm=SimpleNamespace(
            latent_stream=_Stream("latent"), memset_stream=_Stream("memset"), **events
        ),
        megamoe=ws,
        shared_gate_up_reserve=0,
        moe_back_planes=None,
        moe_router_tf32=moe_router_tf32,
    )


def _run(
    case: str, cuda_graph: bool = True, tree=None, moe_router_tf32: bool = True
) -> SimpleNamespace:
    """One ``latent_moe`` call of ``tree`` (modeling.py by default) on the case's rank: the plan,
    the logger, the block, the bucket writers' arguments, the workspace and the output."""
    counts, rank, chunk_tokens = _CASES[case]
    plan, seen = _Plan(), {}
    log = _Logger(plan)
    block = _block(tree or _tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan, seen)})
    # Weights, inputs and workspaces exist before the step: built outside the recording.
    gen = torch.Generator().manual_seed(2026)
    layer = _layer(gen)
    x = torch.randn(counts[rank], _HIDDEN, generator=gen).to(torch.bfloat16)
    ws = _workspace(rank, chunk_tokens)
    rt = _rt(plan, ws, moe_router_tf32)
    st = SimpleNamespace(cuda_graph=cuda_graph, all_rank_num_tokens=counts)
    with _Record(plan):
        out = block["latent_moe"](layer, _LAYER, x, st, rt)
    assert tuple(out.routed.shape) == tuple(out.shared.shape) == (counts[rank], _HIDDEN)
    return SimpleNamespace(plan=plan, log=log, block=block, seen=seen, ws=ws, out=out)


# Every generation graph step ends so: MegaMoE's rows normed and up-projected on the current
# stream; the shared experts on the aux stream from the router's end, joined at the end.
_GRAPH_TAIL = [
    "log",
    "rmsnorm@main",
    "latent_up@main",
    "wait:shared_fork@aux",
    "shared@aux",
    "record:shared_join@aux",
    "wait:shared_join@main",
]
_EAGER_TAIL = ["rmsnorm@main", "latent_up@main", "shared@main"]
# The chunk code normalizes each chunk's rows itself: the tails after it.
_GRAPH_CHUNK_TAIL = [op for op in _GRAPH_TAIL if op != "rmsnorm@main"]
_EAGER_CHUNK_TAIL = [op for op in _EAGER_TAIL if op != "rmsnorm@main"]
# A one-chunk graph step of the bucket path: the quantization after latent_down on the latent
# stream, the routing on the current stream after the router GEMM, MegaMoE after the wait.
_GRAPH_BUCKET = [
    "upcast_fp32@main",
    "log",
    "record:latent_fork@main",
    "wait:latent_fork@latent",
    "latent_down@latent",
    "megamoe_quantize@latent",
    "record:latent_ready@latent",
    "router_gemm@main",
    "record:shared_fork@main",
    "route_topk_out@main",
    "wait:latent_ready@main",
    "megamoe_in_bucket@main",
    *_GRAPH_TAIL,
]
_EAGER_BUCKET = [
    "upcast_fp32@main",
    "log",
    "router_gemm@main",
    "latent_down@main",
    "route_topk_out@main",
    "megamoe_quantize@main",
    "megamoe_in_bucket@main",
    *_EAGER_TAIL,
]
# The chunk code of the other cases, the same on both kinds of step.
_CHUNK_OPS = {
    "staged": ["noaux_tc@main", "fp4_quantize@main", "megamoe@main", "rmsnorm@main"],
    "no_rows": ["megamoe@main", "rmsnorm@main"],
    "chunks": [
        "route_topk_out@main",
        "megamoe_quantize@main",
        "megamoe_in_bucket@main",
        "rmsnorm@main",
        "torch copy_@main",
    ]
    * 4,
}


def _cast(case: str) -> list:
    """The router's FP32 widening by the entry (an empty step launches nothing), then its line."""
    return (["upcast_fp32@main"] if _rows(case) else []) + ["log"]


def _late_cast(case: str, bounds=None) -> bool:
    """Whether the case's graph step widens the router's input after the fork: its rows (the
    step's, ``x``; no mix hands it the FP32 copy here) in moe_widen_after_fork's bounds."""
    first, last = bounds or _constants(_tree())["_MOE_WIDEN_LATE_M"]
    return first <= _rows(case) <= last


def _tf32(case: str, bounds=None) -> bool:
    """Whether the case's graph step runs router_tf32_splitk: the bucket path, its rows (``x``'s)
    in the lever's bounds (as committed, or ``bounds``)."""
    first, last = bounds or _constants(_tree())["_MOE_ROUTER_TF32_M"]
    return case in _BUCKET_CASES and first <= _rows(case) <= last


def _graph_plan(case: str, bounds=None, tf32_bounds=None, tf32: bool = True) -> list:
    """The case's graph step: moe_widen_after_fork's rows ``bounds``, router_tf32_splitk's
    ``tf32_bounds`` (``tf32``: its runtime check passed)."""
    plan = _graph_ops(case, bounds)
    if tf32 and _tf32(case, tf32_bounds):
        plan = ["router_tf32@main" if op == "router_gemm@main" else op for op in plan]
    return plan


def _graph_ops(case: str, bounds=None) -> list:
    if case in _BUCKET_CASES:
        plan = list(_GRAPH_BUCKET)
    else:
        plan = [
            *_cast(case),
            "record:latent_fork@main",
            "wait:latent_fork@latent",
            "latent_down@latent",
            "record:latent_ready@latent",
            "router_gemm@main",
            "record:shared_fork@main",
            "wait:latent_ready@main",
            *_CHUNK_OPS[case],
            *_GRAPH_CHUNK_TAIL,
        ]
    if _late_cast(case, bounds):  # the cast and its line after the latent stream's work
        cast = _cast(case)
        assert plan[: len(cast)] == cast, plan
        plan = plan[len(cast) :]
        at = plan.index("record:latent_ready@latent") + 1
        plan[at:at] = cast
    return plan


def _eager_plan(case: str) -> list:
    if case in _BUCKET_CASES:
        return _EAGER_BUCKET
    return [
        *_cast(case),
        "router_gemm@main",
        "latent_down@main",
        *_CHUNK_OPS[case],
        *_EAGER_CHUNK_TAIL,
    ]


def _bucket_order_problems(plan: _Plan) -> list:
    """Each MegaMoE launch on a bucket after the bucket's last two writes before it, the routing
    and the quantization: in the launch stream's order, or through a wait on the writer's
    stream."""
    found = []
    for at, stream, name, _, clock in plan.kernels:
        if name != "megamoe_in_bucket":
            continue
        for writer in ("route_topk_out", "megamoe_quantize"):
            before = [(k[0], k[1]) for k in plan.kernels if k[2] == writer and k[0] < at]
            if not before:
                found.append(f"MegaMoE at {at} launches before any {writer}")
                continue
            w_at, w_stream = max(before)
            if w_stream != stream and clock.get(w_stream, 0) <= w_at:
                found.append(f"MegaMoE at {at} reads the bucket before {writer} at {w_at}")
    return found


def _unjoined(plan: _Plan) -> list:
    """The side streams with work the current stream did not wait for by the step's end (the
    capture of the step would fail)."""
    last = {}
    for at, stream, *_ in plan.kernels:
        last[stream] = at
    return [s for s in ("latent", "aux") if s in last and plan.clock["main"].get(s, 0) <= last[s]]


def _faulty(tree: ast.Module, fault: str) -> ast.Module:
    """``tree`` with one statement of ``_moe_dep_gen_streams`` moved: ``latent_ready`` recorded
    before the quantization on the latent stream, or the wait on it after MegaMoE's launch."""
    tree = copy.deepcopy(tree)
    fn = _function(tree, "_moe_dep_gen_streams")
    if fault == "ready_before_quantization":
        (part,) = [
            s
            for s in fn.body
            if isinstance(s, ast.With) and "c.latent_stream" in ast.unparse(s.items[0])
        ]
        statement, before = "c.latent_ready.record()", "_moe_bucket_quantize("
    else:
        statement, before = "routed = _moe_bucket_launch(L, b, rows, rt)", "c.latent_ready.wait()"
        # The bucket path's branch: the one that launches MegaMoE (a later branch normalizes).
        (part,) = [
            s
            for s in fn.body
            if isinstance(s, ast.If)
            and ast.unparse(s.test) == "b is not None"
            and statement in [ast.unparse(t) for t in s.body]
        ]
    body = list(part.body)
    texts = [ast.unparse(s) for s in body]
    (k,) = [i for i, t in enumerate(texts) if before in t]
    moved = body.pop(texts.index(statement))
    body.insert(k, moved)
    part.body = body
    return tree


def test_the_cases_take_the_paths_this_file_checks():
    block = _run("bucket").block
    assert block["_MOE_TOP_K"] == _TOP_K and block["_MOE_ROUTE_SKIP_M"] == frozenset({32})
    assert block["_MOE_ROUTE_MAX_M"] >= max(_LADDER)
    for case, (counts, _, chunk_tokens) in _CASES.items():
        one_chunk = max(counts) <= chunk_tokens
        into_bucket = one_chunk and block["_moe_bucket_rows"](_rows(case))
        assert into_bucket == (case in _BUCKET_CASES), case
        assert one_chunk == (case != "chunks"), case


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_graph_step_runs_latent_down_on_the_latent_stream(case):
    plan = _run(case).plan
    assert plan.ops == _graph_plan(case)
    latent = [op for op in plan.ops if op.endswith("@latent")]
    front = ["latent_down@latent"] + (["megamoe_quantize@latent"] if case in _BUCKET_CASES else [])
    assert latent == ["wait:latent_fork@latent", *front, "record:latent_ready@latent"], latent


def test_the_latent_stream_forks_after_the_router_cast():
    """The fork follows the FP32 cast and precedes the router GEMM's issue: the GEMM, launched
    behind the cast, takes its SMs before latent_down, which waits for the cast's end. In
    moe_widen_after_fork's rows the cast is issued after the latent stream's work instead, still
    before the GEMM. The GEMM is ``_moe_router_fp32``'s, on the widened input
    (``_moe_router_input``)."""
    tree = _tree()
    body = [ast.unparse(s) for s in _function(tree, "_moe_dep_gen_streams").body]
    late = body.index(
        "late = xf is None and _MOE_WIDEN_LATE_M[0] <= x.shape[0] <= _MOE_WIDEN_LATE_M[1]"
    )
    cast = body.index("if xf is None and (not late):\n    xf = _moe_router_input(x)")
    fork = body.index("c.latent_fork.record()")
    (side,) = [
        k for k, t in enumerate(body) if t.startswith("with torch.cuda.stream(c.latent_stream)")
    ]
    late_cast = body.index("if late:\n    xf = _moe_router_input(x)")
    gemm = body.index("logits = _moe_router_gen(L, xf, tf32)")
    assert late < cast < fork < side < late_cast < gemm, body
    gen = [ast.unparse(s) for s in _function(tree, "_moe_router_gen").body]
    assert gen[1:] == [
        "if tf32:\n    return rubin_router_tf32_partials(xf, L.router)",
        "return rubin_linear(xf, L.router, False)",
    ], gen
    router = _function(tree, "_moe_router_fp32")
    (ret,) = [ast.unparse(s) for s in router.body if isinstance(s, ast.Return)]
    assert (
        ret == "return rubin_linear(_moe_router_input(x) if xf is None else xf, L.router, False)"
    ), ret


@pytest.mark.parametrize("case", _BUCKET_CASES)
def test_megamoe_reads_the_bucket_after_its_two_writers_which_write_disjoint_rows(case):
    run = _run(case)
    counts, _, _ = _CASES[case]
    rows = _rows(case)
    (b,) = [b for b in run.ws.buckets if b.max_tokens >= max(counts)][:1]
    ((ids, weights, route_stream),) = run.seen["route"]
    ((x, act, act_sf, q_ids, q_weights, quant_stream),) = run.seen["quantize"]
    assert (route_stream, quant_stream) == ("main", "latent")
    # The routing writes rows [0, rows) of the bucket's routing: views of its buffers that end
    # there.
    assert ids.data_ptr() == b.topk_idx.data_ptr() and ids.shape[0] == rows
    assert weights.data_ptr() == b.topk_weights.data_ptr() and weights.shape[0] == rows
    # The quantization takes the bucket's whole buffers and the rows: it writes the activation
    # rows [0, rows) and the routing rows [rows, T), none of the routing's.
    bufs = (b.activation, b.activation_sf, b.topk_idx, b.topk_weights)
    assert all(u is v for u, v in zip((act, act_sf, q_ids, q_weights), bufs))
    assert x.shape[0] == rows <= b.max_tokens
    # MegaMoE's launch comes after both: the routing in its stream's order, the quantization
    # through the wait on latent_ready.
    assert not _bucket_order_problems(run.plan)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_graph_step_touches_memory_after_its_earlier_users(case):
    plan = _run(case).plan
    assert not plan.problems, "\n".join(plan.problems)
    assert not _bucket_order_problems(plan)
    assert not _unjoined(plan), _unjoined(plan)


@pytest.mark.parametrize("fault", ["ready_before_quantization", "launch_before_the_wait"])
def test_the_checks_flag_a_bucket_read_before_the_quantization(fault):
    plan = _run("bucket", tree=_faulty(_tree(), fault)).plan
    assert _bucket_order_problems(plan), plan.ops
    if fault == "ready_before_quantization":
        # Nothing waits for the quantization: a capture ends with the latent stream unjoined.
        assert _unjoined(plan) == ["latent"], plan.ops


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_graph_step_returns_what_the_eager_step_returns(case):
    graph, eager = _run(case), _run(case, cuda_graph=False)
    assert torch.equal(graph.out.routed, eager.out.routed)
    assert torch.equal(graph.out.shared, eager.out.shared)
    for g, e in zip(graph.ws.buckets, eager.ws.buckets):
        for field in ("activation", "activation_sf", "topk_idx", "topk_weights"):
            assert torch.equal(getattr(g, field), getattr(e, field)), (g.max_tokens, field)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_an_eager_step_runs_on_the_current_stream_alone(case):
    run = _run(case, cuda_graph=False)
    assert run.plan.ops == _eager_plan(case)
    assert not any(run.plan.clock.values()) and not run.plan.problems
    assert run.log.once == [(_WIDEN, _WIDEN)]


def test_a_graph_step_logs_the_line_once_per_process():
    run = _run("full_bucket")
    assert run.block["_MOE_LATENT_BESIDE_LINE"] == _LINE
    assert run.block["_MOE_ROUTER_WIDEN_LINE"] == _WIDEN
    assert run.log.once == [(_WIDEN, _WIDEN), (_LINE, _LINE)]


@pytest.mark.parametrize("cuda_graph", [True, False], ids=["graph", "eager"])
def test_the_router_gemm_takes_the_entrys_widening_of_the_input(cuda_graph):
    """One widening by the entry on the current stream, then the router GEMM on its output, whose
    bits are ``x.float()``'s; no casting copy."""
    plan = _run("full_bucket", cuda_graph=cuda_graph).plan
    names = [name for _, _, name, _, _ in plan.kernels]
    assert names.count("upcast_fp32") == 1 and "torch float" not in names, names
    ((_, stream, _, (x,), _),) = [k for k in plan.kernels if k[2] == "upcast_fp32"]
    # 96 rows of a graph step: router_tf32_splitk's GEMM; an eager step: rubin_linear's
    router = "router_tf32" if cuda_graph else "router_gemm"
    ((_, _, _, (xf, w), _),) = [k for k in plan.kernels if k[2] == router]
    assert stream == "main" and x.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert names.index("upcast_fp32") < names.index(router)
    assert xf.dtype == torch.float32 and torch.equal(
        xf.view(torch.int32), x.float().view(torch.int32)
    )


def test_a_non_contiguous_router_input_keeps_x_float():
    """``_moe_router_input`` on a non-contiguous input: ``x.float()``, no entry, no line; the
    router GEMM takes those bits."""
    plan, seen = _Plan(), {}
    log = _Logger(plan)
    block = _block(_tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan, seen)})
    gen = torch.Generator().manual_seed(7)
    layer = _layer(gen)
    x = torch.randn(_HIDDEN, 6, generator=gen).to(torch.bfloat16).t()
    assert not x.is_contiguous()
    with _Record(plan):
        xf = block["_moe_router_input"](x)
        logits = block["_moe_router_fp32"](layer, x)
    assert plan.ops == ["torch float@main", "torch float@main", "router_gemm@main"], plan.ops
    assert log.once == []
    assert torch.equal(xf.view(torch.int32), x.float().view(torch.int32))
    assert torch.equal(logits, x.float() @ layer.router.t())


def test_the_graph_front_widens_the_router_input_once_and_only_for_the_router():
    """In ``_moe_dep_gen_streams`` the router's input comes from ``_moe_router_input`` (one call on
    a step: before the fork, or after it in moe_widen_after_fork's rows; no ``.float()`` left) and
    is read only by the router GEMM; ``_moe_router_input`` widens through ``_widen_fp32``, the
    helper the prefill indexer also takes."""
    tree = _tree()
    fn = _function(tree, "_moe_dep_gen_streams")
    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert calls.count("_moe_router_input") == 2 and "x.float" not in calls, calls
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "xf"]
    # one widening on a step, only when the mix did not hand ``xf`` over (router_f32): before the
    # fork unless ``late``, after it if ``late``; the router reads it
    kinds = sorted(type(n.ctx).__name__ for n in reads)
    assert kinds == ["Load", "Load", "Load", "Store", "Store"], reads
    body = [ast.unparse(s) for s in fn.body]
    assert "if xf is None and (not late):\n    xf = _moe_router_input(x)" in body
    assert "if late:\n    xf = _moe_router_input(x)" in body
    assert "logits = _moe_router_gen(L, xf, tf32)" in body
    (ret,) = [
        ast.unparse(s) for s in _function(tree, "_widen_fp32").body if isinstance(s, ast.Return)
    ]
    assert ret == "return rubin_upcast_fp32(x) if x.is_contiguous() else x.float()", ret
    body = [ast.unparse(s) for s in _function(tree, "_moe_router_input").body]
    assert body[1:] == [
        "xf = _widen_fp32(x)",
        "if x.is_contiguous():\n    logger.info_once(_MOE_ROUTER_WIDEN_LINE, key=_MOE_ROUTER_WIDEN_LINE)",
        "return xf",
    ], body


def test_the_runtime_holds_the_latent_stream_and_its_events():
    """``_MoeComm`` holds the latent stream and ``latent_ready``; ``_build_runtime`` binds the
    core's latent side stream and makes one event per event field."""
    tree = _tree()

    def fields(name: str) -> list:
        (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        return [
            (n.target.id, ast.unparse(n.annotation))
            for n in cls.body
            if isinstance(n, ast.AnnAssign)
        ]

    comm = fields("_MoeComm")
    assert comm[:2] == [
        ("latent_stream", "torch.cuda.Stream"),
        ("memset_stream", "torch.cuda.Stream"),
    ]
    events = [name for name, kind in comm[2:] if kind == "torch.cuda.Event"]
    assert len(events) == len(comm) - 2, comm
    assert {"latent_fork", "latent_ready", "shared_fork", "shared_join"} <= set(events)
    assert ("latent", "torch.cuda.Stream") in fields("_Streams")
    (call,) = [
        n
        for n in ast.walk(_function(tree, "_build_runtime"))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_MoeComm"
    ]
    named = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
    assert named == {"latent_stream": "s.latent", "memset_stream": "s.memset"}, named
    (rest,) = [ast.unparse(k.value) for k in call.keywords if k.arg is None]
    assert rest == "{f: torch.cuda.Event() for f in _MoeComm._fields[2:]}", rest


def _with_quant_pdl(block: dict, bounds: tuple) -> dict:
    """``block`` with megamoe_quantize_pdl's rows set to ``bounds`` (the module constant its graph
    step reads)."""
    block["_MOE_QUANT_PDL_M"] = bounds
    return block


def test_megamoe_quantize_pdl_is_a_generation_constant_on_at_96_rows():
    tree = _tree()
    consts = _constants(tree)
    gen, ctx = consts["_CONSTANTS"]["generation"], consts["_CONSTANTS"]["context"]
    assert gen["megamoe_quantize_pdl"] is True and "megamoe_quantize_pdl" not in ctx
    assert (gen["megamoe_quantize_pdl_min_m"], gen["megamoe_quantize_pdl_max_m"]) == (96, 96)
    assert consts["_MOE_QUANT_PDL_M"] == (96, 96)
    details = consts["_DETAILS"]
    for key in ("megamoe_quantize_pdl_min_m", "megamoe_quantize_pdl_max_m"):
        assert details[key] == "megamoe_quantize_pdl", key


@pytest.mark.parametrize("case", sorted(_CASES))
def test_megamoe_quantize_pdl_launches_the_bucket_quantization_programmatically(case):
    """96 rows of the bucket path (full_bucket) launch with PDL on a graph step; 6 rows of it, the
    staging and chunk paths and every eager step launch in stream order."""
    graph, eager = _run(case), _run(case, cuda_graph=False)
    if case in _BUCKET_CASES:
        want = [case == "full_bucket"]
    else:  # the chunk code (several chunks) or no quantization into a bucket (staging, no rows)
        want = [False] * 4 if case == "chunks" else []
    assert graph.seen.get("quantize_pdl", []) == want, graph.seen.get("quantize_pdl")
    assert set(eager.seen.get("quantize_pdl", [])) <= {False}


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("bounds", [(1, 4096), (1, 0)], ids=["every_row", "none"])
def test_megamoe_quantize_pdl_bounds_change_nothing_but_the_launch(case, bounds):
    """With the lever's rows patched to every row count or to none, a graph step runs the same
    plan and leaves the same bucket and outputs as the in-order step."""
    counts, rank, chunk_tokens = _CASES[case]
    runs = []
    for patched in (False, True):
        plan, seen = _Plan(), {}
        log = _Logger(plan)
        block = _block(_tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan, seen)})
        if patched:
            _with_quant_pdl(block, bounds)
        gen = torch.Generator().manual_seed(2026)
        layer = _layer(gen)
        x = torch.randn(counts[rank], _HIDDEN, generator=gen).to(torch.bfloat16)
        ws = _workspace(rank, chunk_tokens)
        st = SimpleNamespace(cuda_graph=True, all_rank_num_tokens=counts)
        with _Record(plan):
            out = block["latent_moe"](layer, _LAYER, x, st, _rt(plan, ws))
        runs.append(SimpleNamespace(plan=plan, seen=seen, ws=ws, out=out))
    base, lever = runs
    assert lever.plan.ops == base.plan.ops == _graph_plan(case)
    assert not lever.plan.problems and not _bucket_order_problems(lever.plan)
    assert torch.equal(lever.out.routed, base.out.routed)
    assert torch.equal(lever.out.shared, base.out.shared)
    for g, e in zip(lever.ws.buckets, base.ws.buckets):
        for field in ("activation", "activation_sf", "topk_idx", "topk_weights"):
            assert torch.equal(getattr(g, field), getattr(e, field)), (g.max_tokens, field)
    pdl = lever.seen.get("quantize_pdl", [])
    if case in _BUCKET_CASES:
        assert pdl == [bounds[0] <= _rows(case) <= bounds[1]], pdl
    else:  # the chunk code (several chunks) or no quantization into a bucket (staging, no rows)
        assert pdl == ([False] * 4 if case == "chunks" else []), pdl


def test_moe_widen_after_fork_is_a_generation_constant_on_at_48_rows():
    consts = _constants(_tree())
    gen, ctx = consts["_CONSTANTS"]["generation"], consts["_CONSTANTS"]["context"]
    assert gen["moe_widen_after_fork"] is True and "moe_widen_after_fork" not in ctx
    assert (gen["moe_widen_after_fork_min_m"], gen["moe_widen_after_fork_max_m"]) == (48, 48)
    assert consts["_MOE_WIDEN_LATE_M"] == (48, 48)
    # The mix stores the FP32 copy only above attn_res_pdl_max_m: the bounds lie below it.
    assert consts["_MOE_WIDEN_LATE_M"][1] <= gen["attn_res_pdl_max_m"]
    for key in ("moe_widen_after_fork_min_m", "moe_widen_after_fork_max_m"):
        assert consts["_DETAILS"][key] == "moe_widen_after_fork", key


def test_a_48_row_graph_step_widens_the_router_input_after_the_fork():
    plan = _run("mid").plan
    ops = plan.ops
    assert ops.index("record:latent_ready@latent") < ops.index("upcast_fp32@main")
    assert ops.index("upcast_fp32@main") < ops.index("router_gemm@main")
    (_, _, _, _, q_clock) = [k for k in plan.kernels if k[2] == "megamoe_quantize"][0]
    (u_at, _, _, _, _) = [k for k in plan.kernels if k[2] == "upcast_fp32"][0]
    # The latent stream waited for the fork only, before the widening was issued.
    assert q_clock.get("main", 0) < u_at, q_clock


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("bounds", [(1, 4096), (1, 0)], ids=["every_row", "none"])
def test_moe_widen_after_fork_bounds_change_nothing_but_the_order(case, bounds):
    """With the lever's rows patched to every row count or to none, a graph step only moves the
    widening (and its line) after the latent stream's work, and leaves the same bucket and outputs
    as with the bounds as committed."""
    runs = []
    for patch in (None, bounds):
        plan, seen = _Plan(), {}
        log = _Logger(plan)
        block = _block(_tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan, seen)})
        if patch is not None:
            block["_MOE_WIDEN_LATE_M"] = patch
        counts, rank, chunk_tokens = _CASES[case]
        gen = torch.Generator().manual_seed(2026)
        layer = _layer(gen)
        x = torch.randn(counts[rank], _HIDDEN, generator=gen).to(torch.bfloat16)
        ws = _workspace(rank, chunk_tokens)
        st = SimpleNamespace(cuda_graph=True, all_rank_num_tokens=counts)
        with _Record(plan):
            out = block["latent_moe"](layer, _LAYER, x, st, _rt(plan, ws))
        runs.append(SimpleNamespace(plan=plan, ws=ws, out=out))
    base, lever = runs
    assert base.plan.ops == _graph_plan(case)
    assert lever.plan.ops == _graph_plan(case, bounds)
    assert not lever.plan.problems and not _bucket_order_problems(lever.plan)
    assert not _unjoined(lever.plan)
    assert torch.equal(lever.out.routed, base.out.routed)
    assert torch.equal(lever.out.shared, base.out.shared)
    for g, e in zip(lever.ws.buckets, base.ws.buckets):
        for field in ("activation", "activation_sf", "topk_idx", "topk_weights"):
            assert torch.equal(getattr(g, field), getattr(e, field)), (g.max_tokens, field)


def test_moe_router_tf32_splitk_is_a_generation_constant_on_at_96_rows():
    consts = _constants(_tree())
    gen, ctx = consts["_CONSTANTS"]["generation"], consts["_CONSTANTS"]["context"]
    assert gen["moe_router_tf32_splitk"] is True and "moe_router_tf32_splitk" not in ctx
    assert (gen["moe_router_tf32_splitk_min_m"], gen["moe_router_tf32_splitk_max_m"]) == (96, 96)
    assert consts["_MOE_ROUTER_TF32_M"] == (96, 96)
    # One row per CTA of the routing launch at the bounds' largest row count.
    assert consts["_MOE_ROUTER_TF32_TAIL"] == (96, 4, 4)
    for key in ("moe_router_tf32_splitk_min_m", "moe_router_tf32_splitk_max_m"):
        assert consts["_DETAILS"][key] == "moe_router_tf32_splitk", key


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("checked", [True, False], ids=["check_passed", "check_failed"])
def test_router_tf32_splitk_runs_on_bucket_graph_steps_in_its_rows_only(case, checked):
    """96 rows of the bucket path (full_bucket) on a graph step whose runtime passed the check:
    the router's partials, routed on the 96-CTA launch; every other step runs rubin_linear and
    routes logits. The outputs and the bucket are the eager step's."""
    graph = _run(case, moe_router_tf32=checked)
    eager = _run(case, cuda_graph=False, moe_router_tf32=checked)
    on = checked and case == "full_bucket"
    assert graph.plan.ops == _graph_plan(case, tf32=checked)
    assert ("router_tf32@main" in graph.plan.ops) == on
    assert graph.seen.get("route_tail", []) == ([(96, 4, 4)] if on else [])
    assert "router_tf32@main" not in eager.plan.ops and "route_tail" not in eager.seen
    assert not graph.plan.problems and not _bucket_order_problems(graph.plan)
    assert torch.equal(graph.out.routed, eager.out.routed)
    assert torch.equal(graph.out.shared, eager.out.shared)
    for g, e in zip(graph.ws.buckets, eager.ws.buckets):
        for field in ("activation", "activation_sf", "topk_idx", "topk_weights"):
            assert torch.equal(getattr(g, field), getattr(e, field)), (g.max_tokens, field)


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("bounds", [(1, 4096), (1, 0)], ids=["every_row", "none"])
def test_router_tf32_splitk_bounds_change_nothing_but_the_router_kernel(case, bounds):
    """With the lever's rows patched to every row count or to none, a graph step swaps only the
    router GEMM (and the routing's input) and leaves the same bucket and outputs."""
    runs = []
    for patch in (None, bounds):
        plan, seen = _Plan(), {}
        log = _Logger(plan)
        block = _block(_tree(), {"torch": _Torch(plan), "logger": log, **_entries(plan, seen)})
        if patch is not None:
            block["_MOE_ROUTER_TF32_M"] = patch
        counts, rank, chunk_tokens = _CASES[case]
        gen = torch.Generator().manual_seed(2026)
        layer = _layer(gen)
        x = torch.randn(counts[rank], _HIDDEN, generator=gen).to(torch.bfloat16)
        ws = _workspace(rank, chunk_tokens)
        st = SimpleNamespace(cuda_graph=True, all_rank_num_tokens=counts)
        with _Record(plan):
            out = block["latent_moe"](layer, _LAYER, x, st, _rt(plan, ws))
        runs.append(SimpleNamespace(plan=plan, ws=ws, out=out))
    base, lever = runs
    assert base.plan.ops == _graph_plan(case)
    assert lever.plan.ops == _graph_plan(case, tf32_bounds=bounds)
    assert not lever.plan.problems and not _bucket_order_problems(lever.plan)
    assert not _unjoined(lever.plan)
    assert torch.equal(lever.out.routed, base.out.routed)
    assert torch.equal(lever.out.shared, base.out.shared)
    for g, e in zip(lever.ws.buckets, base.ws.buckets):
        for field in ("activation", "activation_sf", "topk_idx", "topk_weights"):
            assert torch.equal(getattr(g, field), getattr(e, field)), (g.max_tokens, field)


def test_the_runtime_runs_router_tf32_splitk_only_where_its_check_passed():
    """``_build_runtime`` runs ``_moe_router_tf32_check`` over every MoE layer's router weight
    when the constant is on and hands its verdict to the runtime; a failed check is a WARNING."""
    fn = _function(_tree(), "_build_runtime")
    body = ast.unparse(fn)
    assert "routers = [t for k, t in core.w.items() if k.endswith('_moe_router')]" in body
    assert "why = _moe_router_tf32_check(routers)" in body
    assert "moe_router_tf32 = why is None" in body
    (call,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_Rt"]
    named = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert named["moe_router_tf32"] == "moe_router_tf32"
    check = ast.unparse(_function(_tree(), "_moe_router_tf32_check"))
    # TF32-exact weights, then the partials' sum against rubin_linear at every row of the bounds.
    assert "& 8191" in check and "rubin_linear(x, routers[0], False)" in check
    assert "range(_MOE_ROUTER_TF32_M[0], _MOE_ROUTER_TF32_M[1] + 1)" in check


def test_the_router_tf32_check_refuses_a_runtime_it_cannot_prove():
    """No router, or weights on a device the lever does not run on: the reason, not None."""
    ns = {**_constants(_tree()), "torch": torch, "Optional": Any, **_entries(_Plan(), {})}
    module = ast.Module(body=[_function(_tree(), "_moe_router_tf32_check")], type_ignores=[])
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    check = ns["_moe_router_tf32_check"]
    assert check([]) == "no MoE router"
    assert check([torch.zeros(4, 8)]) == "not an SM107 device"

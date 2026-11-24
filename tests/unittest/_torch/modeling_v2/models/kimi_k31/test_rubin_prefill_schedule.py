# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepGEMM's MQA-logits schedule in the specialized Kimi K3.1 Rubin targets' prefill indexer.

The family DSA metadata plans each prefill chunk's query tiles and builds DeepGEMM's schedule of
every tile once a step (``rubin_mqa_logits_metadata``). The dep16 and tep8 targets' prefill
indexer (``_idx_prefill``, which only ``mla_prefill`` reaches) passes each tile's schedule to
``fp8_mqa_logits``. With it DeepGEMM splits the keys over the SMs; the logits stay the same.

On the CPU (``cpu_only``), each target's own step projection (``_dsa_step_args``) and prefill
indexer run from its modeling.py, with the kernels recorded, over plans the metadata builds:

* every logits call gets the schedule built for exactly its windows, key count and heads: tep8's
  batch plan on every rank (one rank's rows also cut across two requests), its per-chunk TP query
  split, dep16's tiles behind 86k to 283k cached keys and a packed chunk; the tiles cover the rows
  the projection gives the rank;
* every other argument is the unscheduled call's: the tiles of the per-call budget over the
  same rows, the same windows, keys and ``clean_logits``;
* two DSA layers of a step read the same schedules; the next step reads only its own;
* a plan without schedules, and a step the metadata did not chunk, call without one; a step
  whose context rows take the dense Top-K makes no call;
* in each target every ``fp8_mqa_logits`` call passes ``schedule_meta`` and only the prefill
  reaches one: no generation step holds a schedule, and the routing runs prefill eagerly
  (``test_routing_sm107.py``), so no CUDA graph captures one; helix8 makes no such call;
* the entry sends an unscheduled call to DeepGEMM's ``fp8_mqa_logits`` and a
  scheduled one to ``fp8_fp4_mqa_logits`` with no query scale and fp32 logits; its reference
  ignores the schedule and its guard refuses a schedule of another call.

On one sm_107 GPU, each target's prefill indexer over a paged index-key cache, with the
metadata's plan and real schedules: every scheduled logits call equals the same call without the
schedule bit for bit inside the windows, and the step's Top-K equals the Top-K of the same step
run without schedules.
"""

from __future__ import annotations
import __future__

import ast
import copy
import dataclasses
import functools
import importlib
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TARGETS = {
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "dep16": "kimi_k31_pro_nvfp4__sm_107__dep16",
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
}
# The layouts whose prefill target scores context rows (helix8 serves generation steps only).
_PREFILL = ("tep8", "dep16")
# The per-call logits budget (rows x keys of one call) and the indexer's heads.
_BUDGET = 1 << 29
_HEADS = 32
_TOKENS_PER_BLOCK = 64
_TEP8 = 8
_DEP16 = SimpleNamespace(tp_size=16, tp_rank=5, enable_attention_dp=True)
# The DSA layers two runs of the indexer stand for.
_LAYERS = (3, 7)


def _tep8(rank: int) -> SimpleNamespace:
    return SimpleNamespace(tp_size=_TEP8, tp_rank=rank, enable_attention_dp=False)


# --- the targets' code, run from modeling.py ---


@functools.lru_cache(maxsize=None)
def _tree(layout: str) -> ast.Module:
    path = _FAMILY / _TARGETS[layout] / "modeling.py"
    return ast.parse(path.read_text(), filename=str(path))


def _constants(layout: str, names: tuple[str, ...]) -> dict:
    """``names`` and the ``_K31*`` tables they derive from, evaluated from modeling.py."""
    ns: dict = {"MappingProxyType": MappingProxyType}
    for node in _tree(layout).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in names:
            ns[target.id] = eval(compile(ast.Expression(node.value), "<constant>", "eval"), ns)
        elif target.id.startswith("_K31"):
            try:
                ns[target.id] = eval(compile(ast.Expression(node.value), "<constant>", "eval"), ns)
            except NameError:
                continue
    missing = set(names) - set(ns)
    assert not missing, f"{layout}: modeling.py defines no {sorted(missing)}"
    return {n: ns[n] for n in names}


def _from_source(layout: str, defs: tuple[str, ...], consts: tuple[str, ...], ns: dict) -> dict:
    """The functions and classes ``defs`` of the target's modeling.py, executed over ``ns``."""
    ns = dict(ns, **_constants(layout, consts))
    body = [
        copy.deepcopy(n)
        for n in _tree(layout).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in defs
    ]
    missing = set(defs) - {n.name for n in body}
    assert not missing, f"{layout}: modeling.py defines no {sorted(missing)}"
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, f"{layout} modeling.py", "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _projection(layout: str):
    """The target's ``_dsa_step_args``: the step state its blocks read off the metadata."""
    defs = ("_DsaStep", "_sparse_args", "_rope_args", "_dsa_step_args")
    consts = ("_DSA_TOPK", "_DSA_TOPK_OUT")
    ns = {"torch": torch, "Any": Any, "NamedTuple": NamedTuple}
    return _from_source(layout, defs, consts, ns)["_dsa_step_args"]


def _prefill(layout: str, kernels: dict):
    """The target's ``_idx_prefill`` over ``kernels`` in place of its entries."""
    consts = ("_IDX_DIM", "_IDX_TOPK", "_IDX_GVR_DETERMINISTIC")
    ns = {"torch": torch, **kernels}
    return _from_source(layout, ("_idx_prefill_topk", "_idx_prefill"), consts, ns)["_idx_prefill"]


class _Logits:
    """A logits result's shape and layout, without values (the recorded Top-K reads nothing)."""

    def __init__(self, rows: int, keys: int):
        self.shape = (rows, keys)

    def stride(self, dim: int) -> int:
        return (self.shape[1], 1)[dim]

    def data_ptr(self) -> int:
        return 0


class _Kernels:
    """The entries the prefill indexer calls, recording their arguments; nothing is computed."""

    def __init__(self):
        self.calls: list[dict] = []
        self.gathers: list[tuple] = []
        self.allgathers: list[tuple] = []

    def namespace(self) -> dict:
        return {
            "k31_gather_k_cache_paged": self.gather,
            "fp8_mqa_logits": self.mqa_logits,
            "indexer_topk_prefill": lambda *a: None,
            "rubin_gvr_topk_prefill": lambda *a: None,
            "allgather_module": self.allgather,
        }

    def gather(self, k_cache, slot_fp8, slot_scale, k_token_start, num_k_tokens, head_dim):
        self.gathers.append((k_token_start, num_k_tokens, head_dim))
        return torch.zeros(num_k_tokens, 1, dtype=torch.uint8), torch.zeros(num_k_tokens, 1)

    def mqa_logits(self, q, k, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits, **kw):
        self.calls.append(
            dict(q=q, k=k, w=weights, ks=cu_seqlen_ks, ke=cu_seqlen_ke, clean=clean_logits, kw=kw)
        )
        return _Logits(q.shape[0], k.shape[0])

    def allgather(self, input, mapping, dim, sizes):
        self.allgathers.append((input.shape[0], tuple(sizes)))
        return input.new_zeros((sum(sizes),) + tuple(input.shape[1:]))


class _Schedules:
    """Stands in for ``rubin_mqa_logits_metadata``: a distinct tensor per schedule, kept with the
    arguments it was built from."""

    def __init__(self):
        self.built: dict[int, tuple] = {}

    def __call__(self, cu_seqlen_ks, cu_seqlen_ke, num_kv_tokens, num_heads):
        meta = torch.tensor([len(self.built)], dtype=torch.int32)
        self.built[id(meta)] = (
            meta,
            cu_seqlen_ks.clone(),
            cu_seqlen_ke.clone(),
            num_kv_tokens,
            num_heads,
        )
        return meta


# --- the metadata's plan ---


def _metadata_module():
    return importlib.import_module(f"{_PKG}.ops.attention.rubin_dsa_metadata")


def _bare(**attrs):
    """A family DSA metadata object holding only ``attrs``."""
    cls = _metadata_module().RubinDSAMetadata
    obj = object.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


@dataclasses.dataclass(frozen=True)
class _Step:
    """One prefill step of one rank: (cached keys, new rows) per context request."""

    name: str
    layout: str
    requests: tuple[tuple[int, int], ...]
    rank: int = 3
    chunked: bool = True  # MLA chunked prefill
    reuse: bool = True  # KV cache reuse
    skip: bool = False  # the context rows take the dense Top-K

    @property
    def mapping(self):
        return _tep8(self.rank) if self.layout == "tep8" else _DEP16

    @property
    def rows(self) -> int:
        return sum(q for _, q in self.requests)

    @property
    def kv_tokens(self) -> int:
        return sum(c + q for c, q in self.requests)


def _plan(step: _Step, monkeypatch, schedules=None, device: str = "cpu", md=None):
    """The metadata's prefill plan of ``step`` (``md`` reused when given, as the engine does)."""
    meta_mod = _metadata_module()
    engine_indexer = importlib.import_module(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.indexer"
    )
    if schedules is not None:
        monkeypatch.setattr(meta_mod, "rubin_mqa_logits_metadata", schedules)
    monkeypatch.setattr(
        engine_indexer.Indexer, "recompute_context_kv_gather_mappings", lambda md, p: None
    )
    q_lens = [q for _, q in step.requests]
    cached_indptr, kv_indptr = [0], [0]
    for c, q in step.requests:
        cached_indptr.append(cached_indptr[-1] + c)
        kv_indptr.append(kv_indptr[-1] + c + q)
    attrs = dict(
        sparse_metadata_params=SimpleNamespace(q_split_threshold=8192),
        enable_context_mla_with_cached_kv=step.reuse or step.chunked,
        runtime_features=SimpleNamespace(chunked_prefill=step.chunked),
        compress_ratios=[1],
        host_ctx_cached_token_indptr=torch.tensor(cached_indptr),
        host_ctx_kv_indptr=torch.tensor(kv_indptr),
        mapping=step.mapping,
        max_num_tokens=max(step.rows, 16384),
        kv_cache_manager=SimpleNamespace(max_seq_len=1 << 20),
        indexer_prefill_q_split=None,
        indexer_max_chunk_size=16384,
        skip_indexer_for_ctx_reqs=step.skip,
        cu_seqlen_ks=torch.zeros(1, dtype=torch.int32, device=device),
    )
    if md is None:
        md = _bare(**attrs)
    else:
        for key, value in attrs.items():
            setattr(md, key, value)
    params = SimpleNamespace(
        num_contexts=len(step.requests),
        seq_lens=torch.tensor(q_lens, dtype=torch.int32),
        kv_lens=torch.tensor([c + q for c, q in step.requests], dtype=torch.int32),
    )
    meta_mod._plan_prefill(md, params)
    return md


class _Md(SimpleNamespace):
    """What the step projection reads of the metadata: the plan's fields, the rest None."""

    def __getattr__(self, name):
        return None

    def get_indexer_max_seq_len(self):
        return 1 << 20

    def get_empty(self, buffers, shape, cache_name, dtype, capture_graph):
        return torch.zeros(shape[0], 1, dtype=dtype)


def _project(step: _Step, md, windows=None):
    """``(st, rt)`` of the step as the target's projection gives them to its blocks."""
    nc, nct = len(step.requests), step.rows
    ks, ke = windows if windows is not None else (torch.zeros(nct, dtype=torch.int32),) * 2
    view = _Md(
        is_ragged_verify=False,
        skip_indexer_for_gen_reqs=False,
        skip_indexer_for_ctx_reqs=md.skip_indexer_for_ctx_reqs,
        indexer_prefill_chunks=md.indexer_prefill_chunks,
        indexer_prefill_q_split=md.indexer_prefill_q_split,
        sparse_metadata_params=md.sparse_metadata_params,
        mapping=md.mapping,
        req_idx_per_token=torch.zeros(nct, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.zeros(nc, 1, dtype=torch.int32),
        kv_lens_cuda_runtime=torch.zeros(nc, dtype=torch.int32),
        block_table=torch.zeros(nc, 1, dtype=torch.int32),
        topk_indices_buffer=torch.zeros(nct, 1, dtype=torch.int32),
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        num_ctx_kv_tokens=step.kv_tokens,
        slot_mapping_fp8_fullkv=torch.zeros(step.kv_tokens, dtype=torch.int64),
        slot_mapping_scale_fullkv=torch.zeros(step.kv_tokens, dtype=torch.int64),
        max_seq_len=1 << 20,
        beam_width=1,
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=_TOKENS_PER_BLOCK,
            kv_cache_pool_pointers=None,
            kv_cache_pool_mapping=None,
        ),
    )
    pools = SimpleNamespace(tokens_per_block=_TOKENS_PER_BLOCK)
    dsa = _projection(step.layout)(view, pools, nct, nc, nct, 0)
    st = SimpleNamespace(dsa=dsa, num_ctx_tokens=nct, md=view)
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(idx_k_cache={i: f"layer {i}" for i in _LAYERS}))
    return st, rt


def _run(step: _Step, st, rt):
    """The target's prefill indexer over the step's layers, with recorded kernels."""
    kernels = _Kernels()
    run = _prefill(step.layout, kernels.namespace())
    nct = step.rows
    q8 = torch.arange(nct, dtype=torch.int32)[:, None, None].expand(nct, _HEADS, 1)
    w = torch.arange(nct, dtype=torch.int32)[:, None].expand(nct, _HEADS)
    k8 = torch.zeros(step.kv_tokens, 1, dtype=torch.uint8)
    ks = torch.zeros(step.kv_tokens, 1)
    for i in _LAYERS:
        run(i, q8, w, k8, ks, st, rt)
    return kernels


def _rows(call) -> tuple[int, int]:
    """The step rows a recorded call scored (its queries carry their row numbers)."""
    q = call["q"]
    return int(q[0, 0, 0]), int(q[-1, 0, 0]) + 1


def _rank_rows(step: _Step, st, chunk, index: int) -> tuple[int, int]:
    """The chunk rows this rank scores, as the projection gives them."""
    if step.layout == "tep8":
        q_start, q_end, _ = st.dsa.chunk_q_split[index]
        return q_start, q_end
    return 0, chunk.token_end - chunk.token_start


def _budget_tiles(q_start: int, q_end: int, keys: int) -> list[tuple[int, int]]:
    """The query tiles of rows ``[q_start, q_end)`` of an unscheduled call: rows x keys of one
    call within the per-call budget."""
    tile = max(1, min(q_end - q_start, _BUDGET // max(1, keys)))
    return [(c0, min(c0 + tile, q_end)) for c0 in range(q_start, q_end, tile)]


# --- CPU: the schedule reaches every call it was built for ---

_STEPS = (
    _Step("tep8 batch plan rank 0, 16k rows, no prefix", "tep8", ((0, 16384),), rank=0),
    _Step("tep8 batch plan rank 3, 16k rows behind 64k", "tep8", ((65536, 16384),), rank=3),
    _Step("tep8 batch plan rank 7, 16k rows behind 283k", "tep8", ((282624, 16384),), rank=7),
    _Step(
        "tep8 batch plan rank 4, its rows cut across two requests",
        "tep8",
        ((70000, 9000), (0, 7384)),
        rank=4,
    ),
    _Step(
        "tep8 TP query split per chunk rank 3, 16k behind 64k (no MLA chunked prefill)",
        "tep8",
        ((65536, 16384),),
        rank=3,
        chunked=False,
    ),
    _Step("dep16 16k rows behind 86k", "dep16", ((86016, 16384),)),
    _Step("dep16 16k rows behind 131k", "dep16", ((131072, 16384),)),
    _Step("dep16 16k rows behind 197k", "dep16", ((196608, 16384),)),
    _Step("dep16 16k rows behind 283k", "dep16", ((282624, 16384),)),
    _Step("dep16 packed chunk of 8 prompts", "dep16", tuple((0, 2048) for _ in range(8))),
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("step", _STEPS, ids=lambda s: s.name)
def test_every_logits_call_gets_the_schedule_built_for_it(step, monkeypatch):
    schedules = _Schedules()
    md = _plan(step, monkeypatch, schedules)
    st, rt = _project(step, md)
    kernels = _run(step, st, rt)
    chunks = md.indexer_prefill_chunks
    assert chunks, "the metadata planned no chunk"

    # The unscheduled calls: per chunk with keys, the budget's tiles
    # of the rank's rows, over the chunk's gathered keys.
    expected = []
    for index, chunk in enumerate(chunks):
        keys = chunk.k_token_end - chunk.k_token_start
        if keys <= 0:
            continue
        q_start, q_end = _rank_rows(step, st, chunk, index)
        assert (chunk.q_start, chunk.q_end) == (q_start, q_end), (index, chunk.q_start, chunk.q_end)
        # The tiles cover the rank's rows of the chunk once, in order, under the budget.
        assert chunk.tiles == tuple(_budget_tiles(q_start, q_end, keys)), (index, chunk.tiles)
        assert chunk.tiles[0][0] == q_start and chunk.tiles[-1][1] == q_end
        assert all(a[1] == b[0] for a, b in zip(chunk.tiles, chunk.tiles[1:]))
        expected += [(chunk, c0, c1, keys, j) for j, (c0, c1) in enumerate(chunk.tiles)]
    assert len(kernels.calls) == len(_LAYERS) * len(expected) > 0
    assert len(schedules.built) == len(expected), "one schedule per tile, once a step"

    for n, call in enumerate(kernels.calls):
        chunk, c0, c1, keys, j = expected[n % len(expected)]
        t0 = chunk.token_start
        # Every argument but the schedule: the tile's rows and windows, the chunk's keys.
        assert _rows(call) == (t0 + c0, t0 + c1), n
        assert call["w"].shape[0] == c1 - c0 and int(call["w"][0, 0]) == t0 + c0
        assert call["ks"].data_ptr() == chunk.cu_seqlen_ks[c0:c1].data_ptr()
        assert call["ke"].data_ptr() == chunk.cu_seqlen_ke[c0:c1].data_ptr()
        assert call["ks"].shape == (c1 - c0,) and call["ke"].shape == (c1 - c0,)
        assert call["k"].shape[0] == keys and call["clean"] is False
        # The schedule: the metadata's object for this tile, built for these windows, keys
        # and heads; every layer of the step reads the same one.
        assert set(call["kw"]) == {"schedule_meta"}, call["kw"].keys()
        meta = call["kw"]["schedule_meta"]
        assert meta is chunk.mqa_meta[j], (n, j)
        built, ks, ke, num_kv, heads = schedules.built[id(meta)]
        assert torch.equal(ks, call["ks"]) and torch.equal(ke, call["ke"])
        assert (num_kv, heads) == (call["k"].shape[0], call["q"].shape[1]) == (keys, _HEADS)

    # Each chunk's keys are gathered once per layer.
    assert kernels.gathers == [
        (c.k_token_start, c.k_token_end - c.k_token_start, 128)
        for _ in _LAYERS
        for c in chunks
        if c.k_token_end > c.k_token_start
    ]


@pytest.mark.cpu_only
def test_each_tep8_rank_scores_its_eighth_and_gathers_the_steps_top_k():
    """Ranks 0..7 of a 16k step: each scores its eighth of the rows, then all-gathers them."""
    seen = []
    for rank in range(_TEP8):
        step = _Step(f"rank {rank}", "tep8", ((65536, 16384),), rank=rank)
        with pytest.MonkeyPatch.context() as mp:
            md = _plan(step, mp, _Schedules())
            st, rt = _project(step, md)
            kernels = _run(step, st, rt)
        assert md.indexer_prefill_q_split == (rank * 2048, (rank + 1) * 2048, [2048] * _TEP8)
        assert st.dsa.chunk_q_split == ((0, 2048, None),)
        rows = {_rows(c) for c in kernels.calls}
        assert rows == {(rank * 2048, (rank + 1) * 2048)}, rows
        assert kernels.allgathers == [(2048, (2048,) * _TEP8)] * len(_LAYERS)
        seen += sorted(rows)
    assert seen == [(r * 2048, (r + 1) * 2048) for r in range(_TEP8)]


@pytest.mark.cpu_only
def test_the_next_step_reads_only_its_own_schedules(monkeypatch):
    """The metadata object is reused across steps; the second step's calls carry none of the
    first step's schedules."""
    first = _Step("first", "dep16", ((86016, 16384),))
    second = _Step("second", "dep16", ((282624, 16384),))
    s1, s2 = _Schedules(), _Schedules()
    md = _plan(first, monkeypatch, s1)
    k1 = _run(first, *_project(first, md))
    md = _plan(second, monkeypatch, s2, md=md)
    k2 = _run(second, *_project(second, md))
    ids1 = {id(c["kw"]["schedule_meta"]) for c in k1.calls}
    ids2 = {id(c["kw"]["schedule_meta"]) for c in k2.calls}
    assert ids1 == set(s1.built) and ids2 == set(s2.built)
    assert not ids1 & ids2
    assert len(k2.calls) == len(_LAYERS) * 10  # 10 tiles of 1795 rows behind 283k


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", _PREFILL)
def test_a_plan_without_schedules_calls_without_one_and_nothing_else_changes(layout, monkeypatch):
    requests = ((65536, 16384),) if layout == "tep8" else ((282624, 16384),)
    step = _Step("no schedules", layout, requests)
    md = _plan(step, monkeypatch, _Schedules())
    scheduled = _run(step, *_project(step, md))
    md.indexer_prefill_chunks = [
        dataclasses.replace(c, mqa_meta=()) for c in md.indexer_prefill_chunks
    ]
    plain = _run(step, *_project(step, md))
    assert len(plain.calls) == len(scheduled.calls) > 0
    for a, b in zip(plain.calls, scheduled.calls):
        assert a["kw"] == {"schedule_meta": None}
        assert _rows(a) == _rows(b) and a["k"].shape == b["k"].shape
        assert torch.equal(a["ks"], b["ks"]) and torch.equal(a["ke"], b["ke"])
        assert a["clean"] is b["clean"] is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", _PREFILL)
def test_a_step_the_metadata_did_not_chunk_runs_one_call_without_a_schedule(layout, monkeypatch):
    """Two short prompts without cache reuse or chunked prefill: one pass over the step's own
    keys."""
    step = _Step("one pass", layout, ((0, 9), (0, 5)), chunked=False, reuse=False)
    schedules = _Schedules()
    md = _plan(step, monkeypatch, schedules)
    assert md.indexer_prefill_chunks is None and not schedules.built
    ks = torch.tensor([0] * 9 + [9] * 5, dtype=torch.int32)
    ke = torch.tensor(list(range(1, 10)) + list(range(10, 15)), dtype=torch.int32)
    kernels = _run(step, *_project(step, md, (ks, ke)))
    assert len(kernels.calls) == len(_LAYERS) and not kernels.gathers
    for call in kernels.calls:
        assert call["kw"] == {"schedule_meta": None}
        assert _rows(call) == (0, 14) and call["k"].shape[0] == 14
        assert torch.equal(call["ks"], ks) and torch.equal(call["ke"], ke)


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", _PREFILL)
def test_a_dense_top_k_step_builds_and_reads_no_schedule(layout, monkeypatch):
    step = _Step("dense Top-K", layout, ((0, 1024),), skip=True)
    schedules = _Schedules()
    md = _plan(step, monkeypatch, schedules)
    assert not schedules.built
    assert all(c.mqa_meta == () for c in md.indexer_prefill_chunks)
    st, rt = _project(step, md)
    assert st.dsa.skip_ctx
    assert not _run(step, st, rt).calls


# --- CPU: where the call is, and what reaches it ---


def _scopes(tree: ast.Module):
    """``(qualified name, node)`` of every module-level function and class method."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield f"{node.name}.{item.name}", item


def _callers(tree: ast.Module) -> dict[str, set[str]]:
    """Name -> the functions and methods that call it by name."""
    found: dict[str, set[str]] = {}
    for scope, node in _scopes(tree):
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                found.setdefault(inner.func.id, set()).add(scope)
    return found


def _reach(tree: ast.Module, roots: set[str]) -> set[str]:
    """The module-level functions a set of scopes reaches through calls by name."""
    calls: dict[str, set[str]] = {}
    for scope, node in _scopes(tree):
        calls[scope] = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
    seen, todo = set(), list(roots)
    while todo:
        scope = todo.pop()
        if scope in seen:
            continue
        seen.add(scope)
        todo += [c for c in calls.get(scope, ()) if c in calls]
    return seen


def _reaching(tree: ast.Module, target: str) -> set[str]:
    """The functions and methods that reach ``target`` through calls by name."""
    callers = _callers(tree)
    seen, todo = set(), [target]
    while todo:
        name = todo.pop()
        for scope in callers.get(name, ()):
            if scope not in seen:
                seen.add(scope)
                todo.append(scope)
    return seen


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", _PREFILL)
def test_the_logits_call_passes_the_schedule_and_only_the_prefill_reaches_it(layout):
    tree = _tree(layout)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "fp8_mqa_logits"
    ]
    assert len(calls) == 1, f"{layout}: {len(calls)} fp8_mqa_logits calls"
    assert [k.arg for k in calls[0].keywords] == ["schedule_meta"]
    callers = _callers(tree)
    assert callers["fp8_mqa_logits"] == {"_idx_prefill_topk"}
    assert callers["_idx_prefill_topk"] == {"_idx_prefill"}
    # Every scope that reaches the prefill indexer is the prefill target or under it.
    reaching = _reaching(tree, "_idx_prefill")
    assert {"mla_prefill", "PrefillTarget.forward"} <= reaching, sorted(reaching)
    assert not any(s.startswith("DecodeTarget.") or s == "mla_generate" for s in reaching)
    assert callers["mla_prefill"] == {"PrefillTarget.forward"}
    decode = {s for s, _ in _scopes(tree) if s.startswith("DecodeTarget.")} | {"mla_generate"}
    reached = _reach(tree, decode)
    assert not reached & {"_idx_prefill", "_idx_prefill_topk", "mla_prefill"}, sorted(reached)


@pytest.mark.cpu_only
def test_helix8_makes_no_prefill_logits_call():
    tree = _tree("helix8")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = {
        a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert "fp8_mqa_logits" not in names | imported


# --- CPU: the entry ---


def _entry_module():
    return importlib.import_module(f"{_PKG}.ops.attention.fp8_mqa_logits")


def _entry_args():
    q = torch.randint(-3, 4, (6, 2, 128)).float().to(torch.float8_e4m3fn)
    k = torch.randint(-3, 4, (9, 128)).float().to(torch.float8_e4m3fn)
    k_scale = torch.full((9,), 0.5)
    w = torch.randint(-3, 4, (6, 2)).float() / 16
    ks = torch.tensor([0, 0, 0, 4, 4, 4], dtype=torch.int32)
    ke = torch.tensor([1, 2, 3, 6, 8, 9], dtype=torch.int32)
    return q, k, k_scale, w, ks, ke


@pytest.mark.cpu_only
def test_the_entry_sends_each_call_to_its_deepgemm_function(monkeypatch):
    mod = _entry_module()
    seen = []
    monkeypatch.setattr(
        mod, "_deep_gemm_fp8_mqa_logits", lambda *a, **k: seen.append(("unscheduled", a, k)) or "L"
    )
    monkeypatch.setattr(
        mod, "_deep_gemm_fp8_fp4_mqa_logits", lambda *a, **k: seen.append(("unified", a, k)) or "U"
    )
    q, k, k_scale, w, ks, ke = _entry_args()
    meta = torch.tensor([5, 7], dtype=torch.int32)
    assert mod.fp8_mqa_logits(q, k, k_scale, w, ks, ke, False) == "L"
    assert mod.fp8_mqa_logits(q, k, k_scale, w, ks, ke, True, schedule_meta=meta) == "U"
    (kind, a, kw), (kind2, a2, kw2) = seen
    # Without a schedule: DeepGEMM's fp8_mqa_logits with the unscheduled call's arguments.
    assert kind == "unscheduled" and set(kw) == {"clean_logits"} and kw["clean_logits"] is False
    assert len(a) == 5 and a[0] is q and len(a[1]) == 2
    assert a[1][0] is k and a[1][1] is k_scale
    assert all(x is y for x, y in zip(a[2:], (w, ks, ke)))
    # With one: fp8_fp4_mqa_logits with no query scale, fp32 logits and the schedule.
    assert kind2 == "unified"
    assert set(kw2) == {"clean_logits", "logits_dtype", "schedule_meta"}
    assert kw2["clean_logits"] is True and kw2["logits_dtype"] == torch.float32
    assert kw2["schedule_meta"] is meta
    assert len(a2) == 5 and a2[0][0] is q and a2[0][1] is None
    assert a2[1][0] is k and a2[1][1] is k_scale
    assert all(x is y for x, y in zip(a2[2:], (w, ks, ke)))


@pytest.mark.cpu_only
def test_the_entry_reference_ignores_the_schedule():
    op = _entry_module().fp8_mqa_logits
    args = _entry_args()
    plain = op.reference(*args, False)
    scheduled = op.reference(*args, False, schedule_meta=torch.tensor([1], dtype=torch.int32))
    assert torch.equal(plain, scheduled)


@pytest.mark.cpu_only
def test_the_entry_guard_refuses_a_schedule_of_another_call(monkeypatch):
    mod = _entry_module()

    def built(ks, ke, num_kv_tokens, num_heads):
        return torch.cat([ks, ke, torch.tensor([num_kv_tokens, num_heads], dtype=torch.int32)])

    monkeypatch.setattr(mod, "_deep_gemm_mqa_logits_metadata", built)
    q, k, k_scale, w, ks, ke = _entry_args()
    op = mod.fp8_mqa_logits
    op.is_valid(q, k, k_scale, w, ks, ke, False, schedule_meta=built(ks, ke, 9, 2))
    op.is_valid(q, k, k_scale, w, ks, ke, False)
    for other in (built(ks, ke - 1, 9, 2), built(ks, ke, 10, 2), built(ks, ke, 9, 4)):
        with pytest.raises(AssertionError, match="another call"):
            op.is_valid(q, k, k_scale, w, ks, ke, False, schedule_meta=other)


# --- GPU: bit for bit against the step without schedules ---

_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin targets' prefill indexer runs on sm_107",
)

_GPU_STEPS = (
    _Step("tep8 rank 3 of a 16k step behind 64k", "tep8", ((65536, 16384),), rank=3),
    _Step("tep8 rank 3 of a 16k step behind 283k", "tep8", ((282624, 16384),), rank=3),
    _Step("dep16 16k behind 86k", "dep16", ((86016, 16384),)),
    _Step("dep16 16k behind 283k", "dep16", ((282624, 16384),)),
    _Step("dep16 packed chunk of 8 prompts", "dep16", tuple((0, 2048) for _ in range(8))),
)


def _index_cache(kv_tokens: int, gen: torch.Generator):
    """A paged index-key cache holding ``kv_tokens`` keys on random slots: the [pages, P, 1, 132]
    page view and the int64 byte offsets of each key and of its fp32 scale."""
    page_bytes = _TOKENS_PER_BLOCK * 132
    pages = -(-kv_tokens // _TOKENS_PER_BLOCK) + 8
    base = torch.randint(0, 256, (pages, page_bytes), generator=gen, device="cuda")
    base = base.to(torch.uint8)
    slots = torch.randperm(pages * _TOKENS_PER_BLOCK, generator=gen, device="cuda")[:kv_tokens]
    page, pos = slots // _TOKENS_PER_BLOCK, slots % _TOKENS_PER_BLOCK
    fp8 = page * page_bytes + pos * 128
    scale = page * page_bytes + _TOKENS_PER_BLOCK * 128 + pos * 4
    raw = torch.randn(kv_tokens, 128, generator=gen, device="cuda")
    amax = raw.abs().amax(dim=1).clamp_min(1e-4)
    scales = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    keys = (raw / scales[:, None]).to(torch.float8_e4m3fn)
    flat = base.view(-1)
    flat[(fp8[:, None] + torch.arange(128, device="cuda")).reshape(-1)] = keys.view(
        torch.uint8
    ).reshape(-1)
    flat[(scale[:, None] + torch.arange(4, device="cuda")).reshape(-1)] = (
        scales.contiguous().view(torch.uint8).reshape(-1)
    )
    cache = base.view(pages, _TOKENS_PER_BLOCK, 1, 132)
    return cache, fp8.to(torch.int64), scale.to(torch.int64)


@_ON_SM107
@pytest.mark.parametrize("step", _GPU_STEPS, ids=lambda s: s.name)
def test_the_scheduled_prefill_is_the_unscheduled_one_bit_for_bit(step, monkeypatch):
    model = importlib.import_module(f"{_PKG}.{_TARGETS[step.layout]}.modeling")
    md = _plan(step, monkeypatch, device="cuda")
    chunks = md.indexer_prefill_chunks
    assert chunks and all(len(c.mqa_meta) == len(c.tiles) > 0 for c in chunks)
    gen = torch.Generator(device="cuda").manual_seed(sum(map(ord, step.name)))
    cache, slot_fp8, slot_scale = _index_cache(step.kv_tokens, gen)
    nct = step.rows
    q8 = torch.randn(nct, _HEADS, 128, generator=gen, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(nct, _HEADS, generator=gen, device="cuda")
    rt = SimpleNamespace(dsa_pools=SimpleNamespace(idx_k_cache={3: cache}))

    def state(plan):
        dsa = SimpleNamespace(
            skip_ctx=False,
            topk_dense=None,
            topk_out=torch.full((nct, 2048), -7, dtype=torch.int32, device="cuda"),
            chunks=plan,
            num_ctx_kv_tokens=step.kv_tokens,
            cu_ks=None,
            cu_ke=None,
            k_slot_fp8=slot_fp8,
            k_slot_scale=slot_scale,
            # One GPU: each chunk's rows are this rank's (the batch plan); no TP gather.
            chunk_q_split=tuple((0, c.token_end - c.token_start, None) for c in plan),
            q_split=None,
        )
        return SimpleNamespace(dsa=dsa, num_ctx_tokens=nct, md=SimpleNamespace(mapping=None))

    real = model.fp8_mqa_logits
    compared = []

    def logits_against_unscheduled(q, k, k_scale, w, ks, ke, clean_logits, *, schedule_meta=None):
        out = real(q, k, k_scale, w, ks, ke, clean_logits, schedule_meta=schedule_meta)
        assert schedule_meta is not None, "a planned tile ran without its schedule"
        plain = real(q, k, k_scale, w, ks, ke, clean_logits)
        cols = torch.arange(out.shape[1], device="cuda")[None, :]
        window = (cols >= ks.long()[:, None]) & (cols < ke.long()[:, None])
        compared.append(
            torch.equal(
                torch.where(window, out.view(torch.int32), 0),
                torch.where(window, plain.view(torch.int32), 0),
            )
        )
        return out

    monkeypatch.setattr(model, "fp8_mqa_logits", logits_against_unscheduled)
    scheduled = model._idx_prefill(3, q8, w, None, None, state(chunks), rt)
    monkeypatch.setattr(model, "fp8_mqa_logits", real)
    bare = [dataclasses.replace(c, mqa_meta=()) for c in chunks]
    plain = model._idx_prefill(3, q8, w, None, None, state(bare), rt)
    torch.cuda.synchronize()
    tiles = sum(len(c.tiles) for c in chunks)
    assert len(compared) == tiles and all(compared), compared
    for c in chunks:
        rows = slice(c.token_start, c.token_end)
        assert not torch.eq(scheduled[rows], -7).any(), "rows left unscored"
        assert torch.equal(scheduled[rows], plain[rows]), (c.token_start, c.token_end)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generation server's steps through tep8's and dep16's DSA step projection (CPU, from source).

A disaggregated generation server runs without block reuse and without chunked prefill, so its
metadata has no cached-KV context MLA. The engine's DSA metadata then holds the context rows'
full-KV gather slots (``slot_mapping_fp8_fullkv`` / ``slot_mapping_scale_fullkv``) only once it
has prepared a step with context requests: its allocation creates them only under cached-KV
context MLA, and the draft forward that also aliases them runs only with a separate draft cache,
which the DFlash2 drafter does not use. A generation server prepares no context step, from its
first forward (the attention warmup's generation request) on.

The family metadata's step preparation (``_prepare_indexer``, ``_plan_prefill`` and the engine's
``recompute_context_kv_gather_mappings``) and tep8's and dep16's ``_dsa_step_args`` run from their
source, with the device leaves stood in.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_BACKENDS = _PKG_ROOT / "_torch/attention/backends"
_ENGINE_DSA = _BACKENDS / "sparse/dsa"
_TARGETS = {
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "dep16": "kimi_k31_pro_nvfp4__sm_107__dep16",
}
_SLOTS = ("slot_mapping_fp8_fullkv", "slot_mapping_scale_fullkv")
_FLAG = "enable_context_mla_with_cached_kv"


def _scope(path: Path, cls: str | None = None) -> list[ast.stmt]:
    tree = ast.parse(path.read_text())
    if cls is None:
        return tree.body
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body


def _defs(path: Path, names: set[str], cls: str | None = None) -> list[ast.stmt]:
    """The definitions ``names`` (functions, classes, constants) of ``path`` (of class ``cls``) in
    file order, decorators dropped."""
    out, seen = [], set()
    for node in _scope(path, cls):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            out.append(node)
            seen.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & names:
                out.append(copy.deepcopy(node))
                seen |= targets
    assert not names - seen, f"{path.name}: {sorted(names - seen)} not found"
    return out


def _run(nodes: list[ast.stmt], ns: dict, filename: str) -> dict:
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), filename, "exec"), ns)
    return ns


# --- what the engine's metadata holds on a generation server ---


def _guards(fn: ast.FunctionDef, attr: str) -> list[bool]:
    """For each ``self.<attr> = ...`` in ``fn``: whether it sits under ``if self.<_FLAG>:``."""
    found = []

    def visit(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.If):
            inner = guarded or ast.unparse(node.test) == f"self.{_FLAG}"
            for child in node.body:
                visit(child, inner)
            for child in node.orelse:
                visit(child, guarded)
            return
        if isinstance(node, ast.Assign):
            found.extend(guarded for t in node.targets if ast.unparse(t) == f"self.{attr}")
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    for stmt in fn.body:
        visit(stmt, False)
    return found


def test_the_allocation_creates_the_slots_only_under_cached_kv_context_mla():
    """The engine's ``create_buffers_for_indexer`` (the family metadata inherits it) assigns the
    slots only under ``enable_context_mla_with_cached_kv``, and no class of the chain declares
    them: a metadata built without the flag has no such attribute until a step sets it."""
    alloc = next(
        n
        for n in _scope(_ENGINE_DSA / "metadata.py", "DSAtrtllmAttentionMetadata")
        if isinstance(n, ast.FunctionDef) and n.name == "create_buffers_for_indexer"
    )
    for attr in _SLOTS:
        guards = _guards(alloc, attr)
        assert guards and all(guards), f"{attr}: assigned outside the flag ({guards})"
    chain = (
        (_FAMILY / "ops/attention/rubin_dsa_metadata.py", "RubinDSAMetadata"),
        (_ENGINE_DSA / "metadata.py", "DSAtrtllmAttentionMetadata"),
        (_BACKENDS / "trtllm.py", "TrtllmAttentionMetadata"),
        (_BACKENDS / "interface.py", "AttentionMetadata"),
    )
    for path, cls in chain:
        declared = {
            ast.unparse(t)
            for n in _scope(path, cls)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        }
        assert not declared & set(_SLOTS), f"{cls} declares {declared & set(_SLOTS)}"


def _prepared(nc: int, ng: int, next_n: int, *, cached_kv_mla: bool = False) -> SimpleNamespace:
    """One step's indexer preparation by the family metadata's code: ``nc`` fresh context
    requests of 300 tokens, then ``ng`` generation requests of ``next_n`` rows behind 1000 cached
    tokens, on what the allocation leaves (the slots None under cached-KV context MLA, else
    absent). The device leaves (cache slot writes, decode schedules, prefill tiles) stand in."""
    seq = [300] * nc + [next_n] * ng
    past = [0] * nc + [1000] * ng
    params = SimpleNamespace(
        num_contexts=nc,
        seq_lens=torch.tensor(seq),
        new_kv_tokens=torch.tensor(seq),
        kv_lens=torch.tensor(seq) + torch.tensor(past),
        cached_kv_tokens=torch.tensor(past),
        _num_past_tokens_tensor=torch.tensor(past),
    )
    indexer = SimpleNamespace(
        build_indexer_params=lambda md: params,
        prepare_for_update_k_cache=lambda md, p: None,
    )
    engine = _run(
        _defs(_ENGINE_DSA / "indexer.py", {"recompute_context_kv_gather_mappings"}, "Indexer"),
        {"torch": torch, "Indexer": indexer, "Optional": Any, "IndexerParams": Any},
        "indexer.py",
    )
    indexer.recompute_context_kv_gather_mappings = engine["recompute_context_kv_gather_mappings"]
    family = _run(
        _defs(
            _FAMILY / "ops/attention/rubin_dsa_metadata.py",
            {"_prepare_indexer", "_plan_prefill", "_index_slots_in_model"},
        ),
        {
            "torch": torch,
            "Indexer": indexer,
            "_effective_compress_ratio_divisor": lambda ratio: 1,
            "_select_indexer_compress_ratio": lambda ratios: 1,
            "compute_cu_seqlen_kv_bounds_with_cache": lambda lens, nc, nct, *a: (
                torch.zeros(nct, dtype=torch.int32),
                torch.zeros(nct, dtype=torch.int32),
            ),
            "maybe_pin_memory": lambda t: t,
            "select_indexer_chunk_size": lambda size, max_k: size,
            "split_prefill_chunks": lambda lens, size, start_idx: [list(range(len(lens)))],
            "_decode_schedules": lambda md, lens: None,
        },
        "rubin_dsa_metadata.py",
    )
    rows = sum(seq)
    md = SimpleNamespace(
        num_contexts=nc,
        num_ctx_tokens=sum(seq[:nc]),
        num_generations=ng,
        seq_lens=torch.tensor(seq),
        compress_ratios=[1],
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(rows, dtype=torch.int32),
        sparse_metadata_params=SimpleNamespace(q_split_threshold=8192),
        runtime_features=SimpleNamespace(chunked_prefill=cached_kv_mla),
        indexer_max_chunk_size=16384,
        kv_lens_cuda_runtime=torch.tensor(seq) + torch.tensor(past),
        get_indexer_kv_lens=lambda lens: lens,
        skip_indexer_for_ctx_reqs=False,
        slot_mapping_fp8=torch.zeros(rows, dtype=torch.int64),
        slot_mapping_scale=torch.zeros(rows, dtype=torch.int64),
    )
    setattr(md, _FLAG, cached_kv_mla)
    if cached_kv_mla:  # what create_buffers_for_indexer leaves under the flag
        for attr in _SLOTS:
            setattr(md, attr, None)
    family["_prepare_indexer"](md)
    return md


def test_a_generation_step_leaves_the_slots_unset():
    """A generation step of one request verifying 7 drafts, and one of 12 requests: no slots."""
    for ng in (1, 12):
        md = _prepared(0, ng, 8)
        assert not any(hasattr(md, attr) for attr in _SLOTS), ng


def test_a_context_or_mixed_step_sets_the_slots():
    """With context requests the plan aliases the step's own slot mappings (no cached KV)."""
    for nc, ng in ((1, 0), (2, 0), (1, 2)):
        md = _prepared(nc, ng, 8)
        assert md.slot_mapping_fp8_fullkv is md.slot_mapping_fp8, (nc, ng)
        assert md.slot_mapping_scale_fullkv is md.slot_mapping_scale, (nc, ng)


def test_under_cached_kv_context_mla_a_generation_step_keeps_them_none():
    """With cached-KV context MLA the allocation creates the slots; a generation step leaves them
    None."""
    md = _prepared(0, 1, 8, cached_kv_mla=True)
    assert all(getattr(md, attr) is None for attr in _SLOTS)


# --- the products' step projection on that metadata ---


def _projection(layout: str):
    """``_dsa_step_args`` of ``layout`` and what it reads, run from the product's source."""
    names = {
        "_DsaStep",
        "_sparse_args",
        "_rope_args",
        "_dsa_step_args",
        "_DSA_TOPK",
        "_DSA_TOPK_OUT",
    }
    path = _FAMILY / _TARGETS[layout] / "modeling.py"
    ns = {
        "torch": torch,
        "Any": Any,
        "NamedTuple": NamedTuple,
        "_DsaPools": object,
        "_K31_DSA": {"index_topk": 2048},
    }
    return _run(_defs(path, names), ns, f"{_TARGETS[layout]}/modeling.py")["_dsa_step_args"]


class _StepMd(SimpleNamespace):
    """A step's metadata as the projection reads it: the buffers it slices are tensors, the
    op-argument fields it only hands through read as None, and a name in ``_absent`` (what the
    engine's metadata does not hold at this step) raises AttributeError as the engine's does."""

    def __getattr__(self, name: str):
        if name.startswith("__") or name in self.__dict__.get("_absent", ()):
            raise AttributeError(f"'RubinDSAMetadata' object has no attribute '{name}'")
        return None

    def get_indexer_max_seq_len(self) -> int:
        return 4096

    def get_empty(self, buffers, shape, cache_name, dtype, capture_graph=False):
        return torch.empty(shape, dtype=dtype)


def _step_md(nc: int, nct: int, ng: int, next_n: int, absent=()) -> _StepMd:
    rows, seqs = nct + ng * next_n, nc + ng

    def ints(*shape):
        return torch.zeros(*shape, dtype=torch.int32)

    md = _StepMd(
        _absent=frozenset(absent),
        is_ragged_verify=False,
        is_cuda_graph=False,
        skip_indexer_for_gen_reqs=False,
        skip_indexer_for_ctx_reqs=False,
        slot_mapping_fp8=torch.zeros(rows, dtype=torch.int64),
        slot_mapping_scale=torch.zeros(rows, dtype=torch.int64),
        gen_indexer_kv_lens_cuda_runtime=ints(ng),
        indexer_k_cache_block_offsets=ints(seqs, 4),
        scheduler_metadata_buffer=ints(64, 2),
        kv_lens_cuda_runtime=ints(seqs),
        topk_indices_buffer=ints(rows, 2048),
        radix_aux_indices=ints(1),
        radix_aux_logits=ints(1),
        req_idx_per_token=ints(rows),
        block_table=ints(seqs, 4),
        cu_seqlen_ks=ints(max(nct, 1)),
        cu_seqlen_ke=ints(max(nct, 1)),
        num_ctx_kv_tokens=nct,
        max_seq_len=4096,
        beam_width=1,
        kv_cache_manager=SimpleNamespace(
            kv_cache_pool_pointers=None, kv_cache_pool_mapping=None, tokens_per_block=64
        ),
    )
    if nc:
        md.slot_mapping_fp8_fullkv = torch.zeros(nct, dtype=torch.int64)
        md.slot_mapping_scale_fullkv = torch.zeros(nct, dtype=torch.int64)
    return md


def _pools() -> SimpleNamespace:
    return SimpleNamespace(tokens_per_block=64)


# (layout, generation requests, rows per request): one request (b1 k7), a TEP8 graph batch,
# 12 requests (b12 k7) and a one-row step.
_GEN_STEPS = (("tep8", 1, 8), ("tep8", 4, 8), ("dep16", 12, 8), ("dep16", 1, 1))


@pytest.mark.parametrize(
    "layout, ng, next_n", _GEN_STEPS, ids=["tep8_b1k7", "tep8_b4k7", "dep16_b12k7", "dep16_b1"]
)
def test_a_generation_step_projects_without_the_slots(layout, ng, next_n):
    """The generation server's first forward and every later one: the projection reads nothing
    the prepared metadata lacks, and its step carries no full-KV slots."""
    absent = [attr for attr in _SLOTS if not hasattr(_prepared(0, ng, next_n), attr)]
    assert absent == list(_SLOTS)
    md = _step_md(0, 0, ng, next_n, absent=absent)
    st = _projection(layout)(md, _pools(), ng * next_n, 0, 0, ng)
    assert st.k_slot_fp8 is None and st.k_slot_scale is None
    assert st.next_n == next_n


@pytest.mark.parametrize("layout", sorted(_TARGETS))
@pytest.mark.parametrize("ng", [0, 2], ids=["context", "mixed"])
def test_a_context_step_hands_its_slots_through(layout, ng):
    """The projection hands a context step's full-KV gather slots to its blocks."""
    md = _step_md(1, 300, ng, 8)
    st = _projection(layout)(md, _pools(), 300 + 8 * ng, 1, 300, ng)
    assert st.k_slot_fp8 is md.slot_mapping_fp8_fullkv
    assert st.k_slot_scale is md.slot_mapping_scale_fullkv

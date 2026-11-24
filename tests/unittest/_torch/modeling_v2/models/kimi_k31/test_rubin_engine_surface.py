# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the specialized Kimi K3.1 Rubin targets need from the engine, and what they leave in it.

The targets live in modeling_v2; outside it the engine carries only the runtime changes they
need, each with a reason. Read from the source with ``ast`` unless a check says otherwise:

* the engine has what the targets inherit or call: the DFlash worker methods the drafter's
  worker keeps, the DSA metadata methods the family metadata keeps (with the device-refresh
  switch of the speculative update), the cache-provider flag the engine's DSA code reads, and
  every engine name the targets and the runtime helpers import;
* no engine file names the model (the usage allowlist excepted);
* helix CP warmup runs no context row through a target (its prefill target refuses them): the
  general and memory-pool warmups, the attention warmup's context batches and the Mamba warmup
  are off under helix; the autotuner tunes on generation requests; the KV cache estimation
  refuses context parallelism;
* helix8's step projection hands the key-sharded Top-K a uniform verify step whose rows carry
  their verify bounds, and the CP merge a step without them (runs the projection on CPU).

A known rule break is listed in ``_PENDING`` with the follow-up that removes it; an entry that
matches nothing fails.
"""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_REPO = _PKG_ROOT.parent
_MV2 = "tensorrt_llm/_torch/_experimental/modeling_v2"
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_FAMILY_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_TARGETS = {
    "cep": "kimi_k31_pro_nvfp4__sm_107__cep",
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "dep16": "kimi_k31_pro_nvfp4__sm_107__dep16",
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
}
# The runtime helpers whose engine imports must resolve (cache manager, metadata, backend).
_HELPERS = (
    "ops/attention/rubin_dsa_backend.py",
    "ops/attention/rubin_dsa_metadata.py",
    "ops/attention/rubin_dsa_cache.py",
    "ops/ssm/rubin_kda_dsa_cache_manager.py",
    "ops/ssm/rubin_cache_policy.py",
    "ops/ssm/rubin_kda_metadata.py",
)
# The engine DFlash worker methods the drafter's worker inherits or overrides.
_WORKER_INHERITS = (
    "prepare_managed_draft_cache",
    "snapshot_managed_draft_history",
    "_prepare_managed_history",
    "_lazy_init_ctx_buffers",
    "_advance_ctx_len",
    "_assign_slot",
    "_write_ctx_len",
    "_store_prefill_context",
    "_store_context_kv_paged",
    "_prepare_paged_latent_gen_inputs",
    "_prepare_kv_for_draft_forward",
    "_apply_kv_rewind_after_draft",
    "_prepare_next_new_tokens",
    "_forward_impl",
    "_draft_reads_target_metadata",
    "set_draft_model",
    "set_target_model",
    "target_logits",
    "target_head_rows",
    "target_vocab_parallel_head",
)
# The engine DSA metadata methods the family metadata keeps or extends.
_DSA_METADATA_KEEPS = (
    "prepare",
    "on_update_kv_lens",
    "update_for_spec_dec",
    "token_major_gen_view",
    "helix_spec_row_kv_lens",
)
# Engine packages built as extensions or vendored at build time: no source to read.
_COMPILED = ("tensorrt_llm.bindings", "tensorrt_llm.deep_gemm")
_K31_NAME = re.compile(r"KimiK31|kimi_k31|K3\.1|\bK31|k31_")
_NAME_ALLOWLIST = frozenset({"tensorrt_llm/usage/architecture_allowlist.py"})
_PENDING: dict[tuple[str, str], str] = {
    ("names", r"tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer\.py$"): (
        "EPLB recognizes models by checkpoint architecture (moe_model_arch_list); a model-declared "
        "EPLB capability would remove the K3.1 entry"
    ),
}


def _check(rule: str, found: set[str]) -> None:
    """Fail on a rule break no pending entry covers, and on a pending entry that matches nothing."""
    pending = {p: item for (r, p), item in _PENDING.items() if r == rule}
    new = sorted(f for f in found if not any(re.match(p, f) for p in pending))
    gone = sorted(
        f"{p!r} ({i})" for p, i in pending.items() if not any(re.match(p, f) for f in found)
    )
    assert not new, f"{rule}: " + "; ".join(new)
    assert not gone, f"{rule}: fixed, remove from _PENDING: " + "; ".join(gone)


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _engine(rel: str) -> Path:
    return _REPO / rel


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    assert found, f"no class {name}"
    return found[0]


def _method_names(cls: ast.ClassDef) -> set[str]:
    return {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _function(scope: list[ast.stmt], name: str) -> ast.FunctionDef:
    found = [
        n
        for n in scope
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    assert found, f"no function {name}"
    return found[0]


def _module_file(dotted: str) -> Path | None:
    parts = dotted.split(".")
    base = _PKG_ROOT.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_names(path: Path) -> set[str]:
    """Names a module binds at its top level, in any branch (what ``from it import x`` finds)."""
    names = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


# --- what the targets inherit or call ---


def test_the_engine_has_what_the_targets_inherit_and_call():
    """The engine methods the targets' worker and metadata keep exist, and so does every engine
    name the targets and the runtime helpers import."""
    found = set()
    worker = _class(_parse(_engine("tensorrt_llm/_torch/speculative/dflash.py")), "DFlashWorker")
    base = _class(_parse(_engine("tensorrt_llm/_torch/speculative/interface.py")), "SpecWorkerBase")
    have = _method_names(worker) | _method_names(base)
    found |= {f"DFlashWorker lacks {m}" for m in _WORKER_INHERITS if m not in have}
    dsa = _parse(_engine("tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py"))
    trt = _parse(_engine("tensorrt_llm/_torch/attention/backends/trtllm.py"))
    meta = _class(dsa, "DSAtrtllmAttentionMetadata")
    have = _method_names(meta) | _method_names(_class(trt, "TrtllmAttentionMetadata"))
    found |= {f"DSAtrtllmAttentionMetadata lacks {m}" for m in _DSA_METADATA_KEEPS if m not in have}
    update = _function(meta.body, "update_for_spec_dec")
    if "refresh_device_metadata" not in {a.arg for a in update.args.args + update.args.kwonlyargs}:
        found.add("update_for_spec_dec takes no refresh_device_metadata")
    cache = _engine("tensorrt_llm/_torch/attention/backends/sparse/dsa/cache_manager.py")
    if "provides_dsa_indexer_k_cache" not in {
        n.value for n in ast.walk(_parse(cache)) if isinstance(n, ast.Constant)
    }:
        found.add("the engine DSA code reads no provides_dsa_indexer_k_cache")
    files = [_FAMILY / t / "modeling.py" for t in _TARGETS.values()]
    files += [_FAMILY / h for h in _HELPERS if (_FAMILY / h).is_file()]
    for path in files:
        for node in ast.walk(_parse(path)):
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                continue
            if not node.module.startswith("tensorrt_llm") or node.module.startswith(_FAMILY_PKG):
                continue
            if node.module.startswith(_COMPILED):
                continue
            source = _module_file(node.module)
            if source is None:
                found.add(f"{path.name} imports the missing module {node.module}")
                continue
            names = _top_names(source)
            for alias in node.names:
                sub = _module_file(f"{node.module}.{alias.name}")
                if alias.name not in names and sub is None:
                    found.add(f"{path.name}: {node.module} has no {alias.name}")
    _check("base", found)


# --- what the targets leave in the engine ---


def _engine_python_files() -> list[Path]:
    return sorted(
        p
        for p in _PKG_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and not p.relative_to(_REPO).as_posix().startswith(_MV2)
    )


def test_engine_files_do_not_name_the_model():
    """Outside modeling_v2 only the usage allowlist names K3.1."""
    found = set()
    for path in _engine_python_files():
        rel = path.relative_to(_REPO).as_posix()
        if rel not in _NAME_ALLOWLIST and _K31_NAME.search(path.read_text()):
            found.add(rel)
    _check("names", found)


# --- helix CP warmup runs no context row ---


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


def _guards(node: ast.AST, parents: dict[int, ast.AST], stop: ast.AST) -> list[tuple[str, bool]]:
    """The ``if`` tests above a node, with whether the node is in the true branch."""
    out = []
    child = node
    while child is not stop:
        parent = parents[id(child)]
        if isinstance(parent, ast.If):
            out.append((ast.unparse(parent.test), any(child is s for s in parent.body)))
        child = parent
    return out


# The attention warmup's one ungated batch: one generation request of a full verify block.
_GENERATION_ONLY_SHAPE = "(1 + self._config.max_total_draft_tokens, 1)"


def test_helix_warmup_runs_no_context_row():
    """Under helix CP the engine's warmup sends no context row through the target, so helix8's
    prefill target may refuse them; the KV cache estimation refuses context parallelism."""
    path = _engine("tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/warmup.py")
    tree = _parse(path)
    runner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and "_warmup_scheduled" in _method_names(n)
    )
    scheduled = _function(runner.body, "_warmup_scheduled")
    parents = _parents(scheduled)
    problems = []
    agree = [
        n
        for n in ast.walk(scheduled)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "can_run_general_warmup"
    ]
    if not (agree and "not self.mapping.has_cp_helix()" in ast.unparse(agree[0].value)):
        problems.append("the general warmup is not off under helix")
    for node in ast.walk(scheduled):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        guards = _guards(node, parents, scheduled)
        if callee in ("self._general_warmup", "self._run_mamba_hybrid_warmup"):
            off = any(
                ("can_run_general_warmup" in test and branch)
                or ("has_cp_helix()" in test and not branch)
                for test, branch in guards
            )
            if not off:
                problems.append(f"{callee} may run under helix")
        if callee == "self._run_autotuner_warmup":
            helix = any("has_cp_helix()" in test and branch for test, branch in guards)
            gen_only = any(
                k.arg == "generation_only" and ast.unparse(k.value) == "True" for k in node.keywords
            )
            if helix and not gen_only:
                problems.append("the helix autotuner warmup runs context requests")
            if not helix and not any("has_cp_helix()" in t and not b for t, b in guards):
                problems.append("an autotuner warmup is not split on helix")
    impl = _function(runner.body, "_general_warmup_impl")
    if "if self.mapping.has_cp_helix() and num_tokens != num_gen_tokens:" not in ast.unparse(impl):
        problems.append("a general warmup config with context rows runs under helix")
    attention = _function(runner.body, "_run_attention_warmup")
    a_parents = _parents(attention)
    for node in ast.walk(attention):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "warmup_requests_configs.append"
        ):
            gated = any(
                "can_run_general_warmup" in t and b for t, b in _guards(node, a_parents, attention)
            )
            shape = ast.unparse(node.args[0])
            if not gated and shape != _GENERATION_ONLY_SHAPE:
                problems.append(
                    f"the attention warmup runs {shape} without the general-warmup gate"
                )
    util = _parse(_engine("tensorrt_llm/_torch/pyexecutor/_util.py"))
    estimate = [
        n
        for n in ast.walk(util)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_token_num_for_estimation"
    ]
    estimate_src = ast.unparse(estimate[0]) if estimate else ""
    if not (
        ("cp_type" in estimate_src or "_cp_splits_kv" in estimate_src) and "raise" in estimate_src
    ):
        problems.append("the KV cache estimation does not refuse context parallelism")
    assert not problems, "; ".join(problems)


# --- helix8's key-sharded Top-K on the steps it must serve ---


def _helix_projection():
    """helix8's ``_dsa_step_args`` and what it needs, run from the target's source."""
    torch = pytest.importorskip("torch")
    tree = _parse(_FAMILY / _TARGETS["helix8"] / "modeling.py")
    names = {"_DsaStep", "_sparse_args", "_rope_args", "_token_major_sparse_args", "_dsa_step_args"}
    consts = {"_DSA_TOPK", "_DSA_MAX_ATOM_N", "_DSA_INDEX_HEADS", "_DSA_TOPK_OUT"}
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            body.append(copy.deepcopy(node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in consts for t in node.targets
        ):
            body.append(copy.deepcopy(node))
    ns: dict[str, Any] = {
        "torch": torch,
        "Any": Any,
        "NamedTuple": NamedTuple,
        "_DsaPools": object,
        "_K31_DSA": {"index_topk": 2048, "index_num_heads": 32},
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, "helix8 modeling.py", "exec"), ns)
    return ns["_dsa_step_args"], torch


def _key_shard_next_n(rows: int, ng: int, bounds: bool) -> int:
    """The verify width the metadata's key-shard decode prep sets (0: the CP merge serves it):
    uniform verify groups, with per-row bounds unless each group is one row."""
    if rows % ng:
        return 0
    next_n = rows // ng
    return next_n if bounds or next_n == 1 else 0


class _Md(SimpleNamespace):
    """The metadata attributes the projection reads, as the helix metadata provides them; the
    op-argument fields it only hands through read as None."""

    def __getattr__(self, name):
        return None

    def helix_key_shard_inputs(self, rows):
        return ("table", self.gen_indexer_kv_lens_cuda_runtime, "sched", "row_lens")

    def helix_spec_row_kv_lens(self, rows):
        return self.bounds if self._helix_spec_tokens_valid else None

    def helix_sparse_row_lens_out(self, rows):
        return "counts"

    def get_indexer_max_seq_len(self):
        return 4096

    def get_empty(self, buffers, shape, cache_name, dtype, capture_graph):
        return ("topk", shape)

    def token_major_gen_view(self, token_sparse=False):
        return None


def _metadata(torch, ng: int, next_n: int, *, bounds: bool, graph: bool) -> _Md:
    rows = ng * next_n
    zeros = torch.zeros(max(rows, 1), dtype=torch.int32)
    md = _Md(
        is_ragged_verify=False,
        skip_indexer_for_gen_reqs=False,
        key_shard_world=8,
        key_shard_next_n=_key_shard_next_n(rows, ng, bounds),
        key_shard_max_local_len=1024,
        _helix_spec_tokens_valid=bounds,
        bounds=zeros[:rows],
        use_expanded_buffers_for_mtp=False,
        gen_indexer_kv_lens_cuda_runtime=zeros[:ng],
        indexer_k_cache_block_offsets=torch.zeros(ng, 4, dtype=torch.int32),
        scheduler_metadata_buffer=zeros,
        scheduler_metadata_buffer_full_next_n=zeros,
        kv_lens_cuda_2d=torch.zeros(ng, next_n, dtype=torch.int32),
        kv_lens_cuda_runtime=zeros[:ng],
        slot_mapping_fp8=zeros,
        slot_mapping_scale=zeros,
        topk_indices_buffer=zeros,
        radix_aux_indices=zeros,
        radix_aux_logits=zeros,
        req_idx_per_token=zeros,
        block_table=torch.zeros(ng, 4, dtype=torch.int32),
        cuda_graph_buffers=None,
        is_cuda_graph=graph,
        kv_cache_manager=SimpleNamespace(
            kv_cache_pool_pointers=None, kv_cache_pool_mapping=None, tokens_per_block=64
        ),
        helix_position_offsets=zeros,
        helix_is_inactive_rank=zeros,
        helix_local_slots=zeros,
        max_seq_len=4096,
        beam_width=1,
    )
    return md


def _pools():
    return SimpleNamespace(tokens_per_block=64)


# (step kind, generation requests, verify width, rows carry their verify bounds, graph step)
_HELIX_STEPS = (
    ("k5 verify", 16, 6, True, True),
    ("k7 verify", 12, 8, True, True),
    ("one row per request", 16, 1, False, True),
)


@pytest.mark.parametrize(
    "kind, ng, next_n, bounds, graph", _HELIX_STEPS, ids=[s[0] for s in _HELIX_STEPS]
)
def test_helix_key_shard_serves_every_uniform_spec_step(kind, ng, next_n, bounds, graph):
    """helix8's step projection hands the key-sharded Top-K a uniform verify step whose rows
    carry their verify bounds, and a step of one row per request."""
    project, torch = _helix_projection()
    md = _metadata(torch, ng, next_n, bounds=bounds, graph=graph)
    st = project(md, _pools(), ng * next_n, ng)
    assert st.ks_next_n == next_n, f"{kind}: the key-sharded Top-K does not serve it"
    assert st.ks_row_lens is not None and st.ks_local_len > 0, kind
    if next_n > 1:
        assert st.row_bounds is not None, f"{kind}: no per-row bounds for the merge's mask"


def test_helix_steps_without_key_shard_inputs_take_the_cp_merge():
    """A multi-row step whose rows carry no verify bounds takes the CP merge (eager)."""
    project, torch = _helix_projection()
    md = _metadata(torch, 16, 6, bounds=False, graph=False)
    st = project(md, _pools(), 96, 16)
    assert st.ks_next_n == 0

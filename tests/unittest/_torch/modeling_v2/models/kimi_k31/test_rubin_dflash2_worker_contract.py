# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash worker and draft-container contracts of the specialized SM107 targets."""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional

import pytest
import torch

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


@dataclass(frozen=True)
class _Case:
    target: str
    helix: bool = False
    container: str = "DraftModel"

    @property
    def id(self) -> str:
        name = self.target.rsplit("__", 1)[-1]
        return name if self.container == _CONTAINER else f"{name}-{self.container}"


# The worker and the drafter container every specialized target's modeling.py defines.
_WORKER, _CONTAINER = "DFlashWorker", "DraftModel"
_WORKERS = (
    _Case("kimi_k31_pro_nvfp4__sm_107__cep"),
    _Case("kimi_k31_pro_nvfp4__sm_107__dep16"),
    _Case("kimi_k31_pro_nvfp4__sm_107__helix8", helix=True),
    _Case("kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1", helix=True),
    _Case("kimi_k31_pro_nvfp4__sm_107__helix16_wide", helix=True),
    _Case("kimi_k31_pro_nvfp4__sm_107__tep8"),
)
# What the engine reads from a DFlash2 drafter container (the first attribute of each chain).
_CONTAINER_NAMES = (
    "supports_unified_draft_cache",
    "_kv_factor",
    "_num_attn_layers",
    "_head_dim",
    "context_kv_fp8",
    "context_kv_only",
    "_helix_mapping",
    "fc",
    "hidden_norm",
    "config",
    "_uses_lilicorr",
    "has_candidate_selector",
    "draft_model_full",
    "mask_token_id",
    "lm_head",
    "logits_processor",
    "project_target_hidden",
    "precompute_context_kv",
    "dflash_forward",
)
# The keyword arguments the worker calls dflash_forward with.
_DFLASH_FORWARD_KW = (
    "noise_embedding",
    "query_positions",
    "num_ctx_per_req",
    "ctx_k_cache",
    "ctx_v_cache",
    "ctx_cache_batch_idx",
    "ctx_kv_cache",
    "ctx_page_table",
    "block_placement",
)
_ENGINE_SPEC = "/_torch/speculative/"
_MISSING = object()


# --- reading the source ---


def _package_root() -> Path:
    """The ``tensorrt_llm`` package under test, located without importing it."""
    module = sys.modules.get("tensorrt_llm")
    paths = list(getattr(module, "__path__", None) or ())
    if not paths:
        try:
            spec = importlib.util.find_spec("tensorrt_llm")
        except ValueError:
            spec = None
        paths = list((spec and spec.submodule_search_locations) or ())
    if paths:
        return Path(paths[0]).resolve()
    return Path(__file__).resolve().parents[6] / "tensorrt_llm"


_ROOT = _package_root()
_FAMILY = _ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"


def _modeling(case: _Case) -> Path:
    return _FAMILY / case.target / "modeling.py"


@functools.lru_cache(maxsize=None)
def _parsed(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(), filename=path)


def _module_name(path: Path) -> Optional[str]:
    """The dotted name of a module of the package, None for a file outside it."""
    try:
        rel = path.resolve().relative_to(_ROOT).with_suffix("")
    except ValueError:
        return None
    parts = ["tensorrt_llm", *rel.parts]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _module_file(name: str) -> Optional[Path]:
    if not name.startswith("tensorrt_llm."):
        return None
    rel = Path(*name.split(".")[1:])
    for candidate in (_ROOT / rel.with_suffix(".py"), _ROOT / rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=None)
def _imports(path: str) -> dict:
    """Local name -> (absolute module, imported attribute or None) for a module's imports."""
    file = Path(path)
    name = _module_name(file)
    package = (
        None if name is None else name if file.name == "__init__.py" else name.rsplit(".", 1)[0]
    )
    out = {}
    for node in ast.walk(_parsed(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    out[alias.asname] = (alias.name, None)
                else:
                    out[alias.name.split(".")[0]] = (alias.name.split(".")[0], None)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                if package is None:
                    continue
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                out[alias.asname or alias.name] = (module, alias.name)
    return out


@dataclass(frozen=True)
class _Def:
    path: Path
    node: Any

    @property
    def engine(self) -> bool:
        return _ENGINE_SPEC in self.path.as_posix()


def _find_def(path: Path, name: str, depth: int = 0) -> Optional[_Def]:
    """The top-level class or function ``name`` of a module, following its imports."""
    for node in _parsed(str(path)).body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return _Def(path, node)
    imported = _imports(str(path)).get(name)
    if imported and imported[1] and depth < 8:
        target = _module_file(imported[0])
        if target is not None:
            return _find_def(target, imported[1], depth + 1)
    return None


def _mro(cls: _Def) -> list[_Def]:
    """The class and the bases resolvable in the source tree, depth first, left to right."""
    out, seen = [], set()

    def visit(c: _Def) -> None:
        key = (str(c.path), c.node.name)
        if key in seen:
            return
        seen.add(key)
        out.append(c)
        for base in c.node.bases:
            found = None
            if isinstance(base, ast.Name):
                found = _find_def(c.path, base.id)
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                module = _imports(str(c.path)).get(base.value.id)
                target = _module_file(module[0]) if module and module[1] is None else None
                found = _find_def(target, base.attr) if target else None
            if found is not None and isinstance(found.node, ast.ClassDef):
                visit(found)

    visit(cls)
    return out


def _method(mro: list[_Def], name: str) -> Optional[_Def]:
    for cls in mro:
        for node in cls.node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return _Def(cls.path, node)
    return None


def _self_targets(fn: ast.AST) -> list[tuple[str, ast.AST]]:
    out = []
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for t in target.elts if isinstance(target, ast.Tuple) else [target]:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    out.append((t.attr, getattr(node, "value", None)))
    return out


def _provided(mro: list[_Def]) -> set[str]:
    """Every attribute name a class and its resolvable bases define or assign."""
    names = set()
    for cls in mro:
        for node in cls.node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                names.update(attr for attr, _ in _self_targets(node))
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _constant(mro: list[_Def], name: str):
    """The constant the most derived class binds ``name`` to, _MISSING if none does."""
    for cls in mro:
        for node in cls.node.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return node.value.value if isinstance(node.value, ast.Constant) else _MISSING
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for attr, value in _self_targets(node):
                    if attr == name:
                        return value.value if isinstance(value, ast.Constant) else _MISSING
    return _MISSING


def _engine_worker_bases(path: Path) -> set[str]:
    """Names a module binds to the engine's DFlashWorker."""
    return {
        local
        for local, (module, attr) in _imports(str(path)).items()
        if module == "tensorrt_llm._torch.speculative.dflash" and attr == "DFlashWorker"
    }


def _worker_classes(path: Path) -> set[str]:
    """Classes of a module that subclass the engine's DFlashWorker, directly or in the module."""
    tree = _parsed(str(path))
    bases = _engine_worker_bases(path)
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name in found:
                continue
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else None
                attr = base.attr if isinstance(base, ast.Attribute) else None
                if name in bases or name in found or attr == "DFlashWorker":
                    found.add(node.name)
                    changed = True
                    break
    return found


def _target_dirs() -> list[Path]:
    """The specialized SM107 targets in the tree: the ones the contract covers."""
    return sorted(p for p in _FAMILY.glob("kimi_k31_*__sm_107__*") if (p / "modeling.py").is_file())


def _cases() -> list:
    return [pytest.param(case, id=case.id) for case in _WORKERS]


def _source_cases() -> list:
    return [pytest.param(case, id=case.id) for case in _WORKERS]


# --- the catalog (source only) ---


@pytest.mark.cpu_only
def test_catalog_lists_every_dflash2_worker():
    listed = {(c.target, _WORKER) for c in _WORKERS}
    shipped = {
        (target.name, cls)
        for target in _target_dirs()
        for cls in _worker_classes(target / "modeling.py")
    }
    missing = sorted(shipped - listed)
    assert not missing, f"targets ship DFlash2 workers the contract does not list: {missing}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", _source_cases())
def test_catalog_entry_names_its_classes(case):
    path = _modeling(case)
    if not path.is_file():
        pytest.fail(f"{case.target} has no modeling.py")
    assert _WORKER in _worker_classes(path), f"{case.target} defines no {_WORKER} worker"
    container = _find_def(path, case.container)
    assert container is not None and isinstance(container.node, ast.ClassDef), (
        f"{case.target} defines no {case.container} drafter container"
    )


def _hook_call(fn: ast.AST, receiver: str) -> bool:
    """Whether ``fn`` calls ``<receiver>._apply_force_accepted_tokens(..., spec_metadata=...)``."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_apply_force_accepted_tokens"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
            and any(k.arg == "spec_metadata" for k in node.keywords)
        ):
            return True
    return False


def _accept_path_problems(path: Path, worker: str) -> list[str]:
    mro = _mro(_find_def(path, worker))
    problems = []
    greedy = _method(mro, "_sample_and_accept_draft_tokens_base")
    if greedy is not None and not greedy.engine and not _hook_call(greedy.node, "self"):
        problems.append(f"{greedy.path.name}:{greedy.node.lineno} greedy acceptance skips the hook")
    entry = _method(mro, "sample_and_accept_draft_tokens")
    if entry is None or entry.engine:
        return problems
    returns = [n for n in ast.walk(entry.node) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        problems.append(f"{entry.path.name}:{entry.node.lineno} returns no acceptance")
    for ret in returns:
        call = ret.value
        where = f"{entry.path.name}:{ret.lineno}"
        if not isinstance(call, ast.Call):
            problems.append(f"{where} returns something other than an acceptance call")
        elif (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "sample_and_accept_draft_tokens"
        ):
            continue  # the engine's acceptance (super() or SpecWorkerBase), which calls the hook
        elif isinstance(call.func, ast.Name):
            fn = _find_def(entry.path, call.func.id)
            args = (
                fn.node.args.args
                if fn is not None and not isinstance(fn.node, ast.ClassDef)
                else []
            )
            if not args or not _hook_call(fn.node, args[0].arg):
                problems.append(
                    f"{where} {call.func.id} does not end in the forced-acceptance hook"
                )
        elif (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        ):
            method = _method(mro, call.func.attr)
            if method is None or not (method.engine or _hook_call(method.node, "self")):
                problems.append(f"{where} self.{call.func.attr} does not end in the hook")
        else:
            problems.append(f"{where} dispatches to a call the contract cannot follow")
    return problems


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", _source_cases())
def test_both_accept_paths_reach_the_force_hook_in_source(case):
    """(a), read off the source: the greedy acceptance and every return of the sampled dispatch
    end in ``_apply_force_accepted_tokens(..., spec_metadata=...)``; the GPU test runs them."""
    problems = _accept_path_problems(_modeling(case), _WORKER)
    assert not problems, "\n".join(problems)


def _container_problems(path: Path, container: str) -> list[str]:
    found = _find_def(path, container)
    if found is None or not isinstance(found.node, ast.ClassDef):
        return [f"no {container} class"]
    mro = _mro(found)
    problems = []
    missing = sorted(set(_CONTAINER_NAMES) - _provided(mro))
    if missing:
        problems.append(f"{container} lacks {missing}")
    if _constant(mro, "supports_unified_draft_cache") is not True:
        problems.append("supports_unified_draft_cache is not bound to True")
    if _constant(mro, "_kv_factor") != 1:
        problems.append("_kv_factor is not bound to 1 (one latent per token and layer)")
    if _constant(mro, "_uses_lilicorr") not in (False, _MISSING):
        problems.append("_uses_lilicorr is bound to something other than False")
    if _constant(mro, "unified_draft_cache_when_aggregated") is not True:
        problems.append("unified_draft_cache_when_aggregated is not bound to True")
    forward = _method(mro, "dflash_forward")
    if forward is not None and forward.node.args.kwarg is None:
        params = {a.arg for a in forward.node.args.args + forward.node.args.kwonlyargs}
        if not set(_DFLASH_FORWARD_KW) <= params:
            problems.append(f"dflash_forward lacks {sorted(set(_DFLASH_FORWARD_KW) - params)}")
    return problems


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", _source_cases())
def test_container_provides_what_the_engine_reads_in_source(case):
    """(f), read off the source: the container class (with the bases the tree holds) defines or
    assigns every attribute the engine reads, with the fixed values of the unified draft cache."""
    problems = _container_problems(_modeling(case), case.container)
    assert not problems, "\n".join(problems)


_ENGINE_IMPORT = "from tensorrt_llm._torch.speculative.dflash import DFlashWorker as _Engine\n"
_KEEPS_HOOK = """
def sampled(worker, logits, attn_metadata, spec_metadata):
    num = worker.verify(logits)
    return worker._apply_force_accepted_tokens(num, 0, 4, spec_metadata=spec_metadata)

class DFlashWorker(_Engine):
    def _sample_and_accept_draft_tokens_base(self, logits, drafts, nc, bs, spec_metadata):
        return logits, self._apply_force_accepted_tokens(drafts, nc, 4, spec_metadata=spec_metadata)

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        if spec_metadata.use_rejection_sampling:
            return sampled(self, logits, attn_metadata, spec_metadata)
        return super().sample_and_accept_draft_tokens(logits, attn_metadata, spec_metadata)
"""


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "edit, problems",
    [
        ((), 0),
        (
            (
                (
                    "worker._apply_force_accepted_tokens(num, 0, 4, spec_metadata=spec_metadata)",
                    "num",
                ),
            ),
            1,
        ),
        (
            (
                (
                    "self._apply_force_accepted_tokens(drafts, nc, 4, spec_metadata=spec_metadata)",
                    "drafts",
                ),
            ),
            1,
        ),
        ((("spec_metadata=spec_metadata)\n\nclass", ")\n\nclass"),), 1),
    ],
    ids=["keeps", "sampled-skips", "greedy-skips", "no-metadata"],
)
def test_accept_path_check_flags_a_path_without_the_hook(tmp_path, edit, problems):
    source = _ENGINE_IMPORT + _KEEPS_HOOK
    for old, new in edit:
        assert old in source
        source = source.replace(old, new)
    path = tmp_path / "modeling.py"
    path.write_text(source)
    assert _worker_classes(path) == {"DFlashWorker"}
    assert len(_accept_path_problems(path, "DFlashWorker")) == problems


@pytest.mark.cpu_only
def test_container_check_flags_a_duck_typed_container_that_falls_short(tmp_path):
    bound = {"context_kv_fp8", "dflash_forward", "supports_unified_draft_cache", "_kv_factor"}
    names = "\n".join(
        f"    {n} = None" for n in _CONTAINER_NAMES if n not in bound | {"_uses_lilicorr"}
    )
    path = tmp_path / "modeling.py"
    path.write_text(
        "class DraftModel:\n"
        f"{names}\n"
        "    supports_unified_draft_cache = True\n"
        "    unified_draft_cache_when_aggregated = True\n"
        "    _kv_factor = 1\n"
        "    _uses_lilicorr = False\n"
        "    def dflash_forward(self, noise_embedding, query_positions):\n"
        "        pass\n"
    )
    problems = _container_problems(path, "DraftModel")
    assert any("context_kv_fp8" in p for p in problems), problems
    assert any("dflash_forward lacks" in p for p in problems), problems


def _metadata_overrides(paths) -> tuple[list[str], list[str]]:
    """The ``update_for_spec_dec`` overrides of the modules' classes, and those that do not take
    ``refresh_device_metadata``."""
    overrides, missing = [], []
    for path in paths:
        for cls in ast.walk(_parsed(str(path))):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if (
                    isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and fn.name == "update_for_spec_dec"
                ):
                    where = f"{path.name}:{cls.name}"
                    overrides.append(where)
                    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
                    if "refresh_device_metadata" not in names and fn.args.kwarg is None:
                        missing.append(where)
    return overrides, missing


def _guided_decoder_forwarded(path: Path) -> bool:
    """Whether the target's model class (``ModelingV2*``) defines ``set_guided_decoder`` returning
    ``self.spec_worker.set_guided_decoder(<its argument>)``."""
    for cls in _parsed(str(path)).body:
        if not (isinstance(cls, ast.ClassDef) and cls.name.startswith("ModelingV2")):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "set_guided_decoder"):
                continue
            args = [a.arg for a in fn.args.args]
            for node in ast.walk(fn):
                call = node.value if isinstance(node, ast.Return) else None
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "set_guided_decoder"
                    and ast.unparse(call.func.value) == "self.spec_worker"
                    and len(args) == 2
                    and [ast.unparse(a) for a in call.args] == [args[1]]
                ):
                    return True
    return False


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", _cases())
def test_the_model_attaches_a_guided_decoder_to_its_worker(case):
    """The engine attaches a one-model drafter's grammar decoder through the model's
    ``set_guided_decoder`` and refuses to start when it is missing; a specialized target's model
    hands it to its DFlash2 worker, whose verification applies the mask."""
    assert _guided_decoder_forwarded(_modeling(case)), (
        f"{case.target}: the model class does not forward set_guided_decoder to its spec worker"
    )


@pytest.mark.cpu_only
def test_guided_decoder_check_flags_a_model_without_the_forward(tmp_path):
    good, bad = (
        tmp_path / "good.py",
        tmp_path / "bad.py",
    )  # one file each: the parse is cached by path
    good.write_text(
        "class ModelingV2Good:\n"
        "    def set_guided_decoder(self, guided_decoder):\n"
        "        return self.spec_worker.set_guided_decoder(guided_decoder)\n"
    )
    bad.write_text("class ModelingV2Bad:\n    def load_weights(self, weights): ...\n")
    assert _guided_decoder_forwarded(good)
    assert not _guided_decoder_forwarded(bad)


@pytest.mark.cpu_only
def test_family_metadata_takes_the_refresh_switch():
    """A specialized worker's drafter reads no target attention metadata, so its draft step calls
    ``update_for_spec_dec(refresh_device_metadata=False)``: every family attention metadata class
    that overrides the hook takes the keyword."""
    overrides, missing = _metadata_overrides(sorted((_FAMILY / "ops" / "attention").glob("*.py")))
    if not overrides:
        pytest.skip("no family attention metadata class overrides update_for_spec_dec")
    assert not missing, f"update_for_spec_dec without refresh_device_metadata: {missing}"


@pytest.mark.cpu_only
def test_metadata_switch_check_flags_an_override_without_it(tmp_path):
    path = tmp_path / "metadata.py"
    path.write_text(
        "class Takes:\n    def update_for_spec_dec(self, refresh_device_metadata=True): ...\n"
        "class Kwargs:\n    def update_for_spec_dec(self, **kwargs): ...\n"
        "class Lacks:\n    def update_for_spec_dec(self): ...\n"
    )
    overrides, missing = _metadata_overrides([path])
    assert len(overrides) == 3 and missing == ["metadata.py:Lacks"]


# --- the worker on a GPU: a toy drafter of the checkpoint's structure ---

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the DFlash2 worker's kernels are certified on sm_103 / sm_107",
)

_H, _NH, _QL, _KVL, _NOPE, _ROPE_D, _V, _I, _VOCAB, _L = 128, 6, 32, 512, 16, 64, 16, 256, 512, 2
_WINDOW, _GROUP, _TAPS, _PAGE = 8, 16, 2, 32
# Three requests and one CUDA-graph padding row (the shared dummy slot).
_BATCH, _DUMMY_SLOT, _PAGES_PER_ROW = 4, 3, 6
_FP8 = torch.float8_e4m3fn


def _draft_config(block_size: int):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    cfg = PretrainedConfig(
        architectures=["DFlash2DraftModel"],
        hidden_size=_H,
        num_attention_heads=_NH,
        num_key_value_heads=_NH,
        q_lora_rank=_QL,
        kv_lora_rank=_KVL,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE_D,
        v_head_dim=_V,
        intermediate_size=_I,
        hidden_act="silu",
        num_hidden_layers=_L,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB,
        max_position_embeddings=4096,
        rope_parameters=dict(
            rope_type="yarn",
            rope_theta=50000.0,
            factor=32.0,
            original_max_position_embeddings=32768,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
        rope_theta=50000.0,
        sliding_window=_WINDOW,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * _L,
        is_causal=True,
        dflash_config=dict(
            attention_mode="mla",
            block_size=block_size,
            conv_kernel_size=_TAPS,
            conv_group_size=_GROUP,
            selector_rank=8,
            # The candidates per slot that the verification kernels hold
            # (k31_dflash2_proposal_load's TOP_K, the targets' DRAFT_DFLASH selector_top_k).
            selector_top_k=32,
            mask_token_id=_VOCAB - 1,
            target_layer_ids=[0, 1],
        ),
    )
    cfg.model_type = "qwen3"
    cfg.torch_dtype = torch.bfloat16
    return ModelConfig(pretrained_config=cfg, mapping=Mapping(world_size=1, tp_size=1, rank=0))


def _weights(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=None):
        scale = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {}
    for i in range(_L):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = norm(_H)
        w[p + "post_attention_layernorm.weight"] = norm(_H)
        w[p + "self_attn.q_a_proj.weight"] = rnd(_QL, _H)
        w[p + "self_attn.q_a_layernorm.weight"] = norm(_QL)
        w[p + "self_attn.q_b_proj.weight"] = rnd(_NH * (_NOPE + _ROPE_D), _QL)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(_KVL + _ROPE_D, _H)
        w[p + "self_attn.kv_a_layernorm.weight"] = norm(_KVL)
        w[p + "self_attn.kv_b_proj.weight"] = rnd(_NH * (_NOPE + _V), _KVL)
        w[p + "self_attn.o_proj.weight"] = rnd(_H, _NH * _V)
        w[p + "mlp.gate_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.up_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.down_proj.weight"] = rnd(_H, _I)
        for conv in ("attention_conv", "mlp_conv"):
            base = 0.2 * torch.randn(2, _TAPS, _H, generator=g)
            base[:, 0] += 1.0
            w[p + conv + ".base_kernel"] = base.to(torch.bfloat16)
            w[p + conv + ".kernel_projection.weight"] = rnd(
                2 * _TAPS * _H // _GROUP, _H, scale=0.02
            )
    w["fc.weight"] = rnd(_H, 2 * _H)
    w["hidden_norm.weight"] = norm(_H)
    w["norm.weight"] = norm(_H)
    w["candidate_selector.predecessor_codebook"] = rnd(_VOCAB, 8)
    w["candidate_selector.successor_codebook"] = rnd(_VOCAB, 8)
    w["candidate_selector.hidden_projection.weight"] = rnd(8, _H)
    return w


class _ToyRt(NamedTuple):
    """What a drafter reads off the target's runtime (``core._rt``): the TP group's all-reduce, and
    ``draft``, which its ``bind_runtime`` replaces with the drafter's own."""

    all_reduce: Any
    draft: Any


class _ToyCore(torch.nn.Module):
    """The target core a specialized drafter is built on, with what the drafter reads off it: the
    target's model config (``spec_config``), the helix mapping (``cp_mapping``, the model
    config's), the per-rank widths (``dims.kda_tp``), the weight table ``w`` (the embedding the
    drafter shares and the drafter's own ``d_*`` rows) and the runtime (``_rt``)."""

    def __init__(self, embed: torch.Tensor, model_config: Any, draft_rows: dict) -> None:
        super().__init__()
        from tensorrt_llm._torch.distributed import AllReduce

        self.model_config = model_config
        self.cp_mapping = model_config.mapping
        self.dims = SimpleNamespace(kda_tp=model_config.mapping.tp_size)
        rows = {"embed": embed, **draft_rows}
        self.w = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v, requires_grad=False) for k, v in rows.items()}
        )
        # One GPU: the all-reduce of a one-rank TP group returns its input.
        self._rt = _ToyRt(AllReduce(mapping=model_config.mapping, dtype=torch.bfloat16), None)


def _draft_rows(history_only: bool) -> dict:
    """The drafter's ``core.w["d_*"]`` rows from the toy checkpoint, in the layout the target's
    weight loader (``w3_rows``) writes on one rank: ``d_kv_a`` and ``d_kv_a_norm`` stack every
    layer's latent projection and norm, ``kv_b`` holds every head's ``k_nope`` rows then every
    head's ``v`` rows, ``gate_up`` the gate rows then the up rows. A history-only drafter
    (``context_kv_only``) has the history's rows only."""
    w = _weights()
    layers = range(_L)
    rows = {
        "d_fc": w["fc.weight"],
        "d_hidden_norm": w["hidden_norm.weight"],
        "d_kv_a": torch.cat([w[f"layers.{j}.self_attn.kv_a_proj_with_mqa.weight"] for j in layers]),
        "d_kv_a_norm": torch.stack(
            [w[f"layers.{j}.self_attn.kv_a_layernorm.weight"] for j in layers]
        ),
    }
    if not history_only:
        rows["d_final_norm"] = w["norm.weight"]
        rows["d_sel_pred"] = w["candidate_selector.predecessor_codebook"]
        rows["d_sel_succ"] = w["candidate_selector.successor_codebook"]
        rows["d_sel_hidden_proj"] = w["candidate_selector.hidden_projection.weight"]
        for j in layers:
            p, d = f"layers.{j}.", f"d_l{j}_"
            kv_b = w[p + "self_attn.kv_b_proj.weight"].view(_NH, _NOPE + _V, _KVL)
            rows[d + "in_norm"] = w[p + "input_layernorm.weight"]
            rows[d + "post_norm"] = w[p + "post_attention_layernorm.weight"]
            rows[d + "q_a"] = w[p + "self_attn.q_a_proj.weight"]
            # helix16 reads q_a and the layer's kv_a as one [q_a ; kv_a] bank.
            rows[d + "q_kv_a"] = torch.cat(
                (w[p + "self_attn.q_a_proj.weight"], w[p + "self_attn.kv_a_proj_with_mqa.weight"])
            )
            rows[d + "q_a_norm"] = w[p + "self_attn.q_a_layernorm.weight"]
            rows[d + "q_b"] = w[p + "self_attn.q_b_proj.weight"]
            rows[d + "kv_b"] = torch.cat(
                (kv_b[:, :_NOPE].reshape(-1, _KVL), kv_b[:, _NOPE:].reshape(-1, _KVL))
            )
            rows[d + "o"] = w[p + "self_attn.o_proj.weight"]
            rows[d + "gate_up"] = torch.cat(
                (w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"])
            )
            rows[d + "down"] = w[p + "mlp.down_proj.weight"]
            rows[d + "attn_conv_base"] = w[p + "attention_conv.base_kernel"]
            rows[d + "attn_conv_proj"] = w[p + "attention_conv.kernel_projection.weight"]
            rows[d + "mlp_conv_base"] = w[p + "mlp_conv.base_kernel"]
            rows[d + "mlp_conv_proj"] = w[p + "mlp_conv.kernel_projection.weight"]
    return {k: v.contiguous().cuda() for k, v in rows.items()}


def _target_mapping(case: _Case):
    """The target's Mapping on this one GPU: one rank, of a CP8 helix group for a helix case (the
    drafter shards its history by the target's helix mapping; the CP16 target takes it too)."""
    from tensorrt_llm.mapping import CpType, Mapping

    if case.helix:
        return Mapping(world_size=8, rank=0, cp_size=8, cp_config={"cp_type": CpType.HELIX})
    return Mapping(world_size=1, tp_size=1, rank=0)


def _module(case: _Case):
    return importlib.import_module(f"{_PACKAGE}.{case.target}.modeling")


def _build_container(case: _Case, module, block: int, monkeypatch, *, history_only=False):
    """The case's drafter container at toy size, with seeded weights, on the GPU."""
    draft_config = _draft_config(block)
    g = torch.Generator().manual_seed(7)
    embed = (torch.randn(_VOCAB, _H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    spec = SimpleNamespace(
        max_draft_len=block - 1,
        attention_backend="VANILLA",
        context_kv_only=history_only,
        context_kv_dtype="fp8",
    )
    model_config = SimpleNamespace(spec_config=spec, extra_attrs={}, mapping=_target_mapping(case))
    # The drafter holds no weight of its own: its rows sit in the core's table, as the target's
    # weight loader fills them.
    core = _ToyCore(embed, model_config, _draft_rows(history_only))
    # As the shell builds it: the drafter's ModelConfig from draft_config_for, then the container
    # on the core.
    monkeypatch.setattr(module, "draft_config_for", lambda _: draft_config)
    lm_head = torch.nn.Linear(_H, _VOCAB, bias=False, dtype=torch.bfloat16)
    try:
        container = module.DraftModel(core, module.draft_config_for(model_config), lm_head, None)
    except (TypeError, AttributeError) as error:
        pytest.fail(
            f"{case.target}.DraftModel no longer builds from (core, model_config, lm_head, "
            f"logits_processor): {error}; update _build_container to its constructor"
        )
    container = container.cuda()
    # A drafter may hold the target core by weak reference only (it shares the embedding).
    object.__setattr__(container, "_contract_core", core)
    embed_tokens = container.draft_model_full.model.embed_tokens
    if embed_tokens is not None and hasattr(embed_tokens, "weight"):
        with torch.no_grad():
            embed_tokens.weight.copy_(embed)
    # The worker sizes the drafter's window workspace for its largest step.
    container._max_block_rows = _BATCH * block
    return container


def _build_worker(case: _Case, module, container, k: int):
    from tensorrt_llm.mapping import Mapping

    spec_config = SimpleNamespace(
        max_draft_len=k,
        attention_backend="VANILLA",
        context_kv_dtype="fp8",
        context_kv_only=False,
        use_rejection_sampling=False,
        tokens_per_gen_step=k + 1,
    )
    worker = getattr(module, _WORKER)(spec_config, Mapping(world_size=1, tp_size=1, rank=0))
    worker.set_draft_model(container)
    return worker


@dataclass
class _Rig:
    """One worker, its toy drafter, the unified-cache state of a generation step, and the
    step's inputs, all in fixed tensors: a captured step reads the same storage at replay."""

    worker: Any
    container: Any
    k: int
    pools: list
    logits: torch.Tensor
    draft_tokens: torch.Tensor
    captured: torch.Tensor
    slots: torch.Tensor
    attn: SimpleNamespace
    spec: SimpleNamespace

    @property
    def block(self) -> int:
        return self.k + 1

    def state(self) -> dict:
        w = self.worker
        return {
            "ctx_len": w._ctx_len,
            "ctx_position_offset": w._ctx_position_offset,
            "seq_lens": self.attn._seq_lens_cuda,
            **{f"pool{i}": p for i, p in enumerate(self.pools)},
        }


def _rig(case: _Case, k: int, pool_dtype: torch.dtype, monkeypatch, force: int = 0) -> _Rig:
    torch.manual_seed(0)
    module = _module(case)
    block = k + 1
    container = _build_container(case, module, block, monkeypatch)
    worker = _build_worker(case, module, container, k)
    worker.force_num_accepted_tokens = float(force)
    dev = torch.device("cuda")
    pages = _BATCH * _PAGES_PER_ROW
    pools = [
        torch.zeros(pages, 1, 1, _PAGE, _KVL + _ROPE_D, dtype=pool_dtype, device=dev)
        for _ in range(_L)
    ]
    # The unified draft cache as _lazy_init_ctx_buffers binds it (slots 0..2, dummy slot 3).
    worker._ctx_kv_manager = SimpleNamespace(has_standalone_draft_cache=True)
    worker._ctx_buf_inited = True
    worker._compute_block_size = block
    worker._dummy_slot = _DUMMY_SLOT
    worker._max_ctx = 1024
    worker._ctx_page_size = _PAGE
    worker._ctx_len = torch.zeros(_BATCH, dtype=torch.long, device=dev)
    worker._ctx_position_offset = torch.zeros(_BATCH, dtype=torch.long, device=dev)
    worker._ctx_block_tables = torch.zeros(_BATCH, _PAGES_PER_ROW, dtype=torch.int32, device=dev)
    worker._ctx_block_counts = torch.full((_BATCH,), _PAGES_PER_ROW, dtype=torch.long, device=dev)
    worker._ctx_kv_buf = pools
    worker._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device=dev)
    worker._ctx_k_buf = worker._ctx_v_buf = None
    slots = torch.zeros(_BATCH, dtype=torch.long, device=dev)
    worker._batch_to_slot_buffer = SimpleNamespace(tensor=slots)
    # Then the drafter's runtime, as the worker's _lazy_init_ctx_buffers binds it.
    # A drafter keyed by context slot (helix16's headtp ring) also takes the worker's slots.
    keyed = "slot_table" in inspect.signature(container.bind_runtime).parameters
    container.bind_runtime(
        _BATCH * worker._compute_block_size,
        worker._ctx_page_size,
        **(dict(slots=worker._ctx_len.shape[0], slot_table=slots) if keyed else {}),
    )
    rows = _BATCH * block
    captured = torch.zeros(rows, 2 * _H, dtype=torch.bfloat16, device=dev)
    attn = SimpleNamespace(
        num_seqs=_BATCH,
        num_contexts=0,
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.zeros(_BATCH, dtype=torch.int32, device=dev),
        _seq_lens=torch.zeros(_BATCH, dtype=torch.int32),
    )
    draft_tokens = torch.zeros(_BATCH * k, dtype=torch.int32, device=dev)
    spec = SimpleNamespace(
        runtime_draft_len=k,
        draft_tokens=draft_tokens,
        is_ragged_verify=False,
        enable_penalty=False,
        use_rejection_sampling=False,
        is_all_greedy_sample=True,
        is_cuda_graph=False,
        batch_slot_ids=None,
        dummy_slot_row=0,
        max_num_requests=_BATCH,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n],
    )
    return _Rig(
        worker=worker,
        container=container,
        k=k,
        pools=pools,
        logits=torch.zeros(rows, _VOCAB, dtype=torch.float32, device=dev),
        draft_tokens=draft_tokens,
        captured=captured,
        slots=slots,
        attn=attn,
        spec=spec,
    )


def _inputs(rig: _Rig, seed: int, ctx_lens: tuple[int, int, int]) -> dict:
    """One generation step's inputs and the cache state it starts from (host tensors)."""
    g = torch.Generator().manual_seed(seed)
    k, block = rig.k, rig.block
    logits = torch.randn(_BATCH * block, _VOCAB, generator=g) * 3.0
    target = logits.view(_BATCH, block, _VOCAB).argmax(-1)
    drafts = torch.randint(0, _VOCAB, (_BATCH, k), generator=g)
    for row in range(_BATCH):
        agree = (seed + row) % (k + 1)  # the leading drafts that match: 0 .. k accepted
        drafts[row, :agree] = target[row, :agree]
    real = torch.randperm(_DUMMY_SLOT, generator=g)
    ctx_len = torch.zeros(_BATCH, dtype=torch.long)
    ctx_len[real] = torch.tensor(ctx_lens, dtype=torch.long)
    pools = []
    for _ in range(_L):
        latent = torch.randn(_BATCH * _PAGES_PER_ROW, 1, 1, _PAGE, _KVL + _ROPE_D, generator=g)
        pools.append(latent.to(torch.bfloat16).to(rig.pools[0].dtype))
    host = {
        "logits": logits,
        "draft_tokens": drafts.reshape(-1).to(torch.int32),
        "captured": (torch.randn(_BATCH * block, 2 * _H, generator=g) * 0.5).to(torch.bfloat16),
        # Rows 0..2 are requests, row 3 the padding row on the dummy slot.
        "slots": torch.cat([real, torch.tensor([_DUMMY_SLOT])]),
        "ctx_len": ctx_len,
        "ctx_position_offset": torch.zeros(_BATCH, dtype=torch.long),
        "block_tables": torch.randperm(_BATCH * _PAGES_PER_ROW, generator=g)
        .view(_BATCH, _PAGES_PER_ROW)
        .to(torch.int32),
        "pools": pools,
    }
    # Pinned, so a step's inputs are staged asynchronously, as the executor stages them.
    return {
        name: [t.pin_memory() for t in value] if name == "pools" else value.pin_memory()
        for name, value in host.items()
    }


_STEP_INPUTS = ("logits", "draft_tokens", "captured", "slots", "block_tables")


def _load(rig: _Rig, inputs: dict, names=None) -> None:
    """Copy inputs into the rig's fixed tensors, asynchronously."""
    w = rig.worker
    targets = {
        "logits": rig.logits,
        "draft_tokens": rig.draft_tokens,
        "captured": rig.captured,
        "slots": rig.slots,
        "block_tables": w._ctx_block_tables,
        "ctx_len": w._ctx_len,
        "ctx_position_offset": w._ctx_position_offset,
    }
    for name in names or (*targets, "pools"):
        if name == "pools":
            for dst, src in zip(rig.pools, inputs["pools"]):
                dst.copy_(src, non_blocking=True)
        else:
            targets[name].copy_(inputs[name], non_blocking=True)


def _step(rig: _Rig) -> dict:
    """One generation step of the worker: acceptance, the fused drafter-input preparation from
    the unified draft cache, the drafter's block forward."""
    w, c = rig.worker, rig.container
    accepted, num_accepted = w.sample_and_accept_draft_tokens(rig.logits, rig.attn, rig.spec)
    inputs = w._prepare_paged_latent_gen_inputs(
        accepted,
        num_accepted,
        rig.attn,
        rig.spec,
        c,
        _BATCH * rig.block,
        c.mask_token_id,
        getattr(c, "mask_token_embedding", None),
    )
    hidden = c.dflash_forward(**{name: inputs[name] for name in _DFLASH_FORWARD_KW})
    return {
        "accepted": accepted,
        "num_accepted": num_accepted,
        "num_ctx": inputs["num_ctx_per_req"],
        "query_positions": inputs["query_positions"],
        "hidden": hidden,
    }


def _snapshot(rig: _Rig, out: dict) -> dict:
    snap = {}
    for name, t in out.items():
        if name == "accepted":
            # Only each row's accepted prefix is defined; acceptance leaves the rest unwritten.
            defined = torch.arange(t.shape[1], device=t.device) < out["num_accepted"].unsqueeze(1)
            t = torch.where(defined, t, torch.full_like(t, -1))
        snap[f"out.{name}"] = t.clone()
    snap.update({f"state.{name}": t.clone() for name, t in rig.state().items()})
    return snap


def _assert_same(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for name in want:
        a, b = got[name], want[name]
        if a.dtype == _FP8:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"{name} differs"
    dummy = want["state.ctx_len"][_DUMMY_SLOT].item()
    assert dummy == 0, f"the dummy slot holds a context of {dummy}"


_A_CTX, _B_CTX = (30, 63, 97), (33, 64, 150)


def _cuda_case(case: _Case) -> None:
    if case.helix:
        pytest.skip("the helix drafter runs across CP ranks: its capture is the multi-rank test's")


# --- (a) both acceptance paths end in the forced-acceptance hook ---


def _spy_on_hook(worker) -> list:
    calls = []
    inner = worker._apply_force_accepted_tokens

    def hook(num_accepted, num_contexts, runtime_draft_len, spec_metadata=None):
        calls.append((runtime_draft_len, spec_metadata))
        return inner(num_accepted, num_contexts, runtime_draft_len, spec_metadata=spec_metadata)

    worker._apply_force_accepted_tokens = hook
    return calls


@_GPU
@pytest.mark.parametrize("k", [5, 7])
@pytest.mark.parametrize("case", _cases())
def test_both_accept_paths_call_the_force_hook(case, k, monkeypatch):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    rig = _rig(case, k, torch.bfloat16, monkeypatch, force=k)
    _load(rig, _inputs(rig, 11, _A_CTX))
    calls = _spy_on_hook(rig.worker)
    _, num = rig.worker.sample_and_accept_draft_tokens(rig.logits, rig.attn, rig.spec)
    assert len(calls) == 1, f"greedy acceptance called the hook {len(calls)} times"
    assert calls[0][0] == k and calls[0][1] is rig.spec
    assert torch.equal(num.cpu(), torch.full((_BATCH,), k + 1, dtype=num.dtype))

    calls.clear()
    rows = _BATCH * (k + 1)
    dev = rig.logits.device
    sampled = SimpleNamespace(
        **{
            **vars(rig.spec),
            **dict(
                use_rejection_sampling=True,
                is_all_greedy_sample=False,
                advanced_sampling_mode=AdvancedSamplingMode.FULL,
                temperatures=torch.ones(rows, device=dev),
                top_ks=torch.zeros(rows, dtype=torch.int32, device=dev),
                top_ps=torch.ones(rows, device=dev),
                min_ps=torch.zeros(rows, device=dev),
                request_temperatures=torch.ones(_BATCH, device=dev),
                request_seeds=torch.arange(_BATCH, dtype=torch.int64, device=dev) + 1234,
                request_offsets=torch.zeros(_BATCH, dtype=torch.int64, device=dev),
                rng_accept_slot=0,
                batch_slot_ids=torch.arange(_BATCH, dtype=torch.int64, device=dev),
                draft_probs=None,
                num_seq_slots=_BATCH,
            ),
        }
    )
    _, num = rig.worker.sample_and_accept_draft_tokens(rig.logits, rig.attn, sampled)
    assert len(calls) == 1, f"sampled block verification called the hook {len(calls)} times"
    assert calls[0][0] == k and calls[0][1] is sampled
    assert torch.equal(num.cpu(), torch.full((_BATCH,), k + 1, dtype=num.dtype))


# --- (b) capture on inputs A, replay on B = eager on B; (g) no compile after warmup ---


class _CompileLog:
    """Counts Triton JIT cache misses and CuTe DSL compiles while active."""

    def __init__(self):
        self.compiled: list[str] = []
        self._undo: list = []

    def __enter__(self):
        import triton

        def hook(*args, **kwargs):
            fn = kwargs.get("fn")
            self.compiled.append(f"triton:{getattr(fn, 'name', fn)}")
            return False

        knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
        if knobs is not None and hasattr(knobs, "jit_cache_hook"):
            self._undo.append((knobs, "jit_cache_hook", knobs.jit_cache_hook))
            knobs.jit_cache_hook = hook
        else:
            from triton.runtime.jit import JITFunction

            self._undo.append((JITFunction, "cache_hook", JITFunction.cache_hook))
            JITFunction.cache_hook = hook
        try:
            import cutlass.cute as cute
        except ImportError:
            return self
        log = self.compiled

        class _Counting:
            def __init__(self, inner):
                self._inner = inner

            def __call__(self, fn, *args, **kwargs):
                log.append(f"cute:{getattr(fn, '__name__', fn)}")
                return self._inner(fn, *args, **kwargs)

            def __getitem__(self, options):
                return _Counting(self._inner[options])

            def __getattr__(self, name):
                return getattr(self._inner, name)

        self._undo.append((cute, "compile", cute.compile))
        cute.compile = _Counting(cute.compile)
        return self

    def __exit__(self, *exc):
        for owner, name, value in reversed(self._undo):
            setattr(owner, name, value)
        return False


def _warm(rig: _Rig, inputs: dict) -> None:
    _load(rig, inputs)
    _step(rig)
    torch.cuda.synchronize()


def _capture(rig: _Rig, inputs: dict):
    _load(rig, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _step(rig)
    torch.cuda.synchronize()
    return graph, out


@_GPU
@pytest.mark.parametrize("force", [0, 3], ids=["natural", "forced3"])
@pytest.mark.parametrize("pool_dtype", [_FP8, torch.bfloat16], ids=["fp8_history", "bf16_history"])
@pytest.mark.parametrize("k", [5, 7])
@pytest.mark.parametrize("case", _cases())
def test_replay_on_new_inputs_equals_eager(case, k, pool_dtype, force, monkeypatch):
    _cuda_case(case)
    rig = _rig(case, k, pool_dtype, monkeypatch, force)
    a, b = _inputs(rig, 1, _A_CTX), _inputs(rig, 2, _B_CTX)
    _warm(rig, a)
    with _CompileLog() as log:
        graph, out = _capture(rig, a)
        _load(rig, b)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _snapshot(rig, out)
        _load(rig, b)
        eager = _snapshot(rig, _step(rig))
    _assert_same(replayed, eager)
    assert not log.compiled, f"(g) compiled after the warmup pass: {log.compiled}"


# --- (c) overlap order = sequential; (d) no host sync in a step ---


def _host_advance(rig: _Rig, before: dict, num_accepted: torch.Tensor) -> torch.Tensor:
    """The context lengths after a step, as the host accounts them once it knows the acceptance."""
    ctx_len = before["ctx_len"].clone()
    for row, slot in enumerate(before["slots"].tolist()):
        if slot != _DUMMY_SLOT:
            ctx_len[slot] = min(ctx_len[slot].item() + int(num_accepted[row]), rig.worker._max_ctx)
    ctx_len[_DUMMY_SLOT] = 0
    return ctx_len


@_GPU
@pytest.mark.parametrize("force", [0, 3], ids=["natural", "forced3"])
@pytest.mark.parametrize("k", [5, 7])
@pytest.mark.parametrize("case", _cases())
def test_two_steps_in_overlap_order_equal_two_sequential_steps(case, k, force, monkeypatch):
    """Sequential: step one, the host learns its acceptance and sets the context lengths it
    accounts, then step two. Overlap order: step two is staged before step one's results reach
    the host, so it reads the lengths step one advanced on the device; no sync in between."""
    _cuda_case(case)
    rig = _rig(case, k, _FP8, monkeypatch, force)
    first, second = _inputs(rig, 3, _A_CTX), _inputs(rig, 4, _B_CTX)
    # Step two keeps step one's requests, slots, pages and draft cache.
    second["slots"] = first["slots"]
    second["block_tables"] = first["block_tables"]
    _warm(rig, first)

    _load(rig, first)
    out = _step(rig)
    torch.cuda.synchronize()
    accounted = _host_advance(rig, first, out["num_accepted"].cpu())
    rig.worker._ctx_len.copy_(accounted.cuda())
    _load(rig, second, _STEP_INPUTS)
    sequential = _snapshot(rig, _step(rig))

    _load(rig, first)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _step(rig)
        _load(rig, second, _STEP_INPUTS)
        overlapped = _snapshot(rig, _step(rig))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _assert_same(overlapped, sequential)


# --- (e) a specialized target refuses a private context ring ---


@_GPU
@pytest.mark.parametrize("case", _cases())
def test_specialized_worker_refuses_a_cache_that_is_not_unified(case, monkeypatch):
    module = _module(case)
    container = _build_container(case, module, 8, monkeypatch)
    worker = _build_worker(case, module, container, 7)
    spec = SimpleNamespace(max_num_requests=_BATCH, request_ids=[], is_cuda_graph=False)
    attn = SimpleNamespace(num_seqs=0, num_contexts=0, kv_cache_manager=None, max_seq_len=4096)
    for manager in (None, SimpleNamespace(has_standalone_draft_cache=False)):
        with pytest.raises(Exception) as refused:
            worker._lazy_init_ctx_buffers(container, spec, attn, manager)
        assert refused.type in (RuntimeError, ValueError, NotImplementedError) and (
            "unified" in str(refused.value).lower()
        ), f"refused with {refused.type.__name__}: {refused.value} (want: needs the unified cache)"


# --- (f) the container carries what the engine reads ---


@_GPU
@pytest.mark.parametrize("case", _cases())
def test_container_carries_what_the_engine_reads(case, monkeypatch):
    module = _module(case)
    container = _build_container(case, module, 8, monkeypatch)
    missing = [name for name in _CONTAINER_NAMES if not hasattr(container, name)]
    assert not missing, f"{_CONTAINER} lacks {missing}"
    assert container.supports_unified_draft_cache is True
    assert container._kv_factor == 1
    assert container._num_attn_layers == _L
    assert container._head_dim == _KVL + _ROPE_D
    assert not container._uses_lilicorr
    assert container.has_candidate_selector is True
    assert container.config.max_position_embeddings > 0
    assert container.draft_model_full.model.embed_tokens is not None
    for name in ("project_target_hidden", "precompute_context_kv", "dflash_forward"):
        assert callable(getattr(container, name))
    params = inspect.signature(container.dflash_forward).parameters
    assert set(_DFLASH_FORWARD_KW) <= set(params) or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    assert getattr(container, "unified_draft_cache_when_aggregated", False) is True
    assert container.context_kv_fp8 is True, "the drafter history is FP8 (the recipe's)"
    history = _build_container(case, module, 8, monkeypatch, history_only=True)
    assert history.context_kv_only is True
    for name in ("capture_projection_blocks", "finish_capture_projection"):
        assert callable(getattr(history, name, None)), f"a history-only drafter needs {name}"

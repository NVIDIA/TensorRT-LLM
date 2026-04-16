# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-weight-loading pipeline snapshot caching for AutoDeploy.

The cache boundary sits between the SHARDING and WEIGHT_LOAD stages.  All
parameters in the cached artifact live on the ``meta`` device — no real weights
are stored.  On restore the pipeline skips FACTORY, EXPORT, and SHARDING and
jumps straight to WEIGHT_LOAD, which loads and materialises the weights.

The current artifact format uses AD IR (AutoDeploy Intermediate Representation).
The FX graph is serialized as compressed structured JSON (``ad_ir.json.gz``), capturing every
node's op, target, args/kwargs, and placeholder shape metadata.  On restore the
``torch.fx.Graph`` is rebuilt node-by-node via the Graph construction API — **no
re-tracing** — so all structural and metadata invariants are preserved.

See ``ad_ir.py`` for the IR data model and serialization utilities.
AD-pipeline hooks are rebuilt from declarative ``hook_specs`` embedded in the IR.
Source-model hooks are replayed from a fresh factory-built model and validated
against the saved hook fingerprint.
"""

import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm.version import __version__ as TRTLLM_VERSION

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from .ad_ir import (
    BUFFERS_FILE_NAME,
    GRAPH_SCHEMA_VERSION,
    HOOK_SCHEMA_VERSION,
    IR_FILE_NAME,
    IRGraph,
    build_graph_module,
    extract_ir,
    hydrate_shapes,
    load_ir,
    save_ir,
)
from .interface import (
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)

# Bump when the cache contract itself changes (layout, manifest fields,
# save/restore protocol). Restore rejects any mismatch.
CACHE_CONTRACT_VERSION = 2
MANIFEST_FILE_NAME = "manifest.json"

# Module path prefix for the auto_deploy package. Used by the producer-hash
# closure walk to bound which source files count toward cache invalidation.
_AD_MANAGED_MODULE_PREFIX = "tensorrt_llm._torch.auto_deploy."

# Sub-packages that introduce pipeline-managed hooks (sharding, quantization,
# dedup, alias). A hook whose defining module starts with one of these prefixes
# MUST match a declarative HookSpec, otherwise the save is rejected.
# Hooks from elsewhere — including custom-model modules under
# ``auto_deploy/models/custom/`` — are treated as source-model hooks and go
# through the factory-replay fingerprint path.
_AD_PIPELINE_HOOK_PREFIXES: Tuple[str, ...] = (
    "tensorrt_llm._torch.auto_deploy.transform.",
    "tensorrt_llm._torch.auto_deploy.export.",
)


def _canonicalize_for_hash(value: Any) -> Any:
    """Normalize nested objects into a deterministic structure for hashing."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")

    if isinstance(value, dict):
        return {
            str(key): _canonicalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Enum):
        return _canonicalize_for_hash(value.value)
    if hasattr(value, "to_dict"):
        return _canonicalize_for_hash(value.to_dict())
    return value


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonicalize_for_hash(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any, sort_keys: bool = True) -> None:
    """Write *payload* to *path* atomically with fsync.

    Writes to ``<path>.tmp``, fsyncs the file, renames into place, and fsyncs
    the parent directory so power-loss leaves either the old manifest or the
    new one — never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory fd (POSIX only)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_publish_rank_dir(tmp_rank_dir: Path, rank_dir: Path) -> None:
    """Atomically replace *rank_dir* with the contents of *tmp_rank_dir*.

    Pattern: if ``rank_dir`` exists, move it aside to ``rank_dir.old.<pid>``
    first, then rename the temp dir into place. The old dir is only removed
    after the rename succeeds. A concurrent reader never observes a missing
    ``rank_dir``; at worst it sees the previous snapshot.
    """
    old_dir: Optional[Path] = None
    if rank_dir.exists():
        old_dir = rank_dir.with_name(f"{rank_dir.name}.old.{os.getpid()}")
        rank_dir.rename(old_dir)
    try:
        tmp_rank_dir.rename(rank_dir)
        _fsync_dir(rank_dir.parent)
    except OSError:
        # Roll back: put the old dir back so the cache remains consistent.
        if old_dir is not None and old_dir.exists() and not rank_dir.exists():
            old_dir.rename(rank_dir)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _repo_git_sha() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_repo_root()))
    except ValueError:
        return str(resolved)


def _describe_source_object(obj: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "module": getattr(obj, "__module__", type(obj).__module__),
        "qualname": getattr(obj, "__qualname__", type(obj).__qualname__),
    }

    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        source = None

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except (OSError, TypeError):
        source_file = None

    if source is not None:
        payload["source_hash"] = _hash_text(source)
    elif source_file is not None:
        file_path = Path(source_file)
        payload["source_file"] = _repo_relative_path(file_path)
        try:
            payload["source_file_hash"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            payload["source_file_hash"] = None

    if source_file is not None and "source_file" not in payload:
        payload["source_file"] = _repo_relative_path(Path(source_file))

    return payload


def _describe_file(path: Path) -> Dict[str, Any]:
    """Return repo-relative path + SHA256. Raises if the file cannot be read.

    Missing files are a hard error: silently weakening the producer hash would
    let stale artifacts restore after a file rename/deletion.
    """
    payload: Dict[str, Any] = {"path": _repo_relative_path(path)}
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


# Per-process cache of module-closure file lists, keyed by transform module.
# A closure is a pure function of the import graph in that module, so it only
# needs to be recomputed when a relevant file changes on disk.
_MODULE_CLOSURE_CACHE: Dict[str, Tuple[Path, ...]] = {}


def _auto_deploy_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_auto_deploy_module(module: Any) -> bool:
    module_name = getattr(module, "__name__", "")
    if not module_name.startswith(_AD_MANAGED_MODULE_PREFIX):
        return False
    # Only modules with a real source file participate in the closure hash.
    source_file = getattr(module, "__file__", None)
    return bool(source_file and Path(source_file).exists())


def _transform_module_closure(root_module_name: str) -> Tuple[Path, ...]:
    """Return the set of auto_deploy source files reachable from *root_module_name*.

    The closure is computed by walking ``sys.modules`` starting from the root
    module, following each module's public attributes that resolve to other
    loaded modules under the ``auto_deploy`` package. Third-party dependencies
    (``torch``, ``transformers``, ...) are excluded — their versions are covered
    by dedicated manifest fields.
    """
    cached = _MODULE_CLOSURE_CACHE.get(root_module_name)
    if cached is not None:
        return cached

    root_module = sys.modules.get(root_module_name)
    if root_module is None:
        # Import if not already loaded. Fail if the module path is invalid —
        # we cannot safely produce a hash for a non-existent transform module.
        root_module = importlib.import_module(root_module_name)

    visited: Set[str] = set()
    files: Set[Path] = set()
    stack: List[Any] = [root_module]

    while stack:
        mod = stack.pop()
        mod_name = getattr(mod, "__name__", None)
        if not mod_name or mod_name in visited:
            continue
        if not _is_auto_deploy_module(mod):
            continue
        visited.add(mod_name)
        files.add(Path(mod.__file__).resolve())

        for attr_name in dir(mod):
            if attr_name.startswith("__"):
                continue
            try:
                obj = getattr(mod, attr_name)
            except AttributeError:
                continue
            # Follow submodule references.
            if inspect.ismodule(obj):
                if _is_auto_deploy_module(obj) and obj.__name__ not in visited:
                    stack.append(obj)
                continue
            # Follow classes and functions to their defining module.
            obj_module_name = getattr(obj, "__module__", None)
            if not obj_module_name or not obj_module_name.startswith(_AD_MANAGED_MODULE_PREFIX):
                continue
            obj_module = sys.modules.get(obj_module_name)
            if obj_module is not None and obj_module.__name__ not in visited:
                stack.append(obj_module)

    sorted_files = tuple(sorted(files))
    _MODULE_CLOSURE_CACHE[root_module_name] = sorted_files
    return sorted_files


def _build_producer_hash(prefix_transform_names: Sequence[str]) -> str:
    """Hash the source closure of each in-prefix transform.

    Only transforms in the cache prefix (at or before the boundary) contribute.
    Post-boundary transforms (``fuse_moe``, ``fuse_gemm``, ``compile``, ...) are
    excluded: they run fresh on every invocation after restore and their code
    changes do not affect the cached artifact.

    For transforms whose module lives under the ``auto_deploy`` package, we
    also hash every ``auto_deploy`` module reachable via its import graph —
    this captures library helpers (``library/sharding.py``, ``utils/_graph.py``,
    etc.) without blanket-hashing unrelated directories.

    For transforms defined outside the package (e.g. unit-test fixtures), we
    fall back to hashing the transform source alone. Production pipelines do
    not register transforms outside the package, so this path exists only to
    keep tests usable.
    """
    entries: List[Dict[str, Any]] = []
    for name in prefix_transform_names:
        transform_cls = TransformRegistry.get(name)
        transform_mod_name = getattr(transform_cls, "__module__", "")
        entry: Dict[str, Any] = {
            "name": name,
            "transform_source": _describe_source_object(transform_cls),
        }
        if transform_mod_name.startswith(_AD_MANAGED_MODULE_PREFIX):
            closure_files = _transform_module_closure(transform_mod_name)
            entry["closure_files"] = [_describe_file(p) for p in closure_files]
        else:
            entry["closure_files"] = []  # out-of-package transform: source-only
        entries.append(entry)
    return _hash_payload({"prefix_transforms": entries})


# ---------------------------------------------------------------------------
# HookSpec: declarative representation of load hooks
# ---------------------------------------------------------------------------


def _hook_defining_module(hook: Any) -> Optional[str]:
    """Return the module path where the hook's implementation is defined.

    For a ``functools.partial``, the underlying function is used. For a bound
    method, ``__self__``'s type defines the hook. For a plain function/closure,
    ``__module__`` is used directly.
    """
    if isinstance(hook, partial):
        return _hook_defining_module(hook.func)
    bound_self = getattr(hook, "__self__", None)
    if bound_self is not None:
        return type(bound_self).__module__
    return getattr(hook, "__module__", None)


def _identify_hook(hook: Any, scope: str = "root") -> Optional[Dict[str, Any]]:
    """Pattern-match a live hook object and return its declarative HookSpec dict.

    Returns ``None`` if the hook cannot be identified. Callers must treat a
    ``None`` return from an AD-managed module as a save rejection.
    """
    # --- functools.partial-based hooks ---
    func = getattr(hook, "func", None)
    kw = getattr(hook, "keywords", {})

    if func is not None:
        func_name = getattr(func, "__name__", "")

        # _load_hook (TP sharding) from sharding.py
        if func_name == "_load_hook" and "param_key" in kw:
            return _identify_shard_tp_hook(kw, scope)

        # _load_hook_remove from sharding.py
        if func_name == "_load_hook_remove" and "param_key" in kw:
            return {
                "type": "remove",
                "scope": scope,
                "position": "pre",
                "param_key": kw["param_key"],
            }

        # _load_hook_for_deduplication from export.py
        if func_name == "_load_hook_for_deduplication":
            return {
                "type": "dedup",
                "scope": scope,
                "position": "pre",
                "param_key_remaining": kw.get("param_key_remaining", ""),
                "param_key_removed": kw.get("param_key_removed", ""),
            }

        # shard_load_hook (quantization-aware sharding) — bound method via partial
        if func_name == "shard_load_hook" and "weight_name" in kw:
            return _identify_shard_quant_hook(func, kw, scope)

        # Quantization load_hook — partial(self.load_hook, weight_name=...)
        if func_name == "load_hook" and "weight_name" in kw:
            return _identify_quant_load_hook(func, kw, scope)

        # Quantization convert_amax_hook — partial(self.convert_amax_hook, ...)
        if func_name == "convert_amax_hook" and "scale_name" in kw:
            return _identify_quant_amax_hook(func, kw, scope)

        # Quantization post_load_hook — partial(self.post_load_hook, weight_name=...)
        if func_name == "post_load_hook" and "weight_name" in kw:
            return _identify_quant_post_load_hook(func, kw, scope)

    # --- closure-based hooks ---
    qualname = getattr(hook, "__qualname__", "")

    # aliasing_load_pre_hook from export.py (via _build_aliasing_load_pre_hook)
    if "aliasing_load_pre_hook" in qualname:
        return _identify_alias_hook(hook, scope)

    # Anything else: classify by defining module.
    # - Hooks defined under the AD pipeline prefixes (transform/, export/)
    #   that did not match a declarative pattern are unknown and must block
    #   the save — those hooks can only be restored via a registered HookSpec.
    # - Everything else (source-model hooks, including custom-model hooks
    #   under auto_deploy/models/custom/) is validated against a fresh
    #   factory-built model during restore.
    if callable(hook):
        hook_module = _hook_defining_module(hook) or ""
        if any(hook_module.startswith(p) for p in _AD_PIPELINE_HOOK_PREFIXES):
            # Unknown pipeline-managed hook — refuse to save. Returning None
            # makes collect_hook_specs set has_unknown=True.
            return None
        return {
            "type": "source_model",
            "scope": scope,
            "position": "pre",
            "hook_identity": _describe_hook_identity(hook),
        }

    return None


def _identify_shard_tp_hook(kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Extract HookSpec for a ``_load_hook`` partial (TP sharding)."""
    f_split = kw.get("f_split")
    param_shape = list(kw.get("param_shape", []))

    spec: Dict[str, Any] = {
        "type": "shard_tp",
        "scope": scope,
        "position": "pre",
        "param_key": kw["param_key"],
        "param_shape": param_shape,
    }

    # Try to extract the split parameters from f_split
    if isinstance(f_split, partial):
        # partial(_split_tensor_for_tp, dim=..., rank=..., world_size=..., min_local_shape=...)
        split_kw = f_split.keywords
        spec["dim"] = split_kw.get("dim", 0)
        spec["rank"] = split_kw.get("rank", 0)
        spec["world_size"] = split_kw.get("world_size", 1)
        spec["min_local_shape"] = split_kw.get("min_local_shape", 1)
        spec["fused_weight_dims"] = None
    elif callable(f_split):
        # Nested closure — try to extract from closure cells or defaults
        defaults = getattr(f_split, "__defaults__", None) or ()
        code = getattr(f_split, "__code__", None)

        fused_dims = None
        dim = 0
        if defaults:
            # f_split(t, fused_dims=fused_weight_dims, d=dim) has defaults
            fused_dims = defaults[0] if len(defaults) >= 1 else None
            dim = defaults[1] if len(defaults) >= 2 else 0

        if fused_dims is not None and isinstance(fused_dims, (list, tuple)):
            spec["fused_weight_dims"] = list(fused_dims)
            spec["dim"] = dim
            # Extract rank/world_size from the closure cells
            cells = _extract_closure_vars(f_split)
            spec["rank"] = cells.get("rank", 0)
            spec["world_size"] = cells.get("world_size", 1)
            spec["min_local_shape"] = cells.get("min_local_shape", 1)
        else:
            # slice_tensor lambda t: t[start:end] — BMM sharding
            spec["type"] = "shard_slice"
            cells = _extract_closure_vars(f_split)
            defaults = getattr(f_split, "__defaults__", ()) or ()
            spec["start_idx"] = cells.get("start_idx", defaults[0] if defaults else 0)
            spec["end_idx"] = cells.get("end_idx", defaults[1] if len(defaults) > 1 else 0)
            # Also check if it's a simple closure with start_idx/end_idx
            if code and code.co_freevars:
                if "start_idx" in code.co_freevars or "end_idx" in code.co_freevars:
                    pass  # already extracted from cells
            spec.pop("fused_weight_dims", None)
    else:
        spec["dim"] = 0
        spec["rank"] = 0
        spec["world_size"] = 1
        spec["min_local_shape"] = 1
        spec["fused_weight_dims"] = None

    return spec


def _identify_shard_quant_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Extract HookSpec for a ``shard_load_hook`` partial (quant sharding)."""
    # func is a bound method like instance.shard_load_hook
    self_obj = getattr(func, "__self__", None)
    quant_class = type(self_obj).__name__ if self_obj is not None else "unknown"
    fused_weight_dims = getattr(self_obj, "fused_weight_dims", None)

    return {
        "type": "shard_quant",
        "scope": scope,
        "position": "pre",
        "quant_class": quant_class,
        "weight_name": kw.get("weight_name", ""),
        "weight_original_shape": list(kw.get("weight_original_shape", [])),
        "dim": kw.get("dim", 0),
        "rank": kw.get("rank", 0),
        "world_size": kw.get("world_size", 1),
        "min_local_shape": kw.get("min_local_shape", 1),
        "fused_weight_dims": list(fused_weight_dims) if fused_weight_dims else None,
    }


def _quant_instance_payload(self_obj: Any) -> Dict[str, Any]:
    """Capture the subset of a quant-info instance that its hooks read.

    We dump Pydantic model state when available, else copy public attrs. The
    payload is JSON-roundtripped at save time; anything that doesn't survive
    raises, so no silently-dropped fields make it into the cache.
    """
    if self_obj is None:
        return {}
    if _has_model_dump(self_obj):
        try:
            data = self_obj.model_dump(mode="json")
            json.dumps(data)
            return data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Pipeline cache: quant-info instance {type(self_obj).__qualname__} "
                f"has non-JSON-serializable fields: {exc}"
            ) from exc
    # Plain objects: extract public attributes that are JSON-safe.
    payload: Dict[str, Any] = {}
    for attr_name in dir(self_obj):
        if attr_name.startswith("_") or callable(getattr(self_obj, attr_name, None)):
            continue
        try:
            val = getattr(self_obj, attr_name)
            json.dumps(val)
            payload[attr_name] = val
        except (TypeError, ValueError, AttributeError):
            continue
    return payload


def _has_model_dump(obj: Any) -> bool:
    return hasattr(obj, "model_dump") and callable(obj.model_dump)


def _identify_quant_load_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Extract HookSpec for a quantization ``load_hook`` partial."""
    self_obj = getattr(func, "__self__", None)
    quant_class = type(self_obj).__qualname__ if self_obj is not None else "unknown"
    quant_module = type(self_obj).__module__ if self_obj is not None else ""
    return {
        "type": "quant_load",
        "scope": scope,
        "position": "pre",
        "quant_class": quant_class,
        "quant_module": quant_module,
        "instance_payload": _quant_instance_payload(self_obj),
        "weight_name": kw.get("weight_name", ""),
    }


def _identify_quant_amax_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Extract HookSpec for a quantization ``convert_amax_hook`` partial."""
    self_obj = getattr(func, "__self__", None)
    quant_class = type(self_obj).__qualname__ if self_obj is not None else "unknown"
    quant_module = type(self_obj).__module__ if self_obj is not None else ""
    return {
        "type": "quant_amax",
        "scope": scope,
        "position": "pre",
        "quant_class": quant_class,
        "quant_module": quant_module,
        "instance_payload": _quant_instance_payload(self_obj),
        "scale_name": kw.get("scale_name", ""),
        "amax_name": kw.get("amax_name", ""),
    }


def _identify_quant_post_load_hook(
    func: Callable, kw: Dict[str, Any], scope: str
) -> Dict[str, Any]:
    """Extract HookSpec for a quantization ``post_load_hook`` partial."""
    self_obj = getattr(func, "__self__", None)
    quant_class = type(self_obj).__qualname__ if self_obj is not None else "unknown"
    quant_module = type(self_obj).__module__ if self_obj is not None else ""
    return {
        "type": "quant_post_load",
        "scope": scope,
        "position": "post",
        "quant_class": quant_class,
        "quant_module": quant_module,
        "instance_payload": _quant_instance_payload(self_obj),
        "weight_name": kw.get("weight_name", ""),
    }


def _identify_alias_hook(hook: Callable, scope: str) -> Optional[Dict[str, Any]]:
    """Extract HookSpec for an aliasing load hook."""
    cells = _extract_closure_vars(hook)
    aliased_groups = cells.get("aliased_groups")
    if aliased_groups is not None and isinstance(aliased_groups, list):
        return {
            "type": "alias",
            "scope": scope,
            "position": "pre",
            "aliased_groups": aliased_groups,
        }
    return None


def _describe_hook_identity(hook: Any) -> Dict[str, Any]:
    """Build a JSON-serializable identity for a live hook callable."""
    if isinstance(hook, partial):
        return {
            "kind": "partial",
            "callable": _describe_hook_identity(hook.func),
            "args_hash": _hash_payload({"args": list(hook.args)}),
            "keywords_hash": _hash_payload({"keywords": dict(hook.keywords or {})}),
            "keyword_names": sorted(str(key) for key in (hook.keywords or {}).keys()),
        }

    bound_self = getattr(hook, "__self__", None)
    func = getattr(hook, "__func__", None)
    if func is not None:
        return {
            "kind": "bound_method" if bound_self is not None else "function",
            "callable": _describe_source_object(func),
            "owner_type": None if bound_self is None else type(bound_self).__qualname__,
            "owner_module": None if bound_self is None else type(bound_self).__module__,
        }

    if inspect.isfunction(hook):
        return {
            "kind": "function",
            "callable": _describe_source_object(hook),
        }

    if hasattr(hook, "__call__"):
        return {
            "kind": "callable_object",
            "callable": _describe_source_object(type(hook).__call__),
            "owner_type": type(hook).__qualname__,
            "owner_module": type(hook).__module__,
        }

    return {
        "kind": "unknown",
        "repr": repr(hook),
    }


def _extract_closure_vars(func: Callable) -> Dict[str, Any]:
    """Extract variables captured in a closure's cells."""
    result: Dict[str, Any] = {}
    closure = getattr(func, "__closure__", None)
    code = getattr(func, "__code__", None)
    if closure is None or code is None:
        return result
    for name, cell in zip(code.co_freevars, closure):
        try:
            result[name] = cell.cell_contents
        except ValueError:
            pass
    return result


def collect_hook_specs(gm: GraphModule) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect declarative HookSpec dicts from all load hooks on *gm*.

    Returns:
        A tuple ``(specs, has_unknown)`` where *has_unknown* is ``True`` if any
        hook could not be identified.
    """
    specs: List[Dict[str, Any]] = []
    has_unknown = False
    unknown_counts: Dict[str, int] = {}

    def _collect_from_module(mod: nn.Module, scope: str) -> None:
        nonlocal has_unknown
        for hook in mod._load_state_dict_pre_hooks.values():
            with_module = bool(getattr(hook, "with_module", False))
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is not None:
                spec["with_module"] = with_module
                specs.append(spec)
            else:
                qualname = getattr(hook_obj, "__qualname__", repr(hook_obj))
                unknown_counts[qualname] = unknown_counts.get(qualname, 0) + 1
                has_unknown = True

        for hook in mod._load_state_dict_post_hooks.values():
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is not None:
                spec["position"] = "post"
                specs.append(spec)
            else:
                qualname = getattr(hook_obj, "__qualname__", repr(hook_obj))
                unknown_counts[qualname] = unknown_counts.get(qualname, 0) + 1
                has_unknown = True

    _collect_from_module(gm, "root")
    for name, child in gm.named_modules():
        if name:  # skip root, already collected
            _collect_from_module(child, name)

    for qualname, count in unknown_counts.items():
        ad_logger.warning(f"Pipeline cache: {count} unrecognized hook(s) of type {qualname}")

    return specs, has_unknown


def _source_model_hook_spec_key(spec: Mapping[str, Any]) -> str:
    normalized = {
        "type": spec.get("type"),
        "scope": spec.get("scope"),
        "position": spec.get("position"),
        "with_module": bool(spec.get("with_module", False)),
        "hook_identity": spec.get("hook_identity"),
    }
    return json.dumps(_canonicalize_for_hash(normalized), sort_keys=True, separators=(",", ":"))


def _collect_live_source_model_hooks(model: nn.Module) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    def _collect_from_module(mod: nn.Module, scope: str) -> None:
        for hook in mod._load_state_dict_pre_hooks.values():
            with_module = bool(getattr(hook, "with_module", False))
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is None or spec.get("type") != "source_model":
                continue
            spec["with_module"] = with_module
            records.append(
                {
                    "scope": scope,
                    "position": "pre",
                    "with_module": with_module,
                    "hook_fn": hook_obj,
                    "spec": spec,
                }
            )

        for hook in mod._load_state_dict_post_hooks.values():
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is None or spec.get("type") != "source_model":
                continue
            spec["position"] = "post"
            records.append(
                {
                    "scope": scope,
                    "position": "post",
                    "with_module": False,
                    "hook_fn": hook_obj,
                    "spec": spec,
                }
            )

    _collect_from_module(model, "root")
    for name, child in model.named_modules():
        if name:
            _collect_from_module(child, name)

    return records


# ---------------------------------------------------------------------------
# HookSpec restore: rebuild live hooks from specs
# ---------------------------------------------------------------------------


def reattach_hooks(gm: GraphModule, specs: List[Dict[str, Any]]) -> None:
    """Rebuild and register load hooks on *gm* from declarative HookSpec dicts."""
    for spec in specs:
        if spec.get("type") == "source_model":
            continue

        hook_fn = _rebuild_hook(spec)
        if hook_fn is None:
            raise ValueError(
                f"Pipeline cache: could not rebuild hook type={spec.get('type')!r} "
                f"scope={spec.get('scope')!r}."
            )

        scope = spec.get("scope", "root")
        position = spec.get("position", "pre")

        target_mod = gm if scope == "root" else gm.get_submodule(scope)

        if position == "pre":
            with_module = bool(spec.get("with_module", False))
            if with_module and not bool(spec.get("requires_module", False)):
                hook_fn = _wrap_pre_hook_with_module_adapter(hook_fn)
            target_mod._register_load_state_dict_pre_hook(
                hook_fn,
                with_module=with_module,
            )
        else:
            target_mod.register_load_state_dict_post_hook(hook_fn)


def _rebuild_hook(spec: Dict[str, Any]) -> Optional[Callable]:
    """Rebuild a single live hook from its HookSpec dict.

    Returns ``None`` only for ``source_model`` specs (handled by the
    factory-replay path). Every other supported ``type`` must return a
    callable; unsupported types raise so the caller can propagate a miss.
    """
    hook_type = spec.get("type")

    if hook_type == "shard_tp":
        return _rebuild_shard_tp_hook(spec)
    if hook_type == "shard_slice":
        return _rebuild_shard_slice_hook(spec)
    if hook_type == "shard_quant":
        return _rebuild_shard_quant_hook(spec)
    if hook_type == "dedup":
        return _rebuild_dedup_hook(spec)
    if hook_type == "alias":
        return _rebuild_alias_hook(spec)
    if hook_type == "remove":
        return _rebuild_remove_hook(spec)
    if hook_type in ("quant_load", "quant_amax", "quant_post_load"):
        return _rebuild_quant_hook(spec)
    if hook_type == "source_model":
        return None
    raise ValueError(f"Pipeline cache: unknown hook spec type {hook_type!r}")


def _wrap_pre_hook_with_module_adapter(hook_fn: Callable) -> Callable:
    """Adapt a module-agnostic pre-hook for ``with_module=True`` registration."""

    def with_module_adapter(
        module: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        *args,
        **kwargs,
    ) -> None:
        del module
        hook_fn(state_dict, prefix, *args, **kwargs)

    return with_module_adapter


def _rebuild_shard_tp_hook(spec: Dict[str, Any]) -> Callable:
    from ..transform.library.sharding import _load_hook, _split_tensor_for_tp

    dim = spec["dim"]
    rank = spec["rank"]
    world_size = spec["world_size"]
    min_local_shape = spec.get("min_local_shape", 1)
    fused_weight_dims = spec.get("fused_weight_dims")

    if fused_weight_dims:

        def f_split(
            t: torch.Tensor,
            fused_dims: list = fused_weight_dims,
            d: int = dim,
        ) -> torch.Tensor:
            return torch.cat(
                [
                    _split_tensor_for_tp(w, dim, rank, world_size, min_local_shape)
                    for w in torch.split(t, fused_dims, dim=d)
                ],
                dim=d,
            )

    else:
        f_split = partial(
            _split_tensor_for_tp,
            dim=dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
        )

    return partial(
        _load_hook,
        f_split=f_split,
        param_key=spec["param_key"],
        param_shape=torch.Size(spec["param_shape"]),
    )


def _rebuild_shard_slice_hook(spec: Dict[str, Any]) -> Callable:
    from ..transform.library.sharding import _load_hook

    start_idx = spec["start_idx"]
    end_idx = spec["end_idx"]

    def slice_tensor(t: torch.Tensor) -> torch.Tensor:
        return t[start_idx:end_idx]

    return partial(
        _load_hook,
        f_split=slice_tensor,
        param_key=spec["param_key"],
        param_shape=torch.Size(spec["param_shape"]),
    )


def _rebuild_shard_quant_hook(spec: Dict[str, Any]) -> Callable:
    """Rebuild a ``shard_load_hook`` partial for a QuantizationShardingMixin subclass.

    The shard_load_hook methods on the current sharding-info classes read only
    ``self.fused_weight_dims`` (plus their method kwargs). We rebuild the class
    with just the fields we persisted; any future state dependency must show
    up as a field on the spec so restore stays explicit.
    """
    from ..transform.interface import Stages
    from ..transform.library.sharding import (
        FineGrainedFP8WeightShardingInfo,
        FP4WeightShardingInfo,
        FP8WeightShardingInfo,
        ShardingTransformConfig,
        SplitDimension,
        WeightShardingInfo,
    )

    _QUANT_CLASSES: Dict[str, type] = {
        "WeightShardingInfo": WeightShardingInfo,
        "FP8WeightShardingInfo": FP8WeightShardingInfo,
        "FineGrainedFP8WeightShardingInfo": FineGrainedFP8WeightShardingInfo,
        "FP4WeightShardingInfo": FP4WeightShardingInfo,
    }

    quant_class = spec.get("quant_class", "")
    cls = _QUANT_CLASSES.get(quant_class)
    if cls is None:
        raise ValueError(
            f"Pipeline cache: unknown quant sharding class {quant_class!r}. "
            f"Known classes: {sorted(_QUANT_CLASSES)}"
        )

    fused_weight_dims = spec.get("fused_weight_dims")
    # ShardingInfo classes are Pydantic models; target_node/config/split_dim
    # are required fields on the base type but shard_load_hook reads only
    # fused_weight_dims. Constructing with placeholders keeps the contract
    # explicit: if a future refactor makes shard_load_hook depend on any of
    # the other fields, the restored hook will produce different tensors and
    # a round-trip test must catch it.
    instance = cls(
        target_node="",
        config=ShardingTransformConfig(stage=Stages.SHARDING),
        split_dim=SplitDimension.COLUMN,
        fused_weight_dims=tuple(fused_weight_dims) if fused_weight_dims else None,
    )

    return partial(
        instance.shard_load_hook,
        weight_name=spec["weight_name"],
        weight_original_shape=torch.Size(spec["weight_original_shape"]),
        dim=spec["dim"],
        rank=spec["rank"],
        world_size=spec["world_size"],
        min_local_shape=spec.get("min_local_shape", 1),
    )


def _rebuild_dedup_hook(spec: Dict[str, Any]) -> Callable:
    from ..export.export import _load_hook_for_deduplication

    return partial(
        _load_hook_for_deduplication,
        param_key_remaining=spec["param_key_remaining"],
        param_key_removed=spec["param_key_removed"],
    )


def _rebuild_alias_hook(spec: Dict[str, Any]) -> Callable:
    from ..export.export import _build_aliasing_load_pre_hook

    return _build_aliasing_load_pre_hook(spec["aliased_groups"])


def _rebuild_remove_hook(spec: Dict[str, Any]) -> Callable:
    from ..transform.library.sharding import _load_hook_remove

    return partial(_load_hook_remove, param_key=spec["param_key"])


def _resolve_nested_class(mod: Any, class_path: str) -> Optional[type]:
    """Resolve a possibly-nested class name (``"Outer.Inner"``) on *mod*."""
    cls: Any = mod
    for part in class_path.split("."):
        cls = getattr(cls, part, None)
        if cls is None:
            return None
    return cls if isinstance(cls, type) else None


def _rebuild_quant_hook(spec: Dict[str, Any]) -> Callable:
    """Rebuild a quantization load/amax/post_load hook from its spec.

    The instance is reconstructed with the JSON payload captured at save time:
    Pydantic models use ``model_validate``; plain classes get ``__new__`` +
    ``setattr``. We raise on any failure so the restore path remains
    fail-closed.
    """
    quant_module = spec.get("quant_module", "")
    quant_class = spec.get("quant_class", "")

    try:
        mod = importlib.import_module(quant_module)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Pipeline cache: cannot import module {quant_module!r} for hook restore: {exc}"
        ) from exc

    cls = _resolve_nested_class(mod, quant_class)
    if cls is None:
        raise ValueError(
            f"Pipeline cache: cannot reconstruct quant class {quant_module}.{quant_class}"
        )

    payload = spec.get("instance_payload", {}) or {}
    instance: Any
    if _has_model_dump(cls) or hasattr(cls, "model_validate"):
        try:
            instance = cls.model_validate(payload)
        except Exception as exc:  # Pydantic ValidationError + any subclass
            raise ValueError(
                f"Pipeline cache: Pydantic validation failed for {quant_class}: {exc}"
            ) from exc
    else:
        instance = cls.__new__(cls)
        for attr_name, value in payload.items():
            setattr(instance, attr_name, value)

    hook_type = spec["type"]
    if hook_type == "quant_load":
        method = getattr(instance, "load_hook", None)
        if method is None:
            raise ValueError(f"Pipeline cache: {quant_class}.load_hook missing")
        return partial(method, weight_name=spec["weight_name"])
    if hook_type == "quant_amax":
        method = getattr(instance, "convert_amax_hook", None)
        if method is None:
            raise ValueError(f"Pipeline cache: {quant_class}.convert_amax_hook missing")
        return partial(method, scale_name=spec["scale_name"], amax_name=spec["amax_name"])
    if hook_type == "quant_post_load":
        method = getattr(instance, "post_load_hook", None)
        if method is None:
            raise ValueError(f"Pipeline cache: {quant_class}.post_load_hook missing")
        return partial(method, weight_name=spec["weight_name"])
    raise ValueError(f"Pipeline cache: unknown quant hook type {hook_type!r}")


# ---------------------------------------------------------------------------
# PipelineSnapshotManager
# ---------------------------------------------------------------------------


class PipelineSnapshotManager:
    """Save and restore pre-weight-loading AutoDeploy pipeline snapshots."""

    def __init__(
        self,
        factory: Optional[ModelFactory],
        config: StrictInferenceOptimizerConfig,
        shared_config: SharedConfig,
        pipeline_cache_config: Optional[Any],
    ) -> None:
        self._factory = factory
        self._config = config
        self._shared_config = shared_config
        self._cache_config = pipeline_cache_config
        self._enabled = bool(getattr(pipeline_cache_config, "enabled", False))
        self._root = Path(getattr(pipeline_cache_config, "root", Path("."))).expanduser()
        self._repo_git_sha = _repo_git_sha()
        self._boundary_info = self._build_boundary_info()
        if self._enabled:
            self._check_world_size_consistency()

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._boundary_info)

    def _check_world_size_consistency(self) -> None:
        """Sanity-check the distributed state matches the SharedConfig."""
        if dist.is_available() and dist.is_initialized():
            actual = dist.get_world_size()
            if actual != self._shared_config.world_size:
                raise ValueError(
                    f"Pipeline cache: distributed world_size={actual} disagrees with "
                    f"SharedConfig.world_size={self._shared_config.world_size}."
                )

    def _build_boundary_info(self) -> Dict[str, Dict[str, Any]]:
        if not self._enabled:
            return {}

        boundary_name = getattr(self._cache_config, "boundary", None)
        if not boundary_name:
            ad_logger.warning("Pipeline cache enabled but no boundary name configured; disabling.")
            return {}

        ordered_items = list(self._config.items())
        transform_names = {name for name, _ in ordered_items}
        if boundary_name not in transform_names:
            ad_logger.warning(
                f"Pipeline cache boundary {boundary_name!r} is not in the optimizer config "
                f"(known transforms: {sorted(transform_names)}); disabling the cache."
            )
            return {}

        info: Dict[str, Dict[str, Any]] = {}
        for idx, (name, transform_config) in enumerate(ordered_items):
            if name != boundary_name:
                continue
            if transform_config.stage > Stages.SHARDING:
                raise ValueError(
                    "Pre-weight-loading pipeline snapshots only support boundaries "
                    f"through sharding. Got '{name}' at stage '{transform_config.stage.value}'."
                )

            transform_prefix = [
                {
                    "name": prefix_name,
                    "config": prefix_config,
                }
                for prefix_name, prefix_config in ordered_items[: idx + 1]
            ]
            prefix_transform_names = [prefix_name for prefix_name, _ in ordered_items[: idx + 1]]
            transform_prefix_hash = _hash_payload({"transforms": transform_prefix})
            producer_hash = _build_producer_hash(prefix_transform_names)
            cache_identity = {
                "cache_contract_version": CACHE_CONTRACT_VERSION,
                "graph_schema_version": GRAPH_SCHEMA_VERSION,
                "hook_schema_version": HOOK_SCHEMA_VERSION,
                "boundary_name": name,
                "model_identifier": None
                if self._factory is None
                else self._factory.get_pipeline_cache_model_identifier(),
                "checkpoint_fingerprint": None
                if self._factory is None
                else self._factory.get_pipeline_cache_checkpoint_fingerprint(),
                "factory_type": None if self._factory is None else type(self._factory).__name__,
                "model_kwargs": getattr(self._factory, "model_kwargs", {}),
                "transform_prefix_hash": transform_prefix_hash,
                "producer_hash": producer_hash,
                "mapping": self._mapping_payload(),
                "trtllm_version": TRTLLM_VERSION,
                "repo_git_sha": self._repo_git_sha,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "world_size": self._shared_config.world_size,
            }
            info[name] = {
                "producer_hash": producer_hash,
                "transform_index": idx,
                "transform_prefix_hash": transform_prefix_hash,
                "cache_key": _hash_payload(cache_identity),
            }
            break  # single-boundary invariant

        return info

    def _mapping_payload(self) -> Dict[str, Any]:
        mapping = self._shared_config.mapping
        if mapping is not None and hasattr(mapping, "to_dict"):
            payload = mapping.to_dict()
            payload.pop("rank", None)
            return payload
        return {"world_size": self._shared_config.world_size}

    def maybe_restore(self, cm: CachedSequenceInterface) -> Tuple[Optional[nn.Module], int]:
        """Restore the highest valid boundary if one exists.

        Restore is fail-closed and collective: any local failure (missing
        sidecars, checksum mismatch, hook rebuild error, FakeTensorProp error,
        graph.lint failure) causes this rank to report miss to the collective,
        and a single rank miss forces all ranks to rebuild.
        """
        if not self.enabled:
            return None, 0

        boundaries = sorted(
            self._boundary_info.items(),
            key=lambda item: item[1]["transform_index"],
            reverse=True,
        )
        for boundary_name, boundary_info in boundaries:
            local_restore_ready, manifest = self._validate_manifest(boundary_name, boundary_info)

            if not self._collective_bool_and(local_restore_ready):
                continue

            local_restore_success = False
            module: Optional[nn.Module] = None
            if local_restore_ready and manifest is not None:
                try:
                    module, ir = self._load_module(boundary_name, manifest)
                    if isinstance(module, GraphModule) and ir.hook_specs:
                        reattach_hooks(module, ir.hook_specs)
                        ad_logger.debug(f"Reattached {len(ir.hook_specs)} hook(s) from AD IR")

                    if ir.source_model_hooks_required:
                        self._replay_source_model_hooks(module, ir)

                    self._restore_sidecars(boundary_name, module, cm, ir)

                    # Structural validation: lint the graph before we hand it back.
                    if isinstance(module, GraphModule):
                        module.graph.lint()
                    local_restore_success = True
                except Exception as exc:
                    ad_logger.warning(
                        f"Failed to restore AutoDeploy pipeline snapshot for "
                        f"'{boundary_name}': {exc}"
                    )
                    module = None

            if not self._collective_bool_and(local_restore_success):
                continue

            assert module is not None
            next_index = int(boundary_info["transform_index"]) + 1
            ad_logger.info(f"Restored AutoDeploy pipeline snapshot for '{boundary_name}'")
            return module, next_index

        return None, 0

    def _validate_manifest(
        self, boundary_name: str, boundary_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Rank-local manifest validation. Returns ``(ready, manifest_or_none)``."""
        if not self._has_complete_boundary_snapshot(boundary_name):
            return False, None

        manifest_path = self._rank_dir(boundary_name).joinpath(MANIFEST_FILE_NAME)
        try:
            manifest = _read_json(manifest_path)
        except (json.JSONDecodeError, OSError) as exc:
            ad_logger.warning(f"Ignoring invalid pipeline cache manifest {manifest_path}: {exc}")
            return False, None

        if not self._manifest_matches(boundary_name, boundary_info, manifest):
            return False, manifest

        if manifest.get("has_unserializable_hooks", False):
            ad_logger.info(
                f"Skipping boundary '{boundary_name}': artifact has unserializable hooks"
            )
            return False, manifest

        if manifest.get("source_model_hooks_required", False) and self._factory is None:
            ad_logger.info(
                f"Skipping boundary '{boundary_name}': source-model hooks require a "
                "factory for restore."
            )
            return False, manifest

        return True, manifest

    def maybe_save(
        self,
        transform_name: str,
        transform_idx: int,
        transform_config: TransformConfig,
        mod: nn.Module,
        cm: CachedSequenceInterface,
    ) -> None:
        """Save a snapshot after a configured boundary transform completes."""
        if not self.enabled or transform_name not in self._boundary_info:
            return

        if transform_config.stage > Stages.SHARDING:
            return

        if not isinstance(mod, GraphModule):
            # TODO(nvchenghaoz): extend this to more generalized solution.
            ad_logger.warning(
                f"Skipping pipeline snapshot for '{transform_name}' because the module is not a "
                f"GraphModule ({type(mod).__name__})."
            )
            return

        self._barrier()

        rank_dir = self._rank_dir(transform_name)
        tmp_rank_dir = self._tmp_rank_dir(transform_name)
        boundary_dir = self._boundary_dir(transform_name)

        shutil.rmtree(tmp_rank_dir, ignore_errors=True)
        tmp_rank_dir.mkdir(parents=True, exist_ok=True)

        local_save_success = False
        try:
            hook_specs, has_unknown = collect_hook_specs(mod)
            if has_unknown:
                raise ValueError("graph contains unserializable load hooks")
            source_model_hooks_required = any(
                spec.get("type") == "source_model" for spec in hook_specs
            )
            # Restore rebuilds only a plain module tree plus FX nodes/weights, so any
            # residual call_module would require submodule forward() behavior we do not serialize.
            call_module_targets = self._find_call_module_targets(mod)
            if call_module_targets:
                raise ValueError(
                    f"graph contains unsupported call_module targets: {sorted(call_module_targets)}"
                )

            ir, real_buffers = extract_ir(
                mod,
                hook_specs=hook_specs,
                autodeploy_meta={
                    "cm_structural_state": {
                        "active_args": list(cm.info._active_args),
                        "active_host_prep_args": sorted(cm.info._active_host_prep_args),
                        "use_flattened_layout": cm.info._use_flattened_layout,
                    },
                },
                source_model_hooks_required=source_model_hooks_required,
            )
            checksums = save_ir(ir, real_buffers, tmp_rank_dir)

            manifest = self._build_manifest(
                transform_name,
                transform_idx,
                transform_config,
                hook_spec_count=len(hook_specs),
                has_unserializable_hooks=False,
                source_model_hooks_required=source_model_hooks_required,
                sidecar_checksums=checksums,
            )
            _write_json_atomic(tmp_rank_dir / MANIFEST_FILE_NAME, manifest)
            _fsync_dir(tmp_rank_dir)
            local_save_success = True
        except Exception as exc:
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)
            ad_logger.warning(
                f"Skipping AutoDeploy pipeline snapshot for '{transform_name}' because saving the "
                f"snapshot failed: {exc}"
            )
        all_ranks_saved = self._collective_bool_and(local_save_success)
        if not all_ranks_saved:
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)
            shutil.rmtree(rank_dir, ignore_errors=True)
            self._barrier()
            return

        boundary_dir.mkdir(parents=True, exist_ok=True)
        _fsync_dir(boundary_dir)
        _atomic_publish_rank_dir(tmp_rank_dir, rank_dir)
        self._barrier()
        ad_logger.info(f"Saved AutoDeploy pipeline snapshot for '{transform_name}' to {rank_dir}")

    def _build_manifest(
        self,
        transform_name: str,
        transform_idx: int,
        transform_config: TransformConfig,
        hook_spec_count: int = 0,
        has_unserializable_hooks: bool = False,
        source_model_hooks_required: bool = False,
        sidecar_checksums: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        boundary_info = self._boundary_info[transform_name]
        return {
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "hook_schema_version": HOOK_SCHEMA_VERSION,
            "boundary_name": transform_name,
            "transform_index": transform_idx,
            "boundary_stage": transform_config.stage.value,
            "cache_key": boundary_info["cache_key"],
            "transform_prefix_hash": boundary_info["transform_prefix_hash"],
            "producer_hash": boundary_info["producer_hash"],
            "model_identifier": None
            if self._factory is None
            else self._factory.get_pipeline_cache_model_identifier(),
            "checkpoint_fingerprint": None
            if self._factory is None
            else self._factory.get_pipeline_cache_checkpoint_fingerprint(),
            "trtllm_version": TRTLLM_VERSION,
            "repo_git_sha": self._repo_git_sha,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "mapping": self._mapping_payload(),
            "weights_materialized": False,
            "source_model_hooks_required": source_model_hooks_required,
            "has_unserializable_hooks": has_unserializable_hooks,
            "hook_spec_count": hook_spec_count,
            "sidecar_checksums": dict(sidecar_checksums or {}),
            "rank": self._shared_config.local_rank,
            "world_size": self._shared_config.world_size,
        }

    def _manifest_matches(
        self,
        boundary_name: str,
        boundary_info: Dict[str, Any],
        manifest: Mapping[str, Any],
    ) -> bool:
        expected = {
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "hook_schema_version": HOOK_SCHEMA_VERSION,
            "boundary_name": boundary_name,
            "cache_key": boundary_info["cache_key"],
            "transform_prefix_hash": boundary_info["transform_prefix_hash"],
            "producer_hash": boundary_info["producer_hash"],
            "world_size": self._shared_config.world_size,
            "rank": self._shared_config.local_rank,
            "trtllm_version": TRTLLM_VERSION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "repo_git_sha": self._repo_git_sha,
        }
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                ad_logger.info(
                    f"Pipeline snapshot manifest mismatch for '{boundary_name}': "
                    f"{key}={manifest.get(key)!r} != {expected_value!r}"
                )
                return False
        return True

    def _load_module(
        self, boundary_name: str, manifest: Mapping[str, Any]
    ) -> Tuple[nn.Module, IRGraph]:
        """Load a cached GraphModule via AD IR. Raises on any failure.

        The caller must treat any exception as a cache miss.
        """
        rank_dir = self._rank_dir(boundary_name)
        checksums = dict(manifest.get("sidecar_checksums", {}))
        result = load_ir(rank_dir, expected_checksums=checksums)
        if result is None:
            raise ValueError(f"Pipeline cache IR file missing in {rank_dir}")
        ir, real_buffers = result
        gm = build_graph_module(ir, real_buffers)
        hydrate_shapes(gm, ir)
        return gm, ir

    def _replay_source_model_hooks(self, mod: nn.Module, ir: IRGraph) -> None:
        """Replay source-model hooks from a fresh factory-built model and validate them."""
        if self._factory is None:
            raise ValueError("Source-model hooks require a model factory for restore.")

        expected_specs = [spec for spec in ir.hook_specs if spec.get("type") == "source_model"]
        expected_keys = sorted(_source_model_hook_spec_key(spec) for spec in expected_specs)

        original_model = self._factory.build_model("meta")
        try:
            source_hook_records = _collect_live_source_model_hooks(original_model)
            actual_keys = sorted(
                _source_model_hook_spec_key(record["spec"]) for record in source_hook_records
            )
            if actual_keys != expected_keys:
                raise ValueError(
                    "Source-model hook fingerprint mismatch between cached artifact and "
                    "freshly built factory model."
                )

            for record in source_hook_records:
                scope = record["scope"]
                target_mod = mod if scope == "root" else mod.get_submodule(scope)
                if record["position"] == "pre":
                    target_mod._register_load_state_dict_pre_hook(
                        record["hook_fn"],
                        with_module=record["with_module"],
                    )
                else:
                    target_mod.register_load_state_dict_post_hook(record["hook_fn"])
        finally:
            del original_model

    def _restore_sidecars(
        self,
        boundary_name: str,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        ir: IRGraph,
    ) -> None:
        cm_structural = (
            ir.autodeploy_meta.get("cm_structural_state") if ir.autodeploy_meta else None
        )
        if cm_structural:
            active_args = tuple(cm_structural.get("active_args", []))
            for arg_name in active_args:
                if arg_name not in cm.info.available_args:
                    raise ValueError(f"Unknown SequenceInfo active arg '{arg_name}' in snapshot.")
            cm.info._active_args = active_args
            cm.info._active_host_prep_args = set(cm_structural.get("active_host_prep_args", []))
            cm.info._use_flattened_layout = bool(cm_structural.get("use_flattened_layout", False))

    def _rank_dir(self, boundary_name: str) -> Path:
        return self._boundary_dir(boundary_name) / f"rank_{self._shared_config.local_rank}"

    def _boundary_dir(self, boundary_name: str) -> Path:
        cache_key = self._boundary_info[boundary_name]["cache_key"]
        return self._root / cache_key / boundary_name

    def _tmp_rank_dir(self, boundary_name: str) -> Path:
        cache_key = self._boundary_info[boundary_name]["cache_key"]
        return self._root / (
            f".{cache_key}.{boundary_name}.rank_{self._shared_config.local_rank}.tmp.{os.getpid()}"
        )

    def _has_complete_boundary_snapshot(self, boundary_name: str) -> bool:
        """Verify every peer rank directory has manifest + IR (+ buffers if declared).

        A manifest without its IR (or with a missing declared-buffer file) is
        treated as incomplete — the file sets must match what the manifest
        describes, or we fall back to rebuild.
        """
        boundary_dir = self._boundary_dir(boundary_name)
        for rank in range(self._shared_config.world_size):
            rank_dir = boundary_dir / f"rank_{rank}"
            manifest_path = rank_dir / MANIFEST_FILE_NAME
            if not manifest_path.exists():
                return False
            ir_path = rank_dir / IR_FILE_NAME
            if not ir_path.exists():
                return False
            try:
                manifest = _read_json(manifest_path)
            except (json.JSONDecodeError, OSError):
                return False
            checksums = manifest.get("sidecar_checksums", {}) or {}
            if BUFFERS_FILE_NAME in checksums:
                if not (rank_dir / BUFFERS_FILE_NAME).exists():
                    return False
        return True

    @staticmethod
    def _find_call_module_targets(gm: GraphModule) -> List[str]:
        return sorted({str(node.target) for node in gm.graph.nodes if node.op == "call_module"})

    @staticmethod
    def _collective_bool_and(local_value: bool) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return local_value

        backend = dist.get_backend()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        agreed = torch.tensor(1 if local_value else 0, dtype=torch.int32, device=device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        return bool(agreed.item())

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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

"""Trusted pre-weight-loading pipeline cache for AutoDeploy.

This cache is an AutoDeploy transform. Place ``pipeline_cache`` at the boundary
where the graph should be snapshotted, normally after sharding and before
``load_weights``. The transform saves the incoming module on a miss and the
optimizer asks the same transform for a restore before running the prefix so a
hit skips the transforms before the cache point.

The main artifact is a ``torch.save`` structural FX payload for the
``GraphModule`` or GraphModule-bearing wrapper. Load hooks are never part of the
artifact contract: they are scrubbed before save, rebuilt from ``hooks.json``,
or replayed from a fresh factory-built source model.
"""

import hashlib
import importlib
import inspect
import json
import os
import pickle
import shutil
import stat
import sys
import types
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import Field, model_validator
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.graph import Graph, _custom_builtins, _FindNodesLookupTable, _PyTreeCodeGen
from torch.fx.graph_module import _CodeOnlyModule, reduce_graph_module
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import Node

from tensorrt_llm.version import __version__ as TRTLLM_VERSION

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils._graph import named_graphmodules
from ..utils.logger import ad_logger
from ..utils.node_utils import invalidate_weight_node_cache
from .interface import (
    BaseTransform,
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

MANIFEST_FILE_NAME = "manifest.json"
MODULE_FILE_NAME = "module.pt"
HOOKS_FILE_NAME = "hooks.json"
SOURCE_HOOKS_FILE_NAME = "source_hooks.json"
SIDECARS_FILE_NAME = "sidecars.json"

_AD_MANAGED_MODULE_PREFIX = "tensorrt_llm._torch.auto_deploy."
_AD_PIPELINE_HOOK_PREFIXES: Tuple[str, ...] = (
    "tensorrt_llm._torch.auto_deploy.transform.",
    "tensorrt_llm._torch.auto_deploy.export.",
)
_MODULE_CLOSURE_CACHE: Dict[str, Tuple[Path, ...]] = {}
CacheKeyExtraValue = Union[str, int, float, bool, None]


def _default_pipeline_cache_root() -> str:
    return str(Path.home() / ".cache" / "tensorrt_llm" / "auto_deploy" / "pipeline_cache")


class PipelineCacheConfig(TransformConfig):
    """Configuration for the trusted torch-save pipeline cache transform."""

    model_config = {
        "extra": "forbid",
    }

    enabled: bool = Field(
        default=False,
        description="Whether to enable the trusted torch-save pipeline cache transform.",
    )
    run_per_gm: ClassVar[bool] = False
    run_graph_cleanup: ClassVar[bool] = False
    requires_clean_graph: ClassVar[bool] = False
    run_shape_prop: ClassVar[bool] = False
    requires_shape_prop: ClassVar[bool] = False
    skip_on_error: ClassVar[bool] = False
    debug_visualize_dir: ClassVar[Optional[str]] = None
    expect_mem_change: ClassVar[bool] = False
    root: Optional[str] = Field(
        default=None,
        description=(
            "Trusted cache root. Defaults to ~/.cache/tensorrt_llm/auto_deploy/"
            "pipeline_cache when the transform is enabled."
        ),
    )
    trust_cache_root: bool = Field(
        default=True,
        description="Whether to trust the cache root for restore with torch.load(weights_only=False).",
    )
    strict_root_permissions: bool = Field(
        default=False,
        description="Reject cache roots writable by group or other users when true.",
    )
    cache_key_extra: Dict[str, CacheKeyExtraValue] = Field(
        default_factory=dict,
        description="Additional JSON-serializable identity payload included in the cache key.",
    )

    @model_validator(mode="after")
    def validate_enabled_cache(self) -> "PipelineCacheConfig":
        if not self.enabled:
            return self
        if self.root in (None, ""):
            self.root = _default_pipeline_cache_root()
        if not self.root:
            raise ValueError("pipeline_cache.root is required when pipeline_cache is enabled.")
        if not self.trust_cache_root:
            raise ValueError(
                "pipeline_cache uses torch.load(weights_only=False); set "
                "trust_cache_root=true only for trusted cache roots."
            )
        if self.stage > Stages.SHARDING:
            raise ValueError(
                "pipeline_cache must be placed at or before the sharding stage so restore can "
                "resume before weight loading."
            )
        return self


def _canonicalize_for_hash(value: Any) -> Any:
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
    if isinstance(value, Enum):
        return _canonicalize_for_hash(value.value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if hasattr(value, "to_dict"):
        return _canonicalize_for_hash(value.to_dict())
    return value


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonicalize_for_hash(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any, sort_keys: bool = True) -> None:
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
    old_dir: Optional[Path] = None
    if rank_dir.exists():
        old_dir = rank_dir.with_name(f"{rank_dir.name}.old.{os.getpid()}")
        rank_dir.rename(old_dir)
    try:
        tmp_rank_dir.rename(rank_dir)
        _fsync_dir(rank_dir.parent)
    except OSError:
        if old_dir is not None and old_dir.exists() and not rank_dir.exists():
            old_dir.rename(rank_dir)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_repo_root()))
    except ValueError:
        return str(resolved)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return {
        "path": _repo_relative_path(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _is_auto_deploy_module(module: Any) -> bool:
    module_name = getattr(module, "__name__", "")
    if not module_name.startswith(_AD_MANAGED_MODULE_PREFIX):
        return False
    source_file = getattr(module, "__file__", None)
    return bool(source_file and Path(source_file).exists())


def _transform_module_closure(root_module_name: str) -> Tuple[Path, ...]:
    cached = _MODULE_CLOSURE_CACHE.get(root_module_name)
    if cached is not None:
        return cached

    root_module = sys.modules.get(root_module_name)
    if root_module is None:
        root_module = importlib.import_module(root_module_name)

    visited: Set[str] = set()
    files: Set[Path] = set()
    stack: List[Any] = [root_module]
    while stack:
        mod = stack.pop()
        mod_name = getattr(mod, "__name__", None)
        if not mod_name or mod_name in visited or not _is_auto_deploy_module(mod):
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
            if inspect.ismodule(obj):
                if _is_auto_deploy_module(obj) and obj.__name__ not in visited:
                    stack.append(obj)
                continue
            obj_module_name = getattr(obj, "__module__", None)
            if not obj_module_name or not obj_module_name.startswith(_AD_MANAGED_MODULE_PREFIX):
                continue
            obj_module = sys.modules.get(obj_module_name)
            if obj_module is not None and obj_module.__name__ not in visited:
                stack.append(obj_module)

    result = tuple(sorted(files))
    _MODULE_CLOSURE_CACHE[root_module_name] = result
    return result


def _build_producer_hash(prefix_transform_names: Sequence[str]) -> str:
    entries: List[Dict[str, Any]] = []
    for name in prefix_transform_names:
        transform_cls = TransformRegistry.get(name)
        transform_mod_name = getattr(transform_cls, "__module__", "")
        entry: Dict[str, Any] = {
            "name": name,
            "transform_source": _describe_source_object(transform_cls),
        }
        if transform_mod_name.startswith(_AD_MANAGED_MODULE_PREFIX):
            entry["closure_files"] = [
                _describe_file(path) for path in _transform_module_closure(transform_mod_name)
            ]
        else:
            entry["closure_files"] = []
        entries.append(entry)
    return _hash_payload({"prefix_transforms": entries})


def _build_factory_producer_hash(factory: Optional[ModelFactory]) -> Optional[str]:
    if factory is None:
        return None

    factory_cls = type(factory)
    factory_mod_name = getattr(factory_cls, "__module__", "")
    entry: Dict[str, Any] = {
        "factory_type": f"{factory_mod_name}.{factory_cls.__qualname__}",
        "factory_source": _describe_source_object(factory_cls),
    }
    if factory_mod_name.startswith(_AD_MANAGED_MODULE_PREFIX):
        entry["closure_files"] = [
            _describe_file(path) for path in _transform_module_closure(factory_mod_name)
        ]
    else:
        entry["closure_files"] = []
    return _hash_payload({"factory": entry})


def _hook_defining_module(hook: Any) -> Optional[str]:
    if isinstance(hook, partial):
        return _hook_defining_module(hook.func)
    bound_self = getattr(hook, "__self__", None)
    if bound_self is not None:
        return type(bound_self).__module__
    return getattr(hook, "__module__", None)


def _extract_closure_vars(func: Callable) -> Dict[str, Any]:
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


def _has_model_dump(obj: Any) -> bool:
    return hasattr(obj, "model_dump") and callable(obj.model_dump)


def _quant_instance_payload(self_obj: Any) -> Dict[str, Any]:
    if self_obj is None:
        return {}
    if _has_model_dump(self_obj):
        data = self_obj.model_dump(mode="json")
        json.dumps(data)
        return data
    payload: Dict[str, Any] = {}
    for attr_name in dir(self_obj):
        if attr_name.startswith("_") or callable(getattr(self_obj, attr_name, None)):
            continue
        try:
            value = getattr(self_obj, attr_name)
            json.dumps(value)
        except (TypeError, ValueError, AttributeError):
            continue
        payload[attr_name] = value
    return payload


def _describe_hook_identity(hook: Any) -> Dict[str, Any]:
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
        return {"kind": "function", "callable": _describe_source_object(hook)}
    if hasattr(hook, "__call__"):
        return {
            "kind": "callable_object",
            "callable": _describe_source_object(type(hook).__call__),
            "owner_type": type(hook).__qualname__,
            "owner_module": type(hook).__module__,
        }
    return {"kind": "unknown", "repr": repr(hook)}


def _identify_alias_hook(hook: Callable, scope: str) -> Optional[Dict[str, Any]]:
    cells = _extract_closure_vars(hook)
    aliased_groups = cells.get("aliased_groups")
    if aliased_groups is None or not isinstance(aliased_groups, list):
        return None
    return {
        "type": "alias",
        "scope": scope,
        "position": "pre",
        "aliased_groups": aliased_groups,
    }


def _identify_shard_tp_hook(kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    f_split = kw.get("f_split")
    spec: Dict[str, Any] = {
        "type": "shard_tp",
        "scope": scope,
        "position": "pre",
        "param_key": kw["param_key"],
        "param_shape": list(kw.get("param_shape", [])),
    }

    if isinstance(f_split, partial):
        split_kw = f_split.keywords or {}
        spec["dim"] = split_kw.get("dim", 0)
        spec["rank"] = split_kw.get("rank", 0)
        spec["world_size"] = split_kw.get("world_size", 1)
        spec["min_local_shape"] = split_kw.get("min_local_shape", 1)
        spec["fused_weight_dims"] = None
        return spec

    if callable(f_split):
        defaults = getattr(f_split, "__defaults__", None) or ()
        fused_dims = defaults[0] if len(defaults) >= 1 else None
        dim = defaults[1] if len(defaults) >= 2 else 0
        if isinstance(fused_dims, (list, tuple)):
            cells = _extract_closure_vars(f_split)
            spec["fused_weight_dims"] = list(fused_dims)
            spec["dim"] = dim
            spec["rank"] = cells.get("rank", 0)
            spec["world_size"] = cells.get("world_size", 1)
            spec["min_local_shape"] = cells.get("min_local_shape", 1)
            return spec

        cells = _extract_closure_vars(f_split)
        spec["type"] = "shard_slice"
        spec["start_idx"] = cells.get("start_idx", defaults[0] if defaults else 0)
        spec["end_idx"] = cells.get("end_idx", defaults[1] if len(defaults) > 1 else 0)
        return spec

    spec["dim"] = 0
    spec["rank"] = 0
    spec["world_size"] = 1
    spec["min_local_shape"] = 1
    spec["fused_weight_dims"] = None
    return spec


def _identify_shard_quant_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    self_obj = getattr(func, "__self__", None)
    fused_weight_dims = getattr(self_obj, "fused_weight_dims", None)
    return {
        "type": "shard_quant",
        "scope": scope,
        "position": "pre",
        "quant_class": type(self_obj).__name__ if self_obj is not None else "unknown",
        "weight_name": kw.get("weight_name", ""),
        "weight_original_shape": list(kw.get("weight_original_shape", [])),
        "dim": kw.get("dim", 0),
        "rank": kw.get("rank", 0),
        "world_size": kw.get("world_size", 1),
        "min_local_shape": kw.get("min_local_shape", 1),
        "fused_weight_dims": list(fused_weight_dims) if fused_weight_dims else None,
    }


def _identify_quant_load_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    self_obj = getattr(func, "__self__", None)
    return {
        "type": "quant_load",
        "scope": scope,
        "position": "pre",
        "quant_class": type(self_obj).__qualname__ if self_obj is not None else "unknown",
        "quant_module": type(self_obj).__module__ if self_obj is not None else "",
        "instance_payload": _quant_instance_payload(self_obj),
        "weight_name": kw.get("weight_name", ""),
    }


def _identify_quant_amax_hook(func: Callable, kw: Dict[str, Any], scope: str) -> Dict[str, Any]:
    self_obj = getattr(func, "__self__", None)
    return {
        "type": "quant_amax",
        "scope": scope,
        "position": "pre",
        "quant_class": type(self_obj).__qualname__ if self_obj is not None else "unknown",
        "quant_module": type(self_obj).__module__ if self_obj is not None else "",
        "instance_payload": _quant_instance_payload(self_obj),
        "scale_name": kw.get("scale_name", ""),
        "amax_name": kw.get("amax_name", ""),
    }


def _identify_quant_post_load_hook(
    func: Callable, kw: Dict[str, Any], scope: str
) -> Dict[str, Any]:
    self_obj = getattr(func, "__self__", None)
    return {
        "type": "quant_post_load",
        "scope": scope,
        "position": "post",
        "quant_class": type(self_obj).__qualname__ if self_obj is not None else "unknown",
        "quant_module": type(self_obj).__module__ if self_obj is not None else "",
        "instance_payload": _quant_instance_payload(self_obj),
        "weight_name": kw.get("weight_name", ""),
    }


def _identify_hook(hook: Any, scope: str = "root") -> Optional[Dict[str, Any]]:
    func = getattr(hook, "func", None)
    kw = getattr(hook, "keywords", {}) or {}
    if func is not None:
        func_name = getattr(func, "__name__", "")
        if func_name == "_load_hook" and "param_key" in kw:
            return _identify_shard_tp_hook(kw, scope)
        if func_name == "_load_hook_remove" and "param_key" in kw:
            return {
                "type": "remove",
                "scope": scope,
                "position": "pre",
                "param_key": kw["param_key"],
            }
        if func_name == "_load_hook_for_deduplication":
            return {
                "type": "dedup",
                "scope": scope,
                "position": "pre",
                "param_key_remaining": kw.get("param_key_remaining", ""),
                "param_key_removed": kw.get("param_key_removed", ""),
            }
        if func_name == "shard_load_hook" and "weight_name" in kw:
            return _identify_shard_quant_hook(func, kw, scope)
        if func_name == "load_hook" and "weight_name" in kw:
            return _identify_quant_load_hook(func, kw, scope)
        if func_name == "convert_amax_hook" and "scale_name" in kw:
            return _identify_quant_amax_hook(func, kw, scope)
        if func_name == "post_load_hook" and "weight_name" in kw:
            return _identify_quant_post_load_hook(func, kw, scope)

    if "aliasing_load_pre_hook" in getattr(hook, "__qualname__", ""):
        return _identify_alias_hook(hook, scope)

    if callable(hook):
        hook_module = _hook_defining_module(hook) or ""
        if any(hook_module.startswith(prefix) for prefix in _AD_PIPELINE_HOOK_PREFIXES):
            return None
        return {
            "type": "source_model",
            "scope": scope,
            "position": "pre",
            "hook_identity": _describe_hook_identity(hook),
        }
    return None


def collect_hook_specs(gm: GraphModule) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect declarative hook specs from load hooks on ``gm`` and children."""
    specs: List[Dict[str, Any]] = []
    has_unknown = False
    unknown_counts: Dict[str, int] = {}

    def collect_from_module(mod: nn.Module, scope: str) -> None:
        nonlocal has_unknown
        for hook in mod._load_state_dict_pre_hooks.values():
            with_module = bool(getattr(hook, "with_module", False))
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is None:
                qualname = getattr(hook_obj, "__qualname__", repr(hook_obj))
                unknown_counts[qualname] = unknown_counts.get(qualname, 0) + 1
                has_unknown = True
                continue
            spec["with_module"] = with_module
            specs.append(spec)

        for hook in mod._load_state_dict_post_hooks.values():
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is None:
                qualname = getattr(hook_obj, "__qualname__", repr(hook_obj))
                unknown_counts[qualname] = unknown_counts.get(qualname, 0) + 1
                has_unknown = True
                continue
            spec["position"] = "post"
            specs.append(spec)

    collect_from_module(gm, "root")
    for name, child in gm.named_modules():
        if name:
            collect_from_module(child, name)

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

    def collect_from_module(mod: nn.Module, scope: str) -> None:
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

    collect_from_module(model, "root")
    for name, child in model.named_modules():
        if name:
            collect_from_module(child, name)
    return records


def _wrap_pre_hook_with_module_adapter(hook_fn: Callable) -> Callable:
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


def _resolve_nested_class(mod: Any, class_path: str) -> Optional[type]:
    cls: Any = mod
    for part in class_path.split("."):
        cls = getattr(cls, part, None)
        if cls is None:
            return None
    return cls if isinstance(cls, type) else None


def _rebuild_shard_tp_hook(spec: Dict[str, Any]) -> Callable:
    from ..transform.library.sharding import _load_hook, _split_tensor_for_tp

    dim = spec["dim"]
    rank = spec["rank"]
    world_size = spec["world_size"]
    min_local_shape = spec.get("min_local_shape", 1)
    fused_weight_dims = spec.get("fused_weight_dims")
    if fused_weight_dims:

        def f_split(
            t: torch.Tensor, fused_dims: list = fused_weight_dims, d: int = dim
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
    from ..transform.interface import Stages
    from ..transform.library.sharding import (
        FineGrainedFP8WeightShardingInfo,
        FP4WeightShardingInfo,
        FP8WeightShardingInfo,
        ShardingTransformConfig,
        SplitDimension,
        WeightShardingInfo,
    )

    quant_classes: Dict[str, Type[Any]] = {
        "WeightShardingInfo": WeightShardingInfo,
        "FP8WeightShardingInfo": FP8WeightShardingInfo,
        "FineGrainedFP8WeightShardingInfo": FineGrainedFP8WeightShardingInfo,
        "FP4WeightShardingInfo": FP4WeightShardingInfo,
    }
    cls = quant_classes.get(spec.get("quant_class", ""))
    if cls is None:
        raise ValueError(
            f"Pipeline cache: unknown quant sharding class {spec.get('quant_class')!r}."
        )

    fused_weight_dims = spec.get("fused_weight_dims")
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


def _rebuild_quant_hook(spec: Dict[str, Any]) -> Callable:
    quant_module = spec.get("quant_module", "")
    quant_class = spec.get("quant_class", "")
    try:
        mod = importlib.import_module(quant_module)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Pipeline cache: cannot import {quant_module!r}: {exc}") from exc

    cls = _resolve_nested_class(mod, quant_class)
    if cls is None:
        raise ValueError(
            f"Pipeline cache: cannot reconstruct quant class {quant_module}.{quant_class}"
        )

    payload = spec.get("instance_payload", {}) or {}
    if hasattr(cls, "model_validate"):
        instance = cls.model_validate(payload)
    else:
        instance = cls.__new__(cls)
        for attr_name, value in payload.items():
            setattr(instance, attr_name, value)

    hook_type = spec["type"]
    if hook_type == "quant_load":
        return partial(instance.load_hook, weight_name=spec["weight_name"])
    if hook_type == "quant_amax":
        return partial(
            instance.convert_amax_hook, scale_name=spec["scale_name"], amax_name=spec["amax_name"]
        )
    if hook_type == "quant_post_load":
        return partial(instance.post_load_hook, weight_name=spec["weight_name"])
    raise ValueError(f"Pipeline cache: unknown quant hook type {hook_type!r}")


def _rebuild_hook(spec: Dict[str, Any]) -> Optional[Callable]:
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


def reattach_hooks(gm: GraphModule, specs: List[Dict[str, Any]]) -> None:
    """Rebuild and register AD-managed load hooks on ``gm``."""
    for spec in specs:
        if spec.get("type") == "source_model":
            continue
        hook_fn = _rebuild_hook(spec)
        if hook_fn is None:
            raise ValueError(f"Pipeline cache: could not rebuild hook {spec!r}.")

        scope = spec.get("scope", "root")
        target_mod = gm if scope == "root" else gm.get_submodule(scope)
        if spec.get("position", "pre") == "pre":
            with_module = bool(spec.get("with_module", False))
            if with_module and not bool(spec.get("requires_module", False)):
                hook_fn = _wrap_pre_hook_with_module_adapter(hook_fn)
            target_mod._register_load_state_dict_pre_hook(hook_fn, with_module=with_module)
        else:
            target_mod.register_load_state_dict_post_hook(hook_fn)


def _snapshot_and_clear_load_hooks(model: nn.Module) -> List[Tuple[nn.Module, Any, Any]]:
    records: List[Tuple[nn.Module, Any, Any]] = []
    for module in model.modules():
        records.append(
            (
                module,
                module._load_state_dict_pre_hooks.copy(),
                module._load_state_dict_post_hooks.copy(),
            )
        )
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_post_hooks.clear()
    return records


def _restore_load_hooks(records: List[Tuple[nn.Module, Any, Any]]) -> None:
    for module, pre_hooks, post_hooks in records:
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_pre_hooks.update(pre_hooks)
        module._load_state_dict_post_hooks.clear()
        module._load_state_dict_post_hooks.update(post_hooks)


_FORWARD_HOOK_ATTRS = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


def _snapshot_and_clear_forward_hooks(model: nn.Module) -> List[Tuple[nn.Module, Dict[str, Any]]]:
    records: List[Tuple[nn.Module, Dict[str, Any]]] = []
    for module in model.modules():
        hook_state = {
            attr: getattr(module, attr).copy()
            for attr in _FORWARD_HOOK_ATTRS
            if hasattr(module, attr)
        }
        if not any(hook_state.values()):
            continue
        records.append((module, hook_state))
        for attr in hook_state:
            getattr(module, attr).clear()
    return records


def _restore_forward_hooks(records: List[Tuple[nn.Module, Dict[str, Any]]]) -> None:
    for module, hook_state in records:
        for attr, hooks in hook_state.items():
            current_hooks = getattr(module, attr)
            current_hooks.clear()
            current_hooks.update(hooks)


_CACHED_GRAPHMODULE_IMPORT_BLOCK = (
    "\n".join(sorted({builtin.import_str for builtin in _custom_builtins.values()})) + "\n"
)
_GRAPHMODULE_PICKLE_EXCLUDED_ATTRS = {
    "_graph",
    "_tracer_cls",
    "_tracer_extras",
}
_GRAPHMODULE_STRUCTURAL_EXCLUDED_ATTRS = {
    "_graph",
    "_code",
    "_lineno_map",
    "_prologue_start",
    "_tracer_cls",
    "_tracer_extras",
}
_GRAPHMODULE_STRUCTURAL_FORCE_RESTORE_ATTRS = (
    "meta",
    "shape_env",
    "_replace_hooks",
    "_create_node_hooks",
    "_erase_node_hooks",
    "_deepcopy_hooks",
    "_non_persistent_buffers_set",
    "_graphmodule_cls_name",
)
_GRAPH_TARGET_REF_KEY = "__pipeline_cache_graph_target_ref__"
_PLACEHOLDER_META_VALUE_KEY = "__pipeline_cache_placeholder_meta_value__"
_NODE_META_DROP_KEYS = {
    "bias_nodes",
    "eager_input_vals",
    "from_node",
    "scale_nodes",
    "weight_nodes",
}
_DROP_SELF_REFERENCE = object()
_SHARDING_TRANSFORM_CONTAINER_ATTR = "_sharding_transform_container"


def _cacheable_sharding_transform_container(value: Any) -> Any:
    from .library.sharding import ShardingTransformContainer

    if not isinstance(value, ShardingTransformContainer):
        return value

    config = value.config.model_copy(deep=True)
    if hasattr(config, "mapping"):
        config.mapping = None
    dist_config = getattr(config, "dist_config", None)
    if hasattr(dist_config, "to_dict") and hasattr(type(dist_config), "from_dict"):
        config.dist_config = type(dist_config).from_dict(dist_config.to_dict())

    candidate = ShardingTransformContainer(config=config)
    if _is_pickleable(candidate):
        return candidate
    return _DROP_SELF_REFERENCE


def _cacheable_graphmodule_attr(key: str, value: Any) -> Any:
    if key == _SHARDING_TRANSFORM_CONTAINER_ATTR:
        return _cacheable_sharding_transform_container(value)
    return value


def _contains_direct_self_reference(value: Any, module: nn.Module) -> bool:
    return value is module or getattr(value, "__self__", None) is module


def _strip_direct_self_references(value: Any, module: nn.Module) -> Tuple[Any, bool]:
    if _contains_direct_self_reference(value, module):
        return _DROP_SELF_REFERENCE, True
    if isinstance(value, dict):
        stripped = value.copy()
        for key, item in list(stripped.items()):
            if _contains_direct_self_reference(key, module) or _contains_direct_self_reference(
                item, module
            ):
                del stripped[key]
        return stripped, len(stripped) != len(value)
    if isinstance(value, list):
        stripped = [item for item in value if not _contains_direct_self_reference(item, module)]
        return stripped, len(stripped) != len(value)
    if isinstance(value, tuple):
        stripped = tuple(
            item for item in value if not _contains_direct_self_reference(item, module)
        )
        return stripped, len(stripped) != len(value)
    if isinstance(value, set):
        stripped = type(value)(
            item for item in value if not _contains_direct_self_reference(item, module)
        )
        return stripped, len(stripped) != len(value)
    return value, False


def _snapshot_and_clear_direct_self_references(
    root: nn.Module,
) -> List[Tuple[nn.Module, str, Any]]:
    records: List[Tuple[nn.Module, str, Any]] = []
    for module in root.modules():
        for key, value in list(module.__dict__.items()):
            sanitized, changed = _strip_direct_self_references(value, root)
            if not changed:
                continue
            records.append((module, key, value))
            if sanitized is _DROP_SELF_REFERENCE:
                del module.__dict__[key]
            else:
                module.__dict__[key] = sanitized
    return records


def _restore_direct_self_references(records: List[Tuple[nn.Module, str, Any]]) -> None:
    for module, key, value in reversed(records):
        module.__dict__[key] = value


def _sanitized_graphmodule_body(gm: GraphModule, excluded_attrs: Set[str]) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    for key, value in gm.__dict__.items():
        if key in excluded_attrs:
            continue
        sanitized, _ = _strip_direct_self_references(_cacheable_graphmodule_attr(key, value), gm)
        if sanitized is not _DROP_SELF_REFERENCE:
            body[key] = sanitized
    return body


def _is_pickleable(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=2)
    except Exception:
        return False
    return True


def _resolve_qualified_attr(module_name: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _encode_graph_target_ref(target: Any) -> Dict[str, str]:
    if isinstance(target, torch._ops.OpOverload):
        return {
            "kind": "torch_op_overload",
            "namespace": target.namespace,
            "opname": target._opname,
            "overload": target._overloadname,
        }
    if isinstance(target, torch._ops.OpOverloadPacket):
        namespace, opname = target._qualified_op_name.split("::", 1)
        return {
            "kind": "torch_op_packet",
            "namespace": namespace,
            "opname": opname,
        }

    module_name = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if isinstance(module_name, str) and isinstance(qualname, str) and "<locals>" not in qualname:
        try:
            resolved = _resolve_qualified_attr(module_name, qualname)
        except (AttributeError, ImportError, ValueError):
            resolved = None
        if resolved is target:
            return {
                "kind": "importable",
                "module": module_name,
                "qualname": qualname,
            }

    raise ValueError(
        "Pipeline cache: graph target is not pickleable and cannot be restored by import: "
        f"{target!r} ({type(target).__module__}.{type(target).__qualname__})."
    )


def _decode_graph_target_ref(ref: Mapping[str, str]) -> Any:
    if ref.get("kind") == "torch_op_overload":
        packet = getattr(getattr(torch.ops, ref["namespace"]), ref["opname"])
        return getattr(packet, ref["overload"])
    if ref.get("kind") == "torch_op_packet":
        return getattr(getattr(torch.ops, ref["namespace"]), ref["opname"])
    if ref.get("kind") == "importable":
        return _resolve_qualified_attr(ref["module"], ref["qualname"])
    raise ValueError(f"Pipeline cache: unknown graph target reference {ref!r}.")


def _graphmodule_bound_method_specs(gm: GraphModule) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for name, value in gm.__dict__.items():
        if not inspect.ismethod(value) or getattr(value, "__self__", None) is not gm:
            continue
        try:
            function_ref = _encode_graph_target_ref(value.__func__)
        except ValueError:
            ad_logger.warning(
                "Pipeline cache: dropping non-importable GraphModule bound method "
                f"{name!r} during structural save."
            )
            continue
        specs.append({"name": name, "function_ref": function_ref})
    return specs


def _restore_graphmodule_bound_methods(
    module: GraphModule, specs: Sequence[Mapping[str, Any]]
) -> None:
    for spec in specs:
        name = spec.get("name")
        function_ref = spec.get("function_ref")
        if not isinstance(name, str) or not isinstance(function_ref, Mapping):
            raise ValueError(f"pipeline cache module payload has invalid bound method: {spec!r}")
        setattr(module, name, types.MethodType(_decode_graph_target_ref(function_ref), module))


def _encode_graph_target(target: Any) -> Dict[str, Any]:
    if _is_pickleable(target):
        return {"kind": "literal", "value": target}
    return {"kind": "ref", "value": _encode_graph_target_ref(target)}


def _decode_graph_target(spec: Mapping[str, Any]) -> Any:
    kind = spec.get("kind")
    if kind == "literal":
        return spec["value"]
    if kind == "ref":
        return _decode_graph_target_ref(spec["value"])
    raise ValueError(f"Pipeline cache: unknown graph target spec {spec!r}.")


def _make_graph_target_ref(spec: Mapping[str, str]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (_GRAPH_TARGET_REF_KEY, tuple(sorted(spec.items())))


def _is_graph_target_ref(target: Any) -> bool:
    return (
        isinstance(target, tuple)
        and len(target) == 2
        and target[0] == _GRAPH_TARGET_REF_KEY
        and isinstance(target[1], tuple)
    )


def _rebuild_find_nodes_lookup_table(graph: Graph) -> None:
    lookup_table = _FindNodesLookupTable()
    for node in graph.nodes:
        lookup_table.insert(node)
    graph._find_nodes_lookup_table = lookup_table


def _meta_value_to_spec(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (list, tuple)):
        items = [_meta_value_to_spec(item) for item in value]
        if all(item is not None for item in items):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": items,
            }
        return None
    if isinstance(value, dict):
        items = {key: _meta_value_to_spec(item) for key, item in value.items()}
        if all(item is not None for item in items.values()):
            return {"kind": "dict", "items": items}
        return None

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "kind": "tensor",
            "shape": _concrete_shape(shape),
            "dtype": dtype,
        }
    if _is_pickleable(value):
        return {
            "kind": "literal",
            "value": value,
        }
    return None


def _meta_value_from_spec(spec: Mapping[str, Any], fake_mode: FakeTensorMode) -> Any:
    kind = spec.get("kind")
    if kind == "none":
        return None
    if kind == "tuple":
        return tuple(_meta_value_from_spec(item, fake_mode) for item in spec["items"])
    if kind == "list":
        return [_meta_value_from_spec(item, fake_mode) for item in spec["items"]]
    if kind == "dict":
        return {key: _meta_value_from_spec(item, fake_mode) for key, item in spec["items"].items()}
    if kind == "tensor":
        tensor = torch.empty(tuple(spec["shape"]), dtype=spec["dtype"], device="meta")
        return fake_mode.from_tensor(tensor, static_shapes=True)
    if kind == "literal":
        return spec["value"]
    raise ValueError(f"Pipeline cache: unknown placeholder meta value spec {spec!r}.")


def _concrete_shape_dim(dim: Any) -> int:
    try:
        return int(dim)
    except (TypeError, ValueError):
        node = getattr(dim, "node", None)
        hint = getattr(node, "hint", None)
        if hint is not None:
            return int(hint)
    raise ValueError(f"Pipeline cache: cannot serialize symbolic shape dimension {dim!r}.")


def _concrete_shape(shape: Any) -> Tuple[int, ...]:
    return tuple(_concrete_shape_dim(dim) for dim in shape)


def _sanitize_node_meta_for_pickling(node: torch.fx.Node) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in node.meta.items():
        if key in _NODE_META_DROP_KEYS:
            continue
        if key == "val":
            if node.op in ("placeholder", "get_attr"):
                spec = _meta_value_to_spec(value)
                if spec is not None:
                    sanitized[_PLACEHOLDER_META_VALUE_KEY] = spec
            continue
        if _is_pickleable(value):
            sanitized[key] = value
    return sanitized


@contextmanager
def _sanitize_graph_node_meta_for_pickling(graph: Graph):
    records = []
    try:
        for node in graph.nodes:
            sanitized = _sanitize_node_meta_for_pickling(node)
            if sanitized == node.meta:
                continue
            records.append((node, node.meta))
            node.meta = sanitized
        yield
    finally:
        for node, meta in records:
            node.meta = meta


def _restore_placeholder_meta_values(graph: Graph) -> None:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    for node in graph.nodes:
        spec = node.meta.pop(_PLACEHOLDER_META_VALUE_KEY, None)
        if spec is not None:
            node.meta["val"] = _meta_value_from_spec(spec, fake_mode)


def _encode_graph_arg(value: Any) -> Dict[str, Any]:
    if isinstance(value, Node):
        return {"kind": "node", "name": value.name}
    if isinstance(value, torch.Size):
        return {"kind": "torch_size", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, immutable_list):
        return {"kind": "immutable_list", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, immutable_dict):
        return {
            "kind": "immutable_dict",
            "items": [(key, _encode_graph_arg(item)) for key, item in value.items()],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [(key, _encode_graph_arg(item)) for key, item in value.items()],
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _encode_graph_arg(value.start),
            "stop": _encode_graph_arg(value.stop),
            "step": _encode_graph_arg(value.step),
        }
    if _is_pickleable(value):
        return {"kind": "literal", "value": value}
    raise ValueError(
        f"Pipeline cache: graph argument is not pickleable: {value!r} ({_short_type_name(value)})."
    )


def _decode_graph_arg(spec: Mapping[str, Any], nodes_by_name: Mapping[str, Node]) -> Any:
    kind = spec.get("kind")
    if kind == "node":
        return nodes_by_name[spec["name"]]
    if kind == "torch_size":
        return torch.Size(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "tuple":
        return tuple(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "list":
        return [_decode_graph_arg(item, nodes_by_name) for item in spec["items"]]
    if kind == "immutable_list":
        return immutable_list(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "immutable_dict":
        return immutable_dict(
            [(key, _decode_graph_arg(item, nodes_by_name)) for key, item in spec["items"]]
        )
    if kind == "dict":
        return {key: _decode_graph_arg(item, nodes_by_name) for key, item in spec["items"]}
    if kind == "slice":
        return slice(
            _decode_graph_arg(spec["start"], nodes_by_name),
            _decode_graph_arg(spec["stop"], nodes_by_name),
            _decode_graph_arg(spec["step"], nodes_by_name),
        )
    if kind == "literal":
        return spec["value"]
    raise ValueError(f"Pipeline cache: unknown graph argument spec {spec!r}.")


def _graph_to_state(graph: Graph) -> Dict[str, Any]:
    nodes = []
    for node in graph.nodes:
        node_state = {
            "name": node.name,
            "op": node.op,
            "target": _encode_graph_target(node.target),
            "args": _encode_graph_arg(node.args),
            "kwargs": _encode_graph_arg(node.kwargs),
            "type": node.type if _is_pickleable(node.type) else None,
            "meta": _sanitize_node_meta_for_pickling(node),
        }
        nodes.append(node_state)
    return {
        "nodes": nodes,
        "codegen": _graph_codegen_to_state(graph),
        "co_fields": getattr(graph, "_co_fields", {}),
    }


def _graph_codegen_to_state(graph: Graph) -> Dict[str, Any]:
    codegen = getattr(graph, "_codegen", None)
    pytree_info = getattr(codegen, "pytree_info", None)
    if pytree_info is not None and _is_pickleable(pytree_info):
        return {
            "kind": "pytree",
            "pytree_info": pytree_info,
        }
    return {"kind": "default"}


def _graph_codegen_from_state(state: Mapping[str, Any]) -> Optional[Any]:
    if state.get("kind") == "pytree":
        return _PyTreeCodeGen(state["pytree_info"])
    return None


def _graph_from_state(state: Mapping[str, Any]) -> Graph:
    graph = Graph()
    nodes_by_name: Dict[str, Node] = {}
    for node_state in state["nodes"]:
        args = _decode_graph_arg(node_state["args"], nodes_by_name)
        kwargs = _decode_graph_arg(node_state["kwargs"], nodes_by_name)
        node = graph.create_node(
            node_state["op"],
            _decode_graph_target(node_state["target"]),
            args=args,
            kwargs=kwargs,
            name=node_state["name"],
            type_expr=node_state.get("type"),
        )
        node.meta = dict(node_state.get("meta", {}))
        nodes_by_name[node.name] = node

    codegen = _graph_codegen_from_state(state.get("codegen", {}))
    if codegen is not None:
        graph._codegen = codegen
    if state.get("co_fields") is not None:
        graph._co_fields = state["co_fields"]
    _restore_placeholder_meta_values(graph)
    _rebuild_find_nodes_lookup_table(graph)
    return graph


@contextmanager
def _sanitize_graph_namespace_for_pickling(graph: Graph):
    namespace = getattr(graph, "_graph_namespace", None)
    obj_to_name = getattr(namespace, "_obj_to_name", None)
    if not isinstance(obj_to_name, dict):
        yield
        return

    original = obj_to_name
    namespace._obj_to_name = {key: value for key, value in original.items() if _is_pickleable(key)}
    try:
        yield
    finally:
        namespace._obj_to_name = original


def _mark_cached_shape_metadata_invalid(module: nn.Module) -> None:
    for _, graph_module in named_graphmodules(module):
        invalidate_weight_node_cache(graph_module)
        autodeploy_meta = graph_module.meta.get("_autodeploy", {})
        history = autodeploy_meta.get("transform_history", {})
        for key, info in list(history.items()):
            if hasattr(info, "model_copy"):
                history[key] = info.model_copy(update={"has_valid_shapes": False})
            elif isinstance(info, dict):
                history[key] = {**info, "has_valid_shapes": False}


@contextmanager
def _encode_unpickleable_graph_targets(graph: Graph):
    records = []
    try:
        for node in graph.nodes:
            if _is_pickleable(node.target):
                continue
            encoded_target = _make_graph_target_ref(_encode_graph_target_ref(node.target))
            records.append((node, node.target))
            object.__setattr__(node, "target", encoded_target)
        if records:
            _rebuild_find_nodes_lookup_table(graph)
        yield
    finally:
        for node, target in records:
            object.__setattr__(node, "target", target)
        if records:
            _rebuild_find_nodes_lookup_table(graph)


def _decode_unpickleable_graph_targets(graph: Graph) -> None:
    for node in graph.nodes:
        target = node.target
        if _is_graph_target_ref(target):
            object.__setattr__(node, "target", _decode_graph_target_ref(dict(target[1])))
    _rebuild_find_nodes_lookup_table(graph)


def _short_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _find_unpickleable_path(
    value: Any,
    path: str = "payload",
    seen: Optional[Set[int]] = None,
    depth: int = 0,
) -> Optional[str]:
    if _is_pickleable(value):
        return None
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return None
    seen.add(value_id)
    if depth > 12:
        return f"{path}: {value!r} ({_short_type_name(value)})"

    if isinstance(value, Mapping):
        for key, item in value.items():
            result = _find_unpickleable_path(key, f"{path}.key({key!r})", seen, depth + 1)
            if result is not None:
                return result
            result = _find_unpickleable_path(item, f"{path}[{key!r}]", seen, depth + 1)
            if result is not None:
                return result
        return f"{path}: {value!r} ({_short_type_name(value)})"

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            result = _find_unpickleable_path(item, f"{path}[{idx}]", seen, depth + 1)
            if result is not None:
                return result
        return f"{path}: {value!r} ({_short_type_name(value)})"

    if isinstance(value, (set, frozenset)):
        for idx, item in enumerate(value):
            result = _find_unpickleable_path(item, f"{path}[set:{idx}]", seen, depth + 1)
            if result is not None:
                return result
        return f"{path}: {value!r} ({_short_type_name(value)})"

    if inspect.ismethod(value):
        result = _find_unpickleable_path(value.__self__, f"{path}.__self__", seen, depth + 1)
        if result is not None:
            return result
        return _find_unpickleable_path(value.__func__, f"{path}.__func__", seen, depth + 1)

    getstate = getattr(value, "__getstate__", None)
    if callable(getstate):
        try:
            state = getstate()
        except Exception:
            state = None
        if state is not None and state is not value:
            result = _find_unpickleable_path(state, f"{path}.__getstate__()", seen, depth + 1)
            if result is not None:
                return result

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        result = _find_unpickleable_path(value_dict, f"{path}.__dict__", seen, depth + 1)
        if result is not None:
            return result

    return f"{path}: {value!r} ({_short_type_name(value)})"


@contextmanager
def _use_cached_graphmodule_code_for_pickling():
    """Avoid regenerating large FX sources when pickling an already compiled GraphModule."""

    original_reduce = GraphModule.__reduce__
    original_recursion_limit = sys.getrecursionlimit()

    def _reduce_graphmodule_with_cached_code(self):
        if "_graph" not in self.__dict__ or "_code" not in self.__dict__:
            return original_reduce(self)

        body = _sanitized_graphmodule_body(self, _GRAPHMODULE_PICKLE_EXCLUDED_ATTRS)
        return reduce_graph_module, (body, _CACHED_GRAPHMODULE_IMPORT_BLOCK)

    GraphModule.__reduce__ = _reduce_graphmodule_with_cached_code
    if original_recursion_limit < 10000:
        sys.setrecursionlimit(10000)
    try:
        yield
    finally:
        if sys.getrecursionlimit() != original_recursion_limit:
            sys.setrecursionlimit(original_recursion_limit)
        GraphModule.__reduce__ = original_reduce


@contextmanager
def _make_graph_picklable_without_graphmodule(graph: Graph):
    """Avoid serializing Graph.owning_module, which points back to the GraphModule."""

    owning_module = graph.owning_module
    tracer_cls = getattr(graph, "_tracer_cls", None)
    tracer_extras = getattr(graph, "_tracer_extras", None)
    graph.owning_module = None
    graph._tracer_cls = None
    graph._tracer_extras = None
    try:
        yield
    finally:
        graph.owning_module = owning_module
        graph._tracer_cls = tracer_cls
        graph._tracer_extras = tracer_extras


class _GraphModulePlaceholder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


def _graphmodule_to_structural_payload(gm: GraphModule) -> Dict[str, Any]:
    body = _sanitized_graphmodule_body(gm, _GRAPHMODULE_STRUCTURAL_EXCLUDED_ATTRS)
    return {
        "class_name": body.get("_graphmodule_cls_name", type(gm).__name__),
        "body": body,
        "bound_methods": _graphmodule_bound_method_specs(gm),
        "graph_state": _graph_to_state(gm.graph),
    }


def _is_graphmodule_structural_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and "body" in payload and "graph_state" in payload


def _is_module_tree_structural_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and "module" in payload and "graphmodules" in payload


def _graphmodule_from_structural_payload(payload: Mapping[str, Any]) -> GraphModule:
    graph_state = payload.get("graph_state")
    body = payload.get("body")
    if not isinstance(graph_state, Mapping) or not isinstance(body, dict):
        raise ValueError("pipeline cache module payload is missing GraphModule state.")

    graph = _graph_from_state(graph_state)
    root = _CodeOnlyModule(body)
    module = GraphModule(root, graph, class_name=payload.get("class_name", "GraphModule"))
    for key, value in body.items():
        if key in _GRAPHMODULE_STRUCTURAL_EXCLUDED_ATTRS:
            continue
        if not hasattr(module, key):
            setattr(module, key, value)
    for key in _GRAPHMODULE_STRUCTURAL_FORCE_RESTORE_ATTRS:
        if key in body:
            setattr(module, key, body[key])
    _restore_graphmodule_bound_methods(module, payload.get("bound_methods", []))
    _mark_cached_shape_metadata_invalid(module)
    return module


def _save_graphmodule_structural(gm: GraphModule, module_file: Any) -> None:
    payload = _graphmodule_to_structural_payload(gm)
    try:
        torch.save(payload, module_file)
    except Exception as exc:
        detail = _find_unpickleable_path(payload)
        if detail:
            raise ValueError(f"{exc}; first unpickleable payload path: {detail}") from exc
        raise


def _load_graphmodule_structural(module_file: Any) -> GraphModule:
    payload = torch.load(module_file, map_location="cpu", weights_only=False)
    if isinstance(payload, GraphModule):
        return payload
    if not _is_graphmodule_structural_payload(payload):
        raise ValueError(
            f"pipeline cache module has unsupported payload shape: {type(payload).__name__}"
        )
    return _graphmodule_from_structural_payload(payload)


def _replace_submodule(module: nn.Module, target: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = target.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    parent._modules[child_name] = replacement


def _named_graphmodule_roots(module: nn.Module) -> List[Tuple[str, GraphModule]]:
    roots: List[Tuple[str, GraphModule]] = []
    for name, graph_module in named_graphmodules(module):
        if any(parent == "" or name.startswith(f"{parent}.") for parent, _ in roots):
            continue
        roots.append((name, graph_module))
    return roots


def _save_module_structural(module: nn.Module, module_file: Any) -> None:
    if isinstance(module, GraphModule):
        _save_graphmodule_structural(module, module_file)
        return

    graphmodules = _named_graphmodule_roots(module)
    if not graphmodules:
        raise ValueError(
            "pipeline_cache only supports GraphModule or nn.Module wrappers containing "
            "GraphModule children."
        )

    graphmodule_payloads = [
        {"name": name, "payload": _graphmodule_to_structural_payload(graph_module)}
        for name, graph_module in graphmodules
    ]
    try:
        for name, _ in graphmodules:
            _replace_submodule(module, name, _GraphModulePlaceholder(name))
        payload = {
            "module": module,
            "graphmodules": graphmodule_payloads,
        }
        torch.save(payload, module_file)
    except Exception as exc:
        detail = _find_unpickleable_path(payload if "payload" in locals() else graphmodule_payloads)
        if detail:
            raise ValueError(f"{exc}; first unpickleable payload path: {detail}") from exc
        raise
    finally:
        for name, graph_module in graphmodules:
            _replace_submodule(module, name, graph_module)


def _load_module_structural(module_file: Any) -> nn.Module:
    payload = torch.load(module_file, map_location="cpu", weights_only=False)
    if isinstance(payload, GraphModule):
        return payload
    if _is_graphmodule_structural_payload(payload):
        return _graphmodule_from_structural_payload(payload)
    if not _is_module_tree_structural_payload(payload):
        raise ValueError(
            f"pipeline cache module has unsupported payload shape: {type(payload).__name__}"
        )

    module = payload.get("module")
    if not isinstance(module, nn.Module):
        raise ValueError("pipeline cache module tree payload is missing the root module.")
    graphmodule_payloads = payload.get("graphmodules")
    if not isinstance(graphmodule_payloads, list):
        raise ValueError("pipeline cache module tree payload is missing GraphModule states.")

    for item in graphmodule_payloads:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError(f"pipeline cache module tree has invalid GraphModule entry: {item!r}")
        graph_module = _graphmodule_from_structural_payload(item["payload"])
        _replace_submodule(module, item["name"], graph_module)
    _mark_cached_shape_metadata_invalid(module)
    return module


def _clear_load_hooks(model: nn.Module) -> None:
    for module in model.modules():
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_post_hooks.clear()


def _validate_pre_weight_snapshot(model: nn.Module) -> None:
    materialized_params = [
        name for name, param in model.named_parameters() if param.device.type != "meta"
    ]
    if materialized_params:
        raise ValueError(
            "pipeline_cache only supports pre-weight-loading snapshots; materialized "
            f"parameters found: {materialized_params[:5]}"
        )


def _cm_sidecar(cm: CachedSequenceInterface) -> Dict[str, Any]:
    return {
        "cm_structural_state": {
            "active_args": list(cm.info._active_args),
            "active_host_prep_args": sorted(cm.info._active_host_prep_args),
            "use_flattened_layout": cm.info._use_flattened_layout,
        },
    }


def _restore_cm_sidecar(cm: CachedSequenceInterface, sidecars: Mapping[str, Any]) -> None:
    cm_structural = sidecars.get("cm_structural_state")
    if not cm_structural:
        return
    active_args = tuple(cm_structural.get("active_args", []))
    for arg_name in active_args:
        if arg_name not in cm.info.available_args:
            raise ValueError(f"Unknown SequenceInfo active arg '{arg_name}' in pipeline cache.")
    cm.info._active_args = active_args
    cm.info._active_host_prep_args = set(cm_structural.get("active_host_prep_args", []))
    cm.info._use_flattened_layout = bool(cm_structural.get("use_flattened_layout", False))


def _dist_config_payload(shared_config: SharedConfig) -> Dict[str, Any]:
    dist_config = getattr(shared_config, "dist_config", None)
    if dist_config is not None and hasattr(dist_config, "to_dict"):
        payload = dist_config.to_dict()
        payload.pop("rank", None)
        return payload
    return {"world_size": shared_config.world_size}


def _transform_config_payload(name: str, config: TransformConfig) -> Dict[str, Any]:
    payload = config.model_dump(mode="python") if hasattr(config, "model_dump") else dict(config)
    return payload


def _optimizer_items(
    optimizer_config: Optional[InferenceOptimizerConfig],
    shared_config: SharedConfig,
    fallback_name: str,
    fallback_config: TransformConfig,
) -> List[Tuple[str, TransformConfig]]:
    config = optimizer_config or getattr(shared_config, "transform_config", None)
    if config:
        return list(config.items())
    return [(fallback_name, fallback_config)]


def _cache_transform_index(items: Sequence[Tuple[str, TransformConfig]], fallback_name: str) -> int:
    for idx, (name, _) in enumerate(items):
        if name == fallback_name:
            return idx
    return len(items) - 1


def _callable_accepts_keyword(method: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _factory_model_identifier(
    factory: Optional[ModelFactory],
    non_prefix_transform_names: Sequence[str],
) -> Any:
    if factory is None:
        return None
    method = getattr(factory, "get_pipeline_cache_model_identifier", None)
    if callable(method):
        if _callable_accepts_keyword(method, "non_prefix_transform_names"):
            return method(non_prefix_transform_names=non_prefix_transform_names)
        return method()
    model_kwargs_method = getattr(factory, "_pipeline_cache_structural_model_kwargs", None)
    if callable(model_kwargs_method):
        if _callable_accepts_keyword(model_kwargs_method, "non_prefix_transform_names"):
            model_kwargs = model_kwargs_method(
                non_prefix_transform_names=non_prefix_transform_names
            )
        else:
            model_kwargs = model_kwargs_method()
    else:
        model_kwargs = getattr(factory, "model_kwargs", {})
    return {
        "factory_type": f"{type(factory).__module__}.{type(factory).__qualname__}",
        "model": getattr(factory, "model", None),
        "model_kwargs": model_kwargs,
    }


def _factory_checkpoint_fingerprint(factory: Optional[ModelFactory]) -> Any:
    if factory is None:
        return None
    method = getattr(factory, "get_pipeline_cache_checkpoint_fingerprint", None)
    if callable(method):
        return method()
    return {"model": getattr(factory, "model", None)}


def _python_version_payload() -> Dict[str, Any]:
    return {
        "version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
    }


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


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@TransformRegistry.register("pipeline_cache")
class PipelineCache(BaseTransform):
    """Transform that snapshots/restores the model at its configured pipeline position."""

    config: PipelineCacheConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return PipelineCacheConfig

    def maybe_restore(
        self,
        cm: CachedSequenceInterface,
        factory: Optional[ModelFactory],
        shared_config: SharedConfig,
        transform_index: int,
        optimizer_config: InferenceOptimizerConfig,
    ) -> Optional[nn.Module]:
        """Return a cached module for this transform point, or ``None`` on a miss."""
        if not self.config.enabled:
            return None
        context = self._build_context(factory, shared_config, optimizer_config, transform_index)
        if not _collective_bool_and(self._validate_manifest(context)[0]):
            return None

        local_success = False
        module: Optional[nn.Module] = None
        try:
            module = self._load_module(context, cm, factory)
            local_success = True
        except Exception as exc:
            ad_logger.warning(f"Failed to restore AutoDeploy pipeline cache: {exc}")
            module = None

        if not _collective_bool_and(local_success):
            return None
        assert module is not None
        ad_logger.info(f"Restored AutoDeploy pipeline cache from {self._rank_dir(context)}")
        return module

    def _apply_to_full_model(
        self,
        model: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        items = _optimizer_items(None, shared_config, self.get_transform_key(), self.config)
        transform_index = _cache_transform_index(items, self.get_transform_key())
        context = self._build_context(factory, shared_config, dict(items), transform_index)
        saved = self._save_module(context, model, cm)
        info = TransformInfo(
            skipped=not saved,
            num_matches=1 if saved else 0,
            is_clean=True,
            has_valid_shapes=True,
        )
        return model, info

    def _build_context(
        self,
        factory: Optional[ModelFactory],
        shared_config: SharedConfig,
        optimizer_config: Optional[InferenceOptimizerConfig],
        transform_index: int,
    ) -> Dict[str, Any]:
        root = Path(str(self.config.root)).expanduser()
        self._validate_cache_root(root)
        items = _optimizer_items(
            optimizer_config,
            shared_config,
            self.get_transform_key(),
            self.config,
        )
        prefix_items = list(items[:transform_index])
        non_prefix_transform_names = [name for name, _ in items[transform_index:]]
        transform_prefix = [
            {"name": name, "config": _transform_config_payload(name, config)}
            for name, config in prefix_items
        ]
        prefix_transform_names = [name for name, _ in prefix_items]
        transform_prefix_hash = _hash_payload({"transforms": transform_prefix})
        producer_hash = _build_producer_hash(prefix_transform_names)
        factory_producer_hash = _build_factory_producer_hash(factory)
        identity = {
            "boundary_name": self.get_transform_key(),
            "model_identifier": _factory_model_identifier(factory, non_prefix_transform_names),
            "checkpoint_fingerprint": _factory_checkpoint_fingerprint(factory),
            "factory_type": None
            if factory is None
            else f"{type(factory).__module__}.{type(factory).__qualname__}",
            "transform_prefix_hash": transform_prefix_hash,
            "producer_hash": producer_hash,
            "factory_producer_hash": factory_producer_hash,
            "dist_config": _dist_config_payload(shared_config),
            "trtllm_version": TRTLLM_VERSION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python": _python_version_payload(),
            "world_size": shared_config.world_size,
            "cache_key_extra": self.config.cache_key_extra,
        }
        return {
            "root": root,
            "cache_key": _hash_payload(identity),
            "identity": identity,
            "transform_index": transform_index,
            "transform_prefix_hash": transform_prefix_hash,
            "producer_hash": producer_hash,
            "factory_producer_hash": factory_producer_hash,
            "shared_config": shared_config,
        }

    def _validate_cache_root(self, root: Path) -> None:
        if not self.config.trust_cache_root:
            raise ValueError("pipeline_cache.trust_cache_root must be true when enabled.")
        root.mkdir(parents=True, exist_ok=True)
        if not self.config.strict_root_permissions:
            return
        mode = root.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                f"Pipeline cache root {root} is group/other writable; disable "
                "strict_root_permissions only for a trusted fleet cache root."
            )

    def _save_module(
        self,
        context: Mapping[str, Any],
        model: nn.Module,
        cm: CachedSequenceInterface,
    ) -> bool:
        _barrier()
        rank_dir = self._rank_dir(context)
        tmp_rank_dir = self._tmp_rank_dir(context)
        boundary_dir = self._boundary_dir(context)
        shutil.rmtree(tmp_rank_dir, ignore_errors=True)
        tmp_rank_dir.mkdir(parents=True, exist_ok=True)

        local_save_success = False
        try:
            _validate_pre_weight_snapshot(model)
            hook_specs, has_unknown = collect_hook_specs(model)
            if has_unknown:
                raise ValueError("graph contains unserializable AD-managed load hooks")
            ad_hook_specs = [spec for spec in hook_specs if spec.get("type") != "source_model"]
            source_hook_specs = [spec for spec in hook_specs if spec.get("type") == "source_model"]
            sidecars = _cm_sidecar(cm)

            hook_records = _snapshot_and_clear_load_hooks(model)
            forward_hook_records = _snapshot_and_clear_forward_hooks(model)
            self_ref_records = _snapshot_and_clear_direct_self_references(model)
            try:
                with open(tmp_rank_dir / MODULE_FILE_NAME, "wb") as module_file:
                    _save_module_structural(model, module_file)
                    module_file.flush()
                    os.fsync(module_file.fileno())
            finally:
                _restore_direct_self_references(self_ref_records)
                _restore_forward_hooks(forward_hook_records)
                _restore_load_hooks(hook_records)

            _write_json_atomic(tmp_rank_dir / HOOKS_FILE_NAME, ad_hook_specs)
            _write_json_atomic(tmp_rank_dir / SOURCE_HOOKS_FILE_NAME, source_hook_specs)
            _write_json_atomic(tmp_rank_dir / SIDECARS_FILE_NAME, sidecars)
            checksums = {
                file_name: _sha256_file(tmp_rank_dir / file_name)
                for file_name in (
                    MODULE_FILE_NAME,
                    HOOKS_FILE_NAME,
                    SOURCE_HOOKS_FILE_NAME,
                    SIDECARS_FILE_NAME,
                )
            }
            manifest = self._build_manifest(
                context,
                hook_spec_count=len(ad_hook_specs),
                source_model_hooks_required=bool(source_hook_specs),
                file_checksums=checksums,
            )
            _write_json_atomic(tmp_rank_dir / MANIFEST_FILE_NAME, manifest)
            _fsync_dir(tmp_rank_dir)
            local_save_success = True
        except Exception as exc:
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)
            ad_logger.warning(f"Skipping AutoDeploy pipeline cache save: {exc}")

        if not _collective_bool_and(local_save_success):
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)
            shutil.rmtree(rank_dir, ignore_errors=True)
            _barrier()
            return False

        boundary_dir.mkdir(parents=True, exist_ok=True)
        _fsync_dir(boundary_dir)
        _atomic_publish_rank_dir(tmp_rank_dir, rank_dir)
        _barrier()
        ad_logger.info(f"Saved AutoDeploy pipeline cache to {rank_dir}")
        return True

    def _build_manifest(
        self,
        context: Mapping[str, Any],
        hook_spec_count: int,
        source_model_hooks_required: bool,
        file_checksums: Mapping[str, str],
    ) -> Dict[str, Any]:
        shared_config = context["shared_config"]
        identity = context["identity"]
        return {
            "requires_weights_only_false": True,
            "boundary_name": self.get_transform_key(),
            "transform_index": context["transform_index"],
            "boundary_stage": self.config.stage.value,
            "cache_key": context["cache_key"],
            "transform_prefix_hash": context["transform_prefix_hash"],
            "producer_hash": context["producer_hash"],
            "factory_producer_hash": identity["factory_producer_hash"],
            "model_identifier": identity["model_identifier"],
            "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
            "factory_type": identity["factory_type"],
            "trtllm_version": TRTLLM_VERSION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python": _python_version_payload(),
            "dist_config": _dist_config_payload(shared_config),
            "weights_materialized": False,
            "source_model_hooks_required": source_model_hooks_required,
            "has_unserializable_hooks": False,
            "hook_spec_count": hook_spec_count,
            "file_checksums": dict(file_checksums),
            "rank": shared_config.local_rank,
            "world_size": shared_config.world_size,
        }

    def _validate_manifest(
        self, context: Mapping[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if not self._has_complete_snapshot(context):
            return False, None
        manifest_path = self._rank_dir(context) / MANIFEST_FILE_NAME
        try:
            manifest = _read_json(manifest_path)
        except (json.JSONDecodeError, OSError) as exc:
            ad_logger.warning(f"Ignoring invalid pipeline cache manifest {manifest_path}: {exc}")
            return False, None

        expected = {
            "requires_weights_only_false": True,
            "boundary_name": self.get_transform_key(),
            "cache_key": context["cache_key"],
            "transform_prefix_hash": context["transform_prefix_hash"],
            "producer_hash": context["producer_hash"],
            "factory_producer_hash": context["identity"]["factory_producer_hash"],
            "world_size": context["shared_config"].world_size,
            "rank": context["shared_config"].local_rank,
            "trtllm_version": TRTLLM_VERSION,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python": _python_version_payload(),
        }
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                ad_logger.info(
                    f"Pipeline cache manifest mismatch: {key}={manifest.get(key)!r} "
                    f"!= {expected_value!r}"
                )
                return False, manifest
        if (
            manifest.get("source_model_hooks_required", False)
            and not context["identity"]["factory_type"]
        ):
            return False, manifest
        try:
            self._verify_file_checksums(context, manifest)
        except ValueError as exc:
            ad_logger.warning(str(exc))
            return False, manifest
        return True, manifest

    def _load_module(
        self,
        context: Mapping[str, Any],
        cm: CachedSequenceInterface,
        factory: Optional[ModelFactory],
    ) -> nn.Module:
        ready, manifest = self._validate_manifest(context)
        if not ready or manifest is None:
            raise ValueError("pipeline cache manifest is not valid")

        rank_dir = self._rank_dir(context)
        module = _load_module_structural(rank_dir / MODULE_FILE_NAME)

        _clear_load_hooks(module)
        ad_hook_specs = _read_json(rank_dir / HOOKS_FILE_NAME)
        source_hook_specs = _read_json(rank_dir / SOURCE_HOOKS_FILE_NAME)
        sidecars = _read_json(rank_dir / SIDECARS_FILE_NAME)

        reattach_hooks(module, ad_hook_specs)
        if source_hook_specs:
            self._replay_source_model_hooks(module, source_hook_specs, factory)
        _restore_cm_sidecar(cm, sidecars)
        for _, graph_module in named_graphmodules(module):
            graph_module.graph.lint()
        return module

    def _replay_source_model_hooks(
        self,
        module: nn.Module,
        expected_specs: List[Dict[str, Any]],
        factory: Optional[ModelFactory],
    ) -> None:
        if factory is None:
            raise ValueError("Source-model hooks require a model factory for restore.")

        expected_keys = sorted(_source_model_hook_spec_key(spec) for spec in expected_specs)
        original_model = factory.build_model("meta")
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
                target_mod = module if scope == "root" else module.get_submodule(scope)
                if record["position"] == "pre":
                    target_mod._register_load_state_dict_pre_hook(
                        record["hook_fn"],
                        with_module=record["with_module"],
                    )
                else:
                    target_mod.register_load_state_dict_post_hook(record["hook_fn"])
        finally:
            del original_model

    def _verify_file_checksums(
        self, context: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        rank_dir = self._rank_dir(context)
        checksums = manifest.get("file_checksums", {}) or {}
        required_files = (
            MODULE_FILE_NAME,
            HOOKS_FILE_NAME,
            SOURCE_HOOKS_FILE_NAME,
            SIDECARS_FILE_NAME,
        )
        for file_name in required_files:
            expected = checksums.get(file_name)
            if not expected:
                raise ValueError(f"Pipeline cache manifest is missing checksum for {file_name}.")
            path = rank_dir / file_name
            if not path.exists():
                raise ValueError(f"Pipeline cache file is missing: {path}")
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"Pipeline cache checksum mismatch for {path}: {actual} != {expected}"
                )

    def _has_complete_snapshot(self, context: Mapping[str, Any]) -> bool:
        boundary_dir = self._boundary_dir(context)
        for rank in range(context["shared_config"].world_size):
            rank_dir = boundary_dir / f"rank_{rank}"
            for file_name in (
                MANIFEST_FILE_NAME,
                MODULE_FILE_NAME,
                HOOKS_FILE_NAME,
                SOURCE_HOOKS_FILE_NAME,
                SIDECARS_FILE_NAME,
            ):
                if not (rank_dir / file_name).exists():
                    return False
        return True

    def _boundary_dir(self, context: Mapping[str, Any]) -> Path:
        return context["root"] / context["cache_key"] / self.get_transform_key()

    def _rank_dir(self, context: Mapping[str, Any]) -> Path:
        return self._boundary_dir(context) / f"rank_{context['shared_config'].local_rank}"

    def _tmp_rank_dir(self, context: Mapping[str, Any]) -> Path:
        shared_config = context["shared_config"]
        return context["root"] / (
            f".{context['cache_key']}.{self.get_transform_key()}.rank_"
            f"{shared_config.local_rank}.tmp.{os.getpid()}"
        )

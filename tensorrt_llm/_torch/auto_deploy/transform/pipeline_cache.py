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
The FX graph is serialized as structured JSON (``ad_ir.json``), capturing every
node's op, target, args/kwargs, and placeholder shape metadata.  On restore the
``torch.fx.Graph`` is rebuilt node-by-node via the Graph construction API — **no
re-tracing** — so all structural and metadata invariants are preserved.

See ``ad_ir.py`` for the IR data model and serialization utilities.
Load hooks are rebuilt from declarative ``hook_specs`` embedded in the IR.
"""

import hashlib
import importlib
import inspect
import json
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm.version import __version__ as TRTLLM_VERSION

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from .ad_ir import IRGraph, build_graph_module, extract_ir, hydrate_shapes, load_ir, save_ir
from .interface import (
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)

_BOUNDARY_CLASS_PRE_WEIGHT_LOAD = "pre_weight_load"

# ---------------------------------------------------------------------------
# Hash / JSON helpers
# ---------------------------------------------------------------------------


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
    if hasattr(value, "value"):
        return _canonicalize_for_hash(value.value)
    if hasattr(value, "to_dict"):
        return _canonicalize_for_hash(value.to_dict())
    return value


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonicalize_for_hash(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, sort_keys: bool = True) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


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
    payload: Dict[str, Any] = {"path": _repo_relative_path(path)}
    try:
        payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        payload["sha256"] = None
    return payload


def _core_producer_files() -> Tuple[Path, ...]:
    this_file = Path(__file__).resolve()
    auto_deploy_root = this_file.parents[1]
    transform_root = this_file.parent
    return (
        auto_deploy_root / "export" / "export.py",
        transform_root / "ad_ir.py",
        transform_root / "interface.py",
        transform_root / "library" / "sharding.py",
        this_file,
    )


def _build_producer_hash(prefix_transform_names: Sequence[str]) -> str:
    transform_sources = [
        {"name": name, **_describe_source_object(TransformRegistry.get(name))}
        for name in prefix_transform_names
    ]
    core_files = [_describe_file(path) for path in _core_producer_files()]
    return _hash_payload(
        {
            "transform_sources": transform_sources,
            "core_files": core_files,
        }
    )


# ---------------------------------------------------------------------------
# HookSpec: declarative representation of load hooks
# ---------------------------------------------------------------------------


def _identify_hook(hook: Any, scope: str = "root") -> Optional[Dict[str, Any]]:
    """Pattern-match a live hook object and return its declarative HookSpec dict.

    Returns ``None`` if the hook cannot be identified.
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

    # Source-model hooks (bound methods or static methods on submodules, replayed via factory).
    # By this point every known AD-pipeline hook pattern has been checked, so any remaining
    # callable on a non-root submodule is a source-model hook — including @staticmethod hooks
    # which lack ``__self__``.
    if scope != "root" and callable(hook):
        self_obj = getattr(hook, "__self__", None)
        return {
            "type": "source_model",
            "scope": scope,
            "position": "pre",
            "class_name": type(self_obj).__name__ if self_obj is not None else "static",
            "method_name": getattr(hook, "__name__", "unknown"),
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
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope)
            if spec is not None:
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


# ---------------------------------------------------------------------------
# HookSpec restore: rebuild live hooks from specs
# ---------------------------------------------------------------------------


def reattach_hooks(gm: GraphModule, specs: List[Dict[str, Any]]) -> None:
    """Rebuild and register load hooks on *gm* from declarative HookSpec dicts."""
    skipped = 0
    for spec in specs:
        if spec.get("type") == "source_model":
            skipped += 1
            continue  # replayed separately via _replay_source_model_hooks

        hook_fn = _rebuild_hook(spec)
        if hook_fn is None:
            ad_logger.warning(
                f"Pipeline cache: could not rebuild hook type={spec.get('type')!r} "
                f"scope={spec.get('scope')!r}"
            )
            continue

        scope = spec.get("scope", "root")
        position = spec.get("position", "pre")

        target_mod = gm if scope == "root" else gm.get_submodule(scope)

        if position == "pre":
            target_mod._register_load_state_dict_pre_hook(hook_fn)
        else:
            target_mod.register_load_state_dict_post_hook(hook_fn)

    if skipped:
        ad_logger.debug(f"Skipped {skipped} source_model hook(s) (replayed via factory)")


def _rebuild_hook(spec: Dict[str, Any]) -> Optional[Callable]:
    """Rebuild a single live hook from its HookSpec dict."""
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
        return None  # replayed via _replay_source_model_hooks
    return None


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


def _rebuild_shard_quant_hook(spec: Dict[str, Any]) -> Optional[Callable]:
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

    cls = _QUANT_CLASSES.get(spec.get("quant_class", ""))
    if cls is None:
        ad_logger.warning(
            f"Pipeline cache: unknown quant sharding class {spec.get('quant_class')!r}"
        )
        return None

    fused_weight_dims = spec.get("fused_weight_dims")
    # The base class requires target_node / config / split_dim, but they
    # are unused by shard_load_hook.  Supply placeholders so Pydantic validation passes.
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


def _rebuild_quant_hook(spec: Dict[str, Any]) -> Optional[Callable]:
    """Rebuild a quantization load/amax/post_load hook from its spec."""
    quant_module = spec.get("quant_module", "")
    quant_class = spec.get("quant_class", "")

    try:
        mod = importlib.import_module(quant_module)
        cls = getattr(mod, quant_class, None)
        if cls is None:
            # Try nested class names (e.g., "Outer.Inner")
            parts = quant_class.split(".")
            cls = mod
            for p in parts:
                cls = getattr(cls, p, None)
                if cls is None:
                    break
    except (ImportError, AttributeError):
        cls = None

    if cls is None:
        ad_logger.warning(
            f"Pipeline cache: cannot reconstruct quant class {quant_module}.{quant_class}"
        )
        return None

    try:
        instance = cls.__new__(cls)
    except Exception:  # noqa: BLE001
        ad_logger.warning(f"Pipeline cache: cannot instantiate {quant_class}")
        return None

    hook_type = spec["type"]
    if hook_type == "quant_load":
        method = getattr(instance, "load_hook", None)
        if method is None:
            return None
        return partial(method, weight_name=spec["weight_name"])
    if hook_type == "quant_amax":
        method = getattr(instance, "convert_amax_hook", None)
        if method is None:
            return None
        return partial(method, scale_name=spec["scale_name"], amax_name=spec["amax_name"])
    if hook_type == "quant_post_load":
        method = getattr(instance, "post_load_hook", None)
        if method is None:
            return None
        return partial(method, weight_name=spec["weight_name"])
    return None


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

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._boundary_info)

    def _build_boundary_info(self) -> Dict[str, Dict[str, Any]]:
        if not self._enabled:
            return {}

        boundaries: Sequence[str] = getattr(self._cache_config, "boundaries", ())
        info: Dict[str, Dict[str, Any]] = {}
        ordered_items = list(self._config.items())
        for idx, (name, transform_config) in enumerate(ordered_items):
            if name not in boundaries:
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

        return info

    def _mapping_payload(self) -> Dict[str, Any]:
        mapping = self._shared_config.mapping
        if mapping is not None and hasattr(mapping, "to_dict"):
            payload = mapping.to_dict()
            payload.pop("rank", None)
            return payload
        return {"world_size": self._shared_config.world_size}

    def maybe_restore(self, cm: CachedSequenceInterface) -> Tuple[Optional[nn.Module], int]:
        """Restore the highest valid boundary if one exists."""
        if not self.enabled:
            return None, 0

        boundaries = sorted(
            self._boundary_info.items(),
            key=lambda item: item[1]["transform_index"],
            reverse=True,
        )
        for boundary_name, boundary_info in boundaries:
            if not self._has_complete_boundary_snapshot(boundary_name):
                continue

            manifest_path = self._rank_dir(boundary_name).joinpath("manifest.json")

            try:
                manifest = _read_json(manifest_path)
            except json.JSONDecodeError:
                ad_logger.warning(f"Ignoring invalid pipeline cache manifest: {manifest_path}")
                continue

            if not self._manifest_matches(boundary_name, boundary_info, manifest):
                continue

            if manifest.get("has_unserializable_hooks", False):
                ad_logger.info(
                    f"Skipping boundary '{boundary_name}': artifact has unserializable hooks"
                )
                continue

            load_result = self._load_module(boundary_name)
            if load_result is None:
                continue
            module, ir = load_result

            # Reattach hooks from IR-embedded hook specs
            if isinstance(module, GraphModule) and ir.hook_specs:
                try:
                    reattach_hooks(module, ir.hook_specs)
                    ad_logger.debug(f"Reattached {len(ir.hook_specs)} hook(s) from AD IR")
                except Exception as exc:  # noqa: BLE001
                    ad_logger.warning(f"Failed to restore hooks from AD IR: {exc}")

            if ir.source_model_hooks_required and self._factory is not None:
                self._replay_source_model_hooks(module)

            self._restore_sidecars(boundary_name, module, cm, ir)

            next_index = int(boundary_info["transform_index"]) + 1
            ad_logger.info(
                f"Restored AutoDeploy pipeline snapshot for '{boundary_name}' "
                f"from {self._rank_dir(boundary_name)}"
            )
            return module, next_index

        return None, 0

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
            ad_logger.warning(
                f"Skipping pipeline snapshot for '{transform_name}' because the module is not a "
                f"GraphModule ({type(mod).__name__})."
            )
            return

        self._barrier()

        rank_dir = self._rank_dir(transform_name)
        rank_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = rank_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

        try:
            hook_specs, has_unknown = collect_hook_specs(mod)

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
                source_model_hooks_required=True,
            )
            save_ir(ir, real_buffers, rank_dir)

            manifest = self._build_manifest(
                transform_name,
                transform_idx,
                transform_config,
                hook_spec_count=len(hook_specs),
                has_unserializable_hooks=has_unknown,
            )
            _write_json(manifest_path, manifest)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(rank_dir, ignore_errors=True)
            ad_logger.warning(
                f"Skipping AutoDeploy pipeline snapshot for '{transform_name}' because saving the "
                f"snapshot failed: {exc}"
            )
            return

        ad_logger.info(f"Saved AutoDeploy pipeline snapshot for '{transform_name}' to {rank_dir}")

    def _build_manifest(
        self,
        transform_name: str,
        transform_idx: int,
        transform_config: TransformConfig,
        hook_spec_count: int = 0,
        has_unserializable_hooks: bool = False,
    ) -> Dict[str, Any]:
        boundary_info = self._boundary_info[transform_name]
        return {
            "boundary_name": transform_name,
            "transform_index": transform_idx,
            "boundary_stage": transform_config.stage.value,
            "boundary_class": _BOUNDARY_CLASS_PRE_WEIGHT_LOAD,
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
            "source_model_hooks_required": True,
            "has_unserializable_hooks": has_unserializable_hooks,
            "hook_spec_count": hook_spec_count,
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
            "boundary_name": boundary_name,
            "boundary_class": _BOUNDARY_CLASS_PRE_WEIGHT_LOAD,
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

    def _load_module(self, boundary_name: str) -> Optional[Tuple[nn.Module, IRGraph]]:
        """Load a cached GraphModule via AD IR.  Returns (module, ir) or None."""
        rank_dir = self._rank_dir(boundary_name)
        try:
            result = load_ir(rank_dir)
            if result is None:
                return None
            ir, real_buffers = result
            gm = build_graph_module(ir, real_buffers)
            hydrate_shapes(gm, ir)
            return gm, ir
        except Exception as exc:  # noqa: BLE001
            ad_logger.warning(
                f"Failed to restore AutoDeploy pipeline snapshot module from {rank_dir}: {exc}"
            )
            return None

    def _replay_source_model_hooks(self, mod: nn.Module) -> None:
        """Replay ``_add_missing_load_hooks`` using a fresh meta-device model."""
        try:
            from ..export.export import _add_missing_load_hooks

            original_model = self._factory.build_model("meta")
            _add_missing_load_hooks(mod, original_model)
            del original_model
            ad_logger.debug("Replayed source-model load hooks onto restored module")
        except Exception as exc:  # noqa: BLE001
            ad_logger.warning(f"Failed to replay source-model hooks: {exc}")

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
        cache_key = self._boundary_info[boundary_name]["cache_key"]
        return self._root / cache_key / boundary_name / f"rank_{self._shared_config.local_rank}"

    def _has_complete_boundary_snapshot(self, boundary_name: str) -> bool:
        cache_key = self._boundary_info[boundary_name]["cache_key"]
        boundary_dir = self._root / cache_key / boundary_name
        for rank in range(self._shared_config.world_size):
            if not (boundary_dir / f"rank_{rank}" / "manifest.json").exists():
                return False
        return True

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

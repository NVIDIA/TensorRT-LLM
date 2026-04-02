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

"""AutoDeploy Intermediate Representation (AD IR) for pipeline cache serialization.

The AD IR captures the FX graph topology, module structure, and placeholder metadata
in a JSON-serializable format.  On restore the ``torch.fx.Graph`` is reconstructed
node-by-node (no re-tracing), so all structural and metadata invariants are preserved.

Typical round-trip::

    ir = extract_ir(graph_module)  # GraphModule -> IR
    data = ir.to_dict()  # IR -> JSON-serializable dict
    ir2 = IRGraph.from_dict(data)  # dict -> IR
    gm2 = build_graph_module(ir2, real_buffers)  # IR -> fresh GraphModule
    hydrate_shapes(gm2, ir2)  # populate FakeTensor metadata on all nodes
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import operator
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

from ..utils.logger import ad_logger

try:
    from pydantic import BaseModel as _PydanticBaseModel
except ImportError:
    _PydanticBaseModel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

AD_IR_FORMAT_VERSION = 4

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}

_STR_TO_DTYPE: Dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dt: torch.dtype) -> str:
    s = _DTYPE_TO_STR.get(dt)
    if s is not None:
        return s
    return str(dt).replace("torch.", "")


def _str_to_dtype(s: str) -> torch.dtype:
    dt = _STR_TO_DTYPE.get(s)
    if dt is not None:
        return dt
    return getattr(torch, s)


# ---------------------------------------------------------------------------
# memory_format / layout helpers
# ---------------------------------------------------------------------------

_MEMORY_FORMAT_TO_STR = {
    torch.contiguous_format: "contiguous_format",
    torch.channels_last: "channels_last",
    torch.channels_last_3d: "channels_last_3d",
    torch.preserve_format: "preserve_format",
}
_STR_TO_MEMORY_FORMAT = {v: k for k, v in _MEMORY_FORMAT_TO_STR.items()}

_LAYOUT_TO_STR = {
    torch.strided: "strided",
    torch.sparse_coo: "sparse_coo",
}
_STR_TO_LAYOUT = {v: k for k, v in _LAYOUT_TO_STR.items()}


# ===================================================================
# Target serialization / resolution
# ===================================================================


def serialize_target(target: Any) -> str:
    """Convert a call_function target to a stable string key."""
    if isinstance(target, torch._ops.OpOverload):
        schema_name = target._schema.name
        overload = target._overloadname
        return f"{schema_name}.{overload}"

    if isinstance(target, torch._ops.OpOverloadPacket):
        return f"{target._qualified_op_name}.__packet__"

    if target is operator.getitem:
        return "operator.getitem"
    if target is getattr:
        return "builtins.getattr"

    module = getattr(target, "__module__", None) or ""
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", "")
    if module and qualname:
        # __qualname__ may contain a class prefix (e.g. "_VariableFunctionsClass.eq"
        # for torch.eq).  If so, the combined "module.qualname" path won't be
        # resolvable by importlib.  Normalize to "module.leaf" when the leaf
        # attribute is directly accessible on the module.
        if "." in qualname:
            leaf = qualname.rsplit(".", 1)[-1]
            try:
                mod = importlib.import_module(module)
                if hasattr(mod, leaf) and getattr(mod, leaf) is target:
                    return f"{module}.{leaf}"
            except (ModuleNotFoundError, ImportError):
                pass
        return f"{module}.{qualname}"

    raise ValueError(f"Cannot serialize target: {target!r}")


def resolve_target(key: str) -> Any:
    """Resolve a serialized target string back to a Python callable."""
    if key == "operator.getitem":
        return operator.getitem
    if key == "builtins.getattr":
        return getattr

    if "::" in key:
        schema_part, overload = key.rsplit(".", 1)
        ns, op_name = schema_part.split("::", 1)
        packet = getattr(getattr(torch.ops, ns), op_name)
        if overload == "__packet__":
            return packet
        try:
            return getattr(packet, overload)
        except AttributeError:
            return packet

    module_path, attr_name = key.rsplit(".", 1)
    parts = attr_name.split(".")
    try:
        mod = importlib.import_module(module_path)
    except (ModuleNotFoundError, ImportError):
        # module_path may include a non-importable class segment
        # (e.g. "torch._VariableFunctionsClass" for key "torch._VariableFunctionsClass.eq").
        # Walk up to the nearest importable prefix.
        segments = module_path.split(".")
        leaf = parts[-1]
        for i in range(len(segments) - 1, 0, -1):
            prefix = ".".join(segments[:i])
            try:
                parent = importlib.import_module(prefix)
            except (ModuleNotFoundError, ImportError):
                continue
            # Try the full attribute chain first
            try:
                obj = parent
                for seg in segments[i:] + parts:
                    obj = getattr(obj, seg)
                return obj
            except AttributeError:
                pass
            # Fallback: the leaf attr may live directly on the parent module
            if hasattr(parent, leaf):
                return getattr(parent, leaf)
        raise ModuleNotFoundError(f"Cannot resolve target: {key}")
    obj = mod
    for part in parts:
        obj = getattr(obj, part)
    return obj


# ===================================================================
# Arg serialization / resolution
# ===================================================================

_REF_KEY = "__ref__"
_TUPLE_KEY = "__tuple__"
_DTYPE_KEY = "__dtype__"
_DEVICE_KEY = "__device__"
_MEMFMT_KEY = "__memory_format__"
_LAYOUT_KEY = "__layout__"
_TORCH_SIZE_KEY = "__torch_size__"
_SLICE_KEY = "__slice__"


def _concretize(val: Any) -> Any:
    """Convert SymInt/SymFloat/SymBool to plain Python scalars."""
    if isinstance(val, torch.SymInt):
        return int(val)
    if isinstance(val, torch.SymFloat):
        return float(val)
    if isinstance(val, torch.SymBool):
        return bool(val)
    return val


def serialize_arg(arg: Any) -> Any:
    """Recursively convert an FX node argument into a JSON-serializable value."""
    if isinstance(arg, Node):
        return {_REF_KEY: arg.name}

    if isinstance(arg, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _concretize(arg)

    if arg is None or isinstance(arg, (bool, int, float, str)):
        return arg

    if isinstance(arg, tuple):
        return {_TUPLE_KEY: [serialize_arg(a) for a in arg]}

    if isinstance(arg, list):
        return [serialize_arg(a) for a in arg]

    if isinstance(arg, torch.dtype):
        return {_DTYPE_KEY: _dtype_to_str(arg)}

    if isinstance(arg, torch.device):
        return {_DEVICE_KEY: str(arg)}

    if isinstance(arg, torch.memory_format):
        s = _MEMORY_FORMAT_TO_STR.get(arg)
        if s is None:
            raise ValueError(f"Unsupported memory_format: {arg}")
        return {_MEMFMT_KEY: s}

    if isinstance(arg, torch.layout):
        s = _LAYOUT_TO_STR.get(arg)
        if s is None:
            raise ValueError(f"Unsupported layout: {arg}")
        return {_LAYOUT_KEY: s}

    if isinstance(arg, torch.Size):
        return {_TORCH_SIZE_KEY: [_concretize(s) for s in arg]}

    if isinstance(arg, slice):
        return {_SLICE_KEY: [arg.start, arg.stop, arg.step]}

    raise TypeError(f"Cannot serialize FX argument of type {type(arg).__name__}: {arg!r}")


def resolve_arg(arg: Any, node_map: Dict[str, Node]) -> Any:
    """Recursively resolve a serialized argument back to an FX-compatible value."""
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return arg

    if isinstance(arg, list):
        return [resolve_arg(a, node_map) for a in arg]

    if isinstance(arg, dict):
        if _REF_KEY in arg:
            name = arg[_REF_KEY]
            if name not in node_map:
                raise KeyError(f"Node reference '{name}' not found in graph")
            return node_map[name]

        if _TUPLE_KEY in arg:
            return tuple(resolve_arg(a, node_map) for a in arg[_TUPLE_KEY])

        if _DTYPE_KEY in arg:
            return _str_to_dtype(arg[_DTYPE_KEY])

        if _DEVICE_KEY in arg:
            return torch.device(arg[_DEVICE_KEY])

        if _MEMFMT_KEY in arg:
            return _STR_TO_MEMORY_FORMAT[arg[_MEMFMT_KEY]]

        if _LAYOUT_KEY in arg:
            return _STR_TO_LAYOUT[arg[_LAYOUT_KEY]]

        if _TORCH_SIZE_KEY in arg:
            return torch.Size(arg[_TORCH_SIZE_KEY])

        if _SLICE_KEY in arg:
            parts = arg[_SLICE_KEY]
            return slice(parts[0], parts[1], parts[2])

        return {k: resolve_arg(v, node_map) for k, v in arg.items()}

    raise TypeError(f"Cannot resolve argument: {arg!r}")


# ===================================================================
# TreeSpec serialization (pickle + base64 — opaque but stable per PyTree version)
# ===================================================================


def _serialize_treespec(spec: Any) -> Optional[str]:
    if spec is None:
        return None
    try:
        buf = io.BytesIO()
        pickle.dump(spec, buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _deserialize_treespec(data: Optional[str]) -> Any:
    if data is None:
        return None
    raw = base64.b64decode(data.encode("ascii"))
    return pickle.loads(raw)  # noqa: S301


# ===================================================================
# IR data model
# ===================================================================


@dataclass
class IRNode:
    """Serializable representation of a single ``torch.fx.Node``."""

    name: str
    op: str
    target: str
    args: list
    kwargs: dict

    val_type: Optional[str] = None
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "args": self.args,
            "kwargs": self.kwargs,
        }
        if self.val_type is not None:
            d["val_type"] = self.val_type
            if self.shape is not None:
                d["shape"] = self.shape
            if self.dtype is not None:
                d["dtype"] = self.dtype
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> IRNode:
        return cls(
            name=d["name"],
            op=d["op"],
            target=d["target"],
            args=d["args"],
            kwargs=d["kwargs"],
            val_type=d.get("val_type"),
            shape=d.get("shape"),
            dtype=d.get("dtype"),
        )


@dataclass
class ParamSpec:
    """Metadata for a single parameter."""

    shape: List[int]
    dtype: str
    requires_grad: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": self.shape, "dtype": self.dtype, "requires_grad": self.requires_grad}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ParamSpec:
        return cls(shape=d["shape"], dtype=d["dtype"], requires_grad=d["requires_grad"])


@dataclass
class BufferSpec:
    """Metadata for a single buffer."""

    shape: List[int]
    dtype: str
    is_meta: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": self.shape, "dtype": self.dtype, "is_meta": self.is_meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BufferSpec:
        return cls(shape=d["shape"], dtype=d["dtype"], is_meta=d["is_meta"])


@dataclass
class IRGraph:
    """Complete serializable representation of a post-sharding ``GraphModule``."""

    format_version: int
    nodes: List[IRNode]
    module_tree: Dict[str, str]
    params: Dict[str, ParamSpec]
    buffers: Dict[str, BufferSpec]
    in_spec: Optional[str] = None
    out_spec: Optional[str] = None
    orig_args: Optional[List[str]] = None
    hook_specs: List[Dict[str, Any]] = field(default_factory=list)
    autodeploy_meta: Dict[str, Any] = field(default_factory=dict)
    source_model_hooks_required: bool = False
    gm_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "nodes": [n.to_dict() for n in self.nodes],
            "module_tree": self.module_tree,
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "buffers": {k: v.to_dict() for k, v in self.buffers.items()},
            "in_spec": self.in_spec,
            "out_spec": self.out_spec,
            "orig_args": self.orig_args,
            "hook_specs": self.hook_specs,
            "autodeploy_meta": self.autodeploy_meta,
            "source_model_hooks_required": self.source_model_hooks_required,
            "gm_attrs": self.gm_attrs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> IRGraph:
        return cls(
            format_version=d["format_version"],
            nodes=[IRNode.from_dict(n) for n in d["nodes"]],
            module_tree=d["module_tree"],
            params={k: ParamSpec.from_dict(v) for k, v in d["params"].items()},
            buffers={k: BufferSpec.from_dict(v) for k, v in d["buffers"].items()},
            in_spec=d.get("in_spec"),
            out_spec=d.get("out_spec"),
            orig_args=d.get("orig_args"),
            hook_specs=d.get("hook_specs", []),
            autodeploy_meta=d.get("autodeploy_meta", {}),
            source_model_hooks_required=d.get("source_model_hooks_required", False),
            gm_attrs=d.get("gm_attrs", {}),
        )


# ===================================================================
# GraphModule custom attribute serialization (generic, registry-free)
# ===================================================================

# Attributes that belong to nn.Module / GraphModule internals and should
# never be captured.  Everything else that is a Pydantic BaseModel is
# automatically serialized.
_GM_INTERNAL_ATTRS = frozenset(
    {
        "training",
        "_parameters",
        "_buffers",
        "_modules",
        "_backward_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_non_persistent_buffers_set",
        "_is_full_backward_hook",
        "meta",
        "_graph",
        "_code",
        "_in_spec",
        "_out_spec",
        "_graphmodule_cls_name",
    }
)


def _extract_pydantic_attrs(gm: GraphModule) -> Dict[str, Dict[str, Any]]:
    """Auto-discover and serialize any Pydantic BaseModel attribute on *gm*.

    The class path is derived from the object itself — no registry required.
    Fields that are not JSON-serializable are silently skipped so that models
    with non-serializable fields (e.g. ``MpiTopology``) degrade gracefully.
    """
    if _PydanticBaseModel is None:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for attr_name, obj in gm.__dict__.items():
        if attr_name in _GM_INTERNAL_ATTRS or attr_name.startswith("__"):
            continue
        if not isinstance(obj, _PydanticBaseModel):
            continue

        class_path = f"{type(obj).__module__}.{type(obj).__qualname__}"
        data = _robust_model_dump(obj)
        if data is not None:
            try:
                json.dumps(data)
                result[attr_name] = {"class": class_path, "data": data}
            except (TypeError, ValueError):
                ad_logger.debug(f"Skipping non-JSON-serializable attr '{attr_name}'")
    return result


def _robust_model_dump(obj: Any) -> Optional[Dict[str, Any]]:
    """Dump a Pydantic model to a JSON-safe dict, skipping problem fields."""
    if _PydanticBaseModel is None or not isinstance(obj, _PydanticBaseModel):
        return None

    try:
        data = obj.model_dump(mode="json")
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        pass

    result: Dict[str, Any] = {}
    for field_name in obj.model_fields:
        try:
            partial = obj.model_dump(mode="json", include={field_name})
            json.dumps(partial)
            result.update(partial)
        except Exception:  # noqa: BLE001
            val = getattr(obj, field_name, None)
            if isinstance(val, _PydanticBaseModel):
                sub = _robust_model_dump(val)
                if sub is not None:
                    result[field_name] = sub
                continue
            ad_logger.debug(f"Skipping field '{field_name}' in {type(obj).__name__}")
    return result if result else None


def _restore_pydantic_attrs(gm: GraphModule, gm_attrs: Dict[str, Dict[str, Any]]) -> None:
    """Restore Pydantic model attributes onto a GraphModule from IR data."""
    for attr_name, entry in gm_attrs.items():
        class_path = entry.get("class", "")
        data = entry.get("data", {})
        try:
            module_path, cls_name = class_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            obj = cls.model_validate(data)
            setattr(gm, attr_name, obj)
            ad_logger.debug(f"Restored attr '{attr_name}' ({cls_name}) from AD IR")
        except Exception as exc:  # noqa: BLE001
            ad_logger.warning(f"Could not restore GraphModule attribute '{attr_name}': {exc}")


# ===================================================================
# Extract: GraphModule -> IRGraph
# ===================================================================


def extract_ir(
    gm: GraphModule,
    hook_specs: Optional[List[Dict[str, Any]]] = None,
    autodeploy_meta: Optional[Dict[str, Any]] = None,
    source_model_hooks_required: bool = False,
) -> Tuple[IRGraph, Dict[str, torch.Tensor]]:
    """Extract an ``IRGraph`` from a live ``GraphModule``.

    Returns:
        A tuple of ``(ir_graph, real_buffers)`` where *real_buffers* is a dict of
        non-meta buffer tensors that must be persisted via ``torch.save``.
    """
    ir_nodes: List[IRNode] = []

    for node in gm.graph.nodes:
        target_str: str
        if node.op in ("placeholder", "output"):
            target_str = str(node.target)
        elif node.op in ("get_attr", "call_module"):
            target_str = str(node.target)
        elif node.op == "call_function":
            target_str = serialize_target(node.target)
        elif node.op == "call_method":
            target_str = str(node.target)
        else:
            raise ValueError(f"Unknown FX node op: {node.op!r}")

        ir_node = IRNode(
            name=node.name,
            op=node.op,
            target=target_str,
            args=serialize_arg(node.args),
            kwargs={k: serialize_arg(v) for k, v in node.kwargs.items()},
        )

        if node.op == "placeholder":
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                ir_node.val_type = "tensor"
                ir_node.shape = [_concretize(s) for s in val.shape]
                ir_node.dtype = _dtype_to_str(val.dtype)
            elif val is None:
                ir_node.val_type = "none"
            elif isinstance(val, (int, float, torch.SymInt, torch.SymFloat)):
                ir_node.val_type = "scalar"
                ir_node.shape = None
                ir_node.dtype = str(type(_concretize(val)).__name__)

        ir_nodes.append(ir_node)

    module_tree: Dict[str, str] = {}
    for name, mod in gm.named_modules():
        if name:
            module_tree[name] = type(mod).__qualname__

    params: Dict[str, ParamSpec] = {}
    for name, p in gm.named_parameters():
        params[name] = ParamSpec(
            shape=[_concretize(s) for s in p.shape],
            dtype=_dtype_to_str(p.dtype),
            requires_grad=p.requires_grad,
        )

    buffer_specs: Dict[str, BufferSpec] = {}
    real_buffers: Dict[str, torch.Tensor] = {}
    for name, b in gm.named_buffers():
        is_meta = b.device.type == "meta"
        buffer_specs[name] = BufferSpec(
            shape=[_concretize(s) for s in b.shape],
            dtype=_dtype_to_str(b.dtype),
            is_meta=is_meta,
        )
        if not is_meta:
            real_buffers[name] = b.detach().cpu()

    in_spec = _serialize_treespec(getattr(gm, "_in_spec", None))
    out_spec = _serialize_treespec(getattr(gm, "_out_spec", None))

    orig_args: Optional[List[str]] = None
    codegen = getattr(gm.graph, "_codegen", None)
    if codegen is not None and hasattr(codegen, "pytree_info"):
        orig_args = list(codegen.pytree_info.orig_args)

    ad_meta: Dict[str, Any] = {}
    if autodeploy_meta is not None:
        ad_meta = autodeploy_meta
    elif hasattr(gm, "meta"):
        ad_meta = gm.meta.get("_autodeploy", {})

    gm_attrs = _extract_pydantic_attrs(gm)

    ir = IRGraph(
        format_version=AD_IR_FORMAT_VERSION,
        nodes=ir_nodes,
        module_tree=module_tree,
        params=params,
        buffers=buffer_specs,
        in_spec=in_spec,
        out_spec=out_spec,
        orig_args=orig_args,
        hook_specs=hook_specs or [],
        autodeploy_meta=ad_meta,
        source_model_hooks_required=source_model_hooks_required,
        gm_attrs=gm_attrs,
    )
    return ir, real_buffers


# ===================================================================
# Build helpers
# ===================================================================


def _ensure_submodule(root: nn.Module, path: str) -> None:
    """Ensure *path* exists as a submodule chain on *root*."""
    parts = path.split(".")
    parent = root
    for part in parts:
        if not hasattr(parent, part):
            setattr(parent, part, nn.Module())
        parent = getattr(parent, part)


def _resolve_args_tuple(args: Any, node_map: Dict[str, Node]) -> Tuple:
    """Resolve serialized args, always returning a tuple for the Graph API."""
    resolved = resolve_arg(args, node_map)
    if isinstance(resolved, list):
        return tuple(resolved)
    if isinstance(resolved, tuple):
        return resolved
    return (resolved,)


def _resolve_kwargs(kwargs: Any, node_map: Dict[str, Node]) -> Dict[str, Any]:
    """Resolve serialized kwargs."""
    if not isinstance(kwargs, dict):
        return {}
    return {k: resolve_arg(v, node_map) for k, v in kwargs.items()}


# ===================================================================
# Build: IRGraph -> GraphModule  (NO re-tracing)
# ===================================================================


def build_graph_module(
    ir: IRGraph,
    real_buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> GraphModule:
    """Reconstruct a ``GraphModule`` from an ``IRGraph``.

    The ``torch.fx.Graph`` is built node-by-node using the graph construction API
    (``graph.placeholder``, ``graph.call_function``, etc.) — **no re-tracing**.
    """
    if real_buffers is None:
        real_buffers = {}

    # --- 1. Build nn.Module skeleton ---
    root = nn.Module()
    for path in sorted(ir.module_tree.keys()):
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        if not hasattr(parent, parts[-1]):
            setattr(parent, parts[-1], nn.Module())

    # --- 2. Register parameters (meta device — weight_load fills later) ---
    for name, spec in ir.params.items():
        parts = name.rsplit(".", 1)
        mod = root.get_submodule(parts[0]) if len(parts) == 2 else root
        pname = parts[-1]
        param = nn.Parameter(
            torch.empty(spec.shape, dtype=_str_to_dtype(spec.dtype), device="meta"),
            requires_grad=spec.requires_grad,
        )
        mod.register_parameter(pname, param)

    # --- 3. Register buffers ---
    for name, spec in ir.buffers.items():
        parts = name.rsplit(".", 1)
        mod = root.get_submodule(parts[0]) if len(parts) == 2 else root
        bname = parts[-1]
        if not spec.is_meta and name in real_buffers:
            buf = real_buffers[name]
        else:
            buf = torch.empty(spec.shape, dtype=_str_to_dtype(spec.dtype), device="meta")
        mod.register_buffer(bname, buf)

    # --- 4. Build torch.fx.Graph node-by-node ---
    graph = Graph()
    node_map: Dict[str, Node] = {}

    # Pre-register all IR node names in the graph namespace so that auto-generated
    # names from helper methods (graph.call_function, etc.) never collide with names
    # we intend to assign.  Without this, a call_function for operator.getitem might
    # auto-generate "getitem_215" before we create the IR node that is supposed to
    # own that name, leading to "Node redefined name" errors in graph.lint().
    for ir_node in ir.nodes:
        graph._graph_namespace.create_name(ir_node.name, None)

    ns = graph._graph_namespace

    for ir_node in ir.nodes:
        node: Node
        # The pre-registration above added ir_node.name to _used_names.
        # Temporarily remove it so create_node → create_name returns the
        # exact name instead of appending a "_1" suffix.
        ns._used_names.discard(ir_node.name)

        if ir_node.op == "placeholder":
            node = graph.create_node("placeholder", ir_node.name, name=ir_node.name)

        elif ir_node.op == "get_attr":
            node = graph.create_node("get_attr", ir_node.target, name=ir_node.name)

        elif ir_node.op == "call_function":
            target = resolve_target(ir_node.target)
            args = _resolve_args_tuple(ir_node.args, node_map)
            kwargs = _resolve_kwargs(ir_node.kwargs, node_map)
            node = graph.create_node("call_function", target, args, kwargs, name=ir_node.name)

        elif ir_node.op == "call_module":
            args = _resolve_args_tuple(ir_node.args, node_map)
            kwargs = _resolve_kwargs(ir_node.kwargs, node_map)
            node = graph.create_node("call_module", ir_node.target, args, kwargs, name=ir_node.name)

        elif ir_node.op == "call_method":
            args = _resolve_args_tuple(ir_node.args, node_map)
            kwargs = _resolve_kwargs(ir_node.kwargs, node_map)
            node = graph.create_node("call_method", ir_node.target, args, kwargs, name=ir_node.name)

        elif ir_node.op == "output":
            full_args = resolve_arg(ir_node.args, node_map)
            if isinstance(full_args, (list, tuple)) and len(full_args) == 1:
                full_args = full_args[0]
            node = graph.create_node("output", "output", (full_args,), name=ir_node.name)

        else:
            raise ValueError(f"Unknown IR node op: {ir_node.op!r}")

        node_map[ir_node.name] = node

    # --- 5. Set up _PyTreeCodeGen if we have pytree metadata ---
    in_spec_obj = _deserialize_treespec(ir.in_spec)
    out_spec_obj = _deserialize_treespec(ir.out_spec)
    if ir.orig_args is not None and in_spec_obj is not None:
        from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

        pytree_info = _PyTreeInfo(
            orig_args=ir.orig_args,
            in_spec=in_spec_obj,
            out_spec=out_spec_obj,
        )
        graph._codegen = _PyTreeCodeGen(pytree_info)

    # --- 6. Create GraphModule ---
    gm = GraphModule(root, graph)

    # GraphModule.__init__ only copies attributes referenced by graph nodes.
    # Re-register any parameters/buffers that exist in the IR but were not
    # referenced via get_attr/call_module nodes (e.g. standalone buffers).
    existing_params = {n for n, _ in gm.named_parameters()}
    for name, spec in ir.params.items():
        if name not in existing_params:
            parts = name.rsplit(".", 1)
            target_mod = gm
            if len(parts) == 2:
                _ensure_submodule(gm, parts[0])
                target_mod = gm.get_submodule(parts[0])
            param = nn.Parameter(
                torch.empty(spec.shape, dtype=_str_to_dtype(spec.dtype), device="meta"),
                requires_grad=spec.requires_grad,
            )
            target_mod.register_parameter(parts[-1], param)

    existing_buffers = {n for n, _ in gm.named_buffers()}
    for name, spec in ir.buffers.items():
        if name not in existing_buffers:
            parts = name.rsplit(".", 1)
            target_mod = gm
            if len(parts) == 2:
                _ensure_submodule(gm, parts[0])
                target_mod = gm.get_submodule(parts[0])
            if not spec.is_meta and name in real_buffers:
                buf = real_buffers[name]
            else:
                buf = torch.empty(spec.shape, dtype=_str_to_dtype(spec.dtype), device="meta")
            target_mod.register_buffer(parts[-1], buf)

    # --- 7. Restore _in_spec / _out_spec ---
    if in_spec_obj is not None:
        gm._in_spec = in_spec_obj
    if out_spec_obj is not None:
        gm._out_spec = out_spec_obj

    # --- 8. Restore Pydantic model attributes ---
    if ir.gm_attrs:
        _restore_pydantic_attrs(gm, ir.gm_attrs)

    return gm


# ===================================================================
# Hydrate: populate FakeTensor metadata via FakeTensorProp
# ===================================================================


def hydrate_shapes(gm: GraphModule, ir: IRGraph) -> None:
    """Populate ``node.meta["val"]`` on all graph nodes via ``FakeTensorProp``.

    Placeholder FakeTensors are created from the shape / dtype stored in the IR.
    ``FakeTensorProp`` then forward-propagates to derive shapes for every node.
    """
    from torch._subclasses import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    from ..utils._graph import enable_python_dispatcher, placeholders_on_meta

    ir_node_map = {n.name: n for n in ir.nodes}

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    placeholder_vals: List[Any] = []

    for fx_node in gm.graph.nodes:
        if fx_node.op != "placeholder":
            continue
        ir_node = ir_node_map.get(fx_node.name)
        if ir_node is None or ir_node.val_type is None:
            placeholder_vals.append(
                fake_mode.from_tensor(torch.empty(0, device="meta"), static_shapes=True)
            )
            continue

        if ir_node.val_type == "tensor":
            meta_t = torch.empty(ir_node.shape, dtype=_str_to_dtype(ir_node.dtype), device="meta")
            fake_t = fake_mode.from_tensor(meta_t, static_shapes=True)
            fx_node.meta["val"] = fake_t
            placeholder_vals.append(fake_t)
        elif ir_node.val_type == "none":
            fx_node.meta["val"] = None
            placeholder_vals.append(None)
        elif ir_node.val_type == "scalar":
            val: Union[int, float] = 0
            if ir_node.dtype == "int":
                val = 0
            elif ir_node.dtype == "float":
                val = 0.0
            fx_node.meta["val"] = val
            placeholder_vals.append(val)
        else:
            placeholder_vals.append(
                fake_mode.from_tensor(torch.empty(0, device="meta"), static_shapes=True)
            )

    try:
        with enable_python_dispatcher():
            if placeholders_on_meta(gm):
                from ..utils._graph import lift_to_meta

                with lift_to_meta(gm):
                    FakeTensorProp(gm, fake_mode).propagate(*placeholder_vals)
            else:
                FakeTensorProp(gm, fake_mode).propagate(*placeholder_vals)
        ad_logger.debug("AD IR: hydrated shape metadata for all graph nodes")
    except Exception as exc:  # noqa: BLE001
        ad_logger.warning(f"AD IR: FakeTensorProp failed during shape hydration: {exc}")


# ===================================================================
# File I/O helpers
# ===================================================================


def save_ir(ir: IRGraph, real_buffers: Dict[str, torch.Tensor], rank_dir: Path) -> None:
    """Write an ``IRGraph`` and its real buffers to *rank_dir*."""
    rank_dir.mkdir(parents=True, exist_ok=True)
    ir_path = rank_dir / "ad_ir.json"
    ir_path.write_text(
        json.dumps(ir.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if real_buffers:
        torch.save(real_buffers, rank_dir / "real_buffers.pt")


def load_ir(rank_dir: Path) -> Optional[Tuple[IRGraph, Dict[str, torch.Tensor]]]:
    """Read an ``IRGraph`` and its real buffers from *rank_dir*.

    Returns ``None`` if the IR file does not exist.
    """
    ir_path = rank_dir / "ad_ir.json"
    if not ir_path.exists():
        return None

    data = json.loads(ir_path.read_text(encoding="utf-8"))
    ir = IRGraph.from_dict(data)

    real_buffers: Dict[str, torch.Tensor] = {}
    buf_path = rank_dir / "real_buffers.pt"
    if buf_path.exists():
        try:
            real_buffers = torch.load(buf_path, weights_only=True)
        except TypeError:
            real_buffers = torch.load(buf_path)

    return ir, real_buffers

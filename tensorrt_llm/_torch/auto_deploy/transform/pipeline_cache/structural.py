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

"""FX GraphModule save/load helpers for the AutoDeploy pipeline cache."""

import importlib
import inspect

# Used only to probe torch.save-compatible cache payloads.
import pickle  # nosec B403
import types
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.graph import Graph, _PyTreeCodeGen
from torch.fx.graph_module import _CodeOnlyModule
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.node import Node

from ...utils._graph import named_graphmodules
from ...utils.dist_config import DistConfig
from ...utils.node_utils import invalidate_weight_node_cache

_GRAPHMODULE_PAYLOAD_KIND = "graphmodule_snapshot"
_MODULE_TREE_PAYLOAD_KIND = "module_tree"


# ShardingTransformContainer carries live distributed objects after construction. The cache only
# needs the serializable config, so rebuild that config into a pre-runtime form before torch.save.
def _cacheable_sharding_transform_container(value: Any) -> Any:
    from ..library.sharding import ShardingTransformContainer

    if not isinstance(value, ShardingTransformContainer):
        return value

    config = value.config.model_copy(deep=True)
    config.mapping = None
    config.dist_config = DistConfig.from_dict(config.dist_config.to_dict())
    return ShardingTransformContainer(config=config)


def _sanitized_graphmodule_body(gm: GraphModule) -> dict[str, Any]:
    # The body is the non-graph part of GraphModule.__dict__. It is saved with torch.save after
    # removing fields that would pull the live FX graph or runtime-only objects back into pickle.
    body: dict[str, Any] = {}
    for key, value in gm.__dict__.items():
        if key == "_graph":
            continue
        sanitized = _cacheable_sharding_transform_container(value)
        if sanitized is not gm and getattr(sanitized, "__self__", None) is not gm:
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


# FX node targets can include torch custom ops. Those are stable by namespace/op/overload, while
# Python callables are only accepted when importing module.qualname resolves to the same object.
def _encode_graph_target_ref(target: Any) -> dict[str, str]:
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


def _encode_graph_target(target: Any) -> dict[str, Any]:
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


# GraphModule can keep bound methods in __dict__ for export/runtime behavior. Serialize those as
# function refs and rebind them after construction to avoid direct self-references in torch.save.
def _graphmodule_bound_method_specs(gm: GraphModule) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, value in gm.__dict__.items():
        if not inspect.ismethod(value) or getattr(value, "__self__", None) is not gm:
            continue
        if _is_exported_program_train_eval_method(name, value):
            continue
        function_ref = _encode_graph_target_ref(value.__func__)
        specs.append({"name": name, "function_ref": function_ref})
    return specs


def _is_exported_program_train_eval_method(name: str, value: Any) -> bool:
    if name not in ("train", "eval"):
        return False
    func = getattr(value, "__func__", None)
    return getattr(func, "__module__", None) == "torch.export.exported_program" and getattr(
        func, "__qualname__", ""
    ).startswith("ExportedProgram.module.<locals>._")


def _restore_graphmodule_bound_methods(
    module: GraphModule, specs: Sequence[Mapping[str, Any]]
) -> None:
    for spec in specs:
        name = spec.get("name")
        function_ref = spec.get("function_ref")
        if not isinstance(name, str) or not isinstance(function_ref, Mapping):
            raise ValueError(f"pipeline cache module payload has invalid bound method: {spec!r}")
        setattr(module, name, types.MethodType(_decode_graph_target_ref(function_ref), module))


# Node metadata often contains live Nodes, modules, or real tensors. Keep durable pickleable
# metadata and rebuild placeholder/get_attr "val" entries as fake tensors from shape/dtype.
def _meta_value_to_spec(value: Any) -> dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "kind": "tensor",
            "shape": _concrete_shape(shape),
            "dtype": dtype,
        }
    return None


def _meta_value_from_spec(spec: Mapping[str, Any], fake_mode: FakeTensorMode) -> Any:
    if spec.get("kind") == "tensor":
        tensor = torch.empty(tuple(spec["shape"]), dtype=spec["dtype"], device="meta")
        return fake_mode.from_tensor(tensor, static_shapes=True)
    raise ValueError(f"Pipeline cache: unknown placeholder meta value spec {spec!r}.")


def _concrete_shape(shape: Any) -> tuple[int, ...]:
    def concrete_dim(dim: Any) -> int:
        try:
            return int(dim)
        except (TypeError, ValueError):
            node = getattr(dim, "node", None)
            hint = getattr(node, "hint", None)
            if hint is not None:
                return int(hint)
        raise ValueError(f"Pipeline cache: cannot serialize symbolic shape dimension {dim!r}.")

    concrete_dims: list[int] = []
    for dim in shape:
        concrete_dims.append(concrete_dim(dim))
    return tuple(concrete_dims)


def _contains_non_durable_meta_ref(value: Any) -> bool:
    if isinstance(value, (Node, torch.Tensor, nn.Module)):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_non_durable_meta_ref(key) or _contains_non_durable_meta_ref(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset, immutable_list)):
        return any(_contains_non_durable_meta_ref(item) for item in value)
    return False


def _sanitize_node_meta_for_pickling(
    node: torch.fx.Node,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sanitized: dict[str, Any] = {}
    placeholder_meta_spec = None
    for key, value in node.meta.items():
        if key == "val":
            if node.op in ("placeholder", "get_attr"):
                placeholder_meta_spec = _meta_value_to_spec(value)
            continue
        if not _contains_non_durable_meta_ref(value) and _is_pickleable(value):
            sanitized[key] = value
    return sanitized, placeholder_meta_spec


def _restore_placeholder_meta_values(
    graph: Graph, placeholder_meta_specs: Mapping[str, Mapping[str, Any]]
) -> None:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    for node in graph.nodes:
        spec = placeholder_meta_specs.get(node.name)
        if spec is not None:
            node.meta["val"] = _meta_value_from_spec(spec, fake_mode)


# Args/kwargs are structurally encoded so Node references survive graph reconstruction by name
# instead of by pickle identity. This is the main reason the cache owns graph_state.
def _encode_graph_arg(value: Any) -> dict[str, Any]:
    if isinstance(value, Node):
        return {"kind": "node", "name": value.name}
    if isinstance(value, torch.Size):
        return {"kind": "torch_size", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, immutable_list):
        return {"kind": "immutable_list", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, immutable_dict):
        return {
            "kind": "immutable_dict",
            "items": [(key, _encode_graph_arg(item)) for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_graph_arg(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_graph_arg(item) for item in value]}
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
        "Pipeline cache: graph argument is not pickleable: "
        f"{value!r} ({type(value).__module__}.{type(value).__qualname__})."
    )


def _decode_graph_arg(spec: Mapping[str, Any], nodes_by_name: Mapping[str, Node]) -> Any:
    kind = spec.get("kind")
    if kind == "node":
        return nodes_by_name[spec["name"]]
    if kind == "torch_size":
        return torch.Size(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "immutable_list":
        return immutable_list(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "immutable_dict":
        return immutable_dict(
            [(key, _decode_graph_arg(item, nodes_by_name)) for key, item in spec["items"]]
        )
    if kind == "tuple":
        return tuple(_decode_graph_arg(item, nodes_by_name) for item in spec["items"])
    if kind == "list":
        return [_decode_graph_arg(item, nodes_by_name) for item in spec["items"]]
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


def _graph_codegen_to_state(graph: Graph) -> dict[str, Any] | None:
    # torch.export graphs carry PyTreeCodeGen state. Without it, restored modules lose their
    # original input/output flattening behavior even though the node list itself is correct.
    codegen = getattr(graph, "_codegen", None)
    pytree_info = getattr(codegen, "pytree_info", None)
    if pytree_info is not None and _is_pickleable(pytree_info):
        return {
            "kind": "pytree",
            "pytree_info": pytree_info,
        }
    return None


def _graph_codegen_from_state(state: Mapping[str, Any] | None) -> Any:
    if state is None:
        return None
    if state.get("kind") == "pytree":
        return _PyTreeCodeGen(state["pytree_info"])
    return None


def _graph_to_state(graph: Graph) -> dict[str, Any]:
    # Raw torch.fx.Graph pickling captures private lookup tables and object identities that are
    # brittle across runs. graph_state stores only the ordered nodes plus structural references.
    nodes = []
    placeholder_meta_specs = {}
    for node in graph.nodes:
        meta, placeholder_meta_spec = _sanitize_node_meta_for_pickling(node)
        if placeholder_meta_spec is not None:
            placeholder_meta_specs[node.name] = placeholder_meta_spec
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": _encode_graph_target(node.target),
                "args": _encode_graph_arg(node.args),
                "kwargs": _encode_graph_arg(node.kwargs),
                "type": node.type if _is_pickleable(node.type) else None,
                "meta": meta,
            }
        )
    state = {
        "nodes": nodes,
    }
    codegen_state = _graph_codegen_to_state(graph)
    if codegen_state is not None:
        state["codegen"] = codegen_state
    if placeholder_meta_specs:
        state["placeholder_meta"] = placeholder_meta_specs
    return state


def _graph_from_state(state: Mapping[str, Any]) -> Graph:
    graph = Graph()
    nodes_by_name: dict[str, Node] = {}
    node_states = state.get("nodes")
    if not isinstance(node_states, Sequence):
        raise ValueError("pipeline cache graph payload is missing nodes.")
    for node_state in node_states:
        if not isinstance(node_state, Mapping):
            raise ValueError(f"pipeline cache graph payload has invalid node: {node_state!r}")
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

    codegen = _graph_codegen_from_state(state.get("codegen"))
    if codegen is not None:
        graph._codegen = codegen
    placeholder_meta_specs = state.get("placeholder_meta", {})
    if not isinstance(placeholder_meta_specs, Mapping):
        raise ValueError("pipeline cache graph payload has invalid placeholder metadata.")
    _restore_placeholder_meta_values(graph, placeholder_meta_specs)
    return graph


def _mark_cached_shape_metadata_invalid(module: nn.Module) -> None:
    # Weight nodes are rebuilt after cache load, so shape-prop history must not claim that cached
    # weight-dependent shapes are still valid for later transforms.
    for _, graph_module in named_graphmodules(module):
        invalidate_weight_node_cache(graph_module)
        autodeploy_meta = graph_module.meta.get("_autodeploy", {})
        history = autodeploy_meta.get("transform_history", {})
        for key, info in list(history.items()):
            history[key] = info.model_copy(update={"has_valid_shapes": False})


class _GraphModulePlaceholder(nn.Module):
    """Temporary stand-in while torch.save serializes an nn.Module wrapper."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name


def _graphmodule_to_structural_payload(gm: GraphModule) -> dict[str, Any]:
    body = _sanitized_graphmodule_body(gm)
    return {
        "type": _GRAPHMODULE_PAYLOAD_KIND,
        "class_name": body.get("_graphmodule_cls_name", type(gm).__name__),
        "body": body,
        "bound_methods": _graphmodule_bound_method_specs(gm),
        "graph_state": _graph_to_state(gm.graph),
    }


def _graphmodule_from_structural_payload(payload: Mapping[str, Any]) -> GraphModule:
    graph_state = payload.get("graph_state")
    body = payload.get("body")
    if not isinstance(graph_state, Mapping) or not isinstance(body, dict):
        raise ValueError("pipeline cache module payload is missing GraphModule state.")
    graph = _graph_from_state(graph_state)
    root = _CodeOnlyModule(body)
    module = GraphModule(root, graph, class_name=payload["class_name"])
    for key, value in body.items():
        if key == "_graph":
            continue
        module.__dict__[key] = value
    module.recompile()
    bound_methods = payload.get("bound_methods", [])
    if not isinstance(bound_methods, Sequence):
        raise ValueError("pipeline cache GraphModule payload has invalid bound methods.")
    _restore_graphmodule_bound_methods(module, bound_methods)
    return module


def _replace_submodule(module: nn.Module, target: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = target.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    parent._modules[child_name] = replacement


def _named_graphmodule_roots(module: nn.Module) -> list[tuple[str, GraphModule]]:
    roots: list[tuple[str, GraphModule]] = []
    for name, graph_module in named_graphmodules(module):
        if any(parent == "" or name.startswith(f"{parent}.") for parent, _ in roots):
            continue
        roots.append((name, graph_module))
    return roots


def save_module_structural(module: nn.Module, module_file: Any) -> None:
    if isinstance(module, GraphModule):
        torch.save(_graphmodule_to_structural_payload(module), module_file)
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
        # Qwen-style wrappers need the root module pickled, but their GraphModule children still
        # use the structural graph_state path. Replace children only for the duration of save.
        for name, _ in graphmodules:
            _replace_submodule(module, name, _GraphModulePlaceholder(name))
        payload = {
            "type": _MODULE_TREE_PAYLOAD_KIND,
            "module": module,
            "graphmodules": graphmodule_payloads,
        }
        torch.save(payload, module_file)
    finally:
        for name, graph_module in graphmodules:
            _replace_submodule(module, name, graph_module)


def load_module_structural(module_file: Any) -> nn.Module:
    payload = torch.load(module_file, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and payload.get("type") == _GRAPHMODULE_PAYLOAD_KIND:
        module = _graphmodule_from_structural_payload(payload)
        _mark_cached_shape_metadata_invalid(module)
        return module
    if not isinstance(payload, Mapping) or payload.get("type") != _MODULE_TREE_PAYLOAD_KIND:
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


def validate_pre_weight_snapshot(model: nn.Module) -> None:
    materialized_params = [
        name for name, param in model.named_parameters() if param.device.type != "meta"
    ]
    if materialized_params:
        raise ValueError(
            "pipeline_cache only supports pre-weight-loading snapshots; materialized "
            f"parameters found: {materialized_params[:5]}"
        )

"""Utility to extract and dump dtype metadata from exported GraphModule.

This module provides functions to extract dtype information (input, output, weight)
for each operation in an exported FX graph. This is useful for comparing precision
profiles between HuggingFace models and AutoDeploy exported graphs.

Usage:
    from tensorrt_llm._torch.auto_deploy.utils.dtype_metadata import dump_dtype_metadata

    # After torch.export
    gm = torch_export_to_gm(model, args, kwargs, ...)
    dump_dtype_metadata(gm, "dtype_metadata.json", layer_filter="layers.0")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.fx import GraphModule, Node


def _get_dtype_str(dtype: Optional[torch.dtype]) -> str:
    """Convert torch dtype to string representation."""
    if dtype is None:
        return "None"
    return str(dtype).replace("torch.", "")


def _extract_dtype_from_meta(node: Node) -> Optional[str]:
    """Extract dtype from node's meta['val']."""
    meta = getattr(node, "meta", {})
    val = meta.get("val")

    if val is None:
        return None

    if isinstance(val, torch.Tensor):
        return _get_dtype_str(val.dtype)

    # Handle FakeTensor or similar
    if hasattr(val, "dtype"):
        return _get_dtype_str(val.dtype)

    # Handle tuple/list of tensors - return first tensor's dtype
    if isinstance(val, (tuple, list)) and len(val) > 0:
        first = val[0]
        if hasattr(first, "dtype"):
            return _get_dtype_str(first.dtype)

    return None


def _extract_input_dtypes(node: Node) -> List[str]:
    """Extract dtypes of all tensor inputs to a node."""
    input_dtypes = []

    for arg in node.args:
        if isinstance(arg, Node):
            dtype = _extract_dtype_from_meta(arg)
            if dtype:
                input_dtypes.append(dtype)
        elif isinstance(arg, (tuple, list)):
            for item in arg:
                if isinstance(item, Node):
                    dtype = _extract_dtype_from_meta(item)
                    if dtype:
                        input_dtypes.append(dtype)

    return input_dtypes


def _get_module_path_from_stack(node: Node) -> str:
    """Extract simplified module path from nn_module_stack."""
    meta = getattr(node, "meta", {})
    nn_module_stack = meta.get("nn_module_stack", {})

    if not nn_module_stack:
        return ""

    if isinstance(nn_module_stack, dict):
        # Get the deepest (last) module path
        paths = list(nn_module_stack.keys())
        if paths:
            path = paths[-1]
            # Remove common prefixes
            for prefix in ["L__self__model.", "L__self__"]:
                if path.startswith(prefix):
                    path = path[len(prefix) :]
                    break
            return path

    return str(nn_module_stack)


def _find_weight_param_for_node(
    gm: GraphModule,
    node: Node,
    module_path: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Find weight parameter dtype for a node if applicable.

    Returns:
        Tuple of (param_name, dtype_str) or (None, None) if not found.
    """
    # Check if this is a linear-like op by looking for get_attr in args
    for arg in node.args:
        if isinstance(arg, Node) and arg.op == "get_attr":
            # Found a get_attr node - this is likely a weight
            param_name = arg.target
            try:
                param = gm.get_parameter(param_name)
                return param_name, _get_dtype_str(param.dtype)
            except AttributeError:
                # Try as buffer
                try:
                    param = gm.get_buffer(param_name)
                    return param_name, _get_dtype_str(param.dtype)
                except AttributeError:
                    pass

            # Try to get dtype from the get_attr node's meta
            dtype = _extract_dtype_from_meta(arg)
            if dtype:
                return param_name, dtype

    return None, None


def extract_dtype_metadata(
    gm: GraphModule,
    layer_filter: Optional[str] = "layers.0",
) -> Dict[str, Any]:
    """Extract dtype info for each op in the graph.

    Args:
        gm: The exported GraphModule
        layer_filter: Optional filter to include only nodes from specific layer(s).
                     Set to None to include all nodes.

    Returns:
        Dict with per-node dtype info:
        {
            "nodes": {
                "node_name": {
                    "op": "call_function",
                    "target": "aten.linear.default",
                    "module_path": "layers.0.self_attn.q_proj",
                    "input_dtypes": ["bfloat16", "bfloat16"],
                    "output_dtype": "bfloat16",
                    "weight_param": "p_layers_0_self_attn_q_proj_weight",
                    "weight_dtype": "bfloat16",
                    "order": 42,
                },
                ...
            },
            "parameters": {
                "p_layers_0_self_attn_q_proj_weight": {
                    "dtype": "bfloat16",
                    "shape": [4096, 4096],
                    "module_hint": "layers.0.self_attn.q_proj",
                },
                ...
            },
            "layer_filter": "layers.0",
            "total_nodes": 150,
            "filtered_nodes": 120,
        }
    """
    # Unwrap if needed
    actual_gm = gm
    if not isinstance(actual_gm, GraphModule):
        if hasattr(gm, "model") and isinstance(gm.model, GraphModule):
            actual_gm = gm.model
        elif hasattr(gm, "graph"):
            actual_gm = gm
        else:
            return {"error": f"Cannot extract GraphModule from {type(gm).__name__}"}

    nodes_info: Dict[str, Dict[str, Any]] = {}
    total_nodes = 0
    filtered_nodes = 0

    for order, node in enumerate(actual_gm.graph.nodes):
        total_nodes += 1

        # Skip placeholder and output nodes for the main comparison
        if node.op in ("placeholder", "output", "get_attr"):
            continue

        module_path = _get_module_path_from_stack(node)

        # Apply layer filter
        if layer_filter and layer_filter not in module_path:
            continue

        filtered_nodes += 1

        # Extract target string
        target_str = (
            str(node.target)
            if not callable(node.target)
            else (getattr(node.target, "__name__", str(node.target)))
        )

        # Extract dtypes
        input_dtypes = _extract_input_dtypes(node)
        output_dtype = _extract_dtype_from_meta(node)
        weight_param, weight_dtype = _find_weight_param_for_node(actual_gm, node, module_path)

        node_info = {
            "op": node.op,
            "target": target_str,
            "module_path": module_path,
            "input_dtypes": input_dtypes,
            "output_dtype": output_dtype,
            "order": order,
        }

        if weight_param:
            node_info["weight_param"] = weight_param
            node_info["weight_dtype"] = weight_dtype

        nodes_info[node.name] = node_info

    # Extract all parameters with dtypes
    parameters_info: Dict[str, Dict[str, Any]] = {}
    for name, param in actual_gm.named_parameters():
        # Try to infer module hint from parameter name
        module_hint = name.replace("p_", "").replace("_weight", "").replace("_bias", "")
        module_hint = module_hint.replace("_", ".")

        # Apply layer filter to parameters too
        if layer_filter and layer_filter not in module_hint:
            continue

        parameters_info[name] = {
            "dtype": _get_dtype_str(param.dtype),
            "shape": list(param.shape),
            "module_hint": module_hint,
        }

    return {
        "nodes": nodes_info,
        "parameters": parameters_info,
        "layer_filter": layer_filter,
        "total_nodes": total_nodes,
        "filtered_nodes": filtered_nodes,
    }


def dump_dtype_metadata(
    gm: GraphModule,
    output_path: Union[str, Path],
    layer_filter: Optional[str] = "layers.0",
) -> Dict[str, Any]:
    """Extract and save dtype metadata to JSON.

    Args:
        gm: The exported GraphModule
        output_path: Path to save the JSON file
        layer_filter: Optional filter for layer (e.g., "layers.0")

    Returns:
        The extracted metadata dict
    """
    metadata = extract_dtype_metadata(gm, layer_filter=layer_filter)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[dtype_metadata] Saved dtype metadata to {output_path}")
    print(
        f"[dtype_metadata] Total nodes: {metadata.get('total_nodes', 0)}, "
        f"Filtered nodes: {metadata.get('filtered_nodes', 0)}, "
        f"Parameters: {len(metadata.get('parameters', {}))}"
    )

    return metadata


def print_dtype_summary(metadata: Dict[str, Any]) -> None:
    """Print a summary of dtype metadata to console."""
    nodes = metadata.get("nodes", {})
    parameters = metadata.get("parameters", {})

    print("\n" + "=" * 70)
    print("AD Dtype Metadata Summary")
    print("=" * 70)

    # Group nodes by module path
    modules: Dict[str, List[str]] = {}
    for node_name, info in nodes.items():
        module_path = info.get("module_path", "unknown")
        if module_path not in modules:
            modules[module_path] = []
        modules[module_path].append(node_name)

    # Print by module
    for module_path in sorted(modules.keys()):
        node_names = modules[module_path]
        print(f"\n[{module_path}] ({len(node_names)} ops)")

        for node_name in node_names[:5]:  # Show first 5 ops per module
            info = nodes[node_name]
            in_dtypes = ", ".join(info.get("input_dtypes", []))
            out_dtype = info.get("output_dtype", "?")
            weight_dtype = info.get("weight_dtype", "")
            target = info.get("target", "?")

            weight_str = f" W={weight_dtype}" if weight_dtype else ""
            print(f"  {target}: [{in_dtypes}] -> {out_dtype}{weight_str}")

        if len(node_names) > 5:
            print(f"  ... and {len(node_names) - 5} more ops")

    print("\n" + "=" * 70)
    print(f"Total: {len(nodes)} ops, {len(parameters)} parameters")
    print("=" * 70)

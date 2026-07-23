"""Utilities for graph debugging and module boundary extraction.

This module provides functions for:
- Loading dumped GraphModules and metadata
- Extracting module boundaries from metadata
- Finding boundary nodes for comparison
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
import torch
from safetensors.torch import load_file as safetensors_load
from torch.fx import GraphModule


def load_debug_artifacts(
    debug_dir: Path,
    stage: str = "final",
) -> Tuple[Optional[GraphModule], Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """Load dumped GraphModule, metadata, and inputs from disk.

    Args:
        debug_dir: Base directory containing debug dumps
        stage: Stage name (e.g., "post_export", "final")

    Returns:
        Tuple of (GraphModule, metadata dict, inputs dict)
    """
    stage_dir = debug_dir / stage

    # Load GraphModule
    gm_path = stage_dir / "gm.pt"
    gm = None
    if gm_path.exists():
        try:
            gm = torch.load(gm_path, map_location="cpu", pickle_module=dill)
        except Exception as e:
            print(f"Warning: Failed to load GraphModule from {gm_path}: {e}")

    # Load metadata
    metadata_path = stage_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata from {metadata_path}: {e}")

    # Load inputs
    inputs_path = stage_dir / "inputs.safetensors"
    inputs = None
    if inputs_path.exists():
        try:
            inputs = safetensors_load(str(inputs_path))
        except Exception as e:
            print(f"Warning: Failed to load inputs from {inputs_path}: {e}")

    return gm, metadata, inputs


def get_module_boundary_nodes(
    metadata: Dict[str, Any],
    module_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Get boundary nodes (first and last) for each module.

    Args:
        metadata: Metadata dict from extract_graph_metadata or loaded from JSON
        module_filter: Optional substring to filter modules (e.g., "self_attn")

    Returns:
        Dict mapping simplified module names to their boundary info:
        {
            "embed_tokens": {"first_node": "...", "last_node": "...", "nodes": [...]},
            "self_attn": {"first_node": "...", "last_node": "...", "nodes": [...]},
            ...
        }
    """
    module_boundaries = metadata.get("module_boundaries", {})

    # Simplify module paths for easier access
    simplified = {}
    for module_path, info in module_boundaries.items():
        # Extract simplified name from path like "L__self__model.layers.0.self_attn"
        simplified_name = _simplify_module_path(module_path)

        if module_filter and module_filter not in simplified_name:
            continue

        simplified[simplified_name] = info

    return simplified


def _simplify_module_path(module_path: str) -> str:
    """Simplify module path for easier comparison.

    Examples:
        "L__self__model.embed_tokens" -> "embed_tokens"
        "L__self__model.layers.0.self_attn" -> "layers.0.self_attn"
        "L__self__model.layers.0.self_attn.q_proj" -> "layers.0.self_attn.q_proj"
    """
    # Remove common prefixes
    prefixes = ["L__self__model.", "L__self__"]
    for prefix in prefixes:
        if module_path.startswith(prefix):
            module_path = module_path[len(prefix) :]
            break
    return module_path


def get_comparison_points(
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    """Get the key comparison points (last node of each major module).

    These are the nodes where we compare HF output vs AD output.

    Args:
        metadata: Metadata dict

    Returns:
        Dict mapping module name to its last (output) node:
        {
            "embed_tokens": "model_embed_tokens_embedding",
            "self_attn": "model_layers_0_self_attn_o_proj_...",
            "block_sparse_moe": "model_layers_0_block_sparse_moe_...",
            "lm_head": "lm_head_torch_linear_simple_5",
        }
    """
    boundaries = get_module_boundary_nodes(metadata)

    # Key modules we care about for comparison
    key_modules = [
        "embed_tokens",
        "self_attn",
        "block_sparse_moe",
        "lm_head",
        "norm",
    ]

    comparison_points = {}
    for key in key_modules:
        # Collect candidates containing the key
        candidates = [
            (module_name, info) for module_name, info in boundaries.items() if key in module_name
        ]
        if not candidates:
            continue

        # Prefer module paths that end with the key (e.g., "...self_attn" over "...self_attn.q_proj")
        exact_matches = [
            (module_name, info) for module_name, info in candidates if module_name.endswith(key)
        ]
        pool = exact_matches if exact_matches else candidates

        # Choose the shortest path from the preferred pool (outermost module)
        chosen_name, chosen_info = min(pool, key=lambda x: len(x[0]))
        comparison_points[key] = chosen_info["last_node"]

        # Debug visibility: which module path was selected
        print(
            f"[comparison_points] key={key} chosen_module={chosen_name} last_node={chosen_info['last_node']}"
        )

    return comparison_points


def select_module_for_key(metadata: Dict[str, Any], key: str) -> Optional[str]:
    """Select the best-matching simplified module name for a given key.

    Deterministic selection:
    - consider modules whose simplified name contains `key`
    - prefer modules whose simplified name endswith `key` (e.g. "layers.0.self_attn" over "layers.0.self_attn.q_proj")
    - choose the shortest name among the preferred pool (outermost)
    """
    boundaries = get_module_boundary_nodes(metadata)
    candidates = [name for name in boundaries.keys() if key in name]
    if not candidates:
        return None
    exact = [n for n in candidates if n.endswith(key)]
    pool = exact if exact else candidates
    return min(pool, key=len)


def get_module_boundary_outputs(
    metadata: Dict[str, Any],
    simplified_module_name: str,
) -> List[str]:
    """Return boundary-crossing output nodes for a given simplified module name.

    Uses `metadata["module_boundary_io"][module_path]["output_nodes"]` and maps module_path -> simplified name.
    """
    module_boundary_io = metadata.get("module_boundary_io", {})
    outputs: List[str] = []
    for module_path, io in module_boundary_io.items():
        if _simplify_module_path(module_path) == simplified_module_name:
            outputs = list(io.get("output_nodes", []))
            break
    return outputs


def get_nodes_for_module(
    metadata: Dict[str, Any],
    module_name: str,
) -> List[str]:
    """Get ordered list of nodes belonging to a specific module.

    Args:
        metadata: Metadata dict
        module_name: Module name (can be partial, e.g., "self_attn")

    Returns:
        List of node names in execution order
    """
    boundaries = get_module_boundary_nodes(metadata, module_filter=module_name)

    # Collect all nodes from matching modules
    all_nodes = []
    for info in boundaries.values():
        all_nodes.extend(info.get("nodes", []))

    # Sort by order if available
    nodes_info = metadata.get("nodes", {})
    all_nodes.sort(key=lambda n: nodes_info.get(n, {}).get("order", float("inf")))

    return all_nodes


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Dict[str, Any]:
    """Compare two tensors and return comparison stats.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with comparison results:
        {
            "match": bool,
            "max_diff": float,
            "mean_diff": float,
            "shape_match": bool,
            "shapes": (shape1, shape2),
        }
    """
    shape1 = tuple(tensor1.shape)
    shape2 = tuple(tensor2.shape)

    if shape1 != shape2:
        return {
            "match": False,
            "max_diff": float("inf"),
            "mean_diff": float("inf"),
            "shape_match": False,
            "shapes": (shape1, shape2),
        }

    diff = (tensor1.float() - tensor2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    match = torch.allclose(tensor1.float(), tensor2.float(), atol=atol, rtol=rtol)

    return {
        "match": match,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "shape_match": True,
        "shapes": (shape1, shape2),
    }

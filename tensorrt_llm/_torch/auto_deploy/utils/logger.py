import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import dill
import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save

from ....logger import Logger


def _get_dtype_or_type(val):
    """Get dtype if tensor-like, otherwise return type name for SymInt/SymFloat etc."""
    if hasattr(val, "dtype"):
        return val.dtype
    else:
        # For SymInt, SymFloat, etc. - return the type name
        return type(val).__name__


def _get_shape_str(val):
    """Get shape as 'dim0xdim1x...' string, or '?' if not available."""
    if hasattr(val, "shape"):
        # Handle symbolic dimensions (SymInt) by converting to str
        dims = [str(int(d)) if str(d).isdigit() else str(d) for d in val.shape]
        return "x".join(dims) if dims else "scalar"
    return "?"


def _get_shape_dtype_str(val):
    """Return 'shape : dtype' string for a value."""
    shape = _get_shape_str(val)
    dtype = _get_dtype_or_type(val)
    return f"{shape} : {dtype}"


def dump_ssa_with_meta(f, mod):
    for node in mod.graph.nodes:
        # Write out IR in traditional SSA style
        if node.op == "placeholder":
            if "val" in node.meta:
                shape_dtype = _get_shape_dtype_str(node.meta["val"])
            else:
                shape_dtype = "? : unknown"
            f.write(f"%{node.name} : {shape_dtype}\n")
        elif node.op in ("call_function", "call_method", "call_module"):
            # Build inputs list in SSA format with shape:dtype info
            input_vars = []
            for arg in node.args:
                if hasattr(arg, "name"):
                    # Look up the arg node's metadata for shape/dtype
                    if hasattr(arg, "meta") and "val" in arg.meta:
                        arg_shape_dtype = _get_shape_dtype_str(arg.meta["val"])
                        input_vars.append(f"%{arg.name} : {arg_shape_dtype}")
                    else:
                        input_vars.append(f"%{arg.name} : ? : unknown")
                else:
                    input_vars.append(str(arg))

            # Handle output shape/dtype (including multi-output)
            if "val" in node.meta:
                out_val = node.meta["val"]
                if isinstance(out_val, (tuple, list)):
                    # Multi-output: (shape1, shape2) : (dtype1, dtype2)
                    shapes = []
                    dtypes = []
                    for v in out_val:
                        if v is not None:
                            shapes.append(_get_shape_str(v))
                            dtypes.append(str(_get_dtype_or_type(v)))
                        else:
                            shapes.append("?")
                            dtypes.append("None")
                    out_info = f"({', '.join(shapes)}) : ({', '.join(dtypes)})"
                else:
                    out_info = _get_shape_dtype_str(out_val)
            else:
                out_info = "? : N/A"
            # Standard SSA notation: %out = op(args) : shape : dtype
            f.write(f"%{node.name} = {node.target}({', '.join(input_vars)}) : {out_info}\n")
        elif node.op == "output":
            # Output assignment in SSA IR
            outputs = node.args[0] if isinstance(node.args[0], (tuple, list)) else [node.args[0]]
            output_vars = []
            for o in outputs:
                if hasattr(o, "name"):
                    output_vars.append(f"%{o.name}")
                else:
                    output_vars.append(str(o))
            f.write(f"output {', '.join(output_vars)}\n")


def extract_graph_metadata(gm: nn.Module) -> Dict[str, Any]:
    """Extract metadata from GraphModule for debugging.

    Returns a dict with:
    - nodes: per-node info (op, target, nn_module_stack, output_shape, order)
    - module_boundaries: first/last node per module path

    If gm is wrapped (e.g., CapturedGraph), tries to unwrap to find the GraphModule.
    """
    from torch.fx import GraphModule

    # Try to unwrap if not a GraphModule (e.g., CapturedGraph wraps in .model)
    actual_gm = gm
    if not isinstance(actual_gm, GraphModule):
        if hasattr(gm, "model") and isinstance(gm.model, GraphModule):
            actual_gm = gm.model
        elif hasattr(gm, "graph"):
            # Has a graph attribute, might still work
            actual_gm = gm
        else:
            return {}

    nodes_info: Dict[str, Any] = {}
    module_to_nodes: Dict[str, list] = {}
    node_name_to_node: Dict[str, Any] = {}
    debug_markers: Dict[str, str] = {}

    for order, node in enumerate(actual_gm.graph.nodes):
        node_name_to_node[node.name] = node
        meta = getattr(node, "meta", {})

        # Extract nn_module_stack (full hierarchy) and deepest module path for display
        nn_module_stack = meta.get("nn_module_stack", {})
        module_stack_paths = []
        if nn_module_stack:
            if isinstance(nn_module_stack, dict):
                # keys are hierarchical module paths (outer->inner in insertion order)
                module_stack_paths = list(nn_module_stack.keys())
            else:
                module_stack_paths = [str(nn_module_stack)]

        deepest_module_path = module_stack_paths[-1] if module_stack_paths else ""

        # Extract output shape from tensor_meta or val
        # Handle SymInt values by converting to int or string
        output_shape = None
        if "tensor_meta" in meta and hasattr(meta["tensor_meta"], "shape"):
            shape = meta["tensor_meta"].shape
            output_shape = [int(d) if str(d).isdigit() else str(d) for d in shape]
        elif "val" in meta and isinstance(meta["val"], torch.Tensor):
            shape = meta["val"].shape
            output_shape = [int(d) if str(d).isdigit() else str(d) for d in shape]

        # Store node info
        target_str = (
            str(node.target)
            if not callable(node.target)
            else (getattr(node.target, "__name__", str(node.target)))
        )
        nodes_info[node.name] = {
            "op": node.op,
            "target": target_str,
            "nn_module_stack": deepest_module_path,
            "nn_module_stack_paths": module_stack_paths,
            "output_shape": output_shape,
            "order": order,
        }

        # Group by all module paths in the stack (ancestor membership), not just the deepest
        for module_path in module_stack_paths:
            if module_path not in module_to_nodes:
                module_to_nodes[module_path] = []
            module_to_nodes[module_path].append(node.name)

        # Record debug markers (tag -> node.name) for marker-based comparisons.
        # After strip_debug_markers, the tags are stored in node.meta["debug_marker_tags"].
        marker_tags = meta.get("debug_marker_tags", [])
        for tag in marker_tags:
            debug_markers[tag] = node.name

    # Compute module boundaries
    module_boundaries = {}
    for module_path, node_names in module_to_nodes.items():
        # Nodes are already in order since we iterated in graph order
        module_boundaries[module_path] = {
            "first_node": node_names[0],
            "last_node": node_names[-1],
            "node_count": len(node_names),
            "nodes": node_names,
        }

    # Compute module boundary-crossing inputs/outputs using graph edges.
    # These are the semantic values that enter/leave a module region.
    module_boundary_io: Dict[str, Dict[str, Any]] = {}
    for module_path, node_names in module_to_nodes.items():
        region = set(node_names)
        input_nodes = set()
        output_nodes = set()

        for node_name in node_names:
            node = node_name_to_node.get(node_name)
            if node is None:
                continue

            # boundary inputs: region nodes that take at least one arg from outside the region
            for in_node in node.all_input_nodes:
                if in_node.name not in region:
                    input_nodes.add(node_name)
                    break

            # boundary outputs: region nodes whose users include a node outside the region
            for user in node.users:
                if user.name not in region:
                    output_nodes.add(node_name)
                    break

        # Deterministic ordering by graph order
        def _order(n: str) -> int:
            return nodes_info.get(n, {}).get("order", 10**18)

        module_boundary_io[module_path] = {
            "input_nodes": sorted(input_nodes, key=_order),
            "output_nodes": sorted(output_nodes, key=_order),
        }

    return {
        "nodes": nodes_info,
        "module_to_nodes": module_to_nodes,
        "module_boundaries": module_boundaries,
        "module_boundary_io": module_boundary_io,
        "debug_markers": debug_markers,
    }


class ADLogger(Logger):
    ENV_VARIABLE = "AUTO_DEPLOY_LOG_LEVEL"
    PREFIX = "TRT-LLM AUTO-DEPLOY"
    DEFAULT_LEVEL = "info"

    DUMP_GRAPHS_ENV = "AD_DUMP_GRAPHS_DIR"
    DUMP_DEBUG_ENV = "AD_DUMP_DEBUG_DIR"

    def __init__(self):
        super().__init__()
        self._dump_dir = os.environ.get(self.DUMP_GRAPHS_ENV)
        self._transform_counter = 0
        self._dump_dir_initialized = False

        # Debug dumping state
        self._debug_dump_dir = os.environ.get(self.DUMP_DEBUG_ENV)
        self._debug_dump_initialized = False
        self._inputs_to_save: Optional[Dict[str, Any]] = None

    @property
    def debug_dump_enabled(self) -> bool:
        """Check if debug dumping is enabled."""
        return self._debug_dump_dir is not None

    def set_debug_inputs(self, inputs: Optional[Dict[str, Any]]) -> None:
        """Set inputs to save with debug dumps."""
        self._inputs_to_save = inputs

    def _init_debug_dump_dir(self) -> Optional[Path]:
        """Initialize debug dump directory (lazy, only on rank 0)."""
        if not self._debug_dump_dir:
            return None

        # Only dump from main process (rank 0) or single-process mode
        if self.rank is not None and self.rank != 0:
            return None

        if not self._debug_dump_initialized:
            dump_dir = Path(self._debug_dump_dir)
            if dump_dir.exists():
                shutil.rmtree(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            self.info(f"Debug dumping enabled to: {dump_dir}")
            self._debug_dump_initialized = True

        return Path(self._debug_dump_dir)

    def dump_debug_artifacts(
        self,
        gm: nn.Module,
        stage_name: str,
        model_path: Optional[str] = None,
    ) -> None:
        """Dump GraphModule and metadata to disk for debugging.

        Args:
            gm: The GraphModule to dump
            stage_name: Name of the stage (e.g., "post_export", "final")
            model_path: Optional HF model path to store in metadata
        """
        from torch.fx import GraphModule

        dump_dir = self._init_debug_dump_dir()
        if dump_dir is None:
            return

        # Try to unwrap if not a GraphModule (e.g., CapturedGraph wraps in .model)
        actual_gm = gm
        if not isinstance(actual_gm, GraphModule):
            if hasattr(gm, "model") and isinstance(gm.model, GraphModule):
                actual_gm = gm.model
                self.info(f"Unwrapped {type(gm).__name__} to get GraphModule for {stage_name}")
            elif hasattr(gm, "graph"):
                actual_gm = gm
                self.info(f"Using {type(gm).__name__} with graph attribute for {stage_name}")
            else:
                self.debug(
                    f"Skipping debug dump for {stage_name}: no GraphModule found in {type(gm).__name__}"
                )
                return

        stage_dir = dump_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Dump GraphModule
        # For stages with weights (e.g., final), move to CPU before saving
        gm_path = stage_dir / "gm.pt"
        try:
            # Save GraphModule using dill to handle lambdas in compiled graphs
            torch.save(actual_gm, gm_path, pickle_module=dill)
            self.info(f"Dumped GraphModule to {gm_path}")
        except Exception as e:
            self.warning(f"Failed to dump GraphModule: {e}")

        # Dump metadata
        metadata = extract_graph_metadata(actual_gm)
        if model_path:
            metadata["model_path"] = model_path
        metadata_path = stage_dir / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self.info(f"Dumped metadata to {metadata_path}")
        except Exception as e:
            self.warning(f"Failed to dump metadata: {e}")

        # Dump inputs if available
        if self._inputs_to_save is not None:
            inputs_path = stage_dir / "inputs.safetensors"
            try:
                # Use safetensors to handle tensor aliasing issues
                # Only save tensor values (convert to CPU and filter non-tensors)
                inputs_to_save = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in self._inputs_to_save.items()
                }
                # Filter out non-tensor values for safetensors
                tensor_inputs = {
                    k: v for k, v in inputs_to_save.items() if isinstance(v, torch.Tensor)
                }
                safetensors_save(tensor_inputs, str(inputs_path))
                self.info(f"Dumped inputs to {inputs_path}")
            except Exception as e:
                self.warning(f"Failed to dump inputs: {e}")

    def dump_graph(self, mod: nn.Module, transform_name: str, stage: str) -> None:
        """Dump the FX graph (SSA-style) to a file after a transform."""
        if not self._dump_dir:
            return

        # Only dump from main process (rank 0) or single-process mode (rank is None)
        if self.rank is not None and self.rank != 0:
            return

        # Lazy directory initialization (only on rank 0 / main process)
        if not self._dump_dir_initialized:
            dump_dir_path = Path(self._dump_dir)
            if dump_dir_path.exists():
                shutil.rmtree(dump_dir_path)
            dump_dir_path.mkdir(parents=True, exist_ok=True)
            self.info(f"Graph dumping enabled to: {self._dump_dir}")
            self._dump_dir_initialized = True

        from torch.fx import GraphModule

        if not isinstance(mod, GraphModule):
            return  # Skip non-GraphModule (e.g., during Factory stage)

        self._transform_counter += 1
        filename = f"{self._transform_counter:03d}_{stage}_{transform_name}.txt"
        filepath = Path(self._dump_dir) / filename

        with open(filepath, "w") as f:
            f.write(f"# Transform: {transform_name}\n")
            f.write(f"# Stage: {stage}\n\n")
            dump_ssa_with_meta(f, mod)


ad_logger = ADLogger()

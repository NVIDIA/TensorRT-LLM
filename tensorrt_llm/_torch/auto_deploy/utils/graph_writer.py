import shutil
from pathlib import Path
from typing import TextIO

import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm import envs

from ....logger import Singleton
from .logger import ADLogger


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


def dump_ssa_with_meta(f: TextIO, mod: GraphModule) -> None:
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


class GraphWriter(metaclass=Singleton):
    DUMP_GRAPHS_ENV = "AD_DUMP_GRAPHS_DIR"

    def __init__(self):
        self._dump_dir = envs.get_env(self.DUMP_GRAPHS_ENV)
        self._logger = ADLogger()
        self._transform_counter = 0
        self._dump_dir_initialized = False

    def dump_graph(self, mod: nn.Module, transform_name: str, stage: str) -> None:
        """Dump the FX graph (SSA-style) to a file after a transform."""
        if not self._dump_dir:
            return

        # Only dump from main process (rank 0) or single-process mode (rank is None)
        if self._logger.rank is not None and self._logger.rank != 0:
            return

        # Lazy directory initialization (only on rank 0 / main process)
        if not self._dump_dir_initialized:
            dump_dir_path = Path(self._dump_dir)
            if dump_dir_path.exists():
                shutil.rmtree(dump_dir_path)
            dump_dir_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Graph dumping enabled to: {self._dump_dir}")
            self._dump_dir_initialized = True

        # Collect all GraphModules (including from submodules)
        graph_modules = []
        for name, submod in mod.named_modules():
            if isinstance(submod, GraphModule):
                graph_modules.append((name if name else "(root)", submod))

        if not graph_modules:
            return  # No GraphModules found

        self._transform_counter += 1
        filename = f"{self._transform_counter:03d}_{stage}_{transform_name}.txt"
        filepath = Path(self._dump_dir) / filename

        with open(filepath, "w") as f:
            f.write(f"# Transform: {transform_name}\n")
            f.write(f"# Stage: {stage}\n")
            f.write(f"# GraphModules found: {len(graph_modules)}\n\n")

            for module_name, gm in graph_modules:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"# GraphModule: {module_name}\n")
                f.write(f"{'=' * 80}\n\n")
                dump_ssa_with_meta(f, gm)


graph_writer = GraphWriter()

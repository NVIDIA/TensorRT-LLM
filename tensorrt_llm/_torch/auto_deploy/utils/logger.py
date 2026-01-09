import os
import shutil
from pathlib import Path

import torch.nn as nn

from ....logger import Logger


def _get_dtype_or_type(val):
    """Get dtype if tensor-like, otherwise return type name for SymInt/SymFloat etc."""
    if hasattr(val, "dtype"):
        return val.dtype
    else:
        # For SymInt, SymFloat, etc. - return the type name
        return type(val).__name__


def dump_ssa_with_meta(f, mod):
    for node in mod.graph.nodes:
        # Write out IR in traditional SSA style
        if node.op == "placeholder":
            if "val" in node.meta:
                dtype = _get_dtype_or_type(node.meta["val"])
            else:
                dtype = "unknown"
            f.write(f"%{node.name} : {dtype}\n")
        elif node.op in ("call_function", "call_method", "call_module"):
            # Build inputs list in SSA format
            input_vars = []
            for arg in node.args:
                if hasattr(arg, "name"):
                    input_vars.append(f"%{arg.name}")
                else:
                    input_vars.append(str(arg))
            if "val" in node.meta:
                out_dtype = _get_dtype_or_type(node.meta["val"])
            else:
                out_dtype = "N/A"
            # Standard SSA notation: %out = op(args) : out_dtype
            f.write(f"%{node.name} = {node.target}({', '.join(input_vars)}) : {out_dtype}\n")
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


class ADLogger(Logger):
    ENV_VARIABLE = "AUTO_DEPLOY_LOG_LEVEL"
    PREFIX = "TRT-LLM AUTO-DEPLOY"
    DEFAULT_LEVEL = "info"

    DUMP_GRAPHS_ENV = "AD_DUMP_GRAPHS_DIR"

    def __init__(self):
        super().__init__()
        self._dump_dir = os.environ.get(self.DUMP_GRAPHS_ENV)
        self._transform_counter = 0
        self._dump_dir_initialized = False

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

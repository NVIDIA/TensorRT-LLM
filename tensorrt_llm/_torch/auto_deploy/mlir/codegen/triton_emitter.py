# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton kernel generation from fused MLIR subgraphs.

Walks a ``FusibleSubgraph`` in topological order, emits Triton expressions
for each op via the ``_EMIT`` table, writes the generated source to a temp
file (required by Triton's ``@jit`` for ``inspect.getsourcelines``), and
registers the result as a ``torch.library.custom_op`` for FX graph
compatibility.

Supports two grid modes:
    - **Standard (1D grid)**: one program per row, processes the full last
      dimension.
    - **Grouped (2D grid)**: one program per (row, group) pair, used for
      patterns like gated RMSNorm where reductions operate on sub-slices
      of the last dimension (controlled by ``group_size`` on reduction ops).

All arithmetic is performed in f32 for numerical stability; loads upcast
from the original dtype and stores downcast back.

Generated kernels are cached by subgraph hash via ``KernelCache`` to avoid
redundant compilation.
"""

import atexit
import textwrap
from typing import Callable, List

import torch
from xdsl.dialects.builtin import FloatAttr, TensorType
from xdsl.ir import SSAValue

from .kernel_cache import KernelCache

# Module-level kernel cache shared across calls (keyed by subgraph hash)
_kernel_cache = KernelCache()

# Track temp files so they can be cleaned up at process exit.
# These files must persist while the process is alive because Triton's @jit
# calls inspect.getsourcelines() lazily during kernel compilation.
_temp_files: list[str] = []


def _cleanup_temp_files():
    import os

    for path in _temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass
    _temp_files.clear()


atexit.register(_cleanup_temp_files)

# ---------------------------------------------------------------------------
# Emission table: op name -> lambda producing Triton expression string
# ---------------------------------------------------------------------------

_EMIT = {
    # All computation is in f32 (loads upcast, stores downcast), so no per-op
    # dtype management needed. This matches hand-written normalization kernels.
    "ad.add": lambda a, b: f"({a} + {b})",
    "ad.mul": lambda a, b: f"({a} * {b})",
    "ad.sub": lambda a, b: f"({a} - {b})",
    "ad.neg": lambda a: f"(-{a})",
    "ad.pow": None,  # handled specially — needs attribute extraction for exponent
    "ad.rsqrt": lambda a: f"(1.0 / tl.sqrt({a}))",
    "ad.sqrt": lambda a: f"tl.sqrt({a})",
    "ad.silu": lambda a: f"({a} * tl.sigmoid({a}))",
    "ad.gelu": lambda a: f"({a} * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf({a} * 0.7071067811865476)))",
    "ad.relu": lambda a: f"tl.maximum({a}, 0)",
    "ad.tanh": lambda a: f"tl.extra.cuda.libdevice.tanh({a})",
    "ad.sigmoid": lambda a: f"tl.sigmoid({a})",
    "ad.exp": lambda a: f"tl.extra.cuda.libdevice.exp({a})",
    "ad.softplus": lambda a: f"tl.extra.cuda.libdevice.log(1.0 + tl.extra.cuda.libdevice.exp({a}))",
    "ad.reduce_sum": lambda a: f"tl.sum({a}, 0)",
    "ad.reduce_mean": lambda a, ncols: f"(tl.sum({a}, 0) * (1.0 / {ncols}))",
    "ad.splat": None,  # handled specially — just inline the scalar value
    "ad.cast": lambda a, dt: f"{a}.to({dt})",
}

# Triton dtype name for MLIR element types
_TRITON_DTYPE_MAP = {
    "f16": "tl.float16",
    "f32": "tl.float32",
    "bf16": "tl.bfloat16",
    "f64": "tl.float64",
}

# torch dtype string for MLIR element types (used in codegen source strings)
_TORCH_DTYPE_STR_MAP = {
    "f16": "torch.float16",
    "f32": "torch.float32",
    "bf16": "torch.bfloat16",
    "f64": "torch.float64",
}


def _mlir_elem_to_triton_str(tensor_type: TensorType) -> str:
    """Return Triton dtype string for a TensorType's element type."""
    elem_str = str(tensor_type.element_type)
    return _TRITON_DTYPE_MAP.get(elem_str, "tl.float32")


def _mlir_elem_to_torch_dtype_str(tensor_type: TensorType) -> str:
    """Return torch dtype string for a TensorType's element type (for codegen source)."""
    elem_str = str(tensor_type.element_type)
    return _TORCH_DTYPE_STR_MAP.get(elem_str, "torch.bfloat16")


def _validate_reduction_attrs(op) -> None:
    """Validate that a reduction op targets the last dim (dim=-1) as required by the row kernel.

    The Triton row kernel only supports last-dimension reductions. Any other axis
    would silently miscompile, so we reject unsupported configurations here.
    """
    dim = op.attributes["dim"].value.data
    if dim != -1:
        raise NotImplementedError(
            f"{op.name}: reduction over axis {dim} is not supported "
            "(only dim=-1 is supported by the row-wise Triton emitter)"
        )


def _get_tensor_rank(val: SSAValue) -> int:
    """Return the rank (number of dims) of an SSAValue's TensorType."""
    if isinstance(val.type, TensorType):
        return len(val.type.get_shape())
    return 0


def _is_broadcast_input(val: SSAValue, max_rank: int) -> bool:
    """Return True if this input is lower rank (broadcast/weight)."""
    return _get_tensor_rank(val) < max_rank


def _get_ncols(inputs: List[SSAValue]) -> int:
    """Return the maximum last-dim size across all highest-rank inputs.

    This determines ``N_COLS`` for the generated Triton kernel.  We take the
    *maximum* last-dim among all inputs at the highest rank, because some
    subgraphs mix narrow (e.g. ``(-1, 1)`` gating scalars) and wide
    (e.g. ``(-1, 2048)`` hidden-state) tensors.  The kernel must process
    the full row width.
    """
    max_rank = 0
    for inp in inputs:
        if isinstance(inp.type, TensorType):
            max_rank = max(max_rank, len(inp.type.get_shape()))

    ncols = 0
    for inp in inputs:
        if isinstance(inp.type, TensorType):
            shape = inp.type.get_shape()
            if len(shape) == max_rank:
                ncols = max(ncols, shape[-1])

    if ncols > 0:
        return ncols

    # Fallback: use the last dim of any input
    for inp in inputs:
        if isinstance(inp.type, TensorType):
            shape = inp.type.get_shape()
            if shape:
                return shape[-1]
    raise ValueError("Cannot determine N_COLS from subgraph inputs")


def _detect_group_size(subgraph) -> int:
    """Detect if the subgraph uses grouped reduction.

    Scans for ``AdReduceMean`` ops with ``group_size > 0``. Returns the group
    size if found (all must agree), or 0 for standard full-dim mode.
    """
    from ..dialect import AdReduceMean as _AdReduceMean

    group_size = 0
    for op in subgraph.ops:
        if isinstance(op, _AdReduceMean):
            gs_attr = op.attributes.get("group_size")
            if gs_attr is not None:
                gs = gs_attr.value.data
                if gs > 0:
                    if group_size > 0 and gs != group_size:
                        raise ValueError(f"Mixed group_sizes in subgraph: {group_size} vs {gs}")
                    group_size = gs
    return group_size


def generate_kernel_from_subgraph(subgraph) -> Callable:
    """Generate a Triton kernel from a FusibleSubgraph and register it as a custom op.

    Walks subgraph ops in topological order, emits Triton expressions via the
    ``_EMIT`` table, writes the generated kernel + launcher to a temp ``.py``
    file (Triton's ``@jit`` requires ``inspect.getsourcelines``), imports it,
    and registers it as a ``torch.library.custom_op``.

    Results are cached by subgraph hash via ``KernelCache`` so repeated calls
    with the same subgraph skip compilation entirely.

    Args:
        subgraph: A ``FusibleSubgraph`` with ops, inputs, and outputs.

    Returns:
        A wrapper callable that dispatches to the registered
        ``torch.ops.auto_deploy.mlir_fused_<hash>`` op.  Takes torch tensors
        matching subgraph inputs and returns a tuple of output tensors.
    """
    sg_hash = KernelCache.hash_subgraph(subgraph)

    # Check cache first
    cached = _kernel_cache.get(sg_hash)
    if cached is not None:
        return cached

    n_inputs = len(subgraph.inputs)
    n_outputs = len(subgraph.outputs)
    ncols = _get_ncols(subgraph.inputs)

    # Detect grouped reduction mode (e.g. gated RMSNorm with group_size > 0).
    # In grouped mode, the kernel processes one group per program instead of one
    # full row, using a 2D grid (seq_len, ngroups).
    group_size = _detect_group_size(subgraph)
    grouped_mode = group_size > 0

    # Determine max rank among inputs to detect broadcast inputs
    max_rank = max((_get_tensor_rank(inp) for inp in subgraph.inputs), default=2)

    # Map SSAValue -> variable name
    val_names: dict[int, str] = {}

    # Assign names to subgraph inputs
    for i, inp in enumerate(subgraph.inputs):
        val_names[id(inp)] = f"v{i}"

    # Track which inputs are broadcast (lower rank) or narrow (same rank but
    # last dim < N_COLS, e.g. a gating scalar of shape (-1, 1) in a subgraph
    # whose row width is 2048).  Both categories need a load pattern that
    # avoids reading past the end of the actual data.
    # Scalar-like inputs (rank-0 OR broadcast with last-dim 1, e.g. shape [1])
    # need a single-element load; Triton broadcasts the scalar automatically.
    broadcast_flags = [_is_broadcast_input(inp, max_rank) for inp in subgraph.inputs]
    scalar_flags = []
    for i, inp in enumerate(subgraph.inputs):
        rank = _get_tensor_rank(inp)
        if rank == 0:
            scalar_flags.append(True)
        elif broadcast_flags[i] and isinstance(inp.type, TensorType):
            shape = inp.type.get_shape()
            # Broadcast input whose last dim is 1 (e.g. layer_scalar shape [1])
            # must be loaded as a single element, not a vector.
            scalar_flags.append(not shape or shape[-1] == 1)
        else:
            scalar_flags.append(False)
    narrow_flags = []
    for inp in subgraph.inputs:
        if isinstance(inp.type, TensorType):
            inp_last_dim = inp.type.get_shape()[-1] if inp.type.get_shape() else 0
            narrow_flags.append(not _is_broadcast_input(inp, max_rank) and 0 < inp_last_dim < ncols)
        else:
            narrow_flags.append(False)

    # Build kernel body lines.
    # All computation is done in f32 for numerical stability (matching hand-written
    # kernels). Loads upcast to f32; stores downcast to the original dtype.
    body_lines = []

    # Load all subgraph inputs and upcast to f32.
    # In grouped mode, full-row inputs are offset by both row and group:
    #   ptr + pid_row * row_stride + pid_group * N_COLS + offs
    # Broadcast (1D) inputs (e.g. weights) are offset by group only:
    #   ptr + pid_group * N_COLS + offs
    for i, inp in enumerate(subgraph.inputs):
        if scalar_flags[i]:
            # Rank-0 (scalar) tensor: load single element, Triton broadcasts automatically.
            body_lines.append(f"    v{i} = tl.load(in{i}_ptr).to(tl.float32)")
        elif broadcast_flags[i]:
            if grouped_mode:
                body_lines.append(
                    f"    v{i} = tl.load(in{i}_ptr + group_off + offs, mask=mask).to(tl.float32)"
                )
            else:
                body_lines.append(f"    v{i} = tl.load(in{i}_ptr + offs, mask=mask).to(tl.float32)")
        elif narrow_flags[i]:
            inp_last_dim = inp.type.get_shape()[-1]
            body_lines.append(
                f"    v{i} = tl.load(in{i}_ptr + pid * {inp_last_dim}).to(tl.float32)"
            )
        else:
            body_lines.append(
                f"    v{i} = tl.load(in{i}_ptr + row_off + offs, mask=mask).to(tl.float32)"
            )

    # Process ops in topological order
    temp_counter = 0
    for op in subgraph.ops:
        op_name = op.name
        emitter = _EMIT.get(op_name)

        if op_name == "ad.splat":
            # Inline the constant value
            float_val = op.attributes["value"]
            if isinstance(float_val, FloatAttr):
                scalar = float_val.value.data
            else:
                scalar = float(str(float_val))
            result_name = f"t{temp_counter}"
            body_lines.append(f"    {result_name} = {scalar}")
            temp_counter += 1
            # Map the result SSAValue
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.pow":
            # pow(base, exponent) — exponent is an attribute, not an operand
            base_val = op.operands[0]
            base_name = val_names[id(base_val)]
            exp_attr = op.attributes["exponent"]
            if isinstance(exp_attr, FloatAttr):
                exp_val = exp_attr.value.data
            else:
                exp_val = float(str(exp_attr))
            result_name = f"t{temp_counter}"
            body_lines.append(
                f"    {result_name} = tl.extra.cuda.libdevice.pow({base_name}, {exp_val})"
            )
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.reduce_mean":
            # reduce_mean(input, ncols) — row-wise only
            _validate_reduction_attrs(op)
            input_val = op.operands[0]
            input_name = val_names[id(input_val)]
            result_name = f"t{temp_counter}"
            expr = _EMIT["ad.reduce_mean"](input_name, ncols)
            body_lines.append(f"    {result_name} = {expr}")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.reduce_sum":
            # reduce_sum(input) — row-wise only
            _validate_reduction_attrs(op)
            input_val = op.operands[0]
            input_name = val_names[id(input_val)]
            result_name = f"t{temp_counter}"
            expr = _EMIT["ad.reduce_sum"](input_name)
            body_lines.append(f"    {result_name} = {expr}")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.cast":
            input_val = op.operands[0]
            input_name = val_names[id(input_val)]
            # Get target dtype from the result type
            result_type = op.results[0].type
            triton_dt = _mlir_elem_to_triton_str(result_type)
            result_name = f"t{temp_counter}"
            expr = _EMIT["ad.cast"](input_name, triton_dt)
            body_lines.append(f"    {result_name} = {expr}")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif emitter is not None:
            # Standard elementwise op
            operand_names = [val_names[id(v)] for v in op.operands]
            result_name = f"t{temp_counter}"
            expr = emitter(*operand_names)
            body_lines.append(f"    {result_name} = {expr}")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name
        else:
            raise ValueError(f"Unsupported op for Triton codegen: {op_name}")

    # Store all subgraph outputs, downcasting from f32 to the original dtype.
    # Outputs are always contiguous (allocated via torch.empty_like), so use
    # ``out_row_off`` (= pid * N_COLS) rather than ``row_off`` (= pid * row_stride).
    # These differ when inputs are non-contiguous (e.g. from aten.chunk).
    for i, out in enumerate(subgraph.outputs):
        out_name = val_names[id(out)]
        out_rank = _get_tensor_rank(out)
        # Determine original output dtype for downcast from f32
        if isinstance(out.type, TensorType):
            out_dt = _mlir_elem_to_triton_str(out.type)
        else:
            out_dt = "tl.bfloat16"
        cast = f".to({out_dt})" if out_dt != "tl.float32" else ""
        if out_rank < max_rank:
            body_lines.append(f"    tl.store(out{i}_ptr + pid, {out_name}{cast})")
        else:
            body_lines.append(
                f"    tl.store(out{i}_ptr + out_row_off + offs, {out_name}{cast}, mask=mask)"
            )

    # Build parameter lists
    in_ptr_params = [f"in{i}_ptr" for i in range(n_inputs)]
    out_ptr_params = [f"out{i}_ptr" for i in range(n_outputs)]
    all_ptr_params = in_ptr_params + out_ptr_params

    kernel_name = f"fused_kernel_{sg_hash}"
    constexpr_params = ["row_stride: tl.constexpr", "N_COLS: tl.constexpr", "BLOCK_N: tl.constexpr"]
    if grouped_mode:
        constexpr_params.append("FEAT_SIZE: tl.constexpr")
    params_str = ",\n    ".join(all_ptr_params + constexpr_params)

    if grouped_mode:
        # 2D grid: (seq_len, ngroups). Each program processes one group of one row.
        # row_off uses input row_stride (may be non-contiguous).
        # out_row_off uses FEAT_SIZE (output is always contiguous).
        preamble = textwrap.dedent("""\
        pid_row = tl.program_id(0)
        pid_group = tl.program_id(1)
        pid = pid_row
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N_COLS
        group_off = pid_group * N_COLS
        row_off = pid_row * row_stride + group_off
        out_row_off = pid_row * FEAT_SIZE + group_off""")
    else:
        preamble = textwrap.dedent("""\
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N_COLS
        row_off = pid * row_stride
        out_row_off = pid * N_COLS""")

    preamble_lines = ["    " + line for line in preamble.splitlines()]

    kernel_src = (
        f"@triton.jit\n"
        f"def {kernel_name}(\n"
        f"    {params_str},\n"
        f"):\n" + "\n".join(preamble_lines) + "\n" + "\n".join(body_lines) + "\n"
    )

    # Find the reference input for shape in launcher + fake impl.
    # Among highest-rank inputs, pick the one with the largest last dimension.
    # This ensures we use a full-row activation tensor (e.g. shape (-1, 2048)),
    # not a narrow gating scalar (e.g. shape (-1, 1)).
    ref_input_idx = 0
    ref_input_rank = 0
    ref_input_ncols = 0
    for i_inp, inp in enumerate(subgraph.inputs):
        rank = _get_tensor_rank(inp)
        inp_ncols = (
            inp.type.get_shape()[-1]
            if isinstance(inp.type, TensorType) and inp.type.get_shape()
            else 0
        )
        if rank > ref_input_rank or (rank == ref_input_rank and inp_ncols > ref_input_ncols):
            ref_input_rank = rank
            ref_input_ncols = inp_ncols
            ref_input_idx = i_inp

    # Build launcher function
    in_tensor_params = [f"input{i}" for i in range(n_inputs)]
    ref = f"input{ref_input_idx}"
    out_alloc_lines = []
    # Determine output shapes and dtypes from subgraph output MLIR types
    for i, out in enumerate(subgraph.outputs):
        out_rank = _get_tensor_rank(out)
        out_dtype_str = (
            _mlir_elem_to_torch_dtype_str(out.type)
            if isinstance(out.type, TensorType)
            else f"{ref}.dtype"
        )
        if out_rank < max_rank:
            out_alloc_lines.append(
                f"    out{i} = torch.empty("
                f"{ref}.shape[:-1] + (1,), device={ref}.device, dtype={out_dtype_str})"
            )
        else:
            out_alloc_lines.append(f"    out{i} = torch.empty_like({ref}, dtype={out_dtype_str})")

    launch_args = [f"input{i}" for i in range(n_inputs)] + [f"out{i}" for i in range(n_outputs)]

    return_tuple = ", ".join(f"out{i}" for i in range(n_outputs))

    launcher_name = f"launch_{sg_hash}"
    if grouped_mode:
        launcher_src = (
            f"def {launcher_name}({', '.join(in_tensor_params)}):\n"
            f"    feat_size = {ref}.size(-1)\n"
            f"    seq_len = {ref}.numel() // feat_size\n"
            f"    row_stride = {ref}.stride(-2) if {ref}.dim() >= 2 else feat_size\n"
            f"    group_size = {group_size}\n"
            f"    ngroups = feat_size // group_size\n"
            f"    BLOCK_N = triton.next_power_of_2(group_size)\n"
            + "\n".join(out_alloc_lines)
            + "\n"
            f"    grid = (seq_len, ngroups)\n"
            f"    {kernel_name}[grid](\n"
            f"        {', '.join(launch_args)},\n"
            f"        row_stride=row_stride,\n"
            f"        N_COLS=group_size,\n"
            f"        BLOCK_N=BLOCK_N,\n"
            f"        FEAT_SIZE=feat_size,\n"
            f"        num_warps=4,\n"
            f"        num_stages=3,\n"
            f"    )\n"
            f"    return ({return_tuple},)\n"
        )
    else:
        launcher_src = (
            f"def {launcher_name}({', '.join(in_tensor_params)}):\n"
            f"    feat_size = {ref}.size(-1)\n"
            f"    seq_len = {ref}.numel() // feat_size\n"
            f"    row_stride = {ref}.stride(-2) if {ref}.dim() >= 2 else feat_size\n"
            f"    BLOCK_N = triton.next_power_of_2(feat_size)\n" + "\n".join(out_alloc_lines) + "\n"
            f"    grid = (seq_len,)\n"
            f"    {kernel_name}[grid](\n"
            f"        {', '.join(launch_args)},\n"
            f"        row_stride=row_stride,\n"
            f"        N_COLS=feat_size,\n"
            f"        BLOCK_N=BLOCK_N,\n"
            f"        num_warps=4,\n"
            f"        num_stages=3,\n"
            f"    )\n"
            f"    return ({return_tuple},)\n"
        )

    # Compile the kernel + launcher by writing to a temp file.
    # Triton's @jit requires inspect.getsourcelines() to work, which needs
    # the function to live in an actual .py file on disk.
    full_src = (
        "import triton\n"
        "import triton.language as tl\n"
        "import torch\n\n" + kernel_src + "\n" + launcher_src
    )

    import logging as _logging
    import os as _os

    _logging.getLogger("mlir_codegen").info("Generated kernel %s:\n%s", sg_hash, full_src)

    # Optional: dump kernel source to a directory for offline inspection.
    # Controlled by the AD_DUMP_KERNELS_DIR environment variable.
    _kernel_dump_dir = _os.environ.get("AD_DUMP_KERNELS_DIR")
    if _kernel_dump_dir:
        _dump_path = _os.path.join(_kernel_dump_dir, f"triton_gen_{sg_hash}.py")
        _os.makedirs(_kernel_dump_dir, exist_ok=True)
        with open(_dump_path, "w") as _f:
            _f.write(full_src)

    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"triton_gen_{sg_hash}_", delete=False
    ) as f:
        f.write(full_src)
        tmp_path = f.name
    _temp_files.append(tmp_path)

    spec = importlib.util.spec_from_file_location(f"_triton_gen_{sg_hash}", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    launcher_fn = getattr(mod, launcher_name)

    # Register as torch.library.custom_op
    op_name = f"auto_deploy::mlir_fused_{sg_hash}"

    # Build the custom op dynamically
    tensor_annotations = ", ".join(f"input{i}: torch.Tensor" for i in range(n_inputs))
    return_annotation = "tuple[" + ", ".join("torch.Tensor" for _ in range(n_outputs)) + "]"

    # Build fake return expressions using explicit output shapes from MLIR types.
    # Dynamic dims (-1) are resolved from the highest-rank input (the activation
    # tensor, not a weight/bias), using the matching dimension index.
    ref_input = f"input{ref_input_idx}"

    fake_returns = []
    for i, out in enumerate(subgraph.outputs):
        if isinstance(out.type, TensorType):
            shape_parts = []
            for d, s in enumerate(out.type.get_shape()):
                if s < 0:
                    shape_parts.append(f"{ref_input}.shape[{d}]")
                else:
                    shape_parts.append(str(s))
            shape_expr = "(" + ", ".join(shape_parts) + (",)" if len(shape_parts) == 1 else ")")
            out_dtype_str = _mlir_elem_to_torch_dtype_str(out.type)
            fake_returns.append(
                f"torch.empty({shape_expr}, device={ref_input}.device, dtype={out_dtype_str})"
            )
        else:
            fake_returns.append(f"torch.empty_like({ref_input})")

    fake_returns_str = ", ".join(fake_returns)

    reg_src = (
        f'@torch.library.custom_op("{op_name}", mutates_args=())\n'
        f"def the_op({tensor_annotations}) -> {return_annotation}:\n"
        f"    return launcher_fn({', '.join(in_tensor_params)})\n"
        f"\n"
        f"@the_op.register_fake\n"
        f"def _({tensor_annotations}):\n"
        f"    return ({fake_returns_str},)\n"
    )

    reg_globals = {"torch": torch, "launcher_fn": launcher_fn}
    exec(reg_src, reg_globals)  # noqa: S102  # nosec B102

    # Build a wrapper that calls via torch.ops
    op_short_name = f"mlir_fused_{sg_hash}"

    def make_wrapper(name):
        def wrapper(*args):
            op_fn = getattr(torch.ops.auto_deploy, name)
            return op_fn(*args)

        return wrapper

    wrapper = make_wrapper(op_short_name)

    _kernel_cache.put(sg_hash, wrapper)

    return wrapper

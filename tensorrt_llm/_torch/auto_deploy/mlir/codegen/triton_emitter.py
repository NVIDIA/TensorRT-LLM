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

"""Template-based Triton kernel generation from fused MLIR ops.

Two operating modes:
    - ``preexisting``: Map fused ops to existing custom ops (FlashInfer/Triton).
    - ``generate``: Generate Triton kernels from MLIR op semantics, registered
      as proper ``torch.library.custom_op`` for FX graph compatibility.

String-based codegen:
    - ``generate_kernel_from_subgraph()``: Walk a ``FusibleSubgraph`` in topo
      order, emit Triton expressions per op, compile via ``exec()``, and
      register as ``torch.library.custom_op``.
"""

import textwrap
from typing import Callable, List

import torch
from xdsl.dialects.builtin import FloatAttr, TensorType
from xdsl.ir import SSAValue

from .kernel_cache import KernelCache

# Cache for registered generated ops (module-level to survive across instances)
_generated_op_cache: dict = {}

# Module-level kernel cache shared across calls
_kernel_cache = KernelCache()

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
    "ad.gelu": lambda a: f"({a} * 0.5 * (1.0 + tl.math.erf({a} * 0.7071067811865476)))",
    "ad.relu": lambda a: f"tl.maximum({a}, 0)",
    "ad.tanh": lambda a: f"tl.math.tanh({a})",
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


def _mlir_elem_to_triton_str(tensor_type: TensorType) -> str:
    """Return Triton dtype string for a TensorType's element type."""
    elem_str = str(tensor_type.element_type)
    return _TRITON_DTYPE_MAP.get(elem_str, "tl.float32")


def _get_tensor_rank(val: SSAValue) -> int:
    """Return the rank (number of dims) of an SSAValue's TensorType."""
    if isinstance(val.type, TensorType):
        return len(val.type.get_shape())
    return 0


def _is_broadcast_input(val: SSAValue, max_rank: int) -> bool:
    """Return True if this input is lower rank (broadcast/weight)."""
    return _get_tensor_rank(val) < max_rank


def _get_ncols(inputs: List[SSAValue]) -> int:
    """Return the last-dim size from the highest-rank input."""
    for inp in inputs:
        if isinstance(inp.type, TensorType):
            shape = inp.type.get_shape()
            if len(shape) >= 2:
                return shape[-1]
    # Fallback: use the last dim of any input
    for inp in inputs:
        if isinstance(inp.type, TensorType):
            shape = inp.type.get_shape()
            if shape:
                return shape[-1]
    raise ValueError("Cannot determine N_COLS from subgraph inputs")


def generate_kernel_from_subgraph(subgraph) -> Callable:
    """Generate a Triton kernel from a FusibleSubgraph and register it.

    Walks the subgraph ops in topological order, emits Triton expressions,
    wraps them in kernel boilerplate, compiles via ``exec()``, and registers
    the result as a ``torch.library.custom_op``.

    Args:
        subgraph: A ``FusibleSubgraph`` with ops, inputs, and outputs.

    Returns:
        A callable that takes torch tensors (matching subgraph inputs) and
        returns a tuple of output tensors.
    """
    sg_hash = KernelCache.hash_subgraph(subgraph)

    # Check cache first
    cached = _kernel_cache.get(sg_hash)
    if cached is not None:
        return cached

    # Also check if the custom op was already registered in _generated_op_cache
    cache_key = f"mlir_fused_{sg_hash}"
    if cache_key in _generated_op_cache:
        return _generated_op_cache[cache_key]

    n_inputs = len(subgraph.inputs)
    n_outputs = len(subgraph.outputs)
    ncols = _get_ncols(subgraph.inputs)

    # Determine max rank among inputs to detect broadcast inputs
    max_rank = max((_get_tensor_rank(inp) for inp in subgraph.inputs), default=2)

    # Map SSAValue -> variable name
    val_names: dict[int, str] = {}

    # Assign names to subgraph inputs
    for i, inp in enumerate(subgraph.inputs):
        val_names[id(inp)] = f"v{i}"

    # Track which inputs are broadcast (1D weights)
    broadcast_flags = [_is_broadcast_input(inp, max_rank) for inp in subgraph.inputs]

    # Build kernel body lines.
    # All computation is done in f32 for numerical stability (matching hand-written
    # kernels). Loads upcast to f32; stores downcast to the original dtype.
    body_lines = []

    # Load all subgraph inputs and upcast to f32
    for i, inp in enumerate(subgraph.inputs):
        if broadcast_flags[i]:
            body_lines.append(f"    v{i} = tl.load(in{i}_ptr + offs, mask=mask).to(tl.float32)")
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
            body_lines.append(f"    {result_name} = tl.math.pow({base_name}, {exp_val})")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.reduce_mean":
            # reduce_mean(input, ncols)
            input_val = op.operands[0]
            input_name = val_names[id(input_val)]
            result_name = f"t{temp_counter}"
            expr = _EMIT["ad.reduce_mean"](input_name, ncols)
            body_lines.append(f"    {result_name} = {expr}")
            temp_counter += 1
            for r in op.results:
                val_names[id(r)] = result_name

        elif op_name == "ad.reduce_sum":
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

    # Store all subgraph outputs, downcasting from f32 to the original dtype
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
                f"    tl.store(out{i}_ptr + row_off + offs, {out_name}{cast}, mask=mask)"
            )

    # Build parameter lists
    in_ptr_params = [f"in{i}_ptr" for i in range(n_inputs)]
    out_ptr_params = [f"out{i}_ptr" for i in range(n_outputs)]
    all_ptr_params = in_ptr_params + out_ptr_params

    kernel_name = f"fused_kernel_{sg_hash}"
    params_str = ",\n    ".join(
        all_ptr_params
        + ["row_stride: tl.constexpr", "N_COLS: tl.constexpr", "BLOCK_N: tl.constexpr"]
    )

    preamble = textwrap.dedent("""\
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N_COLS
    row_off = pid * row_stride""")

    preamble_lines = ["    " + line for line in preamble.splitlines()]

    kernel_src = (
        f"@triton.jit\n"
        f"def {kernel_name}(\n"
        f"    {params_str},\n"
        f"):\n" + "\n".join(preamble_lines) + "\n" + "\n".join(body_lines) + "\n"
    )

    # Find the highest-rank input index for shape reference in launcher + fake impl.
    # This avoids using a 1D weight as the shape reference when the activation
    # tensor (higher rank) should be used.
    ref_input_idx = 0
    ref_input_rank = 0
    for i_inp, inp in enumerate(subgraph.inputs):
        rank = _get_tensor_rank(inp)
        if rank > ref_input_rank:
            ref_input_rank = rank
            ref_input_idx = i_inp

    # Build launcher function
    in_tensor_params = [f"input{i}" for i in range(n_inputs)]
    ref = f"input{ref_input_idx}"
    out_alloc_lines = []
    # Determine output shapes from subgraph output types
    for i, out in enumerate(subgraph.outputs):
        out_rank = _get_tensor_rank(out)
        if out_rank < max_rank:
            out_alloc_lines.append(
                f"    out{i} = torch.empty("
                f"{ref}.shape[:-1] + (1,), device={ref}.device, dtype={ref}.dtype)"
            )
        else:
            out_alloc_lines.append(f"    out{i} = torch.empty_like({ref})")

    launch_args = [f"input{i}" for i in range(n_inputs)] + [f"out{i}" for i in range(n_outputs)]

    return_tuple = ", ".join(f"out{i}" for i in range(n_outputs))

    launcher_name = f"launch_{sg_hash}"
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

    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"triton_gen_{sg_hash}_", delete=False
    ) as f:
        f.write(full_src)
        tmp_path = f.name

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
            fake_returns.append(
                f"torch.empty({shape_expr}, device={ref_input}.device, dtype={ref_input}.dtype)"
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
    exec(reg_src, reg_globals)  # noqa: S102

    # Build a wrapper that calls via torch.ops
    op_short_name = f"mlir_fused_{sg_hash}"

    def make_wrapper(name):
        def wrapper(*args):
            op_fn = getattr(torch.ops.auto_deploy, name)
            return op_fn(*args)

        return wrapper

    wrapper = make_wrapper(op_short_name)

    _generated_op_cache[cache_key] = wrapper
    _kernel_cache.put(sg_hash, wrapper)

    return wrapper

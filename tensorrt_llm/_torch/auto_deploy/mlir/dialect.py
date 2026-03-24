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

"""AutoDeploy MLIR dialect definition using xDSL.

Defines the ``ad`` dialect with ops that mirror the FX graph operations
relevant to post-sharding fusions. Uses MLIR builtin ``TensorType`` with
dynamic dims (``-1``) for shape representation.

Ops:
    - ``ad.add``: Elementwise add (maps to ``aten.add.Tensor``)
    - ``ad.rmsnorm``: RMSNorm with eps attribute (backend-agnostic)
    - ``ad.to_dtype``: Dtype cast (maps to ``aten.to.dtype``)
    - ``ad.opaque``: Catch-all for unmodeled FX ops
    - ``ad.graph_input`` / ``ad.graph_output``: Graph boundaries
    - ``ad.mul``, ``ad.sub``, ``ad.neg``: Elementwise arithmetic primitives
    - ``ad.pow``, ``ad.rsqrt``, ``ad.sqrt``: Power/root primitives
    - ``ad.silu``, ``ad.gelu``, ``ad.relu``, ``ad.tanh``, ``ad.sigmoid``: Activation primitives
    - ``ad.exp``, ``ad.softplus``: Elementwise math primitives
    - ``ad.reduce_sum``, ``ad.reduce_mean``: Reduction primitives
    - ``ad.cast``: Dtype cast primitive
    - ``ad.splat``: Constant scalar splat
"""

import torch
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
)
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
    var_result_def,
)

# ---------------------------------------------------------------------------
# FP8 type definitions (not yet in xDSL upstream)
# ---------------------------------------------------------------------------


@irdl_attr_definition
class Float8E4M3FNType(ParametrizedAttribute, TypeAttribute):
    """FP8 E4M3 type (matches ``torch.float8_e4m3fn`` / MLIR ``f8E4M3FN``)."""

    name = "ad.f8E4M3FN"


@irdl_attr_definition
class Float8E5M2Type(ParametrizedAttribute, TypeAttribute):
    """FP8 E5M2 type (matches ``torch.float8_e5m2`` / MLIR ``f8E5M2``)."""

    name = "ad.f8E5M2"


# ---------------------------------------------------------------------------
# Dtype mapping helpers
# ---------------------------------------------------------------------------

_TORCH_TO_MLIR_DTYPE = {
    torch.float16: Float16Type,
    torch.bfloat16: BFloat16Type,
    torch.float32: Float32Type,
    torch.float64: Float64Type,
    torch.float8_e4m3fn: Float8E4M3FNType,
    torch.float8_e5m2: Float8E5M2Type,
    torch.int8: lambda: IntegerType(8),
    torch.int16: lambda: IntegerType(16),
    torch.int32: lambda: IntegerType(32),
    torch.int64: lambda: IntegerType(64),
    torch.bool: lambda: IntegerType(1),
}

# Reverse mapping for types that have a unique class (float types).
# IntegerType uses width to disambiguate, handled separately in mlir_to_torch_dtype.
_MLIR_CLS_TO_TORCH_DTYPE = {
    Float16Type: torch.float16,
    BFloat16Type: torch.bfloat16,
    Float32Type: torch.float32,
    Float64Type: torch.float64,
    Float8E4M3FNType: torch.float8_e4m3fn,
    Float8E5M2Type: torch.float8_e5m2,
}

_MLIR_INT_WIDTH_TO_TORCH_DTYPE = {
    1: torch.bool,
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}


def torch_dtype_to_mlir(dtype: torch.dtype) -> Attribute:
    """Convert a PyTorch dtype to the corresponding MLIR element type."""
    ctor = _TORCH_TO_MLIR_DTYPE.get(dtype)
    if ctor is None:
        raise ValueError(f"Unsupported torch dtype for MLIR conversion: {dtype}")
    return ctor()


def mlir_to_torch_dtype(mlir_type: Attribute) -> torch.dtype:
    """Convert an MLIR element type back to a PyTorch dtype."""
    for mlir_cls, torch_dt in _MLIR_CLS_TO_TORCH_DTYPE.items():
        if isinstance(mlir_type, mlir_cls):
            return torch_dt
    if isinstance(mlir_type, IntegerType):
        width = mlir_type.width.data
        dt = _MLIR_INT_WIDTH_TO_TORCH_DTYPE.get(width)
        if dt is not None:
            return dt
    raise ValueError(f"Unsupported MLIR type for torch conversion: {mlir_type}")


def tensor_type_from_meta(fake_tensor: torch.Tensor) -> TensorType:
    """Build an MLIR ``TensorType`` from a FakeTensor's shape and dtype.

    Dynamic dimensions (from ``torch.export`` symbolic ints) are mapped to ``-1``.
    """
    shape = []
    for s in fake_tensor.shape:
        if isinstance(s, (torch.SymInt,)):
            shape.append(-1)
        else:
            shape.append(int(s))
    elem = torch_dtype_to_mlir(fake_tensor.dtype)
    return TensorType(elem, shape)


# ---------------------------------------------------------------------------
# Op definitions
# ---------------------------------------------------------------------------


@irdl_op_definition
class AdAdd(IRDLOperation):
    """Elementwise addition — mirrors ``aten.add.Tensor``."""

    name = "ad.add"
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdRMSNorm(IRDLOperation):
    """RMSNorm — backend-agnostic, maps to any rmsnorm variant.

    All FX rmsnorm variants (flashinfer, triton, torch) lower to this single op.
    Backend choice happens at codegen/lowering time.
    """

    name = "ad.rmsnorm"
    input = operand_def(AnyAttr())
    weight = operand_def(AnyAttr())
    eps = attr_def(FloatAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdToDtype(IRDLOperation):
    """Dtype cast — mirrors ``aten.to.dtype``."""

    name = "ad.to_dtype"
    input = operand_def(AnyAttr())
    target_dtype = attr_def(IntegerAttr)  # stores torch dtype enum value
    output = result_def(AnyAttr())


@irdl_op_definition
class AdOpaque(IRDLOperation):
    """Catch-all for FX ops not explicitly modeled in the dialect.

    Preserves graph connectivity without needing per-op MLIR definitions.
    The ``op_name`` attribute stores the original FX target for round-trip.
    The ``node_key`` attribute is a unique key for metadata side-table lookup.
    """

    name = "ad.opaque"
    inputs = var_operand_def(AnyAttr())
    op_name = attr_def(StringAttr)
    node_key = attr_def(StringAttr)
    outputs = var_result_def(AnyAttr())


@irdl_op_definition
class AdGraphInput(IRDLOperation):
    """Graph input placeholder — mirrors FX ``placeholder`` nodes."""

    name = "ad.graph_input"
    input_name = attr_def(StringAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdGraphOutput(IRDLOperation):
    """Graph output — mirrors FX ``output`` nodes."""

    name = "ad.graph_output"
    inputs = var_operand_def(AnyAttr())


# ---------------------------------------------------------------------------
# Primitive ops for elementwise fusion
# ---------------------------------------------------------------------------


@irdl_op_definition
class AdMul(IRDLOperation):
    """Elementwise multiplication — mirrors ``aten.mul.Tensor``."""

    name = "ad.mul"
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSub(IRDLOperation):
    """Elementwise subtraction — mirrors ``aten.sub.Tensor``."""

    name = "ad.sub"
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdNeg(IRDLOperation):
    """Elementwise negation — mirrors ``aten.neg``."""

    name = "ad.neg"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdPow(IRDLOperation):
    """Elementwise power — mirrors ``aten.pow.Tensor_Scalar``."""

    name = "ad.pow"
    base = operand_def(AnyAttr())
    exponent = attr_def(FloatAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdRsqrt(IRDLOperation):
    """Elementwise reciprocal square root — mirrors ``aten.rsqrt``."""

    name = "ad.rsqrt"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSqrt(IRDLOperation):
    """Elementwise square root — mirrors ``aten.sqrt``."""

    name = "ad.sqrt"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSilu(IRDLOperation):
    """SiLU activation — mirrors ``aten.silu``."""

    name = "ad.silu"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdGelu(IRDLOperation):
    """GELU activation — mirrors ``aten.gelu``."""

    name = "ad.gelu"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdRelu(IRDLOperation):
    """ReLU activation — mirrors ``aten.relu``."""

    name = "ad.relu"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdTanh(IRDLOperation):
    """Tanh activation — mirrors ``aten.tanh``."""

    name = "ad.tanh"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSigmoid(IRDLOperation):
    """Sigmoid activation — mirrors ``aten.sigmoid``."""

    name = "ad.sigmoid"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdExp(IRDLOperation):
    """Elementwise exponential — mirrors ``aten.exp``."""

    name = "ad.exp"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSoftplus(IRDLOperation):
    """Softplus activation — mirrors ``aten.softplus``."""

    name = "ad.softplus"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())


@irdl_op_definition
class AdReduceSum(IRDLOperation):
    """Reduction sum along a dimension — mirrors ``aten.sum.dim_IntList``."""

    name = "ad.reduce_sum"
    input = operand_def(AnyAttr())
    dim = attr_def(IntegerAttr)
    keepdim = attr_def(IntegerAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdReduceMean(IRDLOperation):
    """Reduction mean along a dimension — mirrors ``aten.mean.dim``."""

    name = "ad.reduce_mean"
    input = operand_def(AnyAttr())
    dim = attr_def(IntegerAttr)
    keepdim = attr_def(IntegerAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdCast(IRDLOperation):
    """Dtype cast — mirrors ``aten.to.dtype`` as a primitive op."""

    name = "ad.cast"
    input = operand_def(AnyAttr())
    target_dtype = attr_def(IntegerAttr)
    output = result_def(AnyAttr())


@irdl_op_definition
class AdSplat(IRDLOperation):
    """Constant splat — creates a tensor filled with a scalar value."""

    name = "ad.splat"
    # TODO(suyogg): FloatAttr only supports floating-point constants. Integer and
    # boolean literals are currently unrepresentable. Change to AnyAttr() and update
    # fx_to_mlir.py to build IntegerAttr/BoolAttr based on the Python value type.
    value = attr_def(FloatAttr)
    output = result_def(AnyAttr())


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------

AD_OPS = [
    AdAdd,
    AdRMSNorm,
    AdToDtype,
    AdOpaque,
    AdGraphInput,
    AdGraphOutput,
    AdMul,
    AdSub,
    AdNeg,
    AdPow,
    AdRsqrt,
    AdSqrt,
    AdSilu,
    AdGelu,
    AdRelu,
    AdTanh,
    AdSigmoid,
    AdExp,
    AdSoftplus,
    AdReduceSum,
    AdReduceMean,
    AdCast,
    AdSplat,
]


AD_TYPES = [
    Float8E4M3FNType,
    Float8E5M2Type,
]


def register_ad_dialect(ctx):
    """Register all ``ad`` dialect ops and types with the given xDSL context."""
    for op_cls in AD_OPS:
        ctx.load_op(op_cls)
    for type_cls in AD_TYPES:
        ctx.load_attr_or_type(type_cls)

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

"""FX graph → MLIR (xDSL) converter.

Walks an FX ``GraphModule`` topologically and emits ``ad`` dialect ops.
Tensor shapes/dtypes are extracted from ``node.meta["val"]`` (FakeTensor).
FX metadata is stored in a Python-side dict keyed by a unique node name
for round-trip fidelity.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.fx import GraphModule, Node
from xdsl.dialects.builtin import (
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import Block, Region, SSAValue

from ..utils.node_utils import is_op
from .dialect import (
    AdAdd,
    AdGelu,
    AdGraphInput,
    AdGraphOutput,
    AdMul,
    AdNeg,
    AdOpaque,
    AdPow,
    AdReduceMean,
    AdRelu,
    AdRMSNorm,
    AdRsqrt,
    AdSilu,
    AdSplat,
    AdSqrt,
    AdSub,
    AdTanh,
    AdToDtype,
    register_ad_dialect,
    tensor_type_from_meta,
)

# All rmsnorm FX targets that map to the single ad.rmsnorm op
_RMSNORM_TARGETS = []


def _init_rmsnorm_targets():
    """Lazily collect rmsnorm op targets (avoids import-time torch.ops resolution)."""
    global _RMSNORM_TARGETS
    if _RMSNORM_TARGETS:
        return
    for attr_name in [
        "flashinfer_rms_norm",
        "triton_rms_norm",
        "torch_rmsnorm",
    ]:
        op = getattr(torch.ops.auto_deploy, attr_name, None)
        if op is not None:
            _RMSNORM_TARGETS.append(op)


def _is_rmsnorm(node: Node) -> bool:
    """Return True if *node* calls any registered rmsnorm variant."""
    _init_rmsnorm_targets()
    return is_op(node, _RMSNORM_TARGETS) if _RMSNORM_TARGETS else False


def _get_fake_tensor(node: Node) -> Optional[torch.Tensor]:
    """Extract the FakeTensor from node metadata, if available."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val
    return None


def _result_type_for_node(node: Node) -> TensorType:
    """Return an MLIR TensorType for *node*'s output tensor."""
    ft = _get_fake_tensor(node)
    if ft is not None:
        return tensor_type_from_meta(ft)
    # Fallback: unranked f32 tensor (should not happen with proper shape prop)
    from xdsl.dialects.builtin import Float32Type

    return TensorType(Float32Type(), [-1])


class FXToMLIRConverter:
    """Convert an FX ``GraphModule`` to an xDSL ``ModuleOp`` using the ``ad`` dialect.

    After conversion:
    - ``self.mlir_module``: the xDSL ``ModuleOp``
    - ``self.metadata``: dict mapping ``node_name → node.meta`` for round-trip
    """

    def __init__(self, gm: GraphModule):
        self._gm = gm
        # node name → MLIR SSAValue (for wiring operands)
        self._value_map: Dict[str, SSAValue] = {}
        # node name → node.meta (for round-trip)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.mlir_module: Optional[ModuleOp] = None

    def convert(self) -> ModuleOp:
        """Run the conversion and return the MLIR module."""
        from xdsl.context import Context as MLContext

        ctx = MLContext()
        register_ad_dialect(ctx)

        block = Block()
        region = Region([block])

        for node in self._gm.graph.nodes:
            self._convert_node(node, block)

        self.mlir_module = ModuleOp(region)
        return self.mlir_module

    # ------------------------------------------------------------------
    # Per-node conversion
    # ------------------------------------------------------------------

    def _convert_node(self, node: Node, block: Block) -> None:
        """Dispatch to the appropriate handler based on FX node op type."""
        if node.op == "placeholder":
            self._convert_placeholder(node, block)
        elif node.op == "call_function":
            self._convert_call_function(node, block)
        elif node.op == "output":
            self._convert_output(node, block)
        elif node.op == "get_attr":
            # Parameters/buffers — treat as graph inputs for MLIR purposes
            self._convert_get_attr(node, block)
        # call_module and call_method are rare in exported graphs; skip for prototype

    def _convert_placeholder(self, node: Node, block: Block) -> None:
        """Convert FX placeholder → ``ad.graph_input``."""
        result_type = _result_type_for_node(node)
        op = AdGraphInput.build(
            attributes={"input_name": StringAttr(node.name)},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_get_attr(self, node: Node, block: Block) -> None:
        """Convert FX get_attr → ``ad.graph_input`` (parameters as inputs)."""
        result_type = _result_type_for_node(node)
        op = AdGraphInput.build(
            attributes={"input_name": StringAttr(node.name)},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        # Tag as get_attr so MLIR→FX can reconstruct it correctly
        self.metadata[node.name]["_fx_op"] = "get_attr"
        self.metadata[node.name]["_fx_target"] = node.target

    def _convert_output(self, node: Node, block: Block) -> None:
        """Convert FX output → ``ad.graph_output``."""
        # FX output args can be a nested tuple; flatten for MLIR but store
        # the original structure template for faithful reconstruction.
        flat_args = self._flatten_args(node.args[0])
        operands = [self._resolve_operand(a) for a in flat_args if isinstance(a, Node)]
        op = AdGraphOutput.build(operands=[operands])
        block.add_op(op)
        # Store the original output structure as metadata so MLIR→FX can rebuild it
        self.metadata["__output_structure__"] = node.args[0]

    def _convert_call_function(self, node: Node, block: Block) -> None:
        """Route call_function nodes to precise or opaque lowering."""
        import operator as op_module

        if is_op(node, torch.ops.aten.add.Tensor):
            self._convert_add(node, block)
        elif is_op(node, torch.ops.aten.mul.Tensor):
            self._convert_binary_elementwise(node, block, AdMul)
        elif is_op(node, torch.ops.aten.sub.Tensor):
            self._convert_binary_elementwise(node, block, AdSub)
        elif is_op(node, [torch.ops.aten.neg, torch.ops.aten.neg.default]):
            self._convert_unary_elementwise(node, block, AdNeg)
        elif is_op(node, [torch.ops.aten.silu, torch.ops.aten.silu.default]):
            self._convert_unary_elementwise(node, block, AdSilu)
        elif is_op(node, [torch.ops.aten.gelu, torch.ops.aten.gelu.default]):
            self._convert_unary_elementwise(node, block, AdGelu)
        elif is_op(node, [torch.ops.aten.relu, torch.ops.aten.relu.default]):
            self._convert_unary_elementwise(node, block, AdRelu)
        elif is_op(node, [torch.ops.aten.tanh, torch.ops.aten.tanh.default]):
            self._convert_unary_elementwise(node, block, AdTanh)
        elif is_op(node, [torch.ops.aten.rsqrt, torch.ops.aten.rsqrt.default]):
            self._convert_unary_elementwise(node, block, AdRsqrt)
        elif is_op(node, [torch.ops.aten.sqrt, torch.ops.aten.sqrt.default]):
            self._convert_unary_elementwise(node, block, AdSqrt)
        elif is_op(node, torch.ops.aten.pow.Tensor_Scalar):
            self._convert_pow(node, block)
        elif is_op(node, torch.ops.aten.mean.dim):
            self._convert_mean(node, block)
        elif _is_rmsnorm(node):
            self._convert_rmsnorm(node, block)
        elif is_op(node, torch.ops.aten.to.dtype):
            self._convert_to_dtype(node, block)
        elif node.target is op_module.getitem:
            self._convert_getitem(node, block)
        else:
            self._convert_opaque(node, block)

    # ------------------------------------------------------------------
    # Precise lowerings
    # ------------------------------------------------------------------

    def _convert_add(self, node: Node, block: Block) -> None:
        """``aten.add.Tensor`` → ``ad.add``. Handles scalar second arg via splat."""
        result_type = _result_type_for_node(node)
        lhs = self._resolve_operand(node.args[0], block, result_type)
        rhs = self._resolve_operand(node.args[1], block, result_type)
        op = AdAdd.build(operands=[lhs, rhs], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_binary_elementwise(self, node: Node, block: Block, op_cls) -> None:
        """Generic binary elementwise: two inputs, one output. Handles scalar args via splat."""
        result_type = _result_type_for_node(node)
        lhs = self._resolve_operand(node.args[0], block, result_type)
        rhs = self._resolve_operand(node.args[1], block, result_type)
        op = op_cls.build(operands=[lhs, rhs], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_pow(self, node: Node, block: Block) -> None:
        """``aten.pow.Tensor_Scalar`` → ``ad.pow``."""
        base_val = self._resolve_operand(node.args[0])
        exponent = float(node.args[1])
        result_type = _result_type_for_node(node)
        op = AdPow.build(
            operands=[base_val],
            attributes={"exponent": FloatAttr(exponent, Float64Type())},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_mean(self, node: Node, block: Block) -> None:
        """``aten.mean.dim`` → ``ad.reduce_mean``."""
        input_val = self._resolve_operand(node.args[0])
        dims = node.args[1]  # list of ints
        keepdim = node.args[2] if len(node.args) > 2 else False
        # Only single-dim reductions are supported; fall back to opaque for multi-axis.
        if isinstance(dims, (list, tuple)):
            if len(dims) != 1:
                self._convert_opaque(node, block)
                return
            dim = dims[0]
        else:
            dim = dims
        result_type = _result_type_for_node(node)
        op = AdReduceMean.build(
            operands=[input_val],
            attributes={
                "dim": IntegerAttr(dim, IntegerType(64)),
                "keepdim": IntegerAttr(1 if keepdim else 0, IntegerType(1)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_unary_elementwise(self, node: Node, block: Block, op_cls) -> None:
        """Generic unary elementwise: one input, one output."""
        input_val = self._resolve_operand(node.args[0])
        result_type = _result_type_for_node(node)
        op = op_cls.build(operands=[input_val], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_rmsnorm(self, node: Node, block: Block) -> None:
        """Any rmsnorm variant → ``ad.rmsnorm`` (backend-agnostic)."""
        input_val = self._resolve_operand(node.args[0])
        weight_val = self._resolve_operand(node.args[1])
        eps_val = float(node.args[2]) if len(node.args) > 2 else 1e-5
        eps_attr = FloatAttr(eps_val, Float64Type())
        result_type = _result_type_for_node(node)
        # If no FakeTensor metadata (e.g., from match_rmsnorm_pattern),
        # use the input tensor's type — rmsnorm preserves shape.
        if result_type.get_shape() == (-1,) and isinstance(input_val.type, TensorType):
            result_type = input_val.type
        op = AdRMSNorm.build(
            operands=[input_val, weight_val],
            attributes={"eps": eps_attr},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_to_dtype(self, node: Node, block: Block) -> None:
        """``aten.to.dtype`` → ``ad.to_dtype``."""
        input_val = self._resolve_operand(node.args[0])
        # args[1] is the target dtype — can be torch.dtype or int
        target_dtype = node.args[1]
        if isinstance(target_dtype, int):
            dtype_int = target_dtype
        else:
            # torch.dtype → store a stable identifier (str hash works for round-trip)
            _DTYPE_TO_INT = {
                torch.float16: 5,
                torch.bfloat16: 15,
                torch.float32: 6,
                torch.float64: 7,
            }
            dtype_int = _DTYPE_TO_INT.get(target_dtype, hash(str(target_dtype)) % (2**31))
        dtype_attr = IntegerAttr(dtype_int, IntegerType(64))
        result_type = _result_type_for_node(node)
        op = AdToDtype.build(
            operands=[input_val],
            attributes={"target_dtype": dtype_attr},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_getitem(self, node: Node, block: Block) -> None:
        """Handle ``operator.getitem`` nodes that index into tuple results.

        If the container op in MLIR has multiple results, map this getitem to
        the corresponding result. Otherwise, fall through to opaque handling.
        """
        container = node.args[0]
        index = node.args[1]

        if isinstance(container, Node):
            # Check if the container op has a multi-result mapping
            multi_key = f"{container.name}__result_{index}"
            if multi_key in self._value_map:
                self._value_map[node.name] = self._value_map[multi_key]
                self._store_meta(node)
                return

            # If container has a single result, pass through to opaque
            # (getitem on a single-result op is valid for tensor indexing)
            if container.name in self._value_map:
                # Treat as opaque — it may be tensor indexing
                self._convert_opaque(node, block)
                return

        # Fallback to opaque
        self._convert_opaque(node, block)

    # ------------------------------------------------------------------
    # Opaque (catch-all)
    # ------------------------------------------------------------------

    def _convert_opaque(self, node: Node, block: Block) -> None:
        """Unmodeled FX op → ``ad.opaque``."""
        # Build MLIR operands from Node args, tracking arg structure for round-trip.
        # _args_template stores the original args structure with Node positions replaced
        # by integer indices into the MLIR operands list.
        operands = []
        operand_idx = [0]  # mutable counter

        def _build_template(arg):
            if isinstance(arg, Node) and arg.name in self._value_map:
                operands.append(self._value_map[arg.name])
                idx = operand_idx[0]
                operand_idx[0] += 1
                return ("__mlir_operand__", idx)
            elif isinstance(arg, (tuple, list)):
                return type(arg)(_build_template(a) for a in arg)
            else:
                return arg

        args_template = tuple(_build_template(a) for a in node.args)
        kwargs_template = {k: _build_template(v) for k, v in node.kwargs.items()}

        # Determine result types
        val = node.meta.get("val")
        result_types: List[TensorType] = []
        if isinstance(val, torch.Tensor):
            result_types = [tensor_type_from_meta(val)]
        elif isinstance(val, (tuple, list)):
            for v in val:
                if isinstance(v, torch.Tensor):
                    result_types.append(tensor_type_from_meta(v))
        # If still empty but node has users, create a dummy result so downstream
        # nodes can reference it (prevents "value not found" errors).
        if not result_types and node.users:
            from xdsl.dialects.builtin import Float32Type

            result_types = [TensorType(Float32Type(), [-1])]

        # Build a human-readable op name from the FX target
        op_name = (
            str(node.target)
            if not callable(node.target)
            else getattr(node.target, "__name__", str(node.target))
        )

        op = AdOpaque.build(
            operands=[operands],
            attributes={
                "op_name": StringAttr(op_name),
                "node_key": StringAttr(node.name),
            },
            result_types=[result_types],
        )
        block.add_op(op)

        # Map results
        if len(op.outputs) == 1:
            self._value_map[node.name] = op.outputs[0]
        elif len(op.outputs) > 1:
            for i, res in enumerate(op.outputs):
                self._value_map[f"{node.name}__result_{i}"] = res
            # Also map the base name to first result for simple consumers
            self._value_map[node.name] = op.outputs[0]

        self._store_meta(node)
        # Store original target and args/kwargs templates for round-trip reconstruction
        # (must be after _store_meta which initializes the metadata dict)
        self.metadata[node.name]["_original_target"] = node.target
        self.metadata[node.name]["_args_template"] = args_template
        self.metadata[node.name]["_kwargs_template"] = kwargs_template

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_operand(self, arg, block: Block = None, result_type=None) -> SSAValue:
        """Look up the MLIR SSAValue for a given FX node or constant.

        For scalar constants (int, float), emits an ``AdSplat`` op to the given
        block and returns its result. This enables ``aten.add(tensor, 1e-5)``
        to lower to ``ad.splat(1e-5) → ad.add(tensor, splat)`` instead of
        falling through to ``ad.opaque``.
        """
        if isinstance(arg, Node):
            val = self._value_map.get(arg.name)
            if val is None:
                raise ValueError(
                    f"MLIR value not found for FX node '{arg.name}'. "
                    "This may indicate an unsupported node type or ordering issue."
                )
            return val
        if isinstance(arg, (int, float)) and block is not None and result_type is not None:
            # Scalar constant → emit AdSplat
            scalar_val = float(arg)
            splat_type = result_type  # broadcast to match the result shape
            splat_op = AdSplat.build(
                attributes={"value": FloatAttr(scalar_val, Float64Type())},
                result_types=[splat_type],
            )
            block.add_op(splat_op)
            return splat_op.output
        raise TypeError(f"Cannot resolve non-Node operand to SSAValue: {type(arg)}")

    def _store_meta(self, node: Node) -> None:
        """Store FX node metadata in the side-table for round-trip."""
        self.metadata[node.name] = dict(node.meta)

    @staticmethod
    def _flatten_args(args) -> list:
        """Flatten potentially nested args (tuples/lists) into a flat list."""
        if isinstance(args, (tuple, list)):
            flat = []
            for a in args:
                flat.extend(FXToMLIRConverter._flatten_args(a))
            return flat
        return [args]

    # Keep old name as alias for backward compat within the class
    _flatten_output_args = _flatten_args

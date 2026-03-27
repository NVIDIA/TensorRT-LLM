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

"""MLIR (xDSL) → FX graph converter.

Walks an xDSL ``ModuleOp`` and reconstructs a PyTorch FX ``GraphModule``.
Fused ops are represented as ``ad.opaque`` with metadata pointing to
auto-generated Triton kernels registered as ``torch.library.custom_op``.
"""

import operator
from typing import Any, Callable, Dict, Optional

import torch
from torch.fx import Graph, GraphModule, Node
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from .dialect import (
    AdAdd,
    AdCast,
    AdExp,
    AdGatedRMSNorm,
    AdGelu,
    AdGraphInput,
    AdGraphOutput,
    AdMul,
    AdNeg,
    AdOpaque,
    AdPow,
    AdReduceMean,
    AdReduceSum,
    AdRelu,
    AdRMSNorm,
    AdRsqrt,
    AdSigmoid,
    AdSilu,
    AdSoftplus,
    AdSplat,
    AdSqrt,
    AdSub,
    AdTanh,
    AdToDtype,
    mlir_to_torch_dtype,
)


class MLIRToFXConverter:
    """Convert an xDSL ``ModuleOp`` back to an FX ``GraphModule``.

    Args:
        original_gm: The original FX GraphModule (used for module references
            and parameter access).
    """

    def __init__(self, original_gm: GraphModule, **kwargs):
        self._original_gm = original_gm
        # MLIR SSAValue → FX Node
        self._value_map: Dict[SSAValue, Node] = {}
        # Old FX node name → new FX Node (for opaque arg reconstruction)
        self._node_name_map: Dict[str, Node] = {}

    def convert(
        self,
        mlir_module: ModuleOp,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> GraphModule:
        """Convert the MLIR module to an FX GraphModule.

        Args:
            mlir_module: The xDSL module to convert.
            metadata: Optional node metadata side-table (from ``FXToMLIRConverter``).

        Returns:
            A new FX GraphModule.
        """
        metadata = metadata or {}

        # Preserve the original graph's CodeGen (carries pytree_info needed by
        # downstream transforms like gather_logits_before_lm_head).
        original_codegen = self._original_gm.graph._codegen
        graph = Graph()
        graph._codegen = original_codegen

        # Walk MLIR ops in order (they are in topological order within the block)
        for mlir_op in mlir_module.body.block.ops:
            try:
                self._convert_op(mlir_op, graph, metadata)
            except ValueError as e:
                op_name = mlir_op.name if hasattr(mlir_op, "name") else type(mlir_op).__name__
                node_key = ""
                if isinstance(mlir_op, AdOpaque):
                    node_key = f" node_key='{mlir_op.node_key.data}'"
                raise ValueError(f"Error converting MLIR op '{op_name}'{node_key}: {e}") from e

        # Build _node_name_map for ALL nodes by scanning value_map.
        # This ensures opaque arg reconstruction can find nodes by their
        # original FX names (including precisely-lowered ops like add/rmsnorm).
        self._build_node_name_map_from_graph(graph)

        # Build the new GraphModule reusing the original module's parameters.
        # Copy callable attributes (e.g. get_input_embeddings) from the
        # original GM that aren't present on the new one, so downstream code
        # that calls these methods still works.
        new_gm = GraphModule(self._original_gm, graph)
        for attr_name in dir(self._original_gm):
            if (
                not attr_name.startswith("_")
                and callable(getattr(self._original_gm, attr_name, None))
                and not hasattr(new_gm, attr_name)
            ):
                setattr(new_gm, attr_name, getattr(self._original_gm, attr_name))
        return new_gm

    # ------------------------------------------------------------------
    # Op dispatch
    # ------------------------------------------------------------------

    def _convert_op(
        self,
        mlir_op: Operation,
        graph: Graph,
        metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Dispatch MLIR op to the appropriate FX reconstruction handler."""
        if isinstance(mlir_op, AdGraphInput):
            self._convert_graph_input(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdAdd):
            self._convert_add(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdMul):
            self._convert_binary_elementwise(mlir_op, graph, metadata, torch.ops.aten.mul.Tensor)
        elif isinstance(mlir_op, AdSub):
            self._convert_binary_elementwise(mlir_op, graph, metadata, torch.ops.aten.sub.Tensor)
        elif isinstance(mlir_op, AdNeg):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.neg.default)
        elif isinstance(mlir_op, AdSilu):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.silu.default)
        elif isinstance(mlir_op, AdGelu):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.gelu.default)
        elif isinstance(mlir_op, AdRelu):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.relu.default)
        elif isinstance(mlir_op, AdTanh):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.tanh.default)
        elif isinstance(mlir_op, AdSigmoid):
            self._convert_unary_elementwise(
                mlir_op, graph, metadata, torch.ops.aten.sigmoid.default
            )
        elif isinstance(mlir_op, AdExp):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.exp.default)
        elif isinstance(mlir_op, AdSoftplus):
            self._convert_softplus(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdRsqrt):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.rsqrt.default)
        elif isinstance(mlir_op, AdSqrt):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.sqrt.default)
        elif isinstance(mlir_op, AdReduceMean):
            self._convert_reduce(mlir_op, graph, metadata, torch.ops.aten.mean.dim)
        elif isinstance(mlir_op, AdReduceSum):
            self._convert_reduce(mlir_op, graph, metadata, torch.ops.aten.sum.dim_IntList)
        elif isinstance(mlir_op, AdSplat):
            self._convert_splat(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdCast):
            self._convert_cast(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdPow):
            self._convert_pow(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdRMSNorm):
            self._convert_rmsnorm(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdGatedRMSNorm):
            self._convert_gated_rmsnorm(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdToDtype):
            self._convert_to_dtype(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdGraphOutput):
            self._convert_graph_output(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdOpaque):
            self._convert_opaque(mlir_op, graph, metadata)

    # ------------------------------------------------------------------
    # Individual converters
    # ------------------------------------------------------------------

    def _convert_graph_input(self, op: AdGraphInput, graph: Graph, metadata: dict) -> None:
        """Reconstruct FX placeholder or get_attr from ``ad.graph_input``."""
        name = op.input_name.data
        node_meta = metadata.get(name, {})
        if node_meta.get("_fx_op") == "get_attr":
            # Reconstruct as get_attr (parameter/buffer access)
            target = node_meta.get("_fx_target", name)
            node = graph.get_attr(target)
        else:
            node = graph.placeholder(name)
        self._restore_meta(node, name, metadata)
        self._map_value(op.output, node, name)

    def _convert_add(self, op: AdAdd, graph: Graph, metadata: dict) -> None:
        """Reconstruct ``aten.add.Tensor`` from ``ad.add``."""
        lhs = self._resolve(op.lhs)
        rhs = self._resolve(op.rhs)
        node = graph.call_function(torch.ops.aten.add.Tensor, args=(lhs, rhs))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_binary_elementwise(self, op, graph: Graph, metadata: dict, aten_target) -> None:
        """Reconstruct a binary elementwise aten op from an ``ad.*`` op."""
        lhs = self._resolve(op.lhs)
        rhs = self._resolve(op.rhs)
        node = graph.call_function(aten_target, args=(lhs, rhs))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_unary_elementwise(self, op, graph: Graph, metadata: dict, aten_target) -> None:
        """Reconstruct a unary elementwise aten op from an ``ad.*`` op."""
        input_node = self._resolve(op.input)
        node = graph.call_function(aten_target, args=(input_node,))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_reduce(self, op, graph: Graph, metadata: dict, aten_target) -> None:
        """Reconstruct a reduction aten op from ``ad.reduce_mean`` or ``ad.reduce_sum``."""
        input_node = self._resolve(op.input)
        dim = op.dim.value.data
        keepdim = bool(op.keepdim.value.data)
        node = graph.call_function(aten_target, args=(input_node, [dim], keepdim))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_splat(self, op, graph: Graph, metadata: dict) -> None:
        """Reconstruct a scalar constant from ``ad.splat``."""
        from xdsl.dialects.builtin import FloatAttr as FA

        val = op.value
        scalar = val.value.data if isinstance(val, FA) else float(str(val))
        # Create a constant node — splat values appear as scalar args downstream
        node = graph.call_function(torch.ops.aten.scalar_tensor.default, args=(scalar,))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_cast(self, op, graph: Graph, metadata: dict) -> None:
        """Reconstruct dtype cast from ``ad.cast``."""
        input_node = self._resolve(op.input)
        result_type = op.output.type
        if hasattr(result_type, "element_type"):
            target_dtype = mlir_to_torch_dtype(result_type.element_type)
        else:
            target_dtype = op.target_dtype.value.data
        node = graph.call_function(torch.ops.aten.to.dtype, args=(input_node, target_dtype))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_softplus(self, op, graph: Graph, metadata: dict) -> None:
        """Reconstruct softplus from ``ad.softplus`` (beta=1, threshold=20)."""
        input_node = self._resolve(op.input)
        node = graph.call_function(torch.ops.aten.softplus.default, args=(input_node,))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_pow(self, op, graph: Graph, metadata: dict) -> None:
        """Reconstruct pow from ``ad.pow``."""
        from xdsl.dialects.builtin import FloatAttr as FA

        base_node = self._resolve(op.base)
        exp_attr = op.exponent
        exp_val = exp_attr.value.data if isinstance(exp_attr, FA) else float(str(exp_attr))
        node = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, args=(base_node, exp_val))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_rmsnorm(self, op: AdRMSNorm, graph: Graph, metadata: dict) -> None:
        """Reconstruct rmsnorm call from ``ad.rmsnorm``.

        Maps back to ``flashinfer_rms_norm`` by default (the standard backend).
        """
        input_node = self._resolve(op.input)
        weight_node = self._resolve(op.weight)
        eps = op.eps.value.data
        node = graph.call_function(
            torch.ops.auto_deploy.flashinfer_rms_norm,
            args=(input_node, weight_node, eps),
        )
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_gated_rmsnorm(self, op: AdGatedRMSNorm, graph: Graph, metadata: dict) -> None:
        """Reconstruct gated rmsnorm call from ``ad.gated_rmsnorm``.

        Maps back to ``triton_rmsnorm_gated`` for unfused instances.
        """
        input_node = self._resolve(op.input)
        weight_node = self._resolve(op.weight)
        gate_node = self._resolve(op.gate)
        eps = op.eps.value.data
        group_size = op.group_size.value.data
        norm_before_gate = bool(op.norm_before_gate.value.data)
        node = graph.call_function(
            torch.ops.auto_deploy.triton_rmsnorm_gated,
            args=(input_node, weight_node, gate_node, eps, group_size, norm_before_gate),
        )
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_to_dtype(self, op: AdToDtype, graph: Graph, metadata: dict) -> None:
        """Reconstruct ``aten.to.dtype`` from ``ad.to_dtype``."""
        from .dialect import TensorType, mlir_to_torch_dtype

        input_node = self._resolve(op.input)
        # Recover target dtype from the MLIR output type for full type coverage
        # (int, bool, fp8, etc.), falling back to the stored integer for plain floats.
        result_type = op.output.type
        if isinstance(result_type, TensorType):
            target_dtype = mlir_to_torch_dtype(result_type.element_type)
        else:
            dtype_int = op.target_dtype.value.data
            _INT_TO_DTYPE = {
                5: torch.float16,
                15: torch.bfloat16,
                6: torch.float32,
                7: torch.float64,
            }
            target_dtype = _INT_TO_DTYPE.get(dtype_int, dtype_int)
        node = graph.call_function(torch.ops.aten.to.dtype, args=(input_node, target_dtype))
        self._restore_meta_from_op(node, op, metadata)
        self._map_value(op.output, node)

    def _convert_graph_output(self, op: AdGraphOutput, graph: Graph, metadata: dict) -> None:
        """Reconstruct FX output node from ``ad.graph_output``.

        Uses the original output structure stored by FXToMLIRConverter to
        faithfully reconstruct nested tuples expected by the graph's CodeGen.
        """
        # Build a flat list of resolved FX nodes for the MLIR operands
        flat_nodes = [self._resolve(inp) for inp in op.inputs]

        # If the original output structure was stored, reconstruct it
        output_structure = metadata.get("__output_structure__")
        if output_structure is not None:
            idx = [0]  # mutable counter for flat_nodes consumption

            def _rebuild(template):
                if isinstance(template, Node):
                    node = flat_nodes[idx[0]]
                    idx[0] += 1
                    return node
                elif isinstance(template, (tuple, list)):
                    rebuilt = [_rebuild(item) for item in template]
                    return type(template)(rebuilt)
                else:
                    # Non-Node constants (None, int, etc.) — pass through
                    return template

            result = _rebuild(output_structure)
        else:
            result = tuple(flat_nodes)

        graph.output(result)

    def _convert_opaque(self, op: AdOpaque, graph: Graph, metadata: dict) -> None:
        """Reconstruct original FX node from ``ad.opaque`` using stored metadata."""
        node_key = op.node_key.data
        meta = metadata.get(node_key, {})

        original_target = meta.get("_original_target")
        args_template = meta.get("_args_template")
        kwargs_template = meta.get("_kwargs_template", {})

        # Resolve MLIR operands to new FX nodes
        mlir_operands = [self._resolve(inp) for inp in op.inputs]

        if original_target is not None and args_template is not None:
            # Reconstruct args from template, substituting MLIR operand indices
            new_args = self._instantiate_template(args_template, mlir_operands)
            new_kwargs = self._instantiate_template(kwargs_template, mlir_operands)
            node = graph.call_function(original_target, args=new_args, kwargs=new_kwargs)
        else:
            # Fallback: use MLIR operands as flat args
            op_name = op.op_name.data
            target = self._resolve_target(op_name) or _opaque_placeholder
            node = graph.call_function(target, args=tuple(mlir_operands))

        self._restore_meta(node, node_key, metadata)

        # Map outputs and register by node_key for name-based lookup.
        # Generated fused kernels always return a tuple (even for single output),
        # so they always need getitem extraction.
        is_fused_kernel = node_key.startswith("mlir_fused_")
        if len(op.outputs) == 1 and not is_fused_kernel:
            self._map_value(op.outputs[0], node, node_key)
        elif len(op.outputs) >= 1:
            # Also map the base node by name
            self._node_name_map[node_key] = node
            # Propagate per-element "val" metadata to getitem nodes
            node_val = node.meta.get("val")
            for i, res in enumerate(op.outputs):
                getitem_node = graph.call_function(operator.getitem, args=(node, i))
                if isinstance(node_val, (tuple, list)) and i < len(node_val):
                    getitem_node.meta["val"] = node_val[i]
                self._map_value(res, getitem_node)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_node_name_map_from_graph(self, graph: Graph) -> None:
        """Build a complete name→node map from the new graph's nodes.

        FX auto-assigns names to nodes. We match them to original names by
        scanning the new graph and matching against the names already recorded
        in `_node_name_map` from opaque/graph_input conversions. For any new
        node whose name matches a metadata key, we record it.
        """
        for node in graph.nodes:
            if node.name and node.name not in self._node_name_map:
                self._node_name_map[node.name] = node

    def _map_value(self, ssa_val: SSAValue, fx_node: Node, name: str = "") -> None:
        """Register an MLIR value → FX node mapping (and name-based map for opaque reconstruction)."""
        self._value_map[ssa_val] = fx_node
        if name:
            self._node_name_map[name] = fx_node

    @staticmethod
    def _instantiate_template(template, mlir_operands: list):
        """Rebuild args/kwargs from a template, substituting MLIR operand markers.

        The template contains ``("__mlir_operand__", idx)`` markers at positions
        where FX Node args were. These are replaced with the corresponding
        new FX nodes from ``mlir_operands``.
        """

        def _rebuild(item):
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__mlir_operand__":
                idx = item[1]
                return mlir_operands[idx]
            elif isinstance(item, tuple):
                return tuple(_rebuild(a) for a in item)
            elif isinstance(item, list):
                return [_rebuild(a) for a in item]
            elif isinstance(item, dict):
                return {k: _rebuild(v) for k, v in item.items()}
            else:
                return item

        return _rebuild(template)

    def _resolve(self, ssa_val: SSAValue) -> Node:
        """Look up the FX Node for an MLIR SSAValue."""
        node = self._value_map.get(ssa_val)
        if node is None:
            producer = ssa_val.owner
            producer_name = producer.name if hasattr(producer, "name") else type(producer).__name__
            raise ValueError(
                f"FX node not found for MLIR value produced by '{producer_name}' "
                f"({type(producer).__name__}). "
                "This may indicate an unsupported op or ordering issue."
            )
        return node

    @staticmethod
    def _restore_meta(node: Node, key: str, metadata: Dict[str, Dict[str, Any]]) -> None:
        """Restore FX node.meta from the metadata side-table."""
        if key in metadata:
            node.meta.update(metadata[key])

    @staticmethod
    def _restore_meta_from_op(
        node: Node, mlir_op: Operation, metadata: Dict[str, Dict[str, Any]]
    ) -> None:
        """Try to restore metadata using various key strategies."""
        # For precisely-lowered ops, try the original node name from metadata
        # We search metadata for an entry whose "val" shape matches
        # For now, no metadata restoration for precisely-lowered ops
        # (shape prop will recompute it)
        pass

    @staticmethod
    def _resolve_target(op_name: str) -> Optional[Callable]:
        """Try to resolve an op name string to a callable target."""
        # Handle common patterns like "aten.add.Tensor"
        parts = op_name.split(".")
        if len(parts) >= 2 and parts[0] == "aten":
            try:
                op = getattr(torch.ops.aten, parts[1])
                if len(parts) >= 3:
                    op = getattr(op, parts[2])
                return op
            except AttributeError:
                pass
        return None


def _opaque_placeholder(*args, **kwargs):
    """Placeholder function for opaque ops that couldn't be resolved."""
    raise RuntimeError(
        "Opaque op placeholder was called. This indicates an MLIR → FX "
        "conversion issue where the original op target could not be resolved."
    )

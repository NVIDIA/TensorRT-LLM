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
Fused ops (e.g., ``ad.fused_add_rmsnorm``) are mapped to either pre-existing
custom ops or Triton-generated kernels depending on the codegen mode.
"""

import operator
from typing import Any, Callable, Dict, Literal, Optional

import torch
from torch.fx import Graph, GraphModule, Node
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from .codegen.triton_emitter import TritonCodegen
from .dialect import (
    AdAdd,
    AdFusedAddRMSNorm,
    AdGelu,
    AdGraphInput,
    AdGraphOutput,
    AdMul,
    AdNeg,
    AdOpaque,
    AdRelu,
    AdRMSNorm,
    AdRsqrt,
    AdSilu,
    AdSqrt,
    AdSub,
    AdTanh,
    AdToDtype,
)


class MLIRToFXConverter:
    """Convert an xDSL ``ModuleOp`` back to an FX ``GraphModule``.

    Args:
        original_gm: The original FX GraphModule (used for module references
            and parameter access).
        codegen_mode: ``"preexisting"`` to map fused ops to existing custom ops,
            ``"generate"`` to use Triton-generated kernels.
    """

    def __init__(
        self,
        original_gm: GraphModule,
        codegen_mode: Literal["preexisting", "generate"] = "preexisting",
    ):
        self._original_gm = original_gm
        self._codegen = TritonCodegen(mode=codegen_mode)
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
            self._convert_op(mlir_op, graph, metadata)

        # Build _node_name_map for ALL nodes by scanning value_map.
        # This ensures opaque arg reconstruction can find nodes by their
        # original FX names (including precisely-lowered ops like add/rmsnorm).
        self._build_node_name_map_from_graph(graph)

        # Build the new GraphModule reusing the original module's parameters
        new_gm = GraphModule(self._original_gm, graph)
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
        elif isinstance(mlir_op, AdRsqrt):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.rsqrt.default)
        elif isinstance(mlir_op, AdSqrt):
            self._convert_unary_elementwise(mlir_op, graph, metadata, torch.ops.aten.sqrt.default)
        elif isinstance(mlir_op, AdRMSNorm):
            self._convert_rmsnorm(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdToDtype):
            self._convert_to_dtype(mlir_op, graph, metadata)
        elif isinstance(mlir_op, AdFusedAddRMSNorm):
            self._convert_fused_add_rmsnorm(mlir_op, graph, metadata)
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

    def _convert_to_dtype(self, op: AdToDtype, graph: Graph, metadata: dict) -> None:
        """Reconstruct ``aten.to.dtype`` from ``ad.to_dtype``."""
        input_node = self._resolve(op.input)
        dtype_int = op.target_dtype.value.data
        # Convert the stored int back to a torch.dtype
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

    def _convert_fused_add_rmsnorm(
        self, op: AdFusedAddRMSNorm, graph: Graph, metadata: dict
    ) -> None:
        """Reconstruct fused add+rmsnorm as a custom op call.

        Uses TritonCodegen to select between pre-existing (FlashInfer) or
        generated (Triton) implementations.
        """
        x_node = self._resolve(op.x)
        residual_node = self._resolve(op.residual)
        weight_node = self._resolve(op.weight)
        eps = op.eps.value.data

        impl_fn = self._codegen.get_fused_add_rmsnorm_impl(op)

        fused_node = graph.call_function(impl_fn, args=(x_node, residual_node, weight_node, eps))

        # Extract tuple results via getitem (matches existing FX convention)
        norm_out = graph.call_function(operator.getitem, args=(fused_node, 0))
        add_out = graph.call_function(operator.getitem, args=(fused_node, 1))

        self._map_value(op.norm_result, norm_out)
        self._map_value(op.add_result, add_out)

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

        # Map outputs and register by node_key for name-based lookup
        if len(op.outputs) == 1:
            self._map_value(op.outputs[0], node, node_key)
        elif len(op.outputs) > 1:
            # Also map the base node by name
            self._node_name_map[node_key] = node
            for i, res in enumerate(op.outputs):
                getitem_node = graph.call_function(operator.getitem, args=(node, i))
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
            raise ValueError(
                "FX node not found for MLIR value. "
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

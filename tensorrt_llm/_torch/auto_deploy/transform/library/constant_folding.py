# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generalized constant folding transform for FX graphs.

This transform identifies subgraphs where all inputs are constants (parameters/buffers)
and folds them into materialized constant tensors at compile time.

Algorithm:
1. Topological traversal of the graph
2. Forward propagation of "constantness" - a node is constant if all its inputs are constant
3. Identify fold boundaries where constant nodes have dynamic users
4. Evaluate constant subgraphs and replace with materialized constants
5. Remove unused parameters and buffers
"""

import operator
from typing import Any, Callable, Dict, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# Pure, deterministic operations that can be safely folded
_FOLDABLE_ATEN_OPS: Set[Callable] = {
    # Unary math
    torch.ops.aten.exp.default,
    torch.ops.aten.log.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.abs.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.silu.default,
    # Binary math
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.pow.Tensor_Tensor,
    # Reshaping
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.t.default,
    # Type conversion
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    # Indexing (with constant indices)
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    operator.getitem,
    # Reductions (on constant tensors)
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean.dim,
}

_FOLDABLE_TORCH_OPS: Set[Callable] = {
    torch.exp,
    torch.log,
    torch.neg,
    torch.abs,
    torch.sqrt,
    torch.rsqrt,
    torch.sin,
    torch.cos,
    torch.tanh,
    torch.sigmoid,
    torch.relu,
    operator.neg,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.getitem,
}


def _is_foldable_op(node: Node) -> bool:
    """Check if a node's operation is safe to fold."""
    if node.op != "call_function":
        return False
    target = node.target
    return target in _FOLDABLE_ATEN_OPS or target in _FOLDABLE_TORCH_OPS


def _is_foldable_method(node: Node) -> bool:
    """Check if a method call is safe to fold."""
    if node.op != "call_method":
        return False
    method_name = node.target
    foldable_methods = {
        "exp",
        "log",
        "neg",
        "abs",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "view",
        "reshape",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
        "expand",
        "contiguous",
        "clone",
        "float",
        "half",
        "bfloat16",
        "to",
        "sum",
        "mean",
    }
    return method_name in foldable_methods


def _get_attr_value(gm: GraphModule, target: str) -> torch.Tensor:
    """Retrieve the value of a get_attr node."""
    parts = target.split(".")
    obj = gm
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _evaluate_node(
    gm: GraphModule,
    node: Node,
    env: Dict[Node, Any],
) -> Any:
    """Evaluate a single node given the environment of computed values."""
    if node.op == "get_attr":
        return _get_attr_value(gm, node.target)

    elif node.op == "call_function":
        args = tuple(env[arg] if isinstance(arg, Node) else arg for arg in node.args)
        kwargs = {k: env[v] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
        return node.target(*args, **kwargs)

    elif node.op == "call_method":
        self_obj = env[node.args[0]] if isinstance(node.args[0], Node) else node.args[0]
        method = getattr(self_obj, node.target)
        args = tuple(env[arg] if isinstance(arg, Node) else arg for arg in node.args[1:])
        kwargs = {k: env[v] if isinstance(v, Node) else v for k, v in node.kwargs.items()}
        return method(*args, **kwargs)

    else:
        raise ValueError(f"Cannot evaluate node with op: {node.op}")


def _classify_and_evaluate_constants(
    gm: GraphModule,
) -> Tuple[Dict[Node, bool], Dict[Node, Any]]:
    """Classify nodes and evaluate constant nodes in a single forward pass.

    Since the graph is topologically sorted, we can classify and evaluate
    all nodes in one traversal. When we visit a node, all its inputs have
    already been processed.

    A node is constant if:
    - It's a get_attr (parameter/buffer)
    - It's a foldable operation with all constant inputs

    Returns:
        Tuple of:
        - is_constant: Dict mapping nodes to True (constant) or False (dynamic)
        - constant_values: Dict mapping constant nodes to their evaluated tensor values
    """
    is_constant: Dict[Node, bool] = {}
    constant_values: Dict[Node, Any] = {}

    with torch.no_grad():
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                # Model inputs are dynamic
                is_constant[node] = False

            elif node.op == "get_attr":
                # Parameters and buffers are constants
                is_constant[node] = True
                try:
                    constant_values[node] = _get_attr_value(gm, node.target)
                except Exception as e:
                    ad_logger.warning(f"Failed to get attr {node.target}: {e}")
                    is_constant[node] = False

            elif node.op == "output":
                # Output nodes are always considered dynamic
                is_constant[node] = False

            elif _is_foldable_op(node) or _is_foldable_method(node):
                # A foldable op is constant if ALL its inputs are constant
                # Since the torch graph is always top-sorted, we can just check all_input_nodes
                all_inputs_constant = all(
                    is_constant.get(inp, False) for inp in node.all_input_nodes
                )
                is_constant[node] = all_inputs_constant

                # If constant, evaluate immediately
                if all_inputs_constant:
                    try:
                        constant_values[node] = _evaluate_node(gm, node, constant_values)
                    except Exception as e:
                        ad_logger.warning(f"Failed to evaluate node {node}: {e}")
                        is_constant[node] = False

            else:
                # Non-foldable operations are dynamic
                is_constant[node] = False

    return is_constant, constant_values


def _find_fold_points(gm: GraphModule, is_constant: Dict[Node, bool]) -> list[Node]:
    """Find nodes that should be materialized as constants.

    A fold point is a constant node that has at least one dynamic user.
    This is the boundary where we need to materialize the constant value.
    """
    fold_points = []

    for node in gm.graph.nodes:
        if not is_constant.get(node, False):
            continue

        # Skip get_attr nodes - they're already materialized
        if node.op == "get_attr":
            continue

        # Check if any user is dynamic
        has_dynamic_user = any(not is_constant.get(user, False) for user in node.users)

        if has_dynamic_user:
            fold_points.append(node)

    return fold_points


def _generate_constant_name(gm: GraphModule, fold_point: Node, counter: int) -> str:
    """Generate a unique name for a folded constant."""
    # Try to create a descriptive name based on the original computation
    base_name = f"_folded_const_{counter}"

    # Ensure uniqueness
    name = base_name
    suffix = 0
    while hasattr(gm, name):
        suffix += 1
    name = f"{base_name}_{suffix}"

    return name


def _replace_with_constant(
    gm: GraphModule,
    fold_point: Node,
    value: torch.Tensor,
    name: str,
) -> None:
    """Replace a fold point with a materialized constant."""
    # Register the computed value as a buffer (non-trainable constant)
    gm.register_buffer(name, value)

    # Create a new get_attr node
    with gm.graph.inserting_before(fold_point):
        new_node = gm.graph.get_attr(name)

    # Copy metadata if present
    new_node.meta = fold_point.meta.copy() if fold_point.meta else {}

    # Replace all uses of the fold point with the new constant
    fold_point.replace_all_uses_with(new_node)


def _del_attr_by_name(obj: Any, name: str) -> None:
    """Delete a nested attribute by dotted name."""
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    delattr(obj, parts[-1])


def _remove_unused_params_and_buffers(gm: GraphModule) -> int:
    """Remove parameters and buffers that are no longer referenced in the graph."""
    # Collect all get_attr targets still in use
    used_attrs = {str(node.target) for node in gm.graph.nodes if node.op == "get_attr"}

    removed_count = 0

    def _maybe_remove(name: str) -> None:
        nonlocal removed_count
        if name in used_attrs:
            return
        try:
            _del_attr_by_name(gm, name)
            removed_count += 1
            ad_logger.debug(f"Removed unused attribute: {name}")
        except AttributeError:
            ad_logger.warning(f"Failed to delete unused attribute {name} from GraphModule.")

    # Check all parameters and buffers
    for name, _ in list(gm.named_parameters()):
        _maybe_remove(name)
    for name, _ in list(gm.named_buffers()):
        _maybe_remove(name)

    return removed_count


@TransformRegistry.register("constant_folding")
class ConstantFolding(BaseTransform):
    """Generalized constant folding transform."""

    # Class-level configuration defaults
    MAX_FOLD_NUMEL: int = 10_000_000  # Maximum elements in a folded constant
    MAX_FOLD_BYTES: Optional[int] = None  # Maximum size in bytes (None = no limit)

    def _should_fold(self, value: torch.Tensor) -> bool:
        """Check if a tensor should be folded based on size constraints."""
        if value.numel() > self.MAX_FOLD_NUMEL:
            ad_logger.debug(
                f"Skipping fold: tensor has {value.numel()} elements (max: {self.MAX_FOLD_NUMEL})"
            )
            return False

        if self.MAX_FOLD_BYTES is not None:
            size_bytes = value.numel() * value.element_size()
            if size_bytes > self.MAX_FOLD_BYTES:
                ad_logger.debug(
                    f"Skipping fold: tensor is {size_bytes} bytes (max: {self.MAX_FOLD_BYTES})"
                )
                return False

        return True

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Apply constant folding to the graph."""
        is_constant, constant_values = _classify_and_evaluate_constants(gm)

        num_constant_nodes = sum(1 for v in is_constant.values() if v)
        ad_logger.debug(f"Found {num_constant_nodes} constant nodes in graph")

        # Find fold points (constant nodes with dynamic users)
        fold_points = _find_fold_points(gm, is_constant)
        ad_logger.debug(f"Found {len(fold_points)} fold points")

        if not fold_points:
            return gm, TransformInfo(
                skipped=False,
                num_matches=0,
                is_clean=True,
                has_valid_shapes=True,
            )

        # Replace each fold point with its materialized constant
        num_folded = 0
        for i, fold_point in enumerate(fold_points):
            value = constant_values.get(fold_point)
            if value is None:
                ad_logger.warning(f"Fold point {fold_point} was not evaluated, skipping")
                continue

            # Check size constraints
            if not self._should_fold(value):
                continue

            # Generate a name for the new constant
            name = _generate_constant_name(gm, fold_point, i)

            # Replace with materialized constant
            _replace_with_constant(gm, fold_point, value, name)

            num_folded += 1
            ad_logger.debug(
                f"Folded {fold_point.target} -> {name} (shape: {value.shape}, dtype: {value.dtype})"
            )

        if num_folded > 0:
            gm.graph.eliminate_dead_code()
            num_removed = _remove_unused_params_and_buffers(gm)
            gm.recompile()
            ad_logger.info(
                f"Constant folding: folded {num_folded} subgraphs, "
                f"removed {num_removed} unused params/buffers"
            )
        else:
            ad_logger.debug("Constant folding: no subgraphs folded")

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_folded,
            is_clean=num_folded == 0,
            has_valid_shapes=True,
        )

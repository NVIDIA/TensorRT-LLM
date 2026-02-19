# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformation for fusing Add + (optional Cast) + RMSNorm via direct FX graph manipulation."""

import operator
from typing import List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import flashinfer_fused_add_rms_norm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("fuse_add_rms_norm")
class FuseAddRMSNorm(BaseTransform):
    """Fuse (add + optional cast + RMSNorm) into one fused op.

    Uses direct FX graph manipulation instead of the inductor pattern matcher
    to correctly handle patterns where intermediate nodes (add, rms_norm) have
    multiple users in the graph.

    Pattern 1 (without cast):
        %add = aten.add(%x, %residual)
        %norm = flashinfer_rms_norm(%add, %weight, eps)

    Pattern 2 (with cast):
        %add = aten.add(%x, %residual)
        %cast = aten.to.dtype(%add, bfloat16)
        %norm = flashinfer_rms_norm(%cast, %weight, eps)

    Both are replaced with:
        %fused = flashinfer_fused_add_rms_norm(%x, %residual, %weight, eps)
        %norm_out = getitem(%fused, 0)   # norm result  (replaces %norm)
        %add_out  = getitem(%fused, 1)   # add result   (replaces %add)
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        num_matches = 0

        # --- Step 1: collect (add_node, optional cast_node, norm_node) triples ---
        matches: List[Tuple[Node, Optional[Node], Node]] = []

        for node in graph.nodes:
            # Match flashinfer_rms_norm (handles both overload packet and .default)
            if not is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
                continue

            input_to_norm = node.args[0]
            cast_node: Optional[Node] = None

            # Check for an optional aten.to.dtype cast between add and norm
            if isinstance(input_to_norm, Node) and is_op(input_to_norm, torch.ops.aten.to.dtype):
                cast_node = input_to_norm
                input_to_norm = cast_node.args[0]

            # The (possibly unwrapped) input must be an aten.add.Tensor
            if not isinstance(input_to_norm, Node) or not is_op(
                input_to_norm, torch.ops.aten.add.Tensor
            ):
                continue

            add_node = input_to_norm
            matches.append((add_node, cast_node, node))

        # --- Step 2: apply fusions ---
        erased: set = set()  # track erased node ids to skip stale matches

        for add_node, cast_node, norm_node in matches:
            # Safety: skip if a node in this match was already consumed
            if id(add_node) in erased or id(norm_node) in erased:
                continue

            # Original operands
            add_lhs = add_node.args[0]  # e.g. previous residual
            add_rhs = add_node.args[1]  # e.g. attention/MoE output
            weight = norm_node.args[1]
            eps = norm_node.args[2]

            # Insert the fused call right before the norm node. Using
            # inserting_before(norm_node) ensures correct topological order:
            # fused_node → norm_out → add_out all appear before norm_node.
            with graph.inserting_before(norm_node):
                # flashinfer_fused_add_rms_norm(x, residual, weight, eps):
                #   residual += x        →  residual becomes add result
                #   x = rms_norm(residual) →  x becomes norm result
                # returns (x, residual) = (norm_result, add_result)
                fused_node = graph.call_function(
                    flashinfer_fused_add_rms_norm,
                    args=(add_rhs, add_lhs, weight, eps),
                )
                norm_out = graph.call_function(operator.getitem, args=(fused_node, 0))
                add_out = graph.call_function(operator.getitem, args=(fused_node, 1))

            # Rewire all consumers of the original norm → norm_out
            norm_node.replace_all_uses_with(norm_out)

            # Erase norm first so cast_node (if present) loses its only user
            graph.erase_node(norm_node)
            erased.add(id(norm_node))

            # Erase cast_node *before* replacing add's uses, otherwise
            # replace_all_uses_with would rewrite cast's input to add_out
            # which sits after cast in the graph → topological violation.
            if cast_node is not None:
                if len(cast_node.users) == 0:
                    graph.erase_node(cast_node)
                    erased.add(id(cast_node))
                else:
                    # Rare: cast has users besides norm. Redirect them to a
                    # new cast placed after add_out so the ordering is valid.
                    with graph.inserting_before(list(cast_node.users)[0]):
                        new_cast = graph.call_function(
                            cast_node.target,
                            args=(add_out, *cast_node.args[1:]),
                            kwargs=cast_node.kwargs,
                        )
                    cast_node.replace_all_uses_with(new_cast)
                    graph.erase_node(cast_node)
                    erased.add(id(cast_node))

            # Rewire all consumers of the original add → add_out
            # (includes the residual connection to the *next* layer's add)
            add_node.replace_all_uses_with(add_out)

            graph.erase_node(add_node)
            erased.add(id(add_node))

            num_matches += 1

        # Clean up any remaining dead code
        if num_matches > 0:
            eliminate_dead_code(gm)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info

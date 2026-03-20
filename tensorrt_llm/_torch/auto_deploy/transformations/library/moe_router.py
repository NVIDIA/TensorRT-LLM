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

"""Graph transform to match and replace MoE routing patterns with fused custom ops.

Matches the pattern:
    probs = softmax(logits, dim=-1, dtype=float32)
    topk_weights, topk_indices = topk(probs, k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  [optional]
    topk_weights = topk_weights.to(input_dtype)  [optional]

And replaces with:
    topk_weights, topk_indices = torch.ops.auto_deploy.<backend>_moe_router(logits, k, normalize)
"""

import operator

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from .._graph import canonicalize_graph

_BACKEND_OPS = {
    "triton": torch.ops.auto_deploy.triton_moe_router,
    "torch": torch.ops.auto_deploy.torch_moe_router,
}


def fuse_moe_router(gm: GraphModule, backend: str = "triton") -> None:
    """Match and replace MoE routing patterns with fused custom ops.

    Scans the FX graph for softmax -> topk -> optional normalize -> optional cast
    patterns and replaces them with a single fused MoE router custom op.

    Args:
        gm: Input graph module to transform.
        backend: Backend to use ("triton" or "torch").
    """
    if backend.lower() not in _BACKEND_OPS:
        raise ValueError(f"Invalid backend, must be one of {list(_BACKEND_OPS)}, got {backend}")

    ad_logger.info(f"Starting MoE router pattern matching with backend: {backend}")
    router_op = _BACKEND_OPS[backend.lower()]

    graph = gm.graph
    num_patterns = 0

    for node in list(graph.nodes):
        # Step 1: Find topk nodes
        if not is_op(node, torch.ops.aten.topk):
            continue

        topk_node = node
        topk_input = topk_node.args[0]
        top_k = topk_node.args[1]

        # Step 2: Check that the topk input comes from softmax
        if not isinstance(topk_input, Node):
            continue
        if not is_op(topk_input, torch.ops.aten._softmax):
            continue

        softmax_node = topk_input
        logits_node = softmax_node.args[0]

        # Verify softmax produces float32 (the dtype argument)
        # aten._softmax signature: _softmax(input, dim, half_to_float)
        # We just need the logits input

        # Step 3: Find the getitem users of topk (topk returns a tuple)
        topk_weights_getitem = None
        topk_indices_getitem = None
        for user in topk_node.users:
            if is_op(user, operator.getitem):
                idx = user.args[1]
                if idx == 0:
                    topk_weights_getitem = user
                elif idx == 1:
                    topk_indices_getitem = user

        if topk_weights_getitem is None or topk_indices_getitem is None:
            continue

        # Step 4: Detect optional normalization pattern on the weights
        # Pattern: weights / weights.sum(dim=-1, keepdim=True)
        normalize = False
        weight_output_node = topk_weights_getitem
        cast_node = None

        # Walk through users to find normalization and/or cast
        for user in list(topk_weights_getitem.users):
            # Check for sum -> div normalization
            if is_op(user, torch.ops.aten.sum):
                # This is the sum in the normalization: weights.sum(dim=-1, keepdim=True)
                # The div node should use both topk_weights_getitem and this sum
                for sum_user in user.users:
                    if is_op(sum_user, torch.ops.aten.div):
                        normalize = True
                        weight_output_node = sum_user
                        break

        # Step 5: Check for optional dtype cast after normalization
        for user in list(weight_output_node.users):
            if is_op(user, torch.ops.aten.to) or is_op(user, torch.ops.aten._to_copy):
                cast_node = user
                break

        # Step 6: Insert the fused router op
        with graph.inserting_before(topk_node):
            router_call = graph.call_function(
                router_op,
                args=(logits_node, top_k, normalize),
            )

            # Create getitem nodes for the two outputs
            new_weights_getitem = graph.call_function(operator.getitem, args=(router_call, 0))
            new_indices_getitem = graph.call_function(operator.getitem, args=(router_call, 1))

        # Step 7: Replace uses
        # Replace the indices output
        topk_indices_getitem.replace_all_uses_with(new_indices_getitem)

        # Replace the weights output (accounting for normalization and cast)
        if cast_node is not None:
            # If there's a cast, the final consumer uses the cast node
            # We need to insert a cast after our new weights
            with graph.inserting_after(new_weights_getitem):
                # Copy the cast operation
                new_cast = graph.call_function(
                    cast_node.target,
                    args=(new_weights_getitem,) + cast_node.args[1:],
                    kwargs=dict(cast_node.kwargs),
                )
            cast_node.replace_all_uses_with(new_cast)
        elif normalize:
            weight_output_node.replace_all_uses_with(new_weights_getitem)
        else:
            topk_weights_getitem.replace_all_uses_with(new_weights_getitem)

        num_patterns += 1

    if num_patterns > 0:
        graph.eliminate_dead_code()
        canonicalize_graph(gm)

    ad_logger.info(f"Found {num_patterns} MoE router patterns")

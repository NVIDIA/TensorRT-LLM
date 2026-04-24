"""Transform for multi-stream execution of MoE layers that have shared experts and routed experts."""

from typing import Callable, List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
)
from ...utils.node_utils import has_shape, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _find_merge_node(moe_node: Node) -> Optional[Node]:
    """Walk forward from a MoE op to find the node that merges shared and routed outputs.

    The merge node is where the shared-expert output and the routed-expert
    output are first combined.  In the common case this is an
    ``aten.add.Tensor``, but when upstream passes (e.g. MLIR elementwise
    fusion) have fused the merge ``add`` into a larger kernel, it may be an
    opaque fused op instead.

    The search is a breadth-first traversal of the user graph starting from
    the MoE node.  We track the "MoE forward cone" — nodes reachable purely
    through the MoE output chain.  The merge node is the first node
    encountered that has at least one ``call_function`` input *outside* this
    cone, meaning it also consumes the shared-expert branch.
    """
    moe_cone: Set[Node] = {moe_node}
    visited: Set[Node] = set()
    queue = list(moe_node.users.keys())
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)

        # Check if any input to *n* is a tensor-producing call_function node
        # outside the cone.  The ``has_shape`` guard filters out scalar /
        # SymInt nodes (e.g. ``aten.sym_size.int``) that feed into reshape
        # ops — those are shape computations, not data branches.
        for inp in n.all_input_nodes:
            if inp not in moe_cone and inp.op == "call_function" and has_shape(inp):
                return n

        # *n* is purely on the MoE/routed path — extend the cone.
        moe_cone.add(n)
        queue.extend(n.users.keys())
    return None


def _get_ancestors(node: Node) -> Set[Node]:
    """Return the set of all nodes reachable by walking backwards from *node*."""
    ancestors: Set[Node] = set()
    queue = list(node.all_input_nodes)
    while queue:
        n = queue.pop()
        if n in ancestors:
            continue
        ancestors.add(n)
        queue.extend(n.all_input_nodes)
    return ancestors


def _execute_shared_expert_in_aux_stream(
    gm: GraphModule, moe_ops: List[Callable]
) -> Tuple[GraphModule, int]:
    """Move shared-expert computation to the auxiliary CUDA stream.

    For each MoE fused op in the graph:
      1. Walk forward to find the merge node where shared-expert output and
         routed-expert output are combined.  This is typically an
         ``aten.add.Tensor`` but may be a fused op (e.g. MLIR kernel) that
         has absorbed the merge add along with subsequent elementwise ops.
      2. Classify the merge node's inputs into *routed* (descended from the
         MoE node), *pass-through* (an ancestor of the MoE node, such as a
         residual connection), or *shared* (the shared-expert output).
      3. Trace the shared-expert branch backwards to collect all its
         computation nodes and identify the fork point (the latest common
         ancestor shared with the MoE / routing path).
      4. Insert ``begin_aux_stream_passthrough`` before the first shared-expert
         op to switch to the auxiliary CUDA stream.
      5. Insert ``end_aux_stream_passthrough`` after the last shared-expert op
         to switch back to the main stream.
      6. Insert ``wait_aux_stream_passthrough`` on the routed-branch input
         just before the merge node so the main stream waits for the auxiliary
         stream to finish before merging outputs.
    """
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating.
    target_nodes = [n for n in graph.nodes if is_op(n, moe_ops)]
    if not target_nodes:
        return gm, 0

    node_order = {node: i for i, node in enumerate(graph.nodes)}

    for moe_node in target_nodes:
        # ---- Step 1: Find the merge node. ----
        merge_node = _find_merge_node(moe_node)
        if merge_node is None:
            ad_logger.warning(
                f"No merge node found downstream of MoE node {moe_node.name}; "
                "skipping multi-stream transform for this node."
            )
            continue

        # ---- Step 2: Classify merge node inputs. ----
        # The merge node may have >2 inputs when upstream passes (e.g. MLIR
        # elementwise fusion) have fused the merge add with subsequent ops
        # like residual add + RMSNorm.  We classify each input as:
        #   - *routed*:      descended from the MoE node.
        #   - *pass-through*: an ancestor of the MoE node (e.g. residual).
        #   - *shared*:      the shared-expert output (neither of the above).
        moe_ancestors = _get_ancestors(moe_node)
        moe_ancestors.add(moe_node)

        routed_output: Optional[Node] = None
        shared_output: Optional[Node] = None
        for arg in merge_node.all_input_nodes:
            arg_ancestors = _get_ancestors(arg)
            if moe_node in arg_ancestors or arg is moe_node:
                routed_output = arg
            elif arg in moe_ancestors or arg.op != "call_function":
                # Pass-through: either an ancestor of the MoE node (e.g.
                # residual connection) or a non-computation node (e.g.
                # parameter, placeholder) — leave untouched.
                pass
            else:
                shared_output = arg

        if routed_output is None or shared_output is None:
            ad_logger.warning(
                f"Could not classify merge node inputs for MoE node "
                f"{moe_node.name} (routed={routed_output}, shared={shared_output}); "
                "skipping multi-stream transform for this node."
            )
            continue

        shared_nodes: List[Node] = []
        fork_point: Optional[Node] = None
        visited: Set[Node] = set()
        queue = [shared_output]

        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)

            # Skip static weight / parameter nodes.
            if n.op == "get_attr":
                continue

            if n in moe_ancestors:
                # This node is on the MoE / routing path — candidate fork point.
                if fork_point is None or node_order.get(n, 0) > node_order.get(fork_point, 0):
                    fork_point = n
                continue

            shared_nodes.append(n)
            for inp in n.all_input_nodes:
                queue.append(inp)

        if not shared_nodes or fork_point is None:
            ad_logger.warning(
                f"Could not identify shared-expert subgraph for MoE node "
                f"{moe_node.name}; skipping multi-stream transform for this node."
            )
            continue

        # Order shared nodes by their position in the graph.
        shared_nodes.sort(key=lambda n: node_order.get(n, 0))
        first_shared = shared_nodes[0]

        # Sanity check: the first shared op must directly consume the fork
        # point so we can wire begin_aux_stream_passthrough into it.
        if fork_point not in first_shared.all_input_nodes:
            ad_logger.warning(
                f"First shared-expert op ({first_shared.name}) does not directly "
                f"consume fork point ({fork_point.name}); skipping."
            )
            continue

        # ---- Step 4: Insert begin_aux before the first shared-expert op. ----
        # NOTE: do NOT bake ``torch.cuda.current_device()`` into the graph —
        # that would hard-code device 0 and break on other ranks in a
        # multi-GPU setup.  Omitting ``device`` lets the passthrough
        # functions resolve the device at **runtime** (default ``-1``).
        with graph.inserting_before(first_shared):
            begin_aux_node = graph.call_function(
                begin_aux_stream_passthrough,
                args=(fork_point,),
            )

        # Create a data dependency: first_shared reads begin_aux output
        # instead of fork_point.
        first_shared.args = tuple(
            begin_aux_node if arg is fork_point else arg for arg in first_shared.args
        )

        # ---- Step 5: Insert end_aux after the last shared-expert op. ----
        with graph.inserting_after(shared_output):
            end_aux_node = graph.call_function(
                end_aux_stream_passthrough,
                args=(shared_output,),
            )

        # Replace shared-expert input to the merge node with end_aux output.
        merge_node.args = tuple(
            end_aux_node if arg is shared_output else arg for arg in merge_node.args
        )

        # ---- Step 6: Insert wait_aux before the merge node. ----
        with graph.inserting_before(merge_node):
            wait_aux_node = graph.call_function(
                wait_aux_stream_passthrough,
                args=(routed_output,),
            )

        merge_node.args = tuple(
            wait_aux_node if arg is routed_output else arg for arg in merge_node.args
        )

        num_replaced += 1

    return gm, num_replaced


@TransformRegistry.register("multi_stream_moe")
class MultiStreamMOE(BaseTransform):
    """Multi-stream execution of MoE layers that have shared experts and routed experts."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        base_ops = [
            torch.ops.auto_deploy.trtllm_moe_fused,
            torch.ops.auto_deploy.triton_moe_fused,
            torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
            torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused,
            torch.ops.auto_deploy.trtllm_nvfp4_trtllm_gen_moe_fused,
            torch.ops.auto_deploy.trtllm_quant_finegrained_fp8_moe_fused,
        ]

        # Ensure that aux stream and events for the current device are added to the CudaStreamManager.
        cuda_stream_manager.add_device(torch.cuda.current_device())
        gm, num_matches = _execute_shared_expert_in_aux_stream(gm, base_ops)
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info

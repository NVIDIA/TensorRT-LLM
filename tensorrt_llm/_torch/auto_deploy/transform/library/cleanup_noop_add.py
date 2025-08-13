from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_noop_add")
class CleanupNoopAdd(BaseTransform):
    """Eliminate add nodes from the graph that are no-ops.

    This would be any node that is just adding 0 to the input tensor. We can safely remove those.

    NOTE: this function has one failure mode when the op ``out = tensor + zero_tensor`` is used
    in such a way that``out`` will be broadcast to the shape of zero_tensor. After removing this op
    then, out won't have the right shape anymore. This should be a rare case and we can handle it
    when it comes up or disable this transform.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        for node in gm.graph.nodes:
            # looking for add nodes
            if not is_op(node, torch.ops.aten.add):
                continue
            # only handling this parameter combination for now
            if len(node.all_input_nodes) != 2:
                continue

            # check if any of the input nodes is just a constant tensor with value 0
            if is_op(node.all_input_nodes[0], torch.ops.aten.zeros):
                zero_node, true_node = node.all_input_nodes
            elif is_op(node.all_input_nodes[1], torch.ops.aten.zeros):
                true_node, zero_node = node.all_input_nodes
            else:
                continue

            # do the replacement and clean-up
            node.replace_all_uses_with(true_node)
            gm.graph.erase_node(node)
            num_matches += 1

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=num_matches)

        return gm, info

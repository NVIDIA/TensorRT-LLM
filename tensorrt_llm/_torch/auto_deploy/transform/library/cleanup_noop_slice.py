from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_noop_slice")
class CleanupNoopSlice(BaseTransform):
    """Remove no-op slice nodes from the graph.

    Those will be nodes that are used to represent a slice operation like ``t[:, :5]``. The graph IR
    will represent it as ``t[:][:5]``, i.e., two nodes and the first slice being a no-op. This
    function gets rid of such instances.
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
            # looking for slice nodes
            if not is_op(node, torch.ops.aten.slice):
                continue
            # only handling this parameter combination for now
            # 4 args will be (input, dim, start, end)
            if len(node.args) != 4 or len(node.kwargs) != 0:
                continue
            # check if dim is just an integer
            if not isinstance(node.args[1], int):
                continue
            # check if the slice op is indeed a no-op
            if node.args[2] != 0 or node.args[3] != torch.iinfo(torch.long).max:
                continue
            # extract input tensor node and remove the slice node
            in_node = node.args[0]
            assert [in_node] == node.all_input_nodes, "Slice node has unexpected input nodes."
            node.replace_all_uses_with(in_node)
            gm.graph.erase_node(node)
            num_matches += 1

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=num_matches)

        return gm, info

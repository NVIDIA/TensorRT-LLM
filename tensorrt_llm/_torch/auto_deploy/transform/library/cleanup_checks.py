from typing import Tuple

import torch
from torch.fx import Graph, GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_checks")
class CleanupChecks(BaseTransform):
    """This transformations removes shape checks and assertions from the graph."""

    _check_ops = {
        torch.ops.aten._assert_scalar,
        torch.ops.aten.sym_constrain_range,
        torch.ops.aten.sym_constrain_range_for_size,
        torch.ops.aten._assert_tensor_metadata,
        # torch.ops.aten._functional_sym_constrain_range,
        # torch.ops.aten._functional_sym_constrain_range_for_size
    }

    def _apply(
        self, gm: GraphModule, cm: CachedSequenceInterface, factory: ModelFactory
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0

        graph: Graph = gm.graph
        for node in reversed(graph.nodes):
            if len(node.users) > 0 or not is_op(node, self._check_ops):
                continue
            graph.erase_node(node)
            num_matches += 1

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=num_matches)

        return gm, info

import math
from typing import List, Tuple

import torch
from torch.fx import Graph, GraphModule
from torch.utils._sympy.value_ranges import ValueRanges

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


# TODO (lucaslie): consider reconfiguring this transform to run before we switch to flattened
# sequences which is done in update_in_out_nodes at the moment.
@TransformRegistry.register("cleanup_input_constraints")
class CleanupInputConstraints(BaseTransform):
    """Cleanup input constraints from the graph.

    This transformations updates the input constraints of the graph. Specifically, we want to
    account for flattened sequences and hence the max constraint should be updated to reflect the
    flattened sequence length.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph: Graph = gm.graph
        input_node = graph.find_nodes(op="placeholder")[0]
        sym_shape: torch.Size = input_node.meta["val"].shape

        # get expressions in the symbolic shape
        vrs: List[ValueRanges] = []
        for s in sym_shape:
            if isinstance(s, int):
                vrs.append(ValueRanges(0, s))
            elif isinstance(s, torch.SymInt):
                vrs.append(gm.range_constraints[s.node.expr])
            else:
                raise TypeError(f"Unexpected type {type(s)} in symbolic shape.")

        # update the max constraint for each vr
        max_total = math.prod(vr.upper for vr in vrs)
        for vr in vrs:
            object.__setattr__(vr, "upper", max_total)

        # store info object about the transform
        info = TransformInfo(skipped=False, num_matches=len(vrs))

        return gm, info

import math
from typing import List, Tuple

import torch
from torch.fx import Graph, GraphModule
from torch.utils._sympy.value_ranges import ValueRanges

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import has_shape
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
        placeholders = graph.find_nodes(op="placeholder")
        if len(placeholders) < 2:
            return gm, TransformInfo(skipped=True, num_matches=0)
        input_node = placeholders[1]
        if not has_shape(input_node):
            ad_logger.debug(
                f"Input node {input_node.name} does not have a shape. Skipping cleanup_input_constraints transform."
            )
            return gm, TransformInfo(skipped=True, num_matches=0)
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
        # NOTE: this is more a heuristic anyway than a strict constraint. We just want to make sure
        # that this never gets triggered. So we multiply by 1000 to be safe. Not that it has to
        # be a symint (not an int) --> so that's why we use a heuristic based on the existing
        # symint values instead of just using e.g. max_num_tokens...
        max_total = math.prod(vr.upper for vr in vrs) * 1000
        for vr in vrs:
            object.__setattr__(vr, "upper", max_total)

        # store info object about the transform
        info = TransformInfo(
            skipped=False,
            num_matches=len(vrs),
            is_clean=len(vrs) == 0,
            has_valid_shapes=len(vrs) == 0,
        )

        return gm, info

from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_identity_dtype_cast")
class CleanupIdentityDtypeCast(BaseTransform):
    """Remove identity dtype casts where input dtype already matches the target dtype.

    During FX tracing, explicit `.to(dtype)` calls in model code are preserved even when
    the tensor is already in the target dtype (e.g., bf16 -> bf16). These generate
    unnecessary elementwise copy kernels at runtime. This transform removes them.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        nodes_to_erase = []

        for node in gm.graph.nodes:
            if not is_op(node, torch.ops.aten.to.dtype):
                continue

            if len(node.args) < 2:
                continue

            input_node = node.args[0]
            target_dtype = node.args[1]

            input_meta = input_node.meta.get("val", None)
            if input_meta is None:
                continue

            if not hasattr(input_meta, "dtype"):
                continue

            if input_meta.dtype == target_dtype:
                node.replace_all_uses_with(input_node)
                nodes_to_erase.append(node)
                num_matches += 1

        for node in nodes_to_erase:
            gm.graph.erase_node(node)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

        return gm, info

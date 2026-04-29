from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# Positional argument indices for aten.to.dtype:
#   to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False,
#            bool copy=False, MemoryFormat? memory_format=None)
_TO_DTYPE_ARG_SELF = 0
_TO_DTYPE_ARG_DTYPE = 1
_TO_DTYPE_ARG_NON_BLOCKING = 2
_TO_DTYPE_ARG_COPY = 3
_TO_DTYPE_ARG_MEMORY_FORMAT = 4


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

            if len(node.args) < _TO_DTYPE_ARG_DTYPE + 1:
                continue

            input_node = node.args[_TO_DTYPE_ARG_SELF]
            target_dtype = node.args[_TO_DTYPE_ARG_DTYPE]

            input_meta = input_node.meta.get("val", None)
            if input_meta is None:
                continue

            if not hasattr(input_meta, "dtype"):
                continue

            # `copy=True` forces a materialized copy and `memory_format` may
            # change the tensor's stride layout, both of which are observable
            # even when the source and target dtypes match. Skip elimination
            # in those cases to preserve semantics.
            copy = node.kwargs.get("copy", False)
            if len(node.args) > _TO_DTYPE_ARG_COPY:
                copy = node.args[_TO_DTYPE_ARG_COPY]
            if copy:
                continue

            memory_format = node.kwargs.get("memory_format", None)
            if len(node.args) > _TO_DTYPE_ARG_MEMORY_FORMAT:
                memory_format = node.args[_TO_DTYPE_ARG_MEMORY_FORMAT]
            if memory_format is not None:
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

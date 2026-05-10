from typing import Any, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _get_positional_or_kwarg(node: Node, pos: int, name: str, default: Any = None) -> Any:
    """Return ``node.args[pos]`` if present, else ``node.kwargs[name]`` (or default).

    FX records args either positionally or by keyword depending on how the op
    was called at trace time, so both spellings need to be checked.
    """
    if len(node.args) > pos:
        return node.args[pos]
    return node.kwargs.get(name, default)


@TransformRegistry.register("cleanup_identity_dtype_cast")
class CleanupIdentityDtypeCast(BaseTransform):
    """Remove identity dtype casts where input dtype already matches the target dtype.

    Handles the three dtype-cast spellings commonly produced by tracing /
    functionalization:

    * ``aten.to.dtype(self, dtype, ...)`` — explicit ``.to(dtype)`` calls.
    * ``aten._to_copy.default(self, dtype=..., ...)`` — functionalized form
      that always materializes a copy.
    * ``prims.convert_element_type.default(self, dtype)`` — canonical torch.
      compile / torch.export primitive for dtype conversion.

    Elimination only proceeds when the cast is semantically an identity:
    source dtype equals target dtype and no other observable attribute
    (copy flag, layout, device, pin_memory, memory_format) diverges from
    the input.
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
            result = self._try_eliminate(node)
            if result is None:
                continue
            input_node = result
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

    # ------------------------------------------------------------------
    # Per-op dispatch
    # ------------------------------------------------------------------

    def _try_eliminate(self, node: Node) -> Optional[Node]:
        """Return the upstream input node to fold into, or None to keep the node."""
        if is_op(node, torch.ops.aten.to.dtype):
            return self._try_eliminate_to_dtype(node)
        if is_op(node, torch.ops.aten._to_copy.default):
            return self._try_eliminate_to_copy(node)
        if is_op(node, torch.ops.prims.convert_element_type.default):
            return self._try_eliminate_convert_element_type(node)
        return None

    # ------------------------------------------------------------------
    # aten.to.dtype
    # ------------------------------------------------------------------

    def _try_eliminate_to_dtype(self, node: Node) -> Optional[Node]:
        # to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False,
        #          bool copy=False, MemoryFormat? memory_format=None)
        if len(node.args) < 2:
            return None

        input_node = node.args[0]
        target_dtype = node.args[1]

        input_dtype = _get_input_dtype(input_node)
        if input_dtype is None or input_dtype != target_dtype:
            return None

        # `copy=True` forces a materialized copy and `memory_format` may
        # change the tensor's stride layout, both of which are observable
        # even when the source and target dtypes match. Skip elimination
        # in those cases to preserve semantics.
        copy = _get_positional_or_kwarg(node, 3, "copy", False)
        if copy:
            return None

        memory_format = _get_positional_or_kwarg(node, 4, "memory_format", None)
        if memory_format is not None:
            return None

        return input_node

    # ------------------------------------------------------------------
    # aten._to_copy.default
    # ------------------------------------------------------------------

    def _try_eliminate_to_copy(self, node: Node) -> Optional[Node]:
        # _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
        #          Device? device=None, bool? pin_memory=None, bool non_blocking=False,
        #          MemoryFormat? memory_format=None)
        if len(node.args) < 1:
            return None
        input_node = node.args[0]

        input_meta = _get_val_meta(input_node)
        if input_meta is None or not hasattr(input_meta, "dtype"):
            return None

        # Only eliminate when the caller actually requested a dtype and it
        # matches the source. Missing dtype kwarg means the caller wanted
        # the copy for some other reason (layout/device pinning); don't
        # fold in that case.
        target_dtype = node.kwargs.get("dtype", None)
        if target_dtype is None or input_meta.dtype != target_dtype:
            return None

        # Any of the other kwargs changing would make this an observable
        # materialization, not an identity. Keep this list in sync with
        # the _to_copy schema.
        if node.kwargs.get("memory_format", None) is not None:
            return None
        if node.kwargs.get("pin_memory", None):
            return None

        target_layout = node.kwargs.get("layout", None)
        if target_layout is not None and target_layout != getattr(input_meta, "layout", None):
            return None

        target_device = node.kwargs.get("device", None)
        if target_device is not None:
            meta_device = getattr(input_meta, "device", None)
            if meta_device is None or torch.device(target_device) != meta_device:
                return None

        return input_node

    # ------------------------------------------------------------------
    # prims.convert_element_type.default
    # ------------------------------------------------------------------

    def _try_eliminate_convert_element_type(self, node: Node) -> Optional[Node]:
        # convert_element_type(Tensor a, ScalarType dtype) -> Tensor
        if len(node.args) < 2:
            return None

        input_node = node.args[0]
        target_dtype = node.args[1]

        input_dtype = _get_input_dtype(input_node)
        if input_dtype is None or input_dtype != target_dtype:
            return None

        return input_node


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------


def _get_val_meta(input_node: Any) -> Any:
    """Return ``input_node.meta['val']`` (a FakeTensor) or None."""
    if not isinstance(input_node, Node):
        return None
    return input_node.meta.get("val", None)


def _get_input_dtype(input_node: Any) -> Optional[torch.dtype]:
    """Return the dtype of ``input_node``'s FakeTensor meta, or None."""
    meta = _get_val_meta(input_node)
    if meta is None:
        return None
    return getattr(meta, "dtype", None)

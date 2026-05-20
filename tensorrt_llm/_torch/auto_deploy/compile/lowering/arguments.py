from __future__ import annotations

from typing import Any

from torch.fx import Node


class OpArgumentResolver:
    """Schema-aware read access for FX call_function node arguments."""

    def get(self, node: Node, *names: str) -> tuple[Any, ...]:
        from ...utils.node_utils import extract_op_args

        return tuple(extract_op_args(node, *names))

    def one(self, node: Node, name: str) -> Any:
        (value,) = self.get(node, name)
        return value

    def as_dict(self, node: Node, *names: str) -> dict[str, Any]:
        return dict(zip(names, self.get(node, *names)))

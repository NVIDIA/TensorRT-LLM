"""The modeling-bringup meaning of the engine's opaque node ``type`` field.

The generic execution engine treats a node's ``type`` as an opaque string; it
is this modeling-bringup policy table that gives ``"stage"`` and ``"goal"``
their meaning. Every node runs the ``coder ⇄ reviewer`` loop. A ``stage`` node
(``runs_qa=True``) additionally runs a QA step and closes only on QA APPROVE —
the accuracy-convergence gate; a ``goal`` node (``runs_qa=False``) closes on
Reviewer APPROVE. ``is_replan_unit`` marks the node type the replan phase
operates on, which is the ``stage`` only.

This module is pure policy: it depends on nothing beyond the standard library
so it never pulls in the engine (``orchestration``) or any other subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeTypePolicy:
    """The gate policy a modeling-bringup node type maps to.

    Attributes:
        node_type: The modeling-bringup node type, ``"stage"`` or ``"goal"``.
        runs_qa: When ``True`` the node runs a QA step on top of the
            ``coder ⇄ reviewer`` loop and closes on QA APPROVE (the
            accuracy-convergence gate); when ``False`` it closes on Reviewer
            APPROVE.
        is_replan_unit: Whether the replan phase operates on this node type
            (``stage`` only).
    """

    node_type: str
    runs_qa: bool
    is_replan_unit: bool


STAGE_POLICY = NodeTypePolicy("stage", runs_qa=True, is_replan_unit=True)
GOAL_POLICY = NodeTypePolicy("goal", runs_qa=False, is_replan_unit=False)

_POLICY_BY_TYPE: dict[str, NodeTypePolicy] = {
    STAGE_POLICY.node_type: STAGE_POLICY,
    GOAL_POLICY.node_type: GOAL_POLICY,
}

VALID_NODE_TYPES: frozenset[str] = frozenset(_POLICY_BY_TYPE)


def policy_for_type(node_type: str) -> NodeTypePolicy:
    """Return the gate policy for a modeling-bringup ``node_type``.

    Args:
        node_type: The node type, which must be ``"stage"`` or ``"goal"`` —
            the modeling-bringup vocabulary is exactly these two.

    Returns:
        The :class:`NodeTypePolicy` for ``node_type``.

    Raises:
        ValueError: If ``node_type`` is neither ``"stage"`` nor ``"goal"``.
            The message names the offending type.
    """
    try:
        return _POLICY_BY_TYPE[node_type]
    except KeyError:
        valid = ", ".join(sorted(VALID_NODE_TYPES))
        raise ValueError(
            f"unknown modeling-bringup node type {node_type!r}; expected one of: {valid}"
        ) from None


__all__ = [
    "NodeTypePolicy",
    "STAGE_POLICY",
    "GOAL_POLICY",
    "VALID_NODE_TYPES",
    "policy_for_type",
]

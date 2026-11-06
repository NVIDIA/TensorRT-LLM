"""Tests for the modeling-bringup node type-to-gate policy table."""

from __future__ import annotations

import dataclasses

import pytest

from agent_flow.workflows.modeling_bringup import node_policy


def test_stage_policy_runs_qa_and_is_replan_unit():
    """A ``stage`` node runs QA (accuracy gate) and is the replan unit."""
    policy = node_policy.policy_for_type("stage")
    assert policy is node_policy.STAGE_POLICY
    assert policy.node_type == "stage"
    assert policy.runs_qa is True
    assert policy.is_replan_unit is True


def test_goal_policy_skips_qa_and_is_not_replan_unit():
    """A ``goal`` node closes on Reviewer APPROVE and is not a replan unit."""
    policy = node_policy.policy_for_type("goal")
    assert policy is node_policy.GOAL_POLICY
    assert policy.node_type == "goal"
    assert policy.runs_qa is False
    assert policy.is_replan_unit is False


def test_policy_for_unknown_type_raises_value_error_naming_type():
    """An unknown type is rejected with a ``ValueError`` naming the offender."""
    with pytest.raises(ValueError, match="bogus") as excinfo:
        node_policy.policy_for_type("bogus")
    assert "bogus" in str(excinfo.value)


def test_node_type_policy_is_frozen():
    """``NodeTypePolicy`` is immutable so the shared constants can't be mutated."""
    assert dataclasses.is_dataclass(node_policy.NodeTypePolicy)
    with pytest.raises(dataclasses.FrozenInstanceError):
        node_policy.STAGE_POLICY.runs_qa = False  # type: ignore[misc]


def test_valid_node_types_is_exactly_stage_and_goal():
    """The modeling-bringup vocabulary is exactly ``stage`` and ``goal``."""
    assert node_policy.VALID_NODE_TYPES == frozenset({"stage", "goal"})
    assert isinstance(node_policy.VALID_NODE_TYPES, frozenset)

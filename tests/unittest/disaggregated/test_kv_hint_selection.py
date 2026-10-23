# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private action selection preserves admitted metadata without side effects."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.disaggregation.orchestration._kv_hints import _select_kv_fetch
from tensorrt_llm.kv_hints.protocol import snapshot_kv_hints

pytestmark = pytest.mark.cpu_only


def _wire() -> dict:
    """Return the producer's kv.fetch fixture plus an opaque future action."""
    return {
        "protocol_version": "0.1",
        "message_id": "msg-123",
        "actions": [
            {
                "action_id": "a1",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://127.0.0.1:23280",
                    "block_hashes": [11, 22],
                },
            },
            {
                "action_id": "future",
                "action_type": "kv.future_action",
                "action_version": "7.3",
                "payload": {"nested": {"enabled": True}, "items": [1, None, "x"]},
            },
        ],
    }


def test_first_fetch_selection_preserves_full_action_and_envelope() -> None:
    """Selection keeps metadata, counts ignored actions and offers no retry cursor."""
    wire = _wire()
    first = wire["actions"][0]
    first["action_version"] = "future-provider-version"
    first["optional_field"] = {"keep": [1, 2]}
    wire["producer_metadata"] = {"trace": "router"}
    wire["actions"] = [wire["actions"][1], first, dict(first, action_id="later")]
    snapshot = snapshot_kv_hints(wire)
    versions = frozenset({"future-provider-version"})
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        selected = _select_kv_fetch(snapshot, supported_action_versions=versions)
        assert _select_kv_fetch(snapshot, supported_action_versions=versions) == selected
    count.assert_not_called()
    assert selected.unknown_actions == 1
    assert selected.unsupported_version_actions == 0
    assert selected.extra_fetch_actions == 1
    assert selected.selected_action.envelope().model_dump() == dict(wire, actions=[first])
    assert snapshot.envelope().model_dump() == wire
    selected.selected_action.envelope().actions[0].payload.clear()
    assert selected.selected_action.envelope().actions[0].payload == first["payload"]
    with pytest.raises(FrozenInstanceError):
        selected.extra_fetch_actions = 0


def test_selection_without_fetch_does_not_invent_a_source() -> None:
    """Absent or unrecognized-only actions produce no selected action."""
    versions = frozenset({"1.0"})
    absent = _select_kv_fetch(None, supported_action_versions=versions)
    assert absent.selected_action is None
    assert absent.unknown_actions == 0
    assert absent.unsupported_version_actions == 0
    assert absent.extra_fetch_actions == 0
    wire = _wire()
    wire["actions"] = wire["actions"][1:]
    selected = _select_kv_fetch(snapshot_kv_hints(wire), supported_action_versions=versions)
    assert selected.selected_action is None
    assert selected.unknown_actions == 1
    assert selected.unsupported_version_actions == 0
    assert selected.extra_fetch_actions == 0


@pytest.mark.parametrize(
    ("action_versions", "supported_versions", "selected_index", "unsupported_count", "extra_count"),
    [
        (["2.0", "1.0"], {"1.0"}, 1, 1, 0),
        (["2.0", "1.0"], {"1.0", "2.0"}, 0, 0, 1),
        (["1.0", "2.0"], {"1.0", "2.0"}, 0, 0, 1),
        (["2.0", "3.0"], {"1.0"}, None, 2, 0),
        (["1.0"], set(), None, 1, 0),
        (["1.0", "2.0", "1.0"], {"1.0"}, 0, 1, 1),
        (["1.00", "1.0"], {"1.0"}, 1, 1, 0),
    ],
)
def test_selection_uses_consumer_action_versions(
    action_versions: list[str],
    supported_versions: set[str],
    selected_index: int | None,
    unsupported_count: int,
    extra_count: int,
) -> None:
    """Compatibility filtering preserves producer order and disjoint action counts."""
    wire = _wire()
    fetch = wire["actions"][0]
    wire["actions"] = [
        dict(fetch, action_id=f"fetch-{index}", action_version=version)
        for index, version in enumerate(action_versions)
    ]
    snapshot = snapshot_kv_hints(wire)
    selected = _select_kv_fetch(snapshot, supported_action_versions=frozenset(supported_versions))
    assert selected.unknown_actions == 0
    assert selected.unsupported_version_actions == unsupported_count
    assert selected.extra_fetch_actions == extra_count
    if selected_index is None:
        assert selected.selected_action is None
    else:
        assert selected.selected_action.envelope().model_dump() == dict(
            wire, actions=[wire["actions"][selected_index]]
        )
    assert snapshot.envelope().model_dump() == wire


def test_selection_leaves_payload_validation_to_consumer() -> None:
    """An opaque payload does not redirect selection to a later action."""
    wire = _wire()
    fetch = wire["actions"][0]
    first = dict(fetch, action_id="missing-provider-fields", payload={})
    wire["actions"] = [first, dict(fetch, action_id="later")]
    snapshot = snapshot_kv_hints(wire)
    versions = frozenset({"1.0"})
    selected = _select_kv_fetch(snapshot, supported_action_versions=versions)
    assert selected.selected_action.envelope().model_dump() == dict(wire, actions=[first])
    assert selected.unsupported_version_actions == 0
    assert selected.extra_fetch_actions == 1
    assert _select_kv_fetch(snapshot, supported_action_versions=versions) == selected

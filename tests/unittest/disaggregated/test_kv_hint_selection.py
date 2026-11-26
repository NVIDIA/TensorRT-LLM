# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private candidate collection preserves admitted metadata without side effects."""

import tracemalloc
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.disaggregation.orchestration._kv_hints import _collect_kv_fetch_candidates
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


def test_candidates_preserve_order_metadata_and_duplicate_action_ids() -> None:
    """Each supported action keeps its position and metadata, even when IDs repeat."""
    wire = _wire()
    first = wire["actions"][0]
    first["optional_field"] = {"keep": [1, 2]}
    later = dict(first, payload={"block_hashes": [33], "source_control_endpoint": "opaque"})
    wire["producer_metadata"] = {"trace": ["router"]}
    wire["actions"] = [wire["actions"][1], first, later]
    snapshot = snapshot_kv_hints(wire)
    capabilities = {"store": frozenset({("kv.fetch", "1.0")})}

    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        result = _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities)
        materialized = [
            candidate.snapshot().envelope().model_dump() for candidate in result.candidates
        ]

    count.assert_not_called()
    assert isinstance(result.candidates, tuple)
    assert materialized == [dict(wire, actions=[first]), dict(wire, actions=[later])]
    assert all(candidate.families == frozenset({"store"}) for candidate in result.candidates)
    assert result.unknown_actions == 1
    assert result.unsupported_version_actions == 0
    assert snapshot.envelope().model_dump() == wire


def test_family_capabilities_match_types_versions_and_count_each_action_once() -> None:
    """Overlapping family support does not duplicate candidates or ignored counts."""
    wire = _wire()
    fetch, future = wire["actions"]
    versions = ["2.0", "1.0", "3.0", "1.00", "1.0"]
    wire["actions"] = [
        dict(fetch, action_id=f"fetch-{index}", action_version=version)
        for index, version in enumerate(versions)
    ] + [future]
    capabilities = {
        "store": frozenset({("kv.fetch", "1.0"), ("kv.publish", "3.0")}),
        "peer": frozenset({("kv.fetch", "1.0"), ("kv.fetch", "2.0")}),
        "future": frozenset({("kv.future_action", "7.3")}),
    }
    snapshot = snapshot_kv_hints(wire)

    result = _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities)
    reversed_result = _collect_kv_fetch_candidates(
        snapshot, family_capabilities=dict(reversed(list(capabilities.items())))
    )

    assert result == reversed_result
    assert [candidate.families for candidate in result.candidates] == [
        frozenset({"peer"}),
        frozenset({"store", "peer"}),
        frozenset({"store", "peer"}),
    ]
    assert [
        candidate.snapshot().envelope().actions[0].action_id for candidate in result.candidates
    ] == [
        "fetch-0",
        "fetch-1",
        "fetch-4",
    ]
    assert result.unknown_actions == 1
    assert result.unsupported_version_actions == 2


@pytest.mark.parametrize(
    "capabilities",
    [
        {},
        {"store": frozenset()},
        {"store": frozenset({("kv.fetch", "2.0")})},
        {"store": frozenset({("kv.publish", "1.0")})},
    ],
)
def test_no_matching_family_produces_no_candidates(
    capabilities: dict[str, frozenset[tuple[str, str]]],
) -> None:
    """A missing exact capability never invents a candidate or a source."""
    result = _collect_kv_fetch_candidates(
        snapshot_kv_hints(_wire()), family_capabilities=capabilities
    )
    assert result.candidates == ()
    assert result.unknown_actions == 1
    assert result.unsupported_version_actions == 1


def test_absent_or_unknown_only_envelope_produces_no_candidates() -> None:
    """Absent input and unknown actions have different, disjoint counters."""
    capabilities = {"store": frozenset({("kv.fetch", "1.0")})}
    absent = _collect_kv_fetch_candidates(None, family_capabilities=capabilities)
    assert absent.candidates == ()
    assert absent.unknown_actions == 0
    assert absent.unsupported_version_actions == 0

    wire = _wire()
    wire["actions"] = wire["actions"][1:]
    unknown = _collect_kv_fetch_candidates(
        snapshot_kv_hints(wire), family_capabilities=capabilities
    )
    assert unknown.candidates == ()
    assert unknown.unknown_actions == 1
    assert unknown.unsupported_version_actions == 0

    wire["actions"] = []
    empty = _collect_kv_fetch_candidates(snapshot_kv_hints(wire), family_capabilities=capabilities)
    assert empty == absent


def test_candidates_do_not_expose_mutable_input_or_result_state() -> None:
    """Caller changes and decoded-envelope changes cannot alter retained candidates."""
    wire = _wire()
    wire["producer_metadata"] = {"trace": ["router"]}
    snapshot = snapshot_kv_hints(wire)
    capabilities = {"store": frozenset({("kv.fetch", "1.0")})}
    result = _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities)
    candidate = result.candidates[0]
    expected = candidate.snapshot().envelope().model_dump()

    capabilities.clear()
    wire["producer_metadata"]["trace"].clear()
    wire["actions"][0]["payload"].clear()
    decoded = candidate.snapshot().envelope()
    decoded.actions[0].payload.clear()
    decoded.__pydantic_extra__["producer_metadata"]["trace"].clear()

    assert candidate.families == frozenset({"store"})
    assert candidate.snapshot().envelope().model_dump() == expected
    assert snapshot.envelope().actions[0].payload == expected["actions"][0]["payload"]
    with pytest.raises(FrozenInstanceError):
        candidate.families = frozenset()
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()


def test_collection_does_not_validate_payloads_or_choose_an_action() -> None:
    """An opaque first payload stays first; all supported candidates remain available."""
    wire = _wire()
    fetch = wire["actions"][0]
    first = dict(fetch, action_id="missing-provider-fields", payload={})
    wire["actions"] = [first, dict(fetch, action_id="later")]
    snapshot = snapshot_kv_hints(wire)
    capabilities = {"store": frozenset({("kv.fetch", "1.0")})}

    result = _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities)

    assert [
        candidate.snapshot().envelope().actions[0].action_id for candidate in result.candidates
    ] == [
        "missing-provider-fields",
        "later",
    ]
    assert result.candidates[0].snapshot().envelope().actions[0].payload == {}
    assert result.unsupported_version_actions == 0
    assert _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities) == result


def test_collection_does_not_copy_large_envelope_metadata_per_candidate() -> None:
    """Retaining many candidates must not multiply the envelope's shared metadata."""
    wire = _wire()
    wire["producer_metadata"] = {"padding": "x" * 32_000}
    fetch = dict(wire["actions"][0], payload={})
    wire["actions"] = [dict(fetch, action_id=str(index)) for index in range(200)]
    snapshot = snapshot_kv_hints(wire)
    assert snapshot is not None
    capabilities = {"store": frozenset({("kv.fetch", "1.0")})}

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        result = _collect_kv_fetch_candidates(snapshot, family_capabilities=capabilities)
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert len(result.candidates) == 200
    # Eagerly copying 32 KB of metadata for 200 candidates retains over 6 MB.
    assert retained - before < 2 * 1024 * 1024
    selected = result.candidates[0].snapshot().envelope()
    assert selected.__pydantic_extra__["producer_metadata"] == wire["producer_metadata"]
    assert len(selected.actions) == 1

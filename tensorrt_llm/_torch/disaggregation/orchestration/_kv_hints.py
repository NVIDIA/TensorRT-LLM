# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private action selection for hint-to-source orchestration."""

from dataclasses import dataclass

from tensorrt_llm.kv_hints.protocol import KvHintsSnapshot


@dataclass(frozen=True, slots=True)
class _KvActionSelection:
    """Pure action selection result for one orchestration admission decision.

    Attributes:
        selected_action: Immutable one-action envelope, including its envelope
            metadata and the complete selected action. Action-to-source
            integration uses this value to prepare the backend hint.
        unknown_actions: Number of actions with unrecognized action types.
        unsupported_version_actions: Number of kv.fetch actions whose versions
            the configured action consumer does not support.
        extra_fetch_actions: Number of additional supported kv.fetch actions ignored.
    """

    selected_action: KvHintsSnapshot | None
    unknown_actions: int = 0
    unsupported_version_actions: int = 0
    extra_fetch_actions: int = 0


def _select_kv_fetch(
    snapshot: KvHintsSnapshot | None,
    *,
    supported_action_versions: frozenset[str],
) -> _KvActionSelection:
    """Select the first fetch whose action version the consumer supports.

    Provider payloads remain opaque. Selection is private planning policy,
    separate from the public envelope format.

    Args:
        snapshot: Immutable admitted metadata. Orchestration calls this once per
            request and accounts for the returned action counts on its owner.
        supported_action_versions: Exact versions supported by the configured
            action consumer, independent of the envelope protocol version.
            Keep this set fixed for the request; rejection must not trigger
            reselection with a different set. An empty set selects nothing.

    Returns:
        The complete selected action and envelope metadata, plus ignored counts.
        Unsupported fetch versions are skipped before selection. After
        selection, runtime or provider rejection never advances to a later
        action. Payload validation belongs to the consumer and provider.
        This function has no counters or other process/rank side effects.
    """
    if snapshot is None:
        return _KvActionSelection(None)
    envelope = snapshot.envelope()
    selected = None
    unknown_actions = 0
    unsupported_version_actions = 0
    extra_fetch_actions = 0
    for action in envelope.actions:
        if action.action_type != "kv.fetch":
            unknown_actions += 1
        elif action.action_version not in supported_action_versions:
            unsupported_version_actions += 1
        elif selected is None:
            selected = action
        else:
            extra_fetch_actions += 1
    selected_action = None
    if selected is not None:
        envelope.actions = [selected]
        selected_action = KvHintsSnapshot(envelope.model_dump_json().encode("utf-8"))
    return _KvActionSelection(
        selected_action=selected_action,
        unknown_actions=unknown_actions,
        unsupported_version_actions=unsupported_version_actions,
        extra_fetch_actions=extra_fetch_actions,
    )

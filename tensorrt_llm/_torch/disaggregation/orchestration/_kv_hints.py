# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private capability matching for hint-to-source orchestration."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from tensorrt_llm.kv_hints.protocol import KvHintsSnapshot


@dataclass(frozen=True, slots=True)
class _KvFetchCandidate:
    """One supported action with its possible backend families.

    Candidates share the immutable input envelope and retain action positions.
    Materialize a one-action snapshot only when a consumer needs it, so retained
    candidates do not each copy the envelope metadata.

    Attributes:
        families: Families that support this action's type and version. This
            unordered set does not select or authorize a concrete binding.
    """

    families: frozenset[str]
    _envelope_snapshot: KvHintsSnapshot = field(repr=False)
    _action_index: int

    def snapshot(self) -> KvHintsSnapshot:
        """Materialize this action without exposing mutable retained state.

        Returns:
            An immutable one-action envelope with all envelope and action
            metadata preserved. Each call returns an independent snapshot.
        """
        envelope = self._envelope_snapshot.envelope()
        envelope.actions = [envelope.actions[self._action_index]]
        return KvHintsSnapshot(envelope.model_dump_json().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _KvFetchCandidates:
    """Capability matches and per-action counts for one admitted envelope.

    Attributes:
        candidates: All supported fetch actions in producer order, one candidate
            per action even when several families support it.
        unknown_actions: Actions whose type is not recognized by this consumer.
        unsupported_version_actions: Fetch actions whose exact type/version pair
            is not supported by any configured family.
    """

    candidates: tuple[_KvFetchCandidate, ...]
    unknown_actions: int = 0
    unsupported_version_actions: int = 0


def _collect_kv_fetch_candidates(
    snapshot: KvHintsSnapshot | None,
    *,
    family_capabilities: Mapping[str, frozenset[tuple[str, str]]],
) -> _KvFetchCandidates:
    """Collect supported fetch actions without selecting a source or binding.

    Args:
        snapshot: Immutable admitted metadata. Orchestration collects candidates
            once per request and accounts for the returned counts.
        family_capabilities: Assembly-supplied family IDs mapped to supported
            (action_type, action_version) pairs. Keep this capability snapshot
            fixed for the request. A family can have multiple bindings; matching
            a family does not prove a specific binding's capability or authority.

    Returns:
        All supported fetch actions in producer order, with unordered family
        sets and disjoint unknown/unsupported counts. Count each action once,
        regardless of the number of matching families. Empty capabilities match
        nothing. Action identifiers do not deduplicate producer entries.

        Orchestration applies the first-candidate policy and resolves one trusted
        binding before provider contact. Authorization, payload validation or
        provider rejection must not trigger another candidate, family or binding.
        This helper interprets no provider payloads and performs no I/O or global
        accounting. It does not classify retained candidates as ignored.
    """
    if snapshot is None:
        return _KvFetchCandidates(())
    capabilities = tuple(
        (family, frozenset(actions)) for family, actions in family_capabilities.items()
    )
    envelope = snapshot.envelope()
    candidates = []
    unknown_actions = 0
    unsupported_version_actions = 0
    for index, action in enumerate(envelope.actions):
        if action.action_type != "kv.fetch":
            unknown_actions += 1
            continue
        action_key = (action.action_type, action.action_version)
        families = frozenset(family for family, actions in capabilities if action_key in actions)
        if not families:
            unsupported_version_actions += 1
            continue
        candidates.append(_KvFetchCandidate(families, snapshot, index))
    return _KvFetchCandidates(
        candidates=tuple(candidates),
        unknown_actions=unknown_actions,
        unsupported_version_actions=unsupported_version_actions,
    )

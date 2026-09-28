# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the cache-transfer contract promises, pinned.

These are the invariants the rest of the design leans on: that ``reports_pending`` answers one
question and only one -- whether a report is still owed -- that success cannot be spelled with a
report still outstanding, and that the two things nobody can recompute -- whether a piece ends the
series, and how each group's ids are to be read -- have to be stated rather than defaulted.
"""

import dataclasses

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import (
    Attempt,
    CacheKind,
    Cancelled,
    Chunk,
    Delivered,
    Failed,
    Fetches,
    Publishes,
    TokenRange,
)

pytestmark = pytest.mark.cpu_only


def _owes_a_report(outcome) -> bool:
    """No conclusion yet, or one that is still owed a report.

    Local to the tests on purpose. It used to be exported from the contract, where the name read
    like permission to hand memory back -- which is the one thing it never meant.
    """
    return outcome is None or outcome.reports_pending


def _chunk(**overrides) -> Chunk:
    fields = dict(
        block_ids_per_layer_groups=[np.array([1, 2], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=8),
        is_last=True,
    )
    fields.update(overrides)
    return Chunk(**fields)


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def test_token_range_may_be_empty_but_not_reversed():
    assert TokenRange(start=8, end=8).start == 8
    with pytest.raises(ValueError):
        TokenRange(start=10, end=3)
    with pytest.raises(ValueError):
        TokenRange(start=-1, end=5)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_is_last_has_no_default():
    """Getting it wrong ends a transfer early and in silence, so it cannot be left out."""
    with pytest.raises(TypeError):
        Chunk(
            block_ids_per_layer_groups=[],
            kind_per_layer_group=[],
            token_range=TokenRange(start=0, end=8),
        )


def test_every_layer_group_is_given_a_kind():
    """Both kinds are arrays of integers, so a missing kind cannot be noticed by reading the ids."""
    with pytest.raises(ValueError):
        _chunk(
            block_ids_per_layer_groups=[
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ],
        )


def test_kinds_stay_in_step_with_the_block_lists():
    """The pairing is positional, which is the only thing that ties a slot id to being a slot."""
    chunk = _chunk(
        block_ids_per_layer_groups=[
            np.array([1, 2], dtype=np.int64),
            np.array([9], dtype=np.int64),
        ],
        kind_per_layer_group=[CacheKind.PAGED, CacheKind.STATE],
    )
    assert chunk.kind_per_layer_group[1] is CacheKind.STATE


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_delivered_can_never_be_owed_a_report():
    """Success on the strength of a report that has not arrived is refused."""
    assert Delivered(token_end=8).reports_pending is False
    with pytest.raises(ValueError):
        Delivered(token_end=8, reports_pending=True)


def test_delivered_pending_is_spelled_out_not_absent():
    """Every member answers ``pending``, so reading it never needs a default to fall back on."""
    for outcome in (
        Delivered(token_end=1),
        Failed(reason="x", reports_pending=False),
        Cancelled(by_peer=False, reports_pending=False),
    ):
        assert outcome.reports_pending is False


def test_cancelled_is_pending_until_told_otherwise():
    """A cancellation stops the ask, not the writers already holding the destination."""
    assert Cancelled().reports_pending is True
    assert Cancelled().by_peer is False


def test_outcomes_are_frozen():
    """A stored outcome must not be mutated into a stale claim."""
    outcome = Delivered(token_end=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.token_end = 5


# ---------------------------------------------------------------------------
# Whether a report is still owed
# ---------------------------------------------------------------------------


def test_no_conclusion_counts_as_owed():
    """A transfer that has not concluded may be queued and about to write."""
    assert _owes_a_report(None) is True


def test_a_delivery_owes_nothing_further():
    assert _owes_a_report(Delivered(token_end=8)) is False


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (Failed(reason="peer died", reports_pending=True), True),
        (Failed(reason="peer died", reports_pending=False), False),
        (Cancelled(reports_pending=True), True),
        (Cancelled(reports_pending=False), False),
    ],
)
def test_what_is_owed_does_not_follow_from_the_kind_of_ending(outcome, expected):
    """How a delivery ended and what it still owes are separate; neither is read off the other."""
    assert _owes_a_report(outcome) is expected


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def test_poll_alone_satisfies_attempt():
    class OnePiece:
        def poll(self):
            return None

    assert isinstance(OnePiece(), Attempt)


def test_reading_and_offering_are_separate_to_implement():
    """A store that only answers reads is asked for nothing it cannot do."""

    class ReadOnly:
        def fetch(self, extent, *, src=None):
            raise NotImplementedError

    assert isinstance(ReadOnly(), Fetches)
    assert not isinstance(ReadOnly(), Publishes)

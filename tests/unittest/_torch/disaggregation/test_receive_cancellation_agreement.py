# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-neutral logical agreement must not substitute for physical evidence."""

import pytest

from tensorrt_llm._torch.disaggregation.orchestration.cancellation import (
    ReceiveCancellation,
    ReceiveCancellationVote,
    agree_receive_cancellation,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("votes", [[], [None], [ReceiveCancellationVote(False, True), None]])
def test_missing_participant_cannot_commit_or_release(
    votes: list[ReceiveCancellationVote | None],
) -> None:
    assert agree_receive_cancellation(ReceiveCancellation.PENDING, votes) == (
        ReceiveCancellation.PENDING,
        False,
    )


@pytest.mark.parametrize("prior_failure", [False, True])
def test_complete_outcome_vote_does_not_wait_for_access_end(prior_failure: bool) -> None:
    votes = [ReceiveCancellationVote(False, True), ReceiveCancellationVote(prior_failure, False)]
    expected = ReceiveCancellation.FAILED if prior_failure else ReceiveCancellation.CANCELLED
    assert agree_receive_cancellation(ReceiveCancellation.PENDING, votes) == (expected, False)


@pytest.mark.parametrize("committed", [ReceiveCancellation.CANCELLED, ReceiveCancellation.FAILED])
@pytest.mark.parametrize("late_failure", [False, True])
def test_late_evidence_changes_release_not_committed_outcome(
    committed: ReceiveCancellation, late_failure: bool
) -> None:
    assert agree_receive_cancellation(
        committed,
        [ReceiveCancellationVote(late_failure, False), ReceiveCancellationVote(False, True)],
    ) == (committed, False)
    assert agree_receive_cancellation(
        committed,
        [ReceiveCancellationVote(late_failure, True), ReceiveCancellationVote(False, True)],
    ) == (committed, True)
    assert agree_receive_cancellation(committed, [None]) == (committed, False)


def test_safe_first_vote_can_commit_and_retire_together() -> None:
    assert agree_receive_cancellation(
        ReceiveCancellation.PENDING, [ReceiveCancellationVote(False, True)]
    ) == (ReceiveCancellation.CANCELLED, True)


def test_unsupported_request_cannot_enter_agreement() -> None:
    with pytest.raises(ValueError, match="Unsupported cancellation"):
        agree_receive_cancellation(ReceiveCancellation.NOT_HANDLED, [])

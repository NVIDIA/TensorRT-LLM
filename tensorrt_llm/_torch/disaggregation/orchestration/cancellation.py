# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-level cancellation agreement, independent of backend evidence collection."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ReceiveCancellation(Enum):
    """Logical cancellation disposition; no member authorizes memory reuse."""

    NOT_HANDLED = "not_handled"
    PENDING = "pending"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class ReceiveCancellationVote:
    """Facts supplied by one qualified owner after closing further admission."""

    prior_failure: bool
    access_ended: bool


def agree_receive_cancellation(
    committed: ReceiveCancellation,
    votes: Sequence[ReceiveCancellationVote | None],
) -> tuple[ReceiveCancellation, bool]:
    """Commit one logical outcome and separately agree physical retirement.

    Args:
        committed: PENDING until the first complete vote; terminal afterwards.
        votes: One entry per required participant; None means no vote yet.

    Returns:
        The immutable logical outcome and whether every participant supplied
        access-end evidence. The binding must still close its real owners.

    Raises:
        ValueError: An unsupported request was included in agreement.
    """
    if committed is ReceiveCancellation.NOT_HANDLED:
        raise ValueError("Unsupported cancellation cannot enter group agreement")
    if not votes or any(vote is None for vote in votes):
        return committed, False
    present = [vote for vote in votes if vote is not None]
    if committed is ReceiveCancellation.PENDING:
        committed = (
            ReceiveCancellation.FAILED
            if any(vote.prior_failure for vote in present)
            else ReceiveCancellation.CANCELLED
        )
    return committed, all(vote.access_ended for vote in present)

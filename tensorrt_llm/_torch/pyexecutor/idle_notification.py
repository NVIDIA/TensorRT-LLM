# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered idle-request notification before the existing request collective."""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .executor_request_queue import RequestQueueItem


class IdleRequestNotification:
    """Coordinate a leader's blocking fetch with lossless follower wakeups.

    The caller owns notification transport, socket lifetime and rank-wide
    failure containment. All ranks must enter this protocol in the same order.
    """

    _WAIT_FOR_REQUEST = -1

    def __init__(
        self,
        send_notice: Callable[[tuple[int, bool]], None],
        receive_notice: Callable[[], tuple[int, bool]],
    ) -> None:
        self._send_notice = send_notice
        self._receive_notice = receive_notice
        self._epoch = 0

    def count(
        self,
        rank: int,
        requests: list["RequestQueueItem"],
        broadcast: Callable[[int], int],
        fetch: Callable[[], list["RequestQueueItem"]] | None,
    ) -> tuple[list["RequestQueueItem"], int]:
        """Publish idle state, fetch on the leader, then resume the count."""
        proposed = 0
        if rank == 0:
            proposed = (
                self._WAIT_FOR_REQUEST if fetch is not None and not requests else len(requests)
            )
        count = broadcast(proposed)
        if type(count) is not int or count < self._WAIT_FOR_REQUEST:
            raise ValueError("Invalid shared idle-request count")
        if count == self._WAIT_FOR_REQUEST:
            if rank == 0:
                if fetch is None:
                    raise ValueError("The idle leader must provide its queue fetch")
                try:
                    requests = fetch()
                    if not isinstance(requests, list):
                        raise TypeError("The request queue must return its original request list")
                except Exception:
                    self._send_notice((self._epoch, False))
                    raise
                self._send_notice((self._epoch, True))
            else:
                epoch, ready = self._receive_notice()
                if type(epoch) is not int or type(ready) is not bool or epoch != self._epoch:
                    raise ValueError("Out-of-order idle-request notification")
                if not ready:
                    raise RuntimeError("Leader request fetch failed")
            count = broadcast(len(requests) if rank == 0 else 0)
            if type(count) is not int or count < 0:
                raise ValueError("Invalid post-notification request count")
        self._epoch += 1
        return requests, count

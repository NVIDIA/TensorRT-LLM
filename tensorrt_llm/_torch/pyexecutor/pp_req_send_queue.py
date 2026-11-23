# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded queue of the pipeline-parallel request-payload sends.

With ``TRTLLM_PP_REQ_SEND_ASYNC=1`` every PP rank but the last forwards each new-request
payload to the next PP rank with a non-blocking send. By default at most one such send is in
flight: before posting the next payload the rank waits on the previous send, which cannot
complete before the next rank has posted the matching receive at the top of its own loop. Each
arrival therefore couples a rank's loop to the downstream rank's loop top.

A queue of depth ``N`` lets up to ``N`` request sends be in flight; the rank waits on the oldest
one only when ``N`` are already outstanding. The executor uses ``DEFAULT_PP_REQ_SEND_DEPTH``
(1), which is exactly the single-handle behaviour.

Correctness does not depend on the depth:

* The payload is serialized when the send is posted (``pkl5.Intracomm.isend`` pickles before it
  calls ``MPI_Isend`` in mpi4py 3.1 and 4.x, ``TorchDist.isend_object`` builds its byte tensor
  first). One exception: with pickle protocol 5, ``pkl5`` sends an out-of-band buffer of at
  least ``MPI4PY_PICKLE_THRESHOLD`` bytes (0.25 MiB by default), such as a contiguous numpy
  array, straight from that object's memory, so an in-place change of such an array before its
  send completes would reach the receiver. No request payload exposes such a buffer (the
  C++ request pickles to lists and bytes, torch tensors pickle through bytes). The window is
  one request send at the default depth and ``N`` sends at depth ``N``.
* The serialized buffers are owned by the request handle (``mpi4py`` keeps each one in the
  low-level ``Request.ob_buf``, which the ``pkl5.Request`` tuple holds); the queue keeps every
  handle referenced until it has been waited on.
* MPI does not let messages with the same communicator, source and tag overtake one another,
  and the receiver always names the source and the dedicated payload tag, so the next rank
  receives the payloads in the order they were sent.
* Every rank calls the request broadcast once per loop iteration in the same order, so the
  ``i``-th payload a rank receives is still the ``i``-th payload the first rank sent.

How far a rank can run ahead of the next one is also bounded by the other per-iteration PP
channels (the sample-state relay, the lagged reports of the first rank); a depth at or above
that bound no longer limits the lead.
"""

import os
from collections import deque
from typing import Any, Deque, Mapping, Optional

PP_REQ_SEND_ASYNC_ENV = "TRTLLM_PP_REQ_SEND_ASYNC"
DEFAULT_PP_REQ_SEND_DEPTH = 1
LOG_PREFIX = "[pp_req_send_depth]"


def pp_req_send_async_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``TRTLLM_PP_REQ_SEND_ASYNC=1`` selects the non-blocking request send."""
    environ = os.environ if environ is None else environ
    return environ.get(PP_REQ_SEND_ASYNC_ENV, "0") == "1"


class PPRequestSendQueue:
    """The request-payload send handles a PP rank has posted and not yet waited on.

    Only the executor thread uses an instance, so it takes no lock.
    """

    def __init__(self, depth: int = DEFAULT_PP_REQ_SEND_DEPTH):
        if depth < 1:
            raise ValueError(f"request send depth {depth} must be a positive integer")
        self.depth = depth
        self._handles: Deque[Any] = deque()
        self.num_posted = 0
        self.num_waited = 0
        self.max_in_flight = 0

    def __len__(self) -> int:
        return len(self._handles)

    def full(self) -> bool:
        """Whether posting another send would put more than ``depth`` in flight."""
        return len(self._handles) >= self.depth

    def wait_oldest(self) -> None:
        """Wait on the oldest send in flight and release its handle."""
        handle = self._handles.popleft()
        handle.wait()
        self.num_waited += 1

    def push(self, handle: Any) -> None:
        """Track a posted send; ``None`` (a blocking send already completed) is not tracked."""
        if handle is None:
            return
        if self.full():
            raise RuntimeError(
                f"{LOG_PREFIX} posted a request send with {len(self._handles)} in flight "
                f"(depth {self.depth}); wait on the oldest first"
            )
        self._handles.append(handle)
        self.num_posted += 1
        self.max_in_flight = max(self.max_in_flight, len(self._handles))

    def drain(self) -> int:
        """Wait on every send still in flight, oldest first; returns how many it waited on.

        A no-op at depth 1, where the loop never waited on its last request send either: the
        default teardown stays as it was.
        """
        if self.depth <= DEFAULT_PP_REQ_SEND_DEPTH:
            return 0
        drained = 0
        while self._handles:
            self.wait_oldest()
            drained += 1
        return drained

    def summary(self) -> str:
        """One line on the sends this queue tracked."""
        return (
            f"{LOG_PREFIX} depth={self.depth}: posted {self.num_posted}, "
            f"waited {self.num_waited}, max in flight {self.max_in_flight}, "
            f"in flight {len(self._handles)}"
        )

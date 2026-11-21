# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded, non-blocking receives over an mpi4py communicator.

Control channels that must not hang a rank -- the sleep/wakeup ACKs and the
memory monitor's report gathering -- poll with ``Iprobe`` instead of blocking
in ``recv``, which some MPI implementations implement by spinning a CPU core,
and give up at a deadline. A reply that arrives after its request was
abandoned is recognised by the caller's ``accept`` predicate and dropped.
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["message_ready", "recv_until", "try_recv"]


def message_ready(comm: Any, source: int, tag: int) -> bool:
    """Whether a message from ``source`` with ``tag`` can be received now.

    Accepts both mpi4py's lower-case object API (``iprobe``) and the
    buffer API (``Iprobe``).
    """
    if hasattr(comm, "iprobe"):
        return bool(comm.iprobe(source=source, tag=tag))
    if hasattr(comm, "Iprobe"):
        return bool(comm.Iprobe(source=source, tag=tag))
    raise RuntimeError("communicator does not support a nonblocking probe")


def try_recv(
    comm: Any,
    source: int,
    tag: int,
    accept: Callable[[Any], bool] | None = None,
) -> tuple[bool, Any]:
    """Receive the first ready message ``accept`` keeps, without waiting.

    Ready messages that ``accept`` rejects are received and dropped.
    Returns ``(True, message)``, or ``(False, None)`` once nothing is ready.
    """
    while message_ready(comm, source, tag):
        message = comm.recv(source=source, tag=tag)
        if accept is None or accept(message):
            return True, message
    return False, None


def recv_until(
    comm: Any,
    source: int,
    tag: int,
    deadline: float,
    *,
    poll_interval_s: float,
    accept: Callable[[Any], bool] | None = None,
    what: str = "message",
) -> Any:
    """Receive a message ``accept`` keeps before a ``time.monotonic()`` deadline.

    Raises:
        TimeoutError: no such message arrived by ``deadline``; the message
            names ``what`` and the source rank.
    """
    while True:
        received, message = try_recv(comm, source, tag, accept)
        if received:
            return message
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {what} from rank {source} by deadline {deadline:.6f}"
            )
        time.sleep(poll_interval_s)

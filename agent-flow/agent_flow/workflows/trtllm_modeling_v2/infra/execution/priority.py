# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The class a job queues at in a session's spool: accuracy first, then features, then perf.

The spool scheduler starts a waiting job of a higher ``priority`` first. The
class is set where the job's purpose is known (:func:`job_priority`) and read
where the job is submitted (:func:`current_job_priority`), so the code
between needs no argument for it.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator

#: Checks on the way to the first unit's accuracy: its boot and benchmark.
ACCURACY = 2
#: Every other check: the feature units, the variants, the proofs, delivery.
FEATURE = 1
#: The tune campaign's measurements and candidates.
PERF = 0

_local = threading.local()


@contextlib.contextmanager
def job_priority(priority: int) -> Iterator[None]:
    """Jobs submitted in this thread, inside the block, queue at ``priority``."""
    previous = getattr(_local, "priority", None)
    _local.priority = priority
    try:
        yield
    finally:
        _local.priority = previous


def current_job_priority() -> int:
    """The class this thread's jobs queue at; :data:`PERF` (the scheduler's 0) when none was set."""
    priority = getattr(_local, "priority", None)
    return PERF if priority is None else int(priority)


__all__ = ["ACCURACY", "FEATURE", "PERF", "current_job_priority", "job_priority"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast-path reports a kernel module makes from inside a call: ``taken`` (an engagement line), ``missed`` (a covered
shape that cannot run: raises under ``modeling_v2='require'``), ``uncovered`` and ``capture_missed``.

The policy (logging once, raising under require) is the target's: ``modeling.py`` installs its functions here at import,
so a kernel reports without importing the target module. Until then every report is a no-op.

``work`` is the gate-e work-done counter (no install needed): the code that does an item's work calls
``work(item, detail)`` from its host code each time it does it. Per rank, the first call logs the item's engagement
line and every power-of-two count logs the running total; the totals are logged once more at process exit. Under CUDA
graphs the host code runs at capture and on eager steps only, so the count is captures + eager calls (replays are not
counted), and every line says so.
"""

from __future__ import annotations

import atexit
import threading
from typing import Callable, Dict, Optional


def _noop(*args, **kwargs) -> None:
    return None


taken: Callable[..., None] = _noop
missed: Callable[..., None] = _noop
uncovered: Callable[..., None] = _noop
capture_missed: Callable[..., None] = _noop


def install(
    *,
    on_taken: Callable[[str, Optional[int], str], None],
    on_missed: Callable[[str, Optional[int], str], None],
    on_uncovered: Callable[[str, str], None],
    on_capture_missed: Callable[[str, Optional[int], str], None],
) -> None:
    """Route this module's reports to the target's policy."""
    global taken, missed, uncovered, capture_missed
    taken, missed, uncovered, capture_missed = on_taken, on_missed, on_uncovered, on_capture_missed


#: The tag every work line carries (the eng kit greps it with the item id).
TAG = "[M3 sm107 tp2]"
#: What a count counts (said in every line).
HOST_COUNT = "host count: CUDA-graph captures + eager calls, replays not counted"

_work_lock = threading.Lock()
_work_counts: Dict[str, int] = {}
_work_exit_registered = [False]


def _rank() -> int:
    from tensorrt_llm._utils import mpi_rank

    return mpi_rank()


def _info(message: str) -> None:
    from tensorrt_llm.logger import logger

    logger.info(message)


def work(item: str, detail: str = "") -> int:
    """Count one unit of ``item``'s work on this rank and return the running count. Logs the engagement line at the
    first count (with ``detail``) and the running total at every power of two."""
    with _work_lock:
        count = _work_counts.get(item, 0) + 1
        _work_counts[item] = count
        if not _work_exit_registered[0]:
            _work_exit_registered[0] = True
            atexit.register(log_work_totals)
    if count & (count - 1) == 0:
        if count == 1:
            what = f" ({detail})" if detail else ""
            _info(f"{TAG} rank {_rank()}: {item} engaged{what}; work 1 ({HOST_COUNT})")
        else:
            _info(f"{TAG} rank {_rank()}: {item} work {count} ({HOST_COUNT})")
    return count


def work_count(item: str) -> int:
    """This rank's running count of ``item`` (0 if it never ran)."""
    with _work_lock:
        return _work_counts.get(item, 0)


def work_totals() -> Dict[str, int]:
    """Every item's running count on this rank."""
    with _work_lock:
        return dict(_work_counts)


def log_work_totals() -> None:
    """The final per-rank counts, one line (registered at exit by the first ``work`` call)."""
    totals = work_totals()
    if totals:
        body = ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        _info(f"{TAG} rank {_rank()}: work-done final ({HOST_COUNT}): {body}")


def reset_work() -> None:
    """Forget every count (tests)."""
    with _work_lock:
        _work_counts.clear()

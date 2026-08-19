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
"""Phase-0 GIL tick probe (design doc Section 5.3).

Run INSIDE the gen-side process under a representative decode load (e.g.
import and call :func:`start_probe` from a site near the executor loop, or
run this module standalone for a no-load baseline):

    python3 -m tensorrt_llm._torch.disaggregation.bounce_v2.probe [seconds]

It models the reactor's 1 ms wait and reports the tick-interval distribution.
GATE: p99 < 5 ms, p99.9 < 10 ms, max < 20 ms, and the fraction of any 21 ms
sliding window spent >1 ms late < 50%. A failing gate selects the
subprocess-reactor variant BEFORE any deployment relies on the in-process
reactor.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

__all__ = ["run_probe", "start_probe"]

_TICK_S = 0.001
_WINDOW_NS = 21_000_000  # the 8 x 128 MiB pipeline slack at 50 GB/s
_LATE_NS = 2_000_000  # a tick counts late past 1 ms of intended sleep


def _probe_loop(out: list, stop: threading.Event, tick: float = _TICK_S) -> None:
    last = time.monotonic_ns()
    while not stop.is_set():
        time.sleep(tick)  # models the reactor's 1 ms poll cap
        now = time.monotonic_ns()
        out.append(now - last)
        last = now


def start_probe() -> "tuple[list, threading.Event, threading.Thread]":
    """Start the probe thread inside an existing process; returns
    ``(intervals_ns, stop_event, thread)`` — set the event, join, then feed
    the intervals to :func:`run_probe`'s reporting via ``report()``."""
    out: list = []
    stop = threading.Event()
    thread = threading.Thread(target=_probe_loop, args=(out, stop), daemon=True)
    thread.start()
    return out, stop, thread


def report(intervals_ns: "list[int]") -> bool:
    """Print the distribution + gate verdict; returns pass/fail."""
    arr = np.asarray(intervals_ns, dtype=np.int64)
    if arr.size == 0:
        print("probe: no samples")
        return False
    ms = arr / 1e6
    p50, p99, p999 = (float(np.percentile(ms, p)) for p in (50, 99, 99.9))
    worst = float(ms.max())
    # Fraction of 21 ms sliding windows dominated by late ticks.
    late = arr > _LATE_NS
    ends = np.cumsum(arr)
    late_ns = np.cumsum(np.where(late, arr, 0))
    starts = np.searchsorted(ends, ends - _WINDOW_NS, side="left")
    window_late = late_ns - np.where(starts > 0, late_ns[starts - 1], 0)
    window_total = ends - np.where(starts > 0, ends[starts - 1], 0)
    frac = float((window_late / np.maximum(window_total, 1) > 0.5).mean())
    ok = p99 < 5.0 and p999 < 10.0 and worst < 20.0 and frac < 0.5
    print(
        f"probe: n={arr.size} p50={p50:.3f} ms p99={p99:.3f} ms "
        f"p99.9={p999:.3f} ms max={worst:.3f} ms late-window-frac={frac:.3f}"
    )
    print(f"probe: GATE {'PASS' if ok else 'FAIL'} (p99<5, p99.9<10, max<20, late-frac<0.5)")
    return ok


def run_probe(duration_s: float = 60.0) -> bool:
    """Run the probe for ``duration_s`` seconds and report."""
    out, stop, thread = start_probe()
    time.sleep(duration_s)
    stop.set()
    thread.join()
    return report(out)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    sys.exit(0 if run_probe(seconds) else 1)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1d.4 (stub): Fault-injection driver for the WideEP FT MVP prototype.

Launches ``kill_and_survive_worker.py`` under ``mpirun``, parses the per-rank
JSON-line event stream, SIGKILLs a victim rank at the configured wall-clock
moment, and emits the per-event timeline JSON described in
``mvp-prototype-plan.md`` §4.4.

Per-event timeline schema (``--output``):

    {
      "config": {...},
      "events": {
         "t_kill"                       : <sec since loop_start>,
         "t_watchdog_fires"             : <first survivor's mark_failed>,
         "t_mark_failed_propagated"     : <every survivor saw it>,
         "t_iteration_boundary"         : <next reconfigure-triggered iter>,
         "t_reconfigure_done"           : <reconfigure_mask_only returned>,
         "t_first_new_request_completed": <first iter past reconfigure>
      },
      "raw_events": [...one per stream line...],
      "exit_codes": {<rank>: <code>}
    }

The wall-clock origin is ``t = 0`` at the *first survivor's* loop_start
(``loop_start`` events are barriered so all survivors see ~the same instant).

Production PR 1d.4 wraps this in a pytest fixture, supports kill-during-
dispatch / kill-during-combine / etc. via CUDA-event hooks into the kernel,
adds per-token correctness assertions, and integrates with the test suite.
This stub is a one-off CLI tool; assertions are command-line warnings, not
test framework integration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Optional


def _find_mpirun() -> str:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        sys.exit("error: mpirun not found in PATH")
    return mpirun


def _build_argv(args: argparse.Namespace, worker_path: str) -> list[str]:
    """Construct the mpirun command line.

    Per audit-1a Day 2 finding (and the prototype plan §3): without
    ``--mca orte_enable_recovery 1`` mpirun terminates survivors on any
    abnormal child exit, so this flag is non-optional for the prototype.
    """
    mpirun = _find_mpirun()
    return [
        mpirun,
        "-np",
        str(args.np),
        "--mca",
        "orte_enable_recovery",
        "1",
        # Stream stdout per-rank with a tag so we can demultiplex in real time.
        "--tag-output",
        sys.executable,
        worker_path,
        "--iterations",
        str(args.iterations),
        "--iter-sleep-sec",
        str(args.iter_sleep_sec),
    ]


# Event stream parsing -------------------------------------------------------
#
# mpirun --tag-output prefixes each line like "[1,RANK]<stdout>:". Strip that
# tag, then attempt to parse JSON; non-JSON lines are forwarded to the
# driver's stderr so the user can see worker errors.

_TAG_RE = re.compile(r"^\[1,(\d+)\]<\w+>:(.*)$")


def _parse_tagged_line(line: str) -> Optional[tuple[int, dict[str, Any]]]:
    m = _TAG_RE.match(line.rstrip())
    if m is None:
        return None
    rank = int(m.group(1))
    body = m.group(2).strip()
    if not body.startswith("{"):
        return None
    try:
        evt = json.loads(body)
    except json.JSONDecodeError:
        return None
    return rank, evt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=4, help="Number of MPI ranks")
    parser.add_argument("--victim-rank", type=int, default=2, help="Rank to SIGKILL mid-run")
    parser.add_argument(
        "--kill-after-iter",
        type=int,
        default=3,
        help="SIGKILL the victim after their N-th heartbeat (which fires every 20 iter)",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--iter-sleep-sec", type=float, default=0.05)
    parser.add_argument(
        "--budget-sec",
        type=float,
        default=10.0,
        help="Recovery budget; warn if t_first_new_request_completed exceeds this",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the timeline JSON (default: stdout pretty-print only)",
    )
    args = parser.parse_args()

    if not 0 <= args.victim_rank < args.np:
        sys.exit(f"--victim-rank {args.victim_rank} not in [0, {args.np})")

    # 1d.0 is real in this branch; remind the operator the env var is required.
    if os.environ.get("TLLM_FAULT_TOLERANCE_MODE") != "1":
        print(
            "warning: TLLM_FAULT_TOLERANCE_MODE != '1'; the SIGKILL'd rank's "
            "signal handler will MPI_Abort and the survivors will die. "
            "Set TLLM_FAULT_TOLERANCE_MODE=1 in the calling shell.",
            file=sys.stderr,
        )

    worker_path = os.path.join(os.path.dirname(__file__), "kill_and_survive_worker.py")
    cmd = _build_argv(args, worker_path)

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    # State accumulated as the event stream arrives.
    raw_events: list[dict[str, Any]] = []
    pid_by_rank: dict[int, int] = {}
    loop_start_by_rank: dict[int, float] = {}
    watchdog_fired_at: Optional[tuple[int, float]] = None  # (discoverer, monotonic_sec)
    propagated_to: set[int] = set()
    reconfigure_done_at: dict[int, float] = {}
    first_post_reconfigure_iter_at: dict[int, float] = {}
    kill_at_monotonic: Optional[float] = None
    killed = False

    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            parsed = _parse_tagged_line(raw_line)
            if parsed is None:
                # Non-JSON noise (mpirun banner, worker stderr, etc.)
                sys.stderr.write(raw_line)
                continue
            rank, evt = parsed
            evt["_rank"] = rank
            raw_events.append(evt)
            etype = evt.get("event")
            wt = evt.get("wall_time_sec")
            if etype == "worker_start":
                pid_by_rank[rank] = int(evt["pid"])
            elif etype == "loop_start":
                loop_start_by_rank[rank] = float(wt)
                # Inject the SIGKILL once enough iterations have run on the
                # victim. The worker emits heartbeats every 20 iterations;
                # we kill after its (kill_after_iter)-th heartbeat to make
                # the kill point reproducible across runs.
            elif etype == "heartbeat" and rank == args.victim_rank and not killed:
                if int(evt.get("iteration", 0)) >= args.kill_after_iter * 20:
                    victim_pid = pid_by_rank.get(args.victim_rank)
                    if victim_pid is not None:
                        kill_at_monotonic = time.monotonic()
                        os.kill(victim_pid, signal.SIGKILL)
                        killed = True
                        sys.stderr.write(
                            f"[driver] SIGKILL pid={victim_pid} (rank={args.victim_rank})\n"
                        )
            elif etype == "watchdog_marked_failed":
                if watchdog_fired_at is None:
                    watchdog_fired_at = (rank, float(wt))
            elif etype == "broadcast_received":
                propagated_to.add(rank)
            elif etype == "reconfigure_done":
                reconfigure_done_at.setdefault(rank, float(wt))
                first_post_reconfigure_iter_at.setdefault(
                    rank,
                    float(wt) + args.iter_sleep_sec,  # conservative estimate
                )
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        raise

    rc = proc.wait()
    sys.stderr.write(f"[driver] mpirun exited with code {rc}\n")

    # Aggregate the timeline ------------------------------------------------
    # Origin: first loop_start observed across survivors.
    survivors = [r for r in range(args.np) if r != args.victim_rank]
    survivor_starts = [loop_start_by_rank[r] for r in survivors if r in loop_start_by_rank]
    if not survivor_starts:
        sys.exit("error: no survivor reported loop_start; run did not progress")
    origin = min(survivor_starts)

    def _rel(t: Optional[float]) -> Optional[float]:
        return None if t is None else round(t - origin, 4)

    full_propagated = propagated_to.union({watchdog_fired_at[0]} if watchdog_fired_at else set())
    all_propagated = full_propagated.issuperset(set(survivors))
    propagated_at = (
        max(
            (
                ev["wall_time_sec"]
                for ev in raw_events
                if ev["event"] in ("watchdog_marked_failed", "broadcast_received")
                and ev["_rank"] in survivors
            ),
            default=None,
        )
        if all_propagated
        else None
    )
    first_reconfigure_at = min(reconfigure_done_at.values(), default=None)
    last_reconfigure_at = max(reconfigure_done_at.values(), default=None)
    first_post_recon_at = min(first_post_reconfigure_iter_at.values(), default=None)

    timeline = {
        "config": vars(args),
        "events": {
            "t_kill": _rel(kill_at_monotonic),
            "t_watchdog_fires": _rel(watchdog_fired_at[1] if watchdog_fired_at else None),
            "t_mark_failed_propagated": _rel(propagated_at),
            "t_iteration_boundary": _rel(first_reconfigure_at),
            "t_reconfigure_done": _rel(last_reconfigure_at),
            "t_first_new_request_completed": _rel(first_post_recon_at),
        },
        "raw_events": raw_events,
        "exit_codes": {"mpirun": rc},
        "victim_rank": args.victim_rank,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(timeline, fh, indent=2, default=str)
        sys.stderr.write(f"[driver] wrote timeline to {args.output}\n")

    # Pretty print the headline
    print()
    print("─" * 72)
    print("WideEP FT MVP prototype — kill-and-survive timeline")
    print("─" * 72)
    for k, v in timeline["events"].items():
        v_str = f"{v:7.3f} s" if v is not None else "  <not observed>"
        print(f"  {k:<30s}: {v_str}")
    print("─" * 72)
    final = timeline["events"]["t_first_new_request_completed"]
    if final is None:
        print(f"  < {args.budget_sec:.0f} s recovery budget   : ✗ DID NOT RECOVER")
        return 2
    if final < args.budget_sec:
        print(f"  < {args.budget_sec:.0f} s recovery budget   : ✓ PASS ({final:.3f} s)")
        return 0
    print(f"  < {args.budget_sec:.0f} s recovery budget   : ✗ FAIL ({final:.3f} s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

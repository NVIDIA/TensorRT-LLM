# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The controller's own deadline: inside a job it stops in time and says how to continue.

The flow submits no job for its controller. Whoever does (a person's ``sbatch``, a site's
script), a controller that finds itself inside a Slurm job stops starting turns
:data:`STOP_BEFORE_S` before that job's end (the run's own stop: turns in flight are left,
and their tasks start again as the same attempt), says in one line the command that
continues the run, and exits with :data:`EXIT_CONTINUE`. That status means one thing: the
same command, run again, goes on from the workspace. So a script that submits itself again
when the command exits 75 is all a site needs to keep a controller in jobs of its own.
"""

from __future__ import annotations

import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..files import read_json, write_json_atomic
from . import slurm as slurm_mod
from .scheduler import utc_iso

#: The exit status of a controller that stopped for its limit: to be continued.
EXIT_CONTINUE = 75
#: How long before its job's end a controller stops starting turns and hands over.
STOP_BEFORE_S = 600.0
#: How long before the end it exits whatever still runs: a job killed at its limit hands nothing over.
EXIT_BEFORE_S = 120.0
#: The reason of a run stopped for its controller's limit ends with this.
LIMIT_STOP = "stopped for the limit"
#: ``<workspace>/controller/handover.json``: what the last controller said when it stopped.
DIRNAME = "controller"
HANDOVER = "handover.json"
MODULE = "agent_flow.workflows.trtllm_modeling_v2.cli"


def job_end(environ: Mapping[str, str], *, runner: slurm_mod.Run = slurm_mod._run) -> float | None:
    """When the job this process runs in ends (epoch seconds); ``None`` outside a job.

    Slurm's own figure in the environment, else what the queue says of the job.
    """
    value = environ.get("SLURM_JOB_END_TIME", "")
    if value.isdigit() and int(value) > 0:
        return float(value)
    job = environ.get("SLURM_JOB_ID", "")
    if not job:
        return None
    try:
        done = runner(["squeue", "-h", "-j", job, "-o", "%e"])
    except Exception:
        return None
    return slurm_mod._epoch(done.stdout.strip().splitlines()[0]) if done.stdout.strip() else None


class Limit:
    """The watch on a controller's own end: stop the run, say what continues it, leave in time.

    At ``end - before_s`` it calls ``stop`` with the reason (again until the run takes it), so
    that nothing more starts, and then ``hand_over``, which says in one line what continues the
    run. If the process is still there at ``end - exit_before_s`` it calls ``leave``: a job
    killed at its limit would have said nothing.
    """

    def __init__(
        self,
        *,
        end: float,
        stop: Callable[[str], bool],
        hand_over: Callable[[], str],
        leave: Callable[[int], None],
        clock: Callable[[], float] = time.time,
        before_s: float = STOP_BEFORE_S,
        exit_before_s: float = EXIT_BEFORE_S,
        poll_s: float = 5.0,
    ) -> None:
        self.end, self.stop, self.hand_over, self.leave = end, stop, hand_over, leave
        self.clock, self.before_s, self.exit_before_s, self.poll_s = (
            clock,
            before_s,
            exit_before_s,
            poll_s,
        )
        #: Whether the limit was reached, and the line that says what continues the run.
        self.reached = False
        self.said = ""
        #: Why the run stops for the limit before it is near: nothing more can start by then.
        self.early = ""
        self._early = threading.Event()
        self._cancel = threading.Event()
        self._handed = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="mv2-limit", daemon=True)

    @property
    def reason(self) -> str:
        why = self.early or f"the controller's job ends at {utc_iso(self.end)}"
        return f"{why}: {LIMIT_STOP}"

    def now(self, why: str) -> None:
        """Stop the run for the limit now: what is left of the job is of no use to it."""
        self.early = why
        self._early.set()

    def start(self) -> "Limit":
        self._thread.start()
        return self

    def _wait_until(self, moment: float) -> bool:
        """Wait for ``moment``; False when the watch was cancelled first."""
        while self.clock() < moment and not self._early.is_set():
            if self._cancel.wait(min(self.poll_s, max(0.01, moment - self.clock()))):
                return False
        return not self._cancel.is_set()

    def _watch(self) -> None:
        if not self._wait_until(self.end - self.before_s):
            return
        self.reached = True
        taken = self.stop(self.reason)
        self.said = self.hand_over()
        self._handed.set()
        # Stopped early, a run that does not return is left as soon as at the limit itself.
        last = self.end - self.exit_before_s
        if self.early:
            last = min(last, self.clock() + self.exit_before_s)
        while self.clock() < last:
            if self._cancel.wait(self.poll_s):
                return
            # A run that had not started its loop yet takes the stop once it does.
            taken = taken or self.stop(self.reason)
        self.leave(EXIT_CONTINUE)

    def finish(self) -> bool:
        """The run returned: stop watching. Whether it returned because the limit was reached.

        A hand-over under way is waited for: the process must not go before what continues the
        run was said.
        """
        self._cancel.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            if self.reached:
                self._handed.wait(max(0.0, self.end - self.exit_before_s - self.clock()))
            self._thread.join(timeout=self.poll_s + 5)
        return self.reached


def hand_over(
    environ: Mapping[str, str],
    command: Sequence[str],
    *,
    end: float,
    workspace: str | Path,
    clock: Callable[[], float] = time.time,
    why: str = "",
) -> str:
    """What continues the run, in one line, left where the monitor reads it."""
    job = environ.get("SLURM_JOB_ID", "?")
    line = (
        f"this controller runs inside job {job}, which ends at {utc_iso(end)}: "
        f"{why + '; ' if why else ''}{LIMIT_STOP}; continue with: {shlex.join(command)}"
    )
    try:
        target = Path(workspace) / DIRNAME / HANDOVER
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, {"at": clock(), "job": job, "end": end, "line": line})
    except OSError:
        pass
    return line


def watch_limit(
    workspace: str | Path,
    argv: Sequence[str],
    *,
    stop: Callable[[str], bool],
    leave: Callable[[int], None],
    environ: Mapping[str, str] | None = None,
    say: Callable[[str], None] = print,
    command: Sequence[str] | None = None,
) -> Limit | None:
    """Start the watch on this controller's own end; ``None`` when it runs in no job.

    ``argv`` is what followed the program's name; ``command`` replaces the whole command for
    a caller that is not the flow's own CLI.
    """
    environ = os.environ if environ is None else environ
    end = job_end(environ)
    if end is None:
        return None
    # What continues a run resumes it: the same command with `--clean` would start it over.
    typed = (
        list(command)
        if command is not None
        else [sys.executable, "-m", MODULE, *(word for word in argv if word != "--clean")]
    )

    def over() -> str:
        line = hand_over(environ, typed, end=end, workspace=workspace, why=limit.early)
        say(f"→ {line}")
        return line

    say(
        f"→ this controller runs inside job {environ.get('SLURM_JOB_ID', '?')}, which ends at "
        f"{utc_iso(end)}: it stops starting turns {STOP_BEFORE_S / 60:.0f} minutes before that, "
        f"says the command that continues, and exits {EXIT_CONTINUE}"
    )
    limit = _LIMITS[Path(workspace).resolve()] = Limit(
        end=end, stop=stop, hand_over=over, leave=leave
    )
    return limit.start()


#: The watch on each workspace this process drives, for whoever finds a reason to stop early.
_LIMITS: dict[Path, Limit] = {}


def continue_now(workspace: str | Path, why: str) -> bool:
    """Stop the run of ``workspace`` now and say "to be continued"; False where no job's end is watched.

    For a run that can start nothing more before its job ends (the pool's ``late``): the
    same stop, hand-over line and exit status as at the limit, without waiting for it.
    """
    limit = _LIMITS.get(Path(workspace).resolve())
    if limit is not None:
        limit.now(why)
    return limit is not None


def last_handover(workspace: str | Path) -> dict | None:
    """What the last controller of ``workspace`` said when it stopped for its limit."""
    return read_json(Path(workspace) / DIRNAME / HANDOVER)


__all__ = [
    "EXIT_BEFORE_S",
    "EXIT_CONTINUE",
    "LIMIT_STOP",
    "STOP_BEFORE_S",
    "Limit",
    "continue_now",
    "hand_over",
    "job_end",
    "last_handover",
    "watch_limit",
]

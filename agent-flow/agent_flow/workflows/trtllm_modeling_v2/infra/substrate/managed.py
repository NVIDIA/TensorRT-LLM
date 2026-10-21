# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Managed sessions: the run submits, renews and releases its own compute sessions.

Where this host reaches Slurm (``sbatch`` on ``PATH``) and the run is asked
to (``--manage-sessions``), it does what a driver otherwise does with the
broker by hand, for the sessions ``environment.sessions`` declares. What the
run submitted is kept in ``<workspace>/managed_sessions.json``.
"""

from __future__ import annotations

import getpass
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scheduler import SpoolPaths, write_text_atomic
from .sessions import (
    LIVE_WATCHER_S,
    SessionSpec,
    acknowledge,
    install_session_entry,
    pending_requests,
    plan_successor,
    sessions_from_task,
)

#: What the run submitted, under its workspace.
MANAGED_FILENAME = "managed_sessions.json"
#: Runs one argv and returns its exit code and output.
Runner = Callable[[Sequence[str]], tuple[int, str]]
#: Queue states of a job that still holds or waits for an allocation.
_ALIVE = ("PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED")


def _run(argv: Sequence[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def render_batch(session: SessionSpec, *, after: str | None = None) -> str:
    """The batch script that starts ``session``: its entry once per node, in its container."""
    submit = dict(session.submit)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=mv2flow-{session.name}",
        f"#SBATCH --nodes={session.nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --output={Path(session.spool) / 'session-%j.out'}",
    ]
    if session.gpus_per_node:
        lines.append(f"#SBATCH --gpus-per-node={session.gpus_per_node}")
    if session.time_limit:
        lines.append(f"#SBATCH --time={session.time_limit}")
    for key in ("partition", "account", "qos"):
        if submit.get(key):
            lines.append(f"#SBATCH --{key}={submit[key]}")
    if after:
        lines.append(f"#SBATCH --dependency=afterany:{after}")
    srun = ["srun"]
    if submit.get("container_image"):
        srun.append(f"--container-image={submit['container_image']}")
        if submit.get("container_name"):
            srun.append(f"--container-name={submit['container_name']}")
        if submit.get("mounts"):
            srun.append(f"--container-mounts={','.join(str(m) for m in submit['mounts'])}")
    srun += ["bash", str(session.entry_path)]
    lines.append(" ".join(shlex.quote(part) for part in srun))
    return "\n".join(lines) + "\n"


class ManagedSessions:
    """The run's own sessions on Slurm: what to submit, renew and release, and what it did."""

    def __init__(
        self,
        task_data: Mapping[str, Any],
        workspace: Path,
        *,
        run: Runner = _run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.task_data = task_data
        self.workspace = Path(workspace)
        self.run = run
        self.which = which
        self.record_path = self.workspace / MANAGED_FILENAME

    def unavailable(self) -> str:
        """Why the run cannot manage its sessions here; ``""`` when it can."""
        if not sessions_from_task(self.task_data):
            return "the task declares no session (environment.sessions)"
        missing = [tool for tool in ("sbatch", "squeue", "scancel") if not self.which(tool)]
        if missing:
            return f"this host has no {', '.join(missing)}"
        return ""

    def submitted(self) -> list[dict[str, Any]]:
        """Every job the run submitted, oldest first."""
        try:
            data = json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []

    def alive(self) -> dict[str, str] | None:
        """The run's jobs still in the queue, by id, with their state; ``None`` when Slurm cannot say.

        Asked by job name and user rather than by id: an id Slurm has
        forgotten makes ``squeue -j`` fail for every id it is given.
        """
        ids = {str(job["job_id"]) for job in self.submitted()}
        if not ids:
            return {}
        names = ",".join(f"mv2flow-{s.name}" for s in sessions_from_task(self.task_data))
        code, out = self.run(["squeue", "-h", "-u", getpass.getuser(), "-n", names, "-o", "%i %T"])
        if code != 0:
            return None
        states = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ids and parts[1] in _ALIVE:
                states[parts[0]] = parts[1]
        return states

    def _submit(self, session: SessionSpec, after: str | None) -> str | None:
        install_session_entry(self.task_data, session.name)
        script = Path(session.spool) / f"managed-{session.name}.sbatch"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(render_batch(session, after=after), encoding="utf-8")
        code, out = self.run(["sbatch", "--parsable", str(script)])
        job_id = out.strip().split(";")[0].strip() if code == 0 else ""
        if not job_id.isdigit():
            return None
        jobs = self.submitted()
        jobs.append({"session": session.name, "job_id": job_id, "after": after, "at": time.time()})
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.record_path, json.dumps(jobs, indent=1))
        return job_id

    def tick(self) -> list[str]:
        """Submit what the sessions lack now; one line per thing done, or failed."""
        if self.unavailable():
            return []
        said = []
        queued = self.alive()
        if queued is None:
            # Not knowing what is queued, submitting could start a session twice.
            return []
        mine = {str(job["job_id"]): job["session"] for job in self.submitted()}
        waiting = {mine[j] for j in queued if j in mine}
        for session in sessions_from_task(self.task_data):
            live = SpoolPaths(session.spool).live_watchers(LIVE_WATCHER_S)
            if live or session.name in waiting:
                continue
            job = self._submit(session, None)
            said.append(
                f"session {session.name}: no watcher and no job in the queue: submitted job {job}"
                if job
                else f"session {session.name}: sbatch refused the submission"
            )
            if job:
                waiting.add(session.name)
        for request in pending_requests(self.task_data):
            name = str(request.get("session"))
            plan = plan_successor(self.task_data, name)
            session = next(s for s in sessions_from_task(self.task_data) if s.name == name)
            job = self._submit(session, plan.after)
            if job is None:
                said.append(f"session {name}: sbatch refused the successor")
                continue
            acknowledge(
                session.spool,
                job_id=job,
                by="managed",
                request_id=request.get("request_id"),
                session=name,
                after=plan.after,
            )
            said.append(f"session {name}: successor job {job} submitted ({plan.reason})")
        return said

    def release(self) -> list[str]:
        """Cancel every job the run submitted that is still in the queue; one line each."""
        said = []
        for job_id, state in sorted((self.alive() or {}).items()):
            code, out = self.run(["scancel", job_id])
            said.append(
                f"job {job_id} ({state}) cancelled"
                if code == 0
                else f"job {job_id} ({state}) not cancelled: {out.strip()[:200]}"
            )
        return said


__all__ = ["MANAGED_FILENAME", "ManagedSessions", "render_batch"]

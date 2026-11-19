"""Bash-invokable wrapper that launches a long command DETACHED and records it.

The agent runs long work via ``python -m agent_flow.jobs.run_detached … -- <cmd>``
instead of a raw blocking ``srun``, so the job survives the orchestrator being
paused or stopped. It writes a name-first EXPECTED registry row before launching
(so a crash between submit and id-record still leaves the job recoverable by its
deterministic name), launches detached, then records the handle id.

Re-attach on resume is the AGENT's job, not this wrapper's: on a resumed turn the
agent reads its own ``jobs.json``, polls the recorded job with ``squeue``/``sacct``
itself, and resubmits (re-runs this wrapper) only if the job is gone. The wrapper
just launches + records; it does not decide attach-vs-resubmit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from agent_flow.jobs.registry import JobEntry, JobRegistry


def _launch(kind: str, name: str, command: list[str], sentinel_dir: Path) -> dict:
    """Launch ``command`` detached; return the kind-specific handle dict."""
    if kind == "slurm":
        p = subprocess.run(
            ["sbatch", f"--job-name={name}", "--parsable", *command],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"job_id": p.stdout.strip().split(";")[0], "name": name}
    if kind == "local":
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        sentinel = sentinel_dir / f"{name}.sentinel.json"
        out = sentinel_dir / f"{name}.out"
        # start_new_session → new session, survives orchestrator death; sentinel captures exit code.
        script = (
            f'{" ".join(shlex.quote(a) for a in command)} > "{out}" 2>&1; '
            f'printf \'{{"exit_code": %d}}\' "$?" > "{sentinel}"'
        )
        proc = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"pid": proc.pid, "sentinel": str(sentinel), "name": name}
    raise ValueError(f"unknown kind {kind!r}")


def main(argv: list[str] | None = None) -> int:
    """Parse args, record name-first, launch detached, then record the handle id."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--handle", required=True)
    ap.add_argument("--kind", required=True, choices=["slurm", "local"])
    ap.add_argument("--name", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    command = ns.command[1:] if ns.command and ns.command[0] == "--" else ns.command

    reg = JobRegistry(Path(ns.registry))
    # name-first: EXPECTED row on disk BEFORE we launch (crash-recoverable by name).
    reg.upsert(
        JobEntry(handle_key=ns.handle, kind=ns.kind, handle={"name": ns.name}, state="EXPECTED")
    )
    handle = _launch(ns.kind, ns.name, command, Path(ns.registry).parent / "_detached")
    reg.upsert(JobEntry(handle_key=ns.handle, kind=ns.kind, handle=handle, state="SUBMITTED"))
    print(f"[run_detached] {ns.kind} {ns.handle} -> {handle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

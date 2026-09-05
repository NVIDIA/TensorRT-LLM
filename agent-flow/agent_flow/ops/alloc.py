"""Resolve a held allocation by NAME, not by a hard-coded job id.

A scheduler job id changes whenever the allocation is preempted, fails or is
resubmitted. Hard-coding it meant every change needed a code edit plus a
restart of everything already running, and a stale dashboard reported a
healthy allocation as gone.

Resolution order: ``$JOBID`` (an explicit override always wins), then the
newest job matching the allocation's ``job_name`` — preferring a RUNNING one —
then the ``job_id`` recorded in the config.
"""

from __future__ import annotations

import os
import subprocess

from agent_flow.ops.config import OpsConfig


def resolve(cfg: OpsConfig, key: str = "") -> str:
    """Current job id of allocation ``key`` (default: the config's default)."""
    if os.environ.get("JOBID"):
        return os.environ["JOBID"]
    key = key or cfg.default_allocation
    alloc = cfg.allocations.get(key)
    if alloc is None:
        return ""
    if alloc.job_name:
        found = _by_name(alloc.job_name)
        if found:
            return found
    return alloc.job_id


def _by_name(job_name: str) -> str:
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", job_name, "-o", "%i %T"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    rows = [line.split() for line in out.splitlines() if line.strip()]
    if not rows:
        return ""
    for jid, state in rows:
        if state == "RUNNING":
            return jid
    return rows[-1][0]

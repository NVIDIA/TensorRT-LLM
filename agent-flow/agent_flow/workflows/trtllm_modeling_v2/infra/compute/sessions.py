# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entry scripts of the pool's watchers.

A watcher is a process on a GPU node whose main process is the spool scheduler; a lead is the
same program with no devices, where steps across nodes can be started. :func:`render_entry`
renders either entry script, for the holders the flow starts itself (:mod:`.pool`).
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Mapping

from . import SCHEDULER_PATH
from .scheduler import utc_iso

#: The oldest interpreter the scheduler runs on: a lead runs on its host's own.
LEAD_MIN_PYTHON = (3, 8)


# Variables whose declared value EXTENDS the inherited one. A plain `PATH: <venv>/bin:/usr/bin`
# drops `/usr/local/mpi/bin`, and the next launcher line dies `mpirun: command not found` (rc
# 127). `PYTHONPATH` already holds the gates' rank hook (`checks/check_runtime.RANK_HOOK_DIR`), the
# scheduler's resolve hook and the job's tree; unset, it must not leave a trailing `:`, which
# Python reads as the working directory.
MERGED_ENV_VARS: tuple[str, ...] = ("PATH", "PYTHONPATH")
#: Merged variables whose inherited value is appended only when it is set.
_SEPARATOR_WHEN_SET: tuple[str, ...] = ("PYTHONPATH",)


def prepend_to_inherited(name: str, value: str) -> str:
    """``/a:/b`` -> ``/a:/b:"$PATH"`` (the value shell-quoted), idempotent against a ``:$PATH``.

    The expansion stays outside the quoting so that the shell expands it. ``PYTHONPATH`` becomes
    ``/a:/b"${PYTHONPATH:+:$PYTHONPATH}"``: nothing is appended when it is unset or empty.
    """
    for suffix in (f":${name}", f":${{{name}}}"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value or value in (f"${name}", f"${{{name}}}"):
        return f'"${name}"'
    if name in _SEPARATOR_WHEN_SET:
        return f'{shlex.quote(value)}"${{{name}:+:${name}}}"'
    return f'{shlex.quote(value)}:"${name}"'


def env_assignments(env: Mapping[str, str]) -> list[str]:
    """Stable shell assignments; path variables prepend rather than erase inherited values."""
    return [
        f"{key}={prepend_to_inherited(key, str(value)) if key in MERGED_ENV_VARS else shlex.quote(str(value))}"
        for key, value in sorted(env.items())
    ]


def render_entry(
    task_data: Mapping[str, Any],
    *,
    spool: str,
    session: str,
    gpus_per_node: int | None = None,
    lead: bool = False,
    host_python: str = "python3",
) -> str:
    """The entry script of a watcher on ``spool``: set the run's environment, become the scheduler.

    This is where the environment of every job a watcher starts is made: the spec's
    ``environment.env``, and the directory of ``environment.python`` first on ``PATH`` unless
    the spec states a ``PATH`` of its own.

    With ``lead`` it is the lead's: run where steps across the holder's nodes can be started
    (on Slurm, the batch host, outside the container), with ``host_python``, no devices and
    nothing of the container's environment. The holder's name reaches both as ``MV2FLOW_HOLDER``.
    """
    block = task_data.get("environment") or {}
    declared = {} if lead else {str(k): str(v) for k, v in (block.get("env") or {}).items()}
    if not lead:
        declared.setdefault("PYTHONNOUSERSITE", "1")
    declared["MV2FLOW_SESSION"] = session
    if task_data.get("trtllm_repo_path"):
        declared["MV2FLOW_REPO"] = str(task_data["trtllm_repo_path"])
    python = str(block.get("python") or "python3")
    if not lead and os.path.isabs(python):
        # As activating the engine's install would: every job of this watcher finds what is
        # beside the interpreter (`pip`, the launcher) before the image's own. The engine
        # looks its install up that way when it compiles a kernel at load.
        declared.setdefault("PATH", os.path.dirname(python))
    watch = ["--session", session]
    if lead:
        watch += ["--gpus", "0", "--lead"]
    elif gpus_per_node is not None:
        watch += ["--gpus", str(gpus_per_node)]
    what = "lead" if lead else "watcher"
    lines = [
        "#!/bin/bash",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        f"# Entry script of the {what} of {session!r}, rendered at {utc_iso()} from the run's spec.",
        "# Do not edit it: the pool renders it again at every start.",
        "#",
        *(
            [
                "# Run it once per holder, where steps across its nodes can be started. It holds",
                "# no device: it forms the holder's multi-node jobs and runs their scripts.",
            ]
            if lead
            else [
                "# Run it once on each node of a holder. The scheduler it execs is that node's",
                "# watcher; work arrives as files in the spool.",
            ]
        ),
        "set -u",
        f"SPOOL={shlex.quote(spool)}",
        *(f"export {assignment}" for assignment in env_assignments(declared)),
        'export MV2FLOW_SPOOL="$SPOOL"',
        "ulimit -c 0",
        'mkdir -p "$SPOOL"',
    ]
    if lead:
        py, least = shlex.quote(host_python), ".".join(map(str, LEAD_MIN_PYTHON))
        lines += [
            "",
            "# The lead runs on this host's own interpreter: one that is too old is said here,",
            "# not as a syntax error of the scheduler.",
            f"if ! {py} -c 'import sys; sys.exit(sys.version_info < {LEAD_MIN_PYTHON})' 2>/dev/null; then",
            f"  echo \"error: the holder's lead needs Python {least} or newer on $(hostname), and "
            f"{host_python} is $({py} -V 2>&1): name a newer one "
            f'(environment.slurm.host_python)" >&2',
            "  exit 127",
            "fi",
            f'exec {shlex.quote(host_python)} {shlex.quote(str(SCHEDULER_PATH))} watch "$SPOOL" '
            + " ".join(shlex.quote(a) for a in watch),
        ]
        return "\n".join(lines) + "\n"
    lines += [
        "{",
        '  echo "=== watcher entry $(date -Is) on $(hostname) ==="',
        '  echo "session=$MV2FLOW_SESSION holder=${MV2FLOW_HOLDER:--} spool=$SPOOL '
        'job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"',
        f"  {shlex.quote(python)} -c \"import sys; print('python', sys.version.split()[0], sys.executable)\" || true",
        f"  PYTHONNOUSERSITE=1 {shlex.quote(python)} -c "
        "\"import tensorrt_llm; print('trtllm', tensorrt_llm.__version__, tensorrt_llm.__file__)\" 2>&1 | tail -3",
        "  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>&1",
        "  free -g 2>&1 | head -2; df -h /dev/shm 2>&1 | tail -1",
        "  cat /sys/fs/cgroup/memory.max 2>/dev/null",
        '} > "$SPOOL/watcher_entry.$(hostname).log" 2>&1',
        "",
        f'exec "${{MV2FLOW_SCHEDULER_PYTHON:-python3}}" {shlex.quote(str(SCHEDULER_PATH))} watch "$SPOOL" '
        + " ".join(shlex.quote(a) for a in watch),
    ]
    return "\n".join(lines) + "\n"


def write_entry(target: Path, text: str) -> Path:
    """Write an entry script (executable) at ``target``, renamed into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_text(text, encoding="utf-8")
    staging.chmod(0o755)
    os.replace(staging, target)
    return target


__all__ = ["LEAD_MIN_PYTHON", "MERGED_ENV_VARS", "env_assignments", "render_entry", "write_entry"]

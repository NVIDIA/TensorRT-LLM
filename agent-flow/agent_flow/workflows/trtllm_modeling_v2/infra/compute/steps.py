# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A world across the nodes of a session (a step), and the script of servers a job runs.

A job that spans nodes is run once by the session's lead, each world a step of the allocation
(README, "anything wider than one node"). A step's shell is a fresh one: it reads
:data:`SESSION_ENV` and is handed :data:`HANDED_ON` and the check's variables by value.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

#: What a job's shell holds that every rank of its steps needs.
HANDED_ON: tuple[str, ...] = (
    "PYTHONPATH",
    "MV2FLOW_TREE",
    "MV2FLOW_READS_DIR",
    "MV2FLOW_RESOLVE_LOG",
    "MV2FLOW_RESOLVE_MODULES",
    "MV2FLOW_SPOOL",
    "MV2FLOW_SPOOL_JOB",
)
#: Import records one file a node (``<log>.<node>``): ranks of several nodes appending to one
#: file on a network file system lose lines.
RESOLVE_PER_HOST = "MV2FLOW_RESOLVE_PER_HOST"
#: The session entry's exports and setup, which a step's shell reads (``sessions.py``).
SESSION_ENV = "session_env.sh"
ALL_HOSTS = '"$MV2FLOW_JOB_HOSTS"'
FIRST_HOST = '"${MV2FLOW_JOB_HOSTS%%,*}"'


def step_line(
    step_args: Sequence[str],
    argv: str,
    *,
    ranks: int,
    nodes: int = 1,
    hosts: str = FIRST_HOST,
    assignments: Sequence[str] = (),
    devices: Sequence[int | str] = (),
    path: str = "",
) -> str:
    """``srun`` of ``argv`` as ``ranks`` tasks on ``nodes`` of ``hosts``, each seeing ``devices``.

    ``hosts``, ``argv`` and ``assignments`` are shell words; ``path`` goes before the
    container's own ``PATH``.
    """
    flags = [
        "--overlap",
        "--mpi=none" if ranks == 1 else "--mpi=pmix",
        "--kill-on-bad-exit=1",
        f"--nodes={nodes}",
        f"--ntasks={ranks}",
        f"--ntasks-per-node={ranks // nodes}",
    ]
    inner = (
        f'[ -r "$1/{SESSION_ENV}" ] && . "$1/{SESSION_ENV}"; cd "$2" && shift 2 && '
        + (f"export PATH={path} && " if path else "")
        + 'exec env "$@"'
    )
    words = [
        "srun",
        *flags,
        f"--nodelist={hosts}",
        *(shlex.quote(str(word)) for word in step_args),
        "bash",
        "-c",
        shlex.quote(inner),
        "step",
        '"${MV2FLOW_SPOOL:-}"',
        '"$PWD"',
        *(f'{name}="${{{name}:-}}"' for name in HANDED_ON),
        f"{RESOLVE_PER_HOST}=1",
        *assignments,
        "CUDA_VISIBLE_DEVICES=" + ",".join(str(d) for d in devices),
        argv,
    ]
    return " ".join(words)


@dataclass(frozen=True)
class World:
    """One server of a job: its ranks, its first node and node count, its devices on each."""

    name: str
    ranks: int
    node: int
    nodes: int
    devices: tuple[int, ...]
    argv: str
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def hosts(self) -> str:
        return '"' + ",".join(f"${{HOSTS[{self.node + n}]}}" for n in range(self.nodes)) + '"'


#: :meth:`RunEnvironment.step`, or :meth:`RunEnvironment.world` on one node.
Step = Callable[..., str]


def servers_script(
    step: Step, worlds: Sequence[World], *, directory: Path, nodes: int, client: str
) -> Path:
    """Write the script that starts ``worlds``, then runs ``client``; its path.

    A world's output is in ``<world>.log``, its end in ``exit.<world>``; the client's exit code
    is the script's, which then ends its own launchers (never a rank).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    held = ",".join(str(d) for d in range(1 + max(d for w in worlds for d in w.devices)))
    lines = [
        "#!/bin/bash",
        f"# {len(worlds)} server(s) on "
        + (f"{nodes} nodes, run once by the session's lead: " if nodes > 1 else "one node: ")
        + "; ".join(
            f"{w.name} {w.ranks} rank(s) on node(s) {w.node}"
            + (f"-{w.node + w.nodes - 1}" if w.nodes > 1 else "")
            + f" device(s) {','.join(map(str, w.devices))}"
            for w in worlds
        ),
        "set -u",
        # Each launcher in a process group of its own, so that it can be ended as one.
        "set -m",
        f"D={shlex.quote(str(directory))}",
        "launchers=()",
        'IFS=, read -r -a HOSTS <<< "${MV2FLOW_JOB_HOSTS:-}"'
        if nodes > 1
        else f'IFS=, read -r -a DEVICES <<< "${{CUDA_VISIBLE_DEVICES:-{held}}}"',
        'rm -f "$D"/exit.* "$D"/address.*',
        *(
            "{ "
            + step(
                w.argv, ranks=w.ranks, nodes=w.nodes, hosts=w.hosts, devices=w.devices, env=w.env
            )
            + f' > "$D/{w.name}.log" 2>&1; echo $? > "$D/exit.{w.name}"; }} &\nlaunchers+=($!)'
            for w in worlds
        ),
        step(client, ranks=1, hosts='"${HOSTS[0]}"'),
        "rc=$?",
        'for launcher in "${launchers[@]}"; do kill -TERM -- "-$launcher" 2>/dev/null; done',
        "wait",
        'exit "$rc"',
    ]
    script = directory / "servers.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script

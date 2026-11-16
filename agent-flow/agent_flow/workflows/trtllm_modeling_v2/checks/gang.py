# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session wider than one node: where each role instance runs, and the script that runs it.

A deployment that holds more devices than one node has is one job of whole nodes
(``nodes=M``). The pool's lead runs the job's script once, where it runs itself, under
``MV2FLOW_JOB_HOSTS``; the script starts everything else as steps of the allocation, each
built by the pool's step line (``RunEnvironment.step``):

1. **Place.** For each role instance, the step its world will run as, with a command
   that only writes where each rank is and which devices it was given. Then one task in
   the container checks those files (``session_driver.py --placed``): every rank
   answered, and no two instances were given the same device of one host. A clash ends
   the script here, before any load.
2. **Load.** One step per role instance: one world of that instance's ranks, on its
   slice of the reserved hosts, each rank shown its instance's devices of its own host.
   Every world fills each of its nodes with the same number of ranks (:func:`plan_layout`).
3. **Proxy and clients.** One task runs the session's driver attached to those servers
   (``session_driver.py --attach``): it starts the proxy over them, runs every check as
   a client and prints the same records as a session on one node.
4. **Stop.** The script ends its own launchers; a rank is never signalled from here.

Nothing here mounts anything: a step enters the container the holder's watcher step
made, which carries every mount.

A job runs a pin, never a moving tree (``infra/compute/pintree.py``). The lead's job is
told its pin (``MV2FLOW_PIN``); a step on another node has no tree of it yet. So every
step runs under the pin's own prelude, as each node's step of any job of
several nodes does: it makes (or finds) the pin's tree on its node, checks its libraries
against the pin's build record, writes that node's tree record for the origin judge, and
only then becomes the step's command inside that tree. Without a pin (a directory
nothing can pin) a step enters the tree named when the job was written.

A step's shell is a fresh one: it has nothing of what the pool's watcher gives a job it
starts itself. So the script writes what its own job was given for the origin judge
(:data:`ORIGIN_VARIABLES`: where each process reports what it imported, and from which
tree) into ``origin.env``, every launched shell reads it, and the pool's origin hook is
put on ``PYTHONPATH`` beside the tree: every rank a step starts reports its origin to
the same record the job's stated ranks are counted from.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..infra.compute import pintree
from ..infra.compute import slurm as slurm_mod
from ..infra.compute.environment import RunEnvironment
from ..spec.deployments import Deployment, Instance
from .harness.session_driver import ENGINE_ROLE_ENV
from .harness.workload_io import RANK_RECORDS_ENV

#: The job's script, and what it says when it cannot go on.
SCRIPT = "gang.sh"
GANG_MARKER = "MV2FLOW_GANG"
#: The role names ``trtllm-serve --server_role`` takes.
SERVER_ROLES = {"ctx": "context", "gen": "generation"}
#: What the pool's watcher exports to a job it starts, that a rank's origin record needs
#: (``infra/compute/scheduler.py:_job_environment``, ``resolvehook/sitecustomize.py``).
ORIGIN_VARIABLES = (
    "MV2FLOW_RESOLVE_LOG",
    "MV2FLOW_RESOLVE_MODULES",
    "MV2FLOW_TREE",
    "MV2FLOW_SPOOL",
    "MV2FLOW_SPOOL_JOB",
)
ORIGIN_FILE = "origin.env"
#: What a step's command starts with: the pin's prelude when the job runs a pin,
#: nothing when it runs a directory as it is.
PINNED = '${PINNED[@]+"${PINNED[@]}"}'


@dataclass(frozen=True)
class Slot:
    """The devices one role instance holds on one node of the job."""

    #: Index into the job's reserved hosts, in the order the lead names them.
    node: int
    devices: tuple[int, ...]


Layout = Mapping[str, tuple[Slot, ...]]


def plan_layout(deployment: Deployment, per_node: int) -> tuple[int, dict[str, tuple[Slot, ...]]]:
    """``(nodes, layout)``: each role instance's devices, on nodes of ``per_node``.

    Instances take devices in their order, one device per rank, and no device is given
    twice. One that fits in what is left of a node goes there, so small servers share a
    node; one that does not starts on a fresh node; one wider than a node takes whole
    nodes. So every world fills each of its nodes with the same number of ranks, which is
    what a step lays out. ``ValueError`` for a world wider than a node whose ranks are no
    whole number of nodes.
    """
    if per_node < 1:
        raise ValueError("a node has at least one device")
    node, used = 0, 0
    layout: dict[str, tuple[Slot, ...]] = {}
    for instance in deployment.instances():
        ranks = instance.server.ranks
        if used and ranks > per_node - used:
            node, used = node + 1, 0
        if ranks <= per_node:
            layout[instance.name] = (Slot(node, tuple(range(used, used + ranks))),)
            used += ranks
        elif ranks % per_node:
            raise ValueError(
                f"`{instance.name}` of `{deployment.name}` is one world of {ranks} ranks, and "
                f"a node has {per_node} devices: a world wider than a node takes whole nodes "
                f"(a multiple of {per_node} ranks)"
            )
        else:
            whole = ranks // per_node
            layout[instance.name] = tuple(
                Slot(node + n, tuple(range(per_node))) for n in range(whole)
            )
            node, used = node + whole - 1, per_node
        if used == per_node:
            node, used = node + 1, 0
    return node + (1 if used else 0), layout


def describe(layout: Layout) -> str:
    """The layout in one line: ``ctx0: node 0 devices 0-3; gen0: node 1 devices 0,1``."""
    parts = []
    for name, slots in layout.items():
        where = ", ".join(
            f"node {slot.node} device(s) {','.join(map(str, slot.devices))}" for slot in slots
        )
        parts.append(f"{name}: {where}")
    return "; ".join(parts)


def serve_command(model: str, instance: Instance, options: str | Path, address: Path) -> str:
    """What each rank of ``instance``'s world runs: the launcher over ``trtllm-serve`` past one rank."""
    words = [
        *(["trtllm-llmapi-launch"] if instance.server.ranks > 1 else []),
        "trtllm-serve",
        str(model),
        "--config",
        str(options),
        # Bound to every interface: the clients' task may be on another node.
        "--host",
        "0.0.0.0",  # nosec B104
        "--port",
        "0",
        "--report_addr",
        str(address),
        "--num_serve_frontends=1",
    ]
    if instance.role in SERVER_ROLES:
        words += ["--server_role", SERVER_ROLES[instance.role]]
    return shlex.join(words)


def _prelude(python: str) -> str:
    """The pin's prelude for one step, as the lead's shell writes it into its array.

    ``pintree.step_argv`` with the pin, the job and the job's tree record as the lead's own
    job has them; the spool is handed on too, for the sweep of unused trees.
    """
    words = pintree.step_argv("@PIN@", "@JOB@", "@RECORD@", python)
    said = " ".join(shlex.quote(str(word)) for word in words)
    said = said.replace("@PIN@", '"$MV2FLOW_PIN"').replace("@RECORD@", '"$record"')
    said = said.replace("@JOB@", '"${MV2FLOW_SPOOL_JOB:-session}"')
    return f'env "MV2FLOW_SPOOL=${{MV2FLOW_SPOOL:-}}" {said}'


def _rank_script(directory: Path, instance: Instance, serve: str) -> str:
    """What one rank of ``instance`` does: take its devices of its own host, then place or load."""
    name, table = instance.name, shlex.quote(str(directory / f"devices.{instance.name}"))
    single = "export TLLM_WORKER_USE_SINGLE_PROCESS=1\n" if instance.server.ranks <= 1 else ""
    # The engine's own role variable, in every rank (`session_driver.ENGINE_ROLE_ENV`): the
    # serve command sets it in rank 0 alone. One engine has no role, and borrows none.
    role = SERVER_ROLES.get(instance.role)
    engine = f"export {ENGINE_ROLE_ENV}={role}" if role else f"unset {ENGINE_ROLE_ENV}"
    return f"""#!/bin/bash
# One rank of {name}: shown the devices its instance holds on this rank's own host.
set -u
# Where this rank reports the origin of what it imports, as its job was told.
[ -f {shlex.quote(str(directory / ORIGIN_FILE))} ] && . {shlex.quote(str(directory / ORIGIN_FILE))}
host=${{SLURMD_NODENAME:-$(hostname -s)}}
rank=${{SLURM_PROCID:-${{PMIX_RANK:-0}}}}
devices=$(awk -v h="$host" '$1 == h {{print $2}}' {table})
if [ -z "$devices" ]; then
  echo "{GANG_MARKER} {name} rank $rank: $host is no host of its layout ({table})" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=$devices
export MV2FLOW_SERVER_ROLE={shlex.quote(instance.role)} MV2FLOW_INSTANCE={instance.index}
{engine}
# Each rank keeps its engine's record in a file of its own: the world shares one stream.
export {RANK_RECORDS_ENV}={shlex.quote(str(directory))}
{single}if [ "${{1:-}}" = place ]; then
  held=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$devices" 2>/dev/null | tr '\\n' ' ')
  printf '%s %s %s %s\\n' "$host" "$rank" "$devices" "$held" > {shlex.quote(str(directory))}/placed.{name}.$rank
  exit 0
fi
: > {shlex.quote(str(directory))}/load.{name}.$rank
exec {serve}
"""


def render(
    environment: RunEnvironment,
    deployment: Deployment,
    layout: Layout,
    nodes: int,
    *,
    directory: Path,
    model: str,
    options: Mapping[str, str | Path],
    tree: str | None,
    driver: Sequence[str],
    env: Mapping[str, str],
    pythonpath: Sequence[str] = (),
) -> dict[str, str]:
    """Every file of the job, by name; :data:`SCRIPT` is what the lead runs.

    ``driver`` is the session driver's command over its spec, without a mode; ``env``
    is exported inside every launched shell, as is ``pythonpath`` after the tree.
    Raises the environment's ``EnvironmentError`` when it has no way to run a step.
    """
    directory = Path(directory)
    quoted = shlex.quote(str(directory))
    files: dict[str, str] = {}

    def inner(command: str, role: str | None = None) -> str:
        return slurm_mod.inner_script(
            command,
            python=environment.python,
            tree=tree,
            pythonpath=pythonpath,
            env=env,
            role=role,
        )

    def hosts_of(slots: Sequence[Slot]) -> str:
        return '"' + ",".join(f"${{HOSTS[{slot.node}]}}" for slot in slots) + '"'

    tables: list[str] = []
    place: list[str] = []
    load: list[str] = []
    for instance in deployment.instances():
        name, slots = instance.name, layout[instance.name]
        address = directory / f"address.{name}"
        serve = serve_command(model, instance, options[instance.role], address)
        files[f"serve.{name}.sh"] = _rank_script(directory, instance, serve)
        files[f"rank.{name}.sh"] = inner(f'bash {quoted}/serve.{name}.sh "$@"', role=name)
        tables.append(f"# {name}: {describe({name: slots})}")
        rows = " ".join(f'"${{HOSTS[{s.node}]}} {",".join(map(str, s.devices))}"' for s in slots)
        tables.append(f"printf '%s\\n' {rows} > \"$D/devices.{name}\"")

        def step(command: str, slots: Sequence[Slot] = slots) -> str:
            return environment.step(
                ranks=sum(len(s.devices) for s in slots),
                nodes=len(slots),
                hosts=hosts_of(slots),
                command=command,
            )

        place.append(
            f"{step(f'{PINNED} bash {quoted}/rank.{name}.sh place')} || gave_up 3 'place {name}'"
        )
        load.append(
            f'{{ {step(f"{PINNED} bash {quoted}/rank.{name}.sh")} > "$D/{name}.log" 2>&1; '
            f'echo $? > "$D/exit.{name}"; }} &\nlaunchers+=($!)'
        )
    command = shlex.join([str(word) for word in driver])
    origin = shlex.quote(str(directory / ORIGIN_FILE))
    files["driver.sh"] = (
        "#!/bin/bash\n# The session's driver, told what its job was told of origins.\n"
        f'[ -f {origin} ] && . {origin}\nexec {command} "$@"\n'
    )
    files["check.sh"] = inner(f"bash {quoted}/driver.sh --placed")
    files["clients.sh"] = inner(f"bash {quoted}/driver.sh --attach")

    def one_task(script: str) -> str:
        return environment.step(
            ranks=1, nodes=1, hosts='"${HOSTS[0]}"', command=f"{PINNED} bash {quoted}/{script}"
        )

    lines = [
        "#!/bin/bash",
        f"# The session of `{deployment.name}` on {nodes} nodes, run once by the holder's lead.",
        f"# {describe(layout)}",
        "set -u",
        # Each launcher in a process group of its own, so that it can be ended as one.
        "set -m",
        f"D={quoted}",
        "launchers=()",
        "gave_up() {",
        f'  echo "{GANG_MARKER} nothing was loaded: $2 failed (exit $1)"',
        '  exit "$1"',
        "}",
        'IFS=, read -r -a HOSTS <<< "${MV2FLOW_JOB_HOSTS:-}"',
        f'if [ "${{#HOSTS[@]}}" -lt {nodes} ]; then',
        f'  echo "{GANG_MARKER} nothing was loaded: this session needs {nodes} nodes and the job '
        f"has ${{#HOSTS[@]}} (${{MV2FLOW_JOB_HOSTS:-none}}): it is run by a holder's lead as a "
        f'job of nodes={nodes}"',
        "  exit 2",
        "fi",
        'rm -f "$D"/placed.* "$D"/exit.* "$D"/address.* "$D"/load.* "$D"/devices.*',
        'rm -f "$D"/rank-engine.*',
        "# The pin this job runs, when it runs one: every step makes (or finds) its tree on",
        "# its own node and checks its libraries there, before it becomes its command.",
        "PINNED=()",
        'if [ -n "${MV2FLOW_PIN:-}" ]; then',
        '  record="${MV2FLOW_SPOOL:-$D}/logs/${MV2FLOW_SPOOL_JOB:-session}.tree.jsonl"',
        f"  PINNED=({_prelude(environment.python)})",
        "fi",
        "# What this job was given for the origin judge, for every shell a step launches.",
        f"for name in {' '.join(ORIGIN_VARIABLES)}; do",
        '  [ -n "${!name:-}" ] && printf \'export %s=%q\\n\' "$name" "${!name}"',
        f'done > "$D/{ORIGIN_FILE}"',
        "# Where each role instance runs: its devices on each of its hosts.",
        *tables,
        "# Before any load: every rank says where it is and which devices it was given.",
        *place,
        f"{one_task('check.sh')} || gave_up 3 'the device check'",
        "# One step per role instance: one world of its ranks, on its slice of the hosts.",
        *load,
        "# The proxy and every check, as clients of those servers.",
        one_task("clients.sh"),
        "rc=$?",
        "# This script's own launchers; a rank is never signalled from here.",
        'for launcher in "${launchers[@]}"; do kill -TERM -- "-$launcher" 2>/dev/null; done',
        "wait",
        'exit "$rc"',
    ]
    files[SCRIPT] = "\n".join(lines) + "\n"
    return files


def write(directory: Path, files: Mapping[str, str]) -> Path:
    """Write the job's files into ``directory``; the path of the script the lead runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory / SCRIPT


__all__ = [
    "GANG_MARKER",
    "ORIGIN_FILE",
    "ORIGIN_VARIABLES",
    "SCRIPT",
    "Layout",
    "Slot",
    "describe",
    "plan_layout",
    "render",
    "serve_command",
    "write",
]

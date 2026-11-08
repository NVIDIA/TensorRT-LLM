#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One engine across the nodes of a multi-node spool job.

A member wider than one node (``tep8`` where a node holds four devices) runs as **one** MPI world
whose ranks span a multi-node job's nodes (``scheduler.py``, MULTI-NODE JOBS). The job's command
runs on every node; this program, at the place in it where a single node would start its ranks,
decides what each node does:

- **node 0 leads.** It runs the payload with the gang's MPI environment: ``MV2FLOW_GANG_MPI_HOSTS``
  (every host and its slots, for ``mpirun --host``) and Open MPI's remote launch agent set to this
  program's ``agent`` subcommand. With ``--ranks`` the payload is the ranks' launcher, started by
  one ``mpirun`` over every host; without it the payload is one process that starts them itself
  (``server_runtime.rank_launch`` reads the hosts).
- **every other node serves.** It runs the daemon command ``mpirun`` asks the agent to start on
  its host, in the environment the job's command line gave it on that node, and returns its exit
  code, until node 0's part ends.

The agent opens no connection and needs no ``ssh`` or ``srun`` inside the container: it hands the
command to the node through the job's directory on the shared spool and waits for it to end. The
daemons it starts connect back to ``mpirun`` over TCP, exactly as under ``ssh``::

    <spool>/gangs/<job>/relay/launch-agent   the agent mpirun runs (written by node 0)
    <spool>/gangs/<job>/relay/<rank>/<id>.json   a daemon command for node <rank>
    <spool>/gangs/<job>/relay/<rank>/<id>.rc     its exit code, written when it ends
    <spool>/gangs/<job>/relay/<rank>/<id>.cancel the agent was stopped: stop the command

Outside a multi-node job ``lead`` runs the payload as one node would. This module is standard
library only and runs by path (``python3 gang.py lead ...``) inside the container.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

try:
    from ..files import read_json, write_json_atomic, write_text_atomic
    from .server_runtime import GANG_MPI_HOSTS_ENV, rank_launch
except ImportError:  # run by path: the sibling modules are on sys.path[0]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from files import read_json, write_json_atomic, write_text_atomic  # type: ignore[no-redef]
    from server_runtime import GANG_MPI_HOSTS_ENV, rank_launch  # type: ignore[no-redef]

#: The scheduler exports the multi-node job's directory under the spool to each node's part.
GANG_DIR_ENV = "MV2FLOW_JOB_GANG_DIR"
NODES_ENV = "MV2FLOW_JOB_NODES"
NODE_RANK_ENV = "MV2FLOW_JOB_NODE_RANK"
HOSTS_ENV = "MV2FLOW_JOB_HOSTS"

RELAY_DIRNAME = "relay"
AGENT_NAME = "launch-agent"
#: The part whose end ends the job: node 0 runs the payload.
LEADER_PART = Path("parts") / "0.json"

#: Open MPI's remote launch agent, as PRRTE (Open MPI 5) and ORTE (Open MPI 4) name it. A tree
#: spawn would have each daemon start the next through the agent as well; one level is enough.
AGENT_PARAMS: tuple[str, ...] = ("PRTE_MCA_plm_ssh_agent", "OMPI_MCA_plm_rsh_agent")
NO_TREE_SPAWN: dict[str, str] = {
    "PRTE_MCA_plm_ssh_no_tree_spawn": "1",
    "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
}

POLL_S = 0.2
#: SIGTERM to SIGKILL for a relayed command still running when node 0's part ends.
STOP_GRACE_S = 10.0
#: How long a relayed command's failure waits for node 0 to end before it fails this part.
FAILURE_GRACE_S = 60.0
#: The exit code ``lead`` and ``agent`` use for a gang they cannot run in.
RC_USAGE = 2
#: The exit code ``agent`` reports when the node it relays to ended before the command did.
RC_NODE_GONE = 255
#: The exit code a request without a command gets, as a shell's for a command it cannot run.
RC_NO_COMMAND = 127


class GangError(RuntimeError):
    """The job is not the multi-node job this command needs."""


def _say(message: str) -> None:
    print(f"[gang] {message}", file=sys.stderr, flush=True)


def gang_dir(environ: Mapping[str, str]) -> Path:
    value = environ.get(GANG_DIR_ENV, "")
    if not value:
        raise GangError(
            f"{GANG_DIR_ENV} is not set: this job's session runs a scheduler older than this "
            f"launcher; start the session again from this checkout (`session render`)"
        )
    return Path(value)


def job_nodes(environ: Mapping[str, str]) -> int:
    """How many nodes the job holds: 1 outside a multi-node job."""
    try:
        return max(1, int(environ.get(NODES_ENV) or 1))
    except ValueError:
        return 1


def node_slots(environ: Mapping[str, str], ranks: int | None, nodes: int) -> int:
    """Ranks per node: this node's devices (a multi-node job holds every device of each node).

    TensorRT-LLM puts rank ``r`` on node ``r // devices`` and device ``r % devices``; giving each
    host as many slots as it has devices makes ``mpirun --map-by slot`` agree with it.
    """
    visible = [d for d in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if visible:
        slots = len(visible)
    elif ranks:
        slots = -(-ranks // nodes)
    else:
        raise GangError(
            "CUDA_VISIBLE_DEVICES is not set, so the ranks per node are unknown: a multi-node "
            "job's part holds every device of its node"
        )
    if ranks is not None and not (slots * (nodes - 1) < ranks <= slots * nodes):
        raise GangError(
            f"{ranks} ranks do not span {nodes} nodes of {slots} devices: each node but the "
            f"last is filled, and the last holds at least one rank"
        )
    return slots


def _listed(directory: Path, name: str) -> bool:
    """Whether ``directory`` holds ``name``, read by listing it.

    Another node writes these files. On NFS a lookup of a missing name can be answered from the
    client's cache for many seconds; opening the directory revalidates it.
    """
    try:
        return name in os.listdir(directory)
    except FileNotFoundError:
        return False


def mpi_hosts(hosts: Sequence[str], slots: int) -> str:
    """``hostA:4,hostB:4``: what ``mpirun --host`` takes."""
    return ",".join(f"{host}:{slots}" for host in hosts)


def agent_environment(agent: Path) -> dict[str, str]:
    """The variables that make ``mpirun`` start its remote daemons through ``agent``."""
    return {**{name: str(agent) for name in AGENT_PARAMS}, **NO_TREE_SPAWN}


def _inbox(root: Path, *parts: str) -> Path | None:
    """``root/relay/<parts>``, made when missing; ``None`` once the job's directory is gone.

    The scheduler owns ``root`` (the job's ``gangs/<name>``): it is never made here, or a job that
    already ended would leave a directory that holds the spool's nodes for ever.
    """
    path = root / RELAY_DIRNAME
    try:
        path.mkdir(exist_ok=True)
        for part in parts:
            path = path / part
            path.mkdir(exist_ok=True)
    except FileNotFoundError:
        return None
    return path


def write_agent(directory: Path, python: str = sys.executable) -> Path:
    """The agent ``mpirun`` runs, as an executable file: it takes no arguments of its own."""
    relay = _inbox(directory)
    if relay is None:
        raise GangError(f"{directory} is gone: the multi-node job has ended")
    path = relay / AGENT_NAME
    text = (
        "#!/bin/sh\n"
        f'exec {shlex.quote(python)} {shlex.quote(str(Path(__file__).resolve()))} agent "$@"\n'
    )
    write_text_atomic(path, text, mode=0o755)
    return path


def lead(
    argv: Sequence[str],
    *,
    nodes: int,
    ranks: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str]] | None:
    """What this node runs: ``(argv, env)`` to exec, or ``None`` when it serves (node > 0).

    ``nodes`` is how many nodes the command was built for; a job of another size is refused
    rather than squeezed onto it.
    """
    env = dict(os.environ if environ is None else environ)
    held = job_nodes(env)
    if held != nodes:
        raise GangError(
            f"this command spans {nodes} node(s) and its job holds {held}: submit it with "
            f"`nodes={nodes}` to a session of that many nodes"
        )
    if nodes == 1:
        launch = rank_launch(ranks, argv[0]) if ranks and ranks > 1 else []
        return [*launch, *(argv[1:] if launch else argv)], env
    root = gang_dir(env)
    rank = int(env.get(NODE_RANK_ENV) or 0)
    if rank > 0:
        return None
    hosts = [h for h in env.get(HOSTS_ENV, "").split(",") if h]
    if len(hosts) != nodes:
        raise GangError(f"{HOSTS_ENV}={env.get(HOSTS_ENV, '')!r} does not name {nodes} hosts")
    spread = mpi_hosts(hosts, node_slots(env, ranks, nodes))
    env.update({GANG_MPI_HOSTS_ENV: spread, **agent_environment(write_agent(root))})
    if ranks:
        launch = rank_launch(ranks, argv[0], hosts=spread)
        return [*launch, *argv[1:]], env
    return list(argv), env


# ------------------------------------------------------------------- the relay


def _rank_of(host: str, hosts: Sequence[str]) -> int | None:
    """Which node ``host`` names: as given, else by its short name (Open MPI may drop a domain)."""
    if host in hosts:
        return hosts.index(host)
    short = [h.split(".", 1)[0] for h in hosts]
    name = host.split(".", 1)[0]
    return short.index(name) if name in short else None


def agent(host: str, words: Sequence[str], environ: Mapping[str, str] | None = None) -> int:
    """Run ``words`` on ``host`` through its node's part, as ``ssh host words...`` would.

    Returns the command's exit code, or :data:`RC_NODE_GONE` when the node's part ended first.
    ``mpirun`` does not wait for its agents once its daemons are told to exit: an agent whose
    ``mpirun`` is gone withdraws its command and exits too, rather than outlive the job.
    """
    env = os.environ if environ is None else environ
    root = gang_dir(env)
    hosts = [h for h in env.get(HOSTS_ENV, "").split(",") if h]
    rank = _rank_of(host, hosts)
    command = " ".join(words)
    if rank is None:
        _say(f"agent: {host!r} is none of this job's hosts {hosts}")
        return RC_NODE_GONE
    if rank == 0:
        return subprocess.call(["/bin/sh", "-c", command], stdin=subprocess.DEVNULL)
    inbox = _inbox(root, str(rank))
    if inbox is None:
        _say(f"agent: {root} is gone: the multi-node job has ended")
        return RC_NODE_GONE
    name = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    rc_file, cancel = inbox / f"{name}.rc", inbox / f"{name}.cancel"

    def withdraw(signum, frame):  # noqa: ARG001
        cancel.touch()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, withdraw)
    parent = os.getppid()
    write_json_atomic(
        inbox / f"{name}.json", {"host": host, "command": command, "submitted": time.time()}
    )
    while True:
        if _listed(inbox, rc_file.name):
            text = rc_file.read_text(encoding="utf-8").strip()
            return int(text) if text.lstrip("-").isdigit() else RC_NODE_GONE
        if _listed(root / "parts", f"{rank}.json") or not root.is_dir():
            _say(f"agent: node {rank} ({host}) ended before its command did: {command[:200]}")
            return RC_NODE_GONE
        if os.getppid() != parent:
            cancel.touch()
            return RC_NODE_GONE
        time.sleep(POLL_S)


class _Relayed:
    """One command this node runs for node 0's ``mpirun``."""

    def __init__(self, name: str, command: str) -> None:
        self.name = name
        self.command = command
        self.proc = subprocess.Popen(["/bin/sh", "-c", command], stdin=subprocess.DEVNULL)
        self.cancelled = False


def serve(
    root: Path,
    rank: int,
    *,
    poll_s: float = POLL_S,
    grace_s: float = STOP_GRACE_S,
    failure_grace_s: float = FAILURE_GRACE_S,
) -> int:
    """Run node ``rank``'s relayed commands until node 0's part ends; 0, or a failure's code.

    Node 0's part carries the job's verdict: a daemon that ends as its world ends is no failure,
    and one still running then is stopped. A command that fails while node 0 still runs fails
    this part, and with it the job's other parts, only if node 0 has not ended
    ``failure_grace_s`` later: ``mpirun`` normally stops the job itself when a daemon is lost.
    """
    inbox = _inbox(root, str(rank))
    if inbox is None:
        return 0  # node 0's part, and the job, have ended
    running: dict[str, _Relayed] = {}
    failed, failed_at = 0, 0.0

    def stop(signum, frame):  # noqa: ARG001
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    _say(f"node {rank}: serving {inbox} until node 0's part ends")
    try:
        while True:
            ended = _listed(root / LEADER_PART.parent, LEADER_PART.name) or not root.is_dir()
            try:
                present = set(os.listdir(inbox))
            except FileNotFoundError:
                present, ended = set(), True
            for entry in () if ended else sorted(present):
                name, suffix = os.path.splitext(entry)
                if suffix != ".json" or name in running or f"{name}.rc" in present:
                    continue
                request = inbox / entry
                command = str((read_json(request) or {}).get("command") or "")
                if not command:
                    write_text_atomic(inbox / f"{name}.rc", f"{RC_NO_COMMAND}\n")
                    continue
                _say(f"node {rank}: running {name}: {command[:300]}")
                running[name] = _Relayed(name, command)
            for name, job in list(running.items()):
                if not job.cancelled and f"{name}.cancel" in present:
                    job.cancelled = True
                    job.proc.terminate()
                rc = job.proc.poll()
                if rc is None:
                    continue
                del running[name]
                write_text_atomic(inbox / f"{name}.rc", f"{rc}\n")
                _say(f"node {rank}: {name} exited {rc}")
                if rc != 0 and not job.cancelled and not failed:
                    failed, failed_at = (rc if rc > 0 else 128 - rc), time.monotonic()
            if ended:
                _stop(running.values(), grace_s)
                for name, job in running.items():
                    write_text_atomic(inbox / f"{name}.rc", f"{job.proc.returncode}\n")
                running.clear()
                return 0
            if failed and time.monotonic() - failed_at >= failure_grace_s:
                _say(f"node {rank}: a relayed command failed ({failed}) and node 0 runs on")
                return failed
            time.sleep(poll_s)
    finally:
        _stop(running.values(), grace_s)


def _stop(jobs, grace_s: float) -> None:
    alive = [job.proc for job in jobs if job.proc.poll() is None]
    for proc in alive:
        proc.terminate()
    deadline = time.monotonic() + grace_s
    for proc in alive:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gang.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("lead", help="node 0 runs the payload; the others serve its daemons")
    run.add_argument("--nodes", type=int, required=True, help="nodes the command spans")
    run.add_argument("--ranks", type=int, help="start this many ranks of the payload's launcher")
    run.add_argument("argv", nargs=argparse.REMAINDER, help="-- the payload")
    relay = sub.add_parser("agent", help="Open MPI's remote launch agent (node 0 only)")
    relay.add_argument("host")
    relay.add_argument("words", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "agent":
            return agent(args.host, args.words)
        payload = list(args.argv[1:] if args.argv[:1] == ["--"] else args.argv)
        if not payload:
            _say("lead: no payload after --")
            return RC_USAGE
        planned = lead(payload, nodes=args.nodes, ranks=args.ranks)
        if planned is None:
            return serve(gang_dir(os.environ), int(os.environ[NODE_RANK_ENV]))
    except GangError as exc:
        _say(str(exc))
        return RC_USAGE
    command, env = planned
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(command[0], command, env)
    return 0  # unreachable: exec replaces this process


__all__ = [
    "AGENT_PARAMS",
    "GANG_DIR_ENV",
    "GangError",
    "LEADER_PART",
    "NO_TREE_SPAWN",
    "RC_NODE_GONE",
    "RELAY_DIRNAME",
    "agent",
    "agent_environment",
    "job_nodes",
    "lead",
    "main",
    "mpi_hosts",
    "node_slots",
    "serve",
    "write_agent",
]


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How GPU work is launched — resolved once, then handed to every agent.

The ``environment`` block of ``task.yaml`` becomes a :class:`RunEnvironment`, whose ``prefix``
every GPU command is prepended with; the agent prompts carry that string. ``local`` runs on this
host's claimed devices, ``session_spool`` through a compute session's spool. This module only
builds commands; :mod:`.executor` runs them.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..files import read_json
from . import GANG_PATH, SPOOLCTL_PATH
from .gang import AGENT_PARAMS, NO_TREE_SPAWN
from .scheduler import SpoolPaths
from .server_runtime import MULTI_TASK_SLURM_VARS, rank_launch
from .sessions import (
    GATE_ROLES,
    MERGED_ENV_VARS,
    SessionSpec,
    describe_sessions,
    env_assignments,
    sessions_from_task,
)

#: The wrapper that makes a pre-launched MPI world work: rank 0 runs the leader and the task,
#: every other rank a worker.
LLMAPI_LAUNCH = "trtllm-llmapi-launch"

#: Set at one rank (:attr:`RunEnvironment.launch_env`): the LLM API then runs its worker in its
#: own process instead of spawning it over MPI.
SINGLE_PROCESS_WORKER_ENV = "TLLM_WORKER_USE_SINGLE_PROCESS"

# The launch rule is TensorRT-LLM's and turns on the task count alone: one task scrubs the MPI
# environment and runs bare; more run under `mpirun` and the launcher. One task is the special
# case because `trtllm-bench` and the LLM API import mpi4py at load, which initializes MPI
# whenever PMI variables are present (in a PMIx-managed step, always); the process then dies
# spawning a worker, which pyxis forbids. `SLURM_PROCID` and `SLURM_LOCALID` survive the scrub:
# they name the rank rather than bootstrap MPI, and code downstream reads them. So do the
# MPI *settings* (:data:`MPI_SETTINGS_KEPT`, and every MPI variable the run's environment
# declares): they bootstrap nothing, and a one-task process that starts `mpirun` itself (the
# tree's disaggregated launcher does) needs them -- a container running as root, for one,
# cannot start its workers without `OMPI_ALLOW_RUN_AS_ROOT`.
MPI_SETTINGS_KEPT: tuple[str, ...] = (
    "OMPI_ALLOW_RUN_AS_ROOT",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
)
#: The prefixes of MPI variables a run's environment may declare and keep through the scrub.
_DECLARED_MPI_PREFIXES: tuple[str, ...] = ("OMPI_MCA_", "PMIX_MCA_", "OMPI_ALLOW_")
#: What node 0 of a multi-node job sets for the remote launch (``gang.py``), kept the same way.
GANG_AGENT_SETTINGS: tuple[str, ...] = (*AGENT_PARAMS, *NO_TREE_SPAWN)


def single_task_scrub(keep: Sequence[str] = ()) -> str:
    """The shell loop that unsets the MPI bootstrap environment, but ``keep`` and the settings."""
    kept = ["SLURM_PROCID", "SLURM_LOCALID", *MPI_SETTINGS_KEPT, *sorted(set(keep))]
    tests = " && ".join(f'[ "$v" != {name} ]' for name in dict.fromkeys(kept))
    return f'for v in ${{!PMI@}} ${{!PMIX@}} ${{!MPI@}} ${{!OMPI@}} ${{!SLURM@}}; do {tests} && unset "$v"; done'


SINGLE_TASK_MPI_SCRUB = single_task_scrub()

# Above one task, under `mpirun`, the misleading variables are the Slurm task ones:
# `trtllm-llmapi-launch` reads `SLURM_PROCID` before `OMPI_COMM_WORLD_RANK` for its rank and
# `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE` for `mpi_world_size`. Right under `srun`, wrong
# under an `mpirun` inside a one-task allocation, which is what a `session_spool` job is.
# Unscrubbed, every rank reads rank 0 of a world of 1 and becomes a `mgmn_leader_node` waiting
# for workers that never come: no error, no exit, a hang.
MULTI_TASK_SLURM_SCRUB = "env " + " ".join(f"-u {name}" for name in MULTI_TASK_SLURM_VARS)


@dataclass(frozen=True)
class RunEnvironment:
    """A resolved, immutable description of how to launch GPU work."""

    kind: str
    python: str
    env: dict[str, str] = field(default_factory=dict)
    devices: tuple[int, ...] = ()
    world_size: int = 1
    # The image the GPU jobs run in, if any: an interpreter path above is a path inside it.
    container_image: str | None = None
    # ``session_spool`` only: the spool a long-lived compute session watches.
    spool: str | None = None
    # Every declared session an agent may submit to.
    sessions: tuple[SessionSpec, ...] = ()
    # ``perf.dev_spools``: the sessions the tuners submit to; no other prompt names them.
    dev_spools: tuple[str, ...] = ()
    # The most devices one node holds (``environment.gpus_per_node``); ``None`` when undeclared.
    gpus_per_node: int | None = None
    # The nodes one engine's ranks span: more than one only for a member wider than a node,
    # whose commands run in a multi-node job (:meth:`at_width`, ``gang.py``).
    nodes: int = 1

    @property
    def ntasks(self) -> int:
        """One task per rank. This is the number the whole launch turns on."""
        return max(1, self.world_size)

    @property
    def needs_llmapi_launch(self) -> bool:
        """Whether the payload must be wrapped in ``trtllm-llmapi-launch``."""
        return self.ntasks > 1

    def at_width(self, world_size: int) -> "RunEnvironment":
        """This environment for one engine of ``world_size`` ranks, across as many nodes as it needs.

        An engine wider than one node (``gpus_per_node``) spans ``ceil(world / per node)`` nodes:
        its commands run in a multi-node job of that many nodes, launched by ``gang.py``.
        """
        size = max(1, int(world_size))
        return replace(self, world_size=size, nodes=nodes_for_width(size, self.gpus_per_node))

    @property
    def launch_env(self) -> dict[str, str]:
        """``env``, plus ``TLLM_WORKER_USE_SINGLE_PROCESS=1`` at one rank unless ``env`` sets it.

        A one-rank engine has no launcher to create its worker, so the LLM API would spawn it with
        ``MPI_Comm_spawn``, which a container runtime such as pyxis forbids (``MPI_ERR_SPAWN``).
        """
        env = {key: str(value) for key, value in self.env.items()}
        if self.ntasks == 1:
            env.setdefault(SINGLE_PROCESS_WORKER_ENV, "1")
        return env

    @property
    def prefix(self) -> str:
        """The literal string every GPU command is prepended with."""
        parts: list[str] = []
        if self.devices:
            parts.append("CUDA_VISIBLE_DEVICES=" + ",".join(str(d) for d in self.devices))
        parts.extend(env_assignments(self.launch_env))
        return " ".join(parts)

    def _scrub_and_exec(self, argv: str) -> str:
        """``argv`` behind a scrub of the MPI environment, in this process."""
        declared = [k for k in self.env if str(k).startswith(_DECLARED_MPI_PREFIXES)]
        return f"bash -c {shlex.quote(f'{single_task_scrub(declared)}; exec {argv}')}"

    def _mpirun_launch(self, argv: str) -> str:
        """``argv`` behind an ``mpirun`` that creates the ranks here.

        Without it the LLM API finds a world size of 1, tries to create the other ranks itself,
        and dies with ``MPI_ERR_SPAWN``. ``--oversubscribe``: Open MPI takes its slot count from
        Slurm, and a spool session is **one task** that owns the node, so ``-n 4`` asks for four
        slots against one and dies before any Python runs. Raising the session's
        ``ntasks_per_node`` instead would start four spool watchers racing for every job.
        """
        launch = f"{shlex.join(rank_launch(self.ntasks, LLMAPI_LAUNCH))} {argv}"
        # The launcher is installed beside the interpreter; a session's PATH need not hold it
        # (`mpirun` reports `trtllm-llmapi-launch` not found), so its directory goes first.
        if os.path.isabs(self.python):
            return f'PATH={shlex.quote(os.path.dirname(self.python))}:"$PATH" {launch}'
        return launch

    def _gang_launch(self, argv: str, *, single_process: bool) -> str:
        """``argv`` as one engine across the ``nodes`` of a multi-node job (``gang.py lead``).

        The command runs on every node of the job. Node 0 runs ``argv``: under ``mpirun`` over
        every node's devices when this is engine work, as one process (which may start the
        ranks itself through ``server_runtime.rank_launch``) when ``single_process``. Every
        other node lends its devices: it starts the daemons node 0's ``mpirun`` asks for, in the
        environment this same command line gives it there.
        """
        lead = [self.python, str(GANG_PATH), "lead", "--nodes", str(self.nodes)]
        if single_process:
            # The launch agent node 0 sets is an MPI setting, kept like the declared ones.
            kept = [k for k in self.env if str(k).startswith(_DECLARED_MPI_PREFIXES)]
            scrub = single_task_scrub([*kept, *GANG_AGENT_SETTINGS])
            payload = f"bash -c {shlex.quote(f'{scrub}; exec {argv}')}"
            return f"{shlex.join(lead)} -- {payload}"
        launch = f"{shlex.join([*lead, '--ranks', str(self.ntasks)])} -- {LLMAPI_LAUNCH} {argv}"
        if os.path.isabs(self.python):
            return f'PATH={shlex.quote(os.path.dirname(self.python))}:"$PATH" {launch}'
        return launch

    def command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return the command line ``argv`` prefixed for this environment.

        ``single_process`` is for work that is **not** engine work, such as a probe that imports
        the build and exits: one process whatever the world size, no launcher, the MPI
        environment scrubbed. Across ``nodes`` it runs on node 0 alone.
        """
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        if single_process and self.ntasks > 1:
            # Re-asked at one task: the launcher follows from the task count, and shrinking only
            # the payload would still start one copy of the probe per rank.
            return replace(self, world_size=1).command(argv, single_process=True)
        if self.nodes > 1:
            argv = self._gang_launch(argv, single_process=single_process)
        elif self.needs_llmapi_launch:
            argv = self._mpirun_launch(argv)
        elif single_process:
            # The process inherits a step's PMI variables: a spool script runs inside the
            # session's own allocation.
            argv = self._scrub_and_exec(argv)
        prefix = self.prefix
        return f"{prefix} {argv}".strip()

    def python_command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return a Python invocation of ``argv`` prefixed for this environment."""
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        return self.command(f"{shlex.quote(self.python)} {argv}", single_process=single_process)

    @property
    def _widest_nodes(self) -> int:
        """The nodes this run's widest engine spans (``world_size`` is the widest member's)."""
        return nodes_for_width(self.world_size, self.gpus_per_node)

    def _describe_gang(self) -> str:
        """How to run an engine wider than one node; empty when every engine fits one."""
        nodes = self._widest_nodes
        if nodes <= 1 or not self.gpus_per_node:
            return ""
        wide = self.at_width(self.world_size)
        server = shlex.join([self.python, str(GANG_PATH), "lead", "--nodes", str(nodes)])
        narrow = replace(self, world_size=self.gpus_per_node, nodes=1)
        # Only a session of that many nodes ever starts the job: anywhere else it waits for ever.
        holding = ", ".join(
            f"`{s.name}`: `SPOOL={s.spool}`"
            for s in self.sessions
            if s.role in GATE_ROLES and s.nodes >= nodes
        )
        return (
            f"This run has an engine **wider than one node** ({self.gpus_per_node} GPUs): "
            f"{self.world_size} ranks span {nodes} nodes. Submit its work with "
            f"`--nodes {nodes}` and no `--gpus` (it holds every device of each node) to a "
            f"session of that many nodes{f' ({holding})' if holding else ''}; on any other "
            "session the job waits for nodes that never come. Launch the engine with:\n"
            "\n"
            "```bash\n"
            f"{wide._gang_launch('<your command>', single_process=False)}\n"
            "```\n"
            "\n"
            "The same command runs on every node of the job: node 0 starts every rank with one "
            "`mpirun`, and each other node starts what that `mpirun` asks of it, so never "
            "branch on the node yourself. A process that starts its own ranks (a server) goes "
            f"behind `{server} -- <command>` instead. An engine that fits one node is "
            "launched on one node, with `-n` its own width:\n"
            "\n"
            "```bash\n"
            f"{narrow._mpirun_launch('<your command>')}\n"
            "```\n"
            "\n"
        )

    def describe(self) -> str:
        """The block injected verbatim into every agent's prompt."""
        if self.kind == "session_spool":
            return self._describe_spool()
        device_line = ", ".join(str(d) for d in self.devices) if self.devices else "(none)"
        return (
            "## Running GPU work\n"
            "\n"
            "Every GPU command in this run is launched exactly one way. Prefix it with:\n"
            "\n"
            "```\n"
            f"{self.prefix}\n"
            "```\n"
            "\n"
            f"The Python interpreter is `{self.python}`. The devices claimed for this run "
            f"are: {device_line} (world size {self.world_size}).\n"
            "\n"
            "**Do not construct a launcher command, do not probe for one, and do not call a "
            "test-runner skill.** The environment was resolved before you started and it is "
            "not yours to rediscover. If this prefix does not fit what you need to run, that "
            "is a **symptom to report in your final message** — not a problem to solve.\n"
        )

    def _describe_spool(self) -> str:
        """The prompt block for a long-lived compute session watching a spool."""
        spool = self.spool or "<spool>"
        exports = "\n".join(f"export {part}" for part in env_assignments(self.launch_env))
        if not exports:
            exports = "# no extra environment"
        submit = f"python3 {SPOOLCTL_PATH} submit"
        control = f"python3 {SPOOLCTL_PATH}"
        return (
            "## Running GPU work\n"
            "\n"
            "A compute session is already running and its scheduler watches a spool "
            "directory. You do **not** allocate nodes or start GPU work any other way: "
            "no cluster scheduler commands, no GPU process of your own. Work is a file.\n"
            "\n"
            "Run something and wait for it (prints the log, exits with the job's code):\n"
            "\n"
            "```bash\n"
            f"SPOOL={spool} {submit} --gpus <N|all> --wall <limit> <name> '<command>'\n"
            "```\n"
            "\n"
            "- `--gpus N` asks for N devices; the scheduler exports `CUDA_VISIBLE_DEVICES` to "
            "exactly those. **Never set `CUDA_VISIBLE_DEVICES` yourself.** `--gpus all` holds "
            "the whole node, exclusively — use it for anything that loads the full model. "
            "Without `--gpus`, a `free-` name holds no device and any other name holds the "
            "whole node.\n"
            "- `--wall` is counted from the job's start; time spent queued is free. A job "
            "past its limit is stopped and reported as timed out.\n"
            "- **A job waiting in the queue is not a block.** Other tasks share these "
            "devices; yours starts when they free. Do the rest of your task meanwhile and "
            "poll its log or `status`; the site blocks you only when it refuses the job or "
            "no session runs.\n"
            "- `<name>` must be unique to what you are doing and must start with your "
            "request's prefix when one is set (`$MV2FLOW_JOB_PREFIX`). Its log is "
            f"`{spool}/logs/<name>.log` and **that file is the result**; your recollection of "
            "what scrolled past is not.\n"
            f"- `{control} status` shows what runs, what waits and why. "
            f"`{control} cancel <name>` is the only way to stop a job: never `kill`, "
            "`pkill` or time out a process yourself.\n"
            "\n"
            "The script runs inside the container on the compute node, so it sees the same "
            "filesystem you do. Start it with:\n"
            "\n"
            "```bash\n"
            f"{exports}\n"
            "```\n"
            "\n"
            f"The Python interpreter is `{self.python}`. World size {self.world_size}.\n"
            "\n"
            + describe_sessions(self.sessions)
            + (
                "This run is **multi-rank**, so an engine command has to be launched with "
                "the ranks already in place:\n"
                "\n"
                "```bash\n"
                f"{self._mpirun_launch('<your command>')}\n"
                "```\n"
                "\n"
                "Without the wrapper the LLM API sees a world size of 1, concludes it must "
                "create the other ranks itself, and dies with `MPI_ERR_SPAWN` — which the "
                "container runtime forbids. Do not work around that by reducing the world "
                "size.\n"
                "\n"
                "**Copy that line whole.** The `env -u SLURM_...` prefix and "
                "`--oversubscribe` are not decoration: this `mpirun` runs inside the "
                "session's own one-task Slurm allocation, so without them it either "
                "refuses to start (*not enough slots*) or starts "
                f"{self.ntasks} processes that each believe they are rank 0 of a world "
                "of 1 and hang forever with no error.\n"
                "\n"
                if self.needs_llmapi_launch and self._widest_nodes == 1
                else ""
            )
            + self._describe_gang()
            + "**Do not construct a launcher command, do not probe for one, and do not call a "
            "test-runner skill.** The session was resolved before you started and it is not "
            "yours to rediscover. If this protocol does not fit what you need to run, that "
            "is a **symptom to report in your final message** — not a problem to solve.\n"
        )


def shares_allocation_with_spool(env: "RunEnvironment") -> str | None:
    """Warn when the orchestrator lives inside the allocation it dispatches to.

    The evidence is the ``slurm_job`` of the spool's ``status.json``; only two known, equal job ids
    warn.
    """
    if env.kind != "session_spool" or not env.spool:
        return None
    # A spool job has no SLURM_* variables (they are scrubbed) and carries the session's job id
    # under this name instead.
    mine = os.environ.get("SLURM_JOB_ID") or os.environ.get("MV2FLOW_SPOOL_SESSION_JOBID")
    if not mine:
        return None
    theirs = str((read_json(SpoolPaths(env.spool).snapshot) or {}).get("slurm_job") or "")
    if not theirs or theirs == "unknown" or theirs != mine:
        return None
    return (
        f"the orchestrator is running inside Slurm job {mine}, which is also the "
        f"compute session watching {env.spool}. They will die together, and this "
        f"process needs no GPU: launch it in its own allocation "
        f"(`--gpus-per-node=0`) so a preempted session costs you the work in "
        f"flight and not the run."
    )


def gpus_per_node(data: Mapping[str, Any]) -> int | None:
    """The most devices one node of the run holds: ``environment`` or one of its sessions."""
    env = data.get("environment") or {}
    sizes = [env.get("gpus_per_node")] + [
        s.get("gpus_per_node") for s in env.get("sessions") or () if isinstance(s, Mapping)
    ]
    sizes = [s for s in sizes if isinstance(s, int) and not isinstance(s, bool) and s > 0]
    return max(sizes) if sizes else None


def nodes_for_width(width: int | None, per_node: int | None) -> int:
    """The nodes one engine of ``width`` ranks spans when a node holds ``per_node`` devices.

    One while it fits a node or either is unknown: an undeclared ``gpus_per_node`` keeps every
    engine on one node.
    """
    if not width or not per_node or width <= per_node:
        return 1
    return -(-width // per_node)


def scheduler_owns_devices(kind: str) -> bool:
    """Whether a scheduler, not this process, decides the devices and the host.

    "Not local" rather than a list of remote kinds: a kind added later is remote until someone
    says otherwise, the safe default.
    """
    return kind != "local"


def resolve_environment(
    task_data: Mapping[str, Any],
    *,
    devices: Sequence[int] = (),
    world_size: int = 1,
) -> RunEnvironment:
    """Build a :class:`RunEnvironment` from a validated task spec and preflight's ``devices``."""
    block = dict(task_data.get("environment") or {})
    env = {str(k): str(v) for k, v in (block.get("env") or {}).items()}

    declared = sessions_from_task({"environment": block}) if block.get("sessions") else ()
    return RunEnvironment(
        kind=str(block.get("kind", "local")),
        python=str(block.get("python", "python3")),
        env=env,
        devices=tuple(int(d) for d in devices),
        world_size=max(1, int(world_size)),
        container_image=block.get("container_image"),
        spool=block.get("spool"),
        sessions=declared,
        dev_spools=_dev_spools(task_data.get("perf")),
        gpus_per_node=gpus_per_node(task_data),
    )


def _dev_spools(perf: Any) -> tuple[str, ...]:
    """``perf.dev_spools``, less any spool that also measures (``perf.spools``).

    Only a declared list: the campaign's fallback to its measuring spools is not one, and a gate
    beside a measurement would skew it.
    """
    if not isinstance(perf, Mapping):
        return ()
    measuring = {Path(str(s)) for s in perf.get("spools") or ()}
    return tuple(
        str(s) for s in perf.get("dev_spools") or () if s and Path(str(s)) not in measuring
    )


# Why `TMPDIR` has a maximum length. The LLM API's RPC binds an IPC socket
# under `TMPDIR` named `rpc_test_<uuid4>` (`get_unique_ipc_addr` in
# `tensorrt_llm/executor/rpc/rpc_common.py`, `os.path.join`ed onto
# `tempfile.gettempdir()`). A unix domain socket path is capped by the kernel
# at `sizeof(sockaddr_un.sun_path)`: an `AF_UNIX` bind succeeds at 107
# characters and fails at 108. `len("rpc_test_") + len(str(uuid.uuid4()))` is
# 9 + 36, and the separator `os.path.join` inserts is the third term.
UNIX_SOCKET_PATH_MAX = 107
RPC_SOCKET_BASENAME_LEN = 45
MAX_TMPDIR_LEN = UNIX_SOCKET_PATH_MAX - RPC_SOCKET_BASENAME_LEN - len("/")


class EnvironmentError(ValueError):
    """A declared environment cannot safely run the workload."""


def check_tmpdir(env: Mapping[str, str], *, require_set: bool = True) -> None:
    """Assert ``TMPDIR`` and ``CLAUDE_CODE_TMPDIR`` are set, short, and not ``/tmp``.

    The length bound (:data:`MAX_TMPDIR_LEN`, a unix socket's path) applies to
    ``TMPDIR`` alone: ``CLAUDE_CODE_TMPDIR`` binds no socket. ``env`` is the
    environment **declared in task.yaml**; this check does *not* fall back to
    the orchestrator's own ``os.environ``. ``require_set=False`` checks only
    the variables that are set.
    """
    for name in ("TMPDIR", "CLAUDE_CODE_TMPDIR"):
        value = env.get(name)
        if not value:
            if not require_set:
                continue
            raise EnvironmentError(
                f"{name} is not set. Engine builds and NVRTC scratch default there, and "
                f"on a shared host the default fills to zero mid-run. Set it in "
                f"task.yaml's `environment.env`."
            )
        resolved = Path(value).resolve()
        if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
            raise EnvironmentError(
                f"{name}={value} points into /tmp. Another tenant can fill it to zero "
                f"mid-run; point it at a scratch directory of the run's own."
            )
        if name == "TMPDIR" and len(value) > MAX_TMPDIR_LEN:
            raise EnvironmentError(
                f"{name}={value} is {len(value)} characters. The LLM API binds an RPC "
                f"socket at `$TMPDIR/rpc_test_<uuid4>`, and a unix socket path is "
                f"capped at {UNIX_SOCKET_PATH_MAX} "
                f"(`sizeof(sockaddr_un.sun_path)`), so {name} must be at most "
                f"{MAX_TMPDIR_LEN}. Point it at a shorter path on the same storage — "
                f"this is about the length of the name, not about the filesystem, and "
                f"a short path on the same mount works."
            )


__all__ = [
    "MAX_TMPDIR_LEN",
    "check_tmpdir",
    "EnvironmentError",
    "LLMAPI_LAUNCH",
    "MERGED_ENV_VARS",
    "MULTI_TASK_SLURM_SCRUB",
    "MULTI_TASK_SLURM_VARS",
    "RunEnvironment",
    "MPI_SETTINGS_KEPT",
    "gpus_per_node",
    "nodes_for_width",
    "SINGLE_TASK_MPI_SCRUB",
    "single_task_scrub",
    "resolve_environment",
    "scheduler_owns_devices",
    "shares_allocation_with_spool",
]

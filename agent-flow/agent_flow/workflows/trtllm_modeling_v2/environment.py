# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How GPU work is launched — resolved once, then handed to every agent.

The ``environment`` block in ``task.yaml`` is resolved here into a
:class:`RunEnvironment`, whose ``prefix`` is a literal string that every
GPU command is prepended with. That string — not a policy, not a skill, not
a decision tree — is what the agent prompts carry.

The reason is specific. The predecessor workflow told its agents to route
every test through a chain of skills that classified the request, picked a
container image, checked the environment and finally built an ``srun``
line — four hops, each one an LLM decision, each one able to fail in a way
that reads as "the environment is broken" rather than "this hop guessed
wrong". Runs spent their budget rediscovering how to submit a job. Here
the orchestrator resolves it once, in Python, before any agent exists, and
an agent that finds the prefix does not fit its need is instructed to
*report that as a symptom* rather than to work around it.

Nothing here talks to Slurm. Building the command is all this module does;
running it belongs to :mod:`gates` and to the agents.
"""

from __future__ import annotations

import math
import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

#: ``--mpi=`` values. ``auto`` resolves by task count, which is the only thing
#: that decides it.
MPI_AUTO = "auto"
MPI_NONE = "none"
MPI_PMIX = "pmix"

#: The wrapper that makes a pre-launched MPI world work. Rank 0 runs the
#: leader plus the real task; every other rank runs a worker and waits.
LLMAPI_LAUNCH = "trtllm-llmapi-launch"

# ---------------------------------------------------------------------------
# The launch rules below are TensorRT-LLM's, not this workflow's. They are
# carried over from `.claude/skills/trtllm-test-script-builder/` --- the
# repo's own Slurm script generator --- because that is the tested statement
# of how this engine must be launched, and a second, divergent statement of it
# here would be a bug generator.
#
# Ported as *rules* rather than copied as code: agent-flow is a separate
# installable package and cannot import from a skills directory.
#
# The rule is a function of the task count and nothing else:
#
#   ntasks == 1  ->  --mpi=none, scrub the MPI environment, no wrapper
#   ntasks  > 1  ->  --mpi=<pmix variant>, wrap the payload in the launcher
#
# Why one task is the special case: `trtllm-bench` and the LLM API import
# mpi4py at module load, which auto-initializes MPI whenever PMI variables are
# present. Inside a PMIx-managed step they always are --- so a single-task job
# initializes MPI for no reason and then dies trying to spawn a worker
# dynamically, which is what pyxis forbids. Clearing the variables lets the
# process skip MPI init entirely.
#
# `SLURM_PROCID` and `SLURM_LOCALID` survive the scrub: they identify the rank
# rather than bootstrap MPI, and code downstream reads them. That is true
# under `srun`, which is the only launcher this constant is applied behind;
# `MULTI_TASK_SLURM_SCRUB` below is the case where it is false, and the two
# are halves of one rule rather than two rules.
# ---------------------------------------------------------------------------
SINGLE_TASK_MPI_SCRUB = (
    "for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do "
    '[ "$v" != SLURM_PROCID ] && [ "$v" != SLURM_LOCALID ] && unset "$v"; '
    "done"
)

# ---------------------------------------------------------------------------
# The other half of the rule above, at the other end of the range.
#
# One statement, two ends: **a Slurm step's environment misleads any process
# that `srun` did not launch.** At one task the misleading variables are the
# PMI ones, and a lone process joins an MPI world it never asked for --- that
# is `SINGLE_TASK_MPI_SCRUB`. Above one task, under `mpirun`, the misleading
# variables are the Slurm *task* ones, and this is that case.
#
# `trtllm-llmapi-launch` derives its rank as
#
#     mpi_rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMI_ID:-0}}}}
#
# and, in `mpi_world_size`, reads `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
# Slurm wins over Open MPI in both. That is the right precedence under `srun`,
# which is what the launcher was written for, and exactly wrong under `mpirun`
# *inside* someone else's allocation --- which is precisely what a
# `session_spool` job is: one Slurm task that owns the node and runs `mpirun`
# itself.
#
# Measured, four ranks, one variable changed:
#
#                       SLURM_PROCID   OMPI_COMM_WORLD_RANK   SLURM_NTASKS
#   as-is               0, 0, 0, 0     0, 1, 2, 3             1
#   Slurm vars scrubbed unset          0, 1, 2, 3             unset
#
# So every rank read `mpi_rank=0` and became a leader, and every rank also
# read `mpi_world_size=1`. Four `mgmn_leader_node` processes, each waiting to
# be joined by workers that would never come. No error, no traceback, no exit
# --- the run simply hangs forever. MPI itself was never at fault; a bare
# `mpi4py` probe under the identical `mpirun` line reported `rank 0..3 of 4`.
#
# Scrubbing lets the launcher fall through to the Open MPI variables it should
# be reading here. This belongs ONLY on the `mpirun` branch: under `slurm`,
# `srun --ntasks` really does size the world and `SLURM_PROCID` really is the
# rank, so the same scrub would break a launch that is already correct.
#
# `env -u` rather than a shell loop because this one prefixes a command rather
# than running inside a `bash -c` of its own, and because the set is a fixed
# list of names rather than a glob.
# ---------------------------------------------------------------------------
MULTI_TASK_SLURM_VARS: tuple[str, ...] = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)

MULTI_TASK_SLURM_SCRUB = "env " + " ".join(f"-u {name}" for name in MULTI_TASK_SLURM_VARS)

# ---------------------------------------------------------------------------
# Variables whose declared value EXTENDS the inherited one instead of
# replacing it.
#
# Only `PATH` is on this list, and it earns the exception by being the
# variable that decides *which program runs at all*. A spec that wants its
# own interpreter ahead of everything else writes the obvious line:
#
#     PATH: <checkout>/.venv-3.12/bin:/usr/local/bin:/usr/bin:/bin
#
# which reads as "my venv first" and, as a plain assignment, means "and the
# container's MPI, CUDA and toolchain directories are gone". Measured: that
# exact line deleted `/usr/local/mpi/bin` and the next launcher line died
# with `mpirun: command not found`, rc=127 --- several layers away from the
# spec that caused it, and presenting as a broken probe rather than a
# mis-declared path.
#
# Prepending keeps the property the author was after --- every entry they
# declared still wins the lookup, in the order they declared it --- and drops
# only the part they did not ask for. The alternative considered was to
# reject a `PATH` not ending in `:$PATH`; it teaches every spec author a rule
# to make the obvious line legal, where this makes the obvious line correct.
# ---------------------------------------------------------------------------
MERGED_ENV_VARS: tuple[str, ...] = ("PATH",)


def _prepend_to_inherited(name: str, value: str) -> str:
    """``/a:/b`` -> ``'/a:/b':"$PATH"``, idempotent against an explicit suffix.

    The expansion is deliberately outside the quoting: it has to reach the
    shell as an expansion, and ``shlex.quote`` over the whole value would
    deliver ``$PATH`` as five literal characters — which is the same deletion
    with a stranger spelling. A spec that already wrote the suffix by hand is
    accepted and not doubled.
    """
    for suffix in (f":${name}", f":${{{name}}}"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value or value in (f"${name}", f"${{{name}}}"):
        return f'"${name}"'
    return f'{shlex.quote(value)}:"${name}"'


@dataclass(frozen=True)
class RunEnvironment:
    """A resolved, immutable description of how to launch GPU work."""

    kind: str
    python: str
    env: dict[str, str] = field(default_factory=dict)
    devices: tuple[int, ...] = ()
    world_size: int = 1
    # Slurm only; ``None``/empty for a local run.
    partition: str | None = None
    container_image: str | None = None
    account: str | None = None
    mounts: tuple[str, ...] = ()
    time_limit: str | None = None
    gpus_per_node: int = 0
    # Reuses an already-imported pyxis container on the node instead of
    # re-importing it. A 26 GB image costs 4-8 minutes per job to import, and
    # pyxis skips the import entirely when a container of this name already
    # exists there.
    container_name: str | None = None
    # Which Slurm MPI plugin to request. ``auto`` picks by task count, which
    # is the only thing that actually decides it; an explicit value is for a
    # cluster whose plugin is a versioned variant (``pmix_v5``).
    mpi: str = MPI_AUTO
    # ``session_spool`` only: the directory a long-lived compute session
    # watches for work.
    spool: str | None = None
    # ``environment.sessions``, as ``(name, spool, role)``: every session an
    # agent may submit to, and what each is for.
    sessions: tuple[tuple[str, str, str], ...] = ()

    @property
    def ntasks(self) -> int:
        """One task per rank. This is the number the whole launch turns on."""
        return max(1, self.world_size)

    @property
    def tasks_per_node(self) -> int:
        """Ranks per node, and therefore GPUs per node."""
        return min(self.ntasks, self.gpus_per_node) if self.gpus_per_node else self.ntasks

    @property
    def nodes(self) -> int:
        return math.ceil(self.ntasks / self.tasks_per_node)

    @property
    def mpi_plugin(self) -> str:
        """``--mpi=`` for this launch.

        ``none`` at one task and a PMIx plugin above it — see
        :data:`SINGLE_TASK_MPI_SCRUB` for why one task is the special case.
        """
        if self.mpi != MPI_AUTO:
            return self.mpi
        return MPI_PMIX if self.ntasks > 1 else MPI_NONE

    @property
    def needs_llmapi_launch(self) -> bool:
        """Whether the payload must be wrapped in ``trtllm-llmapi-launch``."""
        return self.ntasks > 1

    @property
    def prefix(self) -> str:
        """The literal string every GPU command is prepended with.

        For ``slurm_broker`` this is the **in-container** prefix. The
        transport is a broker call rather than a shell word, so it cannot
        live in the same string; see :meth:`broker_submit`.
        """
        if self.kind == "slurm":
            return self._slurm_prefix()
        return self._local_prefix()

    def broker_submit(self, command: str = "<your command>", *, script: bool = False) -> str:
        """The literal broker call that launches GPU work, fully resolved.

        ``script=True`` submits ``bash <path>`` instead of the command text.
        That is the form the orchestrator uses, and it exists because this
        template embeds its payload inside a **double-quoted** string: a real
        gate command carries ``--model_dir="/path"`` and
        ``--chat_template_kwargs={"enable_thinking": false}``, and
        interpolating one here produces a call that is mis-parsed or silently
        truncated. A path has no quoting hazard.

        ``slurm_broker`` exists because on some clusters the Slurm CLI is
        unreachable from the process that wants to use it: ``sinfo`` and
        ``sbatch`` fail to contact the controller, and an MCP broker is the
        only path through. That has one consequence worth stating plainly —
        **the orchestrator cannot run GPU work itself**, because it is not
        an MCP client. Gate *execution* therefore moves to an agent, while
        gate *adjudication* stays here: the orchestrator reads the log off
        the shared filesystem and applies its own thresholds, so nothing
        rests on an agent's report of what a number was.

        This template preserves the property the whole module exists for.
        The agent is handed a resolved call, not a decision: no image to
        pick, no partition to discover, no mount list to assemble.
        """
        mounts = ", ".join(f'"{m}"' for m in self.mounts)
        payload = f"bash {command}" if script else command
        return (
            "mcp__slurm-broker__slurm_submit(\n"
            f'    image="{self.container_image}",\n'
            f'    partition="{self.partition}",\n'
            f'    account="{self.account}",\n'
            f"    gpus_per_node={self.gpus_per_node},\n"
            f'    time_limit="{self.time_limit}",\n'
            f"    mounts=[{mounts}],\n"
            f'    command=["bash", "-lc", "{payload}"])'
        )

    def _env_assignments(self) -> list[str]:
        """``KEY=value`` pairs in a stable order, shell-quoted.

        A variable named in :data:`MERGED_ENV_VARS` is emitted as a prepend
        to whatever the host already exports rather than as a replacement;
        see that constant for the failure it prevents.
        """
        parts: list[str] = []
        for key, value in sorted(self.env.items()):
            text = str(value)
            if key in MERGED_ENV_VARS:
                parts.append(f"{key}={_prepend_to_inherited(key, text)}")
            else:
                parts.append(f"{key}={shlex.quote(text)}")
        return parts

    def _local_prefix(self) -> str:
        parts: list[str] = []
        if self.devices:
            parts.append("CUDA_VISIBLE_DEVICES=" + ",".join(str(d) for d in self.devices))
        parts.extend(self._env_assignments())
        return " ".join(parts)

    def _slurm_prefix(self) -> str:
        """An ``srun`` line carrying the allocation, the container and the env.

        **One task per rank.** Omitting ``--ntasks`` is not a smaller version
        of this line, it is a broken one: ``srun`` then launches a single
        task, the MPI world size is 1, and the LLM API concludes it must
        create the other ranks itself —
        ``need_spawn_mpi_workers`` in ``llmapi/mpi_session.py`` is exactly
        ``get_mpi_world_size() == 1 and model_world_size > 1``. Under pyxis
        that dynamic spawn is forbidden and the run dies with
        ``MPI_ERR_SPAWN``. Sizing the allocation without sizing the task count
        produces that failure every time at world size > 1.

        ``--gres`` is **per node**, so it carries ``tasks_per_node`` rather
        than the world size; the two differ as soon as a run spans nodes.

        ``CUDA_VISIBLE_DEVICES`` is deliberately absent: under Slurm the
        allocation *is* the device claim, and setting both means the job
        masks its own allocated devices by index — which presents as
        "device 0 not found" on a node where device 0 was never ours.
        """
        parts = ["srun", "-l"]
        if self.partition:
            parts.append(f"--partition={self.partition}")
        if self.account:
            parts.append(f"--account={self.account}")
        if self.time_limit:
            parts.append(f"--time={self.time_limit}")
        parts.append(f"--nodes={self.nodes}")
        parts.append(f"--ntasks={self.ntasks}")
        parts.append(f"--ntasks-per-node={self.tasks_per_node}")
        parts.append(f"--gres=gpu:{self.tasks_per_node}")
        if self.container_image:
            parts.append(f"--container-image={shlex.quote(self.container_image)}")
        if self.container_name:
            parts.append(f"--container-name={shlex.quote(self.container_name)}")
        if self.mounts:
            parts.append("--container-mounts=" + shlex.quote(",".join(self.mounts)))
        plugin = self.mpi_plugin
        if plugin != MPI_NONE:
            parts.append(f"--mpi={plugin}")
        parts.append("--kill-on-bad-exit=1")
        exports = ",".join(["ALL", *self._env_assignments()])
        parts.append(f"--export={exports}")
        return " ".join(parts)

    def _scrub_and_exec(self, argv: str) -> str:
        """``argv`` behind a scrub of the MPI environment, in this process.

        Shared by the ``srun`` payload and by the launcher-free path, because
        it is one rule and not two: see :data:`SINGLE_TASK_MPI_SCRUB` for why
        a lone process must not inherit the PMI variables.
        """
        return f"bash -c {shlex.quote(f'{SINGLE_TASK_MPI_SCRUB}; exec {argv}')}"

    def _mpirun_launch(self, argv: str) -> str:
        """``argv`` behind an ``mpirun`` that creates the ranks here.

        Same engine rule as :meth:`_slurm_payload`, different transport.
        Inside a spool script or on a local box there is no ``srun`` to create
        the ranks, so ``mpirun`` does it and the wrapper assigns leader/worker
        exactly as it does under Slurm. Without it the LLM API finds a world
        size of 1, concludes it must create the other ranks itself, and dies
        with ``MPI_ERR_SPAWN``.

        Two flags here are about running ``mpirun`` **inside someone else's
        allocation**, which is the one thing this branch does and the ``slurm``
        branch never does. Neither belongs there, and the asymmetry is the
        point in both cases:

        - :data:`MULTI_TASK_SLURM_SCRUB` — the allocation's task variables
          describe the *step*, not the MPI world, and the launcher reads them
          first. See that constant.
        - ``--oversubscribe`` — Open MPI takes its slot count from the resource
          manager, and a spool session is submitted as **one task** that owns
          the node (measured: ``NumTasks=1`` against ``NumCPUs=144`` and
          ``gres/gpu:4``). So ``-n 4`` asks for four slots against one and dies
          before any Python runs, with *"There are not enough slots available
          in the system"*. The allocation really does hold all of the node's
          CPUs and GPUs, so the slot count is a bookkeeping guard rather than a
          real constraint, and oversubscribing it costs nothing.

          Under ``slurm`` the same flag would be a hazard rather than a fix:
          there ``srun --ntasks`` genuinely sizes the world, so a slot shortage
          is a *mis-sized allocation reporting itself*, and oversubscribing
          would hide it and run the ranks somewhere they do not fit.

        Raising the session's own ``ntasks_per_node`` instead was considered
        and is wrong: it gives MPI its slots and simultaneously starts four
        copies of the spool watcher, which then race for every piece of work.
        The session is one process by design.
        """
        return (
            f"{MULTI_TASK_SLURM_SCRUB} mpirun -n {self.ntasks} "
            f"--oversubscribe --allow-run-as-root {LLMAPI_LAUNCH} {argv}"
        )

    def _slurm_payload(self, argv: str) -> str:
        """Wrap ``argv`` the way this task count requires.

        Both branches end up inside ``bash -c`` because both need a shell: one
        to scrub the environment before exec, the other so the wrapper and its
        task read as a single command to ``srun``.
        """
        if self.needs_llmapi_launch:
            return f"bash -c {shlex.quote(f'{LLMAPI_LAUNCH} {argv}')}"
        return self._scrub_and_exec(argv)

    def command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return ``argv`` prefixed for this environment.

        ``argv`` is a command line, not an interpreter invocation — the
        caller decides whether it starts with :attr:`python`, ``pytest``,
        ``trtllm-eval`` or anything else.

        ``single_process`` is for the work that is **not** engine work: a
        read-only probe that imports the build, enumerates it and exits.
        Such a command wants exactly one process whatever the run's world
        size, and the rule for one process already exists here — it is the
        ``ntasks == 1`` branch, which uses no launcher and scrubs the MPI
        environment before exec. Asking for it by name is what keeps launcher
        assembly inside this module: a caller that built its own
        single-process line would be a second, divergent statement of a rule
        this module exists to state once.

        Getting this wrong is not theoretical. The inventory probe went out
        at world size 4 as ``mpirun -n 4 ... trtllm-llmapi-launch``, which is
        right for a gate and wrong for a probe, and the plan phase was handed
        an empty survey.
        """
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        if single_process and self.ntasks > 1:
            # Re-asked at one task rather than answered by a parallel set of
            # rules. `--ntasks`, `--ntasks-per-node`, `--gres`, `--mpi` and
            # the payload wrapper all follow from the task count, and a
            # launch that shrinks only the payload still starts one copy of
            # the probe per rank.
            return replace(self, world_size=1).command(argv, single_process=True)
        if self.kind == "slurm":
            return f"{self._slurm_prefix()} {self._slurm_payload(argv)}"
        if self.needs_llmapi_launch:
            argv = self._mpirun_launch(argv)
        elif single_process:
            # There is no `srun` here to carry the scrub `_slurm_payload`
            # applies, and the process still inherits a step's PMI variables:
            # a spool script runs inside the session's own allocation.
            argv = self._scrub_and_exec(argv)
        prefix = self.prefix
        return f"{prefix} {argv}".strip()

    def python_command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return a Python invocation of ``argv`` prefixed for this environment."""
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        return self.command(f"{shlex.quote(self.python)} {argv}", single_process=single_process)

    def describe(self) -> str:
        """The block injected verbatim into every agent's prompt."""
        if self.kind == "session_spool":
            return self._describe_spool()
        if self.kind == "slurm_broker":
            return self._describe_broker()
        device_line = (
            ", ".join(str(d) for d in self.devices) if self.devices else "(allocated by Slurm)"
        )
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
        """The prompt block for a long-lived compute session watching a spool.

        This is the kind to prefer wherever it is available, and the reason
        is not convenience. Every other kind puts a scheduler between a
        command and its result: an image import (26 GB, 4-8 minutes) per
        command, a queue wait per command, and — under ``slurm_broker`` — a
        transport only an MCP client can reach, which the orchestrator is
        not. A spool has none of that. One allocation is acquired once, and
        work is a **file**, so anything that can write a file can run GPU
        work: the orchestrator, an agent, or a human.

        The session's own scheduler allocates devices, enforces limits and
        removes the allocation's MPI context from every job, so the block
        tells the agent how to *ask* for devices rather than how to pick them.
        """
        from .substrate import SCRIPTS_DIR

        spool = self.spool or "<spool>"
        prefix = self.prefix or "(no extra environment)"
        submit = SCRIPTS_DIR / "spool_submit.sh"
        control = SCRIPTS_DIR / "spoolctl.sh"
        return (
            "## Running GPU work\n"
            "\n"
            "A compute session is already running and its scheduler watches a spool "
            "directory. You do **not** submit jobs, allocate nodes, or touch Slurm — "
            "`srun`/`sbatch` do not even work from here. Work is a file.\n"
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
            f"export {prefix.replace(' ', chr(10) + 'export ')}\n"
            "```\n"
            "\n"
            f"The Python interpreter is `{self.python}`. World size {self.world_size}.\n"
            "\n"
            + self._describe_sessions()
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
                if self.needs_llmapi_launch
                else ""
            )
            + "**Do not construct a launcher command, do not probe for one, and do not call a "
            "test-runner skill.** The session was resolved before you started and it is not "
            "yours to rediscover. If this protocol does not fit what you need to run, that "
            "is a **symptom to report in your final message** — not a problem to solve.\n"
        )

    def _describe_sessions(self) -> str:
        if len(self.sessions) < 2:
            return ""
        from .substrate.sessions import SessionSpec, describe_sessions

        return describe_sessions(
            [SessionSpec(name=n, spool=s, role=r) for n, s, r in self.sessions]
        )

    def _describe_broker(self) -> str:
        """The prompt block for a cluster reachable only through the MCP broker."""
        in_container = self.prefix or "(no extra environment)"
        return (
            "## Running GPU work\n"
            "\n"
            "This cluster's Slurm CLI is **unreachable** from inside this session: "
            "`srun`, `sbatch` and `squeue` fail to contact the controller. Do not try "
            "them, and do not conclude from their failure that the cluster is down.\n"
            "\n"
            "Every GPU command in this run is launched exactly one way — this call, with "
            "only the quoted command changed:\n"
            "\n"
            "```python\n"
            f"{self.broker_submit()}\n"
            "```\n"
            "\n"
            "It returns a `job_id` immediately and does **not** wait. Then:\n"
            "\n"
            "1. Poll `mcp__slurm-broker__slurm_job_status(job_id)` until `JobState` "
            "leaves `PENDING`/`RUNNING`.\n"
            "2. Read the job's log: the `StdOut=` path that status reports, which is "
            "`slurm-<job_id>.out` in your home directory unless the submission set a "
            "`workdir`. That file is the result; your recollection of it is not.\n"
            "\n"
            "**Never use the blocking `slurm_run`.** The MCP layer aborts a tool call "
            "after 300 seconds of silence, and every job here is longer than that — so "
            "`slurm_run` reliably reports a failure for a job that is running fine, and "
            "leaves it running with its output unreachable.\n"
            "\n"
            f"Inside the container, prefix your command with: `{in_container}`\n"
            f"The Python interpreter is `{self.python}`. World size {self.world_size}; "
            f"the allocation carries {self.gpus_per_node} GPU(s) per node.\n"
            "\n"
            "**Do not build a different launcher, do not probe for one, and do not call a "
            "test-runner skill.** If this template does not fit what you need to run, that "
            "is a **symptom to report in your final message** — not a problem to solve.\n"
        )


def image_import_warning(env: "RunEnvironment") -> str | None:
    """Warn when every job will re-import a registry image instead of mounting a file.

    ``container_image`` is passed to pyxis unchanged, and pyxis accepts either
    a **registry reference** or a **local squashfs path**. The difference is
    the single biggest lever on how parallel a run can be.

    A registry reference is fetched and unpacked *per allocation*. Measured on
    the project this was written for: **5-7 minutes** from job start to a
    usable container, for a 26 GB image. Measured again on this run, on a
    loaded cluster and three times in a row: **25-35 minutes**. That second
    number is what makes an isolated job per gate look unaffordable, and it is
    the reason work piles onto one long-lived shared session — which has **one
    failure domain**, so a leaked server, a process-group kill and a PMIx
    collision each took down every other job and the orchestrator with it.

    A squashfs on shared storage mounts in seconds, so an isolated job per
    workload stops being an expensive choice. The run that measured the 5-7
    minutes executed **443 spool jobs**; paying the import on each would have
    cost roughly 37 hours, and at this run's 25-35 minutes it would have been
    several days.

    Returns ``None`` when the image is already a local file, so a correctly
    configured run is silent. An absolute path that does not exist gets its
    own message: pyxis fails to start rather than falling back to a registry,
    so the two are different faults with different fixes.
    """
    image = (env.container_image or "").strip()
    if not image or env.kind not in ("slurm", "slurm_broker", "session_spool"):
        return None
    if image.startswith("/") and Path(image).is_file():
        return None
    if image.startswith("/"):
        return (
            f"container_image points at {image}, which does not exist. pyxis will "
            f"fail to start rather than fall back to a registry."
        )
    return (
        f"container_image is a registry reference ({image.split('/')[-1]}), so every "
        f"allocation re-imports it -- measured at 5-7 minutes for a 26 GB image, and "
        f"25-35 minutes on this cluster under load. Import it once to shared storage "
        f"and point at the file instead:\n"
        f"    enroot import -o <shared>/trtllm.sqsh docker://{image}\n"
        f"then set `environment.container_image` to that path. This is what makes "
        f"an isolated job per gate affordable rather than a trade against startup "
        f"cost -- see `environment.workloads` in task.example.yaml."
    )


def shares_allocation_with_spool(env: "RunEnvironment") -> str | None:
    """Warn when the orchestrator lives inside the allocation it dispatches to.

    The orchestrator is a **long-lived, GPU-free** process: it blocks for hours
    on a role turn and never touches a card. The compute session is a
    short-lived, GPU-heavy one that gets preempted, OOM-killed and torn down.
    Putting the first inside the second couples their lifetimes for no benefit
    at all.

    Measured three times in one run on the project this was written for: the
    session died and took the orchestrator with it, and each time an agent turn
    had already been answered on disk and the answer was thrown away because
    nothing was left reading it — the most expensive of them a twelve-minute
    audit. This run hit it twice more, and neither was a scheduler event: once
    a disagg probe's worker teardown took the step down, and once an agent's
    ``pkill -9`` caught the watcher — which *is* the allocation's main process,
    so killing it ends the allocation. Both took every queued job with them.

    The evidence is ``watcher.jobid`` in the spool, written every pass by the
    session's scheduler (``substrate/scheduler.py``). It is a **separate file
    from** ``watcher.heartbeat`` on purpose: submitters parse the heartbeat as
    ``alive=<timestamp>``, so anything else on that line breaks every one of
    them.

    Returns a message when both job ids are known **and equal**, and ``None``
    otherwise. Unknown is not a warning: an orchestrator launched in its own
    job also has ``SLURM_JOB_ID`` set, and crying wolf at the correct
    configuration is how a warning gets ignored.
    """
    if env.kind != "session_spool" or not env.spool:
        return None
    # A job the spool scheduler started has no SLURM_* variables (they are
    # scrubbed) and carries the session's job id under this name instead.
    mine = os.environ.get("SLURM_JOB_ID") or os.environ.get("MV2FLOW_SPOOL_SESSION_JOBID")
    if not mine:
        return None
    marker = Path(env.spool) / "watcher.jobid"
    try:
        theirs = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not theirs or theirs == "unknown" or theirs != mine:
        return None
    return (
        f"the orchestrator is running inside Slurm job {mine}, which is also the "
        f"compute session watching {env.spool}. They will die together, and this "
        f"process needs no GPU: launch it in its own allocation "
        f"(`--gpus-per-node=0`) so a preempted session costs you the work in "
        f"flight and not the run."
    )


#: Workloads that may carry their own launch substrate. Closed on purpose: a
#: name the orchestrator does not dispatch under is a typo that would silently
#: never apply, and nothing downstream would say so. Enforced by
#: :func:`~.task_schema._validate_workloads`.
WORKLOAD_KINDS: tuple[str, ...] = ("gate",)


def resolve_environment(
    task_data: Mapping[str, Any],
    *,
    devices: Sequence[int] = (),
    world_size: int = 1,
    workload: str | None = None,
) -> RunEnvironment:
    """Build a :class:`RunEnvironment` from a validated task spec.

    ``devices`` comes from preflight's claim, not from the spec: which
    cards are usable is a property of the host at this moment, and the
    spec cannot know it.

    ``workload`` selects an optional override from ``environment.workloads``,
    because **the right substrate depends on how long the job runs**. A shared
    session imports a 26 GB container once and then costs nothing per job,
    which is the correct trade for a 1-minute entry test; one onboarding ran
    443 of them, and paying a 5-7 minute import each would have cost 37 hours.
    It is the wrong trade for a 10-15 minute gate, where the import is 30-50%
    overhead and **isolation is the thing you actually want** — a shared
    session has one failure domain, so a gate that leaks a server, kills a
    process group or collides on PMIx takes every other job and the
    orchestrator with it. See :func:`shares_allocation_with_spool` for the
    five times that happened across two runs.

    Absent an override this returns exactly what it always did.
    """
    block = dict(task_data.get("environment") or {})
    if workload:
        override = (block.get("workloads") or {}).get(workload) or {}
        # `env` merges key-by-key; everything else replaces. A workload that
        # needs a different substrate almost never needs a different *whole*
        # environment, and declaring mounts and accounts again to change one key
        # is how the two drift apart.
        merged_env = {**(block.get("env") or {}), **(override.get("env") or {})}
        block = {**block, **{k: v for k, v in override.items() if k != "env"}}
        block["env"] = merged_env
    env = {str(k): str(v) for k, v in (block.get("env") or {}).items()}
    from .substrate.sessions import sessions_from_task

    declared = sessions_from_task({"environment": block}) if block.get("sessions") else ()
    return RunEnvironment(
        kind=str(block.get("kind", "local")),
        python=str(block.get("python", "python3")),
        env=env,
        devices=tuple(int(d) for d in devices),
        world_size=max(1, int(world_size)),
        partition=block.get("partition"),
        container_image=block.get("container_image"),
        account=block.get("account"),
        mounts=tuple(str(m) for m in (block.get("mounts") or ())),
        time_limit=block.get("time_limit"),
        gpus_per_node=int(block.get("gpus_per_node") or 0),
        container_name=block.get("container_name"),
        mpi=str(block.get("mpi") or MPI_AUTO),
        spool=block.get("spool"),
        sessions=tuple((s.name, s.spool, s.role) for s in declared),
    )


__all__ = [
    "LLMAPI_LAUNCH",
    "MERGED_ENV_VARS",
    "MPI_AUTO",
    "MPI_NONE",
    "MPI_PMIX",
    "MULTI_TASK_SLURM_SCRUB",
    "MULTI_TASK_SLURM_VARS",
    "RunEnvironment",
    "SINGLE_TASK_MPI_SCRUB",
    "WORKLOAD_KINDS",
    "image_import_warning",
    "resolve_environment",
    "shares_allocation_with_spool",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How GPU work is launched — resolved once, then handed to every agent.

The ``environment`` block in ``task.yaml`` is resolved here into a
:class:`RunEnvironment`, whose ``prefix`` is a literal string that every
GPU command is prepended with. That string — not a policy, not a skill, not
a decision tree — is what the agent prompts carry.

The orchestrator resolves the launch once, in Python, before any agent
exists. Routing every test through a chain of skills that classify the
request, pick a container image, check the environment and build an
``srun`` line would be four hops, each one an LLM decision, each one able to
fail in a way that reads as "the environment is broken" rather than "this
hop guessed wrong", and agents would spend their budget rediscovering how
to submit a job. An agent that finds the prefix does not fit its need is
instructed to *report that as a symptom* rather than to work around it.

Two kinds exist: ``local`` (this host, the devices preflight claimed) and
``session_spool`` (a long-lived compute session whose scheduler watches a
spool directory). Nothing here talks to a scheduler. Building the command is
all this module does; running it belongs to :mod:`execution`.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .substrate import SPOOLCTL_PATH
from .substrate.sessions import (
    MERGED_ENV_VARS,
    SessionSpec,
    describe_sessions,
    prepend_to_inherited,
    sessions_from_task,
)

#: The wrapper that makes a pre-launched MPI world work. Rank 0 runs the
#: leader plus the real task; every other rank runs a worker and waits.
LLMAPI_LAUNCH = "trtllm-llmapi-launch"

#: Set at one rank (:attr:`RunEnvironment.launch_env`): the LLM API then runs
#: its worker in its own process instead of spawning it over MPI.
SINGLE_PROCESS_WORKER_ENV = "TLLM_WORKER_USE_SINGLE_PROCESS"

# ---------------------------------------------------------------------------
# The launch rules below are TensorRT-LLM's, not this workflow's. They are a
# function of the task count and nothing else:
#
#   ntasks == 1  ->  scrub the MPI environment, no wrapper
#   ntasks  > 1  ->  mpirun creates the ranks, the payload runs under the launcher
#
# Why one task is the special case: `trtllm-bench` and the LLM API import
# mpi4py at module load, which auto-initializes MPI whenever PMI variables are
# present. Inside a PMIx-managed step they always are --- so a single-task job
# initializes MPI for no reason and then dies trying to spawn a worker
# dynamically, which is what pyxis forbids. Clearing the variables lets the
# process skip MPI init entirely.
#
# `SLURM_PROCID` and `SLURM_LOCALID` survive the scrub: they identify the rank
# rather than bootstrap MPI, and code downstream reads them.
# `MULTI_TASK_SLURM_SCRUB` below is the case where they mislead, and the two
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
# What four ranks under such an `mpirun` see:
#
#                       SLURM_PROCID   OMPI_COMM_WORLD_RANK   SLURM_NTASKS
#   as-is               0, 0, 0, 0     0, 1, 2, 3             1
#   Slurm vars scrubbed unset          0, 1, 2, 3             unset
#
# Unscrubbed, every rank reads `mpi_rank=0` and becomes a leader, and every
# rank also reads `mpi_world_size=1`: four `mgmn_leader_node` processes, each
# waiting to be joined by workers that never come. No error, no traceback, no
# exit --- the run hangs forever. MPI itself is not at fault; a bare `mpi4py`
# probe under the identical `mpirun` line reports `rank 0..3 of 4`.
#
# Scrubbing lets the launcher fall through to the Open MPI variables it should
# be reading here.
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


@dataclass(frozen=True)
class RunEnvironment:
    """A resolved, immutable description of how to launch GPU work."""

    kind: str
    python: str
    env: dict[str, str] = field(default_factory=dict)
    devices: tuple[int, ...] = ()
    world_size: int = 1
    # The image the compute sessions run (``session_spool``); see
    # :func:`image_import_warning`.
    container_image: str | None = None
    # ``session_spool`` only: the directory a long-lived compute session
    # watches for work.
    spool: str | None = None
    # ``environment.sessions``, as ``(name, spool, role)``: every session an
    # agent may submit to, and what each is for.
    sessions: tuple[tuple[str, str, str], ...] = ()
    # ``perf.dev_spools``: the tune campaign's own sessions. A gate borrows one
    # only while it stands idle (``execution``); no prompt names them.
    dev_spools: tuple[str, ...] = ()

    @property
    def ntasks(self) -> int:
        """One task per rank. This is the number the whole launch turns on."""
        return max(1, self.world_size)

    @property
    def needs_llmapi_launch(self) -> bool:
        """Whether the payload must be wrapped in ``trtllm-llmapi-launch``."""
        return self.ntasks > 1

    @property
    def launch_env(self) -> dict[str, str]:
        """The variables every command of this environment runs with.

        ``env``, and at one rank ``TLLM_WORKER_USE_SINGLE_PROCESS=1`` unless
        ``env`` says otherwise: a one-rank engine has no launcher to create its
        worker, so the LLM API would spawn it with ``MPI_Comm_spawn``, which a
        container runtime such as pyxis forbids (``MPI_ERR_SPAWN``). Derived
        rather than stored, so a :func:`dataclasses.replace` to another world
        size and the prompt's prefix (:meth:`describe`) say the same.
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
        parts.extend(self._env_assignments())
        return " ".join(parts)

    def _env_assignments(self) -> list[str]:
        """``KEY=value`` pairs in a stable order, shell-quoted.

        A variable named in :data:`MERGED_ENV_VARS` is emitted as a prepend
        to whatever the host already exports rather than as a replacement;
        see that constant for the failure it prevents.
        """
        parts: list[str] = []
        for key, text in sorted(self.launch_env.items()):
            if key in MERGED_ENV_VARS:
                parts.append(f"{key}={prepend_to_inherited(key, text)}")
            else:
                parts.append(f"{key}={shlex.quote(text)}")
        return parts

    def _scrub_and_exec(self, argv: str) -> str:
        """``argv`` behind a scrub of the MPI environment, in this process.

        See :data:`SINGLE_TASK_MPI_SCRUB` for why a lone process must not
        inherit the PMI variables.
        """
        return f"bash -c {shlex.quote(f'{SINGLE_TASK_MPI_SCRUB}; exec {argv}')}"

    def _mpirun_launch(self, argv: str) -> str:
        """``argv`` behind an ``mpirun`` that creates the ranks here.

        Inside a spool script or on a local box ``mpirun`` creates the ranks
        and the wrapper assigns leader/worker. Without it the LLM API finds a
        world size of 1, concludes it must create the other ranks itself, and
        dies with ``MPI_ERR_SPAWN``.

        Two flags here are about running ``mpirun`` **inside someone else's
        allocation**:

        - :data:`MULTI_TASK_SLURM_SCRUB` — the allocation's task variables
          describe the *step*, not the MPI world, and the launcher reads them
          first. See that constant.
        - ``--oversubscribe`` — Open MPI takes its slot count from the resource
          manager, and a spool session is submitted as **one task** that owns
          the node (e.g. ``NumTasks=1`` against ``NumCPUs=144`` and
          ``gres/gpu:4``). So ``-n 4`` asks for four slots against one and dies
          before any Python runs, with *"There are not enough slots available
          in the system"*. The allocation really does hold all of the node's
          CPUs and GPUs, so the slot count is a bookkeeping guard rather than a
          real constraint, and oversubscribing it costs nothing.

        Raising the session's own ``ntasks_per_node`` instead would be
        wrong: it gives MPI its slots and simultaneously starts four
        copies of the spool watcher, which then race for every piece of work.
        The session is one process by design.
        """
        return (
            f"{MULTI_TASK_SLURM_SCRUB} mpirun -n {self.ntasks} "
            f"--oversubscribe --allow-run-as-root {LLMAPI_LAUNCH} {argv}"
        )

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

        Getting this wrong fails quietly: an inventory probe launched at
        world size 4 as ``mpirun -n 4 ... trtllm-llmapi-launch`` (right for a
        gate, wrong for a probe) hands the plan phase an empty survey.
        """
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        if single_process and self.ntasks > 1:
            # Re-asked at one task rather than answered by a parallel set of
            # rules: the launcher follows from the task count, and a launch
            # that shrinks only the payload still starts one copy of the
            # probe per rank.
            return replace(self, world_size=1).command(argv, single_process=True)
        if self.needs_llmapi_launch:
            argv = self._mpirun_launch(argv)
        elif single_process:
            # The process inherits a step's PMI variables: a spool script runs
            # inside the session's own allocation.
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
        """The prompt block for a long-lived compute session watching a spool.

        A job per scheduler submission would pay an image import (26 GB, 4-8
        minutes) and a queue wait per command, and where the Slurm CLI is
        unreachable from this host it cannot be submitted at all. A spool has
        none of that. One allocation is acquired once, and work is a
        **file**, so anything that can write a file can run GPU work: the
        orchestrator, an agent, or a human.

        The session's own scheduler allocates devices, enforces limits and
        removes the allocation's MPI context from every job, so the block
        tells the agent how to *ask* for devices rather than how to pick them.
        """
        spool = self.spool or "<spool>"
        prefix = self.prefix or "(no extra environment)"
        submit = f"python3 {SPOOLCTL_PATH} submit"
        control = f"python3 {SPOOLCTL_PATH}"
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
            "- A spool that will not start the job soon (its session is handing over, has "
            "too little time left, or its watcher is gone) prints a `spoolctl: warning:` "
            "naming the sessions of the same kind that can take it now; `--pick` sends the "
            "job to whichever can start it soonest and prints which, so read that spool.\n"
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

        return describe_sessions(
            [SessionSpec(name=n, spool=s, role=r) for n, s, r in self.sessions]
        )


def image_import_warning(env: "RunEnvironment") -> str | None:
    """Warn when every job will re-import a registry image instead of mounting a file.

    ``container_image`` is passed to pyxis unchanged, and pyxis accepts either
    a **registry reference** or a **local squashfs path**. The difference is
    the single biggest lever on how parallel a run can be.

    A registry reference is fetched and unpacked *per allocation*: **5-7
    minutes** from job start to a usable container for a 26 GB image, and
    **25-35 minutes** on a loaded cluster. That cost makes an isolated job per
    gate look unaffordable and pushes work onto one long-lived shared session
    — which has **one failure domain**, so a leaked server, a process-group
    kill or a PMIx collision takes down every other job and the orchestrator
    with it.

    A squashfs on shared storage mounts in seconds, so an isolated job per
    workload stops being an expensive choice. An onboarding runs hundreds of
    spool jobs; paying the import on each costs tens of hours (443 jobs at 5-7
    minutes is roughly 37 hours).

    Returns ``None`` when the image is already a local file, so a correctly
    configured run is silent. An absolute path that does not exist gets its
    own message: pyxis fails to start rather than falling back to a registry,
    so the two are different faults with different fixes.
    """
    image = (env.container_image or "").strip()
    if not image or env.kind != "session_spool":
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
        f"then set `environment.container_image` to that path."
    )


def shares_allocation_with_spool(env: "RunEnvironment") -> str | None:
    """Warn when the orchestrator lives inside the allocation it dispatches to.

    The orchestrator is a **long-lived, GPU-free** process: it blocks for hours
    on a role turn and never touches a card. The compute session is a
    short-lived, GPU-heavy one that gets preempted, OOM-killed and torn down.
    Putting the first inside the second couples their lifetimes for no benefit
    at all.

    When the session dies it takes the orchestrator with it, and an agent turn
    already answered on disk is thrown away because nothing is left reading
    it. Not every such death is a scheduler event: a disagg probe's worker
    teardown can take the step down, and an agent's ``pkill -9`` can catch the
    watcher — which *is* the allocation's main process, so killing it ends the
    allocation. Each takes every queued job with it.

    The evidence is ``watcher.jobid`` in the spool, written every pass by the
    session's scheduler (``infra/substrate/scheduler.py``). It is a **separate file
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


def scheduler_owns_devices(kind: str) -> bool:
    """Whether a scheduler, not this process, decides the devices and the host.

    Only ``local`` work runs on the orchestrator's own machine, on the
    devices preflight claimed for it. Under ``session_spool`` the session's
    scheduler hands devices out per job, and the job may run on a host other
    than this one. Three rules follow, and callers ask this rather
    than comparing kinds so that they cannot disagree:

    * preflight claims no devices and probes none of this host's cards;
    * no prompt names a device, because the scheduler exports
      ``CUDA_VISIBLE_DEVICES`` itself and an agent that sets it masks the
      devices it was given;
    * a file a job reads lives on the shared filesystem, because this
      process's temp dir is not on the host that reads it.

    Phrased as "not local" rather than as a list of remote kinds on purpose:
    a kind added later is remote until someone says otherwise, and that is
    the safe default.
    """
    return kind != "local"


def resolve_environment(
    task_data: Mapping[str, Any],
    *,
    devices: Sequence[int] = (),
    world_size: int = 1,
) -> RunEnvironment:
    """Build a :class:`RunEnvironment` from a validated task spec.

    ``devices`` comes from preflight's claim, not from the spec: which
    cards are usable is a property of the host at this moment, and the
    spec cannot know it.
    """
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
        sessions=tuple((s.name, s.spool, s.role) for s in declared),
        dev_spools=_dev_spools(task_data.get("perf")),
    )


def _dev_spools(perf: Any) -> tuple[str, ...]:
    """``perf.dev_spools``, less any spool that also measures (``perf.spools``).

    Only a list declared as such: the campaign's fallback to its measuring
    spools is not one, and a gate beside a measurement would skew it.
    """
    if not isinstance(perf, Mapping):
        return ()
    measuring = {Path(str(s)) for s in perf.get("spools") or ()}
    return tuple(
        str(s) for s in perf.get("dev_spools") or () if s and Path(str(s)) not in measuring
    )


__all__ = [
    "LLMAPI_LAUNCH",
    "MERGED_ENV_VARS",
    "MULTI_TASK_SLURM_SCRUB",
    "MULTI_TASK_SLURM_VARS",
    "RunEnvironment",
    "SINGLE_TASK_MPI_SCRUB",
    "image_import_warning",
    "resolve_environment",
    "scheduler_owns_devices",
    "shares_allocation_with_spool",
]

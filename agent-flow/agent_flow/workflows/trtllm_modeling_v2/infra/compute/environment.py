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
from . import SPOOLCTL_PATH
from .scheduler import SpoolPaths
from .server_runtime import MULTI_TASK_SLURM_VARS, rank_launch
from .sessions import (
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
# they name the rank rather than bootstrap MPI, and code downstream reads them.
SINGLE_TASK_MPI_SCRUB = (
    "for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do "
    '[ "$v" != SLURM_PROCID ] && [ "$v" != SLURM_LOCALID ] && unset "$v"; '
    "done"
)

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
    # The image compute sessions run (``session_spool``); see :func:`image_import_warning`.
    container_image: str | None = None
    # ``session_spool`` only: the spool a long-lived compute session watches.
    spool: str | None = None
    # Every declared session an agent may submit to.
    sessions: tuple[SessionSpec, ...] = ()
    # ``perf.dev_spools``: the sessions the tuners submit to; no other prompt names them.
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
        return f"bash -c {shlex.quote(f'{SINGLE_TASK_MPI_SCRUB}; exec {argv}')}"

    def _mpirun_launch(self, argv: str) -> str:
        """``argv`` behind an ``mpirun`` that creates the ranks here.

        Without it the LLM API finds a world size of 1, tries to create the other ranks itself,
        and dies with ``MPI_ERR_SPAWN``. ``--oversubscribe``: Open MPI takes its slot count from
        Slurm, and a spool session is **one task** that owns the node, so ``-n 4`` asks for four
        slots against one and dies before any Python runs. Raising the session's
        ``ntasks_per_node`` instead would start four spool watchers racing for every job.
        """
        return f"{shlex.join(rank_launch(self.ntasks, LLMAPI_LAUNCH))} {argv}"

    def command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return the command line ``argv`` prefixed for this environment.

        ``single_process`` is for work that is **not** engine work, such as a probe that imports
        the build and exits: one process whatever the world size, no launcher, the MPI
        environment scrubbed.
        """
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        if single_process and self.ntasks > 1:
            # Re-asked at one task: the launcher follows from the task count, and shrinking only
            # the payload would still start one copy of the probe per rank.
            return replace(self, world_size=1).command(argv, single_process=True)
        if self.needs_llmapi_launch:
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
                if self.needs_llmapi_launch
                else ""
            )
            + "**Do not construct a launcher command, do not probe for one, and do not call a "
            "test-runner skill.** The session was resolved before you started and it is not "
            "yours to rediscover. If this protocol does not fit what you need to run, that "
            "is a **symptom to report in your final message** — not a problem to solve.\n"
        )


def image_import_warning(env: "RunEnvironment") -> str | None:
    """Warn when every job will re-import a registry image instead of mounting a file.

    A missing absolute path gets its own message: pyxis then fails to start rather than fall back
    to a registry, a different fault with a different fix.
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
                f"mid-run; point it at scratch or lustre instead."
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
    "SINGLE_TASK_MPI_SCRUB",
    "image_import_warning",
    "resolve_environment",
    "scheduler_owns_devices",
    "shares_allocation_with_spool",
]

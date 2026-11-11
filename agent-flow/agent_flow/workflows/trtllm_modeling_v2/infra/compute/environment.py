# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How GPU work is launched — resolved once, then handed to every agent.

The ``environment`` block of ``task.yaml`` becomes a :class:`RunEnvironment`, whose ``prefix``
every GPU command is prepended with; the agent prompts carry that string. The one kind is
``local``: commands run on this host's claimed devices. This module only builds commands;
:mod:`.executor` runs them.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .server_runtime import MULTI_TASK_SLURM_VARS, rank_launch

# Variables whose declared value EXTENDS the inherited one. A plain `PATH: <venv>/bin:/usr/bin`
# drops `/usr/local/mpi/bin`, and the next launcher line dies `mpirun: command not found` (rc
# 127). `PYTHONPATH` already holds the gates' rank hook (`checks/check_runtime.RANK_HOOK_DIR`) and the
# tree a check measures; unset, it must not leave a trailing `:`, which
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


def single_task_scrub(keep: Sequence[str] = (), *, rank: bool = True) -> str:
    """The shell loop that unsets the MPI bootstrap environment, but ``keep`` and the settings.

    ``rank``: the variables that name this process's rank stay. Behind a launch prefix they
    are the orchestrator's, not the launched program's, and go with the rest.
    """
    named = ("SLURM_PROCID", "SLURM_LOCALID") if rank else ()
    kept = [*named, *MPI_SETTINGS_KEPT, *sorted(set(keep))]
    tests = " && ".join(f'[ "$v" != {name} ]' for name in dict.fromkeys(kept))
    return f'for v in ${{!PMI@}} ${{!PMIX@}} ${{!MPI@}} ${{!OMPI@}} ${{!SLURM@}}; do {tests} && unset "$v"; done'


# Above one task, under `mpirun`, the misleading variables are the Slurm task ones:
# `trtllm-llmapi-launch` reads `SLURM_PROCID` before `OMPI_COMM_WORLD_RANK` for its rank and
# `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE` for `mpi_world_size`. Right under `srun`, wrong
# under an `mpirun` inside a one-task allocation, which is where an orchestrator started in an
# allocation runs.
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
        declared = [k for k in self.env if str(k).startswith(_DECLARED_MPI_PREFIXES)]
        return f"bash -c {shlex.quote(f'{single_task_scrub(declared)}; exec {argv}')}"

    def _mpirun_launch(self, argv: str) -> str:
        """``argv`` behind an ``mpirun`` that creates the ranks here.

        Without it the LLM API finds a world size of 1, tries to create the other ranks itself,
        and dies with ``MPI_ERR_SPAWN``. ``--oversubscribe``: Open MPI takes its slot count from
        Slurm, and inside a one-task allocation ``-n 4`` asks for four slots against one and
        dies before any Python runs.
        """
        launch = f"{shlex.join(rank_launch(self.ntasks, LLMAPI_LAUNCH))} {argv}"
        # The launcher is installed beside the interpreter; the inherited PATH need not hold it
        # (`mpirun` reports `trtllm-llmapi-launch` not found), so its directory goes first.
        if os.path.isabs(self.python):
            return f'PATH={shlex.quote(os.path.dirname(self.python))}:"$PATH" {launch}'
        return launch

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
            # The process inherits a step's PMI variables when the orchestrator runs inside
            # an allocation.
            argv = self._scrub_and_exec(argv)
        prefix = self.prefix
        return f"{prefix} {argv}".strip()

    def python_command(self, argv: str | Sequence[str], *, single_process: bool = False) -> str:
        """Return a Python invocation of ``argv`` prefixed for this environment."""
        if not isinstance(argv, str):
            argv = " ".join(shlex.quote(str(part)) for part in argv)
        return self.command(f"{shlex.quote(self.python)} {argv}", single_process=single_process)


def resolve_environment(
    task_data: Mapping[str, Any],
    *,
    devices: Sequence[int] = (),
    world_size: int = 1,
) -> RunEnvironment:
    """Build a :class:`RunEnvironment` from a validated task spec and preflight's ``devices``."""
    block = dict(task_data.get("environment") or {})
    env = {str(k): str(v) for k, v in (block.get("env") or {}).items()}

    return RunEnvironment(
        kind=str(block.get("kind", "local")),
        python=str(block.get("python", "python3")),
        env=env,
        devices=tuple(int(d) for d in devices),
        world_size=max(1, int(world_size)),
        container_image=block.get("container_image"),
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


def check_tmpdir(env: Mapping[str, str]) -> None:
    """Assert that a ``TMPDIR`` or ``CLAUDE_CODE_TMPDIR`` the spec declares is short and not ``/tmp``.

    The length bound (:data:`MAX_TMPDIR_LEN`, a unix socket's path) applies to ``TMPDIR``
    alone: ``CLAUDE_CODE_TMPDIR`` binds no socket. ``env`` is the environment declared in
    task.yaml; one it does not declare is the system's, which preflight probes.
    """
    for name in ("TMPDIR", "CLAUDE_CODE_TMPDIR"):
        value = env.get(name)
        if not value:
            continue
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

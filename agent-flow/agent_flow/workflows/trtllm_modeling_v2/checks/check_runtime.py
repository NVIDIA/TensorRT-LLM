# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-side check launch protocol, driver workloads and tuning isolation."""

from __future__ import annotations

import os
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from ..infra.compute.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ..infra.compute.environment import RunEnvironment

# A boot gate that needs longer than this is an accuracy run with no anchor.
BOOT_TIMEOUT_S = 1800


MODELING_V2_ENV = "TRTLLM_MODELING_V2"


MODE_REQUIRE = "require"


#: Turns on each target's first-forward step-contract check
#: (each target's ``_check_step_contract``, which calls its ``step_args``):
#: a renamed metadata field then fails the gate loudly instead of computing a
#: wrong answer. Off in a served engine; every gate run sets it, as
#: the in-tree modeling_v2 accuracy tests do.
STEP_CONTRACT_ENV = "TRTLLM_MODELING_V2_VALIDATE"


ShellRunner = Callable[[str, Mapping[str, str] | None, int], CommandResult]


_LOCAL_EXECUTOR = LocalCommandExecutor()


def default_shell_runner(
    command: str,
    env: Mapping[str, str] | None = None,
    timeout: int = BOOT_TIMEOUT_S,
) -> CommandResult:
    """Run a shell command on this host, including the task's launch prefix."""
    return _LOCAL_EXECUTOR.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))


#: Set in every engine gate's environment, so each rank's worker prints its record.
RANK_FACTS_ENV = "MV2FLOW_RANK_FACTS"


#: Put first on every engine gate's ``PYTHONPATH``: its ``sitecustomize`` has
#: each rank's worker print the record.
RANK_HOOK_DIR = Path(__file__).resolve().parent / "harness" / "rankhook"


def with_mode(environment: RunEnvironment, mode: str) -> RunEnvironment:
    """Return a copy of ``environment`` with the modeling_v2 switch set.

    Exported into the launch prefix, not into this process: worker ranks
    receive the environment as it stood when MPI initialized. The same prefix
    puts :data:`RANK_HOOK_DIR` first on ``PYTHONPATH`` and sets
    :data:`RANK_FACTS_ENV`, so every rank's worker prints what its engine runs.
    """
    python_path = [str(RANK_HOOK_DIR)]
    if environment.env.get("PYTHONPATH"):
        python_path.append(str(environment.env["PYTHONPATH"]))
    return replace(
        environment,
        env={
            **environment.env,
            MODELING_V2_ENV: mode,
            STEP_CONTRACT_ENV: "1",
            RANK_FACTS_ENV: "1",
            "PYTHONPATH": os.pathsep.join(python_path),
        },
    )


def gate_workdir(workdir: str | Path) -> Path:
    """``workdir``, made: where a gate writes the files its job reads.

    On the shared filesystem when the job runs on another host; the workflow
    passes ``<workspace>/gates/<unit>``.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


TRITON_CACHE_DIR_ENV = "TRITON_CACHE_DIR"


#: The run's own Triton cache, under its workspace, when the task names none.
RUN_TRITON_CACHE_DIRNAME = "triton-cache"


def default_triton_cache(task_data: dict[str, Any], workspace: str | Path) -> None:
    """Point ``environment.env``'s ``TRITON_CACHE_DIR`` at the run's workspace, unless set.

    Triton otherwise caches under the user's home, which every run on the
    host shares.
    """
    block = task_data.setdefault("environment", {})
    env = block.get("env")
    if env is None:
        env = block["env"] = {}
    env.setdefault(TRITON_CACHE_DIR_ENV, str(Path(workspace).resolve() / RUN_TRITON_CACHE_DIRNAME))


def in_tree(runner: Callable[..., Any], tree: Path | None) -> Callable[..., Any]:
    """Run every command from inside ``tree``, with it first on ``PYTHONPATH``.

    An MPI worker resolves ``tensorrt_llm`` from its working directory ahead
    of ``PYTHONPATH``, and a script's own directory leads its ``sys.path``, so
    both are set. The tree goes ahead of what the command inherits: a
    session's resolve hook is on it.
    """
    if tree is None:
        return runner
    root = shlex.quote(str(Path(tree).resolve()))
    prefix = f'cd {root} && export PYTHONPATH={root}"${{PYTHONPATH:+:$PYTHONPATH}}" && '

    def run(command: str, env: Mapping[str, str] | None = None, timeout: int = 1800):
        return runner(prefix + command, env, timeout)

    return run


__all__ = [
    "in_tree",
    "BOOT_TIMEOUT_S",
    "MODELING_V2_ENV",
    "MODE_REQUIRE",
    "RANK_FACTS_ENV",
    "RANK_HOOK_DIR",
    "RUN_TRITON_CACHE_DIRNAME",
    "STEP_CONTRACT_ENV",
    "ShellRunner",
    "TRITON_CACHE_DIR_ENV",
    "default_shell_runner",
    "default_triton_cache",
    "gate_workdir",
    "with_mode",
]

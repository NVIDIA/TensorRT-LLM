# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-side check launch protocol, driver workloads and tuning isolation."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.compute.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ..infra.compute.environment import RunEnvironment
from .harness import workload_io

# A boot gate that needs longer than this is an accuracy run with no anchor.
BOOT_TIMEOUT_S = 1800


MODELING_V2_ENV = "TRTLLM_MODELING_V2"


MODE_REQUIRE = "require"


MODE_OFF = "off"


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


# ------------------------------------------------------------ the engine driver
#
# The measured interpreter constructs public LLM/SamplingParams from one JSON
# spec and emits MV2FLOW_RECORD results; rank observations are a separate seam.

#: The driver script, run as a file under the measured tree's interpreter.
ENGINE_DRIVER = Path(__file__).resolve().parent / "harness" / "engine_driver.py"


def gate_workdir(workdir: str | Path) -> Path:
    """``workdir``, made: where a gate writes the files its job reads.

    On the shared filesystem when the job runs on another host; the workflow
    passes ``<workspace>/gates/<unit>``.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


#: The tuning a process does for itself, pinned across the legs of a gate that
#: compares them exactly: the AutoTuner's tactic cache, and every
#: ``@triton.autotune`` kernel's chosen config (cached under
#: ``TRITON_CACHE_AUTOTUNING=1`` in one ``TRITON_CACHE_DIR``).
AUTOTUNER_CACHE_ENV = "TLLM_AUTOTUNER_CACHE_PATH"


TRITON_CACHE_AUTOTUNING_ENV = "TRITON_CACHE_AUTOTUNING"


TRITON_CACHE_DIR_ENV = "TRITON_CACHE_DIR"


#: The file Triton writes one kernel's autotuning result to, in its cache.
TRITON_AUTOTUNE_RESULT_GLOB = "*.autotune.json"


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


def pin_tuning(directory: str | Path) -> dict[str, str]:
    """The environment that makes every leg of one gate run share its tuning choices.

    The AutoTuner and ``@triton.autotune`` choose by timing in each process, so
    two legs can differ in the last bit. Call it once per gate run, before the
    first leg: it deletes the choices a previous run left. Merge the result
    over ``environment.env``, not under it.
    """
    root = Path(directory) / "tuning"
    triton = root / "triton-cache"
    autotuner = root / "autotuner.cache"
    autotuner.unlink(missing_ok=True)
    for result in triton.rglob(TRITON_AUTOTUNE_RESULT_GLOB):
        result.unlink(missing_ok=True)
    triton.mkdir(parents=True, exist_ok=True)
    return {
        AUTOTUNER_CACHE_ENV: str(autotuner),
        TRITON_CACHE_AUTOTUNING_ENV: "1",
        TRITON_CACHE_DIR_ENV: str(triton),
    }


def write_driver_spec(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]] = (),
    dataset: Mapping[str, Any] | None = None,
    max_tokens: int = 64,
    concurrency: int = 1,
    ignore_eos: bool = False,
    reuse_probe: bool = False,
    phases: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write public engine options and the workload to materialize before engine launch."""
    spec = {
        **config,
        "requests": [dict(r) for r in requests],
        "dataset": dict(dataset) if dataset else None,
        "max_tokens": int(max_tokens),
        "concurrency": int(concurrency),
        "ignore_eos": bool(ignore_eos),
        "reuse_probe": bool(reuse_probe),
    }
    if phases:
        spec["phases"] = [dict(phase) for phase in phases]
    if not dataset:
        spec["requests"] = workload_io.normalize_requests(spec)
        spec.pop("dataset")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def engine_driver_command(
    environment: RunEnvironment, spec_path: str | Path, *, script: Path = ENGINE_DRIVER
) -> str:
    """Normalize offline data in one CPU process, then launch the public request entry."""
    path = Path(spec_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    prefix = ""
    if spec.get("dataset"):
        normalized = path.with_suffix(".requests.json")
        normalizer = ENGINE_DRIVER.with_name("workload_io.py")
        prefix = (
            environment.python_command(
                [str(normalizer), str(path), str(normalized)], single_process=True
            )
            + " && "
        )
        path = normalized
    return prefix + environment.python_command([str(script), "--spec", str(path)])


def parse_driver_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_RECORD`` the engine driver printed, in request order."""
    records = workload_io.parse_marked(output, workload_io.RECORD_MARKER)
    return sorted(records, key=lambda r: int(r.get("index", 0)))


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
    "AUTOTUNER_CACHE_ENV",
    "BOOT_TIMEOUT_S",
    "ENGINE_DRIVER",
    "MODELING_V2_ENV",
    "MODE_OFF",
    "MODE_REQUIRE",
    "RANK_FACTS_ENV",
    "RANK_HOOK_DIR",
    "RUN_TRITON_CACHE_DIRNAME",
    "STEP_CONTRACT_ENV",
    "ShellRunner",
    "TRITON_AUTOTUNE_RESULT_GLOB",
    "TRITON_CACHE_AUTOTUNING_ENV",
    "TRITON_CACHE_DIR_ENV",
    "default_shell_runner",
    "default_triton_cache",
    "engine_driver_command",
    "gate_workdir",
    "parse_driver_records",
    "pin_tuning",
    "with_mode",
    "write_driver_spec",
]

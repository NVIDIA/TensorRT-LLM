# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run's shared pool of Slurm GPU nodes.

The pool is a set of holder jobs, each holding one whole node and doing
nothing else. The orchestrator keeps ``nodes`` of them submitted from the
moment the run starts — while the planner is still planning — and cancels
them when the run ends. The planner may grow the pool, up to ``max_nodes``.
Workers run GPU commands as steps of a holder job through :mod:`.gpu_run`,
which locks GPUs and serves waiting commands by their task's priority class.

Everything a command needs to find the pool is in ``<workspace>/gpu_pool/
pool.json``, written by the orchestrator.
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import gpu_run

__all__ = [
    "GpuPool",
    "PoolConfig",
    "PoolConfigError",
    "pool_config",
]

# How often the keeper checks that every holder job is submitted.
KEEP_SECONDS = 120

Run = Callable[[list[str]], subprocess.CompletedProcess[str]]


class PoolConfigError(ValueError):
    """``slurm-environment.gpu_pool`` in ``task.yaml`` is not usable."""


@dataclass(frozen=True)
class PoolConfig:
    """How the pool's holder jobs are submitted and its commands run.

    Attributes:
        nodes: How many holder jobs (whole nodes) to keep submitted at first.
        max_nodes: The most the planner may grow the pool to; ``None`` leaves
            the bound to the orchestrator.
        gpus_per_node: GPUs each holder job holds.
        account: Slurm account of the holder jobs.
        partition: Slurm partition of the holder jobs.
        qos: Slurm QOS of the holder jobs, if any.
        hours: Time limit of one holder job; the keeper submits a new one when
            it ends.
        reserve: Free GPUs that ``perf`` work may not take.
        container_image: The container every command runs in.
        container_mounts: Paths mounted at the same path in the container.
    """

    nodes: int
    gpus_per_node: int
    account: str
    partition: str
    container_image: str
    qos: str | None = None
    hours: float = 24
    reserve: int = 2
    container_mounts: tuple[str, ...] = ()
    max_nodes: int | None = None


def pool_config(task_data: dict) -> PoolConfig | None:
    """The pool ``task.yaml`` asks for, or ``None`` when it asks for none.

    Raises:
        PoolConfigError: Listing every problem with ``gpu_pool``.
    """
    slurm = task_data.get("slurm-environment")
    if not isinstance(slurm, dict) or slurm.get("gpu_pool") is None:
        return None
    raw = slurm["gpu_pool"]
    if not isinstance(raw, dict):
        raise PoolConfigError("`slurm-environment.gpu_pool` must be a mapping")
    problems: list[str] = []
    known = {
        "nodes",
        "max_nodes",
        "gpus_per_node",
        "account",
        "partition",
        "qos",
        "hours",
        "reserve",
        "container_mounts",
    }
    problems += [f"`gpu_pool.{key}` is not a known field" for key in sorted(set(raw) - known)]

    def integer(key: str, default: int | None, minimum: int) -> int:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            problems.append(f"`gpu_pool.{key}` must be an integer of at least {minimum}")
            return minimum
        return value

    def text(key: str, default: str | None) -> str | None:
        value = raw.get(key, default)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            problems.append(f"`gpu_pool.{key}` must be a non-empty string")
            return None
        return value

    nodes = integer("nodes", None, 0)
    max_nodes = None
    if raw.get("max_nodes") is not None:
        max_nodes = integer("max_nodes", None, max(nodes, 1))
    gpus_per_node = integer("gpus_per_node", 4, 1)
    reserve = integer("reserve", 2, 0)
    account = text("account", None)
    if account is None:
        problems.append("`gpu_pool.account` is required")
    partition = text("partition", slurm.get("slurm_partition"))
    if partition is None:
        problems.append("`gpu_pool.partition` (or `slurm_partition`) is required")
    image = slurm.get("docker_image")
    if not isinstance(image, str) or not image.strip():
        problems.append(
            "the pool runs commands in `slurm-environment.docker_image`, which is unset"
        )
    hours = raw.get("hours", 24)
    if isinstance(hours, bool) or not isinstance(hours, int | float) or hours <= 0:
        problems.append("`gpu_pool.hours` must be a positive number")
    mounts = raw.get("container_mounts", [])
    if not isinstance(mounts, list) or not all(isinstance(m, str) and m for m in mounts):
        problems.append("`gpu_pool.container_mounts` must be a list of paths")
        mounts = []
    if problems:
        raise PoolConfigError("; ".join(problems))
    if nodes == 0:
        return None
    return PoolConfig(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        account=account,
        partition=partition,
        container_image=image,
        qos=text("qos", None),
        hours=float(hours),
        reserve=reserve,
        container_mounts=tuple(mounts),
        max_nodes=max_nodes,
    )


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


class GpuPool:
    """The orchestrator's side of the pool: write it, keep it, release it.

    Args:
        directory: The pool's directory, ``<workspace>/gpu_pool``.
        config: How to submit and use the holder jobs.
        workspace: The run's workspace; commands look up task priorities in
            its ``gpu_priorities.json``.
        mounts: Paths every command needs mounted besides the configured ones
            (the workspace and the TensorRT-LLM checkout).
        run: Runs a Slurm command; :func:`subprocess.run` by default.
    """

    def __init__(
        self,
        *,
        directory: Path,
        config: PoolConfig,
        workspace: Path,
        mounts: tuple[Path, ...],
        run: Run | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.config = config
        self.workspace = Path(workspace)
        self.mounts = tuple(dict.fromkeys([*config.container_mounts, *map(str, mounts)]))
        self._run = run or _run
        # Holder jobs to keep submitted; the planner may raise it.
        self.nodes = config.nodes
        digest = hashlib.sha1(str(self.workspace.resolve()).encode()).hexdigest()[:8]
        self.job_name = f"mv2pool-{digest}"

    @property
    def holder_path(self) -> Path:
        return self.directory / "holder.sbatch"

    @property
    def config_path(self) -> Path:
        return self.directory / "pool.json"

    def write(self) -> None:
        """Write ``pool.json`` and the holder job's script."""
        config = self.config
        for sub in ("locks", "queue", "logs"):
            (self.directory / sub).mkdir(parents=True, exist_ok=True)
        data = {
            **asdict(config),
            "job_name": self.job_name,
            "container_mounts": list(self.mounts),
            "priorities": str(self.workspace / "gpu_priorities.json"),
        }
        self.config_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        minutes = int(config.hours * 60)
        lines = [
            "#!/bin/bash",
            "# Holds one node of the run's GPU pool; commands run as steps of this job.",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={config.account}",
            f"#SBATCH --partition={config.partition}",
            *([f"#SBATCH --qos={config.qos}"] if config.qos else []),
            "#SBATCH --nodes=1",
            f"#SBATCH --gres=gpu:{config.gpus_per_node}",
            f"#SBATCH --time={minutes // 60}:{minutes % 60:02d}:00",
            f"#SBATCH --output={self.directory / 'logs' / 'holder_%j.out'}",
            'echo "[pool] job=$SLURM_JOB_ID node=$SLURM_NODELIST start=$(date -u +%FT%TZ)"',
            "exec sleep infinity",
        ]
        self.holder_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _jobs(self) -> list[tuple[str, str]] | None:
        """``(job id, state)`` of the holder jobs running or queued.

        ``None`` when Slurm cannot be asked.
        """
        out = self._run(
            [
                "squeue",
                "-h",
                "-u",
                getpass.getuser(),
                "-n",
                self.job_name,
                "-t",
                "R,PD",
                "-o",
                "%i %t",
            ]
        )
        if out.returncode != 0:
            return None
        return [
            (fields[0], fields[1] if len(fields) > 1 else "")
            for fields in (line.split() for line in out.stdout.splitlines())
            if fields
        ]

    def top_up(self) -> str | None:
        """Submit holder jobs until ``nodes`` are submitted.

        Stops at the first refused submission, so it is not repeated in a
        burst. Returns what happened, for the log, or ``None`` when nothing did.
        """
        jobs = self._jobs()
        if jobs is None:
            return None
        notes = []
        for _ in range(self.nodes - len(jobs)):
            out = self._run(["sbatch", str(self.holder_path)])
            if out.returncode != 0:
                lines = (out.stderr or out.stdout).strip().splitlines()
                notes.append(f"holder job not submitted: {lines[-1] if lines else 'sbatch failed'}")
                break
            notes.append(out.stdout.strip())
        return "; ".join(notes) or None

    def status(self) -> str:
        """The pool as the planner sees it: nodes, free GPUs and who waits."""
        jobs = self._jobs()
        if jobs is None:
            return f"GPU pool: {self.nodes} node(s) wanted; Slurm did not answer."
        running = [job for job, state in jobs if state == "R"]
        queued = len(jobs) - len(running)
        gpus = self.config.gpus_per_node
        free = [gpu_run.free_gpus(self.directory / "locks", job, gpus) for job in running]
        lines = [
            f"GPU pool: {self.nodes} node(s) of {gpus} GPUs wanted, {len(running)} running"
            + (f", {queued} queued in Slurm" if queued else "")
            + f"; free GPUs per running node: {free or 'none'}."
        ]
        now = time.time()
        waiting = sorted(gpu_run.tickets(self.directory / "queue"), key=lambda t: t["since"])
        lines += [
            f"- waiting: `{t.get('task', '?')}` ({t.get('class', '?')}) for {t['gpus']} GPU(s), "
            f"{(now - t['since']) / 60:.0f} min"
            for t in waiting
        ] or ["- no command is waiting for GPUs."]
        return "\n".join(lines)

    async def keep(self, log: Callable[[str], None]) -> None:
        """Keep the holder jobs submitted until cancelled."""
        while True:
            note = await asyncio.to_thread(self.top_up)
            if note:
                log(f"gpu pool: {note}")
            await asyncio.sleep(KEEP_SECONDS)

    def release(self) -> None:
        """Cancel every holder job, running or queued."""
        self._run(["scancel", "-u", getpass.getuser(), "-n", self.job_name])

    def command(self, python: str) -> str:
        """The command prefix a worker runs GPU work with."""
        module = "agent_flow.workflows.modeling_v2_onboard.gpu_run"
        return f"{shlex.quote(python)} -m {module} --pool {shlex.quote(str(self.directory))}"

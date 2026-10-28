# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a tune task into its workloads and serving configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..checks.check_config import checkpoint_remote_code
from .measure import BenchPart, BenchSpec
from .serving_config import derive_serving_config, write_serving_config

PACKAGE = Path(__file__).resolve().parent.parent
TARGET_SELECTION = {"modeling_v2": "require"}


@dataclass
class TuneSpec:
    workdir: Path
    repo: Path
    model: str
    target: str
    identity_configs: list[Path]
    isl: int
    osl: int
    concurrency: list[int]
    reference: str
    spools: list[str]
    dev_spools: list[str]
    env_script: Path
    noise_repeats: int = 0
    campaign: dict = field(default_factory=dict)
    embargo: list[str] = field(default_factory=list)
    sister_engines: list[str] = field(default_factory=list)
    wall: str = "2h"
    #: The perf configs, when ``perf.configs`` names several; every measurement covers all.
    parts: list[BenchPart] = field(default_factory=list)
    #: Stop dispatching once every headline metric is this multiple of the stock reference's.
    goal_ratio: float | None = None
    #: The commit to tune. Empty means the checkout's ``HEAD``, measured in place; a commit is
    #: measured in the campaign's own ``best`` worktree, so commits landing on the checkout
    #: meanwhile (later feature work) never reach a measurement.
    base_commit: str = ""
    #: ``perf.sol``: whether a turn derives the speed-of-light ceilings, and the GPU part to price.
    sol: dict = field(default_factory=lambda: {"enabled": True})
    #: Stop dispatching once every headline metric is this fraction of its SOL ceiling.
    goal_pct_of_sol: float | None = None
    #: The task the campaign serves: the contract its keep gate is (``keep_gate.py``).
    task: dict = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: Mapping[str, Any], workspace: Path, *, target: str = "") -> "TuneSpec":
        perf = task.get("perf") or {}
        workload = perf.get("workload") or {}
        env = task.get("environment") or {}
        session_spools = [s.get("spool") for s in env.get("sessions") or [] if s.get("spool")]
        spools = [str(s) for s in perf.get("spools") or session_spools]
        workdir = Path(workspace).resolve() / "tune"
        isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
        concurrency = [int(c) for c in workload.get("concurrency", [1, 128])]
        identity = [Path(p).resolve() for p in perf.get("identity_configs") or []]
        parts = [_part(task, workdir, entry) for entry in perf.get("configs") or []]
        if parts:
            identity = []
            isl, osl, concurrency = parts[0].isl, parts[0].osl, list(parts[0].concurrencies)
        elif not identity:
            config = {
                **TARGET_SELECTION,
                **derive_serving_config(
                    isl=isl, osl=osl, concurrency=concurrency, draft_len=_draft_len(task)
                ),
            }
            identity = [write_serving_config(workdir / "configs" / "serving.yaml", config)]
        goal = perf.get("goal") if isinstance(perf.get("goal"), Mapping) else {}
        return cls(
            # Absolute: every path in a job script is read on a compute node from
            # the job's own cwd (the tree measured), not from where the CLI ran.
            workdir=workdir,
            repo=Path(task.get("trtllm_repo_path") or ".").resolve(),
            model=str(perf.get("model") or task.get("checkpoint_path")),
            target=str(perf.get("target") or target),
            identity_configs=identity,
            isl=isl,
            osl=osl,
            concurrency=concurrency,
            reference=str(perf.get("reference", "stock")),
            spools=[str(Path(s).resolve()) for s in spools],
            dev_spools=[str(Path(s).resolve()) for s in perf.get("dev_spools") or spools],
            env_script=Path(
                perf.get("env_script") or Path(workspace) / "tune" / "env.sh"
            ).resolve(),
            noise_repeats=int(perf.get("noise_repeats", 0)),
            campaign=dict(perf.get("campaign") or {}),
            embargo=[
                str(e)
                for e in (task.get("source_embargo") or []) + (perf.get("source_embargo") or [])
            ],
            sister_engines=[str(s) for s in perf.get("sister_engines") or []],
            wall=str(perf.get("wall", "2h")),
            parts=parts,
            goal_ratio=float(goal["ratio_to_reference"]) if "ratio_to_reference" in goal else None,
            sol={"enabled": True, **dict(perf.get("sol") or {})},
            goal_pct_of_sol=float(goal["pct_of_sol"]) if "pct_of_sol" in goal else None,
            task=dict(task),
        )

    def describe_workload(self) -> str:
        if not self.parts:
            return (
                f"ISL {self.isl} / OSL {self.osl}, concurrency {self.concurrency}, CUDA graphs on."
            )
        rows = [
            f"`{p.name}`: ISL {p.isl} / OSL {p.osl}, concurrency {list(p.concurrencies)}"
            + (f", constraint {dict(p.constraint)}" if p.constraint else "")
            for p in self.parts
        ]
        return "Serving validation covers each perf config: " + "; ".join(rows) + "."

    def bench(self) -> BenchSpec:
        return BenchSpec(
            bench_py=str(PACKAGE / "tune" / "harness" / "bench.py"), model=self.model, env_script=str(self.env_script),
            isl=self.isl, osl=self.osl, concurrencies=self.concurrency, tree=str(self.repo), wall=self.wall,
            parts=tuple(self.parts),
        )  # fmt: skip


def _draft_len(task: Mapping[str, Any]) -> int:
    """The draft length a speculative target verifies, or 0 without one."""
    from ..checks.check_config import resolve_max_draft_len
    from ..spec.task_schema import SPECULATORS

    speculators = [f for f in task.get("features") or [] if f in SPECULATORS]
    if not speculators:
        return 0
    return resolve_max_draft_len(task.get("accuracy") or {}, speculators[-1])


def _part(task: Mapping[str, Any], workdir: Path, entry: Mapping[str, Any]) -> BenchPart:
    """One ``perf.configs`` entry: its workload, and its configs (derived when it names none)."""
    name = str(entry["name"])
    workload = entry.get("workload") or {}
    isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
    concurrency = [int(c) for c in workload.get("concurrency", [1])]
    configs = [Path(p).resolve() for p in entry.get("identity_configs") or []]
    if not configs:
        config = {
            **TARGET_SELECTION,
            **derive_serving_config(
                isl=isl, osl=osl, concurrency=concurrency, draft_len=_draft_len(task)
            ),
        }
        if checkpoint_remote_code(task.get("checkpoint_path")):
            config["trust_remote_code"] = True
        config.update(entry.get("llm_api_options") or {})
        configs = [write_serving_config(workdir / "configs" / f"serving-{name}.yaml", config)]
    return BenchPart(
        name=name,
        isl=isl,
        osl=osl,
        concurrencies=concurrency,
        configs=configs,
        constraint={k: float(v) for k, v in (entry.get("constraint") or {}).items()},
    )

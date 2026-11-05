# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a tune task into its workloads and serving configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..checks.check_config import checkpoint_remote_code, options_capture
from ..spec.task_schema import DEFAULT_PERF_CONCURRENCY
from .harness.bench import AGENTX_DATASET, AGENTX_DURATION_S, resolve_config
from .measure import BenchPart, BenchSpec
from .schema import AGENTX_WORKLOAD_KEYS
from .serving_config import derive_serving_config, write_serving_config

PACKAGE = Path(__file__).resolve().parent.parent
TARGET_SELECTION = {"modeling_v2": "require"}
#: How far above its average share of the concurrency one attention-DP rank's live requests
#: run: requests finish unevenly, so a rank holds more than its share for whole stretches.
CAPTURE_HEADROOM = 1.5


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
    spools: list[str]
    dev_spools: list[str]
    env_script: Path
    campaign: dict = field(default_factory=dict)
    sister_engines: list[str] = field(default_factory=list)
    wall: str = "2h"
    #: The perf configs, when ``perf.configs`` names several; every measurement covers all.
    parts: list[BenchPart] = field(default_factory=list)
    #: Stop dispatching once every headline metric is this multiple of the baseline's.
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
    #: ``perf.source_workspace``: the workspace whose core verdicts establish the tested source the
    #: campaign imports (an onboarding beside a standalone tune); None for the campaign's own.
    source_workspace: Path | None = None

    @classmethod
    def from_task(cls, task: Mapping[str, Any], workspace: Path, *, target: str = "") -> "TuneSpec":
        perf = task.get("perf") or {}
        workload = perf.get("workload") or {}
        env = task.get("environment") or {}
        session_spools = [s.get("spool") for s in env.get("sessions") or [] if s.get("spool")]
        spools = [str(s) for s in perf.get("spools") or session_spools]
        workdir = Path(workspace).resolve() / "tune"
        isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
        concurrency = [int(c) for c in workload.get("concurrency", DEFAULT_PERF_CONCURRENCY)]
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
            spools=[str(Path(s).resolve()) for s in spools],
            dev_spools=[str(Path(s).resolve()) for s in perf.get("dev_spools") or spools],
            env_script=Path(
                perf.get("env_script") or Path(workspace) / "tune" / "env.sh"
            ).resolve(),
            campaign=dict(perf.get("campaign") or {}),
            sister_engines=[str(s) for s in perf.get("sister_engines") or []],
            wall=str(perf.get("wall", "2h")),
            parts=parts,
            goal_ratio=float(goal["ratio_to_baseline"]) if "ratio_to_baseline" in goal else None,
            sol={"enabled": True, **dict(perf.get("sol") or {})},
            goal_pct_of_sol=float(goal["pct_of_sol"]) if "pct_of_sol" in goal else None,
            task=dict(task),
            source_workspace=(
                Path(perf["source_workspace"]).resolve() if perf.get("source_workspace") else None
            ),
        )

    def describe_workload(self) -> str:
        if not self.parts:
            return (
                f"ISL {self.isl} / OSL {self.osl}, concurrency {self.concurrency}, CUDA graphs on."
            )
        rows = [
            (
                f"`{p.name}`: AgentX trace replay of {p.agentx['dataset']} for "
                f"{p.agentx['duration_s']} s per point (about ISL {p.isl} / OSL {p.osl} per "
                f"request, prefixes reused across turns), concurrency {list(p.concurrencies)}"
                if p.client == "agentx"
                else f"`{p.name}`: ISL {p.isl} / OSL {p.osl}, concurrency {list(p.concurrencies)}"
            )
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

    def serving_options(self) -> list[tuple[str, list[int], dict]]:
        """Each perf config's name, concurrencies and the LLM-API options its server reads."""
        return [
            (name, list(concurrencies), resolve_config(list(paths), []))
            for name, (_, _, concurrencies, paths) in self.bench()
            .workloads(self.identity_configs)
            .items()
        ]


def _draft_len(task: Mapping[str, Any]) -> int:
    """The draft length a speculative target verifies, or 0 without one."""
    from ..checks.check_config import resolve_max_draft_len
    from ..spec.task_schema import SPECULATORS

    speculators = [f for f in task.get("features") or [] if f in SPECULATORS]
    if not speculators:
        return 0
    return resolve_max_draft_len(task.get("accuracy") or {}, speculators[-1])


def _derived_options(task: Mapping[str, Any], entry: Mapping[str, Any]) -> dict:
    """The serving config a ``perf.configs`` entry that names no configs runs with."""
    workload = entry.get("workload") or {}
    isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
    concurrency = [int(c) for c in workload.get("concurrency", [1])]
    config = {
        **TARGET_SELECTION,
        **derive_serving_config(
            isl=isl, osl=osl, concurrency=concurrency, draft_len=_draft_len(task)
        ),
    }
    if checkpoint_remote_code(task.get("checkpoint_path")):
        config["trust_remote_code"] = True
    config.update(entry.get("llm_api_options") or {})
    return config


def _part(task: Mapping[str, Any], workdir: Path, entry: Mapping[str, Any]) -> BenchPart:
    """One ``perf.configs`` entry: its workload, and its configs (derived when it names none)."""
    name = str(entry["name"])
    workload = entry.get("workload") or {}
    isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
    concurrency = [int(c) for c in workload.get("concurrency", [1])]
    configs = [Path(p).resolve() for p in entry.get("identity_configs") or []]
    if not configs:
        config = _derived_options(task, entry)
        configs = [write_serving_config(workdir / "configs" / f"serving-{name}.yaml", config)]
    client = str(workload.get("client", "random"))
    agentx = {}
    if client == "agentx":
        agentx = {k: workload[k] for k in AGENTX_WORKLOAD_KEYS if workload.get(k) is not None}
        agentx.setdefault("dataset", AGENTX_DATASET)
        agentx.setdefault("duration_s", AGENTX_DURATION_S)
        # One corpus and client cache per campaign, shared by every measurement.
        agentx["hf_home"] = str(
            Path(agentx.get("hf_home") or workdir / "agentx-cache" / "hf").resolve()
        )
    return BenchPart(
        name=name,
        isl=isl,
        osl=osl,
        concurrencies=concurrency,
        configs=configs,
        constraint={k: float(v) for k, v in (entry.get("constraint") or {}).items()},
        client=client,
        agentx=agentx,
    )


def serving_options(task: Mapping[str, Any]) -> list[tuple[str, list[int], dict]]:
    """:meth:`TuneSpec.serving_options` read from ``task`` alone, writing nothing.

    A config whose files do not read is left out: the campaign's own start says why.
    """
    perf = task.get("perf") or {}
    entries = perf.get("configs")
    if not entries:
        workload = perf.get("workload") or {}
        concurrency = workload.get("concurrency", list(DEFAULT_PERF_CONCURRENCY))
        entries = [
            {
                "name": "",
                "workload": {**workload, "concurrency": concurrency},
                "identity_configs": perf.get("identity_configs"),
            }
        ]
    found = []
    for entry in entries:
        concurrency = [int(c) for c in (entry.get("workload") or {}).get("concurrency", [1])]
        files = [Path(p).resolve() for p in entry.get("identity_configs") or []]
        try:
            options = resolve_config(files, []) if files else _derived_options(task, entry)
        except (OSError, yaml.YAMLError):
            continue
        found.append((str(entry.get("name") or ""), concurrency, options))
    return found


def capture_shortfalls(
    configs: Iterable[tuple[str, Sequence[int], Mapping[str, Any]]],
) -> list[tuple[str, str]]:
    """Each perf config whose decode runs eager for part of its load: ``(why, fix)``.

    CUDA graphs off, or a largest captured batch below the live requests one rank holds: the
    config's concurrency over its attention-DP ranks, times :data:`CAPTURE_HEADROOM` (never
    above the concurrency or the engine's ``max_batch_size``). A step past the largest batch
    runs eager, and the campaign measures that instead of the captured path.
    """
    found = []
    for name, concurrency, options in configs:
        where = f"perf config `{name}`" if name else "the perf config"
        try:
            capture = options_capture(options)
        except ValueError:
            continue  # the LLM API refuses it; the server's boot says so
        if capture is None:
            found.append(
                (
                    f"{where} turns CUDA graphs off (`cuda_graph_config: null`): every decode "
                    "step runs eager, a path no deployment serves",
                    "drop `cuda_graph_config: null` and capture up to the concurrency",
                )
            )
            continue
        top = max(int(c) for c in concurrency)
        ranks = (
            max(1, int(options.get("tensor_parallel_size") or 1))
            if options.get("enable_attention_dp")
            else 1
        )
        share = -(-top // ranks)
        live = min(top, math.ceil(share * CAPTURE_HEADROOM))
        if options.get("max_batch_size"):
            live = min(live, int(options["max_batch_size"]))
        largest = capture.batch_sizes[-1]
        if largest >= live:
            continue
        load = (
            f"concurrency {top} over {ranks} attention-DP ranks is {share} live requests per "
            f"rank on average, and a rank holds up to {live} as requests finish unevenly"
            if ranks > 1
            else f"concurrency {top} keeps up to {live} requests live"
        )
        default = "" if "cuda_graph_config" in options else " (the LLM API default: it sets none)"
        found.append(
            (
                f"{where} captures CUDA graphs up to batch {largest}{default}, but {load}: "
                f"every decode step above {largest} runs eager",
                f"capture every batch up to at least {live}, e.g. `cuda_graph_config: "
                f"{{enable_padding: true, max_batch_size: {live}}}`, with the engine's "
                "`max_batch_size` at least as large",
            )
        )
    return found

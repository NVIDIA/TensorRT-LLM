# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy of an aggregated deployment: ``trtllm-eval <benchmark>``, held to its anchor.

One benchmark is one run of the tree's own evaluator on the deployment's options, behind
the deployment's launch prefix when it has one and on this host's ranks when it has none.
The score is read from what the evaluator prints and compared, one-sided, with the floor
its anchor sets (``checks/anchors.py``). Every rank's engine is held to the deployment's
options (``checks/attest.py``); a benchmark with no anchor is a score with no verdict.
"""

from __future__ import annotations

import os
import re
import shlex
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment, env_assignments, single_task_scrub
from ..infra.files import write_text_atomic
from ..spec.deployments import Benchmark, Deployment
from . import anchors as anchors_mod
from . import attest, datasets, servers
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

ACCURACY_TIMEOUT_S = 4 * 60 * 60
#: How often, in answered requests, the benchmark prints its running score
#: (``TLLM_EVAL_PARTIAL_SCORES_EVERY`` in ``tensorrt_llm/evaluate/lm_eval.py``). The lines
#: keep a healthy run's log growing and carry the sample count scored.
EVAL_PROGRESS_EVERY = 50
EVAL_PROGRESS_ENV = "TLLM_EVAL_PARTIAL_SCORES_EVERY"
# ``lm-eval <task> <filter> accuracy: 90.67`` and ``lm-eval <task> average accuracy: 90.67``:
# the two shapes ``tensorrt_llm/evaluate/lm_eval.py`` emits at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# ``MMLU weighted average accuracy: 74.98 (14042)``: MMLU does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
#: A line that may carry a benchmark's score: a metric's name, then a number. What a
#: verdict quotes when no score parses, for whoever decides how to read it.
_SCORE_LIKE = re.compile(r"(?i)\b(accuracy|score|rouge\w*|exact_match|f1|pass@\d+|bleu)\b[^\n]*?\d")
#: How many of those lines a verdict quotes, the last ones printed.
SCORE_LINES = 8


def eval_command(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    *,
    workdir: str | Path,
    dataset_path: str = "",
) -> list[str]:
    """``trtllm-eval`` on the deployment's options with the Modeling V2 switch on, as shell words."""
    options = {**deployment.server.options, "modeling_v2": MODE_REQUIRE}
    target = gate_workdir(workdir) / f"eval.{deployment.name}.{benchmark.name}.yaml"
    write_text_atomic(target, yaml.safe_dump(options, sort_keys=True), mode=0o600)
    parts = ["trtllm-eval", f"--model={task_data['checkpoint_path']}", f"--config={target}"]
    parts.append(benchmark.name)
    if dataset_path:
        parts.append(f"--dataset_path={dataset_path}")
    arguments = list(benchmark.args)
    # Logged samples and the inference text stay under the check's directory, for an audit.
    if benchmark.name == "gsm8k" and "--log_samples" in arguments:
        output = None
        for index, argument in enumerate(arguments):
            if argument.startswith("--output_path="):
                output = argument.split("=", 1)[1]
            elif argument == "--output_path" and index + 1 < len(arguments):
                output = arguments[index + 1]
        if output is None:
            output = str(target.parent / f"{target.stem}.samples.{time.time_ns()}")
            arguments.append(f"--output_path={output}")
        if not any(arg == "--output_dir" or arg.startswith("--output_dir=") for arg in arguments):
            arguments.append(f"--output_dir={Path(output) / 'inference'}")
    return [shlex.quote(part) for part in (*parts, *arguments)]


def launch_command(
    environment: RunEnvironment, deployment: Deployment, benchmark: Benchmark, parts: list[str]
) -> str:
    """``parts`` as this deployment launches it: behind its prefix, else on this host's ranks.

    A launch prefix places the ranks itself, so nothing of the orchestrator's devices or
    of its MPI and scheduler identity goes behind it; without one the run's environment
    starts the ranks here. The evaluator runs offline and prints its running score.
    """
    env = {
        EVAL_PROGRESS_ENV: str(EVAL_PROGRESS_EVERY),
        **environment.env,
        **deployment.env,
        **benchmark.env,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    environment = with_mode(replace(environment, env=env), MODE_REQUIRE)
    launch = deployment.server.launch.strip()
    if not launch:
        return environment.command(" ".join(parts))
    # The prefix is the site's own words (it may begin with assignments): run as written,
    # with the LLM API's launcher before the program where it has made several ranks. The
    # evaluator is the one installed beside the run's interpreter, and nothing of the
    # orchestrator's scheduler identity goes behind the prefix.
    program = " ".join([*servers.launcher(deployment.server), *parts])
    beside = os.path.dirname(environment.python) if os.path.isabs(environment.python) else ""
    path = f'PATH={shlex.quote(beside)}:"$PATH"; ' if beside else ""
    inner = f"{single_task_scrub(list(env), rank=False)}; {path}{launch} {program}"
    return f"{' '.join(env_assignments(environment.env))} bash -c {shlex.quote(inner)}"


def parse_eval_score(stdout: str, pattern: str | None = None) -> float | None:
    """Extract the final accuracy from an evaluator's output.

    ``pattern`` (a benchmark's ``score_pattern``) is a regular expression whose first
    group is the score; without one, MMLU prints its own weighted average and everything
    else comes through lm-eval. The **last** match wins: a long run also prints
    partial-score estimates.
    """
    patterns = (re.compile(pattern),) if pattern else (_MMLU_SCORE, _LM_EVAL_SCORE)
    for compiled in patterns:
        for match in reversed(list(compiled.finditer(stdout))):
            try:
                return float(match.group(1))
            except (IndexError, TypeError, ValueError):
                continue
    return None


def score_lines(output: str, limit: int = SCORE_LINES) -> list[str]:
    """The last ``limit`` distinct lines of ``output`` that look like a score, oldest first."""
    seen: list[str] = []
    for line in reversed(output.splitlines()):
        text = line.strip()[:240]
        if text and _SCORE_LIKE.search(text) and text not in seen:
            seen.append(text)
            if len(seen) == limit:
                break
    return seen[::-1]


def unparsed_score_detail(output: str, pattern: str | None) -> str:
    """Why a run that finished gives no number, with what it printed that looks like one."""
    if pattern:
        head = f"the evaluator exited 0, but `score_pattern` ({pattern!r}) matched no score"
    else:
        head = (
            "the evaluator exited 0 but printed no score in a form this check reads "
            "(lm-eval's `accuracy:` line or MMLU's weighted average)"
        )
    lines = score_lines(output)
    printed = (
        "; the lines it printed that look like a score:\n" + "\n".join(f"  {x}" for x in lines)
        if lines
        else "; it printed no line that looks like a score"
    )
    return (
        f"{head}{printed}\nDecide how this benchmark's score is read: its `score_pattern` names "
        "it (a regular expression whose first group is the score, on the anchor's scale), or "
        "evaluate the output another way and report it. A check with no number is not a pass."
    )


def judged(
    measured: float,
    anchor: anchors_mod.Anchor,
    benchmark: Benchmark,
    *,
    relative_score: float | None,
    output: str,
    how: str = "",
) -> dict[str, Any]:
    """The verdict fields of ``measured`` against ``anchor``'s floor; a score alone without one."""
    if anchor.unit == "fraction" and measured > 1:
        # Compared as it is, a percentage would clear any floor a fraction sets.
        return dict(
            passed=False,
            measured=measured,
            detail=f"measured {measured:.4f}{how}, which is no fraction: the benchmark's `unit` "
            f"says its scores are fractions, and this one is on another scale; state the unit "
            f"the evaluator reports",
        )
    floor, source = anchors_mod.floor(
        anchor, benchmark, relative_score=relative_score, output=output
    )
    if floor is None:
        return dict(
            passed=False,
            explained=True,
            measured=measured,
            detail=f"measured {measured:.4f}{how}; a score with no verdict "
            f"({anchor.source if anchor.kind == anchors_mod.NONE else source})",
        )
    passed = measured >= floor.value
    return dict(
        passed=passed,
        measured=measured,
        reference=floor.reference,
        floor=floor.value,
        se=floor.se,
        samples=floor.samples,
        floor_rule=floor.describe(),
        detail=f"measured {measured:.4f} {'>=' if passed else '<'} {floor.value:.4f}{how} "
        f"({floor.describe()}; count from {source}; anchor: {anchor.kind})",
    )


def run_accuracy_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    anchor: anchors_mod.Anchor,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    relative_score: float | None = None,
) -> GateResult:
    """Score ``benchmark`` on ``deployment`` with ``trtllm-eval`` and hold it to ``anchor``'s floor.

    ``relative_score`` is the score of the deployment a relative anchor names, measured
    as this one is.
    """
    verdict = verdict_factory(inspect_tree(task_data))
    name = f"accuracy[{benchmark.name}]"
    started = time.monotonic()
    fields: dict[str, Any] = {}

    def ran(**more: Any) -> GateResult:
        return verdict(
            name=name,
            deployment=deployment.name,
            duration_s=time.monotonic() - started,
            **{**fields, **more},
        )

    # Looked for where the evaluator will: under the deployment's and the benchmark's own
    # environment, over the run's.
    declared = {**environment.env, **deployment.env, **benchmark.env}
    dataset, said = datasets.find(benchmark.name, benchmark.dataset_path, declared)
    if said and not dataset:
        return ran(passed=False, detail=said, infrastructure=said)
    if timeout is None:
        timeout = int((task_data.get("accuracy") or {}).get("timeout") or ACCURACY_TIMEOUT_S)
    parts = eval_command(task_data, deployment, benchmark, workdir=workdir, dataset_path=dataset)
    command = launch_command(environment, deployment, benchmark, parts)
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    fields.update(command=command, stdout_tail="\n".join(combined.splitlines()[-40:]))
    if result.infrastructure:
        return ran(**lost_session(result))
    if result.timed_out:
        return ran(passed=False, detail=ended_early(result, timeout))
    if not result.ok:
        return ran(passed=False, detail=f"trtllm-eval exited {result.returncode}")
    attested = attest.attest(
        combined,
        {**deployment.server.options, "modeling_v2": MODE_REQUIRE},
        tree=task_data.get("trtllm_repo_path"),
    )
    fields["engine"] = (attested.record(),)
    if attested.failure():
        return ran(passed=False, detail=attested.failure())
    measured = parse_eval_score(combined, benchmark.score_pattern)
    if measured is None:
        return ran(passed=False, detail=unparsed_score_detail(combined, benchmark.score_pattern))
    return ran(
        **judged(measured, anchor, benchmark, relative_score=relative_score, output=combined)
    )


__all__ = [
    "ACCURACY_TIMEOUT_S",
    "eval_command",
    "judged",
    "launch_command",
    "parse_eval_score",
    "run_accuracy_gate",
    "score_lines",
    "unparsed_score_detail",
]

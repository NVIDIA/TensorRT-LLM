# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a benchmark's score is read and judged, and ``trtllm-eval`` in a load of its own.

A deployment's benchmarks are scored through its session's endpoint
(``checks/session.py``). What is here is shared by both ways of taking a score: reading
it from what an evaluator prints (:func:`parse_eval_score`), the reference rows
(:func:`load_reference_score`) and the one-sided judgment against the floor its anchor
sets (:func:`judged`). :func:`run_eval` is the other way: ``trtllm-eval`` on an
aggregated deployment's options, which loads the model once more. It runs for a
benchmark the endpoint evaluator cannot carry or one the spec marks ``through: eval``,
and its line says so.
"""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment
from ..infra.files import write_text_atomic
from ..spec.deployments import REQUIREMENT_OPTION, requirement, requirement_said
from ..spec.task_schema import resolve_per_unit
from . import accuracy_policy, attest
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .gate_records import KIND_FINDING, GateRecord, GateRecords, open_findings
from .verdict import UNKNOWN_TREE_STATE, GateResult, inspect_tree, lost_session, verdict_factory

ACCURACY_TIMEOUT_S = 4 * 60 * 60

#: How often, in answered requests, the benchmark prints its running score
#: (``TLLM_EVAL_PARTIAL_SCORES_EVERY`` in ``tensorrt_llm/evaluate/lm_eval.py``).
#: The lines keep a healthy run's log growing, so a non-fatal traceback early
#: on does not read as a hang, and they carry the sample count scored.
EVAL_PROGRESS_EVERY = 50
EVAL_PROGRESS_ENV = "TLLM_EVAL_PARTIAL_SCORES_EVERY"


# ``lm-eval <task> <filter> accuracy: 90.67`` and
# ``lm-eval <task> average accuracy: 90.67`` — the two shapes
# ``tensorrt_llm/evaluate/lm_eval.py`` emits at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")


# ``MMLU weighted average accuracy: 74.98 (14042)`` — the MMLU path has its
# own reporting because it does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")


def eval_command(
    task_data: Mapping[str, Any],
    deployment: Any,
    benchmark: Any,
    *,
    workdir: str | Path,
    dataset_path: str | None = None,
) -> list[str]:
    """``trtllm-eval`` on the deployment's options under its Modeling V2 requirement, as shell words."""
    options = {
        **deployment.server.options,
        REQUIREMENT_OPTION: requirement(deployment.server.options),
    }
    target = gate_workdir(workdir) / f"eval.{deployment.name}.{benchmark.name}.yaml"
    write_text_atomic(target, yaml.safe_dump(options, sort_keys=True), mode=0o600)
    parts = ["trtllm-eval", f"--model={task_data['checkpoint_path']}", f"--config={target}"]
    parts.append(benchmark.name)
    if dataset_path:
        parts.append(f"--dataset_path={dataset_path}")
    arguments = [str(arg) for arg in benchmark.args]
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


#: A line that may carry a benchmark's score: a metric's name, then a number. What a
#: verdict quotes when no score parses, for whoever decides how to read it.
_SCORE_LIKE = re.compile(r"(?i)\b(accuracy|score|rouge\w*|exact_match|f1|pass@\d+|bleu)\b[^\n]*?\d")
#: How many of those lines a verdict quotes, the last ones printed.
SCORE_LINES = 8


def parse_eval_score(stdout: str, pattern: str | None = None) -> float | None:
    """Extract the final accuracy from a ``trtllm-eval`` run's output.

    ``pattern`` (``accuracy.score_pattern``) is a regular expression whose
    first group is the score; without one, MMLU prints its own weighted
    average and everything else comes through lm-eval. The **last** match
    wins: a long run also prints partial-score estimates.
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
        head = f"trtllm-eval exited 0, but `accuracy.score_pattern` ({pattern!r}) matched no score"
    else:
        head = (
            "trtllm-eval exited 0 but printed no score in a form this gate reads "
            "(lm-eval's `accuracy:` line or MMLU's weighted average)"
        )
    lines = score_lines(output)
    printed = (
        "; the lines it printed that look like a score:\n" + "\n".join(f"  {x}" for x in lines)
        if lines
        else "; it printed no line that looks like a score"
    )
    return (
        f"{head}{printed}\nDecide how this benchmark's score is read: `accuracy.score_pattern` "
        "names it (a regular expression whose first group is the score, on the anchor's scale), "
        "or evaluate the output another way and report it. A gate with no number is not a pass."
    )


#: The keys an in-tree accuracy test selects a reference row by, and the value a row that
#: leaves one out stands for.
REFERENCE_SPEC_DEFAULTS: dict[str, Any] = {
    "dtype": "auto",
    "quant_algo": None,
    "kv_cache_quant_algo": None,
    "spec_dec_algo": None,
    "extra_acc_spec": None,
}


def load_reference_score(
    reference_yaml: str | Path,
    model_key: str,
    *,
    discriminators: Mapping[str, Any] | None = None,
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    The row is the one the in-tree accuracy test would take: the first whose
    ``dtype``, ``quant_algo``, ``kv_cache_quant_algo``, ``spec_dec_algo`` and
    ``extra_acc_spec`` equal the ones asked for, a key the row leaves out
    counting as ``auto`` for ``dtype`` and unset for the rest. A row's other
    keys (``num_samples``, ``threshold``, ...) do not select. A discriminator
    outside those five narrows the match to rows that carry it with that
    value. Raises ``KeyError`` when no row matches: the anchor is never
    invented.
    """
    path = Path(reference_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"accuracy reference file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if model_key not in data:
        raise KeyError(
            f"{path} has no entry for {model_key!r}. The anchor comes from a written "
            f"source or from the user — it is never invented. Add the entry (citing "
            f"where the score came from) before the accuracy gate can run."
        )

    wanted = {k: v for k, v in (discriminators or {}).items() if v is not None}
    asked = {key: wanted.get(key, default) for key, default in REFERENCE_SPEC_DEFAULTS.items()}
    narrowing = {k: v for k, v in wanted.items() if k not in REFERENCE_SPEC_DEFAULTS}
    entries = [e for e in data[model_key] or [] if isinstance(e, dict)]
    for entry in entries:
        selects = all(
            entry.get(key, default) == asked[key]
            for key, default in REFERENCE_SPEC_DEFAULTS.items()
        )
        if selects and all(entry.get(k) == v for k, v in narrowing.items()) and "accuracy" in entry:
            return float(entry["accuracy"])
    available = [
        {k: e[k] for k in (*REFERENCE_SPEC_DEFAULTS, *narrowing) if k in e} for e in entries
    ]
    raise KeyError(
        f"{path}:{model_key} has no entry matching {wanted or '{}'}. Available "
        f"discriminator sets: {available}"
    )


def run_eval(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    deployment: Any,
    benchmark: Any,
    anchor: Any,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    relative_score: float | None = None,
    dataset_path: str | None = None,
) -> GateResult:
    """Score ``benchmark`` on an aggregated ``deployment`` with ``trtllm-eval``, in a load of its own.

    One job at the deployment's own width: the evaluator loads the model, scores, and
    exits. Every rank's engine is held to the deployment's options (``checks/attest.py``)
    and the score to ``anchor``'s floor (:func:`judged`). ``relative_score`` is the score
    of the deployment a relative anchor names, measured as this one is.
    """
    verdict = verdict_factory(inspect_tree(task_data))
    accuracy = task_data.get("accuracy") or {}
    started = time.monotonic()
    fields: dict[str, Any] = {}

    def ran(**more: Any) -> GateResult:
        return verdict(
            name=f"accuracy[{benchmark.name}]",
            member=deployment.shape,
            duration_s=time.monotonic() - started,
            **{**fields, **more},
        )

    if timeout is None:
        timeout = int(accuracy.get("timeout") or ACCURACY_TIMEOUT_S)
    parts = eval_command(
        task_data, deployment, benchmark, workdir=workdir, dataset_path=dataset_path
    )
    env = {
        EVAL_PROGRESS_ENV: str(EVAL_PROGRESS_EVERY),
        **environment.env,
        **deployment.env,
        **benchmark.env,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    mode = requirement(deployment.server.options)
    if requirement_said(deployment):
        fields["warnings"] = (requirement_said(deployment),)
    launch = with_mode(replace(environment, env=env, world_size=deployment.ranks), mode)
    command = launch.command(" ".join(parts))
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
        {**deployment.server.options, REQUIREMENT_OPTION: mode},
        tree=task_data.get("trtllm_repo_path"),
        target=mode == MODE_REQUIRE,
    )
    fields["engine"] = (attested.record(),)
    if attested.failure():
        return ran(passed=False, detail=attested.failure())
    measured = parse_eval_score(combined, benchmark.score_pattern)
    if measured is None:
        return ran(passed=False, detail=unparsed_score_detail(combined, benchmark.score_pattern))
    return ran(
        **judged(
            measured,
            anchor,
            benchmark,
            accuracy_policy.resolve_floor_policy(accuracy),
            relative_score=relative_score,
            output=combined,
            how=" by `trtllm-eval`, in a load of its own",
        )
    )


def judged(
    measured: float,
    anchor: Any,
    benchmark: Any,
    policy: accuracy_policy.FloorPolicy,
    *,
    relative_score: float | None = None,
    output: str = "",
    how: str = "",
) -> dict[str, Any]:
    """The verdict fields of ``measured`` against the floor ``anchor`` sets; one-sided.

    ``anchor`` and ``benchmark`` are a deployment's (``checks/anchors.py``,
    ``spec/deployments.py``); ``relative_score`` is the score of the deployment a
    relative anchor names, measured as this one is. ``how`` says the way the score was
    taken (through the endpoint, by ``trtllm-eval`` in a load of its own).
    """
    from . import anchors as anchors_mod

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
        anchor, benchmark, policy, relative_score=relative_score, output=output
    )
    if floor is None:
        return dict(
            passed=False,
            measured=measured,
            detail=f"measured {measured:.4f}{how}, but no floor can be computed under it "
            f"({source}). A floor that cannot be computed is not a pass",
        )
    passed = measured >= floor.value
    return dict(
        passed=passed,
        measured=measured,
        reference=floor.reference,
        tolerance=floor.tolerance,
        floor=floor.value,
        se=floor.se,
        samples=floor.samples,
        floor_rule=floor.describe(),
        detail=f"measured {measured:.4f} {'>=' if passed else '<'} {floor.value:.4f}{how} "
        f"({floor.describe()}; count from {source}; anchor: {anchor.kind}, {anchor.source})",
    )


def check_systematic_shift(
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> accuracy_policy.ShiftFinding | None:
    """The check across the deployments scored on one benchmark (:func:`.accuracy_policy.systematic_shift`).

    ``results`` are their accuracy lines, each named by its deployment (``member``), and
    ``unit`` the benchmark. Lines without a measurement or an SE are left out.
    """
    policy = accuracy_policy.resolve_floor_policy(accuracy)
    readings = {
        str(r.member): (float(r.measured), float(r.reference), float(r.se))
        for r in results
        if r.measured is not None and r.reference is not None and r.se is not None
    }
    return accuracy_policy.systematic_shift(
        unit,
        readings,
        min_members=policy.shift_min_members,
        acknowledged=policy.acknowledged,
    )


#: The key a systematic-shift finding is recorded under, per benchmark.
SHIFT_FINDING = "systematic_shift"


def note_systematic_shift(
    records: GateRecords,
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> accuracy_policy.ShiftFinding | None:
    """Run :func:`check_systematic_shift` and record its outcome either way.

    A benchmark with no shift records a resolved entry, so a finding raised on an
    earlier state of the target stops blocking once a later one is clean.
    """
    finding = check_systematic_shift(results, accuracy, unit)
    tree = results[-1].tree_state if results else UNKNOWN_TREE_STATE
    records.append(
        GateRecord(
            kind=KIND_FINDING,
            key={"finding": SHIFT_FINDING, "unit": str(unit)},
            tree=tree,
            passed=finding is None or bool(finding.acknowledged),
            values={
                "detail": finding.describe() if finding else "no systematic shift",
                "members": {
                    str(r.member): {"measured": r.measured, "reference": r.reference, "se": r.se}
                    for r in results
                },
            },
        )
    )
    return finding


def blocking_findings(records: GateRecords, accuracy: Mapping[str, Any]) -> list[GateRecord]:
    """Findings that refuse delivery under the CURRENT ``accuracy.floor.acknowledge``."""
    return open_findings(records, accuracy_policy.resolve_floor_policy(accuracy).acknowledged)


def resolve_model_key(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """The HuggingFace repo name the run's anchor rows are keyed by.

    ``accuracy.model_key`` when the spec states one, otherwise the last two components
    of the checkpoint path (``<org>/<model>``). A benchmark that states its own is read
    by ``checks/anchors.py``.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    explicit = resolve_per_unit(accuracy, "model_key", unit)
    if explicit:
        return str(explicit)
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


__all__ = [
    "ACCURACY_TIMEOUT_S",
    "SHIFT_FINDING",
    "blocking_findings",
    "check_systematic_shift",
    "eval_command",
    "judged",
    "load_reference_score",
    "note_systematic_shift",
    "parse_eval_score",
    "resolve_model_key",
    "run_eval",
    "score_lines",
    "unparsed_score_detail",
]

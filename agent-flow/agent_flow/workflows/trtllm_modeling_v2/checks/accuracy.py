# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy evaluation, reference lookup and cross-member findings."""

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
from ..spec.deployments import Deployment
from ..spec.task_schema import resolve_per_unit, resolve_samples
from . import accuracy_policy, attest
from .check_config import check_configuration, deployment_of
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir, with_mode
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

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


def resolve_discriminators(
    accuracy: Mapping[str, Any], unit: str | None
) -> Mapping[str, Any] | None:
    """Pick the accuracy-reference discriminators for ``unit``; ``None`` when there are none.

    Accepts either a flat mapping (the whole run) or a mapping from unit
    name to mapping, with ``default`` as the fallback key.
    """
    block = accuracy.get("discriminators")
    if not isinstance(block, dict):
        return None
    # Per-unit form is recognised by its values being mappings; a flat form
    # like {"quant_algo": "NVFP4"} has scalar values.
    if any(isinstance(v, dict) for v in block.values()):
        chosen = block.get(unit)
        if chosen is None:
            chosen = block.get("default")
        return chosen if isinstance(chosen, dict) else None
    return block


def eval_command(
    task_data: Mapping[str, Any],
    config: Mapping[str, Any],
    benchmark: str,
    *,
    workdir: str | Path,
    unit: str | None = None,
    member: str | None = None,
) -> list[str]:
    """Public trtllm-eval CLI using the same resolved configuration the verdict attests."""
    accuracy = dict(task_data.get("accuracy") or {})
    options = dict(config["llm"])
    options.pop("model")
    target = gate_workdir(workdir) / f"eval.{unit or 'unit'}.{member or 'member'}.yaml"
    write_text_atomic(target, yaml.safe_dump(options, sort_keys=True), mode=0o600)
    parts = [
        "trtllm-eval",
        shlex.quote(f"--model={task_data['checkpoint_path']}"),
        shlex.quote(f"--config={target}"),
        benchmark,
    ]
    if accuracy.get("dataset_path"):
        parts.append(shlex.quote(f"--dataset_path={accuracy['dataset_path']}"))
    arguments = [str(arg) for arg in resolve_per_unit(accuracy, "benchmark_args", unit) or ()]
    # Logged samples and the inference text stay under the gate's directory, for an audit.
    if benchmark == "gsm8k" and "--log_samples" in arguments:
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
    parts.extend(shlex.quote(arg) for arg in arguments)
    return parts


def eval_harness_command(environment: RunEnvironment, parts: Sequence[str]) -> str:
    """Launch the installed public CLI; the one upstream rank hook observes workers.

    The benchmark prints its running score every :data:`EVAL_PROGRESS_EVERY`
    answers unless the environment already sets how often.
    """
    progress = {EVAL_PROGRESS_ENV: str(EVAL_PROGRESS_EVERY)}
    environment = replace(
        environment,
        env={**progress, **environment.env, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
    )
    return environment.command(" ".join(parts))


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


def load_reference_score(
    reference_yaml: str | Path,
    model_key: str,
    *,
    discriminators: Mapping[str, Any] | None = None,
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    An entry matches when its discriminators (``quant_algo``,
    ``kv_cache_quant_algo``, ``spec_dec_algo``) are exactly the ones asked
    for, no more, no less. Raises ``KeyError`` when no entry, or more than
    one, matches: the anchor is never invented.
    """
    path = Path(reference_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"accuracy reference file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if model_key not in data:
        raise KeyError(
            f"{path} has no entry for {model_key!r}. The anchor comes from a written "
            f"source or from the user — it is never invented. Add the entry (citing "
            f"where the score came from) before the accuracy gate can run: "
            f"`{model_key}:` then `  - accuracy: <score>`, committed to the checkout. A "
            f"score under the gate's own protocol comes from an implementation that already "
            f"serves the checkpoint, e.g. `trtllm-eval --model <checkpoint> <benchmark>` on "
            f"a TensorRT-LLM whose existing model class loads it, on the same GPU type."
        )

    wanted = {k: v for k, v in (discriminators or {}).items() if v is not None}
    entries = data[model_key] or []
    matches = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        present = {k: v for k, v in entry.items() if k != "accuracy"}
        if present == wanted:
            matches.append(entry)

    if not matches:
        available = [
            {k: v for k, v in e.items() if k != "accuracy"} for e in entries if isinstance(e, dict)
        ]
        raise KeyError(
            f"{path}:{model_key} has no entry matching {wanted or '{}'}. Available "
            f"discriminator sets: {available}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"{path}:{model_key} has {len(matches)} entries matching {wanted or '{}'}; "
            f"the anchor must be unambiguous"
        )
    return float(matches[0]["accuracy"])


def resolve_floor(
    accuracy: Mapping[str, Any],
    unit: str | None,
    benchmark: str,
    reference: float,
    output: str = "",
    relative: bool = False,
) -> tuple[accuracy_policy.Floor | None, str]:
    """The floor ``reference`` gets under ``accuracy.floor``, and where its count came from.

    ``relative``: ``reference`` is the unit's own score at this member, on this target.
    """
    policy = accuracy_policy.resolve_floor_policy(accuracy)
    samples, source = resolve_samples(accuracy, unit, benchmark, output)
    return accuracy_policy.floor_for(policy, reference, samples, relative), source


def run_accuracy_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    reference: float,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    unit: str | None = None,
    member: str | None = None,
    relative: bool = False,
    deployment: Deployment | None = None,
) -> GateResult:
    """Run ``trtllm-eval`` on ``deployment``'s options and compare against the anchor. One-sided.

    Without ``deployment`` it is the one ``unit`` at ``member`` names.

    The floor is ``reference - k * SE`` by default (:func:`resolve_floor`), a
    fixed ``reference - tolerance`` under ``accuracy.floor.rule: tolerance``.
    ``relative``: ``reference`` is the unit's own score at this member on this
    target (a configuration variant's ``anchor: relative``), and the SE is that
    of the difference. Every rank's engine is held to the deployment's options
    (``checks/attest.py``).
    """
    accuracy = dict(task_data.get("accuracy") or {})
    timeout = int(accuracy.get("timeout", ACCURACY_TIMEOUT_S)) if timeout is None else timeout
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    tolerance = float(accuracy.get("tolerance", 0.0))
    env = with_mode(environment, MODE_REQUIRE)
    _verdict = verdict_factory(inspect_tree(task_data))
    planned, _ = resolve_floor(accuracy, unit, benchmark, reference, relative=relative)

    def floor_fields(floor: accuracy_policy.Floor | None) -> dict[str, Any]:
        if floor is None:
            return {}
        return {
            "floor": floor.value,
            "se": floor.se,
            "samples": floor.samples,
            "floor_rule": floor.describe(),
        }

    name = f"accuracy[{benchmark}]"
    common = dict(name=name, member=member, reference=reference, tolerance=tolerance)
    if deployment is None:
        deployment = deployment_of(task_data, unit, member, environment.world_size)
    config = check_configuration(task_data, deployment, max_tokens=64)
    config["llm"]["modeling_v2"] = MODE_REQUIRE
    parts = eval_command(task_data, config, benchmark, unit=unit, member=member, workdir=workdir)
    command = eval_harness_command(env, parts)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(command=command, duration_s=elapsed, stdout_tail=tail, **common)

    if result.infrastructure:
        return _verdict(**lost_session(result), **ran, **floor_fields(planned))
    if result.timed_out:
        return _verdict(
            passed=False, detail=ended_early(result, timeout), **ran, **floor_fields(planned)
        )
    if not result.ok:
        return _verdict(
            passed=False,
            detail=f"trtllm-eval exited {result.returncode}",
            **ran,
            **floor_fields(planned),
        )
    options = dict(config["llm"])
    options.pop("model")
    attested = attest.attest(combined, options, tree=task_data.get("trtllm_repo_path"))
    ran["engine"] = (attested.record(),)
    failure = attested.failure()
    if failure:
        return _verdict(passed=False, detail=failure, **ran, **floor_fields(planned))

    pattern = resolve_per_unit(accuracy, "score_pattern", unit)
    measured = parse_eval_score(combined, pattern)
    if measured is None:
        return _verdict(
            passed=False,
            detail=unparsed_score_detail(combined, pattern),
            **ran,
            **floor_fields(planned),
        )

    floor, source = resolve_floor(accuracy, unit, benchmark, reference, combined, relative)
    if floor is None:
        return _verdict(
            passed=False,
            detail=f"measured {measured:.4f}, but the statistical floor needs the sample "
            f"count and there is {source}. A floor that cannot be computed is not a pass",
            measured=measured,
            **ran,
        )
    passed = measured >= floor.value
    return _verdict(
        passed=passed,
        detail=(
            f"measured {measured:.4f} {'>=' if passed else '<'} {floor.value:.4f} "
            f"({floor.describe()}; count from {source}); score validation only; "
            + f"every rank's engine matched its {attested.keys} declared option(s)"
        ),
        measured=measured,
        **ran,
        **floor_fields(floor),
    )


def resolve_reference(
    task_data: Mapping[str, Any],
    *,
    model_key: str,
    discriminators: Mapping[str, Any] | None = None,
    unit: str | None = None,
) -> float:
    """Look up this run's anchor, resolving the reference path against the repo.

    ``discriminators`` overrides ``accuracy.discriminators`` for ``unit``.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    reference_yaml = Path(str(resolve_per_unit(accuracy, "reference_yaml", unit)))
    if not reference_yaml.is_absolute():
        reference_yaml = Path(str(task_data["trtllm_repo_path"])) / reference_yaml
    if discriminators is None:
        discriminators = resolve_discriminators(accuracy, unit)
    return load_reference_score(reference_yaml, model_key, discriminators=discriminators)


def resolve_model_key(task_data: Mapping[str, Any], unit: str | None = None) -> str:
    """The HuggingFace repo name this unit's anchor row is keyed by.

    An explicit ``accuracy.model_key`` when the spec states one (per unit,
    :func:`~.spec.task_schema.resolve_per_unit`), otherwise the last two components of the
    checkpoint path (``<org>/<model>``). This is the one implementation: the
    workflow's ``_model_key`` delegates here, so preflight's anchor check and
    the gate resolve the same key.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    explicit = resolve_per_unit(accuracy, "model_key", unit)
    if explicit:
        return str(explicit)
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


__all__ = [
    "resolve_model_key",
    "ACCURACY_TIMEOUT_S",
    "eval_command",
    "eval_harness_command",
    "load_reference_score",
    "parse_eval_score",
    "resolve_discriminators",
    "resolve_floor",
    "resolve_reference",
    "run_accuracy_gate",
    "score_lines",
    "unparsed_score_detail",
]

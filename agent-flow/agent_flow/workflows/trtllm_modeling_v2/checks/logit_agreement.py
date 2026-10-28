# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The logit gate: the unit's engine predicts what the Hugging Face reference predicts, position by position.

The reference (``checks/logit_reference.py``) holds the HF implementation's
top-k at every prompt position and at each of N greedy continuation steps, and
how far it moves from its own fp32 self (its *spread*). The engine is the boot
gate's, without the speculator (prompt log-probs need every context row);
``harness/logit_driver.py`` is the job. Compared at every position whose target is not
a media placeholder, up to the first continuation step where the greedy tokens
differ, each number is held to the reference's spread times
``threshold_factor``:

- top-1 agreement, pooled: the disagreement rate at most factor times the
  reference's own;
- mean |delta log-prob| of the reference's top-1 token, per prompt: at most
  factor times that prompt's spread, or the pooled spread when that is larger;
- |delta log-prob| at every position: at most factor times the pooled
  spread's largest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..checks.logit_reference import (
    Agreement,
    Reference,
    ReferenceError,
    ReferencePrompt,
    file_digest,
    load_reference,
)
from ..infra.compute.environment import RunEnvironment
from .check_config import (
    EngineLimits,
    check_configuration,
    member_llm_api_overlay,
    speculation_flags,
)
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    engine_driver_command,
    gate_workdir,
    with_mode,
)
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

GATE_NAME = "logits"


#: The job, run as a file under the measured tree's interpreter.
LOGIT_DRIVER = Path(__file__).resolve().parent / "harness" / "logit_driver.py"


DEFAULT_THRESHOLD_FACTOR = 3.0


#: The least each limit is before the factor, so a reference whose two
#: precisions happened to agree exactly still leaves room for rounding.
MIN_DISAGREEMENT = 0.001


MIN_MEAN_DELTA = 0.005


MIN_MAX_DELTA = 0.05


#: What of a member's served engine this gate does not run (:func:`unseen`), by kind.
UNSEEN_ATTENTION_DP = "attention_dp"


UNSEEN_SPECULATION = "speculation"


@dataclass(frozen=True)
class LogitsSpec:
    """``accuracy.logits``: whether the contract owes the logit gate, and against what."""

    enabled: bool = False
    reference: str = ""
    #: The reference prompts to send; empty for every one the unit can take.
    prompts: tuple[str, ...] = ()
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR
    timeout: int = BOOT_TIMEOUT_S

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any]) -> "LogitsSpec":
        block = dict((task_data.get("accuracy") or {}).get("logits") or {})
        reference = str(block.get("reference") or "")
        default = cls()
        return cls(
            enabled=bool(reference) and bool(block.get("enabled", True)),
            reference=reference,
            prompts=tuple(str(p) for p in block.get("prompts") or ()),
            threshold_factor=float(block.get("threshold_factor") or default.threshold_factor),
            timeout=int(block.get("timeout") or default.timeout),
        )


def unseen(member: str, features: Sequence[str]) -> dict[str, str]:
    """What of ``member``'s served engine this gate does not run, by kind, and why.

    The contract leaves a member's benchmark out only where nothing here is
    unseen (``run/validation/contract.py``).
    """
    kinds: dict[str, str] = {}
    if member_llm_api_overlay(member).get("enable_attention_dp"):
        kinds[UNSEEN_ATTENTION_DP] = (
            "attention data parallelism: the logit gate sends one prompt at a time, which one "
            "attention-DP rank serves"
        )
    if speculation_flags(features):
        kinds[UNSEEN_SPECULATION] = "the speculator: the logit gate builds the engine without it"
    return kinds


@dataclass(frozen=True)
class Selection:
    """The reference prompts a unit is sent, why any were left out, and what stops the gate."""

    prompts: tuple[ReferencePrompt, ...] = ()
    notes: tuple[str, ...] = ()
    error: str = ""


def select_prompts(
    reference: Reference, spec: LogitsSpec, *, multimodal: bool, limits: EngineLimits
) -> Selection:
    """The prompts ``spec`` names (every one by default) that this unit's engine can take.

    An image prompt goes only to a unit whose target has a vision path. A
    prompt longer than the engine's sequence limit, or than one context
    chunk without chunked prefill, is left out; both are named in a note.
    """
    if spec.prompts:
        unknown = [p for p in spec.prompts if reference.prompt(p) is None]
        if unknown:
            known = [p.id for p in reference.prompts]
            return Selection(
                error=f"accuracy.logits.prompts names {unknown}, which {reference.path} does "
                f"not hold (it holds {known})"
            )
        chosen = [reference.prompt(p) for p in spec.prompts]
    else:
        chosen = list(reference.prompts)
    kept: list[ReferencePrompt] = []
    notes: list[str] = []
    for prompt in chosen:
        refused = limits.refuses(len(prompt.input_ids), reference.continuation)
        if prompt.images and not multimodal:
            notes.append(f"{prompt.id} left out: an image prompt, and this unit has no vision path")
        elif refused:
            notes.append(f"{prompt.id} left out: {refused}")
        else:
            kept.append(prompt)
    if not kept:
        return Selection(notes=tuple(notes), error="no reference prompt fits this unit")
    if multimodal and not any(p.images for p in kept):
        notes.append(
            "no image prompt: this unit's vision path is not compared; make the reference with "
            "--image (checks/logit_reference.py), which the requirements matrix counts as what "
            "judges a multimodal unit"
        )
    if limits.chunked_prefill and not any(len(p.input_ids) > limits.max_num_tokens for p in kept):
        notes.append(
            f"no prompt crosses a context chunk (max_num_tokens {limits.max_num_tokens}); make "
            f"the reference's long prompt longer"
        )
    return Selection(prompts=tuple(kept), notes=tuple(notes))


def image_path(path: str, repo_root: str | Path | None) -> Path:
    """An image the reference names: as is when absolute, else in the tree under test."""
    candidate = Path(path)
    if candidate.is_absolute() or repo_root is None:
        return candidate
    return Path(repo_root) / candidate


def image_problems(prompts: Sequence[ReferencePrompt], repo_root: str | Path | None) -> list[str]:
    """Every image the prompts send that is missing or is not the file the reference was made with."""
    problems = []
    for prompt in prompts:
        for image in prompt.images:
            path = image_path(image.path, repo_root)
            if not path.is_file():
                problems.append(f"{prompt.id}: image {path} does not exist")
            elif image.sha256 and file_digest(path) != image.sha256:
                problems.append(
                    f"{prompt.id}: {path} is not the image the reference was made with "
                    f"(a git-lfs pointer, or another file)"
                )
    return problems


def driver_requests(
    prompts: Sequence[ReferencePrompt], repo_root: str | Path | None
) -> list[dict[str, Any]]:
    """The driver's requests: a text prompt as its token ids, an image prompt as text and images."""
    requests = []
    for prompt in prompts:
        if prompt.images:
            requests.append(
                {
                    "id": prompt.id,
                    "prompt": prompt.text,
                    "images": [str(image_path(i.path, repo_root)) for i in prompt.images],
                    "render": "raw",
                }
            )
        else:
            requests.append({"id": prompt.id, "token_ids": list(prompt.input_ids)})
    return requests


def write_driver_spec(
    path: str | Path,
    *,
    task_data: Mapping[str, Any],
    config: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    max_tokens: int,
    top_k: int,
    out: str | Path,
) -> Path:
    """Write the driver's spec (see ``harness/logit_driver.py``)."""
    spec = {
        **config,
        "requests": [dict(r) for r in requests],
        "max_tokens": int(max_tokens),
        "top_k": int(top_k),
        "out": str(out),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def driver_command(environment: RunEnvironment, spec_path: str | Path) -> str:
    """The driver's launch over ``spec_path``, as engine work, using the shared public constructor options."""
    return engine_driver_command(environment, spec_path, script=LOGIT_DRIVER)


# ------------------------------------------------------------------ the comparison


@dataclass(frozen=True)
class Position:
    """One compared position: did the top-1 agree, and how far the reference top-1's log-prob moved."""

    position: int
    agreed: bool
    delta: float
    reference_token: int
    reference_logprob: float
    engine_token: int | None
    engine_logprob: float | None
    #: True when the reference's top-1 is not among the engine's log-probs:
    #: ``delta`` is then a lower bound, from the engine's k-th log-prob.
    bounded: bool = False


def compare_position(
    position: int, top_ids: Sequence[int], top_logprobs: Sequence[float], entries: Sequence[Any]
) -> Position | None:
    """The reference's top-1 at ``position`` against the engine's ``[id, log-prob, rank]`` entries.

    The engine returns its top-k (k is the reference's, ``len(top_ids)``)
    and, when it is not among them, the log-prob of the token that actually
    follows. A tie with the engine's best counts as agreeing. A reference
    top-1 outside the engine's top-k is at most its k-th log-prob, which
    bounds the delta from below. ``None`` when the engine returned no usable
    entry there.
    """
    found = {int(e[0]): float(e[1]) for e in entries if e and e[0] is not None}
    if not found:
        return None
    best_token = max(found, key=found.__getitem__)
    best = found[best_token]
    token, logprob = int(top_ids[0]), float(top_logprobs[0])
    if token in found:
        return Position(
            position,
            found[token] >= best,
            abs(found[token] - logprob),
            token,
            logprob,
            best_token,
            found[token],
        )
    ranked = sorted(found.values(), reverse=True)
    kth = ranked[min(len(top_ids), len(ranked)) - 1]
    return Position(
        position, False, max(logprob - kth, 0.0), token, logprob, best_token, None, bounded=True
    )


@dataclass(frozen=True)
class PromptComparison:
    """One prompt: its compared positions, where the continuation diverged, or why it could not be read."""

    prompt: ReferencePrompt
    positions: tuple[Position, ...] = ()
    #: The continuation step at which the engine's greedy token first differed.
    diverged_at: int | None = None
    problem: str = ""

    @property
    def agreement(self) -> Agreement:
        return Agreement.of(
            self.prompt.id, ((p.position, p.agreed, p.delta) for p in self.positions)
        )

    def worst(self) -> Position | None:
        return max(self.positions, key=lambda p: p.delta, default=None)


def compare_prompt(
    prompt: ReferencePrompt, record: Mapping[str, Any] | None, media: frozenset[int]
) -> PromptComparison:
    """The engine's record for ``prompt`` against the reference, at every scored position.

    The engine must have seen the reference's prompt: as many tokens, the
    same text tokens (an image's placeholders may carry other ids).
    """
    if record is None:
        return PromptComparison(prompt, problem="the engine returned no record")
    engine_ids = [int(t) for t in record.get("prompt_token_ids") or ()]
    length = len(prompt.input_ids)
    if len(engine_ids) != length:
        cause = " (its image expanded to another number of placeholders)" if prompt.images else ""
        return PromptComparison(
            prompt,
            problem=f"the engine's prompt is {len(engine_ids)} tokens and the reference's "
            f"{length}{cause}",
        )
    for p, (ours, theirs) in enumerate(zip(prompt.input_ids, engine_ids)):
        if ours not in media and ours != theirs:
            return PromptComparison(
                prompt,
                problem=f"prompt token {p} is {theirs} in the engine and {ours} in the reference",
            )
    prompt_rows = record.get("prompt_logprobs") or []
    if len(prompt_rows) < prompt.prompt_positions:
        return PromptComparison(
            prompt,
            problem=f"the engine returned {len(prompt_rows)} prompt log-probs for "
            f"{prompt.prompt_positions} positions",
        )
    sequence = prompt.sequence
    compared: list[Position] = []
    missing: list[int] = []

    def add(position: int, entries: Sequence[Any]) -> None:
        if sequence[position + 1] in media:
            return
        row = compare_position(
            position, prompt.top_ids[position], prompt.top_logprobs[position], entries
        )
        if row is None:
            missing.append(position)
        else:
            compared.append(row)

    for position in range(prompt.prompt_positions):
        add(position, prompt_rows[position])
    generated = [int(t) for t in record.get("token_ids") or ()]
    step_rows = record.get("logprobs") or []
    diverged = None
    for step, token in enumerate(prompt.continuation_ids):
        if step >= len(generated) or step >= len(step_rows):
            break
        add(prompt.prompt_positions + step, step_rows[step])
        if generated[step] != token:
            diverged = step
            break
    if missing:
        return PromptComparison(
            prompt,
            tuple(compared),
            diverged,
            problem=f"no log-probs at {len(missing)} position(s), the first {missing[0]}",
        )
    return PromptComparison(prompt, tuple(compared), diverged)


@dataclass(frozen=True)
class Limits:
    """The thresholds a comparison is held to: the reference's spread times the factor."""

    factor: float
    disagreement: float
    max_delta: float
    pooled_mean: float

    @classmethod
    def of(cls, reference: Reference, factor: float) -> "Limits":
        spread = reference.spread
        return cls(
            factor=factor,
            disagreement=factor * max(1.0 - spread.top1_agreement, MIN_DISAGREEMENT),
            max_delta=factor * max(spread.max_abs_delta, MIN_MAX_DELTA),
            pooled_mean=spread.mean_abs_delta,
        )

    def mean_for(self, prompt: ReferencePrompt) -> float:
        """The mean |delta| a prompt may reach: its own spread, or the pooled one when larger."""
        return self.factor * max(prompt.spread.mean_abs_delta, self.pooled_mean, MIN_MEAN_DELTA)


def _where(prompt: ReferencePrompt, position: int) -> str:
    step = prompt.step(position)
    kind = f"continuation step {step}" if step is not None else "prompt"
    return f"{prompt.id}@{position} ({kind})"


def describe_position(prompt: ReferencePrompt, row: Position) -> str:
    """One position, as a verdict quotes it."""
    if row.bounded:
        engine = (
            f"not in the engine's top-k (its top-1 {row.engine_token}); |delta| >= {row.delta:.4f}"
        )
    else:
        engine = f"engine {row.engine_logprob:.4f}, |delta| {row.delta:.4f}"
        if not row.agreed:
            engine += f", engine top-1 {row.engine_token}"
    return (
        f"{_where(prompt, row.position)}: reference top-1 {row.reference_token} at "
        f"{row.reference_logprob:.4f}, {engine}"
    )


@dataclass(frozen=True)
class Judgement:
    passed: bool
    detail: str
    breakdown: str = ""


def judge(
    reference: Reference,
    comparisons: Sequence[PromptComparison],
    factor: float = DEFAULT_THRESHOLD_FACTOR,
) -> Judgement:
    """Hold the comparisons to the reference's spread times ``factor``; quote the worst position."""
    limits = Limits.of(reference, factor)
    failures: list[str] = []
    for c in comparisons:
        if c.problem:
            failures.append(f"{c.prompt.id}: {c.problem}")
    pooled = Agreement.pooled(c.agreement for c in comparisons)
    disagreement = 1.0 - pooled.top1_agreement
    if pooled.positions and disagreement > limits.disagreement:
        failures.append(
            f"top-1 agreement {pooled.top1_agreement:.2%} over {pooled.positions} positions, "
            f"under the floor {1 - limits.disagreement:.2%}"
        )
    means = []
    for c in comparisons:
        agreement = c.agreement
        if not agreement.positions:
            continue
        limit = limits.mean_for(c.prompt)
        means.append(
            (agreement.mean_abs_delta / limit, c.prompt.id, agreement.mean_abs_delta, limit)
        )
        if agreement.mean_abs_delta > limit:
            failures.append(
                f"{c.prompt.id}: mean |delta log-prob| {agreement.mean_abs_delta:.4f} over "
                f"{agreement.positions} positions, over its limit {limit:.4f}"
            )
    worst = max(
        ((c, c.worst()) for c in comparisons if c.positions),
        key=lambda item: item[1].delta,
        default=(None, None),
    )
    over = sum(1 for c in comparisons for p in c.positions if p.delta > limits.max_delta)
    if over:
        failures.append(f"{over} position(s) with |delta log-prob| over {limits.max_delta:.4f}")
    if not pooled.positions and not failures:
        failures.append("no position was compared")

    worst_line = (
        f"worst position {describe_position(worst[0].prompt, worst[1])}"
        if worst[1] is not None
        else "no position compared"
    )
    mean_line = ""
    if means:
        _, prompt_id, mean, limit = max(means)
        mean_line = f"mean |delta| up to {mean:.4f} ({prompt_id}, limit {limit:.4f}); "
    numbers = (
        f"{len(comparisons)} prompt(s), {pooled.positions} positions against "
        f"{Path(reference.path).name} at {factor:g}x its {reference.dtype}-vs-"
        f"{reference.spread_dtype} spread: top-1 agreement {pooled.top1_agreement:.2%} (floor "
        f"{1 - limits.disagreement:.2%}); {mean_line}max |delta| {pooled.max_abs_delta:.4f} "
        f"(limit {limits.max_delta:.4f})"
    )
    breakdown = "; ".join(_prompt_line(c, limits) for c in comparisons)
    if failures:
        return Judgement(False, f"{'; '.join(failures)}; {worst_line}. {numbers}", breakdown)
    return Judgement(True, f"{numbers}; {worst_line}", breakdown)


def _prompt_line(comparison: PromptComparison, limits: Limits) -> str:
    if comparison.problem and not comparison.positions:
        return f"{comparison.prompt.id}: {comparison.problem}"
    a = comparison.agreement
    line = (
        f"{comparison.prompt.id}: {a.positions} pos, top-1 {a.top1_agreement:.2%}, mean "
        f"{a.mean_abs_delta:.4f} (limit {limits.mean_for(comparison.prompt):.4f}), max "
        f"{a.max_abs_delta:.4f}"
    )
    if comparison.diverged_at is not None:
        line += f", diverged at step {comparison.diverged_at}"
    return line


def read_records(path: str | Path) -> dict[str, dict[str, Any]] | None:
    """The driver's records by prompt id; ``None`` when it wrote none."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return None
    return {str(r.get("id")): r for r in records if isinstance(r, dict)}


# ------------------------------------------------------------------------ the gate


def run_logit_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    features: Any | None = None,
    unit: str | None = None,
    member: str | None = None,
    spec: LogitsSpec | None = None,
    reference: Reference | None = None,
) -> GateResult:
    """Run the reference's prompts through the unit's engine and compare every position.

    ``features`` are the features the target implements (default the run's):
    the speculator among them is left out, and ``multimodal`` lets the image
    prompts in. ``reference`` defaults to the file ``spec`` names.
    """
    spec = spec or LogitsSpec.from_task(task_data)
    _verdict = verdict_factory(inspect_tree(task_data))

    def refuse(detail: str) -> GateResult:
        return _verdict(name=GATE_NAME, member=member, passed=False, detail=detail)

    if reference is None:
        if not spec.reference:
            return refuse("accuracy.logits names no reference file")
        try:
            reference = load_reference(spec.reference)
        except ReferenceError as exc:
            return refuse(str(exc))
    if not reference.spread.positions:
        return refuse(
            f"{reference.path} records no spread, so the gate has no threshold to hold the "
            f"engine to"
        )

    implemented = [
        str(f) for f in (task_data.get("features") if features is None else features) or ()
    ]
    plain = [f for f in implemented if not speculation_flags([f])]
    multimodal = "multimodal" in plain and task_data.get("scope", "full") != "text_only"
    config = check_configuration(
        task_data,
        environment,
        unit=unit,
        features=plain,
        max_tokens=reference.continuation,
        member=member,
    )
    selection = select_prompts(
        reference, spec, multimodal=multimodal, limits=EngineLimits.from_config(config)
    )
    if selection.error:
        return refuse("; ".join([selection.error, *selection.notes]))
    repo_root = task_data.get("trtllm_repo_path")
    problems = image_problems(selection.prompts, repo_root)
    if problems:
        return refuse("; ".join(problems))

    env = with_mode(environment, MODE_REQUIRE)
    directory = gate_workdir(workdir)
    tag = f"{unit or 'unit'}.{member or 'member'}"
    out = directory / f"logits.{tag}.engine.json"
    out.unlink(missing_ok=True)
    spec_path = write_driver_spec(
        directory / f"logits.{tag}.json",
        task_data=task_data,
        config=config,
        requests=driver_requests(selection.prompts, repo_root),
        max_tokens=reference.continuation,
        top_k=reference.top_k,
        out=out,
    )
    command = driver_command(env, spec_path)

    started = time.monotonic()
    result = runner(command, None, spec.timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        name=GATE_NAME,
        member=member,
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
        warnings=selection.notes,
    )
    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(passed=False, detail=f"timed out after {spec.timeout}s", **ran)
    if not result.ok:
        return _verdict(passed=False, detail=f"exited {result.returncode}", **ran)
    records = read_records(out)
    if records is None:
        return _verdict(
            passed=False, detail=f"the driver exited cleanly and wrote no records at {out}", **ran
        )
    comparisons = [
        compare_prompt(p, records.get(p.id), reference.media_token_ids) for p in selection.prompts
    ]
    judgement = judge(reference, comparisons, spec.threshold_factor)
    return _verdict(
        passed=judgement.passed,
        detail=judgement.detail,
        breakdown=judgement.breakdown,
        explained=True,
        **ran,
    )


__all__ = [
    "DEFAULT_THRESHOLD_FACTOR",
    "GATE_NAME",
    "Judgement",
    "LOGIT_DRIVER",
    "Limits",
    "LogitsSpec",
    "MIN_DISAGREEMENT",
    "MIN_MAX_DELTA",
    "MIN_MEAN_DELTA",
    "Position",
    "PromptComparison",
    "Selection",
    "UNSEEN_ATTENTION_DP",
    "UNSEEN_SPECULATION",
    "compare_position",
    "compare_prompt",
    "describe_position",
    "driver_command",
    "driver_requests",
    "image_path",
    "image_problems",
    "judge",
    "read_records",
    "run_logit_gate",
    "select_prompts",
    "unseen",
    "write_driver_spec",
]

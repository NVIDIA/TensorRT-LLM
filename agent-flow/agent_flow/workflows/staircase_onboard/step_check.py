# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler equivalence: decode-time defects caught without a reference implementation.

THE DEFECT THIS EXISTS FOR

A target read a per-request length its generation steps must not trust: under
the overlap scheduler with speculation, the host prepares step ``n + 1``
before step ``n``'s acceptance is known, and one field of the step metadata
was the host's count from before the correction. Generation steps read and
wrote history at positions ahead of their rows by the number of drafts the
previous step rejected. Context steps were right; runs without the overlap
scheduler were right; runs without speculation were right. Every member
scored within 0.3 points of the anchor, every witness compared the engine
with itself under the same scheduler, and the defect passed every gate for
two days.

THE CHECK

The overlap scheduler and greedy speculation do not change WHAT is computed,
only when the host prepares it. So at batch 1, greedy, one request at a time,
the same prompts must produce the same tokens:

========================  =================================  =================
leg                       configuration                      compared with
========================  =================================  =================
``overlap``               the unit's runtime, as declared,   --
                          speculation on when the target
                          has it
``no_overlap``            the same, overlap scheduler OFF    ``overlap``: token
                                                             identity (exact)
``no_spec``               overlap ON, no speculator          ``overlap``: by
                                                             ``spec_pair``
========================  =================================  =================

The ``no_overlap`` pair is exact by construction: every step runs the same
kernels on the same shapes, so any difference is state the overlap path
reads stale. The ``no_spec`` pair is not: a verify step runs ``k + 1`` rows
where a plain step runs one, and a GEMM's bits depend on its shape, so a
near-tie can flip. It is reported by default (``spec_pair: report``), held to
a minimum agreement prefix (``agreement``) or to identity (``exact``) when
the task declares it.

What makes the comparison exact rather than statistical: one request at a
time (``max_batch_size`` 1), the same prompts in the same order on every leg,
the AutoTuner's tactic cache pinned to one file across the legs
(``TLLM_AUTOTUNER_CACHE_PATH``), and at least 64 generated tokens per prompt
(``ignore_eos`` on by default), so a defect in generation steps has steps to
show in.

THE OPTIONAL STEP TRACE

Token identity says THAT a generation step went wrong. A target that
implements the step-trace hook also says WHERE. The hook is optional, and a
target without it is judged on tokens alone.

* Every leg runs with ``STAIRCASE_STEP_TRACE_DIR=<dir>`` (one directory per
  leg). A target that implements the hook writes ``<dir>/rank<R>.jsonl``, one
  JSON object per forward step per request it served, in step order::

      {
          "request": 0,
          "step": 3,
          "phase": "generation",
          "positions": [131, 132, 133],
          "tokens": [17, 9, 4],
          "digest": "9f2c...",
      }

  ``request`` is the request's ordinal in the workload (requests run one at
  a time, so the order is the workload's); ``step`` counts that request's
  forward steps from 0 (its context step). ``positions`` are the position ids
  the step used for that request's rows. ``digest`` is any target-chosen
  hash of state the step consumed that is NOT a position -- the n-gram
  history an embedding table was indexed with, the cache slots written --
  so a stale read shows as a digest mismatch even when positions agree.
  Every field except ``request`` and ``step`` is optional; only fields both
  legs print are compared.
* The witness compares the ``overlap`` and ``no_overlap`` traces of rank 0
  step by step over the steps both legs ran (the overlap scheduler may run
  one extra, discarded step per request) and names the first step whose
  positions or digest differ.
* A REFERENCE ADAPTER may be declared (``step_trace.reference_command``): a
  command run once after the ``overlap`` leg, given
  ``STAIRCASE_STEP_TRACE_ACCEPTED=<that leg's trace directory>`` and
  ``STAIRCASE_STEP_TRACE_RECORDS=<that leg's records, JSON lines>``, which
  replays the accepted tokens through a reference implementation and writes
  a trace of its own to ``STAIRCASE_STEP_TRACE_DIR``. The target's trace is
  then compared with it in the same way.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environment import RunEnvironment
from .gates import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    GateResult,
    ShellRunner,
    _default_shell_runner,
    _verdict_factory,
    _with_mode,
    engine_driver_command,
    gate_workdir,
    inspect_tree,
    parse_driver_records,
    quickstart_words,
    resolve_runtime,
    speculation_flags,
    write_driver_spec,
)
from .harness import workload_io
from .task_schema import SCHEDULER_EQUIVALENCE_KEYS, TaskSchemaError, load_prompts_file

#: The leg names, in the order they run.
LEG_OVERLAP = "overlap"
LEG_NO_OVERLAP = "no_overlap"
LEG_NO_SPEC = "no_spec"
CONTRAST_LEGS: tuple[str, ...] = (LEG_NO_OVERLAP, LEG_NO_SPEC)

#: How a pair of legs is judged.
PAIR_EXACT = "exact"
PAIR_AGREEMENT = "agreement"
PAIR_REPORT = "report"
PAIR_POLICIES: tuple[str, ...] = (PAIR_EXACT, PAIR_AGREEMENT, PAIR_REPORT)

#: Generated tokens per prompt, by default and at the least.
DEFAULT_MAX_TOKENS = 128
MIN_GENERATED_TOKENS = 64
#: Prompts drawn from the accuracy dataset when the task declares none.
DEFAULT_NUM_PROMPTS = 8
#: The prefix a ``agreement`` pair must share, in tokens.
DEFAULT_AGREEMENT_TOKENS = 16

#: The step-trace hook's environment, see the module docstring.
STEP_TRACE_ENV = "STAIRCASE_STEP_TRACE_DIR"
STEP_TRACE_ACCEPTED_ENV = "STAIRCASE_STEP_TRACE_ACCEPTED"
STEP_TRACE_RECORDS_ENV = "STAIRCASE_STEP_TRACE_RECORDS"
#: Pins the AutoTuner's tactics to one file across the legs.
AUTOTUNER_CACHE_ENV = "TLLM_AUTOTUNER_CACHE_PATH"
#: The fields of a trace record the witness compares, when both legs print them.
TRACE_FIELDS: tuple[str, ...] = ("positions", "tokens", "digest")

#: ``accuracy.scheduler_equivalence`` keys, owned by the schema.
CONFIG_KEYS = SCHEDULER_EQUIVALENCE_KEYS


@dataclass(frozen=True)
class EquivalenceConfig:
    """``accuracy.scheduler_equivalence``, resolved."""

    enabled: bool = True
    #: Units the witness runs in; ``None`` is every unit.
    units: tuple[str, ...] | None = None
    prompts: tuple[str, ...] = ()
    dataset: Mapping[str, Any] | None = None
    source: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    ignore_eos: bool = True
    min_generated_tokens: int = MIN_GENERATED_TOKENS
    legs: tuple[str, ...] = CONTRAST_LEGS
    spec_pair: str = PAIR_REPORT
    agreement_tokens: int = DEFAULT_AGREEMENT_TOKENS
    pin_autotuner: bool = True
    timeout: int = BOOT_TIMEOUT_S
    reference_command: str | None = None

    def applies_to(self, unit: str | None) -> bool:
        """Whether the witness runs in ``unit``."""
        return self.enabled and (self.units is None or str(unit) in self.units)

    def requests(self) -> list[dict[str, Any]]:
        """The declared prompts as engine-driver requests."""
        return [
            {"id": f"e{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]


def applies(accuracy: Mapping[str, Any], unit: str | None) -> bool:
    """Whether ``accuracy.scheduler_equivalence`` runs the witness in ``unit`` (default: every unit)."""
    block = accuracy.get("scheduler_equivalence")
    block = block if isinstance(block, Mapping) else {}
    units = block.get("units")
    return bool(block.get("enabled", True)) and (not units or str(unit) in [str(u) for u in units])


def resolve_equivalence_config(accuracy: Mapping[str, Any]) -> EquivalenceConfig:
    """Read ``accuracy.scheduler_equivalence``; every key is optional.

    With no prompts declared, the workload is the first ``num_prompts``
    (default 8) questions of the accuracy dataset, as the acceptance gate
    draws them. Raises :class:`~.task_schema.TaskSchemaError` when a declared
    prompts file cannot be read, or there is no workload at all.
    """
    from .gates import acceptance_dataset

    block = accuracy.get("scheduler_equivalence")
    block = dict(block) if isinstance(block, Mapping) else {}
    prompts: tuple[str, ...] = ()
    dataset = None
    source = ""
    if block.get("prompts"):
        prompts, source = tuple(str(p) for p in block["prompts"]), "scheduler_equivalence.prompts"
    elif block.get("prompts_file"):
        prompts = load_prompts_file(block["prompts_file"])
        source = f"scheduler_equivalence.prompts_file ({block['prompts_file']})"
    else:

        def setting(key: str) -> Any:
            if key == "num_prompts":
                return block.get(key) or DEFAULT_NUM_PROMPTS
            return block.get(key)

        dataset, source = acceptance_dataset(accuracy, None, setting)
        if dataset is None:
            raise TaskSchemaError(
                "no scheduler-equivalence workload: declare "
                "`accuracy.scheduler_equivalence.prompts`, `prompts_file` or `dataset`"
            )
    units = block.get("units")
    step_trace = block.get("step_trace") or {}
    return EquivalenceConfig(
        enabled=bool(block.get("enabled", True)),
        units=tuple(str(u) for u in units) if units else None,
        prompts=prompts,
        dataset=dataset,
        source=source,
        max_tokens=int(block.get("max_tokens") or DEFAULT_MAX_TOKENS),
        ignore_eos=bool(block.get("ignore_eos", True)),
        min_generated_tokens=int(block.get("min_generated_tokens") or MIN_GENERATED_TOKENS),
        legs=tuple(block.get("legs") or CONTRAST_LEGS),
        spec_pair=str(block.get("spec_pair") or PAIR_REPORT),
        agreement_tokens=int(block.get("agreement_tokens") or DEFAULT_AGREEMENT_TOKENS),
        pin_autotuner=bool(block.get("pin_autotuner", True)),
        timeout=int(block.get("timeout") or BOOT_TIMEOUT_S),
        reference_command=step_trace.get("reference_command") if step_trace else None,
    )


# ------------------------------------------------------------------ comparison


@dataclass(frozen=True)
class PairComparison:
    """Two legs' token ids, prompt by prompt."""

    a: str
    b: str
    policy: str
    #: prompt index -> first differing token, for the prompts that differ.
    differences: Mapping[int, int]
    compared: int
    agreement_tokens: int = DEFAULT_AGREEMENT_TOKENS

    @property
    def identical(self) -> bool:
        """Every compared prompt produced the same token ids."""
        return self.compared > 0 and not self.differences

    @property
    def passed(self) -> bool:
        """Whether the pair meets its policy."""
        if self.policy == PAIR_EXACT:
            return self.identical
        if self.policy == PAIR_AGREEMENT:
            return self.compared > 0 and all(
                at >= self.agreement_tokens for at in self.differences.values()
            )
        return True

    def describe(self) -> str:
        """One clause: how many prompts agree, and where the others first differ."""
        same = self.compared - len(self.differences)
        head = f"{self.a} vs {self.b} ({self.policy}): {same}/{self.compared} identical"
        if not self.differences:
            return head
        shown = ", ".join(f"#{i} at token {at}" for i, at in sorted(self.differences.items())[:8])
        more = f" (+{len(self.differences) - 8} more)" if len(self.differences) > 8 else ""
        return f"{head}; first differences {shown}{more}"


def compare_legs(
    a: str,
    records_a: Sequence[Mapping[str, Any]],
    b: str,
    records_b: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    agreement_tokens: int = DEFAULT_AGREEMENT_TOKENS,
) -> PairComparison:
    """Compare two legs' records prompt by prompt on token ids."""
    from .witness import first_difference

    by_sha_b = {str(r.get("prompt_sha")): r for r in records_b}
    differences: dict[int, int] = {}
    compared = 0
    for index, record in enumerate(records_a):
        other = by_sha_b.get(str(record.get("prompt_sha")))
        if other is None:
            differences[index] = 0
            continue
        compared += 1
        ids_a, ids_b = list(record.get("token_ids") or ()), list(other.get("token_ids") or ())
        if ids_a != ids_b:
            differences[index] = first_difference(ids_a, ids_b)
    return PairComparison(
        a=a,
        b=b,
        policy=policy,
        differences=differences,
        compared=compared,
        agreement_tokens=agreement_tokens,
    )


def read_trace(directory: str | Path, rank: int = 0) -> list[dict[str, Any]]:
    """The step-trace records one rank wrote, in file order; empty when there are none."""
    path = Path(directory) / f"rank{rank}.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and "request" in data and "step" in data:
            records.append(data)
    return records


@dataclass(frozen=True)
class TraceDivergence:
    """The first step at which two traces disagree."""

    request: Any
    step: Any
    field: str
    a: Any
    b: Any

    def describe(self, names: tuple[str, str]) -> str:
        """``request 0 step 3: positions [...] vs [...]``."""
        return (
            f"request {self.request} step {self.step}: {self.field} {names[0]} {self.a!r} vs "
            f"{names[1]} {self.b!r}"
        )


def compare_traces(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]
) -> tuple[TraceDivergence | None, int]:
    """The first differing step over the steps both traces hold, and how many were compared."""
    index_b = {(r.get("request"), r.get("step")): r for r in b}
    compared = 0
    for record in a:
        other = index_b.get((record.get("request"), record.get("step")))
        if other is None:
            continue
        compared += 1
        for name in TRACE_FIELDS:
            if name in record and name in other and record[name] != other[name]:
                return (
                    TraceDivergence(
                        record["request"], record["step"], name, record[name], other[name]
                    ),
                    compared,
                )
    return None, compared


# ------------------------------------------------------------------- the legs


def _leg_spec(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    config: EquivalenceConfig,
    directory: Path,
    *,
    leg: str,
    unit: str | None,
    member: str | None,
    features: Sequence[str],
    concurrency: int = 1,
) -> Path:
    """Write one leg's engine-driver spec."""
    overlap = leg != LEG_NO_OVERLAP
    argv = quickstart_words(
        task_data,
        environment,
        unit=unit,
        features=features,
        max_tokens=config.max_tokens,
        member=member,
        max_batch_size=max(1, concurrency),
        runtime_overrides={"overlap_scheduler": overlap},
    )
    return write_driver_spec(
        directory / f"{leg}.json",
        task_data=task_data,
        argv=argv,
        requests=config.requests(),
        dataset=config.dataset,
        max_tokens=config.max_tokens,
        concurrency=concurrency,
        ignore_eos=config.ignore_eos,
    )


def run_scheduler_equivalence(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    unit: str | None = None,
    member: str | None = None,
    features: Sequence[str] | None = None,
    workdir: str | Path | None = None,
    config: EquivalenceConfig | None = None,
    timeout: int | None = None,
    allow_dirty: bool | None = None,
) -> GateResult:
    """Run the legs of the scheduler-equivalence witness and judge them; see the module docstring.

    ``features`` is what the target implements now; the ``no_spec`` leg drops
    its speculators, and a target with none runs ``overlap`` and
    ``no_overlap`` only.
    """
    name = "scheduler_equivalence"
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=name, member=member, passed=False, detail=state.block_detail())
    accuracy = task_data.get("accuracy") or {}
    try:
        config = config or resolve_equivalence_config(accuracy)
    except TaskSchemaError as exc:
        return _verdict(name=name, member=member, passed=False, detail=str(exc))
    features = list(task_data.get("features") or () if features is None else features)
    speculative = bool(speculation_flags(features))
    plain = [f for f in features if not speculation_flags([f])]
    legs = [LEG_OVERLAP]
    if LEG_NO_OVERLAP in config.legs:
        legs.append(LEG_NO_OVERLAP)
    if LEG_NO_SPEC in config.legs and speculative:
        legs.append(LEG_NO_SPEC)
    notes: list[str] = []
    if not resolve_runtime(task_data, unit).get("overlap_scheduler", True):
        notes.append(
            "the unit's runtime declares the overlap scheduler off; the overlap leg turns it on"
        )

    directory = gate_workdir(workdir, "scheduler-equivalence", task_data) / str(member or "member")
    directory.mkdir(parents=True, exist_ok=True)
    budget = int(timeout or config.timeout)
    started = time.monotonic()
    outputs: dict[str, list[dict[str, Any]]] = {}
    commands: dict[str, str] = {}
    traces: dict[str, Path] = {}
    for leg in legs:
        spec = _leg_spec(
            task_data,
            environment,
            config,
            directory,
            leg=leg,
            unit=unit,
            member=member,
            features=plain if leg == LEG_NO_SPEC else features,
        )
        traces[leg] = directory / "trace" / leg
        traces[leg].mkdir(parents=True, exist_ok=True)
        overlay = {STEP_TRACE_ENV: str(traces[leg])}
        if config.pin_autotuner:
            overlay[AUTOTUNER_CACHE_ENV] = str(directory / "autotuner.cache")
        env = _with_mode(environment, MODE_REQUIRE)
        env = replace(env, env={**env.env, **overlay})
        command = engine_driver_command(env, spec)
        commands[leg] = command
        result = runner(command, None, budget)
        combined = result.stdout + result.stderr
        if not result.ok:
            return _verdict(
                name=name,
                member=member,
                passed=False,
                detail=f"the {leg} leg "
                + ("timed out" if result.timed_out else f"exited {result.returncode}"),
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        records = parse_driver_records(combined)
        if not records:
            return _verdict(
                name=name,
                member=member,
                passed=False,
                detail=f"the {leg} leg printed no {workload_io.RECORD_MARKER} line; a witness "
                f"with no output is not a pass",
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        outputs[leg] = records
        if leg == LEG_OVERLAP:
            short = [r for r in records if int(r.get("tokens", 0)) < config.min_generated_tokens]
            if short:
                return _verdict(
                    name=name,
                    member=member,
                    passed=False,
                    detail=f"{len(short)} of {len(records)} prompt(s) generated fewer than "
                    f"{config.min_generated_tokens} tokens (shortest {min(int(r.get('tokens', 0)) for r in short)}), "
                    f"so generation steps had nothing to show a defect in. Workload: "
                    f"{config.source}; set ignore_eos or longer prompts",
                    command=command,
                    duration_s=time.monotonic() - started,
                )
    elapsed = time.monotonic() - started

    pairs: list[PairComparison] = []
    if LEG_NO_OVERLAP in outputs:
        pairs.append(
            compare_legs(
                LEG_OVERLAP,
                outputs[LEG_OVERLAP],
                LEG_NO_OVERLAP,
                outputs[LEG_NO_OVERLAP],
                policy=PAIR_EXACT,
            )
        )
    if LEG_NO_SPEC in outputs:
        pairs.append(
            compare_legs(
                LEG_OVERLAP,
                outputs[LEG_OVERLAP],
                LEG_NO_SPEC,
                outputs[LEG_NO_SPEC],
                policy=config.spec_pair,
                agreement_tokens=config.agreement_tokens,
            )
        )
    if not speculative and LEG_NO_SPEC in config.legs:
        notes.append("no speculator in this target, so no no_spec leg")
    passed = all(pair.passed for pair in pairs)

    trace_note, trace_failed = _judge_traces(
        traces, outputs, config, environment, runner, budget, directory
    )
    if trace_note:
        notes.append(trace_note)
    passed = passed and not trace_failed

    failing = [pair for pair in pairs if not pair.passed]
    if failing:
        why = (
            "the overlap scheduler changed the greedy output at batch 1. It changes only "
            "when the host prepares a step, not what the step computes, so the target "
            "reads state in generation steps that was prepared before the previous step's "
            "acceptance was known: a length, a position or a history"
            if failing[0].b == LEG_NO_OVERLAP
            else f"speculation changed the greedy output beyond the declared "
            f"`spec_pair: {config.spec_pair}`"
        )
        notes.insert(0, why)
    detail = "; ".join([*(pair.describe() for pair in pairs), *notes])
    detail += (
        f". {len(outputs[LEG_OVERLAP])} prompt(s) from {config.source}, "
        f"{config.max_tokens} tokens each, one at a time"
    )
    differing = [at for pair in failing for at in pair.differences.values()]
    return _verdict(
        name=name,
        member=member,
        passed=passed,
        detail=detail,
        command=commands[LEG_OVERLAP],
        measured=float(min(differing)) if differing else None,
        duration_s=elapsed,
        stdout_tail=_excerpt(outputs, failing) if failing else "",
    )


def _excerpt(
    outputs: Mapping[str, Sequence[Mapping[str, Any]]], failing: Sequence[PairComparison]
) -> str:
    """Both legs' token ids and text around the first difference of the first failing pair."""
    pair = failing[0]
    lines = []
    for index, at in list(sorted(pair.differences.items()))[:4]:
        a = outputs[pair.a][index]
        by_sha = {str(r.get("prompt_sha")): r for r in outputs[pair.b]}
        b = by_sha.get(str(a.get("prompt_sha")), {})
        lo, hi = max(0, at - 8), at + 8
        lines.append(f"prompt #{index}: first differs at token {at}")
        lines.append(f"  {pair.a:<10}: {list(a.get('token_ids') or ())[lo:hi]}")
        lines.append(f"  {pair.b:<10}: {list(b.get('token_ids') or ())[lo:hi]}")
        lines.append(f"  {pair.a:<10} text: {str(a.get('text', ''))[:160]!r}")
        lines.append(f"  {pair.b:<10} text: {str(b.get('text', ''))[:160]!r}")
    return "\n".join(lines)


def _judge_traces(
    traces: Mapping[str, Path],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    config: EquivalenceConfig,
    environment: RunEnvironment,
    runner: ShellRunner,
    timeout: int,
    directory: Path,
) -> tuple[str, bool]:
    """Compare the optional step traces; return a note and whether they failed the witness."""
    overlap = read_trace(traces[LEG_OVERLAP])
    if not overlap:
        return (
            "step trace: not written by the target (optional; see "
            f"step_check.py for the `{STEP_TRACE_ENV}` hook), judged on tokens alone",
            False,
        )
    notes = []
    failed = False
    if LEG_NO_OVERLAP in traces:
        divergence, compared = compare_traces(overlap, read_trace(traces[LEG_NO_OVERLAP]))
        if divergence is not None:
            failed = True
            notes.append(
                f"step trace diverges at {divergence.describe((LEG_OVERLAP, LEG_NO_OVERLAP))}"
            )
        else:
            notes.append(f"step trace: {compared} step(s) agree between overlap and no_overlap")
    if config.reference_command:
        from .witness import contrast_leg_command

        records_path = directory / "overlap.records.jsonl"
        records_path.write_text(
            "".join(json.dumps(r) + "\n" for r in outputs[LEG_OVERLAP]), encoding="utf-8"
        )
        reference_dir = directory / "trace" / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        env = replace(
            environment,
            env={
                **environment.env,
                STEP_TRACE_ENV: str(reference_dir),
                STEP_TRACE_ACCEPTED_ENV: str(traces[LEG_OVERLAP]),
                STEP_TRACE_RECORDS_ENV: str(records_path),
            },
        )
        result = runner(contrast_leg_command(env, config.reference_command), None, timeout)
        if not result.ok:
            return "; ".join([*notes, "the step-trace reference command did not complete"]), True
        divergence, compared = compare_traces(overlap, read_trace(reference_dir))
        if divergence is not None:
            failed = True
            notes.append(
                f"step trace diverges from the reference at {divergence.describe(('target', 'reference'))}"
            )
        else:
            notes.append(f"step trace: {compared} step(s) agree with the reference")
    return "; ".join(notes), failed


__all__ = [
    "AUTOTUNER_CACHE_ENV",
    "CONFIG_KEYS",
    "CONTRAST_LEGS",
    "DEFAULT_AGREEMENT_TOKENS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NUM_PROMPTS",
    "EquivalenceConfig",
    "LEG_NO_OVERLAP",
    "LEG_NO_SPEC",
    "LEG_OVERLAP",
    "MIN_GENERATED_TOKENS",
    "PAIR_AGREEMENT",
    "PAIR_EXACT",
    "PAIR_POLICIES",
    "PAIR_REPORT",
    "PairComparison",
    "STEP_TRACE_ACCEPTED_ENV",
    "STEP_TRACE_ENV",
    "STEP_TRACE_RECORDS_ENV",
    "TRACE_FIELDS",
    "TraceDivergence",
    "applies",
    "compare_legs",
    "compare_traces",
    "read_trace",
    "resolve_equivalence_config",
    "run_scheduler_equivalence",
]

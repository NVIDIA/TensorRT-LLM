# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gates — commands with exit codes, never agent judgements.

1. **boot** — the engine cold-starts under ``TRTLLM_MODELING_V2=require`` and
   a handful of greedy continuations come back; where the run has an HF logit
   reference, their first tokens must be the reference's (:class:`GreedySetting`).
2. **accuracy** — ``trtllm-eval`` against the anchor in TensorRT-LLM's own
   accuracy references. One-sided: ``measured >= reference - k * SE``.
3. **acceptance** — for speculative decoding, where a miscomputed draft path
   produces correct text more slowly: ``acceptance_length`` against the same
   harness with ``TRTLLM_MODELING_V2=off`` instead of ``=require``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import tempfile
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..infra.environment import RunEnvironment, scheduler_owns_devices
from ..infra.execution.commands import CommandResult, CommandSpec, LocalCommandExecutor
from ..infra.vcs import git_output
from ..onboard.delivery import MODELING_V2_PACKAGE
from ..onboard.logit_reference import Reference, load_reference
from ..onboard.logit_reference import ReferenceError as LogitReferenceError
from ..spec.task_schema import (
    BOOT_KEYS,
    PARALLEL_UNIFIED,
    RUNTIME_MARGIN_DEFAULTS,
    TaskSchemaError,
    assembly_units,
    derive_family,
    load_prompts_file,
    parse_parallel_segment,
    resolve_per_unit,
    resolve_samples,
)
from ..spec.variants import _deep_merge, base_unit, is_variant, variant_spec, variants
from . import observables, stats
from .gate_records import (
    KIND_ACCEPTANCE,
    KIND_FINDING,
    KIND_STOCK_UNAVAILABLE,
    RESULTS_FILENAME,
    GateRecord,
    GateRecords,
    open_findings,
)
from .harness import workload_io

# A boot gate that needs longer than this is an accuracy run with no anchor.
BOOT_TIMEOUT_S = 1800
ACCURACY_TIMEOUT_S = 4 * 60 * 60

MODELING_V2_ENV = "TRTLLM_MODELING_V2"
MODE_REQUIRE = "require"
MODE_OFF = "off"

#: Turns on each target's first-forward step-contract check
#: (each target's ``_check_step_contract``, which calls its ``step_args``):
#: a renamed metadata field then fails the gate loudly instead of computing a
#: wrong answer. Off in a served engine; every gate and witness run sets it, as
#: the in-tree modeling_v2 accuracy tests do.
STEP_CONTRACT_ENV = "TRTLLM_MODELING_V2_VALIDATE"

#: What a gate records when it could not ask git anything — no repository, no
#: ``git`` on the host, no commits yet. Never an exception: a gate adjudicates
#: a model, not a toolchain.
UNKNOWN_TREE_STATE = "unknown"

#: Set to ``1`` to let a gate measure a dirty target path anyway; the verdict
#: then carries ``dirty_override`` and says so in its summary.
ALLOW_DIRTY_ENV = "MV2FLOW_ALLOW_DIRTY_GATE"


# ``lm-eval <task> <filter> accuracy: 90.67`` and
# ``lm-eval <task> average accuracy: 90.67`` — the two shapes
# ``tensorrt_llm/evaluate/lm_eval.py`` emits at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# ``MMLU weighted average accuracy: 74.98 (14042)`` — the MMLU path has its
# own reporting because it does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# ``Spec-dec stats: AL (decoded tokens/step, iteration-weighted) 3.351 (...)``
_ACCEPTANCE_LENGTH = re.compile(r"Spec-dec stats:\s*AL\b[^)]*\)\s*([0-9]+(?:\.[0-9]+)?)")
# trtllm-bench reports the same quantity under its own name.
_ACCEPTANCE_LENGTH_ALT = re.compile(r"acceptance[_ ]length\D{0,12}([0-9]+(?:\.[0-9]+)?)", re.I)
# `[0] Accept rate: 1.88 (tokens=32, iterations=17)` --- what
# `examples/llm-api/quickstart_advanced.py` prints, once per request.
# Captured from the parenthetical rather than from the printed ratio so that
# several requests combine correctly (:func:`parse_acceptance`).
_ACCEPT_RATE_PARTS = re.compile(r"Accept rate:\s*[0-9.]+\s*\(tokens=(\d+),\s*iterations=(\d+)\)")

#: Tokens each acceptance prompt may generate when the task declares none.
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256
#: Generation iterations the acceptance workload must reach, summed over its
#: prompts, before the gate reads a rate from it. A few short-answer prompts
#: that each end in EOS after 2-3 tokens reach only a handful of iterations,
#: too few for a rate to mean anything. A task may raise it through
#: ``accuracy.acceptance.min_draft_iterations``.
DEFAULT_MIN_DRAFT_ITERATIONS = 32

ShellRunner = Callable[[str, Mapping[str, str] | None, int], CommandResult]

_LOCAL_EXECUTOR = LocalCommandExecutor()


def describe_tree_state(repo_root: str | Path) -> str:
    """The commit a gate is about to measure, as one token: ``git describe --always``.

    Not ``--dirty``, which refreshes the whole worktree (a minute on a large
    networked checkout): :func:`inspect_tree` appends a marker scoped to the target.
    """
    described = git_output(repo_root, "describe", "--always")
    if described is None:
        return UNKNOWN_TREE_STATE
    return described.strip() or UNKNOWN_TREE_STATE


def target_family_rel(task_data: Mapping[str, Any]) -> str | None:
    """The repo-relative directory holding every target for this family.

    Derived the way :func:`~.preflight.resolve_target_identity` derives it:
    ``target.family``, else the checkpoint's ``architectures[0]``. ``None``
    when neither can be read; a caller then has nothing to guard.
    """
    override = dict(task_data.get("target") or {})
    family = override.get("family")
    if not family:
        config = Path(str(task_data.get("checkpoint_path") or "")) / "config.json"
        try:
            architectures = (json.loads(config.read_text(encoding="utf-8")) or {}).get(
                "architectures"
            )
        except (OSError, ValueError, AttributeError):
            return None
        if not architectures:
            return None
        family = derive_family(str(architectures[0]))
    return f"{MODELING_V2_PACKAGE}/models/{family}"


def dirty_target_files(repo_root: str | Path, target_rel: str) -> tuple[str, ...]:
    """Uncommitted paths under ``target_rel``, as git reports them.

    ``-uall`` so an untracked *file* is named rather than collapsed into its
    directory. An empty tuple means both "clean" and "could not tell".
    """
    status = git_output(repo_root, "status", "--porcelain", "-uall", "--", target_rel)
    if not status:
        return ()
    return tuple(line[3:].strip() for line in status.splitlines() if line.strip())


@dataclass(frozen=True)
class TreeState:
    """What the repository looked like when a gate was about to run."""

    description: str = UNKNOWN_TREE_STATE
    #: Uncommitted paths under the target's own family directory.
    dirty: tuple[str, ...] = ()
    #: True when the run was told to measure a dirty target path anyway.
    override: bool = False

    @property
    def blocked(self) -> bool:
        return bool(self.dirty) and not self.override

    def block_detail(self) -> str:
        shown = ", ".join(self.dirty[:10])
        more = f" (+{len(self.dirty) - 10} more)" if len(self.dirty) > 10 else ""
        return (
            f"the target's own path is dirty at gate time, so this gate would measure "
            f"a tree nobody reviewed: {shown}{more}. By protocol the assembler commits "
            f"before handing back, so uncommitted files here mean another writer is "
            f"live. Let that writer finish, or set {ALLOW_DIRTY_ENV}=1 to measure "
            f"anyway and have the verdict say so."
        )


def inspect_tree(task_data: Mapping[str, Any], *, allow_dirty: bool | None = None) -> TreeState:
    """Read the repo state a gate is about to measure, and whether to refuse.

    The dirtiness question is scoped to the **target's own family path**, not
    to the repo. ``allow_dirty`` overrides the ``MV2FLOW_ALLOW_DIRTY_GATE``
    environment variable.
    """
    repo_root = task_data.get("trtllm_repo_path")
    if not repo_root:
        return TreeState()
    if allow_dirty is None:
        allow_dirty = os.environ.get(ALLOW_DIRTY_ENV, "").strip().lower() in {"1", "true", "yes"}
    described = describe_tree_state(repo_root)
    target_rel = target_family_rel(task_data)
    dirty = dirty_target_files(repo_root, target_rel) if target_rel else ()
    description = f"{described}-dirty" if dirty else described
    return TreeState(description=description, dirty=dirty, override=bool(dirty) and allow_dirty)


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with everything needed to defend it later."""

    name: str
    passed: bool
    detail: str
    #: The parallel configuration this verdict was measured at; ``None`` when
    #: the caller passes no member.
    member: str | None = None
    command: str = ""
    measured: float | None = None
    reference: float | None = None
    tolerance: float | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    #: The repository state this verdict was measured on —
    #: ``git describe --always`` under ``trtllm_repo_path``, suffixed
    #: ``-dirty`` when the target's own path was uncommitted, or
    #: :data:`UNKNOWN_TREE_STATE` where git could not answer.
    tree_state: str = UNKNOWN_TREE_STATE
    #: True when this verdict was measured over an uncommitted target path
    #: because the run said to.
    dirty_override: bool = False
    #: The target's marker for the commit this verdict measured, when the check
    #: ran on a snapshot of one (``run/snapshots.py``): the contract records the
    #: verdict under it, whatever the checkout became while the check ran.
    marker: str = ""
    #: What each command of the gate imported the package under test from, in
    #: order, as its processes reported it: the commit of the snapshot it ran
    #: in, or the path of a checkout no commit pins. Empty where no command
    #: said, and for a gate that ran on no snapshot.
    imported: tuple[str, ...] = ()
    #: Why this verdict says nothing about one tree, when it does not: its
    #: commands imported more than one tree, or a tree no commit pins, or no
    #: snapshot could be made. A void verdict never passes.
    void: str = ""
    #: The tree's Python files the gate's processes imported, relative to it;
    #: empty where a command did not say (``run/snapshots.py:CheckTree.read``).
    reads: tuple[str, ...] = ()
    #: The per-prompt numbers behind ``measured``, where a gate has them.
    breakdown: str = ""
    #: The accuracy floor the measurement was held to, and the binomial
    #: standard error and sample count it was derived from.
    floor: float | None = None
    se: float | None = None
    samples: int | None = None
    #: How ``floor`` was derived, verbatim (see :meth:`.stats.Floor.describe`).
    floor_rule: str = ""
    #: Things the verdict wants a reader to see that do not change it -- a
    #: floor not backed by a measurement, a comparison that had no baseline.
    warnings: tuple[str, ...] = ()
    #: Whether the summary line carries ``detail`` even on a pass: true for a
    #: verdict that rests on more than its headline numbers.
    explained: bool = False
    #: Set when the gate failed because the session under it was lost, not
    #: because of anything it measured; says why. Such a result is not a
    #: verdict about the target: the workflow re-runs the gate once a session
    #: is back instead of recording it or handing it to the assembler.
    infrastructure: str = ""
    #: One record per engine the gate built (:meth:`EngineAttestation.record`):
    #: the configuration it ran against the declared one, the pools it sized
    #: and the device memory it left free. Empty for a gate that built none.
    #: Not hashed: the records are mappings.
    engine: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)

    def summary(self) -> str:
        """One line for the journal, carrying the numbers verbatim."""
        verdict = "VOID" if self.void else "PASS" if self.passed else "FAIL"
        label = f"{self.name}@{self.member}" if self.member else self.name
        tree = f" [tree {self.tree_state}]"
        if self.dirty_override:
            tree = f" [tree {self.tree_state}; DIRTY TARGET PATH measured by override]"
        breakdown = f" [{self.breakdown}]" if self.breakdown else ""
        breakdown += describe_engine(self.engine)
        if self.warnings:
            breakdown += "".join(f" [WARNING: {w}]" for w in self.warnings)
        why = f" -- {self.detail}" if self.detail and (self.explained or not self.passed) else ""
        if self.measured is not None and self.floor is not None:
            return (
                f"[{verdict}] {label}: measured {self.measured:.4f} vs floor "
                f"{self.floor:.4f} [{self.floor_rule}] (margin "
                f"{self.measured - self.floor:+.4f}){why}{breakdown} in "
                f"{self.duration_s:.0f}s{tree}"
            )
        if self.measured is not None and self.reference is not None:
            margin = self.measured - (self.reference - (self.tolerance or 0.0))
            return (
                f"[{verdict}] {label}: measured {self.measured:.4f} vs "
                f"reference {self.reference:.4f} - tol {self.tolerance or 0.0:.2f} "
                f"(margin {margin:+.4f}){why}{breakdown} in {self.duration_s:.0f}s{tree}"
            )
        return f"[{verdict}] {label}: {self.detail}{breakdown} ({self.duration_s:.0f}s){tree}"


def _verdict_factory(state: TreeState) -> Callable[..., GateResult]:
    """A :class:`GateResult` constructor that stamps ``state`` onto every verdict."""

    def verdict(**fields: Any) -> GateResult:
        return GateResult(tree_state=state.description, dirty_override=state.override, **fields)

    return verdict


def lost_session(result: CommandResult) -> dict[str, Any]:
    """The :class:`GateResult` fields of a gate whose session was lost under it."""
    return {
        "passed": False,
        "detail": f"did not run to a verdict: {result.infrastructure}",
        "infrastructure": result.infrastructure,
    }


def measured_on(
    result: GateResult,
    *,
    tree_state: str,
    marker: str = "",
    imported: Sequence[str] = (),
    void: str = "",
    warnings: Sequence[str] = (),
    reads: Sequence[str] = (),
) -> GateResult:
    """``result`` as measured on one pinned tree: what it is, its marker, and what was imported.

    A void result (``void`` says why) does not pass, whatever it measured, and
    its detail leads with why.
    """
    fields: dict[str, Any] = {
        "tree_state": tree_state,
        "marker": marker,
        "imported": tuple(imported),
        "warnings": (*result.warnings, *warnings),
        "reads": tuple(reads),
    }
    if void:
        detail = f"{void}; what it measured: {result.detail}" if result.detail else void
        fields.update(passed=False, void=void, explained=True, detail=detail)
    return replace(result, **fields)


#: The image a multimodal unit's boot sends when the task declares none: a
#: photograph shipped in the repository's own test assets (a git-lfs file),
#: resolved against the tree under test. Never downloaded.
DEFAULT_BOOT_IMAGE = "tests/integration/test_input_files/merlion.png"
#: What the default image prompt asks, and the words any correct description
#: of that photograph (the Merlion fountain in Singapore) contains one of.
DEFAULT_BOOT_IMAGE_PROMPT = "Describe this image in one sentence."
DEFAULT_BOOT_IMAGE_KEYWORDS: tuple[str, ...] = (
    "merlion",
    "lion",
    "statue",
    "fountain",
    "singapore",
    "sculpture",
    "water",
)

#: The logit gate's limits (``checks/logit_agreement.py``), shared with the
#: boot's near-tie margin: how many times its own bf16-vs-fp32 spread an
#: engine may move off the reference, and the least spread that is counted.
LOGIT_THRESHOLD_FACTOR = 3.0
LOGIT_MIN_MAX_DELTA = 0.05
#: ``accuracy.boot.greedy`` when the spec says nothing: four prompts, 24 tokens each.
DEFAULT_GREEDY_PROMPTS = 4
DEFAULT_GREEDY_TOKENS = 24
#: ``quickstart_advanced.py``'s ``--max_num_tokens`` when the argv names none.
QUICKSTART_MAX_NUM_TOKENS = 8192


@dataclass(frozen=True)
class EngineLimits:
    """What one request may be in the unit's engine, read off its quickstart argv."""

    max_num_tokens: int = QUICKSTART_MAX_NUM_TOKENS
    max_seq_len: int | None = None
    chunked_prefill: bool = False

    @classmethod
    def from_words(cls, words: Sequence[str]) -> "EngineLimits":
        """The last ``--max_num_tokens``/``--max_seq_len`` (as argparse keeps it), and chunked prefill."""
        tokens = _flag_value(words, "--max_num_tokens")
        seq_len = _flag_value(words, "--max_seq_len")
        return cls(
            max_num_tokens=int(tokens) if tokens is not None else QUICKSTART_MAX_NUM_TOKENS,
            max_seq_len=int(seq_len) if seq_len is not None else None,
            chunked_prefill="--enable_chunked_prefill" in words,
        )

    def refuses(self, prompt_tokens: int, new_tokens: int) -> str:
        """Why a prompt of ``prompt_tokens`` that generates ``new_tokens`` does not fit; ``""`` when it does."""
        if self.max_seq_len is not None and prompt_tokens + new_tokens > self.max_seq_len:
            return f"{prompt_tokens} + {new_tokens} tokens exceed max_seq_len {self.max_seq_len}"
        if prompt_tokens > self.max_num_tokens and not self.chunked_prefill:
            return (
                f"{prompt_tokens} tokens exceed max_num_tokens {self.max_num_tokens} and "
                f"chunked prefill is off"
            )
        return ""


@dataclass(frozen=True)
class GreedySetting:
    """``accuracy.boot.greedy``: whose greedy tokens the boot holds the engine's to, and how many.

    A differing token the reference's top-1 leads by at most ``tie_factor``
    times the reference's spread is a near-tie, past which the prompt is not
    compared; any other differing token fails the boot.
    """

    reference: str
    prompts: int = DEFAULT_GREEDY_PROMPTS
    tokens: int = DEFAULT_GREEDY_TOKENS
    tie_factor: float = LOGIT_THRESHOLD_FACTOR

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any], unit: str | None) -> "GreedySetting | None":
        """The setting ``unit``'s boot runs under; ``None`` when its boot compares no tokens.

        On wherever ``accuracy.logits`` names a reference, unless
        ``accuracy.boot.greedy.enabled`` (or the unit's own entry) is false. A
        configuration variant compares only when it changes nothing but
        lossless settings (``observables.lossless``), or when its own entry,
        ``accuracy.boot.<variant>.greedy``, says ``enabled: true``.
        """
        accuracy = task_data.get("accuracy") or {}
        if not isinstance(accuracy, Mapping):
            return None
        logits = accuracy.get("logits")
        logits = logits if isinstance(logits, Mapping) else {}
        reference = str(logits.get("reference") or "").strip()
        boot = accuracy.get("boot")
        boot = boot if isinstance(boot, Mapping) else {}
        block = dict(boot.get("greedy") or {}) if isinstance(boot.get("greedy"), Mapping) else {}
        chosen = boot.get(unit) if unit is not None else None
        own = chosen.get("greedy") if isinstance(chosen, Mapping) else None
        if isinstance(own, Mapping):
            block.update(own)
        if (
            is_variant(task_data, unit)
            and not (isinstance(own, Mapping) and own.get("enabled"))
            and not observables.lossless(observables.subject_fields(task_data, unit))
        ):
            return None
        if not reference or block.get("enabled") is False:
            return None
        return cls(
            reference=reference,
            prompts=int(block.get("prompts") or DEFAULT_GREEDY_PROMPTS),
            tokens=int(block.get("tokens") or DEFAULT_GREEDY_TOKENS),
            tie_factor=float(logits.get("threshold_factor") or LOGIT_THRESHOLD_FACTOR),
        )


@dataclass(frozen=True)
class GreedyPrompt:
    """One reference prompt whose first greedy tokens the boot holds the engine's to."""

    id: str
    input_ids: tuple[int, ...]
    #: The reference's first greedy tokens, as many as the boot compares.
    expected: tuple[int, ...]
    #: Per compared step, the reference's top-k token ids and log-probs, best first.
    top_ids: tuple[tuple[int, ...], ...]
    top_logprobs: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class GreedyWorkload:
    """The prompts a boot compares, the margin a differing token is a near-tie within, and why any were left out."""

    source: str
    prompts: tuple[GreedyPrompt, ...] = ()
    tie_margin: float = 0.0
    notes: tuple[str, ...] = ()
    #: Why the boot cannot compare anything; it fails with this.
    error: str = ""

    @property
    def tokens(self) -> int:
        return max((len(p.expected) for p in self.prompts), default=0)

    def requests(self) -> list[dict[str, Any]]:
        """Engine-driver requests: each prompt as its token ids, for its compared tokens."""
        return [
            {"id": f"g{n}", "token_ids": list(p.input_ids), "max_tokens": len(p.expected)}
            for n, p in enumerate(self.prompts)
        ]


def greedy_workload(
    setting: GreedySetting, limits: EngineLimits, reference: Reference | None = None
) -> GreedyWorkload:
    """The reference's text prompts a boot sends: the first ``setting.prompts`` the engine can take.

    ``reference`` defaults to the file ``setting`` names; one that cannot be
    read, or none of whose text prompts fits, is the workload's ``error``.
    """
    source = Path(setting.reference).name
    if reference is None:
        try:
            reference = load_reference(setting.reference)
        except LogitReferenceError as exc:
            return GreedyWorkload(source, error=f"the boot's greedy reference: {exc}")
    tokens = min(setting.tokens, reference.continuation)
    chosen: list[GreedyPrompt] = []
    notes: list[str] = []
    for prompt in reference.prompts:
        if len(chosen) >= setting.prompts or prompt.images:
            continue
        refused = limits.refuses(len(prompt.input_ids), tokens)
        if refused:
            notes.append(f"{prompt.id} left out: {refused}")
            continue
        steps = range(prompt.prompt_positions, prompt.prompt_positions + tokens)
        chosen.append(
            GreedyPrompt(
                id=prompt.id,
                input_ids=prompt.input_ids,
                expected=prompt.continuation_ids[:tokens],
                top_ids=tuple(prompt.top_ids[p] for p in steps),
                top_logprobs=tuple(prompt.top_logprobs[p] for p in steps),
            )
        )
    margin = setting.tie_factor * max(reference.spread.max_abs_delta, LOGIT_MIN_MAX_DELTA)
    if not chosen or not tokens:
        return GreedyWorkload(
            source,
            notes=tuple(notes),
            error=f"no text prompt of the greedy reference {reference.path} fits this unit's "
            f"engine{': ' + '; '.join(notes) if notes else ''}; make a reference with "
            f"shorter prompts, or set accuracy.boot.greedy.enabled: false",
        )
    return GreedyWorkload(source, tuple(chosen), margin, tuple(notes))


def judge_greedy(
    workload: GreedyWorkload, records: Sequence[Mapping[str, Any] | None]
) -> tuple[list[str], list[str]]:
    """The engine's greedy tokens against the reference's: the failures, and the near-ties.

    ``records`` are the engine driver's, one per prompt of ``workload`` in
    order (``None`` where it returned none).
    """
    failures: list[str] = []
    ties: list[str] = []
    for prompt, record in zip(workload.prompts, records):
        if record is None:
            failures.append(f"{prompt.id}: the engine returned no record")
            continue
        seen = record.get("prompt_tokens")
        if isinstance(seen, int) and seen != len(prompt.input_ids):
            failures.append(
                f"{prompt.id}: the engine's prompt is {seen} tokens and the reference's "
                f"{len(prompt.input_ids)}"
            )
            continue
        generated = [int(t) for t in record.get("token_ids") or ()]
        for step, want in enumerate(prompt.expected):
            if step >= len(generated):
                failures.append(
                    f"{prompt.id}: the engine generated {len(generated)} of "
                    f"{len(prompt.expected)} tokens"
                )
                break
            got = generated[step]
            if got == want:
                continue
            ids, logprobs = prompt.top_ids[step], prompt.top_logprobs[step]
            if got in ids:
                gap = logprobs[0] - logprobs[ids.index(got)]
                if gap <= workload.tie_margin:
                    ties.append(
                        f"{prompt.id} step {step}: engine {got}, reference {want}, "
                        f"{gap:.3f} apart in the reference"
                    )
                    break
                behind = f"the engine's token {gap:.3f} behind in the reference"
            else:
                behind = "the engine's token is not in the reference's top-k"
            failures.append(
                f"{prompt.id} step {step}: engine token {got}, reference {want} ({behind}, past "
                f"the near-tie margin {workload.tie_margin:.3f})"
            )
            break
    return failures, ties


@dataclass(frozen=True)
class BootPrompt:
    """One boot prompt with checks of its own, images included."""

    prompt: str
    #: Every one must appear in THIS prompt's generated text (case-insensitive).
    keywords: tuple[str, ...] = ()
    #: At least one must appear in it (case-insensitive).
    keywords_any: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    render: str = "loader"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BootPrompt":
        """Read one entry of a boot ``prompts`` list written as a mapping."""
        return cls(
            prompt=str(data.get("prompt", "")),
            keywords=tuple(str(k) for k in data.get("keywords") or ()),
            keywords_any=tuple(str(k) for k in data.get("keywords_any") or ()),
            images=tuple(str(i) for i in data.get("images") or ()),
            render=str(data.get("render") or "loader"),
        )

    def missing(self, text: str) -> list[str]:
        """What this prompt's checks did not find in ``text``."""
        lowered = (text or "").lower()
        absent = [k for k in self.keywords if k.lower() not in lowered]
        if self.keywords_any and not any(k.lower() in lowered for k in self.keywords_any):
            absent.append(f"any of {list(self.keywords_any)}")
        return absent


@dataclass(frozen=True)
class BootSpec:
    """What the boot gate sends and what it insists on seeing back.

    The keyword list starts **empty on purpose**: keywords are frozen only
    after observing what this model generates. ``prompts`` and ``keywords``
    are the plain form: each keyword must appear in SOME prompt's generated
    text. ``items`` are prompts with checks of their own, and the only ones
    that carry images. Keywords are matched against the generated texts only,
    never against the log: ``72`` matches ``CPUs 72-143`` in an engine log line.
    """

    prompts: tuple[str, ...] = (
        "The capital of France is",
        "In one sentence, photosynthesis is",
    )
    keywords: tuple[str, ...] = ()
    max_tokens: int = 32
    items: tuple[BootPrompt, ...] = ()
    #: Where the spec came from, named in a verdict.
    source: str = "the default boot prompts"
    greedy: GreedySetting | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: str) -> "BootSpec":
        """A spec from a boot block: ``prompts`` (strings or mappings), ``keywords``, ``max_tokens``."""
        plain: list[str] = []
        items: list[BootPrompt] = []
        for entry in data.get("prompts") or ():
            if isinstance(entry, Mapping):
                items.append(BootPrompt.from_mapping(entry))
            else:
                plain.append(str(entry))
        if data.get("prompts_file"):
            plain.extend(load_prompts_file(data["prompts_file"]))
        return cls(
            prompts=tuple(plain) if (plain or items) else cls.prompts,
            keywords=tuple(str(k) for k in data.get("keywords") or ()),
            max_tokens=int(data.get("max_tokens") or cls.max_tokens),
            items=tuple(items),
            source=source,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BootSpec":
        """A workspace ``boot.yaml``; the default spec when there is none."""
        source = Path(path)
        if not source.is_file():
            return cls()
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source} must hold a YAML mapping, got {type(data).__name__}")
        return cls.from_mapping(data, str(source))

    @property
    def has_images(self) -> bool:
        return any(item.images for item in self.items)

    def requests(self, repo_root: str | Path | None) -> list[dict[str, Any]]:
        """Engine-driver requests: the plain prompts, then the items.

        A relative image path is resolved against ``repo_root``, the tree under test.
        """

        def image_path(name: str) -> str:
            path = Path(name)
            if path.is_absolute() or repo_root is None:
                return str(path)
            return str(Path(repo_root) / path)

        requests = [
            {"id": f"b{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]
        for n, item in enumerate(self.items):
            request: dict[str, Any] = {
                "id": f"i{n}",
                "prompt": item.prompt,
                "max_tokens": self.max_tokens,
            }
            if item.images:
                request["images"] = [image_path(i) for i in item.images]
                request["render"] = item.render
            requests.append(request)
        return requests


def resolve_boot_spec(
    task_data: Mapping[str, Any],
    unit: str | None,
    features: Sequence[str] | None = None,
    fallback: str | Path | None = None,
) -> BootSpec:
    """The boot workload ``unit`` runs, declared in task.yaml.

    ``accuracy.boot`` (run-wide keys, with a per-unit entry layered over
    them) when task.yaml declares it, else the workspace ``boot.yaml`` at
    ``fallback``, else the default. A unit whose target implements
    ``multimodal`` (and whose scope is not ``text_only``) and whose spec sends
    no image gets the default image prompt appended. Wherever the run has an
    HF logit reference, the spec carries the greedy comparison
    (:meth:`GreedySetting.from_task`).
    """
    accuracy = task_data.get("accuracy") or {}
    block = accuracy.get("boot") if isinstance(accuracy, Mapping) else None
    greedy = GreedySetting.from_task(task_data, unit)
    if isinstance(block, Mapping) and _declares_workload(block):
        merged = {k: v for k, v in block.items() if k in BOOT_KEYS and k != "greedy"}
        chosen = block.get(unit) if unit is not None else None
        where = "accuracy.boot"
        if isinstance(chosen, Mapping) and _declares_workload(chosen):
            merged.update({k: v for k, v in chosen.items() if k != "greedy"})
            where = f"accuracy.boot.{unit}"
        spec = BootSpec.from_mapping(merged, where)
    elif fallback is not None:
        spec = BootSpec.load(fallback)
    else:
        spec = BootSpec()
    spec = replace(spec, greedy=greedy)
    implemented = {str(f) for f in (features or ())}
    if (
        "multimodal" in implemented
        and task_data.get("scope", "full") != "text_only"
        and not spec.has_images
    ):
        image = BootPrompt(
            prompt=DEFAULT_BOOT_IMAGE_PROMPT,
            keywords_any=DEFAULT_BOOT_IMAGE_KEYWORDS,
            images=(DEFAULT_BOOT_IMAGE,),
        )
        spec = replace(spec, items=(*spec.items, image))
    return spec


def _declares_workload(block: Mapping[str, Any]) -> bool:
    """Whether a level of ``accuracy.boot`` says what to send: any key but ``greedy``, at it or below."""
    for key, value in block.items():
        if key == "greedy":
            continue
        if key in BOOT_KEYS or not isinstance(value, Mapping):
            return True
        if set(value) - {"greedy"}:
            return True
    return False


def _default_shell_runner(
    command: str,
    env: Mapping[str, str] | None = None,
    timeout: int = BOOT_TIMEOUT_S,
) -> CommandResult:
    """Run one command line through the shell, on this host.

    The launch prefix resolved from ``task.yaml`` is shell syntax (env
    assignments, or an ``srun`` line). Nothing here interpolates agent-authored
    text — the prefix comes from the task spec and the command from this
    module.
    """
    return _LOCAL_EXECUTOR.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))


#: The flag vocabularies of the two gate commands; see :func:`topology_flags`.
QUICKSTART_DIALECT = "quickstart"
EVAL_DIALECT = "eval"


def topology_flags(
    segment: str,
    world_size: int,
    member: str | None = None,
    *,
    dialect: str = QUICKSTART_DIALECT,
) -> list[str]:
    """Translate a parallel segment, or the ``member`` served, into a gate command's flags.

    ``unified`` names a family, not a configuration: without a member it
    translates to ``--tp_size`` alone. ``dialect`` names which command the
    flags are for:

    ==================  ===========================  ==========================
    axis                ``quickstart_advanced.py``   ``trtllm-eval``
    ==================  ===========================  ==========================
    tensor parallel     ``--tp_size``                ``--tp_size``
    expert parallel     ``--moe_ep_size``            ``--ep_size``
    attention DP        ``--enable_attention_dp``    **no flag at all**
    ==================  ===========================  ==========================

    Under the ``eval`` dialect attention DP has to arrive through the options
    file (:func:`member_llm_api_overlay`).
    """
    kind, declared = parse_parallel_segment(member or segment)
    size = declared if declared is not None else world_size
    if kind == PARALLEL_UNIFIED:
        return [f"--tp_size={size}"]
    flags = [f"--tp_size={size}"]
    if kind in ("tep", "dep"):
        flags.append(f"--ep_size={size}" if dialect == EVAL_DIALECT else f"--moe_ep_size={size}")
    if kind == "dep" and dialect != EVAL_DIALECT:
        flags.append("--enable_attention_dp")
    return flags


def member_llm_api_overlay(member: str | None) -> dict[str, Any]:
    """The LLM-API settings a member needs that its gate command cannot carry: attention DP."""
    if not member:
        return {}
    kind, _ = parse_parallel_segment(member)
    return {"enable_attention_dp": True} if kind == "dep" else {}


#: Startup cost of a member's topology, cheapest first: ``tep`` partitions the
#: experts and slices attention, ``dep`` replicates attention on every rank,
#: ``tp`` holds a slice of every expert on every rank.
_MEMBER_COST_RANK: dict[str, int] = {"tep": 0, "dep": 1, "tp": 2}

# Unknown cost (``unified`` included) is not cheap cost: it sorts last.
_UNKNOWN_MEMBER_COST = len(_MEMBER_COST_RANK)


def order_members_by_cost(members: Sequence[str]) -> list[str]:
    """Return ``members`` cheapest-first, for a unit gated over several.

    A broken unit fails at every member, so the cheapest one fails first.
    Deterministic in the set of members: ties break on world size, then on the
    name. An unrecognized member sorts last rather than raising.
    """

    def key(member: str) -> tuple[int, int, str]:
        try:
            kind, size = parse_parallel_segment(member)
        except TaskSchemaError:
            return (_UNKNOWN_MEMBER_COST, 0, member)
        return (
            _MEMBER_COST_RANK.get(kind, _UNKNOWN_MEMBER_COST),
            size if size is not None else 0,
            member,
        )

    return sorted(members, key=key)


#: How the stock path says *"I have never heard of this architecture"*: the
#: invariant half of each message, matched in the engine's log.
UNSUPPORTED_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    # `AutoModelForCausalLM` / the multimodal encoder (`_torch/models/modeling_auto.py`)
    # and `tensorrt_llm/models/automodel.py`.
    "Unknown architecture for AutoModelForCausalLM",
    "Unknown architecture for AutoModelForMultimodalEncoder",
    "is not supported in TRT-LLM yet",
    # `get_model_architecture`'s registry miss (`_torch/models/modeling_utils.py`).
    "Unknown model architecture:",
    # `transformers`' refusal of a checkpoint that ships its own code (`auto_map`).
    "contains custom code which must be executed to correctly load the model",
    # `transformers`' refusal of an unknown `model_type` with no code to fall back on.
    "but Transformers does not recognize this architecture",
    # A registration shim: a name registered only so the runtime finds something
    # keyed on it, which raises when built without a modeling_v2 target.
    "has no built-in implementation",
)


def reports_unsupported_architecture(detail: str, output: str) -> bool:
    """Whether this text says *"the stock path does not know this model"*."""
    haystack = f"{detail}\n{output}"
    return any(marker in haystack for marker in UNSUPPORTED_ARCHITECTURE_MARKERS)


def resolve_runtime(task_data: Mapping[str, Any], unit: str | None) -> Mapping[str, Any]:
    """The ``runtime`` block for ``unit``, with the per-unit form merged.

    Accepts the flat form (the whole run) or a mapping from unit name to a
    partial block layered key by key over ``default``. ``extra_flags`` is an
    ordinary key and therefore REPLACES rather than extends.
    """
    block = task_data.get("runtime") or {}
    if not any(isinstance(v, dict) for v in block.values()):
        return block
    merged = dict(block.get("default") or {})
    chosen = block.get(unit)
    if isinstance(chosen, dict):
        merged.update(chosen)
    return merged


def runtime_flags(runtime: Mapping[str, Any], *, capture: CaptureList | None = None) -> list[str]:
    """Translate the ``runtime`` block into ``quickstart_advanced.py`` flags.

    ``overlap_scheduler`` and ``kv_cache_reuse`` are on by default in the
    engine and spelled as negations: wanting them emits nothing, not wanting
    them emits ``--disable_*``. ``capture`` is the unit's decode CUDA-graph
    list (:func:`resolve_capture_list`), spelled as flags with CUDA graphs on.
    """
    flags: list[str] = []
    extra = [str(f) for f in runtime.get("extra_flags", ())]
    if runtime.get("cuda_graph"):
        flags.append(CUDA_GRAPH_FLAG)
        if capture is not None:
            flags.extend(capture.flags(stated=extra))
    if runtime.get("chunked_prefill"):
        flags.append("--enable_chunked_prefill")
    if not runtime.get("overlap_scheduler", True):
        flags.append("--disable_overlap_scheduler")
    if not runtime.get("kv_cache_reuse", True):
        flags.append("--disable_kv_cache_reuse")
    flags.extend(extra)
    return flags


# ------------------------------------------------------- one setting, two launchers
#
# The quickstart gates take runtime settings as `quickstart_advanced.py`
# flags, from the runtime block's `extra_flags`; `trtllm-eval`, `trtllm-serve`
# and the tune campaign take them as LLM-API keys, from the unit's options
# file. Each setting both launchers take is read from both channels and handed
# to the launcher that did not state it; the two channels stating different
# values is refused at preflight (`launch_conflicts`).


@dataclass(frozen=True)
class LaunchKnob:
    """A setting both launchers take: its quickstart flag and its LLM-API key path."""

    flag: str
    path: tuple[str, ...]
    kind: type


LAUNCH_KNOBS: tuple[LaunchKnob, ...] = (
    LaunchKnob("--kv_cache_fraction", ("kv_cache_config", "free_gpu_memory_fraction"), float),
    LaunchKnob("--kv_cache_dtype", ("kv_cache_config", "dtype"), str),
    LaunchKnob("--tokens_per_block", ("kv_cache_config", "tokens_per_block"), int),
    LaunchKnob("--mamba_ssm_cache_dtype", ("kv_cache_config", "mamba_ssm_cache_dtype"), str),
    LaunchKnob("--max_num_tokens", ("max_num_tokens",), int),
    LaunchKnob("--trust_remote_code", ("trust_remote_code",), bool),
)

#: The checkpoint files whose ``auto_map`` says the model's code ships with its weights.
REMOTE_CODE_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def checkpoint_remote_code(checkpoint: str | Path | None) -> bool:
    """Whether the checkpoint ships its own code (an ``auto_map``): every launcher must trust it."""
    if not checkpoint:
        return False
    for name in REMOTE_CODE_FILES:
        path = Path(str(checkpoint)) / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping) and data.get("auto_map"):
            return True
    return False


def _knob_value(knob: LaunchKnob, text: str) -> Any:
    if knob.kind is bool:
        return text.strip().lower() not in ("0", "false", "no")
    try:
        return knob.kind(text)
    except ValueError:
        return text


def flag_settings(flags: Sequence[str]) -> dict[LaunchKnob, Any]:
    """The :data:`LAUNCH_KNOBS` a quickstart argv states, with their values."""
    words = [w for flag in flags for w in shlex.split(str(flag))]
    found: dict[LaunchKnob, Any] = {}
    for index, word in enumerate(words):
        name, eq, text = word.partition("=")
        for knob in LAUNCH_KNOBS:
            if name != knob.flag:
                continue
            if eq:
                found[knob] = _knob_value(knob, text)
            elif knob.kind is bool:
                found[knob] = True
            elif index + 1 < len(words) and not words[index + 1].startswith("--"):
                found[knob] = _knob_value(knob, words[index + 1])
    return found


def option_settings(options: Mapping[str, Any]) -> dict[LaunchKnob, Any]:
    """The :data:`LAUNCH_KNOBS` an LLM-API options mapping states, with their values."""
    found: dict[LaunchKnob, Any] = {}
    for knob in LAUNCH_KNOBS:
        node: Any = options
        for key in knob.path:
            node = node.get(key) if isinstance(node, Mapping) else None
        if node is not None:
            found[knob] = node
    return found


def _unit_options(task_data: Mapping[str, Any], unit: str | None) -> dict[str, Any]:
    return load_llm_api_options(resolve_llm_api_options(task_data.get("accuracy") or {}, unit))


def launch_flags(
    task_data: Mapping[str, Any], unit: str | None, stated: Sequence[str]
) -> list[str]:
    """The quickstart flags for the settings ``unit``'s options state and ``stated`` does not.

    ``--trust_remote_code`` too for a checkpoint that ships its own code.
    """
    have = flag_settings(stated)
    wanted = option_settings(_unit_options(task_data, unit))
    if checkpoint_remote_code(task_data.get("checkpoint_path")):
        wanted.setdefault(LAUNCH_KNOBS[-1], True)
    flags: list[str] = []
    for knob, value in wanted.items():
        if knob in have:
            continue
        if knob.kind is bool:
            if value:
                flags.append(knob.flag)
        else:
            flags.append(f"{knob.flag}={value}")
    return flags


def launch_overlay(task_data: Mapping[str, Any], unit: str | None) -> dict[str, Any]:
    """The LLM-API keys for the settings ``unit``'s runtime flags state, as a nested mapping.

    ``trust_remote_code: true`` too for a checkpoint that ships its own code.
    Merged under the options file's own keys, which win where both state a
    setting.
    """
    stated = flag_settings(resolve_runtime(task_data, unit).get("extra_flags") or ())
    if checkpoint_remote_code(task_data.get("checkpoint_path")):
        stated.setdefault(LAUNCH_KNOBS[-1], True)
    overlay: dict[str, Any] = {}
    for knob, value in stated.items():
        node = overlay
        for key in knob.path[:-1]:
            node = node.setdefault(key, {})
        node[knob.path[-1]] = value
    return overlay


def launch_conflicts(task_data: Mapping[str, Any], unit: str | None) -> list[str]:
    """The settings ``unit``'s runtime flags and its options file state differently, one line each."""
    flags = flag_settings(resolve_runtime(task_data, unit).get("extra_flags") or ())
    options = option_settings(_unit_options(task_data, unit))
    return [
        f"{knob.flag}={flags[knob]} in runtime.extra_flags, but "
        f"{'.'.join(knob.path)}: {options[knob]} in the options file"
        for knob in flags
        if knob in options and flags[knob] != options[knob]
    ]


def merge_options(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` over ``base``, mapping into mapping, so a nested block keeps its other keys."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_options(merged[key], value)
        else:
            merged[key] = value
    return merged


def speculation_flags(features: Any, *, max_draft_len: int = 1) -> list[str]:
    """Translate a speculative feature into its ``quickstart_advanced.py`` flags.

    ``dspark`` gets its own algorithm and never ``MTP``, even though a
    checkpoint may store its DSpark stages as ``mtp.*``: ``MTP`` over those
    weights drafts garbage that verification rejects, which passes accuracy.
    """
    names = {str(f).lower() for f in (features or ())}
    for feature, algo in (("mtp", "MTP"), ("eagle3", "Eagle3"), ("dspark", "DSPARK")):
        if feature in names:
            return [f"--spec_decode_algo={algo}", f"--spec_decode_max_draft_len={max_draft_len}"]
    return []


def resolve_max_draft_len(accuracy: Mapping[str, Any], unit: str | None) -> int:
    """How many draft tokens this unit's gates must ask for, from its options YAML; 1 without one.

    The quickstart gates take it as ``--spec_decode_max_draft_len``, which must
    match the ``speculative_config`` ``trtllm-eval`` reads: a shallower
    speculator accepts a larger fraction of what it drafts. Read in the order
    the engine resolves it: ``max_draft_len``, its deprecated MTP alias
    ``num_nextn_predict_layers``, and a DSpark drafter's ``block_size``.
    """
    options = resolve_llm_api_options(accuracy, unit)
    if not options:
        return 1
    source = Path(options)
    if not source.is_file():
        return 1
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    spec = data.get("speculative_config") or {}
    for key in ("max_draft_len", "num_nextn_predict_layers", "block_size"):
        try:
            return max(1, int(spec[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return 1


def resolve_llm_api_options(accuracy: Mapping[str, Any], unit: str | None) -> str | None:
    """Pick the runtime-configuration YAML for ``unit``.

    ``llm_api_options`` may be a single path (the whole run) or a mapping
    from unit name to path, with ``default`` as the fallback key.
    """
    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        return options.get(unit) or options.get("default")
    return options


def load_llm_api_options(path: str | Path | None) -> dict[str, Any]:
    """The LLM-API options file at ``path`` as a mapping; empty when there is none."""
    if not path or not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


# ------------------------------------------------------------ CUDA-graph capture
#
# `trtllm-eval` (accuracy) reads the decode CUDA-graph batch list from the
# unit's options file; `quickstart_advanced.py` (boot, acceptance, envelope)
# builds `CudaGraphConfig` from its own flags. The list is therefore resolved
# once per unit and spelled as quickstart flags.

#: ``quickstart_advanced.py``'s CUDA-graph switches.
CUDA_GRAPH_FLAG = "--use_cuda_graph"
CAPTURE_BATCHES_FLAG = "--cuda_graph_batch_sizes"
CAPTURE_PADDING_FLAG = "--cuda_graph_padding_enabled"

#: The capture limit ``CudaGraphConfig`` uses when neither ``batch_sizes`` nor
#: ``max_batch_size`` is set (``BaseCudaGraphConfig.validate_base_cuda_graph_config``).
LLM_API_CAPTURE_MAX_BATCH_SIZE = 128


def llm_api_batch_sizes(
    batch_sizes: Sequence[int] | None = None,
    max_batch_size: int = 0,
    enable_padding: bool = False,
) -> tuple[int, ...]:
    """The decode batch sizes ``CudaGraphConfig(...)`` resolves to.

    The LLM API's own rule, restated so the orchestrator never imports
    TensorRT-LLM: ``BaseCudaGraphConfig.validate_base_cuda_graph_config`` and
    ``BaseCudaGraphConfig._generate_cuda_graph_batch_sizes`` in
    ``tensorrt_llm/llmapi/llm_args.py``. Raises ``ValueError`` where it does:
    on a ``max_batch_size`` beside explicit ``batch_sizes`` that is not their
    maximum. The engine's later filter by each launch's own limits
    (``filter_cuda_graph_batch_sizes``) is not part of this list.
    """
    if batch_sizes:
        sizes = sorted(int(b) for b in batch_sizes)
        if max_batch_size and int(max_batch_size) != sizes[-1]:
            raise ValueError(
                f"cuda_graph_config.max_batch_size {max_batch_size} is not "
                f"max(cuda_graph_config.batch_sizes) {sizes[-1]}; the LLM API refuses the pair"
            )
        return tuple(sizes)
    limit = int(max_batch_size) or LLM_API_CAPTURE_MAX_BATCH_SIZE
    if enable_padding:
        generated = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        while generated[-1] + 64 <= limit:
            generated.append(generated[-1] + 64)
    else:
        generated = list(range(1, 32)) + [32, 64, 128]
        generated += [2**i for i in range(8, math.ceil(math.log(limit, 2)))]
    sizes = sorted(size for size in generated if size <= limit)
    if not sizes or sizes[-1] != limit:
        sizes.append(limit)
    return tuple(sizes)


def format_batches(sizes: Sequence[int]) -> str:
    """``1-32, 64, 128``: a batch list with each run of three or more consecutive sizes joined."""
    parts: list[str] = []
    run: list[int] = []
    for size in [*sizes, None]:
        if run and size == run[-1] + 1:
            run.append(size)
            continue
        if len(run) >= 3:
            parts.append(f"{run[0]}-{run[-1]}")
        else:
            parts.extend(str(s) for s in run)
        run = [size] if size is not None else []
    return ", ".join(parts)


@dataclass(frozen=True)
class CaptureList:
    """The decode CUDA-graph batch sizes one engine captures, and whether it pads to them.

    Two lists are equal when they capture the same sizes with the same
    padding; ``source`` says where one was read and takes no part in that.
    """

    batch_sizes: tuple[int, ...]
    enable_padding: bool
    source: str = field(default="", compare=False)

    def describe(self) -> str:
        padded = "padded" if self.enable_padding else "unpadded"
        return f"{len(self.batch_sizes)} batches [{format_batches(self.batch_sizes)}], {padded}"

    def flags(self, stated: Sequence[str] = ()) -> list[str]:
        """The quickstart flags that capture this list, less what ``stated`` already spells.

        A runtime block's own spelling that disagrees with this list is refused
        at preflight (``check_capture_lists``).
        """
        words = _split_flags(stated)
        flags: list[str] = []
        if self.enable_padding and CAPTURE_PADDING_FLAG not in words:
            flags.append(CAPTURE_PADDING_FLAG)
        if not any(_is_capture_batches(word) for word in words):
            flags.append(f"{CAPTURE_BATCHES_FLAG} {' '.join(str(b) for b in self.batch_sizes)}")
        return flags


#: What a user who sets no ``cuda_graph_config`` gets from the LLM API.
LLM_API_DEFAULT_CAPTURE = CaptureList(
    llm_api_batch_sizes(), False, source="the LLM API default, `CudaGraphConfig()`"
)


def _split_flags(flags: Sequence[str]) -> list[str]:
    """``flags`` as words, each entry split as the shell splits it."""
    return [word for flag in flags for word in shlex.split(str(flag))]


def _is_capture_batches(word: str) -> bool:
    return word == CAPTURE_BATCHES_FLAG or word.startswith(f"{CAPTURE_BATCHES_FLAG}=")


def _capture_switches(flags: Sequence[str]) -> tuple[bool, list[int] | None, bool]:
    """``(--use_cuda_graph, the --cuda_graph_batch_sizes values, --cuda_graph_padding_enabled)``.

    Read as ``quickstart_advanced.py``'s argparse reads them: the values run
    to the next option, and the last ``--cuda_graph_batch_sizes`` wins.
    ``None`` for the values when the flag is absent.
    """
    words = _split_flags(flags)
    graphs = padding = False
    sizes: list[int] | None = None
    for index, word in enumerate(words):
        if word == CUDA_GRAPH_FLAG:
            graphs = True
        elif word == CAPTURE_PADDING_FLAG:
            padding = True
        elif word == CAPTURE_BATCHES_FLAG:
            sizes = []
            for value in words[index + 1 :]:
                if value.startswith("-"):
                    break
                sizes.append(int(value))
        elif word.startswith(f"{CAPTURE_BATCHES_FLAG}="):
            sizes = [int(word.split("=", 1)[1])]
    return graphs, sizes, padding


def quickstart_capture(flags: Sequence[str]) -> CaptureList | None:
    """What ``quickstart_advanced.py`` captures from ``flags``; ``None`` without CUDA graphs.

    Under ``--use_cuda_graph`` the script builds ``CudaGraphConfig`` from
    ``--cuda_graph_batch_sizes`` and ``--cuda_graph_padding_enabled`` alone,
    so neither the options file nor ``--max_batch_size`` reaches the list.
    """
    graphs, sizes, padding = _capture_switches(flags)
    if not graphs:
        return None
    return CaptureList(
        llm_api_batch_sizes(sizes, 0, padding), padding, source="quickstart_advanced.py's flags"
    )


def options_capture(
    options: Mapping[str, Any], source: str = "the options file"
) -> CaptureList | None:
    """What an engine built from LLM-API ``options`` captures; ``None`` when it captures nothing.

    Options that do not name ``cuda_graph_config`` get ``CudaGraphConfig()``,
    the field's default in ``TorchLlmArgs``; ``cuda_graph_config: null``
    turns CUDA graphs off. A ``speculative_config.draft_len_schedule`` turns
    padding on and adds its batch thresholds to the list, as ``TorchLlmArgs``'s
    speculative-config validator does (``DecodeCudaGraphConfig._merge_schedule_keys``).
    """
    if "cuda_graph_config" not in options:
        config: Any = {}
        where = f"{LLM_API_DEFAULT_CAPTURE.source}: {source} sets no `cuda_graph_config`"
    else:
        config = options["cuda_graph_config"]
        where = f"`cuda_graph_config` in {source}"
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError(f"`cuda_graph_config` in {source} is not a mapping: {config!r}")
    padding = bool(config.get("enable_padding", False))
    sizes = llm_api_batch_sizes(
        config.get("batch_sizes"), int(config.get("max_batch_size") or 0), padding
    )
    speculative = options.get("speculative_config")
    schedule = speculative.get("draft_len_schedule") if isinstance(speculative, Mapping) else None
    if isinstance(schedule, Mapping) and schedule:
        padding = True
        sizes = tuple(sorted({*sizes, *(int(k) for k in schedule if int(k) <= sizes[-1])}))
        where += " with `speculative_config.draft_len_schedule`"
    return CaptureList(sizes, padding, source=where)


def capture_list_from(
    options: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    source: str = "the options file",
) -> CaptureList | None:
    """The one decode capture list for a unit with these options and this runtime block.

    The options' ``cuda_graph_config`` when they set one, otherwise the LLM
    API's rule over the capture flags the runtime block's ``extra_flags``
    spell. ``None`` when the options set ``cuda_graph_config: null``.
    """
    if "cuda_graph_config" in options:
        return options_capture(options, source)
    _, sizes, padding = _capture_switches(runtime.get("extra_flags", ()))
    stated = {"batch_sizes": sizes, "enable_padding": padding}
    capture = options_capture({**options, "cuda_graph_config": stated}, source)
    if capture is None:
        return None
    if sizes is None and not padding:
        where = f"{LLM_API_DEFAULT_CAPTURE.source}: neither {source} nor the runtime block sets the list"
    else:
        where = "the runtime block's extra_flags"
    return replace(capture, source=where)


def resolve_capture_list(task_data: Mapping[str, Any], unit: str | None) -> CaptureList | None:
    """The one decode CUDA-graph capture list ``unit``'s gates run (:func:`capture_list_from`)."""
    path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    return capture_list_from(
        load_llm_api_options(path),
        resolve_runtime(task_data, unit),
        source=str(path) if path else "the unit (no options file)",
    )


@dataclass(frozen=True)
class LaunchCapture:
    """What each launch path of one unit, or one configuration variant, captures.

    ``quickstart`` is the boot, acceptance and envelope gates' engine, built
    from the gate argv; ``llm_api`` is the accuracy gate's, built by
    ``trtllm-eval`` from ``options``. ``None`` is a path that runs eagerly.
    """

    name: str
    unit: str
    options: str
    resolved: CaptureList | None
    quickstart: CaptureList | None
    llm_api: CaptureList | None

    @property
    def conflicting(self) -> bool:
        """Both paths capture graphs, and not the same ones."""
        return (
            self.quickstart is not None
            and self.llm_api is not None
            and self.quickstart != self.llm_api
        )


def _launch_capture(
    name: str, unit: str, options: Mapping[str, Any], runtime: Mapping[str, Any], source: str
) -> LaunchCapture:
    resolved = capture_list_from(options, runtime, source=source)
    return LaunchCapture(
        name=name,
        unit=unit,
        options=source,
        resolved=resolved,
        quickstart=quickstart_capture(runtime_flags(runtime, capture=resolved)),
        llm_api=options_capture(options, source),
    )


def _launch_configurations(
    task_data: Mapping[str, Any],
) -> list[tuple[str, str, dict[str, Any], Mapping[str, Any], str]]:
    """``(name, unit, options, runtime, source)`` for each unit, then each configuration variant.

    A variant's options are its unit's file with the variant's
    ``llm_api_options`` merged in, as :func:`~.variants.variant_spec` writes
    them, and its runtime block is its unit's with the variant's overlay.
    """
    accuracy = task_data.get("accuracy") or {}
    configurations: list[tuple[str, str, dict[str, Any], Mapping[str, Any], str]] = []
    for unit in assembly_units(task_data):
        path = resolve_llm_api_options(accuracy, unit)
        configurations.append(
            (
                unit,
                unit,
                load_llm_api_options(path),
                resolve_runtime(task_data, unit),
                str(path) if path else "the unit (no options file)",
            )
        )
    for name, entry in variants(task_data).items():
        unit = base_unit(task_data, name)
        path = resolve_llm_api_options(accuracy, unit)
        options = _deep_merge(load_llm_api_options(path), entry.get("llm_api_options") or {})
        # The members it resolves are not read here, so any world size does.
        spec = variant_spec(task_data, name, world_size=1)
        configurations.append(
            (
                name,
                unit,
                options,
                resolve_runtime(spec, name),
                f"{path or 'no options file'} with variants.{name}.llm_api_options",
            )
        )
    return configurations


def launch_captures(task_data: Mapping[str, Any]) -> list[LaunchCapture]:
    """Each unit's capture lists by launch path, then each configuration variant's.

    Raises ``ValueError`` on a ``cuda_graph_config`` the LLM API refuses.
    """
    return [_launch_capture(*configuration) for configuration in _launch_configurations(task_data)]


# ------------------------------------------------------------ KV-cache configuration
#
# The same split as the capture list, for `KvCacheConfig`: the quickstart's
# flags cover only some of its keys. The one that matters is
# `mamba_state_config`: without a snapshot policy the engine turns block reuse
# off for a hybrid linear-attention model (`_validate_and_adjust_mamba_snapshot_config`
# in `tensorrt_llm/_torch/pyexecutor/model_loader.py`). The keys the quickstart
# has no flag for are handed to the harness, which merges them into the
# `KvCacheConfig` the quickstart builds.

#: The ``KvCacheConfig`` keys ``quickstart_advanced.py``'s ``setup_llm`` sets
#: from its own flags, at the flag's default when the argv has none; every
#: other key of the options' ``kv_cache_config`` is handed over.
QUICKSTART_KV_CACHE_KEYS: frozenset[str] = frozenset(
    {
        "enable_block_reuse",  # --disable_kv_cache_reuse
        "free_gpu_memory_fraction",  # --kv_cache_fraction
        "dtype",  # --kv_cache_dtype
        "tokens_per_block",  # --tokens_per_block
        "use_kv_cache_manager_v2",  # --use_kv_cache_manager_v2
        "mamba_ssm_cache_dtype",  # --mamba_ssm_cache_dtype
        "mamba_ssm_stochastic_rounding",  # --mamba_ssm_stochastic_rounding
        "mamba_ssm_philox_rounds",  # --mamba_ssm_philox_rounds
        "event_buffer_max_size",  # --log_kv_cache_events
    }
)

#: The handed-over keys, as a JSON object, in the launch prefix of a gate the
#: quickstart builds (as ``TRTLLM_MODELING_V2`` is). The harness reads it
#: (``checks/harness/engine_driver.py``, ``quickstart_shim.py``); it is absent
#: when the unit's options hand nothing over.
KV_CACHE_CONFIG_ENV = "MV2FLOW_KV_CACHE_CONFIG"

#: The quickstart flag that turns block reuse off.
DISABLE_REUSE_FLAG = "--disable_kv_cache_reuse"


def quickstart_kv_cache_config(options: Mapping[str, Any]) -> dict[str, Any]:
    """The ``kv_cache_config`` keys of LLM-API ``options`` the quickstart has no flag for."""
    block = options.get("kv_cache_config")
    if not isinstance(block, Mapping):
        return {}
    return {str(k): v for k, v in block.items() if str(k) not in QUICKSTART_KV_CACHE_KEYS}


def resolve_kv_cache_config(task_data: Mapping[str, Any], unit: str | None) -> dict[str, Any]:
    """The ``kv_cache_config`` keys ``unit``'s quickstart gates take from its options file."""
    path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    return quickstart_kv_cache_config(load_llm_api_options(path))


def with_kv_cache_config(
    environment: RunEnvironment, task_data: Mapping[str, Any], unit: str | None
) -> RunEnvironment:
    """``environment`` handing ``unit``'s KV-cache keys to the harness (:data:`KV_CACHE_CONFIG_ENV`).

    The variable is left out when the unit's options hand nothing over.
    """
    env = {k: v for k, v in environment.env.items() if k != KV_CACHE_CONFIG_ENV}
    handed = resolve_kv_cache_config(task_data, unit)
    if handed:
        env[KV_CACHE_CONFIG_ENV] = json.dumps(handed, sort_keys=True)
    return replace(environment, env=env)


def has_snapshot_policy(kv_cache: Mapping[str, Any]) -> bool:
    """Whether a ``kv_cache_config`` sets a Mamba state snapshot policy, which block reuse needs.

    The engine's test (``_validate_and_adjust_mamba_snapshot_config``): a
    positive ``mamba_state_config.periodic_snapshot_interval``, or any
    additional snapshot offset. ``mamba_state_cache_interval`` is the
    interval's deprecated alias, and ``block_reuse_config.policy:
    per_conversation`` sets the interval to zero.
    """
    state = kv_cache.get("mamba_state_config")
    state = state if isinstance(state, Mapping) else {}
    interval = state.get("periodic_snapshot_interval", kv_cache.get("mamba_state_cache_interval"))
    reuse = kv_cache.get("block_reuse_config")
    if isinstance(reuse, Mapping) and reuse.get("policy") == "per_conversation":
        interval = 0
    try:
        periodic = int(interval or 0) > 0
    except (TypeError, ValueError):
        periodic = False
    offsets = state.get("additional_snapshot_offsets_from_start") or state.get(
        "additional_snapshot_offsets_from_end"
    )
    return periodic or bool(offsets)


def quickstart_block_reuse(runtime: Mapping[str, Any]) -> bool:
    """Whether the quickstart gates of a unit with this runtime block ask for block reuse."""
    return bool(runtime.get("kv_cache_reuse", True)) and DISABLE_REUSE_FLAG not in _split_flags(
        runtime.get("extra_flags", ())
    )


@dataclass(frozen=True)
class LaunchKvCache:
    """The KV-cache configuration of one unit's, or one configuration variant's, gates.

    ``handed`` is what the quickstart gates take from ``options`` beyond their
    flags; ``quickstart_reuse`` and ``llm_api_reuse`` say whether each launch
    path asks for block reuse, and ``snapshots`` whether the options set a
    Mamba state snapshot policy.
    """

    name: str
    unit: str
    options: str
    handed: Mapping[str, Any]
    quickstart_reuse: bool
    llm_api_reuse: bool
    snapshots: bool


def launch_kv_caches(task_data: Mapping[str, Any]) -> list[LaunchKvCache]:
    """Each unit's KV-cache configuration, then each configuration variant's."""
    launches: list[LaunchKvCache] = []
    for name, unit, options, runtime, source in _launch_configurations(task_data):
        kv_cache = options.get("kv_cache_config")
        kv_cache = kv_cache if isinstance(kv_cache, Mapping) else {}
        launches.append(
            LaunchKvCache(
                name=name,
                unit=unit,
                options=source,
                handed=quickstart_kv_cache_config(options),
                quickstart_reuse=quickstart_block_reuse(runtime),
                llm_api_reuse=bool(kv_cache.get("enable_block_reuse", True)),
                snapshots=has_snapshot_policy(kv_cache),
            )
        )
    return launches


#: The head of what the engine logs when it turns block reuse off for a hybrid
#: model with no snapshot policy: "Disabling KV cache block reuse for the hybrid
#: Mamba model because no Mamba state snapshot policy is configured. ..." The
#: engine turns reuse off for other reasons too (some MLA configurations), with
#: other messages; this one is the reason an options file can fix.
REUSE_DISABLED_MARKER = "Disabling KV cache block reuse for the hybrid Mamba model"


def reuse_disabled_detail(output: str, *, options: str | None) -> str | None:
    """Why a gate fails when its engine turned block reuse off (``options``: the unit's file).

    ``None`` when the log does not say so. The engine logs it only for an
    engine asked for block reuse.
    """
    at = output.find(REUSE_DISABLED_MARKER)
    if at < 0:
        return None
    end = output.find("\n", at)
    line = output[output.rfind("\n", 0, at) + 1 : end if end >= 0 else None].strip()
    where = f"the unit's options file {options}" if options else "an options file for the unit"
    return (
        f"the unit asks for KV-cache block reuse and the engine turned it off: "
        f"{line[:240]!r}. Set `kv_cache_config.mamba_state_config` (a positive "
        f"`periodic_snapshot_interval` or snapshot offsets) in {where} "
        f"(accuracy.llm_api_options); the gates take it from there. Or turn "
        f"`kv_cache_reuse` off for the unit"
    )


# ------------------------------------------------------------ engine attestation
#
# The engine can run something other than what a gate asks: the LLM API fills
# in the arguments and the executor's worker changes them while it loads the
# model. So the harness prints what the engine runs once it is up (one
# `MV2FLOW_ENGINE` record per engine, `checks/harness/engine_facts.py`) and
# what it reused over the workload (one `MV2FLOW_REUSE` record,
# `checks/harness/engine_reuse.py`), and a gate compares them, field by field,
# with what it declared.

#: The fields a gate holds its engine to, in the order a difference is named.
ATTESTED_FIELDS: tuple[str, ...] = (
    "modeling_v2",
    "block_reuse",
    "mamba_snapshot",
    "kv_cache_dtype",
    "mamba_ssm_cache_dtype",
    "cuda_graph",
    "speculative",
    "max_num_tokens",
    "max_batch_size",
    "chunked_prefill",
    "tp",
    "ep",
    "attention_dp",
)

#: The harness's record of one engine (``checks/harness/engine_facts.py``).
ENGINE_MARKER = workload_io.ENGINE_MARKER
#: The harness's record of what one engine reused (``checks/harness/engine_reuse.py``).
REUSE_MARKER = workload_io.REUSE_MARKER
#: What each rank's own worker prints of its engine (``checks/harness/rankhook``).
RANK_ENGINE_MARKER = workload_io.RANK_ENGINE_MARKER
#: Set in every engine gate's environment, so each rank's worker prints its record.
RANK_FACTS_ENV = "MV2FLOW_RANK_FACTS"
#: Put first on every engine gate's ``PYTHONPATH``: its ``sitecustomize`` has
#: each rank's worker print the record.
RANK_HOOK_DIR = Path(__file__).resolve().parent / "harness" / "rankhook"
#: Where an attested value was read (:mod:`.observables`).
RANKS_SOURCE = observables.SOURCE_RANKS
LOG_SOURCE = observables.SOURCE_LOG
PROBE_SOURCE = observables.SOURCE_PROBE
CACHED_TOKENS_SOURCE = observables.SOURCE_CACHED
#: Why an accuracy verdict's engine is not attested: the run printed no engine
#: record (the eval shim found no ``LLM`` class to wrap in the tree's
#: ``trtllm-eval``).
EVAL_NOT_ATTESTED = "the evaluation printed no engine record"

#: ``modeling_v2 newmodel: bound MambaHybridCacheManagerV2 #2: 4096 recurrent
#: slots, ...`` -- what a target prints, once per rank, each time it binds a
#: recurrent-state pool; the highest-numbered one is the pool the engine serves from.
_BIND_LINE = re.compile(r"\bbound (\w+) #(\d+): (\d+) recurrent slots")
#: ``[MemUsageChange] Allocated 29.11 GiB for max tokens in paged KV cache
#: (524288).`` -- the C++ KV-cache manager, once per rank and pool, at INFO.
_KV_ALLOCATED = re.compile(r"Allocated [0-9.]+ GiB for max tokens in paged KV cache \((\d+)\)")

#: What the worker resolves each KV-cache dtype a gate can name to.
_KV_QUANT_FOR: dict[str, str] = {"fp8": "FP8", "fp8_ds_mla": "FP8", "nvfp4": "NVFP4"}
_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
}
_GIB_PER_MIB = 1.0 / 1024.0


def _flag_value(words: Sequence[str], flag: str) -> str | None:
    """The value of the last ``flag`` in ``words``, as ``--f=v`` or ``--f v``; ``None`` if absent."""
    value: str | None = None
    for index, word in enumerate(words):
        if word == flag and index + 1 < len(words) and not words[index + 1].startswith("--"):
            value = words[index + 1]
        elif word.startswith(f"{flag}="):
            value = word.split("=", 1)[1]
    return value


def declared_snapshot_policy(kv_cache: Mapping[str, Any]) -> dict[str, Any]:
    """The snapshot policy handed-over ``kv_cache_config`` keys ask for, as the engine holds it.

    As ``KvCacheConfig``'s validators read the keys: ``mamba_state_cache_interval``
    is the interval's deprecated alias, and ``block_reuse_config.policy:
    per_conversation`` sets the interval to zero.
    """
    state = kv_cache.get("mamba_state_config")
    state = state if isinstance(state, Mapping) else {}
    interval = state.get("periodic_snapshot_interval", kv_cache.get("mamba_state_cache_interval"))
    reuse = kv_cache.get("block_reuse_config")
    if isinstance(reuse, Mapping) and reuse.get("policy") == "per_conversation":
        interval = 0
    return {
        "periodic_snapshot_interval": int(interval or 0),
        "additional_snapshot_offsets_from_start": [
            int(o) for o in state.get("additional_snapshot_offsets_from_start") or ()
        ],
        "additional_snapshot_offsets_from_end": [
            int(o) for o in state.get("additional_snapshot_offsets_from_end") or ()
        ],
    }


def declared_engine(
    words: Sequence[str],
    *,
    mode: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What an engine the quickstart builds from ``words`` must run, field by field.

    ``words`` is the gate's quickstart argv (:func:`quickstart_words`). The
    unit's LLM-API ``options`` add what no flag carries: the snapshot policy
    among the handed-over KV-cache keys, and a cache dtype the argv does not
    state. ``mode`` is the target selection the gate exported. A field that
    neither states is left out: the engine's value is recorded, not judged.
    """
    options = options or {}
    kv_options = options.get("kv_cache_config")
    kv_options = kv_options if isinstance(kv_options, Mapping) else {}
    declared: dict[str, Any] = {}
    if mode:
        declared["modeling_v2"] = mode
    declared["block_reuse"] = DISABLE_REUSE_FLAG not in words
    try:
        declared["mamba_snapshot"] = declared_snapshot_policy(quickstart_kv_cache_config(options))
    except (TypeError, ValueError):
        pass  # the engine refuses such keys itself; there is nothing to compare
    for name, flag, key in (
        ("kv_cache_dtype", "--kv_cache_dtype", "dtype"),
        ("mamba_ssm_cache_dtype", "--mamba_ssm_cache_dtype", "mamba_ssm_cache_dtype"),
    ):
        value = _flag_value(words, flag)
        if value is None:
            value = kv_options.get(key)
        if value is not None:
            declared[name] = str(value)
    capture = quickstart_capture(words)
    declared["cuda_graph"] = (
        None
        if capture is None
        else {"batch_sizes": list(capture.batch_sizes), "enable_padding": capture.enable_padding}
    )
    algorithm = _flag_value(words, "--spec_decode_algo")
    declared["speculative"] = (
        None
        if algorithm is None
        else {
            "algorithm": algorithm,
            "max_draft_len": int(_flag_value(words, "--spec_decode_max_draft_len") or 1),
        }
    )
    for name, flag in (("max_num_tokens", "--max_num_tokens"), ("max_batch_size", BATCH_SIZE_FLAG)):
        value = _flag_value(words, flag)
        if value is not None:
            declared[name] = int(value)
    declared["chunked_prefill"] = "--enable_chunked_prefill" in words
    declared["tp"] = int(_flag_value(words, "--tp_size") or 1)
    # Unset (-1) beside an unset MoE TP width, `Mapping` puts every expert on every rank.
    expert = int(_flag_value(words, "--moe_ep_size") or -1)
    declared["ep"] = expert if expert > 0 else 1
    declared["attention_dp"] = "--enable_attention_dp" in words
    return declared


def engine_declaration(
    task_data: Mapping[str, Any],
    unit: str | None,
    words: Sequence[str],
    *,
    mode: str | None,
) -> dict[str, Any]:
    """:func:`declared_engine` for ``unit``'s gate argv ``words``, over the unit's options file."""
    path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    return declared_engine(words, mode=mode, options=load_llm_api_options(path))


def _dtype_name(value: Any) -> str:
    text = str(value).lower().replace("torch.", "")
    return _DTYPE_ALIASES.get(text, text)


def _show(field_name: str, value: Any) -> str:
    """One field's value as a verdict names it."""
    if value is None:
        return {"cuda_graph": "no CUDA graphs", "speculative": "no speculation"}.get(
            field_name, "none"
        )
    if isinstance(value, bool):
        return "on" if value else "off"
    if field_name == "cuda_graph" and isinstance(value, Mapping):
        sizes = [int(b) for b in value.get("batch_sizes") or ()]
        padded = "padded" if value.get("enable_padding") else "unpadded"
        return f"{len(sizes)} batches [{format_batches(sizes)}], {padded}"
    if field_name == "speculative" and isinstance(value, Mapping):
        return f"{value.get('algorithm')} at draft length {value.get('max_draft_len')}"
    if field_name == "mamba_snapshot" and isinstance(value, Mapping):
        return (
            f"interval {value.get('periodic_snapshot_interval')}, offsets from start "
            f"{value.get('additional_snapshot_offsets_from_start')}, from end "
            f"{value.get('additional_snapshot_offsets_from_end')}"
        )
    return str(value)


def _engine_difference(field_name: str, declared: Any, config: Mapping[str, Any]) -> str | None:
    """How the engine's ``field_name`` differs from ``declared``; ``None`` when it does not.

    The two cache dtypes are compared as asked for (the dtype the LLM API
    holds) and, where the rank-0 worker resolved a named one, as resolved: an
    engine asked for NVFP4 KV that falls back to FP8 ran FP8.
    """
    got = config[field_name]
    if field_name in ("kv_cache_dtype", "mamba_ssm_cache_dtype"):
        want = _dtype_name(declared)
        if _dtype_name(got) != want:
            return f"engine {got}"
        if field_name == "kv_cache_dtype":
            expected = _KV_QUANT_FOR.get(want)
            resolved = config.get("kv_cache_quant")
            if expected and "kv_cache_quant" in config and str(resolved).upper() != expected:
                return f"engine asked for {got} and resolved {resolved}"
        else:
            resolved = config.get("mamba_ssm_cache_dtype_resolved")
            if want != "auto" and resolved is not None and _dtype_name(resolved) != want:
                return f"engine asked for {got} and resolved {resolved}"
        return None
    if field_name == "speculative" and isinstance(declared, Mapping) and isinstance(got, Mapping):
        same = str(declared.get("algorithm")).upper() == str(got.get("algorithm")).upper() and int(
            declared.get("max_draft_len") or 0
        ) == int(got.get("max_draft_len") or 0)
        return None if same else f"engine {_show(field_name, got)}"
    if field_name == "cuda_graph" and isinstance(declared, Mapping) and isinstance(got, Mapping):
        same = [int(b) for b in declared.get("batch_sizes") or ()] == [
            int(b) for b in got.get("batch_sizes") or ()
        ] and bool(declared.get("enable_padding")) == bool(got.get("enable_padding"))
        return None if same else f"engine {_show(field_name, got)}"
    return None if declared == got else f"engine {_show(field_name, got)}"


def parse_engine_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_ENGINE`` record in ``output``, in order."""
    return workload_io.parse_marked(output, ENGINE_MARKER)


def parse_recurrent_slots(output: str) -> list[int]:
    """The recurrent-state slots of the pool each rank serves from: its last bind line's."""
    binds = [(int(n), int(slots)) for _, n, slots in _BIND_LINE.findall(output)]
    if not binds:
        return []
    last = max(n for n, _ in binds)
    return [slots for n, slots in binds if n == last]


def parse_kv_allocations(output: str) -> list[int]:
    """The max tokens of every paged KV-cache pool the C++ manager logged allocating."""
    return [int(tokens) for tokens in _KV_ALLOCATED.findall(output)]


def parse_rank_records(output: str) -> dict[int, dict[str, Any]]:
    """Each rank's last ``MV2FLOW_RANK_ENGINE`` record in ``output``, by rank."""
    ranks: dict[int, dict[str, Any]] = {}
    for record in workload_io.parse_marked(output, RANK_ENGINE_MARKER):
        rank = record.get("rank")
        if isinstance(rank, int) and not isinstance(rank, bool):
            ranks[rank] = record
    return ranks


def parse_reuse_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_REUSE`` record in ``output``, in order."""
    return workload_io.parse_marked(output, REUSE_MARKER)


def _count(value: Any) -> int | None:
    """``value`` as a count, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def reuse_evidence(reuse: Mapping[str, Any]) -> tuple[bool, str] | None:
    """Whether the engine ran block reuse, by what it reused, and where that was read; ``None``.

    The probe answers either way: its second prompt resumed from the cache,
    or the engine computed it from the start. The workload answers only yes:
    a request that resumed shows reuse on, and none resuming may only mean
    that no two requests shared a prefix the engine could resume from.
    """
    probe = reuse.get("probe")
    if isinstance(probe, Mapping) and _count(probe.get("reused")) is not None:
        return _count(probe["reused"]) > 0, PROBE_SOURCE
    if (_count(reuse.get("hits")) or 0) > 0:
        return True, CACHED_TOKENS_SOURCE
    return None


def _holds_recurrent_state(slots: Sequence[int], config: Mapping[str, Any]) -> bool:
    """Whether the engine keeps recurrent state: it bound a recurrent pool, or runs a snapshot policy."""
    if slots:
        return True
    policy = config.get("mamba_snapshot")
    return isinstance(policy, Mapping) and has_snapshot_policy({"mamba_state_config": policy})


@dataclass(frozen=True)
class EngineAttestation:
    """What one engine a gate built ran, held against what the gate declared."""

    #: Which of the gate's engines this is (``require``, ``envelope``, a witness
    #: leg); empty for a gate with one.
    leg: str
    #: Whether the harness printed a record for this engine.
    reported: bool
    declared: Mapping[str, Any] = field(default_factory=dict)
    #: What the engine runs, by field (``config`` of the harness record).
    config: Mapping[str, Any] = field(default_factory=dict)
    #: Where each value of ``config`` was read: ``worker``, ``llm.args`` or
    #: :data:`LOG_SOURCE`.
    sources: Mapping[str, str] = field(default_factory=dict)
    #: One line per declared field the engine ran differently.
    mismatches: tuple[str, ...] = ()
    #: Declared fields the engine's record does not carry.
    not_attested: tuple[str, ...] = ()
    #: Per rank, from the bind lines: the recurrent slots of the pool served from.
    recurrent_slots: tuple[int, ...] = ()
    #: The rank-0 worker's KV-cache pool, in tokens.
    kv_max_tokens: int | None = None
    #: Per rank and pool, from the KV-cache manager's INFO line, when logged.
    kv_log_tokens: tuple[int, ...] = ()
    #: Per device: ``{"device", "free_gib", "total_gib"}`` once the engine was up.
    devices: tuple[Mapping[str, Any], ...] = ()
    memory_source: str = ""
    #: Why the engine is not attested, or what the harness could not read.
    note: str = ""
    #: What the engine reused over the workload: the harness's ``MV2FLOW_REUSE``
    #: record (``checks/harness/engine_reuse.py``), plus ``recurrent`` (the
    #: engine keeps recurrent state) and, when it does, ``snapshot_restores``
    #: (the requests that resumed, each from a recurrent-state snapshot).
    #: Empty when the run printed none.
    reuse: Mapping[str, Any] = field(default_factory=dict)
    #: The ranks whose own worker reported its engine (``MV2FLOW_RANK_ENGINE``).
    ranks: tuple[int, ...] = ()
    #: How many ranks the engine runs: its TP width times its PP width.
    world: int = 1

    def unreported_ranks(self) -> tuple[int, ...]:
        """The engine's ranks whose own worker printed no readable record."""
        return tuple(r for r in range(self.world) if r not in self.ranks)

    def record(self) -> dict[str, Any]:
        """The JSON-able form a :class:`GateResult` carries in ``engine``."""
        return {
            "leg": self.leg,
            "reported": self.reported,
            "declared": dict(self.declared),
            "attested": dict(self.config),
            "sources": dict(self.sources),
            "mismatches": list(self.mismatches),
            "not_attested": list(self.not_attested),
            "pools": {
                "recurrent_slots": list(self.recurrent_slots),
                "kv_max_tokens": self.kv_max_tokens,
                "kv_log_tokens": list(self.kv_log_tokens),
            },
            "memory": {"source": self.memory_source, "devices": [dict(d) for d in self.devices]},
            "reuse": dict(self.reuse),
            "ranks": list(self.ranks),
            "world": self.world,
            "note": self.note,
        }

    def difference(self) -> str:
        """The failing verdict's detail: every field the engine ran differently."""
        where = f" of the {self.leg} leg" if self.leg else ""
        return (
            f"the engine{where} did not run the declared configuration -- "
            + "; ".join(self.mismatches)
            + ". The gate's argv and handed-over KV-cache keys come from the unit's runtime "
            "block and options file; the engine's values were read back once it was up "
            f"({ENGINE_MARKER}). A verdict on this engine would certify a configuration "
            "nobody declared"
        )


def attest_engine(
    output: str,
    declared: Mapping[str, Any] | None,
    *,
    leg: str = "",
    absent: str = "",
) -> EngineAttestation:
    """Hold the engine whose log is ``output`` to ``declared``.

    The last ``MV2FLOW_ENGINE`` record is the engine's and each rank's last
    ``MV2FLOW_RANK_ENGINE`` record its worker's; a field every reporting rank
    carries is held to ``declared`` on each rank. A declared field no record
    carries is not attested, never counted as agreeing. ``declared`` ``None``
    compares nothing, and ``absent`` says why.
    """
    want = dict(declared or {})
    slots = tuple(parse_recurrent_slots(output))
    kv_log = tuple(parse_kv_allocations(output))
    reused = parse_reuse_records(output)
    reuse = dict(reused[-1]) if reused else {}
    records = parse_engine_records(output)
    record = records[-1] if records else None
    ranks = parse_rank_records(output)
    rank_configs = {
        r: dict(rec["config"])
        for r, rec in sorted(ranks.items())
        if isinstance(rec.get("config"), Mapping)
    }
    rank_notes = [
        f"rank {r}'s worker could not be read: {rec.get('error') or rec}"
        for r, rec in sorted(ranks.items())
        if r not in rank_configs
    ]
    harness_read = record is not None and isinstance(record.get("config"), Mapping)
    if not harness_read and not rank_configs:
        if record is not None:
            note = f"the harness could not read the engine: {record.get('error') or record}"
        else:
            note = absent or f"the run printed no {ENGINE_MARKER} record"
        if reuse:
            reuse["recurrent"] = bool(slots)
        return EngineAttestation(
            leg=leg,
            reported=False,
            declared=want,
            not_attested=tuple(f for f in ATTESTED_FIELDS if f in want),
            recurrent_slots=slots,
            kv_log_tokens=kv_log,
            note="; ".join([note, *rank_notes])[:300],
            reuse=reuse,
        )
    record = record if harness_read else {}
    config = dict(record.get("config") or {})
    sources = {str(k): str(v) for k, v in (record.get("sources") or {}).items()}
    if rank_configs:
        lowest = rank_configs[min(rank_configs)]
        for name, value in lowest.items():
            if all(name in c for c in rank_configs.values()):
                config[name] = value
                sources[name] = RANKS_SOURCE
    if "block_reuse" not in config and REUSE_DISABLED_MARKER in output:
        config["block_reuse"] = False
        sources["block_reuse"] = LOG_SOURCE
    evidence = reuse_evidence(reuse)
    if "block_reuse" not in config and evidence is not None:
        config["block_reuse"], sources["block_reuse"] = evidence
    if reuse:
        reuse["recurrent"] = _holds_recurrent_state(slots, config)
        if reuse["recurrent"] and _count(reuse.get("hits")) is not None:
            reuse["snapshot_restores"] = _count(reuse["hits"])
    mismatches: list[str] = []
    unattested: list[str] = []
    for name in ATTESTED_FIELDS:
        if name not in want:
            continue
        if name not in config:
            unattested.append(name)
            continue
        if sources.get(name) == RANKS_SOURCE:
            differing: dict[str, list[int]] = {}
            for rank, rank_config in rank_configs.items():
                difference = _engine_difference(name, want[name], rank_config)
                if difference:
                    differing.setdefault(difference, []).append(rank)
            for difference, at in differing.items():
                mismatches.append(
                    f"{name}: declared {_show(name, want[name])}, {difference} on rank(s) "
                    f"{', '.join(str(r) for r in at)} (read from {RANKS_SOURCE})"
                )
            continue
        difference = _engine_difference(name, want[name], config)
        if difference:
            mismatches.append(
                f"{name}: declared {_show(name, want[name])}, {difference} "
                f"(read from {sources.get(name, 'the engine')})"
            )
    memory = record.get("memory") if isinstance(record.get("memory"), Mapping) else {}
    devices = tuple(
        {
            "device": str(d.get("device")),
            "free_gib": round(float(d["free_mib"]) * _GIB_PER_MIB, 2),
            "total_gib": round(float(d["total_mib"]) * _GIB_PER_MIB, 2),
        }
        for d in memory.get("devices") or ()
        if isinstance(d, Mapping) and d.get("free_mib") is not None and d.get("total_mib")
    )
    pools = record.get("pools") if isinstance(record.get("pools"), Mapping) else {}
    errors = record.get("errors") if isinstance(record.get("errors"), Mapping) else {}
    unread = sorted(set(errors) & set(unattested))
    note = "; ".join(f"{name}: {errors[name]}" for name in unread)
    if errors.get("worker"):
        note = "; ".join(
            filter(None, [note, f"rank 0's worker could not be read: {errors['worker']}"])
        )
    elif not record.get("worker") and not rank_configs:
        note = "; ".join(
            filter(None, [note, "rank 0's worker ran in another process; read from llm.args"])
        )
    note = "; ".join(filter(None, [note, *rank_notes]))
    if memory.get("error") and not devices:
        note = "; ".join(filter(None, [note, f"free memory not read: {memory['error']}"]))
    kv_tokens = pools.get("kv_max_tokens")
    if kv_tokens is None and rank_configs:
        first = ranks[min(rank_configs)].get("pools")
        kv_tokens = first.get("kv_max_tokens") if isinstance(first, Mapping) else None
    world = 1
    for width in ("tp", "pp"):
        value = config.get(width)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            world *= value
    return EngineAttestation(
        leg=leg,
        reported=True,
        declared=want,
        config=config,
        sources=sources,
        mismatches=tuple(mismatches),
        not_attested=tuple(unattested),
        recurrent_slots=slots,
        kv_max_tokens=int(kv_tokens) if isinstance(kv_tokens, (int, float)) else None,
        kv_log_tokens=kv_log,
        devices=devices,
        memory_source=str(memory.get("source") or ""),
        note=note[:300],
        reuse=reuse,
        ranks=tuple(rank_configs),
        world=world,
    )


def reuse_shortfall(attestation: EngineAttestation, *, asks: bool) -> str | None:
    """Why a gate fails when its unit asks for block reuse and the engine reused nothing; ``None``.

    Judged only where the engine had something to reuse: the reuse probe's
    second prompt was computed from the start, or more requests than the
    engine runs at once shared a resumable prefix with an earlier request and
    none resumed. Nothing is judged when the responses did not report what
    they reused.
    """
    reuse = attestation.reuse
    if not asks or not reuse or not _count(reuse.get("reported")):
        return None
    leg = f" of the {attestation.leg} leg" if attestation.leg else ""
    advice = (
        ". So this gate measured the unit without the prefix reuse it serves with. A hybrid "
        "model resumes only where it holds a recurrent-state snapshot: check "
        "`kv_cache_config.mamba_state_config` in the unit's options file, and the log for the "
        "engine's reason; or turn `kv_cache_reuse` off for the unit if it is meant to run "
        "without reuse"
    )
    probe = reuse.get("probe")
    if isinstance(probe, Mapping) and _count(probe.get("reused")) == 0:
        first = _count(reuse.get("min_resume"))
        where = f"at token {first} or later" if first else "past token 0"
        return (
            f"the unit asks for KV-cache block reuse and the engine{leg} reused nothing: the "
            f"reuse probe sent one {probe.get('tokens')}-token prompt twice, the second after "
            f"the first had finished, and the engine computed the second from token 0 "
            f"(cached_tokens 0 on its response), where an engine that reuses resumes it "
            f"{where}" + advice
        )
    resumable = _count(reuse.get("resumable")) or 0
    at_once = _count(reuse.get("concurrency"))
    if at_once is not None and resumable > at_once and not _count(reuse.get("hits")):
        return (
            f"the unit asks for KV-cache block reuse and the engine{leg} reused nothing: "
            f"{resumable} of {reuse.get('requests')} requests shared more than "
            f"{reuse.get('min_resume')} prompt tokens with an earlier request, more than the "
            f"{at_once} the engine runs at once, and every response says it was computed from "
            f"token 0 (cached_tokens 0)" + advice
        )
    return None


def engine_failure(
    output: str,
    attested: EngineAttestation,
    *,
    options: str | None,
    asks: bool,
    prefix: str = "",
) -> str | None:
    """Why a gate fails on the engine it ran, in this order; ``None`` when it does not.

    The engine turned block reuse off (:func:`reuse_disabled_detail`, after
    ``prefix``), reused nothing although the unit ``asks`` for reuse
    (:func:`reuse_shortfall`), or ran a field other than declared.
    """
    reuse_off = reuse_disabled_detail(output, options=options)
    if reuse_off:
        return prefix + reuse_off
    shortfall = reuse_shortfall(attested, asks=asks)
    if shortfall:
        return shortfall
    return attested.difference() if attested.mismatches else None


def _describe_reuse(reuse: Mapping[str, Any]) -> list[str]:
    """The ``[engine ...]`` parts that say what the engine reused."""
    if reuse.get("error"):
        return [f"reuse not read ({str(reuse['error'])[:80]})"]
    parts: list[str] = []
    requests = _count(reuse.get("requests")) or 0
    hits = _count(reuse.get("hits")) or 0
    if requests and not _count(reuse.get("reported")):
        parts.append("reuse not reported by the engine")
    elif hits or _count(reuse.get("resumable")):
        text = f"reuse {hits}/{requests} requests"
        if hits:
            where = ", ".join(f"{p} x{n}" for p, n in (reuse.get("resumed_at") or {}).items())
            what = "recurrent state restored at" if reuse.get("recurrent") else "resumed at"
            text += f", {_count(reuse.get('reused_tokens')) or 0} tokens ({what} {where})"
        parts.append(text)
    probe = reuse.get("probe")
    if isinstance(probe, Mapping):
        resumed = _count(probe.get("reused"))
        if resumed is not None:
            restored = " (snapshot restored)" if resumed and reuse.get("recurrent") else ""
            parts.append(f"reuse probe resumed at {resumed} of {probe.get('tokens')}{restored}")
        else:
            why = probe.get("skipped") or probe.get("error") or "no reading"
            parts.append(f"reuse probe not read ({str(why)[:80]})")
    return parts


def _describe_engine_record(record: Mapping[str, Any]) -> str:
    leg = f"{record['leg']}: " if record.get("leg") else ""
    declared = record.get("declared") or {}
    compared = [name for name in ATTESTED_FIELDS if name in declared]
    differ = record.get("mismatches") or ()
    missing = record.get("not_attested") or ()
    parts: list[str] = []
    if not record.get("reported"):
        parts.append(
            f"NOT ATTESTED ({record.get('note') or 'no record'})"
            if compared
            else "configuration not attested"
        )
    elif compared:
        agreed = len(compared) - len(differ) - len(missing)
        head = f"{agreed}/{len(compared)} fields as declared"
        if differ:
            head += f", {len(differ)} DIFFER"
        if missing:
            head += f", NOT ATTESTED {', '.join(missing)}"
        parts.append(head)
    pools = record.get("pools") or {}
    slots = [int(s) for s in pools.get("recurrent_slots") or ()]
    if slots:
        same = len(set(slots)) == 1
        shown = (
            f"{slots[0]} x{len(slots)}" if same and len(slots) > 1 else "/".join(map(str, slots))
        )
        parts.append(f"recurrent slots {shown}")
    kv_tokens = pools.get("kv_max_tokens")
    if kv_tokens is None and pools.get("kv_log_tokens"):
        kv_tokens = pools["kv_log_tokens"][-1]
    if kv_tokens is not None:
        parts.append(f"KV {int(kv_tokens)} tokens")
    parts.extend(_describe_reuse(record.get("reuse") or {}))
    devices = (record.get("memory") or {}).get("devices") or ()
    if devices:
        low = min(devices, key=lambda d: float(d["free_gib"]))
        parts.append(f"min free {float(low['free_gib']):.2f} of {float(low['total_gib']):.2f} GiB")
    return f"{leg}{'; '.join(parts)}" if parts else ""


def describe_engine(records: Sequence[Mapping[str, Any]]) -> str:
    """A verdict summary's `` [engine ...]`` segment; empty for a gate that built none."""
    shown = [text for text in (_describe_engine_record(r) for r in records) if text]
    return f" [engine {' | '.join(shown)}]" if shown else ""


@dataclass(frozen=True)
class MemoryMargin:
    """How much device memory an engine must leave free once up before its verdict warns.

    ``runtime.min_free_memory_gib`` and ``runtime.min_free_memory_fraction``
    (of the device): the larger of the two is the threshold. The verdict
    still passes under it; its summary carries a ``LOW MARGIN`` warning.
    """

    min_free_gib: float = RUNTIME_MARGIN_DEFAULTS["min_free_memory_gib"]
    min_free_fraction: float = RUNTIME_MARGIN_DEFAULTS["min_free_memory_fraction"]

    def threshold_gib(self, total_gib: float) -> float:
        return max(self.min_free_gib, self.min_free_fraction * total_gib)

    def note(self, records: Sequence[Mapping[str, Any]]) -> str | None:
        """The ``LOW MARGIN`` warning for the tightest device under the threshold; ``None``."""
        tightest: tuple[float, Mapping[str, Any], str] | None = None
        for record in records:
            for device in (record.get("memory") or {}).get("devices") or ():
                free, total = float(device["free_gib"]), float(device["total_gib"])
                short = free - self.threshold_gib(total)
                if short < 0 and (tightest is None or short < tightest[0]):
                    tightest = (short, device, str(record.get("leg") or ""))
        if tightest is None:
            return None
        _, device, leg = tightest
        total = float(device["total_gib"])
        where = f" in the {leg} leg" if leg else ""
        return (
            f"LOW MARGIN: {float(device['free_gib']):.2f} GiB free on device "
            f"{device['device']}{where} once the engine was up, under the "
            f"{self.threshold_gib(total):.2f} GiB threshold (the larger of "
            f"runtime.min_free_memory_gib {self.min_free_gib:g} and "
            f"runtime.min_free_memory_fraction {self.min_free_fraction:g} of {total:.2f} GiB). "
            f"The next change that adds memory to this configuration runs it out"
        )


def resolve_memory_margin(task_data: Mapping[str, Any], unit: str | None) -> MemoryMargin:
    """``unit``'s :class:`MemoryMargin`, from its runtime block."""
    runtime = resolve_runtime(task_data, unit)
    return MemoryMargin(
        min_free_gib=float(
            runtime.get("min_free_memory_gib", RUNTIME_MARGIN_DEFAULTS["min_free_memory_gib"])
        ),
        min_free_fraction=float(
            runtime.get(
                "min_free_memory_fraction", RUNTIME_MARGIN_DEFAULTS["min_free_memory_fraction"]
            )
        ),
    )


def engine_fields(
    attestations: Sequence[EngineAttestation],
    task_data: Mapping[str, Any],
    unit: str | None,
) -> dict[str, Any]:
    """The :class:`GateResult` fields a gate's engines fill: their records, and LOW MARGIN."""
    records = tuple(a.record() for a in attestations)
    note = resolve_memory_margin(task_data, unit).note(records)
    return {"engine": records, "warnings": (note,) if note else ()}


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


def _write_member_options(
    options: str | None,
    overlay: Mapping[str, Any],
    *,
    unit: str | None,
    member: str | None,
    workdir: str | Path | None,
    environment: RunEnvironment,
    task_data: Mapping[str, Any],
    beneath: Mapping[str, Any],
) -> str:
    """Merge a member's own LLM-API settings over its unit's options file; the merged file's path.

    ``beneath`` goes under the unit's file. Written beside the unit's file,
    else in ``workdir``, else in the temp dir (:func:`_refuse_temp_off_host`),
    and replaced whole (:func:`write_atomically`): gates of one member that run
    at once rewrite and read the same file.
    """
    base: dict[str, Any] = {}
    source = Path(options) if options else None
    if source is not None and source.is_file():
        base = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    merged = merge_options(merge_options(beneath, base), overlay)
    stem = f"{unit or 'unit'}.{member or 'member'}.llm_api.yaml"
    if source is not None:
        parent = source.parent
    elif workdir is not None:
        parent = Path(workdir)
        parent.mkdir(parents=True, exist_ok=True)
    else:
        _refuse_temp_off_host("the member's LLM-API options", environment, task_data)
        parent = Path(tempfile.gettempdir())
    target = parent / stem
    header = (
        f"# GENERATED for the `{member}` member of the `{unit}` unit. Do not edit.\n"
        f"# {source if source is not None else '(no unit options file)'} plus the\n"
        f"# settings that member's gate command cannot express as flags:\n"
        f"#   {', '.join(sorted(overlay))}\n"
        f"# See `gates.member_llm_api_overlay` for why each one is here.\n"
    )
    write_atomically(target, header + yaml.safe_dump(merged, sort_keys=True))
    return str(target)


def write_atomically(target: Path, text: str) -> None:
    """Replace ``target`` with ``text`` in one step: a reader sees the old file or the new one.

    The text goes to a temporary file in ``target``'s own directory, which
    ``os.replace`` then renames over it; a rename within one file system is
    atomic. A write that fails leaves ``target`` as it was and no temporary
    file behind.
    """
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    replaced = False
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, target)
        replaced = True
    finally:
        if not replaced:
            Path(handle.name).unlink(missing_ok=True)


def eval_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    benchmark: str,
    *,
    unit: str | None = None,
    member: str | None = None,
    workdir: str | Path | None = None,
) -> list[str]:
    """Build the accuracy gate's ``trtllm-eval`` argv.

    ``trtllm-eval`` takes its runtime configuration as an
    ``--extra_llm_api_options`` YAML: the unit's file, with the member's
    overlay, ``modeling_v2`` and the settings the runtime block states merged
    into a derived copy (:func:`_write_member_options`) under ``workdir`` when
    the unit declares none.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    parts = [
        "trtllm-eval",
        f'--model="{task_data["checkpoint_path"]}"',
        *topology_flags(
            str(task_data["parallel"]),
            environment.world_size,
            member,
            dialect=EVAL_DIALECT,
        ),
    ]
    options = resolve_llm_api_options(accuracy, unit)
    overlay = member_llm_api_overlay(member)
    beneath = launch_overlay(task_data, unit)
    mode = environment.env.get(MODELING_V2_ENV)
    if mode:
        # Target selection is the LLM-API argument `modeling_v2`; the engine
        # does not read the environment switch, and the argument defaults to "off".
        overlay = {**overlay, "modeling_v2": mode}
    if overlay or beneath:
        options = _write_member_options(
            options,
            overlay,
            unit=unit,
            member=member,
            workdir=workdir,
            environment=environment,
            task_data=task_data,
            beneath=beneath,
        )
    if options:
        parts.append(f'--extra_llm_api_options="{options}"')
    for flag in ("max_seq_len", "max_batch_size"):
        if accuracy.get(flag) is not None:
            parts.append(f"--{flag}={accuracy[flag]}")
    parts.append(benchmark)
    if accuracy.get("dataset_path"):
        parts.append(f'--dataset_path="{accuracy["dataset_path"]}"')
    # The protocol flags (`--apply_chat_template`, `--num_fewshot`, ...) are
    # options of the benchmark subcommand, so they go after its name, per unit
    # (the subcommands do not share an option set), and shell-quoted: an
    # argument like `--chat_template_kwargs={"enable_thinking": false}`
    # carries spaces and braces.
    parts.extend(
        shlex.quote(str(arg)) for arg in resolve_per_unit(accuracy, "benchmark_args", unit) or ()
    )
    return parts


def _with_mode(environment: RunEnvironment, mode: str) -> RunEnvironment:
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
# The gates run `checks/harness/engine_driver.py` under the tree's own
# interpreter. It builds the engine with the tree's own quickstart from the
# gate's argv and prints one `MV2FLOW_RECORD` JSON line per request; a gate
# reads those lines and nothing else of the log.

#: The driver script, run as a file under the measured tree's interpreter.
ENGINE_DRIVER = Path(__file__).resolve().parent / "harness" / "engine_driver.py"
#: The accuracy gate's ``trtllm-eval``, run in-process by this script so the
#: engine it builds is reported (``checks/harness/eval_shim.py``).
EVAL_SHIM = Path(__file__).resolve().parent / "harness" / "eval_shim.py"


def eval_harness_command(environment: RunEnvironment, parts: Sequence[str]) -> str:
    """The launch of ``parts`` (:func:`eval_command`'s argv) under the eval shim, as engine work.

    The shim runs the tree's own ``trtllm-eval`` on the argv and adds the
    engine's ``MV2FLOW_ENGINE`` and ``MV2FLOW_REUSE`` records to the output.
    """
    return environment.python_command(f"{shlex.quote(str(EVAL_SHIM))} {' '.join(parts)}")


def options_declaration(options: Mapping[str, Any], *, block_reuse: bool) -> dict[str, Any]:
    """What an engine built from LLM-API ``options`` must run: block reuse, and the cache dtypes named."""
    declared: dict[str, Any] = {"block_reuse": block_reuse}
    kv = options.get("kv_cache_config")
    kv = kv if isinstance(kv, Mapping) else {}
    for name, key in (
        ("kv_cache_dtype", "dtype"),
        ("mamba_ssm_cache_dtype", "mamba_ssm_cache_dtype"),
    ):
        if kv.get(key) is not None:
            declared[name] = str(kv[key])
    return declared


def llm_api_block_reuse(options: Mapping[str, Any]) -> bool:
    """Whether LLM-API ``options`` ask for KV-cache block reuse (``enable_block_reuse``, default on)."""
    kv_cache = options.get("kv_cache_config")
    if not isinstance(kv_cache, Mapping):
        return True
    return bool(kv_cache.get("enable_block_reuse", True))


#: The quickstart flag a declared batch limit replaces.
BATCH_SIZE_FLAG = "--max_batch_size"


def with_max_batch_size(flags: Sequence[str], max_batch_size: int | None) -> list[str]:
    """``flags`` with every ``--max_batch_size`` spelling replaced by one ``=N``; ``None`` keeps them.

    ``--max_batch_size=8``, ``--max_batch_size 8`` as one string, and the flag
    and its value as two are all removed, so the argv states one limit.
    """
    if max_batch_size is None:
        return list(flags)
    kept: list[str] = []
    skip_value = False
    for flag in flags:
        if skip_value:
            skip_value = False
            continue
        word = str(flag)
        if word == BATCH_SIZE_FLAG:
            skip_value = True
            continue
        if word.startswith((f"{BATCH_SIZE_FLAG}=", f"{BATCH_SIZE_FLAG} ")):
            continue
        kept.append(word)
    return [*kept, f"{BATCH_SIZE_FLAG}={int(max_batch_size)}"]


def quickstart_script(task_data: Mapping[str, Any]) -> Path:
    """The measured tree's ``examples/llm-api/quickstart_advanced.py``."""
    return (
        Path(str(task_data["trtllm_repo_path"])) / "examples" / "llm-api" / "quickstart_advanced.py"
    )


def quickstart_flags(
    task_data: Mapping[str, Any], unit: str | None, features: Any | None
) -> list[str]:
    """The flags ``unit``'s quickstart gates run with, as the unit states them.

    Its runtime block and CUDA-graph capture list, the speculator at its draft
    depth, and the settings its options file states that both launchers take
    (:func:`launch_flags`). ``features`` defaults to the run's.
    """
    stated = [
        *runtime_flags(
            resolve_runtime(task_data, unit), capture=resolve_capture_list(task_data, unit)
        ),
        *speculation_flags(
            task_data.get("features") if features is None else features,
            max_draft_len=resolve_max_draft_len(task_data.get("accuracy") or {}, unit),
        ),
    ]
    return [*stated, *launch_flags(task_data, unit, stated)]


def quickstart_words(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    features: Any | None,
    max_tokens: int,
    member: str | None,
) -> list[str]:
    """The quickstart argv as a list of plain words, for the engine driver's spec.

    The topology and :func:`quickstart_flags`, each split as the shell would
    split it. The KV-cache keys no flag carries reach the engine through the
    launch environment instead (:func:`with_kv_cache_config`).
    """
    words = [
        f"--model_dir={task_data['checkpoint_path']}",
        f"--max_tokens={int(max_tokens)}",
        *topology_flags(str(task_data["parallel"]), environment.world_size, member),
    ]
    for flag in quickstart_flags(task_data, unit, features):
        words.extend(shlex.split(str(flag)))
    return words


def _refuse_temp_off_host(
    what: str, environment: RunEnvironment | None, task_data: Mapping[str, Any]
) -> None:
    """Raise when a file in this process's temp dir would be read on another host.

    When a scheduler stands behind the task's environment kind or
    ``environment``'s (:func:`~.environment.scheduler_owns_devices`), the job
    may start on a host whose temp dir is not this one.
    """
    kinds = [str((task_data.get("environment") or {}).get("kind", "local"))]
    if environment is not None:
        kinds.append(environment.kind)
    scheduled = [kind for kind in kinds if scheduler_owns_devices(kind)]
    if scheduled:
        raise ValueError(
            f"{what}: no workdir was given, and under environment kind `{scheduled[0]}` "
            f"the job may run on another host, which cannot read this process's temp "
            f"dir ({tempfile.gettempdir()}). Pass a directory on the shared filesystem; "
            f"the workflow passes <workspace>/gates/<unit>."
        )


def gate_workdir(
    workdir: str | Path | None,
    name: str,
    task_data: Mapping[str, Any],
    *,
    environment: RunEnvironment | None = None,
) -> Path:
    """Where a gate writes the files its job reads.

    ``workdir`` when given; otherwise a directory under this process's temp
    dir named for the gate and the task (its tree and checkpoint), which
    serves a local run only (:func:`_refuse_temp_off_host`).
    """
    if workdir is not None:
        directory = Path(workdir)
    else:
        _refuse_temp_off_host(f"the {name} gate", environment, task_data)
        key = f"{task_data.get('trtllm_repo_path')}|{task_data.get('checkpoint_path')}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        directory = Path(tempfile.gettempdir()) / f"mv2flow-{name}-{digest}"
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
    task_data: Mapping[str, Any],
    argv: Sequence[str],
    requests: Sequence[Mapping[str, Any]] = (),
    dataset: Mapping[str, Any] | None = None,
    max_tokens: int = 64,
    concurrency: int = 1,
    ignore_eos: bool = False,
    reuse_probe: bool = False,
) -> Path:
    """Write one engine-driver spec (see ``checks/harness/engine_driver.py``).

    ``reuse_probe`` has the driver send one prompt twice after the workload
    (``checks/harness/engine_reuse.py``).
    """
    spec = {
        "quickstart": str(quickstart_script(task_data)),
        "quickstart_argv": [str(a) for a in argv],
        "requests": [dict(r) for r in requests],
        "dataset": dict(dataset) if dataset else None,
        "max_tokens": int(max_tokens),
        "concurrency": int(concurrency),
        "ignore_eos": bool(ignore_eos),
    }
    if reuse_probe:
        spec["reuse_probe"] = True
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def engine_driver_command(
    environment: RunEnvironment, spec_path: str | Path, *, script: Path = ENGINE_DRIVER
) -> str:
    """The launch of the driver ``script`` over ``spec_path``, as engine work.

    The quickstart and its argv are spelled on the command line as well as in
    the spec (the driver takes the command line's), so a command read from a
    log says which tree it built from and with which flags.
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    quickstart = spec.get("quickstart")
    argv = [str(a) for a in spec.get("quickstart_argv") or ()]
    named = f" --quickstart {shlex.quote(str(quickstart))}" if quickstart else ""
    flags = f" -- {' '.join(shlex.quote(a) for a in argv)}" if argv else ""
    return environment.python_command(
        f"{shlex.quote(str(script))} --spec {shlex.quote(str(spec_path))}{named}{flags}"
    )


def parse_driver_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_RECORD`` the engine driver printed, in request order."""
    records = workload_io.parse_marked(output, workload_io.RECORD_MARKER)
    return sorted(records, key=lambda r: int(r.get("index", 0)))


def parse_eval_score(stdout: str) -> float | None:
    """Extract the final accuracy from a ``trtllm-eval`` run's output.

    MMLU prints its own weighted average, everything else comes through
    lm-eval. The **last** match wins: a long run also prints partial-score
    estimates.
    """
    for pattern in (_MMLU_SCORE, _LM_EVAL_SCORE):
        matches = pattern.findall(stdout)
        if matches:
            return float(matches[-1])
    return None


@dataclass(frozen=True)
class AcceptanceReading:
    """One leg's acceptance measurement, with the requests it came from."""

    #: The gate's number: iteration-weighted over the requests where they are
    #: known (see :func:`parse_acceptance`), else the summary statistic printed.
    value: float
    #: ``(tokens, iterations)`` per request, in print order. Empty when the
    #: output carried only a summary statistic.
    requests: tuple[tuple[int, int], ...] = ()
    #: A digest of each request's prompt, parallel to ``requests``, where the
    #: run printed one (the engine driver does): what pairs a prompt's rate
    #: with the same prompt's rate in another leg or on another tree.
    prompts: tuple[str, ...] = ()

    def rates(self) -> dict[str, float]:
        """Accepted tokens per iteration, per prompt, keyed by prompt digest."""
        names = self.prompts if len(self.prompts) == len(self.requests) else ()
        return {
            (names[n] if names else str(n)): t / i for n, (t, i) in enumerate(self.requests) if i
        }

    @property
    def draft_iterations(self) -> int | None:
        """Generation iterations over the whole workload; ``None`` if unknown.

        The first streamed iteration of every request is its context step,
        which verifies no draft, so it is not counted.
        """
        if not self.requests:
            return None
        return sum(max(0, iterations - 1) for _, iterations in self.requests)

    def breakdown(self) -> str:
        """Per-prompt rates and their plain mean, beside the weighted value."""
        rates = [(n, t, i) for n, (t, i) in enumerate(self.requests) if i]
        if not rates:
            return ""
        per_prompt = ", ".join(f"[{n}] {t / i:.2f} ({t} tok/{i} it)" for n, t, i in rates)
        mean = sum(t / i for _, t, i in rates) / len(rates)
        return (
            f"per prompt {per_prompt}; prompt mean {mean:.2f}, "
            f"iteration-weighted {self.value:.2f} over {self.draft_iterations} "
            f"generation iterations"
        )


def parse_acceptance(stdout: str) -> AcceptanceReading | None:
    """Extract the acceptance measurement from a driver, eval, bench or quickstart run.

    The engine driver's records come first. Requests combine
    **iteration-weighted**, as ``sum(tokens) / sum(iterations)``, not as the
    mean of their ratios, which would let one short request dominate a long one.
    """
    records = parse_driver_records(stdout)
    if records:
        parts = tuple((int(r.get("tokens", 0)), int(r.get("iterations", 0))) for r in records)
        iterations = sum(i for _, i in parts)
        if iterations:
            return AcceptanceReading(
                value=sum(t for t, _ in parts) / iterations,
                requests=parts,
                prompts=tuple(str(r.get("prompt_sha", n)) for n, r in enumerate(records)),
            )
    for pattern in (_ACCEPTANCE_LENGTH, _ACCEPTANCE_LENGTH_ALT):
        matches = pattern.findall(stdout)
        if matches:
            return AcceptanceReading(value=float(matches[-1]))
    parts = tuple((int(t), int(i)) for t, i in _ACCEPT_RATE_PARTS.findall(stdout))
    iterations = sum(i for _, i in parts)
    if iterations:
        return AcceptanceReading(value=sum(t for t, _ in parts) / iterations, requests=parts)
    return None


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
            f"where the score came from) before the accuracy gate can run."
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


def run_boot_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    spec: BootSpec | None = None,
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    mode: str = MODE_REQUIRE,
    features: Any | None = None,
    unit: str | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
) -> GateResult:
    """Cold-start the engine and check it generates, through the engine driver.

    ``features`` is the features the target under test implements, not those
    the run will finish with (default the run's). ``spec`` is the workload
    (:func:`resolve_boot_spec`); with ``spec.greedy`` the engine's first
    greedy tokens must be the HF reference's up to a near-tie
    (:func:`judge_greedy`). The engine is held to what the gate declared
    (:func:`engine_failure`). ``mode`` other than ``require`` is a control run
    on the stock implementation, recorded as ``boot[<mode>]``.
    """
    spec = spec or BootSpec()
    env = with_kv_cache_config(_with_mode(environment, mode), task_data, unit)
    gate_name = "boot" if mode == MODE_REQUIRE else f"boot[{mode}]"
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=gate_name, member=member, passed=False, detail=state.block_detail())
    argv = quickstart_words(
        task_data,
        environment,
        unit=unit,
        features=task_data.get("features") if features is None else features,
        max_tokens=spec.max_tokens,
        member=member,
    )
    requests = spec.requests(task_data.get("trtllm_repo_path"))
    declared_requests = len(requests)
    limits = EngineLimits.from_words(argv)
    greedy = greedy_workload(spec.greedy, limits) if spec.greedy is not None else None
    if greedy is not None and greedy.error:
        return _verdict(name=gate_name, member=member, passed=False, detail=greedy.error)
    if greedy is not None:
        requests += greedy.requests()
    subjects = observables.subject_fields(task_data, unit)
    chunks = observables.about(subjects, observables.CHUNKED)
    if chunks and not any(
        len(p.input_ids) > limits.max_num_tokens for p in (greedy.prompts if greedy else ())
    ):
        length = observables.chunk_probe_tokens(limits.max_num_tokens, limits.max_seq_len)
        if length is None:
            return _verdict(
                name=gate_name,
                member=member,
                passed=False,
                detail=f"this unit is about chunked prefill, and its engine's limits "
                f"(max_num_tokens {limits.max_num_tokens}, max_seq_len {limits.max_seq_len}) "
                f"leave no room for a prompt the engine would have to split",
            )
        requests.append({"id": "chunk", "probe_tokens": length, "max_tokens": 1})
    spec_path = write_driver_spec(
        gate_workdir(workdir, "boot", task_data, environment=environment)
        / f"boot.{unit or 'unit'}.{member or 'member'}.json",
        task_data=task_data,
        argv=argv,
        requests=requests,
        max_tokens=spec.max_tokens,
        ignore_eos=greedy is not None,
        reuse_probe=True,
    )
    command = engine_driver_command(env, spec_path)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(name=gate_name, member=member, command=command, duration_s=elapsed, stdout_tail=tail)

    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out:
        return _verdict(
            passed=False,
            detail=f"timed out after {timeout}s — a boot gate that takes this long is "
            f"not catching catastrophes any more",
            **ran,
        )
    if not result.ok:
        return _verdict(passed=False, detail=f"exited {result.returncode}", **ran)
    declared = engine_declaration(task_data, unit, argv, mode=mode)
    attested = attest_engine(combined, declared)
    ran.update(engine_fields((attested,), task_data, unit))
    failure = engine_failure(
        combined,
        attested,
        options=resolve_llm_api_options(task_data.get("accuracy") or {}, unit),
        asks=bool(declared.get("block_reuse")),
    )
    if failure:
        return _verdict(passed=False, detail=failure, **ran)
    unobserved = observables.unobserved(attested, subjects)
    if unobserved:
        return _verdict(passed=False, detail=observables.unobserved_detail(unobserved), **ran)

    records = {int(r.get("index", -1)): r for r in parse_driver_records(combined)}
    texts = [
        str(records[n].get("text", "")) if n in records else None for n in range(len(requests))
    ]
    absent = [requests[n]["id"] for n, text in enumerate(texts) if text is None]
    if absent:
        return _verdict(
            passed=False,
            detail=f"the engine exited cleanly but returned no record for {len(absent)} of "
            f"{len(requests)} prompt(s) ({', '.join(absent)})",
            **ran,
        )
    # The reference's prompts are judged by their tokens, which may decode to whitespace.
    compared = [
        records.get(n)
        for n in range(declared_requests, declared_requests + len(greedy.prompts if greedy else ()))
    ]
    texts = texts[:declared_requests]
    empty = [requests[n]["id"] for n, text in enumerate(texts) if not (text or "").strip()]
    if empty:
        return _verdict(
            passed=False,
            detail=f"the engine exited cleanly but generated nothing for {', '.join(empty)}",
            **ran,
        )
    agreed = ""
    if greedy is not None:
        failures, ties = judge_greedy(greedy, compared)
        if failures:
            return _verdict(
                passed=False,
                detail=f"the engine's greedy tokens are not the HF reference's "
                f"({greedy.source}): {'; '.join(failures)}",
                **ran,
            )
        agreed = (
            f"; the first {greedy.tokens} greedy tokens of {len(greedy.prompts)} prompt(s) of "
            f"{greedy.source} are the HF reference's"
            + (f", up to {len(ties)} near-tie(s) ({'; '.join(ties)})" if ties else "")
        )
    chunked = ""
    if chunks:
        shortfall = observables.chunk_shortfall(
            [records.get(n) for n in range(len(requests))], limits.max_num_tokens
        )
        if shortfall:
            return _verdict(passed=False, detail=shortfall, **ran)
        chunked = f"; a prompt past max_num_tokens {limits.max_num_tokens} was served in chunks"
    missing = missing_keywords(spec, texts)
    if missing:
        return _verdict(
            passed=False,
            detail=f"the generated texts are missing {len(missing)} expected keyword(s): "
            f"{missing}. Keywords are matched against generated text only",
            **ran,
        )
    images = sum(len(r.get("images") or ()) for r in requests)
    checked = (
        f"{len(spec.keywords) + sum(1 for i in spec.items if i.keywords or i.keywords_any)} "
        f"keyword check(s) in the generated texts"
        if spec.keywords or any(i.keywords or i.keywords_any for i in spec.items)
        else "non-empty generation"
    )
    sent = f", {images} image(s) sent" if images else ""
    return _verdict(
        passed=True,
        detail=f"engine cold-started and produced {checked} for {len(texts)} prompt(s)"
        f"{sent}; workload: {spec.source}{agreed}{chunked}",
        **ran,
    )


def missing_keywords(spec: BootSpec, texts: Sequence[str | None]) -> list[str]:
    """The keywords ``spec`` expects that ``texts``, its prompts' generated texts in order, lack.

    The run-wide keywords are matched against every text together; a prompt
    with keywords of its own against its own text. Matched against generated
    text only, case-insensitively. A text that is ``None`` was not asked for,
    and its prompt's own keywords are not looked for.
    """
    generated = "\n".join(text or "" for text in texts).lower()
    missing = [k for k in spec.keywords if k.lower() not in generated]
    item_texts = texts[len(spec.prompts) :]
    for item, text in zip(spec.items, item_texts):
        if text is None:
            continue
        for absent_keyword in item.missing(text):
            label = "the image prompt" if item.images else "a prompt"
            missing.append(f"{absent_keyword} (in {label} {item.prompt[:40]!r})")
    return missing


def resolve_floor(
    accuracy: Mapping[str, Any],
    unit: str | None,
    benchmark: str,
    reference: float,
    output: str = "",
) -> tuple[stats.Floor | None, str]:
    """The floor ``reference`` gets under ``accuracy.floor``, and where its count came from."""
    policy = stats.resolve_floor_policy(accuracy)
    samples, source = resolve_samples(accuracy, unit, benchmark, output)
    return stats.floor_for(policy, reference, samples), source


def run_accuracy_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    reference: float,
    runner: ShellRunner = _default_shell_runner,
    timeout: int = ACCURACY_TIMEOUT_S,
    unit: str | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
) -> GateResult:
    """Run ``trtllm-eval`` and compare against the anchor. One-sided.

    The floor is ``reference - k * SE`` by default (:func:`resolve_floor`), a
    fixed ``reference - tolerance`` under ``accuracy.floor.rule: tolerance``.
    The engine is held to the unit's options file (:func:`options_declaration`).
    """
    accuracy = dict(task_data.get("accuracy") or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    tolerance = float(accuracy.get("tolerance", 0.0))
    env = _with_mode(environment, MODE_REQUIRE)
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    planned, _ = resolve_floor(accuracy, unit, benchmark, reference)

    def floor_fields(floor: stats.Floor | None) -> dict[str, Any]:
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
    if state.blocked:
        return _verdict(
            passed=False, detail=state.block_detail(), **common, **floor_fields(planned)
        )

    parts = eval_command(task_data, env, benchmark, unit=unit, member=member, workdir=workdir)
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
            passed=False, detail=f"timed out after {timeout}s", **ran, **floor_fields(planned)
        )
    if not result.ok:
        return _verdict(
            passed=False,
            detail=f"trtllm-eval exited {result.returncode}",
            **ran,
            **floor_fields(planned),
        )
    options = resolve_llm_api_options(accuracy, unit)
    loaded = load_llm_api_options(options)
    asks = llm_api_block_reuse(loaded)
    attested = attest_engine(
        combined, options_declaration(loaded, block_reuse=asks), absent=EVAL_NOT_ATTESTED
    )
    ran["engine"] = engine_fields((attested,), task_data, unit)["engine"]
    failure = engine_failure(combined, attested, options=options, asks=asks)
    if failure:
        return _verdict(passed=False, detail=failure, **ran, **floor_fields(planned))

    measured = parse_eval_score(combined)
    if measured is None:
        return _verdict(
            passed=False,
            detail="trtllm-eval exited 0 but printed no score this module recognizes; "
            "a gate with no number is not a pass",
            **ran,
            **floor_fields(planned),
        )

    floor, source = resolve_floor(accuracy, unit, benchmark, reference, combined)
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
            f"({floor.describe()}; count from {source})"
        ),
        measured=measured,
        **ran,
        **floor_fields(floor),
    )


def check_systematic_shift(
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> stats.ShiftFinding | None:
    """The cross-member check over one unit's accuracy verdicts (:func:`.stats.systematic_shift`).

    Verdicts without a measurement or an SE are left out.
    """
    policy = stats.resolve_floor_policy(accuracy)
    readings = {
        str(r.member): (float(r.measured), float(r.reference), float(r.se))
        for r in results
        if r.measured is not None and r.reference is not None and r.se is not None
    }
    return stats.systematic_shift(
        unit,
        readings,
        min_members=policy.shift_min_members,
        acknowledged=policy.acknowledged,
    )


#: The key a systematic-shift finding is recorded under, per unit.
SHIFT_FINDING = "systematic_shift"


def records_for(workspace: str | Path) -> GateRecords:
    """The results file of a run's workspace."""
    return GateRecords(Path(workspace) / RESULTS_FILENAME)


def note_systematic_shift(
    records: GateRecords,
    results: Sequence[GateResult],
    accuracy: Mapping[str, Any],
    unit: str | None,
) -> stats.ShiftFinding | None:
    """Run :func:`check_systematic_shift` and record its outcome either way.

    A unit with no shift records a resolved entry, so a finding raised by an
    earlier turn of the same unit stops blocking once a later turn is clean.
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
    return open_findings(records, stats.resolve_floor_policy(accuracy).acknowledged)


#: Prompts an acceptance workload must hold before its rates are compared: on
#: a handful, a change that only moves the generated text reads as a
#: regression. ``accuracy.acceptance.min_prompts`` may lower it.
DEFAULT_ACCEPTANCE_MIN_PROMPTS = 32
#: Prompts taken from the accuracy dataset when the task declares none.
DEFAULT_ACCEPTANCE_NUM_PROMPTS = 32
#: The minimum drop, in accepted tokens per iteration, a paired comparison
#: must find before it calls a significant change a regression.
DEFAULT_ACCEPTANCE_MIN_EFFECT = 0.1
#: The share of stock's acceptance a target must reach unless the paired test
#: finds its drop insignificant; a floor derived from a measured reference
#: keeps the same share of it.
MIN_ACCEPTANCE_RATIO = 0.95


@dataclass(frozen=True)
class AcceptanceFloor:
    """An absolute acceptance floor, and the measurement it came from, if any."""

    value: float
    #: Where the reference drafter this floor derives from was measured;
    #: ``None`` when the task declared a number with no measurement behind it.
    source: str | None = None
    #: The measured reference value the floor was derived from, when it was.
    measured: float | None = None

    def warning(self) -> str | None:
        """What a verdict says when the floor is not tied to a measurement."""
        if self.source or self.measured is not None:
            return None
        return (
            f"the acceptance floor {self.value:.2f} is not tied to a measured reference: "
            f"declare `floor_source` naming the reference-drafter measurement it was "
            f"derived from, or `measured_reference` with the value itself. A floor chosen "
            f"below every real drafter's acceptance passes a wrong drafter too"
        )


def resolve_acceptance_floor_spec(
    accuracy: Mapping[str, Any], unit: str | None
) -> AcceptanceFloor | None:
    """The absolute floor ``unit`` declares in ``accuracy.acceptance``, if any.

    ``min_absolute`` is the floor itself. ``measured_reference`` is what a
    reference drafter measured on this workload; with no ``min_absolute``
    the floor is that times :data:`MIN_ACCEPTANCE_RATIO`. ``floor_source``
    names where the measurement was taken.
    """
    block = accuracy.get("acceptance")
    if not isinstance(block, dict):
        return None
    chosen = block.get(unit)
    if chosen is None:
        chosen = block.get("default")
    if isinstance(chosen, dict):
        source = chosen.get("floor_source")
        measured = chosen.get("measured_reference")
        measured = float(measured) if isinstance(measured, (int, float)) else None
        if chosen.get("min_absolute") is not None:
            try:
                return AcceptanceFloor(float(chosen["min_absolute"]), source, measured)
            except (TypeError, ValueError):
                return None
        if measured is not None:
            return AcceptanceFloor(measured * MIN_ACCEPTANCE_RATIO, source, measured)
        return None
    try:
        return AcceptanceFloor(float(chosen))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AcceptanceWorkload:
    """What the acceptance gate sends, and how much of it it needs back."""

    #: Declared prompts; empty when ``dataset`` supplies them.
    prompts: tuple[str, ...]
    max_tokens: int
    min_draft_iterations: int
    #: Where the prompts came from, named in a verdict that blames them.
    source: str
    #: The engine driver's ``dataset`` block, when the prompts come from one.
    dataset: Mapping[str, Any] | None = None
    min_prompts: int = DEFAULT_ACCEPTANCE_MIN_PROMPTS

    @property
    def fingerprint(self) -> str:
        """Names this workload, so two runs of it can be paired."""
        payload = json.dumps(
            {
                "prompts": list(self.prompts),
                "dataset": dict(self.dataset or {}),
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def requests(self) -> list[dict[str, Any]]:
        """The declared prompts as engine-driver requests."""
        return [
            {"id": f"p{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]


def acceptance_dataset(
    accuracy: Mapping[str, Any], unit: str | None, setting: Callable[[str], Any]
) -> tuple[dict[str, Any] | None, str]:
    """The dataset an undeclared acceptance workload is drawn from, and its name.

    ``accuracy.acceptance.dataset`` when declared; otherwise the accuracy
    benchmark's own dataset -- ``accuracy.dataset_path`` when the run keeps
    it locally, else the benchmark's Hugging Face id, read from the local
    cache only (:data:`.harness.workload_io.KNOWN_DATASETS`). A unit whose
    benchmark has no text dataset uses the run-wide benchmark's.
    """
    known: dict[str, Any] = {}
    for scope in (unit, "default"):
        benchmark = resolve_per_unit(accuracy, "benchmark", scope)
        if isinstance(benchmark, str) and benchmark in workload_io.KNOWN_DATASETS:
            known = dict(workload_io.KNOWN_DATASETS[benchmark])
            break
    declared = setting("dataset")
    if declared:
        source, where = str(declared), "accuracy.acceptance.dataset"
    elif accuracy.get("dataset_path") and known:
        source, where = str(accuracy["dataset_path"]), "accuracy.dataset_path"
    elif known:
        source, where = str(known["source"]), "the accuracy benchmark's dataset"
    else:
        return None, ""
    dataset = {
        "source": source,
        "num": int(setting("num_prompts") or DEFAULT_ACCEPTANCE_NUM_PROMPTS),
        "text_field": setting("text_field") or known.get("text_field"),
        "split": setting("split") or known.get("split", "test"),
        "config": setting("dataset_config") or (None if declared else known.get("config")),
        "template": setting("prompt_template") or "{text}",
    }
    return dataset, f"{dataset['num']} prompts from {where} ({source})"


def resolve_acceptance_workload(
    accuracy: Mapping[str, Any], unit: str | None
) -> AcceptanceWorkload:
    """The acceptance workload ``unit`` declares, or the accuracy dataset's prompts.

    Read from ``accuracy.acceptance``, most specific level first: the unit's
    own entry, then its ``default`` entry, then the block's top level.
    ``prompts`` and ``prompts_file`` are one setting, so the level that states
    either wins for both. With neither, the workload is the first
    ``num_prompts`` questions of the accuracy dataset (:func:`acceptance_dataset`).

    Raises :class:`~.task_schema.TaskSchemaError` when a declared prompts
    file cannot be read, or when nothing declares a workload and the
    benchmark has no dataset to draw one from.
    """
    block = accuracy.get("acceptance")
    block = block if isinstance(block, dict) else {}
    levels = [
        (f"accuracy.acceptance.{name}", block.get(name))
        for name in (unit, "default")
        if name is not None
    ]
    levels.append(("accuracy.acceptance", block))
    levels = [(where, entry) for where, entry in levels if isinstance(entry, dict)]

    def setting(key: str) -> Any:
        for _, entry in levels:
            if entry.get(key) is not None:
                return entry[key]
        return None

    prompts: tuple[str, ...] = ()
    source = ""
    dataset: dict[str, Any] | None = None
    for where, entry in levels:
        if entry.get("prompts") is not None:
            prompts = tuple(str(p) for p in entry["prompts"])
            source = f"{where}.prompts"
            break
        if entry.get("prompts_file") is not None:
            prompts = load_prompts_file(entry["prompts_file"])
            source = f"{where}.prompts_file ({entry['prompts_file']})"
            break
    if not prompts:
        dataset, source = acceptance_dataset(accuracy, unit, setting)
        if dataset is None:
            raise TaskSchemaError(
                "no acceptance workload: declare `accuracy.acceptance.prompts`, "
                "`prompts_file` or `dataset` (a local path or a cached dataset id); the "
                "benchmark has no dataset this gate knows how to draw prompts from"
            )
    declared_tokens = setting("max_tokens")
    declared_minimum = setting("min_draft_iterations")
    min_prompts = setting("min_prompts")
    return AcceptanceWorkload(
        prompts=prompts,
        max_tokens=int(
            DEFAULT_ACCEPTANCE_MAX_TOKENS if declared_tokens is None else declared_tokens
        ),
        min_draft_iterations=int(
            DEFAULT_MIN_DRAFT_ITERATIONS if declared_minimum is None else declared_minimum
        ),
        source=source,
        dataset=dataset,
        min_prompts=int(DEFAULT_ACCEPTANCE_MIN_PROMPTS if min_prompts is None else min_prompts),
    )


@dataclass(frozen=True)
class RegressionPolicy:
    """How a paired acceptance comparison is judged (``accuracy.acceptance.regression``)."""

    test: str = stats.TEST_T
    alpha: float = stats.DEFAULT_ALPHA
    min_effect: float = DEFAULT_ACCEPTANCE_MIN_EFFECT

    def compare(
        self, baseline: Sequence[float], candidate: Sequence[float]
    ) -> stats.PairedComparison:
        """:func:`.stats.paired_test` under this policy."""
        return stats.paired_test(
            baseline, candidate, test=self.test, alpha=self.alpha, min_effect=self.min_effect
        )


def resolve_regression_policy(accuracy: Mapping[str, Any]) -> RegressionPolicy:
    """Read ``accuracy.acceptance.regression``; every key optional."""
    block = accuracy.get("acceptance")
    block = (block if isinstance(block, dict) else {}).get("regression")
    block = block if isinstance(block, dict) else {}
    return RegressionPolicy(
        test=str(block.get("test") or stats.TEST_T),
        alpha=float(block.get("alpha", stats.DEFAULT_ALPHA)),
        min_effect=float(block.get("min_effect", DEFAULT_ACCEPTANCE_MIN_EFFECT)),
    )


def tree_commit(repo_root: str | Path | None) -> str | None:
    """The full commit ``HEAD`` names in ``repo_root``, or ``None`` if git cannot say."""
    if not repo_root:
        return None
    answer = git_output(repo_root, "rev-parse", "HEAD")
    return answer.strip() if answer and answer.strip() else None


def acceptance_key(
    workload: AcceptanceWorkload,
    *,
    member: str | None,
    speculation: Sequence[str],
) -> dict[str, Any]:
    """What makes two acceptance verdicts comparable across trees.

    The member, the workload and the speculator flags -- not the unit: a later
    unit that keeps speculation on is compared against the earlier unit's verdict.
    """
    return {
        "gate": "acceptance",
        "member": str(member),
        "workload": workload.fingerprint,
        "speculation": " ".join(speculation),
    }


#: Speculators whose draft depth is a setting of the run, so one depth shallower is a
#: configuration the same checkpoint serves (``--spec_decode_algo``, upper-cased).
VARIABLE_DEPTH_SPECULATORS: frozenset[str] = frozenset({"MTP", "EAGLE3"})


def depth_rise(
    deeper: AcceptanceReading, shallower: AcceptanceReading, policy: RegressionPolicy
) -> tuple[stats.PairedComparison | None, str]:
    """Whether ``deeper`` accepts significantly more than ``shallower``, paired per prompt.

    One-sided at ``policy.alpha`` with no minimum effect: a deep position is
    accepted rarely, and what is asked is that it is at all. ``None`` when
    fewer than two prompts were measured by both.
    """
    deep, shallow = deeper.rates(), shallower.rates()
    shared = [p for p in deep if p in shallow]
    if len(shared) < 2:
        return None, f"{len(shared)} prompt(s) measured by both, too few to compare"
    rise = stats.paired_test(
        [deep[p] for p in shared],
        [shallow[p] for p in shared],
        test=policy.test,
        alpha=policy.alpha,
    )
    se = "n/a" if math.isnan(rise.se_diff) else f"{rise.se_diff:.3f}"
    return rise, (
        f"mean per-prompt rise {-rise.mean_diff:+.3f} (SE {se}, one-sided p {rise.p_value:.3g} "
        f"against alpha {rise.alpha}, {rise.down} prompt(s) up, {rise.up} down over {rise.n})"
    )


def run_acceptance_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    features: Any | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
    records: GateRecords | None = None,
) -> GateResult:
    """Measure per-prompt acceptance, and judge it against stock, a floor and the last tree.

    A workload under ``min_prompts`` prompts or ``min_draft_iterations``
    generation iterations fails as too small to measure, not as a low number.
    Then, each where it can be made: against stock (``TRTLLM_MODELING_V2=off``),
    failing when the ratio is under :data:`MIN_ACCEPTANCE_RATIO` and the paired
    test finds the drop significant; where the stock path cannot load the
    checkpoint (recorded in ``records``), against one draft depth shallower
    (:func:`depth_rise`) and a declared floor; against 1.0; and against the
    previous passing verdict in ``records`` (:func:`acceptance_key`).
    """
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = partial(_verdict_factory(state), name="acceptance", member=member)
    if state.blocked:
        return _verdict(passed=False, detail=state.block_detail())
    accuracy = task_data.get("accuracy") or {}
    try:
        workload = resolve_acceptance_workload(accuracy, unit)
    except TaskSchemaError as exc:
        return _verdict(
            passed=False,
            detail=f"the declared acceptance workload cannot be read: {exc}",
        )
    features = task_data.get("features") if features is None else features
    draft_len = resolve_max_draft_len(accuracy, unit)
    speculation = speculation_flags(features, max_draft_len=draft_len)
    argv = quickstart_words(
        task_data,
        environment,
        unit=unit,
        features=features,
        max_tokens=workload.max_tokens,
        member=member,
    )
    directory = gate_workdir(workdir, "acceptance", task_data, environment=environment)
    spec_path = write_driver_spec(
        directory / f"acceptance.{unit or 'unit'}.{member or 'member'}.json",
        task_data=task_data,
        argv=argv,
        requests=workload.requests(),
        dataset=workload.dataset,
        max_tokens=workload.max_tokens,
        reuse_probe=True,
    )
    policy = resolve_regression_policy(accuracy)
    floor_spec = resolve_acceptance_floor_spec(accuracy, unit)
    # One draft depth shallower is a leg of its own only where the depth is a setting:
    # a DSpark drafter's block is the checkpoint's (`resolve_max_draft_len`).
    algorithm = str(_flag_value(argv, "--spec_decode_algo") or "")
    depth_curve = draft_len > 1 and algorithm.upper() in VARIABLE_DEPTH_SPECULATORS

    options = resolve_llm_api_options(accuracy, unit)
    started = time.monotonic()
    readings: dict[str, AcceptanceReading | None] = {}
    commands: dict[str, str] = {}
    attested: list[EngineAttestation] = []
    stock_unavailable = False
    stock_key = {
        "checkpoint": str(task_data.get("checkpoint_path") or ""),
        "arch": str(task_data.get("arch") or ""),
    }
    known = (
        records.latest(KIND_STOCK_UNAVAILABLE, stock_key, passed_only=False)
        if records is not None
        else None
    )
    skipped_stock = ""
    for mode in (MODE_REQUIRE, MODE_OFF):
        if mode == MODE_OFF and known is not None and (floor_spec is not None or depth_curve):
            # Asked once per checkpoint and architecture: the stock path does not load it.
            readings[mode] = None
            stock_unavailable = True
            skipped_stock = (
                f" (not run again: on tree {known.tree} the stock leg could not load this "
                f"checkpoint)"
            )
            break
        command = engine_driver_command(
            with_kv_cache_config(_with_mode(environment, mode), task_data, unit), spec_path
        )
        commands[mode] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        if result.infrastructure:
            return _verdict(
                passed=False,
                detail=(
                    f"the {MODELING_V2_ENV}={mode} leg did not run to a verdict: "
                    f"{result.infrastructure}"
                ),
                infrastructure=result.infrastructure,
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        if not result.ok:
            # A genuinely new architecture has no built-in modeling: the stock
            # leg is impossible rather than broken.
            unavailable = mode == MODE_OFF and reports_unsupported_architecture(
                f"exited {result.returncode}", "\n".join(combined.splitlines()[-200:])
            )
            if unavailable and records is not None:
                records.append(
                    GateRecord(
                        kind=KIND_STOCK_UNAVAILABLE,
                        key=stock_key,
                        tree=state.description,
                        passed=False,
                        values={"tail": "\n".join(combined.splitlines()[-5:])[:500]},
                    )
                )
            if unavailable and (floor_spec is not None or depth_curve):
                readings[mode] = None
                stock_unavailable = True
                break
            return _verdict(
                passed=False,
                detail=(
                    f"the {MODELING_V2_ENV}={mode} leg "
                    + ("timed out" if result.timed_out else f"exited {result.returncode}")
                    + (
                        " -- this checkpoint has no built-in implementation, so the "
                        "stock leg cannot run, and there is no shallower depth to compare "
                        "with (draft depth 1, or a drafter whose depth is the checkpoint's). "
                        "Declare `accuracy.acceptance.<unit>."
                        "min_absolute` (or `measured_reference`) to gate on an absolute "
                        "floor instead."
                        if unavailable
                        else ""
                    )
                ),
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        declared = engine_declaration(task_data, unit, argv, mode=mode)
        attested.append(attest_engine(combined, declared, leg=mode))
        failure = engine_failure(
            combined,
            attested[-1],
            options=options,
            asks=bool(declared.get("block_reuse")),
            prefix=f"the {MODELING_V2_ENV}={mode} leg: ",
        )
        if failure:
            return _verdict(
                passed=False,
                detail=failure,
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
                engine=engine_fields(attested, task_data, unit)["engine"],
            )
        readings[mode] = parse_acceptance(combined)
    shallower = draft_len - 1 if stock_unavailable and depth_curve else None
    if shallower is not None:
        leg = f"depth {shallower}"
        shallow_argv = [*argv, f"--spec_decode_max_draft_len={shallower}"]
        shallow_spec = write_driver_spec(
            directory / f"acceptance.{unit or 'unit'}.{member or 'member'}.depth{shallower}.json",
            task_data=task_data,
            argv=shallow_argv,
            requests=workload.requests(),
            dataset=workload.dataset,
            max_tokens=workload.max_tokens,
        )
        command = engine_driver_command(
            with_kv_cache_config(_with_mode(environment, MODE_REQUIRE), task_data, unit),
            shallow_spec,
        )
        commands[leg] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        ran = dict(
            command=command,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(combined.splitlines()[-40:]),
        )
        if result.infrastructure:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg did not run to a verdict: {result.infrastructure}",
                infrastructure=result.infrastructure,
                **ran,
            )
        if not result.ok:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg (the target one draft depth shallower, which this "
                f"depth's acceptance is compared with where no stock speculator exists) "
                + ("timed out" if result.timed_out else f"exited {result.returncode}"),
                **ran,
            )
        declared = engine_declaration(task_data, unit, shallow_argv, mode=MODE_REQUIRE)
        attested.append(attest_engine(combined, declared, leg=leg))
        if attested[-1].mismatches:
            return _verdict(
                passed=False,
                detail=attested[-1].difference(),
                engine=engine_fields(attested, task_data, unit)["engine"],
                **ran,
            )
        readings[leg] = parse_acceptance(combined)
    elapsed = time.monotonic() - started
    engine = engine_fields(attested, task_data, unit)

    target_reading = readings.get(MODE_REQUIRE)
    stock_reading = readings.get(MODE_OFF)
    target = target_reading.value if target_reading is not None else None
    stock = stock_reading.value if stock_reading is not None else None
    breakdown = target_reading.breakdown() if target_reading is not None else ""
    base = dict(duration_s=elapsed, engine=engine["engine"])

    def too_small(mode: str) -> GateResult | None:
        """The verdict for a leg whose workload cannot carry a comparison."""
        reading = readings.get(mode)
        if reading is None or not reading.requests:
            return None
        counted = reading.draft_iterations or 0
        if len(reading.requests) < workload.min_prompts:
            why = (
                f"workload too small to compare acceptance: the {MODELING_V2_ENV}={mode} leg "
                f"measured {len(reading.requests)} prompt(s), fewer than the "
                f"{workload.min_prompts} a paired comparison needs; on a handful of prompts "
                f"a change that only moves the generated text reads as a regression"
            )
        elif counted < workload.min_draft_iterations:
            why = (
                f"workload too short to measure acceptance: the {MODELING_V2_ENV}={mode} "
                f"leg ran {counted} generation iteration(s) over {len(reading.requests)} "
                f"prompt(s), fewer than the {workload.min_draft_iterations} this gate "
                f"needs, so its rate ({reading.value:.2f}) says nothing about the draft "
                f"path -- the answers end before drafts are verified"
            )
        else:
            return None
        return _verdict(
            passed=False,
            detail=(
                f"{why}. Workload: {workload.source}, max_tokens {workload.max_tokens}. "
                f"This is a property of the workload, not of the target."
            ),
            command=commands[mode],
            breakdown=reading.breakdown(),
            **base,
        )

    if target is None:
        return _verdict(
            passed=False,
            detail=f"could not read acceptance from the require leg (off={stock})",
            command=commands[MODE_REQUIRE],
            **base,
        )
    for mode in (MODE_REQUIRE, MODE_OFF):
        short = too_small(mode)
        if short is not None:
            return short

    warnings: list[str] = []
    notes: list[str] = []
    if target <= 1.0:
        return _verdict(
            passed=False,
            detail=f"acceptance_length {target:.4f} at or below 1.0 means every draft was "
            f"rejected — the draft path is computing the wrong thing",
            command=commands[MODE_REQUIRE],
            measured=target,
            reference=stock if stock is not None else (floor_spec.value if floor_spec else None),
            breakdown=breakdown,
            **base,
        )

    reference: float | None
    shallow_reading = readings.get(f"depth {shallower}") if shallower is not None else None
    if shallower is not None:
        if shallow_reading is None or not shallow_reading.requests:
            return _verdict(
                passed=False,
                detail=f"could not read per-prompt acceptance from the depth {shallower} leg",
                command=commands[f"depth {shallower}"],
                **base,
            )
        reference = shallow_reading.value
        rise, rise_note = depth_rise(target_reading, shallow_reading, policy)
        passed = rise is not None and rise.significant
        notes.append(
            f"acceptance_length {target:.4f} at draft depth {draft_len} vs "
            f"{shallow_reading.value:.4f} at depth {shallower}, the same target on the same "
            f"prompts ({rise_note}): "
            + (
                f"depth {draft_len}'s last draft position is accepted"
                if passed
                else f"depth {draft_len} accepts no more than depth {shallower}: its last draft "
                f"position is never accepted, or too rarely for this workload to show (more "
                f"prompts: accuracy.acceptance.num_prompts)"
            )
            + f". The {MODELING_V2_ENV}=off leg could not run: this checkpoint has no built-in "
            f"implementation{skipped_stock}"
        )
        if floor_spec is not None:
            floor_passed = target >= floor_spec.value
            passed = passed and floor_passed
            notes.append(
                f"declared floor {floor_spec.value:.4f}: "
                + ("met" if floor_passed else f"NOT met by {target:.4f}")
            )
            if floor_spec.warning():
                warnings.append(floor_spec.warning())
    elif stock_unavailable:
        assert floor_spec is not None
        reference = floor_spec.value
        passed = target >= floor_spec.value
        if floor_spec.warning():
            warnings.append(floor_spec.warning())
        derived = (
            f" (= measured reference {floor_spec.measured:.2f} x {MIN_ACCEPTANCE_RATIO:.2f})"
            if floor_spec.measured is not None and floor_spec.source is None
            else ""
        )
        notes.append(
            f"acceptance_length {target:.4f} vs declared floor {floor_spec.value:.4f}{derived}"
            + (f" (from {floor_spec.source})" if floor_spec.source else "")
            + f". The {MODELING_V2_ENV}=off leg could not run: this checkpoint has no "
            f"built-in implementation{skipped_stock}. THIS IS THE WEAKER CLAIM -- it shows the "
            f"draft path accepts at a real rate, not that it matches stock"
        )
    else:
        if stock is None:
            return _verdict(
                passed=False,
                detail=f"could not read acceptance from both legs (require={target}, off={stock})",
                command=commands[MODE_REQUIRE],
                **base,
            )
        stock_breakdown = stock_reading.breakdown() if stock_reading is not None else ""
        if breakdown and stock_breakdown:
            breakdown = f"require: {breakdown}; off: {stock_breakdown}"
        reference = stock
        ratio = target / stock if stock else 0.0
        paired = None
        if target_reading.requests and stock_reading.requests:
            own, theirs = target_reading.rates(), stock_reading.rates()
            shared = [p for p in own if p in theirs]
            if len(shared) >= 2:
                paired = policy.compare([theirs[p] for p in shared], [own[p] for p in shared])
        passed = ratio >= MIN_ACCEPTANCE_RATIO or (paired is not None and not paired.significant)
        notes.append(
            f"acceptance_length {target:.4f} vs stock {stock:.4f} (ratio {ratio:.3f}, floor "
            f"{MIN_ACCEPTANCE_RATIO:.2f})"
            + (f"; {paired.describe()}" if paired is not None else "")
            + (
                ". Clearly above 1.0 but below the reference means subtly wrong, not merely slower"
                if not passed
                else ""
            )
        )

    key = acceptance_key(workload, member=member, speculation=speculation)
    if records is not None and target_reading.requests:
        baseline = records.latest(KIND_ACCEPTANCE, key, exclude_tree=state.description)
        if baseline is None:
            notes.append(
                "no earlier verdict for this member and workload; this one is the baseline"
            )
        else:
            before = {
                str(k): float(v) for k, v in (baseline.values.get("per_prompt") or {}).items()
            }
            now = target_reading.rates()
            shared = [p for p in now if p in before]
            if len(shared) < 2:
                warnings.append(
                    f"the earlier verdict on tree {baseline.tree} shares {len(shared)} prompt(s) "
                    f"with this one, so the two were not compared"
                )
            else:
                compared = policy.compare([before[p] for p in shared], [now[p] for p in shared])
                regressed = compared.regressed
                notes.append(
                    f"against tree {baseline.tree}: "
                    + ("REGRESSED, " if regressed else "")
                    + compared.describe()
                )
                passed = passed and not regressed
        records.append(
            GateRecord(
                kind=KIND_ACCEPTANCE,
                key=key,
                tree=state.description,
                commit=tree_commit(task_data.get("trtllm_repo_path")),
                passed=passed,
                values={
                    "unit": unit,
                    "weighted": target,
                    "per_prompt": target_reading.rates(),
                    "prompts": len(target_reading.requests),
                },
            )
        )

    return _verdict(
        passed=passed,
        detail="; ".join(notes),
        command=commands[MODE_REQUIRE],
        measured=target,
        reference=reference,
        breakdown=breakdown,
        warnings=(*warnings, *engine["warnings"]),
        explained=True,
        **base,
    )


@dataclass
class GateLog:
    """Every gate a run has fired, in order. The journal's raw material."""

    results: list[GateResult] = field(default_factory=list)

    def record(self, result: GateResult) -> GateResult:
        self.results.append(result)
        return result

    def render(self) -> str:
        return "\n".join(result.summary() for result in self.results)


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


__all__ = [
    "LAUNCH_KNOBS",
    "LaunchKnob",
    "REMOTE_CODE_FILES",
    "checkpoint_remote_code",
    "flag_settings",
    "launch_conflicts",
    "launch_flags",
    "launch_overlay",
    "merge_options",
    "option_settings",
    "UNSUPPORTED_ARCHITECTURE_MARKERS",
    "reports_unsupported_architecture",
    "ACCURACY_TIMEOUT_S",
    "ALLOW_DIRTY_ENV",
    "AUTOTUNER_CACHE_ENV",
    "AcceptanceReading",
    "AcceptanceWorkload",
    "BOOT_TIMEOUT_S",
    "AcceptanceFloor",
    "BATCH_SIZE_FLAG",
    "BootPrompt",
    "BootSpec",
    "DEFAULT_BOOT_IMAGE",
    "DEFAULT_BOOT_IMAGE_KEYWORDS",
    "DEFAULT_BOOT_IMAGE_PROMPT",
    "resolve_boot_spec",
    "DEFAULT_ACCEPTANCE_MIN_EFFECT",
    "DEFAULT_ACCEPTANCE_MIN_PROMPTS",
    "DEFAULT_ACCEPTANCE_NUM_PROMPTS",
    "ENGINE_DRIVER",
    "RegressionPolicy",
    "SHIFT_FINDING",
    "acceptance_dataset",
    "acceptance_key",
    "engine_driver_command",
    "gate_workdir",
    "parse_driver_records",
    "pin_tuning",
    "quickstart_script",
    "quickstart_words",
    "resolve_acceptance_floor_spec",
    "resolve_regression_policy",
    "tree_commit",
    "with_max_batch_size",
    "write_driver_spec",
    "blocking_findings",
    "check_systematic_shift",
    "note_systematic_shift",
    "records_for",
    "resolve_floor",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "GateLog",
    "GateResult",
    "MODELING_V2_ENV",
    "MODE_OFF",
    "MODE_REQUIRE",
    "STEP_CONTRACT_ENV",
    "TRITON_AUTOTUNE_RESULT_GLOB",
    "TRITON_CACHE_AUTOTUNING_ENV",
    "TRITON_CACHE_DIR_ENV",
    "RUN_TRITON_CACHE_DIRNAME",
    "default_triton_cache",
    "ShellRunner",
    "TreeState",
    "UNKNOWN_TREE_STATE",
    "describe_tree_state",
    "dirty_target_files",
    "inspect_tree",
    "load_reference_score",
    "measured_on",
    "order_members_by_cost",
    "parse_acceptance",
    "parse_eval_score",
    "resolve_acceptance_workload",
    "resolve_discriminators",
    "resolve_runtime",
    "resolve_llm_api_options",
    "CAPTURE_BATCHES_FLAG",
    "CAPTURE_PADDING_FLAG",
    "CUDA_GRAPH_FLAG",
    "LLM_API_CAPTURE_MAX_BATCH_SIZE",
    "LLM_API_DEFAULT_CAPTURE",
    "CaptureList",
    "LaunchCapture",
    "capture_list_from",
    "format_batches",
    "launch_captures",
    "llm_api_batch_sizes",
    "load_llm_api_options",
    "options_capture",
    "quickstart_capture",
    "resolve_capture_list",
    "runtime_flags",
    "DISABLE_REUSE_FLAG",
    "KV_CACHE_CONFIG_ENV",
    "QUICKSTART_KV_CACHE_KEYS",
    "REUSE_DISABLED_MARKER",
    "LaunchKvCache",
    "has_snapshot_policy",
    "launch_kv_caches",
    "quickstart_block_reuse",
    "quickstart_kv_cache_config",
    "resolve_kv_cache_config",
    "reuse_disabled_detail",
    "with_kv_cache_config",
    "write_atomically",
    "ATTESTED_FIELDS",
    "CACHED_TOKENS_SOURCE",
    "ENGINE_MARKER",
    "EVAL_NOT_ATTESTED",
    "EVAL_SHIM",
    "EngineAttestation",
    "LOG_SOURCE",
    "MemoryMargin",
    "PROBE_SOURCE",
    "RANKS_SOURCE",
    "RANK_ENGINE_MARKER",
    "RANK_FACTS_ENV",
    "RANK_HOOK_DIR",
    "REUSE_MARKER",
    "attest_engine",
    "declared_engine",
    "declared_snapshot_policy",
    "describe_engine",
    "engine_declaration",
    "engine_fields",
    "eval_harness_command",
    "llm_api_block_reuse",
    "parse_engine_records",
    "parse_kv_allocations",
    "parse_recurrent_slots",
    "options_declaration",
    "parse_rank_records",
    "parse_reuse_records",
    "resolve_memory_margin",
    "reuse_evidence",
    "reuse_shortfall",
    "resolve_reference",
    "run_acceptance_gate",
    "run_accuracy_gate",
    "run_boot_gate",
    "target_family_rel",
    "DEFAULT_GREEDY_PROMPTS",
    "DEFAULT_GREEDY_TOKENS",
    "LOGIT_MIN_MAX_DELTA",
    "LOGIT_THRESHOLD_FACTOR",
    "QUICKSTART_MAX_NUM_TOKENS",
    "EngineLimits",
    "GreedyPrompt",
    "GreedySetting",
    "GreedyWorkload",
    "greedy_workload",
    "judge_greedy",
    "missing_keywords",
    "depth_rise",
    "VARIABLE_DEPTH_SPECULATORS",
    "topology_flags",
    "MIN_ACCEPTANCE_RATIO",
    "engine_failure",
    "lost_session",
    "quickstart_flags",
]

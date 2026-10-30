# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gates — commands with exit codes, never agent judgements.

This module is the reason the workflow has no ``--min-score`` and no QA
agent scoring an artifact out of ten. A modeling_v2 target is trusted because
its gates passed, and a gate is a process that either returned zero and
printed a number above its anchor, or did not. Putting that behind an agent
turn adds latency and adds a way for the verdict to be rationalized.

Three gates, in ascending cost:

1. **boot** — minutes, binary. The engine cold-starts under
   ``TRTLLM_MODELING_V2=require``, the weight manifest covers what it
   claims, and a handful of greedy continuations come back. Where the run
   has an HF logit reference, the first greedy tokens of a few of its
   prompts must be the reference's, token for token (:class:`GreedySetting`).
   Catches catastrophes and the first wrong token, not drift in
   distribution. Runs after every assembler turn.
2. **accuracy** — the release criterion. ``trtllm-eval`` against the anchor
   in TensorRT-LLM's own accuracy references. One-sided:
   ``measured >= reference - k * SE``, the SE binomial over the benchmark's
   sample count (``accuracy.floor``; a fixed ``tolerance`` is available as
   an explicit rule). Runs at a unit boundary.
3. **acceptance** — required only when a feature is *distribution-preserving
   by construction*, speculative decoding above all. Rejection sampling
   holds the emitted distribution to the target model's, so a miscomputed
   draft path produces correct text **more slowly**: boot passes, accuracy
   passes, and only speed moves. The detector is ``acceptance_length``
   against the same checkpoint under stock in-tree modeling at the same
   workload and the same ``speculative_config`` — i.e. the same harness
   with ``TRTLLM_MODELING_V2=off`` instead of ``=require``, which makes it
   one variable rather than two harnesses.

``TRTLLM_MODELING_V2=require`` is not optional for any of them. Under
``auto`` a configuration that misses a target's routing criteria silently
gets the built-in implementation, and a number measured that way reads as
modeling_v2's — the single most expensive mistake available here.
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
    BOOT_RENDERS,
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

# Minutes-scale by design. A boot gate that needs longer than this is not a
# boot gate any more — it is an accuracy run with no anchor.
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
#: a model, and failing one because a VCS query failed would attribute a
#: tooling problem to a target.
UNKNOWN_TREE_STATE = "unknown"

#: Set to ``1`` to let a gate measure a dirty target path anyway. For a human
#: driving a gate by hand against uncommitted edits, which is a real and
#: reasonable thing to do — but the resulting verdict carries
#: ``dirty_override`` and says so in its summary, because a number measured
#: that way is not attributable to any commit.
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
#
# This is the pattern the acceptance gate depends on: nothing in this tree
# emits either of the two patterns above, and `trtllm-eval` reports no
# speculative statistic at all, so an acceptance gate built on the eval
# command would read `None` from both legs and fail with "could not read
# acceptance_length", indistinguishable from a broken draft path.
# The quantity is captured from the parenthetical rather than from the
# printed ratio so that several requests combine correctly --- see
# :func:`parse_acceptance_length`.
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

#: The one process-wide local executor. Stateless, so sharing it is free and
#: constructing one per gate would say something about lifetime that is not
#: true.
_LOCAL_EXECUTOR = LocalCommandExecutor()


def describe_tree_state(repo_root: str | Path) -> str:
    """The commit a gate is about to measure, as one token.

    ``git describe --always``: where the tree carries tags it names the state
    the way a human reads a release (``v0.20.0rc0-8958-g66ebfec7e2``) and
    where it does not it falls back to the abbreviated commit, so one field
    covers both and a reader never has to know which they are looking at.

    What it buys is attribution. Gates measure whatever is on disk under
    ``trtllm_repo_path`` when the engine imports it; without a record of what
    that was, an accuracy number can be attributed to a commit it was never
    measured on. The token is not a pin and does not pretend to be one; it is
    what lets a reader tell whether two numbers came from the same code.

    **Deliberately not ``--dirty``**, which is the obvious spelling and the
    wrong one here, for two independent reasons:

    * *Cost.* ``--dirty`` runs a whole-worktree ``diff-index`` refresh, which
      on a large checkout over a networked filesystem takes a minute or more
      where ``describe --always`` takes a fraction of a second. A gate must
      not sit a minute behind a VCS nicety, and with a timeout in front of it
      the field would simply always read ``unknown``.
    * *Information.* A workflow run's repo is dirty essentially always —
      onboarding records, reference YAMLs, workflow fixes — so a repo-wide
      dirty marker is set on every gate of every run and distinguishes
      nothing. The marker worth carrying is scoped to the target, and
      :func:`inspect_tree` appends that one instead, from
      :func:`dirty_target_files`.
    """
    described = git_output(repo_root, "describe", "--always")
    if described is None:
        return UNKNOWN_TREE_STATE
    return described.strip() or UNKNOWN_TREE_STATE


def target_family_rel(task_data: Mapping[str, Any]) -> str | None:
    """The repo-relative directory holding every target for this family.

    Derived the way :func:`~.preflight.resolve_target_identity` derives it —
    ``task.yaml``'s ``target.family`` when it is stated, otherwise the
    checkpoint's own ``architectures[0]`` through
    :func:`~.task_schema.derive_family` — rather than spelled here, because a
    second spelling of the target path is a guard that watches a directory
    nothing writes to.

    The **family** directory and not the run's own ``target_rel``: a run may
    legitimately write routing and shared modules beside its target, and all
    of it is code the gate is about to import.

    ``None`` when the family cannot be derived at all (no override and no
    readable ``config.json``). A caller that gets ``None`` has nothing to
    guard and must not invent a path to guard instead.
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
    directory: the message this feeds exists to say who is writing, and
    ``?? .../models/<family>/`` does not say it.

    An empty tuple means both "clean" and "could not tell" — a deliberate
    collapse. The guard this feeds stops a gate, and stopping a gate because
    git was unavailable would make the workflow unusable on any host without
    a checkout, which includes every host that runs these tests.

    Cheap because of the pathspec: a status scoped to the target takes a
    fraction of a second where a whole-worktree status can take a minute, the
    same asymmetry :func:`describe_tree_state` describes.
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

    Two questions, one place, because a gate wants both at the same instant:
    *what am I measuring* (recorded always) and *is someone else writing it*
    (a hard stop).

    The dirtiness question is scoped to the **target's own family path**, not
    to the repo. A workflow run has onboarding records, workflow fixes and
    reference YAMLs in flight more or less constantly, and a gate that
    refused to run while any of them were uncommitted would never run at all.
    The narrow scope is what makes the check something a run can leave on.

    ``allow_dirty`` overrides the ``MV2FLOW_ALLOW_DIRTY_GATE`` environment
    variable; either way the escape hatch is explicit and lands in the
    verdict, so a number measured over someone else's half-written edit is
    marked as one rather than passing as an ordinary result.
    """
    repo_root = task_data.get("trtllm_repo_path")
    if not repo_root:
        return TreeState()
    if allow_dirty is None:
        allow_dirty = os.environ.get(ALLOW_DIRTY_ENV, "").strip().lower() in {"1", "true", "yes"}
    described = describe_tree_state(repo_root)
    target_rel = target_family_rel(task_data)
    dirty = dirty_target_files(repo_root, target_rel) if target_rel else ()
    # The dirty marker is appended here rather than asked of git, and it is
    # scoped to the target — see :func:`describe_tree_state` for why the
    # repo-wide one is both a minute slower and less informative. A reader
    # seeing `-dirty` on a gate line therefore knows something was
    # uncommitted **in the code that gate measured**, not merely somewhere.
    description = f"{described}-dirty" if dirty else described
    return TreeState(description=description, dirty=dirty, override=bool(dirty) and allow_dirty)


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with everything needed to defend it later."""

    name: str
    passed: bool
    detail: str
    #: The parallel configuration this verdict was measured at. A ``unified``
    #: target is one codebase over a family of them, so a gate line that does
    #: not name the member cannot be told apart from the two members it did
    #: not run — the same class of failure, one level up, as a directory name
    #: that lies. ``None`` when the caller passes no member.
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
    #: :data:`UNKNOWN_TREE_STATE` where git could not answer. A gate measures
    #: whatever is on disk when the engine imports it, so without this a
    #: number is attributable to nothing — another writer may be rewriting the
    #: target while gates are queued against it. See
    #: :func:`describe_tree_state`.
    tree_state: str = UNKNOWN_TREE_STATE
    #: True when this verdict was measured over an uncommitted target path
    #: because the run said to. Recorded rather than merely permitted — the
    #: point of the escape hatch is that the resulting number stays marked.
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
    #: The per-prompt numbers behind ``measured``, where a gate has them. The
    #: acceptance gate's headline is one iteration-weighted rate, and a mean
    #: cannot say that one prompt ended in EOS after two tokens while another
    #: accepted at the ceiling; this rides in the journal line so it can.
    breakdown: str = ""
    #: The accuracy floor the measurement was held to, and the binomial
    #: standard error and sample count it was derived from. Every accuracy
    #: verdict carries the SE when the count is known, whichever rule set
    #: the floor, so a reader can see how many SE a margin is.
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
    #: and the device memory it left free. Empty for a gate that built none,
    #: and on a verdict recorded before engines were attested. Not hashed, so a
    #: verdict hashes as it did before it carried mappings.
    engine: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)

    def summary(self) -> str:
        """One line for the journal, carrying the numbers verbatim.

        The member is part of the label, not an aside: `progress.yaml`, the
        journal and the reviewer's evidence all read this line, and three of
        them per unit only mean three measurements if each says which
        configuration it measured.

        The tree state rides in the same line for the same reason one level
        up: `progress.yaml` and the journal are what a reader consults later,
        and three measurements are only comparable if each says what code it
        measured. So do the engine's attestation, its pool sizes and its free
        memory (:func:`describe_engine`): two verdicts on one configuration
        whose pools differ did not measure the same engine.
        """
        verdict = "VOID" if self.void else "PASS" if self.passed else "FAIL"
        label = f"{self.name}@{self.member}" if self.member else self.name
        tree = f" [tree {self.tree_state}]"
        if self.dirty_override:
            tree = f" [tree {self.tree_state}; DIRTY TARGET PATH measured by override]"
        breakdown = f" [{self.breakdown}]" if self.breakdown else ""
        breakdown += describe_engine(self.engine)
        if self.warnings:
            breakdown += "".join(f" [WARNING: {w}]" for w in self.warnings)
        # The numbers lead; the reasoning follows whenever the verdict is a
        # failure or rests on more than the numbers (a paired comparison, an
        # HTTP status), so the journal line says why without the log.
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
    """A :class:`GateResult` constructor that stamps ``state`` onto every verdict.

    Each gate has several exits — timed out, non-zero, unparsable, passed —
    and the tree state belongs on **all** of them equally: a failure that
    cannot say what tree it failed on costs the same debugging as a pass that
    cannot say what tree it passed on. Binding it once per gate is what keeps
    a later exit from being the one that forgets.
    """

    def verdict(**fields: Any) -> GateResult:
        return GateResult(tree_state=state.description, dirty_override=state.override, **fields)

    return verdict


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

    Stamped on every exit alike, a witness's included, which states no tree of
    its own. A void result (``void`` says why) does not pass, whatever it
    measured, and its detail leads with why.
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


# ------------------------------------------------------- the boot's greedy tokens
#
# A keyword says the engine generated something on topic; the HF logit
# reference (``onboard/logit_reference.py``) says which tokens. Where the run
# has one, the boot sends a few of its text prompts as token ids and holds the
# engine's first greedy tokens to the reference's continuation, token for
# token: a wrong layer, a misread weight or a wrong speculative verify shows
# at the first token it changes, in a boot, minutes before any benchmark.

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
        values: dict[str, int] = {}
        chunked = False
        for n, word in enumerate(words):
            if word == "--enable_chunked_prefill":
                chunked = True
            for flag in ("--max_num_tokens", "--max_seq_len"):
                if word.startswith(f"{flag}="):
                    values[flag] = int(word.split("=", 1)[1])
                elif word == flag and n + 1 < len(words):
                    values[flag] = int(words[n + 1])
        return cls(
            max_num_tokens=values.get("--max_num_tokens", QUICKSTART_MAX_NUM_TOKENS),
            max_seq_len=values.get("--max_seq_len"),
            chunked_prefill=chunked,
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

    The reference is the logit gate's file (``accuracy.logits.reference``),
    read whether or not that gate is on. The boot sends its first ``prompts``
    text prompts the unit's engine can take, as token ids, and compares the
    first ``tokens`` greedy tokens of each. Where the engine's token differs
    from the reference's, the step is a near-tie when the reference's top-1
    leads the engine's token by at most the logit gate's largest
    |delta log-prob| limit (``tie_factor`` times the reference's spread): the
    two then condition on different text, and the prompt is compared no
    further. Anywhere else a different token fails the boot.
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
        configuration variant that changes only lossless settings (a chunk
        size, block reuse, the capture list; ``observables.lossless``)
        compares as its unit does: the reference measures what it serves. Any
        other variant compares none unless its own entry,
        ``accuracy.boot.<variant>.greedy``, says ``enabled: true``: it serves
        another storage type or depth than the checkpoint's own, which the
        reference does not measure.
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

    An image prompt is left to the logit gate and the boot's own image
    prompt. ``reference`` defaults to the file ``setting`` names; one that
    cannot be read, or none of whose text prompts fits, is the workload's
    ``error``.
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
    order (``None`` where it returned none). A prompt passes when its first
    tokens are the reference's, up to the first near-tie if there is one.
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

    The keyword list starts **empty on purpose**. The rule is that
    the first engine run doubles as keyword verification — you freeze the
    cases only after observing the continuations — so an assembler that has
    not yet seen this model generate anything has no business asserting
    what it will say. Until then the gate asserts the weaker, still useful
    thing: the engine cold-starts and produces non-empty output.

    ``prompts`` and ``keywords`` are the plain form: each keyword must appear
    in SOME prompt's generated text. ``items`` are prompts with checks of
    their own, and the only ones that carry images. Keywords are matched
    against the generated texts only, never against the log: ``72`` matches
    ``CPUs 72-143`` in an engine log line.

    ``greedy`` is the HF reference's greedy tokens the engine's must be
    (:class:`GreedySetting`); :func:`resolve_boot_spec` sets it wherever the
    run has a reference.
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
        """Whether any prompt sends an image."""
        return any(item.images for item in self.items)

    def requests(self, repo_root: str | Path | None) -> list[dict[str, Any]]:
        """Engine-driver requests: the plain prompts, then the items.

        A relative image path is resolved against ``repo_root``, the tree
        under test, so a repository test asset names the same file on every
        checkout.
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
    no image gets the default image prompt appended: a multimodal boot that
    sends only text never exercises the vision path, and the first gate that
    does is an hour-long benchmark.

    A block that says only how the greedy tokens are compared
    (``accuracy.boot.greedy``) declares no workload. Wherever the run has an
    HF logit reference, the spec carries the comparison
    (:meth:`GreedySetting.from_task`), whichever workload it has.
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

    The shell is deliberate:
    the launch prefix resolved from ``task.yaml`` *is* shell syntax (env
    assignments, or an ``srun`` line), and re-parsing it into an argv would
    mean reimplementing the shell. Nothing here interpolates agent-authored
    text — the prefix comes from the task spec and the command from this
    module.
    """
    return _LOCAL_EXECUTOR.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))


#: The two gate commands take different flag vocabularies for the same
#: topology; see :func:`topology_flags`. Named rather than spelled inline
#: so a third caller cannot invent a third spelling by accident.
QUICKSTART_DIALECT = "quickstart"
EVAL_DIALECT = "eval"


def topology_flags(
    segment: str,
    world_size: int,
    member: str | None = None,
    *,
    dialect: str = QUICKSTART_DIALECT,
) -> list[str]:
    """Translate a parallel segment into ``quickstart_advanced.py`` flags.

    Every gate is **topology-blind** on its own: a ``tep4`` target served at
    tp2 boots, scores, and produces a self-consistent curve while the
    directory name lies. Deriving the flags from the segment here — rather
    than letting an agent type them — is what keeps the thing measured and
    the thing named in agreement.

    ``member`` is the configuration actually being served, and it is what
    gets translated when it is given. ``unified`` names a **family**, not a
    configuration, and translating the segment alone yields ``--tp_size=N``
    and nothing else: expert parallelism and attention data parallelism —
    the two branches a unified codebase exists to carry — would be switched
    on by no gate. See :func:`~.task_schema.members_for`, which decides what
    the members are.

    Without a member, ``unified`` translates to ``--tp_size`` alone rather
    than raising; the workflow always passes one.

    ``dialect`` names **which command these flags are for**, because the two
    gate commands spell the same topology differently and one of them cannot
    spell it at all:

    ==================  ===========================  ==========================
    axis                ``quickstart_advanced.py``   ``trtllm-eval``
    ==================  ===========================  ==========================
    tensor parallel     ``--tp_size``                ``--tp_size``
    expert parallel     ``--moe_ep_size``            ``--ep_size``
    attention DP        ``--enable_attention_dp``    **no flag at all**
    ==================  ===========================  ==========================

    Handing the quickstart spelling to ``trtllm-eval`` fails with::

        Error: No such option: --moe_ep_size
        (Possible options: --ep_size, --pp_size, --tp_size)

    A ``tp`` member emits only ``--tp_size``, which both commands accept, so
    a dialect mismatch is invisible at that member and at any target that is
    not ``unified``.

    Attention data parallelism has no ``trtllm-eval`` flag, so under the
    ``eval`` dialect it is **not emitted here**; it has to arrive through
    ``--extra_llm_api_options`` as ``enable_attention_dp: true``. Emitting it
    here would fail the command the same way, and silently dropping it would
    serve a ``dep`` member at ``tep`` while the verdict claimed otherwise —
    so :func:`eval_command` overlays it onto the unit's options file instead,
    and that is not optional.
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
    """The LLM-API settings a member needs that its gate command cannot carry.

    Exactly one today: attention data parallelism, which
    ``quickstart_advanced.py`` takes as ``--enable_attention_dp`` and
    ``trtllm-eval`` cannot take at all. Returned as a mapping rather than
    written by the caller so that "what does `dep` mean" has one answer, next
    to the flag table it is the exception to.
    """
    if not member:
        return {}
    kind, _ = parse_parallel_segment(member)
    return {"enable_attention_dp": True} if kind == "dep" else {}


#: Startup cost of a member's topology, cheapest first. The rank is on the
#: PARALLEL KIND rather than on a measured duration, because what a gate pays
#: before it can report anything is dominated by weight loading and by the
#: collectives the engine builds at init, and both follow from the kind:
#:
#: * ``tep`` — the expert table is PARTITIONED across ranks, so each rank
#:   materializes a fraction of the parameters that dominate a MoE checkpoint,
#:   and the attention weights are still tensor-sliced. Cheapest to start.
#: * ``dep`` — the same expert partition, plus attention data parallelism,
#:   which REPLICATES the attention weights on every rank instead of slicing
#:   them. More bytes per rank than ``tep``, fewer than ``tp``.
#: * ``tp`` — no expert partition at all: every rank holds a slice of every
#:   expert, which is the largest per-rank load and the most sharding work at
#:   init. The degenerate member, and the most expensive one.
#:
#: Startup timings on one checkpoint do not transfer to another, which is why
#: the order is derived from the kind and not from measured durations.
_MEMBER_COST_RANK: dict[str, int] = {"tep": 0, "dep": 1, "tp": 2}

# A member whose kind this module does not recognize — including ``unified``,
# which names a family rather than a configuration — sorts last. Unknown cost
# is not cheap cost, and the point of running cheapest first is to fail early.
_UNKNOWN_MEMBER_COST = len(_MEMBER_COST_RANK)


def order_members_by_cost(members: Sequence[str]) -> list[str]:
    """Return ``members`` cheapest-first, for a unit gated over several.

    Pure, total and deterministic: the answer depends on the SET of members
    and never on the order they arrived in, on the filesystem, or on any
    previous run, so two runs of the same unit gate in the same sequence and
    their journals compare line by line. An unrecognized member is ordered
    last rather than raising — this function decides a running order, and a
    gate loop that dies while choosing one has failed worse than a gate loop
    that runs in a poor order.

    Why order matters at all: **a unit is gated at every member, and a broken
    unit fails at all of them.** The first member to fail is the one that
    stops the unit, so making it the cheapest one buys back the rest. Ties
    break on world size ascending, then on the member name, both only so that
    the result is a function of its input.

    The cost model is :data:`_MEMBER_COST_RANK`; the reasoning for the order
    of the kinds lives there, next to the ranks it justifies.
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


#: How the stock path says *"I have never heard of this architecture"*, quoted
#: from the two places that raise it. ``AutoModelForCausalLM.from_config`` and
#: the multimodal-encoder branch beside it, in
#: ``tensorrt_llm/_torch/models/modeling_auto.py``, raise ``ValueError`` after
#: ``get_registered_model_class`` returns ``None``;
#: ``AutoConfig.from_hugging_face`` and
#: ``AutoModelForCausalLM.get_trtllm_model_class`` in
#: ``tensorrt_llm/models/automodel.py`` raise ``NotImplementedError`` after
#: ``MODEL_MAP.get`` returns ``None``. Both registries are keyed on the
#: checkpoint's own ``architectures[0]``, which is the string this workflow
#: exists to introduce.
#:
#: Matched on the message text because that is all the gate has: it runs the
#: engine as a subprocess and reads a log, so there is no exception object to
#: catch. The markers are deliberately the invariant half of each message —
#: the architecture name is interpolated into it and varies per run.
UNSUPPORTED_ARCHITECTURE_MARKERS: tuple[str, ...] = (
    "Unknown architecture for AutoModelForCausalLM",
    "Unknown architecture for AutoModelForMultimodalEncoder",
    "is not supported in TRT-LLM yet",
    # `get_model_architecture`'s own miss, from `_torch/models/modeling_utils.py`:
    #   raise RuntimeError(f"Unknown model architecture: {arch}")
    # A different message from the two `AutoModelFor*` ones above and reached by
    # a different path -- the registry lookup rather than the config loader --
    # so a checkpoint can fail here without matching any of them. The
    # acceptance gate depends on this list: if its `=off` leg dies in the
    # registry and no marker matches, the gate hard-fails rather than falling
    # back to its declared floor.
    "Unknown model architecture:",
    # `transformers`' own refusal, from `resolve_trust_remote_code`. It is the
    # same condition wearing a different coat: a checkpoint that ships an
    # `auto_map` and demands `trust_remote_code=True` is, by construction, one
    # whose architecture neither `transformers` nor TensorRT-LLM has built in —
    # the implementation travels with the weights. The stock path therefore
    # cannot load it, which is exactly what this fallback is for.
    #
    # The substrate gate can raise this rather than any registry message, so
    # a discriminator keyed only on the registry misses it and stops on a
    # checkpoint the stock path was never expected to load.
    "contains custom code which must be executed to correctly load the model",
    # `transformers`' refusal of an unknown `model_type` with NO remote code to
    # fall back on, from `AutoConfig.from_pretrained`. The checkpoint is newer
    # than the installed `transformers` and ships no configuration module, so
    # neither the stock path nor `trust_remote_code` can load it -- the same
    # condition again, one step earlier than the registry. Without this marker
    # the substrate gate stops here instead of proving the substrate on the
    # declared fallback checkpoint.
    "but Transformers does not recognize this architecture",
    # A registration shim: the architecture name is registered only so the
    # runtime can find something keyed on it (a multimodal input processor),
    # and building it without a modeling_v2 target raises. Once the run's own
    # config classes are registered, this is where the `=off` leg of the
    # acceptance gate dies -- after the config loader and the registry have
    # both succeeded -- so without this marker the gate hard-fails instead of
    # falling back to its declared floor.
    "has no built-in implementation",
)


def reports_unsupported_architecture(detail: str, output: str) -> bool:
    """Whether this text says *"the stock path does not know this model"*.

    Text-based so that callers other than the substrate gate can use it. The
    acceptance gate needs exactly this question: its `=off` leg compares
    against built-in modeling, and a genuinely new architecture has none.
    """
    haystack = f"{detail}\n{output}"
    return any(marker in haystack for marker in UNSUPPORTED_ARCHITECTURE_MARKERS)


def resolve_runtime(task_data: Mapping[str, Any], unit: str | None) -> Mapping[str, Any]:
    """The ``runtime`` block for ``unit``, with the per-unit form merged.

    Accepts the flat form (the whole run) or a mapping from unit name to a
    partial block layered over ``default``. Per-unit because the units of one
    run can genuinely DISAGREE about a switch: text units may gate with block
    reuse **on** while the multimodal path requires it **off**, and a run-wide
    value cannot be right for all of them.

    Why this field and not the LLM-API options file, which is already
    per-unit: ``quickstart_advanced.py`` has no ``--extra_llm_api_options``,
    so the **boot gate** -- the gate that runs first and most often -- cannot
    read that file at all. This block is the only channel it has.

    Merging is key-by-key over ``default``, so a unit states only what
    differs. ``extra_flags`` is an ordinary key and therefore REPLACES rather
    than extends: a unit that needs the default flags plus one more restates
    them, which is verbose and unambiguous.
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

    **Two of the four switches are spelled as negations on the command
    line.** ``overlap_scheduler`` and ``kv_cache_reuse`` are on by default
    in the engine, so wanting them means emitting *nothing* and not
    wanting them means emitting ``--disable_*``. That inversion is the
    whole reason this function exists: asked to "turn on block reuse", the
    obvious move is to add a flag, and the flag that exists turns it off.

    Doing it once here also keeps the gate and the prompts in agreement —
    the assembler is told what is enabled, not what was typed.

    ``capture`` is the unit's decode CUDA-graph list
    (:func:`resolve_capture_list`). With CUDA graphs on it is spelled as
    ``--cuda_graph_padding_enabled`` and ``--cuda_graph_batch_sizes``, so
    the quickstart gates capture the list the accuracy gate's options file
    gives ``trtllm-eval``, the way :func:`resolve_max_draft_len` aligns the
    draft depth. A spelling the runtime block's ``extra_flags`` already
    carries is not repeated; see :meth:`CaptureList.flags`.
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
# The quickstart gates (boot, acceptance, the witness legs) take runtime
# settings as `quickstart_advanced.py` flags, from the runtime block's
# `extra_flags`; `trtllm-eval`, `trtllm-serve` and the tune campaign take them
# as LLM-API keys, from the unit's options file. A setting stated in one
# channel was the other launcher's default: a KV fraction lowered in
# `extra_flags` reached the boot but not the logit or accuracy engines. Each
# setting both launchers take is therefore read from both channels and handed
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
    """Whether the checkpoint ships its own code (an ``auto_map``): every launcher must trust it.

    Such a checkpoint's config, tokenizer or processor is built from that code,
    so a launcher without ``trust_remote_code`` fails in its worker, whatever
    the gate measures.
    """
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

    With the checkpoint's own code to trust (:func:`checkpoint_remote_code`),
    ``--trust_remote_code`` too.
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

    What :func:`launch_flags` is for the quickstart, this is for the launchers
    that read an options file; with the checkpoint's own code to trust,
    ``trust_remote_code: true`` too. Merged under the options file's own keys
    (:func:`merge_options`), which win where both state a setting.
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

    Only the speculator features produce flags; everything else in the
    feature list is a property of the assembled target rather than of the
    command that runs it.

    ``dspark`` gets its own algorithm and never ``MTP``, even though a
    checkpoint may store its DSpark stages as ``mtp.*`` with
    ``num_nextn_predict_layers: 3``. ``MTP`` over those weights builds three
    chained one-token heads where the checkpoint trained one 3-stage pass over
    a 5-token block: it fails to load, or it drafts garbage that verification
    rejects -- and the second passes accuracy. The schema refuses ``mtp`` and
    ``dspark`` together, so the order here never has to decide between them.
    """
    names = {str(f).lower() for f in (features or ())}
    for feature, algo in (("mtp", "MTP"), ("eagle3", "Eagle3"), ("dspark", "DSPARK")):
        if feature in names:
            return [f"--spec_decode_algo={algo}", f"--spec_decode_max_draft_len={max_draft_len}"]
    return []


def resolve_max_draft_len(accuracy: Mapping[str, Any], unit: str | None) -> int:
    """How many draft tokens this unit's gates must ask for.

    **The gates read the speculator's depth from two different places, and
    this keeps them in agreement.** `trtllm-eval` takes no speculation flags,
    so the accuracy gate learns the depth from the unit's
    `--extra_llm_api_options` YAML -- `speculative_config.num_nextn_predict_layers`.
    The boot and acceptance gates drive `quickstart_advanced.py`, which cannot
    read that file, so they take it as `--spec_decode_max_draft_len`.

    If the two disagree, accuracy measures one depth while acceptance measures
    another against a different ceiling. The acceptance gate exists precisely
    because boot and accuracy are blind to a miscomputed draft path; an
    acceptance gate measuring a *different depth* than the unit was built for
    is blind in a new way, and it fails toward "green" because a shallower
    speculator accepts a larger fraction of what it drafts.

    So the YAML is the single source of truth and the flag is derived from
    it. A unit with no options file, or none declaring a speculator, gets 1.

    Read in the order the engine resolves it. ``max_draft_len`` is the field;
    ``num_nextn_predict_layers`` is its deprecated MTP alias, which
    ``MTPDecodingConfig`` remaps and drops when both are present; ``block_size``
    is how a DSpark drafter states its depth, and ``TorchLlmArgs`` rejects any
    DSpark config whose ``block_size`` differs from ``max_draft_len``. For
    DSpark this is not a knob at all: quickstart builds the config from the
    flag alone, so a depth of 1 against a drafter trained for a 5-token block
    fails that validation before the engine starts.
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
    from unit name to path. The mapping form exists because **a feature unit
    is not measured under the same configuration as core**: MTP needs a
    ``speculative_config``, and without one the accuracy gate silently
    re-measures the base path while the acceptance gate compares 1.0 against
    1.0 and calls it a pass.
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
# A unit's gates build engines through two launch paths, and each chooses its
# decode CUDA-graph batch list from a different place. `trtllm-eval` (accuracy)
# reads the unit's options file, `cuda_graph_config` included.
# `quickstart_advanced.py` (boot, acceptance, envelope) cannot read that file:
# it builds `CudaGraphConfig(batch_sizes=..., enable_padding=...)` from its own
# flags. Unaligned, the same unit captures 19 padded batches under accuracy and
# 34 unpadded ones under boot, so work certified on one list runs on the other.
# The list is therefore resolved once per unit and spelled as quickstart flags.

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
    ``tensorrt_llm/llmapi/llm_args.py``. Explicit ``batch_sizes`` are sorted
    and kept, and a ``max_batch_size`` beside them must equal their maximum
    (the LLM API raises otherwise, and so does this). Without them the list is
    generated up to ``max_batch_size``, 128 when unset: with padding 1, 2, 4
    and the multiples of 8 to 128, then steps of 64; without it every size to
    31, then 32, 64, 128 and the powers of two below the limit. The limit
    itself is always the last entry.

    The engine later drops the sizes above ``min(max_batch_size,
    max_num_tokens // (1 + draft))`` (``filter_cuda_graph_batch_sizes`` in
    ``tensorrt_llm/_torch/pyexecutor/engine/cuda_graph.py``). That filter
    takes each launch's own limits, so it is not part of this list.
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

        A runtime block's own ``--cuda_graph_batch_sizes`` or
        ``--cuda_graph_padding_enabled`` is kept as written and not repeated.
        Where it disagrees with this list the two launch paths would capture
        different batches, and preflight's ``check_capture_lists`` refuses
        the unit before any gate runs.
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

    The options' ``cuda_graph_config`` when they set one: it is the only
    statement ``trtllm-eval`` reads, and the one a user's ``--config``
    carries. Otherwise the LLM API's rule over the runtime block's flags: the
    ``--cuda_graph_batch_sizes`` and ``--cuda_graph_padding_enabled`` its
    ``extra_flags`` spell, ``CudaGraphConfig()``'s default when they spell
    neither. ``None`` when the options set ``cuda_graph_config: null``.
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


def resolve_capture_list(
    task_data: Mapping[str, Any],
    unit: str | None,
    *,
    runtime: Mapping[str, Any] | None = None,
) -> CaptureList | None:
    """The one decode CUDA-graph capture list ``unit``'s gates run (:func:`capture_list_from`).

    Read from the unit's options file and its runtime block; ``runtime``
    replaces the block, for a gate that overrides one of its switches.
    """
    path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    return capture_list_from(
        load_llm_api_options(path),
        resolve_runtime(task_data, unit) if runtime is None else runtime,
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

    See :func:`_launch_configurations` for a variant's options and runtime.
    Raises ``ValueError`` on a ``cuda_graph_config`` the LLM API refuses.
    """
    return [_launch_capture(*configuration) for configuration in _launch_configurations(task_data)]


# ------------------------------------------------------------ KV-cache configuration
#
# The same split as the capture list, for `KvCacheConfig`. `trtllm-eval`
# (accuracy) builds it from the unit's options file; `quickstart_advanced.py`
# (boot, acceptance, envelope, the witness legs) builds it from its own flags,
# which cover only some of its keys. A key with no flag would be the options
# file's under accuracy and the LLM API default everywhere else. The one that
# matters is `mamba_state_config`: without a snapshot policy the engine turns
# block reuse off for a hybrid linear-attention model
# (`_validate_and_adjust_mamba_snapshot_config` in
# `tensorrt_llm/_torch/pyexecutor/model_loader.py`), whatever the flags ask.
# The keys the quickstart has no flag for are therefore resolved once per unit
# and handed to the harness, which merges them into the `KvCacheConfig` the
# quickstart builds.

#: The ``KvCacheConfig`` keys ``quickstart_advanced.py``'s ``setup_llm`` sets
#: from its own flags. It passes every one of them, at the flag's default when
#: the argv has none, so for these keys the flag is the setting and the options
#: file is not read; every other key of the options' ``kv_cache_config`` is
#: handed over.
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
    """The ``kv_cache_config`` keys ``unit``'s quickstart gates take from its options file.

    The unit's (or configuration variant's) options file, read as
    :func:`resolve_capture_list` reads it; see :func:`quickstart_kv_cache_config`.
    """
    path = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    return quickstart_kv_cache_config(load_llm_api_options(path))


def with_kv_cache_config(
    environment: RunEnvironment, task_data: Mapping[str, Any], unit: str | None
) -> RunEnvironment:
    """``environment`` handing ``unit``'s KV-cache keys to the harness (:data:`KV_CACHE_CONFIG_ENV`).

    The variable is left out when the unit's options hand nothing over, so
    such a unit's launch line is unchanged.
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
    path asks for block reuse (the runtime block, the options'
    ``enable_block_reuse``), and ``snapshots`` whether the options set a Mamba
    state snapshot policy (:func:`has_snapshot_policy`).
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
    engine asked for block reuse, so a gate whose log has it ran without the
    reuse its unit asked for, and its verdict would certify a configuration
    nobody serves.
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
# Every resolver above decides what a gate ASKS of its engine, and the engine
# can run something else: the LLM API validates and fills in the arguments,
# and the executor's worker changes them again while it loads the model -- a
# hybrid model with no snapshot policy runs without block reuse and still
# generates the same text. So the harness prints what the engine runs once it
# is up (one `MV2FLOW_ENGINE` record per engine,
# `checks/harness/engine_facts.py`), and a gate that builds its engine through
# the harness compares it, field by field, with what it declared, and fails on
# any difference. The pools the engine sized and the device memory it left
# free ride in the verdict, so pool sizes compare across runs and a
# configuration one allocation away from running out of memory is named while
# it still passes.
#
# A configuration that says block reuse is on does not show that the engine
# reused anything. So the harness also prints what the engine reused over the
# workload (one `MV2FLOW_REUSE` record, `checks/harness/engine_reuse.py`): the
# prompt tokens each response says the engine resumed from its KV cache, and,
# where a gate asks for it, a probe that sends one prompt twice. The responses
# come from rank 0's executor at every world size, so this also attests block
# reuse where rank 0's worker is another process, and a unit that asks for
# reuse fails a gate whose engine reused nothing although it had a prefix to
# reuse (`reuse_shortfall`).

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
#: What a value every rank's own worker reported names as its source.
RANKS_SOURCE = observables.SOURCE_RANKS

#: What a block-reuse reading taken from the log, not from an object, names as its source.
LOG_SOURCE = observables.SOURCE_LOG
#: What one taken from the reuse probe names: one prompt sent twice, and the
#: engine resumed the second from its KV cache or computed it from the start.
PROBE_SOURCE = observables.SOURCE_PROBE
#: What one taken from the workload names: a request the engine resumed from its KV cache.
CACHED_TOKENS_SOURCE = observables.SOURCE_CACHED
#: Why an accuracy verdict's engine is not attested: the run printed no engine
#: record (the eval shim found no ``LLM`` class to wrap in the tree's
#: ``trtllm-eval``).
EVAL_NOT_ATTESTED = "the evaluation printed no engine record"

#: ``modeling_v2 newmodel: bound MambaHybridCacheManagerV2 #2: 4096 recurrent
#: slots, ...`` -- what a target prints, once per rank, each time it binds a
#: recurrent-state pool. The first bind is usually the KV-capacity estimation's
#: pool; the highest-numbered one is the pool the engine serves from.
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

    The engine's own reading of the same keys (``KvCacheConfig``'s validators):
    ``mamba_state_cache_interval`` is the interval's deprecated alias, and
    ``block_reuse_config.policy: per_conversation`` sets the interval to zero.
    No keys is the default policy, which takes no snapshots.
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

    ``words`` is the gate's quickstart argv (:func:`quickstart_words`), so it
    carries what the resolvers decided: the runtime block's switches and
    flags, the capture list (:func:`resolve_capture_list`), the speculator at
    the unit's depth, the member's topology, a gate's own batch limit. The
    unit's LLM-API ``options`` add what no flag carries: the snapshot policy
    among the KV-cache keys handed over to the engine
    (:func:`quickstart_kv_cache_config`), and a cache dtype the argv does not
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
    """Whether the engine keeps recurrent state.

    Its ranks bound a recurrent pool, or it runs a snapshot policy, which
    only a model with recurrent state has a use for.
    """
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

    The last ``MV2FLOW_ENGINE`` record in ``output`` is the engine's, and
    each rank's last ``MV2FLOW_RANK_ENGINE`` record is that rank's worker's
    own account (:func:`parse_rank_records`). A field every reporting rank
    carries is read from them (:data:`RANKS_SOURCE`) and held to ``declared``
    on each rank, so a rank that runs another value is named. A declared
    field no record carries is listed as not attested, never counted as
    agreeing; without a record every declared field is.
    Block reuse the record cannot carry (its worker ran in another process)
    is read off the log when an engine there said it turned reuse off, and
    otherwise off what the engine reused (:func:`reuse_evidence`). The pools
    come from what every rank logs, the free memory from the record, and
    what the engine reused from the last ``MV2FLOW_REUSE`` record.

    ``declared`` ``None`` compares nothing: a gate whose engine the harness
    does not build records the pools alone, and ``absent`` says why.
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

    Judged only where the engine had something to reuse:

    * the reuse probe ran, and the engine computed its second prompt from the
      start (it resumes the second from the cache when it reuses);
    * or more requests than the engine runs at once shared, with an earlier
      request, more than the shortest prefix it resumes from, and none
      resumed. With no more than that, every such request may have run beside
      the one it shares a prefix with, before there was anything to reuse.

    Nothing is judged when the responses did not report what they reused.
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
    still passes under it; its summary carries a ``LOW MARGIN`` warning,
    because the next change that adds memory to this configuration -- a
    feature's weights, a deeper draft, a larger pool elsewhere -- runs it out
    of memory in a unit that change never touched.
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
    """Pick the accuracy-reference discriminators for ``unit``.

    TensorRT-LLM's accuracy references key each model to a LIST of entries
    discriminated by ``quant_algo`` / ``kv_cache_quant_algo`` /
    ``spec_dec_algo``, and :func:`load_reference_score` matches a row only
    when the discriminator set is EXACTLY the one asked for. Passing none
    therefore means one of two things, both wrong:

    * the model has an undiscriminated row, and a quantized checkpoint
      silently anchors to the bf16 number;
    * the model has none, and the gate dies with ``no entry matching {}``
      at the end of an assembly rather than at spec-validation time.

    Per-unit for the same reason ``llm_api_options`` is: the ``mtp`` unit's
    anchor is the ``spec_dec_algo: MTP`` row, and measuring it against
    core's row compares two different questions.

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
    workdir: str | Path | None = None,
    environment: RunEnvironment | None = None,
    task_data: Mapping[str, Any] | None = None,
    beneath: Mapping[str, Any] | None = None,
) -> str:
    """Merge a member's own LLM-API settings over its unit's options file.

    ``beneath`` goes under the unit's file: settings it states win over it.

    Written beside the unit's file so the two are read together and a reader
    finding the derived one can see what it was derived from. Deterministic
    name, so a re-run overwrites rather than accumulating, and the file a
    verdict was measured under is still on disk afterwards. Gates of one
    member that run at once rewrite and read the same file, so it is replaced
    whole (:func:`write_atomically`): a reader never sees it half written.

    ``workdir`` is the gate's directory, used when the unit has no options
    file; ``environment`` and ``task_data`` decide whether the temp dir may
    stand in for it (see :func:`_refuse_temp_off_host`).
    """
    base: dict[str, Any] = {}
    source = Path(options) if options else None
    if source is not None and source.is_file():
        base = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    merged = merge_options(merge_options(beneath or {}, base), overlay)
    stem = f"{unit or 'unit'}.{member or 'member'}.llm_api.yaml"
    # Beside the unit's own options file, so the derived copy and the file it
    # was derived from are read together. With no unit file there is nowhere
    # natural: the gate's workdir, which the job can read, and on a local run
    # without one the temp dir. The cwd is the one place it must NOT go: the
    # cwd of a test run is the repository, and a generated artifact that lands
    # in the source tree is indistinguishable from a file somebody meant to
    # check in.
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
    """Build the ``trtllm-eval`` argv shared by the accuracy and acceptance gates.

    **`trtllm-eval` and `quickstart_advanced.py` take different flag
    vocabularies for the same runtime configuration.** There is no
    `--use_cuda_graph`, no `--kv_cache_dtype`, no `--mamba_ssm_cache_dtype`
    here; that configuration has to arrive as an `--extra_llm_api_options`
    YAML instead, so the `runtime` block cannot simply be forwarded.

    Left unforwarded, the result is not a weak comparison but a different
    question: the accuracy gate measures the engine's defaults (including a
    `max_seq_len` at the checkpoint's own maximum) while the boot gate
    measures the requested configuration and the anchor was measured at a
    third thing again. The configuration is therefore declared once, as a
    file, and the anchor run and both gates point at the same one.

    ``workdir`` is where a member's derived options file goes when the unit
    declares none; see :func:`_write_member_options`.
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
        # does not read the environment switch. Without this the gate serves
        # the built-in model (the argument defaults to "off") and its verdict
        # reads as the target's.
        overlay = {**overlay, "modeling_v2": mode}
    if overlay or beneath:
        # `trtllm-eval` has no attention-DP flag, so a `dep` member can only
        # be served through the options YAML. Writing a merged copy rather
        # than editing the unit's file keeps one source of truth: the unit
        # states its runtime configuration once, and the member adds only
        # what it alone needs. A `dep` gate run without this serves `tep`
        # while the verdict says `dep`, which is the failure the member set
        # exists to prevent.
        # The settings the unit states as runtime flags go beneath its file
        # (:func:`launch_overlay`), so `trtllm-eval` runs what the boot ran.
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
    # **The protocol lives here, after the benchmark name.**
    # `--apply_chat_template`, `--fewshot_as_multiturn`,
    # `--chat_template_kwargs`, `--num_fewshot` are options of the benchmark
    # subcommand, not of `trtllm-eval` itself, so they cannot go in the YAML
    # and they cannot go before the positional.
    #
    # The protocol changes the question: the same target can score tens of
    # points apart under the default 5-shot-completion protocol and under
    # the chat-template protocol. Neither number is wrong, and a gate that
    # compares a score under one against an anchor measured under the other
    # reports a regression that does not exist.
    # Shell-quoted: the command is assembled into a string and run through a
    # shell, and a protocol argument like
    # `--chat_template_kwargs={"enable_thinking": false}` carries spaces and
    # braces that the shell would otherwise split and glob. `shlex.quote`
    # leaves ordinary flags untouched.
    # Per unit, because the PROTOCOL flags are options of the BENCHMARK
    # subcommand and the subcommands do not share an option set. `gsm8k`
    # takes `--apply_chat_template` and `--fewshot_as_multiturn`; `mmmu`
    # takes neither and click rejects the whole command with "no such
    # option". A run-wide list is therefore not merely imprecise for a
    # multimodal unit -- it cannot run at all.
    parts.extend(
        shlex.quote(str(arg)) for arg in resolve_per_unit(accuracy, "benchmark_args", unit) or ()
    )
    return parts


def _with_mode(environment: RunEnvironment, mode: str) -> RunEnvironment:
    """Return a copy of ``environment`` with the modeling_v2 switch set.

    Exported into the *prefix*, not into the orchestrator's own process:
    worker ranks receive the environment as it stood when MPI initialized,
    so a value set after launch reaches the driver and not them — and a
    driver resolving a modeling_v2 target while its workers resolve the
    built-in is exactly the silent split the switch exists to prevent.

    The same prefix puts :data:`RANK_HOOK_DIR` first on ``PYTHONPATH``, ahead
    of what the job inherits, and sets :data:`RANK_FACTS_ENV`: every rank's
    worker then prints what its engine runs (:func:`parse_rank_records`).
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
# The gates that need per-request structure -- token ids, whole texts,
# iterations, images -- run `checks/harness/engine_driver.py` under the tree's own
# interpreter. It builds the engine with the tree's own quickstart from the
# same argv the quickstart gates use, so the configuration measured is
# identical, and prints one `MV2FLOW_RECORD` JSON line per request. A gate
# reads those lines and nothing else of the log.

#: The driver script, run as a file under the measured tree's interpreter.
ENGINE_DRIVER = Path(__file__).resolve().parent / "harness" / "engine_driver.py"
#: The accuracy gate's ``trtllm-eval``, run in-process by this script so the
#: engine it builds is reported (``checks/harness/eval_shim.py``).
EVAL_SHIM = Path(__file__).resolve().parent / "harness" / "eval_shim.py"


def eval_harness_command(environment: RunEnvironment, parts: Sequence[str]) -> str:
    """The launch of ``parts`` (:func:`eval_command`'s argv) under the eval shim, as engine work.

    The argv is unchanged, ``trtllm-eval`` first: the shim runs the tree's own
    console script on it, in its own interpreter, and adds the engine's
    ``MV2FLOW_ENGINE`` and ``MV2FLOW_REUSE`` records to the output.
    """
    return environment.python_command(f"{shlex.quote(str(EVAL_SHIM))} {' '.join(parts)}")


def options_declaration(options: Mapping[str, Any], *, block_reuse: bool) -> dict[str, Any]:
    """What an engine built from LLM-API ``options`` must run: block reuse, and the cache dtypes named.

    ``trtllm-eval`` builds its engine from the options file alone, so that is
    all the accuracy gate can hold it to; the cache dtypes are checked on
    every rank where each rank reports (:func:`attest_engine`).
    """
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


#: The quickstart flag a declared batch limit replaces. The runtime block
#: usually sets it through ``extra_flags``, so an override has to remove that
#: spelling, not just append another one.
BATCH_SIZE_FLAG = "--max_batch_size"


def with_max_batch_size(flags: Sequence[str], max_batch_size: int | None) -> list[str]:
    """``flags`` with every ``--max_batch_size`` spelling replaced by one ``=N``.

    A token-comparing gate declares the batch limit its legs run at, and the
    runtime block has usually set its own through ``extra_flags`` --
    ``--max_batch_size=8``, ``--max_batch_size 8`` as one string, or the flag
    and its value as two. All three are removed, so the argv states ONE limit.
    ``None`` returns ``flags`` unchanged.
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


def quickstart_words(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    features: Any | None,
    max_tokens: int,
    member: str | None,
    max_batch_size: int | None = None,
    extra: Sequence[str] = (),
    runtime_overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    """The quickstart argv as a list of plain words, for the engine driver's spec.

    The same flags the quickstart gates emit -- topology, the unit's runtime
    block, its CUDA-graph capture list, the speculator at the unit's draft
    depth, the settings its options file states that both launchers take
    (:func:`launch_flags`) -- with each runtime ``extra_flags`` entry split as
    the shell would split it. ``runtime_overrides`` replaces switches of the unit's runtime
    block (a leg that turns the overlap scheduler off, for one). The KV-cache
    keys no flag carries reach the engine through the launch environment
    instead (:func:`with_kv_cache_config`).
    """
    runtime = {**resolve_runtime(task_data, unit), **dict(runtime_overrides or {})}
    stated = [
        *runtime_flags(runtime, capture=resolve_capture_list(task_data, unit, runtime=runtime)),
        *speculation_flags(
            task_data.get("features") if features is None else features,
            max_draft_len=resolve_max_draft_len(task_data.get("accuracy") or {}, unit),
        ),
        *extra,
    ]
    flags = with_max_batch_size(
        [*stated, *launch_flags(task_data, unit, stated)],
        max_batch_size,
    )
    words = [
        f"--model_dir={task_data['checkpoint_path']}",
        f"--max_tokens={int(max_tokens)}",
        *topology_flags(str(task_data["parallel"]), environment.world_size, member),
    ]
    for flag in flags:
        words.extend(shlex.split(str(flag)))
    return words


def _refuse_temp_off_host(
    what: str, environment: RunEnvironment | None, task_data: Mapping[str, Any] | None
) -> None:
    """Raise when a file in this process's temp dir would be read on another host.

    The task's environment kind selects the executor, and the environment
    handed in selects the launcher. When a scheduler stands behind either (see
    :func:`~.environment.scheduler_owns_devices`) the job may start on a host
    whose temp dir is not this one, and it then fails with
    ``FileNotFoundError`` on a file that exists, here. Every caller in this
    package passes a workdir under the workspace, so arriving here under such
    a kind is a caller to fix.
    """
    kinds = [str(((task_data or {}).get("environment") or {}).get("kind", "local"))]
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
    task_data: Mapping[str, Any] | None = None,
    *,
    environment: RunEnvironment | None = None,
) -> Path:
    """Where a gate writes the files its job reads.

    ``workdir`` when given: the orchestrator passes a directory on the shared
    filesystem, because the job may run on another host (a spool session).
    Otherwise a directory under the temp dir named for the gate and the task
    (its tree and checkpoint), so two runs of one gate on one task write the
    same files and their commands compare equal; a fresh one without a task.

    The temp dir is this process's, so that fallback serves a local run
    only: under a kind whose scheduler places the job, ``environment``'s or
    the run's own, this raises instead (see :func:`_refuse_temp_off_host`).
    """
    if workdir is not None:
        directory = Path(workdir)
    else:
        _refuse_temp_off_host(f"the {name} gate", environment, task_data)
        if task_data is None:
            return Path(tempfile.mkdtemp(prefix=f"mv2flow-{name}-"))
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
    host shares, so one run's compiled kernels and autotuning choices are
    read by another's gates. The task's own value wins, and
    :func:`pin_tuning` still overrides it for the legs of a compared gate.
    """
    block = task_data.setdefault("environment", {})
    env = block.get("env")
    if env is None:
        env = block["env"] = {}
    env.setdefault(TRITON_CACHE_DIR_ENV, str(Path(workspace).resolve() / RUN_TRITON_CACHE_DIRNAME))


def pin_tuning(directory: str | Path) -> dict[str, str]:
    """The environment that makes every leg of one gate run share its tuning choices.

    Each leg is a process of its own, and two tuners choose by timing in each
    process: the AutoTuner picks tactics, and ``@triton.autotune`` picks a
    kernel config (the FLA chunk kernels' ``num_warps``, ``BK``, ``BV``).
    Different configs reduce in a different order, so two processes running
    one configuration can differ in the last bit, which greedy decoding turns
    into a different token. With every leg pointed at the caches under
    ``directory/tuning``, the first leg tunes and every later one reads its
    choices.

    Call it once per gate run, before the first leg: it deletes the choices a
    previous run left there, so a run never inherits another's timings.
    Triton's compiled kernels in the same cache stay; a compiled config is the
    same binary whichever run built it. Merge the result over
    ``task.yaml``'s ``environment.env``, not under it: legs tuned apart are not
    a comparison, and the task's own cache stays the one every other gate uses.

    Triton releases that predate ``TRITON_CACHE_AUTOTUNING`` ignore it, and
    Triton never caches a kernel whose configs carry a ``pre_hook``;
    :func:`triton_tuning_note` reports whether a run cached anything.
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


def triton_tuning_note(pins: Mapping[str, str]) -> str:
    """Whether the legs pinned by :func:`pin_tuning` cached any Triton autotuning; empty unpinned."""
    directory = pins.get(TRITON_CACHE_DIR_ENV)
    if not directory:
        return ""
    cached = sum(1 for _ in Path(directory).rglob(TRITON_AUTOTUNE_RESULT_GLOB))
    if cached:
        return f"Triton autotuning shared by the legs ({cached} tuning result(s) cached)"
    return (
        f"no Triton autotuning was cached: no autotuned Triton kernel ran, or this Triton "
        f"ignores {TRITON_CACHE_AUTOTUNING_ENV} and each leg tuned its own"
    )


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
    perf: Mapping[str, Any] | None = None,
    reuse_probe: bool = False,
) -> Path:
    """Write one engine-driver spec (see ``checks/harness/engine_driver.py``).

    ``reuse_probe`` has the driver send one prompt twice after the workload,
    so the gate sees whether the engine resumes the second from its KV cache
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
        "perf": dict(perf) if perf else None,
    }
    if reuse_probe:
        spec["reuse_probe"] = True
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def engine_driver_command(
    environment: RunEnvironment, spec_path: str | Path, quickstart: str | Path | None = None
) -> str:
    """The launch of the engine driver over ``spec_path``, as engine work.

    The quickstart and its argv are spelled on the command line as well as in
    the spec (the driver takes the command line's), so a command read from a
    log says which tree it built from and with which flags. The prompts stay
    in the spec: a long one would not fit an argv string.
    """
    try:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        spec = {}
    if quickstart is None:
        quickstart = spec.get("quickstart")
    argv = [str(a) for a in spec.get("quickstart_argv") or ()]
    named = f" --quickstart {shlex.quote(str(quickstart))}" if quickstart else ""
    flags = f" -- {' '.join(shlex.quote(a) for a in argv)}" if argv else ""
    return environment.python_command(
        f"{shlex.quote(str(ENGINE_DRIVER))} --spec {shlex.quote(str(spec_path))}{named}{flags}"
    )


def parse_driver_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_RECORD`` the engine driver printed, in request order."""
    records = workload_io.parse_marked(output, workload_io.RECORD_MARKER)
    return sorted(records, key=lambda r: int(r.get("index", 0)))


def parse_eval_score(stdout: str) -> float | None:
    """Extract the final accuracy from a ``trtllm-eval`` run's output.

    Both reporting paths are matched because the benchmark decides which
    one runs: MMLU prints its own weighted average, everything else comes
    through lm-eval. The **last** match wins — a long run also prints
    partial-score estimates, and the final line is the one that counts.
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
    #: output carried only a summary statistic, which says nothing about how
    #: much work it was estimated from.
    requests: tuple[tuple[int, int], ...] = ()
    #: A digest of each request's prompt, parallel to ``requests``, where the
    #: run printed one (the engine driver does). What pairs a prompt's rate
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

        The first streamed iteration of every request is its context step:
        it returns the prefill's token and verifies no draft, so it is not
        counted.
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

    The engine driver's records come first: one per prompt, with the prompt's
    digest, so a rate can be paired with the same prompt's rate elsewhere.

    The quickstart form is **iteration-weighted across requests**: the
    quantity is tokens-per-iteration, so several requests combine as
    ``sum(tokens) / sum(iterations)`` and not as the mean of their printed
    ratios. Averaging the ratios would let one short request with an
    unlucky tail dominate a long one, and the whole judgement here is a
    ratio between two runs. The plain per-prompt mean is still reported, in
    :meth:`AcceptanceReading.breakdown`, because it is the number that shows
    a prompt that never drafted.
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


def parse_acceptance_length(stdout: str) -> float | None:
    """Extract ``acceptance_length`` from an eval, bench or quickstart run.

    The number alone; see :func:`parse_acceptance` for how it is combined.
    """
    reading = parse_acceptance(stdout)
    return reading.value if reading is not None else None


def load_reference_score(
    reference_yaml: str | Path,
    model_key: str,
    *,
    discriminators: Mapping[str, Any] | None = None,
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    The file is keyed by HuggingFace repo name and holds a list of entries,
    each optionally discriminated by ``quant_algo`` / ``kv_cache_quant_algo``
    / ``spec_dec_algo``. An entry matches when its discriminators are
    exactly the ones asked for — no more, no less — so an FP8 anchor is
    never silently used for a BF16 run.

    **The anchor is never invented.** If nothing matches, that is a question
    for the plan phase to answer from a written source, not a number for
    this function to estimate.
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

    ``member`` is the parallel configuration to serve at, and it is the only
    thing that makes a ``unified`` target's gate mean anything — see
    :func:`topology_flags`. It also travels into the verdict, so the journal
    line says which of the members this one was.

    ``unit`` selects the ``runtime`` block when the spec states one per unit.
    It has to: `quickstart_advanced.py` takes no `--extra_llm_api_options`, so
    this gate cannot hand the engine the per-unit options file the accuracy
    gate does, and the runtime block is its channel for a unit-specific switch.
    Two things are read from that file all the same: the CUDA-graph capture
    list (:func:`resolve_capture_list`) and the KV-cache keys the quickstart
    has no flag for (:func:`resolve_kv_cache_config`), handed to the engine
    driver. A log in which the engine turned block reuse off fails the gate
    (:func:`reuse_disabled_detail`), and so does an engine that runs any
    field of :data:`ATTESTED_FIELDS` other than the gate declared
    (:func:`attest_engine`); the verdict carries the engine's record either
    way, and a ``LOW MARGIN`` warning when it left little memory free
    (:class:`MemoryMargin`). After the prompts the driver sends the reuse
    probe, one prompt twice: a unit that asks for block reuse fails when the
    engine computed the second from the start (:func:`reuse_shortfall`), and
    at every member the probe attests block reuse from rank 0's responses.

    ``features`` is **the features the target under test actually
    implements**, which is not the same as the features the *run* will
    finish with. Defaulting it to ``task_data["features"]`` means the
    ``core`` unit is gated with the speculator's flags before a speculator
    exists. The rule is that core is green *before* the first feature
    increment; the gate has to agree with it.

    ``spec`` is the workload (see :func:`resolve_boot_spec`): text prompts,
    and for a multimodal unit at least one prompt that sends an image. The
    engine driver builds the engine exactly as ``quickstart_advanced.py``
    does from the same flags and prints one record per prompt; keywords are
    matched against those generated texts only, never against the log, so
    ``72`` cannot pass on ``CPUs 72-143``.

    With ``spec.greedy`` the driver also sends the HF reference's text
    prompts as token ids (:func:`greedy_workload`), every request generates
    its whole budget past any end token, and the engine's first greedy
    tokens must be the reference's up to a near-tie (:func:`judge_greedy`).
    That is judged before the keywords, and a reference that cannot be read
    fails the gate before the engine starts.

    ``mode`` defaults to ``require`` and should stay there for any result
    that will be attributed to modeling_v2. It is a parameter for one
    reason: when a boot gate fails, the next question is always *"is this
    the target or is this the host?"*, and the cheapest answer is the same
    command under ``off`` — the stock implementation, same prompts, same
    runtime flags, one variable.

    ``allow_dirty`` is the escape hatch of :func:`inspect_tree`; see there
    for why a dirty target path stops this gate by default.
    """
    spec = spec or BootSpec()
    env = with_kv_cache_config(_with_mode(environment, mode), task_data, unit)
    # A control run must not be recorded under the same name as the real
    # gate. Two rows both called "boot", one of which measured the stock
    # implementation, is a journal that reads as evidence for a target it
    # never exercised.
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
        return _verdict(
            passed=False,
            detail=f"did not run to a verdict: {result.infrastructure}",
            infrastructure=result.infrastructure,
            **ran,
        )
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
    reuse_off = reuse_disabled_detail(
        combined, options=resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    )
    if reuse_off:
        return _verdict(passed=False, detail=reuse_off, **ran)
    shortfall = reuse_shortfall(attested, asks=bool(declared.get("block_reuse")))
    if shortfall:
        return _verdict(passed=False, detail=shortfall, **ran)
    if attested.mismatches:
        return _verdict(passed=False, detail=attested.difference(), **ran)
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
    mode: str = MODE_REQUIRE,
    unit: str | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
) -> GateResult:
    """Run ``trtllm-eval`` and compare against the anchor. One-sided.

    ``member`` names the parallel configuration measured, which for a
    ``unified`` target is the whole question: one codebase serving three
    configurations is three anchored numbers, not one repeated three times.

    **The floor is statistical by default**: ``reference - k * SE``, where SE
    is the binomial standard error of a score at the reference measured on
    the benchmark's sample count (see :func:`resolve_samples`) and ``k`` is
    ``accuracy.floor.k`` (default 2). A fixed ``reference - tolerance`` is
    the explicit alternative, ``accuracy.floor.rule: tolerance``. Every
    verdict records the SE, the count and the rule.

    The tree-state guard matters most for this gate. It is the
    expensive one — a quarter of an hour per member — and it is the one whose
    output becomes the target's headline number, so it is also the one where
    measuring an unreviewed tree is least recoverable. See
    :func:`inspect_tree` and ``allow_dirty``.

    ``workdir`` holds the member's derived options file when the unit
    declares none; see :func:`eval_command`.

    ``trtllm-eval`` runs under the eval shim (:func:`eval_harness_command`),
    which prints its engine's record and what the engine reused over the
    benchmark. The verdict carries both and the pools the ranks logged; of
    the configuration block reuse and the cache dtypes are held to the
    unit's options file (:func:`options_declaration`, :func:`attest_engine`).
    A unit that asks for reuse fails when the engine reused nothing although
    the benchmark's prompts shared prefixes it could resume from
    (:func:`reuse_shortfall`).
    """
    accuracy = dict(task_data.get("accuracy") or {})
    benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or "gsm8k")
    tolerance = float(accuracy.get("tolerance", 0.0))
    env = _with_mode(environment, mode)
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

    # The moded environment: `eval_command` hands its `modeling_v2` to the
    # engine through the options file, the only channel `trtllm-eval` reads.
    parts = eval_command(task_data, env, benchmark, unit=unit, member=member, workdir=workdir)
    command = eval_harness_command(env, parts)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(command=command, duration_s=elapsed, stdout_tail=tail, **common)

    if result.infrastructure:
        return _verdict(
            passed=False,
            detail=f"did not run to a verdict: {result.infrastructure}",
            infrastructure=result.infrastructure,
            **ran,
            **floor_fields(planned),
        )
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
    reuse_off = reuse_disabled_detail(combined, options=options)
    if reuse_off:
        return _verdict(passed=False, detail=reuse_off, **ran, **floor_fields(planned))
    shortfall = reuse_shortfall(attested, asks=asks)
    if shortfall or attested.mismatches:
        return _verdict(
            passed=False,
            detail=shortfall or attested.difference(),
            **ran,
            **floor_fields(planned),
        )

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
    """The cross-member check over one unit's accuracy verdicts.

    Every member passing its own floor is not the same as every member being
    on the anchor: a defect in code the members share moves all of them the
    same way, and each still clears ``reference - 2 SE``. When **every**
    member sits more than one SE under the reference, that is a finding, and
    delivery refuses it unless ``accuracy.floor.acknowledge.<unit>`` gives a
    reason. Verdicts without a measurement or an SE are left out.
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


#: Prompts an acceptance workload must hold before its rates are compared. On
#: a handful of prompts, a change that only moves the generated text can move
#: the mean rate by a quarter of a token per iteration; at 32 prompts the same
#: change stays within noise under the paired test. A task may lower it
#: through ``accuracy.acceptance.min_prompts``, and the verdict then rests on
#: fewer pairs, visibly.
DEFAULT_ACCEPTANCE_MIN_PROMPTS = 32
#: Prompts taken from the accuracy dataset when the task declares none.
DEFAULT_ACCEPTANCE_NUM_PROMPTS = 32
#: The minimum drop, in accepted tokens per iteration, a paired comparison
#: must find before it calls a significant change a regression.
DEFAULT_ACCEPTANCE_MIN_EFFECT = 0.1


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
    accuracy: Mapping[str, Any], unit: str | None, *, min_ratio: float = 0.95
) -> AcceptanceFloor | None:
    """The absolute floor ``unit`` declares in ``accuracy.acceptance``, if any.

    ``min_absolute`` is the floor itself. ``measured_reference`` is what a
    reference drafter measured on this workload; with no ``min_absolute``
    the floor is that times ``min_ratio``, the same ratio the stock
    comparison uses. ``floor_source`` names where the measurement was taken.
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
            return AcceptanceFloor(measured * min_ratio, source, measured)
        return None
    try:
        return AcceptanceFloor(float(chosen))
    except (TypeError, ValueError):
        return None


def resolve_acceptance_floor(accuracy: Mapping[str, Any], unit: str | None) -> float | None:
    """The absolute `acceptance_length` floor this unit declares, if any.

    Only consulted when the `=off` leg **cannot exist**. See
    :func:`run_acceptance_gate` and :func:`resolve_acceptance_floor_spec`.
    """
    floor = resolve_acceptance_floor_spec(accuracy, unit)
    return floor.value if floor is not None else None


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
    cache only (:data:`.harness.workload_io.KNOWN_DATASETS`). A multimodal
    unit's benchmark has no text dataset, so the run-wide benchmark's is used.
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
        "answer_field": known.get("answer_field"),
        "split": setting("split") or known.get("split", "test"),
        "config": setting("dataset_config") or (None if declared else known.get("config")),
        "template": setting("prompt_template") or "{text}",
    }
    return dataset, f"{dataset['num']} prompts from {where} ({source})"


def resolve_acceptance_workload(
    accuracy: Mapping[str, Any],
    unit: str | None,
    *,
    max_tokens: int | None = None,
) -> AcceptanceWorkload:
    """The acceptance workload ``unit`` declares, or the accuracy dataset's prompts.

    Read from ``accuracy.acceptance``, most specific level first: the unit's
    own entry, then its ``default`` entry, then the block's top level (see
    ``task_schema.ACCEPTANCE_WORKLOAD_KEYS``). ``prompts`` and
    ``prompts_file`` are one setting, so the level that states either wins
    for both. With neither, the workload is the first ``num_prompts``
    (default 32) natural-text questions of the accuracy dataset; see
    :func:`acceptance_dataset`.

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
    if declared_tokens is None:
        declared_tokens = max_tokens if max_tokens is not None else DEFAULT_ACCEPTANCE_MAX_TOKENS
    declared_minimum = setting("min_draft_iterations")
    min_prompts = setting("min_prompts")
    return AcceptanceWorkload(
        prompts=prompts,
        max_tokens=int(declared_tokens),
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

    The member, the workload and the speculator flags -- not the unit: a
    later unit that keeps speculation on is compared against the earlier
    unit's verdict, which is how a regression a later feature introduces in
    the draft path shows up.
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

    One draft position more adds the tokens of every iteration that accepts
    all of them; a position whose draft is never accepted adds nothing. The
    comparison is one-sided at ``policy.alpha`` with no minimum effect: a
    deep position is accepted rarely, and what is asked is that it is at
    all. ``None`` when fewer than two prompts were measured by both.
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
    min_ratio: float = 0.95,
    unit: str | None = None,
    features: Any | None = None,
    max_tokens: int | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
    records: GateRecords | None = None,
) -> GateResult:
    """Measure per-prompt acceptance, and judge it against stock, a floor and the last tree.

    **The workload** is the task's to declare (``accuracy.acceptance.prompts``,
    ``prompts_file`` or ``dataset``; see :func:`resolve_acceptance_workload`)
    and is otherwise the first 32 questions of the accuracy dataset: natural
    text whose answers run long. It goes to the engine driver (see
    :data:`ENGINE_DRIVER`), which runs each prompt alone, streamed, and
    prints its tokens and iterations; the rate of a prompt is its tokens per
    iteration, and the gate's number is iteration-weighted over all of them.

    **Size.** Fewer than ``min_prompts`` prompts (default 32) or fewer than
    ``min_draft_iterations`` generation iterations fails as a workload too
    small or too short to measure, not as a low number.

    **Three comparisons**, each only where it can be made:

    * against stock: the same command under ``TRTLLM_MODELING_V2=off``, in
      the same session. A leg that finds the stock path cannot load the
      checkpoint is recorded in ``records`` per checkpoint and GPU
      architecture, and not run again where another comparison stands in. It fails when the weighted ratio is under
      ``min_ratio`` AND the per-prompt paired test finds the drop
      significant. A new architecture has no stock path. Then, above draft
      depth 1, the second leg is the same target one draft depth shallower,
      on the same prompts, and this depth must accept significantly more
      (:func:`depth_rise`): a draft position never accepted adds nothing, so
      a first verdict is checked rather than taken as the baseline. A
      declared floor (see :func:`resolve_acceptance_floor_spec`) is held as
      well; at depth 1 it is the only comparison, and the verdict warns when
      it is not tied to a measured reference.
    * against 1.0: a target at or under it rejects every draft.
    * against the previous verdict for the same member, workload and
      speculator in ``records``: a paired test over the prompts both
      measured (``accuracy.acceptance.regression``: t or Wilcoxon, alpha,
      minimum effect). A significant drop at least the minimum effect large
      fails; a fixed delta would read noise on a few prompts as a
      regression.

    Every verdict with a reading carries the per-prompt rates. A passing
    reading is appended to ``records`` and becomes the next tree's baseline.

    The two legs differ in exactly one variable and must run in the same
    session; both import the same working tree, so the tree-state guard of
    :func:`inspect_tree` stops a gate whose target path is being edited. Each
    leg's engine is held to the declared configuration (:func:`attest_engine`)
    with its own ``modeling_v2``, so a leg that ran anything else fails the
    gate rather than turn the comparison into one of two variables. Each leg
    ends with the reuse probe, as the boot gate's does
    (:func:`reuse_shortfall`); it sends no prompt the acceptance reading counts.
    """
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name="acceptance", member=member, passed=False, detail=state.block_detail())
    accuracy = task_data.get("accuracy") or {}
    try:
        workload = resolve_acceptance_workload(accuracy, unit, max_tokens=max_tokens)
    except TaskSchemaError as exc:
        return _verdict(
            name="acceptance",
            member=member,
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
    floor_spec = resolve_acceptance_floor_spec(accuracy, unit, min_ratio=min_ratio)
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
                name="acceptance",
                member=member,
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
            # **The stock leg can be impossible rather than broken.** A
            # genuinely new architecture HAS no built-in modeling, so `=off`
            # cannot load it at all -- the check is unavailable exactly when
            # onboarding is doing its job (see
            # :data:`UNSUPPORTED_ARCHITECTURE_MARKERS`). The
            # spec may then declare an absolute floor instead, and the weaker
            # claim is recorded verbatim.

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
                name="acceptance",
                member=member,
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
        reuse_off = reuse_disabled_detail(combined, options=options)
        shortfall = reuse_shortfall(attested[-1], asks=bool(declared.get("block_reuse")))
        if reuse_off or shortfall or attested[-1].mismatches:
            return _verdict(
                name="acceptance",
                member=member,
                passed=False,
                detail=(
                    f"the {MODELING_V2_ENV}={mode} leg: {reuse_off}"
                    if reuse_off
                    else shortfall or attested[-1].difference()
                ),
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
            name="acceptance",
            member=member,
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
    base = dict(name="acceptance", member=member, duration_s=elapsed, engine=engine["engine"])

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
            f" (= measured reference {floor_spec.measured:.2f} x {min_ratio:.2f})"
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
        passed = ratio >= min_ratio or (paired is not None and not paired.significant)
        notes.append(
            f"acceptance_length {target:.4f} vs stock {stock:.4f} (ratio {ratio:.3f}, floor "
            f"{min_ratio:.2f})"
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

    def last(self, name_prefix: str) -> GateResult | None:
        for result in reversed(self.results):
            if result.name.startswith(name_prefix):
                return result
        return None

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

    ``discriminators`` overrides the spec; otherwise they come from
    ``accuracy.discriminators`` for ``unit``. A caller that passes neither
    gets the undiscriminated row, which for a quantized checkpoint is the
    wrong anchor -- see :func:`resolve_discriminators`.
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
    "BOOT_RENDERS",
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
    "parse_acceptance_length",
    "parse_eval_score",
    "resolve_acceptance_floor",
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
    "triton_tuning_note",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gates — commands with exit codes, never agent judgements.

This module is the reason the workflow has no ``--min-score`` and no QA
agent scoring an artifact out of ten. A staircase target is trusted because
its gates passed, and a gate is a process that either returned zero and
printed a number above its anchor, or did not. Putting that behind an agent
turn adds latency and adds a way for the verdict to be rationalized.

Three gates, in ascending cost:

1. **boot** — minutes, binary. The engine cold-starts under
   ``TRTLLM_MODELING_V2=require``, the weight manifest covers what it
   claims, and a handful of greedy continuations come back. Catches
   catastrophes, not accuracy. Runs after every assembler turn.
2. **accuracy** — the release criterion. ``trtllm-eval`` against the anchor
   in TensorRT-LLM's own accuracy references. One-sided:
   ``measured >= reference - k * SE``, the SE binomial over the benchmark's
   sample count (``accuracy.floor``; a fixed ``tolerance`` is still
   available as an explicit rule). Runs at a unit boundary.
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
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from agent_flow.runtime import (
    CommandSpec,
    ExecutionRecord,
    LocalCommandExecutor,
    SpoolCommandExecutor,
)

from . import stats
from .delivery import MODELING_V2_PACKAGE
from .environment import RunEnvironment
from .gate_records import (
    KIND_ACCEPTANCE,
    KIND_FINDING,
    RESULTS_FILENAME,
    GateRecord,
    GateRecords,
    open_findings,
)
from .harness import workload_io
from .task_schema import (
    BOOT_KEYS,
    BOOT_RENDERS,
    PARALLEL_UNIFIED,
    TaskSchemaError,
    derive_family,
    load_prompts_file,
    parse_parallel_segment,
)

# Minutes-scale by design. A boot gate that needs longer than this is not a
# boot gate any more — it is an accuracy run with no anchor.
BOOT_TIMEOUT_S = 1800
ACCURACY_TIMEOUT_S = 4 * 60 * 60

MODELING_V2_ENV = "TRTLLM_MODELING_V2"
MODE_REQUIRE = "require"
MODE_OFF = "off"

#: A target records the metadata of its catalog calls and prints it at exit
#: (``=== MODELINGV2 RUNTIME VALUE DUMP``) only when this is 1, so a served
#: engine stays quiet. Every gate and witness run sets it: the auditor reads
#: those dumps against the catalog contracts.
RUNTIME_VALUE_DUMP_ENV = "TRTLLM_MODELING_V2_VALUE_DUMP"

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
ALLOW_DIRTY_ENV = "STAIRCASE_ALLOW_DIRTY_GATE"

# Seconds. A git query that takes longer than this is a sick host, and the
# gate's answer is `unknown` rather than a stall in front of a multi-hour run.
_GIT_TIMEOUT_S = 15

# ``lm-eval <task> <filter> accuracy: 90.67`` and
# ``lm-eval <task> average accuracy: 90.67`` — the two shapes
# ``tensorrt_llm/evaluate/lm_eval.py`` emits at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# ``MMLU weighted average accuracy: 74.98 (14042)`` — the MMLU path has its
# own reporting because it does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# The sample count a run actually scored, where the eval prints one: the MMLU
# line carries it in parentheses, and lm-eval's partial-score line (printed
# under `TLLM_EVAL_PARTIAL_SCORES_EVERY`) ends at `<total>/<total>`.
_MMLU_SAMPLES = re.compile(r"MMLU weighted average accuracy:\s*[0-9.]+\s*\((\d+)\)")
_PARTIAL_TOTAL = re.compile(r"Partial scores after\s+(\d+)/(\d+)\s+responses")
# `--num_samples=N` or `--num_samples N` among the benchmark arguments.
_NUM_SAMPLES_ARG = re.compile(r"^--num_samples(?:=(\d+))?$")
# ``Spec-dec stats: AL (decoded tokens/step, iteration-weighted) 3.351 (...)``
_ACCEPTANCE_LENGTH = re.compile(r"Spec-dec stats:\s*AL\b[^)]*\)\s*([0-9]+(?:\.[0-9]+)?)")
# trtllm-bench reports the same quantity under its own name.
_ACCEPTANCE_LENGTH_ALT = re.compile(r"acceptance[_ ]length\D{0,12}([0-9]+(?:\.[0-9]+)?)", re.I)
# `[0] Accept rate: 1.88 (tokens=32, iterations=17)` --- what
# `examples/llm-api/quickstart_advanced.py` prints, once per request.
#
# This one is load-bearing rather than a nicety: **nothing in this tree emits
# either of the two patterns above.** `trtllm-eval` reports no speculative
# statistic at all, so an acceptance gate built on the eval command reads
# `None` from both legs and fails every time with "could not read
# acceptance_length", which is indistinguishable from a broken draft path.
# The quantity is captured from the parenthetical rather than from the
# printed ratio so that several requests combine correctly --- see
# :func:`parse_acceptance_length`.
_ACCEPT_RATE_PARTS = re.compile(r"Accept rate:\s*[0-9.]+\s*\(tokens=(\d+),\s*iterations=(\d+)\)")

#: Tokens each acceptance prompt may generate when the task declares none.
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256
#: Generation iterations the acceptance workload must reach, summed over its
#: prompts, before the gate reads a rate from it. The degenerate case this
#: exists for measured 4: three short-answer boot prompts on DeepSeek-V4.1,
#: each ending in EOS after 2-3 tokens. A task may raise it through
#: ``accuracy.acceptance.min_draft_iterations``.
DEFAULT_MIN_DRAFT_ITERATIONS = 32

ShellRunner = Callable[[str, Mapping[str, str] | None, int], "ShellResult"]

#: The one process-wide local executor. Stateless, so sharing it is free and
#: constructing one per gate would say something about lifetime that is not
#: true.
_LOCAL_EXECUTOR = LocalCommandExecutor()


@dataclass(frozen=True)
class ShellResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def shell_result_from(record: ExecutionRecord) -> ShellResult:
    """Narrow an :class:`ExecutionRecord` to what a gate reads.

    A gate adjudicates on four things: did it finish, did it exit zero, and
    what did it print. The record carries more — which executor, how long,
    where the full log is — and that belongs in the run's evidence rather
    than in a threshold comparison, so it is deliberately dropped here
    instead of being smuggled into a gate's decision.
    """
    return ShellResult(
        returncode=record.returncode,
        stdout=record.stdout,
        stderr=record.stderr,
        timed_out=record.timed_out,
    )


def _git(repo_root: str | Path, *args: str) -> str | None:
    """Ask git one question about ``repo_root``. ``None`` if it cannot answer.

    Every failure mode collapses to ``None`` on purpose — no repository, no
    ``git`` binary, no commits yet, a hung filesystem. The callers of this
    function are gates, and a gate that dies because a VCS query died
    reports a tooling failure as a modeling failure. Called with an argv
    rather than through a shell because the only interpolated value is a
    path that may contain anything.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def describe_tree_state(repo_root: str | Path) -> str:
    """The commit a gate is about to measure, as one token.

    ``git describe --always``: where the tree carries tags it names the state
    the way a human reads a release (``v0.20.0rc0-8958-g66ebfec7e2``) and
    where it does not it falls back to the abbreviated commit, so one field
    covers both and a reader never has to know which they are looking at.

    What it buys is attribution. Gates measure whatever is on disk under
    ``trtllm_repo_path`` when the engine imports it; before this there was no
    record of what that was, so an accuracy number could be — and in this
    workflow's own run nearly was — attributed to a commit it was never
    measured on. The token is not a pin and does not pretend to be one; it is
    what lets a reader tell whether two numbers came from the same code.

    **Deliberately not ``--dirty``**, which is the obvious spelling and the
    wrong one here, for two independent reasons:

    * *Cost.* ``--dirty`` runs a whole-worktree ``diff-index`` refresh.
      Measured on this checkout, on the networked filesystem the runs use:
      ``describe --always`` 0.09 s, ``describe --always --dirty`` past 60 s.
      A gate must not sit a minute behind a VCS nicety, and with a timeout
      in front of it the field would simply always read ``unknown``.
    * *Information.* A staircase run's repo is dirty essentially always —
      onboarding records, reference YAMLs, workflow fixes — so a repo-wide
      dirty marker is set on every gate of every run and distinguishes
      nothing. The marker worth carrying is scoped to the target, and
      :func:`inspect_tree` appends that one instead, from
      :func:`dirty_target_files`.
    """
    described = _git(repo_root, "describe", "--always")
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
    ``?? .../models/qwen3_5/`` does not say it.

    An empty tuple means both "clean" and "could not tell" — a deliberate
    collapse. The guard this feeds stops a gate, and stopping a gate because
    git was unavailable would make the workflow unusable on any host without
    a checkout, which includes every host that runs these tests.

    Cheap because of the pathspec: measured at 0.14 s on this checkout
    against the minute a whole-worktree status costs here, which is the same
    asymmetry :func:`describe_tree_state` records.
    """
    status = _git(repo_root, "status", "--porcelain", "-uall", "--", target_rel)
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
    to the repo. A staircase run has onboarding records, workflow fixes and
    reference YAMLs in flight more or less constantly, and a gate that
    refused to run while any of them were uncommitted would never run at all.
    The narrow scope is what makes the check something a run can leave on.

    ``allow_dirty`` overrides the ``STAIRCASE_ALLOW_DIRTY_GATE`` environment
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
    #: that lies. ``None`` only for a caller that predates the member set.
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
    #: number is attributable to nothing: the run that produced this field had
    #: a second agent rewriting the target's ``modeling.py`` while three
    #: accuracy gates were queued against it. See :func:`describe_tree_state`.
    tree_state: str = UNKNOWN_TREE_STATE
    #: True when this verdict was measured over an uncommitted target path
    #: because the run said to. Recorded rather than merely permitted — the
    #: point of the escape hatch is that the resulting number stays marked.
    dirty_override: bool = False
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

    @property
    def threshold(self) -> float | None:
        """The number ``measured`` was held to, where the gate states one (the ledger reads it)."""
        return self.floor

    def summary(self) -> str:
        """One line for the journal, carrying the numbers verbatim.

        The member is part of the label, not an aside: `progress.yaml`, the
        journal and the reviewer's evidence all read this line, and three of
        them per unit only mean three measurements if each says which
        configuration it measured.

        The tree state rides in the same line for the same reason one level
        up: `progress.yaml` and the journal are what a reader consults later,
        and three measurements are only comparable if each says what code it
        measured.
        """
        verdict = "PASS" if self.passed else "FAIL"
        label = f"{self.name}@{self.member}" if self.member else self.name
        tree = f" [tree {self.tree_state}]"
        if self.dirty_override:
            tree = f" [tree {self.tree_state}; DIRTY TARGET PATH measured by override]"
        breakdown = f" [{self.breakdown}]" if self.breakdown else ""
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

    The keyword list starts **empty on purpose**. Staircase's rule is that
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
    """
    accuracy = task_data.get("accuracy") or {}
    block = accuracy.get("boot") if isinstance(accuracy, Mapping) else None
    if isinstance(block, Mapping):
        merged = {k: v for k, v in block.items() if k in BOOT_KEYS}
        chosen = block.get(unit) if unit is not None else None
        where = "accuracy.boot"
        if isinstance(chosen, Mapping):
            merged.update(chosen)
            where = f"accuracy.boot.{unit}"
        spec = BootSpec.from_mapping(merged, where)
    elif fallback is not None:
        spec = BootSpec.load(fallback)
    else:
        spec = BootSpec()
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


def _default_shell_runner(
    command: str,
    env: Mapping[str, str] | None = None,
    timeout: int = BOOT_TIMEOUT_S,
) -> ShellResult:
    """Run one command line through the shell, on this host.

    A thin binding of :class:`~agent_flow.runtime.LocalCommandExecutor` to
    the callable shape this module's gates take. The shell is deliberate:
    the launch prefix resolved from ``task.yaml`` *is* shell syntax (env
    assignments, or an ``srun`` line), and re-parsing it into an argv would
    mean reimplementing the shell. Nothing here interpolates agent-authored
    text — the prefix comes from the task spec and the command from this
    module.
    """
    return shell_result_from(
        _LOCAL_EXECUTOR.execute(CommandSpec(argv=command, env=env, timeout_s=timeout, label="gate"))
    )


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
    configuration, so translating the segment alone emitted ``--tp_size=N``
    and nothing else: expert parallelism and attention data parallelism —
    the two branches a unified codebase exists to carry — were switched on
    by no gate, while the target's Coverage section claimed all three. See
    :func:`~.task_schema.members_for`, which decides what the members are.

    Without a member, ``unified`` keeps its old ``--tp_size`` reading rather
    than raising: the fallback is for a caller that predates the member set,
    and the workflow always passes one.

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

    Measured, on the first ``tep4`` accuracy gate of this run::

        Error: No such option: --moe_ep_size
        (Possible options: --ep_size, --pp_size, --tp_size)

    ``tp4`` passed the same gate minutes earlier because it emits only
    ``--tp_size``, which both commands accept — so the defect was invisible
    at the one member that does not exercise it, and would have been
    invisible for any target that is not ``unified``.

    Attention data parallelism has no ``trtllm-eval`` flag, so under the
    ``eval`` dialect it is **not emitted here**; it has to arrive through
    ``--extra_llm_api_options`` as ``enable_attention_dp: true``. Emitting it
    here would reintroduce the failure this parameter exists to fix, and
    silently dropping it would serve a ``dep`` member at ``tep`` while the
    verdict claimed otherwise — so :func:`eval_command` overlays it onto the
    unit's options file instead, and that is not optional.
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
#: This run measured ``tep4`` at ~180 s against ``tp4``'s 285-438 s, which is
#: consistent with the ordering but is not what the ordering is derived from —
#: two data points on one checkpoint would not survive a different one.
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


def spool_runner(
    spool: str | Path,
    *,
    poll_s: float = 3.0,
    name_prefix: str = "gate",
) -> ShellRunner:
    """Return a :data:`ShellRunner` that executes through a compute session.

    The session is a long-lived allocation whose own main process watches
    ``<spool>/pending`` (see ``spool_watcher.sh``). Work is a
    file, which is the property that matters: **anything able to write a
    file can run GPU work**, including a plain Python process on a host
    with no GPU and no reachable scheduler.

    That is what hands gate execution back to the orchestrator. The
    alternative on such a host is to route every gate through an agent
    holding an MCP tool, and then the verdict rests on that agent's
    account of what a number was. Here it rests on a file.

    Completion is the appearance of ``done/<name>.rc``, written last and
    renamed into place by the watcher, so a reader that sees it can trust
    the log beside it is complete.

    The protocol itself lives in
    :class:`~agent_flow.runtime.SpoolCommandExecutor`: a spool is a way to
    reach an allocation, not a Staircase concept, and a run that stops using
    one should not have to touch a gate.
    """
    executor = SpoolCommandExecutor(spool, poll_s=poll_s, name_prefix=name_prefix)

    def run(
        command: str,
        env: Mapping[str, str] | None = None,
        timeout: int = BOOT_TIMEOUT_S,
    ) -> ShellResult:
        return shell_result_from(
            executor.execute(CommandSpec(argv=command, env=env, timeout_s=timeout, label="gate"))
        )

    return run


def resolve_runtime(task_data: Mapping[str, Any], unit: str | None) -> Mapping[str, Any]:
    """The ``runtime`` block for ``unit``, with the per-unit form merged.

    Accepts the flat form (the whole run) or a mapping from unit name to a
    partial block layered over ``default``. Per-unit because the units of one
    run can genuinely DISAGREE about a switch, which the other per-unit fields
    never had to handle: ``benchmark_args`` differed because the subcommands
    differ, but here `core` and `mtp` were gated with block reuse **on** and
    passed, while the multimodal path requires it **off**. A run-wide value
    cannot be right for all three.

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


def runtime_flags(runtime: Mapping[str, Any]) -> list[str]:
    """Translate the ``runtime`` block into ``quickstart_advanced.py`` flags.

    **Two of the four switches are spelled as negations on the command
    line.** ``overlap_scheduler`` and ``kv_cache_reuse`` are on by default
    in the engine, so wanting them means emitting *nothing* and not
    wanting them means emitting ``--disable_*``. That inversion is the
    whole reason this function exists: asked to "turn on block reuse", the
    obvious move is to add a flag, and the flag that exists turns it off.

    Doing it once here also keeps the gate and the prompts in agreement —
    the assembler is told what is enabled, not what was typed.
    """
    flags: list[str] = []
    if runtime.get("cuda_graph"):
        flags.append("--use_cuda_graph")
    if runtime.get("chunked_prefill"):
        flags.append("--enable_chunked_prefill")
    if not runtime.get("overlap_scheduler", True):
        flags.append("--disable_overlap_scheduler")
    if not runtime.get("kv_cache_reuse", True):
        flags.append("--disable_kv_cache_reuse")
    flags.extend(str(f) for f in runtime.get("extra_flags", ()))
    return flags


def speculation_flags(features: Any, *, max_draft_len: int = 1) -> list[str]:
    """Translate a speculative feature into its ``quickstart_advanced.py`` flags.

    Only the speculator features produce flags; everything else in the
    feature list is a property of the assembled target rather than of the
    command that runs it.

    ``dspark`` gets its own algorithm and never ``MTP``, even though a
    DeepSeek-V4.1 checkpoint stores its DSpark stages as ``mtp.*`` with
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

    **The three gates read the speculator's depth from two different places,
    and before this they disagreed.** `trtllm-eval` takes no speculation
    flags, so the accuracy gate learns the depth from the unit's
    `--extra_llm_api_options` YAML -- `speculative_config.num_nextn_predict_layers`.
    The boot and acceptance gates drive `quickstart_advanced.py`, which cannot
    read that file, so they take it as `--spec_decode_max_draft_len` and it
    was hard-defaulted to 1.

    Measured on this run's `mtp` unit: the options file says 3 and the flag
    said 1, so accuracy would have measured MTP-3 while acceptance measured
    MTP-1 against a ceiling of 2.0 -- and the target's real acceptance is
    2.44-2.59 against a ceiling of 4.0. The acceptance gate exists precisely
    because boot and accuracy are blind to a miscomputed draft path; an
    acceptance gate measuring a *different depth* than the unit was built for
    is blind in a new way, and it fails toward "green" because a shallower
    speculator accepts a larger fraction of what it drafts.

    So the YAML is the single source of truth and the flag is derived from
    it. A unit with no options file, or none declaring a speculator, keeps
    the old answer of 1.

    Read in the order the engine resolves it. ``max_draft_len`` is the field;
    ``num_nextn_predict_layers`` is its deprecated MTP alias, which
    ``MTPDecodingConfig`` remaps and drops when both are present; ``block_size``
    is how a DSpark drafter states its depth, and ``TorchLlmArgs`` rejects any
    DSpark config whose ``block_size`` differs from ``max_draft_len``. For
    DSpark this is not a knob at all: quickstart builds the config from the
    flag alone, so the old default of 1 against a drafter trained for a
    5-token block fails that validation before the engine starts.
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


def resolve_per_unit(accuracy: Mapping[str, Any], key: str, unit: str | None) -> Any:
    """Read ``accuracy[key]``, honouring the per-unit mapping form.

    Several fields of the accuracy block are legitimately different for a
    feature increment than for core, and the mapping form is how a spec says
    so. The scalar form still means "the whole run", so nothing that already
    works changes shape.

    ``benchmark`` and ``reference_yaml`` are the load-bearing case: the
    ``multimodal`` unit's release criterion is a multimodal benchmark, and
    gating it on the text benchmark core was gated on is not a weaker check
    — it is a check that cannot see the vision tower at all, so it passes
    whatever the vision tower does.
    """
    value = accuracy.get(key)
    if isinstance(value, dict):
        chosen = value.get(unit)
        if chosen is None:
            chosen = value.get("default")
        return chosen
    return value


def resolve_llm_api_options(accuracy: Mapping[str, Any], unit: str | None) -> str | None:
    """Pick the runtime-configuration YAML for ``unit``.

    ``llm_api_options`` may be a single path (the whole run) or a mapping
    from unit name to path. The mapping form exists because **a feature unit
    is not measured under the same configuration as core**: MTP needs a
    ``speculative_config``, and without one the accuracy gate silently
    re-measures the base path while the acceptance gate compares 1.0 against
    1.0 and calls it a pass.

    That was live in this run: neither gate carried a speculative config at
    all, so the `mtp` unit's gates were measuring `core`.
    """
    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        return options.get(unit) or options.get("default")
    return options


def resolve_discriminators(
    accuracy: Mapping[str, Any], unit: str | None
) -> Mapping[str, Any] | None:
    """Pick the accuracy-reference discriminators for ``unit``.

    TensorRT-LLM's accuracy references key each model to a LIST of entries
    discriminated by ``quant_algo`` / ``kv_cache_quant_algo`` /
    ``spec_dec_algo``, and :func:`load_reference_score` matches a row only
    when the discriminator set is EXACTLY the one asked for. Passing none --
    which is what this workflow did until now -- therefore means one of two
    things, both wrong:

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
) -> str:
    """Merge a member's own LLM-API settings over its unit's options file.

    Written beside the unit's file so the two are read together and a reader
    finding the derived one can see what it was derived from. Deterministic
    name, so a re-run overwrites rather than accumulating, and the file a
    verdict was measured under is still on disk afterwards.
    """
    base: dict[str, Any] = {}
    source = Path(options) if options else None
    if source is not None and source.is_file():
        base = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    merged = {**base, **overlay}
    stem = f"{unit or 'unit'}.{member or 'member'}.llm_api.yaml"
    # Beside the unit's own options file, so the derived copy and the file it
    # was derived from are read together. With no unit file there is nowhere
    # natural, and the cwd is the one place it must NOT go: the cwd of a test
    # run is the repository, and this wrote `core.dep1.llm_api.yaml` into it.
    # A generated artifact that lands in the source tree is indistinguishable
    # from a file somebody meant to check in.
    parent = source.parent if source is not None else Path(tempfile.gettempdir())
    target = parent / stem
    header = (
        f"# GENERATED for the `{member}` member of the `{unit}` unit. Do not edit.\n"
        f"# {source if source is not None else '(no unit options file)'} plus the\n"
        f"# settings that member's gate command cannot express as flags:\n"
        f"#   {', '.join(sorted(overlay))}\n"
        f"# See `gates.member_llm_api_overlay` for why each one is here.\n"
    )
    target.write_text(header + yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
    return str(target)


def eval_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    benchmark: str,
    *,
    unit: str | None = None,
    member: str | None = None,
) -> list[str]:
    """Build the ``trtllm-eval`` argv shared by the accuracy and acceptance gates.

    **`trtllm-eval` and `quickstart_advanced.py` take different flag
    vocabularies for the same runtime configuration.** There is no
    `--use_cuda_graph`, no `--kv_cache_dtype`, no `--mamba_ssm_cache_dtype`
    here; that configuration has to arrive as an `--extra_llm_api_options`
    YAML instead. So the `runtime` block cannot simply be forwarded — and
    before this existed it was not forwarded at all.

    The consequence was not a weak comparison but a different question: the
    accuracy gate measured the engine's defaults (including
    `max_seq_len=262144`, the checkpoint's own maximum) while the boot gate
    measured the requested configuration and the anchor had been measured at
    a third thing again. Caught on the first real accuracy gate of this run,
    by reading the command the gate had actually issued.

    The fix is to declare the configuration once, as a file, and have the
    anchor run and both gates point at the same one.
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
    mode = environment.env.get(MODELING_V2_ENV)
    if mode:
        # Target selection is the LLM-API argument `modeling_v2`; nothing in
        # the engine reads the environment switch any more. Without this the
        # gate serves the built-in model (the argument defaults to "off") and
        # its verdict reads as the target's.
        overlay = {**overlay, "modeling_v2": mode}
    if overlay:
        # `trtllm-eval` has no attention-DP flag, so a `dep` member can only
        # be served through the options YAML. Writing a merged copy rather
        # than editing the unit's file keeps one source of truth: the unit
        # states its runtime configuration once, and the member adds only
        # what it alone needs. A `dep` gate run without this serves `tep`
        # while the verdict says `dep`, which is the failure the member set
        # exists to prevent.
        options = _write_member_options(options, overlay, unit=unit, member=member)
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
    # This is the difference the run measured: the same target scored 63.50
    # under the default 5-shot-completion protocol and the anchor was 95.53
    # under the chat-template protocol. Neither number is wrong; they answer
    # different questions, and a gate that compares them reports a 27-point
    # regression that does not exist.
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
    """
    from dataclasses import replace

    return replace(
        environment,
        env={
            **environment.env,
            MODELING_V2_ENV: mode,
            RUNTIME_VALUE_DUMP_ENV: "1",
            STEP_CONTRACT_ENV: "1",
        },
    )


# ------------------------------------------------------------ the engine driver
#
# The gates that need per-request structure -- token ids, whole texts,
# iterations, images -- run `harness/engine_driver.py` under the tree's own
# interpreter. It builds the engine with the tree's own quickstart from the
# same argv the quickstart gates use, so the configuration measured is
# identical, and prints one `STAIRCASE_RECORD` JSON line per request. A gate
# reads those lines and nothing else of the log.

#: The driver script, run as a file under the measured tree's interpreter.
ENGINE_DRIVER = Path(__file__).resolve().parent / "harness" / "engine_driver.py"

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
    block, the speculator at the unit's draft depth -- with each runtime
    ``extra_flags`` entry split as the shell would split it.
    ``runtime_overrides`` replaces switches of the unit's runtime block (a
    leg that turns the overlap scheduler off, for one).
    """
    runtime = {**resolve_runtime(task_data, unit), **dict(runtime_overrides or {})}
    flags = with_max_batch_size(
        [
            *runtime_flags(runtime),
            *speculation_flags(
                task_data.get("features") if features is None else features,
                max_draft_len=resolve_max_draft_len(task_data.get("accuracy") or {}, unit),
            ),
            *extra,
        ],
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


def gate_workdir(
    workdir: str | Path | None, name: str, task_data: Mapping[str, Any] | None = None
) -> Path:
    """Where a gate writes the files its job reads.

    ``workdir`` when given: the orchestrator passes a directory on the shared
    filesystem, because the job may run on another host (a spool session).
    Otherwise a directory under the temp dir named for the gate and the task
    (its tree and checkpoint), so two runs of one gate on one task write the
    same files and their commands compare equal; a fresh one without a task.
    """
    if workdir is not None:
        directory = Path(workdir)
    elif task_data is not None:
        key = f"{task_data.get('trtllm_repo_path')}|{task_data.get('checkpoint_path')}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        directory = Path(tempfile.gettempdir()) / f"staircase-{name}-{digest}"
    else:
        return Path(tempfile.mkdtemp(prefix=f"staircase-{name}-"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
) -> Path:
    """Write one engine-driver spec (see ``harness/engine_driver.py``)."""
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
    argv: list[str] = []
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
    """Every ``STAIRCASE_RECORD`` the engine driver printed, in request order."""
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
    this gate cannot read the per-unit options file the other two gates do, and
    the runtime block is its only channel for a unit-specific switch.

    ``features`` is **the features the target under test actually
    implements**, which is not the same as the features the *run* will
    finish with. Defaulting it to ``task_data["features"]`` means the
    ``core`` unit is gated with the speculator's flags before a speculator
    exists. Staircase's rule is that core is green *before* the first feature
    increment; the gate has to agree with it.

    ``spec`` is the workload (see :func:`resolve_boot_spec`): text prompts,
    and for a multimodal unit at least one prompt that sends an image. The
    engine driver builds the engine exactly as ``quickstart_advanced.py``
    does from the same flags and prints one record per prompt; keywords are
    matched against those generated texts only, never against the log, so
    ``72`` cannot pass on ``CPUs 72-143``.

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
    env = _with_mode(environment, mode)
    # A control run must not be recorded under the same name as the real
    # gate. Two rows both called "boot", one of which measured the stock
    # implementation, is a journal that reads as evidence for a target it
    # never exercised.
    gate_name = "boot" if mode == MODE_REQUIRE else f"boot[{mode}]"
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=gate_name, member=member, passed=False, detail=state.block_detail())
    requests = spec.requests(task_data.get("trtllm_repo_path"))
    argv = quickstart_words(
        task_data,
        environment,
        unit=unit,
        features=task_data.get("features") if features is None else features,
        max_tokens=spec.max_tokens,
        member=member,
    )
    spec_path = write_driver_spec(
        gate_workdir(workdir, "boot", task_data)
        / f"boot.{unit or 'unit'}.{member or 'member'}.json",
        task_data=task_data,
        argv=argv,
        requests=requests,
        max_tokens=spec.max_tokens,
    )
    command = engine_driver_command(env, spec_path)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(name=gate_name, member=member, command=command, duration_s=elapsed, stdout_tail=tail)

    if result.timed_out:
        return _verdict(
            passed=False,
            detail=f"timed out after {timeout}s — a boot gate that takes this long is "
            f"not catching catastrophes any more",
            **ran,
        )
    if not result.ok:
        return _verdict(passed=False, detail=f"exited {result.returncode}", **ran)

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
    empty = [requests[n]["id"] for n, text in enumerate(texts) if not (text or "").strip()]
    if empty:
        return _verdict(
            passed=False,
            detail=f"the engine exited cleanly but generated nothing for {', '.join(empty)}",
            **ran,
        )
    generated = "\n".join(text or "" for text in texts).lower()
    missing = [k for k in spec.keywords if k.lower() not in generated]
    item_texts = texts[len(spec.prompts) :]
    for item, text in zip(spec.items, item_texts):
        for absent_keyword in item.missing(text or ""):
            label = "the image prompt" if item.images else "a prompt"
            missing.append(f"{absent_keyword} (in {label} {item.prompt[:40]!r})")
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
        detail=f"engine cold-started and produced {checked} for {len(requests)} prompt(s)"
        f"{sent}; workload: {spec.source}",
        **ran,
    )


def resolve_samples(
    accuracy: Mapping[str, Any],
    unit: str | None,
    benchmark: str,
    output: str = "",
) -> tuple[int | None, str]:
    """How many samples the accuracy score was (or will be) measured on, and from where.

    Most specific first: the count the eval itself printed (what was actually
    scored), then ``--num_samples`` among the unit's benchmark arguments, then
    ``accuracy.samples`` (scalar or per unit), then
    :data:`.stats.KNOWN_BENCHMARK_SAMPLES`. ``(None, reason)`` when none of
    them answers; the statistical floor cannot be computed then, and the
    schema refuses such a task before a run starts.
    """
    if output:
        mmlu = _MMLU_SAMPLES.findall(output)
        if mmlu:
            return int(mmlu[-1]), "the eval's own output"
        partial = _PARTIAL_TOTAL.findall(output)
        if partial:
            return int(partial[-1][1]), "the eval's own output"
    args = [str(a) for a in resolve_per_unit(accuracy, "benchmark_args", unit) or ()]
    for i, arg in enumerate(args):
        matched = _NUM_SAMPLES_ARG.match(arg)
        if not matched:
            continue
        value = matched.group(1) or (args[i + 1] if i + 1 < len(args) else None)
        try:
            return int(str(value)), "--num_samples in benchmark_args"
        except (TypeError, ValueError):
            break
    declared = resolve_per_unit(accuracy, "samples", unit)
    if declared is not None:
        try:
            return int(declared), "accuracy.samples"
        except (TypeError, ValueError):
            pass
    known = stats.KNOWN_BENCHMARK_SAMPLES.get(benchmark)
    if known is not None:
        return known, f"the {benchmark} split size"
    return None, f"no sample count for `{benchmark}`: declare `accuracy.samples`"


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

    This is the gate the tree-state guard was written for. It is the
    expensive one — a quarter of an hour per member — and it is the one whose
    output becomes the target's headline number, so it is also the one where
    measuring an unreviewed tree is least recoverable. See
    :func:`inspect_tree` and ``allow_dirty``.
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

    parts = eval_command(task_data, environment, benchmark, unit=unit, member=member)
    command = env.command(" ".join(parts))

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(command=command, duration_s=elapsed, stdout_tail=tail, **common)

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
#: four prompts a change that only moved the generated text moved the mean by
#: +0.24 / -0.18 / -0.02 across three members; on 32 GSM8K prompts the same
#: change measured -0.06 / +0.05 at paired t -1.4 / +0.8 -- no change. A task
#: may lower it through ``accuracy.acceptance.min_prompts``, and the verdict
#: then rests on fewer pairs, visibly.
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
            f"from the ceiling passed every wrong drafter of the run that had one"
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
    answer = _git(repo_root, "rev-parse", "HEAD")
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
      the same session. It fails when the weighted ratio is under
      ``min_ratio`` AND the per-prompt paired test finds the drop
      significant. A new architecture has no stock path; then the declared
      floor (see :func:`resolve_acceptance_floor_spec`) is used instead, and
      the verdict warns when that floor is not tied to a measured reference.
    * against 1.0: a target at or under it rejects every draft.
    * against the previous verdict for the same member, workload and
      speculator in ``records``: a paired test over the prompts both
      measured (``accuracy.acceptance.regression``: t or Wilcoxon, alpha,
      minimum effect). A significant drop at least the minimum effect large
      fails. This replaces the fixed 0.1 delta that read noise on four
      prompts as a regression.

    Every verdict with a reading carries the per-prompt rates. A passing
    reading is appended to ``records`` and becomes the next tree's baseline.

    The two legs differ in exactly one variable and must run in the same
    session; both import the same working tree, so the tree-state guard of
    :func:`inspect_tree` stops a gate whose target path is being edited.
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
    directory = gate_workdir(workdir, "acceptance", task_data)
    spec_path = write_driver_spec(
        directory / f"acceptance.{unit or 'unit'}.{member or 'member'}.json",
        task_data=task_data,
        argv=argv,
        requests=workload.requests(),
        dataset=workload.dataset,
        max_tokens=workload.max_tokens,
    )
    policy = resolve_regression_policy(accuracy)
    floor_spec = resolve_acceptance_floor_spec(accuracy, unit, min_ratio=min_ratio)

    started = time.monotonic()
    readings: dict[str, AcceptanceReading | None] = {}
    commands: dict[str, str] = {}
    stock_unavailable = False
    for mode in (MODE_REQUIRE, MODE_OFF):
        command = engine_driver_command(_with_mode(environment, mode), spec_path)
        commands[mode] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        if not result.ok:
            # **The stock leg can be impossible rather than broken.** A
            # genuinely new architecture HAS no built-in modeling, so `=off`
            # cannot load it at all -- the check is unavailable exactly when
            # onboarding is doing its job (see
            # `preflight.UNSUPPORTED_ARCHITECTURE_MARKERS`, reused here). The
            # spec may then declare an absolute floor instead, and the weaker
            # claim is recorded verbatim.
            from .preflight import reports_unsupported_architecture

            unavailable = mode == MODE_OFF and reports_unsupported_architecture(
                f"exited {result.returncode}", "\n".join(combined.splitlines()[-200:])
            )
            if unavailable and floor_spec is not None:
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
                        "stock leg cannot run. Declare `accuracy.acceptance.<unit>."
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
        readings[mode] = parse_acceptance(combined)
    elapsed = time.monotonic() - started

    target_reading = readings.get(MODE_REQUIRE)
    stock_reading = readings.get(MODE_OFF)
    target = target_reading.value if target_reading is not None else None
    stock = stock_reading.value if stock_reading is not None else None
    breakdown = target_reading.breakdown() if target_reading is not None else ""
    base = dict(name="acceptance", member=member, duration_s=elapsed)

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
                f"{workload.min_prompts} a paired comparison needs -- on four prompts a "
                f"change that only moved the generated text read as a regression"
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
    if stock_unavailable:
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
            f"built-in implementation. THIS IS THE WEAKER CLAIM -- it shows the draft path "
            f"accepts at a real rate, not that it matches stock"
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
        warnings=tuple(warnings),
        explained=True,
        **base,
    )


@dataclass
class GateLog:
    """Every gate a run has fired, in order. The journal's raw material."""

    results: list[GateResult] = field(default_factory=list)
    #: Called with every result as it is recorded. The workflow points it at
    #: the evidence ledger (``ledger.record_gate_result``), so every verdict a
    #: run produces enters the ledger here and nowhere else.
    sink: Callable[[GateResult], Any] | None = None

    def record(self, result: GateResult) -> GateResult:
        self.results.append(result)
        if self.sink is not None:
            self.sink(result)
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
    "ACCURACY_TIMEOUT_S",
    "ALLOW_DIRTY_ENV",
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
    "resolve_samples",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "GateLog",
    "GateResult",
    "MODELING_V2_ENV",
    "MODE_OFF",
    "MODE_REQUIRE",
    "RUNTIME_VALUE_DUMP_ENV",
    "STEP_CONTRACT_ENV",
    "ShellResult",
    "ShellRunner",
    "TreeState",
    "UNKNOWN_TREE_STATE",
    "describe_tree_state",
    "dirty_target_files",
    "inspect_tree",
    "load_reference_score",
    "order_members_by_cost",
    "parse_acceptance",
    "parse_acceptance_length",
    "parse_eval_score",
    "resolve_acceptance_floor",
    "resolve_acceptance_workload",
    "resolve_discriminators",
    "resolve_per_unit",
    "resolve_runtime",
    "resolve_llm_api_options",
    "resolve_reference",
    "run_acceptance_gate",
    "run_accuracy_gate",
    "run_boot_gate",
    "shell_result_from",
    "spool_runner",
    "target_family_rel",
    "topology_flags",
]

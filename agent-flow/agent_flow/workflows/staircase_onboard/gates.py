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
   ``measured >= reference - tolerance``. Runs at a unit boundary.
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

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from agent_flow.runtime import (
    CommandSpec,
    ExecutionRecord,
    LocalCommandExecutor,
    SpoolCommandExecutor,
)

from .delivery import MODELING_V2_PACKAGE
from .environment import RunEnvironment
from .task_schema import (
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
        if self.measured is not None and self.reference is not None:
            margin = self.measured - (self.reference - (self.tolerance or 0.0))
            return (
                f"[{verdict}] {label}: measured {self.measured:.4f} vs "
                f"reference {self.reference:.4f} - tol {self.tolerance or 0.0:.2f} "
                f"(margin {margin:+.4f}){breakdown} in {self.duration_s:.0f}s{tree}"
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


@dataclass(frozen=True)
class BootSpec:
    """What the boot gate sends and what it insists on seeing back.

    The keyword list starts **empty on purpose**. Staircase's rule is that
    the first engine run doubles as keyword verification — you freeze the
    cases only after observing the continuations — so an assembler that has
    not yet seen this model generate anything has no business asserting
    what it will say. Until then the gate asserts the weaker, still useful
    thing: the engine cold-starts and produces non-empty output.
    """

    prompts: tuple[str, ...] = (
        "The capital of France is",
        "In one sentence, photosynthesis is",
    )
    keywords: tuple[str, ...] = ()
    max_tokens: int = 32

    @classmethod
    def load(cls, path: str | Path) -> "BootSpec":
        source = Path(path)
        if not source.is_file():
            return cls()
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{source} must hold a YAML mapping, got {type(data).__name__}")
        prompts = tuple(str(p) for p in data.get("prompts") or cls.prompts)
        return cls(
            prompts=prompts or cls.prompts,
            keywords=tuple(str(k) for k in data.get("keywords") or ()),
            max_tokens=int(data.get("max_tokens", cls.max_tokens)),
        )


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
        env={**environment.env, MODELING_V2_ENV: mode, RUNTIME_VALUE_DUMP_ENV: "1"},
    )


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
    """Extract the acceptance measurement from an eval, bench or quickstart run.

    The quickstart form is **iteration-weighted across requests**: the
    quantity is tokens-per-iteration, so several requests combine as
    ``sum(tokens) / sum(iterations)`` and not as the mean of their printed
    ratios. Averaging the ratios would let one short request with an
    unlucky tail dominate a long one, and the whole judgement here is a
    ratio between two runs. The plain per-prompt mean is still reported, in
    :meth:`AcceptanceReading.breakdown`, because it is the number that shows
    a prompt that never drafted.
    """
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
) -> GateResult:
    """Cold-start the engine and check it generates.

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
    finish with. Defaulting it to ``task_data["features"]`` — as this
    function originally did with no way to override — means the ``core``
    unit is gated with the speculator's flags before a speculator exists.

    Measured: the core target refused `--spec_decode_algo=MTP` through its
    own fail-fast assert, correctly, and the assembler escalated rather than
    deleting the assert to make the gate pass. Deleting it would have been
    the silent wrong — one-engine MTP is drafted by the model, so a base
    model under that config emits no drafts and the run would have gated a
    speculator that was not there.

    Staircase's rule is that core is green *before* the first feature
    increment; the gate has to agree with it.

    ``mode`` defaults to ``require`` and should stay there for any result
    that will be attributed to modeling_v2. It is a parameter for one
    reason: when a boot gate fails, the next question is always *"is this
    the target or is this the host?"*, and the cheapest answer is the same
    command under ``off`` — the stock implementation, same prompts, same
    runtime flags, one variable. Without this the control has to be
    reconstructed by hand, which is how a run ends up debugging a target
    that was never at fault.

    ``run_accuracy_gate`` already took a ``mode`` for the same reason; the
    boot gate not having one was an inconsistency, not a decision.

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
    repo = Path(str(task_data["trtllm_repo_path"]))
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    prompts = " ".join(f'"{p}"' for p in spec.prompts)
    argv = " ".join(
        [
            str(script),
            f'--model_dir="{task_data["checkpoint_path"]}"',
            f"--max_tokens={spec.max_tokens}",
            *topology_flags(str(task_data["parallel"]), environment.world_size, member),
            *runtime_flags(resolve_runtime(task_data, unit)),
            *speculation_flags(
                task_data.get("features") if features is None else features,
                max_draft_len=resolve_max_draft_len(task_data.get("accuracy") or {}, unit),
            ),
            f"--prompt {prompts}",
        ]
    )
    command = env.python_command(argv)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])

    if result.timed_out:
        return _verdict(
            name=gate_name,
            member=member,
            passed=False,
            detail=f"timed out after {timeout}s — a boot gate that takes this long is "
            f"not catching catastrophes any more",
            command=command,
            duration_s=elapsed,
            stdout_tail=tail,
        )
    if not result.ok:
        return _verdict(
            name=gate_name,
            member=member,
            passed=False,
            detail=f"exited {result.returncode}",
            command=command,
            duration_s=elapsed,
            stdout_tail=tail,
        )

    missing = [k for k in spec.keywords if k not in result.stdout]
    if missing:
        return _verdict(
            name=gate_name,
            member=member,
            passed=False,
            detail=f"generated output is missing {len(missing)} expected keyword(s): {missing}",
            command=command,
            duration_s=elapsed,
            stdout_tail=tail,
        )
    if not result.stdout.strip():
        return _verdict(
            name=gate_name,
            member=member,
            passed=False,
            detail="the engine exited cleanly but produced no output at all",
            command=command,
            duration_s=elapsed,
            stdout_tail=tail,
        )
    checked = f"{len(spec.keywords)} keyword(s)" if spec.keywords else "non-empty generation"
    return _verdict(
        name=gate_name,
        member=member,
        passed=True,
        detail=f"engine cold-started and produced {checked}",
        command=command,
        duration_s=elapsed,
        stdout_tail=tail,
    )


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
    if state.blocked:
        return _verdict(
            name=f"accuracy[{benchmark}]",
            member=member,
            passed=False,
            detail=state.block_detail(),
            reference=reference,
            tolerance=tolerance,
        )

    parts = eval_command(task_data, environment, benchmark, unit=unit, member=member)
    command = env.command(" ".join(parts))

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])

    if result.timed_out:
        return _verdict(
            name=f"accuracy[{benchmark}]",
            member=member,
            passed=False,
            detail=f"timed out after {timeout}s",
            command=command,
            reference=reference,
            tolerance=tolerance,
            duration_s=elapsed,
            stdout_tail=tail,
        )
    if not result.ok:
        return _verdict(
            name=f"accuracy[{benchmark}]",
            member=member,
            passed=False,
            detail=f"trtllm-eval exited {result.returncode}",
            command=command,
            reference=reference,
            tolerance=tolerance,
            duration_s=elapsed,
            stdout_tail=tail,
        )

    measured = parse_eval_score(combined)
    if measured is None:
        return _verdict(
            name=f"accuracy[{benchmark}]",
            member=member,
            passed=False,
            detail="trtllm-eval exited 0 but printed no score this module recognizes; "
            "a gate with no number is not a pass",
            command=command,
            reference=reference,
            tolerance=tolerance,
            duration_s=elapsed,
            stdout_tail=tail,
        )

    floor = reference - tolerance
    passed = measured >= floor
    return _verdict(
        name=f"accuracy[{benchmark}]",
        member=member,
        passed=passed,
        detail=(
            f"measured {measured:.4f} {'>=' if passed else '<'} {floor:.4f} "
            f"(reference {reference:.4f} - tol {tolerance:.2f})"
        ),
        command=command,
        measured=measured,
        reference=reference,
        tolerance=tolerance,
        duration_s=elapsed,
        stdout_tail=tail,
    )


def resolve_acceptance_floor(accuracy: Mapping[str, Any], unit: str | None) -> float | None:
    """The absolute `acceptance_length` floor this unit declares, if any.

    Only consulted when the `=off` leg **cannot exist**. See
    :func:`run_acceptance_gate`.
    """
    block = accuracy.get("acceptance")
    if not isinstance(block, dict):
        return None
    chosen = block.get(unit)
    if chosen is None:
        chosen = block.get("default")
    if isinstance(chosen, dict):
        chosen = chosen.get("min_absolute")
    try:
        return float(chosen)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AcceptanceWorkload:
    """What the acceptance gate sends, and how much work it must get back."""

    prompts: tuple[str, ...]
    max_tokens: int
    min_draft_iterations: int
    #: Where the prompts came from, named in a verdict that blames them.
    source: str


def resolve_acceptance_workload(
    accuracy: Mapping[str, Any],
    unit: str | None,
    *,
    fallback: BootSpec | None = None,
    max_tokens: int | None = None,
) -> AcceptanceWorkload:
    """The acceptance workload ``unit`` declares, or the fallback when none does.

    Read from ``accuracy.acceptance``, most specific level first: the unit's
    own entry, then its ``default`` entry, then the block's top level (see
    ``task_schema.ACCEPTANCE_WORKLOAD_KEYS``). ``prompts`` and
    ``prompts_file`` are one setting, so the level that states either wins
    for both. ``fallback`` (the boot spec) and ``max_tokens`` are what a task
    that declares nothing measures, which is what every task measured before
    the workload could be declared.

    Raises :class:`~.task_schema.TaskSchemaError` when a declared prompts
    file cannot be read.
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

    prompts: tuple[str, ...] | None = None
    source = ""
    for where, entry in levels:
        if entry.get("prompts") is not None:
            prompts = tuple(str(p) for p in entry["prompts"])
            source = f"{where}.prompts"
            break
        if entry.get("prompts_file") is not None:
            prompts = load_prompts_file(entry["prompts_file"])
            source = f"{where}.prompts_file ({entry['prompts_file']})"
            break
    if prompts is None:
        prompts = (fallback or BootSpec()).prompts
        source = "the boot prompts (no `accuracy.acceptance.prompts` declared)"
    declared_tokens = setting("max_tokens")
    declared_minimum = setting("min_draft_iterations")
    if declared_tokens is None:
        declared_tokens = max_tokens if max_tokens is not None else DEFAULT_ACCEPTANCE_MAX_TOKENS
    if declared_minimum is None:
        declared_minimum = DEFAULT_MIN_DRAFT_ITERATIONS
    return AcceptanceWorkload(
        prompts=prompts,
        max_tokens=int(declared_tokens),
        min_draft_iterations=int(declared_minimum),
        source=source,
    )


def run_acceptance_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    min_ratio: float = 0.95,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
) -> GateResult:
    """Compare ``acceptance_length`` against stock in-tree modeling.

    The two runs differ in exactly one variable — ``TRTLLM_MODELING_V2``
    ``off`` versus ``require`` — and must happen **in the same session**.
    A cross-session delta carries session variance into the judgement, and
    staircase measured a bit-identical forward at 94.7688 and 95.0720 a day
    apart.

    ``member`` fixes the topology **both** legs run at, so the one variable
    stays one: a draft path whose acceptance depends on the expert-parallel
    slice is a thing this gate can only see when it is asked at that member.

    The tree-state guard matters here for a reason the other two gates do not
    have: **both legs import the same working tree**, so an edit landing
    between them moves the ratio without moving either leg's correctness, and
    the gate reports it as a draft-path defect. See :func:`inspect_tree`.

    **The workload is the task's to declare** (``accuracy.acceptance.prompts``
    or ``prompts_file``, ``max_tokens``; see
    :func:`resolve_acceptance_workload`). ``spec`` and ``max_tokens`` here are
    only the fallback for a task that declares none. The boot prompts are
    the wrong workload for a model whose answers to them are short:
    DeepSeek-V4.1 answers them in 2-3 tokens and EOS, so every request is a
    context step plus one generation step, the rate reads exactly 1.00 for
    ANY drafter, and a declared floor above 1 can never pass while a broken
    drafter cannot be told from a working one. On long-answer prompts the
    same command measured 1.91 for a drafter with a broken ring index and
    3.24 after the fix.

    **A workload too short to measure is a failure of its own**, reported as
    one: fewer than ``min_draft_iterations`` generation iterations in total
    (from quickstart's ``Accept rate: ... (tokens=T, iterations=I)`` lines)
    fails with "workload too short to measure acceptance" instead of a low
    number that reads as a drafter defect. A leg that prints only a summary
    statistic carries no iteration count and is not held to it.

    Every verdict that has a reading carries the per-prompt rates and their
    plain mean beside the iteration-weighted value it is judged on, in
    :attr:`GateResult.breakdown`.
    """
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name="acceptance", member=member, passed=False, detail=state.block_detail())
    accuracy = task_data.get("accuracy") or {}
    try:
        workload = resolve_acceptance_workload(
            accuracy, unit, fallback=spec or BootSpec(), max_tokens=max_tokens
        )
    except TaskSchemaError as exc:
        return _verdict(
            name="acceptance",
            member=member,
            passed=False,
            detail=f"the declared acceptance workload cannot be read: {exc}",
        )
    # The QUICKSTART command, not the eval command, and not as an
    # optimization. `trtllm-eval` emits no speculative statistic in this
    # tree, so an acceptance gate built on it reads None from both legs and
    # fails identically to a broken draft path. `quickstart_advanced.py`
    # prints `Accept rate: N (tokens=T, iterations=I)` per request, which is
    # the quantity this gate is about.
    #
    # It is also two orders of magnitude cheaper -- minutes against the nine
    # a full gsm8k run costs, twice -- which matters because this gate runs
    # both legs.
    repo = Path(str(task_data["trtllm_repo_path"]))
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    # Shell-quoted: a declared workload is free text, and a code-listing
    # prompt carries the quotes, `$` and backslashes a double-quoted word
    # would expand or end on.
    prompts = " ".join(shlex.quote(p) for p in workload.prompts)
    argv = " ".join(
        [
            str(script),
            f'--model_dir="{task_data["checkpoint_path"]}"',
            # Longer than the boot gate's, deliberately. Acceptance length is
            # a rate, and a 32-token continuation gives it a handful of
            # iterations to be estimated from -- enough to see a draft path
            # collapse to 1.0, not enough to resolve a 5% shortfall.
            f"--max_tokens={workload.max_tokens}",
            *topology_flags(str(task_data["parallel"]), environment.world_size, member),
            *runtime_flags(resolve_runtime(task_data, unit)),
            *speculation_flags(
                task_data.get("features") if features is None else features,
                max_draft_len=resolve_max_draft_len(accuracy, unit),
            ),
            f"--prompt {prompts}",
        ]
    )

    started = time.monotonic()
    readings: dict[str, AcceptanceReading | None] = {}
    commands: dict[str, str] = {}
    stock_unavailable = False
    for mode in (MODE_REQUIRE, MODE_OFF):
        command = _with_mode(environment, mode).python_command(argv)
        commands[mode] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        if not result.ok:
            # **The stock leg can be impossible rather than broken.** This gate
            # compares the target against the same checkpoint under built-in
            # modeling -- and a genuinely new architecture HAS no built-in
            # modeling, so `=off` cannot load it at all. That is the same shape
            # as the substrate gate's problem (see
            # `preflight.UNSUPPORTED_ARCHITECTURE_MARKERS`, reused here): the
            # check is unavailable exactly when onboarding is doing its job.
            #
            # Failing the unit for it would be wrong, and silently skipping the
            # gate would be worse -- it is the ONLY gate that can see a
            # miscomputed draft path, because rejection sampling makes a wrong
            # draft produce correct text more slowly. So the spec may declare an
            # ABSOLUTE floor instead, and the weaker claim is recorded verbatim.
            from .preflight import reports_unsupported_architecture

            floor = resolve_acceptance_floor(accuracy, unit)
            unavailable = mode == MODE_OFF and reports_unsupported_architecture(
                f"exited {result.returncode}", "\n".join(combined.splitlines()[-200:])
            )
            if unavailable and floor is not None:
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
                        "min_absolute` to gate on an absolute floor instead."
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

    def too_short(mode: str) -> GateResult | None:
        """The verdict for a leg whose workload never exercised the drafter."""
        reading = readings.get(mode)
        counted = reading.draft_iterations if reading is not None else None
        if reading is None or counted is None or counted >= workload.min_draft_iterations:
            return None
        return _verdict(
            name="acceptance",
            member=member,
            passed=False,
            detail=(
                f"workload too short to measure acceptance: the {MODELING_V2_ENV}={mode} "
                f"leg ran {counted} generation iteration(s) over {len(reading.requests)} "
                f"prompt(s), fewer than the {workload.min_draft_iterations} this gate "
                f"needs, so its rate ({reading.value:.2f}) says nothing about the draft "
                f"path -- the answers end before drafts are verified. Workload: "
                f"{workload.source}, max_tokens {workload.max_tokens}. Declare "
                f"long-answer prompts in `accuracy.acceptance.prompts` (or "
                f"`prompts_file`); this is a property of the workload, not of the target."
            ),
            command=commands[mode],
            duration_s=elapsed,
            breakdown=reading.breakdown(),
        )

    if stock_unavailable:
        floor = resolve_acceptance_floor(accuracy, unit)
        if target is None:
            return _verdict(
                name="acceptance",
                member=member,
                passed=False,
                detail="could not read acceptance_length from the require leg",
                command=commands[MODE_REQUIRE],
                duration_s=elapsed,
            )
        short = too_short(MODE_REQUIRE)
        if short is not None:
            return short
        return _verdict(
            name="acceptance",
            member=member,
            passed=target >= float(floor),
            detail=(
                f"acceptance_length {target:.4f} vs declared floor {float(floor):.4f}. "
                f"The {MODELING_V2_ENV}=off leg could not run: this checkpoint has no "
                f"built-in implementation, so there is no stock acceptance to compare "
                f"against. THIS IS THE WEAKER CLAIM -- it shows the draft path accepts "
                f"at a real rate, not that it matches stock."
            ),
            command=commands[MODE_REQUIRE],
            measured=target,
            reference=float(floor),
            duration_s=elapsed,
            breakdown=breakdown,
        )
    if target is None or stock is None:
        return _verdict(
            name="acceptance",
            member=member,
            passed=False,
            detail=f"could not read acceptance_length from both legs "
            f"(require={target}, off={stock})",
            command=commands[MODE_REQUIRE],
            duration_s=elapsed,
        )
    for mode in (MODE_REQUIRE, MODE_OFF):
        short = too_short(mode)
        if short is not None:
            return short
    stock_breakdown = stock_reading.breakdown() if stock_reading is not None else ""
    if breakdown and stock_breakdown:
        breakdown = f"require: {breakdown}; off: {stock_breakdown}"
    if target <= 1.0:
        return _verdict(
            name="acceptance",
            member=member,
            passed=False,
            detail=f"acceptance_length {target:.4f} at or below 1.0 means every draft "
            f"was rejected — the draft path is computing the wrong thing",
            command=commands[MODE_REQUIRE],
            measured=target,
            reference=stock,
            duration_s=elapsed,
            breakdown=breakdown,
        )

    ratio = target / stock if stock else 0.0
    passed = ratio >= min_ratio
    return _verdict(
        name="acceptance",
        member=member,
        passed=passed,
        detail=(
            f"acceptance_length {target:.4f} vs stock {stock:.4f} "
            f"(ratio {ratio:.3f}, floor {min_ratio:.2f}). Clearly above 1.0 but below "
            f"the reference means subtly wrong, not merely slower."
        ),
        command=commands[MODE_REQUIRE],
        measured=target,
        reference=stock,
        duration_s=elapsed,
        breakdown=breakdown,
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
    "ACCURACY_TIMEOUT_S",
    "ALLOW_DIRTY_ENV",
    "AcceptanceReading",
    "AcceptanceWorkload",
    "BOOT_TIMEOUT_S",
    "BootSpec",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "GateLog",
    "GateResult",
    "MODELING_V2_ENV",
    "MODE_OFF",
    "MODE_REQUIRE",
    "RUNTIME_VALUE_DUMP_ENV",
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

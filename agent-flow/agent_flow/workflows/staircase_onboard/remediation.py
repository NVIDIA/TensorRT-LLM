# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bounded loop that closes an `UNCOVERED` audit.

WHAT WAS THERE BEFORE

An auditor that found runtime values outside their entries' certified
columns returned `UNCOVERED`, and the run **ended**. The fix then happened
outside the workflow, serially, by hand:

    auditor finds a gap
      -> run closes
      -> a human reads the finding, certifies the entry, re-runs
      -> a fresh, stateless auditor re-scans EVERYTHING
      -> finds the next gap

That is the mechanism behind an onboarding run spending audit passes 53,
54, 55, 56 and 57 rediscovering an overlapping set of gaps. Each pass paid
a full re-scan to learn one increment.

WHY THE OBVIOUS FIX IS WRONG

    while auditor == UNCOVERED:
        dispatch a smith per uncovered value
        re-run the full audit

This is worse than the manual loop, not better. It re-dispatches identical
work when the auditor repeats itself, it fans out one smith per *value*
when seven uncovered batch sizes on one entry are a single piece of work,
and it has no stopping condition that is not "the model changed its mind".

WHAT THIS MODULE IS

The same loop, made **bounded, deduplicated and monotone**. Four rules,
each of which exists because its absence is a specific failure:

  * **Group by entry surface, not by value.** Seven uncovered batch sizes
    on one entry are one verify dispatch that extends one parameterized
    GPU test and writes one receipt, not seven. The merge key is
    ``(entry, arch, axis)``.
  * **Give every item a stable identity.** Natural language cannot say
    whether two rounds found the same gap. The identity is a content hash
    over the canonicalized surface plus the commit and capture it was
    observed under, so "did we already do this?" is answerable.
  * **Never re-dispatch a certified identity.** If a row that was
    certified comes back, something is wrong upstream -- stale receipt,
    stale contract read, a wrong coverage evaluator, or a smith whose fix
    did not actually cover the surface. Dispatching the same smith again
    cannot fix any of those. It is reported as non-progress.
  * **Require monotone convergence.** The unresolved set must shrink. If
    round N's signature equals round N-1's, the loop stops with a named
    verdict instead of spending another GPU hour proving it is stuck.

Everything here is a pure function over plain data. No subprocess, no
agent, no filesystem -- the orchestrator owns those and hands in the
results. That is what makes the convergence rules testable without a GPU.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------- queue states

#: Queued, not yet dispatched.
QUEUE_PENDING = "pending"
#: Dispatched in this round; a crash here is distinguishable from a crash
#: before it, so a resume knows the work may have been lost.
QUEUE_RUNNING = "running"
#: A smith landed the contract, the parameterized test and a fresh receipt.
QUEUE_CERTIFIED = "certified"
#: A smith ran the test and it did not pass. Terminal: re-dispatching the
#: same work cannot change the answer.
QUEUE_FAILED_RECEIPT = "failed_receipt"
#: A smith could not proceed at all. Terminal for the same reason.
QUEUE_BLOCKED = "blocked"
#: Replaced by a wider item covering the same surface.
QUEUE_SUPERSEDED = "superseded"

#: States that still owe work.
OPEN_STATES = (QUEUE_PENDING, QUEUE_RUNNING)
#: States a round must not re-dispatch.
TERMINAL_STATES = (QUEUE_CERTIFIED, QUEUE_FAILED_RECEIPT, QUEUE_BLOCKED, QUEUE_SUPERSEDED)

VALID_STATES = OPEN_STATES + TERMINAL_STATES

# ------------------------------------------------------------------- verdicts

#: The unresolved set stopped shrinking, or a certified row came back. Either
#: way the loop cannot converge and a human has to look.
STOP_NO_PROGRESS = "audit-no-progress"
#: A budget ran out with work still open.
STOP_BUDGET = "audit-remediation-budget"
#: A verify smith reported FAILED_RECEIPT or BLOCKED.
STOP_VERIFICATION_FAILED = "audit-verification-failed"

# -------------------------------------------------------------------- budgets


@dataclass(frozen=True)
class RemediationBudget:
    """Hard limits, so the loop is a bounded cycle rather than a ``while``.

    The defaults are deliberately small. Remediation is the *cheap* half of
    onboarding -- it runs catalog GPU tests, not accuracy gates -- but it is
    still GPU time, and a loop that needs a third round is usually a loop
    whose first round was measuring the wrong thing.
    """

    max_rounds: int = 2
    #: One attempt per identity. A second attempt at byte-identical work
    #: with unchanged evidence is the definition of no progress.
    max_attempts_per_item: int = 1
    max_new_items: int = 16
    max_parallel_verifiers: int = 4


# ----------------------------------------------------------------------- rows


def _normalize_values(values: Iterable[Any]) -> tuple[str, ...]:
    """Canonicalize a surface's values so two spellings of it hash alike.

    Numeric-looking values sort numerically; everything else sorts as text.
    Without this, ``[3, 5, 12]`` and ``[12, 3, 5]`` are different identities
    and the same gap is dispatched twice.
    """
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    numeric = all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in seen) if seen else False
    return tuple(sorted(seen, key=float) if numeric else sorted(seen))


@dataclass(frozen=True)
class UncoveredRow:
    """One finding, as the auditor reported it.

    ``axis`` is the certification dimension the value varies along --
    ``batch_size``, ``num_pages``, ``local_layer_idx``. It is what makes two
    findings on one entry the same piece of work or different ones.
    """

    entry: str
    arch: str
    axis: str
    values: tuple[str, ...] = ()
    #: The contract sentence that excludes the value. Carried into the smith
    #: brief verbatim: it is the difference between "certify batch size 3"
    #: and a dispatch that has to re-derive what was wrong.
    contract_line: str = ""

    @staticmethod
    def parse(raw: Any) -> "UncoveredRow | None":
        """Build a row from one decision-file mapping, or ``None`` if unusable.

        Permissive about shape and strict about content: a row missing the
        entry, the arch or the axis cannot be grouped or dispatched, so it
        is dropped rather than guessed at. The caller reports the drop --
        silently discarding an auditor finding is the one outcome worse than
        having none.
        """
        if not isinstance(raw, Mapping):
            return None
        entry = str(raw.get("entry", "")).strip()
        arch = str(raw.get("arch", "")).strip()
        axis = str(raw.get("axis", "")).strip()
        if not entry or not arch or not axis:
            return None
        values = raw.get("values")
        if values is None:
            values = raw.get("value")
        if isinstance(values, (str, int, float)):
            values = [values]
        if not isinstance(values, Sequence):
            values = []
        return UncoveredRow(
            entry=entry,
            arch=arch,
            axis=axis,
            values=_normalize_values(values),
            contract_line=str(raw.get("contract_line", "")).strip(),
        )


def parse_uncovered_rows(raw: Any) -> tuple[tuple[UncoveredRow, ...], int]:
    """Parse the ``uncovered`` block of an auditor decision.

    Returns the usable rows and how many were dropped, so the caller can say
    so rather than silently narrowing the audit.
    """
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (), 0
    rows: list[UncoveredRow] = []
    dropped = 0
    for candidate in raw:
        row = UncoveredRow.parse(candidate)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    return tuple(rows), dropped


# ---------------------------------------------------------------------- items


@dataclass(frozen=True)
class RemediationItem:
    """One verify dispatch: one entry surface, on one architecture.

    The unit the queue is made of. Several :class:`UncoveredRow` on the same
    ``(entry, arch, axis)`` collapse into one of these, because they are one
    piece of work: a smith extends the parameterized GPU test over the whole
    surface and writes one receipt.
    """

    entry: str
    arch: str
    axis: str
    values: tuple[str, ...] = ()
    contract_line: str = ""
    #: The target commit and runtime capture the surface was observed under.
    #: Part of the identity: the same values under a NEW capture are
    #: legitimately new work (a wrapper change can open a new runtime
    #: branch), while the same values under the same capture are a repeat.
    target_commit: str = ""
    capture_id: str = ""
    status: str = QUEUE_PENDING
    attempts: int = 0

    @property
    def group_key(self) -> tuple[str, str, str]:
        """What makes two findings the same piece of work."""
        return (self.entry, self.arch, self.axis)

    @property
    def identity(self) -> str:
        """Content hash over everything that makes this work distinct.

        Stable across processes and across a resume, which is the whole
        point: it is how round two answers "have I already done this?"
        """
        payload = "|".join(
            [
                self.entry,
                self.arch,
                self.axis,
                ",".join(self.values),
                self.target_commit,
                self.capture_id,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        """The checkpoint form. ``identity`` is derived, never stored."""
        return {
            "entry": self.entry,
            "arch": self.arch,
            "axis": self.axis,
            "values": list(self.values),
            "contract_line": self.contract_line,
            "target_commit": self.target_commit,
            "capture_id": self.capture_id,
            "status": self.status,
            "attempts": self.attempts,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "RemediationItem":
        status = str(raw.get("status", QUEUE_PENDING))
        return RemediationItem(
            entry=str(raw.get("entry", "")),
            arch=str(raw.get("arch", "")),
            axis=str(raw.get("axis", "")),
            values=_normalize_values(raw.get("values") or ()),
            contract_line=str(raw.get("contract_line", "")),
            target_commit=str(raw.get("target_commit", "")),
            capture_id=str(raw.get("capture_id", "")),
            status=status if status in VALID_STATES else QUEUE_PENDING,
            attempts=int(raw.get("attempts", 0) or 0),
        )

    def brief(self) -> str:
        """The dispatch brief a verify smith reads.

        Deliberately the whole surface rather than the uncovered subset:
        a parameterized test written over ``[3, 5, 7]`` and a test written
        over the entry's full observed range are different tests, and only
        the second one stops the next round finding value 9.
        """
        values = ", ".join(self.values) or "(none recorded)"
        line = (
            f"\nThe contract line that excludes them: {self.contract_line}"
            if self.contract_line
            else ""
        )
        return (
            f"axis: {self.axis}\n"
            f"uncovered values on {self.arch}: {values}{line}\n"
            f"Extend the parameterized GPU test over this surface, update the "
            f"contract's certified column, and write a fresh {self.arch} receipt."
        )


def canonicalize(
    rows: Iterable[UncoveredRow],
    *,
    target_commit: str = "",
    capture_id: str = "",
) -> tuple[RemediationItem, ...]:
    """Collapse findings into one item per ``(entry, arch, axis)``.

    This is the step that turns "seven uncovered batch sizes" into one
    dispatch instead of seven, and it is why the queue is a queue of
    *surfaces* rather than of values.
    """
    grouped: dict[tuple[str, str, str], list[UncoveredRow]] = {}
    for row in rows:
        grouped.setdefault((row.entry, row.arch, row.axis), []).append(row)

    items: list[RemediationItem] = []
    for (entry, arch, axis), members in grouped.items():
        values: list[str] = []
        for member in members:
            values.extend(member.values)
        # First non-empty contract line wins: they describe the same column.
        line = next((m.contract_line for m in members if m.contract_line), "")
        items.append(
            RemediationItem(
                entry=entry,
                arch=arch,
                axis=axis,
                values=_normalize_values(values),
                contract_line=line,
                target_commit=target_commit,
                capture_id=capture_id,
            )
        )
    return tuple(sorted(items, key=lambda i: i.group_key))


# ---------------------------------------------------------------------- merge


@dataclass(frozen=True)
class MergeOutcome:
    """What appending a round's findings to the queue actually did."""

    queue: tuple[RemediationItem, ...] = ()
    #: Genuinely new work.
    appended: tuple[RemediationItem, ...] = ()
    #: Already queued or already running; not dispatched again.
    duplicates: tuple[RemediationItem, ...] = ()
    #: An open item grew to cover more of the same surface.
    widened: tuple[RemediationItem, ...] = ()
    #: A surface that was certified came back. Not re-dispatchable -- see
    #: the module docstring.
    regressions: tuple[RemediationItem, ...] = ()
    #: A surface whose previous attempt failed or was blocked.
    exhausted: tuple[RemediationItem, ...] = ()

    @property
    def has_new_work(self) -> bool:
        return bool(self.appended or self.widened)


def merge_into_queue(
    queue: Sequence[RemediationItem],
    incoming: Sequence[RemediationItem],
    *,
    budget: RemediationBudget = RemediationBudget(),
) -> MergeOutcome:
    """Append only genuinely new work, and classify everything else.

    The classification is the useful part. "Nothing was appended" is not one
    condition but four, and they need different verdicts: an auditor
    repeating itself, an auditor widening a surface, a certified row coming
    back, and a row whose attempt already failed.
    """
    by_group: dict[tuple[str, str, str], int] = {}
    working = list(queue)
    for index, item in enumerate(working):
        # Later entries win, so a superseded row never shadows its successor.
        by_group[item.group_key] = index

    appended: list[RemediationItem] = []
    duplicates: list[RemediationItem] = []
    widened: list[RemediationItem] = []
    regressions: list[RemediationItem] = []
    exhausted: list[RemediationItem] = []

    for item in incoming:
        index = by_group.get(item.group_key)
        if index is None:
            if len(appended) >= budget.max_new_items:
                # Over the cap: not silently dropped, reported as exhausted
                # so the caller stops rather than proceeding on a partial set.
                exhausted.append(item)
                continue
            working.append(item)
            by_group[item.group_key] = len(working) - 1
            appended.append(item)
            continue

        existing = working[index]
        covered = set(item.values) <= set(existing.values)

        if existing.status in (QUEUE_FAILED_RECEIPT, QUEUE_BLOCKED):
            exhausted.append(item)
            continue
        if existing.status == QUEUE_CERTIFIED:
            # Same surface, already certified -> upstream is wrong somewhere.
            # A wider surface under the SAME capture is the same signal: the
            # first audit was incomplete.
            regressions.append(item)
            continue
        if covered:
            duplicates.append(item)
            continue
        if existing.attempts >= budget.max_attempts_per_item:
            exhausted.append(item)
            continue

        merged = replace(
            existing,
            values=_normalize_values(list(existing.values) + list(item.values)),
            contract_line=existing.contract_line or item.contract_line,
            status=QUEUE_PENDING,
        )
        working[index] = merged
        widened.append(merged)

    return MergeOutcome(
        queue=tuple(working),
        appended=tuple(appended),
        duplicates=tuple(duplicates),
        widened=tuple(widened),
        regressions=tuple(regressions),
        exhausted=tuple(exhausted),
    )


# ------------------------------------------------------------------ signature


def unresolved_signature(queue: Sequence[RemediationItem]) -> str:
    """A hash of the still-open identity set.

    Round N's signature equal to round N-1's means the loop learned nothing
    and would learn nothing from another round. Comparing signatures is
    cheaper and stricter than comparing prose summaries, which was the only
    thing available before.
    """
    identities = sorted(i.identity for i in queue if i.status in OPEN_STATES)
    if not identities:
        return ""
    return hashlib.sha256("|".join(identities).encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------- round plan


@dataclass(frozen=True)
class RoundDecision:
    """What the orchestrator should do next."""

    #: ``dispatch`` | ``stop`` | ``clean``
    action: str
    #: The closing verdict, when ``action`` is ``stop``.
    verdict: str = ""
    dispatch: tuple[RemediationItem, ...] = ()
    #: Human-readable reason, for the journal and the console.
    detail: str = ""
    workers: int = 1

    @property
    def stopping(self) -> bool:
        return self.action == "stop"


def decide_round(
    merge: MergeOutcome,
    *,
    budget: RemediationBudget,
    round_no: int,
    previous_signature: str = "",
) -> RoundDecision:
    """Decide whether to dispatch another verification round, or stop.

    The order of these checks is the policy, and it is deliberate:

    1. nothing open at all -> the audit is closed;
    2. a certified surface came back -> stop, whatever else is true. It
       means the evidence is untrustworthy, and more rounds cannot repair
       evidence;
    3. the round budget is spent -> stop;
    4. no new work -> stop. The auditor repeated itself;
    5. the unresolved set did not change -> stop. Monotone convergence is
       the contract;
    6. otherwise dispatch, capped at ``max_parallel_verifiers``.
    """
    open_items = tuple(i for i in merge.queue if i.status in OPEN_STATES)

    if not open_items and not merge.regressions:
        return RoundDecision(action="clean", detail="no uncovered surfaces remain")

    if merge.regressions:
        names = ", ".join(f"{i.entry}[{i.axis}]" for i in merge.regressions[:4])
        return RoundDecision(
            action="stop",
            verdict=STOP_NO_PROGRESS,
            detail=(
                f"{len(merge.regressions)} surface(s) reported uncovered after being "
                f"certified in an earlier round: {names}. Re-dispatching the same "
                f"verification cannot fix a stale receipt, a stale contract read or a "
                f"coverage evaluator that is wrong -- and one of those is what this is."
            ),
        )

    if round_no > budget.max_rounds:
        return RoundDecision(
            action="stop",
            verdict=STOP_BUDGET,
            detail=(
                f"remediation budget spent ({budget.max_rounds} round(s)) with "
                f"{len(open_items)} surface(s) still uncovered"
            ),
        )

    if merge.exhausted:
        names = ", ".join(f"{i.entry}[{i.axis}]" for i in merge.exhausted[:4])
        return RoundDecision(
            action="stop",
            verdict=STOP_BUDGET,
            detail=(
                f"{len(merge.exhausted)} surface(s) are past their attempt or item budget: {names}"
            ),
        )

    if not merge.has_new_work:
        return RoundDecision(
            action="stop",
            verdict=STOP_NO_PROGRESS,
            detail=(
                f"the audit reported {len(merge.duplicates)} surface(s), all already "
                f"queued or running. Nothing new was learned this round."
            ),
        )

    signature = unresolved_signature(merge.queue)
    if previous_signature and signature == previous_signature:
        return RoundDecision(
            action="stop",
            verdict=STOP_NO_PROGRESS,
            detail=(
                "the unresolved set is byte-identical to the previous round's. "
                "Remediation has to shrink it to continue."
            ),
        )

    dispatch = tuple(i for i in merge.queue if i.status == QUEUE_PENDING)
    return RoundDecision(
        action="dispatch",
        dispatch=dispatch,
        workers=max(1, min(budget.max_parallel_verifiers, len(dispatch))),
        detail=(
            f"{len(merge.appended)} new and {len(merge.widened)} widened surface(s); "
            f"{len(dispatch)} verify dispatch(es)"
        ),
    )


# ------------------------------------------------------------------- outcomes

#: Smith decisions that certify a surface, and the queue state each maps to.
_DECISION_STATES = {
    "LANDED": QUEUE_CERTIFIED,
    "FAILED_RECEIPT": QUEUE_FAILED_RECEIPT,
    "BLOCKED": QUEUE_BLOCKED,
}


def apply_outcomes(
    queue: Sequence[RemediationItem],
    outcomes: Mapping[str, str | None],
) -> tuple[tuple[RemediationItem, ...], tuple[RemediationItem, ...]]:
    """Fold a round's smith decisions back into the queue.

    ``outcomes`` maps identity to the smith's decision. Returns the new
    queue and the items that failed, which the caller turns into
    ``audit-verification-failed``.

    A missing or unrecognized decision counts as blocked rather than as
    silence: a verify dispatch that ended without saying anything has not
    certified anything, and treating it as success is how an uncertified
    surface reaches a merge.
    """
    updated: list[RemediationItem] = []
    failed: list[RemediationItem] = []
    for item in queue:
        if item.status not in OPEN_STATES:
            updated.append(item)
            continue
        decision = (outcomes.get(item.identity) or "").strip().upper()
        state = _DECISION_STATES.get(decision, QUEUE_BLOCKED)
        landed = replace(item, status=state, attempts=item.attempts + 1)
        updated.append(landed)
        if state in (QUEUE_FAILED_RECEIPT, QUEUE_BLOCKED):
            failed.append(landed)
    return tuple(updated), tuple(failed)


@dataclass
class RemediationState:
    """The queue plus the bookkeeping the convergence rules need.

    Persisted in the run's checkpoint so a resume continues the cycle rather
    than restarting it -- which is what made the manual loop re-scan
    everything each time.
    """

    queue: list[RemediationItem] = field(default_factory=list)
    round_no: int = 0
    signature: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue": [item.as_dict() for item in self.queue],
            "round_no": self.round_no,
            "signature": self.signature,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any] | None) -> "RemediationState":
        raw = raw or {}
        rows = raw.get("queue") or []
        return RemediationState(
            queue=[RemediationItem.from_dict(r) for r in rows if isinstance(r, Mapping)],
            round_no=int(raw.get("round_no", 0) or 0),
            signature=str(raw.get("signature", "")),
        )

    def render(self) -> str:
        """One line per item, for the journal."""
        if not self.queue:
            return "(empty)"
        return "\n".join(
            f"  {item.identity}  {item.status:<14} {item.entry}[{item.axis}] "
            f"-> {', '.join(item.values) or '(none)'}"
            for item in self.queue
        )


__all__ = [
    "OPEN_STATES",
    "QUEUE_BLOCKED",
    "QUEUE_CERTIFIED",
    "QUEUE_FAILED_RECEIPT",
    "QUEUE_PENDING",
    "QUEUE_RUNNING",
    "QUEUE_SUPERSEDED",
    "STOP_BUDGET",
    "STOP_NO_PROGRESS",
    "STOP_VERIFICATION_FAILED",
    "TERMINAL_STATES",
    "VALID_STATES",
    "MergeOutcome",
    "RemediationBudget",
    "RemediationItem",
    "RemediationState",
    "RoundDecision",
    "UncoveredRow",
    "apply_outcomes",
    "canonicalize",
    "decide_round",
    "merge_into_queue",
    "parse_uncovered_rows",
    "unresolved_signature",
]

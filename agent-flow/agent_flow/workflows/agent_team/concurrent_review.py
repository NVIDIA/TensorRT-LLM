"""Snapshot + prompt helpers for the overlapped (concurrent) review mode.

Sequentially, the build phase is ``coder(i) -> reviewer(i) -> verdict ->
coder(i+1)``. A thorough reviewer can take hours, and the coder — plus
whatever machines it has reserved — sits idle for all of it.

With ``--concurrent-review`` the workflow instead:

1. snapshots the coder's checkout at the end of ``coder(i)`` (a git ref
   ``refs/agent-flow/review/iter-<i>`` at the current ``HEAD``),
2. runs ``reviewer(i)`` against that frozen commit on a worker thread, and
3. starts ``coder(i+1)`` immediately, telling it to work in a git worktree
   branched from the snapshot so the reviewer's checkout stays stable.

The verdict is delivered to the already-running coder as a notice entry in
``progress.yaml`` (``source: reviewer``), which the coder polls with the
``read_review_notices`` MCP tool.

This module holds the pieces that have no orchestration state: resolving
the repo to snapshot, taking the snapshot, and building the two prompt
addenda. The pipeline itself lives in ``workflow.py``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Task keys understood by this module. All optional: ``review_snapshot_repo``
# names the checkout the coder edits (``trtllm_repo_path`` is accepted as a
# fallback so modeling-bringup task files work unchanged), and the two
# worktree keys are pass-through hints repeated to the coder verbatim.
TASK_KEY_SNAPSHOT_REPO = "review_snapshot_repo"
TASK_KEY_SNAPSHOT_REPO_FALLBACK = "trtllm_repo_path"
TASK_KEY_WORKTREES_DIR = "worktrees_dir"
TASK_KEY_WORKTREE_RESERVATIONS = "worktree_reservations"
# ``review_checkout`` inverts the split: the coder keeps working IN PLACE in
# the snapshot repo and the REVIEWER gets its own checkout (a git worktree of
# the same repo, pre-populated with whatever build artifacts it needs). The
# orchestrator detaches that worktree at the snapshot commit before the
# reviewer starts. Use it when the coder's tooling hard-wires the main
# checkout (fixed PYTHONPATH, untracked build outputs, container mounts).
TASK_KEY_REVIEW_CHECKOUT = "review_checkout"

REVIEW_REF_PREFIX = "refs/agent-flow/review/iter-"

#: ``source`` value stamped on orchestrator-authored review notices so the
#: coder can tell them apart from ``--feedback`` entries written by a human.
REVIEWER_NOTICE_SOURCE = "reviewer"


class SnapshotError(RuntimeError):
    """Raised when the coder's checkout cannot be frozen for review."""


@dataclass(frozen=True)
class ReviewSnapshot:
    """A frozen commit of the coder's checkout for one review pass."""

    iteration: int
    repo: Path
    commit: str
    ref: str
    #: Uncommitted entries (``git status --porcelain`` lines) that were NOT
    #: part of the snapshot. Only non-empty with ``allow_dirty`` (in-place
    #: mode); the reviewer is told about them.
    dirty: tuple[str, ...] = ()

    @property
    def short_commit(self) -> str:
        return self.commit[:12]


def review_ref(iteration: int) -> str:
    return f"{REVIEW_REF_PREFIX}{iteration}"


def read_task_mapping(task_path: Path) -> dict[str, Any]:
    """Parse ``task.yaml``, returning ``{}`` for anything that is not a mapping.

    Concurrent review is opt-in and best-effort about task keys: a task file
    the workflow cannot parse must not break the run, it just means no repo
    was configured and the workflow falls back to sequential review.
    """
    try:
        data = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_snapshot_repo(task_path: Path, override: str | Path | None = None) -> Path | None:
    """Resolve the checkout to snapshot, or ``None`` when none is configured.

    Precedence: the CLI override, then ``review_snapshot_repo`` in the task
    file, then ``trtllm_repo_path`` (modeling-bringup's name for the same
    thing). The path is not required to exist here; ``snapshot_repo``
    reports that with a clearer message at the point of use.
    """
    if override is not None:
        return Path(override)
    data = read_task_mapping(task_path)
    for key in (TASK_KEY_SNAPSHOT_REPO, TASK_KEY_SNAPSHOT_REPO_FALLBACK):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip())
    return None


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SnapshotError(
            f"`git {' '.join(args)}` failed in {repo} "
            f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def snapshot_repo(repo: Path, iteration: int, *, allow_dirty: bool = False) -> ReviewSnapshot:
    """Freeze ``repo`` at its current ``HEAD`` under a per-iteration ref.

    Raises :class:`SnapshotError` when the repo is missing, is not a git
    checkout, or has a dirty working tree. A dirty tree means the commit the
    reviewer would read is *not* what the coder actually built, so the
    caller must fall back to a sequential review for that iteration rather
    than review a lie.

    With ``allow_dirty`` (in-place mode, see ``TASK_KEY_REVIEW_CHECKOUT``)
    the coder never stops editing this tree anyway, so the reviewer reviews
    the commit and is told which uncommitted entries it did not see.
    """
    if not repo.is_dir():
        raise SnapshotError(f"snapshot repo does not exist: {repo}")

    _git(repo, "rev-parse", "--git-dir")
    dirty = _git(repo, "status", "--porcelain")
    if dirty and not allow_dirty:
        first = dirty.splitlines()[:5]
        raise SnapshotError(
            f"working tree in {repo} is dirty ({len(dirty.splitlines())} "
            f"entries, e.g. {'; '.join(first)}). The reviewer can only read "
            f"committed state, so an uncommitted tree cannot be frozen."
        )

    commit = _git(repo, "rev-parse", "HEAD")
    ref = review_ref(iteration)
    _git(repo, "update-ref", ref, commit)
    return ReviewSnapshot(
        iteration=iteration,
        repo=repo,
        commit=commit,
        ref=ref,
        dirty=tuple(dirty.splitlines()) if dirty else (),
    )


def checkout_snapshot(checkout: Path, snapshot: ReviewSnapshot) -> None:
    """Detach the reviewer's worktree ``checkout`` at ``snapshot.commit``.

    ``checkout`` must be a git worktree of the same repository as the
    snapshot (same ``--git-common-dir``), must not BE the snapshot repo, and
    must have no tracked modifications (untracked/ignored files such as
    linked build artifacts are fine). Raises :class:`SnapshotError`.
    """
    if not checkout.is_dir():
        raise SnapshotError(f"review checkout does not exist: {checkout}")
    if checkout.resolve() == snapshot.repo.resolve():
        raise SnapshotError(
            f"review checkout {checkout} is the snapshot repo itself; it must "
            f"be a separate worktree"
        )
    common = Path(
        _git(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()
    expected = Path(
        _git(snapshot.repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()
    if common != expected:
        raise SnapshotError(
            f"review checkout {checkout} is not a worktree of {snapshot.repo} "
            f"(git common dir {common} != {expected})"
        )
    modified = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if modified:
        raise SnapshotError(
            f"review checkout {checkout} has tracked modifications "
            f"({modified.splitlines()[:3]}); refusing to move it"
        )
    _git(checkout, "checkout", "--quiet", "--detach", snapshot.commit)


def snapshot_exists(repo: Path, ref: str) -> bool:
    """Whether ``ref`` still resolves in ``repo`` (used on resume)."""
    try:
        _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except SnapshotError:
        return False
    return True


def reviewer_addendum(snapshot: ReviewSnapshot, *, review_checkout: str | None = None) -> str:
    """Prompt addendum telling the reviewer its checkout is frozen.

    With ``review_checkout`` (in-place mode) the frozen tree is the
    reviewer's own worktree, already detached at the snapshot commit, and
    the coder's main checkout is explicitly NOT stable.
    """
    if review_checkout:
        where = (
            f"\n\n**Concurrent review — review commit `{snapshot.commit}` "
            f"(ref `{snapshot.ref}`) from YOUR OWN checkout "
            f"`{review_checkout}`,** which the orchestrator has detached at "
            f"that commit (`git -C {review_checkout} log -1` shows it). Read, "
            f"build and run from `{review_checkout}` only. The Coder has "
            f"already moved on to the next iteration and is editing "
            f"`{snapshot.repo}` IN PLACE, so that tree is not stable and is "
            f"not what you are reviewing; never run from it. The task's "
            f"rules say how to launch container jobs from your checkout. "
            f"Do not expect, wait for, or request changes during this "
            f"review: nothing you ask for can land before your verdict. "
            f"Your APPROVE/REJECT applies to this commit only; work the "
            f"Coder does after it gets a separate review pass."
        )
        if snapshot.dirty:
            sample = "; ".join(snapshot.dirty[:5])
            where += (
                f"\n\nNOTE: {len(snapshot.dirty)} uncommitted entries in the "
                f"Coder's tree were NOT part of this snapshot (e.g. {sample}). "
                f"Review the commit; if the Coder's status claims results "
                f"from uncommitted code, say so in the report."
            )
        return where + (
            "\n\nWrite your rolling status snapshot with `update_status` as "
            "usual — in concurrent mode your `update_status` writes a "
            "per-iteration file that the Coder reads, so it will not collide "
            "with the Coder's own status.md."
        )
    return (
        f"\n\n**Concurrent review — the checkout is FROZEN for this "
        f"review.** Review commit `{snapshot.commit}` of `{snapshot.repo}` "
        f"(also reachable as `{snapshot.ref}`). The Coder has already moved "
        f"on to the next iteration and is working elsewhere, in a separate "
        f"git worktree. Do not expect, wait for, or request changes to the "
        f"checkout during this review: nothing you ask for can land before "
        f"your verdict, and anything that appears to change under you is "
        f"not part of what you are reviewing. Inspect the snapshot with "
        f"`git -C {snapshot.repo} show/diff {snapshot.ref}`, and if you need "
        f"to build or run, do it from a checkout of `{snapshot.ref}` rather "
        f"than the live working tree. Your APPROVE/REJECT applies to this "
        f"commit only; work the Coder does after it gets a separate review "
        f"pass.\n\n"
        f"Write your rolling status snapshot with `update_status` as usual — "
        f"in concurrent mode your `update_status` writes a per-iteration "
        f"file that the Coder reads, so it will not collide with the "
        f"Coder's own status.md."
    )


def coder_addendum(
    snapshot: ReviewSnapshot,
    *,
    worktrees_dir: str | None = None,
    reservations: str | None = None,
    review_checkout: str | None = None,
) -> str:
    """Prompt addendum telling the coder a review is running against a frozen tree.

    With ``review_checkout`` (in-place mode) the coder keeps working in the
    main checkout; the only constraints are history hygiene and committing
    before the turn ends.
    """
    if review_checkout:
        return (
            f"\n\n**Concurrent review in progress (in-place mode).** The "
            f"review of iteration {snapshot.iteration} is running right now "
            f"against commit `{snapshot.commit}` (ref `{snapshot.ref}`) from "
            f"the Reviewer's own checkout `{review_checkout}`. Your working "
            f"tree in `{snapshot.repo}` is NOT frozen: keep working there as "
            f"usual, with these rules:\n"
            f"1. Never rewrite history at or below `{snapshot.commit}` — no "
            f"amend, rebase, reset or force-move of anything reachable from "
            f"`{snapshot.ref}`. Only add commits on top.\n"
            f"2. Do not touch `{review_checkout}` (files, branch, HEAD).\n"
            f"3. Commit everything before you end your turn: uncommitted "
            f"files are invisible to the next review snapshot.\n"
            f"4. Poll `read_review_notices` periodically — at minimum once "
            f"mid-turn and once before you end your turn. The verdict arrives "
            f"as a `REVIEW VERDICT for iteration {snapshot.iteration}` notice "
            f"with the decision and the path to the Reviewer's report. On a "
            f"REJECT, address the Reviewer's feedback as part of this "
            f"iteration.\n\n"
            f"If the verdict has not arrived by the time you would otherwise "
            f"finish, end your turn anyway with your work committed and say "
            f"so in `update_status`; the orchestrator folds the verdict into "
            f"the next iteration."
        )
    if worktrees_dir:
        where = f"a git worktree under `{worktrees_dir}`"
        if reservations:
            where += (
                f", claiming your slot in `{reservations}` first so you do "
                f"not collide with another worker"
            )
    else:
        where = f"a git worktree under `{snapshot.repo}/../worktrees`"
    return (
        f"\n\n**Concurrent review in progress.** The review of iteration "
        f"{snapshot.iteration} is running right now against commit "
        f"`{snapshot.commit}` of `{snapshot.repo}` (ref `{snapshot.ref}`).\n\n"
        f"Until the verdict arrives you MUST NOT modify any file in "
        f"`{snapshot.repo}` — the Reviewer needs a stable source tree. "
        f"Instead:\n"
        f"1. Create {where}, branched from `{snapshot.ref}`, and do all of "
        f"this iteration's work there.\n"
        f"2. Run the code from the worktree, using whatever mechanism the "
        f"task/plan documents for that repo (e.g. `PYTHONPATH`, an editable "
        f"install, or a build directory of its own). Do not repoint the "
        f"main checkout at your worktree.\n"
        f"3. Poll `read_review_notices` periodically — at minimum once "
        f"mid-turn and once before you end your turn. The verdict arrives "
        f"as a `REVIEW VERDICT for iteration {snapshot.iteration}` notice "
        f"with the decision and the path to the Reviewer's report.\n"
        f"4. Once that notice has arrived, the freeze is lifted: fold your "
        f"worktree commits back into `{snapshot.repo}` (fast-forward if the "
        f"main checkout has not moved, otherwise cherry-pick) and continue "
        f"there. On a REJECT, address the Reviewer's feedback as part of "
        f"this iteration.\n\n"
        f"If the verdict has not arrived by the time you would otherwise "
        f"finish, end your turn anyway with your work committed in the "
        f"worktree and say so in `update_status`; the orchestrator folds the "
        f"verdict into the next iteration."
    )


def verdict_notice(
    iteration: int, decision: str | None, report_path: Path, *, in_place: bool = False
) -> str:
    """The notice text injected into the running coder when a review lands."""
    if in_place:
        return (
            f"REVIEW VERDICT for iteration {iteration}: {decision or 'MISSING'}, "
            f"see {report_path}. On REJECT, address the feedback in your "
            f"current iteration."
        )
    return (
        f"REVIEW VERDICT for iteration {iteration}: {decision or 'MISSING'}, "
        f"see {report_path}. The review freeze on the snapshot checkout is "
        f"lifted; fold your worktree commits back into the main checkout "
        f"before continuing."
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit overlap (``--overlap-units``): start the next unit's assembler early, in its own worktree.

Units are assembled strictly in order, and that order is right: core is green
before the first feature. But a unit's *gates* — boot over every member, the
reviewer, accuracy over every member, the witness — take hours, and while they
run the next unit's assembler is idle, although nothing it does depends on
those gates passing. Certification-only rejects never change the target, so
the work it would do is usually valid as it stands.

So, opt-in:

1. As soon as unit N's assembler writes ``UNIT_READY`` (before N's gates),
   unit N+1's assembler starts in a **separate git worktree** branched from
   ``HEAD``, with the main checkout's untracked native build linked in, its
   own commit helper, its own progress file and its own decision path. It
   runs in the background; nothing it writes can be read as unit N's.
2. When N's gates have passed and the loop reaches N+1, the early result is
   **adopted** only if every one of these holds: the turn ended
   ``UNIT_READY``, it committed something, its worktree is clean, its
   commits rebase onto the current ``HEAD`` without conflict (N's later
   turns may have moved it), and the main checkout fast-forwards to them.
   Adoption appends the early decision to the run's progress and goes
   straight to N+1's gates, the way a resume onto an unchanged
   ``UNIT_READY`` does.
3. **Anything else falls back to sequential**: the main checkout is left
   exactly as it was, the early worktree is kept for inspection (with its
   diff saved), and N+1's assembler is dispatched normally.

Not covered, deliberately: a run that ends (or a process that dies) while an
early turn is still running leaves its worktree on disk; a resumed process
never adopts a worktree it did not start, and says so. Remove it with
``git worktree remove --force <workspace>/overlap/<unit>/tree``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .committer import CommitIdentity, CommitPolicy, commit_instruction, install_committer
from .premerge._git import GitError, git

OVERLAP_DIRNAME = "overlap"
BRANCH_PREFIX = "staircase-overlap"

#: Untracked build outputs a worktree needs to import the package, found as
#: the git-ignored entries directly under this directory of the main checkout.
NATIVE_ROOT = "tensorrt_llm"
_NOT_NATIVE = ("__pycache__",)


class OverlapError(RuntimeError):
    """The early worktree could not be prepared."""


@dataclass
class OverlapJob:
    """One early assembler turn, and what came of it."""

    unit: str
    after: str
    root: Path
    tree: Path
    branch: str
    base: str
    commit_script: Path
    thread: threading.Thread | None = None
    decision: str = ""
    entry: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    linked: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AdoptOutcome:
    adopted: bool
    detail: str


def native_artifacts(repo: str | Path) -> list[str]:
    """Git-ignored entries directly under ``tensorrt_llm/`` in ``repo``."""
    out = git(
        repo,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--directory",
        "--",
        f"{NATIVE_ROOT}/",
        check=False,
    )
    found = []
    for line in out.splitlines():
        path = line.strip().rstrip("/")
        parts = path.split("/")
        if len(parts) == 2 and parts[0] == NATIVE_ROOT and parts[1] not in _NOT_NATIVE:
            if not parts[1].endswith(".egg-info"):
                found.append(path)
    return found


class UnitOverlap:
    """The git side of unit overlap: worktrees, linking, integration, cleanup.

    ``turn_runner(job, prompt)`` runs the early turn and returns its typed
    decision entry (a mapping with ``decision``); the workflow supplies one,
    a test supplies a fake.
    """

    def __init__(
        self,
        repo: str | Path,
        workspace: str | Path,
        *,
        policy: CommitPolicy | None = None,
        link_native: bool = True,
        turn_runner: Callable[[OverlapJob, str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.workspace = Path(workspace)
        self.policy = policy or CommitPolicy()
        self.link_native = link_native
        self.turn_runner = turn_runner
        self.jobs: dict[str, OverlapJob] = {}

    # ------------------------------------------------------------- set-up

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        identity: CommitIdentity | None = self.policy.identity
        if identity is not None:
            env.update(GIT_COMMITTER_NAME=identity.name, GIT_COMMITTER_EMAIL=identity.email)
        return env

    def open_worktree(self, unit: str, *, after: str) -> OverlapJob:
        """A worktree for ``unit`` at the main checkout's ``HEAD``, ready to commit into."""
        root = self.workspace / OVERLAP_DIRNAME / unit
        tree = root / "tree"
        if tree.exists():
            raise OverlapError(f"{tree} already exists (a previous early start); remove it first")
        base = git(self.repo, "rev-parse", "HEAD").strip()
        branch = f"{BRANCH_PREFIX}/{unit}-{base[:10]}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            git(
                self.repo,
                "worktree",
                "add",
                "-q",
                "-b",
                branch,
                str(tree),
                base,
                env={
                    **os.environ,
                    "GIT_LFS_SKIP_SMUDGE": "1",
                },
            )
        except GitError as exc:
            raise OverlapError(str(exc)) from exc
        linked = self._link_native(tree) if self.link_native else []
        script = install_committer(
            root,
            tree,
            policy=replace(self.policy, denied_paths=()),
        )
        job = OverlapJob(
            unit=unit,
            after=after,
            root=root,
            tree=tree,
            branch=branch,
            base=base,
            commit_script=script,
            linked=linked,
        )
        self.jobs[unit] = job
        return job

    def _link_native(self, tree: Path) -> list[str]:
        linked = []
        for rel in native_artifacts(self.repo):
            target = tree / rel
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(self.repo / rel)
            linked.append(rel)
        return linked

    # ------------------------------------------------------------- running

    def start(self, job: OverlapJob, prompt: str) -> None:
        """Run the early turn in a daemon thread; the result lands on ``job``."""
        if self.turn_runner is None:
            raise OverlapError("no turn runner configured")

        def run() -> None:
            try:
                entry = self.turn_runner(job, prompt) or {}
            except Exception as exc:  # noqa: BLE001
                # The thread boundary: whatever the turn raised has to reach
                # the adopting thread as a reason to fall back, not vanish
                # with a daemon thread.
                job.error = f"{type(exc).__name__}: {exc}"
                return
            job.entry = dict(entry)
            job.decision = str(entry.get("decision", "")).strip().upper()

        job.thread = threading.Thread(target=run, name=f"overlap-{job.unit}", daemon=True)
        job.thread.start()

    def wait(self, unit: str, timeout: float | None = None) -> OverlapJob | None:
        job = self.jobs.get(unit)
        if job is not None and job.thread is not None:
            job.thread.join(timeout)
        return job

    # ------------------------------------------------------------- landing

    def integrate(self, job: OverlapJob) -> AdoptOutcome:
        """Rebase the early commits onto ``HEAD`` and fast-forward the main checkout.

        Touches the main checkout only in the final fast-forward, and only
        after every check passed; a failure at any step leaves it unchanged.
        """
        if job.thread is not None and job.thread.is_alive():
            return AdoptOutcome(False, "the early turn is still running")
        if job.error:
            return AdoptOutcome(False, f"the early turn failed: {job.error}")
        if job.decision != "UNIT_READY":
            return AdoptOutcome(False, f"the early turn ended {job.decision or 'with no decision'}")
        dirty = [
            line
            for line in git(job.tree, "status", "--porcelain", "--untracked-files=no").splitlines()
            if line.strip()
        ]
        if dirty:
            return AdoptOutcome(False, f"uncommitted work in the early worktree: {dirty[:5]}")
        ahead = git(job.tree, "rev-list", f"{job.base}..HEAD").split()
        if not ahead:
            return AdoptOutcome(False, "the early turn committed nothing")
        head = git(self.repo, "rev-parse", "HEAD").strip()
        if head != job.base:
            try:
                git(job.tree, "rebase", "-q", "--onto", head, job.base, env=self._env())
            except GitError as exc:
                git(job.tree, "rebase", "--abort", check=False)
                return AdoptOutcome(
                    False, f"its commits do not rebase onto HEAD {head[:10]}: {exc}"
                )
        tip = git(job.tree, "rev-parse", "HEAD").strip()
        try:
            git(self.repo, "merge", "--ff-only", "-q", tip)
        except GitError as exc:
            return AdoptOutcome(
                False, f"the main checkout does not fast-forward to {tip[:10]}: {exc}"
            )
        return AdoptOutcome(True, f"{len(ahead)} commit(s) landed; HEAD {head[:10]} -> {tip[:10]}")

    def remove(self, job: OverlapJob) -> None:
        """Remove an adopted job's worktree and branch (its commits are on HEAD now)."""
        git(self.repo, "worktree", "remove", "--force", str(job.tree), check=False)
        git(self.repo, "branch", "-D", job.branch, check=False)
        self.jobs.pop(job.unit, None)

    def keep_for_inspection(self, job: OverlapJob, why: str) -> Path:
        """Leave a job that was not adopted on disk, with its diff and the reason."""
        note = job.root / "NOT_ADOPTED.md"
        diff = git(job.tree, "diff", job.base, check=False) if job.tree.exists() else ""
        (job.root / "early.patch").write_text(diff, encoding="utf-8")
        note.write_text(
            f"# {job.unit}: early start not adopted\n\n- why: {why}\n- worktree: `{job.tree}`\n"
            f"- branch: `{job.branch}` (from `{job.base[:10]}`)\n- diff: `early.patch`\n\n"
            f"Remove with `git -C {self.repo} worktree remove --force {job.tree}` and "
            f"`git -C {self.repo} branch -D {job.branch}`.\n",
            encoding="utf-8",
        )
        self.jobs.pop(job.unit, None)
        return note

    def stale(self, unit: str) -> Path | None:
        """A worktree for ``unit`` this process did not start, if one is on disk."""
        tree = self.workspace / OVERLAP_DIRNAME / unit / "tree"
        return tree if unit not in self.jobs and tree.exists() else None


# ----------------------------------------------------------------- the glue
#
# Everything below needs the orchestrator; it is here rather than in
# workflow.py so that file only gains its two call sites.


def early_prompt(workflow: Any, job: OverlapJob) -> str:
    """Unit ``job.unit``'s first assembler prompt, re-pointed at the worktree."""
    repo = str(workflow.task_data["trtllm_repo_path"])
    body = workflow._assembler_prompt(job.unit, 1).replace(
        f"Repository: {repo}\n", f"Repository: {job.tree}\n", 1
    )
    preface = (
        f"## Early start: `{job.unit}` overlaps `{job.after}`'s gates\n\n"
        f"`{job.after}` is UNIT_READY and is being gated now. You are assembling "
        f"`{job.unit}` ahead of that, in its **own worktree**:\n\n"
        f"- every path that names the repository means `{job.tree}` for this turn — edit "
        f"only there, never in `{repo}`;\n"
        f"- commit only through `{job.commit_script}` (the script in your system prompt "
        f"commits into the other checkout);\n"
        f"- the native build is linked in from the main checkout, so imports work; do not "
        f"rebuild it from here.\n\n"
        f"If `{job.after}`'s gates force a change, your commits are rebased onto it before "
        f"they land; a conflict discards this turn and the unit is assembled again "
        f"afterwards, so keep to `{job.unit}`'s own files.\n\n"
    )
    context = workflow._context_block()
    return (
        preface
        + body
        + (f"\n\n{context}\n" if context else "")
        + "\n\n"
        + commit_instruction(job.commit_script)
    )


def run_early_turn(workflow: Any, job: OverlapJob, prompt: str) -> dict[str, Any] | None:
    """Run the early turn through the run's own transport; return its decision entry."""
    from agent_flow.runtime import TurnRequest

    from .progress import (
        ProgressContext,
        build_progress_tools,
        collect_decision_file,
        decision_path,
        init_progress_file,
        latest_entry,
        mark_decision_dispatch,
    )
    from .roles import resolve_role_agent

    ctx = ProgressContext(path=job.root / "progress.yaml", current_iteration=1)
    init_progress_file(ctx.path)
    decision_file = None
    if workflow._decision_files:
        target = decision_path(job.root, "assembler", 1)
        decision_file = str(target)
        mark_decision_dispatch(job.root, "assembler", 1)
        prompt = (
            f"{prompt}\n\n---\n\n**Write your decision file to exactly this path when you "
            f"finish:**\n\n`{target}`\n"
        )
    layer = None
    if workflow._agent_transport is None:
        layer = workflow._make_agent(
            "assembler",
            workflow.prompts.assembler,
            required_tools=["append_assembler_progress"],
            session_mode="stateless",
        )
        config = getattr(layer, "config", None)
        if config is not None and not workflow._decision_files:
            from agent_flow import AgentLayer

            tools = build_progress_tools(ctx).get("assembler")
            layer = AgentLayer(replace(config, backend=replace(config.backend, tools=tools)))
    what = f"{job.unit} (early start, overlapping {job.after})"
    agent = resolve_role_agent(workflow.task_data, "assembler")
    request = TurnRequest(
        role="assembler",
        what=what,
        prompt=prompt,
        system_prompt=getattr(workflow.prompts, "assembler", ""),
        backend=agent.backend,
        model=agent.model,
        decision_file=decision_file,
        iteration=1,
    )
    report = workflow.journal.record_dispatch("assembler", what, prompt)
    try:
        executor = workflow._turns if layer is None else workflow._turns.bound(layer)
        text = executor.run(request).text
    finally:
        if layer is not None and hasattr(layer, "__exit__"):
            layer.__exit__(None, None, None)
    workflow.journal.record_result(report, text)
    if workflow._decision_files:
        collect_decision_file(ctx, job.root, "assembler", 1)
    return latest_entry(ctx.path, "assembler")


def maybe_start(workflow: Any, state: Any, unit: str, log: Any) -> bool:
    """After ``unit``'s UNIT_READY: start the next unit early, if that is safe."""
    from agent_flow.console import print_message

    from .state import ITEM_DONE
    from .task_schema import assembly_units

    overlap = _overlap_of(workflow)
    if overlap is None:
        return False
    units = assembly_units(workflow.task_data)
    if unit not in units or units.index(unit) + 1 >= len(units):
        return False
    nxt = units[units.index(unit) + 1]
    if nxt in state.units_done or nxt in overlap.jobs:
        return False
    owed = [row for row in state.queue if row.get("deferred_to") and row.get("status") != ITEM_DONE]
    reason = ""
    if owed:
        reason = f"{len(owed)} deferred smith dispatch(es) must land first"
    elif not workflow._target_marker():
        reason = "the target's commit could not be established"
    else:
        dirt = git(
            overlap.repo,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            workflow._preflight.target_rel,
            check=False,
        ).strip()
        if dirt:
            reason = "the target has uncommitted changes, which a worktree would not carry"
    if not reason:
        try:
            job = overlap.open_worktree(nxt, after=unit)
        except (OverlapError, GitError) as exc:
            reason = f"no worktree: {exc}"
    if reason:
        print_message(
            f"[bold yellow]→ overlap: not starting {nxt} early — {reason}[/bold yellow]", log
        )
        workflow.journal.event("overlap not started", f"{nxt}: {reason}")
        return False
    overlap.start(job, early_prompt(workflow, job))
    print_message(
        f"[bold cyan]→ overlap: {unit} is UNIT_READY; {nxt}'s assembler starts now in "
        f"{job.tree} while {unit} is gated[/bold cyan]",
        log,
    )
    workflow.journal.event(
        "overlap started",
        f"{nxt} in {job.tree} from {job.base[:10]}; native links: {len(job.linked)}",
    )
    return True


def adopt(workflow: Any, state: Any, unit: str, log: Any) -> bool:
    """Before ``unit``'s first turn: land its early result, or fall back to sequential.

    Returns True when adopted; the caller's ordinary resume path then sees an
    unchanged ``UNIT_READY`` and goes straight to the gates.
    """
    from agent_flow.console import print_message

    overlap = _overlap_of(workflow)
    if overlap is None:
        return False
    stale = overlap.stale(unit)
    if stale is not None:
        print_message(
            f"[bold yellow]→ overlap: {stale} was left by a previous process and is not "
            f"adopted; {unit} is assembled normally[/bold yellow]",
            log,
        )
        workflow.journal.event("overlap stale worktree ignored", str(stale))
        return False
    job = overlap.jobs.get(unit)
    if job is None:
        return False
    if job.thread is not None and job.thread.is_alive():
        print_message(
            f"[bold cyan]→ overlap: waiting for {unit}'s early turn to finish[/bold cyan]", log
        )
    overlap.wait(unit)
    outcome = overlap.integrate(job)
    if not outcome.adopted:
        note = overlap.keep_for_inspection(job, outcome.detail)
        print_message(
            f"[bold yellow]→ overlap: {unit}'s early start is not adopted ({outcome.detail}); "
            f"falling back to a sequential assembly. Kept for inspection: {note}[/bold yellow]",
            log,
        )
        workflow.journal.event("overlap fell back", f"{unit}: {outcome.detail}")
        return False
    entry = {**job.entry, "iteration": 1, "overlap": True, "unit": unit}
    entry.setdefault("agent", "assembler")
    workflow._progress_ctx.append("assembler", entry)
    state.unit_marker[unit] = workflow._target_marker()
    workflow._resumed_into_unit = True
    workflow._checkpoint(state)
    overlap.remove(job)
    print_message(
        f"[bold green]✔ overlap: {unit}'s early UNIT_READY adopted — {outcome.detail}[/bold green]",
        log,
    )
    workflow.journal.event("overlap adopted", f"{unit}: {outcome.detail}")
    return True


def _overlap_of(workflow: Any) -> UnitOverlap | None:
    """The run's overlap, built on first use; ``None`` unless ``--overlap-units``."""
    if not getattr(workflow, "_overlap_units", False):
        return None
    if getattr(workflow, "_overlap", None) is None:
        workflow._overlap = build_overlap(workflow)
    return workflow._overlap


def build_overlap(workflow: Any) -> UnitOverlap:
    """The run's :class:`UnitOverlap`, wired to the run's transport."""
    repo = workflow.task_data["trtllm_repo_path"]
    overlap = UnitOverlap(
        repo,
        workflow.workspace,
        policy=CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace),
    )
    overlap.turn_runner = lambda job, prompt: run_early_turn(workflow, job, prompt)
    return overlap


__all__ = [
    "AdoptOutcome",
    "BRANCH_PREFIX",
    "OVERLAP_DIRNAME",
    "OverlapError",
    "OverlapJob",
    "UnitOverlap",
    "adopt",
    "build_overlap",
    "early_prompt",
    "maybe_start",
    "native_artifacts",
    "run_early_turn",
]

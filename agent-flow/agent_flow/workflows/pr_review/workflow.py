from __future__ import annotations

import re
from pathlib import Path

from agent_flow import (
    CLAUDE_CODE_DEFAULT_MODEL,
    CODEX_DEFAULT_MODEL,
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
    require_tool_call_stop_hook,
)
from agent_flow.console import print_message, print_rule
from agent_flow.logger import get_logger

from . import vcs
from .discussion import DiscussionContext, build_discussion_tools
from .progress import (
    CODER_DECISIONS,
    REVIEWER_DECISIONS,
    ProgressContext,
    build_progress_tools,
    init_progress_file,
    latest_entry,
    read_progress,
)
from .prompts import DEFAULT_PROMPTS, SOURCING_SYSTEM_PROMPT, PromptBundle
from .sourcing import SourcingContext, build_sourcing_tools
from .state import (
    STAGE1,
    STAGE2,
    STAGE_S1_CODER,
    STAGE_S1_REVIEWER,
    STAGE_S2_CODER,
    STAGE_S2_REVIEWER,
    STATE_FILENAME,
    WorkflowState,
    load_state,
    save_state,
)

# Per-stage wiring: which sub-stage constants, progress list key, agent
# attributes, and the (reviewer, coder) backend pairing apply to each stage.
# Stage 1 reviews with Claude Code (coder = Codex); Stage 2 swaps the roles.
_STAGE_PLAN = {
    STAGE1: {
        "num": 1,
        "reviewer_stage": STAGE_S1_REVIEWER,
        "coder_stage": STAGE_S1_CODER,
        "reviewer_attr": "s1_reviewer",
        "coder_attr": "s1_coder",
        "next_stage": STAGE_S2_REVIEWER,
    },
    STAGE2: {
        "num": 2,
        "reviewer_stage": STAGE_S2_REVIEWER,
        "coder_stage": STAGE_S2_CODER,
        "reviewer_attr": "s2_reviewer",
        "coder_attr": "s2_coder",
        "next_stage": None,
    },
}


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Compose stop hooks that require *every* listed tool to be called.

    ``require_tool_call_stop_hook`` enforces "at least one of the listed names
    was called". Stacking one such hook per tool — each independent — yields
    AND semantics: every per-tool hook must allow the stop, so all listed
    tools must have been called this turn.
    """
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


def _make_agent(
    name: str,
    system_prompt: str,
    tools: list | None = None,
    required_tools: list[str] | None = None,
    backend_kind: str = "claude-code",
    model: str = CLAUDE_CODE_DEFAULT_MODEL,
    session_mode: str = "persistent",
) -> AgentLayer:
    hooks = _compose_required_tools_hooks(required_tools or [])
    return AgentLayer(
        AgentLayerConfig(
            name=name,
            system_prompt=system_prompt,
            backend=BackendConfig(kind=backend_kind, model=model, tools=tools, hooks=hooks),
            session=SessionConfig(mode=session_mode),
        )
    )


class PrReviewWorkflow:
    """Two-stage cross-model PR/MR review, built on ``AgentLayer``.

    Stage 1: reviewer = Claude Code, coder = Codex. Stage 2: reviewer = Codex,
    coder = Claude Code. Each stage cycles reviewer ⇄ coder rounds until they
    converge (reviewer APPROVE + coder AGREE) or the coder stands firm on a
    push-back (coder wins), then advances. The review conversation is kept
    local (``progress.yaml`` + ``discussion.md``); nothing is posted to the
    PR/MR, committed, or pushed.
    """

    def __init__(
        self,
        repo: Path,
        target: str | None = None,
        base: str | None = None,
        workspace: Path | None = None,
        num_rounds: int = 20,
        reviewer_context_reset_interval: int = 2,
        coder_context_reset_interval: int = 2,
        clean: bool = False,
        prompts: PromptBundle | None = None,
    ) -> None:
        self.repo = Path(repo)
        # The target is a GitHub PR or GitLab MR — a number or URL. We don't
        # care which platform; the sourcing agent figures that out and runs the
        # right CLI.
        self.identifier = (target or "").strip()
        if not self.identifier:
            raise ValueError("Provide the PR/MR to review (a number or URL); got an empty target.")
        self.base = base
        self.prompts = prompts or DEFAULT_PROMPTS

        if workspace is None:
            workspace = Path("workspace/pr-review") / _identifier_slug(self.identifier)
        self.workspace = Path(workspace)
        self.pr_context_path = self.workspace / "pr_context.md"
        self.progress_path = self.workspace / "progress.yaml"
        self.discussion_path = self.workspace / "discussion.md"
        self.state_path = self.workspace / STATE_FILENAME
        self.num_rounds = num_rounds
        self.reviewer_context_reset_interval = reviewer_context_reset_interval
        self.coder_context_reset_interval = coder_context_reset_interval

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            # Wipe the workflow's managed files so the constructor proceeds as
            # a fresh run. The user's repo / checkout is left alone.
            for path in (
                self.state_path,
                self.pr_context_path,
                self.progress_path,
                self.discussion_path,
            ):
                path.unlink(missing_ok=True)

        # Resume is auto-detected from the checkpoint's presence; ``--clean``
        # has just wiped it if the user wanted to start over.
        self.resume = self.state_path.is_file()

        if not self.resume:
            # On a fresh run, the managed conversation files must be empty so
            # we don't silently scribble over a prior run the user forgot
            # about. ``pr_context.md`` is exempt — it's regenerated from the
            # PR/MR in ``_init_state``.
            existing: list[Path] = []
            if _progress_has_entries(self.progress_path):
                existing.append(self.progress_path)
            if (
                self.discussion_path.is_file()
                and self.discussion_path.read_text(encoding="utf-8").strip()
            ):
                existing.append(self.discussion_path)
            if existing:
                names = ", ".join(p.name for p in existing)
                raise FileExistsError(
                    f"{names} already contains content in {self.workspace} but "
                    f"no checkpoint was found. Pass --clean to overwrite, or "
                    f"delete the file(s) manually to start fresh."
                )
            init_progress_file(self.progress_path)
            self.discussion_path.write_text("", encoding="utf-8")
            # pr_context.md is (re)written from the resolved PR/MR in ``run``.

        # The tool handlers close over these contexts; updating
        # ``current_stage`` / ``current_round`` before each agent call stamps
        # every entry without the agent having to pass them.
        self._progress_ctx = ProgressContext(path=self.progress_path)
        self._discussion_ctx = DiscussionContext(path=self.discussion_path)
        self._progress_tools = build_progress_tools(self._progress_ctx)
        self._discussion_tools = build_discussion_tools(self._discussion_ctx)

        # The sourcing agent checks the PR/MR out (running gh/glab itself — the
        # orchestrator never shells out to them) and reports its metadata into
        # this context via the ``report_pr_context`` tool. Read back in
        # ``_run_sourcing`` after the agent's turn.
        self._sourcing_ctx = SourcingContext()
        self.sourcing = _make_agent(
            "sourcing",
            SOURCING_SYSTEM_PROMPT,
            tools=build_sourcing_tools(self._sourcing_ctx),
            required_tools=["report_pr_context"],
            backend_kind="claude-code",
            model=CLAUDE_CODE_DEFAULT_MODEL,
            session_mode="stateless",
        )

        # (name, role, backend_kind, model) recipe per agent attribute, reused
        # by the constructor and ``_reset_agent``.
        self._agent_specs: dict[str, tuple[str, str, str, str]] = {
            "s1_reviewer": ("s1_reviewer", "reviewer", "claude-code", CLAUDE_CODE_DEFAULT_MODEL),
            "s1_coder": ("s1_coder", "coder", "codex", CODEX_DEFAULT_MODEL),
            "s2_reviewer": ("s2_reviewer", "reviewer", "codex", CODEX_DEFAULT_MODEL),
            "s2_coder": ("s2_coder", "coder", "claude-code", CLAUDE_CODE_DEFAULT_MODEL),
        }
        self.s1_reviewer = self._make_role_agent(*self._agent_specs["s1_reviewer"])
        self.s1_coder = self._make_role_agent(*self._agent_specs["s1_coder"])
        self.s2_reviewer = self._make_role_agent(*self._agent_specs["s2_reviewer"])
        self.s2_coder = self._make_role_agent(*self._agent_specs["s2_coder"])

    def _make_role_agent(self, name: str, role: str, backend_kind: str, model: str) -> AgentLayer:
        if role == "reviewer":
            tools = self._progress_tools["reviewer"] + self._discussion_tools["reviewer"]
            required = ["append_reviewer_progress", "update_discussion"]
            prompt = self.prompts.reviewer
        else:
            tools = self._progress_tools["coder"] + self._discussion_tools["coder"]
            required = ["append_coder_progress", "update_discussion"]
            prompt = self.prompts.coder
        return _make_agent(
            name,
            prompt,
            tools,
            required_tools=required,
            backend_kind=backend_kind,
            model=model,
        )

    def __enter__(self) -> "PrReviewWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for attr in (*self._agent_specs, "sourcing"):
            layer = getattr(self, attr, None)
            if isinstance(layer, AgentLayer):
                layer.__exit__(None, None, None)

    # ------------------------------------------------------------- orchestration

    def run(self) -> None:
        log = get_logger().console

        state = self._init_state(log)
        if state is None:
            return

        try:
            for stage_key in (STAGE1, STAGE2):
                plan = _STAGE_PLAN[stage_key]
                if state.stage in (plan["reviewer_stage"], plan["coder_stage"]):
                    self._run_review_stage(state, stage_key, log)

            if not state.done:
                state.done = True
                self._checkpoint(state)
            print_message("[bold green]✔ pr-review complete[/bold green]", log)
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — run again to continue from the "
                "last checkpoint, or pass --clean to start fresh[/bold yellow]",
                log,
            )
            raise

    def _run_review_stage(self, state: WorkflowState, stage_key: str, log) -> None:
        """Drive one review stage's reviewer ⇄ coder loop to a conclusion."""
        plan = _STAGE_PLAN[stage_key]
        reviewer_stage = plan["reviewer_stage"]
        coder_stage = plan["coder_stage"]
        reviewer_attr = plan["reviewer_attr"]
        coder_attr = plan["coder_attr"]
        stage_num = plan["num"]
        next_stage = plan["next_stage"]

        for i in range(state.next_round_index, self.num_rounds):
            round_no = i + 1
            print_rule(
                f"[bold cyan]Stage {stage_num} · Round {round_no}/{self.num_rounds}[/bold cyan]",
                log,
            )

            # Each sub-stage checkpoints before advancing, so a crash / Ctrl-C
            # mid-round resumes at the same agent rather than rerunning the
            # other one.
            if state.stage == reviewer_stage:
                if self._should_reset(i, self.reviewer_context_reset_interval):
                    self._reset_agent(reviewer_attr)
                self._run_reviewer(stage_key, round_no, reviewer_attr)
                state.stage = coder_stage
                self._checkpoint(state)

            if state.stage == coder_stage:
                if self._should_reset(i, self.coder_context_reset_interval):
                    self._reset_agent(coder_attr)
                self._run_coder(stage_key, round_no, coder_attr)
                reviewer_decision = self._latest_reviewer_decision(stage_key)
                coder_decision = self._latest_coder_decision(stage_key)
                state.next_round_index = i + 1

                if coder_decision == "STAND_FIRM":
                    print_message(
                        f"[bold yellow]⚑ stage {stage_num}: coder stands firm — "
                        f"push-back is final, advancing[/bold yellow]",
                        log,
                    )
                    self._advance_after_stage(state, next_stage)
                    return

                if reviewer_decision == "APPROVE" and coder_decision == "AGREE":
                    print_message(
                        f"[bold green]✔ stage {stage_num}: reviewer APPROVE + coder "
                        f"AGREE — converged at round {round_no}[/bold green]",
                        log,
                    )
                    self._advance_after_stage(state, next_stage)
                    return

                # Not yet settled — loop back to the reviewer for another round.
                print_message(
                    f"[bold yellow]↻ stage {stage_num}: reviewer "
                    f"{reviewer_decision or 'missing'} / coder "
                    f"{coder_decision or 'missing'} — another round[/bold yellow]",
                    log,
                )
                state.stage = reviewer_stage
                self._checkpoint(state)

        # Round budget exhausted without a settled outcome: advance anyway so a
        # stale checkpoint doesn't auto-resume back into the loop. The coder's
        # latest state stands (it always has the final say).
        print_message(
            f"[bold yellow]⚠ stage {stage_num}: round budget "
            f"({self.num_rounds}) exhausted — advancing with the current "
            f"state[/bold yellow]",
            log,
        )
        self._advance_after_stage(state, next_stage)

    def _advance_after_stage(self, state: WorkflowState, next_stage: str | None) -> None:
        """Transition out of a finished stage (to the next one, or to done)."""
        if next_stage is None:
            state.done = True
            self._checkpoint(state)
        else:
            state.stage = next_stage
            state.next_round_index = 0
            self._checkpoint(state)

    def _init_state(self, log) -> WorkflowState | None:
        """Load or create the workflow state; return ``None`` to no-op.

        On a fresh run the sourcing agent checks the PR/MR out and reports its
        metadata (it runs ``gh``/``glab`` itself); the orchestrator then derives
        the diff context with local ``git`` and writes ``pr_context.md``. On
        resume, the on-disk ``pr_context.md`` is authoritative and the PR/MR is
        not re-sourced.
        """
        if self.resume:
            state = load_state(self.state_path)
            if state.done:
                print_message(
                    "[bold green]✔ workflow already completed; pass --clean to "
                    "rerun from scratch[/bold green]",
                    log,
                )
                return None
            return state

        print_rule(
            f"[bold cyan]Sourcing {self.identifier}[/bold cyan]",
            log,
        )
        metadata = self._run_sourcing()
        pr_ctx = vcs.build_pr_context(self.repo, self.identifier, metadata, self.base)
        self.pr_context_path.write_text(vcs.format_pr_context_md(pr_ctx), encoding="utf-8")
        print_message(
            f"[bold cyan]→ reviewing {self.identifier} "
            f"({pr_ctx.head_branch or '?'} → {pr_ctx.base_branch}); diff against "
            f"{pr_ctx.diff_base_ref}[/bold cyan]",
            log,
        )
        try:
            stat = vcs.diff_stat(self.repo, pr_ctx.diff_base_ref).strip()
            if stat:
                print_message(stat, log)
        except vcs.VcsError:
            # The diff summary is best-effort; the agents recompute the diff
            # themselves, so a failure here must not block the run.
            pass

        state = WorkflowState(
            pr_context_path=str(self.pr_context_path),
            num_rounds=self.num_rounds,
            stage=STAGE_S1_REVIEWER,
        )
        # Checkpoint before running the first stage so a crash mid-stage can be
        # picked up on the next run.
        self._checkpoint(state)
        return state

    def _run_sourcing(self) -> dict[str, str]:
        """Drive the sourcing agent to check the PR/MR out and report its metadata.

        The agent detects whether the target is a GitHub PR or GitLab MR and runs
        the matching CLI itself (the orchestrator no longer does); it reports back
        through the ``report_pr_context`` tool, which fills ``self._sourcing_ctx``.
        Returns the reported metadata dict; raises if the agent never reported (so
        we don't proceed without a base branch).
        """
        self._sourcing_ctx.reset()
        base_line = (
            f"The operator fixed the base branch to `{self.base}`; still report "
            f"the branch the PR/MR actually targets.\n"
            if self.base
            else ""
        )
        self.sourcing(
            f"Source the pull/merge request `{self.identifier}` into the local "
            f"repo at `{self.repo}`.\n\n"
            f"1. Determine whether it is a GitHub PR (use `gh`) or a GitLab MR "
            f"(use `glab`) — from the URL, or from the repo's remotes.\n"
            f"2. Check it out so its changes land in the working tree (e.g. "
            f"`gh pr checkout {self.identifier}` or `glab mr checkout "
            f"{self.identifier}`, run inside `{self.repo}`).\n"
            f"3. Read its metadata — base and head branches, title, author, URL, "
            f"description.\n"
            f"4. Call `report_pr_context` once, as your last action, with what "
            f"you found.\n"
            f"{base_line}\n"
            f"Do NOT post to the PR/MR, and do NOT commit or push."
        )
        if not self._sourcing_ctx.reported:
            raise vcs.VcsError(
                "the sourcing agent finished without calling report_pr_context, "
                "so the PR/MR base branch is unknown. Re-run, or pass --base."
            )
        return self._sourcing_ctx.as_metadata()

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    # --------------------------------------------------- context reset helpers

    @staticmethod
    def _should_reset(i: int, interval: int) -> bool:
        if i == 0 or interval <= 0:
            return False
        return i % interval == 0

    def _reset_agent(self, attr: str) -> None:
        layer = getattr(self, attr, None)
        if not isinstance(layer, AgentLayer):
            return
        layer.__exit__(None, None, None)
        setattr(self, attr, self._make_role_agent(*self._agent_specs[attr]))

    # ---------------------------------------------------- decision helpers

    def _latest_reviewer_decision(self, stage_key: str) -> str | None:
        entry = latest_entry(self.progress_path, stage_key, "reviewer")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in REVIEWER_DECISIONS else None

    def _latest_coder_decision(self, stage_key: str) -> str | None:
        entry = latest_entry(self.progress_path, stage_key, "coder")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in CODER_DECISIONS else None

    # ------------------------------------------------------------------ agents

    def _run_reviewer(self, stage_key: str, round_no: int, agent_attr: str) -> None:
        self._progress_ctx.current_stage = stage_key
        self._progress_ctx.current_round = round_no
        stage_num = _STAGE_PLAN[stage_key]["num"]
        agent = getattr(self, agent_attr)
        agent(
            f"Repo under review: {self.repo}\n"
            f"Review stage {stage_num}, round {round_no}.\n\n"
            f"Start by calling `read_discussion` to load the local review "
            f"conversation so far.\n\n"
            f"Read `{self.pr_context_path}` for the PR/MR under review and the "
            f"exact diff command. Run that command — and read the changed "
            f"files and build/run/test as needed — to judge the **current** "
            f"change (it includes the coder's uncommitted edits from earlier "
            f"rounds).\n\n"
            f'Call `read_latest_progress` with `agent: "coder"` to see the '
            f"Coder's latest response — what they changed and what they pushed "
            f"back on. Engage with it: concede points the Coder rebutted well; "
            f"hold firm only with concrete, actionable justification.\n\n"
            f"Decide APPROVE (no outstanding requests) or REQUEST_CHANGES "
            f"(specific, actionable items). Keep the conversation local — do "
            f"NOT post to the PR/MR, and do NOT commit or push.\n\n"
            f"Before finishing, call **both** required tools: "
            f"`append_reviewer_progress` (with `summary` and `decision`) and "
            f"`update_discussion` (refresh the open threads and their status)."
        )

    def _run_coder(self, stage_key: str, round_no: int, agent_attr: str) -> None:
        self._progress_ctx.current_stage = stage_key
        self._progress_ctx.current_round = round_no
        stage_num = _STAGE_PLAN[stage_key]["num"]
        agent = getattr(self, agent_attr)
        agent(
            f"Repo under review: {self.repo}\n"
            f"Review stage {stage_num}, round {round_no}.\n\n"
            f"Start by calling `read_discussion` to load the local review "
            f"conversation so far.\n\n"
            f"Read `{self.pr_context_path}` for the PR/MR under review and the "
            f"exact diff command; run it to see the current change.\n\n"
            f'Call `read_latest_progress` with `agent: "reviewer"` to fetch '
            f"the Reviewer's latest comments. For each item, either **address "
            f"it** by editing the working tree, or **push back** with a "
            f"recorded rationale — never make a change you disagree with. "
            f"Build/run/test what you change.\n\n"
            f"Keep the conversation local — do NOT post to the PR/MR, and do "
            f"NOT commit or push; leave edits in the working tree.\n\n"
            f"Decide REVISE (made changes, please re-review), AGREE (addressed "
            f"everything; you believe it's good), or STAND_FIRM (you decline "
            f"the remaining requests with rationale — final, ends the stage in "
            f"your favor).\n\n"
            f"Before finishing, call **both** required tools: "
            f"`append_coder_progress` (with `summary` and `decision`) and "
            f"`update_discussion`."
        )


def _identifier_slug(identifier: str) -> str:
    """Make a filesystem-safe workspace slug from a PR/MR number or URL.

    A bare number passes through unchanged (``"42"`` → ``"42"``); a URL is
    reduced to a readable, path-safe form so each PR/MR still gets its own
    isolated workspace directory.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", identifier).strip("-")
    return slug or "target"


def _progress_has_entries(path: Path) -> bool:
    """Return True iff ``path`` holds a progress.yaml with real entries.

    A shell ``{stage1: [], stage2: []}`` left behind by ``init_progress_file``
    does not count (a retry should not be blocked by it). Missing/empty files
    count as no entries. A malformed file ``read_progress`` rejects counts as
    "has content" so the user is forced to ``--clean`` rather than silently
    losing it.
    """
    if not path.is_file():
        return False
    try:
        data = read_progress(path)
    except ValueError:
        return True
    return bool(data[STAGE1] or data[STAGE2])


if __name__ == "__main__":
    from .cli import main

    main()

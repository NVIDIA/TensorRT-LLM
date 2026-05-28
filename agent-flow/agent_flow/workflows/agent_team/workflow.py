from __future__ import annotations

import shutil
from pathlib import Path

from agent_flow import (CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL,
                        AgentLayer, AgentLayerConfig, BackendConfig,
                        SessionConfig, require_tool_call_stop_hook)
from agent_flow.console import print_message, print_rule
from agent_flow.logger import get_logger

from .progress import (BUILD_STAGE, PLAN_STAGE, ProgressContext,
                       append_human_feedback, build_progress_tools,
                       init_progress_file, latest_entry)
from .prompts import DEFAULT_PROMPTS, PromptBundle
from .state import (STAGE_CODER, STAGE_PLAN_DRAFTER, STAGE_PLAN_HUMAN,
                    STAGE_PLAN_REVIEWER, STAGE_QA, STAGE_REVIEWER,
                    STATE_FILENAME, WorkflowState, load_state, save_state)
from .status import StatusContext, build_status_tools

_PLAN_STAGES = (STAGE_PLAN_DRAFTER, STAGE_PLAN_REVIEWER, STAGE_PLAN_HUMAN)


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Compose stop hooks that require *every* listed tool to be called.

    ``require_tool_call_stop_hook`` enforces "at least one of the listed
    names was called". Stacking one such hook per tool — each independent —
    yields AND semantics: every per-tool hook must allow the stop, so all
    listed tools must have been called this turn.
    """
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


def _make_agent(name: str,
                system_prompt: str,
                tools: list | None = None,
                required_tools: list[str] | None = None,
                backend_kind: str = "claude-code",
                model: str = CLAUDE_CODE_DEFAULT_MODEL,
                session_mode: str = "persistent",
                human_input_enabled: bool = False) -> AgentLayer:
    hooks = _compose_required_tools_hooks(required_tools or [])
    return AgentLayer(
        AgentLayerConfig(
            name=name,
            system_prompt=system_prompt,
            backend=BackendConfig(kind=backend_kind,
                                  model=model,
                                  tools=tools,
                                  hooks=hooks),
            session=SessionConfig(mode=session_mode),
            human_input_enabled=human_input_enabled,
        ))


class AgentTeamWorkflow:
    """Plan phase (drafter ↔ reviewer ↔ human) then build phase (coder ↔
    reviewer ↔ qa) loop, both built on AgentLayer."""

    def __init__(
        self,
        workspace: Path,
        num_iterations: int = 100,
        coder_context_reset_interval: int = 2,
        reviewer_context_reset_interval: int = 2,
        min_score: float = 8.0,
        plan_human_review_enabled: bool = False,
        build_human_review_enabled: bool = False,
        clean: bool = False,
        plan: str | Path | None = None,
        acceptance_criteria: str | Path | None = None,
        feedback: str | Path | None = None,
        prompts: PromptBundle | None = None,
    ) -> None:
        self.workspace = workspace
        self.prompts = prompts or DEFAULT_PROMPTS
        self.task_path = workspace / "task.yaml"
        self.plan_path = workspace / "plan.md"
        self.acceptance_criteria_path = workspace / "acceptance-criteria.md"
        self.progress_path = workspace / "progress.yaml"
        self.status_path = workspace / "status.md"
        self.state_path = workspace / STATE_FILENAME
        self.num_iterations = num_iterations
        self.coder_context_reset_interval = coder_context_reset_interval
        self.reviewer_context_reset_interval = reviewer_context_reset_interval
        self.min_score = min_score
        self.plan_human_review_enabled = plan_human_review_enabled
        self.build_human_review_enabled = build_human_review_enabled
        # When set on a fresh run, the matching plan-phase output is
        # populated from this value. If *both* presets are provided the
        # plan phase is skipped entirely and the workflow starts at the
        # Coder; if only one is provided the plan phase still runs so
        # the PlanDrafter fills in the missing file. Ignored when
        # resuming from a checkpoint (on-disk files are preserved).
        self.preset_plan = plan
        self.preset_acceptance_criteria = acceptance_criteria
        # Free-form human guidance the user wants the build-phase agents
        # to take into account. May be a literal string or a path to a
        # file. Appended (not overwritten) to ``progress.yaml``'s
        # ``human_feedback`` list once state is loaded, so re-running
        # with another ``--feedback`` adds another entry.
        self.pending_feedback = feedback

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            # Wipe the workflow's managed files so the constructor
            # proceeds as a fresh run. Other files in the workspace
            # (user code, build outputs) are left alone.
            for path in (self.state_path, self.plan_path,
                         self.acceptance_criteria_path, self.progress_path,
                         self.status_path):
                path.unlink(missing_ok=True)

        # Resume is auto-detected from the checkpoint's presence;
        # ``--clean`` has just wiped it if the user wanted to start over.
        self.resume = self.state_path.is_file()

        # ``--feedback`` only makes sense when resuming: it appends a
        # ``human_feedback`` entry stamped with the *upcoming* iteration so
        # the build-phase agents read it on their next turn. On a fresh run
        # there are no prior iterations to course-correct, and the entry
        # would land before the plan phase has even produced a plan — silently
        # ignored by the build agents that consume it. Reject the
        # combination so the user notices the mismatch; any guidance the
        # user wants the workflow to honor from the start belongs in the
        # task description (--task), not in --feedback.
        if feedback is not None and not self.resume:
            raise ValueError(
                "--feedback is only valid when resuming from a checkpoint, "
                "and no checkpoint was found in the workspace. If you have "
                "guidance you want the workflow to honor from the start of "
                "the task, put it in the task description (--task) instead. "
                "Use --feedback only after a prior run has produced a "
                "checkpoint, to course-correct subsequent iterations.")

        if not self.resume:
            # A preset will overwrite the matching file on purpose, so a
            # non-empty file is fine in that case; everything else must
            # still be empty.
            guarded = [self.progress_path, self.status_path]
            if plan is None:
                guarded.insert(0, self.plan_path)
            if acceptance_criteria is None:
                guarded.insert(0, self.acceptance_criteria_path)
            existing = [
                p for p in guarded
                if p.is_file() and p.read_text(encoding="utf-8").strip()
            ]
            if existing:
                names = ", ".join(p.name for p in existing)
                raise FileExistsError(
                    f"{names} already contains content in {self.workspace} "
                    f"but no checkpoint was found. Pass --clean to "
                    f"overwrite, or delete the file(s) manually to start "
                    f"fresh.")

            if plan is None:
                self.plan_path.write_text("", encoding="utf-8")
            if acceptance_criteria is None:
                self.acceptance_criteria_path.write_text("", encoding="utf-8")
            # When a preset is provided, the matching file is
            # materialized in ``_init_state`` alongside task.yaml.
            init_progress_file(self.progress_path)
            self.status_path.write_text("", encoding="utf-8")
            # task.yaml is (re)written from the resolved task in ``run``.

        # The tool handlers close over this context; updating
        # ``current_iteration`` before each agent call stamps every entry
        # with the right iteration without the agent having to pass it.
        self._progress_ctx = ProgressContext(path=self.progress_path)
        progress_tools = build_progress_tools(self._progress_ctx)

        # Coder and Reviewer share status.md as a rolling scratchpad. The
        # tools mutate ``status.md`` directly; no per-iteration context is
        # needed because the file is overwritten each turn.
        self._status_ctx = StatusContext(path=self.status_path)
        status_tools = build_status_tools(self._status_ctx)

        # PlanDrafter has ``human_input_enabled=True`` unconditionally so
        # it can call the ``ask_human`` MCP tool during the human-review
        # phase. The same agent (and persistent session) is used in the
        # draft phase, but the prompt instructs it not to call
        # ``ask_human`` there.
        #
        # The Coder also gets ``ask_human`` — but only when
        # ``build_human_review_enabled`` is True (opt-in via
        # ``--build-human-review``). The two flags are independent and
        # both off by default: ``plan_human_review_enabled`` (opt-in
        # via ``--plan-human-review``) gates the plan-stage human
        # checkpoint; ``build_human_review_enabled`` gates the Coder's
        # mid-build escape hatch.
        self.plan_drafter = _make_agent(
            "plan_drafter",
            self.prompts.plan_drafter,
            progress_tools["plan_drafter"],
            required_tools=["append_plan_drafter_progress"],
            human_input_enabled=True,
            backend_kind="codex",
            model=CODEX_DEFAULT_MODEL,
        )
        self.plan_reviewer = _make_agent(
            "plan_reviewer",
            self.prompts.plan_reviewer,
            progress_tools["plan_reviewer"],
            required_tools=["append_plan_reviewer_progress"],
        )
        self.coder = _make_agent(
            "coder",
            self.prompts.coder,
            progress_tools["coder"] + status_tools["coder"],
            required_tools=["append_coder_progress", "update_status"],
            human_input_enabled=self.build_human_review_enabled,
        )
        self.reviewer = _make_agent(
            "reviewer",
            self.prompts.reviewer,
            progress_tools["reviewer"] + status_tools["reviewer"],
            required_tools=["append_reviewer_progress", "update_status"],
            backend_kind="codex",
            model=CODEX_DEFAULT_MODEL,
        )
        # QA is stateless so each iteration starts with a fresh session —
        # no carry-over bias from prior runs. The handler set is also
        # narrower (append only, no read_latest_progress, no status.md) so
        # QA can only ground its verdict in task.yaml and
        # acceptance-criteria.md.
        self.qa = _make_agent(
            "qa",
            self.prompts.qa,
            progress_tools["qa"],
            required_tools=["append_qa_progress"],
            session_mode="stateless",
        )
        self._progress_tools = progress_tools
        self._status_tools = status_tools

    def __enter__(self) -> "AgentTeamWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for layer in (self.plan_drafter, self.plan_reviewer, self.coder,
                      self.reviewer, self.qa):
            layer.__exit__(None, None, None)

    # ------------------------------------------------------------- orchestration

    def run(self, task: str) -> None:
        log = get_logger().console

        state = self._init_state(task, log)
        if state is None:
            return

        self._record_pending_feedback(state, log)

        try:
            # ----- PLAN PHASE -----
            if state.stage in _PLAN_STAGES:
                self._run_plan_phase(state, log)

            # ----- BUILD PHASE -----
            for i in range(state.next_iteration_index, self.num_iterations):
                iteration = i + 1
                print_rule(
                    f"[bold cyan]Iteration {iteration}/{self.num_iterations}"
                    f"[/bold cyan]",
                    log,
                )

                # Each stage checkpoints before advancing, so a crash/Ctrl-C
                # mid-iteration resumes at the same agent rather than
                # rerunning earlier stages.
                if state.stage == STAGE_CODER:
                    if self._should_reset_coder(i):
                        self._reset_coder()
                    self._run_coder(iteration)
                    state.stage = STAGE_REVIEWER
                    self._checkpoint(state)

                if state.stage == STAGE_REVIEWER:
                    if self._should_reset_reviewer(i):
                        self._reset_reviewer()
                    self._run_reviewer(iteration)
                    decision = self._latest_reviewer_decision()
                    if decision != "APPROVE":
                        # REJECT (or missing) → loop back to coder, skip QA.
                        print_message(
                            f"[bold yellow]↻ reviewer {decision or 'missing'} "
                            f"— looping back to coder[/bold yellow]",
                            log,
                        )
                        state.next_iteration_index = i + 1
                        state.stage = STAGE_CODER
                        self._checkpoint(state)
                        continue
                    state.stage = STAGE_QA
                    self._checkpoint(state)

                if state.stage == STAGE_QA:
                    # QA is stateless, so its "reset every iteration" need
                    # is satisfied by SessionConfig; no explicit reset call.
                    self._run_qa(iteration)
                    decision = self._latest_qa_decision()
                    score = self._latest_qa_score()
                    state.next_iteration_index = i + 1
                    score_ok = (self.min_score <= 0 or
                                (score is not None and score >= self.min_score))
                    if decision == "APPROVE" and score_ok:
                        state.done = True
                        state.stage = STAGE_CODER
                        self._checkpoint(state)
                        print_message(
                            f"[bold green]✔ QA APPROVE at iteration "
                            f"{iteration}[/bold green]",
                            log,
                        )
                        break
                    if (decision == "APPROVE" and score is not None
                            and score < self.min_score):
                        print_message(
                            f"[bold yellow]↻ qa APPROVE but score "
                            f"{score:.2f} < {self.min_score:.2f} — looping "
                            f"back to coder[/bold yellow]",
                            log,
                        )
                    else:
                        print_message(
                            f"[bold yellow]↻ qa {decision or 'missing'} — "
                            f"looping back to coder[/bold yellow]",
                            log,
                        )
                    state.stage = STAGE_CODER
                    self._checkpoint(state)

            # Budget exhausted without an APPROVE: mark the run done so a
            # stale checkpoint doesn't auto-resume back into the loop.
            if not state.done:
                state.done = True
                self._checkpoint(state)
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — run again to continue from "
                "the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            raise

    def _run_plan_phase(self, state: WorkflowState, log) -> None:
        """Drive the plan phase to completion.

        Loops indefinitely until the human approves (or the AI
        PlanReviewer APPROVEs and human review is disabled), at which
        point ``state.stage`` is advanced to ``STAGE_CODER``.
        """
        j = state.plan_next_iteration_index
        while True:
            plan_iter = j + 1
            print_rule(
                f"[bold magenta]Plan iter {plan_iter}[/bold magenta]",
                log,
            )

            if state.stage == STAGE_PLAN_DRAFTER:
                self._run_plan_drafter(plan_iter, mode="draft")
                state.stage = STAGE_PLAN_REVIEWER
                self._checkpoint(state)

            if state.stage == STAGE_PLAN_REVIEWER:
                self._run_plan_reviewer(plan_iter)
                decision = self._latest_plan_reviewer_decision()
                if decision != "APPROVE":
                    print_message(
                        f"[bold yellow]↻ plan_reviewer "
                        f"{decision or 'missing'} — looping back to "
                        f"plan_drafter[/bold yellow]",
                        log,
                    )
                    state.plan_next_iteration_index = j + 1
                    state.stage = STAGE_PLAN_DRAFTER
                    self._checkpoint(state)
                    j += 1
                    continue
                if not self.plan_human_review_enabled:
                    print_message(
                        "[bold green]✔ plan_reviewer APPROVE — entering "
                        "build phase (plan-stage human review "
                        "disabled)[/bold green]",
                        log,
                    )
                    state.stage = STAGE_CODER
                    state.next_iteration_index = 0
                    self._checkpoint(state)
                    return
                state.stage = STAGE_PLAN_HUMAN
                self._checkpoint(state)

            if state.stage == STAGE_PLAN_HUMAN:
                if not self.plan_human_review_enabled:
                    # Resumed from a STAGE_PLAN_HUMAN checkpoint with
                    # plan-stage human review now disabled — skip
                    # straight to the build phase rather than calling
                    # ask_human.
                    state.stage = STAGE_CODER
                    state.next_iteration_index = 0
                    self._checkpoint(state)
                    return
                self._run_plan_drafter(plan_iter, mode="human")
                decision = self._latest_plan_drafter_decision()
                if decision == "HUMAN_APPROVED":
                    print_message(
                        "[bold green]✔ human approved plan — entering "
                        "build phase[/bold green]",
                        log,
                    )
                    state.stage = STAGE_CODER
                    state.next_iteration_index = 0
                    self._checkpoint(state)
                    return
                # POLISHING / unset / DRAFT_READY: re-invoke the drafter
                # in human mode for another round. The AI PlanReviewer is
                # intentionally not re-run here — the human is the final
                # arbiter once they take over.
                print_message(
                    f"[bold yellow]↻ plan_drafter "
                    f"{decision or 'missing'} (human stage) — re-invoking"
                    f"[/bold yellow]",
                    log,
                )
                state.plan_next_iteration_index = j + 1
                state.stage = STAGE_PLAN_HUMAN
                self._checkpoint(state)

            j += 1

    def _init_state(self, task: str | Path, log) -> WorkflowState | None:
        """Load or create the workflow state; return ``None`` to no-op.

        ``task`` is a path to the YAML file containing the task spec. The
        file is copied verbatim into ``workspace/task.yaml`` on fresh runs;
        on resume, the on-disk file is authoritative and the supplied path
        is read only to detect drift.
        """
        source = Path(task)
        if not source.is_file():
            raise FileNotFoundError(
                f"--task must be a path to an existing YAML file; got "
                f"{source}")
        task_text = source.read_text(encoding="utf-8").strip()

        if self.resume:
            state = load_state(self.state_path)
            if state.done:
                if self.pending_feedback is None:
                    print_message(
                        "[bold green]✔ workflow already completed; pass "
                        "--clean to rerun from scratch, or pass "
                        "--feedback to resume with new guidance."
                        "[/bold green]",
                        log,
                    )
                    return None
                # Re-engage the build phase so the agents can address the
                # new --feedback the user just supplied. The pending
                # feedback itself is appended by ``_record_pending_feedback``
                # right after this method returns.
                #
                # If the prior run exhausted its budget (or QA accepted on
                # the final iteration), ``next_iteration_index`` already
                # equals ``num_iterations`` and the build loop's
                # ``range(next_iteration_index, num_iterations)`` would be
                # empty — the feedback would be appended and the workflow
                # immediately marked done again. Grant a fresh budget of
                # ``self.num_iterations`` more iterations starting from
                # the prior tail so the audit log stays monotonic and the
                # build loop actually runs. When the prior run finished
                # under-budget (e.g. QA accepted at iter 50 of 100), the
                # remaining slack is left intact.
                print_message(
                    "[bold cyan]→ workflow was complete; resuming build "
                    "phase to address new --feedback.[/bold cyan]",
                    log,
                )
                state.done = False
                state.stage = STAGE_CODER
                if state.next_iteration_index >= self.num_iterations:
                    self.num_iterations = (state.next_iteration_index +
                                           self.num_iterations)
                    state.num_iterations = self.num_iterations
                self._checkpoint(state)
            if (self.preset_plan is not None
                    or self.preset_acceptance_criteria is not None):
                print_message(
                    "[bold yellow]⚠ --plan / --acceptance-criteria "
                    "ignored while resuming from checkpoint; pass --clean "
                    "to start fresh.[/bold yellow]",
                    log,
                )
            # task.yaml on disk is the source of truth on resume; compare it
            # against the --task the user just passed to flag drift.
            checkpoint_task_path = Path(state.task_path)
            if checkpoint_task_path.is_file():
                checkpoint_text = checkpoint_task_path.read_text(
                    encoding="utf-8").strip()
                if checkpoint_text and checkpoint_text != task_text:
                    print_message(
                        "[bold yellow]⚠ --task differs from the "
                        "checkpointed task; using the checkpointed task to "
                        "preserve context.[/bold yellow]",
                        log,
                    )
            return state

        self._materialize_task(source)

        if self.preset_plan is not None:
            self._materialize_plan(self._resolve_plan_input(self.preset_plan))
        if self.preset_acceptance_criteria is not None:
            self._materialize_acceptance_criteria(
                self._resolve_acceptance_criteria_input(
                    self.preset_acceptance_criteria))

        initial_stage = STAGE_PLAN_DRAFTER
        if (self.preset_plan is not None
                and self.preset_acceptance_criteria is not None):
            print_message(
                "[bold cyan]→ using user-supplied plan and acceptance "
                "criteria; skipping plan phase and starting at coder"
                "[/bold cyan]",
                log,
            )
            initial_stage = STAGE_CODER
        elif self.preset_plan is not None:
            print_message(
                "[bold cyan]→ user-supplied plan loaded; running plan "
                "phase to generate acceptance-criteria.md[/bold cyan]",
                log,
            )
        elif self.preset_acceptance_criteria is not None:
            print_message(
                "[bold cyan]→ user-supplied acceptance criteria loaded; "
                "running plan phase to generate plan.md[/bold cyan]",
                log,
            )

        state = WorkflowState(task_path=str(self.task_path),
                              num_iterations=self.num_iterations,
                              stage=initial_stage)
        # Checkpoint before running the plan phase so a crash mid-plan can
        # be picked up on the next run.
        self._checkpoint(state)
        return state

    def _record_pending_feedback(self, state: WorkflowState, log) -> None:
        """Append the user's ``--feedback`` argument to ``progress.yaml``.

        The CLI flag is provided per-invocation, so re-running with another
        ``--feedback`` appends another entry rather than overwriting prior
        ones. The entry is stamped with the upcoming iteration and the
        currently-active stage so build-phase agents can ground their next
        turn on it.
        """
        if self.pending_feedback is None:
            return
        text = self._resolve_feedback_input(self.pending_feedback)
        if not text.strip():
            print_message(
                "[bold yellow]⚠ --feedback was empty; nothing recorded."
                "[/bold yellow]",
                log,
            )
            return
        if state.stage in _PLAN_STAGES:
            stage_key = PLAN_STAGE
            iteration = state.plan_next_iteration_index + 1
        else:
            stage_key = BUILD_STAGE
            iteration = state.next_iteration_index + 1
        append_human_feedback(
            self.progress_path,
            summary=text.rstrip() + "\n",
            iteration=iteration,
            stage=stage_key,
        )
        print_message(
            f"[bold cyan]→ recorded human_feedback (stage={stage_key}, "
            f"iteration={iteration}); build-phase agents will read it on "
            f"their next turn.[/bold cyan]",
            log,
        )

    @staticmethod
    def _resolve_feedback_input(feedback: str | Path) -> str:
        """Return the literal feedback text.

        If ``feedback`` is a ``Path`` (or a string that resolves to an
        existing file), return its contents; otherwise treat the string
        as the literal feedback text. ``--feedback`` accepts free-form
        notes, so a long literal string can exceed the OS path-length
        limit and make ``Path.is_file`` raise ``OSError``; treat any
        path-probe error as "not a file" and fall back to the literal.
        """
        if isinstance(feedback, Path):
            return feedback.read_text(encoding="utf-8")
        try:
            candidate = Path(feedback)
            is_file = candidate.is_file()
        except OSError:
            return feedback
        if is_file:
            return candidate.read_text(encoding="utf-8")
        return feedback

    @staticmethod
    def _resolve_plan_input(plan: str | Path) -> str | Path:
        """Decide whether ``plan`` is a path-to-existing-file or content.

        Mirrors the same rule as ``--task``: if the value is a path that
        resolves to an existing file, treat it as the source to copy;
        otherwise treat it as the literal plan text.
        """
        if isinstance(plan, Path):
            return plan
        candidate = Path(plan)
        return candidate if candidate.is_file() else plan

    @staticmethod
    def _resolve_acceptance_criteria_input(criteria: str | Path) -> str | Path:
        """Same path-or-content resolution rule as ``_resolve_plan_input``."""
        if isinstance(criteria, Path):
            return criteria
        candidate = Path(criteria)
        return candidate if candidate.is_file() else criteria

    def _materialize_task(self, source: Path) -> None:
        """Copy ``source`` verbatim into ``workspace/task.yaml``."""
        if source.resolve() != self.task_path.resolve():
            shutil.copyfile(source, self.task_path)

    def _materialize_plan(self, source: str | Path) -> None:
        """Populate ``workspace/plan.md`` from a preset plan.

        Mirrors ``_materialize_task``: a ``Path`` is copied verbatim, a
        string is written directly. Only used when ``--plan`` is set on a
        fresh run.
        """
        if isinstance(source, Path):
            if source.resolve() != self.plan_path.resolve():
                shutil.copyfile(source, self.plan_path)
        else:
            self.plan_path.write_text(source.rstrip() + "\n", encoding="utf-8")

    def _materialize_acceptance_criteria(self, source: str | Path) -> None:
        """Populate ``workspace/acceptance-criteria.md`` from a preset.

        Mirrors ``_materialize_plan``. Only used when
        ``--acceptance-criteria`` is set on a fresh run.
        """
        if isinstance(source, Path):
            if source.resolve() != self.acceptance_criteria_path.resolve():
                shutil.copyfile(source, self.acceptance_criteria_path)
        else:
            self.acceptance_criteria_path.write_text(source.rstrip() + "\n",
                                                     encoding="utf-8")

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    def _should_reset_coder(self, i: int) -> bool:
        if i == 0:
            return False
        # Iteration 1 completes the initial build; reset before iteration 2
        # so refinement starts from a clean context.
        if i == 1:
            return True
        if self.coder_context_reset_interval <= 0:
            return False
        return i % self.coder_context_reset_interval == 0

    def _should_reset_reviewer(self, i: int) -> bool:
        if i == 0 or self.reviewer_context_reset_interval <= 0:
            return False
        return i % self.reviewer_context_reset_interval == 0

    def _reset_coder(self) -> None:
        if not isinstance(self.coder, AgentLayer):
            return
        self.coder.__exit__(None, None, None)
        self.coder = _make_agent(
            "coder",
            self.prompts.coder,
            self._progress_tools["coder"] + self._status_tools["coder"],
            required_tools=["append_coder_progress", "update_status"],
            human_input_enabled=self.build_human_review_enabled,
        )

    def _reset_reviewer(self) -> None:
        if not isinstance(self.reviewer, AgentLayer):
            return
        self.reviewer.__exit__(None, None, None)
        self.reviewer = _make_agent(
            "reviewer",
            self.prompts.reviewer,
            self._progress_tools["reviewer"] + self._status_tools["reviewer"],
            required_tools=["append_reviewer_progress", "update_status"],
            backend_kind="codex",
            model=CODEX_DEFAULT_MODEL,
        )

    def _latest_reviewer_decision(self) -> str | None:
        entry = latest_entry(self.progress_path, "reviewer")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in ("APPROVE", "REJECT") else None

    def _latest_qa_decision(self) -> str | None:
        entry = latest_entry(self.progress_path, "qa")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in ("APPROVE", "REJECT") else None

    def _latest_qa_score(self) -> float | None:
        entry = latest_entry(self.progress_path, "qa")
        if entry is None:
            return None
        score = entry.get("weighted_score")
        try:
            return float(score) if score is not None else None
        except (TypeError, ValueError):
            return None

    def _latest_plan_reviewer_decision(self) -> str | None:
        entry = latest_entry(self.progress_path, "plan_reviewer")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in ("APPROVE", "REJECT") else None

    def _latest_plan_drafter_decision(self) -> str | None:
        entry = latest_entry(self.progress_path, "plan_drafter")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return (d if d in ("DRAFT_READY", "POLISHING",
                           "HUMAN_APPROVED") else None)

    # ------------------------------------------------------------------ agents

    def _run_plan_drafter(self, iteration: int, mode: str) -> None:
        """Invoke the PlanDrafter in either ``draft`` or ``human`` mode."""
        self._progress_ctx.current_iteration = iteration
        if mode == "draft":
            prompt = (
                f"Workspace: {self.workspace}\n"
                f"Plan iteration: {iteration}\n"
                f"Phase: **draft** (PlanReviewer will check your work; do "
                f"NOT call ask_human in this phase).\n\n"
                f"Read `{self.task_path}` for the original task.\n"
                f"Read the current contents of `{self.plan_path}` and "
                f"`{self.acceptance_criteria_path}` — either may already "
                f"hold user-supplied content you should preserve or "
                f"refine; the other will be empty for you to draft from "
                f"scratch.\n"
                f"If this is a re-draft, call `read_latest_progress` with "
                f"`agent: \"plan_reviewer\"` to fetch the latest REJECT "
                f"feedback and address every item.\n\n"
                f"Write your complete implementation plan to "
                f"`{self.plan_path}` and your acceptance-criteria "
                f"checklist (`- [ ] ...`) to "
                f"`{self.acceptance_criteria_path}`. Both files must end "
                f"this turn populated and coherent with each other and "
                f"with `task.yaml`.\n\n"
                f"Before completing your turn, call "
                f"`append_plan_drafter_progress` with `summary` describing "
                f"what you wrote/changed in **both** files and "
                f"`decision: \"DRAFT_READY\"`.")
        elif mode == "human":
            prompt = (
                f"Workspace: {self.workspace}\n"
                f"Plan iteration: {iteration}\n"
                f"Phase: **human review** (PlanReviewer has APPROVEd the "
                f"plan; now obtain the human's approval).\n\n"
                f"Call the `ask_human` tool with:\n"
                f"  - `header`: \"plan-review\"\n"
                f"  - `question`: a self-contained summary of the plan in "
                f"`{self.plan_path}` **and** the acceptance criteria in "
                f"`{self.acceptance_criteria_path}`, plus the explicit "
                f"ask \"Should we proceed with this plan and acceptance "
                f"criteria, or do you want changes?\"\n"
                f"  - `options`: exactly two — "
                f"`{{\"label\": \"Approve\", \"description\": \"Plan and "
                f"criteria look good — proceed to implementation.\"}}` "
                f"and `{{\"label\": \"Request changes\", \"description\": "
                f"\"Provide feedback in the reply text; I will revise.\""
                f"}}`.\n\n"
                f"On reply:\n"
                f"  - exactly `\"Approve\"` → call "
                f"`append_plan_drafter_progress` with "
                f"`decision: \"HUMAN_APPROVED\"` and end the turn.\n"
                f"  - `\"Request changes\"` or any free-form feedback → "
                f"revise `{self.plan_path}` and/or "
                f"`{self.acceptance_criteria_path}` to address it, then "
                f"call `ask_human` again. Loop within this turn until "
                f"the human approves.\n"
                f"  - `\"(no response from human)\"` → stop asking, call "
                f"`append_plan_drafter_progress` with "
                f"`decision: \"POLISHING\"`, and end the turn.")
        else:
            raise ValueError(f"unknown plan_drafter mode: {mode!r}")
        self.plan_drafter(prompt)

    def _run_plan_reviewer(self, iteration: int) -> None:
        self._progress_ctx.current_iteration = iteration
        self.plan_reviewer(
            f"Workspace: {self.workspace}\n"
            f"Plan iteration: {iteration}\n\n"
            f"Read `{self.task_path}` (the user's original intent), "
            f"`{self.plan_path}` (the PlanDrafter's plan), and "
            f"`{self.acceptance_criteria_path}` (the pass/fail checklist "
            f"QA will verify). Call `read_latest_progress` with "
            f"`agent: \"plan_drafter\"` to fetch the PlanDrafter's "
            f"latest summary.\n\n"
            "Decide APPROVE or REJECT covering **both** plan-phase "
            "outputs as a unit: the plan must satisfy task.yaml and be "
            "concrete enough for the Coder to execute, and the "
            "acceptance criteria must be a flat checklist of mechanically "
            "checkable items faithful to task.yaml. Do NOT build, run, or "
            "test code — that belongs to the build-phase Reviewer. This "
            "is a paper review.\n\n"
            "Before completing your turn, call "
            "`append_plan_reviewer_progress` with `summary` and `decision` "
            "(exactly `APPROVE` or `REJECT`). On REJECT, list specific "
            "actionable items the PlanDrafter must address, naming the "
            "file (`plan.md` or `acceptance-criteria.md`) for each item.")

    def _run_coder(self, iteration: int) -> None:
        self._progress_ctx.current_iteration = iteration
        self.coder(
            f"Workspace: {self.workspace}\n"
            f"Iteration: {iteration}\n\n"
            f"Start by calling `read_status` to load the rolling "
            f"`status.md` scratchpad — that is your fastest way to pick up "
            f"where the previous turn left off.\n\n"
            f"Read `{self.task_path}` for the original task from the user, "
            f"`{self.plan_path}` for the build plan, and "
            f"`{self.acceptance_criteria_path}` for the pass/fail "
            f"checklist QA will verify (your definition of done). Call "
            f"`read_latest_progress` with `iterations: 2` to fetch the "
            f"Reviewer's latest REJECT feedback (if any) and the QA's "
            f"latest REJECT report (if any). These are what you must "
            f"address this iteration.\n\n"
            f"Also call `read_human_feedback` to fetch any direct user "
            f"guidance recorded via `--feedback`. Treat those entries as "
            f"high-priority guidance from the human and address every "
            f"unaddressed point this turn.\n\n"
            "Implement or refine the code to address the feedback and "
            "satisfy every acceptance criterion. Before completing your "
            "turn, call **both** required tools: `append_coder_progress` "
            "(with a `summary` of what you built or changed) and "
            "`update_status` (overwriting status.md with a short, clean "
            "snapshot — current status, execution path, what's been tried, "
            "what worked, what didn't, pointers for the next step).")

    def _run_reviewer(self, iteration: int) -> None:
        self._progress_ctx.current_iteration = iteration
        self.reviewer(
            f"Workspace: {self.workspace}\n"
            f"Iteration: {iteration}\n\n"
            f"Start by calling `read_status` to load the rolling "
            f"`status.md` scratchpad so you know what the Coder claims "
            f"the current state is.\n\n"
            f"Read `{self.plan_path}` for the build plan and "
            f"`{self.acceptance_criteria_path}` for the pass/fail "
            f"checklist. Call `read_latest_progress` with "
            f"`agent: \"coder\"` to fetch the Coder's latest summary.\n\n"
            f"Also call `read_human_feedback` to fetch any direct user "
            f"guidance recorded via `--feedback`. When you decide "
            f"APPROVE/REJECT, verify the Coder has actually addressed "
            f"every unaddressed point — if not, REJECT and call them out "
            f"by name.\n\n"
            "Work closely with the Coder: inspect the changed files, then "
            "**build the code, run it, and execute the relevant tests** "
            "against the plan and the acceptance criteria. APPROVE only "
            "when you have seen the change actually build and run "
            "correctly and have evidence the criteria will hold for QA. "
            "REJECT — with specific, actionable feedback citing exact "
            "errors or failing tests — when the build/tests fail, "
            "runtime behavior contradicts the plan, or any acceptance "
            "criterion is clearly unmet. Keep the loop tight: skip long "
            "benchmarks and full-suite stress runs (those belong to QA).\n\n"
            "Before completing your turn, call **both** required tools: "
            "`append_reviewer_progress` (with `summary` and `decision`, "
            "exactly `APPROVE` or `REJECT` — cite the commands you ran "
            "and what you observed in the summary) and `update_status` "
            "(overwriting status.md to reflect the post-review state, "
            "what was actually tested, and what the Coder must address "
            "next on REJECT).")

    def _run_qa(self, iteration: int) -> None:
        self._progress_ctx.current_iteration = iteration
        gate_hint = ""
        if self.min_score > 0:
            gate_hint = (
                f"\n\nThe orchestrator treats an APPROVE as final only when "
                f"`weighted_score` is at least {self.min_score:.1f}/10; "
                f"APPROVE below that floor is downgraded to a loop-back. "
                f"Do not pad the score — if the artifact is not yet that "
                f"good, say REJECT and list the gaps.")
        self.qa(f"Workspace: {self.workspace}\n"
                f"Iteration: {iteration}\n\n"
                f"Read `{self.task_path}` (the user's stated intent — "
                f"ultimate ground truth) and "
                f"`{self.acceptance_criteria_path}` (the pass/fail "
                f"checklist; the operational denominator for "
                f"`weighted_score`). Do NOT read `plan.md`, "
                f"`progress.yaml`, `status.md`, or any other intermediate "
                f"artifact; your verdict must be grounded solely in "
                f"those two specs and the actual code you build and "
                f"run. On any conflict between the criteria and "
                f"`task.yaml`, `task.yaml` wins — call out the gap.\n\n"
                f"Call `read_human_feedback` to fetch any direct user "
                f"guidance recorded via `--feedback`. Human feedback is "
                f"the user's own voice, not a downstream agent's, so "
                f"treat it on par with `task.yaml`: APPROVE requires that "
                f"every unaddressed feedback point has been resolved at "
                f"runtime. Flag any conflicts with `task.yaml` in your "
                f"summary.\n\n"
                "Discover the code under the workspace yourself (ls, grep, "
                "etc.), build it, run tests, and verify every acceptance "
                "criterion at runtime. Do not rely on code review alone.\n\n"
                "Before completing your turn, call the `append_qa_progress` "
                "tool with:\n"
                "- `summary`: per-criterion pass/fail with runtime "
                "evidence, evaluation-criterion scores, strengths, "
                "weaknesses, recommendation\n"
                "- `decision`: exactly `APPROVE` or `REJECT`\n"
                "- `weighted_score`: the weighted average in [0, 10]\n\n"
                "APPROVE ends the workflow (subject to the score floor). "
                "REJECT sends the work back to the Coder; put the gaps they "
                "must fix in `summary`." + gate_hint)


if __name__ == "__main__":
    from .cli import main

    main()

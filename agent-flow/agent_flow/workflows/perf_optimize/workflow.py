from __future__ import annotations

import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

import yaml
from rich.markup import escape

from agent_flow import (
    CLAUDE_CODE_DEFAULT_MODEL,
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
    require_tool_call_stop_hook,
)
from agent_flow.console import print_message, print_rule
from agent_flow.logger import get_logger
from agent_flow.workflows.perf_analyze.sol_methodology import (
    SolMethodology,
    output_instruction,
    projector_instruction,
)
from agent_flow.workflows.perf_analyze.workflow import clear_stale_benchmark_results

from . import gitops, kernel_ledger, reuse, roadmap_schema
from .disagg import disagg_config_path, has_disagg, load_disagg_config, worker_config_yaml
from .execution import (
    ExecutionLayout,
    PerfOptimizeLayout,
    RunFileSystems,
    initialize_remote_execution,
    sync_benchmarker_results_to_control,
    sync_qa_results_to_control,
    sync_run_inputs_to_execution,
)
from .progress import (
    EVALUATOR_DECISIONS,
    INTEGRATOR_DECISIONS,
    OPTIMIZATION_STAGE,
    ProgressContext,
    append_workflow_event,
    build_progress_tools,
    init_progress_file,
    latest_entry,
    read_progress,
)
from .prompts import DEFAULT_PROMPTS, PromptBundle
from .roadmap_schema import RoadmapError
from .state import (
    ROUND_STAGES,
    STAGE_ANALYZER,
    STAGE_BENCHMARKER,
    STAGE_EVALUATOR,
    STAGE_INTEGRATOR,
    STAGE_OPTIMIZER,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_PROJECTOR,
    STAGE_QA,
    STAGE_REPORTER,
    WorkflowState,
    load_state,
    save_state,
)
from .task_schema import (
    OPTIMIZE_DEFAULTS,
    cluster_ssh,
    concurrency_points,
    dump_task_yaml,
    focus_concurrencies,
    is_curve_mode,
    kernel_coverage,
    load_and_validate_task_yaml,
    max_regression_pct,
    sol_enabled,
)


def _progress_has_entries(path: Path) -> bool:
    """Return True iff ``path`` holds a progress.yaml with real entries.

    Distinguishes the empty ``{optimization: []}`` shell left by
    ``init_progress_file`` (which should not block a retry) from real
    entries written by an earlier run (which should). Missing/empty files
    count as no entries. Malformed YAML counts as "has content" so the
    user is forced to ``--clean`` rather than silently losing the bad
    file.
    """
    if not path.is_file():
        return False
    try:
        data = read_progress(path)
    except (ValueError, yaml.YAMLError):
        return True
    return bool(data[OPTIMIZATION_STAGE])


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Compose stop hooks that require *every* listed tool to be called.

    ``require_tool_call_stop_hook`` enforces "at least one of the listed
    names was called". Stacking one such hook per tool — each independent
    — yields AND semantics: every per-tool hook must allow the stop, so
    all listed tools must have been called this turn.
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
    cwd: Path | None = None,
) -> AgentLayer:
    hooks = _compose_required_tools_hooks(required_tools or [])
    return AgentLayer(
        AgentLayerConfig(
            name=name,
            system_prompt=system_prompt,
            backend=BackendConfig(
                kind=backend_kind,
                model=model,
                tools=tools,
                hooks=hooks,
                cwd=cwd,
            ),
            session=SessionConfig(mode=session_mode),
        )
    )


_ROLES = (
    "benchmarker",
    "projector",
    "analyzer",
    "optimizer",
    "evaluator",
    "integrator",
    "qa",
    "reporter",
)


class PerfOptimizeWorkflow:
    """Iterative optimization loop over a TRT-LLM serving setup.

    ``benchmarker`` measures the baseline once; the one-shot
    ``projector`` stage then runs between the baseline and round 1 —
    unless task.yaml sets ``sol.enabled: false`` — writing
    ``sol_projection.md``, the analytical speed-of-light ceiling per the
    ``internal-perf-sol-analysis`` skill (or ``perf-analysis`` where that
    is not installed), that the analyzer weighs when ranking roadmap
    items (and answers for when leaving the roadmap exhausted with
    headroom remaining — the remaining-gap attribution), the optimizer
    aims each item's realization with, and the reporter turns into a
    headroom-captured story closed by a remaining-gap accountability
    breakdown; disabled, that stage is skipped. Then the loop runs
    up to ``max_rounds`` rounds of ``analyzer`` (profile + rank
    ``roadmap.yaml`` by expected benefit) → a batch of up to
    ``max_items_per_round`` pending
    items running serially or concurrently in isolated worktrees (per item:
    ``optimizer`` ⇄ ``evaluator`` — review code/functionality/perf against
    the acceptance gate; the evaluator's three-way verdict either APPROVEs
    the attempt, REJECTs the item terminally, or PUSH_BACKs it to the
    optimizer with feedback, bounded by ``max_attempts_per_item`` — so
    every item keeps its own measured gain, verdict, and revert). No
    agent decides when to stop, and none decides what a round costs
    either. A round that accepted work is re-profiled, as is one whose
    reverted code attempt may have left gitignored build output behind.
    Otherwise it opens **replan-only** — the analyzer re-plans from the
    standing profile and the failed items' evidence without launching a
    server, because the orchestrator knows the runtime has not been
    invalidated since that profile. The loop
    runs the full round budget unless a deterministic break fires — an
    analyzer turn leaves the roadmap with no actionable pending item (a
    roadmap that runs dry *between* items earns one more round first, so
    the campaign always closes on a plan made against its latest
    measurements), or the optional ``optimize.target_improvement_pct``
    is met on the roadmap ledger.
    ``qa`` then runs **once** as the
    campaign's final verification (independent benchmark + optional
    accuracy eval; skipped when nothing was accepted), and ``reporter``
    synthesizes
    ``optimization_report.md`` / ``.html``. All eight roles
    run on the Claude Code backend, with sessions scoped to each role's
    unit of work: the analyzer keeps one session for the whole campaign
    (its roadmap memory), the optimizer's session spans a single item's
    attempts and is reset between items, and the evaluator and QA are
    stateless — every verdict gets fresh eyes. The evaluator and QA
    deliberately never see the SOL projection: their gates stay
    measured-vs-measured.

    The orchestrator — not the agents — owns the TRT-LLM checkout's git
    state (dedicated branch, one commit per accepted item, hard revert of
    rejected/pushed-back attempts) and every ``roadmap.yaml`` lifecycle
    field, driven by the evaluator's structured progress decisions.

    ``reuse_analysis`` seeds a fresh workspace from a previous
    ``perf-analyze`` run or ``perf-optimize`` campaign (see
    :mod:`.reuse`): the baseline report, the SOL projection, and the
    newest profile findings + traces are copied into this workspace's
    round-1 layout, the benchmarker and projector stages are marked done,
    and round 1's analyzer runs **plan-only** — authoring
    ``roadmap.yaml`` from the imported evidence without launching a
    server or a profiler.

    Every transition checkpoints before the next agent runs, so a crash /
    Ctrl-C resumes at the same stage with the same round/attempt indices.
    """

    def __init__(
        self,
        workspace: Path,
        clean: bool = False,
        prompts: PromptBundle | None = None,
        max_rounds_override: int | None = None,
        reuse_analysis: str | Path | None = None,
        sol_methodology: SolMethodology | None = None,
        execution_run_root: str | None = None,
        ssh_known_hosts: str | Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.perf_layout = PerfOptimizeLayout()
        self.execution_run_root = execution_run_root
        self.ssh_known_hosts = ssh_known_hosts
        self.execution_layout: ExecutionLayout | None = None
        self.run_fs: RunFileSystems | None = None
        self.prompts = prompts or DEFAULT_PROMPTS
        # Which SOL methodology skill this session has. Resolved by the CLI
        # before the run (so the projector's prompt matches), and defaulted to
        # the full methodology here so a direct constructor call — the test
        # suite included — never pays a live probe.
        self.sol_methodology = sol_methodology or SolMethodology()
        self.max_rounds_override = max_rounds_override
        self.reuse_analysis = Path(reuse_analysis).expanduser() if reuse_analysis else None
        self.task_path = workspace / self.perf_layout.task
        self.baseline_dir = workspace / self.perf_layout.baseline
        self.baseline_results_path = workspace / self.perf_layout.baseline_report
        self.sol_projection_path = workspace / self.perf_layout.sol_projection
        self.tuning_dir = workspace / self.perf_layout.tuning
        self.tuning_config_path = workspace / self.perf_layout.tuning_live
        self.tuning_accepted_path = workspace / self.perf_layout.tuning_accepted
        self.roadmap_path = workspace / self.perf_layout.roadmap
        self.sol_work_dir = workspace / self.perf_layout.sol_work
        # Where ``--reuse-analysis`` parks its provenance manifest and the
        # source campaign's roadmap (prior art, never the live ledger).
        self.reuse_dir = workspace / self.perf_layout.reuse_analysis
        self.reuse_manifest_path = self.reuse_dir / reuse.MANIFEST_NAME
        self.prior_roadmap_path = self.reuse_dir / reuse.PRIOR_ROADMAP_NAME
        self.rounds_dir = workspace / self.perf_layout.rounds
        self.worktrees_dir = workspace / self.perf_layout.worktrees
        self.final_verification_dir = workspace / self.perf_layout.final_verification
        self.verification_report_path = workspace / self.perf_layout.verification_report
        self.report_path = workspace / self.perf_layout.report
        self.report_html_path = workspace / self.perf_layout.report_html
        self.progress_path = workspace / self.perf_layout.progress
        self.state_path = workspace / self.perf_layout.state
        self._checkpoint_lock = Lock()
        self._progress_lock = Lock()

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            # Remove linked worktrees through git before deleting their
            # directories, otherwise the source repository retains stale
            # worktree registrations.
            if self.state_path.is_file() and self.task_path.is_file():
                try:
                    stale_state = load_state(self.state_path)
                    task_data = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))
                    if not isinstance(task_data, dict):
                        raise ValueError("checkpoint task.yaml is not a mapping")
                    stale_layout = (
                        ExecutionLayout.from_dict(stale_state.execution)
                        if stale_state.execution
                        else ExecutionLayout.local(
                            control_workspace=self.workspace,
                            campaign_repo=str(task_data.get("trtllm_repo_path") or ""),
                        )
                    )
                    stale_fs = RunFileSystems.from_layout(
                        stale_layout,
                        known_hosts=self.ssh_known_hosts,
                    )
                    gitops.use_cluster(stale_layout.remote_host or "")
                    relatives = [
                        self.perf_layout.item_worktree(
                            stale_state.round_index + 1,
                            int(entry["item_index"]) + 1,
                            str(entry["current_item_id"]),
                        )
                        for entry in stale_state.item_batch
                    ]
                    relatives.append(
                        self.perf_layout.integration_worktree(stale_state.round_index + 1)
                    )
                    for relative in relatives:
                        if stale_layout.campaign_repo and stale_fs.exists(relative, on="execution"):
                            gitops.remove_worktree(
                                stale_layout.campaign_repo,
                                stale_fs.path(relative, on="execution"),
                            )
                    stale_fs.close()
                except (OSError, ValueError, yaml.YAMLError, gitops.GitOpsError):
                    # ``--clean`` must still recover from a corrupt checkpoint;
                    # git's later `worktree prune` can remove an orphan record.
                    pass
                finally:
                    gitops.use_cluster("")
            # Wipe the workflow's managed files and directories so the
            # constructor proceeds as a fresh run. The TRT-LLM checkout is
            # NOT touched — abandoned optimization branches are left for
            # the user to inspect/delete.
            for path in (
                self.state_path,
                self.sol_projection_path,
                self.roadmap_path,
                self.report_path,
                self.report_html_path,
                self.progress_path,
            ):
                path.unlink(missing_ok=True)
            for directory in (
                self.baseline_dir,
                self.rounds_dir,
                self.worktrees_dir,
                self.tuning_dir,
                self.final_verification_dir,
                self.reuse_dir,
                self.sol_work_dir,
            ):
                shutil.rmtree(directory, ignore_errors=True)

        # Resume is auto-detected from the checkpoint's presence;
        # ``--clean`` has just wiped it if the user wanted to start over.
        self.resume = self.state_path.is_file()

        if not self.resume:
            # On a fresh run, every managed output must be empty so we
            # don't silently scribble over a prior run the user forgot
            # about. ``task.yaml`` is exempt — it's (re)written from the
            # validated spec in ``_init_state``.
            guarded = [
                self.baseline_results_path,
                self.sol_projection_path,
                self.roadmap_path,
                self.report_path,
                self.report_html_path,
            ]
            existing = [p for p in guarded if p.is_file() and p.read_text(encoding="utf-8").strip()]
            for directory in (self.rounds_dir, self.final_verification_dir):
                if directory.is_dir() and any(directory.iterdir()):
                    existing.append(directory)
            if _progress_has_entries(self.progress_path):
                existing.append(self.progress_path)
            if existing:
                names = ", ".join(p.name for p in existing)
                raise FileExistsError(
                    f"{names} already contains content in {self.workspace} "
                    f"but no checkpoint was found. Pass --clean to "
                    f"overwrite, or delete the file(s) manually to start "
                    f"fresh."
                )

            self.baseline_dir.mkdir(parents=True, exist_ok=True)
            self.baseline_results_path.write_text("", encoding="utf-8")
            # ``sol_projection.md`` is managed unconditionally — the
            # constructor cannot see the task spec (only ``run`` does), so
            # a run without a ``sol`` block simply leaves it blank.
            self.sol_projection_path.write_text("", encoding="utf-8")
            self.roadmap_path.write_text("", encoding="utf-8")
            self.report_path.write_text("", encoding="utf-8")
            # ``report_html_path`` stays absent until the Reporter writes
            # it — keeping it missing rather than blank makes the
            # "Reporter produced HTML" check robust to empty-file edge
            # cases.
            init_progress_file(self.progress_path)
            # task.yaml and the tuning config are materialized from the
            # validated spec in ``_init_state``.

        # The tool handlers close over this context; updating its fields
        # before each agent call stamps every entry with the right loop
        # position without the agent having to pass it.
        self._progress_ctx = ProgressContext(
            path=self.progress_path,
            global_lock=self._progress_lock,
        )
        progress_tools = build_progress_tools(self._progress_ctx)

        for role in _ROLES:
            setattr(
                self,
                role,
                _make_agent(
                    role,
                    getattr(self.prompts, role),
                    progress_tools[role],
                    required_tools=[f"append_{role}_progress"],
                    # Sessions are scoped to each role's unit of work: the
                    # judges (evaluator, qa) are stateless so every verdict
                    # gets fresh eyes, uninfluenced by earlier attempts' /
                    # rounds' conclusions; each item constructs its own
                    # persistent optimizer worker for its attempt loop;
                    # the analyzer keeps campaign-long roadmap memory.
                    session_mode=(
                        "stateless" if role in ("qa", "evaluator", "integrator") else "persistent"
                    ),
                ),
            )
        self._progress_tools = progress_tools

    def __enter__(self) -> "PerfOptimizeWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for role in _ROLES:
            layer = getattr(self, role)
            if hasattr(layer, "__exit__"):
                layer.__exit__(None, None, None)
        if self.run_fs is not None:
            self.run_fs.close()

    # ------------------------------------------------------------- orchestration

    def run(self, task: str) -> None:
        log = get_logger().console

        state = self._init_state(task, log)
        if state is None:
            return

        # Where git commands run, decided once and before the first one is issued.
        #
        # HERE rather than in `_init_state`, because a resumed run returns from that
        # method early — setting it on the fresh-run path only would leave every
        # resume issuing git against the local filesystem, where the checkout does
        # not exist. This is the one point both paths pass through with the task
        # resolved.
        ssh_alias = cluster_ssh(self._task_data())
        gitops.use_cluster(ssh_alias)
        if ssh_alias:
            print_message(
                f"[bold cyan]git via ssh {ssh_alias} — the checkout lives on the "
                f"cluster, not on this host[/bold cyan]",
                log,
            )

        try:
            self._ensure_optimization_branch(state, log)

            # ---- one-shot: baseline ----
            if state.stage == STAGE_BENCHMARKER:
                print_rule("[bold cyan]Benchmarker (baseline)[/bold cyan]", log)
                clear_stale_benchmark_results(self.baseline_dir)
                self._run_benchmarker(state)
                sync_benchmarker_results_to_control(
                    self._execution(),
                    self._filesystems(),
                    self.perf_layout,
                )
                self._require_stage_outputs(STAGE_BENCHMARKER, [self.baseline_results_path])
                self._require_baseline_measurement()
                state.benchmarker_done = True
                state.stage = STAGE_PROJECTOR
                self._checkpoint(state)

            # ---- one-shot: SOL projection (conditional) ----
            if state.stage == STAGE_PROJECTOR:
                # The stage transition is unconditional but execution is
                # gated on the resolved task.yaml — this also covers the
                # resume edge where a checkpoint parked at the projector
                # is re-run after the stage was disabled. A projection
                # imported by ``--reuse-analysis`` arrives already marked
                # done: the ceiling is a property of hardware + model +
                # operating point, so re-deriving it would only restate it.
                if state.projector_done:
                    print_message(
                        f"[dim]projector skipped — reusing the projection "
                        f"imported from {state.reuse_analysis_dir}[/dim]",
                        log,
                    )
                elif self._sol_enabled():
                    print_rule("[bold cyan]Projector[/bold cyan]", log)
                    self._run_projector(state)
                    self._require_stage_outputs(STAGE_PROJECTOR, [self.sol_projection_path])
                    state.projector_done = True
                else:
                    print_message(
                        "[dim]projector skipped — `sol.enabled: false` in task.yaml[/dim]",
                        log,
                    )
                state.stage = STAGE_ANALYZER
                self._checkpoint(state)

            # ---- outer round loop (fixed budget; deterministic breaks) ----
            while state.stage in ROUND_STAGES and state.round_index < state.max_rounds:
                round_no = state.round_index + 1
                print_rule(
                    f"[bold cyan]Optimization round {round_no}/{state.max_rounds}[/bold cyan]",
                    log,
                )

                if state.stage == STAGE_ANALYZER:
                    analysis_dir = self._analysis_dir(state)
                    self._filesystems().makedirs(self._analysis_relative(state), on="both")
                    replan_only = self._replan_only(state)
                    if replan_only:
                        print_message(
                            f"[dim]round {round_no} opens replan-only — the "
                            f"previous round accepted nothing, so "
                            f"{state.last_profiled_analysis_dir} still describes "
                            f"this build[/dim]",
                            log,
                        )
                    self._run_analyzer(state)
                    analyzer_outputs = [
                        self.roadmap_path,
                        analysis_dir / "profile_findings.md",
                    ]
                    enforce_ledger = (
                        self._kernel_coverage() is not None
                        # A replan-only round runs no ncu at all: the
                        # standing ledger still describes this build.
                        and not replan_only
                        # A reused round never ran ncu either, so it can only
                        # carry the ledger the source campaign wrote — hold it
                        # to the contract exactly when the source had one.
                        and (
                            not state.reuse_pending
                            or (analysis_dir / kernel_ledger.LEDGER_FILENAME).is_file()
                        )
                    )
                    if enforce_ledger:
                        analyzer_outputs.append(analysis_dir / kernel_ledger.LEDGER_FILENAME)
                    self._require_stage_outputs(STAGE_ANALYZER, analyzer_outputs)
                    roadmap = self._validate_roadmap()
                    if enforce_ledger:
                        self._validate_kernel_ledger(roadmap, analysis_dir)
                    self._record_nsys_capture(state, analysis_dir)
                    if not replan_only and not state.reuse_pending:
                        # This round's evidence now describes the current
                        # build; a replan round produced none and leaves the
                        # pointer on the analysis it planned from. The
                        # ``--reuse-analysis`` import turn is excluded for the
                        # same reason: the artifacts under ``analysis_dir``
                        # were profiled by *another* run against another
                        # checkout, so they are prior art, not evidence about
                        # this build. Leaving the pointer empty is what makes
                        # round 2 of a reuse campaign profile normally instead
                        # of replanning against a stranger's traces.
                        state.last_profiled_analysis_dir = str(analysis_dir)
                        state.profile_required = False
                    state.reuse_pending = False
                    noise_floor = float(self._optimize_block()["noise_floor_pct"])
                    items = roadmap_schema.top_pending_items(
                        roadmap,
                        state.max_items_per_round,
                        noise_floor,
                        self._allowed_approaches(),
                    )
                    if not items:
                        # The analyzer just profiled the current state and
                        # planned nothing actionable — further rounds would
                        # re-derive the same nothing at full profile cost.
                        state.round_index += 1
                        state.item_index = 0
                        self._conclude_round_loop(
                            state, "roadmap has no actionable pending items", log
                        )
                        break
                    state.stage = STAGE_OPTIMIZER_EVALUATOR
                    parallel = state.item_execution == "parallel"
                    self._prepare_item_batch(state, items, eager_runtime=parallel)
                    if parallel:
                        for item in items:
                            roadmap_schema.mark_in_progress(self.roadmap_path, item["id"])
                    print_message(
                        f"[bold cyan]→ {state.item_execution} optimizer/evaluator batch: "
                        f"{', '.join(str(item['id']) for item in items)}[/bold cyan]",
                        log,
                    )

                if state.stage == STAGE_OPTIMIZER_EVALUATOR:
                    if state.item_execution == "serial":
                        self._run_opt_items_serial(state, log)
                        self._finish_item_batch(state, log)
                    else:
                        self._run_opt_items_parallel(state, log)
                        if any(
                            entry.get("status") == "candidate_ready" for entry in state.item_batch
                        ):
                            state.stage = STAGE_INTEGRATOR
                            self._checkpoint(state)
                        else:
                            self._finalize_failed_parallel_batch(state, log)

                if state.stage == STAGE_INTEGRATOR:
                    self._integrate_batch(state, log)

            # Defensive: a checkpoint parked inside the round ladder with
            # the round budget already spent falls through to the final
            # verification instead of dead-ending.
            if state.stage in ROUND_STAGES:
                state.stage = STAGE_QA
                self._checkpoint(state)

            # ---- one-shot: final verification ----
            if state.stage == STAGE_QA:
                if self._any_accepted_items():
                    print_rule("[bold cyan]QA (final verification)[/bold cyan]", log)
                    self._filesystems().makedirs(
                        self.perf_layout.final_verification,
                        on="both",
                    )
                    clear_stale_benchmark_results(self.final_verification_dir)
                    self._run_qa(state)
                    sync_qa_results_to_control(
                        self._execution(),
                        self._filesystems(),
                        self.perf_layout,
                    )
                    self._require_stage_outputs(STAGE_QA, [self.verification_report_path])
                else:
                    print_message(
                        "[bold yellow]no accepted items — skipping the final "
                        "verification benchmark[/bold yellow]",
                        log,
                    )
                state.stage = STAGE_REPORTER
                self._checkpoint(state)

            # ---- one-shot: report ----
            if state.stage == STAGE_REPORTER:
                print_rule("[bold cyan]Reporter[/bold cyan]", log)
                self._run_reporter(state)
                self._require_stage_outputs(
                    STAGE_REPORTER, [self.report_path, self.report_html_path]
                )
                state.reporter_done = True
                state.done = True
                self._checkpoint(state)
                print_message(
                    f"[bold green]✔ optimization report written to {self.report_path}[/bold green]",
                    log,
                )
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — run again to continue from "
                "the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            raise
        except Exception as exc:
            # Record the abort in the session log — otherwise the log just
            # ends after the last agent's turn and the crash (which only
            # reaches stderr) looks like a silent, deliberate exit.
            print_message(
                f"[bold red]✗ workflow aborted at stage '{state.stage}' "
                f"(round {state.round_index + 1}, attempt "
                f"{state.attempt_index + 1}): "
                f"{escape(f'{type(exc).__name__}: {exc}')}\n"
                f"Run again to continue from the last checkpoint, or pass "
                f"--clean to start fresh.[/bold red]",
                log,
            )
            raise

    # ------------------------------------------------------------ state & setup

    def _configure_execution(
        self,
        state: WorkflowState,
        task_data: dict[str, Any],
        *,
        fresh: bool,
    ) -> None:
        """Bind persisted execution identity to the two filesystem backends."""
        if state.execution:
            layout = ExecutionLayout.from_dict(state.execution)
            if self.execution_run_root and self.execution_run_root.rstrip("/") != layout.run_root:
                raise ValueError(
                    "--execution-run-root cannot change on resume: "
                    f"checkpoint has {layout.run_root!r}, got {self.execution_run_root!r}"
                )
        elif self.execution_run_root:
            host = cluster_ssh(task_data)
            if not host:
                raise ValueError(
                    "--execution-run-root requires slurm-environment.cluster_ssh in task.yaml"
                )
            layout = ExecutionLayout.remote(
                control_workspace=self.workspace,
                remote_host=host,
                run_root=self.execution_run_root,
                campaign_repo=str(task_data.get("trtllm_repo_path") or ""),
            )
            state.execution = layout.to_dict()
        else:
            host = cluster_ssh(task_data)
            if host:
                raise ValueError(
                    "task.yaml selects remote host "
                    f"{host!r} but no --execution-run-root was provided; "
                    "refusing a partial mode where Git is remote but workflow "
                    "files and worktrees are local"
                )
            layout = ExecutionLayout.local(
                control_workspace=self.workspace,
                campaign_repo=str(task_data.get("trtllm_repo_path") or ""),
            )
            state.execution = layout.to_dict()

        self.execution_layout = layout
        self.run_fs = RunFileSystems.from_layout(layout, known_hosts=self.ssh_known_hosts)
        if fresh and layout.mode == "remote":
            initialize_remote_execution(layout, self.run_fs, self.perf_layout)
            sync_run_inputs_to_execution(layout, self.run_fs, self.perf_layout)

    def _filesystems(self) -> RunFileSystems:
        if self.run_fs is None:
            raise RuntimeError("execution filesystems are not configured")
        return self.run_fs

    def _execution(self) -> ExecutionLayout:
        if self.execution_layout is None:
            raise RuntimeError("execution layout is not configured")
        return self.execution_layout

    def _init_state(self, task: str, log) -> WorkflowState | None:
        """Load or create the workflow state; return ``None`` to no-op.

        ``task`` is the path to the input ``task.yaml``. On a fresh run it
        is validated and the normalized spec is written verbatim into
        ``workspace/task.yaml``, and the live tuning config is
        materialized; on resume the checkpointed ``workspace/task.yaml``
        is the source of truth.
        """
        if self.resume:
            state = load_state(self.state_path)
            if state.done:
                print_message(
                    "[bold green]✔ workflow already completed; pass --clean to rerun[/bold green]",
                    log,
                )
                return None
            if (
                self.max_rounds_override is not None
                and self.max_rounds_override != state.max_rounds
            ):
                print_message(
                    f"[bold yellow]⚠ --max-rounds {self.max_rounds_override} ignored on "
                    f"resume; the checkpointed budget ({state.max_rounds}) wins. Pass "
                    f"--clean to start fresh with the new budget.[/bold yellow]",
                    log,
                )
            if self.reuse_analysis is not None:
                print_message(
                    f"[bold yellow]⚠ --reuse-analysis {self.reuse_analysis} ignored on "
                    f"resume; the import is a fresh-run seeding step and this "
                    f"workspace already has a checkpoint. Pass --clean to re-seed "
                    f"from it.[/bold yellow]",
                    log,
                )
            self._configure_execution(state, self._task_data(), fresh=False)
            return state

        # Fresh run: validate + normalize the spec and materialize it into
        # the workspace so the agents read a fully-resolved task.yaml.
        task_data = load_and_validate_task_yaml(
            task,
            max_rounds_override=self.max_rounds_override,
        )
        self.task_path.write_text(dump_task_yaml(task_data), encoding="utf-8")

        # Materialize the live tuning config (the single
        # --extra_llm_api_options every serve in this workflow uses) and
        # its last-accepted snapshot. In a disagg campaign the same file
        # holds the harness config's ctx / gen worker_config instead, so
        # the optimizer still edits exactly one file and the diff /
        # revert / accepted-snapshot machinery applies unchanged.
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        disagg_config = disagg_config_path(task_data)
        extra = task_data.get("extra_llm_api_options")
        if disagg_config is not None:
            self.tuning_config_path.write_text(
                worker_config_yaml(load_disagg_config(disagg_config)), encoding="utf-8"
            )
        elif extra:
            shutil.copyfile(extra, self.tuning_config_path)
        else:
            self.tuning_config_path.write_text("{}\n", encoding="utf-8")
        shutil.copyfile(self.tuning_config_path, self.tuning_accepted_path)

        optimize = task_data["optimize"]
        state = WorkflowState(
            task_path=str(self.task_path),
            max_rounds=int(optimize["max_rounds"]),
            max_attempts_per_item=int(optimize["max_attempts_per_item"]),
            max_items_per_round=int(optimize["max_items_per_round"]),
            item_execution=str(optimize["item_execution"]),
            stage=STAGE_BENCHMARKER,
        )
        self._configure_execution(state, task_data, fresh=True)
        if self.reuse_analysis is not None:
            self._seed_from_reuse(state, log)
        # Checkpoint before running the first stage so a crash mid-stage
        # can be picked up on the next run.
        self._checkpoint(state)
        return state

    def _seed_from_reuse(self, state: WorkflowState, log) -> None:
        """Import a previous run's analysis and skip the stages it covers.

        Copies the ``--reuse-analysis`` source's baseline report (+ result
        JSONs), SOL projection (+ ``sol_work/``) and newest profile
        findings (+ traces, kernel ledger) into this workspace's canonical
        paths, then marks the stages those artifacts stand in for as done:
        an imported baseline skips the benchmarker, an imported projection
        skips the projector, and imported findings put round 1's analyzer
        in plan-only mode (``reuse_pending``). Whatever the source lacks is
        simply produced normally.
        """
        source = self.reuse_analysis
        if source is None:
            return
        if self.workspace.resolve() == source.resolve():
            raise reuse.ReuseError(
                f"--reuse-analysis source is this run's own workspace ({source}); "
                f"point it at the previous run's workspace instead."
            )
        discovered = reuse.discover(source)
        imported = reuse.import_analysis(
            discovered,
            workspace=self.workspace,
            baseline_dir=self.baseline_dir,
            analysis_dir=self.rounds_dir / "round_1" / "analysis",
            sol_projection_path=self.sol_projection_path,
            sol_work_dir=self.sol_work_dir,
            reuse_dir=self.reuse_dir,
        )
        state.reuse_analysis_dir = str(source)
        if imported.baseline_report:
            state.benchmarker_done = True
            state.stage = STAGE_PROJECTOR
        if imported.sol_projection:
            # Only meaningful when the task enables the stage at all; the
            # flag is what makes the projector gate skip it, and a task
            # with ``sol.enabled: false`` skips it regardless.
            state.projector_done = self._sol_enabled()
        if imported.findings:
            state.reuse_pending = True
            self._record_nsys_capture(state, self.rounds_dir / "round_1" / "analysis")
        print_message(
            f"[bold cyan]reusing analysis from {source}: "
            f"{imported.summary()} (manifest: {imported.manifest_path})[/bold cyan]",
            log,
        )
        if not imported.baseline_report:
            print_message(
                "[yellow]no baseline report in the reuse source — measuring the "
                "baseline normally[/yellow]",
                log,
            )
        if not imported.findings:
            print_message(
                "[yellow]no profile findings in the reuse source — round 1 will "
                "profile normally[/yellow]",
                log,
            )
        elif self._kernel_coverage() is not None and not imported.kernel_ledger:
            print_message(
                f"[yellow]reused analysis carries no "
                f"{kernel_ledger.LEDGER_FILENAME} — the per-kernel coverage "
                f"contract is not enforced for the reused round (round 2+ "
                f"re-profiles under it as usual)[/yellow]",
                log,
            )

    def _ensure_optimization_branch(self, state: WorkflowState, log) -> None:
        """Create (fresh run) or check out (resume) the optimization branch.

        The checkpoint records the branch name *before* the branch is
        created, so a crash between the two resumes into ``checkout`` of
        the recorded name rather than a second ``checkout -b``.
        """
        repo = self._trtllm_repo_path()
        if not repo:
            raise RuntimeError(
                f"trtllm_repo_path missing from {self.task_path}; cannot manage the "
                f"optimization branch."
            )
        if state.campaign_git_branch:
            gitops.checkout(repo, state.campaign_git_branch)
            return
        if not gitops.is_git_repo(repo):
            raise RuntimeError(
                f"trtllm_repo_path ({repo}) is not a git repository. perf-optimize "
                f"needs git to commit accepted optimizations and revert rejected "
                f"ones — clone the checkout with git and retry."
            )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        state.campaign_git_branch = f"perf-optimize/{self.workspace.name}-{timestamp}"
        state.campaign_git_base_commit = gitops.rev_parse_head(repo)
        self._checkpoint(state)
        gitops.create_branch(repo, state.campaign_git_branch)
        print_message(
            f"[bold cyan]optimizing on branch {state.campaign_git_branch} "
            f"(base {state.campaign_git_base_commit[:12]})[/bold cyan]",
            log,
        )

    def _checkpoint(self, state: WorkflowState) -> None:
        with self._checkpoint_lock:
            save_state(self.state_path, state)

    def _update_batch_item(
        self, state: WorkflowState, item_id: str, **updates: Any
    ) -> dict[str, Any]:
        """Update one batch row and checkpoint it as one locked operation."""
        with self._checkpoint_lock:
            for entry in state.item_batch:
                if entry["current_item_id"] == item_id:
                    entry.update(updates)
                    save_state(self.state_path, state)
                    return dict(entry)
        raise RuntimeError(f"item batch has no item {item_id!r}")

    def _remove_worktree_best_effort(self, repo: str, path: str, log) -> None:
        """Retry a failed worktree cleanup once without aborting the campaign."""
        try:
            gitops.remove_worktree(repo, path)
            return
        except (gitops.GitOpsError, OSError):
            time.sleep(1)

        try:
            gitops.remove_worktree(repo, path)
        except (gitops.GitOpsError, OSError) as exc:
            print_message(
                "[yellow]⚠ worktree cleanup failed after one retry; "
                f"leaving it in place and continuing: {escape(path)}\n"
                f"{escape(str(exc))}[/yellow]",
                log,
            )

    def _prepare_item_batch(
        self,
        state: WorkflowState,
        items: list[dict[str, Any]],
        *,
        eager_runtime: bool = True,
    ) -> None:
        """Checkpoint a batch and optionally create every item runtime eagerly."""
        repo = self._trtllm_repo_path()
        base = gitops.rev_parse_head(repo)
        round_no = state.round_index + 1
        batch: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            item_id = str(item["id"])
            safe = self.perf_layout._safe_item_id(item_id)
            branch = f"{state.campaign_git_branch}-round-{round_no}-item-{index + 1}-{safe}"
            batch.append(
                {
                    "current_item_id": item_id,
                    "item_index": index,
                    "attempt_index": 0,
                    "approach_violation": "",
                    "item_branch": branch,
                    "item_base_commit": base,
                    "phase": STAGE_OPTIMIZER,
                    "status": "pending",
                    "candidate_commit": "",
                    "last_error": "",
                    "finalized": False,
                }
            )
        state.item_batch = batch
        state.batch_started = False
        state.batch_completed = False
        self._checkpoint(state)
        if eager_runtime:
            for entry in state.item_batch:
                self._ensure_item_runtime(state, entry)

    def _ensure_item_runtime(self, state: WorkflowState, entry: dict[str, Any]) -> None:
        """Create any missing worktree/config/progress parts for a batch row."""
        worktree_relative = self._item_worktree_relative(state, entry)
        worktree = self._execution_path(worktree_relative)
        if not self._filesystems().exists(worktree_relative, on="execution"):
            parent = str(PurePosixPath(worktree_relative).parent)
            self._filesystems().makedirs(parent, on="execution")
            gitops.create_worktree(
                self._trtllm_repo_path(),
                worktree,
                str(entry["item_branch"]),
                str(entry["item_base_commit"]),
            )
        local_state = self._local_item_state(state, entry)
        self._filesystems().makedirs(self._item_relative(local_state), on="both")
        live, accepted = self._state_tuning_relatives(local_state)
        self._filesystems().makedirs(
            self.perf_layout.item_tuning_dir(
                local_state.round_index + 1,
                local_state.item_index + 1,
                local_state.current_item_id,
            ),
            on="execution",
        )
        if not self._filesystems().exists(live, on="execution"):
            self._filesystems().copy_file(
                self.perf_layout.tuning_accepted,
                live,
                source_side="execution",
                destination_side="execution",
            )
        if not self._filesystems().exists(accepted, on="execution"):
            self._filesystems().copy_file(
                self.perf_layout.tuning_accepted,
                accepted,
                source_side="execution",
                destination_side="execution",
            )
        progress_path = self._item_progress_path(local_state)
        if not progress_path.exists():
            init_progress_file(progress_path)

    def _local_item_state(self, state: WorkflowState, entry: dict[str, Any]) -> WorkflowState:
        """Build a worker-local state view for the existing item helpers."""
        return replace(
            state,
            current_item_id=str(entry["current_item_id"]),
            item_index=int(entry["item_index"]),
            attempt_index=int(entry.get("attempt_index", 0)),
            approach_violation=str(entry.get("approach_violation", "")),
            item_worktree_path=self._execution_path(self._item_worktree_relative(state, entry)),
            item_branch=str(entry["item_branch"]),
            item_base_commit=str(entry["item_base_commit"]),
            item_batch=[],
            stage=str(entry.get("phase", STAGE_OPTIMIZER)),
        )

    def _run_opt_items_parallel(self, state: WorkflowState, log) -> None:
        for entry in state.item_batch:
            self._ensure_item_runtime(state, entry)
        item_ids = [str(entry["current_item_id"]) for entry in state.item_batch]
        if not state.batch_started:
            append_workflow_event(
                self.progress_path,
                self._progress_lock,
                event="batch_started",
                round_no=state.round_index + 1,
                item_ids=item_ids,
                summary=f"Started {len(item_ids)} parallel optimizer/evaluator item loops.",
            )
            state.batch_started = True
            self._checkpoint(state)

        errors: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=max(1, len(state.item_batch)),
            thread_name_prefix="perf-opt-item",
        ) as executor:
            futures = {
                executor.submit(self._run_opt_item, state, dict(entry), log): entry
                for entry in state.item_batch
                if entry.get("status") not in ("candidate_ready", "failed")
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:  # preserve other completed item results
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} parallel optimization item(s) failed; "
                f"first error: {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]

        if not state.batch_completed:
            append_workflow_event(
                self.progress_path,
                self._progress_lock,
                event="batch_completed",
                round_no=state.round_index + 1,
                item_ids=item_ids,
                summary="All parallel optimizer/evaluator item loops reached a terminal state.",
            )
            state.batch_completed = True
            self._checkpoint(state)

    def _run_opt_items_serial(self, state: WorkflowState, log) -> None:
        """Run the preselected batch one item at a time, accepting directly."""
        repo = self._trtllm_repo_path()
        for entry in state.item_batch:
            item_id = str(entry["current_item_id"])
            if entry.get("finalized"):
                continue

            status = str(entry.get("status", "pending"))
            if status not in ("candidate_ready", "failed"):
                if status == "pending":
                    # The previous serial item may have advanced the campaign.
                    # Delay freezing this item's base/config until it actually runs.
                    self._update_batch_item(
                        state,
                        item_id,
                        item_base_commit=gitops.rev_parse_head(repo),
                    )
                    roadmap_schema.mark_in_progress(self.roadmap_path, item_id)
                self._ensure_item_runtime(state, entry)
                self._run_opt_item(state, dict(entry), log)

            self._finalize_serial_item(state, item_id, log)
            met, _ = self._target_met()
            if met:
                break

    def _sync_evaluator_approved_result_to_control(
        self,
        state: WorkflowState,
        item_state: WorkflowState,
        *,
        candidate_commit: str,
    ) -> None:
        """Archive only an approved item's changed code/config in control."""
        result_dir = self.perf_layout.item_result_dir(
            item_state.round_index + 1,
            item_state.item_index + 1,
            item_state.current_item_id,
        )
        if candidate_commit:
            patch = gitops.format_patch(
                item_state.item_worktree_path,
                item_state.item_base_commit,
                candidate_commit,
            )
            if patch:
                self._filesystems().write_text(
                    f"{result_dir}/code.patch",
                    patch + ("" if patch.endswith("\n") else "\n"),
                    on="control",
                )

        live, accepted = self._state_tuning_relatives(item_state)
        if self._filesystems().read_text(live, on="execution") != self._filesystems().read_text(
            accepted, on="execution"
        ):
            self._filesystems().copy_file(
                live,
                f"{result_dir}/config_snapshot.yaml",
                source_side="execution",
                destination_side="control",
            )

    def _sync_integrator_result_to_control(
        self,
        state: WorkflowState,
        *,
        decision: str,
        included: set[str],
        integration_base_commit: str,
        result_commit: str,
        integration_worktree: str,
        integration_config_relative: str,
        verdict: dict[str, Any],
    ) -> None:
        """Write the parallel integration patch, config and structured result."""
        round_no = state.round_index + 1
        integration_dir = self.perf_layout.integration_dir(round_no)
        integrated_patch = f"{integration_dir}/integrated.patch"
        final_config = f"{integration_dir}/config_snapshot.yaml"

        integrated_patch_path: str | None = None
        final_config_path: str | None = None
        if decision != "REJECT":
            if result_commit and result_commit != integration_base_commit:
                patch = gitops.format_patch(
                    integration_worktree,
                    integration_base_commit,
                    result_commit,
                )
                if patch:
                    self._filesystems().write_text(
                        integrated_patch,
                        patch + ("" if patch.endswith("\n") else "\n"),
                        on="control",
                    )
                    integrated_patch_path = integrated_patch
            self._filesystems().copy_file(
                integration_config_relative,
                final_config,
                source_side="execution",
                destination_side="control",
            )
            final_config_path = final_config

        items: list[dict[str, Any]] = []
        for entry in state.item_batch:
            item_id = str(entry["current_item_id"])
            result_dir = self.perf_layout.item_result_dir(
                round_no,
                int(entry["item_index"]) + 1,
                item_id,
            )
            patch_path = f"{result_dir}/code.patch"
            config_path = f"{result_dir}/config_snapshot.yaml"
            items.append(
                {
                    "item_id": item_id,
                    "candidate_commit": str(entry.get("candidate_commit") or ""),
                    "patch_path": (
                        patch_path if self._filesystems().exists(patch_path, on="control") else None
                    ),
                    "config_path": (
                        config_path
                        if self._filesystems().exists(config_path, on="control")
                        else None
                    ),
                    "integration_status": "included" if item_id in included else "dropped",
                }
            )

        payload = {
            "round": round_no,
            "decision": decision,
            "base_commit": integration_base_commit,
            "items": items,
            "included_item_ids": sorted(included),
            "dropped_item_ids": sorted(
                str(entry["current_item_id"])
                for entry in state.item_batch
                if str(entry["current_item_id"]) not in included
            ),
            "result_commit": result_commit or None,
            "integrated_patch_path": integrated_patch_path,
            "final_config_path": final_config_path,
            "measured_value": verdict.get("measured_value"),
            "measured_gain_pct": verdict.get("measured_gain_pct"),
            "curve": verdict.get("curve") if isinstance(verdict.get("curve"), list) else [],
        }
        self._filesystems().write_text(
            f"{integration_dir}/integration_result.yaml",
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            on="control",
        )

    def _finalize_serial_item(self, state: WorkflowState, item_id: str, log) -> None:
        """Promote one terminal serial worker result into the campaign state."""
        entry = next(
            (row for row in state.item_batch if row["current_item_id"] == item_id),
            None,
        )
        if entry is None:
            raise RuntimeError(f"serial batch has no item {item_id!r}")
        if entry.get("finalized"):
            return

        status = str(entry.get("status", ""))
        if status not in ("candidate_ready", "failed"):
            raise RuntimeError(
                f"serial item {item_id!r} reached finalization with status {status!r}"
            )

        attempts = int(entry.get("attempts", 0))
        gain = entry.get("measured_gain_pct")
        repo = self._trtllm_repo_path()
        if status == "candidate_ready":
            # Persist the stale-profile decision before mutating the accepted
            # campaign state so a crash cannot resume into replan-only mode.
            state.profile_required = True
            self._checkpoint(state)
            if entry.get("candidate_commit"):
                gitops.fast_forward(repo, str(entry["item_branch"]))
            local_state = self._local_item_state(state, entry)
            candidate_config, _ = self._state_tuning_relatives(local_state)
            for destination in (
                self.perf_layout.tuning_live,
                self.perf_layout.tuning_accepted,
            ):
                self._filesystems().copy_file(
                    candidate_config,
                    destination,
                    source_side="execution",
                    destination_side="execution",
                )
                self._filesystems().copy_file(
                    candidate_config,
                    destination,
                    source_side="execution",
                    destination_side="control",
                )
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="accepted",
                attempts=attempts,
                measured_gain_pct=gain,
            )
            value = entry.get("measured_value")
            if value is not None:
                curve = entry.get("curve") if isinstance(entry.get("curve"), list) else None
                roadmap_schema.set_current_best(
                    self.roadmap_path,
                    float(value),
                    str(
                        (self._attempt_dir(local_state) / "evaluation.md").relative_to(
                            self.workspace
                        )
                    ),
                    curve=curve,
                )
            self._record_nsys_capture(state, self._attempt_dir(local_state) / "profile")
            final_status = "accepted"
            print_message(
                f"[bold green]✔ serial evaluator APPROVE — {item_id} accepted[/bold green]",
                log,
            )
        else:
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="failed",
                attempts=attempts,
                measured_gain_pct=gain,
            )
            final_status = "failed"

        self._update_batch_item(
            state,
            item_id,
            status=final_status,
            phase="complete",
            finalized=True,
        )
        worktree_relative = self._item_worktree_relative(state, entry)
        worktree = self._execution_path(worktree_relative)
        if self._filesystems().exists(worktree_relative, on="execution"):
            self._remove_worktree_best_effort(repo, worktree, log)

    def _run_opt_item(self, state: WorkflowState, entry: dict[str, Any], log) -> None:
        """Run one existing optimizer/evaluator attempt loop in isolation."""
        item_state = self._local_item_state(state, entry)
        item_id = item_state.current_item_id
        repo = item_state.item_worktree_path
        agent_cwd = Path(repo) if self._execution().mode == "local" else self._item_dir(item_state)
        progress_path = self._item_progress_path(item_state)
        progress_ctx = ProgressContext(
            path=progress_path,
            global_path=self.progress_path,
            global_lock=self._progress_lock,
        )
        tools = build_progress_tools(progress_ctx)
        optimizer = _make_agent(
            f"optimizer-{item_id}",
            self.prompts.optimizer,
            tools["optimizer"],
            required_tools=["append_optimizer_progress"],
            cwd=agent_cwd,
        )
        evaluator = _make_agent(
            f"evaluator-{item_id}",
            self.prompts.evaluator,
            tools["evaluator"],
            required_tools=["append_evaluator_progress"],
            session_mode="stateless",
            cwd=agent_cwd,
        )
        live_relative, accepted_relative = self._state_tuning_relatives(item_state)
        try:
            while item_state.attempt_index < item_state.max_attempts_per_item:
                attempt_no = item_state.attempt_index + 1
                item_state.stage = str(entry.get("phase", STAGE_OPTIMIZER))
                self._filesystems().makedirs(self._attempt_relative(item_state), on="both")

                if item_state.stage == STAGE_OPTIMIZER:
                    gitops.reset_to(repo, item_state.item_base_commit)
                    self._filesystems().copy_file(
                        accepted_relative,
                        live_relative,
                        source_side="execution",
                        destination_side="execution",
                    )
                    self._update_batch_item(
                        state,
                        item_id,
                        phase=STAGE_OPTIMIZER,
                        attempt_index=item_state.attempt_index,
                        status="running",
                    )
                    self._run_optimizer(item_state, agent=optimizer, progress_ctx=progress_ctx)
                    self._require_stage_outputs(
                        STAGE_OPTIMIZER,
                        [self._attempt_dir(item_state) / "optimization_summary.md"],
                    )
                    violation = self._detect_approach_violation(item_state)
                    if violation is not None:
                        gitops.reset_to(repo, item_state.item_base_commit)
                        self._filesystems().copy_file(
                            accepted_relative,
                            live_relative,
                            source_side="execution",
                            destination_side="execution",
                        )
                        if attempt_no >= item_state.max_attempts_per_item:
                            self._update_batch_item(
                                state,
                                item_id,
                                status="failed",
                                phase="complete",
                                attempts=attempt_no,
                                approach_violation=violation,
                            )
                            return
                        item_state.attempt_index += 1
                        item_state.approach_violation = violation
                        entry["phase"] = STAGE_OPTIMIZER
                        self._update_batch_item(
                            state,
                            item_id,
                            phase=STAGE_OPTIMIZER,
                            attempt_index=item_state.attempt_index,
                            approach_violation=violation,
                        )
                        continue
                    item_state.approach_violation = ""
                    entry["phase"] = STAGE_EVALUATOR
                    self._update_batch_item(
                        state,
                        item_id,
                        phase=STAGE_EVALUATOR,
                        approach_violation="",
                    )

                item_state.stage = STAGE_EVALUATOR
                cached_verdict = latest_entry(progress_path, "evaluator")
                cached_attempt = (
                    cached_verdict.get("attempt") if cached_verdict is not None else None
                )
                evaluation_path = self._attempt_dir(item_state) / "evaluation.md"
                if not (self._is_nonempty(evaluation_path) and cached_attempt == attempt_no):
                    clear_stale_benchmark_results(self._attempt_dir(item_state))
                    self._run_evaluator(item_state, agent=evaluator, progress_ctx=progress_ctx)
                    self._require_stage_outputs(
                        STAGE_EVALUATOR,
                        [evaluation_path],
                    )
                decision = self._latest_evaluator_decision(progress_path)
                gain = self._latest_evaluator_measured_gain(progress_path)
                value = self._latest_evaluator_measured_value(progress_path)
                curve = self._latest_evaluator_curve(progress_path)
                if decision == "APPROVE":
                    commit = ""
                    if not gitops.worktree_clean(repo):
                        commit = gitops.commit_all(
                            repo,
                            f"perf-optimize: {item_id} candidate-ready "
                            f"[round {item_state.round_index + 1}]",
                        )
                    elif gitops.rev_parse_head(repo) != item_state.item_base_commit:
                        # Resume after a commit succeeded but candidate export failed.
                        commit = gitops.rev_parse_head(repo)
                    self._sync_evaluator_approved_result_to_control(
                        state,
                        item_state,
                        candidate_commit=commit,
                    )
                    self._update_batch_item(
                        state,
                        item_id,
                        status="candidate_ready",
                        phase="complete",
                        attempts=attempt_no,
                        candidate_commit=commit,
                        measured_gain_pct=gain,
                        measured_value=value,
                        curve=curve,
                    )
                    return
                if decision == "REJECT" or attempt_no >= item_state.max_attempts_per_item:
                    gitops.reset_to(repo, item_state.item_base_commit)
                    self._filesystems().copy_file(
                        accepted_relative,
                        live_relative,
                        source_side="execution",
                        destination_side="execution",
                    )
                    self._update_batch_item(
                        state,
                        item_id,
                        status="failed",
                        phase="complete",
                        attempts=attempt_no,
                        measured_gain_pct=gain,
                    )
                    return

                gitops.reset_to(repo, item_state.item_base_commit)
                self._filesystems().copy_file(
                    accepted_relative,
                    live_relative,
                    source_side="execution",
                    destination_side="execution",
                )
                item_state.attempt_index += 1
                entry["phase"] = STAGE_OPTIMIZER
                self._update_batch_item(
                    state,
                    item_id,
                    phase=STAGE_OPTIMIZER,
                    attempt_index=item_state.attempt_index,
                    status="running",
                )
        except BaseException as exc:
            self._update_batch_item(
                state,
                item_id,
                status="error",
                phase=item_state.stage,
                attempt_index=item_state.attempt_index,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            optimizer.__exit__(None, None, None)
            evaluator.__exit__(None, None, None)

    def _finalize_failed_parallel_batch(self, state: WorkflowState, log) -> None:
        """Record an all-failed parallel batch and close its round."""
        for entry in state.item_batch:
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                str(entry["current_item_id"]),
                status="failed",
                attempts=int(entry.get("attempts", 0)),
                measured_gain_pct=entry.get("measured_gain_pct"),
            )
        self._finish_item_batch(state, log)

    def _integrate_batch(self, state: WorkflowState, log) -> None:
        """Combine candidate-ready items and trust the Integrator's verdict."""
        candidates: list[dict[str, Any]] = []
        for entry in state.item_batch:
            if entry.get("status") != "candidate_ready":
                continue
            candidate = dict(entry)
            item_state = self._local_item_state(state, entry)
            live_config, _ = self._state_tuning_relatives(item_state)
            candidate["candidate_config_path"] = self._execution_path(live_config)
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError("integrator stage has no candidate-ready items")

        repo = self._trtllm_repo_path()
        integration_base_commit = gitops.rev_parse_head(repo)
        round_no = state.round_index + 1
        integration_relative = self.perf_layout.integration_dir(round_no)
        self._filesystems().makedirs(integration_relative, on="both")
        integration_dir = self._control_path(integration_relative)
        worktree_relative = self.perf_layout.integration_worktree(round_no)
        worktree = self._execution_path(worktree_relative)
        if not state.integration_branch:
            state.integration_branch = f"{state.campaign_git_branch}-round-{round_no}-integration"
            self._checkpoint(state)
        if not self._filesystems().exists(worktree_relative, on="execution"):
            self._filesystems().makedirs(
                str(PurePosixPath(worktree_relative).parent),
                on="execution",
            )
            gitops.create_worktree(
                repo,
                worktree,
                state.integration_branch,
                gitops.rev_parse_head(repo),
            )

        integration_tuning_relative = f"{integration_relative}/tuning"
        self._filesystems().makedirs(integration_tuning_relative, on="execution")
        integration_config_relative = f"{integration_tuning_relative}/extra_llm_api_options.yaml"
        integration_config = self._execution_path(integration_config_relative)
        if not self._filesystems().exists(integration_config_relative, on="execution"):
            self._filesystems().copy_file(
                self.perf_layout.tuning_accepted,
                integration_config_relative,
                source_side="execution",
                destination_side="execution",
            )
        manifest_path = integration_dir / "candidate_manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "round": round_no,
                    "campaign_base_commit": integration_base_commit,
                    "candidates": candidates,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        report_path = integration_dir / "integration.md"
        verdict = latest_entry(self.progress_path, "integrator")
        cached_verdict = (
            self._is_nonempty(report_path)
            and verdict is not None
            and verdict.get("round") == round_no
        )
        if not cached_verdict:
            clear_stale_benchmark_results(integration_dir)
            self._stamp_progress(state, round_no=round_no)
            self.integrator(
                self._disagg_directive() + f"Workspace: {self.workspace}\n"
                f"Round: {round_no}\n"
                f"Integration worktree: {worktree}\n"
                f"Integration branch: {state.integration_branch}\n"
                f"Candidate manifest: {manifest_path}\n"
                f"Live integration config: {integration_config}\n"
                f"Campaign accepted config (base): {self.tuning_accepted_path}\n"
                f"Task: {self.task_path}\n"
                f"Roadmap/current_best: {self.roadmap_path}\n"
                f"Reference benchmark results: {self._reference_result_dir()}\n"
                f"Write the integration report to: {report_path}\n\n"
                f"Read the manifest in order. Cherry-pick every non-empty "
                f"candidate_commit into the integration branch, resolve only "
                f"merge conflicts/minimal combination defects, and combine the "
                f"candidate config files into the live integration config. Commit "
                f"any conflict-resolution code before finishing.\n\n"
                f"Launch and benchmark this combined state using "
                f"`--extra_llm_api_options {integration_config}` and the same "
                f"Evaluator measurement/Pareto rules. Compute the combined "
                f"required gain as `max(noise_floor_pct, best standalone "
                f"measured_gain_pct - noise_floor_pct)` from task.yaml and the "
                f"manifest. Your threshold and curve-gate conclusion are "
                f"authoritative; the Python orchestrator will not recalculate "
                f"them. You may diagnose/remediate at most twice. If the combined "
                f"state still fails, retain and validate only the highest standalone "
                f"gain candidate (manifest order breaks ties); if that also fails, "
                f"restore the base and REJECT.\n\n"
                f"Call `append_integrator_progress` exactly once with the final "
                f"APPROVE, FALLBACK_BEST, or REJECT decision and all required "
                f"fields. Leave precisely the accepted code/config state in the "
                f"integration worktree and integration config."
            )
        self._require_stage_outputs(STAGE_INTEGRATOR, [report_path])
        verdict = latest_entry(self.progress_path, "integrator")
        if verdict is None or verdict.get("decision") not in INTEGRATOR_DECISIONS:
            raise RuntimeError("integrator finished without a structured verdict")

        decision = str(verdict["decision"])
        included = {str(item_id) for item_id in verdict.get("included_item_ids", [])}
        result_commit = ""
        if decision != "REJECT":
            # Persist the stale-profile decision before mutating the accepted
            # campaign state so a crash cannot resume into replan-only mode.
            state.profile_required = True
            self._checkpoint(state)
            if not gitops.worktree_clean(worktree):
                result_commit = gitops.commit_all(
                    worktree,
                    f"perf-optimize: integrate round {round_no} candidates",
                )
            else:
                result_commit = gitops.rev_parse_head(worktree)
        self._sync_integrator_result_to_control(
            state,
            decision=decision,
            included=included,
            integration_base_commit=integration_base_commit,
            result_commit=result_commit,
            integration_worktree=worktree,
            integration_config_relative=integration_config_relative,
            verdict=verdict,
        )
        if decision != "REJECT":
            gitops.fast_forward(repo, state.integration_branch)
            for destination in (
                self.perf_layout.tuning_live,
                self.perf_layout.tuning_accepted,
            ):
                self._filesystems().copy_file(
                    integration_config_relative,
                    destination,
                    source_side="execution",
                    destination_side="execution",
                )
                self._filesystems().copy_file(
                    integration_config_relative,
                    destination,
                    source_side="execution",
                    destination_side="control",
                )

        for entry in state.item_batch:
            item_id = str(entry["current_item_id"])
            accepted = decision != "REJECT" and item_id in included
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="accepted" if accepted else "failed",
                attempts=int(entry.get("attempts", 0)),
                measured_gain_pct=entry.get("measured_gain_pct"),
            )
        if decision != "REJECT":
            curve = verdict.get("curve") if isinstance(verdict.get("curve"), list) else None
            roadmap_schema.set_current_best(
                self.roadmap_path,
                float(verdict["measured_value"]),
                str(report_path.relative_to(self._control_path(""))),
                curve=curve,
            )
            self._record_nsys_capture(state, integration_dir / "profile")
        print_message(
            f"[bold green]Integrator verdict: {decision}; included "
            f"{', '.join(sorted(included)) or 'none'}[/bold green]",
            log,
        )
        self._finish_item_batch(state, log)

    def _finish_item_batch(self, state: WorkflowState, log) -> None:
        """Clean batch worktrees and deterministically close the round."""
        repo = self._trtllm_repo_path()
        for entry in state.item_batch:
            relative = self._item_worktree_relative(state, entry)
            if self._filesystems().exists(relative, on="execution"):
                self._remove_worktree_best_effort(repo, self._execution_path(relative), log)
        integration_relative = self.perf_layout.integration_worktree(state.round_index + 1)
        if self._filesystems().exists(integration_relative, on="execution"):
            self._remove_worktree_best_effort(
                repo,
                self._execution_path(integration_relative),
                log,
            )

        state.round_index += 1
        state.item_index = 0
        state.attempt_index = 0
        state.current_item_id = ""
        state.approach_violation = ""
        state.item_batch = []
        state.batch_started = False
        state.batch_completed = False
        state.integration_worktree_path = ""
        state.integration_branch = ""

        met, cumulative = self._target_met()
        if met:
            self._conclude_round_loop(
                state,
                f"target_improvement_pct reached (cumulative {cumulative:+.2f}%)",
                log,
            )
            return
        if state.round_index >= state.max_rounds:
            reason = "round budget exhausted"
            if state.profile_required:
                reason += " before the accepted runtime could be re-profiled"
            self._conclude_round_loop(state, reason, log)
            return
        state.stage = STAGE_ANALYZER
        self._checkpoint(state)

    # ------------------------------------------------------ deterministic gates

    def _detect_approach_violation(self, state: WorkflowState | None = None) -> str | None:
        """Return why the attempt violates ``optimize.approaches``, or ``None``.

        Purely mechanical checks, run after every optimizer turn: with
        ``config`` disallowed, the live tuning config must still match
        the accepted snapshot; with ``code`` disallowed, the checkout's
        worktree must still be clean. With both approaches allowed (the
        default) neither check runs.
        """
        approaches = self._allowed_approaches()
        active_state = state or WorkflowState(task_path=str(self.task_path))
        if "config" not in approaches:
            tuning_config, tuning_accepted = self._state_tuning_relatives(active_state)
            live = self._filesystems().read_text(tuning_config, on="execution")
            accepted = self._filesystems().read_text(tuning_accepted, on="execution")
            if live != accepted:
                return (
                    "the attempt changed tuning/extra_llm_api_options.yaml, but "
                    "'config' is not in optimize.approaches"
                )
        if "code" not in approaches:
            repo = self._state_repo_path(active_state)
            if repo and not gitops.worktree_clean(repo):
                return (
                    "the attempt changed the TRT-LLM checkout, but 'code' is "
                    "not in optimize.approaches"
                )
        return None

    def _replan_only(self, state: WorkflowState) -> bool:
        """Whether the round about to open should re-plan instead of re-profile.

        True exactly when the standing profile is known to remain current:
        nothing was accepted, no reverted code attempt could have left a
        rebuilt gitignored artifact behind, and this campaign has produced
        a real profile to plan from. Rejected config attempts are
        hard-reverted (``git reset --hard`` + ``clean -fd`` + the last
        accepted tuning config), so their runtime state is the one the
        standing analysis already describes. What *has* changed is the
        evidence — a batch of items now measured dead — so the round still
        runs an analyzer turn; it just plans from the standing profile and
        those verdicts rather than re-deriving the same traces.

        Round 1 (nothing profiled yet), the ``--reuse-analysis`` import
        turn, and a resumed pre-field checkpoint (whose profile currency
        is unknown, so ``load_state`` sets ``profile_required``) all
        profile normally.
        """
        return (
            state.round_index > 0
            and not state.reuse_pending
            and not state.profile_required
            and bool(state.last_profiled_analysis_dir)
        )

    def _conclude_round_loop(self, state: WorkflowState, reason: str, log) -> None:
        """End the optimization loop and park at the final verification.

        Callers have already counted the concluding round into
        ``round_index`` (so it reads as "rounds ran" from here on).
        """
        print_message(
            f"[bold green]✔ optimization loop done ({reason})[/bold green]",
            log,
        )
        state.stage = STAGE_QA
        self._checkpoint(state)

    # ---------------------------------------------------------------- gates

    @staticmethod
    def _is_nonempty(path: Path) -> bool:
        """True iff ``path`` exists and holds non-whitespace content."""
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def _disagg_directive(self) -> str:
        """The disagg override every stage prompt opens with, or ``""``.

        The role prompts are composed of two layers: a system prompt built
        from shared fragments, and the per-stage instruction this
        orchestrator writes. ``DISAGG_CAMPAIGN`` supersedes the
        single-server guidance in the *first* layer — but the second layer
        also names ``trtllm-serve``, ``--extra_llm_api_options`` and a
        readiness poll, and it arrives last and reads as the more specific
        of the two. Without this the agent is handed a contradiction and
        the disagg section can lose on specificity.

        So the stage prompt states the mode up front and points at the
        section that governs it, rather than every stage's instruction
        growing a disagg variant of its own.
        """
        if not has_disagg(self._task_data()):
            return ""
        config = disagg_config_path(self._task_data())
        return (
            f"⚠️ **This campaign is DISAGGREGATED** (harness config: `{config}`). "
            f"Nothing below that mentions `trtllm-serve`, `--extra_llm_api_options` "
            f"or polling a server applies — your system prompt's "
            f"*Disaggregated serving* section replaces all of it.\n\n"
        )

    def _require_baseline_measurement(self) -> None:
        """Fail loudly when the baseline stage produced no measurement.

        ``_require_stage_outputs`` only proves the report exists. A report
        can exist and still carry no numbers: the benchmarker is required
        to write one saying so when the server never served a request,
        which is the honest outcome and exactly what happened on a
        checkpoint the engine could not load.

        Advancing past that is pure waste — every later stage replays the
        same operating point against the same broken config, so a
        campaign that cannot measure its baseline burns a full allocation
        per stage to rediscover it. The signal has to be the artifact, not
        the prose: at least one result JSON under ``baseline/`` must carry
        the target metric.
        """
        metric = str(self._optimize_block()["target_metric"])
        for path in sorted(self.baseline_dir.rglob("*.json")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(data, dict) and metric in data:
                return
        raise RuntimeError(
            f"the baseline stage produced no measurement: no JSON under "
            f"{self.baseline_dir} carries '{metric}'. Read "
            f"{self.baseline_results_path} — the benchmarker records the blocker "
            f"there. Every later stage replays this same operating point, so the "
            f"campaign is stopped rather than spending an allocation per stage on "
            f"a configuration that cannot serve a request. Fix the blocker and "
            f"re-run to retry the baseline, or pass --clean to start over."
        )

    def _require_stage_outputs(self, stage: str, paths: list[Path]) -> None:
        """Fail loudly if a stage finished without its required deliverable.

        An agent's turn can end early — e.g. after only launching a server
        and recording an interim progress entry — having written no
        deliverable. Without this gate the workflow would advance to a
        downstream stage that has nothing to work with. Raising here
        leaves the checkpoint un-advanced (``stage`` still names this
        role), so simply re-running the workflow retries the same stage;
        ``--clean`` starts over.
        """
        missing = [p.name for p in paths if not self._is_nonempty(p)]
        if missing:
            raise RuntimeError(
                f"{stage} stage finished but left required output "
                f"empty/missing: {', '.join(missing)}. The stage did not "
                f"complete its work (it likely ended its turn before writing "
                f"its deliverable). Re-run the workflow to retry this stage, "
                f"or pass --clean to start over."
            )

    def _validate_roadmap(self) -> dict[str, Any]:
        """Structurally validate roadmap.yaml as part of the analyzer gate."""
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError as exc:
            raise RuntimeError(
                f"analyzer stage finished but roadmap.yaml failed validation:\n{exc}\n"
                f"Re-run the workflow to retry the analyzer stage, or pass --clean "
                f"to start over."
            ) from exc
        if self._curve_mode():
            # The per-point gate pairs curves by concurrency, so a
            # baseline curve at the wrong points poisons every later gain.
            points = self._curve_points()
            baseline = roadmap.get("baseline")
            curve = baseline.get("curve") if isinstance(baseline, dict) else None
            curve_points = (
                [p.get("concurrency") for p in curve] if isinstance(curve, list) else None
            )
            if curve_points != points:
                raise RuntimeError(
                    f"analyzer stage finished but roadmap.yaml's baseline.curve "
                    f"does not cover the task's concurrency points: expected "
                    f"{points}, got {curve_points}. Re-run the workflow to "
                    f"retry the analyzer stage, or pass --clean to start over."
                )
        return roadmap

    def _validate_kernel_ledger(self, roadmap: dict[str, Any], analysis_dir: Path) -> None:
        """Validate the round's kernel ledger as part of the analyzer gate.

        No-op unless the task declares ``profile.kernel_coverage``. With
        the contract active, the ledger must be shape-valid, every
        ``disposition: item`` ref must name a real roadmap item, and the
        enumerated rows must reach the declared coverage target — the
        deterministic teeth behind "every kernel's optimization and
        fusion possibility was considered". Raising leaves the checkpoint
        parked at the analyzer, so re-running retries the stage.
        """
        coverage = self._kernel_coverage()
        if coverage is None:
            return
        ledger_path = analysis_dir / kernel_ledger.LEDGER_FILENAME
        try:
            ledger = kernel_ledger.load_ledger(ledger_path)
            problems = kernel_ledger.cross_validate(
                ledger, roadmap, float(coverage["coverage_target_pct"])
            )
        except kernel_ledger.LedgerError as exc:
            raise RuntimeError(
                f"analyzer stage finished but {kernel_ledger.LEDGER_FILENAME} failed "
                f"validation:\n{exc}\nEvery kernel above the coverage bar must "
                f"carry both dispositions (faster / fusion). Re-run the workflow "
                f"to retry the analyzer stage, or pass --clean to start over."
            ) from exc
        if problems:
            bullet = "\n  - "
            raise RuntimeError(
                f"analyzer stage finished but {ledger_path} failed the coverage "
                f"contract:{bullet}{bullet.join(problems)}\n"
                f"Re-run the workflow to retry the analyzer stage, or pass "
                f"--clean to start over."
            )

    # ------------------------------------------------------------ shared lookups

    def _task_data(self) -> dict[str, Any]:
        """Best-effort read of the resolved ``workspace/task.yaml``."""
        try:
            data = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, yaml.YAMLError):
            return {}

    def _trtllm_repo_path(self) -> str:
        repo = self._task_data().get("trtllm_repo_path")
        return str(repo) if repo else ""

    def _curve_mode(self) -> bool:
        """Whether the resolved task spec runs in Pareto-curve mode."""
        return is_curve_mode(self._task_data())

    def _curve_points(self) -> list[int]:
        """The configured concurrency points (ascending in curve mode)."""
        return concurrency_points(self._task_data())

    def _sol_enabled(self) -> bool:
        """Whether the resolved task spec enables the projector stage.

        On by default — only ``sol.enabled: false`` turns it off.
        """
        return sol_enabled(self._task_data())

    def _focus_points(self) -> list[int] | None:
        """``optimize.focus_concurrencies`` when set, else ``None``.

        ``None`` means every configured point is scored (the default);
        the driving prompts only spell the focus rule out when a real
        subset is configured.
        """
        return focus_concurrencies(self._task_data())

    def _regression_budget(self) -> float | None:
        """``optimize.max_regression_pct`` when set, else ``None`` (strict)."""
        return max_regression_pct(self._task_data())

    def _kernel_coverage(self) -> dict[str, Any] | None:
        """``profile.kernel_coverage`` (defaults merged) when set, else ``None``."""
        return kernel_coverage(self._task_data())

    def _latest_kernel_ledger(self) -> Path | None:
        """The highest-round ``kernel_ledger.yaml``, or ``None``.

        The reporter's coverage proof is the final analyzer state; rounds
        are scanned numerically so ``round_10`` outranks ``round_9``.
        """
        candidates: list[tuple[int, Path]] = []
        for path in self.rounds_dir.glob(f"round_*/analysis/{kernel_ledger.LEDGER_FILENAME}"):
            match = re.fullmatch(r"round_(\d+)", path.parent.parent.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates)[1]

    def _optimize_block(self) -> dict[str, Any]:
        block = self._task_data().get("optimize")
        merged = dict(OPTIMIZE_DEFAULTS)
        if isinstance(block, dict):
            merged.update(block)
        return merged

    def _allowed_approaches(self) -> tuple[str, ...]:
        """``optimize.approaches`` as a tuple, defensively defaulted.

        ``_task_data`` is best-effort, so a malformed value degrades to
        "everything allowed" (matching every other knob's fallback) —
        the restriction is only ever narrowed by a validated spec.
        """
        value = self._optimize_block().get("approaches")
        if isinstance(value, list) and value:
            return tuple(str(entry) for entry in value)
        return tuple(roadmap_schema.APPROACHES)

    def _accuracy_block(self) -> dict[str, Any] | None:
        block = self._task_data().get("accuracy")
        return block if isinstance(block, dict) else None

    def _profile_methods(self) -> tuple[str, ...]:
        """``profile.methods`` from the resolved spec, defensively defaulted.

        The resolved ``task.yaml`` always carries the block, so the
        fallback only covers a hand-edited or unreadable file — and it
        defaults to nsys so the final profile is captured rather than
        silently skipped.
        """
        profile = self._task_data().get("profile")
        methods = profile.get("methods") if isinstance(profile, dict) else None
        if isinstance(methods, list) and methods:
            return tuple(str(entry) for entry in methods)
        return ("nsys",)

    def _record_nsys_capture(self, state: WorkflowState, directory: Path) -> None:
        """Point ``last_nsys_dir`` at ``directory`` when it holds a capture.

        Called after the stages that may produce an nsys profile — the
        analyzer's round profile, and an accepted attempt's
        accept-evidence capture — so the next evaluator's kernel
        comparison always names the freshest trace of the accepted state.
        A stage that captured nothing leaves the pointer unchanged. The
        caller checkpoints.
        """
        if any(directory.glob("*.nsys-rep")) or (directory / "nsys_stats.txt").is_file():
            state.last_nsys_dir = str(directory)

    def _any_accepted_items(self) -> bool:
        """True iff the roadmap records at least one accepted item.

        Gates the final verification: with zero accepts the final state
        IS the baseline, and an independent re-measurement of it buys
        nothing. Defensive on an unreadable roadmap (e.g. the loop never
        got past the benchmarker) — no accepts, nothing to verify.
        """
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return False
        return any(item.get("status") == "accepted" for item in roadmap.get("items", []))

    @staticmethod
    def _normalized_gain_pct(reference: float, measured: float, metric: str) -> float | None:
        """Signed % gain of ``measured`` vs ``reference``, positive = better.

        Mirrors the prompts' measurement protocol: throughput metrics
        improve upward, ``*_ms`` latency metrics improve downward.
        """
        if reference == 0:
            return None
        if metric.endswith("_ms"):
            return (reference - measured) / reference * 100.0
        return (measured - reference) / reference * 100.0

    def _target_met(self) -> tuple[bool, float | None]:
        """Whether ``optimize.target_improvement_pct`` is met, plus the gain.

        Computed deterministically from the roadmap ledger —
        ``current_best`` vs ``baseline`` on the target metric, both
        advanced only by accepted (evaluator-measured) items. Curve mode
        averages the per-point gains over the concurrency points the two
        curves share — restricted to ``optimize.focus_concurrencies``
        when configured, like every other curve→scalar derivation —
        falling back to the scalar means when either side carries no
        curve. Returns ``(False, None)`` when no target is set or the
        ledger is unreadable/incomplete.
        """
        optimize = self._optimize_block()
        target = optimize.get("target_improvement_pct")
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return (False, None)
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return (False, None)
        baseline = roadmap.get("baseline")
        best = roadmap.get("current_best")
        if not isinstance(baseline, dict) or not isinstance(best, dict):
            return (False, None)
        metric = str(roadmap.get("target_metric") or optimize["target_metric"])

        gains: list[float] = []
        base_curve = baseline.get("curve")
        best_curve = best.get("curve")
        if isinstance(base_curve, list) and isinstance(best_curve, list):
            focus = self._focus_points()
            reference_by_point = {
                point["concurrency"]: float(point["value"]) for point in base_curve
            }
            for point in best_curve:
                if focus is not None and point["concurrency"] not in focus:
                    continue
                reference = reference_by_point.get(point["concurrency"])
                if reference is None:
                    continue
                gain = self._normalized_gain_pct(reference, float(point["value"]), metric)
                if gain is not None:
                    gains.append(gain)
        if gains:
            cumulative = sum(gains) / len(gains)
        else:
            cumulative = self._normalized_gain_pct(
                float(baseline["value"]), float(best["value"]), metric
            )
        if cumulative is None:
            return (False, None)
        return (cumulative >= float(target), cumulative)

    def _reference_result_dir(self) -> Path:
        """Directory holding the reference measurement's result JSON(s).

        The last accepted attempt's directory, derived from
        ``current_best.source`` (the evaluation.md path the orchestrator
        recorded on accept), or ``baseline/`` while nothing has been
        accepted — the evaluator diffs its full metric set against the
        result JSONs found here.
        """
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return self.baseline_dir
        best = roadmap.get("current_best")
        source = best.get("source") if isinstance(best, dict) else None
        if isinstance(source, str) and source.strip():
            parent = Path(source).parent
            if parent.is_absolute():
                candidates = [parent]
            else:
                # Workspace-relative per the roadmap spec; pre-fix state
                # files stored the path already workspace-prefixed, so try
                # it as-is (CWD-relative) too.
                candidates = [self.workspace / parent, parent]
            for candidate in candidates:
                if candidate.is_dir() and candidate != self.workspace:
                    return candidate
        return self.baseline_dir

    def _trtllm_hint(self) -> str:
        """Best-effort grep root for the source-search hints in prompts."""
        repo = self._trtllm_repo_path()
        if repo:
            return f"{repo}/tensorrt_llm"
        return "<trtllm_repo_path>/tensorrt_llm"

    # -------------------------------------------------------- decision readers

    def _latest_evaluator_decision(self, path: Path | None = None) -> str | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in EVALUATOR_DECISIONS else None

    def _latest_evaluator_measured_gain(self, path: Path | None = None) -> float | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        try:
            value = entry.get("measured_gain_pct")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _latest_evaluator_measured_value(self, path: Path | None = None) -> float | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        try:
            value = entry.get("measured_value")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _latest_evaluator_curve(self, path: Path | None = None) -> list[dict[str, Any]] | None:
        """The latest evaluator entry's per-point curve, or ``None``.

        Agent-supplied data: returned only when every point is well
        shaped (all four fields, numeric, int concurrency), so
        ``set_current_best`` never raises on it — a malformed curve
        degrades to the scalar path instead of crashing the accept.
        """
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None or not isinstance(entry.get("curve"), list) or not entry["curve"]:
            return None
        curve: list[dict[str, Any]] = []
        for point in entry["curve"]:
            if not isinstance(point, dict):
                return None
            concurrency = point.get("concurrency")
            if isinstance(concurrency, bool) or not isinstance(concurrency, int):
                return None
            try:
                curve.append(
                    {
                        "concurrency": concurrency,
                        "value": float(point["value"]),
                        "tok_s_user": float(point["tok_s_user"]),
                        "tok_s_gpu": float(point["tok_s_gpu"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                return None
        if [p["concurrency"] for p in curve] != sorted({p["concurrency"] for p in curve}):
            return None
        return curve

    # ------------------------------------------------------------- round paths

    def _control_path(self, relative: str) -> Path:
        if self.run_fs is None:
            return self.workspace / relative
        return Path(self.run_fs.path(relative, on="control"))

    def _execution_path(self, relative: str) -> str:
        if self.run_fs is None:
            return str(self.workspace / relative)
        return self.run_fs.path(relative, on="execution")

    def _round_relative(self, state: WorkflowState) -> str:
        return self.perf_layout.round_dir(state.round_index + 1)

    def _analysis_relative(self, state: WorkflowState) -> str:
        return self.perf_layout.analysis_dir(state.round_index + 1)

    def _item_relative(self, state: WorkflowState) -> str:
        return self.perf_layout.item_dir(
            state.round_index + 1,
            state.item_index + 1,
            state.current_item_id,
        )

    def _item_worktree_relative(
        self,
        state: WorkflowState,
        entry: dict[str, Any],
    ) -> str:
        legacy_path = str(entry.get("item_worktree_path") or "")
        if legacy_path and self.run_fs is not None:
            execution_root = self._execution().execution_workspace.rstrip("/") + "/"
            if legacy_path.startswith(execution_root):
                return legacy_path.removeprefix(execution_root)
        return self.perf_layout.item_worktree(
            state.round_index + 1,
            int(entry["item_index"]) + 1,
            str(entry["current_item_id"]),
        )

    def _attempt_relative(self, state: WorkflowState) -> str:
        return self.perf_layout.attempt_dir(
            state.round_index + 1,
            state.item_index + 1,
            state.current_item_id,
            state.attempt_index + 1,
        )

    def _round_dir(self, state: WorkflowState) -> Path:
        return self._control_path(self._round_relative(state))

    def _analysis_dir(self, state: WorkflowState) -> Path:
        return self._control_path(self._analysis_relative(state))

    def _item_dir(self, state: WorkflowState) -> Path:
        """Directory for the item currently in the optimizer ⇄ evaluator loop.

        Namespaced per item so a round applying several items never
        collides attempt directories (a stale ``optimization_summary.md``
        from item 1 would otherwise satisfy item 2's output gate). The
        analyzer authors the ids, so they are sanitized for path use.
        """
        return self._control_path(self._item_relative(state))

    def _attempt_dir(self, state: WorkflowState) -> Path:
        return self._control_path(self._attempt_relative(state))

    def _item_progress_path(self, state: WorkflowState) -> Path:
        relative = self.perf_layout.item_progress(
            state.round_index + 1,
            state.item_index + 1,
            state.current_item_id,
        )
        return self._control_path(relative)

    def _state_repo_path(self, state: WorkflowState) -> str:
        return state.item_worktree_path or self._trtllm_repo_path()

    def _state_tuning_paths(self, state: WorkflowState) -> tuple[Path, Path]:
        live, accepted = self._state_tuning_relatives(state)
        return Path(self._execution_path(live)), Path(self._execution_path(accepted))

    def _state_tuning_relatives(self, state: WorkflowState) -> tuple[str, str]:
        if not state.item_worktree_path:
            return self.perf_layout.tuning_live, self.perf_layout.tuning_accepted
        round_no = state.round_index + 1
        item_index = state.item_index + 1
        return (
            self.perf_layout.item_tuning_live(round_no, item_index, state.current_item_id),
            self.perf_layout.item_tuning_accepted(round_no, item_index, state.current_item_id),
        )

    # ------------------------------------------------------------------ agents

    def _stamp_progress(
        self,
        state: WorkflowState,
        *,
        round_no: int | None = None,
        with_attempt: bool = False,
        ctx: ProgressContext | None = None,
    ) -> None:
        """Position the progress context for the next agent call."""
        progress_ctx = ctx or self._progress_ctx
        data = read_progress(progress_ctx.path)
        progress_ctx.current_step = len(data[OPTIMIZATION_STAGE]) + 1
        progress_ctx.current_round = round_no if round_no is not None else state.round_index
        progress_ctx.current_attempt = state.attempt_index + 1 if with_attempt else None
        progress_ctx.current_item_id = state.current_item_id if with_attempt else ""

    def _run_benchmarker(self, state: WorkflowState) -> None:
        self._stamp_progress(state, round_no=0)
        if self._curve_mode():
            points = self._curve_points()
            load_instruction = (
                f"then run `benchmark_serving.py` **once per concurrency "
                f"point {points}**, sequentially ascending, against the same "
                f"server (Pareto-curve mode — do not relaunch between "
                f"points). Use the **canonical `benchmark_serving.py` "
                f"command in your system prompt** — fill in the paths and "
                f"`benchmark` values, keep the other flags as given, and do "
                f"not improvise. Pass "
                f"`--result-dir {self.baseline_dir}/concurrency_<c>` for the "
                f"run at point `<c>` so each point's result JSON lands under "
                f"`baseline/`"
            )
            baseline_note = (
                "naming the target metric's per-point values and their "
                "**mean** explicitly — the mean becomes the roadmap's "
                "`baseline.value` and the per-point rows become "
                "`baseline.curve`. "
            )
        else:
            load_instruction = (
                f"then run `benchmark_serving.py` at the single "
                f"configured operating point from the `benchmark` block. Use the "
                f"**canonical `benchmark_serving.py` command in your system "
                f"prompt** — fill in the paths and `benchmark` values, keep the "
                f"other flags as given, and do not improvise. Pass "
                f"`--result-dir {self.baseline_dir}` so the result JSON lands "
                f"under `baseline/`"
            )
            baseline_note = (
                "naming the target metric's value explicitly — it becomes "
                "the roadmap's `baseline.value`. "
            )
        self.benchmarker(
            self._disagg_directive() + f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` for the spec — resolve `checkpoint_path`, "
            f"`trtllm_repo_path`, and the `benchmark` / `optimize` blocks.\n\n"
            f"Then **load the `perf-optimization-casebook` skill** (via the "
            f"`Skill` tool) as read-only reference, as your system prompt "
            f"directs, so your Configuration/Notes are grounded in known "
            f"TRT-LLM performance precedents.\n\n"
            f"Launch `trtllm-serve` with "
            f"`--extra_llm_api_options {self.tuning_config_path}` (the live "
            f"tuning config — always passed in this workflow), poll it to "
            f"readiness, {load_instruction}, and tear the server down "
            f"(always).\n\n"
            f"Do **all** of this within this single turn — poll readiness in "
            f"the foreground and do not yield to a background poll.\n\n"
            f"`Write` your baseline report to `{self.baseline_results_path}` "
            f"using the required structure in your system prompt "
            f"(Configuration / Metrics / Notes), {baseline_note}"
            f"Record the **exact** serve and benchmark commands so every "
            f"later stage can replay the same load.\n\n"
            f"Before completing your turn, call `append_benchmarker_progress` "
            f"with a `summary` of the commands you ran, the operating point, "
            f"the headline metrics, and the files you wrote."
        )

    def _run_projector(self, state: WorkflowState) -> None:
        self._stamp_progress(state, round_no=0)
        output = output_instruction(
            self.sol_methodology,
            str(self.sol_projection_path),
            f"{self.workspace}/sol_work/peaks.json",
            "the Analyzer's per-round measured\u2194SOL correlation",
        )
        self.projector(
            f"Workspace: {self.workspace}\n\n"
            f"You run once per campaign — your projection guides the "
            f"Analyzer's roadmap ranking and the Reporter's headroom story "
            f"for every later round.\n\n"
            f"Read `{self.task_path}` — the `sol` block (optional `gpu` "
            f"part-name hint) and the `benchmark` block "
            f"— and `{self.baseline_results_path}` (or call "
            f'`read_latest_progress` with `agent: "benchmarker"`) to recover '
            f"the measured baseline operating point, GPU, and headline "
            f"metrics. The parallel mapping (tp/pp/ep) comes from "
            f"`{self.tuning_config_path}` — the live tuning config every "
            f"server in this workflow runs with.\n\n"
            f"{projector_instruction(self.sol_methodology)}\n\n"
            f"Do **all** of this within this single turn; the stage only "
            f"counts as done once `sol_projection.md` is written.\n\n"
            f"{output}\n\n"
            f"Before completing your turn, call `append_projector_progress` "
            f"with a `summary` of the sources you used, the mapping, the "
            f"headline SOL ceiling and baseline-vs-SOL gap, and the files "
            f"you wrote."
        )

    def _baseline_curve_note(self) -> str:
        """How to seed ``baseline`` / ``current_best``, in curve mode.

        Empty in scalar mode — the analyzer's roadmap contract already
        covers the single-value case.
        """
        if not self._curve_mode():
            return ""
        focus = self._focus_points()
        if focus:
            mean_scope = (
                f"the **mean over the scored subset "
                f"`optimize.focus_concurrencies` {focus}** (the "
                f"campaign's focus regime — expected gains target it too)"
            )
        else:
            mean_scope = f"the **mean** across the concurrency points {self._curve_points()}"
        return (
            f" Pareto-curve mode: `baseline.value` is {mean_scope} "
            f"and `baseline.curve` carries the per-point "
            f"`{{concurrency, value, tok_s_user, tok_s_gpu}}` rows "
            f"for **all** configured points from the baseline "
            f"report's curve summary table — seed "
            f"`current_best` equal, curve included."
        )

    def _run_reused_analyzer(self, state: WorkflowState) -> None:
        """Plan-only analyzer turn over imported artifacts (no profiling).

        Runs in place of round 1's profile when ``--reuse-analysis``
        seeded the workspace: every trace this round would have captured
        is already on disk, so the analyzer's whole job is to turn that
        evidence into ``roadmap.yaml``. It launches no server and runs no
        profiler — the entire point of the reuse is not to spend that GPU
        time again. Round 2 then profiles for real: imported traces
        describe another run's build, so they never set the "last
        profiled" pointer the replan rule keys off (see the guard beside
        that assignment in ``run``).
        """
        analysis_dir = self._analysis_dir(state)
        findings_path = analysis_dir / "profile_findings.md"
        prior_roadmap_context = ""
        if self.prior_roadmap_path.is_file():
            prior_roadmap_context = (
                f"The source run was itself a perf-optimize campaign: its "
                f"roadmap is parked at `{self.prior_roadmap_path}` as "
                f"**read-only prior art**. Its `accepted` / `failed` items "
                f"and their `measured_gain_pct` describe *that* campaign's "
                f"checkout, not this one — never copy its statuses, "
                f"`current_best`, or ids into the roadmap you write. Use it "
                f"the way you would use evidence: carry forward the pending "
                f"items its findings still support, and do not re-propose "
                f"what it recorded as failed unless this checkout changes "
                f"the premise (say so in `evidence` when you do).\n\n"
            )
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Read `{self.sol_projection_path}` (imported with the rest) "
                f"as **optional context**: the projected ceiling, % of SOL "
                f"headroom, and bound mix inform how you rank items and "
                f"sanity-bound their `expected_gain_pct`; the imported "
                f"measured evidence still outranks it. Any measured↔SOL "
                f"correlation the source produced is already in "
                f"`{analysis_dir}` — do not re-derive it.\n\n"
            )
        self.analyzer(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: 1 (**reused analysis** — no profiling this round)\n"
            f"Analysis directory (already populated): {analysis_dir}\n\n"
            f"This campaign was launched with "
            f"`--reuse-analysis {state.reuse_analysis_dir}`: a previous run's "
            f"analysis has been imported into this workspace, so **round 1 "
            f"skips profiling entirely**. Do **not** launch `trtllm-serve`, "
            f"do **not** run nsys / ncu / the torch profiler, and do **not** "
            f"run the benchmark — every trace this round would have captured "
            f"is already on disk, and re-deriving it is exactly the cost the "
            f"reuse exists to avoid.\n\n"
            f"Read `{self.task_path}` (the spec this campaign runs under), "
            f"`{self.reuse_manifest_path}` (what was imported, and from "
            f"where), `{findings_path}` **in full** plus the traces and "
            f"summaries beside it in `{analysis_dir}`, and "
            f"`{self.baseline_results_path}` (the imported baseline "
            f"measurement — the anchor for the roadmap's `baseline` "
            f"block).\n\n"
            + projection_context
            + prior_roadmap_context
            + f"Then **load the `perf-optimization-casebook` skill** (via the "
            f"`Skill` tool) as your system prompt directs, and tag each "
            f"roadmap item's `casebook_ref` with the matching *bottleneck "
            f"signal → candidate pattern* row.\n\n"
            f"Two checks you still owe — both read-only, neither needs a "
            f"GPU: verify the imported analysis actually describes **this** "
            f"task (same model/checkpoint, parallel mapping in "
            f"`{self.tuning_config_path}`, and operating point as "
            f"`{self.task_path}`), and run the **dormant-capability sweep** "
            f"per your system prompt (checkpoint config + weight index, "
            f"unset serving knobs, gated code paths — inspect files and "
            f"`grep -rn`/`rg` under `{self._trtllm_hint()}`; launch "
            f"nothing). Where the imported evidence does not fit this task, "
            f"say so plainly rather than planning on it — a mismatch is a "
            f"finding, not a blocker to hide.\n\n"
            f"`Write` `{self.roadmap_path}` from scratch per the roadmap "
            f"contract in your system prompt — the `baseline` block from "
            f"`{self.baseline_results_path}` (the target metric's value), "
            f"`current_best` seeded equal to it, and `items` ordered by "
            f"`expected_gain_pct` descending, each grounded in the imported "
            f"evidence with a quantified `expected_gain_rationale`."
            f"{self._baseline_curve_note()}\n\n"
            f"Then **extend** `{findings_path}` — keep every imported line "
            f"verbatim (it is the record of a run you did not make) and "
            f"append two sections: `## Reused analysis` (what you reused, "
            f"from where per the manifest, how well it fits this task, and "
            f"what it does not cover) and `## Dormant capabilities` (the "
            f"sweep's outcome).\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` naming the reuse source, which imported "
            f"artifacts you planned from, the fit check's outcome, and the "
            f"roadmap items you authored with their expected gains."
        )

    def _run_analyzer(self, state: WorkflowState) -> None:
        round_no = state.round_index + 1
        self._stamp_progress(state, round_no=round_no)
        analysis_dir = self._analysis_dir(state)
        if state.reuse_pending:
            self._run_reused_analyzer(state)
            return
        if self._replan_only(state):
            self._run_replan_analyzer(state)
            return
        if round_no == 1:
            curve_note = self._baseline_curve_note()
            round_context = (
                f"This is **round 1**: run the **dormant-capability sweep** "
                f"per your system prompt (checkpoint config + weight index, "
                f"unset serving knobs, gated code paths — record the outcome "
                f"under `## Dormant capabilities` in `profile_findings.md`), "
                f"then author `{self.roadmap_path}` from scratch "
                f"per the roadmap contract in your system prompt — the `baseline` "
                f"block from `{self.baseline_results_path}` (the target metric's "
                f"value), `current_best` seeded equal to it, and `items` ordered "
                f"by `expected_gain_pct` descending.{curve_note}"
            )
        else:
            profile_reason = (
                "**the standing profile is stale or unproven for the current "
                "runtime** — accepted work changed the campaign, or the "
                "checkpoint has not established a current local profile"
            )
            round_context = (
                f"This is **round {round_no}**: the roadmap at "
                f"`{self.roadmap_path}` already exists, and {profile_reason}. "
                f"That is why this round profiles rather than re-planning. "
                f'Call `read_latest_progress` with `agent: "evaluator"` for '
                f"the verdicts on the items that **failed** — a REJECT is "
                f"measured evidence about this runtime, and re-proposing what "
                f"it disproved wastes the round. Re-profile the **current** "
                f"build and update the roadmap in place: re-order / revise "
                f"still-pending items, add newly exposed ones, mark stale "
                f"pending items `obsolete`. Never rewrite accepted/failed "
                f"history, `baseline`, `current_best`, or existing ids."
            )
        coverage = self._kernel_coverage()
        if coverage is not None:
            ledger_path = analysis_dir / kernel_ledger.LEDGER_FILENAME
            ncu_scope = (
                f"under the **per-kernel coverage contract** in your system "
                f"prompt (it supersedes Run C's top-kernel targeting): "
                f"enumerate every kernel at/above "
                f"{coverage['min_share_pct']}% of GPU time from the fresh "
                f"kern_sum (extend until "
                f"{coverage['coverage_target_pct']}% is covered; group "
                f"honestly-shared rows), capture them over bounded ncu "
                f"passes (re-filtering each pass on the still-missing "
                f"stems), and answer both questions per kernel — faster? "
                f"fusible? — each with a roadmap item or an evidence-backed "
                f"dismissal. `Write` the ledger to `{ledger_path}` per the "
                f"contract; the orchestrator validates it (both dispositions "
                f"per row, item refs resolving into `{self.roadmap_path}`, "
                f"coverage ≥ target) and an incomplete ledger aborts the "
                f"stage. Mirror the rows as the `## Kernel disposition "
                f"ledger` section of your findings"
            )
            ncu_artifacts = "the per-pass `server_ncu_pass<k>.ncu-rep` reports + their summaries"
        else:
            ncu_scope = (
                "on the top nsys kernels: keep the canonical ncu flags "
                "(`--launch-count` bounded), and classify each profiled "
                "kernel (SOL%, bound class, occupancy, stalls)"
            )
            ncu_artifacts = "`server_ncu.ncu-rep` + its summaries"
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Also read `{self.sol_projection_path}` (or call "
                f'`read_latest_progress` with `agent: "projector"`) as '
                f"**optional context**: the projected SOL ceiling, % of SOL "
                f"headroom, and compute/memory/launch bound mix can inform "
                f"how you rank roadmap items and sanity-bound their "
                f"`expected_gain_pct`, but measured trace evidence always "
                f"outranks the projection — note where the profile confirms "
                f"or contradicts it. After profiling, run the **measured↔SOL "
                f"correlation** per your system prompt: load the "
                f"`internal-perf-sol-analysis` skill (via the `Skill` tool; "
                f"fully-qualified "
                f"`trtllm-agent-toolkit:internal-perf-sol-analysis` if the "
                f"bare name is not found), build "
                f"`{analysis_dir}/regions.json` from this round's traces "
                f"(structural facts only — never invented params or "
                f"measured_ms), run the skill's `sol_calc.py analyze` "
                f"against the Projector's "
                f"`{self.workspace}/sol_work/peaks.json`, write "
                f"`{analysis_dir}/sol.json`, and transcribe the joined "
                f"per-op table into the findings' **SOL correlation "
                f"(measured vs ceiling)** section (or `Correlation "
                f"unavailable: <reason>` when a precondition fails). If you "
                f"leave the roadmap with no "
                f"actionable pending item while projected headroom remains, "
                f"close `profile_findings.md` with the **Remaining-gap "
                f"attribution** section per your system prompt — every part "
                f"of the gap gets a new item or an evidence-backed reason it "
                f"cannot be closed in this campaign.\n\n"
            )
        self.analyzer(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no}\n"
            f"Analysis directory (write your artifacts here): {analysis_dir}\n\n"
            f"Read `{self.task_path}` and `{self.baseline_results_path}` to "
            f"recover the serve + benchmark commands and operating point.\n\n"
            f"{round_context}\n\n"
            + projection_context
            + f"Early on, **load the `perf-optimization-casebook` skill** (via "
            f"the `Skill` tool) as read-only reference, as your system prompt "
            f"directs — tag each roadmap item's `casebook_ref` with the "
            f"matching *bottleneck signal → candidate pattern* row.\n\n"
            f"First **verify this checkout's profiling knobs** with "
            f"`grep -rn`/`rg` via `Bash` under `{self._trtllm_hint()}` as your "
            f"system prompt directs, then profile the current build under the "
            f"methods in `profile.methods`: relaunch `trtllm-serve` with "
            f"`--extra_llm_api_options {self.tuning_config_path}` (the live "
            f"tuning config), replay the canonical benchmark load"
            + (
                f" (Pareto-curve mode: one replay at the largest concurrency "
                f"point, {self._curve_points()[-1]}, only)"
                if self._curve_mode() and self._curve_points()
                else ""
            )
            + f", and drive "
            f"nsys from the **canonical `nsys profile` command in your system "
            f"prompt** (don't improvise nsys flags). Then run the **ncu "
            f"per-kernel deep dive** (Run C in your system prompt) — load "
            f"the `perf-nsight-compute-analysis` skill "
            f"(via the `Skill` tool; fully-qualified "
            f"`trtllm-agent-toolkit:perf-nsight-compute-analysis` if the bare "
            f"name is not found) as the capture + interpretation "
            f"methodology — {ncu_scope}. Save the traces, `nsys stats` "
            f"output, and {ncu_artifacts} under "
            f"`{analysis_dir}`. Tear every server "
            f"down.\n\n"
            f"Do **all** of this within this single turn — poll readiness in "
            f"the foreground and do not yield to a background poll.\n\n"
            f"`Write` `{analysis_dir / 'profile_findings.md'}` (Profiling "
            f"setup / nsys timeline / Torch profiler / ncu kernel analysis / "
            + ("SOL correlation / " if self._sol_enabled() else "")
            + f"Ranked bottleneck "
            f"hypotheses / Caveats), then `Write` the updated "
            f"`{self.roadmap_path}` — items ordered by expected benefit, "
            f"every item grounded across the analyses (nsys timeline, ncu "
            f"kernel analysis"
            + (", SOL correlation" if self._sol_enabled() else "")
            + ") with a quantified `expected_gain_rationale`.\n\n"
            "Before completing your turn, call `append_analyzer_progress` "
            "with a `summary` of which profilers ran, the trace files "
            "produced, and the roadmap items you added / re-ordered / marked "
            "obsolete with their expected gains."
        )

    def _run_replan_analyzer(self, state: WorkflowState) -> None:
        """Replan-only analyzer turn: no server, no profiler (see ``_replan_only``).

        Opens after a round that accepted nothing and made no code attempt
        capable of leaving ignored build output behind. Its config edits
        were hard-reverted, so the standing analysis still describes the
        runtime and re-deriving it would buy the campaign nothing. What the
        round *did* produce is verdicts — items now measured dead — and
        turning those into roadmap edits is this turn's whole job.
        """
        round_no = state.round_index + 1
        analysis_dir = self._analysis_dir(state)
        profiled_dir = state.last_profiled_analysis_dir
        prev_round_dir = self.rounds_dir / f"round_{state.round_index}"
        attempted = [d for d in prev_round_dir.glob("item_*") if d.is_dir()]
        if len(attempted) == 1:
            attempted_note = (
                f"Its one attempted item is the `item_*` directory under `{prev_round_dir}`"
            )
        elif attempted:
            attempted_note = (
                f"Its {len(attempted)} attempted items are the `item_*` "
                f"directories under `{prev_round_dir}`"
            )
        else:
            attempted_note = f"Its attempted items are under `{prev_round_dir}`"
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"`{self.sol_projection_path}` and any measured↔SOL "
                f"correlation already in `{profiled_dir}` remain valid for "
                f"this build — read them, do not re-derive them. If you "
                f"leave the roadmap with no actionable pending item while "
                f"projected headroom remains, close this round's findings "
                f"with the **Remaining-gap attribution** section per your "
                f"system prompt — every part of the gap gets a new item or "
                f"an evidence-backed reason it cannot be closed in this "
                f"campaign.\n\n"
            )
        self.analyzer(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} (**replan only** — no profiling this round)\n"
            f"Analysis directory (write your artifacts here): {analysis_dir}\n\n"
            f"Round {state.round_index} accepted **nothing**. "
            f"{attempted_note}, and the orchestrator hard-reverted every one "
            f"of them (`git reset --hard` plus the last accepted tuning "
            f"config). None was a code attempt that could leave rebuilt "
            f"gitignored output behind, so the runtime remains the state the "
            f"analysis in `{profiled_dir}` describes. Do **not** launch "
            f"`trtllm-serve`, "
            f"do **not** run nsys / ncu / the torch profiler, and do **not** "
            f"run the benchmark: a fresh profile of an unchanged build "
            f"would reproduce those traces at full GPU cost. Plan from them "
            f"instead.\n\n"
            f"What *has* changed is the evidence. Call "
            f'`read_latest_progress` with `agent: "evaluator"` (raise '
            f"`steps` until it reaches back through round "
            f"{state.round_index}) for each attempt's `decision`, "
            f"`reason_category`, and measured gain, and read the "
            f"`evaluation.md` files under `{prev_round_dir}`. Those verdicts "
            f"are measurements of **this** build: an item that failed on a "
            f"perf shortfall bounds what its bottleneck is worth, and one "
            f"that failed on functionality disproves the reasoning that "
            f"ranked it. An attempt the orchestrator auto-rejected for "
            f"violating the approach restriction never reached the "
            f"evaluator and has no `evaluation.md` — its "
            f"`optimization_summary.md` is the record, and what it proves is "
            f"about the item's *realizability* under this campaign's "
            f"`optimize.approaches`, not about the bottleneck.\n\n"
            + projection_context
            + f"Then update `{self.roadmap_path}` **in place** against that "
            f"evidence: mark `obsolete` every pending item the round's "
            f"verdicts disprove or whose premise they undercut, revise the "
            f"`expected_gain_pct` / `evidence` of pending items the "
            f"measurements bound, re-order what survives, and add items the "
            f"failures themselves imply (a REJECT often names the real "
            f"constraint) — **load the `perf-optimization-casebook` skill** "
            f"(via the `Skill` tool) as your system prompt directs before "
            f"authoring any, and tag each new item's `casebook_ref` with the "
            f"matching *bottleneck signal → candidate pattern* row. Never "
            f"rewrite `accepted` / `failed` history, "
            f"`baseline`, `current_best`, or existing ids; new items get "
            f"fresh ids continuing the sequence.\n\n"
            f"**If the evidence leaves nothing actionable, leave the roadmap "
            f"with no actionable pending item and say so.** The orchestrator "
            f"reads that as the campaign's end and closes the loop — the "
            f"correct outcome for a plateau. Do not invent items to keep the "
            f"loop alive; an unfounded item costs a full benchmark to "
            f"disprove.\n\n"
            f"`Write` `{analysis_dir / 'profile_findings.md'}` as this "
            f"round's record — a short **replan note**, not a profiling "
            f"report: which analysis you planned from (`{profiled_dir}`), "
            f"each failed item with the verdict and reason category that "
            f"killed it, what you changed in the roadmap and why, and what "
            f"remains actionable (or why nothing does).\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` naming the round that accepted nothing, the "
            f"verdicts you planned from, and the items you marked obsolete / "
            f"revised / added with their expected gains."
        )

    def _run_optimizer(
        self,
        state: WorkflowState,
        *,
        agent: AgentLayer | None = None,
        progress_ctx: ProgressContext | None = None,
    ) -> None:
        round_no = state.round_index + 1
        attempt_no = state.attempt_index + 1
        self._stamp_progress(state, round_no=round_no, with_attempt=True, ctx=progress_ctx)
        attempt_dir = self._attempt_dir(state)
        repo = self._state_repo_path(state)
        tuning_config, _ = self._state_tuning_paths(state)
        retry_context = ""
        if attempt_no > 1 and state.approach_violation:
            allowed = ", ".join(f"`{a}`" for a in self._allowed_approaches())
            retry_context = (
                f"\n\nThis is a **retry** (attempt {attempt_no} of "
                f"{state.max_attempts_per_item}): the orchestrator "
                f"auto-REJECTED the previous attempt **without evaluation** "
                f"because {state.approach_violation}, and has already "
                f"reverted the checkout and the tuning config to the last "
                f"accepted state. There is no evaluator feedback for it. "
                f"Re-implement the item strictly through the allowed "
                f"approach(es) — {allowed} — per the approach restriction in "
                f"your system prompt; if the item cannot be realized that "
                f"way, make no change and record the blocker in your summary."
            )
        elif attempt_no > 1:
            retry_context = (
                f"\n\nThis is a **retry** (attempt {attempt_no} of "
                f"{state.max_attempts_per_item}): the Evaluator PUSHED BACK "
                f"the previous attempt and the orchestrator has already "
                f"reverted the checkout and the tuning config to the last "
                f"accepted state. First call `read_latest_progress` with "
                f'`agent: "evaluator"` and read the previous attempt\'s '
                f"`evaluation.md` under `{self._item_dir(state)}` — then fix "
                f"the PUSH_BACK reason, not a different problem."
            )
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Also read `{self.sol_projection_path}` (or call "
                f'`read_latest_progress` with `agent: "projector"`) as '
                f"**context, not spec**: where the item leaves you a choice "
                f"of realization variants or knob values, aim at the binding "
                f"ceiling per the SOL guidance in your system prompt, and "
                f"record the `SOL alignment:` line in your summary — the "
                f"item's `how_to_apply` outranks the projection, and the "
                f"projection never expands the item.\n\n"
            )
        verdict_context = ""
        if state.round_index > 0 or (state.item_execution == "serial" and state.item_index > 0):
            # Verdicts land after the roadmap is authored, so an earlier
            # item's REJECT can invalidate a premise this item's text
            # still carries — the re-profile only corrects it next round.
            verdict_context = (
                f"Earlier items' verdicts may have corrected facts this "
                f"item's text still relies on (the roadmap predates them): "
                f"skim the completed items' `evaluation.md` files under "
                f"`{self.rounds_dir}` — their Verdict / `Gap implication:` "
                f"lines outrank this item's `evidence` where they conflict, "
                f"and a premise they disprove is a blocker to record in "
                f"your summary, not a claim to re-assert.\n\n"
            )
        (agent or self.optimizer)(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} — item {state.item_index + 1} of at most "
            f"{state.max_items_per_round} this round — attempt {attempt_no} "
            f"of {state.max_attempts_per_item}\n"
            f"Roadmap item to implement: **{state.current_item_id}** (read it in "
            f"`{self.roadmap_path}`)\n"
            f"Optimization branch: `{state.item_branch or state.campaign_git_branch}` "
            f"at `{repo}`\n"
            f"Attempt directory (write your artifacts here): {attempt_dir}"
            f"{retry_context}\n\n"
            f"Read `{self.task_path}` and the roadmap item, then **load the "
            f"`perf-optimization-casebook` skill** (via the `Skill` tool) as "
            f"your system prompt directs and implement **exactly this one "
            f"item** following its `how_to_apply` and the matched casebook "
            f"case: `approach: config` → edit `{tuning_config}`; "
            f"`approach: code` → edit the source under `{repo}` under "
            f"the git discipline in your system prompt (installed-package "
            f"check first; locate code paths with shell `grep -rn`/`rg` via "
            f"`Bash`; never commit).\n\n"
            + projection_context
            + verdict_context
            + f"Then smoke-check: launch `trtllm-serve` with "
            f"`--extra_llm_api_options {tuning_config}`, poll to "
            f"readiness in the foreground within this turn, send one "
            f"completion request, and tear the server down (always). Do "
            f"**not** run the full benchmark — measuring is the Evaluator's "
            f"job.\n\n"
            f"`Write` your summary to "
            f"`{attempt_dir / 'optimization_summary.md'}` using the required "
            f"structure in your system prompt (What changed / Files touched / "
            f"Mapping to the roadmap item / Expected gain / Smoke check / "
            f"Risks).\n\n"
            f"Before completing your turn, call `append_optimizer_progress` "
            f"with a `summary` of the item you implemented, what you changed, "
            f"the smoke-check result, and any risks or blockers."
        )

    def _evaluator_capture_context(self, state: WorkflowState) -> str:
        """The accept-evidence capture directive for this attempt, or ``""``.

        Composed by the orchestrator so the stateless evaluator gets the
        two things it cannot know: whether nsys is configured at all, and
        where the previous capture of the accepted state lives (the
        deterministic ``last_nsys_dir`` pointer).
        """
        if "nsys" not in self._profile_methods():
            return ""
        profile_dir = self._attempt_dir(state) / "profile"
        if state.last_nsys_dir:
            compare = (
                f"compare it against the previous capture of the accepted "
                f"state at `{state.last_nsys_dir}`"
            )
        else:
            compare = (
                "there is no previous capture to compare against — report "
                "this capture's kernel picture on its own"
            )
        curve_note = ""
        if self._curve_mode() and self._curve_points():
            curve_note = (
                f" (Pareto-curve mode: one replay at the largest concurrency "
                f"point, {self._curve_points()[-1]}, only)"
            )
        return (
            f"**Accept-evidence duty — only if your verdict is APPROVE.** "
            f"After your clean measurement and gate arithmetic, capture the "
            f"accepted state per the accept-evidence procedure in your "
            f"system prompt: tear down the measurement server, relaunch "
            f"under the canonical `nsys profile` wrap, replay the canonical "
            f"load once{curve_note}, tear down, and save the `.nsys-rep`, "
            f"the `nsys stats` output as `nsys_stats.txt`, and the replay "
            f"log into `{profile_dir}`. In `evaluation.md`'s *Kernel "
            f"evidence* section, {compare}, and state whether the item's "
            f"claimed mechanism is visible in the trace. On REJECT or "
            f"PUSH_BACK, skip the capture entirely.\n\n"
        )

    def _run_evaluator(
        self,
        state: WorkflowState,
        *,
        agent: AgentLayer | None = None,
        progress_ctx: ProgressContext | None = None,
    ) -> None:
        round_no = state.round_index + 1
        attempt_no = state.attempt_index + 1
        self._stamp_progress(state, round_no=round_no, with_attempt=True, ctx=progress_ctx)
        attempt_dir = self._attempt_dir(state)
        repo = self._state_repo_path(state)
        tuning_config, tuning_accepted = self._state_tuning_paths(state)
        optimize = self._optimize_block()
        reference_dir = self._reference_result_dir()
        if self._curve_mode():
            points = self._curve_points()
            focus = self._focus_points()
            budget = self._regression_budget()
            if budget is not None:
                regress_rule = (
                    f"regress by more than the task's declared regression "
                    f"budget `optimize.max_regression_pct` = {budget} "
                    f"(name any point kept inside it)"
                )
            else:
                regress_rule = "regress by more than the noise floor"
            if focus:
                mean_rule = (
                    f"the mean over the **scored subset "
                    f"`optimize.focus_concurrencies` {focus}** must pass "
                    f"both thresholds AND no point (scored or not) may "
                    f"{regress_rule}"
                )
                mean_fields = (
                    f"`measured_gain_pct` (the mean of per-point gains over "
                    f"the scored subset {focus}), "
                    f"`measured_value` (the mean of per-point values over "
                    f"that subset), and "
                    "`curve` (the per-point rows for ALL points)"
                )
            else:
                mean_rule = f"the mean must pass both thresholds AND no point may {regress_rule}"
                mean_fields = (
                    "`measured_gain_pct` (the mean of per-point gains), "
                    "`measured_value` (the mean of per-point values), and "
                    "`curve` (the per-point rows)"
                )
            measure_instruction = (
                f"then measure with the **canonical `benchmark_serving.py` "
                f"command in your system prompt** once per concurrency point "
                f"{points}, sequentially ascending over the same server, "
                f"passing `--result-dir {attempt_dir}/concurrency_<c>` per "
                f"point. Curve mode: apply the **Pareto gate** — per-point "
                f"gains vs `current_best.curve` (same concurrency), "
                f"{mean_rule} — per the acceptance gate in your "
                f"system prompt"
            )
            full_diff_note = (
                f"the reference result JSONs for the full-metric diff are "
                f"under `{reference_dir}` (per-point `concurrency_<c>/` "
                f"subdirectories; diff at the largest point)"
            )
            progress_fields = (
                "with all six fields — `summary`, `decision` "
                "(APPROVE|REJECT|PUSH_BACK), `reason_category` "
                "(none|code_quality|functionality|perf_shortfall), "
                f"{mean_fields} — exactly as measured; the "
                "orchestrator acts on them"
            )
        else:
            measure_instruction = (
                f"then measure with the "
                f"**canonical `benchmark_serving.py` command in your system "
                f"prompt** at the configured operating point, passing "
                f"`--result-dir {attempt_dir}`. Compute `measured_gain_pct` "
                f"against `current_best.value` per the measurement protocol, "
                f"and apply the acceptance gate"
            )
            full_diff_note = (
                f"the reference result JSON for the full-metric diff is under `{reference_dir}`"
            )
            progress_fields = (
                "with all five fields — `summary`, `decision` "
                "(APPROVE|REJECT|PUSH_BACK), `reason_category` "
                "(none|code_quality|functionality|perf_shortfall), "
                "`measured_gain_pct`, `measured_value` — exactly as "
                "measured; the orchestrator acts on them"
            )
        if attempt_no >= state.max_attempts_per_item:
            attempt_note = (
                " This is the item's **final attempt**: PUSH_BACK is not "
                "available (the orchestrator treats it as REJECT) — decide "
                "APPROVE or REJECT."
            )
        else:
            attempt_note = ""
        (agent or self.evaluator)(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} — item {state.item_index + 1} of at most "
            f"{state.max_items_per_round} this round — attempt {attempt_no} "
            f"of {state.max_attempts_per_item}\n"
            f"Roadmap item under review: **{state.current_item_id}** (read it in "
            f"`{self.roadmap_path}`)\n"
            f"Optimization branch: `{state.item_branch or state.campaign_git_branch}` "
            f"at `{repo}`\n"
            f"Attempt directory (write your artifacts here): {attempt_dir}\n"
            f"Acceptance gate: accept_fraction={optimize['accept_fraction']}, "
            f"noise_floor_pct={optimize['noise_floor_pct']}, "
            f"target_metric={optimize['target_metric']}"
            + (
                f", focus_concurrencies={self._focus_points()}"
                if self._curve_mode() and self._focus_points()
                else ""
            )
            + (
                f", max_regression_pct={self._regression_budget()}"
                if self._curve_mode() and self._regression_budget() is not None
                else ""
            )
            + "\n\n"
            f"Read `{self.task_path}`, the roadmap item and `current_best` in "
            f"`{self.roadmap_path}`, and the Optimizer's "
            f"`{attempt_dir / 'optimization_summary.md'}`.\n\n"
            f"Review the change (`git -C {repo} diff` + `--stat` "
            f"and `git status --porcelain`; diff "
            f"`{tuning_config}` against "
            f"`{tuning_accepted}` for config edits), verify "
            f"functionality (launch `trtllm-serve` with "
            f"`--extra_llm_api_options {tuning_config}`, poll to "
            f"readiness in the foreground within this turn, send completion "
            f"requests; targeted tests for code items — locate them with "
            f"shell `grep -rn`/`rg` via `Bash`), {measure_instruction}. "
            f"Tear every server down (always).\n\n"
            f"`Write` your report to `{attempt_dir / 'evaluation.md'}` using "
            f"the required structure (Change review / Functionality / "
            f"Performance / Kernel evidence / Verdict), showing the gate "
            f"arithmetic and the full-metric diff — {full_diff_note}."
            f"{attempt_note}\n\n"
            f"{self._evaluator_capture_context(state)}"
            f"Before completing your turn, call `append_evaluator_progress` "
            f"{progress_fields}."
        )

    def _run_qa(self, state: WorkflowState) -> None:
        self._stamp_progress(state)
        accuracy = self._accuracy_block()
        if accuracy:
            accuracy_context = (
                f"`task.yaml` **has** an `accuracy` block: run its `command` "
                f"verbatim against the live server, record the score under "
                f"`{self.final_verification_dir}`, and compare it to "
                f"`baseline_score` / `max_drop_pct` as your system prompt "
                f"directs."
            )
        else:
            accuracy_context = (
                "`task.yaml` has **no** `accuracy` block: skip the accuracy "
                'step entirely and note "accuracy: not configured" in your '
                "report."
            )
        if self._curve_mode():
            points = self._curve_points()
            focus = self._focus_points()
            if focus:
                mean_scope = (
                    f"the **mean over the scored subset `optimize.focus_concurrencies` {focus}**"
                )
            else:
                mean_scope = "the **mean across concurrency points**"
            benchmark_instruction = (
                f"run the **canonical `benchmark_serving.py` command in "
                f"your system prompt** once per concurrency point {points}, "
                f"sequentially ascending over the same server, with "
                f"`--result-dir {self.final_verification_dir}/concurrency_<c>` "
                f"per point"
            )
            cumulative_instruction = (
                f"Compute `cumulative_improvement_pct` from your own "
                f"measurement — {mean_scope} of "
                f"the per-point gain vs the roadmap's `baseline.curve` entry "
                f"with the same concurrency"
            )
            progress_fields = (
                f"with `summary`, `cumulative_improvement_pct` ({mean_scope} "
                f"vs baseline.curve), and `curve` (your per-point rows, all "
                f"points) from your own measurement"
            )
        else:
            benchmark_instruction = (
                f"run the **canonical `benchmark_serving.py` command in "
                f"your system prompt** at the configured operating point with "
                f"`--result-dir {self.final_verification_dir}`"
            )
            cumulative_instruction = (
                "Compute `cumulative_improvement_pct` from your own "
                "measurement vs the roadmap's `baseline.value`"
            )
            progress_fields = (
                "with both fields — `summary` and "
                "`cumulative_improvement_pct` — from your own measurement"
            )
        self.qa(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Campaign: the optimization loop is over ({state.round_index} "
            f"round(s) ran); the system under test is the final accepted "
            f"state.\n"
            f"Verification directory (write your artifacts here): "
            f"{self.final_verification_dir}\n\n"
            f"You are the campaign's final verification. Ground yourself "
            f"ONLY in `{self.task_path}`, `{self.roadmap_path}`, and your own "
            f"runs this turn — do not read other agents' reports or progress "
            f"entries.\n\n"
            f"Launch `trtllm-serve` with "
            f"`--extra_llm_api_options {self.tuning_config_path}` (the live "
            f"tuning config), poll to readiness in the foreground within this "
            f"turn, {benchmark_instruction}, and send a few completion requests "
            f"as a sanity check. {accuracy_context} Tear every server down "
            f"(always).\n\n"
            f"{cumulative_instruction}.\n\n"
            f"`Write` your report to `{self.verification_report_path}` using "
            f"the required structure (Independent benchmark / Sanity / "
            f"Accuracy / Conclusion).\n\n"
            f"Before completing your turn, call `append_qa_progress` "
            f"{progress_fields}."
        )

    def _run_reporter(self, state: WorkflowState) -> None:
        self._stamp_progress(state)
        if self._curve_mode():
            pareto_section = "Pareto Improvement / "
            pareto_chart = ", the Pareto improvement chart,"
            focus = self._focus_points()
            if focus:
                pareto_headline = (
                    f" (curve mode with `optimize.focus_concurrencies` "
                    f"{focus}: the ledger means and the headline score the "
                    f"focus subset — say so wherever a mean is presented — "
                    f"with every point still shown in the Pareto "
                    f"Improvement section)"
                )
            else:
                pareto_headline = (
                    " (curve mode: the mean across concurrency points, with the "
                    "per-point curve in the Pareto Improvement section)"
                )
        else:
            pareto_section = ""
            pareto_chart = ""
            pareto_headline = ""
        if self.verification_report_path.is_file():
            headline_source = (
                f"comes from the final verification's independent "
                f"measurement (`{self.verification_report_path}`)"
            )
        else:
            headline_source = (
                "comes from the roadmap ledger (`current_best` vs "
                "`baseline`) — the final verification did not run (no "
                "accepted items), so say so"
            )
        if state.last_nsys_dir:
            after_profile = (
                f"`{state.last_nsys_dir}` holds the freshest nsys capture "
                f"of the final accepted state — prefer it as the 'after' "
                f"side of the kernel comparison"
            )
        else:
            after_profile = (
                "no nsys capture postdates the last accepted item — the "
                "kernel comparison falls back to the latest round profile "
                "and must say which accepted items it misses"
            )
        if self._sol_enabled():
            projection_read = (
                f" `{self.sol_projection_path}` (the Projector's SOL "
                f"ceiling — the Projection vs Measured section follows the "
                f"SOL guidance in your system prompt: how much of the "
                f"projected headroom the campaign captured, closed by the "
                f"remaining-gap accountability breakdown — every part of "
                f"the remaining gap attributed to cited evidence or "
                f"explicitly marked unexplained — honestly marked "
                f"unavailable when the projection is),"
            )
            projection_section = "Projection vs Measured / "
        else:
            projection_read = ""
            projection_section = ""
        coverage_read = ""
        coverage_section = ""
        if self._kernel_coverage() is not None:
            final_ledger = self._latest_kernel_ledger()
            ledger_name = (
                f"`{final_ledger}`"
                if final_ledger is not None
                else "the final round's `analysis/kernel_ledger.yaml` (none "
                "was written — the section must say the ledger is "
                "unavailable)"
            )
            coverage_read = (
                f" {ledger_name} (the final round's per-kernel disposition "
                f"ledger — the Kernel Coverage section per your system "
                f"prompt: every kernel's faster/fusion disposition resolved "
                f"to its campaign outcome, the still-pending refs itemized "
                f"as the untried tail),"
            )
            coverage_section = "Kernel Coverage / "
        reuse_read = ""
        if state.reuse_analysis_dir:
            reuse_read = (
                f" `{self.reuse_manifest_path}` (this campaign was launched "
                f"with `--reuse-analysis {state.reuse_analysis_dir}` — the "
                f"baseline and the round-1 profile were **measured by that "
                f"run, not this one**; say so in Configuration and name what "
                f"was imported, so no reader mistakes an inherited "
                f"measurement for one this campaign made),"
            )
        self.reporter(
            f"Workspace: {self.workspace}\n"
            f"Optimization branch: `{state.campaign_git_branch}` — base commit "
            f"`{state.campaign_git_base_commit}` in `trtllm_repo_path`\n\n"
            f"The campaign is over ({state.round_index} round(s) ran). Read "
            f"**all** inputs listed in your system prompt: `{self.task_path}`, "
            f"`{self.baseline_results_path}`,"
            f"{reuse_read}{projection_read}{coverage_read} "
            f"`{self.roadmap_path}` (final "
            f"statuses, expected vs measured gains, baseline/current_best), "
            f"every `optimization_summary.md` / `evaluation.md` under "
            f"`{self.rounds_dir}`, every round's "
            f"`analysis/profile_findings.md` + `analysis/nsys_stats.txt` and "
            f"every accepted attempt's `profile/nsys_stats.txt` (the "
            f"kernel-level before/after evidence; {after_profile}), "
            f"`{self.verification_report_path}` when it exists (the final "
            f"verification's independent benchmark + accuracy), "
            f"`{self.progress_path}` (the chronological trail the "
            f"trajectory is reconstructed from), "
            f"`{self.tuning_accepted_path}` (the final accepted config), and "
            f"— read-only — `git -C <trtllm_repo_path> log --oneline` and "
            f"`git diff --stat` over `{state.campaign_git_base_commit[:12]}..HEAD` for "
            f"the code-diff summary. Launch no servers and run no "
            f"benchmarks.\n\n"
            f"`Write` `{self.report_path}` with every required section "
            f"(Executive Summary / Configuration / Baseline / Optimization "
            f"Trajectory / {pareto_section}"
            f"Applied Optimizations / Kernel-Level Comparison / "
            f"{coverage_section}"
            f"Failed Attempts / Final Verification / {projection_section}"
            f"Config & Code Diff "
            f"Summary / Remaining Roadmap / Durable facts for the next "
            f"campaign), then `Write` "
            f"`{self.report_html_path}` "
            f"mirroring it 1:1 (self-contained, interactive, with the "
            f"required trajectory line chart{pareto_chart} "
            f"and kernel before/after bars — "
            f"see your system prompt). The headline cumulative improvement "
            f"{headline_source}{pareto_headline}; "
            f"expected vs measured is reported faithfully for every item, "
            f"failures included.\n\n"
            f"Before completing your turn, call `append_reporter_progress` "
            f"with a `summary` of the cumulative improvement headline, the "
            f"accepted/failed item counts, and confirmation that both files "
            f"were written."
        )


if __name__ == "__main__":
    from .cli import main

    main()

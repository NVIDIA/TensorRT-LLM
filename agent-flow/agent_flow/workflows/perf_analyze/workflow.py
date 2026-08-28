from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

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

from .progress import (
    ANALYSIS_STAGE,
    ProgressContext,
    build_progress_tools,
    init_progress_file,
    read_progress,
)
from .prompts import DEFAULT_PROMPTS, PromptBundle
from .sol_methodology import SolMethodology, output_instruction, projector_instruction
from .state import (
    STAGE_ANALYZER,
    STAGE_BENCHMARKER,
    STAGE_PROJECTOR,
    STAGE_REPORTER,
    STATE_FILENAME,
    WorkflowState,
    load_state,
    save_state,
)
from .task_schema import (
    concurrency_points,
    dump_task_yaml,
    is_curve_mode,
    load_and_validate_task_yaml,
    sol_enabled,
)


def clear_stale_benchmark_results(result_dir: Path) -> list[str]:
    """Remove leftover benchmark result artifacts under ``result_dir``.

    A measuring stage killed mid-sweep (walltime, crash) leaves partial
    per-point ``concurrency_<c>/`` directories and ``openai-*.json``
    result files behind, and ``benchmark_serving.py`` names result JSONs
    by timestamp — so on redispatch the stage would re-measure into the
    same directory and a later reader could pick up a stale file. Purging
    at dispatch keeps every measurement directory single-sourced.

    Only benchmark result artifacts are touched — logs, reports, and
    other stage files in the same directory are kept (and the scan is
    top-level only, so nested attempt directories are never affected).
    Returns the removed entry names (sorted, logged when non-empty); a
    missing ``result_dir`` is a no-op.
    """
    if not result_dir.is_dir():
        return []
    removed: list[str] = []
    for entry in sorted(result_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("concurrency_"):
            shutil.rmtree(entry)
            removed.append(entry.name + "/")
        elif entry.is_file() and entry.name.startswith("openai-") and entry.name.endswith(".json"):
            entry.unlink()
            removed.append(entry.name)
    if removed:
        print_message(
            f"[dim]Cleared stale benchmark result artifacts in {result_dir}: "
            f"{', '.join(removed)}[/dim]",
            get_logger().console,
        )
    return removed


def _progress_has_entries(path: Path) -> bool:
    """Return True iff ``path`` holds a progress.yaml with real entries.

    Distinguishes the empty ``{analysis: []}`` shell left by
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
    return bool(data[ANALYSIS_STAGE])


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
            ),
            session=SessionConfig(mode=session_mode),
        )
    )


class PerfAnalyzeWorkflow:
    """Linear benchmarker → projector → analyzer → reporter pipeline.

    Serves a model checkpoint with ``trtllm-serve``, benchmarks and
    profiles it with ``benchmark_serving.py`` (nsys + torch profiler +
    ncu per-kernel deep dive), and synthesizes a report whose headline
    is the main performance bottleneck. The projector stage — on unless
    ``task.yaml`` sets ``sol.enabled: false`` — derives an analytical
    speed-of-light (SOL) ceiling between the benchmarker and the analyzer
    (``sol_projection.md`` plus the machine-readable
    ``sol_work/peaks.json``, following the ``internal-perf-sol-analysis``
    skill, or ``perf-analysis`` where that is not installed); the analyzer
    then correlates its measured per-op times against that ceiling
    (``sol_calc.py analyze``) and the later stages weigh both. Disabled,
    those steps are skipped. The analyzer role is perf-optimize's diagnosis
    stage without the roadmap — perf-optimize extends this pipeline with
    the optimization loop. All roles run on the Claude Code backend.

    The pipeline is one-shot (no review loop): each stage runs exactly
    once and checkpoints before advancing, so a crash / Ctrl-C resumes at
    the same stage rather than rerunning the previous one.
    """

    def __init__(
        self,
        workspace: Path,
        clean: bool = False,
        prompts: PromptBundle | None = None,
        sol_methodology: SolMethodology | None = None,
    ) -> None:
        self.workspace = workspace
        self.prompts = prompts or DEFAULT_PROMPTS
        # Which SOL methodology skill this session has. Resolved by the CLI
        # before the run (so the projector's prompt matches), and defaulted to
        # the full methodology here so a direct constructor call — the test
        # suite included — never pays a live probe.
        self.sol_methodology = sol_methodology or SolMethodology()
        self.task_path = workspace / "task.yaml"
        self.benchmark_results_path = workspace / "benchmark_results.md"
        self.sol_projection_path = workspace / "sol_projection.md"
        self.profile_findings_path = workspace / "profile_findings.md"
        self.report_path = workspace / "performance_report.md"
        self.report_html_path = workspace / "performance_report.html"
        self.progress_path = workspace / "progress.yaml"
        self.state_path = workspace / STATE_FILENAME

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            # Wipe the workflow's managed files so the constructor proceeds
            # as a fresh run. Other files in the workspace (run artifacts
            # such as serve.log, *.nsys-rep, torch_trace/) are left alone.
            for path in (
                self.state_path,
                self.benchmark_results_path,
                self.sol_projection_path,
                self.profile_findings_path,
                self.report_path,
                self.report_html_path,
                self.progress_path,
            ):
                path.unlink(missing_ok=True)

        # Resume is auto-detected from the checkpoint's presence;
        # ``--clean`` has just wiped it if the user wanted to start over.
        self.resume = self.state_path.is_file()

        if not self.resume:
            # On a fresh run, every managed output file must be empty so we
            # don't silently scribble over a prior run the user forgot
            # about. ``task.yaml`` is exempt — it's (re)written from the
            # validated spec in ``_init_state``.
            guarded = [
                self.benchmark_results_path,
                self.sol_projection_path,
                self.profile_findings_path,
                self.report_path,
                self.report_html_path,
            ]
            existing = [p for p in guarded if p.is_file() and p.read_text(encoding="utf-8").strip()]
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

            self.benchmark_results_path.write_text("", encoding="utf-8")
            # ``sol_projection.md`` is managed unconditionally — the
            # constructor cannot see the task spec (only ``run`` does), so
            # a run without a ``sol`` block simply leaves it blank.
            self.sol_projection_path.write_text("", encoding="utf-8")
            self.profile_findings_path.write_text("", encoding="utf-8")
            self.report_path.write_text("", encoding="utf-8")
            # ``report_html_path`` stays absent until the Reporter writes
            # it — keeping it missing rather than blank makes the
            # "Reporter produced HTML" check robust to empty-file edge
            # cases.
            init_progress_file(self.progress_path)
            # task.yaml is (re)written from the validated spec in ``run``.

        # The tool handlers close over this context; updating
        # ``current_step`` before each agent call stamps every entry with
        # the right step without the agent having to pass it.
        self._progress_ctx = ProgressContext(path=self.progress_path)
        progress_tools = build_progress_tools(self._progress_ctx)

        self.benchmarker = _make_agent(
            "benchmarker",
            self.prompts.benchmarker,
            progress_tools["benchmarker"],
            required_tools=["append_benchmarker_progress"],
        )
        # Constructed unconditionally (the stage gate lives in ``run``);
        # the backend client is lazy, so a skipped projector costs nothing.
        self.projector = _make_agent(
            "projector",
            self.prompts.projector,
            progress_tools["projector"],
            required_tools=["append_projector_progress"],
        )
        self.analyzer = _make_agent(
            "analyzer",
            self.prompts.analyzer,
            progress_tools["analyzer"],
            required_tools=["append_analyzer_progress"],
        )
        self.reporter = _make_agent(
            "reporter",
            self.prompts.reporter,
            progress_tools["reporter"],
            required_tools=["append_reporter_progress"],
        )
        self._progress_tools = progress_tools

    def __enter__(self) -> "PerfAnalyzeWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for layer in (self.benchmarker, self.projector, self.analyzer, self.reporter):
            layer.__exit__(None, None, None)

    # ------------------------------------------------------------- orchestration

    def run(self, task: str) -> None:
        log = get_logger().console

        state = self._init_state(task, log)
        if state is None:
            return

        try:
            # Each stage checkpoints before advancing, so a crash / Ctrl-C
            # mid-stage resumes at the same agent rather than rerunning the
            # previous one.
            if state.stage == STAGE_BENCHMARKER:
                print_rule("[bold cyan]Benchmarker[/bold cyan]", log)
                clear_stale_benchmark_results(self.workspace)
                self._run_benchmarker()
                self._require_stage_outputs(STAGE_BENCHMARKER, [self.benchmark_results_path])
                state.benchmarker_done = True
                state.stage = STAGE_PROJECTOR
                self._checkpoint(state)

            if state.stage == STAGE_PROJECTOR:
                # The stage transition is unconditional but execution is
                # gated on the resolved task.yaml — this also covers the
                # resume edge where a checkpoint parked at the projector
                # is re-run after the stage was disabled.
                if self._sol_enabled():
                    print_rule("[bold cyan]Projector[/bold cyan]", log)
                    self._run_projector()
                    self._require_stage_outputs(STAGE_PROJECTOR, [self.sol_projection_path])
                    state.projector_done = True
                else:
                    print_message(
                        "[dim]projector skipped — `sol.enabled: false` in task.yaml[/dim]",
                        log,
                    )
                state.stage = STAGE_ANALYZER
                self._checkpoint(state)

            if state.stage == STAGE_ANALYZER:
                print_rule("[bold cyan]Analyzer[/bold cyan]", log)
                self._run_analyzer()
                self._require_stage_outputs(STAGE_ANALYZER, [self.profile_findings_path])
                state.analyzer_done = True
                state.stage = STAGE_REPORTER
                self._checkpoint(state)

            if state.stage == STAGE_REPORTER:
                print_rule("[bold cyan]Reporter[/bold cyan]", log)
                self._run_reporter()
                self._require_stage_outputs(
                    STAGE_REPORTER, [self.report_path, self.report_html_path]
                )
                state.reporter_done = True
                state.done = True
                self._checkpoint(state)
                print_message(
                    f"[bold green]✔ performance report written to {self.report_path}[/bold green]",
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

    def _init_state(self, task: str, log) -> WorkflowState | None:
        """Load or create the workflow state; return ``None`` to no-op.

        ``task`` is the path to the input ``task.yaml``. On a fresh run it
        is validated and the normalized spec is written verbatim into
        ``workspace/task.yaml``; on resume the checkpointed
        ``workspace/task.yaml`` is the source of truth.
        """
        if self.resume:
            state = load_state(self.state_path)
            if state.done:
                print_message(
                    "[bold green]✔ workflow already completed; pass --clean to rerun[/bold green]",
                    log,
                )
                return None
            return state

        # Fresh run: validate + normalize the spec and materialize it into
        # the workspace so the agents read a fully-resolved task.yaml.
        task_data = load_and_validate_task_yaml(task)
        self.task_path.write_text(dump_task_yaml(task_data), encoding="utf-8")

        state = WorkflowState(task_path=str(self.task_path), stage=STAGE_BENCHMARKER)
        # Checkpoint before running the first stage so a crash mid-stage
        # can be picked up on the next run.
        self._checkpoint(state)
        return state

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    @staticmethod
    def _is_nonempty(path: Path) -> bool:
        """True iff ``path`` exists and holds non-whitespace content."""
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def _require_stage_outputs(self, stage: str, paths: list[Path]) -> None:
        """Fail loudly if a stage finished without its required deliverable.

        An agent's turn can end early — e.g. after only launching a server
        and recording an interim progress entry — having written no
        deliverable. Without this gate the workflow would mark the stage
        ``*_done`` and advance to a downstream stage that has nothing to
        work with (the Reporter once synthesized a verdict against an empty
        ``profile_findings.md``). Raising here leaves the checkpoint
        un-advanced (``stage`` still names this role), so simply re-running
        the workflow retries the same stage; ``--clean`` starts over.
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

    # ------------------------------------------------------------------ agents

    def _run_benchmarker(self) -> None:
        self._progress_ctx.current_step = 1
        if self._curve_mode():
            points = self._curve_points()
            load_instruction = (
                f"then run `benchmark_serving.py` **once per concurrency "
                f"point {points}**, sequentially ascending, against the same "
                f"server (Pareto-curve mode — do not relaunch between "
                f"points), passing `--result-dir {self.workspace}/"
                f"concurrency_<c>` for the run at point `<c>`"
            )
        else:
            load_instruction = (
                "then run `benchmark_serving.py` at the single configured "
                "operating point from the `benchmark` block"
            )
        self.benchmarker(
            f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` for the spec — resolve `checkpoint_path`, "
            f"`trtllm_repo_path`, the optional `extra_llm_api_options` path, "
            f"and the `benchmark` block.\n\n"
            f"Then **load the `perf-optimization-casebook` skill** (via the "
            f"`Skill` tool) as read-only reference, as your system prompt "
            f"directs, so your Configuration/Notes are grounded in known "
            f"TRT-LLM performance precedents.\n\n"
            f"Launch `trtllm-serve` (passing `--extra_llm_api_options` when "
            f"set), poll it to "
            f"readiness, {load_instruction}. Use the "
            f"**canonical `benchmark_serving.py` command in your system "
            f"prompt** — fill in the paths and `benchmark` values, keep the "
            f"other flags as given, and do not improvise. Capture "
            f"the result JSON of every run and tear the server down "
            f"(always).\n\n"
            f"Do **all** of this within this single turn — poll readiness in "
            f"the foreground and do not yield to a background poll; the stage "
            f"only counts as done once `benchmark_results.md` is written.\n\n"
            f"`Write` your clean benchmark report to "
            f"`{self.benchmark_results_path}` using the required structure in "
            f"your system prompt (Configuration / Metrics / Notes). Record "
            f"the **exact** serve and benchmark commands so the Analyzer can "
            f"replay the same load.\n\n"
            f"Before completing your turn, call `append_benchmarker_progress` "
            f"with a `summary` of the commands you ran, the operating point, "
            f"the headline metrics, and the files you wrote."
        )

    def _run_projector(self) -> None:
        self._progress_ctx.current_step = 2
        output = output_instruction(
            self.sol_methodology,
            str(self.sol_projection_path),
            f"{self.workspace}/sol_work/peaks.json",
            "the Analyzer's measured\u2194SOL correlation",
        )
        self.projector(
            f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` — the `sol` block (optional `gpu` "
            f"part-name hint), the `benchmark` block, and "
            f"the optional `extra_llm_api_options` path — and "
            f"`{self.benchmark_results_path}` (or call `read_latest_progress` "
            f'with `agent: "benchmarker"`) to recover the measured operating '
            f"point, GPU, and headline metrics.\n\n"
            f"{projector_instruction(self.sol_methodology)}\n\n"
            f"Do **all** of this within this single turn; the stage only "
            f"counts as done once `sol_projection.md` is written.\n\n"
            f"{output}\n\n"
            f"Before completing your turn, call `append_projector_progress` "
            f"with a `summary` of the sources you used, the mapping, the "
            f"headline SOL ceiling and measured-vs-SOL gap, and the files "
            f"you wrote."
        )

    def _run_analyzer(self) -> None:
        self._progress_ctx.current_step = 3
        curve_context = ""
        if self._curve_mode():
            largest = self._curve_points()[-1]
            curve_context = (
                f"Pareto-curve mode: the Benchmarker measured the points "
                f"{self._curve_points()}; you profile **only the largest "
                f"concurrency point, {largest}** — one replay per profiler at "
                f"`--max-concurrency {largest}`, as your system prompt "
                f"directs.\n\n"
            )
        projection_context = ""
        correlation_instruction = ""
        findings_sections = (
            "Profiling setup / nsys timeline / Torch profiler / ncu kernel "
            "analysis / Ranked bottleneck hypotheses / Caveats"
        )
        if self._sol_enabled():
            projection_context = (
                f"Also read `{self.sol_projection_path}` (or call "
                f'`read_latest_progress` with `agent: "projector"`) as '
                f"**optional context**: the projected SOL ceiling, % of SOL "
                f"headroom, and compute/memory/launch bound mix can inform "
                f"how you rank hypotheses, but measured trace evidence always "
                f"outranks the projection — note where the profile confirms "
                f"or contradicts it.\n\n"
            )
            correlation_instruction = (
                f"Then run the **measured↔SOL correlation** per your system "
                f"prompt: load the `internal-perf-sol-analysis` skill (via "
                f"the `Skill` tool; fully-qualified "
                f"`trtllm-agent-toolkit:internal-perf-sol-analysis` if the "
                f"bare name is not found), build "
                f"`{self.workspace}/sol_work/regions.json` from your traces "
                f"(structural facts only — never invented params or "
                f"measured_ms), and run the skill's `sol_calc.py analyze` "
                f"against the Projector's "
                f"`{self.workspace}/sol_work/peaks.json`, writing "
                f"`{self.workspace}/sol_work/sol.json`. Transcribe the "
                f"joined per-op table into the findings' **SOL correlation "
                f"(measured vs ceiling)** section; if a precondition fails, "
                f"record `Correlation unavailable: <reason>` there instead.\n\n"
            )
            findings_sections = (
                "Profiling setup / nsys timeline / Torch profiler / ncu "
                "kernel analysis / SOL correlation / Ranked bottleneck "
                "hypotheses / Caveats"
            )
        self.analyzer(
            f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` and `{self.benchmark_results_path}` (or "
            f'call `read_latest_progress` with `agent: "benchmarker"`) to '
            f"recover the exact serve + benchmark commands and operating "
            f"point — replay the **same** load so the profile matches the "
            f"baseline.\n\n"
            + curve_context
            + projection_context
            + f"Early on, **load the `perf-optimization-casebook` skill** (via "
            f"the `Skill` tool) as read-only reference, as your system prompt "
            f"directs, and match each ranked bottleneck hypothesis against its "
            f"*bottleneck signal → candidate pattern* index so the Reporter "
            f"inherits a known precedent.\n\n"
            f"First **verify this checkout's profiling knobs** with "
            f"`grep -rn`/`rg` via `Bash` under "
            f"`{self._trtllm_hint()}` — `py_executor.py` for both "
            f"`TLLM_PROFILE_START_STOP` (the iteration-window gate) and the "
            f"torch-trace env var (e.g. `TLLM_TORCH_PROFILE_TRACE`), and "
            f"`openai_server.py` for whether a `/start_profile` endpoint even "
            f"exists — use the names you find. Then run the profilers listed "
            f"in `profile.methods` (default all three): **nsys** (GPU "
            f"timeline), the **torch profiler**, and **ncu** (per-kernel deep "
            f"dive) — nsys and torch gated server-side by "
            f"`TLLM_PROFILE_START_STOP` over `profile.nsys_iter_range` (not by "
            f"the client's `--profile` flag), ncu over the same window via "
            f"`--profile-from-start off`. Drive nsys from the **canonical "
            f"`nsys profile` command in your system prompt** (don't improvise "
            f"nsys flags): it keeps `--capture-range-end=stop` so the window "
            f"lands in steady-state load without tearing the engine down, and "
            f"the replayed benchmark keeps `--no-test-input`. Run ncu last, "
            f"per your system prompt's Run C: **load the "
            f"`perf-nsight-compute-analysis` skill** (via the `Skill` tool; "
            f"fully-qualified "
            f"`trtllm-agent-toolkit:perf-nsight-compute-analysis` if the bare "
            f"name is not found) as the capture + interpretation methodology, "
            f"target the top kernels from the nsys table, keep the canonical "
            f"ncu flags (`--launch-count` bounded), and classify each "
            f"profiled kernel (SOL%, bound class, occupancy, stalls). Save "
            f"`server_nsys.nsys-rep`, the `nsys stats` output, the torch traces "
            f"under `torch_trace/`, `server_ncu.ncu-rep` + its "
            f"`ncu_details.txt` / `ncu_raw.csv` summaries, and "
            f"`perf_metrics.json` if available. Tear "
            f"every server down.\n\n"
            + correlation_instruction
            + f"Do **all** of this within this single turn — poll readiness in "
            f"the foreground and do not yield to a background poll; the stage "
            f"only counts as done once `profile_findings.md` is written.\n\n"
            f"`Write` your findings to `{self.profile_findings_path}` using "
            f"the required structure ({findings_sections}), citing the "
            f"trace files. **Do not** issue the final verdict — rank "
            f"hypotheses.\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` of which profilers ran, the trace files "
            f"produced, and your ranked hypotheses with key evidence."
        )

    def _run_reporter(self) -> None:
        self._progress_ctx.current_step = 4
        pareto_section = "Pareto Curve / " if self._curve_mode() else ""
        if self._sol_enabled():
            reads = (
                f"Read `{self.task_path}`, `{self.benchmark_results_path}`, "
                f"`{self.profile_findings_path}`, and "
                f"`{self.sol_projection_path}` in full."
            )
            weighing = (
                "weighing the Analyzer's ranked hypotheses against the "
                "benchmark numbers, the ncu per-kernel analysis (each hot "
                "kernel's bound class), and the SOL projection (plus the "
                "per-op correlation table in the findings, when present) — "
                "the projected headroom sizes the win; if the projection "
                "declares itself unavailable, say so in the report and "
                "weigh measured evidence only. Ground each recommendation "
                "in the nsys timeline + ncu kernel analysis + SOL "
                "correlation per your system prompt"
            )
            sections = (
                f"(Executive Summary / Configuration / Benchmark Results / "
                f"{pareto_section}Profiling Findings / Projection vs Measured "
                f"/ Main Bottleneck / Recommendations)"
            )
        else:
            reads = (
                f"Read `{self.task_path}`, `{self.benchmark_results_path}`, "
                f"and `{self.profile_findings_path}` in full."
            )
            weighing = (
                "weighing the Analyzer's ranked hypotheses against the "
                "benchmark numbers and the ncu per-kernel analysis (each "
                "hot kernel's bound class); ground each recommendation in "
                "the nsys timeline + ncu kernel analysis per your system "
                "prompt"
            )
            sections = (
                f"(Executive Summary / Configuration / Benchmark Results / "
                f"{pareto_section}Profiling Findings / Main Bottleneck / "
                f"Recommendations)"
            )
        self.reporter(
            f"Workspace: {self.workspace}\n\n"
            f"{reads} Decide the **single "
            f"dominant** bottleneck using the taxonomy in your system prompt, "
            f"{weighing}.\n\n"
            f"`Write` `{self.report_path}` with every required section "
            f"{sections}, then "
            f"`Write` `{self.report_html_path}` mirroring it 1:1 "
            f"(self-contained, interactive, with the required top-kernel "
            f"share bars — see your system prompt). The "
            f"Executive Summary and Main Bottleneck must name **exactly one** "
            f"headline category.\n\n"
            f"Before completing your turn, call `append_reporter_progress` "
            f"with a `summary` of the main bottleneck, the evidence backing "
            f"it, and confirmation that both files were written."
        )

    def _task_data(self) -> dict[str, Any]:
        """Parse the resolved ``task.yaml`` on disk, or ``{}`` if unreadable.

        The resolved spec is written by ``_init_state`` before any stage
        runs, so by the time a driving prompt needs it the file exists;
        the fallback keeps prompt construction total anyway.
        """
        try:
            data = yaml.safe_load(self.task_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

    def _sol_enabled(self) -> bool:
        """Whether the resolved task spec enables the projector stage.

        On by default — only ``sol.enabled: false`` turns it off.
        """
        return sol_enabled(self._task_data())

    def _curve_mode(self) -> bool:
        """Whether the resolved task spec runs in Pareto-curve mode."""
        return is_curve_mode(self._task_data())

    def _curve_points(self) -> list[int]:
        """The configured concurrency points (ascending in curve mode)."""
        return concurrency_points(self._task_data())

    def _trtllm_hint(self) -> str:
        """Best-effort ``trtllm_repo_path`` for the analyzer's grep hint.

        Falls back to a generic phrase if the resolved ``task.yaml`` isn't
        readable yet (it always is by the time the analyzer runs).
        """
        repo = self._task_data().get("trtllm_repo_path")
        if repo:
            return f"{repo}/tensorrt_llm"
        return "<trtllm_repo_path>/tensorrt_llm"


if __name__ == "__main__":
    from .cli import main

    main()

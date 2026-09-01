"""Tests for the perf-analyze orchestration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from agent_flow import CLAUDE_CODE_DEFAULT_MODEL
from agent_flow.workflows.perf_analyze import progress as progress_module
from agent_flow.workflows.perf_analyze import state as state_module
from agent_flow.workflows.perf_analyze import workflow as workflow_module
from agent_flow.workflows.perf_analyze.prompts import build_perf_analyze_prompts
from agent_flow.workflows.perf_analyze.sol_methodology import SolMethodology

Workflow = workflow_module.PerfAnalyzeWorkflow


# --------------------------------------------------------------------- helpers


def _write_task(tmp_path, sol: bool | None = None) -> Path:
    """Write a minimal-but-valid input task.yaml; return its path.

    ``sol=None`` (the default) writes no ``sol`` block at all, which is
    the production default — the projector stage runs. ``sol=True`` adds
    the block with a ``gpu`` hint (still enabled); ``sol=False`` writes
    the opt-out that skips the stage.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    spec: dict = {"checkpoint_path": str(ckpt), "trtllm_repo_path": str(repo)}
    if sol is True:
        spec["sol"] = {"gpu": "H100"}
    elif sol is False:
        spec["sol"] = {"enabled": False}
    task = tmp_path / "input_task.yaml"
    task.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task


def _write_ws_task(ws: Path, sol: bool | None = None, concurrency=None) -> None:
    """Seed a resolved ``task.yaml`` in the workspace (for resume tests).

    Stage gating on resume reads this file (not the ``--task`` input), so
    resume tests control the ``sol`` block here — with the same
    ``None`` / ``True`` / ``False`` meaning as :func:`_write_task`.
    ``concurrency`` seeds the ``benchmark`` block (a list turns on
    Pareto-curve mode).
    """
    spec: dict = {"checkpoint_path": "/ckpt", "trtllm_repo_path": "/repo"}
    if sol is True:
        spec["sol"] = {"gpu": "H100"}
    elif sol is False:
        spec["sol"] = {"enabled": False}
    if concurrency is not None:
        spec["benchmark"] = {"concurrency": concurrency}
    (ws / "task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


def _stub_agents(workflow):
    """Replace the workflow's agent entry points with recorders.

    Returns ``trace`` — the order in which stages executed. Real backends
    are never invoked; each stub appends the role's progress entry (with
    the pipeline's fixed step numbers: projector=2, analyzer=3, reporter=4).
    """
    trace: list[str] = []

    def _append(entry: dict) -> None:
        data = progress_module.read_progress(workflow.progress_path)
        data["analysis"].append(entry)
        progress_module.write_progress(workflow.progress_path, data)

    def benchmarker():
        trace.append("benchmarker")
        workflow.benchmark_results_path.write_text("# benchmark\n", encoding="utf-8")
        _append({"step": 1, "agent": "benchmarker", "summary": "b"})

    def projector():
        trace.append("projector")
        workflow.sol_projection_path.write_text("# projection\n", encoding="utf-8")
        _append({"step": 2, "agent": "projector", "summary": "d"})

    def analyzer():
        trace.append("analyzer")
        workflow.profile_findings_path.write_text("# findings\n", encoding="utf-8")
        _append({"step": 3, "agent": "analyzer", "summary": "p"})

    def reporter():
        trace.append("reporter")
        workflow.report_path.write_text("# report\n", encoding="utf-8")
        workflow.report_html_path.write_text("<html></html>", encoding="utf-8")
        _append({"step": 4, "agent": "reporter", "summary": "r"})

    workflow._run_benchmarker = benchmarker
    workflow._run_projector = projector
    workflow._run_analyzer = analyzer
    workflow._run_reporter = reporter
    return trace


# ---------------------------------------------------------- fresh-start guard


def test_fresh_start_raises_when_benchmark_results_non_empty(tmp_path):
    (tmp_path / "benchmark_results.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="benchmark_results.md"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_report_non_empty(tmp_path):
    (tmp_path / "performance_report.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="performance_report.md"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_sol_projection_non_empty(tmp_path):
    (tmp_path / "sol_projection.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="sol_projection.md"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_report_html_non_empty(tmp_path):
    (tmp_path / "performance_report.html").write_text("<html></html>", encoding="utf-8")
    with pytest.raises(FileExistsError, match="performance_report.html"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_progress_has_real_entries(tmp_path):
    progress_module.write_progress(
        tmp_path / "progress.yaml",
        {"analysis": [{"step": 1, "agent": "benchmarker", "summary": "from a prior run"}]},
    )
    with pytest.raises(FileExistsError, match="progress.yaml"):
        Workflow(workspace=tmp_path)


def test_fresh_start_allows_progress_shell(tmp_path):
    """An empty ``{analysis: []}`` shell must not block a retry."""
    progress_module.init_progress_file(tmp_path / "progress.yaml")
    workflow = Workflow(workspace=tmp_path)
    try:
        assert workflow.resume is False
        assert progress_module.read_progress(tmp_path / "progress.yaml")["analysis"] == []
    finally:
        workflow.close()


def test_fresh_start_allows_empty_files(tmp_path):
    (tmp_path / "benchmark_results.md").write_text("\n  \n", encoding="utf-8")
    (tmp_path / "performance_report.md").write_text("", encoding="utf-8")
    workflow = Workflow(workspace=tmp_path)
    try:
        assert workflow.resume is False
    finally:
        workflow.close()


# ------------------------------------------------------- stubbed orchestration


def test_fresh_run_executes_stages_in_order(tmp_path):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # No `sol` block → the projector runs anyway; it is on by default.
    assert trace == ["benchmarker", "projector", "analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert state.benchmarker_done is True
    assert state.projector_done is True
    assert state.analyzer_done is True
    assert state.reporter_done is True

    # The normalized spec with resolved defaults is materialized for the agents.
    resolved = yaml.safe_load((ws / "task.yaml").read_text(encoding="utf-8"))
    assert "serve" not in resolved
    assert resolved["sol"] == {"enabled": True}
    assert resolved["benchmark"]["random_input_len"] == 1024
    assert resolved["profile"]["methods"] == ["nsys", "torch", "ncu"]


def test_fresh_run_with_sol_hint_runs_projector(tmp_path):
    task = _write_task(tmp_path, sol=True)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector", "analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert state.projector_done is True

    # The user's `gpu` hint survives the merged-in gate default.
    resolved = yaml.safe_load((ws / "task.yaml").read_text(encoding="utf-8"))
    assert resolved["sol"] == {"enabled": True, "gpu": "H100"}


def test_fresh_run_with_sol_disabled_skips_projector(tmp_path):
    """``sol.enabled: false`` is the opt-out from the default-on stage."""
    task = _write_task(tmp_path, sol=False)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert state.projector_done is False
    # The projection file stays blank rather than half-written.
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == ""


def test_resume_from_analyzer_skips_benchmarker(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            benchmarker_done=True,
            stage=state_module.STAGE_ANALYZER,
        ),
    )
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True


def test_resume_at_projector_with_block_runs_it(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_task(ws, sol=True)
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            benchmarker_done=True,
            stage=state_module.STAGE_PROJECTOR,
        ),
    )
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["projector", "analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.projector_done is True
    assert state.done is True


def test_resume_at_projector_when_disabled_skips_forward(tmp_path):
    """Checkpoint parked at the projector, but the stage was turned off.

    The stage transition is unconditional while execution is gated on the
    resolved task.yaml, so this resume must skip straight to the analyzer
    instead of deadlocking on a stage that can never run.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_task(ws, sol=False)
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            benchmarker_done=True,
            stage=state_module.STAGE_PROJECTOR,
        ),
    )
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["analyzer", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.projector_done is False
    assert state.done is True


def test_resume_from_reporter_skips_benchmarker_and_analyzer(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            benchmarker_done=True,
            analyzer_done=True,
            stage=state_module.STAGE_REPORTER,
        ),
    )
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["reporter"]


def test_already_done_checkpoint_short_circuits(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            done=True,
            stage=state_module.STAGE_REPORTER,
        ),
    )
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run("ignored")
    finally:
        workflow.close()
    assert trace == []


# ----------------------------------------------------- stage-output gate


def test_analyzer_without_findings_blocks_advance(tmp_path):
    """A analyzer that writes no profile_findings.md must not reach the reporter.

    Reproduces the real failure where the analyzer only launched a server
    (recording an interim progress entry) and yielded; the workflow then
    ran the reporter against an empty profile_findings.md.
    """
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def analyzer_no_output():
        trace.append("analyzer")
        # Records nothing on disk — mirrors an agent that yielded before
        # writing its deliverable.

    workflow._run_analyzer = analyzer_no_output
    try:
        with pytest.raises(RuntimeError, match="profile_findings.md"):
            workflow.run(str(task))
    finally:
        workflow.close()

    # Reporter never ran, and the checkpoint stays parked at the analyzer
    # so a re-run retries it rather than skipping ahead.
    assert trace == ["benchmarker", "projector", "analyzer"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.benchmarker_done is True
    assert state.analyzer_done is False
    assert state.stage == state_module.STAGE_ANALYZER
    assert state.done is False


def test_projector_without_projection_blocks_advance(tmp_path):
    """A projector that writes no sol_projection.md must not advance.

    Even a failed projection run is required to leave an honest
    "projection unavailable" file; an empty deliverable parks the
    checkpoint at the projector so a re-run retries the stage.
    """
    task = _write_task(tmp_path, sol=True)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def projector_no_output():
        trace.append("projector")
        # Writes nothing — mirrors an agent that yielded before its
        # deliverable (e.g. mid-bootstrap).

    workflow._run_projector = projector_no_output
    try:
        with pytest.raises(RuntimeError, match="sol_projection.md"):
            workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.benchmarker_done is True
    assert state.projector_done is False
    assert state.stage == state_module.STAGE_PROJECTOR
    assert state.done is False


def test_reporter_requires_both_md_and_html(tmp_path):
    """The reporter must produce both the markdown and the HTML companion."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def reporter_md_only():
        trace.append("reporter")
        workflow.report_path.write_text("# report\n", encoding="utf-8")
        # Deliberately omits performance_report.html.

    workflow._run_reporter = reporter_md_only
    try:
        with pytest.raises(RuntimeError, match="performance_report.html"):
            workflow.run(str(task))
    finally:
        workflow.close()

    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.reporter_done is False
    assert state.done is False
    assert state.stage == state_module.STAGE_REPORTER


# ------------------------------------------------------------------- --clean


def test_clean_overwrites_stale_managed_files(tmp_path):
    # Without --clean, stale content blocks a fresh start.
    (tmp_path / "performance_report.md").write_text("# stale\n", encoding="utf-8")
    (tmp_path / "sol_projection.md").write_text("# stale projection\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        Workflow(workspace=tmp_path)

    # With --clean, the managed files are reset and construction succeeds.
    workflow = Workflow(workspace=tmp_path, clean=True)
    try:
        assert workflow.resume is False
        assert (tmp_path / "performance_report.md").read_text(encoding="utf-8") == ""
        assert (tmp_path / "sol_projection.md").read_text(encoding="utf-8") == ""
    finally:
        workflow.close()


# --------------------------------------------------------------- agent wiring


def test_all_agents_use_claude_code_backend(tmp_path):
    workflow = Workflow(workspace=tmp_path / "ws")
    try:
        for layer in (
            workflow.benchmarker,
            workflow.projector,
            workflow.analyzer,
            workflow.reporter,
        ):
            assert layer.config.backend.kind == "claude-code"
            assert layer.config.backend.model == CLAUDE_CODE_DEFAULT_MODEL
            # Each role is gated by a required-tool stop hook.
            assert layer.config.backend.hooks is not None
    finally:
        workflow.close()


def test_no_role_wires_an_external_mcp_server(tmp_path):
    """No role ships a hosted endpoint.

    Internal knowledge is reached through a skill/subagent the session
    may or may not have, never a URL baked into this package -- a
    site-specific endpoint here would be dead for everyone else and
    would leak the site.
    """
    workflow = Workflow(workspace=tmp_path / "ws")
    try:
        for layer in (
            workflow.benchmarker,
            workflow.projector,
            workflow.analyzer,
            workflow.reporter,
        ):
            assert not layer.config.backend.extra_mcp_servers, layer.config.name
    finally:
        workflow.close()


def test_analyzer_orchestration_prompt_avoids_removed_builtin_tools(tmp_path):
    """The per-turn analyzer instruction must not name the removed tools.

    The agents run on the CLI's ``default`` toolset, which no longer
    includes ``Grep``/``Glob``. The orchestration prompt the workflow
    hands the analyzer each turn must steer source search to shell
    ``grep`` via ``Bash`` rather than instructing a nonexistent tool.
    """
    workflow = Workflow(workspace=tmp_path / "ws")

    captured: dict[str, str] = {}

    def capture(prompt: str) -> None:
        captured["prompt"] = prompt

    original = workflow.analyzer
    workflow.analyzer = capture
    try:
        workflow._run_analyzer()
    finally:
        workflow.analyzer = original
        workflow.close()

    prompt = captured["prompt"]
    for name in ("Grep", "Glob"):
        assert not re.search(rf"\b{name}\b", prompt), name
    assert "grep" in prompt
    assert "`Bash`" in prompt


def test_orchestration_prompts_tell_serving_roles_to_load_casebook(tmp_path):
    """Driving prompts reinforce the casebook load for both serving roles.

    The per-turn benchmarker/analyzer messages must reinforce the system
    prompt: load the ``perf-optimization-casebook`` skill as read-only
    reference before analyzing the TRT-LLM run. Reinforcing it in the
    driving prompt (not only the system prompt) makes the proactive load
    reliable each turn.
    """
    workflow = Workflow(workspace=tmp_path / "ws")

    captured: dict[str, str] = {}

    def _capture(role: str):
        def _cap(prompt: str) -> None:
            captured[role] = prompt

        return _cap

    originals = (workflow.benchmarker, workflow.analyzer)
    workflow.benchmarker = _capture("benchmarker")
    workflow.analyzer = _capture("analyzer")
    try:
        workflow._run_benchmarker()
        workflow._run_analyzer()
    finally:
        workflow.benchmarker, workflow.analyzer = originals
        workflow.close()

    for role in ("benchmarker", "analyzer"):
        assert "perf-optimization-casebook" in captured[role], role
        assert "`Skill` tool" in captured[role], role


def test_orchestration_prompts_keep_serving_roles_in_one_turn(tmp_path):
    """Both serving roles' driving prompts carry the single-turn rule.

    A serving role that ends its turn to wait for a background poll is
    never re-invoked, so the stage would advance with its deliverable
    still empty. The analyzer always carried the reminder in its driving
    prompt; the benchmarker inherits it from perf-optimize, whose
    benchmarker instruction has carried it from the start.
    """
    workflow = Workflow(workspace=tmp_path / "ws")

    captured: dict[str, str] = {}

    def _capture(role: str):
        def _cap(prompt: str) -> None:
            captured[role] = prompt

        return _cap

    originals = (workflow.benchmarker, workflow.analyzer)
    workflow.benchmarker = _capture("benchmarker")
    workflow.analyzer = _capture("analyzer")
    try:
        workflow._run_benchmarker()
        workflow._run_analyzer()
    finally:
        workflow.benchmarker, workflow.analyzer = originals
        workflow.close()

    for role in ("benchmarker", "analyzer"):
        assert "single turn" in captured[role], role
        assert "foreground" in captured[role], role


def test_each_agent_has_its_progress_tools(tmp_path):
    workflow = Workflow(workspace=tmp_path / "ws")
    expected = {
        "benchmarker": "append_benchmarker_progress",
        "projector": "append_projector_progress",
        "analyzer": "append_analyzer_progress",
        "reporter": "append_reporter_progress",
    }
    try:
        for role, append_name in expected.items():
            layer = getattr(workflow, role)
            tool_names = [t.name for t in layer.config.backend.tools]
            assert append_name in tool_names
            assert "read_latest_progress" in tool_names
    finally:
        workflow.close()


def test_slurm_prompts_augment_only_serving_roles():
    base = build_perf_analyze_prompts(include_slurm_environment=False)
    slurm = build_perf_analyze_prompts(include_slurm_environment=True)
    assert "slurm-environment" in slurm.benchmarker
    assert "slurm-environment" in slurm.analyzer
    # The reporter never launches a server, so it is unchanged; neither
    # does the projector (it runs locally — under Slurm on the login
    # node, where it degrades to unmeasured latency constants).
    assert slurm.reporter == base.reporter
    assert slurm.projector == base.projector
    assert "slurm-environment" not in base.benchmarker


# -------------------------------------------------------- projector prompts


def _capture_prompt(workflow, role: str, run_method: str) -> str:
    """Run ``run_method`` with the role's agent replaced by a recorder."""
    captured: dict[str, str] = {}

    def _cap(prompt: str) -> None:
        captured["prompt"] = prompt

    original = getattr(workflow, role)
    setattr(workflow, role, _cap)
    try:
        getattr(workflow, run_method)()
    finally:
        setattr(workflow, role, original)
    return captured["prompt"]


def test_projector_driving_prompt_is_skill_based(tmp_path):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _write_ws_task(ws, sol=True)
    try:
        prompt = _capture_prompt(workflow, "projector", "_run_projector")
    finally:
        workflow.close()

    # The methodology is the SOL skill, named by the spelling this session
    # actually loaded (resolved in Python before the run): peaks come from
    # the skill's calculator, latency constants from its channel
    # measurement, and the architecture from the checkpoint's config.json.
    assert "internal-perf-sol-analysis" in prompt
    assert "`Skill` tool" in prompt
    assert "peaks calculator" in prompt
    assert "measure_channels.py" in prompt
    assert "config.json" in prompt
    # No trace of the removed dlsim cross-check.
    assert "dlsim" not in prompt
    # It derives the SOL ceiling with shown arithmetic.
    assert "speed-of-light" in prompt
    assert "SOL ceiling" in prompt
    assert "numbers substituted" in prompt
    assert "sol_projection.md" in prompt
    # Single-turn discipline.
    assert "single turn" in prompt
    # The internal-knowledge route lives in the system prompt, not here.
    assert "internal-glean-search" not in prompt
    assert "internal-glean-specialist" not in prompt
    # Honest degradation, never fabrication.
    assert "Projection unavailable" in prompt
    assert "never fabricate" in prompt
    # The methodology was resolved in Python, so this message names one
    # skill: the default is the SOL skill, quoted by the spelling to load.
    assert "**load the `internal-perf-sol-analysis` skill**" in prompt
    assert "perf-analysis` skill**" not in prompt.replace("sol-analysis` skill**", "")
    # The machine-readable peaks file is persisted for the analyzer.
    assert "sol_work/peaks.json" in prompt
    # No removed builtin tools.
    for name in ("Grep", "Glob"):
        assert not re.search(rf"\b{name}\b", prompt), name


def test_analyzer_prompt_mentions_projection_only_when_enabled(tmp_path):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    try:
        _write_ws_task(ws, sol=False)
        without = _capture_prompt(workflow, "analyzer", "_run_analyzer")
        _write_ws_task(ws, sol=True)
        with_sol = _capture_prompt(workflow, "analyzer", "_run_analyzer")
    finally:
        workflow.close()

    assert "sol_projection.md" not in without
    assert "sol_projection.md" in with_sol
    assert "optional context" in with_sol
    # The projection never outranks measured trace evidence.
    assert "outranks" in with_sol
    # The measured↔SOL correlation is instructed only when the projector
    # stage ran: skill load, regions from traces, analyze vs the peaks
    # file, and the dedicated findings section.
    for marker in (
        "sol_calc.py analyze",
        "internal-perf-sol-analysis",
        "regions.json",
        "sol_work/peaks.json",
        "SOL correlation",
        "Correlation unavailable",
    ):
        assert marker not in without, marker
        assert marker in with_sol, marker


def test_analyzer_prompt_instructs_the_ncu_deep_dive(tmp_path):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    try:
        _write_ws_task(ws, sol=False)
        without = _capture_prompt(workflow, "analyzer", "_run_analyzer")
        _write_ws_task(ws, sol=True)
        with_sol = _capture_prompt(workflow, "analyzer", "_run_analyzer")
    finally:
        workflow.close()

    # The ncu deep dive is not SOL-gated — both variants instruct it:
    # skill-driven capture + interpretation on the top nsys kernels, with
    # the artifacts saved next to the other traces, and the findings'
    # dedicated section in the required structure.
    for prompt in (without, with_sol):
        assert "perf-nsight-compute-analysis" in prompt
        assert "trtllm-agent-toolkit:perf-nsight-compute-analysis" in prompt
        assert "server_ncu.ncu-rep" in prompt
        assert "ncu kernel analysis" in prompt
        assert "default all three" in prompt


def test_reporter_prompt_mentions_projection_only_when_enabled(tmp_path):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    try:
        _write_ws_task(ws, sol=False)
        without = _capture_prompt(workflow, "reporter", "_run_reporter")
        _write_ws_task(ws, sol=True)
        with_sol = _capture_prompt(workflow, "reporter", "_run_reporter")
    finally:
        workflow.close()

    assert "sol_projection.md" not in without
    assert "Projection vs Measured" not in without
    assert "sol_projection.md" in with_sol
    assert "Projection vs Measured" in with_sol
    # The projection weighs into the verdict, honestly degrading.
    assert "SOL projection" in with_sol
    assert "unavailable" in with_sol


# ------------------------------------------------------ curve-mode prompts


def test_driving_prompts_switch_on_concurrency_list(tmp_path):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    try:
        _write_ws_task(ws, concurrency=64)
        scalar = {
            role: _capture_prompt(workflow, role, f"_run_{role}")
            for role in ("benchmarker", "analyzer", "reporter")
        }
        _write_ws_task(ws, concurrency=[8, 32, 128])
        curve = {
            role: _capture_prompt(workflow, role, f"_run_{role}")
            for role in ("benchmarker", "analyzer", "reporter")
        }
    finally:
        workflow.close()

    # Benchmarker: one run per point with per-point result dirs — only in
    # curve mode.
    assert "single configured operating point" in scalar["benchmarker"]
    assert "concurrency_<c>" not in scalar["benchmarker"]
    assert "once per concurrency point [8, 32, 128]" in curve["benchmarker"]
    assert "concurrency_<c>" in curve["benchmarker"]

    # Analyzer: pinned to the largest point in curve mode only.
    assert "largest" not in scalar["analyzer"]
    assert "largest concurrency point, 128" in curve["analyzer"]
    assert "--max-concurrency 128" in curve["analyzer"]

    # Reporter: the Pareto Curve section is named in curve mode only.
    assert "Pareto Curve" not in scalar["reporter"]
    assert "Pareto Curve" in curve["reporter"]


# ------------------------------------------------- stale benchmark artifacts


def test_clear_stale_benchmark_results_removes_only_result_artifacts(tmp_path):
    (tmp_path / "concurrency_8").mkdir()
    (tmp_path / "concurrency_8" / "openai-infqps-concurrency8-m-1.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "concurrency_256").mkdir()  # pre-created, never filled
    (tmp_path / "openai-infqps-concurrency64-m-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "serve.log").write_text("log", encoding="utf-8")
    (tmp_path / "run_benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "analysis").mkdir()
    (tmp_path / "item_1_opt-001").mkdir()  # nested stage dirs: top-level scan only

    removed = workflow_module.clear_stale_benchmark_results(tmp_path)

    assert removed == [
        "concurrency_256/",
        "concurrency_8/",
        "openai-infqps-concurrency64-m-1.json",
    ]
    assert not (tmp_path / "concurrency_8").exists()
    assert not (tmp_path / "concurrency_256").exists()
    assert not (tmp_path / "openai-infqps-concurrency64-m-1.json").exists()
    assert (tmp_path / "serve.log").is_file()
    assert (tmp_path / "run_benchmarks.sh").is_file()
    assert (tmp_path / "analysis").is_dir()
    assert (tmp_path / "item_1_opt-001").is_dir()


def test_clear_stale_benchmark_results_missing_dir_is_noop(tmp_path):
    assert workflow_module.clear_stale_benchmark_results(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# Which SOL methodology skill the session has is resolved in Python before the
# run (``resolve_sol_methodology``) and handed to the workflow, so the
# projector is told to load a skill that is actually there. Nothing else in
# the pipeline changes: the analyzer's correlation and the reporter's weighing
# already degrade on their own when the projection or the peaks file is
# missing.
# --------------------------------------------------------------------------- #


def _projector_prompt_for(tmp_path, methodology: SolMethodology) -> str:
    ws = tmp_path / f"ws-{methodology.name}-{methodology.probed}"
    workflow = Workflow(workspace=ws, sol_methodology=methodology)
    _write_ws_task(ws, sol=True)
    try:
        return _capture_prompt(workflow, "projector", "_run_projector")
    finally:
        workflow.close()


def test_projector_driving_prompt_names_the_resolved_fallback_skill(tmp_path):
    """The open-source-toolkit case: `perf-analysis` stands in."""
    prompt = _projector_prompt_for(
        tmp_path, SolMethodology(name="reduced", skill="trtllm-agent-toolkit:perf-analysis")
    )
    # The loaded spelling is quoted verbatim — the agent never has to guess
    # between the bare and the plugin-qualified name.
    assert "Load the `trtllm-agent-toolkit:perf-analysis` skill" in prompt
    assert "not installed in this session" in prompt
    # No calculator, so no peaks file — nothing downstream reads one.
    assert "peaks calculator you do not have" in prompt
    assert "do **not** write" in prompt
    assert "measure_channels.py" not in prompt
    # And it still degrades honestly rather than inventing a ceiling.
    assert "Projection unavailable" in prompt
    assert "never fabricate numbers" in prompt


def test_projector_driving_prompt_fails_open_when_the_probe_could_not_run(tmp_path):
    """An unreachable probe must not silently downgrade a stage the user asked for."""
    prompt = _projector_prompt_for(tmp_path, SolMethodology(probed=False))
    assert "load the `internal-perf-sol-analysis` skill" in prompt
    # ...and the agent is handed both spellings, since the name is a guess.
    assert "trtllm-agent-toolkit:internal-perf-sol-analysis" in prompt
    assert "if the bare name is not found" in prompt

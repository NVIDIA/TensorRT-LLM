"""Tests for the perf-optimize orchestration."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow import CLAUDE_CODE_DEFAULT_MODEL
from agent_flow.workflows.perf_analyze.sol_methodology import SolMethodology
from agent_flow.workflows.perf_optimize import progress as progress_module
from agent_flow.workflows.perf_optimize import roadmap_schema, task_schema
from agent_flow.workflows.perf_optimize import state as state_module
from agent_flow.workflows.perf_optimize import workflow as workflow_module

Workflow = workflow_module.PerfOptimizeWorkflow

_ROLES = (
    "benchmarker",
    "projector",
    "analyzer",
    "optimizer",
    "evaluator",
    "qa",
    "reporter",
)
_AGENT_ROLES = (*_ROLES, "integrator")


# --------------------------------------------------------------------- helpers


def _write_task(tmp_path, extra: dict | None = None) -> Path:
    """Write a minimal-but-valid input task.yaml; return its path."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    data = {"checkpoint_path": str(ckpt), "trtllm_repo_path": str(repo)}
    data.update(extra or {})
    task = tmp_path / "input_task.yaml"
    task.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return task


def _sol_extra(tmp_path) -> dict:
    """A ``sol`` task block with the projector's ``gpu`` hint.

    The stage is on by default, so this only adds the hint — see
    :func:`_sol_off_extra` for the opt-out.
    """
    return {"sol": {"gpu": "H100"}}


def _sol_off_extra() -> dict:
    """A ``sol`` task block that turns the projector stage off."""
    return {"sol": {"enabled": False}}


class FakeGitOps:
    """Records git calls; pretends the checkout is a git repo.

    ``worktree_clean`` always returns ``False`` (the accept path sees
    the optimizer's edits and commits them).
    """

    def __init__(self, is_repo: bool = True):
        self.calls: list[tuple] = []
        self.is_repo_flag = is_repo
        # Which host the workflow routed git to, or "" for local. Recorded rather
        # than swallowed so a test can assert the choice was made — the failure it
        # guards is a resumed run silently issuing git against a local path that
        # does not exist.
        self.cluster: str | None = None

    def use_cluster(self, ssh_alias: str) -> None:
        self.calls.append(("use_cluster", ssh_alias))
        self.cluster = ssh_alias

    def is_git_repo(self, repo):
        self.calls.append(("is_git_repo", str(repo)))
        return self.is_repo_flag

    def worktree_clean(self, repo):
        self.calls.append(("worktree_clean", str(repo)))
        return str(repo).split("/")[-1] == "integration"

    def current_branch(self, repo):
        return "main"

    def rev_parse_head(self, repo):
        self.calls.append(("rev_parse_head", str(repo)))
        return "b" * 40

    def create_branch(self, repo, name):
        self.calls.append(("create_branch", name))

    def checkout(self, repo, name):
        self.calls.append(("checkout", name))

    def commit_all(self, repo, message):
        self.calls.append(("commit_all", message))
        return "c" * 40

    def discard_uncommitted(self, repo):
        self.calls.append(("discard_uncommitted", str(repo)))

    def create_worktree(self, repo, path, branch, base):
        self.calls.append(("create_worktree", str(path), branch, base))
        Path(path).mkdir(parents=True, exist_ok=True)

    def remove_worktree(self, repo, path):
        self.calls.append(("remove_worktree", str(path)))
        shutil.rmtree(path, ignore_errors=True)

    def reset_to(self, repo, commit):
        self.calls.append(("reset_to", str(repo), commit))

    def fast_forward(self, repo, branch):
        self.calls.append(("fast_forward", str(repo), branch))

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


@pytest.fixture
def fake_git(monkeypatch):
    fake = FakeGitOps()
    monkeypatch.setattr(workflow_module, "gitops", fake)
    return fake


def _item(item_id: str = "opt-001", gain: float = 10.0, **overrides) -> dict:
    item = {
        "id": item_id,
        "title": f"Optimization {item_id}",
        "category": "launch-host",
        "approach": "config",
        "evidence": ["nsys: 31% GPU idle (nsys_stats.txt)"],
        "expected_gain_pct": gain,
        "expected_gain_rationale": "idle share x typical recovery",
        "how_to_apply": "edit tuning/extra_llm_api_options.yaml",
        "status": "pending",
        "attempts": 0,
        "measured_gain_pct": None,
    }
    item.update(overrides)
    return item


def _write_baseline_result_json(baseline_dir, value: float = 100.0) -> None:
    """Write the result JSON a measured baseline always leaves behind.

    ``benchmark_serving.py --save-result`` produces it, and the workflow
    treats it as the proof that the baseline was measured rather than
    merely reported on.
    """
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "result.json").write_text(
        json.dumps({"output_throughput": value, "completed": 1}), encoding="utf-8"
    )


def _stub_agents(
    workflow,
    *,
    analyzer_items: list[list[dict]] | None = None,
    evaluator_verdicts: list[tuple] | None = None,
    baseline_curve: list[dict] | None = None,
    evaluator_curve: list[dict] | None = None,
):
    """Replace the workflow's agent entry points with recorders.

    Returns ``trace`` — the order in which stages executed. Real backends
    are never invoked; each stub writes the stage's deliverable and
    appends the role's progress entry (the evaluator entries carry the
    structured decision fields the orchestrator branches on).

    - ``analyzer_items`` — items the analyzer stub *adds* to the roadmap,
      one list per invocation (later invocations default to adding none).
    - ``evaluator_verdicts`` — ``(decision, reason, gain, value)`` per
      evaluator invocation (the last one repeats).
    - ``baseline_curve`` — curve the analyzer stub writes on
      ``baseline``/``current_best`` (Pareto-curve mode runs).
    - ``evaluator_curve`` — ``curve`` field every evaluator entry carries.
    """
    trace: list[str] = []
    items_per_round = list(analyzer_items if analyzer_items is not None else [[_item()]])
    verdicts = list(evaluator_verdicts or [("APPROVE", "none", 8.4, 108.4)])
    counters = {"analyzer": 0, "evaluator": 0}

    def _append(entry: dict, local_path: Path | None = None) -> None:
        with workflow._progress_lock:
            data = progress_module.read_progress(workflow.progress_path)
            global_entry = dict(entry)
            global_entry["step"] = len(data["optimization"]) + 1
            data["optimization"].append(global_entry)
            progress_module.write_progress(workflow.progress_path, data)
        if local_path is not None:
            data = progress_module.read_progress(local_path)
            data["optimization"].append(entry)
            progress_module.write_progress(local_path, data)

    def benchmarker(state):
        trace.append("benchmarker")
        workflow.baseline_results_path.write_text("# baseline\n", encoding="utf-8")
        # A real benchmarker always leaves the result JSON its numbers came
        # from; the orchestrator gates on it, because a report can exist and
        # still carry no measurement.
        _write_baseline_result_json(workflow.baseline_dir)
        _append({"step": 1, "agent": "benchmarker", "summary": "b"})

    def projector(state):
        trace.append("projector")
        workflow.sol_projection_path.write_text("# SOL Projection\n", encoding="utf-8")
        _append({"step": 1, "agent": "projector", "summary": "p"})

    def analyzer(state):
        trace.append("analyzer")
        findings = workflow._analysis_dir(state) / "profile_findings.md"
        findings.write_text("# findings\n", encoding="utf-8")
        try:
            data = roadmap_schema.load_roadmap(workflow.roadmap_path)
        except roadmap_schema.RoadmapError:
            baseline: dict = {"value": 100.0, "source": "baseline/benchmark_results.md"}
            if baseline_curve is not None:
                baseline["curve"] = [dict(p) for p in baseline_curve]
            data = {
                "version": 1,
                "target_metric": "output_throughput",
                "baseline": baseline,
                "current_best": {k: v for k, v in baseline.items()},
                "items": [],
            }
        idx = counters["analyzer"]
        counters["analyzer"] += 1
        if idx < len(items_per_round):
            data["items"].extend(items_per_round[idx])
        roadmap_schema.save_roadmap(workflow.roadmap_path, data)
        _append({"step": 1, "agent": "analyzer", "summary": "a"})

    def optimizer(state, *, agent=None, progress_ctx=None):
        trace.append("optimizer")
        summary = workflow._attempt_dir(state) / "optimization_summary.md"
        summary.write_text("# summary\n", encoding="utf-8")
        _append(
            {
                "step": progress_ctx.current_step if progress_ctx else 1,
                "agent": "optimizer",
                "summary": "o",
                "attempt": state.attempt_index + 1,
                "item_id": state.current_item_id,
            },
            progress_ctx.path if progress_ctx else None,
        )

    def evaluator(state, *, agent=None, progress_ctx=None):
        trace.append("evaluator")
        report = workflow._attempt_dir(state) / "evaluation.md"
        report.write_text("# evaluation\n", encoding="utf-8")
        idx = counters["evaluator"]
        counters["evaluator"] += 1
        decision, reason, gain, value = verdicts[min(idx, len(verdicts) - 1)]
        entry = {
            "step": 1,
            "agent": "evaluator",
            "summary": "e",
            "decision": decision,
            "reason_category": reason,
            "measured_gain_pct": gain,
            "measured_value": value,
        }
        if evaluator_curve is not None:
            entry["curve"] = [dict(p) for p in evaluator_curve]
        entry["attempt"] = state.attempt_index + 1
        entry["item_id"] = state.current_item_id
        _append(entry, progress_ctx.path if progress_ctx else None)

    def integrator(prompt):
        state = state_module.load_state(workflow.state_path)
        candidates = [
            entry for entry in state.item_batch if entry.get("status") == "candidate_ready"
        ]
        integration_repo = Path(state.integration_worktree_path)
        if (integration_repo / ".git").exists():
            for candidate in candidates:
                commit = candidate.get("candidate_commit")
                if commit:
                    subprocess.run(
                        ["git", "-C", str(integration_repo), "cherry-pick", str(commit)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
        integration_dir = workflow._round_dir(state) / "integration"
        (integration_dir / "integration.md").write_text("# integration\n", encoding="utf-8")
        included = [str(entry["current_item_id"]) for entry in candidates]
        best = candidates[0] if candidates else {}
        verdict = {
            "agent": "integrator",
            "summary": "integrated",
            "decision": "APPROVE",
            "included_item_ids": included,
            "dropped_item_ids": [],
            "remediation_attempts": 0,
            "measured_gain_pct": float(best.get("measured_gain_pct") or 0),
            "measured_value": float(best.get("measured_value") or 100),
            "required_gain_pct": 0.0,
            "best_candidate_id": included[0] if included else "",
        }
        if best.get("curve"):
            verdict["curve"] = best["curve"]
        _append(verdict)

    def qa(state):
        trace.append("qa")
        workflow.verification_report_path.parent.mkdir(parents=True, exist_ok=True)
        workflow.verification_report_path.write_text("# verification\n", encoding="utf-8")
        _append(
            {
                "step": 1,
                "agent": "qa",
                "summary": "q",
                "cumulative_improvement_pct": 8.4,
            }
        )

    def reporter(state):
        trace.append("reporter")
        workflow.report_path.write_text("# report\n", encoding="utf-8")
        workflow.report_html_path.write_text("<html></html>", encoding="utf-8")
        _append({"step": 1, "agent": "reporter", "summary": "r"})

    workflow._run_benchmarker = benchmarker
    workflow._run_projector = projector
    workflow._run_analyzer = analyzer
    workflow._run_optimizer = optimizer
    workflow._run_evaluator = evaluator
    workflow.integrator = integrator
    workflow._run_qa = qa
    workflow._run_reporter = reporter
    return trace


# ------------------------------------------------------- stubbed orchestration


def test_happy_path_one_accepted_item(tmp_path, fake_git):
    """One item, accepted, and the closing profile the accept earns.

    Not one *round*: the roadmap runs dry with the accept outstanding, so
    the loop spends a second round profiling the build it is about to
    close on. Every campaign that accepts anything ends this way.
    """
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert state.benchmarker_done is True
    # No `sol` block → the projector stage still runs; it is on by
    # default and only `sol.enabled: false` turns it off.
    assert state.projector_done is True
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") != ""
    assert state.reporter_done is True
    assert state.campaign_git_branch.startswith("perf-optimize/")
    assert state.campaign_git_base_commit == "b" * 40

    # The orchestrator recorded the accepted item's lifecycle.
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "accepted"
    assert item["attempts"] == 1
    assert item["measured_gain_pct"] == pytest.approx(8.4)
    assert roadmap["current_best"]["value"] == pytest.approx(108.4)
    assert roadmap["current_best"]["source"] == "rounds/round_1/integration/integration.md"

    # One branch, one accept commit, no reverts.
    assert fake_git.count("create_branch") == 1
    assert fake_git.count("commit_all") == 1
    assert fake_git.count("discard_uncommitted") == 0
    (commit_message,) = [c[1] for c in fake_git.calls if c[0] == "commit_all"]
    assert "opt-001" in commit_message
    # The normalized spec and the tuning config pair were materialized.
    resolved = yaml.safe_load((ws / "task.yaml").read_text(encoding="utf-8"))
    assert resolved["optimize"]["max_rounds"] == 5
    assert resolved["optimize"]["max_items_per_round"] == 3
    assert resolved["optimize"]["item_execution"] == "parallel"
    assert (ws / "tuning" / "extra_llm_api_options.yaml").read_text(encoding="utf-8") == "{}\n"
    assert (ws / "tuning" / "extra_llm_api_options.accepted.yaml").read_text(
        encoding="utf-8"
    ) == "{}\n"


def test_integrator_verdict_is_not_recomputed_by_python(tmp_path, fake_git):
    """The prototype treats the Integrator's structured gate as authoritative."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    integrator_prompts: list[str] = []

    def authoritative_integrator(prompt):
        integrator_prompts.append(prompt)
        state = state_module.load_state(workflow.state_path)
        integration_dir = workflow._round_dir(state) / "integration"
        (integration_dir / "integration.md").write_text("# authoritative\n", encoding="utf-8")
        data = progress_module.read_progress(workflow.progress_path)
        data["optimization"].append(
            {
                "step": len(data["optimization"]) + 1,
                "agent": "integrator",
                "summary": "agent accepts despite a deliberately odd reported gate",
                "decision": "APPROVE",
                "included_item_ids": ["opt-001"],
                "dropped_item_ids": [],
                "remediation_attempts": 0,
                "measured_gain_pct": -50.0,
                "measured_value": 50.0,
                "required_gain_pct": 999.0,
                "best_candidate_id": "opt-001",
            }
        )
        progress_module.write_progress(workflow.progress_path, data)

    workflow.integrator = authoritative_integrator
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    roadmap = roadmap_schema.load_roadmap(workflow.roadmap_path)
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap["current_best"]["value"] == 50.0
    active_repo = str(ws / "worktrees" / "round_1" / "integration")
    assert f"Active runtime checkout: `{active_repo}`" in integrator_prompts[0]
    assert (
        f'export PYTHONPATH="{active_repo}${{PYTHONPATH:+:$PYTHONPATH}}"' in integrator_prompts[0]
    )


def test_relative_workspace_resolves_reference_result_dir(tmp_path, fake_git, monkeypatch):
    """A relative ``--workspace`` (the CLI default shape) round-trips.

    ``current_best.source`` is stored workspace-relative per the roadmap
    spec, so the evaluator's reference dir resolves to the accepted
    attempt instead of silently falling back to ``baseline/``.
    """
    monkeypatch.chdir(tmp_path)
    task = _write_task(tmp_path)
    ws = Path("workspace") / "perf-optimize"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    source = roadmap["current_best"]["source"]
    assert source == "rounds/round_1/integration/integration.md"
    reference = workflow._reference_result_dir()
    assert reference == ws / "rounds" / "round_1" / "integration"
    assert reference != workflow.baseline_dir

    # Pre-fix ledgers stored the path with the workspace prefix already
    # baked in — the reader still resolves those (as CWD-relative) so an
    # in-flight campaign survives the upgrade.
    legacy = dict(roadmap)
    legacy["current_best"] = dict(roadmap["current_best"])
    legacy["current_best"]["source"] = str(ws / source)
    roadmap_schema.save_roadmap(ws / "roadmap.yaml", legacy)
    assert workflow._reference_result_dir() == ws / "rounds" / "round_1" / "integration"


def test_sol_run_executes_projector_once_before_round_one(tmp_path, fake_git):
    """The projector runs exactly once, between the baseline and round 1.

    Later rounds never re-run it — the SOL ceiling is a property of the
    hardware + model + operating point, not of the applied optimizations.
    """
    task = _write_task(
        tmp_path,
        {
            **_sol_extra(tmp_path),
            "optimize": {"max_items_per_round": 1},
        },
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("APPROVE", "none", 8.4, 108.4), ("APPROVE", "none", 4.0, 112.7)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert state.projector_done is True
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == "# SOL Projection\n"


def test_git_routes_over_ssh_when_the_task_names_a_cluster(tmp_path, fake_git):
    """The routing is chosen before the first git command, on BOTH paths.

    A run whose ``slurm-environment`` carries ``cluster_ssh`` is running off-cluster:
    the checkout exists only on the cluster, so every git command has to travel over
    ssh. Absent, nothing changes — which is what every other test here relies on.

    The resume case is the one that decides where this call belongs. ``_init_state``
    returns early when resuming, so setting the route there would leave every resumed
    run issuing git against a local path that does not exist — and `is_git_repo`
    swallows its error, so the run would report "not a git repository" and point at
    the wrong machine.
    """
    slurm = {
        "slurm-environment": {
            "slurm_partition": "batch",
            "docker_image": "/img.sqsh",
            "cluster_ssh": "me@login-01",
        }
    }
    task = _write_task(tmp_path, slurm)
    ws = tmp_path / "ws"

    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    workflow.run(str(task))
    assert fake_git.cluster == "me@login-01"

    # RESUME: an UNFINISHED run reads the checkpointed task.yaml rather than the
    # input file, and must still route remotely. Rewinding `done` is what makes this
    # a resume — a completed workspace returns from `_init_state` as None and never
    # reaches any git command, which is correct and would make this vacuous.
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    state.done = False
    state.stage = state_module.STAGE_REPORTER
    state_module.save_state(ws / state_module.STATE_FILENAME, state)

    fake_git.cluster = None
    resumed = Workflow(workspace=ws)
    _stub_agents(resumed)
    resumed.run(str(task))
    assert fake_git.cluster == "me@login-01", (
        "a resumed run must route to the cluster too; `_init_state` returns early on "
        "resume, so a route set only on the fresh path would be lost here — and "
        "`is_git_repo` swallows its error, so the run would blame the wrong machine"
    )


def test_git_stays_local_without_a_cluster_ssh(tmp_path, fake_git):
    """No `cluster_ssh` means the historical behaviour, explicitly asserted."""
    task = _write_task(tmp_path)
    workflow = Workflow(workspace=tmp_path / "ws")
    _stub_agents(workflow)
    workflow.run(str(task))
    assert fake_git.cluster == "", "an ordinary run must not reach for an ssh host"


def test_resume_parked_at_projector_with_block_runs_it(tmp_path, fake_git):
    """A checkpoint parked at the projector resumes into it (sol set)."""
    task = _write_task(tmp_path, _sol_extra(tmp_path))
    ws = tmp_path / "ws"

    # First run: the benchmarker completes, then the projector dies
    # without output — the checkpoint parks at the projector stage.
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def projector_no_output(state):
        trace.append("projector")
        # Writes nothing — mirrors an agent that yielded before its
        # deliverable.

    workflow._run_projector = projector_no_output
    try:
        with pytest.raises(RuntimeError, match="sol_projection.md"):
            workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_PROJECTOR
    assert state.projector_done is False
    assert state.benchmarker_done is True

    # Resume: the projector runs (benchmarker is not re-run) and the
    # campaign completes.
    workflow = Workflow(workspace=ws)
    assert workflow.resume is True
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.projector_done is True
    assert state.done is True


def test_sol_disabled_skips_the_projector(tmp_path, fake_git):
    """``sol.enabled: false`` is the opt-out from the default-on stage."""
    task = _write_task(tmp_path, _sol_off_extra())
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert "projector" not in trace
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.projector_done is False
    # The managed output stays blank rather than half-written.
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == ""


def test_resume_parked_at_projector_when_disabled_skips_forward(tmp_path, fake_git):
    """The gate re-checks the resolved task.yaml on resume.

    A checkpoint parked at the projector whose stage was since turned
    off (here: disabled from the start — the checkpoint was seeded)
    skips it instead of dead-ending.
    """
    task = _write_task(tmp_path, _sol_off_extra())
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        workflow._init_state(str(task), None)  # materializes task.yaml + tuning
    finally:
        workflow.close()
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            benchmarker_done=True,
            campaign_git_branch="perf-optimize/seeded",
            campaign_git_base_commit="a" * 40,
            stage=state_module.STAGE_PROJECTOR,
        ),
    )
    (ws / "baseline").mkdir(exist_ok=True)
    (ws / "baseline" / "benchmark_results.md").write_text("# baseline\n", encoding="utf-8")
    _write_baseline_result_json(ws / "baseline")

    workflow = Workflow(workspace=ws)
    assert workflow.resume is True
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["analyzer", "optimizer", "evaluator", "analyzer", "qa", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.projector_done is False
    assert state.done is True


def test_evaluator_pushback_then_approve_retries_optimizer(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        evaluator_verdicts=[
            ("PUSH_BACK", "perf_shortfall", 0.2, 100.2),
            ("APPROVE", "none", 9.0, 109.0),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "accepted"
    assert item["attempts"] == 2
    assert item["measured_gain_pct"] == pytest.approx(9.0)
    assert roadmap["current_best"]["value"] == pytest.approx(109.0)

    # The pushed-back attempt was reverted exactly once, then committed once.
    assert fake_git.count("reset_to") >= 2
    assert fake_git.count("commit_all") == 1
    item_dir = ws / "rounds" / "round_1" / "item_1_opt-001"
    assert (item_dir / "attempt_1" / "evaluation.md").is_file()
    assert (item_dir / "attempt_2" / "evaluation.md").is_file()


def test_evaluator_reject_is_terminal_and_skips_final_verification(tmp_path, fake_git):
    """REJECT fails the item without burning the remaining attempts.

    And with nothing accepted, the final state IS the baseline — the
    final-verification benchmark is skipped.
    """
    task = _write_task(tmp_path)  # default max_attempts_per_item = 3
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        evaluator_verdicts=[("REJECT", "functionality", 0.0, 0.0)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The trailing analyzer is round 2's free replan turn: the roadmap
    # ran dry on a build nothing was accepted against, so the loop spends
    # a GPU-less round on the verdict before closing.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "failed"
    assert item["attempts"] == 1
    assert roadmap["current_best"]["value"] == pytest.approx(100.0)
    assert fake_git.count("reset_to") >= 2
    assert fake_git.count("commit_all") == 0
    assert not (ws / "final_verification").exists()
    assert state_module.load_state(ws / state_module.STATE_FILENAME).done is True


def test_pushback_attempts_exhausted_marks_failed(tmp_path, fake_git):
    task = _write_task(tmp_path, {"optimize": {"max_attempts_per_item": 2}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        evaluator_verdicts=[
            ("PUSH_BACK", "perf_shortfall", 0.3, 100.3),
            ("PUSH_BACK", "functionality", 0.0, 0.0),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The second PUSH_BACK landed on the final attempt → treated as
    # REJECT; nothing was accepted, so qa never runs. The trailing
    # analyzer is round 2's free replan turn on the dry roadmap.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "failed"
    assert item["attempts"] == 2
    # A failed item never advances the accepted-measurement watermark.
    assert roadmap["current_best"]["value"] == pytest.approx(100.0)
    assert fake_git.count("reset_to") >= 2
    assert fake_git.count("commit_all") == 0


def test_missing_evaluator_decision_counts_as_pushback(tmp_path, fake_git):
    task = _write_task(tmp_path, {"optimize": {"max_attempts_per_item": 2}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def evaluator_without_decision(state, **kwargs):
        trace.append("evaluator")
        (workflow._attempt_dir(state) / "evaluation.md").write_text("# eval\n", encoding="utf-8")
        path = kwargs["progress_ctx"].path
        data = progress_module.read_progress(path)
        data["optimization"].append({"step": 1, "agent": "evaluator", "summary": "no decision"})
        progress_module.write_progress(path, data)

    workflow._run_evaluator = evaluator_without_decision
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # No valid decision gets the push-back benefit of the doubt (one
    # retry), then fails at the attempt cap; nothing accepted → no qa.
    # The trailing analyzer is round 2's free replan turn on the dry
    # roadmap.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "failed"
    assert item["attempts"] == 2
    assert fake_git.count("reset_to") >= 2


def test_fixed_rounds_run_until_roadmap_exhausted(tmp_path, fake_git):
    """No agent stops the loop.

    Rounds keep running while actionable items remain, and the
    roadmap-exhausted break concludes the campaign.
    """
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("APPROVE", "none", 8.4, 108.4),
            ("APPROVE", "none", 3.0, 111.6),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Round 1 applies opt-001 (item budget 1), round 2 re-profiles and
    # applies opt-002. The roadmap is drained, but opt-002 changed the
    # build the plan was made against, so round 3 profiles what the
    # accept exposed before the break concludes the loop into the
    # one-shot final verification.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    assert (ws / "final_verification" / "verification_report.md").is_file()
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"
    assert roadmap["current_best"]["value"] == pytest.approx(111.6)
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.round_index == 3
    assert state.done is True


def test_round_that_accepts_nothing_opens_the_next_one_replan_only(tmp_path, fake_git):
    """A rejected attempt is hard-reverted, so there is nothing new to profile.

    The round still runs an analyzer turn — the verdicts it produced are
    evidence — but the orchestrator opens it in replan-only mode rather
    than paying to re-derive traces of a build whose tracked state is
    exactly what the standing analysis already describes.
    """
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    modes: list[bool] = []
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("REJECT", "perf_shortfall", -1.0, 99.0),
            ("APPROVE", "none", 6.0, 106.0),
        ],
    )
    original_analyzer = workflow._run_analyzer

    def analyzer_recording_mode(state):
        modes.append(workflow._replan_only(state))
        original_analyzer(state)

    workflow._run_analyzer = analyzer_recording_mode
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Round 1 profiles (nothing analyzed yet), round 2 re-plans (opt-001
    # was reverted), round 3 profiles again (opt-002 changed the build).
    assert modes == [False, True, False]
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    # Rejected parallel candidates are discarded with their isolated
    # worktree; the campaign checkout itself is never reverted.
    assert fake_git.count("discard_uncommitted") == 0
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    # Replan rounds plan from the standing profile and never advance it;
    # round 3's re-profile is the newest analysis of the current build.
    assert state.last_profiled_analysis_dir == str(ws / "rounds" / "round_3" / "analysis")


def test_dry_roadmap_without_accepts_replans_once_before_concluding(tmp_path, fake_git):
    """A dry roadmap is never the mid-round conclusion — it earns a replan turn.

    The roadmap ran out against a plan written *before* the round's
    verdicts existed, and a replan round costs no GPU time, so the loop
    spends one: the analyzer gets its chance to turn "opt-001 is dead"
    into the item nobody planned. Only when *that* turn also finds
    nothing does the top-of-loop break end the campaign — on a plan made
    against the measurements, not on the plan simply running out.

    Note the budget: ``max_items_per_round`` is at its default 3 and the
    roadmap holds one item, so the round never reaches its item cap. The
    replan turn has to come from the dry-roadmap path itself, or a small
    roadmap silently loses every round after the first.
    """
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    modes: list[bool] = []
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0)]],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -1.0, 99.0)],
    )
    original_analyzer = workflow._run_analyzer

    def analyzer_recording_mode(state):
        modes.append(workflow._replan_only(state))
        original_analyzer(state)

    workflow._run_analyzer = analyzer_recording_mode
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Round 2's analyzer runs, and runs free — and no final verification,
    # since the campaign accepted nothing to verify.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    assert modes == [False, True]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    # Two rounds counted, three left unspent — the campaign stopped
    # because the analyzer had nothing to say, not because it ran out.
    assert state.round_index == 2
    assert state.max_rounds == 5


def test_rejected_isolated_code_attempt_keeps_campaign_profile_current(tmp_path, fake_git):
    """Rejected worker artifacts disappear with their isolated worktree."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    modes: list[bool] = []
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0, approach="code")]],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -1.0, 99.0)],
    )
    original_analyzer = workflow._run_analyzer
    original_discard = fake_git.discard_uncommitted
    state_seen_at_revert: list[bool] = []

    def analyzer_recording_mode(state):
        modes.append(workflow._replan_only(state))
        original_analyzer(state)

    def discard_reading_checkpoint(repo):
        checkpoint = state_module.load_state(ws / state_module.STATE_FILENAME)
        state_seen_at_revert.append(checkpoint.profile_required)
        original_discard(repo)

    workflow._run_analyzer = analyzer_recording_mode
    fake_git.discard_uncommitted = discard_reading_checkpoint
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The worker checkout is removed without mutating the campaign checkout,
    # so round 2 can replan against round 1's still-current profile.
    assert modes == [False, True]
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    assert state_seen_at_revert == []
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.profile_required is False
    assert state.last_profiled_analysis_dir == str(ws / "rounds" / "round_1" / "analysis")


def test_a_dry_roadmap_out_of_rounds_concludes_without_another_turn(tmp_path, fake_git):
    """The replan turn a dry roadmap earns is still bounded by `max_rounds`."""
    task = _write_task(tmp_path, {"optimize": {"max_rounds": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0)]],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -1.0, 99.0)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector", "analyzer", "optimizer", "evaluator", "reporter"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.round_index == 1


def test_dry_roadmap_after_an_accept_respects_the_round_budget(tmp_path, fake_git):
    """The extra profile the accept earns is still bounded by `max_rounds`.

    Sibling of the happy path, which spends its second round profiling
    what the accept changed. Here there is no second round to spend, so
    the loop concludes and says why the build it closed on went
    unprofiled.
    """
    task = _write_task(tmp_path, {"optimize": {"max_rounds": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.round_index == 1
    # The campaign closed on a build no analyzer ever saw, and the profile
    # currency marker preserves that fact.
    assert state.profile_required is True


def test_target_improvement_reached_concludes_loop(tmp_path, fake_git):
    """The optional early stop is orchestrator-enforced off the ledger."""
    task = _write_task(tmp_path, {"optimize": {"target_improvement_pct": 5.0}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("APPROVE", "none", 8.4, 108.4)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The whole selected batch runs before the target is checked.
    assert trace[0:3] == ["benchmarker", "projector", "analyzer"]
    assert trace.count("optimizer") == 2
    assert trace.count("evaluator") == 2
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"
    assert state_module.load_state(ws / state_module.STATE_FILENAME).done is True


def test_multiple_items_applied_in_one_round(tmp_path, fake_git):
    """Several items share one analyzer profile + QA pass by default.

    Each still gets its own evaluation, accept commit, and roadmap
    bookkeeping.
    """
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 2}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("APPROVE", "none", 8.4, 108.4),
            ("APPROVE", "none", 3.0, 111.6),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Both items share round 1's profile; the integrated accept makes round
    # 2 profile the resulting campaign state before final verification.
    assert trace[0:3] == ["benchmarker", "projector", "analyzer"]
    assert trace.count("analyzer") == 2
    assert trace.count("optimizer") == 2
    assert trace.count("evaluator") == 2
    assert trace[-2:] == ["qa", "reporter"]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    one = roadmap_schema.find_item(roadmap, "opt-001")
    two = roadmap_schema.find_item(roadmap, "opt-002")
    assert one["status"] == "accepted"
    assert sorted([one["measured_gain_pct"], two["measured_gain_pct"]]) == [3.0, 8.4]
    assert two["status"] == "accepted"
    # The Integrator reports the combined state once for the whole batch.
    assert roadmap["current_best"]["value"] in (108.4, 111.6)
    assert roadmap["current_best"]["source"] == "rounds/round_1/integration/integration.md"
    # Per-item artifact dirs and one accept commit per item, in order.
    round_dir = ws / "rounds" / "round_1"
    assert (round_dir / "item_1_opt-001" / "attempt_1" / "evaluation.md").is_file()
    assert (round_dir / "item_2_opt-002" / "attempt_1" / "evaluation.md").is_file()
    events = [
        entry.get("event")
        for entry in progress_module.read_progress(workflow.progress_path)["optimization"]
        if entry.get("agent") == "optimizer_evaluator"
    ]
    assert events == ["batch_started", "batch_completed"]
    for item_dir in (round_dir / "item_1_opt-001", round_dir / "item_2_opt-002"):
        local_agents = [
            entry["agent"]
            for entry in progress_module.read_progress(item_dir / "progress.yaml")["optimization"]
        ]
        assert local_agents == ["optimizer", "evaluator"]
    messages = [c[1] for c in fake_git.calls if c[0] == "commit_all"]
    assert len(messages) == 2
    assert any("opt-001" in message for message in messages)
    assert any("opt-002" in message for message in messages)


def test_serial_items_reuse_worker_and_accept_directly(tmp_path, fake_git):
    """Serial mode keeps v3 item artifacts but bypasses batch integration."""
    task = _write_task(
        tmp_path,
        {"optimize": {"item_execution": "serial", "max_rounds": 1}},
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("APPROVE", "none", 8.4, 108.4),
            ("APPROVE", "none", 3.0, 111.6),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "optimizer",
        "evaluator",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.item_execution == "serial"
    assert state.item_batch == []

    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"
    assert roadmap["current_best"]["value"] == pytest.approx(111.6)
    assert roadmap["current_best"]["source"] == (
        "rounds/round_1/item_2_opt-002/attempt_1/evaluation.md"
    )

    global_entries = progress_module.read_progress(workflow.progress_path)["optimization"]
    assert [
        entry["agent"] for entry in global_entries if entry["agent"] in ("optimizer", "evaluator")
    ] == ["optimizer", "evaluator", "optimizer", "evaluator"]
    assert not any(
        entry["agent"] in ("optimizer_evaluator", "integrator") for entry in global_entries
    )
    round_dir = ws / "rounds" / "round_1"
    for item_dir in (round_dir / "item_1_opt-001", round_dir / "item_2_opt-002"):
        local_entries = progress_module.read_progress(item_dir / "progress.yaml")["optimization"]
        assert [entry["agent"] for entry in local_entries] == ["optimizer", "evaluator"]
    assert not (round_dir / "integration").exists()

    calls = [call[0] for call in fake_git.calls]
    first_fast_forward = calls.index("fast_forward")
    second_worktree = [i for i, name in enumerate(calls) if name == "create_worktree"][1]
    assert first_fast_forward < second_worktree
    assert fake_git.count("fast_forward") == 2


def test_serial_target_stops_before_unstarted_batch_items(tmp_path, fake_git):
    task = _write_task(
        tmp_path,
        {
            "optimize": {
                "item_execution": "serial",
                "max_rounds": 1,
                "target_improvement_pct": 5.0,
            }
        },
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("APPROVE", "none", 8.4, 108.4)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace.count("optimizer") == 1
    assert trace.count("evaluator") == 1
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "pending"
    assert not (ws / "rounds/round_1/item_2_opt-002").exists()
    assert fake_git.count("create_worktree") == 1


def test_serial_resume_finalizes_candidate_without_rerunning_worker(tmp_path, fake_git):
    task = _write_task(
        tmp_path,
        {"optimize": {"item_execution": "serial", "max_rounds": 1}},
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    first_trace = _stub_agents(workflow)

    def crash_before_finalization(state, item_id, log):
        raise RuntimeError("simulated crash before serial finalization")

    workflow._finalize_serial_item = crash_before_finalization
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            workflow.run(str(task))
    finally:
        workflow.close()

    interrupted = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert interrupted.item_batch[0]["status"] == "candidate_ready"
    assert interrupted.item_batch[0]["finalized"] is False
    assert first_trace.count("optimizer") == 1
    assert first_trace.count("evaluator") == 1

    resumed = Workflow(workspace=ws)
    resume_trace = _stub_agents(resumed)
    try:
        resumed.run("ignored-on-resume")
    finally:
        resumed.close()

    assert "optimizer" not in resume_trace
    assert "evaluator" not in resume_trace
    assert resume_trace == ["qa", "reporter"]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"


def test_worktree_cleanup_retries_once(tmp_path, monkeypatch):
    from agent_flow.workflows.perf_optimize import gitops as gitops_module

    attempts: list[str] = []
    sleeps: list[int] = []

    def _remove(repo, path):
        attempts.append(str(path))
        if len(attempts) == 1:
            raise gitops_module.GitOpsError("transient NFS error")

    monkeypatch.setattr(workflow_module.gitops, "remove_worktree", _remove)
    monkeypatch.setattr(workflow_module.time, "sleep", sleeps.append)

    workflow = Workflow(workspace=tmp_path / "ws")
    worktree = tmp_path / "item-worktree"
    workflow._remove_worktree_best_effort("repo", str(worktree), None)

    assert attempts == [str(worktree), str(worktree)]
    assert sleeps == [1]


def test_worktree_cleanup_failure_is_logged_and_suppressed(tmp_path, monkeypatch):
    from agent_flow.workflows.perf_optimize import gitops as gitops_module

    attempts: list[str] = []
    messages: list[str] = []

    def _remove(repo, path):
        attempts.append(str(path))
        raise gitops_module.GitOpsError("directory not empty")

    monkeypatch.setattr(workflow_module.gitops, "remove_worktree", _remove)
    monkeypatch.setattr(workflow_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        workflow_module,
        "print_message",
        lambda text, log=None: messages.append(text),
    )

    workflow = Workflow(workspace=tmp_path / "ws")
    worktree = tmp_path / "item-worktree"
    workflow._remove_worktree_best_effort("repo", str(worktree), None)

    assert attempts == [str(worktree), str(worktree)]
    assert len(messages) == 1
    assert "leaving it in place and continuing" in messages[0]
    assert str(worktree) in messages[0]
    assert "directory not empty" in messages[0]


def test_item_and_round_budgets_cap_the_campaign(tmp_path, fake_git):
    """The budgets bound the loop even with actionable items remaining."""
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1, "max_rounds": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # One item in the single budgeted round, then the final verification —
    # opt-002 stays pending despite being actionable.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "qa",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "pending"
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.max_items_per_round == 1
    assert state.round_index == 1
    assert state.done is True


def test_rejected_item_advances_to_next_item_in_same_round(tmp_path, fake_git):
    """A terminally rejected item is reverted, then the round moves on."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("REJECT", "functionality", 0.0, 0.0),
            ("APPROVE", "none", 4.0, 104.0),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # opt-001's REJECT is terminal (no retries despite the attempt
    # budget); the round continued with opt-002 without a fresh analyzer
    # profile.
    assert trace[0:3] == ["benchmarker", "projector", "analyzer"]
    assert trace.count("analyzer") == 2
    assert trace.count("optimizer") == 2
    assert trace.count("evaluator") == 2
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    statuses = {
        roadmap_schema.find_item(roadmap, item_id)["status"] for item_id in ("opt-001", "opt-002")
    }
    assert statuses == {"failed", "accepted"}
    assert roadmap["current_best"]["value"] == pytest.approx(104.0)
    assert fake_git.count("reset_to") >= 2
    assert fake_git.count("commit_all") == 1
    round_dir = ws / "rounds" / "round_1"
    assert (round_dir / "item_1_opt-001" / "attempt_1" / "evaluation.md").is_file()
    assert (round_dir / "item_2_opt-002" / "attempt_1" / "evaluation.md").is_file()


def test_each_parallel_item_uses_its_own_optimizer_session(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("PUSH_BACK", "perf_shortfall", 0.2, 100.2),  # opt-001 attempt 1
            ("APPROVE", "none", 8.4, 108.4),  # opt-001 attempt 2
            ("APPROVE", "none", 3.0, 111.6),  # opt-002 attempt 1
        ],
    )
    resets: list[int] = []
    workflow.optimizer.reset_session = lambda: resets.append(len(trace))
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[0:3] == ["benchmarker", "projector", "analyzer"]
    assert trace.count("optimizer") == 3
    assert trace.count("evaluator") == 3
    # The legacy campaign-wide optimizer is not used/reset; each worker owns
    # and closes its own persistent optimizer layer.
    assert resets == []


def test_item_dir_sanitizes_analyzer_authored_ids(tmp_path):
    """Path separators in analyzer-authored ids must not escape the round dir."""
    workflow = Workflow(workspace=tmp_path / "ws")
    try:
        state = state_module.WorkflowState(
            task_path="t",
            current_item_id="opt 001/../weird",
            item_index=1,
            stage=state_module.STAGE_OPTIMIZER,
        )
        item_dir = workflow._item_dir(state)
        assert item_dir.name == "item_2_opt-001-..-weird"
        assert item_dir.parent.name == "round_1"
    finally:
        workflow.close()


def test_no_actionable_items_goes_straight_to_reporter(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    # The only item promises less than the 1.0 default noise floor.
    trace = _stub_agents(workflow, analyzer_items=[[_item(gain=0.5)]])
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Nothing actionable, nothing accepted: no optimizer, no final
    # verification — straight to the reporter.
    assert trace == ["benchmarker", "projector", "analyzer", "reporter"]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "pending"
    assert state_module.load_state(ws / state_module.STATE_FILENAME).done is True


def test_exhausted_roadmap_after_accepts_still_runs_final_verification(tmp_path, fake_git):
    """A drained roadmap concludes the loop, but accepts still get verified.

    Round 2's analyzer plans nothing new and its only pending item is
    rejected — the loop ends mid-round, yet the final verification runs
    because round 1 accepted an item.
    """
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    # Round 1 plans+accepts the only item; round 2's analyzer adds none.
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)], []],
        evaluator_verdicts=[
            ("APPROVE", "none", 8.4, 108.4),
            ("REJECT", "perf_shortfall", 0.1, 108.5),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Round 2's reject drains the roadmap mid-round → round 3 gets the
    # free replan turn (round 2 accepted nothing, so it costs no GPU) →
    # that turn plans nothing, which is what concludes the loop → qa runs
    # (an item was accepted in round 1).
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    assert (ws / "final_verification" / "verification_report.md").is_file()


# ------------------------------------------------------- approach restriction


def test_restricted_run_never_dispatches_disallowed_items(tmp_path, fake_git):
    """approaches: [code] — a higher-gain config item is skipped, not applied."""
    task = _write_task(tmp_path, {"optimize": {"approaches": ["code"]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[
            [
                _item("opt-001", gain=15.0),  # config (default) — disallowed
                _item("opt-002", gain=8.0, approach="code"),
            ]
        ],
        evaluator_verdicts=[("APPROVE", "none", 7.0, 107.0)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Exactly one item ran: opt-002. The config item leads the roadmap
    # but never reaches the optimizer, this round or after the accept.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "pending"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"
    assert roadmap["current_best"]["value"] == pytest.approx(107.0)


def test_only_disallowed_pending_items_goes_straight_to_reporter(tmp_path, fake_git):
    task = _write_task(tmp_path, {"optimize": {"approaches": ["code"]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    # Well above the noise floor, but config-only in a code-only run.
    trace = _stub_agents(workflow, analyzer_items=[[_item(gain=15.0)]])
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector", "analyzer", "reporter"]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "pending"


def test_tuning_edit_in_code_only_run_is_auto_rejected(tmp_path, fake_git):
    """Every attempt edits the tuning YAML → fails without ever evaluating."""
    task = _write_task(tmp_path, {"optimize": {"approaches": ["code"], "max_attempts_per_item": 2}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow, analyzer_items=[[_item(approach="code")]])

    original_optimizer = workflow._run_optimizer

    def optimizer_editing_tuning(state, **kwargs):
        original_optimizer(state, **kwargs)
        workflow._state_tuning_paths(state)[0].write_text(
            "cuda_graph_config: {}\n", encoding="utf-8"
        )

    workflow._run_optimizer = optimizer_editing_tuning
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Both attempts were auto-rejected before the evaluator; nothing was
    # accepted, so the campaign reports without a final verification.
    # The trailing analyzer profiles: although both attempts only violated
    # the tuning restriction in this stub, they were code-approach turns,
    # so the orchestrator conservatively assumes ignored build output may
    # have changed. There is no evaluator verdict for it to read.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "optimizer",
        "analyzer",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "failed"
    assert item["attempts"] == 2
    assert item["measured_gain_pct"] is None
    # Each auto-reject reverted everything: the live tuning config is
    # back to the accepted snapshot, and the checkout was cleaned.
    assert workflow.tuning_config_path.read_text(encoding="utf-8") == "{}\n"
    assert fake_git.count("reset_to") >= 2
    assert state_module.load_state(ws / state_module.STATE_FILENAME).approach_violation == ""


def test_auto_reject_then_clean_retry_reaches_evaluator(tmp_path, fake_git):
    task = _write_task(tmp_path, {"optimize": {"approaches": ["code"]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item(approach="code")]],
        evaluator_verdicts=[("APPROVE", "none", 8.0, 108.0)],
    )

    original_optimizer = workflow._run_optimizer
    seen_violations: list[str] = []

    def optimizer_violating_once(state, **kwargs):
        seen_violations.append(state.approach_violation)
        original_optimizer(state, **kwargs)
        if len(seen_violations) == 1:
            workflow._state_tuning_paths(state)[0].write_text("kv: 0.9\n", encoding="utf-8")

    workflow._run_optimizer = optimizer_violating_once
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    # The retry attempt saw the violation as feedback; attempt 1 did not.
    assert seen_violations[0] == ""
    assert "tuning/extra_llm_api_options.yaml" in seen_violations[1]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-001")
    assert item["status"] == "accepted"
    assert item["attempts"] == 2
    assert roadmap["current_best"]["value"] == pytest.approx(108.0)


def test_code_edit_in_config_only_run_is_auto_rejected(tmp_path, fake_git):
    """approaches: [config] + dirty worktree after the attempt → auto-reject."""
    task = _write_task(
        tmp_path, {"optimize": {"approaches": ["config"], "max_attempts_per_item": 1}}
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    # FakeGitOps.worktree_clean always reports dirty — exactly the violation.
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The trailing analyzer profiles: the disallowed checkout edit may
    # have rebuilt a gitignored runtime artifact before it was reverted.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "analyzer",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "failed"
    assert fake_git.count("reset_to") >= 2


def test_clean_config_only_run_reaches_evaluator(tmp_path, fake_git):
    fake_git.worktree_clean = lambda repo: True
    task = _write_task(tmp_path, {"optimize": {"approaches": ["config"]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    # A clean worktree means a config-only accept: nothing to commit.
    assert fake_git.count("commit_all") == 0


def test_invalid_roadmap_blocks_advance(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def analyzer_with_bad_roadmap(state):
        trace.append("analyzer")
        findings = workflow._analysis_dir(state) / "profile_findings.md"
        findings.write_text("# findings\n", encoding="utf-8")
        workflow.roadmap_path.write_text("- not\n- a\n- roadmap\n", encoding="utf-8")

    workflow._run_analyzer = analyzer_with_bad_roadmap
    try:
        with pytest.raises(RuntimeError, match="roadmap.yaml failed validation"):
            workflow.run(str(task))
    finally:
        workflow.close()

    # The optimizer never ran, and the checkpoint stays parked at the
    # analyzer so a re-run retries it rather than skipping ahead.
    assert trace == ["benchmarker", "projector", "analyzer"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_ANALYZER
    assert state.done is False


# ------------------------------------------------------------------ curve mode

_WF_CURVE = [
    {"concurrency": 8, "value": 90.0, "tok_s_user": 20.0, "tok_s_gpu": 90.0},
    {"concurrency": 32, "value": 110.0, "tok_s_user": 12.0, "tok_s_gpu": 110.0},
]


def test_curve_mode_accept_records_current_best_curve(tmp_path, fake_git):
    task = _write_task(tmp_path, {"benchmark": {"concurrency": [32, 8]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    measured = [
        {"concurrency": 8, "value": 94.1, "tok_s_user": 21.0, "tok_s_gpu": 94.1},
        {"concurrency": 32, "value": 116.2, "tok_s_user": 12.6, "tok_s_gpu": 116.2},
    ]
    _stub_agents(
        workflow,
        evaluator_verdicts=[("APPROVE", "none", 5.5, 105.15)],
        baseline_curve=_WF_CURVE,
        evaluator_curve=measured,
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap["baseline"]["curve"] == _WF_CURVE
    assert roadmap["current_best"]["value"] == pytest.approx(105.15)
    assert roadmap["current_best"]["curve"] == measured


def test_curve_mode_accept_without_curve_degrades_to_scalar(tmp_path, fake_git):
    task = _write_task(tmp_path, {"benchmark": {"concurrency": [8, 32]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(
        workflow,
        evaluator_verdicts=[("APPROVE", "none", 5.5, 105.15)],
        baseline_curve=_WF_CURVE,
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    # No curve in the evaluator entry: the watermark degrades to scalar
    # rather than keeping a stale curve.
    assert roadmap["current_best"]["value"] == pytest.approx(105.15)
    assert "curve" not in roadmap["current_best"]


# Per-point measurements shared by the focus-scored target tests:
# +40% at c=8, +2% at c=32 vs ``_WF_CURVE`` — the unfiltered mean (+21%)
# clears a 5% target that the c=32 focus mean (+2%) does not.
_WF_FOCUS_MEASURED = [
    {"concurrency": 8, "value": 126.0, "tok_s_user": 28.0, "tok_s_gpu": 126.0},
    {"concurrency": 32, "value": 112.2, "tok_s_user": 12.2, "tok_s_gpu": 112.2},
]


def test_target_improvement_scores_focus_subset_only(tmp_path, fake_git):
    """The early stop honors ``optimize.focus_concurrencies``.

    Like every other curve→scalar derivation, ``_target_met`` averages
    only the focus points — a big swing at a non-focus point must not
    conclude the loop while the declared focus point sits below target.
    """
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 32]},
            "optimize": {"target_improvement_pct": 5.0, "focus_concurrencies": [32]},
        },
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("APPROVE", "none", 2.0, 112.2),
            ("APPROVE", "none", 2.0, 112.2),
        ],
        baseline_curve=_WF_CURVE,
        evaluator_curve=_WF_FOCUS_MEASURED,
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The focus mean (+2% at c=32) stays below the 5% target, so opt-002
    # is still applied and the campaign ends on roadmap exhaustion — not
    # on the unfiltered +21% mean after opt-001.
    assert trace.count("optimizer") == 2
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"


def test_target_improvement_met_on_focus_subset_concludes(tmp_path, fake_git):
    """The focus-scored mean crossing the target still stops the loop."""
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 32]},
            "optimize": {"target_improvement_pct": 5.0, "focus_concurrencies": [8]},
        },
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("APPROVE", "none", 40.0, 126.0)],
        baseline_curve=_WF_CURVE,
        evaluator_curve=_WF_FOCUS_MEASURED,
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The target is checked only after the selected parallel batch completes.
    assert trace.count("optimizer") == 2
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-002")["status"] == "accepted"
    assert state_module.load_state(ws / state_module.STATE_FILENAME).done is True


def test_curve_mode_validate_roadmap_rejects_point_mismatch(tmp_path, fake_git):
    task = _write_task(tmp_path, {"benchmark": {"concurrency": [8, 32]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    wrong_points = [
        {"concurrency": 8, "value": 90.0, "tok_s_user": 20.0, "tok_s_gpu": 90.0},
        {"concurrency": 64, "value": 120.0, "tok_s_user": 10.0, "tok_s_gpu": 120.0},
    ]
    trace = _stub_agents(workflow, baseline_curve=wrong_points)
    try:
        with pytest.raises(RuntimeError, match="does not cover the task's concurrency points"):
            workflow.run(str(task))
    finally:
        workflow.close()
    # Parked at the analyzer for a retry, like any roadmap-validation failure.
    assert trace == ["benchmarker", "projector", "analyzer"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_ANALYZER


def test_curve_mode_validate_roadmap_rejects_missing_baseline_curve(tmp_path, fake_git):
    task = _write_task(tmp_path, {"benchmark": {"concurrency": [8, 32]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)  # analyzer writes a scalar-only baseline
    try:
        with pytest.raises(RuntimeError, match="does not cover the task's concurrency points"):
            workflow.run(str(task))
    finally:
        workflow.close()


def test_latest_evaluator_curve_returns_none_on_malformed_entries(tmp_path, fake_git):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    try:
        good = [
            {"concurrency": 8, "value": 94.1, "tok_s_user": 21.0, "tok_s_gpu": 94.1},
            {"concurrency": 32, "value": 116.2, "tok_s_user": 12.6, "tok_s_gpu": 116.2},
        ]
        for curve, expected in [
            (good, good),
            (None, None),  # no curve field at all
            ([], None),  # empty
            ("oops", None),  # not a list
            ([{"concurrency": 8, "value": 1.0}], None),  # missing fields
            (
                [{"concurrency": "8", "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0}],
                None,
            ),  # non-int concurrency
            (
                [
                    {"concurrency": 32, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
                    {"concurrency": 8, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
                ],
                None,
            ),  # not ascending
        ]:
            entry = {"step": 1, "agent": "evaluator", "summary": "e"}
            if curve is not None:
                entry["curve"] = curve
            progress_module.write_progress(workflow.progress_path, {"optimization": [entry]})
            assert workflow._latest_evaluator_curve() == expected, curve
    finally:
        workflow.close()


def test_resume_mid_round_starts_at_evaluator(tmp_path, monkeypatch):
    fake = FakeGitOps()  # accept path commits
    monkeypatch.setattr(workflow_module, "gitops", fake)

    ws = tmp_path / "ws"
    ws.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    # Seed the workspace as an interrupted round-2 attempt-2 evaluation.
    (ws / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(ckpt),
                "trtllm_repo_path": str(repo),
                "optimize": dict(task_schema.OPTIMIZE_DEFAULTS),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    tuning = ws / "tuning"
    tuning.mkdir()
    (tuning / "extra_llm_api_options.yaml").write_text("{}\n", encoding="utf-8")
    (tuning / "extra_llm_api_options.accepted.yaml").write_text("{}\n", encoding="utf-8")
    roadmap_schema.save_roadmap(
        ws / "roadmap.yaml",
        {
            "version": 1,
            "target_metric": "output_throughput",
            "baseline": {"value": 100.0, "source": "baseline/benchmark_results.md"},
            "current_best": {"value": 108.4, "source": "rounds/round_1/attempt_1/evaluation.md"},
            "items": [
                _item("opt-001", status="accepted", attempts=1, measured_gain_pct=8.4),
                _item("opt-002", gain=5.0, status="in_progress", attempts=1),
            ],
        },
    )
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            max_rounds=3,
            max_attempts_per_item=3,
            round_index=1,
            campaign_git_branch="perf-optimize/seeded",
            campaign_git_base_commit="a" * 40,
            benchmarker_done=True,
            item_batch=[
                {
                    "current_item_id": "opt-002",
                    "item_index": 0,
                    "attempt_index": 1,
                    "approach_violation": "",
                    "item_worktree_path": str(ws / "worktrees/round_2/item_1_opt-002"),
                    "item_branch": "perf-optimize/seeded/round-2/item-opt-002",
                    "item_base_commit": "b" * 40,
                    "phase": state_module.STAGE_EVALUATOR,
                    "status": "running",
                }
            ],
            batch_started=True,
            stage=state_module.STAGE_OPTIMIZER_EVALUATOR,
        ),
    )
    (ws / "worktrees/round_2/item_1_opt-002").mkdir(parents=True)
    item_dir = ws / "rounds/round_2/item_1_opt-002"
    (item_dir / "tuning").mkdir(parents=True)
    for name in ("extra_llm_api_options.yaml", "extra_llm_api_options.accepted.yaml"):
        (item_dir / "tuning" / name).write_text("{}\n", encoding="utf-8")
    progress_module.init_progress_file(item_dir / "progress.yaml")

    workflow = Workflow(workspace=ws)
    assert workflow.resume is True
    trace = _stub_agents(
        workflow,
        # The resumed roadmap already carries opt-001, and the closing
        # re-profile of the build the accept changed plans nothing new.
        analyzer_items=[[]],
        evaluator_verdicts=[("APPROVE", "none", 4.0, 112.7)],
    )
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["evaluator", "analyzer", "qa", "reporter"]
    # The seeded branch was checked out, never re-created.
    assert fake.count("checkout") == 1
    assert ("checkout", "perf-optimize/seeded") in fake.calls
    assert fake.count("create_branch") == 0
    assert fake.count("commit_all") == 1
    # The evaluator ran against the seeded round/item/attempt position.
    assert (ws / "rounds" / "round_2" / "item_1_opt-002" / "attempt_2" / "evaluation.md").is_file()
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    item = roadmap_schema.find_item(roadmap, "opt-002")
    assert item["status"] == "accepted"
    assert item["attempts"] == 2
    assert roadmap["current_best"]["value"] == pytest.approx(112.7)


def test_resume_redispatch_purges_stale_attempt_benchmark_results(tmp_path, monkeypatch):
    """A redispatched evaluator starts from a purged attempt dir.

    An evaluator killed mid-sweep leaves partial per-point results in its
    attempt dir; the purge keeps stale result JSONs from ever being read
    as fresh measurements.
    """
    fake = FakeGitOps()
    monkeypatch.setattr(workflow_module, "gitops", fake)

    ws = tmp_path / "ws"
    ws.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ws / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(ckpt),
                "trtllm_repo_path": str(repo),
                "optimize": dict(task_schema.OPTIMIZE_DEFAULTS),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    tuning = ws / "tuning"
    tuning.mkdir()
    (tuning / "extra_llm_api_options.yaml").write_text("{}\n", encoding="utf-8")
    (tuning / "extra_llm_api_options.accepted.yaml").write_text("{}\n", encoding="utf-8")
    roadmap_schema.save_roadmap(
        ws / "roadmap.yaml",
        {
            "version": 1,
            "target_metric": "output_throughput",
            "baseline": {"value": 100.0, "source": "baseline/benchmark_results.md"},
            "current_best": {"value": 108.4, "source": "rounds/round_1/attempt_1/evaluation.md"},
            "items": [
                _item("opt-001", status="accepted", attempts=1, measured_gain_pct=8.4),
                _item("opt-002", gain=5.0, status="in_progress", attempts=1),
            ],
        },
    )
    state_module.save_state(
        ws / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            task_path=str(ws / "task.yaml"),
            max_rounds=3,
            max_attempts_per_item=3,
            round_index=1,
            campaign_git_branch="perf-optimize/seeded",
            campaign_git_base_commit="a" * 40,
            benchmarker_done=True,
            item_batch=[
                {
                    "current_item_id": "opt-002",
                    "item_index": 0,
                    "attempt_index": 1,
                    "approach_violation": "",
                    "item_worktree_path": str(ws / "worktrees/round_2/item_1_opt-002"),
                    "item_branch": "perf-optimize/seeded/round-2/item-opt-002",
                    "item_base_commit": "b" * 40,
                    "phase": state_module.STAGE_EVALUATOR,
                    "status": "running",
                }
            ],
            batch_started=True,
            stage=state_module.STAGE_OPTIMIZER_EVALUATOR,
        ),
    )
    (ws / "worktrees/round_2/item_1_opt-002").mkdir(parents=True)
    item_dir = ws / "rounds/round_2/item_1_opt-002"
    (item_dir / "tuning").mkdir(parents=True)
    for name in ("extra_llm_api_options.yaml", "extra_llm_api_options.accepted.yaml"):
        (item_dir / "tuning" / name).write_text("{}\n", encoding="utf-8")
    progress_module.init_progress_file(item_dir / "progress.yaml")

    # The killed sweep's leftovers, next to an artifact that must survive.
    attempt_dir = ws / "rounds" / "round_2" / "item_1_opt-002" / "attempt_2"
    stale_point_dir = attempt_dir / "concurrency_8"
    stale_point_dir.mkdir(parents=True)
    (stale_point_dir / "openai-infqps-concurrency8-m-1.json").write_text("{}", encoding="utf-8")
    stale_json = attempt_dir / "openai-infqps-concurrency64-m-1.json"
    stale_json.write_text("{}", encoding="utf-8")
    summary = attempt_dir / "optimization_summary.md"
    summary.write_text("# summary\n", encoding="utf-8")

    workflow = Workflow(workspace=ws)
    assert workflow.resume is True
    trace = _stub_agents(
        workflow,
        # The resumed roadmap already carries opt-001, and the closing
        # re-profile of the build the accept changed plans nothing new.
        analyzer_items=[[]],
        evaluator_verdicts=[("APPROVE", "none", 4.0, 112.7)],
    )
    try:
        workflow.run("ignored-on-resume")
    finally:
        workflow.close()

    assert trace == ["evaluator", "analyzer", "qa", "reporter"]
    assert not stale_point_dir.exists()
    assert not stale_json.exists()
    assert summary.is_file()


# ------------------------------------------- evaluator accept-evidence duty


class _RecordingAgent:
    """Stands in for an agent attribute; records the messages it is sent."""

    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)

    def __exit__(self, *exc) -> None:
        return None


def _evaluator_state(ws) -> state_module.WorkflowState:
    return state_module.WorkflowState(
        task_path=str(ws / "task.yaml"),
        current_item_id="opt-001",
        campaign_git_branch="perf-optimize/test-branch",
        stage=state_module.STAGE_EVALUATOR,
    )


def test_run_evaluator_includes_accept_evidence_duty_when_nsys_configured(tmp_path, fake_git):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.evaluator = recorder
    try:
        (ws / "task.yaml").write_text(
            yaml.safe_dump({"profile": {"methods": ["nsys", "torch"]}}),
            encoding="utf-8",
        )
        state = _evaluator_state(ws)
        # Before any capture exists, the duty says so instead of naming a
        # comparison directory.
        workflow._run_evaluator(state)
        first = recorder.messages[0]
        assert "Accept-evidence duty" in first
        assert "only if your verdict is APPROVE" in first
        assert str(workflow._attempt_dir(state) / "profile") in first
        assert "no previous capture" in first
        # Once the orchestrator has recorded a capture, the duty names it
        # as the comparison reference.
        state.last_nsys_dir = str(ws / "rounds" / "round_1" / "analysis")
        workflow._run_evaluator(state)
        second = recorder.messages[1]
        assert state.last_nsys_dir in second
        assert "no previous capture" not in second
    finally:
        workflow.close()


def test_run_evaluator_has_no_duty_without_nsys(tmp_path, fake_git):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.evaluator = recorder
    try:
        (ws / "task.yaml").write_text(
            yaml.safe_dump({"profile": {"methods": ["torch"]}}), encoding="utf-8"
        )
        workflow._run_evaluator(_evaluator_state(ws))
        assert "Accept-evidence duty" not in recorder.messages[0]
    finally:
        workflow.close()


def test_run_evaluator_marks_the_final_attempt(tmp_path, fake_git):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.evaluator = recorder
    try:
        (ws / "task.yaml").write_text(
            yaml.safe_dump({"profile": {"methods": ["torch"]}}), encoding="utf-8"
        )
        state = _evaluator_state(ws)
        workflow._run_evaluator(state)
        assert "final attempt" not in recorder.messages[0]
        state.attempt_index = 2  # attempt 3 of the default 3
        workflow._run_evaluator(state)
        assert "final attempt" in recorder.messages[1]
        assert "PUSH_BACK is not available" in recorder.messages[1]
    finally:
        workflow.close()


def _analyzer_state(ws, **overrides) -> state_module.WorkflowState:
    """A round-2 analyzer state; overrides pick the round's opening mode."""
    data = {
        "task_path": str(ws / "task.yaml"),
        "round_index": 1,
        "profile_required": False,
        "last_profiled_analysis_dir": str(ws / "rounds" / "round_1" / "analysis"),
        "campaign_git_branch": "perf-optimize/test-branch",
        "stage": state_module.STAGE_ANALYZER,
    }
    data.update(overrides)
    return state_module.WorkflowState(**data)


def test_replan_only_round_forbids_profiling_and_briefs_the_verdicts(tmp_path, fake_git):
    """The prompt states the unchanged build as fact, not as a choice to weigh."""
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.analyzer = recorder
    try:
        (ws / "task.yaml").write_text(yaml.safe_dump({"sol": {"enabled": False}}), encoding="utf-8")
        # Two items were attempted last round and all of them reverted.
        for item in ("item_1_opt-001", "item_2_opt-002"):
            (ws / "rounds" / "round_1" / item).mkdir(parents=True)
        workflow._run_analyzer(_analyzer_state(ws))
        message = recorder.messages[0]

        assert "**replan only**" in message
        assert "accepted **nothing**" in message
        assert "Its 2 attempted items are" in message
        # The stronger premise behind the mode: no code attempt could
        # have left a rebuilt ignored artifact behind, so the standing
        # analysis still describes the runtime.
        assert "None was a code attempt" in message
        assert "runtime remains the state" in message
        assert "byte-identical" not in message
        # The three spends the round exists to avoid.
        assert "Do **not** launch `trtllm-serve`" in message
        assert "do **not** run nsys / ncu / the torch profiler" in message
        assert "do **not** run the benchmark" in message
        # …and the evidence it plans from instead.
        assert str(ws / "rounds" / "round_1" / "analysis") in message
        assert 'read_latest_progress` with `agent: "evaluator"' in message
        # A plateau is a legitimate outcome; padding the roadmap is not.
        assert "leave the roadmap with no actionable pending item" in message
        assert "Do not invent items to keep the loop alive" in message
    finally:
        workflow.close()


def test_profiling_round_names_what_moved_the_build_since_the_last_analysis(tmp_path, fake_git):
    """A round only re-profiles because something was accepted — say how much."""
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.analyzer = recorder
    try:
        (ws / "task.yaml").write_text(yaml.safe_dump({"sol": {"enabled": False}}), encoding="utf-8")
        workflow._run_analyzer(_analyzer_state(ws, profile_required=True))
        message = recorder.messages[0]

        assert "**replan only**" not in message
        assert "Re-profile the **current** build" in message
        # Failed items are measurements too — the round should not re-propose
        # what the evaluator already disproved.
        assert 'read_latest_progress` with `agent: "evaluator"' in message
    finally:
        workflow.close()


def test_profiling_round_does_not_invent_an_accept_for_unknown_runtime(tmp_path, fake_git):
    """The conservative profile gate is distinct from the measured accept count."""
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.analyzer = recorder
    try:
        (ws / "task.yaml").write_text(yaml.safe_dump({"sol": {"enabled": False}}), encoding="utf-8")
        workflow._run_analyzer(_analyzer_state(ws, profile_required=True))
        message = recorder.messages[0]

        assert "**replan only**" not in message
        assert "**0 item(s) have been accepted" not in message
        assert "standing profile is stale or unproven" in message
        assert "checkpoint has not established a current local profile" in message
        assert "Re-profile the **current** build" in message
    finally:
        workflow.close()


def test_accept_records_last_nsys_dir_from_captures(tmp_path, fake_git):
    """The analyzer profile, then each accept's capture, advance the pointer."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    original_analyzer = workflow._run_analyzer
    original_evaluator = workflow._run_evaluator
    seen_at_evaluation_time: list[str] = []
    seen_at_analysis_time: list[str] = []

    def analyzer_with_capture(state):
        seen_at_analysis_time.append(state.last_nsys_dir)
        original_analyzer(state)
        (workflow._analysis_dir(state) / "nsys_stats.txt").write_text("k\n", encoding="utf-8")

    def evaluator_with_capture(state, **kwargs):
        seen_at_evaluation_time.append(state.last_nsys_dir)
        original_evaluator(state, **kwargs)
        profile_dir = workflow._attempt_dir(state) / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "nsys_stats.txt").write_text("k\n", encoding="utf-8")

    workflow._run_analyzer = analyzer_with_capture
    workflow._run_evaluator = evaluator_with_capture
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[-1] == "reporter"
    # The evaluator judged with the analyzer's round profile as reference…
    assert seen_at_evaluation_time == [str(ws / "rounds" / "round_1" / "analysis")]
    # Item-local captures are candidates only. The closing analyzer sees the
    # last accepted campaign capture, then records round 2 as the freshest.
    round_1_analysis = ws / "rounds" / "round_1" / "analysis"
    assert seen_at_analysis_time == ["", str(round_1_analysis)]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    expected = ws / "rounds" / "round_2" / "analysis"
    assert state.last_nsys_dir == str(expected)


def test_crash_mid_accept_is_logged_and_leaves_resumable_checkpoint(tmp_path, monkeypatch):
    """An orchestrator crash is recorded in the session log, not just stderr.

    Regression: a `GitOpsError` from the accept commit killed the run with
    the session log simply ending after the evaluator's turn, which read
    as a deliberate exit. The abort must be logged and the checkpoint left
    at the evaluator stage so a plain re-run retries the accept.
    """
    from agent_flow.workflows.perf_optimize import gitops as gitops_module

    fake = FakeGitOps()

    def _boom(repo, message):
        raise gitops_module.GitOpsError("`git commit` failed: pre-commit hook")

    fake.commit_all = _boom
    monkeypatch.setattr(workflow_module, "gitops", fake)

    messages: list[str] = []
    real_print_message = workflow_module.print_message

    def _capture(text, log=None):
        messages.append(text)
        real_print_message(text, log)

    monkeypatch.setattr(workflow_module, "print_message", _capture)

    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        with pytest.raises(RuntimeError, match="parallel optimization item"):
            workflow.run(str(task))
    finally:
        workflow.close()

    assert any("workflow aborted" in m and "pre-commit hook" in m for m in messages)
    # The batch checkpoint retains the failed worker for a targeted resume.
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_OPTIMIZER_EVALUATOR
    assert state.item_batch[0]["current_item_id"] == "opt-001"
    assert state.item_batch[0]["status"] == "error"
    assert state.done is False


def test_already_done_checkpoint_short_circuits(tmp_path, fake_git):
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


# ----------------------------------------------------- stage-output gates


def test_optimizer_without_summary_blocks_advance(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def optimizer_no_output(state, **kwargs):
        trace.append("optimizer")
        # Writes nothing — mirrors an agent that yielded before its deliverable.

    workflow._run_optimizer = optimizer_no_output
    try:
        with pytest.raises(RuntimeError, match="optimization_summary.md"):
            workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["benchmarker", "projector", "analyzer", "optimizer"]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_OPTIMIZER_EVALUATOR
    assert state.item_batch[0]["phase"] == state_module.STAGE_OPTIMIZER
    assert state.done is False


def test_reporter_requires_both_md_and_html(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    def reporter_md_only(state):
        trace.append("reporter")
        workflow.report_path.write_text("# report\n", encoding="utf-8")

    workflow._run_reporter = reporter_md_only
    try:
        with pytest.raises(RuntimeError, match="optimization_report.html"):
            workflow.run(str(task))
    finally:
        workflow.close()

    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_REPORTER
    assert state.done is False


# ---------------------------------------------------------------- git setup


def test_non_git_checkout_aborts_fresh_run(tmp_path, monkeypatch):
    fake = FakeGitOps(is_repo=False)
    monkeypatch.setattr(workflow_module, "gitops", fake)
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        with pytest.raises(RuntimeError, match="not a git repository"):
            workflow.run(str(task))
    finally:
        workflow.close()


def test_branch_is_checkpointed_before_creation(tmp_path, monkeypatch):
    """A crash between checkpoint and checkout -b must resume cleanly."""
    ws = tmp_path / "ws"

    class AssertingGit(FakeGitOps):
        def create_branch(self, repo, name):
            state = state_module.load_state(ws / state_module.STATE_FILENAME)
            assert state.campaign_git_branch == name
            assert state.campaign_git_base_commit == "b" * 40
            super().create_branch(repo, name)

    fake = AssertingGit()
    monkeypatch.setattr(workflow_module, "gitops", fake)
    task = _write_task(tmp_path)
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()
    assert fake.count("create_branch") == 1


# ---------------------------------------------------------- fresh-start guard


@pytest.mark.parametrize(
    "filename",
    ["sol_projection.md", "roadmap.yaml", "optimization_report.md", "optimization_report.html"],
)
def test_fresh_start_raises_when_managed_file_non_empty(tmp_path, filename):
    (tmp_path / filename).write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match=re.escape(filename)):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_baseline_non_empty(tmp_path):
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "benchmark_results.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="benchmark_results.md"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_rounds_dir_non_empty(tmp_path):
    (tmp_path / "rounds" / "round_1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="rounds"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_final_verification_non_empty(tmp_path):
    (tmp_path / "final_verification").mkdir()
    (tmp_path / "final_verification" / "verification_report.md").write_text(
        "# stale\n", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="final_verification"):
        Workflow(workspace=tmp_path)


def test_fresh_start_raises_when_progress_has_real_entries(tmp_path):
    progress_module.write_progress(
        tmp_path / "progress.yaml",
        {"optimization": [{"step": 1, "agent": "benchmarker", "summary": "prior run"}]},
    )
    with pytest.raises(FileExistsError, match="progress.yaml"):
        Workflow(workspace=tmp_path)


def test_fresh_start_allows_progress_shell_and_empty_files(tmp_path):
    progress_module.init_progress_file(tmp_path / "progress.yaml")
    (tmp_path / "roadmap.yaml").write_text("\n  \n", encoding="utf-8")
    workflow = Workflow(workspace=tmp_path)
    try:
        assert workflow.resume is False
    finally:
        workflow.close()


def test_clean_wipes_managed_files_and_dirs(tmp_path):
    (tmp_path / "sol_projection.md").write_text("# stale\n", encoding="utf-8")
    (tmp_path / "roadmap.yaml").write_text("# stale\n", encoding="utf-8")
    (tmp_path / "rounds" / "round_1").mkdir(parents=True)
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning" / "extra_llm_api_options.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "final_verification").mkdir()
    (tmp_path / "final_verification" / "verification_report.md").write_text(
        "# stale\n", encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        Workflow(workspace=tmp_path)

    workflow = Workflow(workspace=tmp_path, clean=True)
    try:
        assert workflow.resume is False
        assert (tmp_path / "sol_projection.md").read_text(encoding="utf-8") == ""
        assert (tmp_path / "roadmap.yaml").read_text(encoding="utf-8") == ""
        assert not (tmp_path / "rounds").exists()
        assert not (tmp_path / "tuning").exists()
        assert not (tmp_path / "final_verification").exists()
    finally:
        workflow.close()


# --------------------------------------------------------------- agent wiring


def test_all_agents_use_claude_code_backend_with_scoped_sessions(tmp_path):
    workflow = Workflow(workspace=tmp_path / "ws")
    try:
        for role in _AGENT_ROLES:
            layer = getattr(workflow, role)
            assert layer.config.backend.kind == "claude-code", role
            assert layer.config.backend.model == CLAUDE_CODE_DEFAULT_MODEL, role
            assert layer.config.backend.hooks is not None, role
            # The judges are stateless (fresh eyes per verdict); the
            # optimizer's persistent session is additionally reset per
            # item by the orchestrator (covered by
            # test_optimizer_session_resets_at_item_boundaries_not_retries).
            expected_mode = (
                "stateless" if role in ("qa", "evaluator", "integrator") else "persistent"
            )
            assert layer.config.session.mode == expected_mode, role
    finally:
        workflow.close()


def test_each_agent_has_its_progress_tools(tmp_path):
    workflow = Workflow(workspace=tmp_path / "ws")
    try:
        for role in _AGENT_ROLES:
            layer = getattr(workflow, role)
            tool_names = [t.name for t in layer.config.backend.tools]
            assert f"append_{role}_progress" in tool_names, role
            assert "read_latest_progress" in tool_names, role
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
        for role in _AGENT_ROLES:
            assert getattr(workflow, role).config.backend.extra_mcp_servers is None, role
    finally:
        workflow.close()


def test_user_tuning_config_is_copied_into_workspace(tmp_path, fake_git):
    extra = tmp_path / "extra.yaml"
    extra.write_text("cuda_graph_config: {}\n", encoding="utf-8")
    task = _write_task(tmp_path, {"extra_llm_api_options": str(extra)})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()
    live = (ws / "tuning" / "extra_llm_api_options.yaml").read_text(encoding="utf-8")
    accepted = (ws / "tuning" / "extra_llm_api_options.accepted.yaml").read_text(encoding="utf-8")
    assert live == "cuda_graph_config: {}\n"
    assert accepted == live


def test_max_rounds_override_applies_on_fresh_run(tmp_path, fake_git):
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, max_rounds_override=1)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()
    # Budget of 1 round: opt-002 is actionable but never gets a round 2.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.max_rounds == 1


# ---------------------------------------------------------- real-git end-to-end


def _init_real_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _real_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_accept_with_real_git_commits_the_code_change(tmp_path):
    """End-to-end against real git: accepted code edits become one commit."""
    repo = _init_real_repo(tmp_path)
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    original_optimizer = workflow._run_optimizer

    def optimizer_editing_repo(state, **kwargs):
        (Path(state.item_worktree_path) / "src.py").write_text(
            "x = 2  # optimized\n", encoding="utf-8"
        )
        original_optimizer(state, **kwargs)

    workflow._run_optimizer = optimizer_editing_repo
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[-1] == "reporter"
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert _real_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == state.campaign_git_branch
    assert "opt-001" in _real_git(repo, "log", "-1", "--pretty=%s")
    assert _real_git(repo, "status", "--porcelain") == ""
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 2  # optimized\n"


def test_accept_survives_installed_precommit_hook(tmp_path):
    """A failing pre-commit hook in the checkout must not abort the accept.

    Regression: TRT-LLM checkouts have `pre-commit install`ed hooks that
    reformat files and exit non-zero; the orchestrator's accept commit
    aborted mid-`_accept_attempt`, crashing the run right after the
    evaluator's APPROVE — before the roadmap/state advanced to QA.
    """
    repo = _init_real_repo(tmp_path)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho '# reformatted' >> src.py\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)

    original_optimizer = workflow._run_optimizer

    def optimizer_editing_repo(state, **kwargs):
        (Path(state.item_worktree_path) / "src.py").write_text(
            "x = 2  # optimized\n", encoding="utf-8"
        )
        original_optimizer(state, **kwargs)

    workflow._run_optimizer = optimizer_editing_repo
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # The run made it past the accept and all the way to the reporter.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    assert "opt-001" in _real_git(repo, "log", "-1", "--pretty=%s")
    assert _real_git(repo, "status", "--porcelain") == ""
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "accepted"


def test_reject_with_real_git_reverts_the_code_change(tmp_path):
    """End-to-end against real git: a terminally rejected item's edits are wiped."""
    repo = _init_real_repo(tmp_path)
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow, evaluator_verdicts=[("REJECT", "functionality", 0.0, 0.0)])

    original_optimizer = workflow._run_optimizer

    def optimizer_editing_repo(state, **kwargs):
        item_repo = Path(state.item_worktree_path)
        (item_repo / "src.py").write_text("x = 999  # broken\n", encoding="utf-8")
        (item_repo / "leftover.py").write_text("junk\n", encoding="utf-8")
        original_optimizer(state, **kwargs)

    workflow._run_optimizer = optimizer_editing_repo
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[-1] == "reporter"
    # The rejected attempt's edits are gone; no commit was made.
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (repo / "leftover.py").exists()
    assert _real_git(repo, "status", "--porcelain") == ""
    assert _real_git(repo, "log", "-1", "--pretty=%s") == "init"
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    assert roadmap_schema.find_item(roadmap, "opt-001")["status"] == "failed"


# ------------------------------------------------------------ driving prompts


def _capture_driving_prompts(
    tmp_path, extra: dict | None = None, sol_methodology: SolMethodology | None = None
) -> dict[str, str]:
    """Build a workflow, then capture each role's per-turn instruction."""
    task = _write_task(tmp_path, extra)
    ws = tmp_path / f"ws-{sol_methodology.name}" if sol_methodology else tmp_path / "ws"
    workflow = Workflow(workspace=ws, sol_methodology=sol_methodology)
    validated = task_schema.load_and_validate_task_yaml(str(task))
    workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")

    state = state_module.WorkflowState(
        task_path=str(workflow.task_path),
        round_index=0,
        attempt_index=0,
        current_item_id="opt-001",
        campaign_git_branch="perf-optimize/test-branch",
        campaign_git_base_commit="a" * 40,
        item_worktree_path=str(ws / "worktrees" / "round_1" / "item_1_opt-001"),
        item_branch="perf-optimize/test-branch-round-1-item-1-opt-001",
        stage=state_module.STAGE_OPTIMIZER,
    )

    captured: dict[str, str] = {}

    def _capture(role: str):
        def _cap(prompt: str) -> None:
            captured[role] = prompt

        return _cap

    originals = {role: getattr(workflow, role) for role in _ROLES}
    for role in _ROLES:
        setattr(workflow, role, _capture(role))
    try:
        for role in _ROLES:
            getattr(workflow, f"_run_{role}")(state)
    finally:
        for role, original in originals.items():
            setattr(workflow, role, original)
        workflow.close()
    return captured


def test_driving_prompts_avoid_removed_builtin_tools(tmp_path):
    captured = _capture_driving_prompts(tmp_path)
    for role, prompt in captured.items():
        for name in ("Grep", "Glob"):
            assert not re.search(rf"\b{name}\b", prompt), (role, name)


def test_driving_prompts_reinforce_casebook_for_serving_analysis_roles(tmp_path):
    captured = _capture_driving_prompts(tmp_path)
    for role in ("benchmarker", "analyzer", "optimizer"):
        assert "perf-optimization-casebook" in captured[role], role
        assert "`Skill` tool" in captured[role], role


def test_optimizer_prompt_names_item_branch_and_attempt_dir(tmp_path):
    captured = _capture_driving_prompts(tmp_path)
    prompt = captured["optimizer"]
    active_repo = str(tmp_path / "ws" / "worktrees" / "round_1" / "item_1_opt-001")
    tuning_dir = tmp_path / "ws" / "rounds" / "round_1" / "item_1_opt-001" / "tuning"
    active_tuning = str(tuning_dir / "extra_llm_api_options.yaml")
    accepted_tuning = str(tuning_dir / "extra_llm_api_options.accepted.yaml")
    assert "opt-001" in prompt
    assert "perf-optimize/test-branch" in prompt
    assert "attempt_1" in prompt
    assert "never commit" in prompt
    assert f"Active runtime checkout: `{active_repo}`" in prompt
    assert f'export PYTHONPATH="{active_repo}${{PYTHONPATH:+:$PYTHONPATH}}"' in prompt
    assert active_tuning in prompt
    assert accepted_tuning not in prompt


def test_evaluator_prompt_carries_the_gate_knobs(tmp_path):
    captured = _capture_driving_prompts(
        tmp_path, {"optimize": {"accept_fraction": 0.7, "noise_floor_pct": 2.0}}
    )
    prompt = captured["evaluator"]
    active_repo = str(tmp_path / "ws" / "worktrees" / "round_1" / "item_1_opt-001")
    tuning_dir = tmp_path / "ws" / "rounds" / "round_1" / "item_1_opt-001" / "tuning"
    active_tuning = str(tuning_dir / "extra_llm_api_options.yaml")
    accepted_tuning = str(tuning_dir / "extra_llm_api_options.accepted.yaml")
    assert "accept_fraction=0.7" in prompt
    assert "noise_floor_pct=2.0" in prompt
    assert "current_best" in prompt
    assert "opt-001" in prompt
    # The three-way decision vocabulary and the full-metric diff reference.
    assert "APPROVE|REJECT|PUSH_BACK" in prompt
    assert "full-metric diff" in prompt
    assert f"Active runtime checkout: `{active_repo}`" in prompt
    assert f'export PYTHONPATH="{active_repo}${{PYTHONPATH:+:$PYTHONPATH}}"' in prompt
    assert f"Active tuning config: `{active_tuning}`" in prompt
    assert f"Accepted tuning config snapshot: `{accepted_tuning}`" in prompt


def test_qa_prompt_conditions_accuracy_on_task_block(tmp_path):
    without = _capture_driving_prompts(tmp_path)["qa"]
    assert "**no** `accuracy` block" in without
    assert "not configured" in without

    with_accuracy = _capture_driving_prompts(
        tmp_path, {"accuracy": {"command": "trtllm-eval ...", "baseline_score": 0.6}}
    )["qa"]
    assert "**has** an `accuracy` block" in with_accuracy


_FOCUS_EXTRA = {
    "benchmark": {"concurrency": [8, 32, 128], "num_prompts": [32, 128, 512]},
    "optimize": {"focus_concurrencies": [32, 128]},
}


def test_focus_concurrencies_reach_the_scoring_roles(tmp_path):
    captured = _capture_driving_prompts(tmp_path, _FOCUS_EXTRA)
    # Evaluator: the gate line and the mean rule name the scored subset;
    # every point is still measured and protected by no-regress.
    evaluator = captured["evaluator"]
    assert "focus_concurrencies=[32, 128]" in evaluator
    assert "scored subset" in evaluator
    assert "no point (scored or not) may" in evaluator
    assert "[8, 32, 128]" in evaluator  # still measures every point
    # Analyzer: round-1 authoring derives the ledger value from the subset.
    analyzer = captured["analyzer"]
    assert "`optimize.focus_concurrencies` [32, 128]" in analyzer
    # QA: the cumulative number is the focus-subset mean.
    qa = captured["qa"]
    assert "scored subset" in qa
    # Reporter: headline means are declared focus-scoped.
    assert "focus_concurrencies" in captured["reporter"]


def test_evaluator_prompt_carries_the_regression_budget(tmp_path):
    captured = _capture_driving_prompts(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 32, 128], "num_prompts": [32, 128, 512]},
            "optimize": {"max_regression_pct": 8.0},
        },
    )
    prompt = captured["evaluator"]
    assert "max_regression_pct=8.0" in prompt
    assert "declared regression budget" in prompt
    # Strict default: no budget language when the key is absent.
    strict = _capture_driving_prompts(
        tmp_path,
        {"benchmark": {"concurrency": [8, 32, 128], "num_prompts": [32, 128, 512]}},
    )["evaluator"]
    assert "max_regression_pct" not in strict
    assert "noise floor" in strict


def test_curve_prompts_without_focus_score_all_points(tmp_path):
    captured = _capture_driving_prompts(
        tmp_path,
        {"benchmark": {"concurrency": [8, 32, 128], "num_prompts": [32, 128, 512]}},
    )
    for role in ("analyzer", "evaluator", "qa", "reporter"):
        assert "focus_concurrencies" not in captured[role], role
        assert "scored subset" not in captured[role], role


def test_optimizer_prompt_does_not_treat_parallel_siblings_as_earlier_verdicts(tmp_path):
    # First item of round 1: no earlier verdicts exist — no pointer.
    first = _capture_driving_prompts(tmp_path)["optimizer"]
    assert "Earlier items' verdicts" not in first

    # Parallel siblings run from one frozen base, so a sibling's verdict
    # cannot be assumed to exist when this prompt is authored.
    task = _write_task(tmp_path)
    ws = tmp_path / "ws2"
    workflow = Workflow(workspace=ws)
    validated = task_schema.load_and_validate_task_yaml(str(task))
    workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")
    state = state_module.WorkflowState(
        task_path=str(workflow.task_path),
        round_index=0,
        item_index=1,
        attempt_index=0,
        current_item_id="opt-002",
        campaign_git_branch="perf-optimize/test-branch",
        campaign_git_base_commit="a" * 40,
        stage=state_module.STAGE_OPTIMIZER,
    )
    captured: dict[str, str] = {}
    original = workflow.optimizer
    workflow.optimizer = lambda prompt: captured.__setitem__("optimizer", prompt)
    try:
        workflow._run_optimizer(state)
    finally:
        workflow.optimizer = original
        workflow.close()
    prompt = captured["optimizer"]
    assert "Earlier items' verdicts" not in prompt

    # Serial items really are ordered. A later item in the same round can
    # and should consume the completed predecessors' failure evidence.
    state.item_execution = "serial"
    workflow = Workflow(workspace=ws)
    workflow.optimizer = lambda prompt: captured.__setitem__("optimizer", prompt)
    try:
        workflow._run_optimizer(state)
    finally:
        workflow.close()
    assert "Earlier items' verdicts" in captured["optimizer"]


def test_reporter_prompt_points_at_base_commit_and_inputs(tmp_path):
    captured = _capture_driving_prompts(tmp_path)
    prompt = captured["reporter"]
    assert "a" * 12 in prompt  # abbreviated base commit
    assert "roadmap.yaml" in prompt
    assert "verification_report.md" in prompt
    assert "profile/nsys_stats.txt" in prompt
    assert "Launch no servers" in prompt


def test_projector_prompt_drives_the_sol_skill_over_optimize_artifacts(tmp_path):
    captured = _capture_driving_prompts(tmp_path, _sol_extra(tmp_path))
    prompt = captured["projector"]
    # The skill is the methodology, named by the spelling this session
    # actually loaded (resolved in Python before the campaign started).
    assert "internal-perf-sol-analysis" in prompt
    assert "`Skill` tool" in prompt
    assert "measure_channels.py" in prompt
    # perf-optimize's artifact layout: the baseline lives under
    # baseline/, and the parallel mapping comes from the live tuning
    # config.
    assert "baseline/benchmark_results.md" in prompt
    assert "tuning/extra_llm_api_options.yaml" in prompt
    assert "sol_projection.md" in prompt
    # The model architecture comes from the checkpoint's config.json;
    # no trace of the removed dlsim cross-check.
    assert "config.json" in prompt
    assert "dlsim" not in prompt
    # Placement context: once per campaign, for the analyzer/reporter.
    assert "once per campaign" in prompt
    # The machine-readable peaks file is persisted for the analyzer's
    # per-round correlation.
    assert "sol_work/peaks.json" in prompt
    # The SOL skill is internal-only, so open-source toolkit builds strip it.
    # Which methodology this run has was resolved in Python before the
    # campaign started, so the message names one skill and the spelling to
    # load it by.
    assert "**load the `internal-perf-sol-analysis` skill**" in prompt
    assert "perf-analysis` skill**" not in prompt.replace("sol-analysis` skill**", "")
    # The internal-knowledge route lives in the system prompt, not here.
    assert "internal-glean-search" not in prompt
    assert "internal-glean-specialist" not in prompt


def test_analyzer_optimizer_reporter_prompts_point_at_projection_iff_sol(tmp_path):
    without = _capture_driving_prompts(tmp_path, _sol_off_extra())
    assert "sol_projection.md" not in without["analyzer"]
    assert "sol_projection.md" not in without["optimizer"]
    assert "Projection vs Measured" not in without["reporter"]
    assert "sol_calc.py analyze" not in without["analyzer"]
    assert "SOL correlation" not in without["analyzer"]

    with_sol = _capture_driving_prompts(tmp_path, _sol_extra(tmp_path))
    analyzer = with_sol["analyzer"]
    assert "sol_projection.md" in analyzer
    # Context, not evidence: the trace outranks the projection.
    assert "optional context" in analyzer
    assert "measured trace evidence always outranks the projection" in analyzer
    # The measured↔SOL correlation runs after profiling: regions from
    # this round's traces, joined against the projector's peaks file,
    # into the findings' dedicated section.
    assert "sol_calc.py analyze" in analyzer
    assert "internal-perf-sol-analysis" in analyzer
    assert "regions.json" in analyzer
    assert "sol_work/peaks.json" in analyzer
    assert "SOL correlation" in analyzer
    assert "Correlation unavailable" in analyzer
    # An exhausted roadmap owes the remaining-gap attribution.
    assert "Remaining-gap attribution" in analyzer
    optimizer = with_sol["optimizer"]
    assert "sol_projection.md" in optimizer
    # Context, not spec: aim at the binding ceiling, never grow the item.
    assert "context, not spec" in optimizer
    assert "binding ceiling" in optimizer
    assert "never expands the item" in optimizer
    assert "SOL alignment:" in optimizer
    reporter = with_sol["reporter"]
    assert "sol_projection.md" in reporter
    # The section slots between Final Verification and the diff summary,
    # and closes with the remaining-gap accountability breakdown.
    assert "Final Verification / Projection vs Measured / Config & Code Diff" in reporter
    assert "remaining-gap accountability" in reporter

    # The evaluator and QA judge on measurements alone — no projection
    # pointer even when the sol block is set.
    assert "sol_projection.md" not in with_sol["evaluator"]
    assert "sol_projection.md" not in with_sol["qa"]


def test_analyzer_prompt_instructs_the_ncu_deep_dive(tmp_path):
    without = _capture_driving_prompts(tmp_path, _sol_off_extra())["analyzer"]
    with_sol = _capture_driving_prompts(tmp_path, _sol_extra(tmp_path))["analyzer"]
    # The ncu deep dive is not SOL-gated — every *profiling* prompt covers
    # the top nsys kernels under ncu with the skill as capture +
    # interpretation methodology, saves the report next to the other
    # traces, and the findings carry the dedicated section.
    for prompt in (without, with_sol):
        assert "perf-nsight-compute-analysis" in prompt
        assert "trtllm-agent-toolkit:perf-nsight-compute-analysis" in prompt
        assert "server_ncu.ncu-rep" in prompt
        assert "ncu kernel analysis" in prompt
        # Roadmap items are grounded across the analyses, not the
        # timeline alone.
        assert "nsys timeline, ncu kernel analysis" in prompt
    # The SOL correlation joins the grounding list only when the
    # projector stage ran.
    assert "nsys timeline, ncu kernel analysis, SOL correlation" not in without
    assert "nsys timeline, ncu kernel analysis, SOL correlation" in with_sol


def test_optimizer_retry_prompt_distinguishes_auto_reject_from_evaluator_reject(tmp_path):
    """An auto-rejected retry must not be pointed at nonexistent evaluator feedback."""
    task = _write_task(tmp_path, {**_sol_off_extra(), "optimize": {"approaches": ["code"]}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    validated = task_schema.load_and_validate_task_yaml(str(task))
    workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")

    captured: dict[str, str] = {}
    original_optimizer = workflow.optimizer
    workflow.optimizer = lambda prompt: captured.__setitem__("optimizer", prompt)
    base = dict(
        task_path=str(workflow.task_path),
        round_index=0,
        attempt_index=1,  # attempt 2 — a retry
        current_item_id="opt-001",
        campaign_git_branch="perf-optimize/test-branch",
        campaign_git_base_commit="a" * 40,
        stage=state_module.STAGE_OPTIMIZER,
    )
    try:
        workflow._run_optimizer(
            state_module.WorkflowState(
                **base,
                approach_violation=(
                    "the attempt changed tuning/extra_llm_api_options.yaml, but "
                    "'config' is not in optimize.approaches"
                ),
            )
        )
        auto_rejected = captured["optimizer"]
        workflow._run_optimizer(state_module.WorkflowState(**base))
        evaluator_rejected = captured["optimizer"]
    finally:
        workflow.optimizer = original_optimizer
        workflow.close()

    assert "auto-REJECTED" in auto_rejected
    assert "without evaluation" in auto_rejected
    assert "`code`" in auto_rejected  # names the allowed approaches
    assert "read_latest_progress" not in auto_rejected
    assert "read_latest_progress" in evaluator_rejected
    assert "PUSHED BACK" in evaluator_rejected
    assert "auto-REJECTED" not in evaluator_rejected


# ------------------------------------------------------- per-kernel coverage


_KC_EXTRA = {"profile": {"kernel_coverage": {}}}


def _ledger_yaml(faster_ref: str = "opt-001") -> str:
    return yaml.safe_dump(
        {
            "version": 1,
            "source": "rounds/round_1/analysis/nsys_stats.txt",
            "coverage": {
                "enumerated_share_pct": 96.0,
                "other_share_pct": 4.0,
                "min_share_pct": 0.5,
            },
            "kernels": [
                {
                    "kernel": "gdn_bf16_state",
                    "full_name": "void tensorrt_llm::kernels::gdn<...>",
                    "share_pct": 96.0,
                    "ncu": {
                        "duration_us": 41.2,
                        "sm_sol_pct": 12.1,
                        "mem_sol_pct": 78.5,
                        "occupancy_pct": 62.0,
                        "bound": "memory",
                    },
                    "faster": {"disposition": "item", "ref": faster_ref},
                    "fusion": {
                        "disposition": "dismissed",
                        "neighbors": "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace)",
                        "ref": "already-fused: neighbors are inside this kernel",
                    },
                }
            ],
        },
        sort_keys=False,
    )


def _stub_agents_with_ledger(workflow, faster_ref: str = "opt-001", **kwargs):
    """`_stub_agents` plus an analyzer that also writes a kernel ledger."""
    trace = _stub_agents(workflow, **kwargs)
    original_analyzer = workflow._run_analyzer

    def analyzer_with_ledger(state):
        original_analyzer(state)
        ledger = workflow._analysis_dir(state) / "kernel_ledger.yaml"
        ledger.write_text(_ledger_yaml(faster_ref), encoding="utf-8")

    workflow._run_analyzer = analyzer_with_ledger
    return trace


def test_kernel_coverage_run_completes_with_valid_ledger(tmp_path, fake_git):
    task = _write_task(tmp_path, _KC_EXTRA)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents_with_ledger(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    # The resolved spec carries the merged contract bars.
    resolved = yaml.safe_load((ws / "task.yaml").read_text(encoding="utf-8"))
    assert resolved["profile"]["kernel_coverage"] == {
        "min_share_pct": 0.5,
        "coverage_target_pct": 95.0,
    }


def test_kernel_coverage_waives_the_ledger_on_a_replan_only_round(tmp_path, fake_git):
    """The contract demands a fresh ledger from a profile, not from a replan.

    A replan-only round runs no ncu at all — the standing ledger still
    describes the build, because the round that preceded it accepted
    nothing. Enforcing the contract there would abort the stage over an
    artifact the round was told not to produce.
    """
    task = _write_task(tmp_path, {**_KC_EXTRA, "optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -1.0, 99.0)],
    )
    original_analyzer = workflow._run_analyzer

    def analyzer_with_first_round_ledger(state):
        original_analyzer(state)
        if state.round_index == 0:
            ledger = workflow._analysis_dir(state) / "kernel_ledger.yaml"
            ledger.write_text(_ledger_yaml("opt-001"), encoding="utf-8")

    workflow._run_analyzer = analyzer_with_first_round_ledger
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Both items were rejected, so the campaign ends without a final
    # verification — and without ever aborting on the ledger gate. Rounds
    # 2 and 3 are both replan-only (nothing was accepted at any point),
    # so the waiver has to hold for each of them.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "reporter",
    ]
    assert (ws / "rounds" / "round_1" / "analysis" / "kernel_ledger.yaml").is_file()
    # The waiver was exercised, not accidentally satisfied.
    assert not (ws / "rounds" / "round_2" / "analysis" / "kernel_ledger.yaml").exists()
    assert not (ws / "rounds" / "round_3" / "analysis" / "kernel_ledger.yaml").exists()


def test_kernel_coverage_missing_ledger_blocks_advance(tmp_path, fake_git):
    """The analyzer gate treats a missing ledger like a missing deliverable."""
    task = _write_task(tmp_path, _KC_EXTRA)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)  # writes findings + roadmap, never the ledger
    try:
        with pytest.raises(RuntimeError, match="kernel_ledger.yaml"):
            workflow.run(str(task))
    finally:
        workflow.close()
    # The checkpoint stays parked at the analyzer, so a re-run retries it.
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_ANALYZER


def test_kernel_coverage_unresolved_item_ref_blocks_advance(tmp_path, fake_git):
    """An `item` disposition must point at a real roadmap id."""
    task = _write_task(tmp_path, _KC_EXTRA)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents_with_ledger(workflow, faster_ref="opt-999")
    try:
        with pytest.raises(RuntimeError, match="does not match any roadmap item id"):
            workflow.run(str(task))
    finally:
        workflow.close()
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.stage == state_module.STAGE_ANALYZER


def test_kernel_coverage_below_target_blocks_advance(tmp_path, fake_git):
    task = _write_task(tmp_path, _KC_EXTRA)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(workflow)
    original_analyzer = workflow._run_analyzer

    def analyzer_with_thin_ledger(state):
        original_analyzer(state)
        data = yaml.safe_load(_ledger_yaml())
        data["coverage"]["enumerated_share_pct"] = 80.0
        data["coverage"]["other_share_pct"] = 20.0
        (workflow._analysis_dir(state) / "kernel_ledger.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )

    workflow._run_analyzer = analyzer_with_thin_ledger
    try:
        with pytest.raises(RuntimeError, match="coverage_target_pct"):
            workflow.run(str(task))
    finally:
        workflow.close()
    assert trace[-1] == "analyzer"


def test_without_kernel_coverage_no_ledger_is_required(tmp_path, fake_git):
    """The historical behavior is untouched when the block is absent."""
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.done is True
    assert not list((ws / "rounds").glob("*/analysis/kernel_ledger.yaml"))


def test_kernel_coverage_driving_prompts_name_ledger_and_section(tmp_path):
    captured = _capture_driving_prompts(tmp_path, _KC_EXTRA)
    analyzer = captured["analyzer"]
    assert "kernel_ledger.yaml" in analyzer
    assert "per-kernel coverage contract" in analyzer
    assert "0.5%" in analyzer and "95.0%" in analyzer
    assert "faster? fusible?" in analyzer
    # The default bounded top-kernel wording is superseded, not repeated.
    assert "on the top nsys kernels: keep the canonical ncu flags" not in analyzer
    reporter = captured["reporter"]
    assert "Kernel-Level Comparison / Kernel Coverage / Failed Attempts" in reporter
    # No round produced a ledger in this bare workspace — the reporter is
    # told to report it unavailable rather than hunt for one.
    assert "none was written" in reporter


def test_default_driving_prompts_omit_the_coverage_contract(tmp_path):
    captured = _capture_driving_prompts(tmp_path)
    assert "kernel_ledger.yaml" not in captured["analyzer"]
    assert "Kernel Coverage / " not in captured["reporter"]
    assert "server_ncu.ncu-rep" in captured["analyzer"]


def test_reporter_prompt_names_the_highest_round_ledger(tmp_path, fake_git):
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.reporter = recorder
    try:
        task = _write_task(tmp_path, _KC_EXTRA)
        validated = task_schema.load_and_validate_task_yaml(str(task))
        workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")
        for round_name in ("round_2", "round_10"):
            analysis = ws / "rounds" / round_name / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "kernel_ledger.yaml").write_text(_ledger_yaml(), encoding="utf-8")
        state = state_module.WorkflowState(
            task_path=str(workflow.task_path),
            campaign_git_branch="perf-optimize/test-branch",
            campaign_git_base_commit="a" * 40,
            stage=state_module.STAGE_REPORTER,
        )
        workflow._run_reporter(state)
        prompt = recorder.messages[0]
        # Numeric round ordering: round_10 outranks round_2.
        assert str(ws / "rounds" / "round_10" / "analysis" / "kernel_ledger.yaml") in prompt
        assert "none was written" not in prompt
    finally:
        workflow.close()


# ------------------------------------------------ profile-aware round control


def test_rejected_parallel_batch_replans_without_reprofile(tmp_path, fake_git):
    """A fully rejected batch earns a replan without another profile.

    Every attempt is hard-reverted, leaving the build byte-identical to
    what the round's analyzer profiled — so the round keeps pulling
    pending items instead of paying for a re-profile that would re-derive
    the same findings.
    """
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[
            [
                _item("opt-001", gain=10.0),
                _item("opt-002", gain=8.0),
                _item("opt-003", gain=5.0),
            ]
        ],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -0.4, 99.6)],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace.count("benchmarker") == 1
    assert trace.count("projector") == 1
    assert trace.count("analyzer") == 2
    assert trace.count("optimizer") == 3
    assert trace.count("evaluator") == 3
    assert trace.count("qa") == 0
    assert trace.count("reporter") == 1
    assert (ws / "rounds" / "round_2").exists()
    roadmap = roadmap_schema.load_roadmap(ws / "roadmap.yaml")
    for item_id in ("opt-001", "opt-002", "opt-003"):
        assert roadmap_schema.find_item(roadmap, item_id)["status"] == "failed"
    round_1 = ws / "rounds" / "round_1"
    assert (round_1 / "item_1_opt-001").is_dir()
    assert (round_1 / "item_2_opt-002").is_dir()
    assert (round_1 / "item_3_opt-003").is_dir()
    # Nothing accepted -> no final verification, and the campaign ends on
    # the roadmap-exhausted break rather than the round budget.
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.round_index == 2
    assert state.done is True


def test_item_budget_reprofiles_remaining_items(tmp_path, fake_git):
    """A full three-item batch leaves the fourth item for round two."""
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 3}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[
            [
                _item("opt-001", gain=10.0),
                _item("opt-002", gain=8.0),
                _item("opt-003", gain=5.0),
                _item("opt-004", gain=4.0),
            ]
        ],
        evaluator_verdicts=[
            ("REJECT", "perf_shortfall", -0.4, 99.6),
            ("REJECT", "perf_shortfall", -0.2, 99.8),
            ("APPROVE", "none", 8.4, 108.4),
            ("APPROVE", "none", 3.0, 111.6),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace.count("projector") == 1
    assert trace.count("analyzer") == 3
    assert trace.count("optimizer") == 4
    assert trace.count("evaluator") == 4
    assert trace.count("qa") == 1
    assert trace.count("reporter") == 1
    round_1 = ws / "rounds" / "round_1"
    assert (round_1 / "item_1_opt-001").is_dir()
    assert (round_1 / "item_2_opt-002").is_dir()
    assert (round_1 / "item_3_opt-003").is_dir()
    assert (ws / "rounds" / "round_2" / "analysis" / "profile_findings.md").is_file()
    assert (ws / "rounds" / "round_2" / "item_1_opt-004").is_dir()
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.round_index == 3


def test_item_budget_is_checkpointed(tmp_path):
    state = state_module.WorkflowState(task_path="t", max_items_per_round=2)
    path = tmp_path / state_module.STATE_FILENAME
    state_module.save_state(path, state)
    assert state_module.load_state(path).max_items_per_round == 2


def test_max_items_per_round_one_reprofiles_each_item(tmp_path, fake_git):
    """A one-item batch leaves the second item for the next round."""
    task = _write_task(
        tmp_path,
        {"optimize": {"max_items_per_round": 1}},
    )
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    trace = _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[
            ("APPROVE", "none", 8.4, 108.4),
            ("APPROVE", "none", 3.0, 111.6),
        ],
    )
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Each one-item batch closes the round before the next item is selected.
    assert trace == [
        "benchmarker",
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    assert (ws / "rounds" / "round_1" / "item_1_opt-001").is_dir()
    assert (ws / "rounds" / "round_2" / "item_1_opt-002").is_dir()
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.max_items_per_round == 1


# ------------------------------------------------------------- analysis reuse


def _analyze_workspace(root: Path, *, baseline: bool = True, findings: bool = True) -> Path:
    """A previous perf-analyze workspace to reuse."""
    root.mkdir(parents=True, exist_ok=True)
    if baseline:
        (root / "benchmark_results.md").write_text("# baseline\n", encoding="utf-8")
        (root / "bench_result.json").write_text("{}\n", encoding="utf-8")
    if findings:
        (root / "profile_findings.md").write_text("# findings\n", encoding="utf-8")
        (root / "nsys_stats.txt").write_text("kern_sum\n", encoding="utf-8")
    return root


def test_reuse_analysis_starts_the_campaign_at_the_optimize_stage(tmp_path, fake_git):
    """The imported baseline + findings replace the two GPU stages."""
    source = _analyze_workspace(tmp_path / "prior")
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # No benchmarker: the campaign opens on the projector (the source
    # carried no projection to import) and then the plan-only analyzer.
    assert trace == [
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    analysis = ws / "rounds" / "round_1" / "analysis"
    assert (ws / "baseline" / "benchmark_results.md").read_text(encoding="utf-8") == "# baseline\n"
    assert (ws / "baseline" / "bench_result.json").is_file()
    assert (analysis / "nsys_stats.txt").is_file()
    assert (ws / "reused_analysis" / "manifest.md").is_file()

    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.benchmarker_done is True
    assert state.reuse_analysis_dir == str(source)
    # Consumed by round 1's analyzer, so a resume never re-plans blind.
    assert state.reuse_pending is False
    # The imported profile is the freshest trace of the state under test.
    assert state.last_nsys_dir == str(analysis)


def test_a_reuse_campaign_that_accepts_nothing_still_profiles_its_own_build(tmp_path, fake_git):
    """Round 1's imported traces are prior art, never evidence about this build.

    They were captured by another run against another checkout, so they
    can never stand in for a profile of this campaign's build — and a
    reuse run whose round 1 accepts nothing is exactly where that
    distinction bites: if the import set the "last profiled" pointer,
    every remaining round would open replan-only and the campaign would
    spend its whole budget replanning against a stranger's traces without
    ever profiling the thing it is optimizing.
    """
    source = _analyze_workspace(tmp_path / "prior")
    task = _write_task(tmp_path, {"optimize": {"max_items_per_round": 1}})
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    modes: list[bool] = []
    _stub_agents(
        workflow,
        analyzer_items=[[_item("opt-001", gain=10.0), _item("opt-002", gain=5.0)]],
        evaluator_verdicts=[("REJECT", "perf_shortfall", -1.0, 99.0)],
    )
    original_analyzer = workflow._run_analyzer

    def analyzer_recording_mode(state):
        modes.append(workflow._replan_only(state))
        original_analyzer(state)

    workflow._run_analyzer = analyzer_recording_mode
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    # Round 1 imports, round 2 PROFILES (it is the first look at this
    # build), and only round 3 may replan off round 2's own traces.
    assert modes[:3] == [False, False, True]
    state = state_module.load_state(ws / state_module.STATE_FILENAME)
    assert state.last_profiled_analysis_dir == str(ws / "rounds" / "round_2" / "analysis")


def test_reuse_without_a_baseline_still_measures_one(tmp_path, fake_git):
    """The import is best-effort per artifact."""
    source = _analyze_workspace(tmp_path / "prior", baseline=False)
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[0] == "benchmarker"
    assert state_module.load_state(ws / state_module.STATE_FILENAME).benchmarker_done is True


def test_reuse_without_findings_profiles_normally(tmp_path, fake_git):
    source = _analyze_workspace(tmp_path / "prior", findings=False)
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == [
        "projector",
        "analyzer",
        "optimizer",
        "evaluator",
        "analyzer",
        "qa",
        "reporter",
    ]
    # The analyzer ran for real (it wrote round 1's findings itself).
    assert (ws / "rounds" / "round_1" / "analysis" / "profile_findings.md").is_file()


def test_reuse_of_an_imported_projection_skips_the_projector(tmp_path, fake_git):
    """The SOL ceiling is a property of hardware + model + operating point."""
    source = _analyze_workspace(tmp_path / "prior")
    (source / "sol_projection.md").write_text("# SOL\n", encoding="utf-8")
    (source / "sol_work").mkdir()
    (source / "sol_work" / "peaks.json").write_text("{}\n", encoding="utf-8")
    task = _write_task(tmp_path, _sol_extra(tmp_path))
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert "projector" not in trace
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == "# SOL\n"
    assert (ws / "sol_work" / "peaks.json").is_file()
    assert state_module.load_state(ws / state_module.STATE_FILENAME).projector_done is True


def test_reuse_is_ignored_on_resume(tmp_path, fake_git):
    """The import is a fresh-run seeding step, not a per-run flag."""
    source = _analyze_workspace(tmp_path / "prior")
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    # A checkpoint from an ordinary run already exists.
    first = Workflow(workspace=ws)
    _stub_agents(first)
    first._init_state(str(task), None)
    first.close()

    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace[0] == "benchmarker"
    assert not (ws / "reused_analysis").exists()
    assert state_module.load_state(ws / state_module.STATE_FILENAME).reuse_analysis_dir == ""


def test_reuse_source_may_not_be_the_run_s_own_workspace(tmp_path, fake_git):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=ws)
    _stub_agents(workflow)
    try:
        with pytest.raises(workflow_module.reuse.ReuseError, match="own workspace"):
            workflow.run(str(task))
    finally:
        workflow.close()


def test_reuse_of_an_empty_source_fails_loudly(tmp_path, fake_git):
    source = tmp_path / "prior"
    source.mkdir()
    task = _write_task(tmp_path)
    workflow = Workflow(workspace=tmp_path / "ws", reuse_analysis=source)
    _stub_agents(workflow)
    try:
        with pytest.raises(workflow_module.reuse.ReuseError, match="no reusable analysis"):
            workflow.run(str(task))
    finally:
        workflow.close()


def test_reused_round_without_a_ledger_does_not_enforce_the_coverage_gate(tmp_path, fake_git):
    """A reused round never ran ncu, so it cannot owe a fresh ledger.

    The contract still binds every round the analyzer actually profiles.
    """
    source = _analyze_workspace(tmp_path / "prior")
    extra = dict(_KC_EXTRA)
    extra["optimize"] = {"max_rounds": 1}
    task = _write_task(tmp_path, extra)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    trace = _stub_agents(workflow)  # writes no kernel_ledger.yaml
    try:
        workflow.run(str(task))
    finally:
        workflow.close()

    assert trace == ["projector", "analyzer", "optimizer", "evaluator", "qa", "reporter"]
    assert state_module.load_state(ws / state_module.STATE_FILENAME).done is True


def test_reused_source_ledger_is_still_validated(tmp_path, fake_git):
    """An imported ledger that fails the contract blocks the stage."""
    source = _analyze_workspace(tmp_path / "prior")
    (source / "kernel_ledger.yaml").write_text(_ledger_yaml("opt-does-not-exist"), encoding="utf-8")
    task = _write_task(tmp_path, _KC_EXTRA)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws, reuse_analysis=source)
    _stub_agents(workflow)
    try:
        with pytest.raises(RuntimeError, match="coverage contract"):
            workflow.run(str(task))
    finally:
        workflow.close()


def test_reused_analyzer_prompt_forbids_profiling_and_names_its_inputs(tmp_path):
    task = _write_task(tmp_path)
    ws = tmp_path / "ws"
    workflow = Workflow(workspace=ws)
    recorder = _RecordingAgent()
    workflow.analyzer = recorder
    try:
        validated = task_schema.load_and_validate_task_yaml(str(task))
        workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")
        workflow.prior_roadmap_path.parent.mkdir(parents=True, exist_ok=True)
        workflow.prior_roadmap_path.write_text("version: 1\n", encoding="utf-8")
        state = state_module.WorkflowState(
            task_path=str(workflow.task_path),
            reuse_analysis_dir=str(tmp_path / "prior"),
            reuse_pending=True,
            stage=state_module.STAGE_ANALYZER,
        )
        workflow._run_analyzer(state)
    finally:
        workflow.close()

    prompt = recorder.messages[0]
    # The whole point of the reuse: none of the expensive work reruns.
    assert "skips profiling entirely" in prompt
    assert "do **not** run nsys / ncu" in prompt.replace("Do **not**", "do **not**")
    assert str(tmp_path / "prior") in prompt
    assert str(workflow.reuse_manifest_path) in prompt
    # The source campaign's roadmap is reference material, not the ledger.
    assert str(workflow.prior_roadmap_path) in prompt
    assert "read-only prior art" in prompt
    # It still owes the roadmap and the fit check.
    assert str(workflow.roadmap_path) in prompt
    assert "dormant-capability sweep" in prompt


def test_reused_analyzer_prompt_omits_prior_roadmap_when_absent(tmp_path):
    """A perf-analyze source has no roadmap to offer."""
    task = _write_task(tmp_path)
    workflow = Workflow(workspace=tmp_path / "ws")
    recorder = _RecordingAgent()
    workflow.analyzer = recorder
    try:
        validated = task_schema.load_and_validate_task_yaml(str(task))
        workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")
        workflow._run_analyzer(
            state_module.WorkflowState(
                task_path=str(workflow.task_path),
                reuse_analysis_dir=str(tmp_path / "prior"),
                reuse_pending=True,
                stage=state_module.STAGE_ANALYZER,
            )
        )
    finally:
        workflow.close()

    assert "read-only prior art" not in recorder.messages[0]


def test_reporter_prompt_flags_an_inherited_baseline(tmp_path):
    workflow = Workflow(workspace=tmp_path / "ws")
    recorder = _RecordingAgent()
    workflow.reporter = recorder
    try:
        task = _write_task(tmp_path)
        validated = task_schema.load_and_validate_task_yaml(str(task))
        workflow.task_path.write_text(task_schema.dump_task_yaml(validated), encoding="utf-8")
        workflow._run_reporter(
            state_module.WorkflowState(
                task_path=str(workflow.task_path),
                reuse_analysis_dir=str(tmp_path / "prior"),
                campaign_git_branch="perf-optimize/test-branch",
                campaign_git_base_commit="a" * 40,
                stage=state_module.STAGE_REPORTER,
            )
        )
    finally:
        workflow.close()

    prompt = recorder.messages[0]
    assert str(workflow.reuse_manifest_path) in prompt
    assert "measured by that run, not this one" in prompt.replace("**", "")


# --------------------------------------------------------------------------- #
# The SOL methodology is resolved in Python before the campaign starts and
# handed to the workflow, so the projector is told to load a skill that is
# actually there. Nothing else in the campaign changes: the analyzer's
# correlation and the reporter's weighing already degrade on their own when
# the projection or the peaks file is missing.
# --------------------------------------------------------------------------- #


def test_projector_prompt_names_the_resolved_fallback_skill(tmp_path):
    """The open-source-toolkit case: `perf-analysis` stands in."""
    methodology = SolMethodology(name="reduced", skill="trtllm-agent-toolkit:perf-analysis")
    captured = _capture_driving_prompts(tmp_path, _sol_extra(tmp_path), sol_methodology=methodology)

    projector = captured["projector"]
    assert "Load the `trtllm-agent-toolkit:perf-analysis` skill" in projector
    assert "not installed in this session" in projector
    # No calculator, so no peaks file — nothing downstream reads one.
    assert "peaks calculator you do not have" in projector
    assert "do **not** write" in projector
    assert "measure_channels.py" not in projector
    # And it still degrades honestly rather than inventing a ceiling.
    assert "Projection unavailable" in projector
    assert "never fabricate numbers" in projector

    # Every other role keeps the message it has on a full run (the two
    # captures use different workspaces, so compare with that normalized out).
    full = _capture_driving_prompts(tmp_path, _sol_extra(tmp_path))
    for role in ("analyzer", "optimizer", "reporter"):
        assert captured[role].replace("/ws-reduced", "/ws") == full[role], role


def test_projector_prompt_fails_open_when_the_probe_could_not_run(tmp_path):
    """An unreachable probe must not silently downgrade a stage the user asked for."""
    captured = _capture_driving_prompts(
        tmp_path, _sol_extra(tmp_path), sol_methodology=SolMethodology(probed=False)
    )
    projector = captured["projector"]
    assert "load the `internal-perf-sol-analysis` skill" in projector
    # ...and the agent is handed both spellings, since the name is a guess.
    assert "trtllm-agent-toolkit:internal-perf-sol-analysis" in projector
    assert "if the bare name is not found" in projector


def test_the_cli_offers_no_way_to_suppress_path_checks() -> None:
    """`--paths-prevalidated` is gone, and must not come back.

    It was a flag the adapter passed to say "these paths are on the cluster". The
    spec says that now (`slurm-environment.cluster_ssh`), which is one fact in one
    place instead of two that had to agree — and a flag that could be passed
    WITHOUT the ssh check behind it deleted the guarantee rather than relocating
    it.

    Asserted on the CLI source rather than on `--help`, because a reintroduced
    flag is a source edit and this is the file a reviewer reads.
    """
    from pathlib import Path as _Path

    cli = _Path(__file__).resolve().parents[3] / "agent_flow/workflows/perf_optimize/cli.py"
    src = cli.read_text(encoding="utf-8")

    assert "--paths-prevalidated" not in src
    assert "paths_prevalidated" not in src, (
        "the workflow decides this from the task spec; the CLI must not offer it"
    )


# --------------------------------------------------- the baseline measurement gate
#
# Not disagg-specific: every campaign is gated on the baseline having produced a
# measurement, because a report can exist and still carry no numbers and every
# later stage replays the same operating point.


def _workflow_with_baseline(tmp_path, results: dict | None, *, metric="output_throughput"):
    from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

    baseline = tmp_path / "ws" / "baseline"
    (baseline / "bench" / "concurrency_32").mkdir(parents=True)
    if results is not None:
        (baseline / "bench" / "concurrency_32" / "result.json").write_text(
            yaml.safe_dump(results), encoding="utf-8"
        )
    wf = PerfOptimizeWorkflow.__new__(PerfOptimizeWorkflow)
    wf.baseline_dir = baseline
    wf.baseline_results_path = baseline / "benchmark_results.md"
    wf._optimize_block = lambda: {"target_metric": metric}
    return wf


def test_baseline_gate_passes_when_a_result_json_carries_the_target_metric(tmp_path):
    wf = _workflow_with_baseline(tmp_path, {"output_throughput": 40707.75, "completed": 2560})
    wf._require_baseline_measurement()  # does not raise


def test_baseline_gate_stops_a_campaign_that_measured_nothing(tmp_path):
    """A BLOCKED baseline must not advance: every later stage replays it."""
    wf = _workflow_with_baseline(tmp_path, None)
    with pytest.raises(RuntimeError, match="no measurement"):
        wf._require_baseline_measurement()


def test_baseline_gate_rejects_a_result_json_without_the_target_metric(tmp_path):
    wf = _workflow_with_baseline(tmp_path, {"completed": 0}, metric="median_tpot_ms")
    with pytest.raises(RuntimeError, match="median_tpot_ms"):
        wf._require_baseline_measurement()


def test_baseline_gate_ignores_unreadable_json(tmp_path):
    wf = _workflow_with_baseline(tmp_path, {"output_throughput": 1.0})
    (wf.baseline_dir / "junk.json").write_text("\x00not json\x00", encoding="utf-8")
    wf._require_baseline_measurement()  # the good one still counts

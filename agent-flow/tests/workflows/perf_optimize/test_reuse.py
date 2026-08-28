"""Tests for importing a previous run's analysis into a new workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.perf_optimize import reuse

# --------------------------------------------------------------------- helpers


def _write(path: Path, text: str = "content\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _perf_analyze_workspace(root: Path) -> Path:
    """A completed perf-analyze workspace (flat layout)."""
    _write(root / "benchmark_results.md", "# baseline\n")
    _write(root / "profile_findings.md", "# findings\n")
    _write(root / "sol_projection.md", "# SOL\n")
    _write(root / "sol_work" / "peaks.json", "{}\n")
    _write(root / "server_nsys.nsys-rep", "trace\n")
    _write(root / "nsys_stats.txt", "kern_sum\n")
    _write(root / "concurrency_8" / "result.json", "{}\n")
    _write(root / "bench_result.json", "{}\n")
    # Not analysis artifacts — the import must leave these behind.
    _write(root / "performance_report.md", "# report\n")
    _write(root / "task.yaml", "checkpoint_path: /x\n")
    _write(root / ".perf_analyze_state.json", "{}\n")
    return root


def _perf_optimize_workspace(root: Path, rounds: int = 2) -> Path:
    """A completed perf-optimize campaign workspace (nested layout)."""
    _write(root / "baseline" / "benchmark_results.md", "# baseline\n")
    _write(root / "baseline" / "concurrency_8" / "result.json", "{}\n")
    _write(root / "sol_projection.md", "# SOL\n")
    _write(root / "sol_work" / "peaks.json", "{}\n")
    _write(root / "roadmap.yaml", "version: 1\n")
    for index in range(1, rounds + 1):
        analysis = root / "rounds" / f"round_{index}" / "analysis"
        _write(analysis / "profile_findings.md", f"# findings round {index}\n")
        _write(analysis / "server_nsys.nsys-rep", "trace\n")
        _write(analysis / "kernel_ledger.yaml", f"version: 1  # round {index}\n")
    _write(root / "optimization_report.md", "# report\n")
    return root


def _import_into(source: Path, workspace: Path) -> reuse.ImportedAnalysis:
    return reuse.import_analysis(
        reuse.discover(source),
        workspace=workspace,
        baseline_dir=workspace / "baseline",
        analysis_dir=workspace / "rounds" / "round_1" / "analysis",
        sol_projection_path=workspace / "sol_projection.md",
        sol_work_dir=workspace / "sol_work",
        reuse_dir=workspace / reuse.REUSE_DIRNAME,
    )


# -------------------------------------------------------------------- discover


def test_discover_reads_the_perf_analyze_layout(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "analyze")
    found = reuse.discover(source)
    assert found.baseline_report == source / "benchmark_results.md"
    assert found.findings == source / "profile_findings.md"
    assert found.sol_projection == source / "sol_projection.md"
    assert found.sol_work == source / "sol_work"
    # perf-analyze never writes a roadmap.
    assert found.prior_roadmap is None
    assert found.kernel_ledger is None
    assert found.is_empty is False


def test_discover_reads_the_perf_optimize_layout(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    found = reuse.discover(source)
    assert found.baseline_report == source / "baseline" / "benchmark_results.md"
    assert found.prior_roadmap == source / "roadmap.yaml"
    assert found.kernel_ledger == found.findings.parent / "kernel_ledger.yaml"


def test_discover_picks_the_newest_round_numerically(tmp_path):
    """``round_10`` outranks ``round_9`` — the last state profiled wins."""
    source = _perf_optimize_workspace(tmp_path / "optimize", rounds=10)
    found = reuse.discover(source)
    assert found.findings == (source / "rounds" / "round_10" / "analysis" / "profile_findings.md")


def test_discover_skips_a_trailing_replan_round(tmp_path):
    """The newest round is not always the one that profiled.

    A perf-optimize round opens **replan-only** when the previous one
    accepted nothing: it writes a short replan note into its `analysis/`
    and captures nothing. A plateau campaign typically ends on one, so
    taking the numerically newest findings would import a pointer note
    with no traces beside it — the round before holds the real evidence.
    """
    source = _perf_optimize_workspace(tmp_path / "optimize", rounds=2)
    replan = source / "rounds" / "round_3" / "analysis"
    _write(replan / "profile_findings.md", "# replan note (round 3)\n")

    found = reuse.discover(source)

    assert found.findings == (source / "rounds" / "round_2" / "analysis" / "profile_findings.md")
    assert found.kernel_ledger == found.findings.parent / "kernel_ledger.yaml"


@pytest.mark.parametrize("trace_kind", ["torch", "ncu"])
def test_discover_recognizes_a_profile_that_omitted_nsys(tmp_path, trace_kind):
    """`profile.methods` supports torch-only and ncu-only profiling rounds."""
    source = tmp_path / "optimize"
    _write(source / "baseline" / "benchmark_results.md", "# baseline\n")
    profiled = source / "rounds" / "round_2" / "analysis"
    _write(profiled / "profile_findings.md", "# profiled without nsys\n")
    if trace_kind == "torch":
        _write(profiled / "torch_trace" / "trace.json", "{}\n")
    else:
        _write(profiled / "server_ncu.ncu-rep", "capture\n")
    replan = source / "rounds" / "round_3" / "analysis"
    _write(replan / "profile_findings.md", "# trailing replan note\n")

    found = reuse.discover(source)

    assert found.findings == profiled / "profile_findings.md"


def test_discover_falls_back_to_prose_when_no_round_profiled(tmp_path):
    """Importing findings without traces still beats importing nothing."""
    source = tmp_path / "optimize"
    _write(source / "baseline" / "benchmark_results.md", "# baseline\n")
    _write(source / "rounds" / "round_1" / "analysis" / "profile_findings.md", "# note\n")

    found = reuse.discover(source)

    assert found.findings == (source / "rounds" / "round_1" / "analysis" / "profile_findings.md")


def test_discover_ignores_blank_managed_placeholders(tmp_path):
    """A fresh run pre-creates blank files; those are not artifacts."""
    source = tmp_path / "aborted"
    _write(source / "baseline" / "benchmark_results.md", "")
    _write(source / "sol_projection.md", "   \n")
    _write(source / "roadmap.yaml", "")
    found = reuse.discover(source)
    assert found.baseline_report is None
    assert found.sol_projection is None
    assert found.prior_roadmap is None
    assert found.is_empty is True


def test_discover_rejects_a_non_directory(tmp_path):
    with pytest.raises(reuse.ReuseError, match="not a directory"):
        reuse.discover(tmp_path / "nope")


# ---------------------------------------------------------------------- import


def test_import_from_perf_analyze_lands_in_canonical_paths(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "analyze")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    assert (ws / "baseline" / "benchmark_results.md").read_text(encoding="utf-8") == "# baseline\n"
    assert (analysis / "profile_findings.md").read_text(encoding="utf-8") == "# findings\n"
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == "# SOL\n"
    assert (ws / "sol_work" / "peaks.json").is_file()
    # Result JSONs follow the baseline report (the evaluator diffs its
    # full metric set against them), traces follow the findings.
    assert (ws / "baseline" / "bench_result.json").is_file()
    assert (ws / "baseline" / "concurrency_8" / "result.json").is_file()
    assert (analysis / "server_nsys.nsys-rep").is_file()
    assert (analysis / "nsys_stats.txt").is_file()

    assert imported.baseline_report is True
    assert imported.findings is True
    assert imported.sol_projection is True
    assert imported.sol_work is True
    assert imported.kernel_ledger is False
    assert imported.prior_roadmap is False


def test_import_leaves_the_sources_non_analysis_files_behind(tmp_path):
    """The sibling sweep is an allowlist, not a directory copy.

    A perf-analyze workspace keeps its report, spec and checkpoint next
    to the artifacts; importing those would collide with this run's own
    managed files.
    """
    source = _perf_analyze_workspace(tmp_path / "analyze")
    ws = tmp_path / "ws"
    _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    for stray in ("performance_report.md", "task.yaml", ".perf_analyze_state.json"):
        assert not (analysis / stray).exists()
        assert not (ws / "baseline" / stray).exists()
    assert not (ws / "task.yaml").exists()


def test_import_from_perf_optimize_brings_ledger_and_prior_roadmap(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    # The newest round's findings + its sibling ledger.
    assert "round 2" in (analysis / "profile_findings.md").read_text(encoding="utf-8")
    assert "round 2" in (analysis / "kernel_ledger.yaml").read_text(encoding="utf-8")
    assert imported.kernel_ledger is True
    # The source roadmap is prior art only — never this campaign's ledger.
    assert (ws / reuse.REUSE_DIRNAME / reuse.PRIOR_ROADMAP_NAME).is_file()
    assert not (ws / "roadmap.yaml").exists()
    assert imported.prior_roadmap is True


def test_import_writes_a_manifest_naming_source_and_destinations(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    assert imported.manifest_path == ws / reuse.REUSE_DIRNAME / reuse.MANIFEST_NAME
    manifest = imported.manifest_path.read_text(encoding="utf-8")
    assert str(source) in manifest
    assert "benchmark_results.md" in manifest
    assert "rounds/round_1/analysis/profile_findings.md" in manifest
    # The provenance warning the report is expected to relay.
    assert "measured" in manifest


def test_import_is_best_effort_per_artifact(tmp_path):
    """A source with only findings still saves the profile."""
    source = tmp_path / "partial"
    _write(source / "profile_findings.md", "# findings\n")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    assert imported.findings is True
    assert imported.baseline_report is False
    assert imported.sol_projection is False
    assert not (ws / "baseline" / "benchmark_results.md").exists()


def test_import_raises_when_nothing_is_reusable(tmp_path):
    source = tmp_path / "empty"
    _write(source / "notes.txt", "hello\n")
    with pytest.raises(reuse.ReuseError, match="no reusable analysis"):
        _import_into(source, tmp_path / "ws")


def test_import_summary_lists_what_came_in(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    summary = _import_into(source, tmp_path / "ws").summary()
    assert "baseline benchmark" in summary
    assert "profile findings" in summary
    assert "kernel ledger" in summary

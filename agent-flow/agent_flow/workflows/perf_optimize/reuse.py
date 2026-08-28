"""Import a previous run's analysis artifacts into a perf-optimize workspace.

perf-optimize's expensive stages are the ones that touch the GPU: the
baseline benchmark, the SOL projection, and the analyzer's per-round
profile (serve + benchmark replay + nsys + ncu). When a ``perf-analyze``
run — or an earlier ``perf-optimize`` campaign — already produced those
artifacts for the same model / hardware / operating point, re-deriving
them costs hours of GPU time and answers nothing new.

``--reuse-analysis <dir>`` seeds a fresh workspace from such a run: the
baseline report (plus its result JSONs), the SOL projection (plus
``sol_work/peaks.json``), and the newest profile findings (plus their
traces and, when present, ``kernel_ledger.yaml``) are copied into the new
workspace's round-1 layout, so the campaign starts at the optimize stage.
Round 1's analyzer then runs **plan-only** — it authors ``roadmap.yaml``
from the imported evidence without launching a server or a profiler.

Only *artifacts* are imported, never campaign ledger state: a source
``roadmap.yaml`` is copied aside as read-only prior art
(``reused_analysis/roadmap.yaml``) for the plan-only analyzer to weigh,
because its statuses and ``current_best`` describe the **source**
campaign's checkout, not this one's.

Both workspace layouts are probed per artifact, so the source may be
either workflow's workspace::

    perf-analyze            perf-optimize
    ------------            -------------
    benchmark_results.md    baseline/benchmark_results.md
    profile_findings.md     rounds/round_<n>/analysis/profile_findings.md
    sol_projection.md       sol_projection.md
    sol_work/               sol_work/
    (none)                  roadmap.yaml

The import is deliberately best-effort per artifact: a source with only
a baseline report still saves the baseline benchmark, and a source with
only findings still saves the profile. It fails only when neither of
those two is present — nothing would have been reused.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

BASELINE_REPORT_NAME = "benchmark_results.md"
FINDINGS_NAME = "profile_findings.md"
SOL_PROJECTION_NAME = "sol_projection.md"
SOL_WORK_DIRNAME = "sol_work"
ROADMAP_NAME = "roadmap.yaml"
KERNEL_LEDGER_NAME = "kernel_ledger.yaml"

# Where the import parks the provenance record and the source campaign's
# roadmap (prior art the plan-only analyzer reads, never the live ledger).
REUSE_DIRNAME = "reused_analysis"
MANIFEST_NAME = "manifest.md"
PRIOR_ROADMAP_NAME = "prior_roadmap.yaml"

# Sibling artifacts copied alongside the baseline report: the result
# JSONs the evaluator's full-metric diff reads out of the reference
# directory (see ``PerfOptimizeWorkflow._reference_result_dir``).
_BENCHMARK_FILE_GLOBS = ("*.json",)
_BENCHMARK_DIR_GLOBS = ("concurrency_*",)

# Sibling artifacts copied alongside the profile findings: the traces the
# evaluator's kernel comparison and the reporter's before/after read,
# plus the machine-readable analysis products. Deliberately an allowlist
# — a perf-analyze source keeps these next to its reports and checkpoint,
# so copying the whole directory would drag that run's state and report
# files in as well. ``*.sqlite`` is excluded on purpose: it is a
# multi-GB, regenerable export of the ``.nsys-rep`` next to it.
_ANALYSIS_FILE_GLOBS = (
    "*.nsys-rep",
    "*.ncu-rep",
    "nsys_stats*.txt",
    "*ncu*.txt",
    "*ncu*.csv",
    "*ncu*.md",
    KERNEL_LEDGER_NAME,
    "regions.json",
    "sol.json",
)
_ANALYSIS_DIR_GLOBS = ("torch_trace*",)

_ROUND_DIR_RE = re.compile(r"round_(\d+)")


class ReuseError(ValueError):
    """Raised when a ``--reuse-analysis`` source cannot be imported."""


def _nonempty_file(path: Path) -> bool:
    """True iff ``path`` is a file holding non-whitespace content.

    Both workflows pre-create blank managed files (``roadmap.yaml``,
    ``sol_projection.md``) on a fresh run, so existence alone would
    "import" an empty placeholder over a real artifact's absence.
    """
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if _nonempty_file(candidate):
            return candidate
    return None


# What makes a round's ``analysis/`` a profile rather than a replan note:
# an artifact from any supported profiler. ``profile.methods`` may omit
# nsys, so ncu-only and torch-only rounds count too. A perf-optimize round
# that opened replan-only writes findings without capturing any of these.
_PROFILED_ROUND_FILE_GLOBS = ("*.nsys-rep", "nsys_stats*.txt", "*.ncu-rep")
_PROFILED_ROUND_DIR_GLOBS = ("torch_trace*",)


def _has_trace(analysis_dir: Path) -> bool:
    """True iff ``analysis_dir`` holds a capture, not just prose."""
    return any(
        path.is_file()
        for pattern in _PROFILED_ROUND_FILE_GLOBS
        for path in analysis_dir.glob(pattern)
    ) or any(
        path.is_dir() and any(path.iterdir())
        for pattern in _PROFILED_ROUND_DIR_GLOBS
        for path in analysis_dir.glob(pattern)
    )


def latest_round_findings(source: Path) -> Path | None:
    """Newest *profiling* ``rounds/round_<n>/analysis/profile_findings.md``.

    Rounds are ranked numerically so ``round_10`` outranks ``round_9``
    (mirroring the reporter's kernel-ledger lookup), and the newest round
    that actually profiled wins — not simply the newest round. A
    perf-optimize campaign opens a round **replan-only** when its standing
    runtime profile remains current, and such a round writes a short
    replan note into its ``analysis/`` with no traces beside it; a plateau
    campaign typically *ends* on one. Importing that note would seed the
    new run with a pointer to a directory it does not have and no traces
    at all, when the earlier profiling round holds the real evidence.

    Falls back to the numerically newest findings when no round carries a
    trace — an import of prose beats importing nothing.
    """
    candidates: list[tuple[int, Path]] = []
    for path in source.glob(f"rounds/round_*/analysis/{FINDINGS_NAME}"):
        match = _ROUND_DIR_RE.fullmatch(path.parent.parent.name)
        if match and _nonempty_file(path):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    profiled = [(number, path) for number, path in candidates if _has_trace(path.parent)]
    return max(profiled or candidates)[1]


@dataclass(frozen=True)
class DiscoveredAnalysis:
    """What a ``--reuse-analysis`` source actually offers.

    Every field is ``None`` when the source does not carry that artifact
    — the import copies what it found and says so.
    """

    source: Path
    baseline_report: Path | None = None
    findings: Path | None = None
    sol_projection: Path | None = None
    sol_work: Path | None = None
    prior_roadmap: Path | None = None

    @property
    def kernel_ledger(self) -> Path | None:
        """The findings' sibling ``kernel_ledger.yaml``, when present."""
        if self.findings is None:
            return None
        ledger = self.findings.parent / KERNEL_LEDGER_NAME
        return ledger if _nonempty_file(ledger) else None

    @property
    def is_empty(self) -> bool:
        """True when neither expensive artifact was found.

        The two that justify a reuse are the baseline measurement and the
        profile findings; the SOL projection alone would leave both GPU
        stages to re-run.
        """
        return self.baseline_report is None and self.findings is None


def discover(source: str | Path) -> DiscoveredAnalysis:
    """Locate reusable artifacts in ``source`` (either workspace layout).

    Raises :class:`ReuseError` when ``source`` is not a directory; an
    existing directory with nothing reusable comes back with
    ``is_empty`` set, so the caller can report what it probed for.
    """
    root = Path(source).expanduser()
    if not root.is_dir():
        raise ReuseError(f"--reuse-analysis source is not a directory: {root}")
    findings = latest_round_findings(root) or _first_existing(root / FINDINGS_NAME)
    sol_work = root / SOL_WORK_DIRNAME
    return DiscoveredAnalysis(
        source=root,
        baseline_report=_first_existing(
            root / "baseline" / BASELINE_REPORT_NAME, root / BASELINE_REPORT_NAME
        ),
        findings=findings,
        sol_projection=_first_existing(root / SOL_PROJECTION_NAME),
        sol_work=sol_work if sol_work.is_dir() and any(sol_work.iterdir()) else None,
        prior_roadmap=_first_existing(root / ROADMAP_NAME),
    )


@dataclass
class ImportedAnalysis:
    """What :func:`import_analysis` copied into the new workspace."""

    source: Path
    baseline_report: bool = False
    findings: bool = False
    sol_projection: bool = False
    sol_work: bool = False
    kernel_ledger: bool = False
    prior_roadmap: bool = False
    # ``(source, destination)`` pairs, in copy order — the manifest body.
    copied: list[tuple[Path, Path]] = field(default_factory=list)
    manifest_path: Path | None = None

    def summary(self) -> str:
        """One-line human summary of what the import brought in."""
        parts = [
            name
            for name, present in (
                ("baseline benchmark", self.baseline_report),
                ("profile findings", self.findings),
                ("SOL projection", self.sol_projection),
                ("kernel ledger", self.kernel_ledger),
                ("prior roadmap (as reference)", self.prior_roadmap),
            )
            if present
        ]
        return ", ".join(parts) if parts else "nothing"


def _copy_file(src: Path, dst: Path, imported: ImportedAnalysis) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    imported.copied.append((src, dst))


def _copy_siblings(
    anchor: Path,
    dst_dir: Path,
    file_globs: tuple[str, ...],
    dir_globs: tuple[str, ...],
    imported: ImportedAnalysis,
) -> None:
    """Copy ``anchor``'s allowlisted sibling artifacts into ``dst_dir``.

    ``anchor`` is the artifact whose directory is being harvested (the
    baseline report / the profile findings); it is skipped itself, having
    already been copied to its canonical destination name. Dotfiles are
    skipped too: ``pathlib`` globs match them (unlike a shell), so
    ``*.json`` would otherwise import the source run's
    ``.perf_analyze_state.json`` checkpoint.
    """
    src_dir = anchor.parent
    for pattern in file_globs:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_file() and path != anchor and not path.name.startswith("."):
                _copy_file(path, dst_dir / path.name, imported)
    for pattern in dir_globs:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_dir():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, dst_dir / path.name, dirs_exist_ok=True)
                imported.copied.append((path, dst_dir / path.name))


def _render_manifest(imported: ImportedAnalysis, workspace: Path) -> str:
    lines = [
        "# Reused analysis",
        "",
        f"This campaign was launched with `--reuse-analysis {imported.source}`.",
        "",
        "The artifacts below were copied from that run instead of being",
        "re-derived, so round 1's analyzer plans from them **without**",
        "profiling (no server launch, no nsys/ncu, no benchmark). Every",
        "measurement they contain describes the source run's system — the",
        "baseline numbers this campaign's gains are computed against were",
        "measured there, not here.",
        "",
        "| artifact | source | destination |",
        "| --- | --- | --- |",
    ]
    for src, dst in imported.copied:
        try:
            shown_dst = str(dst.relative_to(workspace))
        except ValueError:
            shown_dst = str(dst)
        lines.append(f"| `{dst.name}` | `{src}` | `{shown_dst}` |")
    if not imported.copied:
        lines.append("| _(nothing)_ | | |")
    lines.append("")
    return "\n".join(lines)


def import_analysis(
    discovered: DiscoveredAnalysis,
    *,
    workspace: Path,
    baseline_dir: Path,
    analysis_dir: Path,
    sol_projection_path: Path,
    sol_work_dir: Path,
    reuse_dir: Path,
) -> ImportedAnalysis:
    """Copy ``discovered``'s artifacts into a fresh perf-optimize workspace.

    Destinations are the workspace's canonical paths, so every downstream
    stage reads imported artifacts exactly where it reads freshly
    produced ones: the baseline report and its result JSONs land in
    ``baseline/``, the findings and their traces in round 1's
    ``analysis/``, the projection at ``sol_projection.md`` (with
    ``sol_work/`` beside it). The source roadmap is parked in
    ``reused_analysis/`` as prior art — never as the live ledger.

    Raises :class:`ReuseError` when the source carries neither a baseline
    report nor profile findings.
    """
    if discovered.is_empty:
        raise ReuseError(
            f"no reusable analysis found in {discovered.source}. Looked for a "
            f"baseline report (baseline/{BASELINE_REPORT_NAME} or "
            f"{BASELINE_REPORT_NAME}) and profile findings "
            f"(rounds/round_<n>/analysis/{FINDINGS_NAME} or {FINDINGS_NAME}); "
            f"pass a perf-analyze or perf-optimize workspace that completed at "
            f"least one of those stages."
        )

    imported = ImportedAnalysis(source=discovered.source)

    if discovered.baseline_report is not None:
        _copy_file(discovered.baseline_report, baseline_dir / BASELINE_REPORT_NAME, imported)
        _copy_siblings(
            discovered.baseline_report,
            baseline_dir,
            _BENCHMARK_FILE_GLOBS,
            _BENCHMARK_DIR_GLOBS,
            imported,
        )
        imported.baseline_report = True

    if discovered.findings is not None:
        _copy_file(discovered.findings, analysis_dir / FINDINGS_NAME, imported)
        _copy_siblings(
            discovered.findings,
            analysis_dir,
            _ANALYSIS_FILE_GLOBS,
            _ANALYSIS_DIR_GLOBS,
            imported,
        )
        imported.findings = True
        imported.kernel_ledger = discovered.kernel_ledger is not None

    if discovered.sol_projection is not None:
        _copy_file(discovered.sol_projection, sol_projection_path, imported)
        imported.sol_projection = True

    if discovered.sol_work is not None:
        sol_work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(discovered.sol_work, sol_work_dir, dirs_exist_ok=True)
        imported.copied.append((discovered.sol_work, sol_work_dir))
        imported.sol_work = True

    if discovered.prior_roadmap is not None:
        _copy_file(discovered.prior_roadmap, reuse_dir / PRIOR_ROADMAP_NAME, imported)
        imported.prior_roadmap = True

    manifest_path = reuse_dir / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_render_manifest(imported, workspace), encoding="utf-8")
    imported.manifest_path = manifest_path
    return imported


__all__ = [
    "BASELINE_REPORT_NAME",
    "FINDINGS_NAME",
    "KERNEL_LEDGER_NAME",
    "MANIFEST_NAME",
    "PRIOR_ROADMAP_NAME",
    "REUSE_DIRNAME",
    "DiscoveredAnalysis",
    "ImportedAnalysis",
    "ReuseError",
    "discover",
    "import_analysis",
    "latest_round_findings",
]

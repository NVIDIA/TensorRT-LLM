#!/usr/bin/env python3
"""Phase 1 Detection: determine whether host overhead is the bottleneck.

Computes metrics M1-M5 from an nsys SQLite trace, applies verdict logic,
and outputs a structured JSON report.

Usage:
    python detect_host_overhead.py --trace /path/to/trace.sqlite

    # With custom thresholds
    python detect_host_overhead.py --trace /path/to/trace.sqlite \
        --gpu-idle-threshold 0.25 --output verdict.json

    # JSON-only output (for piping to other tools)
    python detect_host_overhead.py --trace /path/to/trace.sqlite --json-only
"""

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Default thresholds (see references/thresholds.md)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gpu_idle_ratio": 0.30,
    "launch_overhead_ratio": 0.10,
    "host_prep_exposed_ratio": 0.50,
    "host_prep_perf_impact": 0.05,
    "host_prep_idle_attribution": 0.50,
    "gpu_utilization": 0.60,
    "nccl_ratio_caveat": 0.20,
}

GEN_OVERRIDES = {
    "gpu_idle_ratio": 0.15,
    "gpu_utilization": 0.80,
}


@dataclass
class PhaseMetrics:
    """Metrics for a single phase (aggregate, context, or generation)."""

    iteration_count: int = 0
    total_time_us: float = 0.0
    gpu_active_us: float = 0.0
    gpu_idle_us: float = 0.0
    gpu_idle_ratio: float = 0.0
    gpu_utilization: float = 0.0
    launch_overhead_us: float = 0.0
    launch_overhead_ratio: float = 0.0
    nccl_us: float = 0.0
    nccl_ratio: float = 0.0
    host_prep_total_us: Optional[float] = None
    host_prep_exposed_us: Optional[float] = None
    host_prep_exposed_ratio: Optional[float] = None
    host_prep_perf_impact: Optional[float] = None
    host_prep_idle_attribution: Optional[float] = None


@dataclass
class PhaseVerdict:
    """Verdict for a single phase."""

    verdict: str = "NO"
    crossed_count: int = 0
    applicable_count: int = 0
    host_prep_confirmed: bool = False
    nccl_caveat: bool = False
    metrics: Optional[PhaseMetrics] = None
    crossed_details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------
def _get_analysis_window(conn):
    """Return (window_start_ns, window_end_ns, total_time_us)."""
    cur = conn.cursor()
    cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    row = cur.fetchone()
    if not row or row[0] is None:
        return None, None, 0.0
    start, end = row
    return start, end, (end - start) / 1000.0


def _merge_intervals(intervals):
    """Merge overlapping (start, end) intervals. Returns list of merged."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _compute_gpu_active(conn, window_start=None, window_end=None):
    """Compute precise GPU active time by merging overlapping kernel ranges."""
    cur = conn.cursor()
    query = "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
    params = []
    if window_start is not None and window_end is not None:
        query += " WHERE start >= ? AND start < ?"
        params = [window_start, window_end]
    query += " ORDER BY start"
    cur.execute(query, params)
    intervals = cur.fetchall()
    merged = _merge_intervals(intervals)
    return sum((e - s) / 1000.0 for s, e in merged), merged


def _compute_launch_overhead(conn, window_start=None, window_end=None):
    """Compute total cudaLaunchKernel time in us."""
    cur = conn.cursor()
    query = (
        "SELECT SUM(r.end - r.start) / 1000.0 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id "
        "WHERE s.value = 'cudaLaunchKernel'"
    )
    params = []
    if window_start is not None and window_end is not None:
        query += " AND r.start >= ? AND r.start < ?"
        params = [window_start, window_end]
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else 0.0


def _compute_nccl(conn, window_start=None, window_end=None):
    """Compute total NCCL kernel time in us."""
    cur = conn.cursor()
    query = (
        "SELECT SUM(k.end - k.start) / 1000.0 "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE s.value LIKE '%nccl%'"
    )
    params = []
    if window_start is not None and window_end is not None:
        query += " AND k.start >= ? AND k.start < ?"
        params = [window_start, window_end]
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else 0.0


def _compute_host_prep_exposed(conn, merged_kernel_intervals, window_start=None, window_end=None):
    """Compute M3 host-prep exposed time.

    Intersects NVTX host-prep ranges with GPU idle gaps.

    Returns (host_prep_total_us, host_prep_exposed_us) or (None, None)
    if no matching NVTX ranges found.
    """
    cur = conn.cursor()
    # Find host-prep NVTX ranges (try common names)
    prep_names = ["%_prepare_tp_inputs%", "%_prepare_inputs%", "%prepare_inputs%"]
    nvtx_ranges = []
    for pattern in prep_names:
        cur.execute(
            "SELECT n.start, n.end "
            "FROM NVTX_EVENTS n "
            "JOIN StringIds s ON n.textId = s.id "
            "WHERE n.end > 0 AND s.value LIKE ?"
            + (" AND n.start >= ? AND n.end <= ?" if window_start is not None else ""),
            ([pattern] + ([window_start, window_end] if window_start is not None else [])),
        )
        nvtx_ranges = cur.fetchall()
        if nvtx_ranges:
            break

    if not nvtx_ranges:
        return None, None

    host_prep_total_us = sum((e - s) / 1000.0 for s, e in nvtx_ranges)

    # Compute GPU idle gaps from merged kernel intervals
    idle_gaps = []
    if merged_kernel_intervals:
        for i in range(1, len(merged_kernel_intervals)):
            gap_start = merged_kernel_intervals[i - 1][1]
            gap_end = merged_kernel_intervals[i][0]
            if gap_end > gap_start:
                idle_gaps.append((gap_start, gap_end))

    # Intersect each NVTX range with idle gaps
    exposed_us = 0.0
    for nvtx_s, nvtx_e in nvtx_ranges:
        for gap_s, gap_e in idle_gaps:
            overlap_s = max(nvtx_s, gap_s)
            overlap_e = min(nvtx_e, gap_e)
            if overlap_e > overlap_s:
                exposed_us += (overlap_e - overlap_s) / 1000.0

    return host_prep_total_us, exposed_us


# ---------------------------------------------------------------------------
# Phase classification (NVTX-based)
# ---------------------------------------------------------------------------
_FWD_STEP_RE = re.compile(r"_forward_step\s+(\d+):\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs")


def _get_forward_steps(conn):
    """Get classified forward step NVTX ranges."""
    cur = conn.cursor()
    cur.execute(
        "SELECT n.start, n.end, s.value "
        "FROM NVTX_EVENTS n "
        "JOIN StringIds s ON n.textId = s.id "
        "WHERE n.end > 0 AND s.value LIKE '%[Executor] _forward_step%' "
        "ORDER BY n.start"
    )

    context_ranges = []
    generation_ranges = []
    for start, end, text in cur.fetchall():
        m = _FWD_STEP_RE.search(text)
        if not m:
            continue
        n_ctx = int(m.group(2))
        n_gen = int(m.group(3))
        if n_ctx > 0:
            context_ranges.append((start, end))
        elif n_gen > 0:
            generation_ranges.append((start, end))

    return context_ranges, generation_ranges


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def _apply_verdict(metrics, thresholds):
    """Apply threshold logic to a PhaseMetrics, return PhaseVerdict."""
    v = PhaseVerdict(metrics=metrics)
    crossed = {}

    # M1: GPU idle ratio
    t = thresholds.get("gpu_idle_ratio", DEFAULTS["gpu_idle_ratio"])
    c = metrics.gpu_idle_ratio > t
    crossed["gpu_idle_ratio"] = c
    v.applicable_count += 1
    if c:
        v.crossed_count += 1

    # M2: Launch overhead ratio
    t = thresholds.get("launch_overhead_ratio", DEFAULTS["launch_overhead_ratio"])
    c = metrics.launch_overhead_ratio > t
    crossed["launch_overhead_ratio"] = c
    v.applicable_count += 1
    if c:
        v.crossed_count += 1

    # M4: GPU utilization (below threshold = crossed)
    t = thresholds.get("gpu_utilization", DEFAULTS["gpu_utilization"])
    c = metrics.gpu_utilization < t
    crossed["gpu_utilization"] = c
    v.applicable_count += 1
    if c:
        v.crossed_count += 1

    # M3 sub-metrics (optional)
    if metrics.host_prep_exposed_ratio is not None:
        t = thresholds.get("host_prep_exposed_ratio", DEFAULTS["host_prep_exposed_ratio"])
        c = metrics.host_prep_exposed_ratio > t
        crossed["host_prep_exposed_ratio"] = c
        v.applicable_count += 1
        if c:
            v.crossed_count += 1

    if metrics.host_prep_perf_impact is not None:
        t = thresholds.get("host_prep_perf_impact", DEFAULTS["host_prep_perf_impact"])
        c3b = metrics.host_prep_perf_impact > t
        crossed["host_prep_perf_impact"] = c3b
        v.applicable_count += 1
        if c3b:
            v.crossed_count += 1
    else:
        c3b = False

    if metrics.host_prep_idle_attribution is not None:
        t = thresholds.get("host_prep_idle_attribution", DEFAULTS["host_prep_idle_attribution"])
        c3c = metrics.host_prep_idle_attribution > t
        crossed["host_prep_idle_attribution"] = c3c
        v.applicable_count += 1
        if c3c:
            v.crossed_count += 1
    else:
        c3c = False

    v.host_prep_confirmed = c3b and c3c

    # NCCL caveat
    t = thresholds.get("nccl_ratio_caveat", DEFAULTS["nccl_ratio_caveat"])
    v.nccl_caveat = metrics.nccl_ratio > t

    # Verdict: YES if >= 2 metrics crossed
    v.verdict = "YES" if v.crossed_count >= 2 else "NO"
    v.crossed_details = crossed

    return v


def _compute_phase_metrics(conn, window_start, window_end, total_time_us, compute_m3=True):
    """Compute all metrics for a given time window."""
    m = PhaseMetrics()
    m.total_time_us = total_time_us
    if total_time_us <= 0:
        return m

    gpu_active_us, merged = _compute_gpu_active(conn, window_start, window_end)
    m.gpu_active_us = gpu_active_us
    m.gpu_idle_us = total_time_us - gpu_active_us
    m.gpu_idle_ratio = m.gpu_idle_us / total_time_us
    m.gpu_utilization = gpu_active_us / total_time_us

    m.launch_overhead_us = _compute_launch_overhead(conn, window_start, window_end)
    m.launch_overhead_ratio = m.launch_overhead_us / total_time_us

    m.nccl_us = _compute_nccl(conn, window_start, window_end)
    m.nccl_ratio = (m.nccl_us / gpu_active_us) if gpu_active_us > 0 else 0.0

    if compute_m3:
        prep_total, prep_exposed = _compute_host_prep_exposed(
            conn, merged, window_start, window_end
        )
        if prep_total is not None and prep_total > 0:
            m.host_prep_total_us = prep_total
            m.host_prep_exposed_us = prep_exposed
            m.host_prep_exposed_ratio = prep_exposed / prep_total
            m.host_prep_perf_impact = prep_exposed / total_time_us
            m.host_prep_idle_attribution = (
                (prep_exposed / m.gpu_idle_us) if m.gpu_idle_us > 0 else 0.0
            )

    return m


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------
def detect(trace_path, thresholds=None, compute_m3=True):
    """Run Phase 1 Detection on a trace.

    Returns dict with aggregate_verdict, per-phase verdicts, and all metrics.
    """
    if thresholds is None:
        thresholds = dict(DEFAULTS)

    conn = sqlite3.connect(trace_path)
    try:
        ws, we, total_us = _get_analysis_window(conn)
        if ws is None:
            return {"error": "No GPU kernels found in trace"}

        # Aggregate metrics
        agg_metrics = _compute_phase_metrics(conn, ws, we, total_us, compute_m3)
        agg_verdict = _apply_verdict(agg_metrics, thresholds)

        # Per-phase metrics
        ctx_ranges, gen_ranges = _get_forward_steps(conn)

        phase_results = {}

        if ctx_ranges:
            ctx_start = min(s for s, _ in ctx_ranges)
            ctx_end = max(e for _, e in ctx_ranges)
            ctx_total = (ctx_end - ctx_start) / 1000.0
            ctx_metrics = _compute_phase_metrics(conn, ctx_start, ctx_end, ctx_total, compute_m3)
            ctx_metrics.iteration_count = len(ctx_ranges)
            ctx_thresholds = dict(thresholds)  # context uses default
            ctx_verdict = _apply_verdict(ctx_metrics, ctx_thresholds)
            phase_results["context"] = {
                "iteration_count": len(ctx_ranges),
                "verdict": ctx_verdict.verdict,
                "crossed_count": ctx_verdict.crossed_count,
                "applicable_count": ctx_verdict.applicable_count,
                "host_prep_confirmed": ctx_verdict.host_prep_confirmed,
                "metrics": _metrics_to_dict(ctx_metrics),
            }

        if gen_ranges:
            gen_start = min(s for s, _ in gen_ranges)
            gen_end = max(e for _, e in gen_ranges)
            gen_total = (gen_end - gen_start) / 1000.0
            gen_metrics = _compute_phase_metrics(conn, gen_start, gen_end, gen_total, compute_m3)
            gen_metrics.iteration_count = len(gen_ranges)
            gen_thresholds = dict(thresholds)
            gen_thresholds.update(GEN_OVERRIDES)
            gen_verdict = _apply_verdict(gen_metrics, gen_thresholds)
            phase_results["generation"] = {
                "iteration_count": len(gen_ranges),
                "verdict": gen_verdict.verdict,
                "crossed_count": gen_verdict.crossed_count,
                "applicable_count": gen_verdict.applicable_count,
                "host_prep_confirmed": gen_verdict.host_prep_confirmed,
                "metrics": _metrics_to_dict(gen_metrics),
            }

        # Overall verdict: any YES elevates to YES
        overall = (
            "YES"
            if (
                agg_verdict.verdict == "YES"
                or phase_results.get("context", {}).get("verdict") == "YES"
                or phase_results.get("generation", {}).get("verdict") == "YES"
            )
            else "NO"
        )

        return {
            "verdict": overall,
            "aggregate_metrics": _metrics_to_dict(agg_metrics),
            "crossed_count": agg_verdict.crossed_count,
            "applicable_count": agg_verdict.applicable_count,
            "host_prep_confirmed": agg_verdict.host_prep_confirmed,
            "nccl_caveat": agg_verdict.nccl_caveat,
            "crossed_details": agg_verdict.crossed_details,
            "phases": phase_results,
        }

    finally:
        conn.close()


def _metrics_to_dict(m):
    """Convert PhaseMetrics to a clean dict (drop zero-value optional fields)."""
    d = {
        "gpu_idle_ratio": round(m.gpu_idle_ratio, 4),
        "launch_overhead_ratio": round(m.launch_overhead_ratio, 4),
        "gpu_utilization": round(m.gpu_utilization, 4),
        "nccl_ratio": round(m.nccl_ratio, 4),
    }
    if m.host_prep_exposed_ratio is not None:
        d["host_prep_exposed_ratio"] = round(m.host_prep_exposed_ratio, 4)
        d["host_prep_perf_impact"] = round(m.host_prep_perf_impact, 4)
        d["host_prep_idle_attribution"] = round(m.host_prep_idle_attribution, 4)
    else:
        d["host_prep_exposed_ratio"] = None
        d["host_prep_perf_impact"] = None
        d["host_prep_idle_attribution"] = None
    return d


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
def _format_report(result):
    """Format detection result as a human-readable report."""
    lines = []
    lines.append(f"## Host Overhead Verdict: {result['verdict']}")
    lines.append("")

    # Aggregate
    lines.append("### Aggregate Evidence")
    lines.append("")
    am = result["aggregate_metrics"]
    lines.append("| Metric                          | Value  | Threshold | Crossed? |")
    lines.append("|---------------------------------|--------|-----------|----------|")

    def _row(name, val, threshold, above=True):
        if val is None:
            return f"| {name:<33s}| N/A    | {threshold:<9s} | N/A      |"
        pct = f"{val * 100:.1f}%"
        op = ">" if above else "<"
        crossed = (
            val > float(threshold.strip("%>< ")) / 100
            if above
            else val < float(threshold.strip("%>< ")) / 100
        )
        flag = "YES" if crossed else "NO"
        return f"| {name:<33s}| {pct:<6s} | {op}{threshold:<8s}| {flag:<8s} |"

    lines.append(_row("GPU idle ratio", am["gpu_idle_ratio"], "30%"))
    lines.append(_row("cudaLaunchKernel overhead ratio", am["launch_overhead_ratio"], "10%"))
    lines.append(_row("Host prep exposed ratio (3a)", am.get("host_prep_exposed_ratio"), "50%"))
    lines.append(_row("Host prep perf impact (3b)", am.get("host_prep_perf_impact"), "5%"))
    lines.append(
        _row("Host prep idle attribution (3c)", am.get("host_prep_idle_attribution"), "50%")
    )
    lines.append(_row("GPU utilization (time-based)", am["gpu_utilization"], "60%", above=False))

    lines.append("")
    lines.append(
        f"Metrics crossed: {result['crossed_count']} / {result['applicable_count']} applicable"
    )
    hpc = "YES" if result["host_prep_confirmed"] else "NO"
    lines.append(f"Host prep confirmed bottleneck: {hpc}")

    if result["nccl_caveat"]:
        lines.append("")
        lines.append(
            f"> **Caveat**: NCCL communication accounts for "
            f"{am['nccl_ratio'] * 100:.1f}% of GPU active time."
        )
        lines.append("> GPU idle gaps may be partially caused by communication stalls.")

    # Per-phase
    for phase_name in ["context", "generation"]:
        phase = result.get("phases", {}).get(phase_name)
        if not phase:
            continue
        lines.append("")
        title = phase_name.capitalize()
        lines.append(f"### {title} Phase ({phase['iteration_count']} iterations)")
        lines.append("")
        lines.append(
            f"Phase verdict: {phase['verdict']} "
            f"({phase['crossed_count']}/{phase['applicable_count']}"
            f" crossed)"
        )
        hpc = "YES" if phase.get("host_prep_confirmed") else "NO"
        lines.append(f"Host prep confirmed: {hpc}")

    # Next steps
    lines.append("")
    lines.append("### Next Steps")
    if result["verdict"] == "YES":
        lines.append("Use `perf-host-optimization` skill to profile and optimize.")
    else:
        lines.append("Host overhead is not the bottleneck.")
        lines.append("Use `nsight-compute-analysis` for kernel-level SOL% analysis.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Detection: is host overhead the bottleneck?"
    )
    parser.add_argument("--trace", required=True, help="Path to nsys SQLite trace")
    parser.add_argument("--output", "-o", help="Output file for JSON (default: stdout)")
    parser.add_argument(
        "--json-only", action="store_true", help="Output JSON only (no human-readable report)"
    )
    parser.add_argument(
        "--no-m3", action="store_true", help="Skip M3 computation (faster, SQL-only)"
    )
    parser.add_argument(
        "--gpu-idle-threshold",
        type=float,
        help=f"M1 threshold (default: {DEFAULTS['gpu_idle_ratio']})",
    )
    parser.add_argument(
        "--launch-threshold",
        type=float,
        help=f"M2 threshold (default: {DEFAULTS['launch_overhead_ratio']})",
    )
    parser.add_argument(
        "--gpu-util-threshold",
        type=float,
        help=f"M4 threshold (default: {DEFAULTS['gpu_utilization']})",
    )
    args = parser.parse_args()

    thresholds = dict(DEFAULTS)
    if args.gpu_idle_threshold is not None:
        thresholds["gpu_idle_ratio"] = args.gpu_idle_threshold
    if args.launch_threshold is not None:
        thresholds["launch_overhead_ratio"] = args.launch_threshold
    if args.gpu_util_threshold is not None:
        thresholds["gpu_utilization"] = args.gpu_util_threshold

    result = detect(args.trace, thresholds=thresholds, compute_m3=not args.no_m3)

    # Output JSON
    json_str = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"JSON written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Human-readable report (unless --json-only)
    if not args.json_only:
        print("\n" + _format_report(result), file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host overhead analysis for LLM inference loops using nsys SQLite traces.

Compares inter-iteration overhead between two versions by:
1. Isolating forward step iterations via allreduce kernel grouping
2. Comparing NVTX-instrumented host operations per step
3. Comparing GPU kernel profiles during steady-state generation
4. Identifying scheduling/request-management regressions

Usage:
    python analyze_host_overhead.py \
        --baseline /path/to/baseline/trace.sqlite \
        --target /path/to/target/trace.sqlite \
        --baseline-label "v1.1" \
        --target-label "main" \
        --output /path/to/output/analysis.txt

    # Single trace analysis (no comparison)
    python analyze_host_overhead.py \
        --baseline /path/to/trace.sqlite \
        --baseline-label "v1.1"

    # Mock mode for testing
    python analyze_host_overhead.py --mock
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass


@dataclass
class IterationInfo:
    """Information about a detected forward step iteration."""

    start: int  # nanoseconds
    end: int  # nanoseconds
    kernel_count: int
    wall_time_us: float


@dataclass
class StepInfo:
    """Information about a single [Executor] _forward_step NVTX range."""

    start: int
    end: int
    dur_us: float
    text: str
    ctx_reqs: int
    gen_reqs: int
    step_num: int


def get_trace_epoch(conn):
    """Get the earliest kernel timestamp as the trace epoch."""
    cur = conn.cursor()
    cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else 0


def detect_iterations_by_allreduce(conn, t0, gap_threshold_ns=1_000_000, phase_gap_ns=100_000_000):
    """Detect forward step iterations by grouping allreduce_fusion kernels.

    Args:
        conn: SQLite connection to nsys trace
        t0: Trace epoch (ns)
        gap_threshold_ns: Max gap between allreduce kernels in same iteration (1ms default)
        phase_gap_ns: Min gap between phases (100ms default)

    Returns:
        (iterations, common_size, phases)
        - iterations: list of IterationInfo
        - common_size: most common allreduce count per iteration
        - phases: list of (start_iter_idx, end_iter_idx) tuples
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT k.start, k.end "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE s.value LIKE '%allreduce%' "
        "ORDER BY k.start"
    )
    allreduce_events = cur.fetchall()

    if not allreduce_events:
        return [], 0, []

    # Group into iterations by gap threshold
    iterations = []
    group_start = allreduce_events[0][0]
    group_end = allreduce_events[0][1]
    group_count = 1

    for i in range(1, len(allreduce_events)):
        gap = allreduce_events[i][0] - allreduce_events[i - 1][1]
        if gap > gap_threshold_ns:
            iterations.append(
                IterationInfo(
                    start=group_start,
                    end=group_end,
                    kernel_count=group_count,
                    wall_time_us=(group_end - group_start) / 1000.0,
                )
            )
            group_start = allreduce_events[i][0]
            group_end = allreduce_events[i][1]
            group_count = 1
        else:
            group_end = max(group_end, allreduce_events[i][1])
            group_count += 1

    iterations.append(
        IterationInfo(
            start=group_start,
            end=group_end,
            kernel_count=group_count,
            wall_time_us=(group_end - group_start) / 1000.0,
        )
    )

    # Find most common iteration size
    size_counts = Counter(it.kernel_count for it in iterations)
    common_size = size_counts.most_common(1)[0][0]

    # Detect phases by large gaps between iterations
    phases = []
    phase_start = 0
    for i in range(1, len(iterations)):
        gap = iterations[i].start - iterations[i - 1].end
        if gap > phase_gap_ns:
            phases.append((phase_start, i - 1))
            phase_start = i
    phases.append((phase_start, len(iterations) - 1))

    return iterations, common_size, phases


def find_benchmark_phase(iterations, common_size, phases):
    """Find the benchmark phase: the last phase where most iterations have the common size.

    Returns (start_idx, end_idx) into iterations list, or None.
    """
    for start, end in reversed(phases):
        phase_iters = iterations[start : end + 1]
        common_count = sum(1 for it in phase_iters if it.kernel_count == common_size)
        if common_count >= len(phase_iters) * 0.5 and len(phase_iters) >= 5:
            return start, end
    return None


def detect_iterations_by_nvtx(conn, phase_gap_ns=100_000_000):
    """Fallback iteration detection using _forward_step NVTX ranges.

    Used when allreduce-based detection returns no results (e.g., TP=1).
    Each NVTX _forward_step range maps directly to one iteration.

    Returns (iterations, common_size, phases) with the same signature as
    detect_iterations_by_allreduce (common_size is always 1).
    """
    steps = get_forward_step_nvtx(conn)
    if not steps:
        return [], 0, []

    unique = dedup_tp_steps(steps)
    iterations = [
        IterationInfo(
            start=s.start,
            end=s.end,
            kernel_count=1,
            wall_time_us=s.dur_us,
        )
        for s in unique
    ]

    # Detect phases by large gaps
    phases = []
    phase_start = 0
    for i in range(1, len(iterations)):
        gap = iterations[i].start - iterations[i - 1].end
        if gap > phase_gap_ns:
            phases.append((phase_start, i - 1))
            phase_start = i
    phases.append((phase_start, len(iterations) - 1))

    return iterations, 1, phases


def get_forward_step_nvtx(conn, time_start=None, time_end=None):
    """Get [Executor] _forward_step NVTX ranges, optionally within a time window.

    Returns list of StepInfo with parsed ctx/gen request counts.
    """
    cur = conn.cursor()
    query = (
        "SELECT n.start, n.end, (n.end - n.start)/1000.0, s.value "
        "FROM NVTX_EVENTS n "
        "JOIN StringIds s ON n.textId = s.id "
        "WHERE n.end > 0 AND s.value LIKE '%[Executor] _forward_step%'"
    )
    params = []
    if time_start is not None and time_end is not None:
        query += " AND n.start >= ? AND n.end <= ?"
        params.extend([time_start, time_end])
    query += " ORDER BY n.start"
    cur.execute(query, params)

    _FWD_STEP_RE = re.compile(r"_forward_step\s+(\d+):\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs")

    steps = []
    for start, end, dur, text in cur.fetchall():
        ctx_reqs = 0
        gen_reqs = 0
        step_num = -1
        m = _FWD_STEP_RE.search(text)
        if m:
            step_num = int(m.group(1))
            ctx_reqs = int(m.group(2))
            gen_reqs = int(m.group(3))
        steps.append(StepInfo(start, end, dur, text, ctx_reqs, gen_reqs, step_num))

    return steps


def dedup_tp_steps(steps):
    """De-duplicate NVTX ranges from multiple TP ranks.

    Groups entries with the same step_num within 100us of each other,
    merging all ranks (works for any TP size, not just TP=2).
    """
    if not steps:
        return []
    unique = []
    i = 0
    while i < len(steps):
        group = [steps[i]]
        # Collect all entries from other TP ranks for the same step
        while (
            i + len(group) < len(steps)
            and abs(steps[i + len(group)].start - steps[i].start) < 100_000
            and steps[i + len(group)].step_num == steps[i].step_num
        ):
            group.append(steps[i + len(group)])
        # Merge the group: use min start, max end, max duration
        merged_start = min(s.start for s in group)
        merged_end = max(s.end for s in group)
        merged_dur = max(s.dur_us for s in group)
        unique.append(
            StepInfo(
                merged_start,
                merged_end,
                merged_dur,
                group[0].text,
                group[0].ctx_reqs,
                group[0].gen_reqs,
                group[0].step_num,
            )
        )
        i += len(group)
    return unique


def compute_stats(values):
    """Compute basic statistics for a list of values."""
    if not values:
        return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "count": 0}
    s = sorted(values)
    return {
        "avg": sum(s) / len(s),
        "min": s[0],
        "max": s[-1],
        "p50": s[len(s) // 2],
        "p90": s[int(len(s) * 0.9)],
        "count": len(s),
    }


def get_nvtx_breakdown(conn, time_start, time_end):
    """Get NVTX range breakdown during a time window."""
    cur = conn.cursor()
    cur.execute(
        "SELECT s.value, COUNT(*) AS cnt, "
        "SUM(n.end - n.start)/1000.0 AS total_us, "
        "AVG(n.end - n.start)/1000.0 AS avg_us "
        "FROM NVTX_EVENTS n "
        "JOIN StringIds s ON n.textId = s.id "
        "WHERE n.end > 0 AND n.start >= ? AND n.end <= ? "
        "AND s.value NOT LIKE '%[Executor]%' "
        "GROUP BY s.value ORDER BY total_us DESC LIMIT 30",
        (time_start, time_end),
    )
    return cur.fetchall()


def get_kernel_breakdown(conn, time_start, time_end):
    """Get GPU kernel breakdown during a time window."""
    cur = conn.cursor()
    cur.execute(
        "SELECT s.value, COUNT(*), SUM(k.end - k.start)/1000.0 AS total_us, "
        "AVG(k.end - k.start)/1000.0 AS avg_us "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE k.start >= ? AND k.start < ? "
        "GROUP BY s.value ORDER BY total_us DESC LIMIT 15",
        (time_start, time_end),
    )
    return cur.fetchall()


def get_cuda_api_breakdown(conn, time_start, time_end):
    """Get CUDA API call breakdown during a time window."""
    cur = conn.cursor()
    cur.execute(
        "SELECT s.value, COUNT(*), SUM(r.end - r.start)/1000.0 AS total_us "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id "
        "WHERE r.start >= ? AND r.start < ? "
        "GROUP BY s.value ORDER BY total_us DESC LIMIT 15",
        (time_start, time_end),
    )
    return cur.fetchall()


def analyze_single_trace(conn, label, out, tp_size=1):
    """Full analysis of a single trace. Returns analysis results dict."""
    t0 = get_trace_epoch(conn)
    results = {"label": label}

    out.write(f"\n{'=' * 80}\n")
    out.write(f"ANALYSIS: {label}\n")
    out.write(f"{'=' * 80}\n")

    # --- Iteration detection ---
    iterations, common_size, phases = detect_iterations_by_allreduce(conn, t0)
    iter_method = "allreduce"

    if not iterations:
        # Fallback: use NVTX _forward_step ranges (TP=1 or no allreduce)
        iterations, common_size, phases = detect_iterations_by_nvtx(conn)
        iter_method = "nvtx"

    out.write(f"\n--- Iteration Detection (method: {iter_method}) ---\n")
    out.write(f"  Total iterations: {len(iterations)}\n")
    if iter_method == "allreduce":
        out.write(f"  Most common iteration size: {common_size} allreduce kernels\n")
    out.write(f"  Phases detected: {len(phases)}\n")

    for i, (ps, pe) in enumerate(phases):
        phase_iters = iterations[ps : pe + 1]
        common_count = sum(1 for it in phase_iters if it.kernel_count == common_size)
        t_start = (phase_iters[0].start - t0) / 1e9
        t_end = (phase_iters[-1].end - t0) / 1e9
        out.write(
            f"    Phase {i}: iters {ps}-{pe} ({pe - ps + 1} iters), "
            f"t={t_start:.2f}s-{t_end:.2f}s, "
            f"{common_count}/{pe - ps + 1} have common size\n"
        )

    # Find benchmark phase
    bench = find_benchmark_phase(iterations, common_size, phases)
    if bench is None:
        out.write("\n  WARNING: Could not identify benchmark phase.\n")
        results["bench_found"] = False
        return results

    bench_start, bench_end = bench
    bench_iters = iterations[bench_start : bench_end + 1]
    results["bench_found"] = True
    results["bench_iter_count"] = len(bench_iters)

    wall_times = [it.wall_time_us for it in bench_iters]
    wall_stats = compute_stats(wall_times)
    results["wall_time_stats"] = wall_stats

    t_bench_start = (bench_iters[0].start - t0) / 1e9
    t_bench_end = (bench_iters[-1].end - t0) / 1e9
    out.write(f"\n  Benchmark phase: iters {bench_start}-{bench_end} ({len(bench_iters)} iters)\n")
    out.write(
        f"  Time range: t={t_bench_start:.4f}s - {t_bench_end:.4f}s "
        f"({t_bench_end - t_bench_start:.4f}s)\n"
    )
    out.write(f"  Avg iteration wall time: {wall_stats['avg']:.1f} us\n")
    out.write(f"  P50: {wall_stats['p50']:.1f} us, P90: {wall_stats['p90']:.1f} us\n")

    # Inter-iteration gaps
    gaps = []
    for i in range(1, len(bench_iters)):
        gap = (bench_iters[i].start - bench_iters[i - 1].end) / 1000.0
        gaps.append(gap)
    gap_stats = compute_stats(gaps)
    results["gap_stats"] = gap_stats
    out.write(
        f"\n  Inter-iteration gap: avg={gap_stats['avg']:.1f} us, "
        f"P50={gap_stats['p50']:.1f} us, P90={gap_stats['p90']:.1f} us\n"
    )

    # --- Steady-state NVTX analysis ---
    out.write("\n--- Steady-State Forward Step Analysis ---\n")

    # Find the max gen reqs to identify steady state
    all_steps = get_forward_step_nvtx(conn)
    if all_steps:
        max_gen = max(s.gen_reqs for s in all_steps)
        out.write(f"  Max generation requests observed: {max_gen}\n")

        # Filter to steady-state (0 ctx, max gen)
        steady_steps = [s for s in all_steps if s.ctx_reqs == 0 and s.gen_reqs == max_gen]
        out.write(f"  Steady-state steps (0 ctx, {max_gen} gen): {len(steady_steps)}\n")

        unique_steady = dedup_tp_steps(steady_steps)
        out.write(f"  Unique steady-state steps: {len(unique_steady)}\n")
        results["steady_step_count"] = len(unique_steady)

        if len(unique_steady) >= 5:
            step_durs = [s.dur_us for s in unique_steady]
            dur_stats = compute_stats(step_durs)
            results["step_dur_stats"] = dur_stats
            out.write(
                f"\n  Step duration: avg={dur_stats['avg']:.1f} us, "
                f"P50={dur_stats['p50']:.1f} us, P90={dur_stats['p90']:.1f} us\n"
            )

            step_gaps = []
            for i in range(1, len(unique_steady)):
                gap = (unique_steady[i].start - unique_steady[i - 1].end) / 1000.0
                step_gaps.append(gap)
            sgap_stats = compute_stats(step_gaps)
            results["step_gap_stats"] = sgap_stats
            out.write(
                f"  Inter-step gap: avg={sgap_stats['avg']:.1f} us, "
                f"P50={sgap_stats['p50']:.1f} us, P90={sgap_stats['p90']:.1f} us\n"
            )

            ss_start = unique_steady[0].start
            ss_end = unique_steady[-1].end
            ss_wall = (ss_end - ss_start) / 1000.0
            wall_per_step = ss_wall / len(unique_steady)
            results["ss_wall_per_step"] = wall_per_step
            out.write(f"\n  Steady-state wall time: {ss_wall:.1f} us ({ss_wall / 1e6:.4f}s)\n")
            out.write(f"  Wall time per step: {wall_per_step:.1f} us\n")

            # NVTX breakdown
            out.write("\n--- NVTX Breakdown (steady-state) ---\n")
            nvtx = get_nvtx_breakdown(conn, ss_start, ss_end)
            nvtx_per_step = {}
            for text, cnt, total, avg in nvtx:
                # Divide by tp_size (each TP rank reports independently)
                per_step = total / tp_size / len(unique_steady)
                nvtx_per_step[text] = per_step
                short = (text[:55] + "...") if text and len(text) > 55 else text
                out.write(
                    f"    {short:60s}: {per_step:8.1f} us/step  "
                    f"(total={total / tp_size:.0f}us, cnt={cnt // tp_size})\n"
                )
            results["nvtx_per_step"] = nvtx_per_step

            # GPU kernels
            out.write("\n--- GPU Kernels (steady-state) ---\n")
            kernels = get_kernel_breakdown(conn, ss_start, ss_end)
            total_gpu = sum(t for _, _, t, _ in kernels)
            results["gpu_per_step"] = total_gpu / len(unique_steady)
            kernel_count = sum(c for _, c, _, _ in kernels)
            results["kernels_per_step"] = kernel_count / len(unique_steady)
            for name, cnt, total, avg in kernels:
                name = str(name)
                short = (name[:55] + "...") if len(name) > 55 else name
                per_step = total / len(unique_steady)
                out.write(
                    f"    {short:60s}: {per_step:7.1f} us/step  (cnt={cnt}, total={total:.0f}us)\n"
                )
            out.write(f"  Total GPU per step: {results['gpu_per_step']:.1f} us\n")
            out.write(f"  Kernel launches per step: {results['kernels_per_step']:.1f}\n")

            # CUDA API
            out.write("\n--- CUDA API (steady-state) ---\n")
            apis = get_cuda_api_breakdown(conn, ss_start, ss_end)
            api_per_step = {}
            for name, cnt, total in apis:
                ps = total / len(unique_steady)
                api_per_step[name] = ps
                out.write(f"    {name:50s}: {ps:7.1f} us/step  (cnt={cnt})\n")
            results["api_per_step"] = api_per_step
    else:
        out.write("  No [Executor] _forward_step NVTX ranges found.\n")
        results["steady_step_count"] = 0

    return results


def compare_results(baseline, target, out):
    """Compare two trace analysis results and produce a diff report."""
    out.write(f"\n{'=' * 80}\n")
    out.write(f"COMPARISON: {baseline['label']} vs {target['label']}\n")
    out.write(f"{'=' * 80}\n")

    if not baseline.get("bench_found") or not target.get("bench_found"):
        out.write("\n  Cannot compare: benchmark phase not found in one or both traces.\n")
        return

    # --- Allreduce iteration comparison ---
    out.write("\n--- Allreduce Iteration Comparison ---\n")
    b_wall = baseline["wall_time_stats"]
    t_wall = target["wall_time_stats"]
    delta_wall = t_wall["avg"] - b_wall["avg"]
    pct_wall = (delta_wall / b_wall["avg"] * 100) if b_wall["avg"] else 0
    out.write(
        f"  Benchmark iterations: {baseline['bench_iter_count']} vs {target['bench_iter_count']}\n"
    )
    out.write(
        f"  Avg iteration wall time: {b_wall['avg']:.1f} vs {t_wall['avg']:.1f} us  "
        f"({delta_wall:+.1f} us, {pct_wall:+.1f}%)\n"
    )

    b_gap = baseline["gap_stats"]
    t_gap = target["gap_stats"]
    delta_gap = t_gap["avg"] - b_gap["avg"]
    pct_gap = (delta_gap / b_gap["avg"] * 100) if b_gap["avg"] else 0
    out.write(
        f"  Inter-iter gap avg: {b_gap['avg']:.1f} vs {t_gap['avg']:.1f} us  "
        f"({delta_gap:+.1f} us, {pct_gap:+.1f}%)\n"
    )
    out.write(f"  Inter-iter gap P50: {b_gap['p50']:.1f} vs {t_gap['p50']:.1f} us\n")

    # --- Steady-state comparison ---
    if baseline.get("ss_wall_per_step") and target.get("ss_wall_per_step"):
        out.write("\n--- Steady-State Per-Step Comparison ---\n")
        b_wps = baseline["ss_wall_per_step"]
        t_wps = target["ss_wall_per_step"]
        delta_wps = t_wps - b_wps
        pct_wps = (delta_wps / b_wps * 100) if b_wps else 0
        out.write(
            f"  Wall time per step: {b_wps:.1f} vs {t_wps:.1f} us  "
            f"({delta_wps:+.1f} us, {pct_wps:+.1f}%)\n"
        )

        b_dur = baseline["step_dur_stats"]
        t_dur = target["step_dur_stats"]
        delta_dur = t_dur["avg"] - b_dur["avg"]
        pct_dur = (delta_dur / b_dur["avg"] * 100) if b_dur["avg"] else 0
        out.write(
            f"  Step duration avg: {b_dur['avg']:.1f} vs {t_dur['avg']:.1f} us  "
            f"({delta_dur:+.1f} us, {pct_dur:+.1f}%)\n"
        )

        b_sgap = baseline["step_gap_stats"]
        t_sgap = target["step_gap_stats"]
        delta_sgap = t_sgap["p50"] - b_sgap["p50"]
        pct_sgap = (delta_sgap / b_sgap["p50"] * 100) if b_sgap["p50"] else 0
        out.write(
            f"  Inter-step gap P50: {b_sgap['p50']:.1f} vs {t_sgap['p50']:.1f} us  "
            f"({delta_sgap:+.1f} us, {pct_sgap:+.1f}%)\n"
        )

    # --- NVTX per-step comparison ---
    b_nvtx = baseline.get("nvtx_per_step", {})
    t_nvtx = target.get("nvtx_per_step", {})
    if b_nvtx or t_nvtx:
        out.write("\n--- NVTX Per-Step Comparison (steady-state) ---\n")
        all_ops = sorted(
            set(list(b_nvtx.keys()) + list(t_nvtx.keys())),
            key=lambda k: abs(t_nvtx.get(k, 0) - b_nvtx.get(k, 0)),
            reverse=True,
        )
        out.write(
            f"  {'Operation':<40s} {'Baseline':>10s} {'Target':>10s} "
            f"{'Delta':>10s} {'Delta%':>8s}  Status\n"
        )
        out.write(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}  ------\n")

        for op in all_ops[:20]:
            b_val = b_nvtx.get(op, 0)
            t_val = t_nvtx.get(op, 0)
            delta = t_val - b_val
            pct = (delta / b_val * 100) if b_val > 0 else float("inf")

            short_op = (op[:38] + "..") if len(op) > 40 else op

            if b_val == 0:
                status = "NEW"
            elif t_val == 0:
                status = "REMOVED"
            elif pct > 50:
                status = "REGRESSION"
            elif pct < -30:
                status = "IMPROVED"
            else:
                status = ""

            pct_str = f"{pct:+.0f}%" if b_val > 0 else "NEW"
            out.write(
                f"  {short_op:<40s} {b_val:10.1f} {t_val:10.1f} "
                f"{delta:+10.1f} {pct_str:>8s}  {status}\n"
            )

    # --- GPU comparison ---
    b_gpu = baseline.get("gpu_per_step", 0)
    t_gpu = target.get("gpu_per_step", 0)
    if b_gpu or t_gpu:
        out.write("\n--- GPU Per-Step Comparison ---\n")
        delta_gpu = t_gpu - b_gpu
        pct_gpu = (delta_gpu / b_gpu * 100) if b_gpu else 0
        out.write(
            f"  GPU time per step: {b_gpu:.1f} vs {t_gpu:.1f} us  "
            f"({delta_gpu:+.1f} us, {pct_gpu:+.1f}%)\n"
        )

        b_kps = baseline.get("kernels_per_step", 0)
        t_kps = target.get("kernels_per_step", 0)
        delta_kps = t_kps - b_kps
        out.write(f"  Kernels per step: {b_kps:.1f} vs {t_kps:.1f}  ({delta_kps:+.1f})\n")

    # --- Summary ---
    out.write("\n--- Summary ---\n")
    if baseline.get("ss_wall_per_step") and target.get("ss_wall_per_step"):
        wps_delta = target["ss_wall_per_step"] - baseline["ss_wall_per_step"]
        wps_pct = wps_delta / baseline["ss_wall_per_step"] * 100
        out.write(f"  Per-step wall time regression: {wps_delta:+.1f} us ({wps_pct:+.1f}%)\n")

        dur_delta = target["step_dur_stats"]["avg"] - baseline["step_dur_stats"]["avg"]
        gap_delta = target["step_gap_stats"]["p50"] - baseline["step_gap_stats"]["p50"]
        out.write(f"    Step duration contribution: {dur_delta:+.1f} us\n")
        out.write(f"    Inter-step gap contribution (P50): {gap_delta:+.1f} us\n")

        if gap_delta > dur_delta:
            out.write("\n  CONCLUSION: Regression is primarily in INTER-STEP HOST OVERHEAD.\n")
            out.write("  The gap between forward steps increased more than step execution.\n")
        else:
            out.write("\n  CONCLUSION: Regression is primarily in STEP EXECUTION TIME.\n")
            out.write("  Forward step duration increased more than inter-step gap.\n")

        # Top 3 regressing operations
        if b_nvtx and t_nvtx:
            regressions = []
            for op in all_ops:
                b_val = b_nvtx.get(op, 0)
                t_val = t_nvtx.get(op, 0)
                delta = t_val - b_val
                if delta > 0:
                    regressions.append((op, b_val, t_val, delta))
            regressions.sort(key=lambda x: -x[3])

            out.write("\n  Top regression sources (us/step):\n")
            for op, bv, tv, d in regressions[:5]:
                if bv > 0:
                    out.write(f"    {op}: {bv:.0f} -> {tv:.0f}  (+{d:.0f})\n")
                else:
                    out.write(f"    {op}: NEW  (+{d:.0f})\n")


def mock_output():
    """Return mock analysis data for testing."""
    return json.dumps(
        {
            "baseline": {
                "label": "v1.1",
                "ss_wall_per_step": 3317.4,
                "step_dur_avg": 1500.3,
                "inter_step_gap_p50": 2543.3,
                "nvtx_top_ops": {
                    "_sample_async": 1163.0,
                    "_process_requests": 1056.3,
                    "_prepare_inputs": 815.9,
                    "_update_requests": 412.6,
                    "_fetch_new_requests": 36.4,
                },
                "gpu_per_step": 109.8,
                "kernels_per_step": 6.2,
            },
            "target": {
                "label": "main",
                "ss_wall_per_step": 3977.7,
                "step_dur_avg": 1667.2,
                "inter_step_gap_p50": 4467.6,
                "nvtx_top_ops": {
                    "_prepare_inputs": 871.0,
                    "_update_requests": 722.8,
                    "_sample_async": 720.3,
                    "broadcast_requests": 249.6,
                    "_fetch_new_requests": 270.4,
                },
                "gpu_per_step": 142.8,
                "kernels_per_step": 21.9,
            },
            "regression_pct": 19.9,
            "primary_cause": "inter-step host overhead",
        },
        indent=2,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze host overhead in LLM inference loops from nsys traces"
    )
    parser.add_argument("--baseline", help="Path to baseline nsys sqlite trace")
    parser.add_argument("--target", help="Path to target nsys sqlite trace (optional)")
    parser.add_argument("--baseline-label", default="baseline", help="Label for baseline trace")
    parser.add_argument("--target-label", default="target", help="Label for target trace")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for NVTX deduplication (default: 1)",
    )
    parser.add_argument("--mock", action="store_true", help="Return mock data")
    args = parser.parse_args()

    if args.mock:
        print(mock_output())
        return

    if not args.baseline:
        parser.error("--baseline is required (or use --mock)")

    out_path = args.output or "(stdout)"
    out_file = open(args.output, "w") if args.output else None
    out = out_file if out_file is not None else sys.stdout

    try:
        # Analyze baseline
        conn_b = sqlite3.connect(args.baseline)
        baseline_results = analyze_single_trace(
            conn_b, args.baseline_label, out, tp_size=args.tp_size
        )
        conn_b.close()

        # Analyze target (if provided)
        target_results = None
        if args.target:
            conn_t = sqlite3.connect(args.target)
            target_results = analyze_single_trace(
                conn_t, args.target_label, out, tp_size=args.tp_size
            )
            conn_t.close()

            # Compare
            compare_results(baseline_results, target_results, out)
    finally:
        if out_file is not None:
            out_file.close()

    print(f"\nAnalysis complete. Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

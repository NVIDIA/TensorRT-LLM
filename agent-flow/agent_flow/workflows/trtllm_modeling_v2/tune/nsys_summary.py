#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step summary of an nsys capture of a TensorRT-LLM server (standard library only).

Reads the ``.sqlite`` export (``nsys export --type sqlite`` or ``nsys stats``,
which writes it as a side effect). Each executor step is an NVTX range
``[Executor] _forward_step N: C ctx reqs, T ctx tokens, G gen reqs``; kernels
are assigned to the step during whose host range they were *launched*
(runtime-API correlation), which is what a CUDA-graph replay or an eager chain
of a step actually is.

Per step class (decode / mixed / prefill) it reports: GPU busy time (union of
kernel intervals), the step's GPU span, idle inside the span, kernel count,
and time per kernel family; plus the host ranges the executor spends around
the forward (``_prepare_inputs``, ``_handle_responses``, sampling).

    python nsys_summary.py <capture>.sqlite [--top 15] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict

STEP = re.compile(r"_forward_step (\d+): (\d+) ctx reqs, (\d+) ctx tokens, (\d+) gen reqs")
FAMILIES = [
    ("norm", r"rmsnorm|layernorm|norm_kernel|flashinfernormkernels"),
    ("gemm", r"gemm|nvjet|cublas|splitk|dot_kernel|cutlass|xmma|gemv"),
    ("moe", r"moe|routing|expert|bmm_|fp4_block|finalize|permute"),
    ("attention", r"fmha|attention|mmha|xqa|qsa"),
    ("recurrence", r"gated_delta|delta_rul|recurrent|gdn|chunk_|gating_delta"),
    ("conv1d", r"conv1d"),
    ("quant", r"quantiz|fp4|fp8|e4m3|e2m1"),
    ("triton", r"triton|_kernel$"),
    (
        "elementwise",
        r"elementwise|vectorized|unrolled|reduce_kernel|index|copy|cat|fill|where|arange",
    ),
]


def family(name: str) -> str:
    low = name.lower()
    for fam, pattern in FAMILIES:
        if re.search(pattern, low):
            return fam
    return "other"


def union_us(intervals: list[tuple[int, int]]) -> float:
    total, end = 0, -1
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total / 1e3


def load(db: str):
    con = sqlite3.connect(db)
    cur = con.cursor()
    names = dict(cur.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    nvtx = []
    if "NVTX_EVENTS" in tables:
        for start, end, text, textid in cur.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        ):
            nvtx.append((start, end, text if text is not None else names.get(textid, "")))
    api = {}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for corr, start in cur.execute(
            "SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            api[corr] = start
    cols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    has_graph = {"graphId", "graphNodeId"} <= cols
    query = "SELECT start, end, correlationId, shortName" + (
        ", graphId, graphNodeId" if has_graph else ""
    )
    kernels, graph = [], defaultdict(list)
    for row in cur.execute(query + " FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        start, end, corr, nameid = row[:4]
        name = names.get(nameid, str(nameid))
        if has_graph and row[4] is not None:
            graph[row[4]].append((start, end, row[5], name))
        else:
            kernels.append((start, end, api.get(corr, start), name, False))
    # Kernels replayed from a CUDA graph carry no runtime correlation: group them
    # into replays (one chunk of distinct node ids per replay, by GPU start) and
    # place the k-th replay at the k-th cudaGraphLaunch -- the host time that puts
    # it inside an executor step. Attributing them by GPU start instead charges a
    # replay to whichever step the host was in when it ran (the overlap scheduler
    # runs the host one step ahead), which reads as idle that is not there.
    launches = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        launches = sorted(r[0] for r in cur.execute(
            "SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
            "WHERE s.value LIKE 'cudaGraphLaunch%'"))  # fmt: skip
    replays = []
    for ks in graph.values():
        ks.sort()
        n = len({k[2] for k in ks})
        for i in range(0, len(ks) - n + 1, n):
            chunk = ks[i : i + n]
            replays.append((chunk[0][0], chunk))
    replays.sort(key=lambda r: r[0])
    offset = max(0, len(launches) - len(replays))
    for j, (first, chunk) in enumerate(replays):
        when = launches[j + offset] if j + offset < len(launches) else first
        kernels.extend((k[0], k[1], when, k[3], True) for k in chunk)
    return nvtx, kernels


def summarize(db: str, top: int = 15) -> dict:
    nvtx, kernels = load(db)
    steps = []
    for start, end, text in nvtx:
        m = STEP.search(text or "")
        if m:
            n, ctx, ctx_tok, gen = map(int, m.groups())
            kind = "prefill" if gen == 0 else ("decode" if ctx == 0 else "mixed")
            steps.append(
                {
                    "n": n,
                    "host": (start, end),
                    "kind": kind,
                    "ctx_tokens": ctx_tok,
                    "gen": gen,
                    "k": [],
                }
            )
    steps.sort(key=lambda s: s["host"][0])
    starts = [s["host"][0] for s in steps]
    import bisect

    for k in kernels:
        i = bisect.bisect_right(starts, k[2]) - 1
        if 0 <= i < len(steps) and k[2] <= steps[i]["host"][1] + 50_000_000:
            steps[i]["k"].append(k)
    host = defaultdict(list)
    for start, end, text in nvtx:
        for key in (
            "_prepare_inputs",
            "_handle_responses",
            "_sample_async",
            "_update_requests",
            "_schedule",
            "_fetch_new_requests",
        ):
            if text and text.endswith(key):
                host[key].append((end - start) / 1e3)
    out = {
        "db": db,
        "classes": {},
        "host_us": {k: round(statistics.median(v), 1) for k, v in host.items() if v},
    }
    by_kind = defaultdict(list)
    for s in steps:
        if s["k"]:
            by_kind[s["kind"]].append(s)
    for kind, ss in by_kind.items():

        def core(step):
            # A step with a graph replay is judged on the replay: eager kernels
            # launched in the same host range run around another step's replay.
            g = [k for k in step["k"] if k[4]]
            return g or step["k"]

        busy = [union_us([(k[0], k[1]) for k in core(s)]) for s in ss]
        span = [(max(k[1] for k in core(s)) - min(k[0] for k in core(s))) / 1e3 for s in ss]
        eager = [
            sum((k[1] - k[0]) for k in s["k"] if not k[4]) / 1e3
            for s in ss
            if any(k[4] for k in s["k"])
        ]
        fam = defaultdict(float)
        names = defaultdict(lambda: [0.0, 0])
        for s in ss:
            for k in s["k"]:
                fam[family(k[3])] += (k[1] - k[0]) / 1e3
                names[k[3]][0] += (k[1] - k[0]) / 1e3
                names[k[3]][1] += 1
        n = len(ss)
        out["classes"][kind] = {
            "steps": n,
            "gen_reqs_median": statistics.median(s["gen"] for s in ss),
            "ctx_tokens_median": statistics.median(s["ctx_tokens"] for s in ss),
            "gpu_busy_us": round(statistics.median(busy), 1),
            "gpu_span_us": round(statistics.median(span), 1),
            "idle_in_span_pct": round(
                100 * (1 - statistics.median(busy) / max(statistics.median(span), 1e-9)), 1
            ),
            "kernels_per_step": round(sum(len(s["k"]) for s in ss) / n, 1),
            "graph_replay": any(k[4] for s in ss for k in s["k"]),
            "eager_us_beside_replay": round(statistics.median(eager), 1) if eager else 0.0,
            "family_us_per_step": {
                f: round(t / n, 1) for f, t in sorted(fam.items(), key=lambda x: -x[1])
            },
            "top_kernels": [
                {
                    "name": nm[:120],
                    "us_per_step": round(t / n, 1),
                    "calls_per_step": round(c / n, 1),
                }
                for nm, (t, c) in sorted(names.items(), key=lambda x: -x[1][0])[:top]
            ],
        }
    return out


def render(summary: dict) -> str:
    lines = [f"# nsys summary: {summary['db']}", ""]
    if summary["host_us"]:
        lines.append(
            "host ranges (median us): "
            + ", ".join(f"{k} {v}" for k, v in summary["host_us"].items())
        )
    for kind, c in summary["classes"].items():
        lines += [
            "",
            f"## {kind}: {c['steps']} steps, gen reqs ~{c['gen_reqs_median']}, ctx tokens ~{c['ctx_tokens_median']}",
            f"GPU busy {c['gpu_busy_us']} us, span {c['gpu_span_us']} us, idle in span {c['idle_in_span_pct']}%, "
            f"{c['kernels_per_step']} kernels/step"
            + (
                f" (graph replay; {c['eager_us_beside_replay']} us of eager kernels beside it)"
                if c.get("graph_replay")
                else ""
            ),
            "",
            "| family | us/step |",
            "|---|---|",
        ]
        lines += [f"| {f} | {t} |" for f, t in c["family_us_per_step"].items()]
        lines += ["", "| kernel | us/step | calls/step |", "|---|---|---|"]
        lines += [
            f"| {k['name']} | {k['us_per_step']} | {k['calls_per_step']} |"
            for k in c["top_kernels"]
        ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sqlite")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    summary = summarize(args.sqlite, args.top)
    print(render(summary))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

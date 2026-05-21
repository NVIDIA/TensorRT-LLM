#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse a TRT-LLM server log produced with `print_iter_log: true` and emit
per-iteration timing breakdowns.

For every iteration line that looks like:

    [05/20/2026-23:24:41] [TRT-LLM] [I] iter = 1, global_rank = 0, rank = 0,
        currank_total_requests = 0/1,
        host_step_time = 45810.21428108215ms,
        prev_device_step_time = N/A,
        timestamp = 2026-05-20 23:24:41,
        num_scheduled_requests: 1,
        states = {'num_ctx_requests': 1, 'num_ctx_tokens': 13694,
                  'num_generation_tokens': 0}

this script extracts the timing fields, classifies each iter as `ctx` /
`gen` / `idle`, splits the log into runs separated by long idle gaps, and
prints per-run summary statistics plus a per-iter table for the longest
ctx iterations.

Designed for the conc=1 32-prompt benchmark in this repro dir but works on
any TRT-LLM PyTorch server log.

Usage:
    python3 scripts/parse_iter_log.py logs/agg.log
    python3 scripts/parse_iter_log.py logs/agg.log --top-ctx 5 --first-run-only
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ITER_RE = re.compile(
    r"iter\s*=\s*(?P<iter>\d+),.*?"
    r"host_step_time\s*=\s*(?P<host>[\d.]+|N/A)ms,.*?"
    r"prev_device_step_time\s*=\s*(?P<prev_dev>[\d.]+|N/A)ms,.*?"
    r"timestamp\s*=\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),.*?"
    r"num_scheduled_requests:\s*(?P<scheduled>\d+),.*?"
    r"states\s*=\s*\{(?P<states>[^}]+)\}", re.DOTALL)

CTX_REQ_RE = re.compile(r"'num_ctx_requests':\s*(\d+)")
CTX_TOK_RE = re.compile(r"'num_ctx_tokens':\s*(\d+)")
GEN_TOK_RE = re.compile(r"'num_generation_tokens':\s*(\d+)")


@dataclass
class Iter:
    iter: int
    host_ms: Optional[float]
    prev_device_ms: Optional[float]
    ts: str
    scheduled: int
    num_ctx_requests: int
    num_ctx_tokens: int
    num_gen_tokens: int

    @property
    def phase(self) -> str:
        if self.num_ctx_tokens > 0:
            return "ctx"
        if self.num_gen_tokens > 0:
            return "gen"
        if self.scheduled == 0:
            return "idle"
        return "other"


@dataclass
class Run:
    """One contiguous chunk of activity (no long idle gap)."""
    iters: list[Iter] = field(default_factory=list)

    def gen_iters(self) -> list[Iter]:
        return [i for i in self.iters if i.phase == "gen"]

    def ctx_iters(self) -> list[Iter]:
        return [i for i in self.iters if i.phase == "ctx"]


def _parse_float(s: str) -> Optional[float]:
    if s == "N/A" or s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse(path: Path) -> list[Iter]:
    text = path.read_text(errors="replace")
    out: list[Iter] = []
    for m in ITER_RE.finditer(text):
        s = m.group("states")
        ctx_req = int(CTX_REQ_RE.search(s).group(1)) if CTX_REQ_RE.search(s) else 0
        ctx_tok = int(CTX_TOK_RE.search(s).group(1)) if CTX_TOK_RE.search(s) else 0
        gen_tok = int(GEN_TOK_RE.search(s).group(1)) if GEN_TOK_RE.search(s) else 0
        out.append(
            Iter(
                iter=int(m.group("iter")),
                host_ms=_parse_float(m.group("host")),
                prev_device_ms=_parse_float(m.group("prev_dev")),
                ts=m.group("ts"),
                scheduled=int(m.group("scheduled")),
                num_ctx_requests=ctx_req,
                num_ctx_tokens=ctx_tok,
                num_gen_tokens=gen_tok,
            ))
    return out


def split_bursts(iters: list[Iter]) -> list[Run]:
    """Split when the iter counter resets (engine warmup vs benchmark)."""
    runs: list[Run] = []
    cur = Run()
    last_iter_num = 0
    for it in iters:
        if cur.iters and it.iter <= last_iter_num:
            runs.append(cur)
            cur = Run()
        cur.iters.append(it)
        last_iter_num = it.iter
    if cur.iters:
        runs.append(cur)
    return runs


def split_requests(iters: list[Iter]) -> list[Run]:
    """Group consecutive iters into per-request runs.

    Heuristic: a new request starts at every ctx iter, OR when scheduled drops
    to 0 in between. Works for conc=1 with one ctx chunk per request.
    """
    runs: list[Run] = []
    cur = Run()
    prev_phase = "none"
    for it in iters:
        # New request signaled by entering ctx after gen/idle.
        if it.phase == "ctx" and prev_phase in ("gen", "idle") and cur.iters:
            runs.append(cur)
            cur = Run()
        cur.iters.append(it)
        prev_phase = it.phase
    if cur.iters:
        runs.append(cur)
    return runs


# ---------- formatting ----------

def _pct(values: list[float], pct: float) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    clean.sort()
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * pct / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return clean[int(k)]
    return clean[f] + (clean[c] - clean[f]) * (k - f)


def _stats_line(label: str, values: list[float]) -> str:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return f"  {label}: n=0"
    return (f"  {label}: n={len(clean):>4}  "
            f"min={min(clean):>8.2f}  "
            f"p50={_pct(clean, 50):>8.2f}  "
            f"mean={statistics.fmean(clean):>8.2f}  "
            f"p90={_pct(clean, 90):>8.2f}  "
            f"p99={_pct(clean, 99):>8.2f}  "
            f"max={max(clean):>10.2f}")


def summarize_run(run: Run, run_idx: int, top_ctx: int) -> None:
    ctx = run.ctx_iters()
    gen = run.gen_iters()
    if not run.iters:
        return
    first_ts = run.iters[0].ts
    last_ts = run.iters[-1].ts
    print(f"\n## Run #{run_idx} (iters {run.iters[0].iter}..{run.iters[-1].iter}, "
          f"{len(run.iters)} total, {len(ctx)} ctx + {len(gen)} gen)")
    print(f"   first ts {first_ts}  -> last ts {last_ts}")

    if ctx:
        print(" CTX iters (host_step_time ms):")
        print(_stats_line("host_step_time", [i.host_ms for i in ctx]))
        print(_stats_line("prev_device_step_time", [i.prev_device_ms for i in ctx]))
        print(_stats_line("num_ctx_tokens",
                          [float(i.num_ctx_tokens) for i in ctx]))
        ctx_sorted = sorted(ctx,
                            key=lambda i: (i.host_ms if i.host_ms else 0),
                            reverse=True)[:top_ctx]
        print(f"\n   top {len(ctx_sorted)} slowest ctx iters by host_step_time:")
        print(f"   {'iter':>6} {'ts':>20} {'host ms':>12} {'gpu ms':>12} "
              f"{'ctx_toks':>10}")
        for i in ctx_sorted:
            print(f"   {i.iter:>6} {i.ts:>20} "
                  f"{(i.host_ms or 0):>12.3f} "
                  f"{(i.prev_device_ms if i.prev_device_ms is not None else float('nan')):>12.3f} "
                  f"{i.num_ctx_tokens:>10}")
    if gen:
        # First gen iter often picks up the GPU time of the LAST ctx iter via
        # prev_device_step_time, so split into "first gen" (per-burst) vs rest.
        # Identify "first gen after ctx" by walking the burst.
        first_gens: list[Iter] = []
        steady_gens: list[Iter] = []
        prev_phase = "none"
        for it in run.iters:
            if it.phase == "gen":
                if prev_phase == "ctx":
                    first_gens.append(it)
                else:
                    steady_gens.append(it)
            prev_phase = it.phase
        print("\n GEN iters (all):")
        print(_stats_line("host_step_time", [i.host_ms for i in gen]))
        print(_stats_line("prev_device_step_time",
                          [i.prev_device_ms for i in gen]))
        if first_gens and steady_gens:
            print(" GEN iters (steady-state, excluding first-after-ctx):")
            print(
                _stats_line("host_step_time",
                            [i.host_ms for i in steady_gens]))
            print(
                _stats_line("prev_device_step_time",
                            [i.prev_device_ms for i in steady_gens]))
            print(" GEN iters (first-after-ctx only; carries ctx GPU time):")
            print(
                _stats_line("host_step_time",
                            [i.host_ms for i in first_gens]))
            print(
                _stats_line("prev_device_step_time",
                            [i.prev_device_ms for i in first_gens]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_path", type=Path)
    p.add_argument("--top-ctx",
                   type=int,
                   default=8,
                   help="Show top-N slowest ctx iters per run.")
    p.add_argument("--first-run-only",
                   action="store_true",
                   help="Only summarize the very first run (debug).")
    p.add_argument(
        "--skip-warmup-runs",
        type=int,
        default=2,
        help="Number of initial bursts (iter-counter resets) to skip. "
        "Default 2 covers TRT-LLM engine warmup before the benchmark run.")
    p.add_argument(
        "--skip-warmup-requests",
        type=int,
        default=2,
        help="After burst splitting, drop the first N per-request runs as "
        "warmup. Default 2: benchmark_serving's initial-test-run plus the "
        "first measured request (which often hits cached blocks).")
    p.add_argument(
        "--per-request",
        type=int,
        default=3,
        help="Print a per-request waterfall for the first N kept requests. "
        "0 disables.")
    args = p.parse_args()

    if not args.log_path.is_file():
        print(f"ERROR: {args.log_path} not found", file=sys.stderr)
        return 1

    iters = parse(args.log_path)
    print(f"# Parsed {len(iters)} iter lines from {args.log_path}")
    bursts = split_bursts(iters)
    print(f"# Detected {len(bursts)} engine burst(s) (iter-counter resets)")
    if args.first_run_only:
        bursts = bursts[:1]
    else:
        bursts = bursts[args.skip_warmup_runs:]
        print(f"# Skipped {args.skip_warmup_runs} warmup burst(s); "
              f"{len(bursts)} remain")
    # Within each burst, split into per-request runs (ctx-iter boundaries).
    flat_iters: list[Iter] = []
    for b in bursts:
        flat_iters.extend(b.iters)
    requests = split_requests(flat_iters)
    print(f"# Identified {len(requests)} per-request run(s) in remaining bursts")
    drop = args.skip_warmup_requests
    if drop > 0 and len(requests) > drop:
        print(f"# Dropping first {drop} request(s) as warmup before per-request stats")
        requests = requests[drop:]

    # Aggregate across all kept requests.
    all_ctx = [i for r in requests for i in r.ctx_iters()]
    all_gen = [i for r in requests for i in r.gen_iters()]
    print(f"\n## Aggregate across {len(requests)} requests")
    print(f"   ctx iters total: {len(all_ctx)}    gen iters total: {len(all_gen)}")
    if all_ctx:
        print(" CTX iters (host = scheduler+launch CPU time, prev_device = GPU iter time):")
        print(_stats_line("host_step_time ms ", [i.host_ms for i in all_ctx]))
        print(_stats_line("prev_device  ms   ", [i.prev_device_ms for i in all_ctx]))
        print(_stats_line("num_ctx_tokens    ",
                          [float(i.num_ctx_tokens) for i in all_ctx]))
    if all_gen:
        # Separate first-gen (carries ctx GPU time) vs steady-state.
        first_gens: list[Iter] = []
        steady_gens: list[Iter] = []
        for r in requests:
            prev_phase = "none"
            for it in r.iters:
                if it.phase == "gen":
                    if prev_phase == "ctx":
                        first_gens.append(it)
                    else:
                        steady_gens.append(it)
                prev_phase = it.phase
        if steady_gens:
            print("\n GEN iters (steady-state only; excludes first-after-ctx that"
                  " carries ctx GPU time):")
            print(_stats_line("host_step_time ms ", [i.host_ms for i in steady_gens]))
            print(_stats_line("prev_device  ms   ",
                              [i.prev_device_ms for i in steady_gens]))
        if first_gens:
            print("\n GEN iters (first-after-ctx; prev_device_step_time is the"
                  " context GPU work):")
            print(_stats_line("host_step_time ms ", [i.host_ms for i in first_gens]))
            print(_stats_line("prev_device  ms   ",
                              [i.prev_device_ms for i in first_gens]))

    # Optionally show a few per-request waterfalls.
    if args.per_request > 0:
        print(f"\n## Per-request waterfall for first {args.per_request} kept request(s):")
        for idx, r in enumerate(requests[:args.per_request]):
            ctx = r.ctx_iters()
            gen = r.gen_iters()
            print(f"\n   request #{idx}: {len(ctx)} ctx + {len(gen)} gen iters")
            if ctx:
                c0 = ctx[0]
                print(f"     ctx iter {c0.iter}: host={c0.host_ms:.2f}ms"
                      f"  prev_dev={c0.prev_device_ms if c0.prev_device_ms else float('nan'):.2f}ms"
                      f"  ctx_toks={c0.num_ctx_tokens}")
            if gen:
                # GPU time for ctx = prev_device of iter that is right after ctx.
                first_gen = next((i for i in r.iters if i.phase == "gen"), None)
                if first_gen and first_gen.prev_device_ms is not None:
                    print(f"     ctx GPU work (=prev_device of first-gen iter "
                          f"{first_gen.iter}): {first_gen.prev_device_ms:.2f}ms")
                steady = [i for i in gen if i is not first_gen]
                if steady:
                    sh = [i.host_ms for i in steady if i.host_ms is not None]
                    sd = [i.prev_device_ms for i in steady
                          if i.prev_device_ms is not None]
                    if sh:
                        print(f"     gen p50 host={_pct(sh, 50):.3f}ms  "
                              f"mean host={statistics.fmean(sh):.3f}ms")
                    if sd:
                        print(f"     gen p50 dev ={_pct(sd, 50):.3f}ms  "
                              f"mean dev ={statistics.fmean(sd):.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

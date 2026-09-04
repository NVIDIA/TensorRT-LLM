# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fit the equal-cost context chunking parameters from worker iteration logs.

Equal-cost context chunking (KV cache manager V2 scheduler, enabled via
TLLM_V2_CTX_COST_KV_OFFSET / TLLM_V2_CTX_COST_KV_DEPTH_THRESHOLD) needs two numbers:

  kv_cost_offset   kv-token equivalents of the KV-independent per-token work
          (a model x kernel x precision property -- measure it, don't guess)
  kv_depth_threshold  the KV depth above which chunks start to shrink
          (a workload property -- a low percentile of the depth distribution)

This script derives both from the per-rank iteration log of an ordinary
token-budget run of the SAME deployment and workload. Enable the log with
TLLM_PROFILE_LOG_RANKS=all on the context workers and run for ~30 minutes;
then:

    python fit_context_cost.py 3_output_CTX_0.log [more logs ...]

Method (robust to eviction/onboard stall outliers):
  1. Find "monopoly spans": consecutive iterations where one rank runs a
     single context request at the full token budget with its KV depth
     advancing chunk by chunk, and that rank's depth dominates all other
     ranks (so the lockstep iteration time is attributable to it).
  2. Theil-Sen within spans: the median of pairwise slopes of iteration
     time vs KV depth gives c2 (time per token*kv); the depth term cancels
     the intercept, and the median rejects stall outliers.
  3. The intercept A = median(T - slope*kv) over deep samples folds the
     KV-independent cost; kv_cost_offset = A / (c2 * max_num_tokens), reported as a
     range for an assumed 0..35% fixed-overhead share of A.
  4. kv_depth_threshold = a percentile of the token-weighted KV-depth distribution over
     all scheduled work (default p50).

The report also includes a piecewise-linearity check, the measured lockstep
waste (share of DP GPU time spent waiting for the straggler rank), and the
predicted equal-cost throughput gain.
"""

import argparse
import re
import statistics as st
import sys

LINE_RE = re.compile(
    r"\[RANK (\d+)\] iter = (\d+),.*?host_step_time = ([0-9.]+)ms, "
    r"prev_device_step_time = (N/A|[0-9.]+m?s?),.*?"
    r"'num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+), "
    r"'num_generation_tokens': (\d+), 'cached_kv_tokens': (\d+)")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else 0


def parse_logs(paths):
    """(iter, rank) -> (device_ms_of_prev_iter, nreq, ctx_tokens, cached_kv)."""
    recs = {}
    for path in paths:
        with open(path, errors="replace") as f:
            for line in f:
                m = LINE_RE.search(line)
                if not m:
                    continue
                dev = m.group(4)
                recs[(int(m.group(2)), int(m.group(1)))] = (
                    None if dev == "N/A" else float(dev.rstrip("ms")),
                    int(m.group(5)),
                    int(m.group(6)),
                    int(m.group(8)),
                )
    return recs


def iteration_times(recs, iters, ranks):
    """Median-across-ranks device time per iteration (lockstep: all equal)."""
    devt = {}
    for it in iters:
        vals = [
            recs[(it + 1, r)][0] for r in ranks
            if (it + 1, r) in recs and recs[(it + 1, r)][0]
        ]
        if len(vals) >= max(1, len(ranks) // 2):
            devt[it] = st.median(vals)
    return devt


def find_monopoly_spans(recs, iters, ranks, devt, max_num_tokens, min_span):
    spans = []
    for r in ranks:
        run = []
        for it in iters:
            rec = recs.get((it, r))
            nxt = recs.get((it + 1, r))
            ok = (rec and nxt and it in devt and rec[1] == 1
                  and rec[2] == max_num_tokens and nxt[1] == 1
                  and nxt[3] == rec[3] + max_num_tokens)
            if ok:
                others = [
                    recs[(it, o)][3] / max(recs[(it, o)][1], 1) for o in ranks
                    if o != r and (it, o) in recs
                ]
                ok = bool(others) and rec[3] > 1.5 * max(others)
            if ok:
                run.append((rec[3], devt[it]))
            else:
                if len(run) >= min_span:
                    spans.append(run)
                run = []
        if len(run) >= min_span:
            spans.append(run)
    return spans


def theil_sen(spans, pair_gap=3):
    slopes = []
    for span in spans:
        pair_slopes = []
        for i in range(len(span)):
            for j in range(i + pair_gap, len(span)):
                dkv = span[j][0] - span[i][0]
                if dkv > 0:
                    pair_slopes.append((span[j][1] - span[i][1]) / dkv)
        if pair_slopes:
            slopes.append(st.median(pair_slopes))
    return st.median(slopes) if slopes else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="context worker log file(s)")
    ap.add_argument("--max-num-tokens", type=int, default=8192)
    ap.add_argument("--kv-depth-threshold-percentile", type=float, default=50.0)
    ap.add_argument("--min-span", type=int, default=6,
                    help="minimum monopoly-span length in iterations")
    ap.add_argument("--warmup-iters", type=int, default=5,
                    help="skip the first N iterations")
    args = ap.parse_args()
    m = args.max_num_tokens

    recs = parse_logs(args.logs)
    if not recs:
        sys.exit("no iteration-log lines found -- was TLLM_PROFILE_LOG_RANKS=all set?")
    iters = sorted({k[0] for k in recs})
    ranks = sorted({k[1] for k in recs})
    devt = iteration_times(recs, iters, ranks)
    print(f"parsed {len(recs)} records: iters {iters[0]}..{iters[-1]}, "
          f"{len(ranks)} ranks, {len(devt)} timed iterations")

    # ---- kv_cost_offset ----
    spans = find_monopoly_spans(recs, iters, ranks, devt, m, args.min_span)
    n_samples = sum(len(s) for s in spans)
    if not spans:
        sys.exit("no monopoly spans found -- run longer, or the workload has "
                 "no deep single-request stretches to attribute time to")
    kv_lo = min(s[0][0] for s in spans)
    kv_hi = max(s[-1][0] for s in spans)
    print(f"\nmonopoly spans: {len(spans)} ({n_samples} samples), "
          f"kv {kv_lo / 1e3:.0f}k..{kv_hi / 1e3:.0f}k")

    slope = theil_sen(spans)  # ms per kv token at N = max_num_tokens
    c2 = slope / m            # ms per token*kv
    flat = [p for s in spans for p in s]
    deep = [t - slope * kv for kv, t in flat if kv > 200_000] or \
           [t - slope * kv for kv, t in flat]
    a = st.median(deep)       # c0 + c1 * max_num_tokens
    offset_hi = a / (c2 * m)
    offset_lo = 0.65 * offset_hi  # if fixed overhead is 35% of A
    kv_cost_offset = round((offset_lo + offset_hi) / 2, -3)
    print(f"slope = {slope * 1e3:.2f} us/kv @N={m}  ->  c2 = {c2 * 1e6:.3f} ns/(token*kv)")
    print(f"A = c0 + c1*{m} = {a:.0f} ms")
    print(f"kv_cost_offset in [{offset_lo / 1e3:.0f}k, {offset_hi / 1e3:.0f}k]  ->  recommended {kv_cost_offset:.0f}")

    # linearity diagnostic: per-depth-band slopes should agree
    print("linearity check (per-band Theil-Sen slope, us/kv):")
    for lo, hi in ((0, 200_000), (200_000, 500_000), (500_000, 1_050_000)):
        band = [[p for p in s if lo <= p[0] < hi] for s in spans]
        band = [b for b in band if len(b) >= args.min_span]
        s_band = theil_sen(band) if band else None
        print(f"  kv {lo // 1000:>4}k-{hi // 1000:<5}k: "
              f"{f'{s_band * 1e3:.2f}' if s_band else 'insufficient data'}")

    # ---- kv_depth_threshold ----
    kvbar = []
    for it in iters:
        if it < args.warmup_iters:
            continue
        for r in ranks:
            rec = recs.get((it, r))
            if rec and rec[2] > 0:
                kvbar.append(rec[3] / max(rec[1], 1) + rec[2] / 2)
    kv_depth_threshold = round(pct(kvbar, args.kv_depth_threshold_percentile), -3)
    print(f"\ntoken-weighted KV-depth percentiles (k): "
          f"p10={pct(kvbar, 10) / 1e3:.0f} p25={pct(kvbar, 25) / 1e3:.0f} "
          f"p50={pct(kvbar, 50) / 1e3:.0f} p90={pct(kvbar, 90) / 1e3:.0f}")
    print(f"kv_depth_threshold (p{args.kv_depth_threshold_percentile:.0f}) -> recommended {kv_depth_threshold:.0f}")

    # ---- expected benefit ----
    waste, gains = [], []
    for it in iters:
        if it < args.warmup_iters:
            continue
        feats = [recs[(it, r)] for r in ranks if (it, r) in recs]
        if len(feats) < len(ranks):
            continue
        costs = [
            a * f[2] / m + c2 * f[2] * (f[3] / max(f[1], 1) + f[2] / 2)
            for f in feats if f[2] > 0
        ]
        if len(costs) < len(ranks):
            continue
        waste.append(1 - st.mean(costs) / max(costs))
        depths = [f[3] / max(f[1], 1) + f[2] / 2 for f in feats]
        gains.append(
            sum(1 / (kv_cost_offset + d) for d in depths) /
            (len(depths) / (kv_cost_offset + max(depths))))
    if waste:
        print(f"\nlockstep waste (1 - mean/max of per-rank cost): "
              f"mean={st.mean(waste) * 100:.1f}%  p90={pct(waste, 90) * 100:.1f}%")
        print(f"predicted equal-cost throughput gain (saturated bound): "
              f"p50={pct(gains, 50):.2f}x")

    print("\nready to paste (context workers):")
    print(f"  TLLM_V2_CTX_COST_KV_OFFSET={kv_cost_offset:.0f}")
    print(f"  TLLM_V2_CTX_COST_KV_DEPTH_THRESHOLD={kv_depth_threshold:.0f}")


if __name__ == "__main__":
    main()

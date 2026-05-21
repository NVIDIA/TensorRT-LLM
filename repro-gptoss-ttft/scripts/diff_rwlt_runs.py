#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join two RWLT runs on (conversation_id, conversation_idx) and compute per-turn deltas.

Use to verify two layouts (agg vs disagg, or two configs) saw the same requests
and to break down the TTFT delta by whether the request was a cold-prefix turn
(low cached_tokens ratio) or a warm-prefix turn (high cached_tokens ratio).

Usage:
    python3 scripts/diff_rwlt_runs.py \
        rwlt-results/agg/rwlt_requests.jsonl \
        rwlt-results/disagg/rwlt_requests.jsonl \
        --label-a agg --label-b disagg

Outputs:
    - Coverage check (matched/unmatched turns, ISL agreement)
    - Per-turn TTFT delta summary (min/p50/mean/p90/p99/max)
    - Split by cold-prefix vs warm-prefix
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_successful(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("success"):
                continue
            key = (r["conversation_id"], r["conversation_idx"])
            # If a (cid, idx) appears more than once (e.g. retried), keep the last.
            out[key] = r
    return out


def _pct(values: list[float], q: float) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    clean.sort()
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return clean[int(k)]
    return clean[f] + (clean[c] - clean[f]) * (k - f)


def _stats_row(label: str, values: list[float], unit: str = "ms") -> str:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return f"  {label:<28}: n=0"
    return (f"  {label:<28}: n={len(clean):>3}  "
            f"min={min(clean):>9.3f}  "
            f"p50={_pct(clean, 50):>9.3f}  "
            f"mean={statistics.fmean(clean):>9.3f}  "
            f"p90={_pct(clean, 90):>9.3f}  "
            f"p99={_pct(clean, 99):>9.3f}  "
            f"max={max(clean):>9.3f}  {unit}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a_path", type=Path, help="rwlt_requests.jsonl from layout A")
    p.add_argument("b_path", type=Path, help="rwlt_requests.jsonl from layout B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--warm-threshold", type=float, default=0.5,
                   help="cached_tokens / input_tokens above which a turn is "
                        "considered warm-prefix. Default 0.5.")
    p.add_argument("--per-turn-out", type=Path, default=None,
                   help="If set, write a TSV with one row per matched turn"
                        " (conversation_id, idx, ISL_a/b, OSL_a/b, cached_a,"
                        " ttft_a, ttft_b, delta_ms, warm). Useful for"
                        " sorting/filtering the per-request comparison.")
    args = p.parse_args()

    a = _load_successful(args.a_path)
    b = _load_successful(args.b_path)
    only_a = sorted(a.keys() - b.keys())
    only_b = sorted(b.keys() - a.keys())
    common = sorted(a.keys() & b.keys())

    print(f"# {args.label_a}: {len(a)} successful turns from {args.a_path.name}")
    print(f"# {args.label_b}: {len(b)} successful turns from {args.b_path.name}")
    print(f"# matched on (conversation_id, conversation_idx): {len(common)}")
    print(f"# only in {args.label_a}: {len(only_a)}")
    print(f"# only in {args.label_b}: {len(only_b)}")
    if only_a:
        print(f"  e.g. {only_a[:3]}")
    if only_b:
        print(f"  e.g. {only_b[:3]}")

    # ISL agreement check.
    isl_mismatch: list[tuple] = []
    cold_deltas: list[float] = []
    warm_deltas: list[float] = []
    all_deltas: list[float] = []
    a_ttft: list[float] = []
    b_ttft: list[float] = []
    osl_deltas: list[int] = []
    per_conv_delta: dict[str, list[float]] = defaultdict(list)
    # New: cache reuse parity and unique-token (real-prefill) analyses.
    cache_diff_blocks: list[int] = []   # cached_b - cached_a
    cache_ratio_diff: list[float] = []  # ratio_b - ratio_a
    unique_a: list[int] = []            # ISL_a - cached_a (real prefill work in agg)
    unique_b: list[int] = []            # ISL_b - cached_b (real prefill work in disagg)
    unique_per_turn: list[tuple[int, int, float]] = []  # (unique_a, unique_b, delta_ms)
    itl_deltas: list[float] = []        # disagg - agg ITL ms (decode speed proxy)
    a_itl: list[float] = []
    b_itl: list[float] = []

    for key in common:
        ra, rb = a[key], b[key]
        isl_a = ra.get("server_input_tokens")
        isl_b = rb.get("server_input_tokens")
        if isl_a is None or isl_b is None or isl_a != isl_b:
            isl_mismatch.append((key, isl_a, isl_b))
        tta = ra.get("ttft")
        ttb = rb.get("ttft")
        if tta is None or ttb is None:
            continue
        delta_ms = (ttb - tta) * 1000.0
        all_deltas.append(delta_ms)
        a_ttft.append(tta * 1000.0)
        b_ttft.append(ttb * 1000.0)
        per_conv_delta[key[0]].append(delta_ms)

        cached_a = ra.get("server_cached_tokens") or 0
        cached_b = rb.get("server_cached_tokens") or 0
        in_a = isl_a or 0
        in_b = isl_b or 0
        ratio_a = (cached_a / in_a) if in_a else 0.0
        ratio_b = (cached_b / in_b) if in_b else 0.0

        cache_diff_blocks.append(cached_b - cached_a)
        cache_ratio_diff.append(ratio_b - ratio_a)

        unique_a_val = max(0, in_a - cached_a)
        unique_b_val = max(0, in_b - cached_b)
        unique_a.append(unique_a_val)
        unique_b.append(unique_b_val)
        unique_per_turn.append((unique_a_val, unique_b_val, delta_ms))

        if ratio_a >= args.warm_threshold:
            warm_deltas.append(delta_ms)
        else:
            cold_deltas.append(delta_ms)

        osl_a = ra.get("server_output_tokens") or 0
        osl_b = rb.get("server_output_tokens") or 0
        osl_deltas.append(osl_b - osl_a)

        ila = ra.get("itl")
        ilb = rb.get("itl")
        if ila is not None and ilb is not None:
            a_itl.append(ila * 1000.0)
            b_itl.append(ilb * 1000.0)
            itl_deltas.append((ilb - ila) * 1000.0)

    print()
    print(f"## ISL agreement: {len(common) - len(isl_mismatch)}/{len(common)} turns match exactly")
    if isl_mismatch:
        print(f"  {len(isl_mismatch)} mismatches; first few:")
        for key, ia, ib in isl_mismatch[:5]:
            print(f"    {key}: {args.label_a}={ia}  {args.label_b}={ib}")

    print()
    print(f"## TTFT distributions (matched turns)")
    print(_stats_row(f"{args.label_a} TTFT", a_ttft))
    print(_stats_row(f"{args.label_b} TTFT", b_ttft))
    print(_stats_row(f"{args.label_b} - {args.label_a}", all_deltas))
    print()
    print(f"## TTFT delta split by prefix-cache hit (warm = cached/input >= {args.warm_threshold})")
    print(_stats_row("cold-prefix turns", cold_deltas))
    print(_stats_row("warm-prefix turns", warm_deltas))
    print()
    print(f"## OSL delta (spec acceptance variance)")
    print(_stats_row(f"{args.label_b} - {args.label_a} OSL", [float(x) for x in osl_deltas], unit="tok"))

    print()
    print(f"## KV cache reuse parity (server-reported cached_tokens)")
    cached_a_vals = [float(c) for c in [(a[k].get('server_cached_tokens') or 0) for k in common]]
    cached_b_vals = [float(c) for c in [(b[k].get('server_cached_tokens') or 0) for k in common]]
    print(_stats_row(f"{args.label_a} cached_tokens", cached_a_vals, unit="tok"))
    print(_stats_row(f"{args.label_b} cached_tokens", cached_b_vals, unit="tok"))
    print(_stats_row(f"{args.label_b} - {args.label_a} cached_tokens",
                     [float(x) for x in cache_diff_blocks], unit="tok"))
    print(_stats_row(f"{args.label_b} - {args.label_a} cache ratio",
                     [r * 100 for r in cache_ratio_diff], unit="%"))
    same_cache = sum(1 for d in cache_diff_blocks if d == 0)
    print(f"  {len(common) - same_cache}/{len(common)} turns differ in cached_tokens")
    if any(d != 0 for d in cache_diff_blocks):
        print(f"  (positive = {args.label_b} reused more blocks than {args.label_a})")

    print()
    print(f"## Real prefill work per turn (input_tokens - cached_tokens)")
    print(_stats_row(f"{args.label_a} unique tokens", [float(x) for x in unique_a], unit="tok"))
    print(_stats_row(f"{args.label_b} unique tokens", [float(x) for x in unique_b], unit="tok"))
    # Bucket TTFT delta by unique-token count (the actual prefill work) to see
    # if disagg overhead scales with real work or with total input.
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    edges = [(0, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 1 << 31)]
    for ua, ub, dms in unique_per_turn:
        u = max(ua, ub)
        for lo, hi in edges:
            if lo <= u < hi:
                buckets[(lo, hi)].append(dms)
                break
    print()
    print(f"## TTFT delta bucketed by max-unique-tokens (real prefill work this turn)")
    print(f"  {'bucket (tok)':>16}  {'n':>5}  {'p50 ms':>9}  {'mean ms':>9}  {'p99 ms':>9}")
    for (lo, hi), vals in sorted(buckets.items()):
        if not vals:
            continue
        clean = sorted(vals)
        p50 = _pct(clean, 50); p99 = _pct(clean, 99); mean = statistics.fmean(clean)
        hi_disp = "inf" if hi >= 1 << 30 else str(hi)
        print(f"  {f'{lo}-{hi_disp}':>16}  {len(clean):>5}  {p50:>9.3f}  {mean:>9.3f}  {p99:>9.3f}")

    if a_itl:
        print()
        print(f"## ITL (mean inter-token latency per turn; decode speed proxy)")
        print(_stats_row(f"{args.label_a} ITL", a_itl))
        print(_stats_row(f"{args.label_b} ITL", b_itl))
        print(_stats_row(f"{args.label_b} - {args.label_a} ITL", itl_deltas))

    if args.per_turn_out is not None:
        args.per_turn_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_turn_out.open("w") as f:
            f.write("\t".join([
                "conversation_id", "conversation_idx",
                f"isl_{args.label_a}", f"isl_{args.label_b}", "isl_match",
                f"osl_{args.label_a}", f"osl_{args.label_b}",
                f"cached_{args.label_a}", f"cached_{args.label_b}",
                f"cache_ratio_{args.label_a}", f"cache_ratio_{args.label_b}",
                "cache_blocks_delta", "warm_prefix",
                f"unique_{args.label_a}", f"unique_{args.label_b}",
                f"ttft_{args.label_a}_ms", f"ttft_{args.label_b}_ms",
                "ttft_delta_ms",
                f"itl_{args.label_a}_ms", f"itl_{args.label_b}_ms",
            ]) + "\n")
            for key in common:
                ra, rb = a[key], b[key]
                tta = ra.get("ttft")
                ttb = rb.get("ttft")
                if tta is None or ttb is None:
                    continue
                isl_a = ra.get("server_input_tokens") or 0
                isl_b = rb.get("server_input_tokens") or 0
                osl_a = ra.get("server_output_tokens") or 0
                osl_b = rb.get("server_output_tokens") or 0
                cached_a = ra.get("server_cached_tokens") or 0
                cached_b = rb.get("server_cached_tokens") or 0
                ratio_a = (cached_a / isl_a) if isl_a else 0.0
                ratio_b = (cached_b / isl_b) if isl_b else 0.0
                warm = ratio_a >= args.warm_threshold
                ila = ra.get("itl")
                ilb = rb.get("itl")
                f.write("\t".join([
                    key[0], str(key[1]),
                    str(isl_a), str(isl_b), str(isl_a == isl_b).lower(),
                    str(osl_a), str(osl_b),
                    str(cached_a), str(cached_b),
                    f"{ratio_a:.4f}", f"{ratio_b:.4f}",
                    str(cached_b - cached_a), str(warm).lower(),
                    str(max(0, isl_a - cached_a)), str(max(0, isl_b - cached_b)),
                    f"{tta * 1000:.3f}", f"{ttb * 1000:.3f}",
                    f"{(ttb - tta) * 1000:.3f}",
                    (f"{ila * 1000:.3f}" if ila is not None else ""),
                    (f"{ilb * 1000:.3f}" if ilb is not None else ""),
                ]) + "\n")
        print(f"\n## Per-turn TSV written to {args.per_turn_out}")
        print(
            f"   useful one-liners:\n"
            f"     sort -k13,13 -g -r {args.per_turn_out} | head -10   # worst-delta turns\n"
            f"     awk -F\\\\t 'NR>1 && $5==\"false\"' {args.per_turn_out} | wc -l  # ISL-mismatched count\n"
            f"     awk -F\\\\t 'NR>1 && $10==\"false\"' {args.per_turn_out} | datamash -W mean 13 perc:50 13 perc:99 13  # cold-prefix delta stats\n"
        )

    print()
    print(f"## Per-conversation median TTFT delta (top 10 by |median|)")
    rows = sorted(
        ((cid, statistics.median(ds), len(ds)) for cid, ds in per_conv_delta.items()),
        key=lambda r: abs(r[1]),
        reverse=True,
    )
    print(f"  {'conversation_id':>30}  {'turns':>6}  median delta ms")
    for cid, med, n in rows[:10]:
        print(f"  {cid:>30}  {n:>6}  {med:>15.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

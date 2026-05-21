#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the cookbook-style stage breakdown from benchmark_serving.py output.

Reads the `*-perf_metrics.json` file that benchmark_serving writes when
`--save-request-time-breakdown` is set, and prints a Markdown-formatted
per-stage breakdown that follows the same stage names as
`simengl/trtllm-cookbook .../aligned_request_lifetime_report_20260512.md`.

Handles both aggregated and disaggregated formats.

Default mode aggregates all requests (dropping the first by default as warmup)
and reports min / p50 / mean / p90 / p99 / max for each stage. Pass
`--request-index N` to instead print a single-request waterfall.

Usage:
    # 32-request aggregate (default; drops index 0 as warmup):
    python3 scripts/cookbook_breakdown.py results/agg/openai-inf-concurrency1-gpt-oss-120b-*-perf_metrics.json

    # Single request:
    python3 scripts/cookbook_breakdown.py <perf_metrics.json> --request-index 1
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


# ---------- Stage definitions (start_field, end_field, label) ----------

AGG_STAGES: list[tuple[str, str, str]] = [
    ("server_arrival_time", "arrival_time",
     "server ingress (server_arrival -> arrival)"),
    ("arrival_time", "first_scheduled_time",
     "queue (arrival -> first_scheduled)"),
    ("first_scheduled_time", "first_token_time",
     "context/prefill + first token (first_scheduled -> first_token)"),
    ("first_token_time", "server_first_token_time",
     "executor -> server token (first_token -> server_first_token)"),
]
AGG_TTFT = ("server_arrival_time", "server_first_token_time",
            "TTFT (server_arrival -> server_first_token)")

# Each stage tuple here is (start_source, start_field, end_source, end_field, label)
# where source is one of: "disagg", "ctx", "gen".
DISAGG_STAGES: list[tuple[str, str, str, str, str]] = [
    ("disagg", "disagg_server_arrival_time", "ctx", "server_arrival_time",
     "disagg ingress (disagg_arrival -> ctx_server_arrival)"),
    ("ctx", "server_arrival_time", "ctx", "arrival_time",
     "prefill ingress (ctx_server_arrival -> ctx_arrival)"),
    ("ctx", "arrival_time", "ctx", "first_scheduled_time",
     "prefill queue (ctx_arrival -> ctx_first_scheduled)"),
    ("ctx", "first_scheduled_time", "ctx", "first_token_time",
     "context compute (ctx_first_scheduled -> ctx_first_token)"),
    ("ctx", "first_token_time", "ctx", "server_first_token_time",
     "prefill finalize (ctx_first_token -> ctx_server_first_token)"),
    ("ctx", "server_first_token_time", "gen", "server_arrival_time",
     "prefill -> decode (ctx_server_first_token -> gen_server_arrival)"),
    ("gen", "server_arrival_time", "gen", "arrival_time",
     "decode ingress (gen_server_arrival -> gen_arrival)"),
    ("gen", "arrival_time", "gen", "kv_cache_transfer_start",
     "decode wait before KV transfer (gen_arrival -> kv_cache_transfer_start)"),
    ("gen", "kv_cache_transfer_start", "gen", "kv_cache_transfer_end",
     "KV transfer (kv_cache_transfer_start -> kv_cache_transfer_end)"),
    ("gen", "kv_cache_transfer_end", "gen", "first_scheduled_time",
     "decode queue (kv_cache_transfer_end -> gen_first_scheduled)"),
    ("gen", "first_scheduled_time", "gen", "first_token_time",
     "decode compute (gen_first_scheduled -> gen_first_token)"),
    ("gen", "first_token_time", "gen", "server_first_token_time",
     "decode -> server token (gen_first_token -> gen_server_first_token)"),
    ("gen", "server_first_token_time", "disagg",
     "disagg_server_first_token_time",
     "disagg egress (gen_server_first_token -> disagg_first_token)"),
]
DISAGG_TTFT = ("disagg", "disagg_server_arrival_time", "disagg",
               "disagg_server_first_token_time",
               "TTFT (disagg_arrival -> disagg_first_token)")


# ---------- Field lookup helpers ----------

def _timing(req: dict[str, Any], source: str) -> dict[str, Any]:
    """Return the timing_metrics dict for the requested source, or {}."""
    if source == "ctx":
        return ((req.get("ctx_perf_metrics") or {}).get("perf_metrics")
                or {}).get("timing_metrics") or {}
    if source == "gen":
        return ((req.get("gen_perf_metrics") or {}).get("perf_metrics")
                or {}).get("timing_metrics") or {}
    if source == "disagg":
        return req
    if source == "agg":
        return ((req.get("perf_metrics") or {}).get("timing_metrics")) or {}
    raise ValueError(source)


def _ms(start: float, end: float) -> float:
    if start is None or end is None or math.isnan(start) or math.isnan(end):
        return float("nan")
    return (end - start) * 1000.0


def _stage_duration_agg(req: dict[str, Any], start_field: str,
                        end_field: str) -> float:
    t = _timing(req, "agg")
    return _ms(t.get(start_field, float("nan")),
               t.get(end_field, float("nan")))


def _stage_duration_disagg(req: dict[str, Any], src_start: str,
                           start_field: str, src_end: str,
                           end_field: str) -> float:
    s = _timing(req, src_start)
    e = _timing(req, src_end)
    return _ms(s.get(start_field, float("nan")),
               e.get(end_field, float("nan")))


# ---------- Formatters ----------

def _percentile(values: list[float], pct: float) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    clean.sort()
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return clean[int(k)]
    return clean[f] + (clean[c] - clean[f]) * (k - f)


def _stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {
            "n": 0,
            "min": float("nan"),
            "p50": float("nan"),
            "mean": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": len(clean),
        "min": min(clean),
        "p50": _percentile(clean, 50),
        "mean": statistics.fmean(clean),
        "p90": _percentile(clean, 90),
        "p99": _percentile(clean, 99),
        "max": max(clean),
    }


def _fmt_stat(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    return f"{v:.3f}"


def _print_agg_aggregate(reqs: list[dict[str, Any]],
                         dropped_warmup: int) -> None:
    ttft_values = [
        _stage_duration_agg(r, AGG_TTFT[0], AGG_TTFT[1]) for r in reqs
    ]
    stage_values: dict[str, list[float]] = {
        label: [_stage_duration_agg(r, s, e) for r in reqs]
        for s, e, label in AGG_STAGES
    }

    ttft_stats = _stats(ttft_values)
    print("## Aggregated server lane (per-request stage breakdown)")
    print()
    print(f"- requests analysed: {len(reqs)} (dropped {dropped_warmup} warmup)")
    print(
        f"- TTFT ms        min={_fmt_stat(ttft_stats['min'])}  "
        f"p50={_fmt_stat(ttft_stats['p50'])}  mean={_fmt_stat(ttft_stats['mean'])}  "
        f"p90={_fmt_stat(ttft_stats['p90'])}  p99={_fmt_stat(ttft_stats['p99'])}  "
        f"max={_fmt_stat(ttft_stats['max'])}")
    print()
    print("| stage | min ms | p50 ms | mean ms | p90 ms | p99 ms | max ms | "
          "p50 % of p50 TTFT |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    p50_ttft = ttft_stats["p50"]
    for label in stage_values:
        s = _stats(stage_values[label])
        pct = (s["p50"] / p50_ttft * 100.0) if (
            not math.isnan(s["p50"]) and not math.isnan(p50_ttft)
            and p50_ttft > 0) else float("nan")
        print(f"| {label} | {_fmt_stat(s['min'])} | {_fmt_stat(s['p50'])} | "
              f"{_fmt_stat(s['mean'])} | {_fmt_stat(s['p90'])} | "
              f"{_fmt_stat(s['p99'])} | {_fmt_stat(s['max'])} | "
              f"{_fmt_stat(pct)}% |")


def _print_disagg_aggregate(reqs: list[dict[str, Any]],
                            dropped_warmup: int) -> None:
    # TTFT is a disagg-level field pair, not a per-source timing field.
    ttft_values = [
        _ms(r.get(DISAGG_TTFT[1], float("nan")),
            r.get(DISAGG_TTFT[3], float("nan"))) for r in reqs
    ]

    stage_values: dict[str, list[float]] = {}
    for src_s, fld_s, src_e, fld_e, label in DISAGG_STAGES:
        stage_values[label] = [
            _stage_duration_disagg(r, src_s, fld_s, src_e, fld_e) for r in reqs
        ]

    ttft_stats = _stats(ttft_values)
    print("## Disaggregated server lane (per-request stage breakdown)")
    print()
    print(f"- requests analysed: {len(reqs)} (dropped {dropped_warmup} warmup)")
    print(
        f"- TTFT ms        min={_fmt_stat(ttft_stats['min'])}  "
        f"p50={_fmt_stat(ttft_stats['p50'])}  mean={_fmt_stat(ttft_stats['mean'])}  "
        f"p90={_fmt_stat(ttft_stats['p90'])}  p99={_fmt_stat(ttft_stats['p99'])}  "
        f"max={_fmt_stat(ttft_stats['max'])}")
    print()
    print("| stage | min ms | p50 ms | mean ms | p90 ms | p99 ms | max ms | "
          "p50 % of p50 TTFT |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    p50_ttft = ttft_stats["p50"]
    for label, vals in stage_values.items():
        s = _stats(vals)
        pct = (s["p50"] / p50_ttft * 100.0) if (
            not math.isnan(s["p50"]) and not math.isnan(p50_ttft)
            and p50_ttft > 0) else float("nan")
        print(f"| {label} | {_fmt_stat(s['min'])} | {_fmt_stat(s['p50'])} | "
              f"{_fmt_stat(s['mean'])} | {_fmt_stat(s['p90'])} | "
              f"{_fmt_stat(s['p99'])} | {_fmt_stat(s['max'])} | "
              f"{_fmt_stat(pct)}% |")


def _print_single_agg(req: dict[str, Any]) -> None:
    t = _timing(req, "agg")
    ttft_ms = _ms(t.get("server_arrival_time", float("nan")),
                  t.get("server_first_token_time", float("nan")))
    print("## Aggregated server lane (single request)")
    print()
    print(f"TTFT (server_arrival -> server_first_token): **{ttft_ms:.3f} ms**")
    print()
    print("| step | start s | end s | duration ms | % of TTFT |")
    print("|---|---:|---:|---:|---:|")
    for s_fld, e_fld, label in AGG_STAGES:
        start = t.get(s_fld, float("nan"))
        end = t.get(e_fld, float("nan"))
        dur = _ms(start, end)
        pct = (dur / ttft_ms * 100.0) if (
            not math.isnan(dur) and ttft_ms > 0) else float("nan")
        print(f"| {label} | {start:.6f} | {end:.6f} | "
              f"{_fmt_stat(dur)} | {_fmt_stat(pct)}% |")


def _print_single_disagg(req: dict[str, Any]) -> None:
    ttft_ms = _ms(req.get("disagg_server_arrival_time", float("nan")),
                  req.get("disagg_server_first_token_time", float("nan")))
    print("## Disaggregated server lane (single request)")
    print()
    print(f"TTFT (disagg_arrival -> disagg_first_token): **{ttft_ms:.3f} ms**")
    print()
    print("| step | start s | end s | duration ms | % of TTFT |")
    print("|---|---:|---:|---:|---:|")
    for src_s, fld_s, src_e, fld_e, label in DISAGG_STAGES:
        s = _timing(req, src_s)
        e = _timing(req, src_e)
        start = s.get(fld_s,
                      req.get(fld_s, float("nan"))) if src_s == "disagg" else \
            s.get(fld_s, float("nan"))
        end = e.get(fld_e,
                    req.get(fld_e, float("nan"))) if src_e == "disagg" else \
            e.get(fld_e, float("nan"))
        dur = _ms(start, end)
        pct = (dur / ttft_ms * 100.0) if (
            not math.isnan(dur) and ttft_ms > 0) else float("nan")
        print(f"| {label} | {start:.6f} | {end:.6f} | "
              f"{_fmt_stat(dur)} | {_fmt_stat(pct)}% |")


def detect_mode(req: dict[str, Any]) -> str:
    if "ctx_perf_metrics" in req or "gen_perf_metrics" in req:
        return "disagg"
    return "agg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument(
        "--request-index",
        type=int,
        default=None,
        help="Index (0-based) of a single request to print a waterfall for."
        " If unset (default), all requests are aggregated.")
    parser.add_argument(
        "--drop-warmup",
        type=int,
        default=1,
        help="Number of leading requests to discard as warmup before aggregating."
        " Default 1. Has no effect in single-request mode.")
    parser.add_argument(
        "--mode",
        choices=["auto", "agg", "disagg"],
        default="auto",
        help="Format hint. 'auto' picks based on the presence of ctx_perf_metrics.",
    )
    args = parser.parse_args()

    if not args.json_file.is_file():
        print(f"ERROR: {args.json_file} not found", file=sys.stderr)
        return 1
    data = json.loads(args.json_file.read_text())
    if not isinstance(data, list):
        print("ERROR: expected JSON array at top level", file=sys.stderr)
        return 1
    if not data:
        print("ERROR: JSON array is empty", file=sys.stderr)
        return 1

    mode = args.mode if args.mode != "auto" else detect_mode(data[0])

    print(f"# Stage breakdown for {args.json_file.name}")
    print()
    print(f"- Total requests in file: {len(data)}")
    print(f"- Detected mode         : {mode}")

    if args.request_index is not None:
        idx = args.request_index
        if idx >= len(data):
            print(
                f"NOTE: only {len(data)} request(s); using index {len(data) - 1}",
                file=sys.stderr)
            idx = len(data) - 1
        req = data[idx]
        print(f"- Single-request waterfall for index {idx} (0-based)")
        print()
        if mode == "agg":
            _print_single_agg(req)
        else:
            _print_single_disagg(req)
        return 0

    drop = max(0, min(args.drop_warmup, len(data) - 1))
    reqs = data[drop:]
    if mode == "agg":
        _print_agg_aggregate(reqs, dropped_warmup=drop)
    else:
        _print_disagg_aggregate(reqs, dropped_warmup=drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

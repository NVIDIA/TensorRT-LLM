#!/usr/bin/env python3
"""CLI: compute per-session real KV-cache hit rate from a run's step JSON.

Companion to ``compute_cache_hit_trace.py`` (offline upper bound). Reads
the engine-measured per-request hit/miss counts already merged into
``step<STEP>.json`` by trace_replay_client (drained from /perf_metrics)
and produces a ``<trace_name>.trace.realcachehit.json`` annotated trace
plus a per-session + per-trace rollup.

Examples::

    # Default output next to the trace file
    python compute_real_cache_hit.py \\
        --step_json .../pareto_server_output/<run_stem>/step8.json

    # Explicit trace + output paths
    python compute_real_cache_hit.py \\
        --step_json .../step8.json \\
        --trace_json .../django__django-16801.trace.json \\
        --out .../django__django-16801.trace.realcachehit.json

The output schema is documented in :mod:`real_cache_hit`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Local package imports — allow running as a script from this directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.io import default_annotated_trace_path, load_trace, write_json
    from analysis.real_cache_hit import (
        REAL_ANNOTATED_TRACE_SUFFIX,
        compute_real_cache_hit,
    )
else:
    from .io import default_annotated_trace_path, load_trace, write_json
    from .real_cache_hit import (
        REAL_ANNOTATED_TRACE_SUFFIX,
        compute_real_cache_hit,
    )


def _resolve_trace_path(step_json: dict, override: Path | None) -> Path:
    """Find the *.trace.json that this step JSON was run against."""
    if override is not None:
        p = override.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--trace_json not found: {p}")
        return p
    src = step_json.get("trace_file")
    if src:
        p = Path(src).expanduser().resolve()
        if p.is_file():
            return p
    # Fallback: trace_dir + single trace
    td = step_json.get("trace_dir")
    if td:
        d = Path(td).expanduser().resolve()
        if d.is_dir():
            cand = sorted(d.glob("*.trace.json"))
            cand = [c for c in cand if not c.name.endswith(".full.trace.json")]
            if len(cand) == 1:
                return cand[0]
    raise FileNotFoundError(
        "could not resolve trace file from step JSON; pass --trace_json")


def _default_real_annotated_path(trace_file: Path,
                                 override: Path | None) -> Path:
    """``<trace_stem>.trace.realcachehit.json`` next to *trace_file* by default."""
    if override is not None:
        return override.expanduser().resolve()
    if trace_file.name.endswith(".trace.json"):
        stem = trace_file.name[:-len(".trace.json")]
        return trace_file.with_name(stem + REAL_ANNOTATED_TRACE_SUFFIX)
    return trace_file.with_suffix(REAL_ANNOTATED_TRACE_SUFFIX)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Compute per-session real KV cache hit/miss block counts from a "
        "Pareto run's step JSON.")
    parser.add_argument(
        "--step_json",
        required=True,
        type=Path,
        help="Path to step<STEP>.json from a run's output directory.")
    parser.add_argument(
        "--trace_json",
        type=Path,
        default=None,
        help="Path to *.trace.json (default: resolve from step_json.trace_file).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <trace_stem>.trace.realcachehit.json "
        "next to the trace file).",
    )
    parser.add_argument(
        "--tokens-per-block",
        dest="tokens_per_block",
        type=int,
        default=32,
        help="KV block size for token-level estimates (engine reports only "
        "block counts per request). Match the runtime config.",
    )
    args = parser.parse_args()

    step_path = args.step_json.expanduser().resolve()
    with step_path.open("r", encoding="utf-8") as f:
        step_json = json.load(f)

    trace_path = _resolve_trace_path(step_json, args.trace_json)
    trace_data = load_trace(trace_path)

    record = compute_real_cache_hit(step_json,
                                    trace_data,
                                    tokens_per_block=args.tokens_per_block)
    out_path = _default_real_annotated_path(trace_path, args.out)
    write_json(out_path, record)

    summary = record["summary"]
    print(f"sessions             : {summary['n_sessions']}")
    print(
        f"assistant events/session: {summary['n_assistant_events_per_session']}"
    )
    print(f"total LLM calls      : {summary['total_llm_calls']}")
    print(f"total hit  blocks    : {summary['total_cache_hit_blocks']}")
    print(f"total miss blocks    : {summary['total_cache_miss_blocks']}")
    print(
        f"trace-level block hit rate: {summary['overall_cache_block_hit_rate']:.4f}"
    )
    if record.get("warnings"):
        print("warnings:")
        for w in record["warnings"]:
            print(f"  - {w}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

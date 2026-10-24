#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Op-level A/B micro-benchmark: is a candidate faster than what the target calls today?

Runs **by path inside the engine container**. A *case file* is a Python file
that defines ``cases()``, yielding :class:`Case` objects. Each case names one
workload shape and maps implementation names to callables::

    from microbench import Case  # this file's directory is on sys.path


    def cases():
        for m in (1, 128, 8192):
            x = torch.randn(m, 10240, device="cuda", dtype=torch.bfloat16)
            ws = [torch.randn(...) for _ in range(48)]  # one copy per layer
            yield Case(
                name=f"hc_mix m={m}",
                impls={
                    "current": lambda i: cur(x, ws[i % 48]),
                    "fused": lambda i: fused(x, ws[i % 48]),
                },
                reference="current",  # the one the others are checked against
                check=lambda a, b: torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2),
            )

Every implementation is called as ``fn(i)`` with a rotating index, so a case
can cycle through per-layer weight copies. A decode step touches each layer's
weights once, and a timing loop that re-reads one 6 MB weight from L2 reports
a speed no forward will see.

Two timings per implementation:

- ``graph_us``: ``reps`` calls captured into one CUDA graph and replayed. This is what a
  decode step under CUDA graphs pays: launch overhead gone, inter-kernel gaps kept.
- ``eager_us``: the same calls launched eagerly. This is what a prefill step (no graph) pays,
  and the gap between the two is the launch cost a fusion removes.

Output is a JSON list of rows (``--out``) and a markdown table on stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass
class Case:
    name: str
    impls: Mapping[str, Callable[[int], Any]]
    reference: str | None = None
    check: Callable[[Any, Any], None] | None = None
    reps: int = 48
    meta: dict = field(default_factory=dict)


def _sync():
    import torch

    torch.cuda.synchronize()


def time_eager(fn: Callable[[int], Any], reps: int, iters: int, warmup: int) -> float:
    import torch

    for i in range(warmup):
        fn(i)
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        for i in range(reps):
            fn(i)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / reps)
    return statistics.median(samples)


def time_graph(fn: Callable[[int], Any], reps: int, iters: int, warmup: int) -> float:
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(max(warmup, 3)):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    _sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            fn(i)
    _sync()
    for _ in range(3):
        graph.replay()
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / reps)
    del graph
    return statistics.median(samples)


def run_case(case: Case, *, iters: int, warmup: int, graph: bool, eager: bool) -> list[dict]:
    rows = []
    outputs = {}
    for name, fn in case.impls.items():
        row: dict[str, Any] = {"case": case.name, "impl": name, **case.meta}
        try:
            outputs[name] = fn(0)
            _sync()
            if eager:
                row["eager_us"] = round(time_eager(fn, case.reps, iters, warmup), 3)
            if graph:
                try:
                    row["graph_us"] = round(time_graph(fn, case.reps, iters, warmup), 3)
                except Exception as exc:  # noqa: BLE001 - a capture failure is a result
                    row["graph_error"] = f"{type(exc).__name__}: {exc}"[:300]
        except Exception as exc:  # noqa: BLE001 - one broken candidate must not hide the rest
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["traceback"] = traceback.format_exc()[-1500:]
        rows.append(row)
    if case.reference and case.check and case.reference in outputs:
        for name, out in outputs.items():
            if name == case.reference:
                continue
            row = next(r for r in rows if r["impl"] == name)
            try:
                case.check(out, outputs[case.reference])
                row["check"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row["check"] = f"FAIL: {exc}"[:400]
    ref = next((r for r in rows if r["impl"] == case.reference), None)
    for row in rows:
        for key in ("graph_us", "eager_us"):
            if ref and key in row and key in ref and row[key]:
                row[f"{key.split('_')[0]}_speedup"] = round(ref[key] / row[key], 3)
    return rows


def load_cases(path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_rows(rows: list[dict]) -> str:
    lines = [
        "| case | impl | graph us | eager us | speedup (graph) | check |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['impl']} | {r.get('graph_us', r.get('graph_error', '-'))} | "
            f"{r.get('eager_us', r.get('error', '-'))} | {r.get('graph_speedup', '')} | {r.get('check', '')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="op-level A/B micro-benchmark")
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--only", default="", help="substring filter on case names")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-eager", action="store_true")
    args = parser.parse_args()

    import torch

    module = load_cases(args.case_file)
    rows: list[dict] = []
    started = time.time()
    for case in module.cases():
        if args.only and args.only not in case.name:
            continue
        rows.extend(
            run_case(
                case,
                iters=args.iters,
                warmup=args.warmup,
                graph=not args.no_graph,
                eager=not args.no_eager,
            )
        )
        torch.cuda.empty_cache()
    env = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "case_file": str(args.case_file),
        "seconds": round(time.time() - started, 1),
    }
    try:
        import flashinfer

        env["flashinfer"] = flashinfer.__version__
    except Exception:  # noqa: BLE001
        pass
    print(format_rows(rows))
    print(json.dumps(env))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"env": env, "rows": rows}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

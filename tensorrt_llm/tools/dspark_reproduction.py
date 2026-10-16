# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public native calibration collection and normal static/enabled comparison.

Run by file path so CPU-only comparison/help does not initialize TRT-LLM/CUDA.
Each ``run`` is one independent model lifetime; order/repeats belong to caller.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _cpu_module():
    path = Path(__file__).with_name("dspark_observations.py")
    name = "_dspark_public_observations"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="One real full-model lifetime, with natural shutdown")
    run.add_argument("--mode", choices=("static", "dynamic", "collect"), required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument(
        "--recorded-source-origin",
        type=Path,
        help="Optional recorded HEAD/full-diff JSON for this immutable checkout.",
    )
    run.add_argument(
        "--recorded-source-origin-sha256",
        help="SHA256 of the exact optional recorded source-origin bytes.",
    )
    run.add_argument("--physical-k", type=int, required=True)
    run.add_argument(
        "--batch-size", type=int, required=True, help="Global fixed-cohort request count"
    )
    run.add_argument("--output-tokens", type=int, required=True)
    run.add_argument(
        "--timeout", type=float, required=True, help="Full startup+model+shutdown budget in seconds"
    )
    run.add_argument("--warmup-cohorts", type=int, default=1)
    run.add_argument("--cost-table", type=Path)
    run.add_argument("--calibration", type=Path)
    run.add_argument("--output", type=Path, required=True, help="Fresh private output directory")
    compare = sub.add_parser("compare", help="Compare closed independent model lifetimes")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        from tensorrt_llm._torch.speculative.dspark_reproduction import run_native

        return run_native(args)
    records = [json.loads(path.read_bytes()) for path in args.runs]
    report = _cpu_module().compare_runs(records)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

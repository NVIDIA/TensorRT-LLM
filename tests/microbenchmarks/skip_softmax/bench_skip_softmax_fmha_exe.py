#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two-pass kernel microbench driver for skip-softmax FMHA on Hopper sm90.

Pass 1 (sparsity)
  Runs `bin/fmha.exe` with `-skip-softmax-stat`. The stat-collecting cubin
  variant (_skipSoftmaxStat) is selected at runtime via Launch_params; the
  binary prints a "Skip-Softmax .: skipped / total" line we parse.

Pass 2 (speedup)
  Re-runs each config without `-skip-softmax-stat` so the production
  _skipSoftmax cubin is selected - no atomic-counter overhead. Median latency
  is computed from N=20 timed runs after M=5 warm-up runs. Speedup =
  latency(threshold=0) / latency(threshold).

Both passes shell out to the same `bin/fmha.exe` produced by
`cpp/kernels/fmha_v2/Makefile`. The two cubin families and the runtime
selection bit are introduced by the wider skip-softmax-stat refactor; if you
build the wheel before applying that refactor pass 1 will pick the regular
_skipSoftmax cubin and report total=0 (no atomics).
"""

from __future__ import annotations

import argparse
import csv
import os  # noqa: F401
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent.parent))  # repo root

from tests.microbenchmarks.skip_softmax.diffusion_configs import diffusion_configs  # noqa: E402
from tests.microbenchmarks.skip_softmax.llm_configs import FmhaConfig, llm_configs  # noqa: E402

DEFAULT_FMHA_EXE = (
    THIS_DIR.parent.parent.parent / "cpp" / "kernels" / "fmha_v2" / "bin" / "fmha.exe"
)

ELAPSED_RE = re.compile(
    r"(?:Elapsed|Fused\s+time)\s*\.+:\s*([0-9.]+)\s*us"
    r"(?:\s*\(([0-9.]+)x\))?"
    r"(?:,\s*([0-9.]+)\s*Tflop/s)?"
)
SKIP_RE = re.compile(r"Skip-Softmax\s*\.+:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)%")


def _dtype_flag(dtype: str) -> List[str]:
    return {
        "fp16": ["-fp16"],
        "bf16": ["-bf16"],
        "e4m3": ["-e4m3"],
    }[dtype]


def _mask_flag(mask: str) -> List[str]:
    if mask == "causal":
        return ["-causal-mask"]
    if mask == "bidirectional":
        # PredefinedAttentionMask.FULL maps to the padding/no-mask path in the
        # TRTLLM torch backend. Leave fmha.exe on its default padding mask path
        # to match that behavior; bidirectional sliding-window is a local mask.
        return []
    raise ValueError(f"Unknown mask: {mask}")


def _build_cmd(
    fmha_exe: Path, cfg: FmhaConfig, threshold: float, *, with_stat: bool, runs: int, warm_up: int
) -> List[str]:
    cmd: List[str] = [
        str(fmha_exe),
        *_dtype_flag(cfg.dtype),
        "-b",
        str(cfg.batch),
        "-h",
        str(cfg.num_heads_q),
        "-d",
        str(cfg.head_size),
        "-s",
        str(cfg.seq_len_kv),
        "-runs",
        str(runs),
        "-warm-up-runs",
        str(warm_up),
        "-skip-checks",
    ]
    if cfg.seq_len_q != cfg.seq_len_kv:
        # Packed-QKV layout requires q == kv length; use contiguous Q + KV
        # for decode-style shapes.
        cmd += ["-s-q", str(cfg.seq_len_q), "-contiguous-q-kv"]
    cmd += _mask_flag(cfg.mask)
    if cfg.num_heads_kv != cfg.num_heads_q:
        cmd += ["-gqa", str(cfg.num_heads_kv)]
    if threshold > 0:
        cmd += ["-skip-softmax-threshold-scale-factor", f"{threshold:g}"]
    if with_stat and threshold > 0:
        cmd += ["-skip-softmax-stat"]
    return cmd


def _run_once(cmd: List[str]) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """Returns (elapsed_us, (skipped, total)). Either may be None if missing."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(f"fmha.exe failed (code {proc.returncode}):\n{out}\n")
        return None, None
    elapsed_us: Optional[float] = None
    skip_pair: Optional[Tuple[int, int]] = None
    m = ELAPSED_RE.search(out)
    if m:
        elapsed_us = float(m.group(1))
    s = SKIP_RE.search(out)
    if s:
        skipped = int(s.group(1))
        total = int(s.group(2))
        skip_pair = (skipped, total)
    return elapsed_us, skip_pair


def pass1_sparsity(cfgs: List[FmhaConfig], fmha_exe: Path, warm_up: int) -> List[dict]:
    rows: List[dict] = []
    for cfg in cfgs:
        for thr in cfg.threshold_sweep:
            cmd = _build_cmd(fmha_exe, cfg, thr, with_stat=True, runs=1, warm_up=warm_up)
            print(f"[pass1] {cfg.name} threshold={thr:g} ...", flush=True)
            _elapsed, skip = _run_once(cmd)
            if thr > 0 and (skip is None or skip[1] == 0):
                raise RuntimeError(
                    f"{cfg.name} threshold={thr:g}: -skip-softmax-stat "
                    f"produced no parsed counters, cmd={cmd}"
                )
            row = {
                "config": cfg.name,
                "dtype": cfg.dtype,
                "batch": cfg.batch,
                "num_heads_q": cfg.num_heads_q,
                "num_heads_kv": cfg.num_heads_kv,
                "head_size": cfg.head_size,
                "seq_len_q": cfg.seq_len_q,
                "seq_len_kv": cfg.seq_len_kv,
                "mask": cfg.mask,
                "threshold_scale_factor": thr,
                "skipped_blocks": skip[0] if skip else None,
                "total_blocks": skip[1] if skip else None,
                "achieved_sparsity": (skip[0] / skip[1] if skip and skip[1] else None),
            }
            rows.append(row)
    return rows


def pass2_speedup(cfgs: List[FmhaConfig], fmha_exe: Path, runs: int, warm_up: int) -> List[dict]:
    rows: List[dict] = []
    for cfg in cfgs:
        baseline_us: Optional[float] = None
        for thr in cfg.threshold_sweep:
            # Median over 3 outer reps, fmha.exe internally averages over `runs`.
            samples: List[float] = []
            for _ in range(3):
                cmd = _build_cmd(fmha_exe, cfg, thr, with_stat=False, runs=runs, warm_up=warm_up)
                elapsed, _ = _run_once(cmd)
                if elapsed is not None:
                    samples.append(elapsed)
            if not samples:
                continue
            elapsed_us = statistics.median(samples)
            if thr == 0:
                baseline_us = elapsed_us
            speedup = (baseline_us / elapsed_us) if baseline_us and elapsed_us else None
            print(
                f"[pass2] {cfg.name} threshold={thr:g} {elapsed_us:.2f}us speedup={speedup:.3f}x"
                if speedup is not None
                else f"[pass2] {cfg.name} threshold={thr:g} {elapsed_us:.2f}us baseline",
                flush=True,
            )
            rows.append(
                {
                    "config": cfg.name,
                    "dtype": cfg.dtype,
                    "threshold_scale_factor": thr,
                    "elapsed_us_median": elapsed_us,
                    "speedup": speedup,
                }
            )
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        sys.stderr.write(f"No rows for {path}\n")
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=("llm", "diffusion", "both"), default="both")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fmha-exe", type=Path, default=DEFAULT_FMHA_EXE)
    ap.add_argument(
        "--runs", type=int, default=20, help="inner-loop timed runs per shell invocation"
    )
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--skip-pass1", action="store_true")
    ap.add_argument("--skip-pass2", action="store_true")
    args = ap.parse_args()

    if not args.fmha_exe.exists():
        sys.stderr.write(
            f"fmha.exe not found at {args.fmha_exe}. Build it first via "
            f"`cd cpp/kernels/fmha_v2 && make fmha.exe` (after the wheel "
            f"build runs setup.py to generate kernel sources).\n"
        )
        return 2

    cfgs: List[FmhaConfig] = []
    if args.config in ("llm", "both"):
        cfgs += llm_configs()
    if args.config in ("diffusion", "both"):
        cfgs += diffusion_configs()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.skip_pass1:
        rows1 = pass1_sparsity(cfgs, args.fmha_exe, warm_up=args.warm_up)
        write_csv(rows1, args.out_dir / "pass1_sparsity.csv")
    if not args.skip_pass2:
        rows2 = pass2_speedup(cfgs, args.fmha_exe, runs=args.runs, warm_up=args.warm_up)
        write_csv(rows2, args.out_dir / "pass2_speedup.csv")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

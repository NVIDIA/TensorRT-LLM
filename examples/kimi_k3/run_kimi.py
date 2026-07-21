#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper for submitting Kimi K3 ("golden prairie") SLURM jobs.

Usage (from anywhere; the repo is located from this file's path):
    examples/kimi_k3/run_kimi.py sanity          # 4-GPU quick pipeline check (~10 min)
    examples/kimi_k3/run_kimi.py sanity-full     # 16-GPU full quality check (~40 min)
    examples/kimi_k3/run_kimi.py gsm8k           # 16-GPU GSM8K eval (~3 hr)

Any extra args are forwarded to sbatch (e.g. --time=01:00:00).

Path resolution — everything defaults to a "workspace" directory expected to
contain the model weights, the exisiting_optimization_work checkout, and the
container image, laid out as siblings of this repo checkout:

    <workspace>/
      tekit-golden-prairie/               <- this repo (any name works)
      goldenprairie-final-weights_vv1/    <- KIMI_K3_MODEL_DIR
      exisiting_optimization_work/        <- KIMI_K3_OPT_WORK_DIR
      *.sqsh                              <- IMAGE (must be exactly one match)

Every path can be overridden individually via the environment:
    KIMI_K3_WORKSPACE      workspace root      (default: parent of this repo)
    KIMI_K3_MODEL_DIR      HF checkpoint       (default: <workspace>/goldenprairie-final-weights_vv1)
    KIMI_K3_OPT_WORK_DIR   optimization work   (default: <workspace>/exisiting_optimization_work)
    KIMI_K3_CACHE_DIR      JIT/HF cache        (default: <repo>/.kimi_k3_cache)
    IMAGE                  container .sqsh     (default: sole *.sqsh in <workspace>)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

MODES = {
    "sanity": {
        "help": "4-GPU pipeline-only check (~10 min, single tray)",
        "script": "sanity_kimi_k3.sbatch",
        "log": "k3-sanity-%j.log",
        "env": {"KIMI_K3_TP": "4", "KIMI_K3_NUM_LAYERS_OVERRIDE": "4"},
        "sbatch_args": ["-N1", "--gpus=4"],
    },
    "sanity-full": {
        "help": "16-GPU full quality check (~40 min)",
        "script": "sanity_kimi_k3.sbatch",
        "log": "k3-sanity-%j.log",
        "env": {},
        "sbatch_args": [],
    },
    "gsm8k": {
        "help": "16-GPU GSM8K eval (~3 hr)",
        "script": "run_gsm8k.sbatch",
        "log": "k3-gsm8k-%j.log",
        "env": {},
        "sbatch_args": ["--time=03:00:00"],
    },
}


def resolve_image(workspace: Path) -> str:
    """Default IMAGE to the sole .sqsh in the workspace, else demand one."""
    candidates = sorted(workspace.glob("*.sqsh"))
    if len(candidates) == 1:
        return str(candidates[0])
    reason = ("no .sqsh found" if not candidates else
              f"{len(candidates)} .sqsh files found: "
              + ", ".join(p.name for p in candidates))
    sys.exit(f"error: cannot infer container image in {workspace} ({reason}); "
             "set IMAGE explicitly")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode", choices=MODES,
        help="; ".join(f"{k}: {v['help']}" for k, v in MODES.items()))
    parser.add_argument(
        "sbatch_args", nargs="*",
        help="extra args forwarded to sbatch (e.g. --time=01:00:00)")
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="resolve paths and print the sbatch command without submitting")
    args = parser.parse_args()
    mode = MODES[args.mode]

    env = os.environ.copy()
    workspace = Path(env.get("KIMI_K3_WORKSPACE", REPO.parent))
    env.setdefault("KIMI_K3_MODEL_DIR",
                   str(workspace / "goldenprairie-final-weights_vv1"))
    env.setdefault("KIMI_K3_OPT_WORK_DIR",
                   str(workspace / "exisiting_optimization_work"))
    env.setdefault("KIMI_K3_CACHE_DIR", str(REPO / ".kimi_k3_cache"))
    if "IMAGE" not in env:
        env["IMAGE"] = resolve_image(workspace)
    env.update(mode["env"])
    env["REPO"] = str(REPO)

    # Preflight: every input path must exist before we burn queue time.
    for key in ("KIMI_K3_OPT_WORK_DIR", "KIMI_K3_MODEL_DIR", "IMAGE"):
        if not Path(env[key]).exists():
            print(f"error: {key}={env[key]} does not exist", file=sys.stderr)
            return 1
    snapshot_data = Path(env["KIMI_K3_OPT_WORK_DIR"]) / "trtllmgen_MOE/flashinfer/data/csrc"
    if not snapshot_data.exists():
        print(f"error: {snapshot_data} missing — set up the flashinfer snapshot "
              "per trtllmgen_MOE/SNAPSHOT_SETUP.md (3rdparty clones + "
              "flashinfer/data symlinks)", file=sys.stderr)
        return 1
    Path(env["KIMI_K3_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    cmd = ["sbatch", "--export=ALL", *mode["sbatch_args"], *args.sbatch_args,
           str(REPO / "examples" / "kimi_k3" / mode["script"])]
    print(f"[{args.mode}] {mode['help']}")
    for key in ("KIMI_K3_MODEL_DIR", "KIMI_K3_OPT_WORK_DIR",
                "KIMI_K3_CACHE_DIR", "IMAGE"):
        print(f"  {key}={env[key]}")
    print(f"  submitting: {' '.join(cmd)}")
    if args.dry_run:
        print("  (dry run — not submitted)")
        return 0

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        print(f"  job id:  {job_id}")
        print(f"  monitor: squeue -j {job_id}")
        print(f"  log:     {REPO / mode['log'].replace('%j', job_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

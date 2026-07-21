#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper for submitting Kimi K3 ("golden prairie") SLURM jobs.

Usage (from anywhere; the repo is located from this file's path):
    examples/kimi_k3/run_kimi.py sanity          # 4-GPU quick pipeline check (~10 min)
    examples/kimi_k3/run_kimi.py sanity-full     # 16-GPU full quality check (~40 min)
    examples/kimi_k3/run_kimi.py gsm8k           # 16-GPU GSM8K eval (~3 hr)

Any extra args are forwarded to sbatch (e.g. --time=01:00:00).
--dry-run resolves paths and prints the sbatch command without submitting.

Configuration — job inputs are read from ~/.config/kimi-bringup.ini
(override the location with KIMI_BRINGUP_CONFIG). To set it up:

    cp examples/kimi_k3/kimi-bringup.ini.example ~/.config/kimi-bringup.ini
    $EDITOR ~/.config/kimi-bringup.ini    # point `workspace` at your dir

The minimal config is a single `workspace` path containing the three inputs
under their standard names; see the example file for per-input overrides.
Resolution order for each input:

    environment variable  >  config file key  >  derived from `workspace`

    input          env var               config key    workspace default
    ------------   -------------------   -----------   -----------------------------------
    checkpoint     KIMI_K3_MODEL_DIR     model_dir     <ws>/goldenprairie-final-weights_vv1
    opt work       KIMI_K3_OPT_WORK_DIR  opt_work_dir  <ws>/exisiting_optimization_work
    container      IMAGE                 image         sole *.sqsh in <ws>
    JIT/HF cache   KIMI_K3_CACHE_DIR     cache_dir     <repo>/.kimi_k3_cache
"""

import argparse
import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(
    os.environ.get("KIMI_BRINGUP_CONFIG", "~/.config/kimi-bringup.ini")
).expanduser()

SETUP_HINT = f"""\
No usable Kimi K3 configuration found.

Set up the config file (one-time):

    cp {REPO}/examples/kimi_k3/kimi-bringup.ini.example {CONFIG_PATH}
    $EDITOR {CONFIG_PATH}    # point `workspace` at your directory

The workspace directory must contain the HF checkpoint
(goldenprairie-final-weights_vv1/), the exisiting_optimization_work checkout,
and the container image (*.sqsh); see the example file for per-input
overrides and run_kimi.py --help for the full resolution rules."""

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


def load_config() -> dict:
    """Return the [paths] section of the config file as a dict ('' if absent)."""
    if not CONFIG_PATH.is_file():
        return {}
    parser = configparser.ConfigParser()
    try:
        parser.read(CONFIG_PATH)
    except configparser.Error as e:
        sys.exit(f"error: cannot parse {CONFIG_PATH}: {e}")
    return dict(parser["paths"]) if parser.has_section("paths") else {}


def infer_image(workspace: Path) -> str | None:
    candidates = sorted(workspace.glob("*.sqsh"))
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        print(f"error: {len(candidates)} .sqsh files in {workspace} "
              f"({', '.join(p.name for p in candidates)}); set `image` in "
              f"{CONFIG_PATH} or IMAGE in the environment", file=sys.stderr)
        sys.exit(1)
    return None


def resolve_inputs() -> dict:
    """Resolve all job inputs: env var > config key > workspace-derived."""
    config = load_config()
    workspace = os.environ.get("KIMI_K3_WORKSPACE") or config.get("workspace")
    ws = Path(workspace).expanduser() if workspace else None

    def pick(env_var, config_key, derive):
        value = os.environ.get(env_var) or config.get(config_key)
        if not value and ws is not None:
            value = derive(ws)
        return str(Path(value).expanduser()) if value else None

    inputs = {
        "KIMI_K3_MODEL_DIR": pick(
            "KIMI_K3_MODEL_DIR", "model_dir",
            lambda w: w / "goldenprairie-final-weights_vv1"),
        "KIMI_K3_OPT_WORK_DIR": pick(
            "KIMI_K3_OPT_WORK_DIR", "opt_work_dir",
            lambda w: w / "exisiting_optimization_work"),
        "IMAGE": pick("IMAGE", "image", infer_image),
        "KIMI_K3_CACHE_DIR": pick(
            "KIMI_K3_CACHE_DIR", "cache_dir", lambda w: None)
            or str(REPO / ".kimi_k3_cache"),
    }

    missing = [k for k, v in inputs.items() if not v]
    if missing:
        print(f"error: unresolved input(s): {', '.join(missing)}\n\n"
              f"{SETUP_HINT}", file=sys.stderr)
        sys.exit(1)
    return inputs


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

    inputs = resolve_inputs()
    env = os.environ.copy()
    env.update(inputs)
    env.update(mode["env"])
    env["REPO"] = str(REPO)

    # Preflight: every input path must exist before we burn queue time.
    ok = True
    for key in ("KIMI_K3_OPT_WORK_DIR", "KIMI_K3_MODEL_DIR", "IMAGE"):
        if not Path(env[key]).exists():
            print(f"error: {key}={env[key]} does not exist", file=sys.stderr)
            ok = False
    if not ok:
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
    for key, value in inputs.items():
        print(f"  {key}={value}")
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

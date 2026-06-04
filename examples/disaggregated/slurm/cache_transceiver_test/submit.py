#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Submit the 2-node KV cache transceiver bandwidth / UCX-tuning job.

Usage:
    python3 submit.py -c config.yaml [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER_PREFIX = "<"


def parse_args():
    ap = argparse.ArgumentParser(description="Submit KV cache transceiver test")
    ap.add_argument("-c", "--config", required=True, help="Path to config YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and print the sbatch command without submitting")
    return ap.parse_args()


def _is_placeholder(v):
    return isinstance(v, str) and v.strip().startswith(PLACEHOLDER_PREFIX)


def validate(cfg):
    errors = []

    slurm = cfg.get("slurm", {})
    for key in ("partition", "account", "job_time", "job_name"):
        if not slurm.get(key) or _is_placeholder(slurm.get(key)):
            errors.append(f"slurm.{key} must be set")

    env = cfg.get("environment", {})
    if not env.get("container_image") or _is_placeholder(env.get("container_image")):
        errors.append("environment.container_image must be set")
    if not env.get("work_dir") or _is_placeholder(env.get("work_dir")):
        errors.append("environment.work_dir must be set")
    if env.get("build_wheel") and (not env.get("trtllm_repo")
                                   or _is_placeholder(env.get("trtllm_repo"))):
        errors.append("environment.trtllm_repo must be set when build_wheel is true")

    n = cfg["hardware"]["gpus_per_node"]
    par = cfg["parallel"]
    if par["ctx_tp"] * par["ctx_pp"] != n:
        errors.append(f"ctx_tp*ctx_pp ({par['ctx_tp']}*{par['ctx_pp']}) != gpus_per_node ({n})")
    if par["gen_tp"] * par["gen_pp"] != n:
        errors.append(f"gen_tp*gen_pp ({par['gen_tp']}*{par['gen_pp']}) != gpus_per_node ({n})")
    if par["ctx_tp"] != par["gen_tp"] or par["ctx_pp"] != par["gen_pp"]:
        errors.append("layout must be symmetric (ctx_tp==gen_tp and ctx_pp==gen_pp) "
                      "for the harness's local verification")

    kv = cfg["kv_cache"]
    if kv["num_kv_heads"] % par["ctx_tp"] != 0:
        errors.append(f"num_kv_heads ({kv['num_kv_heads']}) must be divisible by "
                      f"tensor parallel size ({par['ctx_tp']})")
    if kv["dtype"].upper() not in ("FP8", "HALF", "BF16"):
        errors.append(f"kv_cache.dtype must be FP8|HALF|BF16, got {kv['dtype']}")
    for rl in cfg["test_matrix"]["request_lengths"]:
        if rl > kv["max_tokens_in_buffer"]:
            errors.append(f"request_length {rl} > max_tokens_in_buffer "
                          f"({kv['max_tokens_in_buffer']})")

    # RequestID encoding (shared with report.decode_rid):
    #   rid = case_idx*1_000_000 + reqlen_idx*10_000 + request_index
    # decode_rid recovers reqlen_idx as (rid//10_000) % 100 and request_index as
    # rid % 10_000, so these fields must not overflow their strides or rids
    # alias across cases and bandwidth buckets get silently cross-contaminated.
    tm = cfg["test_matrix"]
    if len(tm["request_lengths"]) > 100:
        errors.append(f"len(request_lengths) ({len(tm['request_lengths'])}) "
                      f"must be <= 100 (RequestID encoding limit)")
    reqs_per_len = tm.get("warmup_requests", 0) + tm.get("num_requests_per_length", 0)
    if reqs_per_len >= 10_000:
        errors.append(f"warmup_requests + num_requests_per_length ({reqs_per_len}) "
                      f"must be < 10000 (RequestID encoding limit)")

    for combination in (cfg["test_matrix"].get("combinations") or cfg["test_matrix"]["combos"]):
        rt = combination.get("runtime")
        be = combination.get("backend")
        if rt not in ("CPP", "PYTHON"):
            errors.append(f"combination runtime must be CPP|PYTHON, got {rt}")
        if be not in ("DEFAULT", "UCX", "NIXL", "MOONCAKE", "MPI"):
            errors.append(f"combination backend invalid: {be}")
        if rt == "PYTHON" and be not in ("DEFAULT", "NIXL"):
            errors.append(f"Python runtime only supports NIXL/DEFAULT, got {be}")

    if not cfg.get("ucx_env_sweep"):
        errors.append("ucx_env_sweep must contain at least one entry")

    run = cfg.get("run", {})
    per_cell = run.get("timeout_per_cell_s", 60)
    max_sweep = run.get("max_sweep_s", 300)
    if max_sweep <= per_cell:
        errors.append(f"run.max_sweep_s ({max_sweep}) must be > "
                      f"run.timeout_per_cell_s ({per_cell}) so a per-cell "
                      f"hang is caught (and recorded) before the sweep cap")

    if errors:
        print("Config validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Defaults.
    cfg.setdefault("hardware", {}).setdefault("gpus_per_node", 4)
    env = cfg.setdefault("environment", {})
    env.setdefault("container_mount", "")
    env.setdefault("trtllm_repo", "")
    env.setdefault("trtllm_wheel_path", "")
    env.setdefault("build_wheel", False)
    env.setdefault("cuda_architectures", "")
    cfg.setdefault("run", {}).setdefault("timeout_per_cell_s", 60)
    cfg["run"].setdefault("max_sweep_s", 180)
    cfg["run"].setdefault("capture_proto_info", True)

    validate(cfg)

    work_dir = env["work_dir"]
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
    resolved = os.path.join(work_dir, "resolved_config.json")
    with open(resolved, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Resolved config: {resolved}")

    n = cfg["hardware"]["gpus_per_node"]
    slurm = cfg["slurm"]
    sbatch_cmd = [
        "sbatch",
        f"--partition={slurm['partition']}",
        f"--account={slurm['account']}",
        f"--time={slurm['job_time']}",
        f"--job-name={slurm['job_name']}",
        "--nodes=2",
        f"--ntasks-per-node={n}",
        f"--gres=gpu:{n}",
        # Keep the batch-level log in work_dir/logs alongside the per-rank logs,
        # CSVs and results.json. Omit --error so sbatch merges stderr into
        # --output (one combined ctt-%j.log). Overrides launch.slurm's #SBATCH
        # --output/--error, which cannot expand work_dir at parse time.
        f"--output={os.path.join(work_dir, 'logs', 'ctt-%j.log')}",
    ]
    for extra in slurm.get("extra_sbatch", []) or []:
        sbatch_cmd.append(extra)
    # Reset TMPDIR to a path that exists on the compute nodes. --export=ALL
    # propagates the submitting shell's TMPDIR, which may point at a login-node
    # -only temp dir; enroot/pyxis (GNU parallel) then fails to import the image
    # with "Parent directory does not exist". A later assignment in --export
    # overrides the ALL-inherited value.
    exports = f"ALL,TMPDIR=/tmp,CTT_CONFIG={resolved},CTT_SCRIPT_DIR={SCRIPT_DIR}"
    sbatch_cmd.append(f"--export={exports}")
    sbatch_cmd.append(os.path.join(SCRIPT_DIR, "launch.slurm"))

    print("sbatch command:")
    print("  " + " ".join(sbatch_cmd))

    if args.dry_run:
        print("(dry-run: not submitting)")
        return

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

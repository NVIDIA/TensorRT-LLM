#!/bin/bash
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
#
# run_workloads.sh - Drive the BOLT profiling workload suite.
#
# Iterates the workloads in a suite YAML and runs each one's `command` with the
# workload fields exported as WL_* env vars. Workloads without a `command` are
# treated as documentation-only placeholders and skipped with a notice.
#
# This indirection keeps cluster/benchmark specifics OUT of the repo: the
# reference suite documents WHAT to run, and you supply HOW via either a
# per-workload `command` in the YAML or a single reusable template in
# BOLT_WORKLOAD_CMD (a shell snippet run for every enabled workload).
#
# Usage:
#   run_workloads.sh <suite.yaml> [--dry-run]
#
# Env:
#   BOLT_WORKLOAD_CMD   shell command run per workload (overrides YAML `command`).
#                       Reads WL_MODEL/WL_GPU/WL_ISL/WL_OSL/WL_CONC/WL_MTP/...
#   PYTHON              python interpreter (default: python3)

set -euo pipefail

# Exported so per-workload `command:`s can locate sibling scripts (e.g. the
# smoke workload) regardless of the caller's working directory.
export BOLT_WORKLOADS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUITE="${1:?usage: run_workloads.sh <suite.yaml> [--dry-run]}"
shift || true
DRY_RUN=0
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=1; done

PYTHON="${PYTHON:-python3}"

[[ -f "$SUITE" ]] || { echo "[ERROR] Suite not found: $SUITE" >&2; exit 1; }

# Emit one shell-eval-able line per enabled workload: a set of WL_* assignments.
# Uses PyYAML if available, else a minimal fallback parser is not attempted --
# PyYAML ships in the TRT-LLM environment.
mapfile -t WL_LINES < <("$PYTHON" - "$SUITE" <<'PY'
import sys, shlex
import yaml

with open(sys.argv[1]) as f:
    doc = yaml.safe_load(f) or {}

defaults = doc.get("defaults", {}) or {}
for wl in doc.get("workloads", []) or []:
    if not wl.get("enabled", True):
        continue
    isl, _, osl = str(wl.get("isl_osl", "/")).partition("/")
    fields = {
        "WL_NAME": wl.get("name", ""),
        "WL_MODEL": wl.get("model", ""),
        "WL_GPU": wl.get("gpu", ""),
        "WL_PRECISION": wl.get("precision", defaults.get("precision", "")),
        "WL_ISL": isl,
        "WL_OSL": osl,
        "WL_CONC": wl.get("concurrency", ""),
        "WL_MTP": wl.get("mtp", ""),
        "WL_MODE": wl.get("mode", ""),
        "WL_CTX_PARALLEL": wl.get("ctx_parallel", ""),
        "WL_GEN_BACKEND": wl.get("gen_backend", ""),
        "WL_GEN_PARALLEL": wl.get("gen_parallel", ""),
        "WL_NODES": wl.get("nodes", ""),
        "WL_MODEL_PATH": wl.get("model_path", ""),
        "WL_TP": wl.get("tp", ""),
        "WL_REASON": wl.get("reason", ""),
        "WL_COMMAND": wl.get("command", ""),
    }
    print(" ".join(f"{k}={shlex.quote(str(v))}" for k, v in fields.items()))
PY
)

[[ ${#WL_LINES[@]} -eq 0 ]] && { echo "[WARNING] No enabled workloads in $SUITE"; exit 0; }

echo "[INFO] $(basename "$SUITE"): ${#WL_LINES[@]} enabled workload(s)"
ran=0 skipped=0
for line in "${WL_LINES[@]}"; do
    # Reset then assign WL_* for this workload.
    unset "${!WL_@}" 2>/dev/null || true
    eval "$line"
    # Export them so the workload command (run as a child `bash -c`) inherits
    # WL_MODEL_PATH / WL_ISL / WL_OSL / WL_TP etc.
    [[ -n "${!WL_@}" ]] && export ${!WL_@}

    cmd="${BOLT_WORKLOAD_CMD:-${WL_COMMAND:-}}"
    echo "----------------------------------------------------------------"
    echo "[INFO] Workload: ${WL_NAME}  (${WL_MODEL} ${WL_GPU} ${WL_ISL}/${WL_OSL} conc=${WL_CONC} ${WL_MODE})"
    echo "[INFO]   ${WL_REASON}"

    if [[ -z "$cmd" ]]; then
        echo "[NOTICE] No command for '${WL_NAME}' -- placeholder, skipping."
        echo "[NOTICE]   Supply BOLT_WORKLOAD_CMD or a per-workload 'command:' to run it."
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] $cmd"
        continue
    fi

    echo "[INFO] Running: $cmd"
    bash -c "$cmd"
    ran=$((ran + 1))
done

echo "================================================================"
echo "[INFO] Workloads run: $ran, placeholders skipped: $skipped"
[[ "$DRY_RUN" -eq 0 && "$ran" -eq 0 ]] && \
    echo "[WARNING] No workloads actually ran -- no .fdata will be produced. Provide commands."
exit 0

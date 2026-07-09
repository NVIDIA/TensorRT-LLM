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
# run_local.sh - Local / customer entry point for the TRT-LLM BOLT flow.
#
# Runs the entire instrument -> profile -> merge -> optimize loop on a single
# node, with NO NVIDIA network access. The only external dependency is a public
# `llvm-bolt` toolchain (see README for install) plus a TensorRT-LLM install
# that was built with ENABLE_BOLT_COMPATIBLE=ON.
#
# Typical end-to-end use (re-BOLT the current install against your workloads):
#
#   export FDATA_OUTPUT_DIR=$PWD/bolt_fdata        # where .fdata is collected
#   scripts/bolt/run_local.sh instrument           # swap in instrumented libs
#   scripts/bolt/run_local.sh profile              # run the workload suite
#   scripts/bolt/run_local.sh merge                # per-PID -> per-lib + YAML
#   scripts/bolt/run_local.sh optimize             # BOLT + install optimized
#
# Or all at once (instrument -> profile -> merge -> optimize):
#   scripts/bolt/run_local.sh all
#
# To consume a profile bundle that already ships in the container (offline
# re-BOLT without profiling), skip instrument/profile/merge and run:
#   scripts/bolt/run_local.sh optimize --profiles /opt/trtllm/bolt
#
# Commands:
#   preflight    check llvm-bolt + that tensorrt/tensorrt_llm import (run first)
#   instrument   setup + back up in-scope ELFs + install instrumented copies
#   profile      run the workload driver against the (instrumented) install
#   merge        merge per-PID .fdata into per-lib .fdata + .yaml
#   optimize     run llvm-bolt with profiles and install the optimized libs
#   all          instrument -> profile -> merge -> optimize
#   reset        remove stage markers and work dirs (start over)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (overridable via env or flags).
export BOLT_WORK_DIR="${BOLT_WORK_DIR:-/opt/trtllm_bolt}"
export FDATA_OUTPUT_DIR="${FDATA_OUTPUT_DIR:-${BOLT_WORK_DIR}/fdata}"
PROFILES_DIR=""          # for `optimize`: where to read profiles from
WORKLOAD_SUITE="${WORKLOAD_SUITE:-${SCRIPT_DIR}/workloads/suite_v0_2.yaml}"

usage() { sed -n '17,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---- arg parsing ----------------------------------------------------------
CMD="${1:-}"; shift || true
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)   export BOLT_WORK_DIR="$2"; shift 2 ;;
        --fdata-dir)  export FDATA_OUTPUT_DIR="$2"; shift 2 ;;
        --profiles)   PROFILES_DIR="$2"; shift 2 ;;
        --workloads)  WORKLOAD_SUITE="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# shellcheck source=scripts/bolt/bolt_lib.sh
source "${SCRIPT_DIR}/bolt_lib.sh"

case "$CMD" in
    instrument)
        bolt_run_stages setup_directories backup_libraries \
                        link_fdata_output_dir instrument_libraries \
                        install_instrumented_libraries
        log_success "Instrumented libraries installed. Now run: $0 profile"
        ;;
    profile)
        log_info "Running workload suite: $WORKLOAD_SUITE"
        log_info "Profiles will be written to: $FDATA_OUTPUT_DIR"
        "${SCRIPT_DIR}/workloads/run_workloads.sh" "$WORKLOAD_SUITE" "${EXTRA_ARGS[@]}"
        ;;
    merge)
        bolt_run_stages merge_fdata_files
        ;;
    optimize)
        [[ -n "$PROFILES_DIR" ]] && export FDATA_INPUT_DIR="$PROFILES_DIR"
        # `optimize` needs the original (non-instrumented) ELFs in
        # $BOLT_WORK_DIR/original. If they aren't there yet (e.g. offline
        # re-BOLT from a shipped bundle), back them up first.
        [[ -f "$BOLT_WORK_DIR/library_paths.txt" ]] || \
            bolt_run_stages setup_directories backup_libraries
        bolt_run_stages optimize_libraries install_optimized_libraries
        ;;
    all)
        bolt_run_stages setup_directories backup_libraries \
                        link_fdata_output_dir instrument_libraries \
                        install_instrumented_libraries
        "${SCRIPT_DIR}/workloads/run_workloads.sh" "$WORKLOAD_SUITE" "${EXTRA_ARGS[@]}"
        bolt_run_stages merge_fdata_files optimize_libraries install_optimized_libraries
        ;;
    preflight)
        # Verify the environment can actually run the BOLT flow before the
        # slow instrument/profile steps. Catches the common failure modes:
        # missing llvm-bolt, and a TRT-LLM install that can't import (broken
        # tensorrt / missing deps). Run from /tmp to avoid source-tree shadowing.
        ok=1
        command -v llvm-bolt   >/dev/null 2>&1 || { log_error "llvm-bolt not on PATH"; ok=0; }
        command -v merge-fdata >/dev/null 2>&1 || { log_error "merge-fdata not on PATH"; ok=0; }
        if ! ( cd /tmp && "${PYTHON:-python3}" -c "import tensorrt, tensorrt_llm" 2>/dev/null ); then
            log_error "import of tensorrt/tensorrt_llm failed."
            log_error "  - container must be runtime-capable (devel/release), not a bare base"
            log_error "  - install the wheel with --no-deps and pin tensorrt (see README 'Running under SLURM')"
            ok=0
        fi
        [[ $ok -eq 1 ]] && log_success "preflight OK" || { log_error "preflight FAILED"; exit 1; }
        ;;
    reset)
        log_warning "Removing $BOLT_WORK_DIR"
        rm -rf "$BOLT_WORK_DIR"
        ;;
    ""|-h|--help)
        usage
        ;;
    *)
        log_error "Unknown command: $CMD"
        usage
        exit 1
        ;;
esac

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
# perf_instrument_hook.sh - BOLT POST_INSTALL_HOOK for the perf-sanity harness.
#
# The perf-sanity harness (jenkins/scripts/perf) runs this once per node, on the
# localid-0 task, AFTER it installs the TRT-LLM wheel and BEFORE any worker rank
# loads the libs (see slurm_install.sh's locked section). It swaps in
# BOLT-instrumented copies of the in-scope TRT-LLM libs, so the ensuing
# perf-sanity workload (agg trtllm-serve, or disagg ctx/gen workers) emits .fdata
# that BOLT can later merge. The fdata output path is baked into the instrumented
# libs here (via run_local.sh's link_fdata_output_dir), so workers dump to it
# automatically -- no per-worker env injection needed.
#
# Wired in via the harness's generic POST_INSTALL_HOOK seam; all BOLT specifics
# live here so the perf harness stays BOLT-agnostic. Config via env (forwarded by
# submit.py's EXTRA_CONTAINER_EXPORTS):
#   BOLT_FDATA_DIR    (required) run-level shared dir; this node writes <dir>/<host>
#   BOLT_LLVM_DIR     (optional) dir to stage/reuse llvm-bolt (default /tmp/bolt-llvm)
#   BOLT_WORK_DIR     (optional) per-node work dir (default /tmp/bolt_work_<jobid>)
#   LLVM_BOLT_VERSION (optional) default 21.1.5

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts/bolt/internal
TOOLKIT="$(dirname "$HERE")"                            # .../scripts/bolt

: "${BOLT_FDATA_DIR:?perf_instrument_hook: BOLT_FDATA_DIR required}"
export BOLT_WORK_DIR="${BOLT_WORK_DIR:-/tmp/bolt_work_${SLURM_JOB_ID:-local}}"
# Per-node fdata dir under the run-level root; slurm_merge.sh fans in <workload>/<host>.
export FDATA_OUTPUT_DIR="${BOLT_FDATA_DIR%/}/$(hostname)"
mkdir -p "$FDATA_OUTPUT_DIR"

# Ensure llvm-bolt / merge-fdata on PATH (self-stage if absent). Race-safe
# extract-then-atomic-rename in case BOLT_LLVM_DIR is a shared path hit by
# multiple nodes.
LLVM_BOLT_VERSION="${LLVM_BOLT_VERSION:-21.1.5}"
BOLT_LLVM_DIR="${BOLT_LLVM_DIR:-/tmp/bolt-llvm}"
if ! command -v llvm-bolt >/dev/null 2>&1; then
    if [ ! -x "$BOLT_LLVM_DIR/bin/llvm-bolt" ]; then
        case "$(uname -m)" in
            aarch64) _a=ARM64 ;;
            x86_64)  _a=X64 ;;
            *) echo "[bolt-hook] unsupported arch $(uname -m)" >&2; exit 1 ;;
        esac
        _tb="LLVM-${LLVM_BOLT_VERSION}-Linux-${_a}.tar.xz"
        _stage="${BOLT_LLVM_DIR}.stage.$$"
        rm -rf "$_stage"; mkdir -p "$_stage"
        echo "[bolt-hook] staging llvm-bolt ${LLVM_BOLT_VERSION} -> $BOLT_LLVM_DIR"
        curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -o "/tmp/${_tb}" \
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_BOLT_VERSION}/${_tb}"
        tar -xJf "/tmp/${_tb}" -C "$_stage" --strip-components=1
        rm -f "/tmp/${_tb}"
        mv -T "$_stage" "$BOLT_LLVM_DIR" 2>/dev/null || rm -rf "$_stage"
    fi
    export PATH="$BOLT_LLVM_DIR/bin:$PATH"
fi
command -v llvm-bolt >/dev/null 2>&1 || { echo "[bolt-hook] llvm-bolt not available" >&2; exit 1; }

echo "[bolt-hook] instrumenting installed TRT-LLM libs on $(hostname); fdata -> $FDATA_OUTPUT_DIR"
bash "$TOOLKIT/run_local.sh" instrument
echo "[bolt-hook] instrumentation complete on $(hostname)"

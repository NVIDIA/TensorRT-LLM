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
# slurm_merge.sh - Cross-workload BOLT merge + package (fan-out step B).
#
# Pairs with the per-workload perf-harness collect runs (the POST_INSTALL_HOOK
# perf_instrument_hook.sh). After N per-workload runs have each written their
# .fdata to $FDATA_ROOT/<workload>/<host>/, this single job:
#   1. Reconstructs the ORIGINAL (pre-instrument) libs by installing the phase-1
#      wheel and backing up the in-scope ELFs (setup_env + backup_libraries; NO
#      instrument -- merge only needs the originals to convert .fdata -> .yaml).
#   2. Gathers EVERY workload's per-host .fdata into one set, tagging the pid
#      field with <workload>-<host> so cross-workload/-node PIDs never collide.
#   3. Merges per-lib, builds the manifest, and packages the promotable bundle.
#
# Submit with: sbatch --nodes=1 scripts/bolt/internal/slurm_merge.sh
#
# Required env (set by BoltProfileGen):
#   WORKSPACE CONTAINER_IMAGE FDATA_ROOT
# Optional: TOOLKIT_HOST BUILDS_HOST MODELS_ROOT BOLT_REF TRIPLE TARBALL_NAME
#           OUT_DIR LLVM_BOLT_VERSION

# ====================== EDIT THESE (cluster specifics) ======================
# NOTE: this merge job is 100% CPU
# (setup_env pip, readelf, merge-fdata, llvm-bolt .fdata->.yaml conversion,
# packaging) and needs NO GPUs -- requesting gpus-per-node=4 holds a full GB200
# node for nothing and queues behind GPU demand. Switch to a CPU partition (or
# --gpus-per-node=0) once we confirm the target cluster accepts GPU-less jobs.
#SBATCH --account=coreai_tensorrt_ci
#SBATCH --job-name=bolt-merge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00

set -euo pipefail

WORKSPACE="${WORKSPACE:?slurm_merge: WORKSPACE required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?slurm_merge: CONTAINER_IMAGE required}"
FDATA_ROOT="${FDATA_ROOT:?slurm_merge: FDATA_ROOT required (holds <workload>/<host>/*.fdata)}"
TOOLKIT_HOST="${TOOLKIT_HOST:-$WORKSPACE/toolkit}"
BUILDS_HOST="${BUILDS_HOST:-$WORKSPACE/builds}"
MODELS_ROOT="${MODELS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models}"
RUNDIR="${RUNDIR:-$WORKSPACE/runs/merge_$SLURM_JOB_ID}"
OUT_DIR="${OUT_DIR:-$FDATA_ROOT/_bundle}"   # where the bundle + manifest land

BOLT_REF="${BOLT_REF:-$(git -C "$TOOLKIT_HOST" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
case "$(uname -m)" in
    aarch64) TRIPLE="${TRIPLE:-aarch64-linux-gnu}" ;;
    x86_64)  TRIPLE="${TRIPLE:-x86_64-linux-gnu}" ;;
    *)       TRIPLE="${TRIPLE:-$(uname -m)-linux-gnu}" ;;
esac

TARBALL_NAME="${TARBALL_NAME:-TensorRT-LLM-GH200.tar.gz}"
CONTAINER_NAME="bolt_merge_${SLURM_JOB_ID}"
GATHERED="$FDATA_ROOT/_merged"
# Merge only needs the toolkit, the phase-1 tarball (+llvm), and the fdata root.
MOUNTS="$TOOLKIT_HOST:/workspace/bolt,$BUILDS_HOST:/builds,$FDATA_ROOT:$FDATA_ROOT,$RUNDIR:$RUNDIR"

mkdir -p "$GATHERED" "$OUT_DIR" "$RUNDIR"
echo "[INFO] merge: FDATA_ROOT=$FDATA_ROOT  REF=$BOLT_REF  TRIPLE=$TRIPLE  OUT=$OUT_DIR"

# Enroot layer cache: reuse the imported container image across jobs instead of
# re-importing. Overridable; matches normal CI.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/home/svc_tensorrt/.cache/enroot}"

# ---- CI self-staging: llvm-bolt --------------------------------------------
LLVM_BOLT_VERSION="${LLVM_BOLT_VERSION:-21.1.5}"
if [ ! -x "$BUILDS_HOST/llvm/bin/llvm-bolt" ]; then
    echo "[INFO] Installing llvm-bolt ${LLVM_BOLT_VERSION} -> $BUILDS_HOST/llvm"
    case "$(uname -m)" in
        aarch64) LLVM_ARCH="ARM64" ;;
        x86_64)  LLVM_ARCH="X64" ;;
        *)       echo "[ERROR] unsupported arch for llvm-bolt" >&2; exit 1 ;;
    esac
    _tb="LLVM-${LLVM_BOLT_VERSION}-Linux-${LLVM_ARCH}.tar.xz"
    # Race-safe extract-then-atomic-rename (shared BUILDS_HOST across jobs).
    _stage="$BUILDS_HOST/.llvm.stage.${SLURM_JOB_ID:-$$}"
    rm -rf "$_stage"; mkdir -p "$_stage"
    curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 \
        -o "/tmp/${_tb}" \
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_BOLT_VERSION}/${_tb}"
    tar -xJf "/tmp/${_tb}" -C "$_stage" --strip-components=1
    rm -f "/tmp/${_tb}"
    mv -T "$_stage" "$BUILDS_HOST/llvm" 2>/dev/null || rm -rf "$_stage"
fi

# ---------------------------------------------------------------------------
# Single in-container step: reconstruct originals -> gather -> merge -> package
# ---------------------------------------------------------------------------
srun --ntasks=1 --ntasks-per-node=1 --nodes=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-name="$CONTAINER_NAME" \
     --container-mounts="$MOUNTS" \
     --container-workdir=/workspace \
     --no-container-mount-home \
     bash -lc '
        set -euo pipefail
        export PATH=/builds/llvm/bin:$PATH
        export BOLT_WORK_DIR='"$RUNDIR"'/work

        # 1) Reconstruct the ORIGINAL libs: install the wheel + back up ELFs.
        #    No instrument -- the collect jobs already produced the .fdata.
        _extract='"$RUNDIR"'/extract
        mkdir -p "$_extract" && tar xzf /builds/'"$TARBALL_NAME"' -C "$_extract" 2>/dev/null || true
        /workspace/bolt/setup_env.sh "$_extract"
        source /workspace/bolt/bolt_lib.sh
        bolt_run_stages setup_directories backup_libraries

        # 2) Gather every workload'\''s per-host .fdata, tagging the pid field with
        #    <workload>-<host> so cross-workload/-node PIDs never collide and the
        #    merge glob (<lib_base>.*.fdata) still matches.
        gathered='"$GATHERED"'
        mkdir -p "$gathered"
        wls=""              # names of workloads that actually produced .fdata
        shopt -s nullglob
        for wldir in '"$FDATA_ROOT"'/*/; do
            wl=$(basename "$wldir")
            case "$wl" in _merged|_bundle) continue ;; esac
            wl_had_fdata=0
            for hostdir in "$wldir"*/; do
                h=$(basename "$hostdir")
                for f in "$hostdir"*.fdata; do
                    [ -e "$f" ] || continue
                    wl_had_fdata=1
                    base=$(basename "$f"); name=${base%.fdata}
                    pid=${name##*.}          # <pid>
                    libbase=${name%.*}       # <lib_base> (dots preserved)
                    cp "$f" "$gathered/${libbase}.${wl}-${h}-${pid}.fdata"
                done
            done
            # Record only workloads that contributed profiles, so the manifest
            # reflects what was ACTUALLY profiled (not the full suite declaration).
            [ "$wl_had_fdata" -eq 1 ] && wls="$wls${wls:+,}$wl"
        done
        shopt -u nullglob   # scope nullglob to the gather loop: merge_fdata_files
                            # relies on no-match globs yielding the literal pattern.
        echo "[INFO] workloads profiled: $wls"
        n=$(ls "$gathered" 2>/dev/null | wc -l)
        echo "[INFO] gathered $n per-(workload,host) .fdata file(s)"
        [ "$n" -gt 0 ] || { echo "[ERROR] no .fdata gathered from '"$FDATA_ROOT"'" >&2; exit 1; }

        # 3) Merge + convert to YAML against the reconstructed originals.
        FDATA_INPUT_DIR="$gathered" bolt_run_stages merge_fdata_files
        echo "[INFO] merged YAML profiles:"; ls -la "$gathered"/*.yaml 2>/dev/null || echo "[WARN] none"

        # 4) Manifest + package the promotable bundle.
        python3 /workspace/bolt/manifest.py build \
            --work-dir "$BOLT_WORK_DIR" --profiles "$gathered" \
            --ref "'"$BOLT_REF"'" --workloads "$wls" \
            -o "$gathered/manifest.json"
        /workspace/bolt/internal/artifactory.sh package \
            "$gathered" "'"$BOLT_REF"'" "'"$TRIPLE"'" "'"$OUT_DIR"'"
     '

BUNDLE="$OUT_DIR/bolt-profile-${BOLT_REF}-${TRIPLE}.tar.gz"
echo "[INFO] Done. Merged profiles under $GATHERED"
if [ -f "$BUNDLE" ]; then
    echo "[INFO] Promotable bundle: $BUNDLE"
else
    echo "[ERROR] Bundle not produced ($BUNDLE) -- merge failed" >&2
    exit 1
fi

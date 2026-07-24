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
# bolt_lib.sh - Self-contained LLVM BOLT engine for TensorRT-LLM native libraries.
#
# This is the single source of truth for the TRT-LLM BOLT recipe (instrument /
# merge / optimize / install). It depends only on a public `llvm-bolt` toolchain
# and a TensorRT-LLM install tree -- NOT on custom-build-flavours, JET, or any
# NVIDIA-internal network. External developers can therefore run the full flow
# offline to re-BOLT their own build against their own workloads.
#
# NOTE(license): the stage structure here is adapted from the NVIDIA-internal
# custom-build-flavours `bolt_prepare/env.sh`. Whether to keep this vendored copy
# as the long-term source of truth, depend on the upstream repo, or replace it is
# an open decision to settle before GA.
#
# Usage:
#   source bolt_lib.sh
#   bolt_run_stages setup_directories backup_libraries instrument_libraries ...
#
# Or drive individual stages directly after sourcing.
#
# Stages (run in order; each is idempotent via a .<stage>.done marker):
#   setup_directories              create work dirs under $BOLT_WORK_DIR
#   backup_libraries               discover + back up in-scope TRT-LLM ELFs
#   link_fdata_output_dir          symlink $FDATA_OUTPUT_DIR -> $BOLT_WORK_DIR/fdata
#   instrument_libraries           llvm-bolt -instrument on backed-up libs
#   install_instrumented_libraries copy instrumented libs over originals
#   merge_fdata_files              merge per-PID .fdata -> per-lib .fdata + .yaml
#   optimize_libraries             llvm-bolt with merged profiles
#   install_optimized_libraries    copy optimized libs over originals

set -eo pipefail

BOLT_WORK_DIR="${BOLT_WORK_DIR:-/opt/trtllm_bolt}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
# Logs go to stderr so functions that emit data on stdout (e.g.
# find_tensorrt_llm_libs, which prints library paths consumed by a `while read`
# loop) are never polluted by log lines.
log_info()    { echo "[INFO] $1" >&2; }
log_success() { echo "[SUCCESS] $1" >&2; }
log_warning() { echo "[WARNING] $1" >&2; }
log_error()   { echo "[ERROR] $1" >&2; }

# ---------------------------------------------------------------------------
# Library discovery
#
# Scope (per design doc):
#   P0: libtensorrt_llm.so, libnvinfer_plugin_tensorrt_llm.so, libth_common.so
#   P1: Python bindings (.so), Triton libtriton_tensorrtllm.so, trtllmExecutorWorker
#
# Unlike the internal env.sh, paths are NOT hard-coded to python3.12/aarch64:
# the tensorrt_llm package dir is resolved at runtime, and libs/ is scanned.
# Extra targets can be injected via BOLT_EXTRA_LIBS (newline- or space-separated
# absolute paths).
# ---------------------------------------------------------------------------

# Resolve the installed tensorrt_llm package directory WITHOUT importing it.
# importlib.util.find_spec only locates the package; it does not execute the
# package __init__ (which pulls in torch/bindings and can fail at runtime,
# especially after a --no-deps wheel install). We only need the on-disk path to
# the bundled libs/, so a successful import is unnecessary.
_trtllm_pkg_dir() {
    local py="${PYTHON:-python3}"
    "$py" - <<'PY' 2>/dev/null || true
import importlib.util, os
try:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec and spec.origin:
        print(os.path.dirname(os.path.abspath(spec.origin)))
except Exception:
    pass
PY
}

# Print the set of candidate in-scope ELF paths, one per line (real paths).
find_tensorrt_llm_libs() {
    local -A seen=()
    local pkg
    pkg="$(_trtllm_pkg_dir)"

    _emit() {
        local p="$1"
        [[ -f "$p" ]] || return 0
        local real
        real="$(readlink -f "$p" 2>/dev/null || echo "$p")"
        [[ -n "$real" && -z "${seen[$real]:-}" ]] || return 0
        seen[$real]=1
        echo "$real"
    }

    if [[ -n "$pkg" ]]; then
        # P0: the three compute-heavy native libs -- the ONLY libs/*.so we
        # instrument. Deliberately NOT `libs/*.so*`: that glob also pulls in
        # out-of-scope bundled libs -- notably the KV-transfer wrappers
        # (libtensorrt_llm_{nixl,ucx,mooncake}_wrapper.so). Instrumenting the
        # transport wrappers is both off-design AND has been observed to stall
        # disaggregated gen-worker bring-up (the gen server never reaches ready,
        # so the disagg router waits forever). Keep the set tight per design.
        local f g
        for f in libtensorrt_llm libnvinfer_plugin_tensorrt_llm libth_common; do
            for g in "$pkg"/libs/"$f".so*; do _emit "$g"; done
        done
        # P1: Python bindings (name encodes interpreter + arch, so glob it).
        for f in "$pkg"/bindings*.so; do _emit "$f"; done
        # P1: Triton backend + executor worker, if present in this layout.
        for f in "$pkg"/libs/libtriton_tensorrtllm.so* "$pkg"/bin/trtllmExecutorWorker; do _emit "$f"; done
    else
        log_warning "tensorrt_llm not importable (PYTHON=${PYTHON:-python3}); relying on BOLT_EXTRA_LIBS only"
    fi

    # Caller-supplied extra targets (absolute paths).
    if [[ -n "${BOLT_EXTRA_LIBS:-}" ]]; then
        local extra
        for extra in $BOLT_EXTRA_LIBS; do _emit "$extra"; done
    fi
}

# A shared library/binary is BOLT-applicable only if it carries relocations
# (.rela.text), which the phase-1 ENABLE_BOLT_COMPATIBLE=ON build emits.
is_bolt_applicable() {
    local lib="$1"
    local sections
    sections="$(readelf -S "$lib" 2>/dev/null)" || true
    echo "$sections" | grep -q '\.rela\.text'
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
setup_directories() {
    log_info "Setting up BOLT working directories under $BOLT_WORK_DIR"
    mkdir -p "$BOLT_WORK_DIR"/{original,instrumented,fdata,bolted}
    log_success "Working directories created"
}

backup_libraries() {
    log_info "Discovering and backing up in-scope TensorRT-LLM libraries"

    local lib_map_file="$BOLT_WORK_DIR/library_paths.txt"
    : > "$lib_map_file"

    local found=0 skipped=0
    local -A seen_base=()
    local lib
    while IFS= read -r lib; do
        [[ -z "$lib" ]] && continue
        if [[ ! -f "$lib" ]]; then
            log_warning "$lib not found, skipping"
            continue
        fi
        if ! is_bolt_applicable "$lib"; then
            log_warning "Skipping $lib (no .rela.text -- not built with ENABLE_BOLT_COMPATIBLE=ON?)"
            skipped=$((skipped + 1))
            continue
        fi
        local base
        base="$(basename "$lib")"
        if [[ -n "${seen_base[$base]:-}" ]]; then
            if [[ "${seen_base[$base]}" != "$lib" ]]; then
                log_error "Duplicate basename '$base' maps to '${seen_base[$base]}' and '$lib'"
                exit 1
            fi
            continue
        fi
        seen_base[$base]="$lib"
        cp "$lib" "$BOLT_WORK_DIR/original/"
        echo "$base:$lib" >> "$lib_map_file"
        log_info "Backed up: $lib"
        found=$((found + 1))
    done < <(find_tensorrt_llm_libs)

    if [[ $skipped -gt 0 ]]; then
        log_warning "Skipped $skipped libraries without .rela.text"
    fi
    if [[ $found -eq 0 ]]; then
        log_error "No BOLT-compatible TensorRT-LLM libraries found."
        log_error "Ensure the install was built with ENABLE_BOLT_COMPATIBLE=ON."
        exit 1
    fi
    log_success "Backed up $found library(ies); map at $lib_map_file"
}

# Instrumented libraries hard-code their fdata output to $BOLT_WORK_DIR/fdata.
# Symlink that to a caller-chosen (e.g. host-mounted / shared-FS) directory so
# profiles can be collected off the container.
link_fdata_output_dir() {
    [[ -z "${FDATA_OUTPUT_DIR:-}" ]] && return 0
    log_info "Linking $BOLT_WORK_DIR/fdata -> $FDATA_OUTPUT_DIR"
    mkdir -p "$FDATA_OUTPUT_DIR"
    rm -rf "$BOLT_WORK_DIR/fdata"
    ln -s "$FDATA_OUTPUT_DIR" "$BOLT_WORK_DIR/fdata"
}

instrument_libraries() {
    log_info "Instrumenting libraries for profiling"
    command -v llvm-bolt >/dev/null 2>&1 || { log_error "llvm-bolt not found"; return 1; }

    local fdata_dir="$BOLT_WORK_DIR/fdata"
    local max_jobs="${BOLT_PARALLEL_JOBS:-1}"
    mkdir -p "$BOLT_WORK_DIR/instrumented/"

    local libs=()
    local lib
    for lib in "$BOLT_WORK_DIR/original/"*.so*; do
        [[ -f "$lib" ]] && libs+=("$lib")
    done
    local total=${#libs[@]}
    [[ $total -eq 0 ]] && { log_error "No backed-up libraries to instrument"; return 1; }

    local fail_marker
    fail_marker="$(mktemp)"; rm -f "$fail_marker"

    local pids=() idx=0
    for lib in "${libs[@]}"; do
        idx=$((idx + 1))
        local base; base="$(basename "$lib")"
        (
            log_info "[$idx/$total] Instrumenting $base"
            if ! llvm-bolt "$lib" -instrument \
                    --instrumentation-file-append-pid \
                    --instrumentation-file="$fdata_dir/${base%.so}" \
                    -o "$BOLT_WORK_DIR/instrumented/$base" 2>&1; then
                echo "$base" >> "$fail_marker"
            fi
        ) &
        pids+=($!)
        if [[ ${#pids[@]} -ge $max_jobs ]]; then
            wait "${pids[0]}"; pids=("${pids[@]:1}")
        fi
    done
    wait "${pids[@]}" 2>/dev/null || true

    if [[ -s "$fail_marker" ]]; then
        log_error "Instrumentation failed for: $(tr '\n' ' ' < "$fail_marker")"
        rm -f "$fail_marker"
        return 1
    fi
    rm -f "$fail_marker"
    log_success "Instrumented $total library(ies)"
}

install_instrumented_libraries() {
    _install_from_dir "$BOLT_WORK_DIR/instrumented" "instrumented"
}

install_optimized_libraries() {
    _install_from_dir "$BOLT_WORK_DIR/bolted" "optimized"
}

# Copy every *.so* in $1 back to its original path from the mapping file.
_install_from_dir() {
    local src_dir="$1" label="$2"
    local lib_map_file="$BOLT_WORK_DIR/library_paths.txt"
    [[ -f "$lib_map_file" ]] || { log_error "Missing $lib_map_file"; exit 1; }

    local installed=0 lib
    for lib in "$src_dir/"*.so*; do
        [[ -f "$lib" ]] || continue
        local base; base="$(basename "$lib")"
        local orig
        orig="$(grep "^${base}:" "$lib_map_file" | head -1 | cut -d: -f2-)"
        if [[ -n "$orig" ]]; then
            cp "$lib" "$orig"
            log_info "Installed $label: $base -> $orig"
            installed=$((installed + 1))
        else
            log_warning "No original path for $base, skipping"
        fi
    done
    [[ $installed -eq 0 ]] && { log_error "No $label libraries installed"; exit 1; }
    log_success "Installed $installed $label library(ies)"
}

merge_fdata_files() {
    log_info "Merging per-PID profiling data"
    command -v merge-fdata >/dev/null 2>&1 || { log_error "merge-fdata not found"; return 1; }

    # FDATA_INPUT_DIR lets callers merge a dir other than the (symlinked)
    # per-run fdata dir -- e.g. the multi-node collector merging a gathered,
    # host-tagged cross-node set. Merged .fdata + .yaml are written there too.
    local fdata_dir="${FDATA_INPUT_DIR:-$BOLT_WORK_DIR/fdata}"
    [[ -d "$fdata_dir" ]] || { log_error "Missing $fdata_dir"; return 1; }
    log_info "Contents of $fdata_dir:"; ls -la "$fdata_dir/" || true

    local merged=0 lib
    for lib in "$BOLT_WORK_DIR/original/"*.so*; do
        [[ -f "$lib" ]] || continue
        local base; base="$(basename "$lib")"
        local lib_base="${base%.so}"
        local parts=( "$fdata_dir/${lib_base}".*.fdata )
        # ${parts[0]:-} (not ${parts[0]}): under `set -u`, if the caller enabled
        # `nullglob` the array is EMPTY when a lib has no .fdata, and a bare
        # ${parts[0]} would abort with "unbound variable". The :- makes the
        # no-profile case fall through to the warning below regardless of glob settings.
        if [[ -e "${parts[0]:-}" ]]; then
            log_info "Merging ${#parts[@]} profile(s) for $base"
            if merge-fdata "${parts[@]}" -o "$fdata_dir/${lib_base}.fdata"; then
                rm -f "${parts[@]}"
                merged=$((merged + 1))
            else
                log_error "merge-fdata failed for $base"
                return 1
            fi
        else
            log_warning "No profiling data for $base"
        fi
    done
    [[ $merged -eq 0 ]] && { log_error "No profiles merged. Did the workload run against instrumented libs?"; return 1; }

    # Convert to rich YAML profiles (preferred by optimize; drift-tolerant).
    command -v llvm-bolt >/dev/null 2>&1 || { log_error "llvm-bolt not found for YAML conversion"; return 1; }
    for lib in "$BOLT_WORK_DIR/original/"*.so*; do
        [[ -f "$lib" ]] || continue
        local base; base="$(basename "$lib")"
        local lib_base="${base%.so}"
        local merged_fdata="$fdata_dir/${lib_base}.fdata"
        [[ -s "$merged_fdata" ]] || continue
        log_info "Converting $base fdata -> YAML"
        llvm-bolt "$lib" -data="$merged_fdata" -w "$fdata_dir/${lib_base}.yaml" -o /dev/null
    done
    log_success "Merged + converted profiles for $merged library(ies)"
}

optimize_libraries() {
    log_info "Optimizing libraries with BOLT"
    command -v llvm-bolt >/dev/null 2>&1 || { log_error "llvm-bolt not found"; return 1; }

    local profile_dir="${FDATA_INPUT_DIR:-$BOLT_WORK_DIR/fdata}"
    [[ -d "$profile_dir" ]] || { log_error "Missing profile dir $profile_dir"; return 1; }
    log_info "Using profiles from: $profile_dir"

    # NOTE: The rich profile-quality / staleness decomposition lives in the
    # internal custom-build-flavours analysis layer and is applied on top of
    # this engine in CI. Here we surface BOLT's own -report-stale + -dyno-stats
    # output, which is enough for a local re-BOLT.
    local optimized=0 lib
    for lib in "$BOLT_WORK_DIR/original/"*.so*; do
        [[ -f "$lib" ]] || continue
        local base; base="$(basename "$lib")"
        local lib_base="${base%.so}"

        local profile=""
        if [[ -s "$profile_dir/${lib_base}.yaml" ]]; then
            profile="$profile_dir/${lib_base}.yaml"
        elif [[ -s "$profile_dir/${lib_base}.fdata" ]]; then
            log_warning "No YAML for $base, falling back to .fdata"
            profile="$profile_dir/${lib_base}.fdata"
        fi

        if [[ -z "$profile" ]]; then
            log_warning "No profile for $base; copying original unmodified"
            cp "$lib" "$BOLT_WORK_DIR/bolted/$base"
            continue
        fi

        log_info "Optimizing $base with $(basename "$profile")"
        if llvm-bolt "$lib" -o "$BOLT_WORK_DIR/bolted/$base" \
                -data="$profile" \
                -lite \
                -infer-stale-profile \
                -report-stale \
                -reorder-blocks=ext-tsp \
                -reorder-functions=hfsort \
                -split-functions \
                -split-all-cold \
                -split-eh \
                -dyno-stats; then
            log_success "Optimized: $base"
            optimized=$((optimized + 1))
        else
            log_error "BOLT failed for $base; copying original as fallback"
            cp "$lib" "$BOLT_WORK_DIR/bolted/$base"
        fi
    done
    [[ $optimized -eq 0 ]] && { log_error "No libraries optimized"; return 1; }
    log_success "Optimized $optimized library(ies)"
}

# ---------------------------------------------------------------------------
# Stage runner (idempotent via .<stage>.done markers)
# ---------------------------------------------------------------------------
bolt_run_stages() {
    mkdir -p "$BOLT_WORK_DIR"
    local stage
    for stage in "$@"; do
        if [[ -f "$BOLT_WORK_DIR/.$stage.done" ]]; then
            log_info "Stage $stage already done; skipping"
            continue
        fi
        log_info "==> Running stage: $stage"
        "$stage" || { log_error "Stage $stage failed"; return 1; }
        touch "$BOLT_WORK_DIR/.$stage.done"
    done
}

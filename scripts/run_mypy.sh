#!/usr/bin/env bash
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

# Wrapper for mypy invoked by the pre-commit type-check hook.
#
# Runs the diagnostic allowlist configured in pyproject.toml, excluding the
# paths, raw names, and resolved import names in scripts/mypy_exclusion.json,
# then reporting only errors absent from the exact diagnostic entries in
# scripts/mypy_baseline.txt. When compiled
# TensorRT-LLM bindings are present (bindings.*.so), also permits automatic
# type-stub installation. Without bindings, keeps the same diagnostics and
# disables automatic stub installation.
# The check is advisory: write the result to mypy_final_report.txt and return
# success to the caller even when mypy or baseline comparison fails. Print the
# final filtered report to stdout when the check finishes.
# Checks changed product Python files by default; explicit paths can be passed
# by pre-commit or on the command line. Pass --full-project to scan all product
# Python files. Pass --update-baseline with --full-project to replace the saved
# entries with the current full scan. Pass --diff-base REF (or set
# MYPY_DIFF_BASE=REF) to keep only diagnostics on added lines relative to REF.
# Changed-file checks default to HEAD; pass --no-diff-filter for file-level output.
#
# Set MYPY_REQUIRE_BINDINGS=1 to record a failed check when the compiled
# bindings cannot be imported.
#
# Importing the bindings needs no GPU, but it does need libcuda.so.1 to resolve;
# on a driverless machine this script falls back to the CUDA stub library (see
# ensure_cuda_driver_stub below).
#
# TODO(TRTLLM-15310): evaluate a stronger type checker than mypy. mypy is the
# reference implementation of the typing spec, not the most capable checker:
# pyright has better inference and narrowing, fuller generics/ParamSpec/overload
# handling, and is considerably faster. ty (Astral, aligns with the ruff already
# used here) and pyrefly (Meta) target the same niche and are worth re-checking
# for maturity. The [tool.mypy] files in pyproject.toml cover the product tree,
# so a candidate can run over the same set as a non-blocking stage
# first to see what it catches that mypy does not.

set -euo pipefail

BASELINE_ACTION="check"
FULL_PROJECT=0
DIFF_BASE="${MYPY_DIFF_BASE:-}"
DIFF_FILTER=1
while (( $# > 0 )); do
    case "$1" in
        --update-baseline)
            BASELINE_ACTION="record"
            shift
            ;;
        --full-project)
            FULL_PROJECT=1
            shift
            ;;
        --diff-base)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "ERROR: --diff-base requires a Git ref" >&2
                exit 2
            fi
            DIFF_BASE="$2"
            shift 2
            ;;
        --no-diff-filter)
            DIFF_FILTER=0
            shift
            ;;
        *)
            break
            ;;
    esac
done
if (( ! FULL_PROJECT && DIFF_FILTER )) && [[ -z "$DIFF_BASE" ]]; then
    DIFF_BASE="HEAD"
fi
if (( DIFF_FILTER == 0 )) && [[ -n "$DIFF_BASE" ]]; then
    echo "ERROR: --no-diff-filter cannot be combined with --diff-base or MYPY_DIFF_BASE" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_FILE="${MYPY_REPORT_PATH:-$PROJECT_DIR/mypy_final_report.txt}"
STUB_DIR=""
MYPY_OUTPUT=""
MYPY_CONFIG_DIR=""
cleanup_stub_dir() {
    local check_status=$?
    if [[ -n "$STUB_DIR" ]]; then
        rm -rf "$STUB_DIR" || true
    fi
    if [[ -n "$MYPY_OUTPUT" ]]; then
        rm -f "$MYPY_OUTPUT" || true
    fi
    if [[ -n "$MYPY_CONFIG_DIR" ]]; then
        rm -rf "$MYPY_CONFIG_DIR" || true
    fi
    printf '\n# Check status: %s\n' "$check_status"
    cat "$REPORT_FILE" >&3 || true
    printf 'mypy report: %s (check status: %s)\n' "$REPORT_FILE" "$check_status" >&3 || true
    trap - EXIT
    exit 0
}
printf '%s\n' \
    '# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.' \
    '# SPDX-License-Identifier: Apache-2.0' \
    '# Command: scripts/run_mypy.sh' \
    '' > "$REPORT_FILE"
exec 3>&1
exec >> "$REPORT_FILE" 2>&1
trap cleanup_stub_dir EXIT

EXCLUSIONS_CONFIG="$SCRIPT_DIR/mypy_exclusion.json"
BASELINE_FILE="$SCRIPT_DIR/mypy_baseline.txt"
cd "$PROJECT_DIR"
EXCLUDE_REGEX="$(python3 "$SCRIPT_DIR/mypy_exclusion.py" exclusions \
    --project-root "$PROJECT_DIR" "$EXCLUSIONS_CONFIG")"

if (( ! FULL_PROJECT )) && [[ "$BASELINE_ACTION" == "record" ]]; then
    echo "ERROR: --update-baseline requires --full-project" >&2
    exit 2
fi
if [[ -n "$DIFF_BASE" && "$BASELINE_ACTION" == "record" ]]; then
    echo "ERROR: --update-baseline cannot be combined with diff-line filtering" >&2
    exit 2
fi
if (( FULL_PROJECT && $# > 0 )); then
    echo "ERROR: --full-project does not accept file paths" >&2
    exit 2
fi
if (( ! FULL_PROJECT && $# == 0 )); then
    CHANGED_PATHS_FILE="$(mktemp)"
    if ! git diff --name-only -z HEAD -- tensorrt_llm > "$CHANGED_PATHS_FILE" ||
        ! git ls-files --others --exclude-standard -z -- tensorrt_llm >> "$CHANGED_PATHS_FILE"; then
        rm -f "$CHANGED_PATHS_FILE"
        echo "ERROR: could not discover changed files" >&2
        exit 2
    fi
    changed_paths=()
    while IFS= read -r -d '' path; do
        changed_paths+=("$path")
    done < "$CHANGED_PATHS_FILE"
    rm -f "$CHANGED_PATHS_FILE"
    set -- "${changed_paths[@]+"${changed_paths[@]}"}"
fi

FILTERED_ARGS=()
KEPT_SOURCES=0
for argument in "$@"; do
    if [[ "$argument" == --* ]]; then
        echo "ERROR: unknown option: $argument" >&2
        exit 2
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != tensorrt_llm/* ]]; then
        echo "Skipping non-product mypy input: $argument"
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != *.py && "$argument" != *.pyi ]]; then
        echo "Skipping non-Python mypy input: $argument"
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ ! -f "$argument" ]]; then
        echo "Skipping missing mypy input: $argument"
        continue
    fi
    if [[ "$argument" == *.py || "$argument" == *.pyi ]]; then
        if python3 "$SCRIPT_DIR/mypy_exclusion.py" exclusions --project-root "$PROJECT_DIR" \
            --matches "$argument" "$EXCLUSIONS_CONFIG"; then
            echo "Skipping excluded mypy source: $argument"
            continue
        fi
        KEPT_SOURCES=$((KEPT_SOURCES + 1))
    fi
    FILTERED_ARGS+=("$argument")
done
if (( ! FULL_PROJECT && KEPT_SOURCES == 0 )); then
    echo "No eligible changed Python files to check"
    exit 0
fi
set -- "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}"
if (( ! FULL_PROJECT )); then
    printf 'Checking %s changed product Python file(s):\n' "$KEPT_SOURCES"
    printf '  %s\n' "$@"
fi

MYPY_CONFIG_DIR="$(mktemp -d)"
MYPY_CONFIG="$MYPY_CONFIG_DIR/mypy.toml"
cat "$PROJECT_DIR/pyproject.toml" > "$MYPY_CONFIG"
if (( ! FULL_PROJECT )); then
    python3 "$SCRIPT_DIR/mypy_exclusion.py" exclusions --explicit-sources-only "$MYPY_CONFIG"
fi
python3 "$SCRIPT_DIR/mypy_exclusion.py" exclusions --import-overrides \
    --project-root "$PROJECT_DIR" "$EXCLUSIONS_CONFIG" >> "$MYPY_CONFIG"

# Make the bindings loadable on a machine with no CUDA driver installed.
#
# bindings.*.so links against the CUDA driver API (the cuMem*/cuMulticast*
# virtual-memory and NVLink-multicast paths), so it carries a DT_NEEDED on
# libcuda.so.1. DT_NEEDED is resolved when the loader maps the library, before
# any of its code runs, so dlopen fails outright where no driver is installed —
# CPU build nodes, for instance. Nothing on the import path calls a driver
# function or needs a device; only symbol resolution does. The CUDA toolkit
# ships a stub libcuda.so for exactly this case, so point the loader at it when
# no real driver is present. Verified on a driverless node: with the stub the
# bindings module imports and initializes cleanly, and torch correctly reports
# zero devices.
ensure_cuda_driver_stub() {
    if python3 -c "import ctypes; ctypes.CDLL('libcuda.so.1')" 2>/dev/null; then
        return  # A real driver is installed; leave the loader path alone.
    fi
    local stub=""
    local candidate
    for candidate in /usr/local/cuda*/targets/*/lib/stubs/libcuda.so \
                     /usr/local/cuda*/lib64/stubs/libcuda.so; do
        # Unmatched globs stay literal; the [[ -f ]] test rejects them.
        if [[ -f "$candidate" ]]; then
            stub="$candidate"
            break
        fi
    done
    if [[ -z "$stub" ]]; then
        return  # Nothing to offer; the import below reports the real error.
    fi
    # A fresh owner-only directory per invocation, not a fixed path: everything
    # on LD_LIBRARY_PATH gets loaded into this process, so a predictable
    # directory under a shared /tmp would let any local user pre-create it and
    # plant their own libcuda.so.1. Removed on exit by the trap above.
    # The stub must be visible under its SONAME, libcuda.so.1.
    STUB_DIR="$(mktemp -d)"
    ln -s "$stub" "$STUB_DIR/libcuda.so.1"
    export LD_LIBRARY_PATH="$STUB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "No CUDA driver on this machine; using the stub at $stub so the" \
         "compiled bindings can be loaded for type checking"
}

ensure_cuda_driver_stub

# What the gate below does and does not establish.
#
# `import tensorrt_llm.bindings` is one coarse check standing in for several
# conditions that fail independently. Worth naming, because "bindings importable"
# reads narrower than what is actually being relied on:
#
#   - the extension was built at all;
#   - the compiled artifacts agree with each other — a bindings.*.so built
#     against a different libth_common.so fails here with an undefined symbol.
#     This is the main reason the check is an import rather than a file test;
#   - the runtime dependencies are installed, so mypy reads real inline types
#     (torch and friends) instead of collapsing them to Any.
#
# It establishes nothing about the .pyi stubs — which are what mypy actually
# consumes, since mypy never imports anything. The bindings stubs are generated
# at build time (tensorrt_llm/bindings/*.pyi) and are not checked in, so a
# stub-generation regression would leave this gate green while the checked code
# is silently compared against a missing or stale API. Asserting stub presence
# separately would close that hole; it is deliberately left as a follow-up
# rather than folded into this check, since the two catch disjoint failures.
if python3 -c 'import tensorrt_llm.bindings'; then
    echo "Compiled bindings detected — running configured mypy checks"
    MYPY_ARGS=(--exclude "$EXCLUDE_REGEX")
else
    if [[ "${MYPY_REQUIRE_BINDINGS:-0}" -eq 1 ]]; then
        echo "ERROR: MYPY_REQUIRE_BINDINGS=1 but 'import tensorrt_llm.bindings'" \
             "failed (traceback above)." >&2
        # The traceback is authoritative; these are hints for the two failures
        # that look alike from the outside (never built vs. built but unloadable).
        if compgen -G "$PROJECT_DIR/tensorrt_llm/bindings*.so" > /dev/null; then
            echo "       note: bindings*.so IS present, so the extension was built" \
                 "and something stopped it from importing. If the traceback names" \
                 "libcuda.so.1, this machine has no CUDA driver and no stub" \
                 "libcuda.so was found under /usr/local/cuda*/**/stubs/." >&2
        else
            echo "       note: no bindings*.so under $PROJECT_DIR/tensorrt_llm," \
                 "so the extension was probably not built." >&2
        fi
        exit 1
    fi
    echo "No compiled bindings — running configured mypy checks without stub installation"
    # Missing imports remain visible through the configured import-not-found code.
    MYPY_ARGS=(--exclude "$EXCLUDE_REGEX" --no-install-types --interactive)
fi
if (( ! FULL_PROJECT )); then
    # Analyze only explicit sources; imported implementation modules become Any.
    MYPY_ARGS+=(--follow-imports=skip)
fi
DIFF_ARGS=()
if [[ -n "$DIFF_BASE" ]]; then
    DIFF_ARGS=(--diff-base "$DIFF_BASE")
fi
PARTIAL_ARGS=()
if (( ! FULL_PROJECT )) || [[ -n "$DIFF_BASE" ]]; then
    PARTIAL_ARGS=(--partial)
fi

MYPY_OUTPUT="$(mktemp)"
MYPY_STATUS=0
mypy --config-file "$MYPY_CONFIG" "${MYPY_ARGS[@]}" \
    --show-error-codes --show-column-numbers --no-pretty \
    --no-color-output "$@" > "$MYPY_OUTPUT" 2>&1 || MYPY_STATUS=$?
if (( ! FULL_PROJECT )); then
    python3 "$SCRIPT_DIR/mypy_exclusion.py" "$BASELINE_ACTION" "$MYPY_OUTPUT" \
        "$BASELINE_FILE" --mypy-exit-code "$MYPY_STATUS" \
        --exclusions-config "$EXCLUSIONS_CONFIG" --project-root "$PROJECT_DIR" \
        "${PARTIAL_ARGS[@]+"${PARTIAL_ARGS[@]}"}" \
        "${DIFF_ARGS[@]+"${DIFF_ARGS[@]}"}"
else
    python3 "$SCRIPT_DIR/mypy_exclusion.py" "$BASELINE_ACTION" "$MYPY_OUTPUT" \
        "$BASELINE_FILE" --mypy-exit-code "$MYPY_STATUS" \
        --exclusions-config "$EXCLUSIONS_CONFIG" --project-root "$PROJECT_DIR" \
        "${PARTIAL_ARGS[@]+"${PARTIAL_ARGS[@]}"}" \
        "${DIFF_ARGS[@]+"${DIFF_ARGS[@]}"}"
fi

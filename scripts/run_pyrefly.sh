#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run Pyrefly on supplied product or test Python files, or the whole project when
# --full-project is used. With no eligible file paths, return success without
# checking. Pre-commit supplies the file list with pass_filenames: true.
# Paths from tool.pyrefly.project-excludes and symbols from
# tool.pyrefly-exclusions in pyproject.toml are removed from the JSON report by
# scripts/pyrefly_exclusion.py. Whole-project checks are always advisory;
# file-list checks are advisory unless --strict is supplied.
# The filtered diagnostic report is written to pyrefly_final_report.json.
# File-list checks print all diagnostics; full-project checks print only the
# report path.
# File-list checks compare with HEAD. Diagnostics are filtered to added and
# modified lines when Git finds them; otherwise they remain at file level. Pass
# --no-diff-filter for file-level output on every supplied file.
# Path and symbol exclusions are enabled by default; use --no-path-module-filter
# to also disable Pyrefly's configured project and import exclusions.

set -euo pipefail

FULL_PROJECT=0
DIFF_BASE=""
DIFF_FILTER=1
PATH_MODULE_FILTER=1
STRICT=0
while (( $# > 0 )); do
    case "$1" in
        --full-project)
            FULL_PROJECT=1
            shift
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Run Pyrefly on supplied product or test Python files, or the whole project.

Pre-commit passes filenames to this script. With no eligible filenames, skip
the check. Use --full-project to check all configured sources explicitly.

Options:
  --full-project           Check all configured product and test Python files.
  --strict                 Fail file-list checks on diagnostics or check failures.
  --no-diff-filter         Keep diagnostics from all lines in supplied files.
  --no-path-module-filter  Include diagnostics from excluded paths and modules.

Path and module exclusions and diff-line filtering are enabled by default.
PYREFLY_REPORT_PATH overrides the report path (default: project root).
EOF
            exit 0
            ;;
        --no-diff-filter)
            DIFF_FILTER=0
            shift
            ;;
        --no-path-module-filter)
            PATH_MODULE_FILTER=0
            shift
            ;;
        *)
            break
            ;;
    esac
done
if (( ! FULL_PROJECT && $# == 0 )); then
    printf 'No file paths supplied; skipping Pyrefly check.\n'
    exit 0
fi
if (( ! FULL_PROJECT && DIFF_FILTER )); then
    DIFF_BASE="HEAD"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXCLUSION_SCRIPT="$SCRIPT_DIR/pyrefly_exclusion.py"
EXCLUSIONS_CONFIG="$PROJECT_DIR/pyproject.toml"
WORKFLOW_SCRIPT="$SCRIPT_DIR/pyrefly_workflow.py"
REPORT_FILE="${PYREFLY_REPORT_PATH:-$PROJECT_DIR/pyrefly_final_report.json}"
if [[ "$REPORT_FILE" != /* ]]; then
    REPORT_FILE="$PROJECT_DIR/$REPORT_FILE"
fi
PYREFLY_OUTPUT=""
PYREFLY_STDERR=""
FILTERED_REPORT=""
DIFF_FILTERED_OUTPUT=""
PYREFLY_CONFIG=""
CANDIDATE_PATHS=""
ALLOWED_PATHS=""
FILTERED_ARG_COUNT=0
SKIP_CHECK=0

cleanup() {
    local status=$?
    for file in "$PYREFLY_OUTPUT" "$PYREFLY_STDERR" "$FILTERED_REPORT" \
                "$DIFF_FILTERED_OUTPUT" "$PYREFLY_CONFIG" \
                "$CANDIDATE_PATHS" "$ALLOWED_PATHS"; do
        if [[ -n "$file" ]]; then
            rm -f "$file" || true
        fi
    done
    if (( SKIP_CHECK )); then
        trap - EXIT
        exit 0
    fi
    if (( FULL_PROJECT )); then
        printf 'Pyrefly full-project report saved to: %s\n' "$REPORT_FILE" >&3
    else
        python3 "$WORKFLOW_SCRIPT" print-diagnostics "$REPORT_FILE" >&3 || true
    fi
    trap - EXIT
    if (( STRICT )); then
        printf 'Pyrefly check exit status: %s\n' "$status" >&3 || true
        exit "$status"
    fi
    if (( FULL_PROJECT )); then
        exit 0
    fi
    printf 'Pyrefly check exit status: %s; advisory script exit status: 0\n' \
        "$status" >&3 || true
    exit 0
}

exec 3>&1
trap cleanup EXIT
cd "$PROJECT_DIR"

if (( FULL_PROJECT && $# > 0 )); then
    printf '{"tool":"pyrefly","report_error":"--full-project does not accept file paths"}\n' \
        > "$REPORT_FILE"
    exit 2
fi
FILTERED_ARGS=()
for argument in "$@"; do
    if [[ "$argument" == --* ]]; then
        printf '{"tool":"pyrefly","report_error":"unknown option: %s"}\n' "$argument" \
            > "$REPORT_FILE"
        exit 2
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != tensorrt_llm/* && "$argument" != tests/* ]]; then
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != *.py && "$argument" != *.pyi ]]; then
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ ! -f "$argument" ]]; then
        continue
    fi
    FILTERED_ARGS+=("$argument")
    FILTERED_ARG_COUNT=$((FILTERED_ARG_COUNT + 1))
done

if (( ! FULL_PROJECT && PATH_MODULE_FILTER && FILTERED_ARG_COUNT > 0 )); then
    CANDIDATE_PATHS="$(mktemp)"
    ALLOWED_PATHS="$(mktemp)"
    printf '%s\0' "${FILTERED_ARGS[@]}" > "$CANDIDATE_PATHS"
    if ! python3 "$EXCLUSION_SCRIPT" filter-input-paths \
        "$CANDIDATE_PATHS" "$ALLOWED_PATHS" "$EXCLUSIONS_CONFIG" \
        --project-root "$PROJECT_DIR"; then
        printf '{"tool":"pyrefly","report_error":"could not filter input paths"}\n' \
            > "$REPORT_FILE"
        exit 2
    fi
    FILTERED_ARGS=()
    while IFS= read -r -d '' path; do
        FILTERED_ARGS+=("$path")
    done < "$ALLOWED_PATHS"
    FILTERED_ARG_COUNT=${#FILTERED_ARGS[@]}
fi

if (( ! FULL_PROJECT && FILTERED_ARG_COUNT == 0 )); then
    SKIP_CHECK=1
    printf 'No eligible supplied product or test Python files; skipping Pyrefly check.\n' >&3
    exit 0
fi
if (( FULL_PROJECT )); then
    STRICT=0
fi

if ! command -v pyrefly >/dev/null 2>&1; then
    printf '{"tool":"pyrefly","report_error":"pyrefly is not installed"}\n' > "$REPORT_FILE"
    exit 127
fi

PYREFLY_OUTPUT="$(mktemp)"
PYREFLY_STDERR="$(mktemp)"
FILTERED_REPORT="$(mktemp)"
PYREFLY_ARGS=(check --output-format=json --summary=none --progress-bar=no
    --relative-to "$PROJECT_DIR")
if (( ! PATH_MODULE_FILTER )); then
    PYREFLY_CONFIG="$(mktemp "$PROJECT_DIR/.pyrefly-config.XXXXXX")"
    python3 "$WORKFLOW_SCRIPT" clear-exclusions-config \
        "$PROJECT_DIR/pyproject.toml" "$PYREFLY_CONFIG"
    PYREFLY_ARGS+=(--config "$PYREFLY_CONFIG")
fi
if (( ! FULL_PROJECT )); then
    PYREFLY_ARGS+=("${FILTERED_ARGS[@]}")
fi

PYREFLY_STATUS=0
pyrefly "${PYREFLY_ARGS[@]}" > "$PYREFLY_OUTPUT" 2> "$PYREFLY_STDERR" || PYREFLY_STATUS=$?

if (( PATH_MODULE_FILTER )); then
    python3 "$EXCLUSION_SCRIPT" filter-report "$PYREFLY_OUTPUT" "$FILTERED_REPORT" \
        "$EXCLUSIONS_CONFIG" --project-root "$PROJECT_DIR" \
        --pyrefly-exit-code "$PYREFLY_STATUS" --stderr "$PYREFLY_STDERR"
else
    python3 "$WORKFLOW_SCRIPT" wrap-unfiltered-report \
        "$PYREFLY_OUTPUT" "$FILTERED_REPORT" "$PYREFLY_STDERR" "$PYREFLY_STATUS"
fi

FINAL_REPORT="$FILTERED_REPORT"
if [[ -n "$DIFF_BASE" ]]; then
    DIFF_FILTERED_OUTPUT="$(mktemp)"
    python3 "$WORKFLOW_SCRIPT" filter-diff-lines \
        "$FILTERED_REPORT" "$DIFF_FILTERED_OUTPUT" "$PROJECT_DIR" "$DIFF_BASE" \
        --files "${FILTERED_ARGS[@]}"
    FINAL_REPORT="$DIFF_FILTERED_OUTPUT"
fi
cp "$FINAL_REPORT" "$REPORT_FILE"
if (( STRICT )); then
    python3 "$WORKFLOW_SCRIPT" report-status "$REPORT_FILE"
    exit 0
fi
exit "$PYREFLY_STATUS"

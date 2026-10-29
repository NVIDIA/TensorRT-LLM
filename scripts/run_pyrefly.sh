#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run Pyrefly on changed product Python files, or the whole project with
# --full-project. The configured Pyrefly baseline suppresses known diagnostics.
# Excluded paths and third-party symbols are removed from the JSON report by
# scripts/pyrefly_exclusion.py. This hook is advisory and always returns success;
# the filtered diagnostic report is written to pyrefly_final_report.json and
# up to eight new diagnostics are printed to stdout when the check finishes.

set -euo pipefail

FULL_PROJECT=0
UPDATE_BASELINE=0
while (( $# > 0 )); do
    case "$1" in
        --full-project)
            FULL_PROJECT=1
            shift
            ;;
        --update-baseline)
            UPDATE_BASELINE=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXCLUSION_SCRIPT="$SCRIPT_DIR/pyrefly_exclusion.py"
EXCLUSIONS_CONFIG="$SCRIPT_DIR/pyrefly_exclusion.json"
REPORT_FILE="${PYREFLY_REPORT_PATH:-$PROJECT_DIR/pyrefly_final_report.json}"
PYREFLY_OUTPUT=""
PYREFLY_STDERR=""
FILTERED_REPORT=""
FILTERED_ARG_COUNT=0

print_report_summary() {
    python3 - "$REPORT_FILE" <<'PY' >&3 || true
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
try:
    report = json.loads(report_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    print(f"Could not read Pyrefly report: {error}")
else:
    if report.get("report_error"):
        print(f"Pyrefly report error: {report['report_error']}")
    result = report.get("result", report)
    errors = result.get("errors", []) if isinstance(result, dict) else []
    if not isinstance(errors, list):
        errors = []
    if not errors:
        print("No new Pyrefly diagnostics.")
    else:
        for error in errors[:8]:
            path = str(error.get("path", "<unknown>"))
            line = error.get("line")
            column = error.get("column")
            location = f"{path}:{line}" if line is not None else path
            if column is not None:
                location += f":{column}"
            kind = error.get("name", "error")
            message = error.get("concise_description") or error.get("description", "")
            print(f"{location}: [{kind}] {str(message).replace(chr(10), ' ')}")
        if len(errors) > 8:
            print(f"Showing 8 of {len(errors)} new diagnostics.")
print(f"See full Pyrefly report at: {report_path}")
PY
}

cleanup() {
    local status=$?
    for file in "$PYREFLY_OUTPUT" "$PYREFLY_STDERR" "$FILTERED_REPORT"; do
        if [[ -n "$file" ]]; then
            rm -f "$file" || true
        fi
    done
    print_report_summary
    printf 'Pyrefly check status: %s\n' "$status" >&3 || true
    trap - EXIT
    exit 0
}

exec 3>&1
trap cleanup EXIT
cd "$PROJECT_DIR"

if (( UPDATE_BASELINE && ! FULL_PROJECT )); then
    printf '{"tool":"pyrefly","report_error":"--update-baseline requires --full-project"}\n' \
        > "$REPORT_FILE"
    exit 2
fi
if (( FULL_PROJECT && $# > 0 )); then
    printf '{"tool":"pyrefly","report_error":"--full-project does not accept file paths"}\n' \
        > "$REPORT_FILE"
    exit 2
fi

if (( ! FULL_PROJECT && $# == 0 )); then
    CHANGED_PATHS_FILE="$(mktemp)"
    if ! git diff --name-only -z HEAD -- tensorrt_llm > "$CHANGED_PATHS_FILE" ||
        ! git ls-files --others --exclude-standard -z -- tensorrt_llm >> "$CHANGED_PATHS_FILE"; then
        rm -f "$CHANGED_PATHS_FILE"
        printf '{"tool":"pyrefly","report_error":"could not discover changed files"}\n' \
            > "$REPORT_FILE"
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
for argument in "$@"; do
    if [[ "$argument" == --* ]]; then
        printf '{"tool":"pyrefly","report_error":"unknown option: %s"}\n' "$argument" \
            > "$REPORT_FILE"
        exit 2
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != tensorrt_llm/* ]]; then
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ "$argument" != *.py && "$argument" != *.pyi ]]; then
        continue
    fi
    if (( ! FULL_PROJECT )) && [[ ! -f "$argument" ]]; then
        continue
    fi
    if [[ "$argument" == *.py || "$argument" == *.pyi ]]; then
        if python3 "$EXCLUSION_SCRIPT" matches "$argument" "$EXCLUSIONS_CONFIG" \
            --project-root "$PROJECT_DIR"; then
            continue
        fi
    fi
    FILTERED_ARGS+=("$argument")
    FILTERED_ARG_COUNT=$((FILTERED_ARG_COUNT + 1))
done

if (( ! FULL_PROJECT && FILTERED_ARG_COUNT == 0 )); then
    printf '{"tool":"pyrefly","pyrefly_exit_code":0,"message":"No eligible changed product Python files to check","errors":[]}\n' \
        > "$REPORT_FILE"
    exit 0
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
if (( UPDATE_BASELINE )); then
    PYREFLY_ARGS+=(--update-baseline)
fi
if (( ! FULL_PROJECT )); then
    PYREFLY_ARGS+=("${FILTERED_ARGS[@]}")
fi

PYREFLY_STATUS=0
pyrefly "${PYREFLY_ARGS[@]}" > "$PYREFLY_OUTPUT" 2> "$PYREFLY_STDERR" || PYREFLY_STATUS=$?
python3 "$EXCLUSION_SCRIPT" filter-report "$PYREFLY_OUTPUT" "$FILTERED_REPORT" \
    "$EXCLUSIONS_CONFIG" --project-root "$PROJECT_DIR" \
    --pyrefly-exit-code "$PYREFLY_STATUS" --stderr "$PYREFLY_STDERR"
cp "$FILTERED_REPORT" "$REPORT_FILE"
exit "$PYREFLY_STATUS"

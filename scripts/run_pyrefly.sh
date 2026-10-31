#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run Pyrefly on changed product Python files, or the whole project with
# --full-project. Add --rebase-upstream to check the current branch (or HEAD
# when detached) in a new worktree rebased on upstream main and emit its diff.
# Excluded paths and third-party symbols are removed from the JSON report by
# scripts/pyrefly_exclusion.py. This hook is advisory and always returns success;
# the filtered diagnostic report is written to pyrefly_final_report.json and
# up to eight diagnostics are printed to stdout when the check finishes.
# Changed-file checks default to HEAD. Pass --full-project to scan all product
# Python files. Diagnostics are filtered to added lines against HEAD by default;
# --rebase-upstream uses its fetched main commit. Pass --no-diff-filter for
# file-level output.
# Path and symbol exclusions are enabled by default; use --no-path-module-filter
# to also disable Pyrefly's configured project and import exclusions.

set -euo pipefail

FULL_PROJECT=0
REBASE_UPSTREAM=0
DIFF_BASE="${PYREFLY_INTERNAL_DIFF_BASE:-}"
DIFF_FILTER=1
PATH_MODULE_FILTER=1
while (( $# > 0 )); do
    case "$1" in
        --full-project)
            FULL_PROJECT=1
            shift
            ;;
        --rebase-upstream)
            REBASE_UPSTREAM=1
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Run Pyrefly on changed product Python files, or the whole project.

Options:
  --full-project           Check all product Python files.
  --rebase-upstream        Rebase a new worktree from the current branch or HEAD
                           on upstream main, report its diff, and check changed files.
  --no-diff-filter         Keep diagnostics from all lines in changed files.
  --no-path-module-filter  Include diagnostics from excluded paths and modules.

Path and module exclusions and diff-line filtering are enabled by default.

Rebase workflow environment:
  Fetches main from git@github.com:NVIDIA/TensorRT-LLM.git, falling back to
  https://github.com/NVIDIA/TensorRT-LLM.git if SSH fetch fails.
  PYREFLY_DIFF_REPORT_PATH      Git diff report directory (default: /tmp)
  PYREFLY_REPORT_PATH           Pyrefly report path (default: project root)
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
if (( ! FULL_PROJECT && DIFF_FILTER )) && [[ -z "$DIFF_BASE" ]]; then
    DIFF_BASE="HEAD"
fi
if (( DIFF_FILTER == 0 )); then
    DIFF_BASE=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXCLUSION_SCRIPT="$SCRIPT_DIR/pyrefly_exclusion.py"
EXCLUSIONS_CONFIG="$SCRIPT_DIR/pyrefly_exclusion.json"
WORKFLOW_SCRIPT="$SCRIPT_DIR/pyrefly_workflow.py"
REPORT_FILE="${PYREFLY_REPORT_PATH:-$PROJECT_DIR/pyrefly_final_report.json}"
PYREFLY_OUTPUT=""
PYREFLY_STDERR=""
FILTERED_REPORT=""
DIFF_FILTERED_OUTPUT=""
PYREFLY_CONFIG=""
PATCH_FILE=""
REBASE_WORKTREE_PATH=""
REBASE_WORKTREE_PARENT=""
REBASE_TEMP_BRANCH=""
FILTERED_ARG_COUNT=0

print_report_summary() {
    python3 "$WORKFLOW_SCRIPT" print-summary "$REPORT_FILE" >&3 || true
}

cleanup_rebase_resources() {
    local status="${1:-$?}"
    trap - EXIT
    if [[ -n "$REBASE_WORKTREE_PATH" ]]; then
        git -C "$PROJECT_DIR" worktree remove --force "$REBASE_WORKTREE_PATH" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n "$REBASE_TEMP_BRANCH" ]] &&
        git -C "$PROJECT_DIR" show-ref --verify --quiet "refs/heads/$REBASE_TEMP_BRANCH"; then
        git -C "$PROJECT_DIR" branch -D "$REBASE_TEMP_BRANCH" >/dev/null 2>&1 || true
    fi
    if [[ -n "$REBASE_WORKTREE_PARENT" ]]; then
        rmdir "$REBASE_WORKTREE_PARENT" >/dev/null 2>&1 || true
    fi
    if [[ -n "$PATCH_FILE" ]]; then
        rm -f "$PATCH_FILE" || true
    fi
    REBASE_WORKTREE_PATH=""
    REBASE_WORKTREE_PARENT=""
    REBASE_TEMP_BRANCH=""
    PATCH_FILE=""
    return "$status"
}

handle_rebase_exit() {
    local status=$?
    local report_path="$REPORT_FILE"
    trap - EXIT
    cleanup_rebase_resources "$status" || true
    if (( status != 0 )); then
        if [[ "$report_path" != /* ]]; then
            report_path="$PROJECT_DIR/$report_path"
        fi
        python3 "$WORKFLOW_SCRIPT" record-rebase-failure \
            "$report_path" "$status" "${BASH_COMMAND:-unknown command}" || true
        printf 'WARNING: advisory rebase-upstream check failed (exit status %s); see report: %s\n' \
            "$status" "$report_path" >&2
    fi
    exit 0
}

run_rebased_check() {
    if (( FULL_PROJECT )); then
        echo "ERROR: --rebase-upstream checks changed files; it cannot be combined with --full-project" >&2
        return 2
    fi
    local upstream_url fallback_url timestamp source_branch worktree_path temp_branch
    local worktree_parent random_suffix requested_path
    local diff_report_dir diff_report check_report upstream_source main_ref main_sha rebased_head
    local requested_paths=("$@")
    upstream_url="git@github.com:NVIDIA/TensorRT-LLM.git"
    fallback_url="https://github.com/NVIDIA/TensorRT-LLM.git"
    timestamp="$(date +%Y%m%d_%H%M%S)_$$"
    source_branch="$(git -C "$PROJECT_DIR" symbolic-ref --quiet --short HEAD || printf 'HEAD')"
    worktree_parent="$(mktemp -d "${TMPDIR:-/tmp}/pyrefly-rebase.XXXXXX")"
    random_suffix="${worktree_parent##*.}"
    worktree_path="$worktree_parent/worktree"
    temp_branch="pyrefly/type-check-rebased-$random_suffix"
    diff_report_dir="${PYREFLY_DIFF_REPORT_PATH:-/tmp}"
    diff_report="$diff_report_dir/pyrefly_rebased_diff_$timestamp.txt"
    check_report="${PYREFLY_REPORT_PATH:-$PROJECT_DIR/pyrefly_rebased_report_$timestamp.json}"
    for path_variable in diff_report check_report; do
        if [[ "${!path_variable}" != /* ]]; then
            printf -v "$path_variable" '%s/%s' "$PROJECT_DIR" "${!path_variable}"
        fi
    done
    if [[ -e "$worktree_path" ]]; then
        echo "ERROR: worktree path already exists: $worktree_path" >&2
        return 2
    fi
    if git -C "$PROJECT_DIR" show-ref --verify --quiet "refs/heads/$temp_branch"; then
        echo "ERROR: temporary branch already exists: $temp_branch" >&2
        rmdir "$worktree_parent" >/dev/null 2>&1 || true
        return 2
    fi

    REBASE_WORKTREE_PATH="$worktree_path"
    REBASE_WORKTREE_PARENT="$worktree_parent"
    REBASE_TEMP_BRANCH="$temp_branch"
    export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
    PATCH_FILE="$(mktemp)"
    git -C "$PROJECT_DIR" diff --binary HEAD > "$PATCH_FILE"
    if git -C "$PROJECT_DIR" fetch "$upstream_url" main; then
        upstream_source="$upstream_url"
    else
        echo "Could not fetch through SSH; falling back to $fallback_url." >&2
        git -C "$PROJECT_DIR" fetch "$fallback_url" main
        upstream_source="$fallback_url"
    fi
    main_ref="FETCH_HEAD"
    main_sha="$(git -C "$PROJECT_DIR" rev-parse "$main_ref^{commit}")"

    git -C "$PROJECT_DIR" worktree add --quiet -b "$temp_branch" "$worktree_path" "$source_branch"
    if ! git -C "$worktree_path" rebase "$main_sha"; then
        echo "Rebase failed; the temporary worktree and branch will be cleaned up." >&2
        return 1
    fi
    if [[ -s "$PATCH_FILE" ]]; then
        if ! git -C "$worktree_path" apply --3way "$PATCH_FILE"; then
            echo "Could not apply current tracked edits after rebase." >&2
            echo "The temporary worktree and branch will be cleaned up." >&2
            return 1
        fi
        git -C "$worktree_path" reset --mixed HEAD
    fi

    local workflow_relative helper_relative
    workflow_relative="scripts/$(basename "${BASH_SOURCE[0]}")"
    if ! git -C "$worktree_path" ls-files --error-unmatch -- "$workflow_relative" >/dev/null 2>&1; then
        cp "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")" "$worktree_path/$workflow_relative"
        git -C "$worktree_path" add -N -- "$workflow_relative"
    fi
    helper_relative="scripts/$(basename "$WORKFLOW_SCRIPT")"
    if ! git -C "$worktree_path" ls-files --error-unmatch -- "$helper_relative" >/dev/null 2>&1; then
        cp "$WORKFLOW_SCRIPT" "$worktree_path/$helper_relative"
        git -C "$worktree_path" add -N -- "$helper_relative"
    fi

    rebased_head="$(git -C "$worktree_path" rev-parse HEAD)"
    local changed_files=() check_files=() path
    while IFS= read -r -d '' path; do
        changed_files+=("$path")
    done < <(git -C "$worktree_path" diff --name-only -z "$main_sha" --)
    for path in "${changed_files[@]}"; do
        if [[ "$path" == tensorrt_llm/* && ( "$path" == *.py || "$path" == *.pyi ) ]]; then
            if (( ${#requested_paths[@]} == 0 )); then
                check_files+=("$path")
            else
                for requested_path in "${requested_paths[@]}"; do
                    if [[ "$path" == "$requested_path" ]]; then
                        check_files+=("$path")
                        break
                    fi
                done
            fi
        fi
    done

    {
        printf '# Source branch: %s\n' "$source_branch"
        printf '# Upstream base: %s (%s)\n' "$upstream_source" "$main_sha"
        printf '# Rebased HEAD: %s\n\n' "$rebased_head"
        printf 'Changed files (%s):\n' "${#changed_files[@]}"
        if (( ${#changed_files[@]} == 0 )); then
            printf '  (none)\n'
        else
            printf '  %s\n' "${changed_files[@]}"
        fi
        printf '\nZero-context code diff (hunks include changed line numbers):\n'
        git -C "$worktree_path" diff --no-ext-diff --no-renames --unified=0 "$main_sha" --
    } > "$diff_report"

    printf 'Rebased %s onto %s at %s\n' "$source_branch" "$upstream_source" "$main_sha"
    printf 'Worktree: %s (branch %s)\n' "$worktree_path" "$temp_branch"
    printf 'Changed files (%s):\n' "${#changed_files[@]}"
    if (( ${#changed_files[@]} == 0 )); then
        printf '  (none)\n'
    else
        printf '  %s\n' "${changed_files[@]}"
    fi
    printf 'Diff and line ranges: %s\n' "$diff_report"
    awk '/^diff --git / || /^@@ / { print }' "$diff_report"

    if (( ${#check_files[@]} > 0 )); then
        local runner_args=() internal_diff_base=""
        if (( ! DIFF_FILTER )); then
            runner_args+=(--no-diff-filter)
        else
            internal_diff_base="$main_sha"
        fi
        if (( ! PATH_MODULE_FILTER )); then
            runner_args+=(--no-path-module-filter)
        fi
        (
            cd "$worktree_path"
            PYREFLY_INTERNAL_DIFF_BASE="$internal_diff_base" PYREFLY_REPORT_PATH="$check_report" \
                bash scripts/run_pyrefly.sh "${runner_args[@]}" "${check_files[@]}"
        )
    else
        printf '%s\n' \
            '{"tool":"pyrefly","pyrefly_exit_code":0,"message":"No changed product Python files to check","errors":[]}' \
            > "$check_report"
    fi
    python3 "$WORKFLOW_SCRIPT" annotate-rebased-report \
        "$check_report" "$source_branch" "$upstream_source" "$main_sha" \
        "$rebased_head" "$worktree_path" "$diff_report" "$DIFF_FILTER" \
        "$PATH_MODULE_FILTER" "${check_files[@]+"${check_files[@]}"}"
    cleanup_rebase_resources
    printf 'Temporary worktree and branch removed after the check.\n'
}

if (( REBASE_UPSTREAM )); then
    trap handle_rebase_exit EXIT
    run_rebased_check "$@"
    exit 0
fi

cleanup() {
    local status=$?
    for file in "$PYREFLY_OUTPUT" "$PYREFLY_STDERR" "$FILTERED_REPORT" \
                "$DIFF_FILTERED_OUTPUT" "$PYREFLY_CONFIG"; do
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

if (( FULL_PROJECT && $# > 0 )); then
    printf '{"tool":"pyrefly","report_error":"--full-project does not accept file paths"}\n' \
        > "$REPORT_FILE"
    exit 2
fi
if [[ -n "$DIFF_BASE" ]] && ! git rev-parse --verify "${DIFF_BASE}^{commit}" >/dev/null 2>&1; then
    printf '{"tool":"pyrefly","report_error":"invalid diff base: %s"}\n' "$DIFF_BASE" \
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
        if (( PATH_MODULE_FILTER )) &&
            python3 "$EXCLUSION_SCRIPT" matches "$argument" "$EXCLUSIONS_CONFIG" \
                --project-root "$PROJECT_DIR"; then
            continue
        fi
    fi
    FILTERED_ARGS+=("$argument")
    FILTERED_ARG_COUNT=$((FILTERED_ARG_COUNT + 1))
done

if (( ! FULL_PROJECT && FILTERED_ARG_COUNT == 0 )); then
    printf '%s\n' \
        '{"tool":"pyrefly","pyrefly_exit_code":0,"message":"No eligible changed product Python files to check","errors":[]}' \
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
if (( ! PATH_MODULE_FILTER )); then
    PYREFLY_CONFIG="$(mktemp "$PROJECT_DIR/.pyrefly-config.XXXXXX.toml")"
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
        "$FILTERED_REPORT" "$DIFF_FILTERED_OUTPUT" "$PROJECT_DIR" "$DIFF_BASE"
    FINAL_REPORT="$DIFF_FILTERED_OUTPUT"
fi
cp "$FINAL_REPORT" "$REPORT_FILE"
exit "$PYREFLY_STATUS"

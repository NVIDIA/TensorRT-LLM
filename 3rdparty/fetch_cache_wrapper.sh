#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FetchContent cache wrapper — injected by CMake as GIT_EXECUTABLE.
#
# Intercepts 'git clone' and 'git submodule update --init' to inject
# '--reference <cache>/<repo>.git' when a matching bare reference repo
# exists.  All other git commands pass through unchanged.
#
# Environment (set by CMake):
#   _TRTLLM_REAL_GIT            Absolute path to the real git binary
#   TRTLLM_FETCHCONTENT_CACHE   Path to the cache directory

set -euo pipefail

REAL_GIT="${_TRTLLM_REAL_GIT:-/usr/bin/git}"
CACHE_DIR="${TRTLLM_FETCHCONTENT_CACHE:-}"

# ── Pass through if no cache ──

if [ -z "$CACHE_DIR" ] || [ ! -d "$CACHE_DIR" ]; then
    exec "$REAL_GIT" "$@"
fi

# ── Find git subcommand (skip global options) ──

subcmd=""
skip_next=false
for arg in "$@"; do
    if $skip_next; then
        skip_next=false
        continue
    fi
    case "$arg" in
        -c|-C|--git-dir|--work-tree|--namespace|--super-prefix|--exec-path)
            skip_next=true ;;
        --git-dir=*|--work-tree=*|--exec-path=*|-C*) ;;
        -*) ;;
        *)
            subcmd="$arg"
            break
            ;;
    esac
done

case "$subcmd" in
    clone)    ;; # handled below
    submodule) ;; # handled below
    *)        exec "$REAL_GIT" "$@" ;;
esac

# ══════════════════════════════════════════════════════════════════════════
# Submodule update --init interception
# ══════════════════════════════════════════════════════════════════════════

if [ "$subcmd" = "submodule" ]; then
    # Parse submodule sub-action and flags
    sub_action=""
    has_init=false
    has_recursive=false
    has_reference=false
    extra_args=()
    paths=()
    past_separator=false
    past_submodule=false
    skip_next=false

    for arg in "$@"; do
        if $skip_next; then
            skip_next=false
            extra_args+=("$arg")
            continue
        fi
        if $past_separator; then
            paths+=("$arg")
            continue
        fi
        if ! $past_submodule; then
            [ "$arg" = "submodule" ] && past_submodule=true
            continue
        fi
        case "$arg" in
            update|init|status|summary|foreach|sync|deinit|set-branch|set-url|absorbgitdirs)
                sub_action="$arg" ;;
            --init)      has_init=true ;;
            --recursive) has_recursive=true ;;
            --reference|--reference=*)
                has_reference=true; extra_args+=("$arg") ;;
            --)          past_separator=true ;;
            --depth|--jobs|-j)
                skip_next=true; extra_args+=("$arg") ;;
            --depth=*|--jobs=*|-j*)
                extra_args+=("$arg") ;;
            -*)          extra_args+=("$arg") ;;
            *)           paths+=("$arg") ;;
        esac
    done

    # Only intercept "submodule update --init"
    if [ "$sub_action" != "update" ] || ! $has_init || $has_reference; then
        exec "$REAL_GIT" "$@"
    fi

    toplevel=$("$REAL_GIT" rev-parse --show-toplevel 2>/dev/null) || exec "$REAL_GIT" "$@"

    if [ ! -f "$toplevel/.gitmodules" ]; then
        exec "$REAL_GIT" "$@"
    fi

    # Register submodules (no clone)
    "$REAL_GIT" submodule init "${paths[@]}" 2>/dev/null || true

    # Update each submodule with per-submodule --reference
    updated_paths=()
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        sub_key="${line%% *}"
        sub_path="${line#* }"
        section="${sub_key%.path}"
        sub_url=$("$REAL_GIT" config --file "$toplevel/.gitmodules" --get "${section}.url" 2>/dev/null) || continue

        # If explicit paths were given, filter
        if [ ${#paths[@]} -gt 0 ]; then
            match=false
            for p in "${paths[@]}"; do
                [ "$p" = "$sub_path" ] && match=true && break
            done
            $match || continue
        fi

        repo_name="$(basename "${sub_url%.git}")"
        ref_repo="$CACHE_DIR/${repo_name}.git"

        if [ -d "$ref_repo" ] && [ -f "$ref_repo/HEAD" ]; then
            echo "-- [fetch-cache] submodule $sub_path: using reference $ref_repo" >&2
            "$REAL_GIT" submodule update --init --reference "$ref_repo" "${extra_args[@]}" -- "$sub_path" || \
                "$REAL_GIT" submodule update --init "${extra_args[@]}" -- "$sub_path" || true
        else
            "$REAL_GIT" submodule update --init "${extra_args[@]}" -- "$sub_path" || true
        fi
        updated_paths+=("$sub_path")
    done < <("$REAL_GIT" config --file "$toplevel/.gitmodules" --get-regexp 'submodule\..*\.path' 2>/dev/null)

    # Handle --recursive: re-enter the wrapper for each submodule
    if $has_recursive; then
        for sub_path in "${updated_paths[@]}"; do
            sub_dir="$toplevel/$sub_path"
            if [ -d "$sub_dir" ] && [ -f "$sub_dir/.gitmodules" ]; then
                (cd "$sub_dir" && "$0" submodule update --init --recursive "${extra_args[@]}")
            fi
        done
    fi
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# Clone interception
# ══════════════════════════════════════════════════════════════════════════

# ── Skip if --reference already present ──

for arg in "$@"; do
    case "$arg" in
        --reference|--reference=*|--reference-if-able|--reference-if-able=*)
            exec "$REAL_GIT" "$@" ;;
    esac
done

# ── Extract clone URL ──

url=""
past_clone=false
skip_next=false
for arg in "$@"; do
    if $skip_next; then
        skip_next=false
        continue
    fi
    if ! $past_clone; then
        [ "$arg" = "clone" ] && past_clone=true
        continue
    fi
    case "$arg" in
        -b|--branch|--depth|-j|--jobs|-o|--origin|--separate-git-dir| \
        --shallow-since|--shallow-exclude|--config|-c|--template| \
        --filter|--reference|--reference-if-able|--server-option|--bundle-uri)
            skip_next=true
            continue ;;
        --branch=*|--depth=*|--jobs=*|--origin=*|--separate-git-dir=*| \
        --shallow-since=*|--shallow-exclude=*|--config=*|--template=*| \
        --filter=*|--reference=*|--reference-if-able=*|--server-option=*|--bundle-uri=*)
            continue ;;
        -*)
            continue ;;
        *)
            url="$arg"
            break
            ;;
    esac
done

if [ -z "$url" ]; then
    exec "$REAL_GIT" "$@"
fi

# ── Map URL to cache repo ──

repo_name="$(basename "${url%.git}")"
ref_repo="$CACHE_DIR/${repo_name}.git"

if [ -d "$ref_repo" ] && [ -f "$ref_repo/HEAD" ]; then
    new_args=()
    injected=false
    for arg in "$@"; do
        new_args+=("$arg")
        if [ "$arg" = "clone" ] && ! $injected; then
            injected=true
            new_args+=("--reference" "$ref_repo")
        fi
    done
    echo "-- [fetch-cache] Using reference: $ref_repo" >&2
    exec "$REAL_GIT" "${new_args[@]}"
else
    exec "$REAL_GIT" "$@"
fi

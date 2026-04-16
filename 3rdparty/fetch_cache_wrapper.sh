#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FetchContent cache wrapper — injected by CMake as GIT_EXECUTABLE.
#
# Intercepts 'git clone' and injects '--reference <cache>/<repo>.git'
# when a matching bare reference repo exists.  All other git commands
# pass through unchanged.
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

if [ "$subcmd" != "clone" ]; then
    exec "$REAL_GIT" "$@"
fi

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

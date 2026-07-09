#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# artifactory.sh - NVIDIA-internal glue to package + promote + pull BOLT profile
# bundles on Artifactory, with a BRANCH-keyed layout so premerge can consume the
# latest postmerge-promoted profiles for its branch.
#
# Subcommands:
#   package <profiles_dir> <ref> <triple> [out_dir]
#       Build <out_dir>/bolt-profile-<ref>-<triple>.tar.gz from the merged
#       .yaml profiles (+ manifest.json if present in profiles_dir, else built
#       when --work-dir is given). NETWORK-FREE -- safe anywhere.
#       (gzip, not zstd: zstd isn't present in the trt-llm containers.)
#
#   promote <bundle.tar.gz> <branch> <triple>
#       Upload the bundle to the branch-keyed path as BOTH a versioned copy and
#       latest.tar.gz (overwrite). Used by POSTMERGE only.
#
#   pull-latest <branch> <triple> <dest_dir>
#       Download + extract latest.tar.gz for <branch>/<triple> into <dest_dir>.
#       Used by PREMERGE (and postmerge fallback is intentionally NOT provided).
#
# The premerge "override" case does NOT use promote: the gen recipe packages a
# bundle locally and apply consumes it directly (run-scoped, never promoted).
#
# Artifactory layout (override via env):
#   <REPO>/<PREFIX>/<branch>/<triple>/bolt-profile-<ref>-<triple>.tar.gz
#   <REPO>/<PREFIX>/<branch>/<triple>/latest.tar.gz
# Defaults match the repo used by jenkins/Build.groovy (sw-tensorrt-generic).
#
# Auth: uses the `jf` (JFrog) CLI. If JF_URL + JF_ACCESS_TOKEN are set, this
# script configures jf automatically; otherwise it assumes `jf` is preconfigured
# (as on CI agents). Jenkins may alternatively use rtUpload/rtDownload with the
# same paths (see promote_path() / latest_path()).

set -euo pipefail

REPO="${BOLT_ARTIFACTORY_REPO:-sw-tensorrt-generic}"
PREFIX="${BOLT_PROFILE_PREFIX:-llm-artifacts/bolt-profiles}"

log()  { echo "[artifactory] $1" >&2; }
die()  { echo "[artifactory][ERROR] $1" >&2; exit 2; }

promote_dir() { echo "${REPO}/${PREFIX}/$1/$2"; }   # <branch> <triple>

_ensure_jf() {
    command -v jf >/dev/null 2>&1 || die "jf (JFrog CLI) not found on PATH"
    if [[ -n "${JF_URL:-}" && -n "${JF_ACCESS_TOKEN:-}" ]]; then
        jf c add --overwrite --url "$JF_URL" --access-token "$JF_ACCESS_TOKEN" \
            --interactive=false bolt-artifactory >/dev/null 2>&1 || true
        jf c use bolt-artifactory >/dev/null 2>&1 || true
    fi
}

cmd_package() {
    local profiles_dir="${1:?package: <profiles_dir> <ref> <triple> [out_dir]}"
    local ref="${2:?ref required}"
    local triple="${3:?triple required (e.g. aarch64-linux-gnu)}"
    local out_dir="${4:-$profiles_dir}"
    [[ -d "$profiles_dir" ]] || die "profiles dir not found: $profiles_dir"

    # Build a manifest if one isn't already present and a work dir was provided.
    if [[ ! -f "$profiles_dir/manifest.json" && -n "${BOLT_WORK_DIR:-}" ]]; then
        local mp; mp="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/manifest.py"
        if [[ -f "$mp" ]]; then
            log "Building manifest.json via manifest.py"
            "${PYTHON:-python3}" "$mp" build \
                --work-dir "$BOLT_WORK_DIR" --profiles "$profiles_dir" \
                --ref "$ref" -o "$profiles_dir/manifest.json" || \
                log "WARNING: manifest build failed; bundle will omit it"
        fi
    fi

    mkdir -p "$out_dir"
    # gzip (not zstd): zstd isn't installed in the trt-llm runtime/devel
    # containers used on the compute nodes, and gzip is always available.
    # Consumers extract by extension (pull-latest), so keep the format in sync.
    local bundle="$out_dir/bolt-profile-${ref}-${triple}.tar.gz"
    # Bundle the .yaml profiles + manifest.json (stored at top level).
    local files=()
    local f
    for f in "$profiles_dir"/*.yaml; do [[ -e "$f" ]] && files+=("$(basename "$f")"); done
    [[ ${#files[@]} -gt 0 ]] || die "no .yaml profiles in $profiles_dir"
    [[ -f "$profiles_dir/manifest.json" ]] && files+=("manifest.json")

    log "Packaging ${#files[@]} file(s) -> $bundle"
    tar -czf "$bundle" -C "$profiles_dir" "${files[@]}"
    echo "$bundle"
}

cmd_promote() {
    local bundle="${1:?promote: <bundle.tar.gz> <branch> <triple>}"
    local branch="${2:?branch required}"
    local triple="${3:?triple required}"
    [[ -f "$bundle" ]] || die "bundle not found: $bundle"
    _ensure_jf
    local dir; dir="$(promote_dir "$branch" "$triple")"
    log "Promoting $(basename "$bundle") -> $dir/ (versioned + latest)"
    jf rt upload "$bundle" "$dir/$(basename "$bundle")" --flat
    jf rt upload "$bundle" "$dir/latest.tar.gz" --flat
    log "Promoted. latest = $dir/latest.tar.gz"
}

cmd_pull_latest() {
    local branch="${1:?pull-latest: <branch> <triple> <dest_dir>}"
    local triple="${2:?triple required}"
    local dest="${3:?dest_dir required}"
    _ensure_jf
    local dir; dir="$(promote_dir "$branch" "$triple")"
    mkdir -p "$dest"
    log "Pulling $dir/latest.tar.gz -> $dest"
    # --flat so the file lands directly in dest; fail cleanly if absent.
    if ! jf rt download "$dir/latest.tar.gz" "$dest/" --flat --fail-no-op; then
        die "no promoted bundle at $dir/latest.tar.gz (branch may have none yet)"
    fi
    tar -xzf "$dest/latest.tar.gz" -C "$dest"
    rm -f "$dest/latest.tar.gz"
    log "Extracted latest bundle into $dest"
}

case "${1:-}" in
    package)      shift; cmd_package "$@" ;;
    promote)      shift; cmd_promote "$@" ;;
    pull-latest)  shift; cmd_pull_latest "$@" ;;
    *) die "usage: artifactory.sh {package|promote|pull-latest} ..." ;;
esac

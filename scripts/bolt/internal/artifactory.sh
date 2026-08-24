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
#   promote <bundle.tar.gz> <branch> <triple> [promote_latest]
#       Upload the bundle to the branch-keyed path as a versioned copy, and (when
#       promote_latest is "true", the default) also overwrite latest.tar.gz.
#       Pass "false" (or set BOLT_PROMOTE_LATEST=false) to publish a versioned
#       bundle for an OLD ref without repointing latest at it. Used by POSTMERGE.
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
# Auth:
#   promote     -> `jf` (JFrog) CLI (write). If JF_URL + JF_ACCESS_TOKEN are set
#                  this configures jf automatically; else assumes jf is
#                  preconfigured. (The CI postmerge path in BoltProfileGen.groovy
#                  instead uploads cluster-side via curl + urm-artifactory-creds.)
#   pull-latest -> anonymous curl (read), matching Build.groovy's phase-1 tarball
#                  download from the same repo; needs no creds and no jf, so it
#                  runs in the premerge build pod as-is.

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
    local bundle="${1:?promote: <bundle.tar.gz> <branch> <triple> [promote_latest]}"
    local branch="${2:?branch required}"
    local triple="${3:?triple required}"
    # Whether to also overwrite latest.tar.gz. Defaults to true (postmerge on the
    # current ref); pass "false" (or set BOLT_PROMOTE_LATEST=false) to publish a
    # versioned bundle for an OLD ref without repointing latest at it.
    local promote_latest="${4:-${BOLT_PROMOTE_LATEST:-true}}"
    [[ -f "$bundle" ]] || die "bundle not found: $bundle"
    _ensure_jf
    local dir; dir="$(promote_dir "$branch" "$triple")"
    log "Promoting $(basename "$bundle") -> $dir/ (versioned$([[ "$promote_latest" == "true" ]] && echo " + latest"))"
    jf rt upload "$bundle" "$dir/$(basename "$bundle")" --flat
    if [[ "$promote_latest" == "true" ]]; then
        jf rt upload "$bundle" "$dir/latest.tar.gz" --flat
        log "Promoted. latest = $dir/latest.tar.gz"
    else
        log "Promoted versioned bundle only (latest NOT updated)."
    fi
}

cmd_pull_latest() {
    local branch="${1:?pull-latest: <branch> <triple> <dest_dir>}"
    local triple="${2:?triple required}"
    local dest="${3:?dest_dir required}"
    local base="${BOLT_ARTIFACTORY_BASE:-https://urm.nvidia.com/artifactory}"
    local url="$base/$(promote_dir "$branch" "$triple")/latest.tar.gz"
    mkdir -p "$dest"
    log "Pulling $url -> $dest"
    # Anonymous download (like jenkins/Build.groovy's phase-1 tarball from the same
    # sw-tensorrt-generic repo): the read path needs no creds, so this works in the
    # build pod without a configured `jf`. Use curl if present, else fall back to
    # wget -- the image-overlay build pod (BuildDockerImage.overlayBoltBundle) has no
    # curl, so hard-requiring curl made the overlay fail (curl: command not found)
    # even though the bundle existed. --retry rides out flaky links; a missing object
    # 404s -> die (fatal for the premerge consumer).
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 5 --retry-all-errors --retry-delay 10 \
             --connect-timeout 60 -o "$dest/latest.tar.gz" "$url" \
            || die "no promoted bundle at $url (branch may have none yet)"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=5 --waitretry=10 --timeout=60 \
             -O "$dest/latest.tar.gz" "$url" \
            || die "no promoted bundle at $url (branch may have none yet)"
    else
        die "neither curl nor wget is available to download $url"
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

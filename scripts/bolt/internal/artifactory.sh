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
#       bundle for an OLD ref without repointing latest at it.
#
#   pull-latest <branch> <triple> <dest_dir>
#       Download + extract a bundle for <branch>/<triple> into <dest_dir>.
#       Honours BOLT_PROFILE_REF: when set, pulls the immutable
#       bolt-profile-<ref>-<triple>.tar.gz instead of the mutable latest.tar.gz,
#       so every consumer in a pipeline can be pinned to one bundle.
#       Exits 0 when the bundle is unpacked, 3 when neither location has it, and
#       4 when the store could not be read.
#
#   resolve-latest <branch> <triple>
#       Print the ref latest.tar.gz currently points at, for a caller that wants
#       to pin the rest of a pipeline to it. Prints nothing when the store has no
#       latest.tar.gz for <branch>/<triple>, or it names no ref; exits 0 either
#       way, and 4 when the store could not be read.
#
# The premerge "override" case does NOT use promote: the gen recipe packages a
# bundle locally and apply consumes it directly (run-scoped, never promoted).
#
# Artifactory layout, the same in the repository of each location:
#   <PREFIX>/<branch>/<triple>/bolt-profile-<ref>-<triple>.tar.gz
#   <PREFIX>/<branch>/<triple>/latest.tar.gz
# PREFIX is llm-artifacts/bolt-profiles unless BOLT_PROFILE_PREFIX replaces it.
# CI's promotion (promoteBundle in jenkins/BoltProfileGen.groovy) writes this
# layout cluster-side with curl, and labels latest.tar.gz with the properties
# bolt.ref, bolt.branch, and bolt.triple in the request that uploads it.
#
# Reads (pull-latest, resolve-latest) go through the read helper,
# jenkins/scripts/fetch_build_artifact.sh, found from the repository root
# relative to this script. They ask for the legacy path
# sw-tensorrt-generic/<PREFIX>/..., which the helper maps to each location: it
# asks the primary location first and the other location only when the file is
# absent there, never after a rejected request, and checks the SHA-1 of a
# download. No read is anonymous on the aws location, and none goes around the
# helper, so a caller provides what the helper reads from its environment:
#   LLM_ARTIFACTS_PRIMARY      aws (the default) or urm
#   ARTIFACTORY_NETRC          netrc file with the credential for the aws location
#   ARTIFACTORY_CREDENTIAL_ID  name of the CI credential behind that file, which
#                              the helper's message names when aws rejects it
#
# Writes:
#   promote -> `jf` (JFrog) CLI, the developer path; CI does not use it. If
#              JF_URL + JF_ACCESS_TOKEN are set this configures jf
#              automatically; else assumes jf is preconfigured. Uploads to the
#              repository BOLT_ARTIFACTORY_REPO (sw-tensorrt-llm-generic-local
#              unless set) on the server jf is configured for.

set -euo pipefail

REPO="${BOLT_ARTIFACTORY_REPO:-sw-tensorrt-llm-generic-local}"
PREFIX="${BOLT_PROFILE_PREFIX:-llm-artifacts/bolt-profiles}"
# The repository name that starts a legacy path, as the read helper takes it.
LEGACY_REPO="sw-tensorrt-generic"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts/bolt/internal
FETCH_HELPER="$HERE/../../../jenkins/scripts/fetch_build_artifact.sh"

# Exit statuses of the read helper, by the names the helper gives them.
# tests/unittest/scripts/test_fetch_build_artifact.py checks that they match.
FETCH_ABSENT=2
FETCH_REJECTED=3
FETCH_TRANSFER_FAILED=4
FETCH_USAGE=64
# Exit statuses of pull-latest and resolve-latest: neither location has the
# bundle, and a read of the store was refused or failed.
BOLT_NO_BUNDLE=3
BOLT_STORE_UNREADABLE=4

log()  { echo "[artifactory] $1" >&2; }
die()  { echo "[artifactory][ERROR] $1" >&2; exit 2; }
# fail <status> <message>: as die, with the exit status of a read.
fail() { echo "[artifactory][ERROR] $2" >&2; exit "$1"; }

# url_host <url>: prints the host, with its port, of a URL.
url_host() {
    local rest="${1#*://}"
    echo "${rest%%/*}"
}

promote_dir() { echo "${REPO}/${PREFIX}/$1/$2"; }       # <branch> <triple>
read_dir()    { echo "${LEGACY_REPO}/${PREFIX}/$1/$2"; }  # <branch> <triple>

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

# Removes the directory that resolve-latest works in, however the script ends.
WORK_DIR=""
cleanup() {
    if [[ -n "$WORK_DIR" ]]; then
        rm -rf -- "$WORK_DIR"
    fi
}
trap cleanup EXIT

# store_read [--properties] <path> <dest-file>, or store_read --exists <path>:
# runs the read helper, which inherits its environment from the caller of this
# script. Returns 0 when the helper wrote <dest-file>, and FETCH_ABSENT when
# neither location has the file (with --properties: has properties for it).
# Any other outcome ends the script with BOLT_STORE_UNREADABLE; the helper has
# logged the cause by then, with the credential's name for a rejection. The
# helper's stdout joins stderr, so that resolve-latest prints nothing but the
# ref. With --exists the stdout is the answer, "<url> <sha1>" of the copy on
# the first location that holds the file, so it is left alone; a caller that
# captures it runs this in a subshell and has to end the script itself.
store_read() {
    local rc=0 reason
    if [[ ! -f "$FETCH_HELPER" ]]; then
        fail "$BOLT_STORE_UNREADABLE" "the read helper is missing: $FETCH_HELPER"
    fi
    if [[ "$1" == --exists ]]; then
        bash "$FETCH_HELPER" "$@" || rc=$?
    else
        bash "$FETCH_HELPER" "$@" >&2 || rc=$?
    fi
    case "$rc" in
        0 | "$FETCH_ABSENT") return "$rc" ;;
        "$FETCH_REJECTED") reason="the primary location rejected the request" ;;
        "$FETCH_TRANSFER_FAILED") reason="the transfer failed" ;;
        "$FETCH_USAGE") reason="the read helper rejected its arguments or its environment" ;;
        *) reason="the read helper ended unexpectedly" ;;
    esac
    fail "$BOLT_STORE_UNREADABLE" \
        "the profile store could not be read: $reason (fetch_build_artifact.sh $* exited $rc)"
}

# Print the ref that <branch>/<triple>'s latest.tar.gz currently points at, or
# nothing when the store has no latest.tar.gz or it names no ref.
#
# CI's promotion labels latest.tar.gz with a bolt.ref property in the same
# request that uploads it, so the label and the bytes can never disagree. A
# latest.tar.gz uploaded another way (`promote` here, for one) carries no
# property, hence the manifest fallback: download the bundle (~1 MB) and read
# `ref` out of manifest.json. A caller that gets nothing back should stay
# unpinned rather than guess.
#
# The label has to be that of the copy that readers get. The helper asks the
# primary location first, for the file and for its properties alike, but a
# file without properties on the primary sends the properties lookup on to the
# other location's copy. So the properties count only when they come from the
# host that holds the file for readers; otherwise the manifest answers.
#
# No read has an overall time limit, so a caller that must not wait on a
# stalled transfer bounds the call itself.
cmd_resolve_latest() {
    local usage="resolve-latest: <branch> <triple>"
    local branch="${1:?$usage}"
    shift
    local triple="${1:?$usage}"
    local path; path="$(read_dir "$branch" "$triple")/latest.tar.gz"
    WORK_DIR="$(mktemp -d)"

    local rc=0 ref="" held="" properties=""
    held="$(store_read --exists "$path")" || rc=$?
    if (( rc == FETCH_ABSENT )); then
        log "no latest.tar.gz for $branch/$triple on either location"
        return 0
    elif (( rc != 0 )); then
        exit "$rc"
    fi

    rc=0
    store_read --properties "$path" "$WORK_DIR/properties.json" || rc=$?
    if (( rc == 0 )); then
        # { "properties": { "bolt.ref": [ "<ref>" ] }, "uri": "<storage URL>" } --
        # fixed, tiny shape, so a sed extraction avoids depending on python3 or
        # jq, neither of which is guaranteed on every pod that might call this.
        properties="$(tr -d ' \n' < "$WORK_DIR/properties.json")"
        if [[ "$(url_host "$(sed -n 's/.*"uri":"\([^"]*\)".*/\1/p' <<<"$properties")")" \
            == "$(url_host "${held%% *}")" ]]; then
            ref="$(sed -n 's/.*"bolt\.ref":\["\([^"]*\)"\].*/\1/p' <<<"$properties")"
        else
            log "the properties of $path are another location's; not using them"
        fi
    fi
    if [[ -n "$ref" ]]; then
        printf '%s\n' "$ref"
        return 0
    fi

    log "no bolt.ref property on $path; falling back to the bundle manifest"
    rc=0
    store_read "$path" "$WORK_DIR/latest.tar.gz" || rc=$?
    if (( rc == FETCH_ABSENT )); then
        log "latest.tar.gz for $branch/$triple is no longer on either location"
        return 0
    fi
    if tar -xzf "$WORK_DIR/latest.tar.gz" -C "$WORK_DIR" manifest.json 2>/dev/null; then
        ref="$(tr -d ' \n' < "$WORK_DIR/manifest.json" \
               | sed -n 's/.*"ref":"\([^"]*\)".*/\1/p')"
    fi
    if [[ -n "$ref" ]]; then
        printf '%s\n' "$ref"
    else
        log "the manifest of $path names no ref"
    fi
    return 0
}

cmd_pull_latest() {
    local usage="pull-latest: <branch> <triple> <dest_dir>"
    local branch="${1:?$usage}"
    shift
    local triple="${1:?$usage}"
    shift
    local dest="${1:?$usage}"
    # BOLT_PROFILE_REF pins this pull to one immutable bundle. It is an env var
    # rather than an argument, so that a caller opts into pinning by exporting it
    # and every call site that does not takes latest.tar.gz.
    local path
    if [[ -n "${BOLT_PROFILE_REF:-}" ]]; then
        path="$(read_dir "$branch" "$triple")/bolt-profile-${BOLT_PROFILE_REF}-${triple}.tar.gz"
        log "Pinned to ${BOLT_PROFILE_REF}"
    else
        path="$(read_dir "$branch" "$triple")/latest.tar.gz"
    fi
    mkdir -p "$dest"
    log "Pulling $path -> $dest"
    # The helper retries a transient failure itself, so one call decides: the
    # bundle is here, neither location has it, or the store cannot be read.
    local rc=0
    store_read "$path" "$dest/latest.tar.gz" || rc=$?
    if (( rc == FETCH_ABSENT )); then
        fail "$BOLT_NO_BUNDLE" \
            "no promoted bundle at $path on either location (branch may have none yet)"
    fi
    tar -xzf "$dest/latest.tar.gz" -C "$dest"
    rm -f "$dest/latest.tar.gz"
    log "Extracted the bundle into $dest"
}

case "${1:-}" in
    package)        shift; cmd_package "$@" ;;
    promote)        shift; cmd_promote "$@" ;;
    pull-latest)    shift; cmd_pull_latest "$@" ;;
    resolve-latest) shift; cmd_resolve_latest "$@" ;;
    *) die "usage: artifactory.sh {package|promote|pull-latest|resolve-latest} ..." ;;
esac

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

# Reads one file under llm-artifacts/ from the primary Artifactory location, and from the secondary
# location only when the primary does not have it. CI reads a build's build_info.txt, packaged
# builds, and waive_list/ with it, and it reads any other file there the same way. Run it with
# --help for the interface and the exit codes.
#
# Credentials go to the aws location only, through curl's --netrc-file; urm is read anonymously.
# The script turns off xtrace that a caller passes on, because a trace would show response URLs,
# which can carry a CDN's signature. Never pass a secret on a command line.

# The steps run through with_retries and cleanup runs from a trap, so shellcheck cannot see that
# they are reachable.
# shellcheck disable=SC2317,SC2329

set +x
set -uo pipefail

readonly PROG="fetch_build_artifact"
readonly MAX_ATTEMPTS=3
readonly AWS_BASE_URL="https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local"
readonly URM_BASE_URL="https://urm.nvidia.com/artifactory/sw-tensorrt-generic"
# Exit codes other than success. Readers that act on one define the same name for it, and
# tests/unittest/scripts/test_fetch_build_artifact.py checks that the values match.
readonly FETCH_ABSENT=2
readonly FETCH_REJECTED=3
readonly FETCH_TRANSFER_FAILED=4
readonly FETCH_USAGE=64

usage() {
    cat <<'EOF'
Usage: fetch_build_artifact.sh [--properties | --exists] <artifact-path> [<dest-file>]

Reads a build artifact from the primary Artifactory location, and from the secondary location
only when the primary does not have it. A rejected request, a server or connection error, or a
checksum mismatch never falls back.

Arguments:
  <artifact-path>  A legacy path (sw-tensorrt-generic/llm-artifacts/... or
                   sw-tensorrt-generic-local/llm-artifacts/...) or a layout path
                   (llm-artifacts/...).
  <dest-file>      Output file, written as <dest-file>.part and renamed when complete.
                   Required unless --exists is given.
  --properties     Write the artifact's properties (storage API JSON) instead of the file.
  --exists         Write no file; print "<url> <sha1>" for the location that holds the artifact.

Environment:
  LLM_ARTIFACTS_PRIMARY      aws (the default) or urm; the other location is the secondary.
  ARTIFACTORY_NETRC          netrc file holding an Artifactory access token for
                             artifactory.nvidia.com; required whenever aws is contacted.
  ARTIFACTORY_CREDENTIAL_ID  Name of the CI credential behind ARTIFACTORY_NETRC, for messages.
  LLM_ARTIFACTS_AWS_URL, LLM_ARTIFACTS_URM_URL
                             Replace a location's base URL, <root>/<repository>.

Exit codes:
  0   Success; stderr gets "fetch_build_artifact: served-by=<aws|urm> path=<layout path>".
  2   Neither location has the artifact (with --properties: has properties for it).
  3   The primary rejected the request; the secondary was not contacted.
  4   The transfer failed after retries: a connection or server error, a persistent CDN
      rejection, a checksum mismatch, or a failure other than "absent" on the secondary.
  64  Usage error, unmappable path, bad LLM_ARTIFACTS_PRIMARY, missing ARTIFACTORY_NETRC,
      or a missing tool.
EOF
}

log() {
    printf '%s: %s\n' "$PROG" "$*" >&2
}

# fail <exit code> <message>...
fail() {
    local code="$1"
    shift
    log "$@"
    exit "$code"
}

usage_error() {
    log "$@"
    usage >&2
    exit "$FETCH_USAGE"
}

# Prints the layout path (llm-artifacts/...) of a legacy or layout artifact path. Fails for any
# other path, for an empty, "." or ".." segment, and for characters that would change the meaning
# of the request URL.
map_path() {
    local path="$1" unsafe='[[:space:];?#]'
    case "$path" in
        sw-tensorrt-generic/*) path="${path#sw-tensorrt-generic/}" ;;
        sw-tensorrt-generic-local/*) path="${path#sw-tensorrt-generic-local/}" ;;
        *) ;;
    esac
    [[ "$path" == llm-artifacts/?* ]] || return 1
    case "/${path}/" in
        *//* | */./* | */../*) return 1 ;;
        *) ;;
    esac
    [[ ! "$path" =~ $unsafe ]] || return 1
    printf '%s\n' "$path"
}

# Prints the lower-case host[:port] of a URL.
url_host() {
    local rest="${1#*://}"
    rest="${rest%%[/?#]*}"
    rest="${rest##*@}"
    printf '%s\n' "${rest,,}"
}

# use_location <aws|urm>: sets LOC, LOC_BASE (<root>/<repository>), LOC_ROOT, LOC_REPO, and
# LOC_HOST. Fails when the base URL does not have that shape.
use_location() {
    LOC="$1"
    if [[ "$LOC" == aws ]]; then
        LOC_BASE="${LLM_ARTIFACTS_AWS_URL:-$AWS_BASE_URL}"
    else
        LOC_BASE="${LLM_ARTIFACTS_URM_URL:-$URM_BASE_URL}"
    fi
    LOC_BASE="${LOC_BASE%/}"
    LOC_ROOT="${LOC_BASE%/*}"
    LOC_REPO="${LOC_BASE##*/}"
    LOC_HOST="$(url_host "$LOC_BASE")"
    [[ "$LOC_ROOT" =~ ^https?://[^/]+ && -n "$LOC_REPO" ]]
}

file_url() {
    printf '%s/%s\n' "$LOC_BASE" "$LAYOUT_PATH"
}

storage_url() {
    printf '%s/api/storage/%s/%s\n' "$LOC_ROOT" "$LOC_REPO" "$LAYOUT_PATH"
}

require_netrc() {
    local need="reading $LAYOUT_PATH from aws needs a netrc file with an Artifactory access token"
    need+=" for artifactory.nvidia.com."
    if [[ -z "${ARTIFACTORY_NETRC:-}" ]]; then
        fail "$FETCH_USAGE" "ARTIFACTORY_NETRC is not set; $need"
    fi
    if [[ ! -f "$ARTIFACTORY_NETRC" || ! -r "$ARTIFACTORY_NETRC" ]]; then
        fail "$FETCH_USAGE" "ARTIFACTORY_NETRC ($ARTIFACTORY_NETRC) is not a readable file; $need"
    fi
}

# http_get <output file> <url> [<curl option>...]
# Sends one GET and follows redirects, with the netrc only when the location is aws; curl applies
# a netrc entry only to the host it names, so a redirect to another host carries no credential.
# Sets HTTP_STATUS (000 when no response arrived) and FINAL_HOST (host[:port] of the last URL
# requested), and returns curl's exit status.
http_get() {
    local output="$1" url="$2" result rc
    shift 2
    if [[ "$LOC" == aws ]]; then
        set -- --netrc-file "$ARTIFACTORY_NETRC" "$@"
    fi
    # Start from an empty file, so that it never holds the body of an earlier attempt.
    : >"$output"
    # -q must come first. It keeps curl from reading a curlrc, which could, for example, turn on
    # verbose output that prints the Authorization header.
    result="$(curl -q --silent --show-error --location --connect-timeout 30 \
        --speed-limit 1 --speed-time 300 "$@" \
        --output "$output" --write-out '%{http_code} %{url_effective}' "$url")"
    rc=$?
    HTTP_STATUS="${result%% *}"
    FINAL_HOST="$(url_host "${result#* }")"
    return "$rc"
}

# Prints checksums.sha1 from a storage API answer. Text tools keep the helper usable on hosts
# without python or jq. The pattern matches the "checksums" key only, never "originalChecksums".
extract_sha1() {
    tr -d ' \t\r\n' <"$1" \
        | sed -n 's/.*"checksums":{[^}]*"sha1":"\([0-9a-fA-F]\{40\}\)".*/\1/p' \
        | tr 'A-F' 'a-f'
}

# Steps make one attempt each and return 0 (done), 1 (transient failure, reason in FAILURE),
# FETCH_ABSENT, or FETCH_REJECTED (status in REJECTED_STATUS).

# Reads the artifact's checksums.sha1 from the storage API into SHA1.
lookup_step() {
    local rc
    http_get "$WORK_DIR/storage.json" "$(storage_url)"
    rc=$?
    if (( rc != 0 )); then
        FAILURE="the storage lookup failed (curl exit $rc)"
        return 1
    fi
    case "$HTTP_STATUS" in
        200)
            SHA1="$(extract_sha1 "$WORK_DIR/storage.json")"
            if [[ -n "$SHA1" ]]; then
                return 0
            fi
            FAILURE="the storage lookup answered without checksums.sha1"
            ;;
        404) return "$FETCH_ABSENT" ;;
        401 | 403)
            REJECTED_STATUS="$HTTP_STATUS"
            return "$FETCH_REJECTED"
            ;;
        *) FAILURE="the storage lookup answered HTTP $HTTP_STATUS" ;;
    esac
    return 1
}

# Downloads the file to PART, and renames it to DEST when its SHA-1 matches SHA1.
download_step() {
    local rc cdn_host=""
    http_get "$PART" "$(file_url)"
    rc=$?
    if (( rc == 0 )) && [[ "$HTTP_STATUS" == 403 && "$FINAL_HOST" != "$LOC_HOST" ]]; then
        # The CDN that the location redirected the download to rejected it, which says nothing
        # about the credential. Ask the location to send the file itself.
        cdn_host="$FINAL_HOST"
        log "$cdn_host rejected the redirected download (HTTP 403);" \
            "downloading from $LOC_HOST without the redirect."
        http_get "$PART" "$(file_url)" -H 'X-JFrog-Download-Redirect-To: None'
        rc=$?
    fi
    if (( rc == 0 )) && [[ "$HTTP_STATUS" == 200 ]]; then
        verify_download
        return
    fi
    rm -f -- "$PART"
    if (( rc != 0 )); then
        FAILURE="the download failed (curl exit $rc)"
    elif [[ -n "$cdn_host" ]]; then
        FAILURE="the download without the CDN redirect answered HTTP $HTTP_STATUS"
    elif [[ "$FINAL_HOST" == "$LOC_HOST" && "$HTTP_STATUS" =~ ^40[13]$ ]]; then
        REJECTED_STATUS="$HTTP_STATUS"
        return "$FETCH_REJECTED"
    else
        FAILURE="the download answered HTTP $HTTP_STATUS from $FINAL_HOST"
    fi
    return 1
}

verify_download() {
    local actual
    actual="$(sha1sum <"$PART")"
    actual="${actual%% *}"
    if [[ "$actual" != "$SHA1" ]]; then
        rm -f -- "$PART"
        FAILURE="checksum mismatch: the download has SHA-1 $actual, the storage API reports $SHA1"
        return 1
    fi
    if ! mv -f -- "$PART" "$DEST"; then
        FAILURE="could not rename $PART to $DEST"
        return 1
    fi
}

# Writes the artifact's properties (storage API JSON) to DEST.
properties_step() {
    local rc
    http_get "$PART" "$(storage_url)?properties"
    rc=$?
    if (( rc == 0 )) && [[ "$HTTP_STATUS" == 200 ]] && mv -f -- "$PART" "$DEST"; then
        return 0
    fi
    rm -f -- "$PART"
    if (( rc != 0 )); then
        FAILURE="the properties lookup failed (curl exit $rc)"
        return 1
    fi
    case "$HTTP_STATUS" in
        404) return "$FETCH_ABSENT" ;;
        401 | 403)
            REJECTED_STATUS="$HTTP_STATUS"
            return "$FETCH_REJECTED"
            ;;
        200) FAILURE="could not rename $PART to $DEST" ;;
        *) FAILURE="the properties lookup answered HTTP $HTTP_STATUS" ;;
    esac
    return 1
}

# with_retries <step>: repeats a step while it fails transiently, at most MAX_ATTEMPTS times in
# total, sleeping 2 s and then 4 s in between. Returns the step's result, or
# FETCH_TRANSFER_FAILED when its last attempt failed transiently.
with_retries() {
    local attempt rc delay
    for (( attempt = 1; ; attempt++ )); do
        "$1"
        rc=$?
        if (( rc != 1 )); then
            return "$rc"
        fi
        if (( attempt >= MAX_ATTEMPTS )); then
            log "attempt $attempt of $MAX_ATTEMPTS on $LOC ($LOC_HOST) failed: $FAILURE;" \
                "giving up on $LAYOUT_PATH."
            return "$FETCH_TRANSFER_FAILED"
        fi
        delay=$(( 2 ** attempt ))
        log "attempt $attempt of $MAX_ATTEMPTS on $LOC ($LOC_HOST) failed: $FAILURE;" \
            "retrying in ${delay}s."
        sleep "$delay"
    done
}

# read_from <aws|urm>: reads the artifact from one location. Returns 0 (served), FETCH_ABSENT,
# FETCH_REJECTED, or FETCH_TRANSFER_FAILED (failed after retries).
read_from() {
    local rc
    use_location "$1"
    if [[ "$LOC" == aws ]]; then
        require_netrc
    fi
    if [[ "$MODE" == properties ]]; then
        with_retries properties_step
        return
    fi
    with_retries lookup_step
    rc=$?
    if (( rc != 0 )) || [[ "$MODE" == exists ]]; then
        return "$rc"
    fi
    # The lookup found the file here, so a failed download is a transfer failure and never
    # sends the read to the other location.
    with_retries download_step
}

# The message names both causes for a 401 and a 403 alike, so that a missing permission is never
# taken for an expired token.
report_rejection() {
    if [[ "$LOC" == aws ]]; then
        log "$LOC_HOST rejected the credentials for $LAYOUT_PATH (HTTP $REJECTED_STATUS)."
        log "the Artifactory access token in credential" \
            "'${ARTIFACTORY_CREDENTIAL_ID:-<unset: ARTIFACTORY_CREDENTIAL_ID>}' needs renewal" \
            "or lacks read permission on $LOC_REPO;" \
            "an Artifactory access token for artifactory.nvidia.com is required."
    else
        log "$LOC_HOST rejected the anonymous request for $LAYOUT_PATH (HTTP $REJECTED_STATUS)."
    fi
}

served() {
    if [[ "$MODE" == exists ]]; then
        printf '%s %s\n' "$(file_url)" "$SHA1"
    fi
    log "served-by=$LOC path=$LAYOUT_PATH"
    exit 0
}

cleanup() {
    rm -rf -- "$WORK_DIR"
    if [[ -n "$PART" ]]; then
        rm -f -- "$PART"
    fi
}

MODE=download
ARGS=()
while (( $# > 0 )); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --properties | --exists)
            if [[ "$MODE" != download && "$MODE" != "${1#--}" ]]; then
                usage_error "--properties and --exists cannot be combined."
            fi
            MODE="${1#--}"
            ;;
        --)
            shift
            ARGS+=("$@")
            break
            ;;
        -?*) usage_error "unknown option: $1" ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

case "$MODE:${#ARGS[@]}" in
    download:2 | properties:2 | exists:1 | exists:2) ;;
    *) usage_error "wrong number of arguments." ;;
esac

if ! LAYOUT_PATH="$(map_path "${ARGS[0]}")"; then
    usage_error "not a file path under llm-artifacts/: '${ARGS[0]}'"
fi

# --exists writes no file, so it ignores <dest-file>.
DEST=""
PART=""
if [[ "$MODE" != exists ]]; then
    DEST="${ARGS[1]}"
    if [[ -z "$DEST" || "$DEST" == */ || -d "$DEST" ]]; then
        fail "$FETCH_USAGE" "<dest-file> must name a file: '$DEST'"
    fi
    if [[ ! -d "$(dirname -- "$DEST")" ]]; then
        fail "$FETCH_USAGE" "the directory of <dest-file> does not exist: '$DEST'"
    fi
    PART="$DEST.part"
fi

PRIMARY="${LLM_ARTIFACTS_PRIMARY:-aws}"
case "$PRIMARY" in
    aws) SECONDARY=urm ;;
    urm) SECONDARY=aws ;;
    *) fail "$FETCH_USAGE" "LLM_ARTIFACTS_PRIMARY must be 'aws' or 'urm', not '$PRIMARY'." ;;
esac

for location in aws urm; do
    override="LLM_ARTIFACTS_${location^^}_URL"
    # Messages show the base URL, so it must not carry credentials.
    authority="${!override:-}"
    authority="${authority#*://}"
    if [[ "${authority%%/*}" == *@* ]]; then
        fail "$FETCH_USAGE" "$override must not contain credentials; ARTIFACTORY_NETRC provides them."
    fi
    if ! use_location "$location"; then
        fail "$FETCH_USAGE" "$override must be an http(s) URL of the form <root>/<repository>, not '$LOC_BASE'."
    fi
done

command -v curl >/dev/null 2>&1 || fail "$FETCH_USAGE" "curl is required."
if [[ "$MODE" == download ]]; then
    command -v sha1sum >/dev/null 2>&1 || fail "$FETCH_USAGE" "sha1sum is required."
fi

LOC=""
SHA1=""
FAILURE=""
REJECTED_STATUS=""
HTTP_STATUS=""
FINAL_HOST=""
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fetch_build_artifact.XXXXXX")" \
    || fail "$FETCH_USAGE" "cannot create a temporary directory."
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

read_from "$PRIMARY"
case $? in
    0) served ;;
    "$FETCH_ABSENT") ;;
    "$FETCH_REJECTED")
        report_rejection
        exit "$FETCH_REJECTED"
        ;;
    *) exit "$FETCH_TRANSFER_FAILED" ;;
esac

read_from "$SECONDARY"
case $? in
    0) served ;;
    "$FETCH_ABSENT")
        if [[ "$MODE" == properties ]]; then
            log "neither $PRIMARY nor $SECONDARY has properties for $LAYOUT_PATH."
        else
            log "neither $PRIMARY nor $SECONDARY has $LAYOUT_PATH."
        fi
        exit "$FETCH_ABSENT"
        ;;
    "$FETCH_REJECTED")
        # A rejection by the secondary is reported like any other failure there.
        report_rejection
        exit "$FETCH_TRANSFER_FAILED"
        ;;
    *) exit "$FETCH_TRANSFER_FAILED" ;;
esac

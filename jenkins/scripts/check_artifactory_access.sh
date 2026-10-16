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

# Checks that the Artifactory identity in ARTIFACTORY_NETRC can do what its CI role needs in the
# build-artifact repository on artifactory.nvidia.com (the aws location). For one probe directory,
# run the publisher role first, then the reader role in every environment that reads build
# artifacts, then the publisher role with --cleanup. --help lists the checks.
#
# Each check prints "PASS <check>" or "FAIL <check>: <reason>" on stdout, and --time prints one
# "TIME <location> <seconds> s <MB/s> MB/s <url>" line per location. Exit status: 0 when every
# check passed, 1 when a check failed, 64 on a usage error.
#
# Environment:
#   ARTIFACTORY_NETRC          netrc file with the identity's Artifactory access token for
#                              artifactory.nvidia.com (required)
#   ARTIFACTORY_CREDENTIAL_ID  name of the CI credential behind that file, used in messages
#   LLM_ARTIFACTS_AWS_URL      base URL (<root>/<repo>) of the aws location; tests override it
#   LLM_ARTIFACTS_URM_URL      base URL of the urm location, read by --time; tests override it
#
# Credentials reach curl only through --netrc-file, and only on requests to the aws location.

set -euo pipefail
# curl's -w timings and awk's number formatting both use "." as the decimal point in the C locale.
export LC_ALL=C

readonly PROG="check_artifactory_access"
readonly DEFAULT_AWS_URL="https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local"
readonly DEFAULT_URM_URL="https://urm.nvidia.com/artifactory/sw-tensorrt-generic"

usage() {
    cat <<'EOF'
Usage:
  check_artifactory_access.sh --role publisher|reader --probe-dir <layout-dir> [--time <artifact-path>]
  check_artifactory_access.sh --role publisher --cleanup --probe-dir <layout-dir>

  --role publisher  In <layout-dir>: create probe.txt, overwrite it with property access.check,
                    read its properties, copy it to probe-copy.txt, list the directory, and
                    delete probe-copy.txt. probe.txt stays for the reader checks.
  --role reader     Read probe.txt with credentials and match its SHA-1 to the storage API,
                    read its properties, and expect HTTP 401 for an anonymous read and HTTP 403
                    for a write with the reader's credentials.
  --cleanup         With --role publisher: delete <layout-dir>/probe.txt and run no other check.
  --time <path>     Also time a download of <path> from aws (with credentials) and from urm
                    (anonymous).

Paths are layout paths (llm-artifacts/...) or legacy paths (sw-tensorrt-generic/llm-artifacts/...).
ARTIFACTORY_NETRC must name a netrc file with an Artifactory access token for
artifactory.nvidia.com.
EOF
}

usage_error() {
    echo "${PROG}: $1" >&2
    usage >&2
    exit 64
}

# Prints the layout path of a legacy (sw-tensorrt-generic[-local]/...) or layout
# (llm-artifacts/...) path. Fails for any other path, for an empty, "." or ".." segment, and for
# characters that would change the meaning of the request URL.
layout_path() {
    local path="$1" unsafe='[[:space:];?#]'
    case "$path" in
        sw-tensorrt-generic/*) path="${path#sw-tensorrt-generic/}" ;;
        sw-tensorrt-generic-local/*) path="${path#sw-tensorrt-generic-local/}" ;;
    esac
    [[ "$path" == llm-artifacts/?* ]] || return 1
    case "/${path}/" in
        *//* | */./* | */../*) return 1 ;;
    esac
    [[ ! "$path" =~ $unsafe ]] || return 1
    printf '%s\n' "$path"
}

# Splits a location base URL (<root>/<repo>) at its last "/" into parsed_root and parsed_repo,
# and sets parsed_host to the root's host (with its port, if any). Messages show the URL, so it
# must not carry credentials.
parse_base_url() {
    local variable="$1" base="$2" authority
    while [[ "$base" == */ ]]; do
        base="${base%/}"
    done
    authority="${base#*://}"
    if [[ "${authority%%/*}" == *@* ]]; then
        usage_error "${variable} must not contain credentials; ARTIFACTORY_NETRC provides them"
    fi
    parsed_root="${base%/*}"
    parsed_repo="${base##*/}"
    if [[ ! "$parsed_root" =~ ^https?://[^/]+ || -z "$parsed_repo" ]]; then
        usage_error "${variable} must look like https://<host>/artifactory/<repo>, not '${base}'"
    fi
    parsed_host="$(url_host "$parsed_root")"
}

# Prints the host (with its port, if any) of a URL.
url_host() {
    local host="${1#*://}"
    host="${host%%[/?#]*}"
    printf '%s\n' "${host##*@}"
}

# Runs one curl request with the given options. Sets http_status (the final status, 000 when no
# response arrived), time_total, size_download, final_url, and curl_error (curl's message when the
# transfer failed). -q must come first: it keeps a .curlrc from adding options such as --netrc or
# --fail, which would change what a check tests.
run_curl() {
    local written rc=0
    written="$(curl -q -sS --connect-timeout 30 \
        -w '%{http_code} %{time_total} %{size_download} %{url_effective}' \
        "$@" 2>"${work_dir}/curl.err")" || rc=$?
    read -r http_status time_total size_download final_url <<<"$written"
    http_status="${http_status:-000}"
    curl_error=""
    if ((rc != 0)); then
        curl_error="$(tail -n 1 "${work_dir}/curl.err")"
        curl_error="${curl_error:-curl exit status ${rc}}"
    fi
}

# Sends one request to the aws location, with the netrc credentials ("auth") or without them
# ("anon"), and keeps the response body in response_file.
aws_request() {
    local auth="$1"
    shift
    : >"$response_file"
    if [[ "$auth" == auth ]]; then
        run_curl --max-time 120 --netrc-file "$ARTIFACTORY_NETRC" -o "$response_file" "$@"
    else
        run_curl --max-time 120 -o "$response_file" "$@"
    fi
}

# Prints why the last request did not get the expected status, with a hint when the aws location
# answered a request with credentials ("auth") with 401 or 403.
status_reason() {
    local expected="$1" auth="$2"
    if [[ "$http_status" == 000 ]]; then
        printf 'got no HTTP response (%s)' "$curl_error"
    elif [[ -n "$curl_error" ]]; then
        printf 'got HTTP %s, but the transfer failed (%s)' "$http_status" "$curl_error"
    else
        printf 'got HTTP %s, expected %s' "$http_status" "$expected"
        if [[ "$auth" == auth && "$http_status" == 401 ]]; then
            printf '; %s did not accept the credentials in ARTIFACTORY_NETRC' "$aws_host"
            if [[ -n "${ARTIFACTORY_CREDENTIAL_ID:-}" ]]; then
                printf " (credential '%s')" "$ARTIFACTORY_CREDENTIAL_ID"
            fi
        elif [[ "$auth" == auth && "$http_status" == 403 ]]; then
            printf '; the identity lacks this permission'
        fi
    fi
}

# Succeeds when the last request completed with one of the given statuses. Otherwise sets
# fail_reason, naming the first status as the expected one.
expect_status() {
    local auth="$1" code
    shift
    if [[ -z "$curl_error" ]]; then
        for code in "$@"; do
            if [[ "$http_status" == "$code" ]]; then
                return 0
            fi
        done
    fi
    fail_reason="$(status_reason "$1" "$auth")"
    return 1
}

# Prints the last response body without whitespace, so that JSON fields match whatever the
# server's formatting.
compact_response() {
    tr -d ' \t\r\n' <"$response_file"
}

# Writes small probe content that is unique to this run.
write_probe() {
    printf 'access check probe, %s at %s on %s (process %s)\n' \
        "$1" "$timestamp" "${HOSTNAME:-unknown host}" "$$" >"$probe_file"
}

file_url() {
    printf '%s/%s/%s/%s\n' "$aws_root" "$aws_repo" "$probe_dir" "$1"
}

# Prints the storage API URL of a file in the probe directory, or of the directory itself.
storage_url() {
    printf '%s/api/storage/%s/%s%s\n' "$aws_root" "$aws_repo" "$probe_dir" "${1:+/$1}"
}

# Runs one check function and prints its PASS or FAIL line.
run_check() {
    local name="$1"
    shift
    fail_reason=""
    checks_run=$((checks_run + 1))
    if "$@"; then
        echo "PASS ${name}"
    else
        echo "FAIL ${name}: ${fail_reason:-the check did not complete}"
        checks_failed=$((checks_failed + 1))
    fi
}

check_create() {
    write_probe created
    aws_request auth -T "$probe_file" "$(file_url probe.txt)"
    expect_status auth 201 200
}

check_overwrite_with_property() {
    write_probe overwritten
    aws_request auth -T "$probe_file" "$(file_url probe.txt);access.check=${timestamp}"
    expect_status auth 201 200
}

check_publisher_properties() {
    local property='"access\.check":\[[^]]*"'"${timestamp}"'"'
    aws_request auth -L "$(storage_url probe.txt)?properties"
    expect_status auth 200 || return 1
    if [[ ! "$(compact_response)" =~ $property ]]; then
        fail_reason="the response has no access.check=${timestamp} property"
        return 1
    fi
}

check_copy() {
    local from="${aws_repo}/${probe_dir}/probe.txt" to="${aws_repo}/${probe_dir}/probe-copy.txt"
    local response
    aws_request auth -X POST "${aws_root}/api/copy/${from}?to=/${to}"
    expect_status auth 200 || return 1
    # The copy API lists what it could not copy as error-level messages in its response.
    response="$(compact_response)"
    if [[ "${response,,}" == *'"level":"error"'* ]]; then
        fail_reason="the copy response reports an error"
        return 1
    fi
}

check_list() {
    aws_request auth -L "$(storage_url)"
    expect_status auth 200 || return 1
    if [[ "$(compact_response)" != *'"uri":"/probe.txt"'* ]]; then
        fail_reason="probe.txt is not among the children of the probe directory"
        return 1
    fi
}

check_delete_copy() {
    aws_request auth -X DELETE "$(file_url probe-copy.txt)"
    expect_status auth 204 200
}

check_read_with_checksum() {
    local sha1_field='"checksums":\{[^}]*"sha1":"([0-9a-fA-F]{40})"' expected actual
    aws_request auth -L "$(storage_url probe.txt)"
    if ! expect_status auth 200; then
        fail_reason="checksum lookup ${fail_reason}"
        return 1
    fi
    if [[ ! "$(compact_response)" =~ $sha1_field ]]; then
        fail_reason="the storage API response has no checksums.sha1"
        return 1
    fi
    expected="${BASH_REMATCH[1],,}"
    # Without the redirect, Artifactory serves the file itself, so the check covers the
    # identity's read permission rather than the CDN.
    aws_request auth -L -H 'X-JFrog-Download-Redirect-To: None' "$(file_url probe.txt)"
    if ! expect_status auth 200; then
        fail_reason="download ${fail_reason}"
        return 1
    fi
    actual="$(sha1sum <"$response_file")"
    actual="${actual%% *}"
    if [[ "$actual" != "$expected" ]]; then
        fail_reason="SHA-1 ${actual} of the download does not match checksums.sha1 ${expected}"
        return 1
    fi
}

check_reader_properties() {
    aws_request auth -L "$(storage_url probe.txt)?properties"
    expect_status auth 200 || return 1
    if [[ "$(compact_response)" != *'"access.check":['* ]]; then
        fail_reason="the response has no access.check property"
        return 1
    fi
}

check_anonymous_read_rejected() {
    aws_request anon -L -H 'X-JFrog-Download-Redirect-To: None' "$(file_url probe.txt)"
    if expect_status anon 401; then
        return 0
    fi
    if [[ "$http_status" == 2* ]]; then
        fail_reason+="; anonymous reads are allowed"
    fi
    return 1
}

check_reader_write_rejected() {
    write_probe "written with the reader's credentials"
    aws_request auth -T "$probe_file" "$(file_url reader-write-probe.txt)"
    if expect_status auth 403; then
        return 0
    fi
    if [[ "$http_status" == 2* ]]; then
        fail_reason+="; the reader identity can write (it created reader-write-probe.txt)"
    fi
    return 1
}

check_delete_probe() {
    aws_request auth -X DELETE "$(file_url probe.txt)"
    expect_status auth 204 200
}

# Downloads an artifact to /dev/null and prints a TIME line with the elapsed seconds and the
# throughput (1 MB = 10^6 bytes), or a FAIL line. On aws the download follows the redirect to the
# CDN, as reads in CI do; after a CDN 403 it is repeated without the redirect, and the TIME line
# covers both attempts.
time_download() {
    local location="$1" url="$2" auth=anon note="" first_attempt
    checks_run=$((checks_run + 1))
    if [[ "$location" == aws ]]; then
        auth=auth
        run_curl -L --netrc-file "$ARTIFACTORY_NETRC" -o /dev/null "$url"
        if [[ -z "$curl_error" && "$http_status" == 403 &&
            "$(url_host "$final_url")" != "$aws_host" ]]; then
            first_attempt="$time_total"
            run_curl -L --netrc-file "$ARTIFACTORY_NETRC" -H 'X-JFrog-Download-Redirect-To: None' \
                -o /dev/null "$url"
            time_total="$(awk -v first="$first_attempt" -v second="$time_total" \
                'BEGIN { printf "%.6f", first + second }')"
            note=" (after a CDN rejection, without the redirect)"
        fi
    else
        run_curl -L -o /dev/null "$url"
    fi
    if [[ -n "$curl_error" || "$http_status" != 200 ]]; then
        echo "FAIL time ${location} (${url}): $(status_reason 200 "$auth")"
        checks_failed=$((checks_failed + 1))
        return 0
    fi
    awk -v location="$location" -v seconds="$time_total" -v bytes="$size_download" \
        -v url="$url" -v note="$note" 'BEGIN {
            rate = seconds > 0 ? bytes / seconds / 1000000 : 0
            printf "TIME %s %.2f s %.1f MB/s %s%s\n", location, seconds, rate, url, note
        }'
}

role=""
probe_dir_arg=""
time_arg=""
cleanup=false
while (($# > 0)); do
    case "$1" in
        --role | --probe-dir | --time)
            (($# >= 2)) || usage_error "$1 needs a value"
            case "$1" in
                --role) role="$2" ;;
                --probe-dir) probe_dir_arg="$2" ;;
                --time) time_arg="$2" ;;
            esac
            shift 2
            ;;
        --cleanup)
            cleanup=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage_error "unknown argument '$1'"
            ;;
    esac
done

case "$role" in
    publisher | reader) ;;
    "") usage_error "--role is required" ;;
    *) usage_error "--role must be publisher or reader, not '${role}'" ;;
esac
[[ -n "$probe_dir_arg" ]] || usage_error "--probe-dir is required"
if [[ "$cleanup" == true ]]; then
    [[ "$role" == publisher ]] || usage_error "--cleanup needs --role publisher"
    [[ -z "$time_arg" ]] || usage_error "--cleanup runs no other check, so it takes no --time"
fi
probe_dir="$probe_dir_arg"
while [[ "$probe_dir" == */ ]]; do
    probe_dir="${probe_dir%/}"
done
probe_dir="$(layout_path "$probe_dir")" ||
    usage_error "--probe-dir must be a directory under llm-artifacts/, not '${probe_dir_arg}'"
time_path=""
if [[ -n "$time_arg" ]]; then
    time_path="$(layout_path "$time_arg")" ||
        usage_error "--time must be an artifact path under llm-artifacts/, not '${time_arg}'"
fi
if [[ -z "${ARTIFACTORY_NETRC:-}" ]]; then
    usage_error "ARTIFACTORY_NETRC is not set"
fi
if [[ ! -f "$ARTIFACTORY_NETRC" || ! -r "$ARTIFACTORY_NETRC" ]]; then
    usage_error "ARTIFACTORY_NETRC (${ARTIFACTORY_NETRC}) is not a readable file"
fi
parse_base_url LLM_ARTIFACTS_AWS_URL "${LLM_ARTIFACTS_AWS_URL:-${DEFAULT_AWS_URL}}"
aws_root="$parsed_root" aws_repo="$parsed_repo" aws_host="$parsed_host"
if [[ -n "$time_path" ]]; then
    parse_base_url LLM_ARTIFACTS_URM_URL "${LLM_ARTIFACTS_URM_URL:-${DEFAULT_URM_URL}}"
    urm_root="$parsed_root" urm_repo="$parsed_repo"
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
response_file="${work_dir}/response"
probe_file="${work_dir}/probe.txt"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
checks_run=0
checks_failed=0

if [[ "$cleanup" == true ]]; then
    echo "${PROG}: publisher cleanup of ${aws_root}/${aws_repo}/${probe_dir}" >&2
    run_check "delete probe.txt" check_delete_probe
elif [[ "$role" == publisher ]]; then
    echo "${PROG}: publisher checks in ${aws_root}/${aws_repo}/${probe_dir}" >&2
    run_check "create probe.txt" check_create
    run_check "overwrite probe.txt with property access.check" check_overwrite_with_property
    run_check "read properties of probe.txt" check_publisher_properties
    run_check "copy probe.txt to probe-copy.txt" check_copy
    run_check "list the probe directory" check_list
    run_check "delete probe-copy.txt" check_delete_copy
else
    echo "${PROG}: reader checks in ${aws_root}/${aws_repo}/${probe_dir}" >&2
    run_check "read probe.txt and match checksums.sha1" check_read_with_checksum
    run_check "read properties of probe.txt" check_reader_properties
    run_check "anonymous read of probe.txt is rejected" check_anonymous_read_rejected
    run_check "write with reader credentials is rejected" check_reader_write_rejected
fi
if [[ -n "$time_path" ]]; then
    time_download aws "${aws_root}/${aws_repo}/${time_path}"
    time_download urm "${urm_root}/${urm_repo}/${time_path}"
fi

if ((checks_failed > 0)); then
    echo "${PROG}: ${checks_failed} of ${checks_run} checks failed" >&2
    exit 1
fi
echo "${PROG}: ${checks_run} of ${checks_run} checks passed" >&2

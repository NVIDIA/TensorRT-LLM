#!/bin/sh
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly DIND_MTU_SCRIPT="${SCRIPT_DIR}/dind_mtu.sh"
readonly TEST_ROOT="$(mktemp -d)"
readonly TEST_BIN="${TEST_ROOT}/bin"
readonly TEST_SYS_CLASS_NET="${TEST_ROOT}/sys/class/net"
readonly TEST_ENTRYPOINT_ARGS="${TEST_ROOT}/entrypoint-args"
readonly SYSTEM_TIMEOUT="$(command -v timeout || true)"

cleanup()
{
    rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT HUP INT TERM

fail()
{
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains()
{
    expected="$1"
    file="$2"
    grep -F -- "${expected}" "${file}" >/dev/null ||
        fail "Expected '${expected}' in ${file}"
}

mkdir -p "${TEST_BIN}" "${TEST_SYS_CLASS_NET}/eth0" "${TEST_SYS_CLASS_NET}/docker0"
printf '%s\n' 1450 > "${TEST_SYS_CLASS_NET}/eth0/mtu"
printf '%s\n' 1450 > "${TEST_SYS_CLASS_NET}/docker0/mtu"

printf '%s\n' \
    '#!/bin/sh' \
    "printf '%s\\n' 'default via 192.0.2.1 dev eth0'" > "${TEST_BIN}/ip"

printf '%s\n' \
    '#!/bin/sh' \
    'printf '\''%s\n'\'' "$*" > "${DIND_TEST_ENTRYPOINT_ARGS}"' > \
    "${TEST_BIN}/dockerd-entrypoint.sh"

printf '%s\n' \
    '#!/bin/sh' \
    'shift' \
    'exec "$@"' > "${TEST_BIN}/timeout"

printf '%s\n' \
    '#!/bin/sh' \
    '[ "${1:-}" = info ]' > "${TEST_BIN}/docker"

chmod +x "${TEST_BIN}/ip" "${TEST_BIN}/dockerd-entrypoint.sh" \
    "${TEST_BIN}/timeout" "${TEST_BIN}/docker" "${DIND_MTU_SCRIPT}"

export DIND_SYS_CLASS_NET="${TEST_SYS_CLASS_NET}"
export DIND_TEST_ENTRYPOINT_ARGS="${TEST_ENTRYPOINT_ARGS}"
export PATH="${TEST_BIN}:${PATH}"

start_output="${TEST_ROOT}/start-output"
"${DIND_MTU_SCRIPT}" start > "${start_output}"
assert_contains "Starting Docker with MTU 1450 from interface eth0" "${start_output}"
assert_contains "--mtu=1450" "${TEST_ENTRYPOINT_ARGS}"

validate_output="${TEST_ROOT}/validate-output"
"${DIND_MTU_SCRIPT}" validate > "${validate_output}"
assert_contains "Verified DIND network MTU: eth0=1450, docker0=1450" "${validate_output}"

printf '%s\n' 1500 > "${TEST_SYS_CLASS_NET}/docker0/mtu"
mismatch_output="${TEST_ROOT}/mismatch-output"
if "${DIND_MTU_SCRIPT}" validate > "${mismatch_output}" 2>&1; then
    fail "Expected mismatched MTUs to fail validation"
fi
assert_contains "DIND MTU mismatch: eth0=1450, docker0=1500" "${mismatch_output}"

if [ -n "${SYSTEM_TIMEOUT}" ]; then
    bounded_bin="${TEST_ROOT}/bounded-bin"
    bounded_output="${TEST_ROOT}/bounded-output"
    probe_log="${TEST_ROOT}/probe-log"
    mkdir -p "${bounded_bin}"

    cp "${TEST_BIN}/ip" "${bounded_bin}/ip"
    printf '%s\n' \
        '#!/bin/sh' \
        'printf '\''probe\n'\'' >> "${DIND_TEST_PROBE_LOG}"' \
        'sleep 10' \
        'exit 1' > "${bounded_bin}/docker"
    chmod +x "${bounded_bin}/ip" "${bounded_bin}/docker"

    export DIND_TEST_PROBE_LOG="${probe_log}"
    if PATH="${bounded_bin}:$(dirname -- "${SYSTEM_TIMEOUT}"):/bin:/usr/bin" \
        DIND_DOCKER_READY_TIMEOUT_SECONDS=3 \
        DIND_DOCKER_INFO_TIMEOUT_SECONDS=1 \
        "${DIND_MTU_SCRIPT}" validate > "${bounded_output}" 2>&1; then
        fail "Expected blocked Docker probes to hit the readiness timeout"
    fi
    assert_contains "Docker daemon did not become ready within 3 seconds" "${bounded_output}"

    probe_count="$(wc -l < "${probe_log}" | tr -d ' ')"
    if [ "${probe_count}" -lt 2 ]; then
        fail "Expected the per-probe timeout to permit a retry, got ${probe_count} probe(s)"
    fi
fi

echo "dind_mtu tests passed"

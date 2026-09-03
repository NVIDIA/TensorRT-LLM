#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Ask, in the stock NGC image and with no TensorRT-LLM involved, which hostnames
# a singleton MPI_Comm_spawn survives.
#
# The Kubernetes test pods answer to their pod name -- 63 characters, containing
# a "---" run -- and under the Open MPI 5 of DLFW 26.08 a singleton spawn there
# fails immediately, while the same image spawns fine under a short name. Both
# 63 characters and consecutive hyphens are legal (HOST_NAME_MAX is 64, and
# RFC 1123 forbids hyphens only at the ends), so the name being invalid does not
# explain the failure, and renaming pods is at best a workaround. This walks a
# series of names that vary length and shape one at a time, so the result says
# which property actually matters.
#
# The probe is a C program rather than mpi4py: it keeps the report to Open MPI
# itself. Everything the test container adds -- our Dockerfile, install_base.sh,
# the wheel -- is out of the picture, so a failure here is reportable to the
# DLFW team as it stands.
#
# Optional env vars:
#   PROBE_TIMEOUT    seconds to allow each spawn (default 60)
#   PROBE_KEEP_NAME  leave the last name in place instead of restoring (default 0)

set -uo pipefail

PROBE_TIMEOUT="${PROBE_TIMEOUT:-60}"
PROBE_KEEP_NAME="${PROBE_KEEP_NAME:-0}"
WORK_DIR="$(mktemp -d)"
ORIGINAL_NAME="$(hostname)"
RESULTS=()

echo "===== environment ====="
echo "hostname:      ${ORIGINAL_NAME} (${#ORIGINAL_NAME} chars)"
echo "HOST_NAME_MAX: $(getconf HOST_NAME_MAX 2>/dev/null || echo '?')"
echo "kernel:        $(uname -srm)"
ompi_info --version 2>&1 | head -2
echo

# PRRTE refuses to fork its DVM as root unless told otherwise, and a singleton
# spawn forks one. Without these every name below would fail identically and the
# comparison would say nothing. Open MPI 4 only checked this in mpirun.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PRTE_ALLOW_RUN_AS_ROOT=1
export PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1

MPICC="$(command -v mpicc || true)"
for candidate in /usr/local/mpi/bin/mpicc /opt/hpcx/ompi/bin/mpicc; do
    [ -n "${MPICC}" ] && break
    [ -x "${candidate}" ] && MPICC="${candidate}"
done
if [ -z "${MPICC}" ]; then
    echo "FATAL: no mpicc found; cannot build the probe"
    exit 1
fi
echo "mpicc:         ${MPICC}"

cat > "${WORK_DIR}/spawn_probe.c" <<'PROBE_SOURCE'
/* Singleton MPI_Comm_spawn probe. The program re-executes itself as the child
   so the child is a real MPI program and the spawn can actually complete. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);
    if (parent != MPI_COMM_NULL) {
        MPI_Barrier(parent);
        MPI_Comm_disconnect(&parent);
        MPI_Finalize();
        return 0;
    }

    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);

    MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);

    MPI_Comm inter = MPI_COMM_NULL;
    int child_err = MPI_SUCCESS;
    double t0 = MPI_Wtime();
    int rc = MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0,
                            MPI_COMM_SELF, &inter, &child_err);
    double elapsed = MPI_Wtime() - t0;

    if (rc == MPI_SUCCESS) {
        MPI_Barrier(inter);
        MPI_Comm_disconnect(&inter);
        printf("PROBE_RESULT ok elapsed=%.2fs host=%s len=%zu\n",
               elapsed, host, strlen(host));
    } else {
        char msg[MPI_MAX_ERROR_STRING] = {0};
        int msglen = 0;
        MPI_Error_string(rc, msg, &msglen);
        printf("PROBE_RESULT fail elapsed=%.2fs host=%s len=%zu rc=%d msg=%s\n",
               elapsed, host, strlen(host), rc, msg);
    }
    fflush(stdout);

    MPI_Finalize();
    return rc == MPI_SUCCESS ? 0 : 1;
}
PROBE_SOURCE

if ! "${MPICC}" -O0 -o "${WORK_DIR}/spawn_probe" "${WORK_DIR}/spawn_probe.c"; then
    echo "FATAL: failed to build the probe"
    exit 1
fi
echo

# Reuse the address the pod already answers to, so every alias resolves to the
# same place and name resolution is never the variable under test.
POD_IP="$(awk -v name="${ORIGINAL_NAME}" '
    /^[[:space:]]*#/ { next }
    { for (i = 2; i <= NF; i++) if ($i == name) { print $1; exit } }
' /etc/hosts 2>/dev/null)"
if [ -z "${POD_IP}" ]; then
    POD_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
fi
echo "pod address:   ${POD_IP:-<unknown>}"
echo

# Repeat a pattern out to exactly the requested length, so length and shape vary
# independently. A trailing hyphen is the one thing RFC 1123 forbids, so the
# last character is forced to a letter and the names stay legal throughout.
make_name() {
    local length="$1" pattern="$2" out=""
    while [ "${#out}" -lt "${length}" ]; do
        out="${out}${pattern}"
    done
    out="${out:0:length}"
    if [ "${out: -1}" = "-" ]; then
        out="${out:0:length-1}x"
    fi
    printf '%s' "${out}"
}

run_probe() {
    local label="$1" name="$2"

    if [ -n "${name}" ]; then
        if ! grep -qw "${name}" /etc/hosts 2>/dev/null; then
            echo "${POD_IP} ${name}" >> /etc/hosts 2>/dev/null
        fi
        if ! hostname "${name}" 2>/dev/null; then
            echo "SKIP  ${label}: cannot set the hostname (needs CAP_SYS_ADMIN)"
            RESULTS+=("${label}|${#name}|skipped (no CAP_SYS_ADMIN)")
            return
        fi
    fi

    local current start output rc elapsed verdict
    current="$(hostname)"
    echo "----- ${label}: ${current} (${#current} chars) -----"
    start="${SECONDS}"
    output="$(timeout -k 5 "${PROBE_TIMEOUT}" "${WORK_DIR}/spawn_probe" 2>&1)"
    rc=$?
    elapsed=$(( SECONDS - start ))
    echo "${output}"
    echo "exit=${rc} wall=${elapsed}s"
    echo

    if [ "${rc}" -eq 0 ]; then
        verdict="spawn OK (${elapsed}s)"
    elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
        verdict="HUNG, killed at ${PROBE_TIMEOUT}s"
    else
        verdict="spawn FAILED (exit ${rc}, ${elapsed}s)"
    fi
    RESULTS+=("${label}|${#current}|${verdict}")
}

# Vary one property at a time: the pod's own name, then 63 characters in three
# shapes, then one shape at shrinking lengths, then the known-good short name as
# a control.
run_probe "as-launched" ""
run_probe "63 chars, --- run"   "$(make_name 63 'cpu---tensorrt-github4-llm-main-l0-test-')"
run_probe "63 chars, single -"  "$(make_name 63 'cpu-tensorrt-github4-llm-main-l0-test-x-')"
run_probe "63 chars, no hyphen" "$(make_name 63 'cputensorrtgithub4llmmainl0testx')"
run_probe "48 chars, single -"  "$(make_name 48 'cpu-tensorrt-github4-llm-main-l0-test-x-')"
run_probe "32 chars, single -"  "$(make_name 32 'cpu-tensorrt-github4-llm-main-l0-test-x-')"
run_probe "16 chars, single -"  "$(make_name 16 'cpu-tensorrt-github4-llm-main-l0-test-x-')"
run_probe "9 chars, control"    "mpi-node0"

if [ "${PROBE_KEEP_NAME}" != "1" ]; then
    hostname "${ORIGINAL_NAME}" 2>/dev/null || true
fi

echo "===== summary ====="
printf '%-22s %7s  %s\n' "variant" "length" "result"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r label length verdict <<< "${row}"
    printf '%-22s %7s  %s\n' "${label}" "${length}" "${verdict}"
done
echo
echo "hostname restored to: $(hostname)"
rm -rf "${WORK_DIR}"

# Always succeed: the point is the table, and a red stage would hide it behind a
# failure the reader then has to dig past.
exit 0

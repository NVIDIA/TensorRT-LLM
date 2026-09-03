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

# Give the test pod a short, resolvable hostname so that a singleton
# MPI_Comm_spawn works under the Open MPI 5 of the DLFW 26.08 base image.
#
# The Kubernetes hostname is the pod name, 63 characters. Under it a singleton
# spawn -- what MpiPoolSession does per worker -- fails with MPI_ERR_UNKNOWN as
# the prte DVM drops the connection. mpi4py.futures swallows that in its manager
# thread, so the worker identity barrier only gives up 300s later with 0/N.
#
# Only system state survives this script: it runs in its own shell, so MPI
# variables belong in docker/common/install_base.sh instead. The replaced name
# likewise reaches later steps through a file, which getHostNodeName() in
# jenkins/L0_Test.groovy reads so its fallback stays unique per pod.
#
# Single-node pods only: on a multi-node Slurm allocation every node would
# answer to the same name.
#
# Optional env var:
#   MPI_POD_HOSTNAME  name to use instead of the default mpi-node0

set -u

SHORT_NAME="${MPI_POD_HOSTNAME:-mpi-node0}"
ORIGINAL_NAME_FILE="/etc/mpi-pod-original-hostname"
CURRENT_NAME="$(hostname)"

if [ "${CURRENT_NAME}" = "${SHORT_NAME}" ]; then
    echo "[mpi-pod-fix] hostname is already ${SHORT_NAME}, nothing to do"
    exit 0
fi

# Reuse whatever address the pod is already known by so the short name resolves
# to the same place. Kubernetes writes "<pod ip> <pod name> ..." into /etc/hosts;
# fall back to hostname -i if that entry is missing.
POD_IP="$(awk -v name="${CURRENT_NAME}" '
    /^[[:space:]]*#/ { next }
    { for (i = 2; i <= NF; i++) if ($i == name) { print $1; exit } }
' /etc/hosts 2>/dev/null)"
if [ -z "${POD_IP}" ]; then
    POD_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
fi

if [ -z "${POD_IP}" ]; then
    echo "[mpi-pod-fix] WARNING: no address found for ${CURRENT_NAME}, leaving the hostname alone"
    exit 0
fi

if ! grep -qw "${SHORT_NAME}" /etc/hosts 2>/dev/null; then
    if ! echo "${POD_IP} ${SHORT_NAME}" >> /etc/hosts 2>/dev/null; then
        echo "[mpi-pod-fix] WARNING: cannot write /etc/hosts, leaving the hostname alone"
        exit 0
    fi
fi

# Record the name we are about to shadow before touching the UTS namespace, so
# that anything needing to tell this pod apart from its neighbours still can.
if ! echo "${CURRENT_NAME}" > "${ORIGINAL_NAME_FILE}" 2>/dev/null; then
    echo "[mpi-pod-fix] WARNING: cannot write ${ORIGINAL_NAME_FILE}, leaving the hostname alone"
    exit 0
fi

# Renaming the UTS namespace needs CAP_SYS_ADMIN, which the test pods add.
if hostname "${SHORT_NAME}" 2>/dev/null; then
    echo "[mpi-pod-fix] hostname ${CURRENT_NAME} (${#CURRENT_NAME} chars) -> $(hostname), resolving to ${POD_IP}"
else
    rm -f "${ORIGINAL_NAME_FILE}"
    echo "[mpi-pod-fix] WARNING: cannot set the hostname (needs CAP_SYS_ADMIN), keeping ${CURRENT_NAME}"
fi

exit 0

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

# Update EFA software stack only on Ubuntu images; skip elsewhere.
. /etc/os-release
if [ "${ID}" != "ubuntu" ]; then
    echo "Skipping EFA installer: OS '${ID}'."
    exit 0
fi

EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.51.0}"
# TODO: Use an internal mirror once the pinned EFA installer is available there.
EFA_INSTALLER_URL="https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

curl --retry 3 --retry-delay 2 -fsSL -o "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    "${EFA_INSTALLER_URL}"
tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
cd aws-efa-installer

apt-get update

# --skip-mpi: NGC images already ship HPC-X / Open MPI; do not overwrite it.
# --skip-rdma-core: use the base image's rdma-core.
# --build-ngc: install the EFA userspace packages without host driver installation,
#              device verification, limit configuration or dependency replacement.
./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc

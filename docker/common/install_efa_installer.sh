#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

# Update EFA software stack only on Ubuntu images; skip elsewhere.
OS=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "${OS}" != "ubuntu" ]; then
    echo "Skipping EFA installer: OS '${OS}'."
    exit 0
fi

EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.51.0}"
EFA_INSTALLER_URL="https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"

WORK_DIR="$(mktemp -d)"
cd "${WORK_DIR}"

curl --retry 3 --retry-delay 2 -fsSL -o "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    "${EFA_INSTALLER_URL}"
tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
cd aws-efa-installer

apt-get update

# --skip-mpi: NGC images already ship HPC-X / Open MPI; do not overwrite it.
# --skip-rdma-core: use the base image's rdma-core.
# --build-ngc: build against the NGC-provided userspace rather than installing
#              the bundled one.
./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc

cd ../../
rm -rf "${WORK_DIR}"

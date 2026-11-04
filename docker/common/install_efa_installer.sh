#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

# EFA Installer 1.51.0 supports --build-ngc only on Ubuntu.
# Other distributions fall back to standard installation, which may change
# system RDMA dependencies. Skip them to preserve the base image's RDMA stack.
. /etc/os-release
if [ "${ID}" != ubuntu ]; then
    echo "Skipping EFA installer: OS '${ID}' version '${VERSION_ID}'."
    exit 0
fi

# NGC PyTorch 26.09 bundles EFA libraries here; the official installer updates
# that existing installation in place. Use the same path for runtime lookup.
EFA_LIB_DIR=/opt/amazon/efa/lib
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.51.0}"
# TODO: Use an internal mirror once the pinned EFA installer is available there.
EFA_INSTALLER_URL="https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

curl --connect-timeout 30 --max-time 1800 --retry 3 --retry-delay 2 \
    -fsSL -o "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    "${EFA_INSTALLER_URL}"
tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
cd aws-efa-installer

# Preserve the base image's MPI and rdma-core.
apt-get update
# --build-ngc handles userspace-only installation for Ubuntu containers.
./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc

# Prefer the EFA runtime over an older distribution libfabric in the loader cache.
printf '%s\n' "${EFA_LIB_DIR}" > /etc/ld.so.conf.d/000_efa.conf
ldconfig

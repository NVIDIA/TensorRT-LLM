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

set -ex

# This script is used for reinstalling CUDA on Rocky Linux 8 with the run file.
# CUDA version is usually aligned with the latest NGC CUDA image tag.
# Only use when public CUDA image is not ready.
CUDA_VER="13.3.0_610.43.02"
CUDA_VER_SHORT="${CUDA_VER%_*}"

NVCC_VERSION_OUTPUT=$(nvcc --version)
OLD_CUDA_VER=$(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1)
echo "The version of pre-installed CUDA is ${OLD_CUDA_VER}."

check_cuda_version() {
    if [ -n "$CUDA_VERSION" ] && [ -n "$CUDA_DRIVER_VERSION" ]; then
        CUDA_VERSION_SHORT=$(echo "$CUDA_VERSION" | cut -d'.' -f1-3)
        ENV_CUDA_VER="${CUDA_VERSION_SHORT}_${CUDA_DRIVER_VERSION}"
        if [ "$ENV_CUDA_VER" = "$CUDA_VER" ]; then
            echo "CUDA version matches ($ENV_CUDA_VER), skipping reinstallation"
            return 0
        fi
    fi
    return 1
}

reinstall_rockylinux_cuda() {
    dnf -y install epel-release
    dnf remove -y "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget --retry-connrefused --timeout=180 --tries=10 --continue https://developer.download.nvidia.com/compute/cuda/${CUDA_VER_SHORT}/local_installers/cuda_${CUDA_VER}_linux.run
    sh cuda_${CUDA_VER}_linux.run --silent --override --toolkit
    rm -f cuda_${CUDA_VER}_linux.run
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  rocky)
    if check_cuda_version; then
        echo "CUDA version matches ($CUDA_VER), skipping reinstallation"
        exit 0
    fi
    echo "Reinstall CUDA for RockyLinux 8..."
    reinstall_rockylinux_cuda
    ;;
  *)
    echo "Skip for other OS..."
    ;;
esac

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

have_headers=false
have_library=false
if printf '#include <gdrapi.h>\n' | cc -E -x c - >/dev/null 2>&1; then
    have_headers=true
fi
if ldconfig -p | grep 'libgdrapi\.so\.2 ' >/dev/null; then
    have_library=true
fi
if ${have_headers} && ${have_library}; then
    echo "GDRCOPY_EXISTING_PRESERVED=1"
    exit 0
fi
if ${have_headers} || ${have_library}; then
    echo "Incomplete GDRCopy installation; refusing to overwrite existing files." >&2
    exit 1
fi

. /etc/os-release
# Match GDRCopy in the NGC PyTorch 26.09 base images.
GDRCOPY_VERSION=2.5.1
ARCH=$(uname -m)
# TODO: Add an internal CI mirror after verifying the pinned 2.5.1 DEB/RPM packages.
DOWNLOAD_ROOT="https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2013.0"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

download() {
    curl --connect-timeout 30 --max-time 300 --retry 3 --retry-delay 2 \
        -fsSL "${DOWNLOAD_ROOT}/$1" -o "${1##*/}"
    sha256sum "${1##*/}"
}

case "${ID}:${VERSION_ID}:${ARCH}" in
    ubuntu:22.04:x86_64|ubuntu:24.04:x86_64)
        package="libgdrapi_${GDRCOPY_VERSION}-1_amd64.Ubuntu${VERSION_ID/./_}.deb"
        download "ubuntu${VERSION_ID/./_}/x64/${package}"
        dpkg -i "${package}"
        ;;
    ubuntu:22.04:aarch64|ubuntu:24.04:aarch64)
        package="libgdrapi_${GDRCOPY_VERSION}-1_arm64.Ubuntu${VERSION_ID/./_}.deb"
        download "ubuntu${VERSION_ID/./_}/aarch64/${package}"
        dpkg -i "${package}"
        ;;
    rocky:8*:x86_64)
        runtime="gdrcopy-${GDRCOPY_VERSION}-1.el8.x86_64.rpm"
        development="gdrcopy-devel-${GDRCOPY_VERSION}-1.el8.noarch.rpm"
        download "rhel8/x64/${runtime}"
        download "rhel8/x64/${development}"
        rpm -qp --requires "${runtime}" "${development}"
        dnf install -y "./${runtime}" "./${development}"
        ;;
    *)
        echo "No GDRCopy ${GDRCOPY_VERSION} package selected for ${ID} ${VERSION_ID} ${ARCH}." >&2
        exit 1
        ;;
esac
ldconfig
printf '#include <gdrapi.h>\n' | cc -E -x c - >/dev/null
ldconfig -p | grep 'libgdrapi\.so\.2 '
echo "GDRCOPY_USERSPACE_INSTALLED=${GDRCOPY_VERSION}"

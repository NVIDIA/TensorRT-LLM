#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFA_PATH="/opt/amazon/efa"
LIBFABRIC_VERSION="2.4.0amzn5.0"

# Only patch the AWS EFA installation bundled with NGC PyTorch images.
if [[ ! -f "${EFA_PATH}/include/rdma/fabric.h" || ! -f "${EFA_PATH}/lib/libfabric.so" ]]; then
    exit 0
fi
installed_version=$(PKG_CONFIG_LIBDIR="${EFA_PATH}/lib/pkgconfig" PKG_CONFIG_PATH= pkg-config --modversion libfabric)
if [[ "${installed_version}" != "${LIBFABRIC_VERSION}" ]]; then
    echo "Revalidate the CUDA VMM backport for AWS libfabric ${installed_version}." >&2
    exit 1
fi

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "${BUILD_DIR}"' EXIT
cd "${BUILD_DIR}"

providers() {
    env -u FI_PROVIDER -u FI_PROVIDER_PATH -u LD_PRELOAD LD_LIBRARY_PATH="$1/lib" \
        "$1/bin/fi_info" -l | awk '/:$/ { print }' | LC_ALL=C sort
}
providers "${EFA_PATH}" > providers.before

curl --fail --location --retry 3 \
    "https://codeload.github.com/aws/libfabric/tar.gz/refs/tags/v${LIBFABRIC_VERSION}" -o libfabric.tar.gz
echo '2b0f027fc4ac1f0731b1c4dca786d9c2d0205ddc9d5f660a5685e8b58fc4213b  libfabric.tar.gz' | sha256sum -c -
tar -xzf libfabric.tar.gz

# Neuron is loaded dynamically by the original library; retain that support.
# These are build-only headers, not a replacement Neuron runtime.
NEURON_COMMIT=e27a3986fa8625d4103a7ce611ee0c85ae79979a
mkdir -p neuron/include/nrt neuron/include/ndl
while read -r checksum header; do
    curl --fail --location --retry 3 \
        "https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/${NEURON_COMMIT}/src/libnrt/include/${header}" \
        -o "neuron/include/${header}"
    echo "${checksum}  neuron/include/${header}" | sha256sum -c -
done <<'HEADERS'
1890d030c4628e5f83413bd30f5a52986e55040e853cbeb0acd064f827886dcf nrt/nrt.h
701bffe3b5c2277deef3df96e9d0ac08d45e819812148196f47e1bf458e218a1 nrt/nrt_experimental.h
19d1b61e30610022df94103f8e19e4cb8948acc6a334401d1d2f8da608a2e995 nrt/nrt_status.h
7f34780056a43c05f6c90b54eac87cd3c1c5b92740a9f06436e106fa0383cff1 ndl/neuron_driver_shared_tensor_batch_op.h
HEADERS

cd "libfabric-${LIBFABRIC_VERSION}"
patch --batch -p1 < "${SCRIPT_DIR}/patches/libfabric-2.4.0amzn5.0-cuda-vmm.patch"
mkdir -p /third-party-source
tar -czf "/third-party-source/libfabric-${LIBFABRIC_VERSION}-cuda-vmm.tar.gz" .
./autogen.sh
# Preserve the bundled providers and dynamic accelerator support.
./configure --prefix="${EFA_PATH}" --libdir="${EFA_PATH}/lib" --disable-static \
    --enable-only --enable-efa --enable-rxm --enable-rxd --enable-shm \
    --enable-udp --enable-tcp --enable-sockets --enable-sm2 --enable-mrail \
    --enable-perf --enable-trace --enable-monitor --enable-hook_debug \
    --enable-hook_hmem --enable-dmabuf_peer_mem \
    --enable-cuda-dlopen --with-cuda=/usr/local/cuda \
    --enable-gdrcopy-dlopen --with-gdrcopy=/usr --with-neuron="${BUILD_DIR}/neuron"
for feature in HAVE_CUDA HAVE_CUDA_CTX_SYNC_MEMOPS HAVE_GDRCOPY HAVE_NEURON; do
    grep -q "^#define ${feature} 1$" config.h
done
make -j"$(nproc)"
make DESTDIR="${BUILD_DIR}/install" install
providers "${BUILD_DIR}/install${EFA_PATH}" > "${BUILD_DIR}/providers.after"
diff -u "${BUILD_DIR}/providers.before" "${BUILD_DIR}/providers.after"
cp -a "${BUILD_DIR}/install${EFA_PATH}/." "${EFA_PATH}/"
ldconfig

#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

ARCH=$(uname -m)
CUDA_ARCHS=${CUDA_ARCHS:-90-real;100-real}
DEEPEP_COMMIT=a0a6d788e6340723da8e42a3cde049fba26fc3eb

# Custom NVSHMEM
curl -fsSL https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz | tar xz
mv nvshmem_src custom_nvshmem
pushd custom_nvshmem
curl -fsSL $GITHUB_URL/deepseek-ai/DeepEP/raw/$DEEPEP_COMMIT/third-party/nvshmem.patch | git apply
sed "s/TRANSPORT_VERSION_MAJOR 3/TRANSPORT_VERSION_MAJOR 103/" -i src/CMakeLists.txt
ln -s /usr/lib/${ARCH}-linux-gnu/libmlx5.so.1 /usr/lib/${ARCH}-linux-gnu/libmlx5.so
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/opt/custom_nvshmem \
    -DGDRCOPY_HOME=/usr/include \
    -DNVSHMEM_SHMEM_SUPPORT=0 \
    -DNVSHMEM_UCX_SUPPORT=0 \
    -DNVSHMEM_USE_NCCL=0 \
    -DNVSHMEM_MPI_SUPPORT=0 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=0 \
    -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    -DNVSHMEM_USE_GDRCOPY=1 \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DNVSHMEM_BUILD_TESTS=0 \
    -DNVSHMEM_BUILD_EXAMPLES=0
cmake --build build -j`nproc`
make -C build install
popd

# DeepEP
TORCH_CUDA_ARCH_LIST="9.0;10.0" NVSHMEM_DIR=/opt/custom_nvshmem pip install git+$GITHUB_URL/deepseek-ai/DeepEP.git@${DEEPEP_COMMIT}

# Clean up
rm -r custom_nvshmem
rm /usr/lib/${ARCH}-linux-gnu/libmlx5.so

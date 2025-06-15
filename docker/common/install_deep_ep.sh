#!/bin/bash

set -euxo pipefail

GITHUB_URL=${GITHUB_MIRROR:-https://github.com}
DEEP_EP_COMMIT=2b266cf6452134f993ab0fcb3ef2d5de7683c561

if [ "$(. /etc/os-release && echo $ID)" == "rocky" ]; then
    echo "Skipping DeepEP installation in the Rocky distribution."
    exit 0
fi
libmlx5_dir=$(dirname $(ldconfig -p | grep libmlx5.so.1 | head -n1 | awk '{print $NF}'))

export NVCC_APPEND_FLAGS="--threads 4"

# Custom NVSHMEM
curl -fsSL https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz | tar xz
pushd nvshmem_src
curl -fsSL $GITHUB_URL/deepseek-ai/DeepEP/raw/$DEEP_EP_COMMIT/third-party/nvshmem.patch | patch -p1
sed "s/TRANSPORT_VERSION_MAJOR 3/TRANSPORT_VERSION_MAJOR 103/" -i src/CMakeLists.txt
ln -s libmlx5.so.1 "$libmlx5_dir/libmlx5.so"
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
    -DCMAKE_CUDA_ARCHITECTURES="90-real;100-real;120-real" \
    -DNVSHMEM_BUILD_TESTS=0 \
    -DNVSHMEM_BUILD_EXAMPLES=0
cmake --build build -j`nproc`
make -C build install
popd

# DeepEP
curl -fsSL $GITHUB_URL/deepseek-ai/DeepEP/archive/$DEEP_EP_COMMIT.tar.gz | tar xz
TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" NVSHMEM_DIR=/opt/custom_nvshmem pip install -v --no-cache-dir ./DeepEP-$DEEP_EP_COMMIT

# Clean up
rm -r nvshmem_src
rm "$libmlx5_dir/libmlx5.so"
rm -r DeepEP-$DEEP_EP_COMMIT

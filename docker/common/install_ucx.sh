#!/bin/bash
set -ex

UCX_VERSION="v1.20.x"
UCX_COMMIT="f656dbdf93e72e60b5d6ca78b9e3d9e744e789bd"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}
git clone -b ${UCX_VERSION} ${UCX_REPO}
cd ucx
git checkout ${UCX_COMMIT}
# UCX v1.20.x GDAKI sources can be built after DOCA 3.3 headers are
# installed, but this commit does not provide a local READ_ONCE fallback
# for the CUDA perf kernel path. Add the small fallback here instead of
# disabling CUDA support for the whole UCX build.
if [ -f src/uct/ib/mlx5/gdaki/gdaki.cuh ] &&
   ! grep -q "TRTLLM_UCX_READ_ONCE_FALLBACK" src/uct/ib/mlx5/gdaki/gdaki.cuh; then
  sed -i '1i\
#ifndef TRTLLM_UCX_READ_ONCE_FALLBACK\
#define TRTLLM_UCX_READ_ONCE_FALLBACK\
#ifndef READ_ONCE\
#define READ_ONCE(x) (*(volatile decltype(x) *)&(x))\
#endif\
#ifndef WRITE_ONCE\
#define WRITE_ONCE(x, v) (*(volatile decltype(x) *)&(x) = (v))\
#endif\
#endif\
' src/uct/ib/mlx5/gdaki/gdaki.cuh
fi
cd ..
tar -czf /third-party-source/ucx-${UCX_VERSION}.tar.gz ucx
cd ucx
./autogen.sh
./contrib/configure-release       \
  --prefix=${UCX_INSTALL_PATH}    \
  --enable-shared                 \
  --disable-static                \
  --disable-doxygen-doc           \
  --enable-optimizations          \
  --enable-cma                    \
  --enable-devel-headers          \
  --with-cuda=${CUDA_PATH}        \
  --with-verbs                    \
  --with-dm                       \
  --enable-mt
make install -j$(nproc)
cd ..
rm -rf ucx  # Remove UCX source to save space
echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"

#!/bin/bash
set -ex

UCX_VERSION="v1.21.x"
UCX_COMMIT="167a4c6a311d9a42e30a37dcc01b8a3e73ea2826"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

mkdir -p /third-party-source

# Drop the trailing slash so this also removes a pre-existing *symlink* at
# ${UCX_INSTALL_PATH} (the NGC PyTorch base image ships /usr/local/ucx as a
# symlink to /opt/hpcx/ucx). Without this, `make install` writes through the
# symlink into /opt/hpcx/ucx, and the later /opt/hpcx/ucx -> /usr/local/ucx
# replacement creates an infinite symlink loop.
rm -rf "${UCX_INSTALL_PATH%/}"
git clone -b ${UCX_VERSION} ${UCX_REPO}
cd ucx
git checkout ${UCX_COMMIT}
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
# Replace any pre-existing UCX (e.g. HPC-X's UCX shipped in the NGC PyTorch
# base image) with a symlink to our freshly-installed one, so every binary
# in the container — PyTorch, NCCL, UCC, plus the C++/Python NIXL stack —
# resolves the same UCX SONAMEs.
for d in /opt/hpcx/ucx /opt/hpcx-*/ucx; do
    # `[ -e ]` catches real dirs/files; `[ -L ]` separately catches symlinks
    # (including broken ones, where -e returns false). Together they replace
    # any pre-existing UCX — real dir, stale symlink to an old UCX, or our
    # own symlink on re-run (idempotent).
    if [ -e "$d" ] || [ -L "$d" ]; then
        echo "Replacing pre-existing UCX at $d with symlink to ${UCX_INSTALL_PATH%/}"
        rm -rf "$d"
        ln -s "${UCX_INSTALL_PATH%/}" "$d"
    fi
done
# Make /usr/local/ucx/lib known to the system dynamic linker so binaries
# without an explicit RPATH (or with $ORIGIN-relative RPATH) still find it.
echo "${UCX_INSTALL_PATH%/}/lib" > /etc/ld.so.conf.d/ucx.conf
ldconfig
cd ..
rm -rf ucx  # Remove UCX source to save space
echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"

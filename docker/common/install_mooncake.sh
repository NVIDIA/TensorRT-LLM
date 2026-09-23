#!/bin/bash
set -ex

# Authenticate the github.com clones below; no-op when no token is available.
source "$(dirname "${BASH_SOURCE[0]}")/github_auth.sh"

MOONCAKE_VERSION="v0.3.7.post2"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
MOONCAKE_INSTALL_PATH="/usr/local/Mooncake"

apt-get update

# https://kvcache-ai.github.io/Mooncake/getting_started/build.html
# libboost-all-dev is removed because it will install a duplicated MPI library
# triton also installed boost so the requirement is already met
apt-get install -y --no-install-recommends \
    build-essential \
    libibverbs-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libnuma-dev \
    libunwind-dev \
    libssl-dev \
    libyaml-cpp-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    pkg-config \
    patchelf

mkdir -p /third-party-source

git clone --depth 1 https://github.com/alibaba/yalantinglibs.git
tar -czf /third-party-source/yalantinglibs.tar.gz yalantinglibs
cd yalantinglibs
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j
make install
cd ../..
rm -rf yalantinglibs

git clone --depth 1 -b ${MOONCAKE_VERSION} ${MOONCAKE_REPO}
tar -czf /third-party-source/Mooncake-${MOONCAKE_VERSION}.tar.gz Mooncake
cd Mooncake
git submodule update --init --recursive --depth 1
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${MOONCAKE_INSTALL_PATH}
make -j
make install
cd ../..
rm -rf Mooncake

echo "export LD_LIBRARY_PATH=${MOONCAKE_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"

# The source build above provides only the C++ transfer engine, which is what
# the cache transceiver links against. MooncakeDistributedStore, the shared CPU
# pool behind the mooncake-store KV cache connector, comes from the Python
# wheel below.
#
# `make install` also emits a `mooncake` package that omits
# libmooncake_store.so, so importing mooncake.store from it fails. It has to go
# before the wheel is installed: CMake writes
# store.cpython-312-x86_64-linux-gnu.so where the wheel writes store.so, and
# importlib prefers the interpreter-tagged suffix, so the broken extension
# would win even after pip reports success. The directory is the one
# mooncake-integration/CMakeLists.txt chose, which this repeats.
MOONCAKE_CMAKE_PACKAGE="$(python3 -c "import sys; print([s for s in sys.path if 'packages' in s][0])")/mooncake"
echo "removing CMake-generated mooncake package: ${MOONCAKE_CMAKE_PACKAGE}"
rm -rf "${MOONCAKE_CMAKE_PACKAGE}"

# The Mooncake Python client (MooncakeDistributedStore) is installed from
# requirements-mooncake.txt, the pin shared with the standalone wheel's
# `tensorrt-llm[mooncake]` extra. That file explains why the pin is independent
# of MOONCAKE_VERSION above and why the CUDA 13 wheel is required.
# The Docker build bind-mounts the file at /opt; a run from a checkout finds it
# relative to this script instead.
MOONCAKE_REQUIREMENTS="${MOONCAKE_REQUIREMENTS:-/opt/requirements-mooncake.txt}"
if [ ! -f "${MOONCAKE_REQUIREMENTS}" ]; then
    MOONCAKE_REQUIREMENTS="$(dirname "${BASH_SOURCE[0]}")/../../requirements-mooncake.txt"
fi
pip3 install --no-cache-dir -r "${MOONCAKE_REQUIREMENTS}"

# Fail the build rather than ship an image whose import is broken.
python3 - <<'PY'
import mooncake.store

mooncake.store.MooncakeDistributedStore()
print(f"mooncake.store OK: {mooncake.store.__file__}")
PY

#!/bin/bash
set -ex

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
# wheel instead, for two reasons.
#
# First, `make install` emits a `mooncake` Python package that omits
# libmooncake_store.so, so importing mooncake.store raises ImportError. It has
# to be removed wherever it landed, and where that is depends on the
# environment: mooncake-integration/CMakeLists.txt picks its install directory
# as the first sys.path entry whose name merely contains "packages".
#
#   - With nvidia-cutlass-dsl installed, that is
#     nvidia_cutlass_dsl/dsl_packages, which nvidia_cutlass_dsl_packages.pth
#     puts at sys.path[0], so it shadows anything pip installs. CUTLASS DSL
#     does not reference `mooncake`, so removing the package is safe.
#   - Without it, the package lands in dist-packages and collides with the
#     wheel: CMake writes store.cpython-312-x86_64-linux-gnu.so, the wheel
#     writes store.so, and importlib prefers the interpreter-tagged suffix, so
#     the broken extension wins even after pip reports success.
#
# Remove the directory outright rather than trying to identify leftovers, since
# pip overwrites __init__.py in the collision case and leaves no marker to key
# on.
python3 - <<'PY'
import os
import shutil
import sys
import sysconfig

paths = sysconfig.get_paths()
for entry in list(sys.path) + [paths["purelib"], paths["platlib"]]:
    if not entry:
        continue
    package = os.path.join(entry, "mooncake")
    if os.path.isdir(package):
        print(f"removing CMake-generated mooncake package: {package}")
        shutil.rmtree(package, ignore_errors=True)
PY

# Second, the `mooncake-transfer-engine` wheel is built against CUDA 12 while
# these images ship CUDA 13 only, so its extensions cannot resolve
# libcudart.so.12. `mooncake-transfer-engine-cuda13` is the same project built
# for CUDA 13. It is versioned independently, with releases starting at 0.3.9,
# so it cannot track MOONCAKE_VERSION above. The store client only has to agree
# with the mooncake_master it connects to, and this wheel supplies both.
MOONCAKE_WHEEL_VERSION="0.3.13"
pip3 install --no-cache-dir "mooncake-transfer-engine-cuda13==${MOONCAKE_WHEEL_VERSION}"

# Fail the build rather than ship an image whose import is broken.
python3 - <<'PY'
from mooncake.store import MooncakeDistributedStore
import mooncake.store

MooncakeDistributedStore()
print(f"mooncake.store OK: {mooncake.store.__file__}")
PY

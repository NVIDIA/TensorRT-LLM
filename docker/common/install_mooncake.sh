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

# The source build above is only useful for the C++ transfer engine, which is
# what the cache transceiver links against. MooncakeDistributedStore -- the
# shared CPU pool behind the mooncake-store KV cache connector -- comes from the
# Python wheel instead, for two reasons.
#
# First, `make install` does emit a `mooncake` Python package, but an unusable
# one: it omits libmooncake_store.so, so importing mooncake.store raises
# ImportError. It must be deleted, and deleting it is not optional in either of
# the two places it can land.
#
# mooncake-integration/CMakeLists.txt chooses its install directory with
#   python3 -c "import sys; print([s for s in sys.path if 'packages' in s][0])"
# i.e. the first sys.path entry whose name merely contains "packages".
#
#   - With nvidia-cutlass-dsl installed (the normal case here: the devel stage
#     removes it, then constraints.txt reinstalls it), that first match is
#     nvidia_cutlass_dsl/dsl_packages, because nvidia_cutlass_dsl_packages.pth
#     does sys.path.insert(0) on it. The broken package then outranks
#     dist-packages on every interpreter start, so no amount of pip installing
#     can fix the import. CUTLASS DSL does not reference `mooncake` at all, so
#     removing it is safe.
#   - Without it, the match is dist-packages itself, and the broken package
#     collides with the wheel. That is the more insidious case: CMake writes
#     store.cpython-312-x86_64-linux-gnu.so while the wheel writes store.so, and
#     importlib prefers the interpreter-tagged suffix, so the broken extension
#     still wins even after pip reports success.
#
# Remove the package outright wherever it landed, before pip installs the real
# one. Nothing legitimate owns a `mooncake` package at this point, and removing
# rather than trying to identify individual leftovers keeps this correct in the
# dist-packages case, where pip would overwrite __init__.py and leave no marker
# to key on.
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

# Second, the `mooncake-transfer-engine` wheel is built against CUDA 12 and
# these images ship CUDA 13 only, so its extensions cannot resolve
# libcudart.so.12. `mooncake-transfer-engine-cuda13` is the same project built
# for CUDA 13. It is versioned independently and its releases start at 0.3.9, so
# it cannot track MOONCAKE_VERSION above; the store client only has to agree
# with the mooncake_master it connects to, and the wheel supplies both.
MOONCAKE_WHEEL_VERSION="0.3.13"
pip3 install --no-cache-dir "mooncake-transfer-engine-cuda13==${MOONCAKE_WHEEL_VERSION}"

# Fail the build rather than ship an image whose import is broken.
python3 - <<'PY'
from mooncake.store import MooncakeDistributedStore
import mooncake.store

MooncakeDistributedStore()
print(f"mooncake.store OK: {mooncake.store.__file__}")
PY

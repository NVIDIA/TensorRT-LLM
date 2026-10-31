#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Authenticate the github.com clone below; no-op when no token is available.
source "$(dirname "${BASH_SOURCE[0]}")/github_auth.sh"

UCX_VERSION="v1.23.x"
UCX_COMMIT="65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0"
UCX_TIMER_FIX_COMMIT="62182080282372e27a5cad45f23b7ac424b7cd94"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}

# Fetch just the pinned commit rather than cloning the whole history
rm -rf ucx
git init -q ucx
git -C ucx remote add origin ${UCX_REPO}
git -C ucx fetch -q --depth 1 origin ${UCX_COMMIT}
git -C ucx checkout -q FETCH_HEAD

# Advance the timer wheel before callbacks to prevent false UD peer timeouts.
git -C ucx fetch -q --depth 2 origin ${UCX_TIMER_FIX_COMMIT}
git -C ucx diff ${UCX_TIMER_FIX_COMMIT}^ ${UCX_TIMER_FIX_COMMIT} > ucx/ucx-timer-fix.patch
git -C ucx apply ucx-timer-fix.patch
rm ucx/ucx-timer-fix.patch

tar -czf /third-party-source/ucx-${UCX_VERSION}.tar.gz ucx
cd ucx
# Pull external/gpunetio shallow, for the same reason: autogen.sh below does a
# full-history `git submodule update --init` on it. With the submodule already
# at the recorded commit that call does nothing.
git submodule update --init --depth 1
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
